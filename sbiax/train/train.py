from typing import Tuple, Optional
from copy import deepcopy
from dataclasses import replace
import os
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.sharding import NamedSharding
import equinox as eqx
from jaxtyping import Key, Array, PyTree
import optax
import numpy as np
from tqdm.auto import tqdm, trange 
import matplotlib.pyplot as plt
import optuna

from .loss import pdf_mse_loss, batch_loss_fn, batch_eval_fn
from .loader import _InMemoryDataLoader, sort_sample
from ..ndes import Ensemble

Optimiser = optax.GradientTransformation 


def apply_ema(
    ema_model: eqx.Module, 
    model: eqx.Module, 
    ema_rate: float = 0.9999
) -> eqx.Module:
    ema_fn = lambda p_ema, p: p_ema * ema_rate + p * (1. - ema_rate)
    m_, _m = eqx.partition(model, eqx.is_inexact_array)
    e_, _e = eqx.partition(ema_model, eqx.is_inexact_array)
    e_ = jtu.tree_map(ema_fn, e_, m_)
    return eqx.combine(e_, _m)


def clip_grad_norm(grads, max_norm):
    norm = jnp.linalg.norm(
        jtu.tree_leaves(
            jtu.tree_map(jnp.linalg.norm, grads)
        )
    )
    factor = jnp.minimum(max_norm, max_norm / (norm + 1e-6))
    return jtu.tree_map(lambda x: x * factor, grads)


@eqx.filter_jit
def make_step(
    nde: eqx.Module, 
    x: Array, 
    y: Array, 
    opt_state: PyTree,
    opt: Optimiser,
    clip_max_norm: float,
    key: Key
) -> Tuple[eqx.Module, PyTree, Array]:
    _fn = eqx.filter_value_and_grad(batch_loss_fn)
    L, grads = _fn(nde, x, y, key=key)
    if clip_max_norm is not None:
        grads = clip_grad_norm(grads, clip_max_norm)
    updates, opt_state = opt.update(grads, opt_state, nde)
    nde = eqx.apply_updates(nde, updates)
    return nde, opt_state, L 


def count_params(nde: eqx.Module) -> int:
    return sum(x.size for x in jtu.tree_leaves(nde) if eqx.is_array(x))


def get_n_split_keys(key: Key, n: int) -> Tuple[Key, Array]:
    key, *keys = jr.split(key, n + 1)
    return key, jnp.asarray(keys)


def partition_and_preprocess_data(
    key: Key, 
    train_data: Tuple[Array, ...], 
    valid_fraction: float, 
    n_batch: int, 
) -> Tuple[Tuple[Array, ...], Tuple[Array, ...], Tuple[int, ...]]:

    # Number of training and validation samples
    n_train_data = len(train_data[0]) 
    n_valid = int(n_train_data * valid_fraction)
    n_train = n_train_data - n_valid

    # Partition dataset into training and validation sets (different split for each NDE!)
    idx = jr.permutation(key, jnp.arange(n_train_data)) 
    is_train, is_valid = jnp.split(idx, [n_train])

    # Simulations, parameters, pdfs (optional)
    data_train = tuple(data[is_train] for data in train_data)
    data_valid = tuple(data[is_valid] for data in train_data)

    # Total numbers of batches
    n_train_data, n_valid_data = len(data_train[0]), len(data_valid[0])
    if n_batch is not None:
        n_train_batches = max(int(n_train_data / n_batch), 1)
        n_valid_batches = max(int(n_valid_data / n_batch), 1)
    else:
        n_train_batches = n_valid_batches = None

    return data_train, data_valid, (n_train_batches, n_valid_batches)


def get_loaders(
    key: Key, 
    data_train: Tuple[Array, ...], 
    data_valid: Tuple[Array, ...], 
    train_mode: str
) -> Tuple[_InMemoryDataLoader, _InMemoryDataLoader]:
    train_dl_key, valid_dl_key = jr.split(key)
    train_dataloader = _InMemoryDataLoader(
        *data_train, train_mode=train_mode, key=train_dl_key
    )
    valid_dataloader = _InMemoryDataLoader(
        *data_valid, train_mode=train_mode, key=valid_dl_key
    )
    return train_dataloader, valid_dataloader


def get_initial_stats() -> dict:
    stats = dict(
        train_losses=[],
        valid_losses=[],
        best_loss=jnp.inf,      # Best valid loss
        best_epoch=0,           # Epoch of best valid loss
        stopping_count=0,       # Epochs since last improvement
        all_valid_loss=jnp.inf, # Validation loss on whole validation set 
        best_nde=None           # Best NDE state
    )
    return stats


def count_epochs_since_best(losses: list[float]) -> int:
    min_idx = jnp.argmin(jnp.array(losses)).item()
    return len(losses) - min_idx - 1


def train_nde(
    key: Key,
    # NDE
    model: eqx.Module,
    train_mode: str,
    # Data
    train_data: Tuple[Array, ...],
    test_data: Tuple[Array, ...] = None, # Independent test data
    # Hyperparameters 
    opt: Optimiser = optax.adam(1e-3),
    valid_fraction: float = 0.1,
    n_epochs: int = 100_000,
    n_batch: int = 100,
    patience: int = 50,
    clip_max_norm: Optional[float] = None,
    # Sharding
    sharding: Optional[NamedSharding] = None,
    # Saving
    results_dir: Optional[str] = None,
    trial: Optional[optuna.trial.Trial] = None,
    tqdm_description: str = "Training",
    show_tqdm: bool = False,
) -> Tuple[eqx.Module, dict]:
    """
    Trains a neural density estimator (NDE) model. 
    
    Supports early stopping, gradient clipping, and Optuna integration for hyperparameter tuning.

    Args:
        `key` (Key): A random key for stochastic operations (e.g., for shuffling, dropout).
        `model` (eqx.Module): The NDE model to be trained.
        `train_mode` (str): Type of NDE training (e.g. neural likelihood estimation, neural posterior estimation).
        `train_data` (Tuple[Array, ...]): Training data tuple consisting of input and target data.
        `test_data` (Tuple[Array, ...], optional): Optional test data for final model validation. Defaults to `None`.
        `opt` (Optimiser, optional): Optimizer for training the model. Defaults to Adam with learning rate `1e-3`.
        `valid_fraction` (float, optional): Fraction of the training data to use for validation. Defaults to `0.1`.
        `n_epochs` (int, optional): Total number of epochs to run the training. Defaults to `100_000`.
        `n_batch` (int, optional): Batch size for training. Defaults to `100`.
        `patience` (int, optional): Number of epochs to wait before early stopping if no improvement is seen. Defaults to `50`.
        `clip_max_norm` (float, optional): Maximum norm for gradient clipping. If `None`, no clipping is applied.
        `sharding` (Optional[NamedSharding], optional): Sharding strategy to partition data across devices. Defaults to `None`.
        `results_dir` (str, optional): Directory to save training results (e.g., model checkpoints and loss plots). Defaults to `None`.
        `trial` (optuna.trial.Trial, optional): Optuna trial for hyperparameter optimization. Can be used to prune unpromising runs. Defaults to `None`.
        `show_tqdm` (bool, optional): Whether to display a progress bar for the training loop. Defaults to `False`.

    Returns:
        Tuple[eqx.Module, dict]:
            - The trained NDE model, either at the last epoch or the epoch with the best validation loss.
            - A dictionary containing training statistics such as loss values, best loss, and the best epoch.
            
    Key Steps:
        1. Partitions and preprocesses the training data into training and validation sets.
        2. Trains the model for `n_epochs` using the specified optimizer and training data.
        3. Tracks training and validation losses and applies early stopping based on validation loss.
        4. Optionally applies gradient clipping and Optuna trial pruning.
        5. Plots and saves the training and validation losses, and saves the best-performing model.
        6. Returns the trained model and relevant training statistics.

    Raises:
        optuna.exceptions.TrialPruned: If the Optuna trial is pruned based on validation loss.
    """

    if not os.path.exists(results_dir):
        os.mkdir(results_dir) 

    # Get training / validation data (frozen per training per NDE)
    key, key_data = jr.split(key)
    (
        data_train, data_valid, (n_train_batches, n_valid_batches)
    ) = partition_and_preprocess_data(
        key_data, train_data, valid_fraction, n_batch=n_batch
    )

    del train_data # Release train_data from memory

    n_params = count_params(model)
    print(f"NDE has n_params={n_params}.")

    opt_state = opt.init(eqx.filter(model, eqx.is_array)) 

    # Stats for training and NDE
    stats = get_initial_stats()

    if show_tqdm:
        epochs = trange(
            n_epochs, desc=tqdm_description, colour="green", unit="epoch"
        )
    else:
        epochs = range(n_epochs)

    for epoch in epochs:

        # Loop through D={d_i} once per epoch, using same validation set
        key, key_loaders = jr.split(key)
        train_dataloader, valid_dataloader = get_loaders(
            key_loaders, data_train, data_valid, train_mode=train_mode
        )

        # Train 
        epoch_train_loss = 0.
        for s, xy in zip(
            range(n_train_batches), train_dataloader.loop(n_batch)
        ):
            key = jr.fold_in(key, s)
            
            if sharding is not None:
                xy = eqx.filter_shard(xy, sharding)

            model, opt_state, train_loss = make_step(
                model, xy.x, xy.y, opt_state, opt, clip_max_norm, key
            )

            epoch_train_loss += train_loss 

        stats["train_losses"].append(epoch_train_loss / (s + 1)) 

        # Validate 
        epoch_valid_loss = 0.
        for s, xy in zip(
            range(n_valid_batches), valid_dataloader.loop(n_batch)
        ):
            key = jr.fold_in(key, s)

            if sharding is not None:
                xy = eqx.filter_shard(xy, sharding)

            valid_loss = batch_eval_fn(model, xy.x, xy.y, key=key)

            epoch_valid_loss += valid_loss

        stats["valid_losses"].append(epoch_valid_loss / (s + 1))

        if show_tqdm:
            epochs.set_postfix(
                ordered_dict={
                    "train" : f"{stats["train_losses"][-1]:.3E}",
                    "valid" : f"{stats["valid_losses"][-1]:.3E}",
                    "best_valid" : f"{stats["best_loss"]:.3E}",
                    "stop" : f"{(patience - stats["stopping_count"] if patience is not None else 0):04d}"
                },
                refresh=True
            )

        # Break training for any broken NDEs
        if not jnp.isfinite(stats["valid_losses"][-1]) or not jnp.isfinite(stats["train_losses"][-1]):
            if show_tqdm:
                epochs.set_description_str(
                    f"\nTraining terminated early at epoch {epoch + 1} (NaN loss).", 
                    # end="\n\n"
                )
            break

        # Optuna can cut this run early 
        if trial is not None:
            # No pruning with multi-objectives
            if len(trial.study.directions) > 1:
                pass
            else:
                trial.report(stats["best_loss"], epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        # Early stopping for NDE training; return best NDE
        if patience is not None:
            better_loss = stats["valid_losses"][-1] < stats["best_loss"]

            # count_epochs_since_best(stats["valid_losses"])

            if better_loss:
                stats["best_loss"] = stats["valid_losses"][-1]
                stats["best_nde"] = deepcopy(model) # Save model with best loss, not just the one at the end of training
                stats["best_epoch"] = epoch - 1 # NOTE: check this
                stats["stopping_count"] = 0
            else:
                stats["stopping_count"] += 1

                if stats["stopping_count"] > patience: 
                    if show_tqdm:
                        epochs.set_description_str(
                            f"Training terminated early at epoch {epoch + 1}; " + 
                            f"valid={stats["valid_losses"][-1]:.3E}, " + 
                            f"train={stats["train_losses"][-1]:.3E}.", 
                        )

                    # NOTE: question of 'best' vs 'last' nde parameters to use (last => converged)
                    # model = stats["best_nde"] # Use best model when quitting, from some better epoch
                    break

    # Plot losses
    epochs = np.arange(0, epoch)
    train_losses = np.asarray(stats["train_losses"][:epoch])
    valid_losses = np.asarray(stats["valid_losses"][:epoch])

    plt.figure()
    plt.title("NDE losses")
    plt.plot(epochs, train_losses, label="train")
    plt.plot(
        epochs,
        valid_losses, 
        label="valid", 
        color=plt.gca().lines[-1].get_color(),
        linestyle=":"
    )
    plt.plot(
        stats["best_epoch"], 
        valid_losses[stats["best_epoch"]],
        marker="x", 
        color="red",
        label=f"Best loss {stats["best_loss"]:.3E}",
        linestyle=""
    )
    plt.legend()
    plt.savefig(os.path.join(results_dir, "losses.png"))
    plt.close()

    # Save NDE model
    eqx.tree_serialise_leaves(
        os.path.join(results_dir, "models/", "cnf.eqx"), model
    )

    # Use test data for validation else just validation set
    if test_data is not None:
        X, Y = test_data 
    else:
        X, Y = data_valid

    xy = sort_sample(train_mode, X, Y) # Arrange for NLE or NPE

    all_valid_loss = batch_eval_fn(model, x=xy.x, y=xy.y, key=key)
    stats["all_valid_loss"] = all_valid_loss

    return model, stats


def plot_losses(ensemble, filename, fisher=False):
    plt.figure()
    plt.title("NDE losses")
    negatives = False
    for nde in ensemble.ndes:
        _losses = nde.fisher_train_losses if fisher else nde.train_losses
        Lt = _losses.train
        Lv = _losses.valid
        if np.any((Lt < 0.) | (Lv < 0.)):
            negatives = True
    plotter = plt.semilogy if not negatives else plt.plot
    for nde in ndes:
        _losses = nde.fisher_train_losses if fisher else nde.train_losses
        plotter(_losses.train, label=nde.name + " (train)")
        plotter(
            _losses.valid, 
            label=nde.name + " (valid)", 
            color=plt.gca().lines[-1].get_color(),
            linestyle=":"
        )
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def train_ensemble(
    key: Key,
    # NDE
    ensemble: Ensemble,
    train_mode: str,
    # Data
    train_data: Tuple[Array, ...],
    test_data: Tuple[Array, ...] = None, # Independent test data
    # Hyperparameters 
    opt: Optimiser = optax.adam(1e-3),
    valid_fraction: float = 0.1,
    n_epochs: int = 100_000,
    n_batch: int = 100,
    patience: int = 50,
    clip_max_norm: Optional[float] = None,
    # Sharding
    sharding: Optional[NamedSharding] = None,
    # Saving
    results_dir: Optional[str] = None,
    tqdm_description: str = "Training",
    show_tqdm: bool = True,
) -> Tuple[eqx.Module, dict]:
    """

    Trains an ensemble of neural density estimator (NDE) models. 
    
    Supports early stopping, gradient clipping, and Optuna integration for hyperparameter tuning.

    Each model in the ensemble is trained independently, and the ensemble's stacking weights are calculated based on validation losses.

    Args:
        `key` (Key): A random key for stochastic operations (e.g., for shuffling, dropout).
        `ensemble` (Ensemble): The ensemble of NDE models to be trained.
        `train_mode` (str): Mode of training, defining how the data is used (e.g., for conditional or unconditional training).
        `train_data` (Tuple[Array, ...]): Training data tuple consisting of input and target data.
        `test_data` (Tuple[Array, ...], optional): Optional test data for final model validation. Defaults to `None`.
        `opt` (Optimiser, optional): Optimizer for training the ensemble models. Defaults to Adam with a learning rate of `1e-3`.
        `valid_fraction` (float, optional): Fraction of the training data to use for validation. Defaults to `0.1`.
        `n_epochs` (int, optional): Total number of epochs to run the training. Defaults to `100_000`.
        `n_batch` (int, optional): Batch size for training. Defaults to `100`.
        `patience` (int, optional): Number of epochs to wait before early stopping if no improvement is seen. Defaults to `50`.
        `clip_max_norm` (float, optional): Maximum norm for gradient clipping. If `None`, no clipping is applied.
        `sharding` (Optional[NamedSharding], optional): Sharding strategy to partition data across devices. Defaults to `None`.
        `results_dir` (str, optional): Directory to save training results (e.g., model checkpoints and loss plots). Defaults to `None`.
        `tqdm_description` (str, optional): Description to show in the progress bar for the training loop. Defaults to `"Training"`.
        `show_tqdm` (bool, optional): Whether to display a progress bar for the training loop. Defaults to `False`.

    Returns:
        Tuple[eqx.Module, dict]:
            - The trained ensemble of NDE models with updated stacking weights.
            - A list of dictionaries containing training statistics for each NDE model, such as loss values, best loss, and the best epoch.

    Key Steps:
        1. Each NDE in the ensemble is trained.
        2. Tracks training and validation losses for each NDE model.
        3. Stacking weights for the ensemble are calculated based on validation losses.
        4. Returns the trained ensemble and training statistics.

    """

    stats = []
    ndes = []
    for n, nde in enumerate(ensemble.ndes):
        key = jr.fold_in(key, n)

        nde, stats_n = train_nde(
            key,
            nde,
            train_mode,
            train_data,
            test_data,
            opt,
            valid_fraction,
            n_epochs,
            n_batch,
            patience,
            clip_max_norm,
            sharding=sharding,
            results_dir=results_dir,
            trial=None,
            tqdm_description=tqdm_description,
            show_tqdm=show_tqdm
        )

        ensemble.ndes[n] = nde
        stats.append(stats_n)
        ndes.append(nde)

    # ensemble = replace(ensemble, ndes=ndes)
    
    weights = ensemble.calculate_stacking_weights(
        losses=[stats[n]["all_valid_loss"] for n, _ in enumerate(ensemble.ndes)]
    )
    ensemble = replace(ensemble, weights=weights)
    print("Weights:", ensemble.weights)

    return ensemble, stats