from typing import Tuple, Optional 
from copy import deepcopy
from dataclasses import replace
import os
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.sharding import NamedSharding, PositionalSharding
import equinox as eqx
from jaxtyping import Key, PRNGKeyArray, Array, PyTree, Float, jaxtyped
from beartype import beartype as typechecker
import optax
import numpy as np
from tqdm.auto import tqdm, trange 
import matplotlib.pyplot as plt
import optuna

from .loss import batch_loss_fn, batch_eval_fn
from .loader import _InMemoryDataLoader, sort_sample
from ..ndes import Ensemble

Optimiser = optax.GradientTransformation 


def shard_batch(
    batch: Tuple[Float[Array, "n x"], Float[Array, "n y"]], 
    sharding: Optional[NamedSharding] = None
) -> Tuple[Float[Array, "n x"], Float[Array, "n y"]]:
    """
    Applies sharding to a batch of data for distributed processing.

    Args:
        batch (`Tuple[Array, Array]`): A tuple containing:
            - Simulation data of shape `(n, x)`.
            - Parameter data of shape `(n, y)`.
        sharding (`jax.sharding.NamedSharding`): An optional `NamedSharding` 
            object to define how the batch is distributed across devices.

    Returns:
        (`Tuple[Array, Array]`): The batch, potentially sharded according to the specified `sharding`.

    Notes:
        - If `sharding` is not provided, the batch remains unchanged.
        - Useful for distributing data across devices in multi-GPU or TPU setups.
    """
    if sharding:
        batch = eqx.filter_shard(batch, sharding)
    return batch


def apply_ema(
    ema_model: eqx.Module, 
    model: eqx.Module, 
    ema_rate: float = 0.9999
) -> eqx.Module:
    """
    Updates an Exponential Moving Average (EMA) model based on the current model parameters.

    Args:
        ema_model: The current EMA model (`eqx.Module`).
        model: The current model (`eqx.Module`) whose parameters are used to update the EMA model.
        ema_rate: The decay rate for the EMA. Defaults to `0.9999`.

    Returns:
        The updated EMA model.

    Notes:
        - EMA is computed using the formula:
          `p_ema = p_ema * ema_rate + p * (1 - ema_rate)`, 
          where `p_ema` are the EMA parameters, and `p` are the model parameters.
        - Inexact arrays are updated while preserving non-trainable state.
    """
    ema_fn = lambda p_ema, p: p_ema * ema_rate + p * (1. - ema_rate)
    m_, _m = eqx.partition(model, eqx.is_inexact_array)
    e_, _e = eqx.partition(ema_model, eqx.is_inexact_array)
    e_ = jtu.tree_map(ema_fn, e_, m_)
    return eqx.combine(e_, _m)


def clip_grad_norm(grads: PyTree, max_norm: float) -> PyTree:
    """
    Clips the gradient norm of a PyTree of gradients to a specified maximum.

    Args:
        grads: A PyTree containing the gradients to be clipped.
        max_norm: The maximum allowable norm for the gradients.

    Returns:
        A PyTree of gradients where the norm is clipped to `max_norm` if it exceeds the limit.

    Notes:
        - Uses L2 norm for clipping.
        - Preserves the relative proportions of gradients while scaling them.
        - Avoids division by zero by adding a small epsilon (`1e-6`) to the denominator.
    """
    norm = jnp.linalg.norm(
        jax.tree.leaves(
            jax.tree.map(jnp.linalg.norm, grads)
        )
    )
    factor = jnp.minimum(max_norm, max_norm / (norm + 1e-6))
    return jax.tree.map(lambda x: x * factor, grads)


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def make_step(
    nde: eqx.Module, 
    x: Float[Array, "b x"], 
    y: Float[Array, "b y"], 
    opt_state: PyTree,
    opt: Optimiser,
    key: Key[jnp.ndarray, "..."],
    *,
    clip_max_norm: Optional[float] = None,
    replicated_sharding: Optional[PositionalSharding] = None,
) -> Tuple[eqx.Module, PyTree, Float[Array, ""]]:
    """
    Performs a single optimization step for a neural density estimator.

    Args:
        nde: The neural density estimator model (`eqx.Module`) being optimized.
        x: The input data of shape `(b, x)`.
        y: The target data of shape `(b, y)`.
        opt_state: The optimizer state (`PyTree`) used to compute parameter updates.
        opt: The optimizer object (`Optimiser`) for computing updates.
        key: A JAX random key for stochastic operations.
        clip_max_norm: An optional float specifying the maximum norm for gradient clipping. Defaults to `None`.
        replicated_sharding: An optional `PositionalSharding` object for distributing computations across devices. Defaults to `None`.

    Returns:
        A tuple containing:
            - The updated model (`eqx.Module`).
            - The updated optimizer state (`PyTree`).
            - The loss value (`Float[Array, ""]`).

    Notes:
        - The function computes the loss and its gradients using `batch_loss_fn`.
        - If `clip_max_norm` is specified, gradient clipping is applied.
        - Supports distributed computations using sharding for model parameters and optimizer states.
    """
    _fn = eqx.filter_value_and_grad(batch_loss_fn)
    if replicated_sharding is not None:
        nde, opt_state = eqx.filter_shard(
            (nde, opt_state), replicated_sharding
        )
    L, grads = _fn(nde, x, y, key=key)
    if clip_max_norm is not None:
        grads = clip_grad_norm(grads, clip_max_norm)
    updates, opt_state = opt.update(grads, opt_state, nde)
    nde = eqx.apply_updates(nde, updates)
    if replicated_sharding is not None:
        nde, opt_state = eqx.filter_shard(
            (nde, opt_state), replicated_sharding
        )
    return nde, opt_state, L 


def count_params(nde: eqx.Module) -> int:
    return sum(x.size for x in jtu.tree_leaves(nde) if eqx.is_array(x))


def get_n_split_keys(key: Key, n: int) -> Tuple[Key, PRNGKeyArray]:
    key, *keys = jr.split(key, n + 1)
    return key, jnp.asarray(keys)


@jaxtyped(typechecker=typechecker)
def partition_and_preprocess_data(
    key: Key[jnp.ndarray, "..."],
    train_data: Tuple[Float[Array, "n x"], Float[Array, "n y"]], 
    valid_fraction: float, 
    n_batch: int, 
) -> Tuple[
    Tuple[Float[Array, "nt x"], Float[Array, "nt y"]], 
    Tuple[Float[Array, "nv x"], Float[Array, "nv y"]], 
    Tuple[int, int]
]:
    """
    Partitions the dataset into training and validation sets, and computes the number of batches.

    Args:
        key: A JAX random key for shuffling the data.
        train_data: A tuple containing:
            - Simulation data of shape `(n, x)`.
            - Parameter data of shape `(n, y)`.
        valid_fraction: The fraction of the dataset to be used for validation (`float`).
        n_batch: The number of samples per batch (`int`).

    Returns:
        A tuple containing:
            - Training data as a tuple of simulation and parameter arrays, 
                with shapes `(nt, x)` and `(nt, y)` respectively.
            - Validation data as a tuple of simulation and parameter arrays, 
                with shapes `(nv, x)` and `(nv, y)` respectively.
            - A tuple with the number of training and validation batches (`int, int`).

    Notes:
        - The dataset is shuffled using the provided random key before partitioning.
        - The training and validation sets are determined based on the `valid_fraction` parameter.
        - If `n_batch` is not `None`, the number of batches is calculated by dividing 
            the number of samples in each set by the batch size, with a minimum of one batch.
        - If `n_batch` is `None`, the number of batches is returned as `None`.

    Example:
        ```python
        import jax.random as jr
        from partitioning import partition_and_preprocess_data

        key = jr.PRNGKey(0)
        simulations = jnp.ones((100, 10))  # 100 samples, 10 features
        parameters = jnp.ones((100, 5))   # 100 samples, 5 parameters
        train_data = (simulations, parameters)
        valid_fraction = 0.2
        n_batch = 10

        train_set, valid_set, batch_counts = partition_and_preprocess_data(
            key, train_data, valid_fraction, n_batch
        )
        ```
    """
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


@jaxtyped(typechecker=typechecker)
def get_loaders(
    key: Key[jnp.ndarray, "..."],
    data_train: Tuple[Float[Array, "nt x"], Float[Array, "nt y"]], 
    data_valid: Tuple[Float[Array, "nv x"], Float[Array, "nv y"]], 
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


@jaxtyped(typechecker=typechecker)
def train_nde(
    key: Key[jnp.ndarray, "..."], 
    # NDE
    model: eqx.Module,
    train_mode: str,
    # Data
    train_data: Tuple[Float[Array, "n x"], Float[Array, "n y"]],
    test_data: Optional[Tuple[Float[Array, "n x"], Float[Array, "n y"]]] = None,
    # Hyperparameters 
    opt: Optimiser = optax.adam(1e-3),
    valid_fraction: float = 0.1,
    n_epochs: int = 100_000,
    n_batch: int = 100,
    patience: int = 50,
    clip_max_norm: Optional[float] = None,
    # Sharding
    sharding: Optional[NamedSharding] = None,
    replicated_sharding: Optional[PositionalSharding] = None,
    # Optuna
    trial: Optional[optuna.trial.Trial] = None,
    # Saving
    results_dir: Optional[str] = None,
    # Progress bar
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
            - A dictionary containing training metrics such as loss values, best loss, and the best epoch.
            
    Key steps:
        1. Partition and preprocesses the training data into training and validation sets.
        2. Train the model for `n_epochs` using the specified optimizer and training data.
        3. Track training and validation losses and applies early stopping based on validation loss.
        4. Optionally apply gradient clipping and Optuna trial pruning.
        5. Plots and saves the training and validation losses, and saves the best-performing model.
        6. Returns the trained model and relevant training metrics.

    Raises:
        optuna.exceptions.TrialPruned: If the Optuna trial is pruned based on validation loss.
    """
    if results_dir is not None:
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
    print("NDE has n_params={}.".format(n_params))

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
                model, 
                xy.x, 
                xy.y, 
                opt_state, 
                opt, 
                key, 
                clip_max_norm=clip_max_norm, 
                replicated_sharding=replicated_sharding
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

            valid_loss = batch_eval_fn(
                model, xy.x, xy.y, key=key, replicated_sharding=replicated_sharding
            )

            epoch_valid_loss += valid_loss

        stats["valid_losses"].append(epoch_valid_loss / (s + 1))

        if show_tqdm:
            epochs.set_postfix_str(
                "t={:.3E} | v={:.3E} | v(best)={:.3E} | stop={:04d}".format(
                    stats["train_losses"][-1],
                    stats["valid_losses"][-1],
                    stats["best_loss"],
                    (patience - stats["stopping_count"] if patience is not None else 0)
                ),
                refresh=True
            )

        # Break training for any broken NDEs
        if not jnp.isfinite(stats["valid_losses"][-1]) or not jnp.isfinite(stats["train_losses"][-1]):
            if show_tqdm:
                epochs.set_description_str(
                    "\nTraining terminated early at epoch {} (NaN loss).".format(epoch + 1), 
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
                            "Training terminated early at epoch {}; valid={:.3E}, train={:.3E}.".format(
                                epoch + 1, stats["valid_losses"][-1], stats["train_losses"][-1]
                            ), 
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
        label="Best loss {:.3E}".format(stats["best_loss"]),
        linestyle=""
    )
    plt.legend()
    if results_dir is not None:
        plt.savefig(os.path.join(results_dir, "losses.png"))
    plt.close()

    # Save NDE model
    if results_dir is not None:
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
    for nde in ensemble.ndes:
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
    train_data: Tuple[Float[Array, "n x"], Float[Array, "n y"]],
    test_data: Optional[Tuple[Float[Array, "nt x"], Float[Array, "nt y"]]] = None,
    # Hyperparameters 
    opt: Optimiser = optax.adam(1e-3),
    valid_fraction: float = 0.1,
    n_epochs: int = 100_000,
    n_batch: int = 100,
    patience: int = 50,
    clip_max_norm: Optional[float] = None,
    # Sharding
    sharding: Optional[NamedSharding] = None,
    replicated_sharding: Optional[NamedSharding] = None,
    # Optuna
    trial: Optional[optuna.trial.Trial] = None,
    # Saving
    results_dir: Optional[str] = None,
    # Progress bar
    tqdm_description: str = "Training",
    show_tqdm: bool = True,
) -> Tuple[Ensemble, dict]:
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
        `trial` (optuna.trial.Trial, optional): Optuna trial for hyperparameter optimization. Can be used to prune unpromising runs. Defaults to `None`.
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
    if trial is not None:
        assert len(ensemble.ndes) == 1, (
            "Can only optimise hyperparameters for single NDE ensembles."
        )

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
            replicated_sharding=replicated_sharding,
            trial=trial,
            results_dir=results_dir,
            tqdm_description=tqdm_description,
            show_tqdm=show_tqdm
        )

        ensemble.ndes[n] = nde

        stats.append(stats_n)
        ndes.append(nde)

    # Calculate weights of NDEs in ensemble (higher log-likelihood is better)
    weights = ensemble.calculate_stacking_weights(
        losses=[
            stats[n]["all_valid_loss"] for n, _ in enumerate(ensemble.ndes)
        ]
    )
    ensemble = replace(ensemble, weights=weights)

    print("Weights:", ensemble.weights)

    return ensemble, stats