import time
import os
from typing import Tuple, Optional, Sequence, Callable, Any, Literal
from functools import partial
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.sharding import NamedSharding, PositionalSharding
import equinox as eqx
from optimistix import minimise, BFGS, LevenbergMarquardt, rms_norm
import optax
from jaxtyping import Key, Array, Float, Scalar, PRNGKeyArray, jaxtyped
from beartype import beartype as typechecker
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
import numpy as np 
from tqdm.auto import trange

from configs.log import setup_module_logger, get_log_level

DATA_PROCESS_TYPE_NN = os.environ.get("DATA_PROCESS_TYPE_NN", None) # Inputs/outputs to compression NN
USE_PRECISION_NN = True if os.environ.get("USE_PRECISION_NN", "").lower() in ("1", "true") else False

assert DATA_PROCESS_TYPE_NN in ["d", "p", "dp", None], (
    "DATA_PROCESS_TYPE_NN={}".format(DATA_PROCESS_TYPE_NN)
)

if USE_PRECISION_NN:
    assert DATA_PROCESS_TYPE_NN in ["d", None], (
        "CANNOT USE DATA_PROCESS_TYPE_NN={} with USE_PRECISION_NN=True".format(DATA_PROCESS_TYPE_NN)
    )

"""
    Tools for compression with neural networks.
    - train a user-defined `eqx.Module` network that compresses a datavector
    to a model-dimensional summary, by minimising a MSE loss.
"""

typecheck = jaxtyped(typechecker=typechecker)

logger, log_figs_dir = setup_module_logger(__name__, level=get_log_level())


def exists(v):
    return v is not None


def loss(
    model: eqx.Module | eqx.nn.MLP, 
    x: Float[Array, "b x"], 
    y: Float[Array, "b y"], 
    key: PRNGKeyArray,
    *,
    precision: Optional[Float[Array, "y y"]] = None
) -> Scalar:

    if precision is None:
        precision = jnp.eye(y.shape[-1])

    def fn(x, y, key):
        y_ = model(x, key=key)

        dy = jnp.subtract(y_, y)

        l = jnp.linalg.multi_dot([dy, precision, dy.T])

        return l

    keys = jr.split(key, len(x))

    return jnp.mean(jax.vmap(fn)(x, y, keys))


def get_batch(
    D: Float[Array, "n x"], 
    Y: Float[Array, "n y"], 
    n: Optional[int], 
    key: Optional[PRNGKeyArray]
) -> Tuple[Float[Array, "b x"], Float[Array, "b y"]]:
    if n is not None and key is not None:
        idx = jr.choice(key, jnp.arange(D.shape[0]), (n,))
        return D[idx], Y[idx]
    else:
        return D, Y


@typecheck
@eqx.filter_jit
def evaluate(
    model: eqx.Module, 
    Xv: Float[Array, "n x"], 
    Yv: Float[Array, "n y"],
    key: PRNGKeyArray,
    *, 
    n_batch: Optional[int],
    precision: Optional[Float[Array, "y y"]] = None,
    sharding: Optional[jax.sharding.NamedSharding] = None,
    replicated_sharding: Optional[PositionalSharding] = None
) -> Scalar:

    model = eqx.nn.inference_mode(model, True)

    x, y = get_batch(Xv, Yv, n=n_batch, key=key)

    if sharding is not None:
        x, y = eqx.filter_shard((x, y), sharding)

    if replicated_sharding is not None:
        model = eqx.filter_shard(model, replicated_sharding)

    l = loss(model, x, y, key=key, precision=precision)

    return l


@typecheck
@eqx.filter_jit
def make_step(
    model: eqx.Module, 
    opt_state: optax.OptState,
    Xt: Float[Array, "n x"], 
    Yt: Float[Array, "n y"],
    key: PRNGKeyArray,
    opt: optax.GradientTransformation, 
    *, 
    n_batch: Optional[int] = None,
    precision: Optional[Float[Array, "y y"]] = None,
    sharding: Optional[jax.sharding.NamedSharding],
    replicated_sharding: Optional[PositionalSharding]
) -> Tuple[eqx.Module, optax.OptState, Scalar]:

    model = eqx.nn.inference_mode(model, False)

    x, y = get_batch(Xt, Yt, n=n_batch, key=key) 
    
    if sharding is not None:
        x, y = eqx.filter_shard((x, y), sharding)

    if replicated_sharding is not None:
        model, opt_state = eqx.filter_shard(
            (model, opt_state), replicated_sharding
        )

    grad_fn = eqx.filter_value_and_grad(partial(loss, precision=precision))

    loss_value, grads = grad_fn(model, x, y, key)

    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    if replicated_sharding is not None:
        model, opt_state = eqx.filter_shard(
            (model, opt_state), replicated_sharding
        )

    return model, opt_state, loss_value


@typecheck
def fit_nn(
    key: PRNGKeyArray,
    model: eqx.Module, 
    train_data: Tuple[Float[Array, "n x"], Float[Array, "n y"]], 
    opt: optax.GradientTransformation, 
    n_batch: Optional[int], 
    patience: Optional[int], 
    n_steps: int = 100_000, 
    valid_fraction: float = 0.1, 
    valid_data: Optional[Sequence[Array]] = None,
    use_tqdm: bool = True,
    *,
    description: str = "Training NN",
    precision: Optional[Float[Array, "y y"]] = None,
    sharding: Optional[NamedSharding] = None,
    replicated_sharding: Optional[PositionalSharding] = None,
) -> Tuple[eqx.Module, Float[np.ndarray, "l 2"]]:
    """
    Trains a neural network model with early stopping.

    Args:
        key: A `PRNGKeyArray`.
        model: The neural network model to be trained, represented as an `eqx.Module`.
        D: The input data matrix (`Array`), where rows are data points and columns are features.
        Y: The target values (`Array`) corresponding to the input data.
        opt: The optimizer to be used for gradient updates, defined as an `optax.GradientTransformation`.
        n_batch: The number of data points per mini-batch for each training step (`int`).
        patience: The number of steps to continue without improvement on the validation loss 
            before early stopping is triggered (`int`).
        n_steps: The maximum number of training steps to perform (`int`, optional). Default is 100,000.
        valid_fraction: The fraction of the data to use for training, with the remainder
            used for validation (`float`, optional). Default is 0.9 (90% training, 10% validation).

    Returns:
        Tuple[`eqx.Module`, `Array`]: 
            - The trained `model` after the optimization process.
            - A 2D array of shape (n_steps, 2), where the first column contains the training loss at each 
            step, and the second column contains the validation loss.
    
    Notes:
        1. The data `D` and targets `Y` are split into training and validation sets based on the 
        `valid_fraction` parameter.
        4. Early stopping occurs if the validation loss does not improve within a specified 
        number of steps (`patience`).
        5. The function returns the trained model and the recorded training/validation loss history.
    """

    if exists(precision) and USE_PRECISION_NN:
        logger.info("Using Fisher precision for NN.")

    D, Y = train_data

    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    if valid_data is not None:
        Xt, Yt = train_data
        Xv, Yv = valid_data
    else:
        n_s, _ = D.shape
        Xt, Xv = jnp.split(D, [n_s - int(valid_fraction * n_s)]) 
        Yt, Yv = jnp.split(Y, [n_s - int(valid_fraction * n_s)])

    if use_tqdm: 
        steps = trange(n_steps, desc=description, colour="magenta")
    else: 
        steps = range(n_steps)

    L = np.zeros((n_steps, 2))
    for step in steps:

        key_t, key_v = jr.split(jr.fold_in(key, step))

        model, opt_state, train_loss = make_step(
            model, 
            opt_state, 
            Xt, 
            Yt, 
            key_t,
            opt=opt, 
            n_batch=n_batch,
            precision=precision, 
            sharding=sharding,
            replicated_sharding=replicated_sharding
        )

        valid_loss = evaluate(
            model, 
            Xv, 
            Yv, 
            key_v, 
            n_batch=n_batch,
            precision=precision, 
            sharding=sharding,
            replicated_sharding=replicated_sharding
        )

        L[step] = train_loss, valid_loss

        if use_tqdm:
            if step > 0:
                steps.set_description_str(
                    "t={:.3E}, v={:.3E}, p={:04d}".format(
                        train_loss.item(), 
                        valid_loss.item(), 
                        int(step - np.argmin(L[:step, 1]))
                    )
                )

        if patience is not None:
            if (step > 0) and (step - np.argmin(L[:step, 1]) > patience):
                if use_tqdm:
                    steps.set_description_str("Stopped at {}".format(step))
                break

    return model, L[:step]


"""
    L-BFGS
"""

@typecheck
@eqx.filter_jit(donate="all-except-first")
def make_step_lbfgs(
    net: eqx.Module, 
    opt_state: optax.OptState, 
    X: Float[Array, "n d"], 
    P: Float[Array, "n p"], 
    *,
    opt: optax.GradientTransformation,
    precision: Optional[Float[Array, "p p"]] = None, 
    replicated_sharding: Optional[jax.sharding.NamedSharding] = None
) -> tuple[eqx.Module, optax.OptState, Scalar]:
    f = partial(loss, x=X, y=P, precision=precision)
    value_and_grad_fn = optax.value_and_grad_from_state(f)
    l, grad = value_and_grad_fn(net, state=opt_state)
    updates, opt_state = opt.update(
        grad, opt_state, net, value=l, grad=grad, value_fn=f 
    )
    net = eqx.apply_updates(net, updates)
    return net, opt_state, l 


def fit_nn_lbfgs(
    key: Key[jnp.ndarray, "..."], 
    model: eqx.Module, 
    train_data: Tuple[Float[Array, "n x"], Float[Array, "n y"]], 
    valid_fraction: float = 0.1, 
    valid_data: Sequence[Array] = None,
    batch_dataset: bool = True,
    *,
    precision: Optional[Float[Array, "y y"]] = None,
    sharding: Optional[NamedSharding] = None,
    replicated_sharding: Optional[PositionalSharding] = None,
) -> Tuple[eqx.Module, Float[np.ndarray, "l 2"]]:

    D, Y = train_data

    y0, static = eqx.partition(model, eqx.is_array)
    
    # Standardise before PCA (don't get tricked by high variance due to units)
    # X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0) # NOTE: already standardised 

    # Fit whitening-PCA to compressed simulations
    # pca = PCA(num_components=D.shape[-1]) 
    # D = (D - jnp.mean(D, axis=0)) / jnp.std(D, axis=0)
    # pca.fit(D) # Fit on fiducial data?
    # D = pca.transform(D)

    # D, preprocess_fn = get_preprocess_fn(D, use_pca=config.use_pca)

    @eqx.filter_jit
    def f(y, args):
        # Dataset to fit network to
        D, Y = args
        # Combine iteration parameters and architecture
        model = eqx.combine(y, static)
        return loss(model, D, Y, precision=precision)

    # BFGS optimisation
    res = minimise(
        f,
        BFGS(rtol=1e-6, atol=1e-6, norm=rms_norm),
        y0=y0,
        max_steps=1_000_000,
        args=(D, Y)
    )

    # Put solution parameters into model
    model = eqx.combine(res.value, static)

    L = np.zeros((1, 2))

    # model = lambda d: pca.transform(model(d))

    return model, L


# class ExtraMLP(eqx.nn.MLP):
#     dropouts: list[eqx.nn.Dropout]
#     layernorms: list[eqx.nn.LayerNorm]

#     def __init__(self, *args, p: float, key: PRNGKeyArray, **kwargs):
#         super().__init__(*args, **kwargs, key=key)
#         dropouts = []
#         layernorms = []
#         for layer in self.layers[:-1]:
#             dropouts.append(eqx.nn.Dropout(p=p))
#             layernorms.append(eqx.nn.LayerNorm(layer.weight.shape[0]))
#         self.dropouts = dropouts + [eqx.nn.Identity()] # Hacky safe-zip
#         self.layernorms = layernorms + [eqx.nn.Identity()] # Hacky safe-zip

#     def __call__(self, x, key=None):
#         for i, (layer, dropout, norm) in enumerate(
#             zip(self.layers, self.dropouts, self.layernorms)
#         ):
#             x = layer(x)
#             x = norm(x)
#             x = dropout(x, key=key)
#             if i != len(self.layers) - 1:
#                 x = self.activation(x) # Don't activate last layer
#         if self.final_activation is not None:
#             x = self.final_activation(x) # ...unless it is required
#         return x


class ExtraMLP(eqx.Module):
    layers: list[eqx.nn.Linear]
    dropouts: list[eqx.nn.Dropout | eqx.nn.Identity]
    layernorms: list[eqx.nn.LayerNorm | eqx.nn.Identity]
    activation: Callable
    final_activation: Optional[Callable]
    p: float

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_sizes: Sequence[int],
        *,
        p: float = 0.0,
        activation: Callable = jax.nn.gelu,
        final_activation: Optional[Callable] = None,
        key: PRNGKeyArray,
        use_bias: bool = True,
    ):
        """Arbitrary-width MLP with per-layer LayerNorm + Dropout (optional).

        Args:
            in_size: input feature dimension
            out_size: output feature dimension
            hidden_sizes: list of hidden layer widths, e.g. [128, 64, 32]
            p: dropout probability (applied to each layer output; 0 disables)
            activation: activation between hidden layers (not after last)
            final_activation: optional activation after final layer
            key: PRNG key
            use_bias: whether Linear layers include bias
        """
        self.p = float(p)
        self.activation = activation
        self.final_activation = final_activation

        # Build Linear stack according to sizes
        if hidden_sizes == [0] or len(list(hidden_sizes)) == 0:
            sizes = [in_size, out_size]
        else:
            sizes = [in_size, *hidden_sizes, out_size]
        n_layers = len(sizes) - 1
        k_lin = jr.split(key, n_layers)

        layers = []
        for i in range(n_layers):
            layers.append(
                eqx.nn.Linear(
                    sizes[i], sizes[i + 1], use_bias=use_bias, key=k_lin[i]
                )
            )
        self.layers = layers

        # Per-layer LayerNorms on the OUTPUT dimension of each layer
        lns = []
        for lyr in self.layers[:-1]:
            out_dim = lyr.weight.shape[0]  # (out, in)
            lns.append(eqx.nn.LayerNorm(out_dim))
        lns.append(eqx.nn.Identity())  # no LN on final layer output (can change if you want)
        self.layernorms = lns

        # Per-layer Dropouts mirroring LayerNorms
        drops = []
        if self.p > 0.0:
            for _ in self.layers[:-1]:
                drops.append(eqx.nn.Dropout(p=self.p))
            drops.append(eqx.nn.Identity())  # no dropout on final layer output
        else:
            drops = [eqx.nn.Identity() for _ in self.layers]
        self.dropouts = drops

    def __call__(self, x, *, key: Optional[PRNGKeyArray] = None):
        # If we have real Dropout modules and a key, split it so masks differ per layer
        if self.p > 0.0 and key is not None:
            k_list = list(jr.split(key, len(self.layers)))
        else:
            k_list = [None] * len(self.layers)

        for i, (layer, norm, drop, k_i) in enumerate(zip(self.layers, self.layernorms, self.dropouts, k_list)):
            x = layer(x)
            x = norm(x)
            x = drop(x, key=k_i)
            if i != len(self.layers) - 1:
                x = self.activation(x)  # no hidden activation after last layer

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x


# def get_ensemble_fn(config, dataset):
#     @eqx.filter_vmap
#     def make_ensemble(key):
#         return eqx.nn.MLP(
#         # return ExtraMLP(
#             dataset.data.shape[-1], 
#             dataset.parameters.shape[-1], 
#             width_size=config.nn.width_size, 
#             depth=config.nn.depth, 
#             use_final_bias=config.nn.use_final_bias,
#             # p=0.3,
#             activation=getattr(jax.nn, config.nn.activation),
#             key=key
#         )
#     return make_ensemble


@eqx.filter_vmap(in_axes=(eqx.if_array(0), None, None))
def evaluate_ensemble(model, x, key=None):
    return model(x, key=key)


class Ensemble(eqx.Module):
    ensemble: eqx.Module

    def __init__(self, ensemble):
        self.ensemble = ensemble

    def __call__(self, x, key=None):
        x = evaluate_ensemble(self.ensemble, x, key)
        return jnp.mean(x, axis=0)


def get_nn_process_fns(
    dataset, # DatasetClass not Dataset dataclass 
    data_process_type_nn: Literal["d", "p", "dp"] | None
) -> tuple[
    Callable[[Float[Array, "..."]], Float[Array, "..."]], 
    Callable[[Float[Array, "..."]], Float[Array, "..."]], 
    Callable[[Float[Array, "..."]], Float[Array, "..."]], 
]:
    assert data_process_type_nn in ["d", "p", "dp", None], (
        "data_process_type_nn={}".format(data_process_type_nn)
    )

    mu_D = dataset.data.mean(axis=0) #jnp.mean(dataset.fiducial_data, axis=0)
    std_D = dataset.data.std(axis=0) #jnp.std(dataset.fiducial_data, axis=0)

    mu_p = dataset.parameters.mean(axis=0)
    std_p = dataset.parameters.std(axis=0)

    def _preprocess_fn_d(d):
        return (d - mu_D) / std_D

    def _preprocess_fn_p(p):
        return (p - mu_p) / std_p

    def _postprocess_fn_p(p):
        return p * std_p + mu_p

    if data_process_type_nn == None:
        preprocess_fn_d = preprocess_fn_p = postprocess_fn_p = lambda x: x

    if data_process_type_nn == "d":
        preprocess_fn_p = postprocess_fn_p = lambda x: x
        preprocess_fn_d = _preprocess_fn_d
    
    if data_process_type_nn == "dp":
        preprocess_fn_p = _preprocess_fn_p
        postprocess_fn_p = _postprocess_fn_p
        preprocess_fn_d = _preprocess_fn_d

    return preprocess_fn_d, preprocess_fn_p, postprocess_fn_p


@typecheck
def get_nn_compressor(
    key: PRNGKeyArray, 
    net: eqx.nn.MLP | eqx.Module,
    config: ConfigDict,
    dataset: Any, #Dataset, 
    *, 
    train: bool = True,
    lbfgs: bool = False, 
    results_dir: str, 
    # net: Optional[eqx.Module] = None
) -> tuple[
    eqx.Module, 
    Callable[[Float[Array, "d"]], Float[Array, "d"]], 
    Callable[[Float[Array, "p"]], Float[Array, "p"]]
]:
    """
        Train neural network compression function
        - Optionally use parameter covariance for chi2 loss
    """

    description = "Fitting NN [{}]".format(dataset.name)

    key = jr.key(int(time.time()))
    net_key, train_key = jr.split(key)

    if USE_PRECISION_NN:
        precision = jnp.linalg.inv(dataset.Finv) # In reality this varies with parameters

        assert DATA_PROCESS_TYPE_NN not in ["dp", "p"], (
            "CANNOT USE DATA_PROCESS_TYPE_NN={} with USE_PRECISION_NN=True".format(DATA_PROCESS_TYPE_NN)
        )

        print("USING PRECISION FOR NN.")
    else:
        precision = None 

        print("NOT USING PRECISION FOR NN.")

    print("D, Y:", dataset.data.shape, dataset.parameters.shape)

    preprocess_fn_d, preprocess_fn_p, postprocess_fn_p = get_nn_process_fns(
        dataset, data_process_type_nn=DATA_PROCESS_TYPE_NN
    )

    train_data = (preprocess_fn_d(dataset.data), preprocess_fn_p(dataset.parameters))

    nn_path = os.path.join(results_dir, "nn.eqx")

    # Train or reload the neural network
    if train:
        if lbfgs:
            net, losses = fit_nn_lbfgs(
                train_key, 
                net, 
                train_data=train_data,
                precision=precision
            )
        else:
            opt = getattr(optax, config.nn.train.opt)(config.nn.train.lr)

            net, losses = fit_nn(
                train_key, 
                net, 
                opt=opt, 
                train_data=train_data,
                precision=precision, 
                n_batch=config.nn.train.n_batch,
                n_steps=config.nn.train.n_steps,
                patience=config.nn.train.patience,
                valid_fraction=config.nn.train.valid_fraction,
                description=description
            )

        plt.figure()
        plt.loglog(losses, color="red" if dataset.name == "tails" else "blue")
        plt.savefig(os.path.join(results_dir, "losses_nn.png"))
        plt.close()

        net = eqx.nn.inference_mode(net, True)

        eqx.tree_serialise_leaves(nn_path, net)
    else:
        net = eqx.tree_deserialise_leaves(nn_path, net)

        net = eqx.nn.inference_mode(net, True)

    # b = dataset.alpha - jnp.mean(jax.vmap(net)(preprocess_fn_d(dataset.fiducial_data)), axis=0)

    # postprocess_fn_p = lambda p: p - b

    return net, preprocess_fn_d, postprocess_fn_p