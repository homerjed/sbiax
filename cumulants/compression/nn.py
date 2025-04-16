from typing import Tuple, Optional, Sequence
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.sharding import NamedSharding, PositionalSharding
import equinox as eqx
from optimistix import minimise, BFGS, LevenbergMarquardt, rms_norm
from jaxtyping import Key, Array, Float, Scalar, jaxtyped
from beartype import beartype as typechecker
import optax
import numpy as np 
from tqdm.auto import trange

from .pca import PCA

"""
    Tools for compression with neural networks.
    - train a user-defined `eqx.Module` network that compresses a datavector
      to a model-dimensional summary, by minimising a MSE loss.
"""

typecheck = jaxtyped(typechecker=typechecker)


def loss(
    model: eqx.Module, 
    x: Float[Array, "b x"], 
    y: Float[Array, "b y"], 
    *,
    precision: Optional[Float[Array, "y y"]] = None
) -> Scalar:
    def fn(x, y):
        y_ = model(x)
        dy = jnp.subtract(y_, y)
        if precision is not None:
            l = jnp.linalg.multi_dot([dy, precision, dy.T]) # NOTE: which transpose?!
        else:
            l = jnp.square(dy)
        return l
    return jnp.mean(jax.vmap(fn)(x, y))


@eqx.filter_jit
def evaluate(
    model: eqx.Module, 
    x: Float[Array, "b x"], 
    y: Float[Array, "b y"],
    *, 
    precision: Optional[Float[Array, "y y"]] = None,
    replicated_sharding: Optional[PositionalSharding] = None
) -> Scalar:
    if replicated_sharding is not None:
        model = eqx.filter_shard(model, replicated_sharding)
    return loss(model, x, y, precision=precision)


@typecheck
@eqx.filter_jit
def make_step(
    model: eqx.Module, 
    opt_state: optax.OptState,
    x: Float[Array, "b x"], 
    y: Float[Array, "b y"],
    opt: optax.GradientTransformation, 
    *, 
    precision: Optional[Float[Array, "y y"]] = None,
    replicated_sharding: Optional[PositionalSharding]
) -> Tuple[eqx.Module, optax.OptState, Scalar]:

    if replicated_sharding is not None:
        model, opt_state = eqx.filter_shard(
            (model, opt_state), replicated_sharding
        )

    grad_fn = eqx.filter_value_and_grad(partial(loss, precision=precision))

    loss_value, grads = grad_fn(model, x, y)

    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    if replicated_sharding is not None:
        model, opt_state = eqx.filter_shard(
            (model, opt_state), replicated_sharding
        )

    return model, opt_state, loss_value


def get_batch(
    D: Float[Array, "n x"], 
    Y: Float[Array, "n y"], 
    n: int, 
    key: Key
) -> Tuple[Float[Array, "b x"], Float[Array, "b y"]]:
    idx = jr.choice(key, jnp.arange(D.shape[0]), (n,))
    return D[idx], Y[idx]


@typecheck
def fit_nn(
    key: Key[jnp.ndarray, "..."], 
    model: eqx.Module, 
    train_data: Tuple[Float[Array, "n x"], Float[Array, "n y"]], 
    opt: optax.GradientTransformation, 
    n_batch: int, 
    patience: Optional[int], 
    n_steps: int = 10_000, 
    valid_fraction: int = 0.9, 
    valid_data: Sequence[Array] = None,
    batch_dataset: bool = True,
    use_tqdm: bool = False,
    *,
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
    D, Y = train_data

    n_s, _ = D.shape

    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    if valid_data is not None:
        Xt, Yt = train_data
        Xv, Yv = valid_data
    else:
        Xt, Xv = jnp.split(D, [int(valid_fraction * n_s)]) 
        Yt, Yv = jnp.split(Y, [int(valid_fraction * n_s)])

    if use_tqdm: 
        steps = trange(n_steps, desc="Training NN", colour="blue")
    else: 
        steps = trange(n_steps)

    L = np.zeros((n_steps, 2))
    for step in steps:
        key_t, key_v = jr.split(jr.fold_in(key, step))

        if batch_dataset:
            x, y = get_batch(Xt, Yt, n=n_batch, key=key_t) # Xt, Yt
        else:
            x, y = Xt, Yt
        
        if sharding is not None:
            x, y = eqx.filter_shard((x, y), sharding)

        model, opt_state, train_loss = make_step(
            model, 
            opt_state, 
            x, 
            y, 
            opt=opt, 
            precision=precision, 
            replicated_sharding=replicated_sharding
        )

        if batch_dataset:
            x, y = get_batch(Xv, Yv, n=n_batch, key=key_v)
        else:
            x, y = Xv, Yv

        if sharding is not None:
            x, y = eqx.filter_shard((x, y), sharding)

        valid_loss = evaluate(
            model, x, y, precision=precision, replicated_sharding=replicated_sharding
        )

        L[step] = train_loss, valid_loss
        steps.set_postfix_str(
            "train={:.3E}, valid={:.3E}".format(train_loss.item(), valid_loss.item())
        )

        if patience is not None:
            if (step > 0) and (step - np.argmin(L[:step, 1]) > patience):
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

# @typecheck
# def fit_nn_lbfgs(
#     key: Key[jnp.ndarray, "..."], 
#     model: eqx.Module, 
#     train_data: Tuple[Float[Array, "n x"], Float[Array, "n y"]], 
#     n_batch: int, 
#     patience: Optional[int], 
#     n_steps: int = 10_000, 
#     valid_fraction: int = 0.9, 
#     valid_data: Sequence[Array] = None,
#     batch_dataset: bool = True,
#     *,
#     precision: Optional[Float[Array, "y y"]] = None,
#     sharding: Optional[NamedSharding] = None,
#     replicated_sharding: Optional[PositionalSharding] = None,
# ) -> Tuple[eqx.Module, Float[np.ndarray, "l 2"]]:

#     D, Y = train_data

#     n_s, _ = D.shape

#     opt = optax.lbfgs()

#     opt_state = opt.init(eqx.filter(model, eqx.is_array))

#     if valid_data is not None:
#         Xt, Yt = train_data
#         Xv, Yv = valid_data
#     else:
#         Xt, Xv = jnp.split(D, [int(valid_fraction * n_s)]) 
#         Yt, Yv = jnp.split(Y, [int(valid_fraction * n_s)])

#     L = np.zeros((n_steps, 2))
#     with trange(n_steps, desc="Training NN (L-BFGS)", colour="blue") as steps:
#         for step in steps:
#             key_t, key_v = jr.split(jr.fold_in(key, step))

#             if batch_dataset:
#                 x, y = get_batch(Xt, Yt, n=n_batch, key=key_t) # Xt, Yt
#             else:
#                 x, y = Xt, Yt
            
#             if sharding is not None:
#                 x, y = eqx.filter_shard((x, y), sharding)

#             model, opt_state, train_loss = make_step_lbfgs(
#                 model, 
#                 opt_state, 
#                 x, 
#                 y, 
#                 opt=opt, 
#                 precision=precision, 
#                 replicated_sharding=replicated_sharding
#             )

#             if batch_dataset:
#                 x, y = get_batch(Xv, Yv, n=n_batch, key=key_v)
#             else:
#                 x, y = Xv, Yv

#             if sharding is not None:
#                 x, y = eqx.filter_shard((x, y), sharding)

#             valid_loss = evaluate(
#                 model, x, y, precision=precision, replicated_sharding=replicated_sharding
#             )

#             L[step] = train_loss, valid_loss
#             steps.set_postfix_str(
#                 "train={:.3E}, valid={:.3E}".format(train_loss.item(), valid_loss.item())
#             )

#             if patience is not None:
#                 if (step > 0) and (step - np.argmin(L[:step, 1]) > patience):
#                     steps.set_description_str("Stopped at {}".format(step))
#                     break

#     return model, L[:step]


def fit_nn_lbfgs(
    key: Key[jnp.ndarray, "..."], 
    model: eqx.Module, 
    train_data: Tuple[Float[Array, "n x"], Float[Array, "n y"]], 
    valid_fraction: int = 0.9, 
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

    # _model = lambda d: pca.transform(model(d))

    return _model, L


if __name__ == "__main__":
    import time
    from configs import (
        cumulants_config, bulk_cumulants_config, 
        get_results_dir, get_posteriors_dir, 
        get_cumulants_sbi_args, get_ndes_from_config
    )
    from cumulants import (
        CumulantsDataset, Dataset, get_data, get_prior, 
        get_compression_fn, get_datavector, get_linearised_data
    )
    from pdfs import BulkCumulantsDataset, get_bulk_dataset
    from cumulants_ensemble import Ensemble
    from sbiax.train import train_ensemble
    from affine import affine_sample
    from utils import plot_moments, plot_latin_moments, plot_summaries, plot_fisher_summaries


    def get_dataset_and_config(bulk_or_tails):
        if bulk_or_tails == "bulk":
            dataset_constructor = BulkCumulantsDataset
            config = bulk_cumulants_config 
        if bulk_or_tails == "tails":
            dataset_constructor = CumulantsDataset
            config = cumulants_config 
        return dataset_constructor, config


    t0 = time.time()

    args = get_cumulants_sbi_args()

    print("TIME:", datetime.datetime.now().strftime("%H:%M %d-%m-%y"))
    print("SEED:", args.seed)
    print("MOMENTS:", args.order_idx)
    print("LINEARISED:", args.linearised)

    """
        Config
    """

    # Bulk / tails constructors for dataset / config
    _dataset, _config = get_dataset_and_config(args.bulk_or_tails) 

    config = _config(
        seed=args.seed, 
        redshift=args.redshift, 
        reduced_cumulants=args.reduced_cumulants,
        sbi_type=args.sbi_type,
        linearised=args.linearised, 
        compression=args.compression,
        order_idx=args.order_idx,
        n_linear_sims=args.n_linear_sims,
        pre_train=args.pre_train
    )

    key = jr.key(config.seed)

    ( 
        model_key, train_key, key_prior, 
        key_datavector, key_state, key_sample
    ) = jr.split(key, 6)

    results_dir = get_results_dir(config, args)

    posteriors_dir = get_posteriors_dir(config, args)

    # Dataset of simulations, parameters, covariance, ...
    cumulants_dataset = _dataset(config, results_dir=results_dir)

    dataset: Dataset = cumulants_dataset.data

    parameter_prior: Distribution = cumulants_dataset.prior

    bulk_pdfs = True # Use PDFs for Finv not cumulants
    bulk_dataset: Dataset = get_bulk_dataset(args, pdfs=bulk_pdfs) # For Fisher forecast comparisons

    print("DATA:", ["{:.3E} {:.3E}".format(_.min(), _.max()) for _ in (dataset.fiducial_data, dataset.data)])
    print("DATA:", [_.shape for _ in (dataset.fiducial_data, dataset.data)])

    """
        Compression
    """

    # Compress simulations
    compression_fn = cumulants_dataset.compression_fn

    X = jax.vmap(compression_fn)(dataset.data, dataset.parameters)

    # Plot summaries
    plot_summaries(X, dataset.parameters, dataset, results_dir)

    plot_moments(dataset.fiducial_data, config, results_dir)

    plot_latin_moments(dataset.data, config, results_dir)