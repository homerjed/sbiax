import os
from dataclasses import dataclass, replace
from functools import partial
from typing import Callable, Optional, Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray, Array, Float, Int, jaxtyped

import equinox as eqx
import optax
from beartype import beartype as typechecker 
from scipy.stats import qmc
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
import tensorflow_probability.substrates.jax.distributions as tfd

from configs.log import setup_module_logger, get_log_level
from data.constants import get_quijote_parameters, get_target_idx, get_F_planck, get_Finv_planck, LOWER, UPPER, ALPHA
# from data.common import get_nn_compressor

typecheck = jaxtyped(typechecker=typechecker)

logger, log_figs_dir = setup_module_logger(__name__, level=get_log_level())

FORCE_NOISELESS_DATAVECTOR = True if os.environ.get("FORCE_NOISELESS_DATAVECTOR", "").lower() in ("1", "true") else False
NON_GAUSSIAN_TEST = True if os.environ.get("NON_GAUSSIAN_TEST", "").lower() in ("1", "true") else False

"""
    Objects common to the PDF and cumulant datasets
"""

def exists(v):
    return v is not None


def add_planck_information_to_Finv(Finv, use_planck=False):
    # Add Fisher information from Planck to any Finv matrix
    if use_planck:
        F_planck = get_F_planck()
        Finv = jnp.linalg.inv(jnp.linalg.inv(Finv) + F_planck)
    return Finv


def hartlap(n_s: int, n_d: int) -> float: 
    return (n_s - n_d - 2) / (n_s - 1)


def get_parameter_strings() -> list[str]:
    (_, _, _, _, _, _, parameter_strings, *_) = get_quijote_parameters()
    return parameter_strings


@typecheck
@dataclass
class Dataset:
    name: Literal["bulk_pdf", "bulk", "tails"]
    alpha: Float[Array, "p"] 
    lower: Float[Array, "p"]
    upper: Float[Array, "p"]
    parameter_strings: list[str]
    Finv: Float[Array, "p p"]
    Cinv: Float[Array, "d d"]
    C: Float[Array, "d d"]
    fiducial_data: Float[Array, "nf d"]
    data: Float[Array, "nl d"]
    parameters: Float[Array, "nl p"]
    derivatives: Float[Array, "500 p d"]


def freeze_out_parameters_dataset(dataset: Dataset) -> Dataset:

    @typecheck
    def _process_latins(
        latins: Float[Array, "n d"], 
        parameters: Float[Array, "n 5"], 
        alpha: Float[Array, "5"], 
        mu: Float[Array, "d"], 
        dmu: Float[Array, "5 d"],
        p_idx: Int[Array, "_"]
    ) -> Float[Array, "n d"]:
        """ 
            Freeze out non-target parameters in latin hypercube simulations
            by removing linear response and adding fiducial linear response.
        """ 

        @typecheck
        def _freeze_parameters(
            pdf_or_cumulant: Float[Array, "d"], 
            p: Float[Array, "p"]
        ) -> Float[Array, "d"]:
            # Freeze the nuisance parameters of the latin hypercube realisations

            # Om, s8 and nuisances at fiducial values
            p0 = jnp.array(
                [_p if (i_p in p_idx) else alpha[i_p] for i_p, _p in enumerate(p)]
            ) 

            mu_p_nu = linearised_model(alpha, alpha_=p, mu=mu, dmu=dmu)
            mu_p_nu_0 = linearised_model(alpha, alpha_=p0, mu=mu, dmu=dmu)

            return pdf_or_cumulant - mu_p_nu + mu_p_nu_0

        return jax.vmap(_freeze_parameters)(latins, parameters)

    p_idx = get_target_idx()

    # Recalculate Fisher information (not marginalising out nuisances, they are known)
    derivatives = dataset.derivatives[:, p_idx, :]  
    _derivatives = jnp.mean(derivatives, axis=0)
    F = jnp.linalg.multi_dot([_derivatives, dataset.Cinv, _derivatives.T])
    Finv = jnp.linalg.inv(F)

    # Remove influence of nuisances from latins by linearisation
    latins_data = _process_latins(
        dataset.data, 
        dataset.parameters, 
        alpha=dataset.alpha, 
        mu=jnp.mean(dataset.fiducial_data, axis=0), 
        dmu=jnp.mean(dataset.derivatives, axis=0), # Must be derivatives for all parameters!
        p_idx=p_idx
    )

    frozen_dataset = Dataset(
        name=dataset.name,
        alpha=dataset.alpha[p_idx],
        lower=dataset.lower[p_idx],
        upper=dataset.upper[p_idx],
        parameter_strings=[
            dataset.parameter_strings[p] for p in p_idx
        ],
        Finv=Finv,
        Cinv=dataset.Cinv,
        C=dataset.C,
        fiducial_data=dataset.fiducial_data,
        data=latins_data,
        parameters=dataset.parameters[:, p_idx],
        derivatives=derivatives # Target-indexed derivatives
    )

    return frozen_dataset 


@typecheck
def get_prior(config: ConfigDict, dataset: Dataset) -> tfd.Distribution:

    if config.linearised:
        logger.info("Using flat prior")

        flat_limit = 1e4
        lower = jnp.ones((dataset.alpha.size,)) * -flat_limit
        upper = jnp.ones((dataset.alpha.size,)) * flat_limit
    else:
        logger.info("Using Quijote uniform prior")
        lower = jnp.asarray(dataset.lower) # Avoid tfp warning
        upper = jnp.asarray(dataset.upper)

    assert jnp.all((upper - lower) > 0.)

    print("FORCING FLAT PRIOR")
    logger.info("FORCING FLAT PRIOR")
    flat_limit = 1e4
    lower = jnp.ones((dataset.alpha.size,)) * -flat_limit
    upper = jnp.ones((dataset.alpha.size,)) * flat_limit

    # print("FORCING QUIJOTE PRIOR")
    # lower = jnp.asarray(dataset.lower) # Avoid tfp warning
    # upper = jnp.asarray(dataset.upper)

    if config.use_planck:
        prior = tfd.MultivariateNormalFullCovariance(
            dataset.alpha, covariance_matrix=get_Finv_planck()
        )
    else:
        prior = tfd.Blockwise(
            [tfd.Uniform(l, u) for l, u in zip(lower, upper)]
        )

    return prior


@typecheck
def get_prior_from_args(args) -> tfd.Distribution:

    if args.linearised:
        logger.info("Using flat prior")

        flat_limit = 1e4
        lower = jnp.ones((5,)) * -flat_limit
        upper = jnp.ones((5,)) * flat_limit
    else:
        logger.info("Using Quijote uniform prior")
        lower = jnp.asarray(LOWER) # Avoid tfp warning
        upper = jnp.asarray(UPPER)

    print("FORCING FLAT PRIOR")
    logger.info("FORCING FLAT PRIOR")
    flat_limit = 1e4
    lower = jnp.ones((5,)) * -flat_limit
    upper = jnp.ones((5,)) * flat_limit

    assert jnp.all((upper - lower) > 0.)

    if args.use_planck:
        prior = tfd.MultivariateNormalFullCovariance(
            ALPHA, covariance_matrix=get_Finv_planck()
        )
    else:
        prior = tfd.Blockwise(
            [tfd.Uniform(l, u) for l, u in zip(lower, upper)]
        )

    return prior


@typecheck
def linearised_model(
    alpha: Float[Array, "p"], 
    alpha_: Float[Array, "p"], 
    mu: Float[Array, "d"], 
    dmu: Float[Array, "p d"]
) -> Float[Array, "d"]:
    return mu + jnp.dot(alpha_ - alpha, dmu)


@typecheck
def sample_prior(
    key: PRNGKeyArray, 
    n_linear_sims: int, 
    alpha: Float[Array, "p"], 
    lower: Float[Array, "p"], 
    upper: Float[Array, "p"],
    *,
    hypercube: bool = True
) -> Float[Array, "n p"]:
    # Forcing Quijote prior for simulating, this prior for inference

    lower = lower.astype(jnp.float32) # Avoid tfp warning
    upper = upper.astype(jnp.float32)

    assert jnp.all((upper - lower) > 0.)

    keys_p = jr.split(key, alpha.size)

    if hypercube:
        logger.info("Hypercube sampling...")

        sampler = qmc.LatinHypercube(d=alpha.size)
        samples = sampler.random(n=n_linear_sims)
        Y = jnp.asarray(qmc.scale(samples, lower, upper))
    else:
        logger.info("Uniform box sampling...")

        Y = jnp.stack(
            [
                jr.uniform(
                    key_p, 
                    (n_linear_sims,), 
                    minval=lower[p], 
                    maxval=upper[p]
                )
                for p, key_p in enumerate(keys_p)
            ], 
            axis=1
        )

    return Y


@typecheck
def get_linearised_data(
    config: ConfigDict, 
    dataset: Dataset,
    *,
    n_linear_sims: Optional[int] = None
) -> tuple[Float[Array, "n d"], Float[Array, "n p"]]:
    """
        Get linearised PDFs and get their MLEs 

        # Pre-train data = Fisher summaries
        X_l, Y_l = get_fisher_summaries(
            summaries_key, 
            n=config.n_linear_sims, 
            parameter_prior=parameter_prior, 
            Finv=dataset.Finv
        )
    """
    
    logger.info("Linearising data...")

    key = jr.key(config.seed)

    key_parameters, key_simulations = jr.split(key)

    if config.n_linear_sims is not None:
        Y = sample_prior(
            key_parameters, 
            config.n_linear_sims if n_linear_sims is None else n_linear_sims, 
            dataset.alpha, 
            dataset.lower, 
            dataset.upper, 
            hypercube=True
        )
    else:
        Y = dataset.parameters

    assert dataset.derivatives.ndim == 3, (
        "Do derivatives [{}] have batch axis? Required.".format(dataset.derivatives.shape)
    )

    dmu = jnp.mean(dataset.derivatives, axis=0)
    mu = jnp.mean(dataset.fiducial_data, axis=0)

    def _simulator(key: PRNGKeyArray, pi: Float[Array, "p"]) -> Float[Array, "d"]:
        # Data model with linearised expectation
        _mu = linearised_model(alpha=dataset.alpha, alpha_=pi, mu=mu, dmu=dmu)
        return jr.multivariate_normal(key, mean=_mu, cov=dataset.C)

    keys = jr.split(key_simulations, len(Y))
    D = jax.vmap(_simulator)(keys, Y) 

    logger.info("... linearised data {} {}".format(D.shape, Y.shape))

    return D, Y # NOTE: only replacing latin hypercube


@typecheck
def non_gaussian_linear_model(
    pi: Float[Array, "p"], 
    idx: Optional[Int[Array, "..."]] = None, 
    *, 
    key: Optional[PRNGKeyArray] = None, 
    dataset: Dataset
) -> Float[Array, "d"]:
    # Non-Gaussian noise with linearised model realisations
    # - `ix` is an index into the fiducial_data
    # - includes noise by default

    mu = jnp.mean(dataset.fiducial_data, axis=0)
    dmu = jnp.mean(dataset.derivatives, axis=0)
    n_s = dataset.fiducial_data.shape[0]

    if idx is None:
        assert key is not None
        idx = jr.choice(key, n_s)

    xi_ = dataset.fiducial_data[idx]

    mu_L = linearised_model(dataset.alpha, pi, mu, dmu)

    return mu_L + (xi_ - mu) * (n_s / (n_s - 1))


@typecheck
def get_non_gaussian_linear_model_data(
    config: ConfigDict, 
    dataset: Dataset,
    *,
    key: Optional[PRNGKeyArray] = None
) -> tuple[Float[Array, "n d"], Float[Array, "n p"]]:

    logger.info("Linearising data...")

    if key is None:
        key = jr.key(config.seed)

    key_parameters, key_simulations = jr.split(key)

    # NOTE: fix indices of noise realisations for training?
    idx = jnp.arange(dataset.parameters.shape[0]) # jr.permutation(key_simulations, jnp.arange(dataset.parameters.shape[0]))

    D = jax.vmap(partial(non_gaussian_linear_model, dataset=dataset))(dataset.parameters, idx)

    logger.info("...linearised data {} {}".format(D.shape, dataset.parameters.shape))

    return D, dataset.parameters # Replacing only hypercube, same parameters as Quijote


@typecheck
def get_datavector(
    key: PRNGKeyArray, 
    config: ConfigDict, 
    dataset: Dataset, 
    n: int = 1, 
    *, 
    use_expectation: bool = False
) -> Float[Array, "... d"]:
    """ Measurement: either Gaussian linear model or not """

    # Choose a linearised model datavector or simply one of the Quijote realisations
    # which corresponds to a non-linearised datavector with Gaussian noise
    if config.use_expectation or use_expectation or FORCE_NOISELESS_DATAVECTOR:
        logger.info("Using expectation (noiseless datavector)...")

        datavector = jnp.mean(dataset.fiducial_data, axis=0, keepdims=True)
    elif NON_GAUSSIAN_TEST:
        logger.info("Using non-Gaussian noise with linear model...")

        sampler = lambda key: non_gaussian_linear_model(dataset.alpha, key=key, dataset=dataset)
        keys = jr.split(key, n)
        datavector = jax.vmap(sampler)(keys)
    else:
        if config.linearised:
            logger.info("Using linearised datavector...")

            mu = jnp.mean(dataset.fiducial_data, axis=0)

            datavector = jr.multivariate_normal(key, mean=mu, cov=dataset.C, shape=(n,))
        else:
            logger.info("Using non-linearised datavector...")

            # datavector = jr.choice(key, dataset.fiducial_data, shape=(n,))
            ix = jr.choice(key, jnp.arange(len(dataset.fiducial_data)), shape=(n,))
            datavector = dataset.fiducial_data[ix]

    if not (n > 1):
        datavector = jnp.squeeze(datavector, axis=0) # Remove batch axis by default

    return datavector 


"""
    Compression
"""


@typecheck
def get_linear_compressor(
    config: ConfigDict, 
    dataset: Dataset
) -> Callable[[Float[Array, "d"], Float[Array, "p"]], Float[Array, "p"]]:
    """ 
        Get Chi^2 minimisation function; compressing datavector 
        at estimated parameters to summary 
    """

    alpha = dataset.alpha
    F_planck = get_F_planck()

    logger.info("Using MAP linear compression" if config.use_planck else "Using MLE linear compression")

    @typecheck
    def mle(
        d: Float[Array, "d"], 
        pi: Float[Array, "p"], 
        Finv: Float[Array, "p p"], 
        mu: Float[Array, "d"], 
        dmu: Float[Array, "p d"], 
        precision: Float[Array, "d d"]
    ) -> Float[Array, "p"]:
        return pi + jnp.linalg.multi_dot([Finv, dmu, precision, d - mu])

    @typecheck
    def map(
        d: Float[Array, "d"], 
        pi: Float[Array, "p"], 
        Finv: Float[Array, "p p"], 
        mu: Float[Array, "d"], 
        dmu: Float[Array, "p d"], 
        precision: Float[Array, "d d"]
    ) -> Float[Array, "p"]:
        F = jnp.linalg.inv(Finv)
        _Finv = jnp.linalg.inv(F + F_planck)
        return pi + jnp.linalg.multi_dot([_Finv, dmu, precision, d - mu]) + jnp.linalg.multi_dot([_Finv, F_planck, alpha - pi])

    @typecheck
    def compressor(
        d: Float[Array, "d"], 
        p: Float[Array, "p"],
        mu: Float[Array, "d"], 
        dmu: Float[Array, "p d"]
    ) -> Float[Array, "p"]: 

        mu_p = linearised_model(
            alpha=alpha, alpha_=p, mu=mu, dmu=dmu
        )

        _estimator_fn = map if config.use_planck else mle

        p_ = _estimator_fn(
            d,
            pi=p,
            Finv=dataset.Finv, 
            mu=mu_p,            
            dmu=dmu, 
            precision=dataset.Cinv
        )

        return p_

    mu = jnp.mean(dataset.fiducial_data, axis=0)
    dmu = jnp.mean(dataset.derivatives, axis=0)

    return partial(compressor, mu=mu, dmu=dmu)


import jax
import jax.numpy as jnp
from jax.scipy.linalg import svd


class PCA:
    """
    Principal Component Analysis (PCA) for dimensionality reduction.

    Attributes:
        num_components (int): Number of principal components to keep.
        mean (jax.Array, optional): Mean of each feature in the training data.
        principal_components (jax.Array, optional): Principal components (eigenvectors) of the training data.
        explained_variance (jax.Array, optional): Explained variance of each principal component.
    """

    def __init__(self, num_components: int):
        self.num_components = num_components
        self.mean = None
        self.principal_components = None
        self.explained_variance = None

    def fit(self, X: jax.Array):
        n, m = X.shape

        if self.mean is None:
            self.mean = X.mean(axis=0)

        X_centred = X - self.mean
        S, self.principal_components = svd(X_centred, full_matrices=True)[1:]

        self.explained_variance = jnp.square(S) / jnp.sum(jnp.square(S))

    def transform(self, X: jax.Array):
        if self.principal_components is None:
            raise RuntimeError("Must fit before transforming.")

        X_centred = X - X.mean(axis=0)
        return jnp.dot(X_centred, self.principal_components[: self.num_components].T)

    def fit_transform(self, X: jax.Array):
        if self.mean is None:
            self.mean = X.mean(axis=0)

        X_centred = X - self.mean

        self.principal_components = svd(X_centred, full_matrices=True)[2]

        return jnp.dot(X_centred, self.principal_components[: self.num_components].T)

    def inverse_transform(self, X_transformed: jax.Array):
        if self.principal_components is None:
            raise RuntimeError("Must fit before transforming.")

        return (
            jnp.dot(X_transformed, self.principal_components[: self.num_components])
            + self.mean
        )


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


@eqx.filter_jit
def evaluate(
    model: eqx.Module, 
    x: Float[Array, "b x"], 
    y: Float[Array, "b y"],
    key: PRNGKeyArray,
    *, 
    precision: Optional[Float[Array, "y y"]] = None,
    replicated_sharding: Optional[PositionalSharding] = None
) -> Scalar:
    model = eqx.nn.inference_mode(model, True)
    if replicated_sharding is not None:
        model = eqx.filter_shard(model, replicated_sharding)
    return loss(model, x, y, key=key, precision=precision)


@typecheck
@eqx.filter_jit
def make_step(
    model: eqx.Module, 
    opt_state: optax.OptState,
    x: Float[Array, "b x"], 
    y: Float[Array, "b y"],
    key: PRNGKeyArray,
    opt: optax.GradientTransformation, 
    *, 
    precision: Optional[Float[Array, "y y"]] = None,
    replicated_sharding: Optional[PositionalSharding]
) -> Tuple[eqx.Module, optax.OptState, Scalar]:

    model = eqx.nn.inference_mode(model, False)

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

    if exists(precision):
        logger.info("Using Fisher precision for NN.")

    D, Y = train_data

    n_s, _ = D.shape

    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    if valid_data is not None:
        Xt, Yt = train_data
        Xv, Yv = valid_data
    else:
        n_train = D.shape[0]
        Xt, Xv = jnp.split(D, [n_train - int(valid_fraction * n_s)]) 
        Yt, Yv = jnp.split(Y, [n_train - int(valid_fraction * n_s)])

    if use_tqdm: 
        steps = trange(n_steps, desc=description, colour="magenta")
    else: 
        steps = range(n_steps)

    L = np.zeros((n_steps, 2))
    for step in steps:
        key_t, key_v = jr.split(jr.fold_in(key, step))

        if exists(n_batch):
            x, y = get_batch(Xt, Yt, n=n_batch, key=key_t) 
        else:
            x, y = Xt, Yt
        
        if sharding is not None:
            x, y = eqx.filter_shard((x, y), sharding)

        model, opt_state, train_loss = make_step(
            model, 
            opt_state, 
            x, 
            y, 
            key_t,
            opt=opt, 
            precision=precision, 
            replicated_sharding=replicated_sharding
        )

        if exists(n_batch):
            x, y = get_batch(Xv, Yv, n=n_batch, key=key_v)
        else:
            x, y = Xv, Yv

        if sharding is not None:
            x, y = eqx.filter_shard((x, y), sharding)

        valid_loss = evaluate(
            model, 
            x, 
            y, 
            key_v, 
            precision=precision, 
            replicated_sharding=replicated_sharding
        )

        L[step] = train_loss, valid_loss
        if use_tqdm:
            steps.set_postfix_str(
                "t={:.3E}, v={:.3E}".format(train_loss.item(), valid_loss.item())
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


@typecheck
def get_nn_compressor(
    key: PRNGKeyArray, 
    config: ConfigDict,
    dataset: Dataset, 
    *, 
    lbfgs: bool = False, 
    results_dir: str, 
    net: Optional[eqx.Module] = None
) -> tuple[eqx.Module, Callable, Callable]:
    """
        Train neural network compression function
        - Optionally use parameter covariance for chi2 loss
    """


    class ExtraMLP(eqx.nn.MLP):
        dropouts: list[eqx.nn.Dropout]
        layernorms: list[eqx.nn.LayerNorm]

        def __init__(self, *args, p: float, key: PRNGKeyArray, **kwargs):
            super().__init__(*args, **kwargs, key=key)
            dropouts = []
            layernorms = []
            for layer in self.layers[:-1]:
                dropouts.append(eqx.nn.Dropout(p=p))
                layernorms.append(eqx.nn.LayerNorm(layer.weight.shape[0]))
            self.dropouts = dropouts + [eqx.nn.Identity()] # Hacky safe-zip
            self.layernorms = layernorms + [eqx.nn.Identity()] # Hacky safe-zip

        def __call__(self, x, key=None):
            for i, (layer, dropout, norm) in enumerate(
                zip(self.layers, self.dropouts, self.layernorms)
            ):
                x = layer(x)
                x = norm(x)
                x = dropout(x, key=key)
                if i != len(self.layers) - 1:
                    x = self.activation(x) # Don't activate last layer
            if self.final_activation is not None:
                x = self.final_activation(x) # ...unless it is required
            return x


    @eqx.filter_vmap
    def make_ensemble(key):
        return eqx.nn.MLP(
        # return ExtraMLP(
            dataset.data.shape[-1], 
            dataset.parameters.shape[-1], 
            width_size=config.nn.width_size, 
            depth=config.nn.depth, 
            use_final_bias=config.nn.use_final_bias,
            # p=0.3,
            activation=getattr(jax.nn, config.nn.activation),
            key=key
        )


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


    mu_D = dataset.data.mean(axis=0) #jnp.mean(dataset.fiducial_data, axis=0)
    std_D = dataset.data.std(axis=0) #jnp.std(dataset.fiducial_data, axis=0)

    mu_p = dataset.parameters.mean(axis=0)
    std_p = dataset.parameters.std(axis=0)

    def get_preprocess_fn(D, use_pca):
        """ 
            Pre-process data, using PCA or not, returning the pre-processing transform 
            for use downstream with measurements.
            - Data is pre-processed here!
        """ 

        # eigvals, eigvecs = jnp.linalg.eigh(
        #     # jnp.cov(D - mu_D, rowvar=False)
        #     jnp.cov(dataset.fiducial_data - mu_D, rowvar=False)
        #     # dataset.C) 
        # )
        # D = ((D - mu_D) @ eigvecs) / np.sqrt(eigvals + 1e-8)

        # if use_pca:
        #     pca = PCA(num_components=D.shape[-1]) 
        #     pca.fit(dataset.data.fiducial_data) # Fit on fiducial data?

        def preprocess_fn(d):
            # d = jnp.asarray(d)
            # if use_pca:
            #     d = pca.transform(d)
            # return ((d - mu_D) @ eigvecs) / np.sqrt(eigvals + 1e-10) 
            return (d - mu_D) / std_D

        return preprocess_fn


    description = "Fitting NN [{}]".format(dataset.name)

    net_key, train_key = jr.split(key)

    keys = jr.split(net_key, config.nn.n_ensemble)

    net = Ensemble(make_ensemble(keys))

    preprocess_fn = get_preprocess_fn(dataset.data, use_pca=config.nn.use_pca)

    print("D, Y:", dataset.data.shape, dataset.parameters.shape)
    
    def preprocess_fn_p(p):
        # Scale parameters into loss / out of net
        # return (p - dataset.alpha) / np.diag(dataset.Finv)
        return (p - mu_p) / std_p

    def postprocess_fn_p(p):
        # Scale parameters into loss / out of net
        # return (p - dataset.alpha) / np.diag(dataset.Finv)
        return p * std_p + mu_p

    preprocess_fn = preprocess_fn_p = postprocess_fn_p = lambda x: x

    train_data = (preprocess_fn(dataset.data), preprocess_fn_p(dataset.parameters))

    precision = jnp.linalg.inv(dataset.Finv) # In reality this varies with parameters

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

    eqx.tree_serialise_leaves(
        os.path.join(results_dir, "nn.eqx"), net
    )

    return net, preprocess_fn, postprocess_fn_p
 

@typecheck
def get_compression_fn(
    key: PRNGKeyArray, 
    config: ConfigDict, 
    dataset: Dataset, 
    *, 
    results_dir: str
) -> Callable[[Float[Array, "d"], Float[Array, "p"]], Float[Array, "p"]]:
    """ 
        Get linear or neural network compressor
    """ 

    logger.info("Getting compression fn ({}) for dataset={}".format(config.compression, dataset.name))

    assert config.compression in ["linear", "nn", "nn-lbfgs"]

    if config.compression == "nn" or config.compression == "nn-lbfgs":

        net, preprocess_fn, postprocess_fn = get_nn_compressor(
            key, 
            config,
            dataset, 
            lbfgs=(config.compression == "nn-lbfgs"), 
            results_dir=results_dir
        )

        def compression_fn(d, p): 
            return postprocess_fn(net(preprocess_fn(d))) # Ignore parameter kwarg for NN

        logger.info("Using NN compression function.")

    if config.compression == "linear":
        compressor = get_linear_compressor(config, dataset)

        def compression_fn(d, p): 
            return compressor(d, p)

        logger.info("Using linear compression function.")

    return compression_fn 