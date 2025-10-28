import time
import os
from dataclasses import dataclass, fields
from functools import partial
from typing import Callable, Optional, Literal, Sequence, Iterable, Any

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray, Array, Float, Int, Scalar, jaxtyped
from beartype import beartype as typechecker 

from scipy.stats import qmc
import numpy as np
from ml_collections import ConfigDict

from configs.log import setup_module_logger, get_log_level
from data.constants import LOWER, UPPER, ALPHA

TYPECHECK = True if os.environ.get("TYPECHECK", "").lower() in ("1", "true") else False
if TYPECHECK:
    typecheck = jaxtyped(typechecker=typechecker)
else:
    typecheck = lambda x: x

Distribution = Any

logger, log_figs_dir = setup_module_logger(__name__, level=get_log_level())

FORCE_NOISELESS_DATAVECTOR = True if os.environ.get("FORCE_NOISELESS_DATAVECTOR", "").lower() in ("1", "true") else False
FORCE_FLAT_PRIOR = True if os.environ.get("FORCE_FLAT_PRIOR", "").lower() in ("1", "true") else False
NON_GAUSSIAN_TEST = True if os.environ.get("NON_GAUSSIAN_TEST", "").lower() in ("1", "true") else False
N_ENSEMBLE_NETS = int(os.environ.get("N_ENSEMBLE_NETS", 10))
NN_TYPE = os.environ.get("NN_TYPE", "NN")

"""
    Objects common to the PDF and cumulant datasets
"""


def exists(v):
    return v is not None


def hartlap(n_s: int, n_d: int) -> float: 
    return (n_s - n_d - 2) / (n_s - 1)


@dataclass(frozen=True)
class BlockwiseUniform:
    lows: tuple[Array, ...]
    highs: tuple[Array, ...]

    def __post_init__(self):
        if len(self.lows) != len(self.highs):
            raise ValueError(
                "lows and highs must have same number of blocks"
            )
        for i, (lo, hi) in enumerate(zip(self.lows, self.highs)):
            if lo.shape != hi.shape:
                raise ValueError(
                    f"Block {i}: low/high shapes differ: {lo.shape} vs {hi.shape}"
                )
            if lo.ndim != 1:
                raise ValueError(
                    f"Block {i}: bounds must be 1-D (got {lo.ndim}D)"
                )
        # Widths must be strictly positive
        widths_ok = all(
            jnp.all(self._width(i) > 0.) for i in range(len(self.lows))
        )
        if not widths_ok:
            raise ValueError(
                "All (high - low) must be strictly positive in every block"
            )

    @classmethod
    def from_bounds(
        cls, 
        bounds: Iterable[tuple[Sequence[float], Sequence[float]]]
    ) -> "BlockwiseUniform":
        lows, highs = [], []
        for lo, hi in bounds:
            lo = jnp.asarray(lo, dtype=jnp.float32)
            hi = jnp.asarray(hi, dtype=jnp.float32)
            lows.append(lo)
            highs.append(hi)
        return cls(tuple(lows), tuple(highs))

    def _width(self, i: int) -> Array:
        return self.highs[i] - self.lows[i]

    @property
    def event_sizes(self) -> tuple[int, ...]:
        return tuple(int(lo.shape[0]) for lo in self.lows)

    @property
    def event_size(self) -> int:
        return int(sum(self.event_sizes))

    @property
    def _split_indices(self):
        # Indices to split concatenated vectors back into blocks
        sizes = np.array(self.event_sizes, dtype=int)
        return tuple(np.cumsum(sizes)[:-1].tolist())

    @property
    def log_volume(self) -> Scalar:
        # log of the hyper-rectangle volume (sum over all dims in all blocks)
        return sum(jnp.sum(jnp.log(self._width(i))) for i in range(len(self.lows)))

    def block_variances(self) -> tuple[Array, ...]:
        """Return per-block variance arrays matching each block's shape.
        
        For a uniform on [low, high] in each dimension, Var = (high - low)^2 / 12.
        """
        return tuple((self._width(i) ** 2.) / 12. for i in range(len(self.lows)))

    def variance(self) -> Array:
        """Return per-dimension variances as a flat vector of shape (event_size,)."""
        return jnp.concatenate(self.block_variances(), axis=0)

    def sample(self, key: PRNGKeyArray, sample_shape: tuple[int, ...] = ()) -> Array:
        """Return samples with shape sample_shape + (event_size,)."""

        keys = jr.split(key, len(self.lows))

        parts = []
        for k, lo, w in zip(
            keys, 
            self.lows, 
            [self._width(i) for i in range(len(self.lows))]
        ):
            u = jr.uniform(
                k, 
                shape=sample_shape + lo.shape, 
                minval=0., 
                maxval=1.
            )

            parts.append(lo + u * w)

        return jnp.concatenate(parts, axis=-1)

    def log_prob(self, x: Array) -> Scalar:
        """x shape (..., event_size). Returns log p with shape (...,)."""
        # Split x into blocks along last axis
        blocks = jnp.split(x, self._split_indices, axis=-1) if len(self.lows) > 1 else (x,)

        in_support = jnp.array(True)
        for xb, lo, hi in zip(blocks, self.lows, self.highs):

            in_b = jnp.all((xb >= lo) & (xb <= hi), axis=-1)  # (...,)

            in_support = jnp.logical_and(in_support, in_b)

        # Constant density inside support: -log(volume); -inf outside.
        logp_inside = -self.log_volume

        return jnp.where(in_support, logp_inside, -jnp.inf)


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
    derivatives: Float[Array, "nd p d"]

    @classmethod
    def from_dict(cls, dataset_dict, *, name):
        assert name in ["bulk_pdf", "bulk", "tails"]

        convert_map = {"parameter_strings": list}

        kwargs = {}
        for f in fields(cls):
            if f.name == "name":
                kwargs[f.name] = name
            else:
                val = dataset_dict[f.name]
                fn = convert_map.get(f.name, jnp.asarray)
                kwargs[f.name] = fn(val)

        return cls(**kwargs)


@typecheck
def get_prior() -> Distribution:

    if FORCE_FLAT_PRIOR:
        logger.info("Forcing flat prior...")
        flat_limit = 1e4
        _lower = jnp.ones((ALPHA.size,)) * -flat_limit
        _upper = jnp.ones((ALPHA.size,)) * flat_limit
    else:
        _lower = LOWER
        _upper = UPPER

    prior = BlockwiseUniform.from_bounds(
        [
            (jnp.atleast_1d(_lower[p]), jnp.atleast_1d(_upper[p])) 
            for p in range(ALPHA.size)
        ]
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
    upper: Float[Array, "p"]
) -> Float[Array, "n p"]:
    # Forcing Quijote prior for simulating, this prior for inference

    assert jnp.all((upper - lower) > 0.)

    seed = jnp.sum(jr.key_data(key))
    sampler = qmc.LatinHypercube(d=alpha.size, rng=np.random.default_rng(int(seed)))
    # sampler = qmc.LatinHypercube(d=alpha.size)

    samples = sampler.random(n=n_linear_sims)
    Y = jnp.asarray(qmc.scale(samples, lower, upper))

    return Y


@typecheck
def get_linearised_data(
    config: ConfigDict, 
    dataset: Dataset,
    *,
    n_linear_sims: Optional[int] = None
) -> tuple[Float[Array, "nf d"], Float[Array, "n d"], Float[Array, "n p"]]:
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
            dataset.upper
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

    keys = jr.split(key_simulations, dataset.fiducial_data.shape[0])
    D0 = jax.vmap(_simulator, in_axes=(0, None))(keys, dataset.alpha) 

    logger.info("... linearised data {} {} {}".format(D0.shape, D.shape, Y.shape))

    return D0, D, Y # NOTE: only replacing latin hypercube


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

    n_s = dataset.fiducial_data.shape[0]

    mu = jnp.mean(dataset.fiducial_data, axis=0)
    dmu = jnp.mean(dataset.derivatives, axis=0)

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

    # NOTE: fix indices of noise realisations for training?
    idx = jnp.arange(dataset.parameters.shape[0]) # jr.permutation(key_simulations, jnp.arange(dataset.parameters.shape[0]))

    _model = partial(non_gaussian_linear_model, dataset=dataset)
    D = jax.vmap(_model)(dataset.parameters, idx)

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

    # Choose a datavector that has a non-Gaussian noise realisation on a linear model prediction
    if NON_GAUSSIAN_TEST:
        logger.info("Using non-Gaussian noise with linear model...")

        sampler = lambda key: non_gaussian_linear_model(dataset.alpha, key=key, dataset=dataset)

        keys = jr.split(key, n)
        datavector = jax.vmap(sampler)(keys)

    # Choose a datavector that has a Gaussian noise realisation on a linear model prediction
    if config.linearised:
        logger.info("Using linearised datavector...")

        mu = jnp.mean(dataset.fiducial_data, axis=0)

        datavector = jr.multivariate_normal(key, mean=mu, cov=dataset.C, shape=(n,))

    # Choose a non-Gaussian non-linear datavector
    if not (
        config.linearised 
        or config.use_expectation 
        or use_expectation 
        or FORCE_NOISELESS_DATAVECTOR
    ):
        logger.info("Using non-linearised datavector...")

        # datavector = jr.choice(key, dataset.fiducial_data, shape=(n,))
        ix = jr.choice(key, len(dataset.fiducial_data), shape=(n,))
        datavector = dataset.fiducial_data[ix]

    # (d,) -> (1, d)
    if datavector.ndim == 1:        
        return datavector[jnp.newaxis, :]

    return datavector 


# def freeze_out_parameters_dataset(dataset: Dataset) -> Dataset:

#     @typecheck
#     def _process_latins(
#         latins: Float[Array, "n d"], 
#         parameters: Float[Array, "n 5"], 
#         alpha: Float[Array, "5"], 
#         mu: Float[Array, "d"], 
#         dmu: Float[Array, "5 d"],
#         p_idx: Int[Array, "_"]
#     ) -> Float[Array, "n d"]:
#         """ 
#             Freeze out non-target parameters in latin hypercube simulations
#             by removing linear response and adding fiducial linear response.
#         """ 

#         @typecheck
#         def _freeze_parameters(
#             pdf_or_cumulant: Float[Array, "d"], 
#             p: Float[Array, "p"]
#         ) -> Float[Array, "d"]:
#             # Freeze the nuisance parameters of the latin hypercube realisations

#             # Om, s8 and nuisances at fiducial values
#             p0 = jnp.array(
#                 [_p if (i_p in p_idx) else alpha[i_p] for i_p, _p in enumerate(p)]
#             ) 

#             mu_p_nu = linearised_model(alpha, alpha_=p, mu=mu, dmu=dmu)
#             mu_p_nu_0 = linearised_model(alpha, alpha_=p0, mu=mu, dmu=dmu)

#             return pdf_or_cumulant - mu_p_nu + mu_p_nu_0

#         return jax.vmap(_freeze_parameters)(latins, parameters)

#     p_idx = get_target_idx()

#     # Recalculate Fisher information (not marginalising out nuisances, they are known)
#     derivatives = dataset.derivatives[:, p_idx, :]  
#     _derivatives = jnp.mean(derivatives, axis=0)
#     F = jnp.linalg.multi_dot([_derivatives, dataset.Cinv, _derivatives.T])
#     Finv = jnp.linalg.inv(F)

#     # Remove influence of nuisances from latins by linearisation
#     latins_data = _process_latins(
#         dataset.data, 
#         dataset.parameters, 
#         alpha=dataset.alpha, 
#         mu=jnp.mean(dataset.fiducial_data, axis=0), 
#         dmu=jnp.mean(dataset.derivatives, axis=0), # Must be derivatives for all parameters!
#         p_idx=p_idx
#     )

#     frozen_dataset = Dataset(
#         name=dataset.name,
#         alpha=dataset.alpha[p_idx],
#         lower=dataset.lower[p_idx],
#         upper=dataset.upper[p_idx],
#         parameter_strings=[
#             dataset.parameter_strings[p] for p in p_idx
#         ],
#         Finv=Finv,
#         Cinv=dataset.Cinv,
#         C=dataset.C,
#         fiducial_data=dataset.fiducial_data,
#         data=latins_data,
#         parameters=dataset.parameters[:, p_idx],
#         derivatives=derivatives # Target-indexed derivatives
#     )

#     return frozen_dataset 



# GET PRIOR
    # if config.linearised:
    #     logger.info("Using flat prior (linearised)")

    #     flat_limit = 1e4
    #     lower = jnp.ones((dataset.alpha.size,)) * -flat_limit
    #     upper = jnp.ones((dataset.alpha.size,)) * flat_limit
    # else:
    #     logger.info("Using Quijote uniform prior")
    #     lower = jnp.asarray(dataset.lower) # Avoid tfp warning
    #     upper = jnp.asarray(dataset.upper)

    # if FORCE_FLAT_PRIOR:
    #     print("FORCING FLAT PRIOR")
    #     logger.info("FORCING FLAT PRIOR")
    #     flat_limit = 1e4
    #     lower = jnp.ones((dataset.alpha.size,)) * -flat_limit
    #     upper = jnp.ones((dataset.alpha.size,)) * flat_limit

    # if FORCE_QUIJOTE_PRIOR:
    #     print("FORCING QUIJOTE PRIOR")
    #     logger.info("Using Quijote uniform prior")
    #     lower = jnp.asarray(dataset.lower) # Avoid tfp warning
    #     upper = jnp.asarray(dataset.upper)

    # assert jnp.all((upper - lower) > 0.)

    # if config.use_planck:
    #     # prior = tfd.MultivariateNormalFullCovariance(
    #     #     dataset.alpha, covariance_matrix=get_Finv_planck()
    #     # )
    #     pass
    # else:
    #     # prior = tfd.Blockwise(
    #     #     [tfd.Uniform(l, u) for l, u in zip(lower, upper)]
    #     # )
    #     prior = BlockwiseUniform.from_bounds(
    #         [
    #             (jnp.atleast_1d(lower[p]), jnp.atleast_1d(upper[p])) 
    #             for p in range(dataset.alpha.size)
    #         ]
    #     )