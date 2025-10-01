import os
from dataclasses import dataclass, fields
from functools import partial
from typing import Callable, Optional, Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import PRNGKeyArray, Array, Float, Int, jaxtyped

from beartype import beartype as typechecker 
from scipy.stats import qmc
from ml_collections import ConfigDict
import tensorflow_probability.substrates.jax.distributions as tfd

from configs.log import setup_module_logger, get_log_level
from data.constants import get_quijote_parameters, get_target_idx, get_F_planck, get_Finv_planck, LOWER, UPPER, ALPHA
from data.nn import get_nn_compressor

typecheck = jaxtyped(typechecker=typechecker)

logger, log_figs_dir = setup_module_logger(__name__, level=get_log_level())

FORCE_NOISELESS_DATAVECTOR = True if os.environ.get("FORCE_NOISELESS_DATAVECTOR", "").lower() in ("1", "true") else False
FORCE_FLAT_PRIOR = True if os.environ.get("FORCE_FLAT_PRIOR", "").lower() in ("1", "true") else False
FORCE_QUIJOTE_PRIOR = True if os.environ.get("FORCE_QUIJOTE_PRIOR", "").lower() in ("1", "true") else False
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
        logger.info("Using flat prior (linearised)")

        flat_limit = 1e4
        lower = jnp.ones((dataset.alpha.size,)) * -flat_limit
        upper = jnp.ones((dataset.alpha.size,)) * flat_limit
    else:
        logger.info("Using Quijote uniform prior")
        lower = jnp.asarray(dataset.lower) # Avoid tfp warning
        upper = jnp.asarray(dataset.upper)

    if FORCE_FLAT_PRIOR:
        print("FORCING FLAT PRIOR")
        logger.info("FORCING FLAT PRIOR")
        flat_limit = 1e4
        lower = jnp.ones((dataset.alpha.size,)) * -flat_limit
        upper = jnp.ones((dataset.alpha.size,)) * flat_limit

    if FORCE_QUIJOTE_PRIOR:
        print("FORCING QUIJOTE PRIOR")
        logger.info("Using Quijote uniform prior")
        lower = jnp.asarray(dataset.lower) # Avoid tfp warning
        upper = jnp.asarray(dataset.upper)

    assert jnp.all((upper - lower) > 0.)

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
        logger.info("Using flat prior (linearised)")

        flat_limit = 1e4
        lower = jnp.ones((ALPHA.size,)) * -flat_limit
        upper = jnp.ones((ALPHA.size,)) * flat_limit
    else:
        logger.info("Using Quijote uniform prior")
        lower = jnp.asarray(LOWER) # Avoid tfp warning
        upper = jnp.asarray(UPPER)

    if FORCE_FLAT_PRIOR:
        print("FORCING FLAT PRIOR")
        logger.info("FORCING FLAT PRIOR")
        flat_limit = 1e4
        lower = jnp.ones((ALPHA.size,)) * -flat_limit
        upper = jnp.ones((ALPHA.size,)) * flat_limit

    if FORCE_QUIJOTE_PRIOR:
        print("FORCING QUIJOTE PRIOR")
        logger.info("Using Quijote uniform prior")
        lower = jnp.asarray(LOWER) # Avoid tfp warning
        upper = jnp.asarray(UPPER)

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
    # Choose a datavector that has a non-Gaussian noise realisation on a linear model prediction
    elif NON_GAUSSIAN_TEST:
        logger.info("Using non-Gaussian noise with linear model...")

        sampler = lambda key: non_gaussian_linear_model(dataset.alpha, key=key, dataset=dataset)

        keys = jr.split(key, n)
        datavector = jax.vmap(sampler)(keys)
    # Choose a datavector that has a Gaussian noise realisation on a linear model prediction
    elif config.linearised:
        logger.info("Using linearised datavector...")

        mu = jnp.mean(dataset.fiducial_data, axis=0)

        datavector = jr.multivariate_normal(key, mean=mu, cov=dataset.C, shape=(n,))
    # Choose a non-Gaussian non-linear datavector
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


# class ExtraMLP(eqx.nn.MLP):
#     dropouts: list[eqx.nn.Dropout]
#     layernorms: list[eqx.nn.LayerNorm]
#     p: float

#     def __init__(self, *args, p: float, key: PRNGKeyArray, **kwargs):
#         super().__init__(*args, **kwargs, key=key)
#         self.p = p
#         dropouts = []
#         layernorms = []

#         for layer in self.layers[:-1]:
#             dropouts.append(eqx.nn.Dropout(p=p))
#             layernorms.append(eqx.nn.LayerNorm(layer.weight.shape[1]))

#         self.dropouts = dropouts + [eqx.nn.Identity()] # Hacky safe-zip
#         self.layernorms = layernorms + [eqx.nn.Identity()] # Hacky safe-zip

#     def __call__(self, x, key=None):
#         for i, (layer, dropout, norm) in enumerate(
#             zip(self.layers, self.dropouts, self.layernorms)
#         ):
#             x = norm(x)

#             x = layer(x)

#             if self.p is not None:
#                 if self.p > 0.:
#                     x = dropout(x, key=key)

#             if i != len(self.layers) - 1:
#                 x = self.activation(x) # Don't activate last layer
                
#         if self.final_activation is not None:
#             x = self.final_activation(x) # ...unless it is required
            
#         return x



class ExtraMLP(eqx.Module):
    layers: list[eqx.nn.Linear]
    # dropouts: list[eqx.nn.Dropout | eqx.nn.Identity]
    layernorms: list[eqx.nn.LayerNorm | eqx.nn.Identity]
    activation: Callable
    final_activation: Optional[Callable]
    # p: float

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_sizes: list[int],
        *,
        p: float = 0.0,
        activation: Callable = jax.nn.gelu,
        final_activation: Optional[Callable] = None,
        use_final_bias: bool = True,
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
        # self.p = p
        self.activation = activation
        self.final_activation = final_activation

        # Build Linear stack according to sizes
        if (
            hidden_sizes == [0] 
            or len(list(hidden_sizes)) == 0 
            or hidden_sizes == [None]
        ):
            sizes = [in_size, out_size]
        else:
            sizes = [in_size, *hidden_sizes, out_size]
        n_layers = len(sizes) - 1
        k_lin = jr.split(key, n_layers)

        layers = []
        for i in range(n_layers):

            if sizes[i + 1] == out_size:
                use_bias = use_final_bias

            layers.append(
                eqx.nn.Linear(sizes[i], sizes[i + 1], use_bias=use_bias, key=k_lin[i])
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
        # drops = []
        # if self.p > 0.0:
        #     for _ in self.layers[:-1]:
        #         drops.append(eqx.nn.Dropout(p=self.p))
        #     drops.append(eqx.nn.Identity())  # no dropout on final layer output
        # else:
        #     drops = [eqx.nn.Identity() for _ in self.layers]
        # self.dropouts = drops

    def __call__(self, x, *, key: Optional[PRNGKeyArray] = None):
        # If we have real Dropout modules and a key, split it so masks differ per layer

        # if self.p > 0.0 and key is not None:
        #     k_list = list(jr.split(key, len(self.layers)))
        # else:
        #     k_list = [None] * len(self.layers)

        for i, (
            layer, 
            norm, 
            # drop, 
            # k_i
        ) in enumerate(
            zip(
                self.layers, 
                self.layernorms, 
                # self.dropouts, 
                # k_list
            )
        ):
            x = layer(x)
            x = norm(x)
            # x = drop(x, key=k_i)
            if i != len(self.layers) - 1:
                x = self.activation(x)  # no hidden activation after last layer

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x


def trunc_init(weight: jax.Array, key: PRNGKeyArray) -> jax.Array:

    out, in_ = weight.shape
    stddev = jnp.sqrt(1 / in_)

    return stddev * jr.truncated_normal(key, shape=(out, in_), lower=-2., upper=2.)


def init_linear_weight(model, init_fn, key):

    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [
        x.weight
        for x in jax.tree.leaves(m, is_leaf=is_linear)
        if is_linear(x)
    ]
    weights = get_weights(model)

    new_weights = [
        init_fn(weight, subkey)
        for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
    ]
    new_model = eqx.tree_at(get_weights, model, new_weights)

    return new_model


def get_default_nn(dataset, config, net_key):

    net = ExtraMLP(
        dataset.data.shape[-1], 
        dataset.parameters.shape[-1], 
        hidden_sizes=config.nn.width_size, 
        # width_size=config.nn.width_size, 
        # depth=config.nn.depth, 
        use_final_bias=config.nn.use_final_bias,
        activation=(
            getattr(jax.nn, config.nn.activation)
            if config.nn.depth > 0 else lambda x: x
        ),
        p=0.,
        key=net_key
    )

    net = init_linear_weight(net, trunc_init, net_key)

    return net


@typecheck
def get_compression_fn(
    key: PRNGKeyArray, 
    config: ConfigDict, 
    dataset: Dataset, 
    *, 
    train: bool = True,
    results_dir: str
) -> Callable[[Float[Array, "d"], Float[Array, "p"]], Float[Array, "p"]]:
    """ 
        Get linear or neural network compressor
    """ 

    logger.info("Getting compression fn ({}) for dataset={}".format(config.compression, dataset.name))

    assert config.compression in ["linear", "nn", "nn-lbfgs"]

    if config.compression == "nn" or config.compression == "nn-lbfgs":

        net_key, train_key = jr.split(key)

        # net = eqx.nn.MLP(
        #     dataset.data.shape[-1], 
        #     dataset.parameters.shape[-1], 
        #     width_size=config.nn.width_size, 
        #     depth=config.nn.depth, 
        #     use_final_bias=config.nn.use_final_bias,
        #     activation=(
        #         getattr(jax.nn, config.nn.activation)
        #         if config.nn.depth > 0 else lambda x: x
        #     ),
        #     key=net_key
        # )

        # net = ExtraMLP(
        #     dataset.data.shape[-1], 
        #     dataset.parameters.shape[-1], 
        #     width_size=config.nn.width_size, 
        #     depth=config.nn.depth, 
        #     use_final_bias=config.nn.use_final_bias,
        #     activation=(
        #         getattr(jax.nn, config.nn.activation)
        #         if config.nn.depth > 0 else lambda x: x
        #     ),
        #     p=0.,
        #     key=net_key
        # )
        net = get_default_nn(dataset, config, net_key)

        print("MLP COMPRESSOR (linearised={}):\n".format(config.linearised), net)

        net, preprocess_fn_d, postprocess_fn_p = get_nn_compressor(
            train_key, 
            net,
            config,
            dataset, 
            lbfgs=(config.compression == "nn-lbfgs"), 
            results_dir=results_dir,
            train=train # If not training, return initialised net and processing fns
        )

        def compression_fn_nn(d, p): 
            return postprocess_fn_p(net(preprocess_fn_d(d))) # Ignore parameter kwarg for NN

        logger.info("Using NN compression function.")

    if config.compression == "linear":
        compressor = get_linear_compressor(config, dataset)

        def compression_fn_linear(d, p): 
            return compressor(d, p)

        logger.info("Using linear compression function.")

    return compression_fn_linear if config.compression == "linear" else compression_fn_nn