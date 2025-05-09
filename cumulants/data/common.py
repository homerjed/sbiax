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
from beartype.door import is_bearable
import numpy as np
from scipy.stats import qmc
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from chainconsumer import Chain, ChainConsumer, Truth
import tensorflow_probability.substrates.jax.distributions as tfd
from tqdm.auto import trange

from data.constants import get_quijote_parameters, get_save_and_load_dirs, get_target_idx
from compression.nn import fit_nn, fit_nn_lbfgs
from compression.pca import PCA
from sbiax.utils import marker

typecheck = jaxtyped(typechecker=typechecker)


"""
    Objects common to the PDF and cumulant datasets
"""


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


def convert_dataset_to_jax(dataset: Dataset) -> Dataset:
    return jax.tree.map(
        lambda a: jnp.asarray(a), 
        dataset, 
        is_leaf=lambda a: isinstance(a, np.ndarray)
    )


def pca_dataset(dataset: Dataset) -> Dataset:

    # Compress simulations as usual 
    # X = jax.vmap(compressor)(dataset.data, dataset.parameters) # Fit PCA to latins
    # X = jax.vmap(compressor, in_axes=(0, None))(dataset.fiducial_data, dataset.alpha) # Fit PCA to fiducials

    # Standardise before PCA (don't get tricked by high variance due to units)
    X = dataset.fiducial_data
    mu_X = jnp.mean(X, axis=0)
    std_X = jnp.std(X, axis=0)

    def _preprocess_fn(X):
        return X #(X - mu_X) / std_X

    # Fit whitening-PCA to compressed simulations
    pca = PCA(num_components=dataset.fiducial_data.shape[-1]) 
    pca.fit(_preprocess_fn(X)) # Fit on fiducial data?
    
    # Reparameterize compression with both transforms
    pca_fn = lambda d: pca.transform(_preprocess_fn(d))

    derivatives = jnp.zeros_like(dataset.derivatives)
    for p in range(dataset.alpha.size):
        derivatives = derivatives.at[:, p, :].set(
            jax.vmap(pca_fn)(dataset.derivatives[:, p, :])
        )

    fiducial_data = jax.vmap(pca_fn)(dataset.fiducial_data)

    C = jnp.cov(fiducial_data, rowvar=False)
    Cinv = jnp.linalg.inv(C)
    _derivatives = jnp.mean(derivatives, axis=0)
    Finv = jnp.linalg.multi_dot([_derivatives, Cinv, _derivatives.T])

    frozen_dataset = Dataset(
        name=dataset.name,
        alpha=dataset.alpha,
        lower=dataset.lower,
        upper=dataset.upper,
        parameter_strings=dataset.parameter_strings,
        Finv=Finv,
        Cinv=Cinv,
        C=C,
        fiducial_data=fiducial_data,
        data=jax.vmap(pca_fn)(dataset.data),
        parameters=dataset.parameters,
        derivatives=derivatives
    )

    return frozen_dataset 


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
        def _freeze_parameters(pdf_or_cumulant: Float[Array, "d"], p: Float[Array, "p"]) -> Float[Array, "d"]:
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
        print("Using flat prior")
        lower = jnp.ones((dataset.alpha.size,)) * -1e-4
        upper = jnp.ones((dataset.alpha.size,)) * 1e-4
    else:
        print("Using Quijote uniform prior")
        lower = jnp.asarray(dataset.lower) # Avoid tfp warning
        upper = jnp.asarray(dataset.upper)

    assert jnp.all((upper - lower) > 0.)

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
        print("Hypercube sampling...")
        sampler = qmc.LatinHypercube(d=alpha.size)
        samples = sampler.random(n=n_linear_sims)
        Y = jnp.asarray(qmc.scale(samples, lower, upper))
    else:
        print("Uniform box sampling...")
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
    dataset: Dataset
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

    key = jr.key(config.seed)

    key_parameters, key_simulations = jr.split(key)

    if config.n_linear_sims is not None:
        Y = sample_prior(
            key_parameters, 
            config.n_linear_sims, 
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

    print("Get linearised data", D.shape, Y.shape)

    return D, Y 


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
    if (not config.use_expectation) or (not use_expectation):
        if config.linearised:
            print("Using linearised datavector")
            mu = jnp.mean(dataset.fiducial_data, axis=0)

            datavector = jr.multivariate_normal(key, mean=mu, cov=dataset.C, shape=(n,))
        else:
            print("Using non-linearised datavector")
            datavector = jr.choice(key, dataset.fiducial_data, shape=(n,))
    else:
        print("Using expectation (noiseless datavector)")
        datavector = jnp.mean(dataset.fiducial_data, axis=0, keepdims=True)

    if not (n > 1):
        datavector = jnp.squeeze(datavector, axis=0) 

    return datavector # Remove batch axis by default


"""
    Compression
"""


@typecheck
def get_linear_compressor(
    config: ConfigDict, dataset: Dataset
) -> Callable[[Float[Array, "d"], Float[Array, "p"]], Float[Array, "p"]]:
    """ 
        Get Chi^2 minimisation function; compressing datavector 
        at estimated parameters to summary 
    """

    @typecheck
    def mle(
        d: Float[Array, "d"], 
        pi: Float[Array, "p"], 
        Finv: Float[Array, "p p"], 
        mu: Float[Array, "d"], 
        dmu: Float[Array, "p d"], 
        precision: Float[Array, "d d"]
    ) -> Float[Array, "p"]:
        """
            Calculates a maximum likelihood estimator (MLE) from a datavector by
            assuming a linear model `mu` in parameters `pi` and using

            Args:
                d (`Array`): The datavector to compress.
                p (`Array`): The estimated parameters of the datavector (e.g. a fiducial set).
                Finv (`Array`): The Fisher matrix. Calculated with a precision matrix (e.g. `precision`) and 
                    theory derivatives.
                mu (`Array`): The model evaluated at the estimated set of parameters `pi`.
                dmu (`Array`): The first-order theory derivatives (for the implicitly assumed linear model, 
                    these are parameter independent!)
                precision (`Array`): The precision matrix - defined as the inverse of the data covariance matrix.

            Returns:
                `Array`: the MLE.
        """
        return pi + jnp.linalg.multi_dot([Finv, dmu, precision, d - mu])

    @typecheck
    def compressor(
        d: Float[Array, "d"], 
        p: Float[Array, "p"],
        mu: Float[Array, "d"], 
        dmu: Float[Array, "p d"]
    ) -> Float[Array, "p"]: 
        mu_p = linearised_model(
            alpha=dataset.alpha, alpha_=p, mu=mu, dmu=dmu
        )
        p_ = mle(
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


@typecheck
def get_nn_compressor(
    key: PRNGKeyArray, 
    dataset: Dataset, 
    data_preprocess_fn: Optional[Callable] = None, 
    *, 
    lbfgs: bool = False, 
    results_dir: str, 
    net: Optional[eqx.Module] = None
) -> tuple[eqx.Module, Callable]:
    """
        Train neural network compression function
        - Optionally use parameter covariance for chi2 loss
    """
    net_key, train_key = jr.split(key)

    if data_preprocess_fn is None:
        data_preprocess_fn = lambda x: x

    if net is None:
        net = eqx.nn.MLP(
            dataset.data.shape[-1], 
            dataset.parameters.shape[-1], 
            width_size=32, 
            depth=3, 
            activation=jax.nn.tanh,
            key=net_key
        )

    def preprocess_fn(x): 
        # Preprocess with covariance?
        return (jnp.asarray(x) - jnp.mean(dataset.data, axis=0)) / jnp.std(dataset.data, axis=0)

    if lbfgs:
        net, losses = fit_nn_lbfgs(
            train_key, 
            net, 
            (preprocess_fn(data_preprocess_fn(dataset.data)), dataset.parameters), 
            # precision=jnp.linalg.inv(dataset.Finv) # In reality this varies with parameters
        )
    else:
        net, losses = fit_nn(
            train_key, 
            net, 
            (preprocess_fn(data_preprocess_fn(dataset.data)), dataset.parameters), 
            opt=optax.adam(1e-3), 
            precision=jnp.linalg.inv(dataset.Finv), # In reality this varies with parameters
            n_batch=500, 
            patience=1000,
            n_steps=50_000
        )

    plt.figure()
    plt.loglog(losses)
    plt.savefig(os.path.join(results_dir, "losses_nn.png"))
    plt.close()

    return net, preprocess_fn


def get_compression_fn(key, config, dataset, *, results_dir):
    """ 
        Get linear or neural network compressor
    """ 
    assert config.compression in ["linear", "nn", "nn-lbfgs"]

    if config.compression == "nn" or config.compression == "nn-lbfgs":

        net, preprocess_fn = get_nn_compressor(
            key, dataset, lbfgs=config.compression == "nn-lbfgs", results_dir=results_dir
        )

        compressor = lambda d, p: net(preprocess_fn(d)) # Ignore parameter kwarg!

    if config.compression == "linear":
        compressor = get_linear_compressor(config, dataset)

    # Fit PCA transform to simulated data and apply after compressing
    # if config.use_pca:

    #     # Compress simulations as usual 
    #     # X = jax.vmap(compressor)(dataset.data, dataset.parameters) # Fit PCA to latins
    #     X = jax.vmap(compressor, in_axes=(0, None))(dataset.fiducial_data, dataset.alpha) # Fit PCA to fiducials

    #     # Standardise before PCA (don't get tricked by high variance due to units)
    #     mu_X = jnp.mean(X, axis=0)
    #     std_X = jnp.std(X, axis=0)

    #     def _preprocess_fn(X):
    #         return (X - mu_X) / std_X

    #     # Fit whitening-PCA to compressed simulations
    #     pca = PCA(num_components=dataset.alpha.size) 
    #     pca.fit(_preprocess_fn(X)) # Fit on fiducial data?
        
    #     # Reparameterize compression with both transforms
    #     compression_fn = lambda d, p: pca.transform(_preprocess_fn(compressor(d, p)))
    # else:
    #     compression_fn = lambda d, p: compressor(d, p)

    compression_fn = lambda d, p: compressor(d, p)

    return compression_fn 