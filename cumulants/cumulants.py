import os
from dataclasses import dataclass, replace
from functools import partial
from typing import Tuple, Callable, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray, Array, Float, jaxtyped

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

from constants import get_quijote_parameters, get_save_and_load_dirs
from nn import fit_nn
from pca import PCA
from sbiax.utils import marker

typecheck = jaxtyped(typechecker=typechecker)


def hartlap(n_s: int, n_d: int) -> float: 
    return (n_s - n_d - 2) / (n_s - 1)


def get_parameter_strings() -> list[str]:
    (_, _, _, _, _, _, parameter_strings, *_) = get_quijote_parameters()
    return parameter_strings


# def remove_nuisances(dataset: Dataset) -> Dataset:
#     # Remove 'nuisance' parameters not constrained by the moments/cumulants
#     return dataset


"""
    Data
"""


@typecheck
@dataclass
class Dataset:
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


@typecheck
def get_raw_data(
    data_dir: str, verbose: bool = False
) -> Tuple[
    Float[np.ndarray, "z 15000 R d"],
    Float[np.ndarray, "z 2000 R d"],
    Float[np.ndarray, "2000 p"],
    Float[np.ndarray, "500 p z R 2 d"]
]:
    """
        Load fiducial and latin PDFs
    """

    # (z, n, d, R)
    fiducial_pdfs = np.load(
        os.path.join(data_dir, "ALL_FIDUCIAL_CUMULANTS.npy") # (z, n, d, R)
    )
    latin_pdfs = np.load(
        os.path.join(data_dir, "ALL_LATIN_CUMULANTS.npy") # (z, n, d, R)
    ) 
    latin_pdfs_parameters = np.loadtxt(
        os.path.join(data_dir, "latin_hypercube_params.txt") # (n, p)
    )
    derivatives = np.load(
        os.path.join(data_dir, "cumulants_derivatives_plus_minus.npy") # (n, p, z, R, pm, d)
    )

    if verbose:
        print("Raw data shapes:", [_.shape for _ in [fiducial_pdfs, latin_pdfs, latin_pdfs_parameters, derivatives]])

    return fiducial_pdfs, latin_pdfs, latin_pdfs_parameters, derivatives


@typecheck
def get_R_and_z_moments(
    z_idx: int, 
    R_idx: list[int], 
    fiducial_pdfs: Float[np.ndarray, "z 15000 R d"], 
    latin_pdfs: Float[np.ndarray, "z 2000 R d"], 
    derivatives: Float[np.ndarray, "500 5 z R d"],
    *, 
    order_idx: Optional[list[int]] = None,
    reduced_cumulants: bool = False,
    verbose: bool = False
) -> tuple[
    Float[np.ndarray, "15000 zRd"],
    Float[np.ndarray, "2000 zRd"],
    Float[np.ndarray, "500 5 zRd"]
]:
    """ 
        Get and stack moments for smoothing scales and redshift. 
        - select for moment order (e.g. var, skewness, kurtosis ... before final reshape)
    """

    def get_R_z(
        simulations: Float[Array, "z n R d"] | Float[Array, "n 5 z R d"], 
        z_idx: int, 
        R_idx: list[int], 
        order_idx: list[int],
        *,
        n_scales: int,
        reduced_cumulants: bool = False,
        are_derivatives: bool = False
    ) -> Array:
        # Obtain fiducial/latin/derivative cumulants at chosen redshift and scales
        # > `simulations` is either simulations or derivatives

        n_cumulants = len(order_idx)

        if verbose:
            print("Are derivatives?", are_derivatives)

        @typecheck
        def _maybe_reduce(
            cumulants: Float[np.ndarray, "c"] | Float[np.ndarray, "c 5"], 
            reduce: bool = False
        ) -> Float[np.ndarray, "c"] | Float[np.ndarray, "c 5"]:
            """ 
                Derivatives; linear operation on two +/- parameter nudged sims => still just divide? 
            """

            # Order of input cumulants (e.g. variance, skewness, kurtosis)
            cumulant_orders = [2, 3, 4] 

            var = cumulants[0] # Broadcast? e.g. [..., :1]

            # Derivatives or sims => choose last axis, last axis is `order_idx` length
            # for cumulant_index in range(cumulants.shape[-1]): 
            for cumulant_index, _ in zip(range(cumulants.shape[0]), order_idx): 

                # Only reduce cumulants of higher order than variance
                if reduce and (cumulant_index > 0):

                    # E.g. for skewness (n=3); skewness_reduced = skewness / (var ** 2)
                    order = cumulant_orders[cumulant_index]

                    # S_n = k_n / (k_2 ** (n - 1)) 
                    cumulants[cumulant_index] = cumulants[cumulant_index] / (var ** (order - 1))

            return cumulants 

        # n_s is number of derivatives or number of simulations (latin / fiducial)
        if are_derivatives:
            n_s, *_ = simulations.shape

            R_z_simulations = np.zeros((n_s, 5, n_scales * n_redshifts * n_cumulants))
        else:
            _, n_s, *_ = simulations.shape

            R_z_simulations = np.zeros((n_s, n_scales * n_redshifts * n_cumulants))

        if verbose:
            bar = trange(
                n_s, desc="reduced_cumulants" if reduced_cumulants else "cumulants"
            ) 
        else: 
            bar = range(n_s)

        for n in bar:
            for z, z_i in enumerate(z_idx):
                for r, r_i in enumerate(R_idx):

                    _slice = z * n_scales + r # NOTE: These must be positions in new array

                    if are_derivatives:
                        # Shape (5, 3)
                        simulation = _maybe_reduce(
                            # Float[np.ndarray, "n 5 z R d"]
                            simulations[n, :, z_i, r_i, order_idx], # Redshift axis is 3rd, parameter axis is 2nd
                            reduce=reduced_cumulants
                        )
                        simulation = np.transpose(simulation)

                        R_z_simulations[n, :, _slice * n_cumulants : (_slice + 1) * n_cumulants] = simulation
                    else:
                        # Shape (3,)
                        simulation = _maybe_reduce(
                            # Float[np.ndarray, "z n R d"]
                            simulations[z_i, n, r_i, order_idx], 
                            reduce=reduced_cumulants
                        )

                        R_z_simulations[n, _slice * n_cumulants : (_slice + 1) * n_cumulants] = simulation

        return R_z_simulations

    if isinstance(z_idx, int):
        z_idx = [z_idx]

    n_scales = len(R_idx)
    n_redshifts = len(z_idx)

    if verbose:
        print("z_idx:", z_idx)
        print("R_idx:", R_idx)

    n_cumulants = len(order_idx)

    fiducial_pdfs_z_R = get_R_z(
        fiducial_pdfs, 
        z_idx=z_idx, 
        R_idx=R_idx, 
        order_idx=order_idx,
        n_scales=n_scales,
        reduced_cumulants=reduced_cumulants
    )
    latin_pdfs_z_R = get_R_z(
        latin_pdfs, 
        z_idx=z_idx, 
        R_idx=R_idx, 
        order_idx=order_idx,
        n_scales=n_scales,
        reduced_cumulants=reduced_cumulants
    )
    derivatives_z_R = get_R_z(
        derivatives, 
        z_idx=z_idx, 
        R_idx=R_idx, 
        order_idx=order_idx,
        n_scales=n_scales,
        reduced_cumulants=reduced_cumulants, 
        are_derivatives=True
    )

    if verbose:
        print(
            "Processed data shapes:", 
            [_.shape for _ in [fiducial_pdfs_z_R, latin_pdfs_z_R, derivatives]]
        )

    return fiducial_pdfs_z_R, latin_pdfs_z_R, derivatives_z_R


def get_cumulant_data(
    config: ConfigDict, *, verbose: bool = False, results_dir: Optional[str] = None
) -> Dataset:

    @typecheck
    def calculate_derivatives(
        derivatives_pm: Float[np.ndarray, "500 5 z R 2 d"], 
        alpha: Float[np.ndarray, "p"], 
        dparams: Float[np.ndarray, "p"], 
        parameter_strings: list[str], 
        parameter_derivative_names: list[list[str]], 
        *, 
        verbose: bool = False
    ) -> Float[np.ndarray, "500 5 z R d"]:

        derivatives = derivatives_pm[..., 1, :] - derivatives_pm[..., 0, :] 

        for p in range(alpha.size):
            if verbose:
                print(
                    "Parameter strings / dp / dp_name", 
                    parameter_strings[p], dparams[p], parameter_derivative_names[p]
                )
            derivatives[:, p, ...] = derivatives[:, p, ...] / dparams[p] # NOTE: OK before or after reducing cumulants

        assert derivatives.ndim == 5, "{}".format(derivatives.shape)

        return derivatives

    data_dir, *_ = get_save_and_load_dirs()

    (
        all_R_values,
        all_redshifts,
        resolution,
        alpha,
        lower,
        upper,
        parameter_strings,
        redshift_strings,
        parameter_derivative_names,
        dparams,
        deltas,
        delta_bin_edges,
        D_deltas 
    ) = get_quijote_parameters()

    R_idx = [all_R_values.index(R) for R in config.scales]
    z_idx = all_redshifts.index(config.redshift)

    if verbose:
        print("z_idx:", z_idx)
        print("R_idx:", R_idx)

    # Raw moments
    (
        fiducial_moments,
        latin_moments, 
        latin_moments_parameters,
        derivatives_pm
    ) = get_raw_data(data_dir, verbose=verbose)

    # Euler derivative from plus minus statistics (NOTE: derivatives: Float[np.ndarray, "500 p z R 2 d"])
    derivatives = calculate_derivatives(
        derivatives_pm, 
        alpha, 
        dparams, 
        parameter_strings=parameter_strings, 
        parameter_derivative_names=parameter_derivative_names, 
        verbose=verbose
    )

    # Grab and stack by redshift and scales
    (
        fiducial_moments_z_R,
        latin_moments_z_R,
        derivatives
    ) = get_R_and_z_moments(
        z_idx, 
        R_idx, 
        fiducial_moments, 
        latin_moments, 
        derivatives=derivatives,
        order_idx=config.order_idx,
        reduced_cumulants=config.reduced_cumulants,
        verbose=verbose
    )

    n_s, n_d = fiducial_moments_z_R.shape 

    # Calculate covariance for datavector of each fiducial pdfs for all chosen scales 
    C = np.cov(fiducial_moments_z_R, rowvar=False) # NOTE: correctly calculates covariance of reduced or not

    # Condition number regularisation
    if config.covariance_epsilon is not None:
        if verbose:
            print("Covariance conditioning...")

        L = jnp.trace(C) / n_d * config.covariance_epsilon

        # U, S, Vt = jnp.linalg.svd(C)
        # L = 0.01 * S.min()
        # L = S.max() / 1000

        C = jnp.identity(n_d) * L + C

    assert np.all(np.isfinite(C)), "Bad covariance."
    assert derivatives.shape[:-1] == (500, 5), "Do derivatives have batch axis? Required."

    # Precision, corrected with Hartlap
    H = hartlap(n_s, n_d)
    Cinv = H * np.linalg.inv(C) # Cinv = jnp.linalg.svd(C) * H

    # Fisher information matrix; all scales, one redshift
    dmu = np.mean(derivatives, axis=0)
    F = np.linalg.multi_dot([dmu, Cinv, dmu.T])
    Finv = np.linalg.inv(F)

    dataset = Dataset(
        alpha=jnp.asarray(alpha),
        lower=jnp.asarray(lower),
        upper=jnp.asarray(upper),
        parameter_strings=parameter_strings,
        Finv=jnp.asarray(Finv),
        Cinv=jnp.asarray(Cinv),
        C=jnp.asarray(C),
        fiducial_data=jnp.asarray(fiducial_moments_z_R),
        data=jnp.asarray(latin_moments_z_R),
        parameters=jnp.asarray(latin_moments_parameters),
        derivatives=jnp.asarray(derivatives)  
    )

    if verbose:
        corr_matrix = np.corrcoef(fiducial_moments_z_R, rowvar=False) + 1e-6 # Log colouring

        print("Covariance condition number: {:.3E}".format(jnp.linalg.cond(C)))
        print("Dtype of moments", fiducial_moments_z_R.dtype)

        plt.figure()
        norm = mcolors.LogNorm(vmin=np.min(corr_matrix[corr_matrix > 0]), vmax=np.max(corr_matrix)) 
        plt.imshow(corr_matrix, norm=norm, cmap="bwr")
        plt.colorbar()
        plt.axis("off")
        plt.savefig("moments_corr_matrix_log.png", bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.imshow(corr_matrix, cmap="bwr", vmin=-1., vmax=1.)
        plt.colorbar()
        plt.axis("off")
        plt.savefig(
            os.path.join(
                results_dir if results_dir is not None else "", "moments_corr_matrix.png"
            ), 
            bbox_inches="tight"
        )
        plt.close()

        # Plot Fisher forecast
        c = ChainConsumer()
        c.add_chain(
            Chain.from_covariance(
                dataset.alpha,
                dataset.Finv,
                columns=dataset.parameter_strings,
                name=r"$F_{\Sigma^{-1}}$",
                color="k",
                linestyle=":",
                shade_alpha=0.
            )
        )
        c.add_marker(
            location=marker(dataset.alpha, parameter_strings=dataset.parameter_strings),
            name=r"$\alpha$", 
            color="#7600bc"
        )
        fig = c.plotter.plot()
        plt.savefig(
            os.path.join(
                results_dir if results_dir is not None else "fisher_forecasts/", 
                "fisher_forecast_{}_z={}_R={}_m={}.png".format(
                    config.linearised, 
                    config.redshift, 
                    "".join(map(str, config.order_idx)),
                    "".join(map(str, config.scales))
                )
            ), 
        )
        plt.close()

    return dataset


@typecheck
def get_prior(config: ConfigDict, dataset: Dataset) -> tfd.Distribution:

    lower = jnp.asarray(dataset.lower)
    upper = jnp.asarray(dataset.upper)

    assert jnp.all((upper - lower) > 0.)

    print("Using flat prior")
    prior = tfd.Blockwise(
        # [tfd.Uniform(lower[p], upper[p]) for p in range(dataset.alpha.size)]
        [tfd.Uniform(-1e4, 1e4) for p in range(dataset.alpha.size)]
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

    # Avoid tfp warning
    lower = lower.astype(jnp.float32)
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
def get_linearised_data(config: ConfigDict, dataset: Dataset) -> Tuple[Float[Array, "n d"], Float[Array, "n p"]]:
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
def get_nonlinearised_data(config: ConfigDict) -> Tuple[Float[Array, "n d"], Float[Array, "n p"]]:
    """
        Get non-linearised PDFs. 
        - use linearised model at a random fiducial pdf noise realisation
    """
    key = jr.key(config.seed)

    assert config.n_linear_sims <= 15_000, (
        "Must use n_linear_sims (={}) <= 15,000. Quijote limit.".format(config.n_linear_sims)
    )

    # NOTE: This is not simply returning fiducials, latins from Quijote...
    if 0:
        dataset: Dataset = get_cumulant_data(config)

        if config.n_linear_sims is not None:
            Y = sample_prior(
                key, 
                config.n_linear_sims, 
                dataset.alpha, 
                dataset.lower, 
                dataset.upper
            )

        assert dataset.derivatives.ndim == 3, (
            "Do derivatives ({}) have batch axis? Required.".format(dataset.derivatives.shape)
        )

        dmu = jnp.mean(dataset.derivatives, axis=0)

        @typecheck
        def _simulator(
            key: PRNGKeyArray, 
            xi_0_i: Float[Array, "d"], 
            pi: Float[Array, "p"]
        ) -> Float[Array, "d"]:
            # Data model for non-linear expectation
            _mu = linearised_model(dataset.alpha, pi, mu=xi_0_i, dmu=dmu)
            return jr.multivariate_normal(key, mean=_mu, cov=dataset.C)

        keys = jr.split(key, len(Y))
        D = jax.vmap(_simulator)(keys, dataset.fiducial_data[:len(Y)], Y) 

    # Use latin dataset straight out of Quijote
    dataset: Dataset = get_cumulant_data(config)

    # Default dataset is non-linear Quijote data
    D = dataset.data
    Y = dataset.parameters

    return D, Y 


@typecheck
def get_data(config: ConfigDict, *, verbose: bool = False, results_dir: Optional[str] = None) -> Dataset:
    """ 
        Get data for linearised-model data or full simulation data. 
        - Start with Quijote default data; linearise or nonlinearise 
          if required
    """

    dataset: Dataset = get_cumulant_data(
        config, verbose=verbose, results_dir=results_dir
    )

    if hasattr(config, "linearised"):
        if config.linearised:
            print("Using linearised model, Gaussian noise.")
            D, Y = get_linearised_data(config, dataset) 

            dataset = replace(dataset, data=D, parameters=Y)

    # E.g. using non-linear model and Gaussian noise or what?
    if hasattr(config, "nonlinearised"):
        if config.nonlinearised:
            print("Using linearised model, non-Gaussian noise.") 
            # NOTE: does `get_datavector()` do this also?
            D, Y = get_nonlinearised_data(config)

            dataset = replace(dataset, data=D, parameters=Y)

    return dataset


@typecheck
def get_datavector(
    key: PRNGKeyArray, config: ConfigDict, dataset: Dataset, n: int = 1
) -> Float[Array, "... d"]:
    """ Measurement: either Gaussian linear model or not """

    # Choose a linearised model datavector or simply one of the Quijote realisations
    # which corresponds to a non-linearised datavector with Gaussian noise
    if not config.use_expectation:
        if config.linearised:
            mu = jnp.mean(dataset.fiducial_data, axis=0)

            print("Using linearised datavector")
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


def get_nn_compressor(key, dataset, data_preprocess_fn=None, *, results_dir):
    net_key, train_key = jr.split(key)

    if data_preprocess_fn is None:
        data_preprocess_fn = lambda x: x

    net = eqx.nn.MLP(
        dataset.data.shape[-1], 
        dataset.parameters.shape[-1], 
        width_size=32, 
        depth=1, 
        activation=jax.nn.tanh,
        key=net_key
    )

    def preprocess_fn(x): 
        # Preprocess with covariance?
        return (jnp.asarray(x) - jnp.mean(dataset.data, axis=0)) / jnp.std(dataset.data, axis=0)

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
    # Get linear or neural network compressor
    if config.compression == "nn":

        net, preprocess_fn = get_nn_compressor(
            key, dataset, results_dir=results_dir
        )

        compressor = lambda d, p: net(preprocess_fn(d)) # Ignore parameter kwarg!

    if config.compression == "linear":
        compressor = get_linear_compressor(config, dataset)

    # Fit PCA transform to simulated data and apply after compressing
    if config.use_pca:

        # Compress simulations as usual 
        X = jax.vmap(compressor)(dataset.data, dataset.parameters)

        # Standardise before PCA (don't get tricked by high variance due to units)
        X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)

        # Fit whitening-PCA to compressed simulations
        pca = PCA(num_components=dataset.alpha.size) 
        pca.fit(X) # Fit on fiducial data?
        
        # Reparameterize compression with both transforms
        compression_fn = lambda d, p: pca.transform(compressor(d, p))
    else:
        compression_fn = lambda d, p: compressor(d, p)

    return compression_fn 


"""
    Dataset
"""


@dataclass
class CumulantsDataset:
    """ 
        Dataset for Simulation-Based Inference with cumulants of the matter PDF 
    """

    config: ConfigDict
    data: Dataset
    prior: tfd.Distribution
    compression_fn: Callable
    results_dir: str

    def __init__(
        self, 
        config: ConfigDict, 
        *, 
        verbose: bool = False, 
        results_dir: Optional[str] = None
    ):
        self.config = config

        self.data = get_data(
            config, verbose=verbose, results_dir=results_dir
        )

        self.prior = get_prior(config, self.data) # Possibly not equal to Quijote prior

        key = jr.key(config.seed)
        self.compression_fn = get_compression_fn(
            key, self.config, self.data, results_dir=results_dir
        )

        self.results_dir = results_dir

    def get_parameter_strings(self):
        return get_parameter_strings()

    def sample_prior(self, key: PRNGKeyArray, n: int, *, hypercube: bool = True) -> Float[Array, "n p"]:
        # Sample Quijote prior which may not be the same as inference prior
        P = sample_prior(
            key, 
            n, 
            self.data.alpha, 
            self.data.lower, 
            self.data.upper, 
            hypercube=hypercube
        )
        return P

    def get_compression_fn(self):
        return self.compression_fn

    def get_datavector(self, key: PRNGKeyArray, n: int = 1) -> Float[Array, "... d"]:
        d = get_datavector(key, self.config, self.data, n)
        return d

    def get_linearised_datavector(self, key: PRNGKeyArray, n: int = 1) -> Float[Array, "... d"]:
        # Sample datavector from linearised Gaussian model
        mu = jnp.mean(self.data.fiducial_data, axis=0) 
        d = jr.multivariate_normal(key, mu, self.data.C, (n,))
        if not (n > 1):
            d = jnp.squeeze(d, axis=0) 
        return d

    def get_linearised_data(self):
        # Get linearised data (e.g. pre-training), where config only sets how many simulations
        return get_linearised_data(self.config, self.data)

    def get_preprocess_fn(self):
        # Get (X, P) preprocessor?
        ...