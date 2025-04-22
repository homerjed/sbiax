import os
from dataclasses import dataclass, replace
from functools import partial
from typing import Callable, Optional

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
from data.common import (
    Dataset,
    get_prior,
    sample_prior,
    get_compression_fn,
    get_nn_compressor,
    get_linear_compressor,
    linearised_model,
    get_linearised_data,
    get_datavector,
    freeze_out_parameters_dataset, 
    hartlap,
    get_parameter_strings
)
from sbiax.utils import marker

typecheck = jaxtyped(typechecker=typechecker)


"""
    Data
"""


@typecheck
def get_raw_data(
    data_dir: str, verbose: bool = False
) -> tuple[
    Float[np.ndarray, "z 15000 R d"],
    Float[np.ndarray, "z 2000 R d"],
    Float[np.ndarray, "2000 p"],
    Float[np.ndarray, "500 z p R 2 d"]
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
        os.path.join(data_dir, "cumulants_derivatives_plus_minus.npy") # (n, z, p, R, pm, d)
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
    derivatives: Float[np.ndarray, "500 z 5 R d"], # Check redshift / parameter axes...
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

                # Order of input cumulants (e.g. variance, skewness, kurtosis)
                cumulant_orders = [2, 3, 4] 

                var = cumulants[0] # Broadcast? e.g. [..., :1]

                # Derivatives or sims => choose last axis, last axis is `order_idx` length
                # Only reduce cumulants of higher order than variance
                if reduce:
                    for cumulant_index, _ in zip(range(cumulants.shape[0]), order_idx): 
                        if (cumulant_index > 0):
                            # E.g. for skewness (n=3); skewness_reduced = skewness / (var ** 2)
                            order = cumulant_orders[cumulant_index]

                            # S_n = k_n / (k_2 ** (n - 1)) 
                            cumulants[cumulant_index] = cumulants[cumulant_index] / (var ** (order - 1))

                # cumulants = cumulants / 100. 

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
                                # Float[np.ndarray, "n z 5 R d"]
                                simulations[n, z_i, :, r_i, order_idx], # Redshift axis is 3rd, parameter axis is 2nd
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
        n_cumulants = len(order_idx)

        if verbose:
            print("z_idx:", z_idx)
            print("R_idx:", R_idx)

        # fiducial_pdfs_z_R = get_R_z(
        #     fiducial_pdfs, 
        #     z_idx=z_idx, 
        #     R_idx=R_idx, 
        #     order_idx=order_idx,
        #     n_scales=n_scales,
        #     reduced_cumulants=reduced_cumulants
        # )
        # latin_pdfs_z_R = get_R_z(
        #     latin_pdfs, 
        #     z_idx=z_idx, 
        #     R_idx=R_idx, 
        #     order_idx=order_idx,
        #     n_scales=n_scales,
        #     reduced_cumulants=reduced_cumulants
        # )
        # derivatives_z_R = get_R_z(
        #     derivatives, 
        #     z_idx=z_idx, 
        #     R_idx=R_idx, 
        #     order_idx=order_idx,
        #     n_scales=n_scales,
        #     reduced_cumulants=reduced_cumulants, 
        #     are_derivatives=True
        # )
    """

    # Do it the same as in bulk PDFS....

    if isinstance(z_idx, int):
        z_idx = [z_idx]

    n_scales = len(R_idx)
    n_redshifts = len(z_idx)
    n_cumulants = len(order_idx)

    if verbose:
        print("z_idx:", z_idx)
        print("R_idx:", R_idx)
        print("order_idx", order_idx)

    @typecheck
    def _maybe_reduce(
        cumulants: Float[np.ndarray, "c"], 
        order_idx: list[int],
        reduce: bool = False
    ) -> Float[np.ndarray, "c"]:
        # Calculate reduced cumulants from cumulants if required
        if reduce:
            # Order of input cumulants (e.g. variance, skewness, kurtosis)
            cumulant_orders = [2, 3, 4] 
            # Only reduce cumulants of higher order than variance
            var = cumulants[0] 
            for cumulant_index, _ in zip(range(cumulants.shape[0]), order_idx): 
                if (cumulant_index > 0):
                    # E.g. for skewness (n=3); skewness_reduced = skewness / (var ** 2)
                    order = cumulant_orders[cumulant_index]
                    # S_n = k_n / (k_2 ** (n - 1)) 
                    cumulants[cumulant_index] = cumulants[cumulant_index] / (var ** (order - 1))
        return cumulants 

    def _get_bar(n_s):
        if verbose:
            bar = trange(
                n_s, desc="reduced_cumulants" if reduced_cumulants else "cumulants"
            ) 
        else: 
            bar = range(n_s)
        return bar

    fiducial_pdfs_z_R = np.zeros((fiducial_pdfs.shape[1], n_scales * n_redshifts * n_cumulants))
    for n in _get_bar(fiducial_pdfs.shape[1]):
        for z, z_i in enumerate(z_idx):
            for r, r_i in enumerate(R_idx):

                _slice = z * n_scales + r # NOTE: These must be positions in new array

                # Shape (3,)
                simulation = _maybe_reduce(
                    # Float[np.ndarray, "z n R d"]
                    fiducial_pdfs[z_i, n, r_i, order_idx], 
                    order_idx=order_idx,
                    reduce=reduced_cumulants
                )

                fiducial_pdfs_z_R[n, _slice * n_cumulants : (_slice + 1) * n_cumulants] = simulation

    latin_pdfs_z_R = np.zeros((latin_pdfs.shape[1], n_scales * n_redshifts * n_cumulants))
    for n in _get_bar(latin_pdfs.shape[1]):
        for z, z_i in enumerate(z_idx):
            for r, r_i in enumerate(R_idx):

                _slice = z * n_scales + r # NOTE: These must be positions in new array

                # Shape (3,)
                simulation = _maybe_reduce(
                    # Float[np.ndarray, "z n R d"]
                    latin_pdfs[z_i, n, r_i, order_idx], 
                    order_idx=order_idx,
                    reduce=reduced_cumulants
                )

                latin_pdfs_z_R[n, _slice * n_cumulants : (_slice + 1) * n_cumulants] = simulation

    derivatives_z_R = np.zeros((derivatives.shape[0], 5, n_scales * n_redshifts * n_cumulants))
    for n in _get_bar(derivatives.shape[0]):
        for z, z_i in enumerate(z_idx):
            for r, r_i in enumerate(R_idx):

                _slice = z * n_scales + r # NOTE: These must be positions in new array

                # Shape (5, 3)
                for p in range(5):
                    simulation = _maybe_reduce(
                        # Float[np.ndarray, "n z 5 R d"]
                        derivatives[n, z_i, p, r_i, order_idx], # Redshift axis is 3rd, parameter axis is 2nd
                        order_idx=order_idx,
                        reduce=reduced_cumulants
                    )
                    derivatives_z_R[n, p, _slice * n_cumulants : (_slice + 1) * n_cumulants] = simulation

    if verbose:
        print(
            "Processed data shapes (fids., latins, derivs.):", 
            [_.shape for _ in [fiducial_pdfs_z_R, latin_pdfs_z_R, derivatives]]
        )

    return fiducial_pdfs_z_R, latin_pdfs_z_R, derivatives_z_R


def get_cumulant_data(
    config: ConfigDict, *, verbose: bool = False, results_dir: Optional[str] = None
) -> Dataset:

    @typecheck
    def calculate_derivatives(
        derivatives_pm: Float[np.ndarray, "500 z p R 2 d"], 
        alpha: Float[np.ndarray, "p"], 
        dparams: Float[np.ndarray, "p"], 
        parameter_strings: list[str], 
        parameter_derivative_names: list[list[str]], 
        *, 
        verbose: bool = False
    ) -> Float[np.ndarray, "500 z 5 R d"]:

        # (n, z, p, R, 2, d) -> (n, z, p, R, d)
        derivatives = derivatives_pm[..., 1, :] - derivatives_pm[..., 0, :] 

        for p in range(alpha.size):
            if verbose:
                print(
                    "Parameter strings / dp / dp_name", 
                    parameter_strings[p], dparams[p], parameter_derivative_names[p]
                )
            derivatives[:, :, p, ...] = derivatives[:, :, p, ...] / dparams[p] # NOTE: OK before or after reducing cumulants

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

    if config.freeze_parameters:
        print("Freezing all but Om, s8")
        dataset = freeze_out_parameters_dataset(dataset)

    if config.use_pca:
        dataset = pca_dataset(dataset)

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
        else:
            print("Using non-linearised model, non-Gaussian noise.")

    # E.g. using non-linear model and Gaussian noise or what?
    if hasattr(config, "nonlinearised"):
        if config.nonlinearised:
            print("Using linearised model, non-Gaussian noise.") 
            # NOTE: does `get_datavector()` do this also?
            D, Y = get_nonlinearised_data(config)

            dataset = replace(dataset, data=D, parameters=Y)

    return dataset


@typecheck
def get_nonlinearised_data(config: ConfigDict) -> tuple[Float[Array, "n d"], Float[Array, "n p"]]:
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

        print("CUMULANTS DATASET")
        print(">DATA:\n\t", ["{:.3E} {:.3E}".format(_.min(), _.max()) for _ in (self.data.fiducial_data, self.data.data)])
        print(">DATA / PARAMETERS:\n\t", [_.shape for _ in (self.data.data, self.data.parameters)])

    def get_parameter_strings(self) -> list[str]:
        return get_parameter_strings()

    def sample_prior(self, key: PRNGKeyArray, n: int, *, hypercube: bool = True) -> Float[Array, "n p"]:
        # Sample Quijote prior which may not be the same as inference prior
        P = sample_prior(
            key, 
            n, 
            alpha=self.data.alpha, 
            lower=self.data.lower, 
            upper=self.data.upper, 
            hypercube=hypercube
        )
        return P

    def get_compression_fn(self) -> Callable[[Float[Array, "d"], Float[Array, "p"]], Float[Array, "p"]]:
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

    def get_linearised_data(self) -> tuple[Float[Array, "n d"], Float[Array, "n p"]]:
        # Get linearised data (e.g. pre-training), where config only sets how many simulations
        return get_linearised_data(self.config, self.data)

    def get_preprocess_fn(self):
        # Get (X, P) preprocessor?
        ...

    # # Condition number regularisation
    # if config.covariance_epsilon is not None:
    #     if verbose:
    #         print("Covariance conditioning...")

    #     L = jnp.trace(C) / n_d * config.covariance_epsilon

    #     # U, S, Vt = jnp.linalg.svd(C)
    #     # L = 0.01 * S.min()
    #     # L = S.max() / 1000

    #     C = jnp.identity(n_d) * L + C