import os
from dataclasses import dataclass, replace
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray, Array, Float, jaxtyped

from beartype import beartype as typechecker 
import numpy as np
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
import tensorflow_probability.substrates.jax.distributions as tfd
from tqdm.auto import trange

from configs.log import setup_module_logger, get_log_level
from data.constants import get_quijote_parameters, get_save_and_load_dirs, ALPHA
from data.common import (
    Dataset,
    get_prior,
    sample_prior,
    get_compression_fn,
    linearised_model,
    get_linearised_data,
    get_datavector,
    freeze_out_parameters_dataset, 
    hartlap,
    get_parameter_strings
)

typecheck = jaxtyped(typechecker=typechecker)

logger, log_figs_dir = setup_module_logger(__name__, level=get_log_level())

FORCE_RECOMPUTE_DATASET = True if os.environ.get("FORCE_RECOMPUTE_DATASET", "").lower() in ("1", "true") else False 
USE_QUIJOTE_TAILS = True if os.environ.get("USE_QUIJOTE_TAILS", "").lower() in ("1", "true") else False
FIDUCIAL_REDUCE = True if os.environ.get("FIDUCIAL_REDUCE", "").lower() in ("1", "true") else False

"""
    Data
"""


@typecheck
def get_raw_data(data_dir: str) -> tuple[
    Float[np.ndarray, "z 15000 R d"],
    Float[np.ndarray, "z 2000 R d"],
    Float[np.ndarray, "2000 p"],
    Float[np.ndarray, "500 z p R 2 d"]
]:
    """
        Load fiducial and latin PDFs loaded out of Quijote files
    """

    # (z, n, d, R)
    fiducial_pdfs = np.load(
        os.path.join(data_dir, "raw/ALL_FIDUCIAL_CUMULANTS.npy") # (z, n, d, R)
    )
    latin_pdfs = np.load(
        os.path.join(data_dir, "raw/ALL_LATIN_CUMULANTS.npy") # (z, n, d, R)
    ) 
    latin_pdfs_parameters = np.loadtxt(
        os.path.join(data_dir, "raw/latin_hypercube_params.txt") # (n, p)
    )
    derivatives = np.load(
        os.path.join(data_dir, "raw/cumulants_derivatives_plus_minus.npy") # (n, z, p, R, pm, d)
    )

    logger.info("Raw data shapes:", [_.shape for _ in [fiducial_pdfs, latin_pdfs, latin_pdfs_parameters, derivatives]])

    return fiducial_pdfs, latin_pdfs, latin_pdfs_parameters, derivatives


@typecheck
def get_R_and_z_moments(
    z_idx: int, 
    R_idx: list[int], 
    fiducial_pdfs: Float[np.ndarray, "z 15000 R d"], 
    latin_pdfs: Float[np.ndarray, "z 2000 R d"], 
    derivatives: Float[np.ndarray, "500 z 5 R d"], # Check redshift / parameter axes...
    *, 
    order_idx: Optional[list[int]] = None
) -> tuple[
    Float[np.ndarray, "15000 zRd"],
    Float[np.ndarray, "2000 zRd"],
    Float[np.ndarray, "500 5 zRd"]
]:
    """ 
        Get and stack moments for smoothing scales and redshift. 
        - select for moment order (e.g. var, skewness, kurtosis ... before final reshape)
    """

    if isinstance(z_idx, int):
        z_idx = [z_idx]

    n_scales = len(R_idx)
    n_redshifts = len(z_idx)
    n_cumulants = len(order_idx)

    assert n_redshifts == 1

    logger.info("z_idx: {}, R_idx: {}, order_idx: {}".format(z_idx, R_idx, order_idx))

    def _get_bar(n_s):
        bar = trange(n_s, desc="cumulants") 
        return bar

    fiducial_pdfs_z_R = np.zeros((fiducial_pdfs.shape[1], n_scales * n_redshifts * n_cumulants))
    fiducial_vars_z_R = np.zeros((fiducial_pdfs.shape[1], n_scales * n_redshifts))
    for n in _get_bar(fiducial_pdfs.shape[1]):
        for z, z_i in enumerate(z_idx):
            for r, r_i in enumerate(R_idx):

                _slice = z * n_scales + r # NOTE: These must be positions in new array

                # Shape (3,), # Float[np.ndarray, "z n R d"]
                simulation = fiducial_pdfs[z_i, n, r_i, order_idx] # NOTE: list order_idx indexing works?

                fiducial_pdfs_z_R[n, _slice * n_cumulants : (_slice + 1) * n_cumulants] = simulation

                fiducial_vars_z_R[n, r] = simulation[0] # Just take variance 
    
    fiducial_vars_z_R = np.mean(fiducial_vars_z_R, axis=0)

    latin_pdfs_z_R = np.zeros((latin_pdfs.shape[1], n_scales * n_redshifts * n_cumulants))
    for n in _get_bar(latin_pdfs.shape[1]):
        for z, z_i in enumerate(z_idx):
            for r, r_i in enumerate(R_idx):

                _slice = z * n_scales + r # NOTE: These must be positions in new array

                # Shape (3,), # Float[np.ndarray, "z n R d"]
                simulation = latin_pdfs[z_i, n, r_i, order_idx] 

                latin_pdfs_z_R[n, _slice * n_cumulants : (_slice + 1) * n_cumulants] = simulation

    derivatives_z_R = np.zeros((derivatives.shape[0], ALPHA.size, n_scales * n_redshifts * n_cumulants))
    for n in _get_bar(derivatives.shape[0]):
        for z, z_i in enumerate(z_idx):
            for r, r_i in enumerate(R_idx):

                _slice = z * n_scales + r # NOTE: These must be positions in new array

                # Shape (5, 3), # Float[np.ndarray, "n z 5 R d"]
                for p in range(ALPHA.size):
                    simulation = derivatives[n, z_i, p, r_i, order_idx] # Redshift axis is 2nd, parameter axis is 3rd

                    derivatives_z_R[n, p, _slice * n_cumulants : (_slice + 1) * n_cumulants] = simulation

    logger.info(
        "Processed data shapes (fids., latins, derivs.):\n\t{}".format( 
            [_.shape for _ in [fiducial_pdfs_z_R, latin_pdfs_z_R, derivatives]]
        )
    )

    if FIDUCIAL_REDUCE:
        logger.info("REDUCING CUMULANTS.")
        for z, z_i in enumerate(z_idx):
            for r, r_i in enumerate(R_idx):

                # Only divide skewness and kurtoses by mean fiducial variance
                # S_n = k_n / var ** (n - 1)
                _vars = np.asarray([fiducial_vars_z_R[r] ** 2., fiducial_vars_z_R[r] ** 3.]) # np.tile(fiducial_vars_z_R[r], (2,)) # Tile to [skew, kurtosis] shape
                fiducial_pdfs_z_R[:, r * n_cumulants : (r + 1) * n_cumulants][:, 1:] /= _vars 
                latin_pdfs_z_R[:, r * n_cumulants : (r + 1) * n_cumulants][:, 1:] /= _vars
                derivatives_z_R[:, :, r * n_cumulants : (r + 1) * n_cumulants][:, :, 1:] /= _vars

    return fiducial_pdfs_z_R, latin_pdfs_z_R, derivatives_z_R


def get_cumulant_data(
    config: ConfigDict, *, results_dir: Optional[str] = None
) -> Dataset:

    @typecheck
    def calculate_derivatives(
        derivatives_pm: Float[np.ndarray, "500 z p R 2 d"], 
        alpha: Float[np.ndarray, "p"], 
        dparams: Float[np.ndarray, "p"], 
        parameter_strings: list[str], 
        parameter_derivative_names: list[list[str]]
    ) -> Float[np.ndarray, "500 z 5 R d"]:

        # (n, z, p, R, 2, d) -> (n, z, p, R, d)
        derivatives = derivatives_pm[..., 1, :] - derivatives_pm[..., 0, :] 

        for p in range(alpha.size):
            logger.info(
                "Parameter strings / dp / dp_name: {} {} {}".format( 
                    parameter_strings[p], dparams[p], parameter_derivative_names[p]
                )
            )
            derivatives[:, :, p, ...] = derivatives[:, :, p, ...] / dparams[p] # NOTE: OK before or after reducing cumulants

        assert derivatives.ndim == ALPHA.size, "{}".format(derivatives.shape)

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

    # Name for dataset to load / save once created
    dataset_identifier_str = "".join(
        [
            # Datavector, model and specification
            "_R" + "".join(map(str, config.scales)),
            "_m" + "".join(map(str, config.order_idx)),
            "_z" + str(config.redshift),
            "_f" if config.freeze_parameters else "_nf",
            "_linearised" if config.linearised else "_nonlinear",
            "_reduced" if FIDUCIAL_REDUCE else "", # Reduction k_n -> S_n with fiducial variance
            "_quijote" # E.g. not calculated from PDFs
            # PDFs dataset
            # "_pdfs" if pdfs else "", 
            # Bulk calcuations 
            # "_with_means" if use_mean else "",
            # "_central" if central_moments else "",
            # "_with_norms" if use_normalisations else "",
            # "_with_means_stacked" if stack_mean else "",
        ]
    )

    # Try loading dataset instead of deriving it again and again NOTE: careful not to load PDFs when yhou need cumulants etc
    dataset_filename = os.path.join(
        data_dir, "datasets/{}_cumulants_dataset{}.npz".format("tails", dataset_identifier_str)
    )

    def generate_dataset():
        # Raw moments
        (
            fiducial_moments,
            latin_moments, 
            latin_moments_parameters,
            derivatives_pm
        ) = get_raw_data(data_dir)

        # Euler derivative from plus minus statistics (NOTE: derivatives: Float[np.ndarray, "500 p z R 2 d"])
        derivatives = calculate_derivatives(
            derivatives_pm, 
            alpha, 
            dparams, 
            parameter_strings=parameter_strings, 
            parameter_derivative_names=parameter_derivative_names
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
            order_idx=config.order_idx
        )

        n_s, n_d = fiducial_moments_z_R.shape 

        # Calculate covariance for datavector of each fiducial pdfs for all chosen scales 
        C = np.cov(fiducial_moments_z_R, rowvar=False) # NOTE: correctly calculates covariance of reduced or not

        assert np.all(np.isfinite(C)), "Bad covariance."
        assert derivatives.shape[:-1] == (500, ALPHA.size), "Do derivatives have batch axis? Required."
        assert fiducial_moments_z_R.shape[0] == 15_000, "Incorrect number of fiducials."

        # Precision, corrected with Hartlap
        H = hartlap(n_s=n_s, n_d=n_d)
        Cinv = H * np.linalg.inv(C) # Cinv = jnp.linalg.svd(C) * H

        # Fisher information matrix; all scales, one redshift
        dmu = np.mean(derivatives, axis=0)
        F = np.linalg.multi_dot([dmu, Cinv, dmu.T])
        Finv = np.linalg.inv(F)

        dataset = Dataset(
            name="tails",
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

        corr_moments = jnp.corrcoef(fiducial_moments_z_R, rowvar=False)

        filename = os.path.join(log_figs_dir, "corr_coeff_moments_tails_quijote.png")
        plt.figure()
        plt.title("Correlation matrix (moments) [tails, quijote]")
        plt.imshow(corr_moments, cmap="coolwarm")
        plt.colorbar()
        plt.savefig(filename)
        plt.close()
        logger.debug("Saved correlation matrix (moments) figure at: \n\t{}".format(filename))

        if config.freeze_parameters:
            print("Freezing all but Om, s8")
            dataset = freeze_out_parameters_dataset(dataset)

        return dataset

    if not FORCE_RECOMPUTE_DATASET:
        try:
            logger.info("Loading dataset:\n\t{}".format(dataset_filename))

            dataset_dict = np.load(dataset_filename, allow_pickle=True) 

            dataset = Dataset(
                name="tails",
                alpha=jnp.asarray(dataset_dict["alpha"]),
                lower=jnp.asarray(dataset_dict["lower"]),
                upper=jnp.asarray(dataset_dict["upper"]),
                parameter_strings=list(dataset_dict["parameter_strings"]),
                Finv=jnp.asarray(dataset_dict["Finv"]),
                Cinv=jnp.asarray(dataset_dict["Cinv"]),
                C=jnp.asarray(dataset_dict["C"]),
                fiducial_data=jnp.asarray(dataset_dict["fiducial_data"]),
                data=jnp.asarray(dataset_dict["data"]),
                parameters=jnp.asarray(dataset_dict["parameters"]),
                derivatives=jnp.asarray(dataset_dict["derivatives"]),
            )

            logger.info("Loaded dataset:\n\t{}".format(dataset_filename))

        except FileNotFoundError:
            logger.info("Generating dataset:\n\t{}".format(dataset_filename))

            dataset = generate_dataset()

            logger.info("Generated dataset:\n\t{}".format(dataset_filename))
    else:
        logger.info("Generating dataset:\n\t{}".format(dataset_filename))

        dataset = generate_dataset()

        logger.info("Generated dataset:\n\t{}".format(dataset_filename))

    return dataset


@typecheck
def get_data(config: ConfigDict, *, results_dir: Optional[str] = None) -> Dataset:
    """ 
        Get data for linearised-model data or full simulation data. 
        - Start with Quijote default data; linearise or nonlinearise 
          if required
    """

    dataset: Dataset = get_cumulant_data(config, results_dir=results_dir)

    if hasattr(config, "linearised"):
        if config.linearised:
            print("Using linearised model, Gaussian noise.")

            D, Y = get_linearised_data(config, dataset) 

            dataset = replace(dataset, data=D, parameters=Y)
        else:
            print("Using non-linearised model, non-Gaussian noise.")

    # # E.g. using non-linear model and Gaussian noise or what?
    # if hasattr(config, "nonlinearised"):
    #     if config.nonlinearised:
    #         print("Using linearised model, non-Gaussian noise.") 
    #         D, Y = get_nonlinearised_data(config)
    #         dataset = replace(dataset, data=D, parameters=Y)

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
        results_dir: Optional[str] = None
    ):
        self.config = config

        self.data = get_data(
            config, results_dir=results_dir
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
    #     if :
    #         print("Covariance conditioning...")

    #     L = jnp.trace(C) / n_d * config.covariance_epsilon

    #     # U, S, Vt = jnp.linalg.svd(C)
    #     # L = 0.01 * S.min()
    #     # L = S.max() / 1000

    #     C = jnp.identity(n_d) * L + C