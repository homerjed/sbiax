import os
from dataclasses import dataclass, replace
from functools import partial
from typing import Tuple, Callable, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray, Float, Array, jaxtyped
from beartype import beartype as typechecker
from ml_collections import ConfigDict

import numpy as np 
import matplotlib.pyplot as plt
from chainconsumer import Chain, ChainConsumer, Truth
import tensorflow_probability.substrates.jax.distributions as tfd
from tqdm.auto import trange

from constants import get_quijote_parameters, get_save_and_load_dirs
from cumulants import get_prior, Dataset
from sbiax.utils import make_df, marker
from configs import bulk_cumulants_config

typecheck = jaxtyped(typechecker=typechecker)


@typecheck
def get_raw_data(
    data_dir: str, verbose: bool = False
) -> tuple[
    Float[np.ndarray, "z 15000 R d"],
    Float[np.ndarray, "z 2000 R d"],
    Float[np.ndarray, "2000 p"],
    Float[np.ndarray, "500 z p R 2 d"],
    Float[np.ndarray, "d"],
    Float[np.ndarray, "d"]
]:
    """
        Load raw files from Quijote for cumulants and their derivatives
    """

    fiducials = np.load(os.path.join(data_dir, "ALL_FIDUCIAL_PDFS.npy"))

    latins = np.load(os.path.join(data_dir, "ALL_LATIN_PDFS.npy"))

    latin_parameters = np.loadtxt(os.path.join(data_dir, "latin_hypercube_params.txt"))

    # Load normalised derivatives (n, p, z, R, pm, d) = (500, 5, 5, 7, 2, 99)
    derivatives = np.load(
        os.path.join(data_dir, f"pdfs_derivatives_plus_minus.npy")
    )
    
    if verbose:
        print("Derivatives:", derivatives.shape)

    deltas = np.load(os.path.join(data_dir, "deltas.npy"))

    DELTA_BIN_EDGES = np.geomspace(1e-2, 1e2, num=100) # 1911.11158 Section 4.1
    D_DELTAS = DELTA_BIN_EDGES[1:] - DELTA_BIN_EDGES[:-1] 

    return fiducials, latins, latin_parameters, derivatives, deltas, D_DELTAS


def get_bulk_cumulants_data(
    config: ConfigDict, 
    *, 
    pdfs: bool = False, # Use PDFs or cumulants for the bulk
    verbose: bool = False, 
    results_dir: Optional[str] = None
) -> Dataset:
    """
        Get dataset for SBI experiments with the cumulants.
        - Cut the PDFs according to a p_min, p_max cut into the CDF which 
          indexes the bins of the PDF.
        - 
    """

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

    p_value_min        = config.p_value_min # Independent of choosing rho/delta for random variable of PDF
    p_value_max        = config.p_value_max 

    n_scales           = len(config.scales)
    n_redshifts        = 1
    n_bins_pdf         = 99
    n_fiducial_pdfs    = 15_000
    n_latin_pdfs       = 2000
    n_derivatives      = 500
    n_p                = alpha.size
    R_idx              = [all_R_values.index(R) for R in config.scales]
    z_idx              = all_redshifts.index(config.redshift)
    n_cumulants        = 3

    fiducials, latins, latin_parameters, derivatives, deltas, D_deltas = get_raw_data(data_dir, verbose=verbose)

    # Euler derivative from plus minus statistics
    derivatives = derivatives[..., 1, :] - derivatives[..., 0, :]
    for p in range(n_p):
        derivatives[:, :, p, ...] = derivatives[:, :, p, ...] / dparams[p] # NOTE: parameter / redshifts axis!!!!!

    # Calculate mean of fiducial PDFS across scales R for cutting with CDF
    fiducial_pdfs_stacked_mean = np.zeros((n_bins_pdf * n_scales,))
    for R in R_idx:
        mean_z_R = jnp.mean(fiducials[z_idx, :, R, :], axis=0) #* D_deltas

        fiducial_pdfs_stacked_mean[R * n_bins_pdf : (R + 1) * n_bins_pdf] = mean_z_R

    # Assuming same shape mean PDF as Cora
    cdf = np.zeros((n_scales * n_bins_pdf,))
    for i in range(1, n_bins_pdf):
        for R, _ in enumerate(config.scales):
            # Fiducial pdfs must be normalised here
            cdf[R * n_bins_pdf + i] = cdf[R * n_bins_pdf + i - 1] \
                                    + fiducial_pdfs_stacked_mean[R * n_bins_pdf + i - 1] * D_deltas[i - 1] # PDFs normalised by default
    if verbose:
        print(f"...CDF min/max {cdf.min():.1f} {cdf.max():.1f}.")

    # Cut indices for each value of R
    cuts = [
        np.where(
            (cdf[R * n_bins_pdf : (R + 1) * n_bins_pdf] > p_value_min) & \
            (cdf[R * n_bins_pdf : (R + 1) * n_bins_pdf] < p_value_max)
        )[0] 
        for R, _ in enumerate(config.scales)
    ]
    if verbose:
        print("CUTS (idx):\n", [(min(cut), max(cut)) for cut in cuts])
        print("CUTS (deltas):\n", ["{:.1f} {:.1f}".format(min(deltas[cut]), max(deltas[cut])) for cut in cuts])
        print("Cut total dim:", sum([_.size for _ in cuts]))

    # Get cut dimensions
    z_cut_dim_totals = 0
    for R, cut_z in zip(config.scales, cuts):
        if verbose:
            print(f" R={R} cut: {cut_z.shape}")
        z_cut_dim_totals += cut_z.shape[0]

    if verbose:
        print("All cuts added:", z_cut_dim_totals)
        print(f"\n>CDF shape: {cdf.shape}") 
        print(f"Total bins kept in CDF cut: {z_cut_dim_totals}/{1 * n_scales * n_bins_pdf}")

    """
        Cut PDFs to bulk density cut
    """

    def normalise_cut_pdf(pdf, D_deltas_cut):
        # Renormalise PDF in bulk region (pdf) which is already normalised
        pdf = pdf #/ D_deltas_cut
        pdf = pdf / jnp.sum(pdf * D_deltas_cut) # Denominator is PDF integral
        return pdf #/ D_deltas_cut

    def moment_n_R(pdf, n, D_deltas_cut, deltas_cut):
        pdf = normalise_cut_pdf(pdf, D_deltas_cut)

        # assert jnp.isclose(jnp.sum(pdf), 1.0, rtol=1e-2, atol=1e-2), (
        #     f"PDF should be normalised to 1.0 but is {jnp.sum(pdf)}"
        # )

        # mu_delta = jnp.sum(pdf * deltas_cut * D_deltas_cut) # = p(x)dx * x
        mu_delta = 0.

        # Calculate moment n
        moment_n = jnp.sum(pdf * D_deltas_cut * ((deltas_cut - mu_delta) ** n)) # = p(x)dx * (x ** n)

        return moment_n

    def moments_to_cumulants(moments, deltas_cut):
        # Bernardeau 2002 Eq. 130
        _delta_ = 0. #deltas_cut.mean() # <delta>

        cumulant_2 = moments[0] - (_delta_) ** 2.
        cumulant_3 = moments[1] - 3. * cumulant_2 * _delta_ - _delta_ ** 3.
        cumulant_4 = moments[2] - 4. * cumulant_3 * _delta_ - 3. * (cumulant_2 ** 2.) - 6. * cumulant_2 * (_delta_ ** 2.) - (_delta_ ** 4.)

        return jnp.asarray([cumulant_2, cumulant_3, cumulant_4]) 

    orders = [2, 3, 4] # Variance, skewness and kurtosis
    cut_dim = sum([cut.size for cut in cuts])

    fiducial_pdfs_z_R_cut = np.zeros((n_fiducial_pdfs, cut_dim))
    fiducial_moments_z_R = np.zeros((n_fiducial_pdfs, n_scales * n_cumulants))
    for n in range(n_fiducial_pdfs):
        
        for R, cut in enumerate(cuts):

            _cut_dim = sum([cut.size for cut in cuts[:R]])

            pdf = fiducials[z_idx, n, R, cut]

            fiducial_pdfs_z_R_cut[n, _cut_dim : _cut_dim + cuts[R].size] = pdf

            for i in range(len(orders)):
                order = orders[i]
                moment = moment_n_R(
                    pdf, n=order, D_deltas_cut=D_deltas[cut], deltas_cut=deltas[cut]
                )
                fiducial_moments_z_R[n, i + R * n_cumulants : (i + 1) + R * n_cumulants] = moment

            # Convert to cumulants
            if config.reduced_cumulants:
                fiducial_moments_z_R[n, R * n_cumulants : (R + 1) * n_cumulants] = moments_to_cumulants(
                    fiducial_moments_z_R[n, R * n_cumulants : (R + 1) * n_cumulants], deltas_cut=deltas[cut]
                )

            if verbose:
                print("\r n={:05d}/{}".format(n, n_fiducial_pdfs), end="")

    latin_pdfs_z_R_cut = np.zeros((n_latin_pdfs, cut_dim))
    latin_moments_z_R = np.zeros((n_latin_pdfs, n_scales * n_cumulants))
    for n in range(n_latin_pdfs):

        for R, cut in enumerate(cuts):

            _cut_dim = sum([cut.size for cut in cuts[:R]])

            pdf = latins[z_idx, n, R, cut]

            latin_pdfs_z_R_cut[n, _cut_dim : _cut_dim + cuts[R].size] = pdf

            for i in range(len(orders)):
                order = orders[i]
                moment = moment_n_R(
                    pdf, order, D_deltas_cut=D_deltas[cut], deltas_cut=deltas[cut]
                )
                latin_moments_z_R[n, i + R * n_cumulants : (i + 1) + R * n_cumulants] = moment

            # Convert to cumulants
            if config.reduced_cumulants:
                latin_moments_z_R[n, R * n_cumulants : (R + 1) * n_cumulants] = moments_to_cumulants(
                    latin_moments_z_R[n, R * n_cumulants : (R + 1) * n_cumulants], deltas_cut=deltas[cut]
                )

            if verbose:
                print("\r n={:05d}/{}".format(n, n_latin_pdfs), end="")

    derivative_pdfs_z_R_cut = np.zeros((n_derivatives, n_p, cut_dim))
    derivative_moments_z_R = np.zeros((n_derivatives, n_p, n_scales * n_cumulants)) 
    for n in range(n_derivatives):

        for R, cut in enumerate(cuts):

            _cut_dim = sum([cut.shape[0] for cut in cuts[:R]])
            
            pdf = derivatives[n, z_idx, :, R, cut].T # Shape (p, cut_dim), raw shape (n_derivatives, n_redshifts, n_params, n_scales, 2, n_bins_pdf)

            derivative_pdfs_z_R_cut[n, :, _cut_dim : _cut_dim + cuts[R].size] = pdf # NOTE: choose the right axis; params or redshift

            for i in range(len(orders)):
                order = orders[i]
                for p in range(n_p):
                    moment_p = moment_n_R(
                        pdf[p], order, D_deltas_cut=D_deltas[cut], deltas_cut=deltas[cut]
                    )
                    derivative_moments_z_R[n, p, i + R * n_cumulants : (i + 1) + R * n_cumulants] = moment_p

            # Convert to cumulants
            if config.reduced_cumulants:
                for p in range(n_p):
                    derivative_moments_z_R[n, p, R * n_cumulants : (R + 1) * n_cumulants] = moments_to_cumulants(
                        derivative_moments_z_R[n, p, R * n_cumulants : (R + 1) * n_cumulants], deltas_cut=deltas[cut]
                    )

            if verbose:
                print("\r n={:05d}/{}".format(n, n_derivatives), end="")

    print(
        "Fiducials, latins (one redshift):\n", 
        fiducial_pdfs_z_R_cut.shape, 
        latin_pdfs_z_R_cut.shape, 
        derivative_pdfs_z_R_cut.shape
    )

    """
        Fisher information matrix calculations
    """

    n_fiducial_pdfs, data_dim_pdfs = fiducial_pdfs_z_R_cut.shape 
    C_pdf = np.cov(fiducial_pdfs_z_R_cut, rowvar=False) 
    H = (n_fiducial_pdfs - data_dim_pdfs - 2.) / (n_fiducial_pdfs - 1.)
    Cinv_pdf = H * np.linalg.inv(C_pdf)
    dmu_pdfs = np.mean(derivative_pdfs_z_R_cut, axis=0)
    F_pdf = jnp.linalg.multi_dot([dmu_pdfs, Cinv_pdf, dmu_pdfs.T])
    Finv_pdf = np.linalg.inv(F_pdf)

    n_fiducial_moments, data_dim_moments = fiducial_moments_z_R.shape
    C_moments = jnp.cov(fiducial_moments_z_R, rowvar=False)
    H = (n_fiducial_moments - data_dim_moments - 2.) / (n_fiducial_moments - 1.)
    Cinv_moments = H * jnp.linalg.inv(C_moments)
    dmu_moments = jnp.mean(derivative_moments_z_R, axis=0)
    F_moments = jnp.linalg.multi_dot([dmu_moments, Cinv_moments, dmu_moments.T])
    Finv_moments = jnp.linalg.inv(F_moments)

    """
        Datasets
    """

    # Cumulants[bulk]
    dataset = Dataset(
        alpha=jnp.asarray(alpha),
        lower=jnp.asarray(lower),
        upper=jnp.asarray(upper),
        parameter_strings=parameter_strings,
        Finv=jnp.asarray(Finv_moments),
        Cinv=jnp.asarray(Cinv_moments),
        C=jnp.asarray(C_moments),
        fiducial_data=jnp.asarray(fiducial_moments_z_R),
        data=jnp.asarray(latin_moments_z_R),
        parameters=jnp.asarray(latin_parameters),
        derivatives=jnp.asarray(derivative_moments_z_R)  
    )

    # PDF[bulk]
    pdf_dataset = Dataset(
        alpha=jnp.asarray(alpha),
        lower=jnp.asarray(lower),
        upper=jnp.asarray(upper),
        parameter_strings=parameter_strings,
        Finv=jnp.asarray(Finv_pdf),
        Cinv=jnp.asarray(Cinv_pdf),
        C=jnp.asarray(C_pdf),
        fiducial_data=jnp.asarray(fiducial_pdfs_z_R_cut),
        data=jnp.asarray(latin_pdfs_z_R_cut),
        parameters=jnp.asarray(latin_parameters),
        derivatives=jnp.asarray(derivative_pdfs_z_R_cut)  
    )

    if verbose:
        def mle_pdf(d, p):
            mu = jnp.mean(fiducial_pdfs_z_R_cut, axis=0)
            dmu = jnp.mean(derivative_pdfs_z_R_cut, axis=0)
            mu_p = mu + jnp.dot(p - alpha, dmu)
            return p + jnp.linalg.multi_dot([Finv_pdf, dmu, Cinv_pdf, d - mu_p])

        def mle_moments(d, p):
            mu = jnp.mean(fiducial_moments_z_R, axis=0)
            dmu = jnp.mean(derivative_moments_z_R, axis=0)
            mu_p = mu + jnp.dot(p - alpha, dmu)
            return p + jnp.linalg.multi_dot([Finv_pdf, dmu, Cinv_moments, d - mu_p])

        X_pdfs = jax.vmap(mle_pdf)(latin_pdfs_z_R_cut, latin_parameters)
        X_moments = jax.vmap(mle_moments)(latin_moments_z_R, latin_parameters)

        fig, axs = plt.subplots(1, dataset.alpha.size, figsize=(2. + 2. * dataset.alpha.size, 2.5))
        for p, ax in enumerate(axs):
            ax.scatter(latin_parameters[:, p], X_moments[:, p], s=0.1)
            ax.scatter(latin_parameters[:, p], X_pdfs[:, p], s=0.1)
            ax.axline((0, 0), slope=1., color="k", linestyle="--")
            ax.set_xlim(dataset.lower[p], dataset.upper[p])
            ax.set_ylim(dataset.lower[p], dataset.upper[p])
            ax.set_xlabel(dataset.parameter_strings[p])
            ax.set_ylabel(dataset.parameter_strings[p] + "'")
        plt.savefig("/project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/scratch/summaries_pdf_moments.png")
        plt.close()

        c = ChainConsumer()
        c.add_chain(
            Chain(
                samples=make_df(X_moments, parameter_strings=parameter_strings), 
                name="X[moments]", 
                color="b", 
                plot_contour=False, 
                plot_cloud=True
            )
        )
        c.add_chain(
            Chain(
                samples=make_df(X_pdfs, parameter_strings=parameter_strings), 
                name="X[pdfs]", 
                color="g", 
                plot_contour=False, 
                plot_cloud=True
            )
        )
        c.add_chain(
            Chain.from_covariance(
                alpha,
                Finv_pdf,
                columns=parameter_strings,
                name=r"$F_{\Sigma^{-1}}$ (PDFs)",
                color="k",
                linestyle=":",
                shade_alpha=0.
            )
        )
        c.add_chain(
            Chain.from_covariance(
                alpha,
                Finv_moments,
                columns=parameter_strings,
                name=r"$F_{\Sigma^{-1}}$ (moments)",
                color="r",
                linestyle=":",
                shade_alpha=0.
            )
        )
        # c.add_chain(
        #     Chain(
        #         samples=make_df(dataset.parameters, parameter_strings=dataset.parameter_strings), 
        #         plot_contour=False, 
        #         plot_cloud=True, 
        #         name="P", 
        #         color="r"
        #     )
        # )
        c.add_marker(
            location=marker(alpha, parameter_strings=parameter_strings),
            name=r"$\alpha$", 
            color="#7600bc"
        )
        fig = c.plotter.plot()
        # fig.suptitle(
        #     r"$k_n/k_2^{n-1}$ SBI & $F_{{\Sigma}}^{{-1}}$" + "\n" +
        #     "z={},\n $n_s$={}, (pre-train $n_s$={}),\n R={} Mpc,\n $k_n$={}".format(
        #             0., 
        #             len(X), 
        #             "[{}]".format(", ".join(map(str, R_values)))
        #         ),
        #     multialignment='center'
        # )
        plt.savefig(
            os.path.join(
                results_dir if results_dir is not None else "fisher_forecasts/", 
                "bulk_fisher_forecast_{}_z={}_R={}_m={}.png".format(
                    config.linearised, 
                    config.redshift, 
                    "".join(map(str, config.order_idx)),
                    "".join(map(str, config.scales))
                )
            ), 
        )
        plt.close()

    return pdf_dataset if pdfs else dataset


"""
    Compression
"""


@typecheck
def linearised_model(
    alpha: Float[Array, "p"], 
    alpha_: Float[Array, "p"], 
    mu: Float[Array, "d"], 
    dmu: Float[Array, "p d"]
) -> Float[Array, "d"]:
    return mu + jnp.dot(alpha_ - alpha, dmu)


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


def get_compression_fn(key, config, dataset, *, results_dir):
    # Get linear or neural network compressor
    if config.compression == "nn":

        net, preprocess_fn = get_nn_compressor(
            key, dataset, results_dir=results_dir
        )

        compressor = lambda d, p: net(preprocess_fn(d)) # Ignore parameter kwarg!

    if config.compression == "linear":
        compressor = get_linear_compressor(config, dataset)

    compression_fn = lambda d, p: compressor(d, p)

    return compression_fn 


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
    Dataset
"""


@dataclass
class BulkCumulantsDataset:
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
        pdfs: bool = False,
        verbose: bool = False, 
        results_dir: Optional[str] = None
    ):
        self.config = config

        self.data = get_bulk_cumulants_data(
            config, pdfs=pdfs, verbose=verbose, results_dir=results_dir
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
            alpha=self.data.alpha, 
            lower=self.data.lower, 
            upper=self.data.upper, 
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


def get_bulk_dataset(args, pdfs=False):
    # Take non-bulk config, get bulk config, get dataset, extract Finv

    config = bulk_cumulants_config(
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

    if pdfs: 
        print("Using PDF dataset for bulk dataset.")
    else:
        print("Using cumulants dataset for bulk dataset.")

    dataset = BulkCumulantsDataset(config, pdfs=pdfs, verbose=True)

    return dataset.data


if __name__ == "__main__":
    from configs import bulk_cumulants_config

    config = bulk_cumulants_config()

    dataset = BulkCumulantsDataset(config, verbose=True)