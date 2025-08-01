import os
from dataclasses import dataclass, replace, asdict
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray, Float, Int, Array, Scalar, jaxtyped
from beartype import beartype as typechecker
from ml_collections import ConfigDict

import numpy as np 
import matplotlib.pyplot as plt
from chainconsumer import Chain, ChainConsumer, Truth
import tensorflow_probability.substrates.jax.distributions as tfd
from tqdm.auto import trange

from configs.log import setup_module_logger, get_log_level
from data.constants import (
    get_quijote_parameters, 
    get_save_and_load_dirs, 
    get_target_idx,
    get_F_planck
)
from data.common import (
    Dataset,
    get_prior,
    sample_prior,
    get_compression_fn,
    get_linearised_data,
    get_non_gaussian_linear_model_data,
    get_datavector,
    freeze_out_parameters_dataset, 
    hartlap,
    get_parameter_strings,
    add_planck_information_to_Finv
)
from sbiax.utils import make_df, marker
from configs.cumulants_configs import bulk_cumulants_config

typecheck = jaxtyped(typechecker=typechecker)

logger, log_figs_dir = setup_module_logger(__name__, level=get_log_level())

FORCE_RECOMPUTE_DATASET = True if os.environ.get("FORCE_RECOMPUTE_DATASET", "").lower() in ("1", "true") else False 
USE_QUIJOTE_TAILS = True if os.environ.get("USE_QUIJOTE_TAILS", "").lower() in ("1", "true") else False
FIDUCIAL_REDUCE = True if os.environ.get("FIDUCIAL_REDUCE", "").lower() in ("1", "true") else False
DEFAULT_RESOLUTION = int(os.environ.get("DEFAULT_RESOLUTION", 1024))
NON_GAUSSIAN_TEST = True if os.environ.get("NON_GAUSSIAN_TEST", "").lower() in ("1", "true") else False

PRINT_FREQ = 500


"""
    Get cumulants of the bulk of the 3D matter PDF
"""


@typecheck
def get_raw_data(data_dir: str) -> tuple[
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

    fiducials = np.load(os.path.join(data_dir, "raw/ALL_FIDUCIAL_PDFS_resolution={}.npy".format(DEFAULT_RESOLUTION)))

    latins = np.load(os.path.join(data_dir, "raw/ALL_LATIN_PDFS_resolution={}.npy".format(DEFAULT_RESOLUTION)))

    latin_parameters = np.loadtxt(os.path.join(data_dir, "raw/latin_hypercube_params.txt"))

    # Load normalised derivatives (n, p, z, R, pm, d) = (500, 5, 5, 7, 2, 99)
    derivatives = np.load(
        os.path.join(data_dir, "raw/pdfs_derivatives_plus_minus_resolution={}.npy".format(DEFAULT_RESOLUTION))
    )

    deltas = np.load(os.path.join(data_dir, "raw/deltas.npy"))

    DELTA_BIN_EDGES = np.geomspace(1e-2, 1e2, num=100) # 1911.11158 Section 4.1, NOTE: This is in rho
    D_DELTAS = DELTA_BIN_EDGES[1:] - DELTA_BIN_EDGES[:-1] 

    logger.debug("Resolution: {}".format(DEFAULT_RESOLUTION)) 
    logger.debug("Fiducials: {}".format(fiducials.shape))
    logger.debug("Latins: {}".format(latins.shape))
    logger.debug("Latins (parameters): {}".format(latin_parameters.shape))
    logger.debug("Derivatives: {}".format(derivatives.shape))

    return fiducials, latins, latin_parameters, derivatives, deltas, D_DELTAS


def get_calculated_cumulants_data(
    config: ConfigDict, 
    *, 
    pdfs: bool = False, # Use PDFs or cumulants for the bulk
    use_means: bool = False,
    use_normalisations: bool = True, # Stack means of bulk of the PDF at each scale with the other cumulants
    stack_means: bool = True,
    full_shape: bool = False,
    results_dir: Optional[str] = None
) -> Dataset:
    """
        Get dataset for SBI experiments with the cumulants.
        - Cut the PDFs according to a p_min, p_max cut into the CDF which 
          indexes the bins of the PDF.
        - Return PDFs of cumulants of the bulk
    """

    logger.info("Getting calculated cumulants for dataset={}".format(config.dataset_name))
    logger.info("Using bulk means..." if use_means else "Not using bulk means...")

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
        _,
        _, 
        _ 
    ) = get_quijote_parameters()

    p_value_min                  = config.p_value_min # Independent of choosing rho/delta for random variable of PDF
    p_value_max                  = config.p_value_max 

    cumulants                    = True                    # Use cumulants over moments (NOTE: check not calculating reduced-cumulants, Quijote uses cumulants)
    use_means                    = use_means               # Use <delta> in calculation of cumulants from moments 
    stack_means                  = stack_means             # Stack bulk mean do bulk datavector For full shape <delta> is very close to zero but <rho> approximately one
    use_normalisations           = use_normalisations      # Stack M_0 normalisation of pdf into datavector ahead of mean M_1 
    normalise                    = False #not use_normalisations  # Divide moments by M_0, don't do this if concatenating M_0 (NOTE: in quijote vs calculation comparison this is ignored in the bulk)
    central_moments              = True                    # Calculate central moments or not (NOTE: 4th cumulant not the same as 4th central moment, but Bernardeau formulae use non-central moments)

    # Value of normalisation of bulk PDF (NOTE: turn this off for the comparison? Divide ALL cumulants by this? => it's off for full-shape)
    fiducial_based_normalisation = config.p_value_max - config.p_value_min 

    n_scales           = len(config.scales)
    n_redshifts        = 1
    n_bins_pdf         = 99
    n_fiducial_pdfs    = 15_000
    n_latin_pdfs       = 2000
    n_derivatives      = 500
    n_p                = alpha.size 
    R_idx              = [all_R_values.index(R) for R in config.scales]
    z_idx              = all_redshifts.index(config.redshift)
    n_cumulants        = 3 # [var, skew, kurt]

    if full_shape:
        bulk_or_tails = "tails"
    else:
        bulk_or_tails = "bulk"

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
            # PDFs dataset
            "_pdfs" if pdfs else "", 
            # Bulk calcuations 
            "_with_means" if use_means else "",
            "_central" if central_moments else "",
            "_with_norms" if use_normalisations else "",
            "_with_means_stacked" if stack_means else "",
        ]
    )

    # Try loading dataset instead of deriving it again and again NOTE: careful not to load PDFs when yhou need cumulants etc
    dataset_filename = os.path.join(
        data_dir, "datasets/{}_cumulants_dataset{}.npz".format(bulk_or_tails, dataset_identifier_str)
    )

    def generate_dataset() -> list[np.ndarray]:

        tqdm_desc_str = config.dataset_name
        if pdfs:
            tqdm_desc_str += " pdfs"
        if full_shape:
            tqdm_desc_str += " full-shape"

        # Get fiducial, derivative and hypercube PDFs
        (
            fiducials,        # Float[np.ndarray, "z 15000 R d"]
            latins,           # Float[np.ndarray, "z 2000 R d"]
            latin_parameters, # Float[np.ndarray, "2000 p"]
            derivatives,      # Float[np.ndarray, "500 z p R 2 d"]
            deltas,           # Float[np.ndarray, "d"]
            D_deltas          # Float[np.ndarray, "d"]
        ) = get_raw_data(data_dir)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # Convert from rho to delta for bin centres, edges and widths ONLY FOR BULKS>>!?
        deltas = deltas - 1.
        delta_bin_edges = np.geomspace(1e-2, 1e2, num=100) - 1. # 1911.11158 Section 4.1, NOTE: This is in delta not rho
        D_deltas = delta_bin_edges[1:] - delta_bin_edges[:-1] 

        if 0:
            # Calculate linear bin widths
            ln_delta_min = np.log(0.01) # Isn't this rho?
            ln_delta_max = np.log(100.)
            dln_delta = (ln_delta_max - ln_delta_min) / n_bins_pdf 

            bin_edges = np.zeros(n_bins_pdf + 1)
            bin_widths = np.zeros(n_bins_pdf)
            mean_bins = np.zeros(n_bins_pdf)
            mean_bins_lin = np.zeros(n_bins_pdf)

            for i in range(n_bins_pdf + 1):
                bin_edges[i] = np.exp(ln_delta_min + i * dln_delta) - 1.

            for i in range(n_bins_pdf):
                bin_widths[i] = bin_edges[i + 1] - bin_edges[i]
                mean_bins[i] = np.sqrt((1. + bin_edges[i + 1]) * (1. + bin_edges[i])) - 1.
                mean_bins_lin[i] = (bin_edges[i + 1] + bin_edges[i]) / 2.

            D_deltas = bin_widths
            deltas = mean_bins_lin
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        def get_cuts_from_fiducial_mean_pdf():
            """ Calculate CDF and cut PDFs """

            # Calculate mean of fiducial PDFS across scales R for cutting with CDF
            fiducial_pdfs_stacked_mean = np.zeros((n_bins_pdf * n_scales,))
            for R, R_i in enumerate(R_idx):
                mean_z_R = jnp.mean(fiducials[z_idx, :, R_i, :], axis=0)

                fiducial_pdfs_stacked_mean[R * n_bins_pdf : (R + 1) * n_bins_pdf] = mean_z_R

            # Assuming same shape mean PDF as Cora
            cdf = np.zeros((n_scales * n_bins_pdf,))
            for i in range(1, n_bins_pdf):
                for R, _ in enumerate(config.scales):
                    # Fiducial pdfs must be normalised here (PDFs normalised by default)
                    p_delta_R_i = fiducial_pdfs_stacked_mean[R * n_bins_pdf + i - 1]

                    cdf[R * n_bins_pdf + i] = cdf[R * n_bins_pdf + i - 1] + p_delta_R_i * D_deltas[i - 1] 

            # Check CDF bounds
            assert np.isclose(cdf.min(), 0.) and np.isclose(cdf.max(), 1.), (
                "CDF: min={} max={}".format(cdf.min(), cdf.max())
            )

            # Cut indices for each scale R
            if full_shape:
                # Calculate cumulants for full shape using all PDF bins
                # cuts = [
                #     np.arange(n_bins_pdf) for R, _ in enumerate(config.scales) 
                # ] 

                logger.info("NOTE: using p min/max for cutting full-shape")

                cuts = [
                    np.where(
                        (cdf[R * n_bins_pdf : (R + 1) * n_bins_pdf] >= p_value_min) & \
                        (cdf[R * n_bins_pdf : (R + 1) * n_bins_pdf] <= p_value_max)
                    )[0] 
                    for R, _ in enumerate(config.scales)
                ]
            else:
                cuts = [
                    np.where(
                        (cdf[R * n_bins_pdf : (R + 1) * n_bins_pdf] >= p_value_min) & \
                        (cdf[R * n_bins_pdf : (R + 1) * n_bins_pdf] <= p_value_max)
                    )[0] 
                    for R, _ in enumerate(config.scales)
                ]

            # Get cut dimensions
            z_cut_dim_totals = 0
            for R, cut_z in zip(config.scales, cuts):
                z_cut_dim_totals += cut_z.shape[0]

                logger.info(f" R={R} cut: {cut_z.shape}")

            logger.debug("CUTS (idx):\n {}".format([(min(cut), max(cut)) for cut in cuts]))
            logger.debug("CUTS (deltas) min/max={:.1f}/{:.1f}:\n".format(deltas.min(), deltas.max())) 
            logger.debug("CUTS:\n{}".format(["{:.1f} {:.1f}".format(min(deltas[cut]), max(deltas[cut])) for cut in cuts]))
            logger.debug(f"\n>CDF shape: {cdf.shape}") 
            logger.debug(f"Total bins kept in CDF cut: {z_cut_dim_totals}/{1 * n_scales * n_bins_pdf}")

            return cuts

        @typecheck
        def normalise_cut_pdf(
            pdf: Float[np.ndarray, "d"], 
            D_deltas_cut: Float[np.ndarray, "d"]
        ) -> Float[np.ndarray, "d"]:
            # Renormalise PDF in bulk region (pdf) which is already normalised

            # logger.debug("PDF (before) {} {}".format(np.sum(pdf), np.sum(pdf * D_deltas_cut)))

            if normalise:
                pdf = pdf / np.sum(pdf * D_deltas_cut) # Denominator is PDF integral

                # logger.debug("PDF (after, normalised) {}, {}".format(np.sum(pdf), np.sum(pdf * D_deltas_cut)))
            else:
                pdf = pdf # Don't normalise: might throw out information for extreme cosmologies

                # logger.debug("PDF (after, no-norm) {}, {}".format(np.sum(pdf), np.sum(pdf * D_deltas_cut)))

            return pdf 

        @typecheck
        def moment_n_R(
            pdf: Float[np.ndarray, "d"], 
            n: int, 
            D_deltas_cut: Float[np.ndarray, "d"], 
            deltas_cut: Float[np.ndarray, "d"]
        ) -> Float[np.ndarray, ""]:

            pdf = normalise_cut_pdf(pdf, D_deltas_cut)

            mu_delta = np.sum(deltas_cut * pdf * D_deltas_cut) if central_moments else jnp.zeros(())

            # Calculate moment n (NOTE: don't centre it around bulk mean of deltas)
            moment_n = np.sum(pdf * D_deltas_cut * ((deltas_cut - mu_delta) ** n))

            return np.asarray(moment_n)

        @typecheck
        def moments_to_cumulants(
            moments: Float[np.ndarray, "k_n"], 
            _delta_: Float[np.ndarray, ""]
        ) -> Float[np.ndarray, "k_n"]:
            # Bernardeau 2002 Eq. 130

            # NOTE: Are these moments incorrectly the central moments? 'moments' arg to this fn is central moments?
            # Central moments == cumulants for n < 4
            cumulant_2 = moments[0] - _delta_ ** 2.
            cumulant_3 = moments[1] - 3. * cumulant_2 * _delta_ - _delta_ ** 3. 
            cumulant_4 = moments[2] - 4. * cumulant_3 * _delta_ - 3. * (cumulant_2 ** 2.) - 6. * cumulant_2 * (_delta_ ** 2.) - _delta_ ** 4.

            cumulants = np.asarray([cumulant_2, cumulant_3, cumulant_4]) 

            return cumulants

        @typecheck
        def _pdf_to_cumulants_bulk(
            cut_pdf: Float[np.ndarray, "d"], # Divide by cut-norm
            deltas: Float[np.ndarray, "d"],
            ddeltas: Float[np.ndarray, "d"],
            cut: Int[np.ndarray, "d"]
        ) -> Float[np.ndarray, "3"]:
            # Bernardeau 2002 Eq. 130

            # Switching from linear to log is just a change of variables encapsulated by widths and means of bins?
            # deltas = mean_bins_lin[cut]
            # ddeltas = bin_widths[cut]

            print("pdfs:", cut_pdf.shape, deltas.shape, ddeltas.shape)

            cut_pdf = cut_pdf / fiducial_based_normalisation

            _delta_ = np.sum(ddeltas * cut_pdf * deltas)

            deltamod = deltas - _delta_

            # Assuming <delta>=0? see Bernardeau eq (130)
            cumulant_2 = np.sum(deltamod ** 2 * cut_pdf * ddeltas)
            cumulant_3 = np.sum(deltamod ** 3 * cut_pdf * ddeltas)
            cumulant_4 = np.sum(deltamod ** 4 * cut_pdf * ddeltas) - (3. * np.sum(deltamod ** 2 * cut_pdf * ddeltas) ** 2)

            cumulants = np.asarray([cumulant_2, cumulant_3, cumulant_4]) 

            return cumulants

        @typecheck
        def intersperse_means(
            means: Float[np.ndarray, "n ... R"], 
            moments: Float[np.ndarray, "n ... Rk"]
        ) -> Float[np.ndarray, "n ... Rkm"] | Float[np.ndarray, "n ... Rk"]:
            # Put means first in moments vectors for all scales i.e. [[mean, var, skew, kurt]_R, ...]
            assert len(means) == len(moments)
            assert means.shape[-1] == n_scales

            means_and_moments = np.zeros(moments.shape[:-1] + (n_scales * (1 + n_cumulants),)) # Mean + cumulants for all scales, additional shape info for derivatives or not

            if fiducial_based_normalisation is not None:
                means = means / fiducial_based_normalisation

            # logger.debug("FULL MEANS MOMENTS: {}".format(means_and_moments.shape))

            # Stack cumulants such that at each scale: M_R = [M_1, k_n]
            for r in range(n_scales):
                _means_and_moments = [
                    means[..., [r]], # Keep last dimension
                    moments[..., r * n_cumulants : (r + 1) * n_cumulants]
                ]

                # logger.debug("MEANS MOMENTS: {}".format([_.shape for _ in _means_and_moments]))

                # Stack on last axis
                means_and_moments[
                    ..., r * (1 + n_cumulants) : (r + 1) * (1 + n_cumulants)
                ] = np.concatenate(_means_and_moments, axis=-1)

            return means_and_moments

        @typecheck
        def intersperse_normalisations(
            normalisations: Float[np.ndarray, "n ... R"], 
            moments: Float[np.ndarray, "n ... Rk"] 
        ) -> Float[np.ndarray, "n ... Rkn"] | Float[np.ndarray, "n ... Rk"]:
            # Put means first in moments vectors for all scales i.e. [[mean, var, skew, kurt]_R, ...]
            assert len(normalisations) == len(moments)
            assert normalisations.shape[-1] == n_scales

            dim = 2 + n_cumulants # Mean and normalisation plus usual cumulants

            normalisations_and_moments = np.zeros(moments.shape[:-1] + (n_scales * dim,)) # Mean + cumulants for all scales, additional shape info for derivatives or not

            if fiducial_based_normalisation is not None:
                normalisations = normalisations / fiducial_based_normalisation

            # logger.debug("NORMALISATIONS AND MOMENTS: {}".format(normalisations_and_moments.shape))

            # Stack cumulants such that at each scale: M_R = [M_0, M_1, k_n]
            for r in range(n_scales):
                _normalisations_and_moments = [
                    normalisations[..., [r]], # Keep last dimension
                    moments[..., r * (n_cumulants + 1) : (r + 1) * (n_cumulants + 1)] # Additional +1 for stacked mean
                ]

                # logger.debug("NORMALISATIONS AND MOMENTS (r): {}".format([_.shape for _ in _normalisations_and_moments]))

                # Stack on last axis
                normalisations_and_moments[
                    ..., r * dim : (r + 1) * dim
                ] = np.concatenate(_normalisations_and_moments, axis=-1)

            return normalisations_and_moments

        """
            Cut PDFs to bulk density cut and calculate moments/cumulants
        """

        cuts = get_cuts_from_fiducial_mean_pdf()

        for scale, cut in zip(config.scales, cuts):
            # Minimum / maximum bins and their deltas
            print("redshift, R: {}, {}".format(config.redshift, scale), deltas[cut.min()], deltas[cut.max()])
    
        print(cuts)

        orders = [2, 3, 4] # Variance, skewness and kurtosis
        cut_dim = sum([cut.size for cut in cuts])

        assert np.all([cut.ndim == 1 for cut in cuts]), (
            "Rank of cut indices arrays not equal to 1, shapes are: {}".format([cut.ndim for cut in cuts])
        )

        fiducial_pdfs_z_R_cut = np.zeros((n_fiducial_pdfs, cut_dim)) # Bulk PDFs
        fiducial_moments_z_R = np.zeros((n_fiducial_pdfs, n_scales * n_cumulants)) # Cumulants of bulk PDFs
        fiducial_vars_z_R = np.zeros((n_fiducial_pdfs, n_scales)) # Variances from cumulants of bulk PDFs
        fiducial_moments_z_R_means = np.zeros((n_fiducial_pdfs, n_scales)) # Means of bulk PDFs
        fiducial_normalisations = np.zeros((n_fiducial_pdfs, n_scales)) # Normalisations of bulk PDFs
        for n in trange(n_fiducial_pdfs, desc="Fiducials [{}]".format(tqdm_desc_str)):
            
            # Using R-th chosen scale and its cut indices into mean fiducial PDF
            # for R, cut in enumerate(cuts):
            for R, (cut, R_i) in enumerate(zip(cuts, R_idx)):

                _cut_dim = sum([_cut.size for _cut in cuts[:R]]) # Cut dimension up to R-th scale

                pdf = fiducials[z_idx, n, R_i, cut] # Cut PDF p(d_i) NOTE: ** this should be R_i from R_idx?!

                fiducial_pdfs_z_R_cut[n, _cut_dim : _cut_dim + cut.size] = pdf

                for i in range(len(orders)): # Cycle through moment orders
                    order = orders[i]

                    moment = moment_n_R(
                        pdf, n=order, D_deltas_cut=D_deltas[cut], deltas_cut=deltas[cut]
                    )

                    fiducial_moments_z_R[n, i + R * n_cumulants : (i + 1) + R * n_cumulants] = moment

                    if i == 0: # Variance
                        fiducial_vars_z_R[n, i + R : (i + 1) + R] = moment

                # Mean and normalisation of PDF
                fiducial_moments_z_R_means[n, R] = np.sum(pdf * D_deltas[cut] * deltas[cut]) # delta_R
                fiducial_normalisations[n, R] = np.sum(pdf * D_deltas[cut])

                # Convert to cumulants (process all orders simultaneously)
                if cumulants:

                    _delta_ = np.asarray(np.sum(pdf * D_deltas[cut] * deltas[cut]))

                    # if full_shape:

                    #     cumulant = moments_to_cumulants(
                    #         fiducial_moments_z_R[n, R * n_cumulants : (R + 1) * n_cumulants], 
                    #         _delta_=_delta_ if use_means else np.zeros(()) # _delta_=_delta_ if central_moments else np.zeros(()) 
                    #     )

                    # else:

                    #     cumulant = _pdf_to_cumulants_bulk(
                    #         pdf, 
                    #         deltas=deltas[cut], 
                    #         ddeltas=D_deltas[cut],
                    #         cut=cut
                    #     )

                    cumulant = _pdf_to_cumulants_bulk(
                        pdf, 
                        deltas=deltas[cut], 
                        ddeltas=D_deltas[cut],
                        cut=cut
                    )

                    fiducial_moments_z_R[n, R * n_cumulants : (R + 1) * n_cumulants] = cumulant               

                    # if n % PRINT_FREQ == 0:
                    #     print("\r n={:05d}/{}".format(n, n_fiducial_pdfs), end="")

        fiducial_vars_z_R = np.mean(fiducial_vars_z_R, axis=0)
        assert fiducial_vars_z_R.shape == (n_scales,), "fiducial_vars_z_R.shape=={}".format(fiducial_vars_z_R.shape)

        if stack_means:
            fiducial_moments_z_R = intersperse_means(fiducial_moments_z_R_means, fiducial_moments_z_R) 

        if use_normalisations:
            fiducial_moments_z_R = intersperse_normalisations(fiducial_normalisations, fiducial_moments_z_R) 

        latin_pdfs_z_R_cut = np.zeros((n_latin_pdfs, cut_dim))
        latin_moments_z_R = np.zeros((n_latin_pdfs, n_scales * n_cumulants))
        latin_moments_z_R_means = np.zeros((n_latin_pdfs, n_scales))
        latin_normalisations = np.zeros((n_latin_pdfs, n_scales))
        for n in trange(n_latin_pdfs, desc="Latins [{}]".format(tqdm_desc_str)):

            # for R, cut in enumerate(cuts):
            for R, (cut, R_i) in enumerate(zip(cuts, R_idx)):

                _cut_dim = sum([_cut.size for _cut in cuts[:R]]) # Cut dimension up to R-th scale

                pdf = latins[z_idx, n, R_i, cut] # NOTE: this should be R_i from R_idx??

                latin_pdfs_z_R_cut[n, _cut_dim : _cut_dim + cut.size] = pdf

                for i in range(len(orders)):
                    order = orders[i]

                    moment = moment_n_R(
                        pdf, n=order, D_deltas_cut=D_deltas[cut], deltas_cut=deltas[cut]
                    )

                    latin_moments_z_R[n, i + R * n_cumulants : (i + 1) + R * n_cumulants] = moment

                # Mean and normalisation of PDF
                latin_moments_z_R_means[n, R] = np.sum(pdf * D_deltas[cut] * deltas[cut])
                latin_normalisations[n, R] = np.sum(pdf * D_deltas[cut])

                # Convert to cumulants
                if cumulants:

                    _delta_ = np.asarray(np.sum(pdf * D_deltas[cut] * deltas[cut]))

                    # if full_shape:
                    #     cumulant = moments_to_cumulants(
                    #         latin_moments_z_R[n, R * n_cumulants : (R + 1) * n_cumulants], 
                    #         _delta_=_delta_ if use_means else np.zeros(()) # _delta_=_delta_ if central_moments else np.zeros(())
                    #     )
                    # else:
                    #     cumulant = _pdf_to_cumulants_bulk(
                    #         pdf, 
                    #         deltas=deltas[cut], 
                    #         ddeltas=D_deltas[cut],
                    #         cut=cut
                    #     )
                    cumulant = _pdf_to_cumulants_bulk(
                        pdf, 
                        deltas=deltas[cut], 
                        ddeltas=D_deltas[cut],
                        cut=cut
                    )

                    latin_moments_z_R[n, R * n_cumulants : (R + 1) * n_cumulants] = cumulant

                # if n % PRINT_FREQ == 0:
                #     print("\r n={:05d}/{}".format(n, n_latin_pdfs), end="")

        if stack_means:
            latin_moments_z_R = intersperse_means(latin_moments_z_R_means, latin_moments_z_R)

        if use_normalisations:
            latin_moments_z_R = intersperse_normalisations(latin_normalisations, latin_moments_z_R) 

        # Including pm axes
        derivative_pdfs_z_R_cut = np.zeros((n_derivatives, n_p, 2, cut_dim))
        derivative_moments_z_R = np.zeros((n_derivatives, n_p, 2, n_scales * n_cumulants)) 
        derivative_moments_z_R_means = np.zeros((n_derivatives, n_p, 2, n_scales)) 
        derivative_normalisations = np.zeros((n_derivatives, n_p, 2, n_scales)) 
        for n in trange(n_derivatives, desc="Derivatives [{}]".format(tqdm_desc_str)):

            # for R, cut in enumerate(cuts):
            for R, (cut, R_i) in enumerate(zip(cuts, R_idx)):

                _cut_dim = sum([_cut.size for _cut in cuts[:R]]) # Cut dimension up to R-th scale
                
                # Including pm axis
                pdf = derivatives[n, z_idx, :, R_i, :, cut] # Shape (p, cut_dim), raw shape (n_derivatives, n_redshifts, n_params, n_scales, 2, n_bins_pdf)

                pdf = np.transpose(pdf, (1, 2, 0)) # Indexing above transposes... shape~(p, p_or_m, n_d)

                # Including pm axis
                derivative_pdfs_z_R_cut[n, :, :, _cut_dim : _cut_dim + cut.size] = pdf # NOTE: choose the right axis; params or redshift

                for i in range(len(orders)):
                    order = orders[i]

                    for p in range(n_p):

                        # Including pm axis
                        for p_or_m in [1, 0]:
                            moment_p = moment_n_R(
                                pdf[p, p_or_m], n=order, D_deltas_cut=D_deltas[cut], deltas_cut=deltas[cut]
                            )

                            derivative_moments_z_R[n, p, p_or_m, i + R * n_cumulants : (i + 1) + R * n_cumulants] = moment_p

                            # Mean and normalisation of PDF
                            derivative_moments_z_R_means[n, p, p_or_m, R] = np.sum(pdf[p, p_or_m] * D_deltas[cut] * deltas[cut]) # PDF at each dp has its own mean
                            derivative_normalisations[n, p, p_or_m, R] = np.sum(pdf[p, p_or_m] * D_deltas[cut]) 

                # Convert moments (or central moments) to cumulants
                if cumulants:
                    for p in range(n_p):

                        # Including pm axis
                        for p_or_m in [1, 0]:

                            _delta_ = np.asarray(np.sum(pdf[p, p_or_m] * D_deltas[cut] * deltas[cut]))

                            # Converting all moments to cumulants at the same time
                            # if full_shape:
                            #     cumulant = moments_to_cumulants(
                            #         derivative_moments_z_R[n, p, p_or_m, R * n_cumulants : (R + 1) * n_cumulants], 
                            #         _delta_=_delta_ if use_means else np.zeros(()) # _delta_=_delta_ if central_moments else np.zeros(())
                            #     )
                            # else:
                            #     cumulant = _pdf_to_cumulants_bulk(
                            #         pdf[p, p_or_m], 
                            #         deltas=deltas[cut], 
                            #         ddeltas=D_deltas[cut], # This is the cut from R-th scale
                            #         cut=cut
                            #     )
                            cumulant = _pdf_to_cumulants_bulk(
                                pdf[p, p_or_m], 
                                deltas=deltas[cut], 
                                ddeltas=D_deltas[cut], # This is the cut from R-th scale
                                cut=cut
                            )

                            derivative_moments_z_R[n, p, p_or_m, R * n_cumulants : (R + 1) * n_cumulants] = cumulant

                # if n % PRINT_FREQ == 0:
                #     print("\r n={:05d}/{}".format(n, n_derivatives), end="")

        if stack_means:
            derivative_moments_z_R = intersperse_means(
                derivative_moments_z_R_means, derivative_moments_z_R
            )

        if use_normalisations:
            derivative_moments_z_R = intersperse_normalisations(
                derivative_normalisations, derivative_moments_z_R
            ) 

        # Euler derivative for PDFs from plus minus statistics
        derivative_pdfs_z_R_cut = derivative_pdfs_z_R_cut[:, :, 1, :] - derivative_pdfs_z_R_cut[:, :, 0, :]
        for p in range(n_p):
            derivative_pdfs_z_R_cut[:, p, ...] = derivative_pdfs_z_R_cut[:, p, ...] / dparams[p] # NOTE: parameter / redshifts axis!!!!!

        # Euler derivative for moments from plus minus statistics
        derivative_moments_z_R = derivative_moments_z_R[:, :, 1, :] - derivative_moments_z_R[:, :, 0, :]
        for p in range(n_p):
            derivative_moments_z_R[:, p, ...] = derivative_moments_z_R[:, p, ...] / dparams[p] # NOTE: parameter / redshifts axis!!!!!

        logger.info(
            "Fiducials: {} \n Latins: {} \n Derivatives: {}".format(
                fiducial_pdfs_z_R_cut.shape, 
                latin_pdfs_z_R_cut.shape, 
                derivative_pdfs_z_R_cut.shape
            )
        )

        if FIDUCIAL_REDUCE:
            logger.info("REDUCING CUMULANTS (assuming using m_0, m_1).")

            assert stack_means and use_normalisations, "Reduction index below here is wrong if this is the case!"

            for r, r_i in enumerate(R_idx):

                print(fiducial_moments_z_R.shape, fiducial_vars_z_R.shape, fiducial_moments_z_R[:, r * 5 : (r + 1) * 5].shape)

                # Only divide skewness and kurtoses by mean fiducial variance (assuming norm/mean included)
                # 5 'cumulants' including m_0, m_1
                _vars = np.asarray([fiducial_vars_z_R[r] ** 2., fiducial_vars_z_R[r] ** 3.]) # np.tile(fiducial_vars_z_R[r], (2,)) # Tile to [skew, kurtosis] shape
                fiducial_moments_z_R[:, r * 5 : (r + 1) * 5][:, 3:] /= _vars
                latin_moments_z_R[:, r * 5 : (r + 1) * 5][:, 3:] /= _vars
                derivative_moments_z_R[:, :, r * 5 : (r + 1) * 5][:, :, 3:] /= _vars

        """
            Datasets
        """

        # Fisher information in cumulants of bulk of the PDF
        n_fiducial_moments, data_dim_moments = fiducial_moments_z_R.shape
        C_moments = np.cov(fiducial_moments_z_R, rowvar=False)

        corr_moments = jnp.corrcoef(fiducial_moments_z_R, rowvar=False)

        filename = os.path.join(log_figs_dir, "corr_coeff_moments.png")
        plt.figure()
        plt.title("Correlation matrix (moments) [{}]".format(bulk_or_tails))
        plt.imshow(corr_moments, cmap="coolwarm")
        plt.colorbar()
        plt.savefig(filename)
        plt.close()
        logger.debug("Saved correlation matrix (moments) figure at: \n\t{}".format(filename))

        # assert C_moments.T == C_moments, "Non-symmetric cumulant covariance."

        # Ill-conditioned matrix when using means
        # print("Conditioning matrix...")
        # C_moments = C_moments + np.eye(data_dim_moments) * 1e-8

        # assert np.all(np.linalg.eigvals(C_moments) > 0)

        H = hartlap(n_s=n_fiducial_moments, n_d=data_dim_moments)
        Cinv_moments = H * np.linalg.inv(C_moments)
        dmu_moments = np.mean(derivative_moments_z_R, axis=0)
        F_moments = np.linalg.multi_dot([dmu_moments, Cinv_moments, dmu_moments.T])
        Finv_moments = np.linalg.inv(F_moments)

        # Cumulants[bulk]
        dataset = Dataset(
            name=bulk_or_tails,
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

        # Remove response in signal from parameters that cannot be constrained at fixed z
        if config.freeze_parameters:
            dataset = freeze_out_parameters_dataset(dataset)

        # If requiring PDFs return dataset for bulk of the PDF (not cumulants of the bulk) NOTE: check this.s... NOTE: check this.s... NOTE: check this.s... NOTE: check this.s...
        return_dataset = dataset

        if pdfs:
            # Fisher information in bulk of the PDF
            _, data_dim_pdfs = fiducial_pdfs_z_R_cut.shape 
            C_pdf = np.cov(fiducial_pdfs_z_R_cut, rowvar=False) 
            H = hartlap(n_s=n_fiducial_pdfs, n_d=data_dim_pdfs) 
            Cinv_pdf = H * np.linalg.inv(C_pdf)
            dmu_pdfs = np.mean(derivative_pdfs_z_R_cut, axis=0)
            F_pdf = jnp.linalg.multi_dot([dmu_pdfs, Cinv_pdf, dmu_pdfs.T])
            Finv_pdf = np.linalg.inv(F_pdf)

            # PDF[bulk]
            pdf_dataset = Dataset(
                name="{}_pdf".format(bulk_or_tails),
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
            
            corr_pdf = np.corrcoef(fiducial_pdfs_z_R_cut, rowvar=False) 

            filename = os.path.join(log_figs_dir, "corr_coeff_pdfs.png")
            plt.figure()
            plt.title("Correlation matrix (PDFs) [{}]".format(bulk_or_tails))
            plt.imshow(corr_pdf, cmap="coolwarm")
            plt.colorbar()
            plt.savefig(filename)
            plt.close()
            logger.debug("Saved correlation matrix (PDFs) figure at: \n\t{}".format(filename))

            if config.freeze_parameters:
                pdf_dataset = freeze_out_parameters_dataset(pdf_dataset)

            return_dataset = pdf_dataset 

            logger.info("Returning PDFs as dataset...")

        # NOTE: whether PDFs or cumulants convert to linearised dataset if so required...
        if config.linearised:
            logger.info("Using linearised dataset [replacing only hypercube]...")

            D, Y = get_linearised_data(config, return_dataset) 

            return_dataset = replace(return_dataset, data=D, parameters=Y)

        if NON_GAUSSIAN_TEST:
            logger.info("Using non-Gaussian linear model dataset [replacing only hypercube]...")

            D, Y = get_linearised_data(config, return_dataset) 

            return_dataset = replace(return_dataset, data=D, parameters=Y)

        # Save return dataset to ensure loading (not creating) next time around
        np.savez(dataset_filename, **asdict(return_dataset))

        logger.info("Saved dataset:\n\t{}".format(dataset_filename))

        return return_dataset # NOTE: why was this here?

    # Create a fresh dataset if required, or generate one if it does not exist
    if not FORCE_RECOMPUTE_DATASET:
        try:
            logger.info("Loading dataset:\n\t{}".format(dataset_filename))

            dataset_dict = np.load(dataset_filename, allow_pickle=True) 

            if pdfs:
                dataset_name = "{}_pdf".format("bulk" if not full_shape else "tails") 
            else:
                dataset_name = "bulk" if not full_shape else "tails"

            return_dataset = Dataset(
                name=dataset_name,
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

            return_dataset = generate_dataset()

            logger.info("Generated dataset:\n\t{}".format(dataset_filename))
    else:
        logger.info("Generating dataset:\n\t{}".format(dataset_filename))

        return_dataset = generate_dataset()

        logger.info("Generated dataset:\n\t{}".format(dataset_filename))

    return return_dataset 


"""
    Dataset
"""


@dataclass
class BulkCumulantsDataset:
    """ 
        Dataset for Simulation-Based Inference with cumulants of the bulk of the matter PDF 
    """

    config: ConfigDict
    data: Dataset
    prior: tfd.Distribution
    compression_fn: Optional[Callable[[Array, Array], Array]]
    results_dir: str

    def __init__(
        self, 
        config: ConfigDict, 
        *, 
        pdfs: bool = False,
        results_dir: Optional[str] = None
    ):
        self.config = config

        self.data = get_calculated_cumulants_data(
            config, 
            pdfs=pdfs,
            use_means=config.use_means,
            use_normalisations=config.use_normalisations,
            stack_means=config.stack_means,
            full_shape=False,
            results_dir=results_dir
        )

        self.prior = get_prior(config, self.data) # Possibly not equal to Quijote prior

        # key = jr.key(config.seed)
        # self.compression_fn = get_compression_fn(
        #     key, self.config, self.data, results_dir=results_dir
        # )
        self.compression_fn = None

        self.results_dir = results_dir

        logger.info("BULK CUMULANT DATASET")
        logger.info(
            ">DATA:\n\t {}".format(
                ["{:.3E} {:.3E}".format(_.min(), _.max()) for _ in (self.data.fiducial_data, self.data.data)]
            )
        )
        logger.info(
            ">DATA / PARAMETERS:\n\t {}".format(
                [_.shape for _ in (self.data.data, self.data.parameters)]
            )
        )

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
        if self.compression_fn is None:
            key = jr.key(self.config.seed)
            fn = get_compression_fn(
                key, self.config, self.data, results_dir=self.results_dir
            )
            assert callable(fn), "Compression function returned is not callable"
            self.compression_fn = fn
        assert self.compression_fn is not None
        return self.compression_fn

    def get_datavector(self, key: PRNGKeyArray, n: int = 1) -> Float[Array, "... d"]:
        d = get_datavector(key, config=self.config, dataset=self.data, n=n)
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


@dataclass
class TailsCumulantsDataset:
    """ 
        Dataset for Simulation-Based Inference with cumulants of the bulk of the matter PDF 
    """

    config: ConfigDict
    data: Dataset
    prior: tfd.Distribution
    compression_fn: Optional[Callable[[Array, Array], Array]]
    results_dir: str

    def __init__(
        self, 
        config: ConfigDict, 
        *, 
        pdfs: bool = False,
        results_dir: Optional[str] = None
    ):
        self.config = config

        self.data = get_calculated_cumulants_data(
            config, 
            pdfs=pdfs,
            use_means=config.use_means,
            use_normalisations=config.use_normalisations,
            stack_means=config.stack_means,
            full_shape=True, # Implies full-shape calculation
            results_dir=results_dir
        )

        self.prior = get_prior(config, self.data) # Possibly not equal to Quijote prior

        # key = jr.key(config.seed)
        # self.compression_fn = get_compression_fn(
        #     key, self.config, self.data, results_dir=results_dir
        # )
        self.compression_fn = None

        self.results_dir = results_dir

        logger.info("TAILS CUMULANT DATASET")
        logger.info(
            ">DATA:\n\t {}".format(
                ["{:.3E} {:.3E}".format(_.min(), _.max()) for _ in (self.data.fiducial_data, self.data.data)]
            )
        )
        logger.info(
            ">DATA / PARAMETERS:\n\t {}".format(
                [_.shape for _ in (self.data.data, self.data.parameters)]
            )
        )

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
        if self.compression_fn is None:
            key = jr.key(self.config.seed)
            fn = get_compression_fn(
                key, self.config, self.data, results_dir=self.results_dir
            )
            assert callable(fn), "Compression function returned is not callable"
            self.compression_fn = fn
        return self.compression_fn

    def get_datavector(self, key: PRNGKeyArray, n: int = 1) -> Float[Array, "... d"]:
        d = get_datavector(key, config=self.config, dataset=self.data, n=n)
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


class BulkPDFsDataset(BulkCumulantsDataset):
    """ 
        Dataset for Simulation-Based Inference with the bulk of the matter PDF 
    """

    def __init__(
        self,
        config: ConfigDict,
        *,
        results_dir: Optional[str] = None
    ):
        super().__init__(config, pdfs=True, results_dir=results_dir)


def get_bulk_dataset(args, pdfs=False):
    # Take non-bulk config, get bulk config, get dataset, extract Finv

    config = bulk_cumulants_config(
        seed=args.seed, 
        redshift=args.redshift, 
        linearised=args.linearised, 
        compression=args.compression,
        order_idx=args.order_idx,
        scales=args.scales,
        n_linear_sims=args.n_linear_sims,
        pre_train=args.pre_train,
        freeze_parameters=args.freeze_parameters
    )

    if pdfs: 
        logger.info("Using PDF dataset for bulk dataset.")
    else:
        logger.info("Using cumulants dataset for bulk dataset.")

    dataset = BulkCumulantsDataset(config, pdfs=pdfs)

    return dataset.data


def get_multi_z_bulk_pdf_fisher_forecast(args):
    # Get bulk PDF dataset for multiple redshifts

    F = np.zeros(())
    for redshift in args.redshifts: #[0.0, 0.5, 1.0]:

        config = bulk_cumulants_config(
            seed=args.seed, 
            redshift=redshift, # Force redshift!
            linearised=args.linearised, 
            compression=args.compression,
            order_idx=args.order_idx,
            scales=args.scales,
            n_linear_sims=args.n_linear_sims,
            pre_train=args.pre_train,
            freeze_parameters=args.freeze_parameters
        )

        logger.info("Using PDF dataset for bulk dataset. z={}".format(redshift))

        dataset = BulkCumulantsDataset(config, pdfs=True)

        F_z = np.linalg.inv(dataset.data.Finv)
        F = F + F_z

    Finv = np.linalg.inv(F)

    return Finv


def load_multi_z_bulk_pdf_fisher_forecast(data_dir, args):
    """
        Load Fisher inverse matrix of PDF dataset, over multiple redshifts, consistently with args
    """
    identifier_str = "".join(
        [
            "_R" + "".join(map(str, args.scales)),
            "_z" + "".join(map(str, args.redshifts)),
            "_f" if args.freeze_parameters else "_nf",
            "_pdfs"
        ]
    )

    Finv_file_path = os.path.join(
        data_dir, "Finv_bulk_pdfs_all_z_{}.npy".format(identifier_str)
    )

    if not FORCE_RECOMPUTE_DATASET:
        try:
            Finv_bulk_pdfs_all_z = np.load(Finv_file_path)
        except:
            Finv_bulk_pdfs_all_z = get_multi_z_bulk_pdf_fisher_forecast(args)

            np.save(Finv_file_path, Finv_bulk_pdfs_all_z)
        else:
            Finv_bulk_pdfs_all_z = get_multi_z_bulk_pdf_fisher_forecast(args)

    # Don't save with Fisher information from Planck
    Finv_bulk_pdfs_all_z = add_planck_information_to_Finv(
        Finv_bulk_pdfs_all_z, use_planck=args.use_planck
    )

    logger.info("Finv bulk PDFs all z loaded from:\n\t{}".format(Finv_file_path))

    return Finv_bulk_pdfs_all_z


if __name__ == "__main__":
    from configs import bulk_cumulants_config
    from sbiax.utils import make_df, marker

    config = bulk_cumulants_config()

    config.use_bulk_means = True

    dataset = BulkCumulantsDataset(config)

    def mle(d):
        return dataset.alpha + jnp.linalg.multi_dot(
            [dataset.Finv, dataset.derivatives.mean(axis=0), dataset.Cinv, d - dataset.fiducial_data.mean(axis=0)]
        )

    X = jax.vmap(mle)(dataset.fiducial_data)

    X_df = make_df(
        X, parameter_strings=dataset.parameter_strings
    )

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
    c.add_chain(
        Chain(samples=X_df, name="X", color="r")
    )
    c.add_marker(
        location=marker(
            dataset.alpha, parameter_strings=dataset.parameter_strings
        ),
        name=r"$\alpha$", 
        color="#7600bc"
    )
    fig = c.plotter.plot()
    plt.savefig("pdfs_test.pdf")
    plt.close()