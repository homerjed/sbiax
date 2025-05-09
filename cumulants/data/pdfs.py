import os
from dataclasses import dataclass, replace, asdict
from functools import partial
from typing import Tuple, Callable, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray, Float, Int, Array, Scalar, jaxtyped
import equinox as eqx
from beartype import beartype as typechecker
from ml_collections import ConfigDict

import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import qmc
from chainconsumer import Chain, ChainConsumer, Truth
import tensorflow_probability.substrates.jax.distributions as tfd
from tqdm.auto import trange

from data.constants import (
    get_quijote_parameters, 
    get_save_and_load_dirs, 
    get_target_idx
)
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
from sbiax.utils import make_df, marker
from configs import bulk_cumulants_config
from compression.nn import fit_nn, fit_nn_lbfgs

typecheck = jaxtyped(typechecker=typechecker)

# Re-compute datasets if so desired
FORCE_RECOMPUTE_DATASET = True if os.environ.get("FORCE_RECOMPUTE_DATASET", "").lower() in ("1", "true") else False 

"""
    Get cumulants of the bulk of the 3D matter PDF
"""


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

    DELTA_BIN_EDGES = np.geomspace(1e-2, 1e2, num=100) # 1911.11158 Section 4.1, NOTE: This is in rho
    D_DELTAS = DELTA_BIN_EDGES[1:] - DELTA_BIN_EDGES[:-1] 

    return fiducials, latins, latin_parameters, derivatives, deltas, D_DELTAS


def get_bulk_cumulants_data(
    config: ConfigDict, 
    *, 
    pdfs: bool = False, # Use PDFs or cumulants for the bulk
    verbose: bool = False, 
    use_mean: bool = False,
    use_bulk_norms: bool = True, # Stack means of bulk of the PDF at each scale with the other cumulants
    stack_bulk_means: bool = True,
    check_cumulants_against_quijote: bool = False,
    results_dir: Optional[str] = None
) -> Dataset:
    """
        Get dataset for SBI experiments with the cumulants.
        - Cut the PDFs according to a p_min, p_max cut into the CDF which 
          indexes the bins of the PDF.
        - Return PDFs of cumulants of the bulk
    """

    print("Using bulk means..." if use_mean else "Not using bulk means...")

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

    cumulants          = True             # Use cumulants over moments (NOTE: check not calculating reduced-cumulants, Quijote uses cumulants)
    use_mean           = use_mean         # Concatenate mean of bulk to datavector
    stack_mean         = stack_bulk_means # For full shape <delta> is very close to zero but <rho> approximately one
    use_normalisations = use_bulk_norms   # Stack M_0 normalisation of pdf into datavector ahead of mean M_1 
    normalise          = False            # Divide moments by M_0

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

    dataset_identifier_str = "".join(
        [
            "_R" + "".join(map(str, config.scales)),
            "_m" + "".join(map(str, config.order_idx)),
            "_z" + str(config.redshift),
            "_f" if config.freeze_parameters else "_nf",
            "_pdfs" if pdfs else "", 
            "_with_means" if use_mean else "",
            "_with_norms" if use_normalisations else "",
            "_with_means_stacked" if stack_mean else ""
        ]
    )

    def generate_dataset():
        (
            fiducials, 
            latins, 
            latin_parameters, 
            derivatives, 
            deltas, 
            D_deltas
        ) = get_raw_data(data_dir, verbose=verbose)

        """
            Calculate CDF and cut PDFs
        """

        # Calculate mean of fiducial PDFS across scales R for cutting with CDF
        fiducial_pdfs_stacked_mean = np.zeros((n_bins_pdf * n_scales,))
        for R in R_idx:
            mean_z_R = jnp.mean(fiducials[z_idx, :, R, :], axis=0)

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

        # Check CDF bounds
        assert np.isclose(cdf.min(), 0.) and np.isclose(cdf.max(), 1.), (
            "CDF: min={} max={}".format(cdf.min(), cdf.max())
        )

        # Cut indices for each value of R
        if check_cumulants_against_quijote:
            cuts = [np.arange(n_bins_pdf) for R, _ in enumerate(config.scales)] # Use all bins
        else:
            cuts = [
                np.where(
                    (cdf[R * n_bins_pdf : (R + 1) * n_bins_pdf] > p_value_min) & \
                    (cdf[R * n_bins_pdf : (R + 1) * n_bins_pdf] < p_value_max)
                )[0] 
                for R, _ in enumerate(config.scales)
            ]
        if verbose:
            print("CUTS (idx):\n", [(min(cut), max(cut)) for cut in cuts])
            print("CUTS (deltas) min/max={:.1f}/{:.1f}:\n".format(
                deltas.min(), deltas.max()), ["{:.1f} {:.1f}".format(min(deltas[cut]), max(deltas[cut])) for cut in cuts]
            )
            print("Cut total dim:", sum([_.size for _ in cuts]))

        # Get cut dimensions
        z_cut_dim_totals = 0
        for R, cut_z in zip(config.scales, cuts):
            z_cut_dim_totals += cut_z.shape[0]
            if verbose:
                print(f" R={R} cut: {cut_z.shape}")

        if verbose:
            print("All cuts added:", z_cut_dim_totals)
            print(f"\n>CDF shape: {cdf.shape}") 
            print(f"Total bins kept in CDF cut: {z_cut_dim_totals}/{1 * n_scales * n_bins_pdf}")

        """
            Cut PDFs to bulk density cut and calculate moments/cumulants
        """

        def normalise_cut_pdf(pdf, D_deltas_cut):
            # Renormalise PDF in bulk region (pdf) which is already normalised

            if verbose:
                print("PDF (before)", jnp.sum(pdf), jnp.sum(pdf * D_deltas_cut))

            if normalise:
                pdf = pdf / jnp.sum(pdf * D_deltas_cut) # Denominator is PDF integral
            else:
                pdf = pdf # Don't normalise: might throw out information for extreme cosmologies

            if verbose:
                print("PDF (after)", jnp.sum(pdf), jnp.sum(pdf * D_deltas_cut))

            return pdf 

        def moment_n_R(pdf, n, D_deltas_cut, deltas_cut):
            pdf = normalise_cut_pdf(pdf, D_deltas_cut)

            # This is rho not delta; => <rho> = 1, must subtract it 
            mu_delta = jnp.sum(deltas_cut * pdf * D_deltas_cut)

            # Calculate moment n (NOTE: don't centre it around bulk mean of deltas)
            moment_n = jnp.sum(pdf * D_deltas_cut * ((deltas_cut - mu_delta) ** n)) # = p(x)dx * (x ** n)

            return moment_n

        @typecheck
        def moments_to_cumulants(moments: Float[np.ndarray, "k_n"], _delta_: Scalar) -> Float[np.ndarray, "k_n"]:
            # Bernardeau 2002 Eq. 130

            cumulant_2 = moments[0] - (_delta_) ** 2.
            cumulant_3 = moments[1] - 3. * cumulant_2 * _delta_ - _delta_ ** 3. 
            cumulant_4 = moments[2] - 4. * cumulant_3 * _delta_ - 3. * (cumulant_2 ** 2.) - 6. * cumulant_2 * (_delta_ ** 2.) - (_delta_ ** 4.)

            cumulants = np.asarray([cumulant_2, cumulant_3, cumulant_4]) 

            return cumulants

        def intersperse_means(means: Float[np.ndarray, "k_n"], moments: Float[np.ndarray, ""]):
            # Put means first in moments vectors for all scales i.e. [[mean, var, skew, kurt]_R, ...]
            assert len(means) == len(moments)
            assert means.shape[-1] == n_scales

            means_and_moments = np.zeros(moments.shape[:-1] + (n_scales * (1 + n_cumulants),)) # Mean + cumulants for all scales, additional shape info for derivatives or not

            print("FULL MEANS MOMENTS", means_and_moments.shape)

            # Stack cumulants such that at each scale: M_R = [M_1, k_n]
            for r in range(n_scales):
                _means_and_moments = [
                    means[..., [r]], # Keep last dimension
                    moments[..., r * n_cumulants : (r + 1) * n_cumulants]
                ]
                print("MEANS MOMENTS", [_.shape for _ in _means_and_moments])

                # Stack on last axis
                means_and_moments[
                    ..., r * (1 + n_cumulants) : (r + 1) * (1 + n_cumulants)
                ] = np.concatenate(_means_and_moments, axis=-1)

            return means_and_moments

        def intersperse_normalisations(normalisations, moments):
            # Put means first in moments vectors for all scales i.e. [[mean, var, skew, kurt]_R, ...]
            assert len(normalisations) == len(moments)
            assert normalisations.shape[-1] == n_scales

            dim = 2 + n_cumulants # Mean and normalisation plus usual cumulants

            normalisations_and_moments = np.zeros(moments.shape[:-1] + (n_scales * dim,)) # Mean + cumulants for all scales, additional shape info for derivatives or not

            print("NORMALISATIONS AND MOMENTS", normalisations_and_moments.shape)

            # Stack cumulants such that at each scale: M_R = [M_0, M_1, k_n]
            for r in range(n_scales):
                _normalisations_and_moments = [
                    normalisations[..., [r]], # Keep last dimension
                    moments[..., r * (n_cumulants + 1) : (r + 1) * (n_cumulants + 1)] # Additional +1 for stacked mean
                ]
                print("NORMALISATIONS MOMENTS", [_.shape for _ in _normalisations_and_moments])

                # Stack on last axis
                normalisations_and_moments[
                    ..., r * dim : (r + 1) * dim
                ] = np.concatenate(_normalisations_and_moments, axis=-1)

            return normalisations_and_moments

        orders = [2, 3, 4] # Variance, skewness and kurtosis
        cut_dim = sum([cut.size for cut in cuts])

        assert np.all([cut.ndim == 1 for cut in cuts]), (
            "Rank of cut indices arrays not equal to 1, shapes are: {}".format([cut.ndim for cut in cuts])
        )

        fiducial_pdfs_z_R_cut = np.zeros((n_fiducial_pdfs, cut_dim)) # Bulk PDFs
        fiducial_moments_z_R = np.zeros((n_fiducial_pdfs, n_scales * n_cumulants)) # Cumulants of bulk PDFs
        fiducial_moments_z_R_means = np.zeros((n_fiducial_pdfs, n_scales)) # Means of bulk PDFs
        fiducial_normalisations = np.zeros((n_fiducial_pdfs, n_scales)) # Normalisations of bulk PDFs
        for n in trange(n_fiducial_pdfs, desc="Fiducials"):
            
            # Using R-th chosen scale and its cut indices into mean fiducial PDF
            for R, cut in enumerate(cuts):

                _cut_dim = sum([cut.size for cut in cuts[:R]]) # Cut dimension up to R-th scale

                pdf = fiducials[z_idx, n, R, cut] # Cut PDF p(d_i)

                fiducial_pdfs_z_R_cut[n, _cut_dim : _cut_dim + cut.size] = pdf

                for i in range(len(orders)): # Cycle through moment orders
                    order = orders[i]

                    moment = moment_n_R(
                        pdf, n=order, D_deltas_cut=D_deltas[cut], deltas_cut=deltas[cut]
                    )

                    fiducial_moments_z_R[n, i + R * n_cumulants : (i + 1) + R * n_cumulants] = moment

                # Mean for calculating cumulants from moments
                if use_mean:
                    _delta_ = jnp.sum(pdf * D_deltas[cut] * deltas[cut]) # delta_R
                else:
                    _delta_ = jnp.zeros() # Zero or one? rho=1 for full PDF but not necessarily for bulk

                if verbose:
                    print("mean", _delta_)

                fiducial_moments_z_R_means[n, R] = _delta_
                fiducial_normalisations[n, R] = jnp.sum(pdf * D_deltas[cut])

                # Convert to cumulants (process all orders simultaneously)
                if cumulants:
                    cumulant = moments_to_cumulants(
                        fiducial_moments_z_R[n, R * n_cumulants : (R + 1) * n_cumulants], 
                        _delta_=_delta_
                    )

                    fiducial_moments_z_R[n, R * n_cumulants : (R + 1) * n_cumulants] = cumulant               

                    if verbose: 
                        print("CUMULANT", cumulant)

                if verbose:
                    print("\r n={:05d}/{}".format(n, n_fiducial_pdfs), end="")

        if stack_mean:
            fiducial_moments_z_R = intersperse_means(fiducial_moments_z_R_means, fiducial_moments_z_R) 

        if use_normalisations:
            fiducial_moments_z_R = intersperse_normalisations(fiducial_normalisations, fiducial_moments_z_R) 

        latin_pdfs_z_R_cut = np.zeros((n_latin_pdfs, cut_dim))
        latin_moments_z_R = np.zeros((n_latin_pdfs, n_scales * n_cumulants))
        latin_moments_z_R_means = np.zeros((n_latin_pdfs, n_scales))
        latin_normalisations = np.zeros((n_latin_pdfs, n_scales))
        for n in trange(n_latin_pdfs, desc="Latins"):

            for R, cut in enumerate(cuts):

                _cut_dim = sum([cut.size for cut in cuts[:R]])

                pdf = latins[z_idx, n, R, cut]

                latin_pdfs_z_R_cut[n, _cut_dim : _cut_dim + cut.size] = pdf

                for i in range(len(orders)):
                    order = orders[i]

                    moment = moment_n_R(
                        pdf, n=order, D_deltas_cut=D_deltas[cut], deltas_cut=deltas[cut]
                    )

                    latin_moments_z_R[n, i + R * n_cumulants : (i + 1) + R * n_cumulants] = moment

                # Mean for calculating cumulants from moments
                if use_mean:
                    _delta_ = jnp.sum(pdf * D_deltas[cut] * deltas[cut])
                else:
                    _delta_ = jnp.zeros()

                if verbose:
                    print("mean", _delta_)

                latin_moments_z_R_means[n, R] = _delta_
                latin_normalisations[n, R] = jnp.sum(pdf * D_deltas[cut])

                # Convert to cumulants
                if cumulants:
                    cumulant = moments_to_cumulants(
                        latin_moments_z_R[n, R * n_cumulants : (R + 1) * n_cumulants], 
                        _delta_=_delta_
                    )

                    latin_moments_z_R[n, R * n_cumulants : (R + 1) * n_cumulants] = cumulant

                if verbose:
                    print("\r n={:05d}/{}".format(n, n_latin_pdfs), end="")

        if stack_mean:
            latin_moments_z_R = intersperse_means(latin_moments_z_R_means, latin_moments_z_R)

        if use_normalisations:
            latin_moments_z_R = intersperse_normalisations(latin_normalisations, latin_moments_z_R) 

        # Including pm axes
        derivative_pdfs_z_R_cut = np.zeros((n_derivatives, n_p, 2, cut_dim))
        derivative_moments_z_R = np.zeros((n_derivatives, n_p, 2, n_scales * n_cumulants)) 
        derivative_moments_z_R_means = np.zeros((n_derivatives, n_p, 2, n_scales)) 
        derivative_normalisations = np.zeros((n_derivatives, n_p, 2, n_scales)) 
        for n in trange(n_derivatives, desc="Derivatives"):

            for R, cut in enumerate(cuts):

                _cut_dim = sum([cut.shape[0] for cut in cuts[:R]])
                
                # Including pm axis
                pdf = derivatives[n, z_idx, :, R, :, cut] # Shape (p, cut_dim), raw shape (n_derivatives, n_redshifts, n_params, n_scales, 2, n_bins_pdf)

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

                            # Mean for calculating cumulants from moments, exists for each plus/minus PDF
                            if use_mean:
                                _delta_ = jnp.sum(pdf[p, p_or_m] * D_deltas[cut] * deltas[cut])
                            else:
                                _delta_ = jnp.zeros() 

                            if verbose:
                                print("mean", _delta_)

                            derivative_moments_z_R_means[n, p, p_or_m, R] = _delta_ # PDF at each dp has its own mean
                            derivative_normalisations[n, p, p_or_m, R] = jnp.sum(pdf[p, p_or_m] * D_deltas[cut]) # PDF at each dp has its own norm 

                # Convert to cumulants
                if cumulants:
                    for p in range(n_p):

                        # Including pm axis
                        for p_or_m in [1, 0]:

                            if use_mean:
                                _delta_ = jnp.sum(pdf[p, p_or_m] * D_deltas[cut] * deltas[cut])
                            else:
                                _delta_ = jnp.zeros()

                            # Converting all moments to cumulants at the same time
                            cumulant = moments_to_cumulants(
                                derivative_moments_z_R[n, p, p_or_m, R * n_cumulants : (R + 1) * n_cumulants], 
                                _delta_=_delta_
                            )

                            derivative_moments_z_R[n, p, p_or_m, R * n_cumulants : (R + 1) * n_cumulants] = cumulant

                if verbose:
                    print("\r n={:05d}/{}".format(n, n_derivatives), end="")

        if stack_mean:
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

        print(
            "Fiducials, latins (one redshift):\n", 
            fiducial_pdfs_z_R_cut.shape, 
            latin_pdfs_z_R_cut.shape, 
            derivative_pdfs_z_R_cut.shape
        )

        """
            Datasets
        """

        # Fisher information in cumulants of bulk of the PDF
        n_fiducial_moments, data_dim_moments = fiducial_moments_z_R.shape
        C_moments = jnp.cov(fiducial_moments_z_R, rowvar=False)

        # assert C_moments.T == C_moments, "Non-symmetric cumulant covariance."

        # Ill-conditioned matrix when using means
        print("Conditioning matrix...")
        C_moments = C_moments + jnp.eye(data_dim_moments) * 1e-8

        # assert np.all(np.linalg.eigvals(C_moments) > 0)

        H = (n_fiducial_moments - data_dim_moments - 2.) / (n_fiducial_moments - 1.)
        Cinv_moments = H * jnp.linalg.inv(C_moments)
        dmu_moments = jnp.mean(derivative_moments_z_R, axis=0)
        F_moments = jnp.linalg.multi_dot([dmu_moments, Cinv_moments, dmu_moments.T])
        Finv_moments = jnp.linalg.inv(F_moments)

        # Cumulants[bulk]
        dataset = Dataset(
            name="bulk",
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

        return_dataset = dataset # If not requiring PDFs

        # If requiring PDFs return dataset for bulk of the PDF (not cumulants of the bulk)
        if (not check_cumulants_against_quijote) or pdfs: # If wanting PDFs or not wanting to check calculated cumulants 

            print("Returning PDFs as dataset...")

            # Fisher information in bulk of the PDF
            _, data_dim_pdfs = fiducial_pdfs_z_R_cut.shape 
            C_pdf = np.cov(fiducial_pdfs_z_R_cut, rowvar=False) 
            H = (n_fiducial_pdfs - data_dim_pdfs - 2.) / (n_fiducial_pdfs - 1.)
            Cinv_pdf = H * np.linalg.inv(C_pdf)
            dmu_pdfs = np.mean(derivative_pdfs_z_R_cut, axis=0)
            F_pdf = jnp.linalg.multi_dot([dmu_pdfs, Cinv_pdf, dmu_pdfs.T])
            Finv_pdf = np.linalg.inv(F_pdf)

            # PDF[bulk]
            pdf_dataset = Dataset(
                name="bulk_pdf",
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

            def mle_pdf(d, p):
                mu = jnp.mean(fiducial_pdfs_z_R_cut, axis=0)
                dmu = jnp.mean(derivative_pdfs_z_R_cut, axis=0)
                mu_p = mu + jnp.dot(p - alpha, dmu)
                return p + jnp.linalg.multi_dot([Finv_pdf, dmu, Cinv_pdf, d - mu_p])

            X_pdfs = jax.vmap(mle_pdf)(latin_pdfs_z_R_cut, latin_parameters)

            if config.freeze_parameters:
                pdf_dataset = freeze_out_parameters_dataset(pdf_dataset)

            return_dataset = pdf_dataset if pdfs else dataset # NOTE: careful with this logic and if ... above

        # Save return dataset to ensure loading (not creating) next time around
        np.savez(
            os.path.join(data_dir, "bulk_cumulants_dataset_{}.npz".format(dataset_identifier_str)), 
            **asdict(return_dataset)
        )
        return return_dataset

    # Try loading dataset instead of deriving it again and again NOTE: careful not to load PDFs when yhou need cumulants etc
    if not FORCE_RECOMPUTE_DATASET:
        try:
            dataset_filename = os.path.join(
                data_dir, "bulk_cumulants_dataset_{}.npz".format(dataset_identifier_str)
            )
            dataset_dict = np.load(dataset_filename, allow_pickle=True) # dataset_dict = jax.tree.map(lambda x: jnp.asarray(x), dataset_dict, is_leaf=lambda x: isinstance(x, np.ndarray))

            return_dataset = Dataset(
                name="bulk_pdf" if pdfs else "bulk",
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

            print("Loaded dataset:\n\t", dataset_filename)
        except FileNotFoundError:
            return_dataset = generate_dataset()
    else:
        return_dataset = generate_dataset()

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
    compression_fn: Callable
    results_dir: str

    def __init__(
        self, 
        config: ConfigDict, 
        *, 
        pdfs: bool = False,
        verbose: bool = False, 
        check_cumulants_against_quijote: bool = False,
        results_dir: Optional[str] = None
    ):
        self.config = config

        self.data = get_bulk_cumulants_data(
            config, 
            pdfs=pdfs,
            use_mean=config.use_bulk_means,
            use_bulk_norms=config.stack_bulk_norms,
            stack_bulk_means=config.stack_bulk_means,
            check_cumulants_against_quijote=check_cumulants_against_quijote, 
            verbose=verbose, 
            results_dir=results_dir
        )

        self.prior = get_prior(config, self.data) # Possibly not equal to Quijote prior

        key = jr.key(config.seed)
        self.compression_fn = get_compression_fn(
            key, self.config, self.data, results_dir=results_dir
        )

        self.results_dir = results_dir

        print("PDFS DATASET")
        print(">DATA:\n\t", ["{:.3E} {:.3E}".format(_.min(), _.max()) for _ in (self.data.fiducial_data, self.data.data)])
        print(">DATA / PARAMETERS:\n\t", [_.shape for _ in (self.data.data, self.data.parameters)])

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


class BulkPDFsDataset(BulkCumulantsDataset):
    """ 
        Dataset for Simulation-Based Inference with the bulk of the matter PDF 
    """

    def __init__(
        self,
        config: ConfigDict,
        *,
        verbose: bool = False,
        check_cumulants_against_quijote: bool = False,
        results_dir: Optional[str] = None
    ):
        super().__init__(
            config,
            pdfs=True,
            verbose=verbose,
            check_cumulants_against_quijote=check_cumulants_against_quijote,
            results_dir=results_dir
        )


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
        pre_train=args.pre_train,
        freeze_parameters=args.freeze_parameters
    )

    if pdfs: 
        print("Using PDF dataset for bulk dataset.")
    else:
        print("Using cumulants dataset for bulk dataset.")

    dataset = BulkCumulantsDataset(config, pdfs=pdfs, verbose=False)

    return dataset.data


def get_multi_z_bulk_pdf_fisher_forecast(args):
    # Get bulk PDF dataset for multiple redshifts

    F = np.zeros(())
    for redshift in [0.0, 0.5, 1.0]:

        config = bulk_cumulants_config(
            seed=args.seed, 
            redshift=redshift, # Force redshift!
            reduced_cumulants=args.reduced_cumulants,
            sbi_type=args.sbi_type,
            linearised=args.linearised, 
            compression=args.compression,
            order_idx=args.order_idx,
            n_linear_sims=args.n_linear_sims,
            pre_train=args.pre_train,
            freeze_parameters=args.freeze_parameters
        )

        print("Using PDF dataset for bulk dataset.")

        dataset = BulkCumulantsDataset(config, pdfs=True, verbose=False)

        F_z = np.linalg.inv(dataset.data.Finv)
        F = F + F_z

    Finv = np.linalg.inv(F)

    return Finv


if __name__ == "__main__":
    from configs import bulk_cumulants_config
    from sbiax.utils import make_df, marker

    config = bulk_cumulants_config()

    config.use_bulk_means = True

    dataset = BulkCumulantsDataset(config, verbose=True)

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






    # if verbose:

    #     # fig, axs = plt.subplots(1, dataset.alpha.size, figsize=(2. + 2. * dataset.alpha.size, 2.5))
    #     # for p, ax in enumerate(axs):
    #     #     ax.scatter(latin_parameters[:, p], X_moments[:, p], s=0.1)
    #     #     ax.scatter(latin_parameters[:, p], X_pdfs[:, p], s=0.1)
    #     #     ax.axline((0, 0), slope=1., color="k", linestyle="--")
    #     #     ax.set_xlim(dataset.lower[p], dataset.upper[p])
    #     #     ax.set_ylim(dataset.lower[p], dataset.upper[p])
    #     #     ax.set_xlabel(dataset.parameter_strings[p])
    #     #     ax.set_ylabel(dataset.parameter_strings[p] + "'")
    #     # plt.savefig("/project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/scratch/summaries_pdf_moments.png")
    #     # plt.close()

    #     if 0:
    #         c = ChainConsumer()
    #         c.add_chain(
    #             Chain(
    #                 samples=make_df(X_moments, parameter_strings=parameter_strings), 
    #                 name="X[moments]", 
    #                 color="b", 
    #                 plot_contour=False, 
    #                 plot_cloud=True
    #             )
    #         )
    #         c.add_chain(
    #             Chain(
    #                 samples=make_df(X_pdfs, parameter_strings=parameter_strings), 
    #                 name="X[pdfs]", 
    #                 color="g", 
    #                 plot_contour=False, 
    #                 plot_cloud=True
    #             )
    #         )
    #         c.add_chain(
    #             Chain.from_covariance(
    #                 alpha,
    #                 Finv_pdf,
    #                 columns=parameter_strings,
    #                 name=r"$F_{\Sigma^{-1}}$ (PDFs)",
    #                 color="k",
    #                 linestyle=":",
    #                 shade_alpha=0.
    #             )
    #         )
    #         c.add_chain(
    #             Chain.from_covariance(
    #                 alpha,
    #                 Finv_moments,
    #                 columns=parameter_strings,
    #                 name=r"$F_{\Sigma^{-1}}$ (moments)",
    #                 color="r",
    #                 linestyle=":",
    #                 shade_alpha=0.
    #             )
    #         )
    #         # c.add_chain(
    #         #     Chain(
    #         #         samples=make_df(dataset.parameters, parameter_strings=dataset.parameter_strings), 
    #         #         plot_contour=False, 
    #         #         plot_cloud=True, 
    #         #         name="P", 
    #         #         color="r"
    #         #     )
    #         # )
    #         c.add_marker(
    #             location=marker(alpha, parameter_strings=parameter_strings),
    #             name=r"$\alpha$", 
    #             color="#7600bc"
    #         )
    #         fig = c.plotter.plot()
    #         # fig.suptitle(
    #         #     r"$k_n/k_2^{n-1}$ SBI & $F_{{\Sigma}}^{{-1}}$" + "\n" +
    #         #     "z={},\n $n_s$={}, (pre-train $n_s$={}),\n R={} Mpc,\n $k_n$={}".format(
    #         #             0., 
    #         #             len(X), 
    #         #             "[{}]".format(", ".join(map(str, R_values)))
    #         #         ),
    #         #     multialignment='center'
    #         # )
    #         plt.savefig(
    #             os.path.join(
    #                 results_dir if results_dir is not None else "fisher_forecasts/", 
    #                 "bulk_fisher_forecast_{}_z={}_R={}_m={}.png".format(
    #                     config.linearised, 
    #                     config.redshift, 
    #                     "".join(map(str, config.order_idx)),
    #                     "".join(map(str, config.scales))
    #                 )
    #             ), 
    #         )
    #         plt.close()


# def get_bulk_cumulants_data(
#     config: ConfigDict, 
#     *, 
#     pdfs: bool = False, # Use PDFs or cumulants for the bulk
#     verbose: bool = False, 
#     use_mean: bool = False,
#     use_bulk_norms: bool = True, # Stack means of bulk of the PDF at each scale with the other cumulants
#     stack_bulk_means: bool = True,
#     check_cumulants_against_quijote: bool = False,
#     results_dir: Optional[str] = None
# ) -> Dataset:
#     """
#         Get dataset for SBI experiments with the cumulants.
#         - Cut the PDFs according to a p_min, p_max cut into the CDF which 
#           indexes the bins of the PDF.
#         - Return PDFs of cumulants of the bulk
#     """

#     print("Using bulk means..." if use_mean else "Not using bulk means...")

#     data_dir, *_ = get_save_and_load_dirs()

#     (
#         all_R_values,
#         all_redshifts,
#         resolution,
#         alpha,
#         lower,
#         upper,
#         parameter_strings,
#         redshift_strings,
#         parameter_derivative_names,
#         dparams,
#         deltas,
#         delta_bin_edges,
#         D_deltas 
#     ) = get_quijote_parameters()

#     p_value_min        = config.p_value_min # Independent of choosing rho/delta for random variable of PDF
#     p_value_max        = config.p_value_max 

#     cumulants          = True             # Use cumulants over moments (NOTE: check not calculating reduced-cumulants, Quijote uses cumulants)
#     use_mean           = use_mean         # Concatenate mean of bulk to datavector
#     stack_mean         = stack_bulk_means # For full shape <delta> is very close to zero but <rho> approximately one
#     use_normalisations = use_bulk_norms   # Stack M_0 normalisation of pdf into datavector ahead of mean M_1 
#     normalise          = False            # Divide moments by M_0

#     n_scales           = len(config.scales)
#     n_redshifts        = 1
#     n_bins_pdf         = 99
#     n_fiducial_pdfs    = 15_000
#     n_latin_pdfs       = 2000
#     n_derivatives      = 500
#     n_p                = alpha.size 
#     R_idx              = [all_R_values.index(R) for R in config.scales]
#     z_idx              = all_redshifts.index(config.redshift)
#     n_cumulants        = 3 # [var, skew, kurt]

#     dataset_identifier_str = "".join(
#         [
#             "_R" + "".join(map(str, config.scales)),
#             "_m" + "".join(map(str, config.order_idx)),
#             "_z" + str(config.redshift),
#             "_f" if config.freeze_parameters else "_nf",
#             "_pdfs" if pdfs else "", 
#             "_with_means" if use_mean else "",
#             "_with_norms" if use_normalisations else "",
#             "_with_means_stacked" if stack_mean else ""
#         ]
#     )

#     def generate_dataset():
#         ...
#         return return_dataset

#     # Try loading dataset instead of deriving it again and again NOTE: careful not to load PDFs when yhou need cumulants etc
#     if not FORCE_RECOMPUTE_DATASET:
#         try:
#             dataset_filename = os.path.join(
#                 data_dir, "bulk_cumulants_dataset_{}.npz".format(dataset_identifier_str)
#             )
#             dataset_dict = np.load(dataset_filename, allow_pickle=True) # dataset_dict = jax.tree.map(lambda x: jnp.asarray(x), dataset_dict, is_leaf=lambda x: isinstance(x, np.ndarray))

#             return_dataset = Dataset(
#                 name="bulk_pdf" if pdfs else "bulk",
#                 alpha=jnp.asarray(dataset_dict["alpha"]),
#                 lower=jnp.asarray(dataset_dict["lower"]),
#                 upper=jnp.asarray(dataset_dict["upper"]),
#                 parameter_strings=list(dataset_dict["parameter_strings"]),
#                 Finv=jnp.asarray(dataset_dict["Finv"]),
#                 Cinv=jnp.asarray(dataset_dict["Cinv"]),
#                 C=jnp.asarray(dataset_dict["C"]),
#                 fiducial_data=jnp.asarray(dataset_dict["fiducial_data"]),
#                 data=jnp.asarray(dataset_dict["data"]),
#                 parameters=jnp.asarray(dataset_dict["parameters"]),
#                 derivatives=jnp.asarray(dataset_dict["derivatives"]),
#             )

#             print("Loaded dataset:\n\t", dataset_filename)
#         except FileNotFoundError:
#             return_dataset = generate_dataset()
#     else:
#         return_dataset = generate_dataset()

#     return return_dataset 
