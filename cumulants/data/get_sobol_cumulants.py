from typing import Literal, Optional, Callable
import os
from dataclasses import dataclass, replace, asdict
import jax
import jax.numpy as jnp
from tqdm.auto import trange
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from jaxtyping import Float, Int, Array, PRNGKeyArray, jaxtyped
from beartype import beartype as typechecker
from ml_collections import ConfigDict
import tensorflow_probability.substrates.jax.distributions as tfd

from configs.log import setup_module_logger, get_log_level
from data.constants import (
    get_scales, get_sobol_scale_numbers, 
    get_save_and_load_dirs, get_quijote_parameters,
    ALL_REDSHIFTS, QUIJOTE_DIR,
    DPARAMS, ALPHA,
    PARAMETER_STRINGS, PARAMETER_DERIVATIVE_STRINGS
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
    get_parameter_strings
)
from configs.cumulants_configs import bulk_cumulants_config

typecheck = jaxtyped(typechecker=typechecker)

logger, log_figs_dir = setup_module_logger(__name__, level=get_log_level())

FORCE_RECOMPUTE_DATASET = True if os.environ.get("FORCE_RECOMPUTE_DATASET", "").lower() in ("1", "true") else False 
FIDUCIAL_REDUCE = True if os.environ.get("FIDUCIAL_REDUCE", "").lower() in ("1", "true") else False
NON_GAUSSIAN_TEST = True if os.environ.get("NON_GAUSSIAN_TEST", "").lower() in ("1", "true") else False

"""
    Load Sobol PDFs for 
    - fiducial
    - latin
    - derivatives
    and calculate the cumulants
"""


def get_fiducials_fixed_lengths(redshift, cdf_cut_lims):

    data_dir = Path(QUIJOTE_DIR) 

    fiducials_dir = data_dir / "matterPDF_fiducial" / "fiducials"

    def get_cdf_of_pdf(pdf, dbins, cdf_cut_lims):
        n_bins_pdf = pdf.size

        p_value_min, p_value_max = cdf_cut_lims

        cdf = np.zeros_like(pdf)
        normalisation = 0.
        for i in range(1, n_bins_pdf):
            p_delta_i = pdf[i - 1]

            dp_i = p_delta_i * dbins[i - 1] 

            cdf[i] = cdf[i - 1] + dp_i

            normalisation += dp_i

        # Check CDF bounds
        assert np.isclose(cdf.min(), 0.) and np.isclose(cdf.max(), 1.), (
            "CDF: min={} max={}".format(cdf.min(), cdf.max())
        )

        # Check PDF is normalised
        assert np.isclose(normalisation, 1.0)

        cut_idx = np.where(
            (cdf >= p_value_min) & (cdf <= p_value_max)
        )[0] 

        return cdf, cut_idx

    def get_pdf_filename_template(redshift, radius_index, realisation):
        filename = 'matter_PDF_linbins_BSQ_z_{:.2f}_extraRind_{}_{}.dat'.format(
            redshift, radius_index, str(realisation)
        )
        return filename


    N_FIDUCIAL_REALISATIONS = len(
        [_ for _ in fiducials_dir.iterdir() if _.is_dir()]
    )

    REDSHIFTS = [redshift]

    R_numbers = get_sobol_scale_numbers()

    # Lengths of cuts for PDF at each redshift and scale
    sizes_all_scales_all_z = dict(
        zip(
            [str(z) for z in REDSHIFTS],
            [
                dict(
                    zip(
                        [str(_) for _ in R_numbers], 
                        [list() for _ in R_numbers]
                    )
                )
            for z in REDSHIFTS
            ]
        )
    )

    for realisation_idx, realisation_path_folder_number in zip(
        bar := trange(
            N_FIDUCIAL_REALISATIONS, desc="Fiducial PDFs", colour="blue"
        ),
        [_ for _ in fiducials_dir.iterdir() if _.is_dir()]
    ):

        realisation_folder_number = int(realisation_path_folder_number.name)

        for i_z, z in enumerate(REDSHIFTS):  

            for i_r, radius in enumerate(R_numbers):

                # Grab physical scale index
                idx = np.squeeze(np.argwhere(R_numbers == radius))

                # Filename of FIDUCIAL PDF
                realisation_path = fiducials_dir / str(realisation_folder_number) / get_pdf_filename_template(
                    redshift=z, radius_index=idx, realisation=realisation_folder_number
                ) 

                try:
                    # Load PDF bin centres and PDF in bins
                    bins, pdf = np.loadtxt(realisation_path).T

                    assert bins.size == pdf.size

                    dbins = bins[1:] - bins[:-1]

                    # Cut each pdf by its own CDF
                    cdf, cdf_cut_idx = get_cdf_of_pdf(pdf, dbins, cdf_cut_lims)

                    cut_pdf = pdf[cdf_cut_idx]

                    sizes_all_scales_all_z[str(z)][str(radius)].append(cut_pdf.size)
                except:
                    pass

    # Dictionary of matched scale-cut sizes
    z_R_lengths = jax.tree.map(
        lambda _R_lengths: min(_R_lengths), # Means not adding a zero
        sizes_all_scales_all_z, 
        is_leaf=lambda l: isinstance(l, list)
    )                    

    return z_R_lengths


def get_fiducials_latins_derivatives_bulk_pdf(redshift, cdf_cut_lims):

    def get_cdf_of_pdf(pdf, dbins, cdf_cut_lims):
        n_bins_pdf = pdf.size

        p_value_min, p_value_max = cdf_cut_lims

        cdf = np.zeros_like(pdf)
        normalisation = 0.
        for i in range(1, n_bins_pdf):
            p_delta_i = pdf[i - 1]

            dp_i = p_delta_i * dbins[i - 1] 

            cdf[i] = cdf[i - 1] + dp_i

            normalisation += dp_i

        # Check CDF bounds
        assert np.isclose(cdf.min(), 0.) and np.isclose(cdf.max(), 1.), (
            "CDF: min={} max={}".format(cdf.min(), cdf.max())
        )

        # Check PDF is normalised
        assert np.isclose(normalisation, 1.0)

        cut_idx = np.where(
            (cdf >= p_value_min) & (cdf <= p_value_max)
        )[0] 

        return cdf, cut_idx

    data_dir = Path(QUIJOTE_DIR) 

    derivatives_dir = data_dir / "matterPDF_fiducial" / "derivatives"
    fiducials_dir = data_dir / "matterPDF_fiducial" / "fiducials"

    z_R_lengths = get_fiducials_fixed_lengths(redshift, cdf_cut_lims)

    all_scales_for_z_dim = sum(_ for _ in z_R_lengths[str(redshift)].values())

    redshifts = [redshift] # Only chosen redshift

    R_numbers = get_sobol_scale_numbers()
    scales = get_scales()

    def get_pdf_filename_template(redshift, radius_index, realisation):
        filename = 'matter_PDF_linbins_BSQ_z_{:.2f}_extraRind_{}_{}.dat'.format(
            redshift, radius_index, str(realisation)
        )
        return filename

    N_FIDUCIAL_REALISATIONS = len(
        [_ for _ in fiducials_dir.iterdir() if _.is_dir()]
    )
    print(N_FIDUCIAL_REALISATIONS)

    """
        FIDUCIALS
    """

    fiducial_pdfs = dict(bulk=np.zeros((N_FIDUCIAL_REALISATIONS, all_scales_for_z_dim)))

    bad_realisations = []

    for realisation_idx, realisation_path_folder_number in zip(
        bar := trange(
            N_FIDUCIAL_REALISATIONS, 
            desc="Fiducial PDFs",
            colour="blue"
        ),
        [_ for _ in fiducials_dir.iterdir() if _.is_dir()]
    ):

        realisation_folder_number = int(realisation_path_folder_number.name)

        fiducial_pdfs_R_z = []
        for i_z, z in enumerate(redshifts):  # ONLY REDSHIFT zero
            
            # Container for all scales, to concatenate
            _pdfs = []     

            bad = False
            for i_r, radius in enumerate(R_numbers):

                # Grab physical scale index
                idx = np.squeeze(np.argwhere(R_numbers == radius))

                # Filename of FIDUCIAL PDF
                realisation_dir = fiducials_dir / str(realisation_folder_number)
                realisation_path = realisation_dir / get_pdf_filename_template(
                    redshift=z, radius_index=idx, realisation=realisation_folder_number
                ) 

                # Length of cut at this radius and redshift
                z_R_length = z_R_lengths[str(z)][str(radius)]

                _pdf_ = np.zeros((z_R_length,))

                try:
                    # Load PDF bin centres and PDF in bins
                    bins, pdf = np.loadtxt(realisation_path).T

                    assert bins.size == pdf.size

                    dbins = bins[1:] - bins[:-1]

                    # Cut each pdf by its own CDF
                    cdf, cdf_cut_idx = get_cdf_of_pdf(pdf, dbins, cdf_cut_lims)

                    cut_pdf = pdf[cdf_cut_idx]
                    cut_bins = bins[cdf_cut_idx]
                    cut_dbins = dbins[cdf_cut_idx] 

                    cut_pdf = cut_pdf[:z_R_length] # Trim

                    # Zero-padded array containing the PDF
                    _pdf_[:len(cut_pdf)] = cut_pdf

                    delta_min = np.minimum(np.min(cut_bins), delta_min)
                    delta_max = np.maximum(np.max(cut_bins), delta_max)

                    fiducial_pdfs_R_z.append(_pdf_)
                    _pdfs.append(_pdf_)

                except FileNotFoundError:
                    bad = True
                    bad_realisations.append(realisation_idx)
                    continue

            # If a good PDF, stack over scale for this redshift and realisation
            if not bad:
                fiducial_pdfs["bulk"][realisation_idx, i_z] = np.concatenate(_pdfs) 
                pass
    """
        DERIVATIVES
    """

    def get_derivative_filename_template(derivative_p_m_name, redshift, realisation, R_i):
        return derivatives_dir / str(realisation) / derivative_p_m_name / (
            'matter_PDF_linbins_BSQ_z_{:.2f}_extraRind_{}_{}.dat'.format(
                redshift, R_i, realisation
            )
        )

    N_DERIVATIVE_REALISATIONS = len(
        [_ for _ in derivatives_dir.iterdir() if _.is_dir()]
    )

    print(N_DERIVATIVE_REALISATIONS)

    derivatives_pdfs = dict(
        bulk=np.zeros((N_DERIVATIVE_REALISATIONS, len(redshifts), ALPHA.size, all_scales_for_z_dim))
    )

    bad_realisations = []

    for i_p in range(ALPHA.size):

        PARAMETERS_PM = PARAMETER_DERIVATIVE_STRINGS[i_p]

        # For each parameter, calculate cumulants for all realisations, plus and minus
        dummy_pdfs = np.zeros((N_DERIVATIVE_REALISATIONS, len(redshifts), all_scales_for_z_dim, 2))

        for realisation, realisation_path_folder_number in zip(
            trange(
                0, N_DERIVATIVE_REALISATIONS, desc=PARAMETER_STRINGS[i_p][1:-1],
                colour="blue"
            ),
            [_ for _ in derivatives_dir.iterdir() if _.is_dir()]
        ):

            realisation_folder_number = int(realisation_path_folder_number.name)

            for pm, parameter_p_or_m in enumerate(PARAMETERS_PM): # Plus then minus 

                for i_z, redshift in enumerate(redshifts):

                    # Container for all scales, to concatenate
                    _pdfs = []     

                    bad = False # if any of the PDFs are not loaded skip the realisation
                    for R_i, R in enumerate(scales):

                        z_R_length = z_R_lengths[str(redshift)][str(R_numbers[R_i])]

                        _pdf_ = np.zeros((z_R_length,))

                        try:
                            derivative_filename = get_derivative_filename_template(
                                parameter_p_or_m, redshift, realisation_folder_number, R_i
                            )

                            bins, pdf = np.loadtxt(derivative_filename).T

                            dbins = bins[1:] - bins[:-1]

                            # Cut each pdf by its own CDF
                            cdf, cdf_cut_idx = get_cdf_of_pdf(pdf, dbins, cdf_cut_lims)

                            cut_pdf = pdf[cdf_cut_idx]
                            cut_bins = bins[cdf_cut_idx]
                            cut_dbins = dbins[cdf_cut_idx] 

                            _pdfs.append(_pdf_)

                            cut_pdf = cut_pdf[:z_R_length] # Trim

                            _pdf_[:len(cut_pdf)] = cut_pdf

                        except FileNotFoundError:
                            bad = True
                            bad_realisations.append(realisation)
                            continue

                    # If a good PDF, stack over scale for this redshift and realisation
                    if not bad:
                        dummy_pdfs[realisation, i_z, :, pm] = np.concatenate(_pdfs) 
                        pass

        # Store the finite difference gradients for each parameter
        derivatives_pdfs["bulk"][:, :, i_p, :] = (dummy_pdfs[..., 0] - dummy_pdfs[..., 1]) / DPARAMS[i_p]

    bad_realisations = list(set(bad_realisations))
    for bad_idx in bad_realisations:
        derivatives_pdfs["bulk"] = np.delete(derivatives_pdfs[cut_name], bad_idx, axis=0)

    return fiducial_pdfs, derivatives_pdfs


def get_fiducials_latins_derivatives_cumulants(bulk_or_tails: Literal["bulk", "tails"], redshift, cdf_cut_lims):

    data_dir = Path(QUIJOTE_DIR) 

    latins_dir = data_dir / "matterPDF_BSQ" / "derivatives"
    derivatives_dir = data_dir / "matterPDF_fiducial" / "derivatives"
    fiducials_dir = data_dir / "matterPDF_fiducial" / "fiducials"

    params = np.loadtxt(latins_dir / "BSQ_params.txt") # Omega_m, Omega_b, h, n_s, sigma_8 

    print("PARAMETERS:", params.shape)

    available_idx_realisations = []
    for _ in data_dir.iterdir():
        try:
            number = int(str(_).split("/")[-1])
        except ValueError:
            continue
        available_idx_realisations.append(number)

    available_idx_realisations = np.asarray(available_idx_realisations)
    print("AVAILABLE REALISATIONS:", available_idx_realisations)

    available_idx_realisations = np.asarray(available_idx_realisations)


    def get_pdf_filename_template(redshift, radius_index, realisation):
        filename = 'matter_PDF_linbins_BSQ_z_{:.2f}_extraRind_{}_{}.dat'.format(
            redshift, radius_index, str(realisation)
        )
        return filename


    n_cumulants = 5 # m_0, m_1, k_2, k_3, k_4

    redshifts = [redshift] # Only chosen redshift

    scale_numbers = get_sobol_scale_numbers()
    scales = get_scales()

    def cut_pdf_to_cumulants(
        cut_pdf: Float[np.ndarray, "d"], # Divide by cut-norm
        deltas: Float[np.ndarray, "d"],
        ddeltas: Float[np.ndarray, "d"],
        prob_norm: float # Max prob - min_prob in CDF cut
    ) -> Float[np.ndarray, "5"]:
        # Bernardeau 2002 Eq. 130

        cut_pdf = cut_pdf / prob_norm # Was 'fiducial_based_normalisation'

        _delta_ = np.sum(ddeltas * cut_pdf * deltas)

        deltamod = deltas - _delta_

        m_0 = np.sum(cut_pdf * ddeltas)
        m_1 = np.sum(cut_pdf * ddeltas * deltas) 

        # Assuming <delta>=0? see Bernardeau eq (130)
        cumulant_2 = np.sum(deltamod ** 2 * cut_pdf * ddeltas)
        cumulant_3 = np.sum(deltamod ** 3 * cut_pdf * ddeltas)
        cumulant_4 = np.sum(deltamod ** 4 * cut_pdf * ddeltas) - (3. * np.sum(deltamod ** 2 * cut_pdf * ddeltas) ** 2)

        cumulants = np.asarray([m_0, m_1, cumulant_2, cumulant_3, cumulant_4]) 

        return cumulants


    def get_cdf_of_pdf(pdf, dbins, cdf_cut_lims):
        n_bins_pdf = pdf.size

        p_value_min, p_value_max = cdf_cut_lims

        cdf = np.zeros_like(pdf)
        normalisation = 0.
        for i in range(1, n_bins_pdf):
            p_delta_i = pdf[i - 1]

            dp_i = p_delta_i * dbins[i - 1] 

            cdf[i] = cdf[i - 1] + dp_i

            normalisation += dp_i

        # Check CDF bounds
        assert np.isclose(cdf.min(), 0.) and np.isclose(cdf.max(), 1.), (
            "CDF: min={} max={}".format(cdf.min(), cdf.max())
        )

        # Check PDF is normalised
        assert np.isclose(normalisation, 1.0)

        cut_idx = np.where(
            (cdf >= p_value_min) & (cdf <= p_value_max)
        )[0] 

        return cdf, cut_idx


    """
        Fiducials
    """

    n_fiducial_pdfs = len(list(fiducials_dir.iterdir()))

    fiducial_cumulants = np.zeros((n_fiducial_pdfs, len(redshifts), len(scales), n_cumulants))
    fiducial_vars = np.zeros((n_fiducial_pdfs, len(redshifts), len(scales), 1))

    bad_realisations = []

    for realisation_idx, realisation in zip(
        bar := trange(
            n_fiducial_pdfs, 
            desc="Fiducial PDFs",
            colour="red" if bulk_or_tails == "tails" else "blue"
        ),
        np.arange(n_fiducial_pdfs) # Assume 0-N are all available
    ):

        for i_z, z in enumerate(redshifts): 

            for i_r, radius in enumerate(scale_numbers):

                # Grab physical scale index
                idx = np.squeeze(np.argwhere(scale_numbers == radius))

                # Filename of FIDUCIAL PDF
                realisation_path = fiducials_dir / str(realisation) / get_pdf_filename_template(
                    redshift=z, radius_index=idx, realisation=realisation
                ) 

                try:
                    # Load PDF bin centres and PDF in bins
                    bins, pdf = np.loadtxt(realisation_path).T

                    # bins = bins - 1. # Rho -> delta

                    assert bins.size == pdf.size

                    dbins = bins[1:] - bins[:-1]

                    # Cut each pdf by its own CDF
                    cdf, cdf_cut_idx = get_cdf_of_pdf(pdf, dbins, cdf_cut_lims)

                    cut_pdf = pdf[cdf_cut_idx]
                    cut_bins = bins[cdf_cut_idx]
                    cut_dbins = dbins[cdf_cut_idx] 

                    cumulants_R_z = cut_pdf_to_cumulants(
                        cut_pdf, 
                        cut_bins, 
                        cut_dbins, 
                        prob_norm=cdf_cut_lims[1] - cdf_cut_lims[0]
                    )

                    fiducial_cumulants[realisation_idx, i_z, i_r] = cumulants_R_z

                    var_R_z = cumulants_R_z[2]
                    fiducial_vars[realisation_idx, i_z, i_r] = var_R_z

                except FileNotFoundError as e:
                    print(e)
                    bad_realisations.append(realisation_idx)

        bar.set_description("cut={}".format(bulk_or_tails))

    fiducial_vars = np.mean(fiducial_vars, axis=0, keepdims=True)

    if FIDUCIAL_REDUCE:
        fiducial_cumulants = fiducial_cumulants / fiducial_vars

    """
        Latins
    """

    n_available_realisations = len(available_idx_realisations)

    cumulants = np.zeros((n_available_realisations, len(redshifts), len(scales), n_cumulants))
    parameters = np.zeros((n_available_realisations, ALPHA.size))

    bad_realisations = []
    for realisation_idx, realisation in zip(
        bar := trange(
            n_available_realisations, 
            colour="red" if bulk_or_tails == "tails" else "blue"
        ),
        available_idx_realisations
    ):

        for i_z, z in enumerate(redshifts): 

            for i_r, radius in enumerate(scale_numbers):

                # Grab physical scale index
                idx = np.squeeze(np.argwhere(scale_numbers == radius))

                # Filename of Sobol sequence PDF
                realisation_path = latins_dir / str(realisation) / get_pdf_filename_template(
                    redshift=z, radius_index=idx, realisation=realisation
                ) 

                try:
                    # Load PDF bin centres and PDF in bins
                    bins, pdf = np.loadtxt(realisation_path).T

                    assert bins.size == pdf.size

                    dbins = bins[1:] - bins[:-1]

                    # Cut each pdf by its own CDF
                    cdf, cdf_cut_idx = get_cdf_of_pdf(pdf, dbins, cdf_cut_lims)

                    print(realisation_idx, "PDF sum / size: {} / {}".format(pdf.sum(), pdf.size))
                    print(realisation_idx, "CDF: {:.1f} {:.1f}".format(cdf.min(), cdf.max()))
                    print(realisation_idx, "CDF cut size: {}, dbins size: {}".format(cdf_cut_idx.size, dbins.size))

                    cut_pdf = pdf[cdf_cut_idx]
                    cut_bins = bins[cdf_cut_idx]
                    cut_dbins = dbins[cdf_cut_idx] 

                    print(realisation_idx, "{}".format(
                        jax.tree.map(lambda a: a.shape, (cut_pdf, cut_bins, cut_dbins)))
                    )

                    cumulants_R_z = cut_pdf_to_cumulants(
                        cut_pdf, 
                        cut_bins, 
                        cut_dbins, 
                        prob_norm=cdf_cut_lims[1] - cdf_cut_lims[0]
                    )
                    print(realisation_idx, "cumulants: {}".format(cumulants_R_z))

                    cumulants[realisation_idx, i_z, i_r] = cumulants_R_z

                    parameters[realisation_idx] = params[realisation_idx]

                except FileNotFoundError as e:
                    print(e)
                    bad_realisations.append(realisation_idx)

        bar.set_description("cut={}".format(bulk_or_tails))

    if FIDUCIAL_REDUCE:
        cumulants = cumulants / fiducial_vars

    """
        Derivatives
    """

    def get_derivative_filename_template(derivative_p_m_name, redshift, realisation, R_i):
        return derivatives_dir / str(realisation) / derivative_p_m_name / (
            'matter_PDF_linbins_BSQ_z_{:.2f}_extraRind_{}_{}.dat'.format(
                redshift, R_i, realisation
            )
        )

    N_DERIVATIVE_REALISATIONS = len(
        [_ for _ in derivatives_dir.iterdir() if _.is_dir()]
    )

    derivatives = np.zeros((N_DERIVATIVE_REALISATIONS, len(redshifts), ALPHA.size, len(scales), n_cumulants))
    for i_p in range(ALPHA.size):

        PARAMETERS_PM = PARAMETER_DERIVATIVE_STRINGS[i_p]

        # For each parameter, calculate cumulants for all realisations
        dummy = np.zeros((N_DERIVATIVE_REALISATIONS, len(redshifts), len(scales), n_cumulants, 2))

        for realisation, realisation_path_folder_number in zip(
            trange(
                0, N_DERIVATIVE_REALISATIONS, 
                desc=PARAMETER_STRINGS[i_p][1:-1]
            ),
            [_ for _ in derivatives_dir.iterdir() if _.is_dir()]
        ):

            # print(realisation_path_folder_number.name)
            realisation_folder_number = int(realisation_path_folder_number.name)

            for pm, parameter_p_or_m in enumerate(PARAMETERS_PM): # Plus then minus 
                
                for i_z, redshift in enumerate(redshifts):

                    for R_i, R in enumerate(scales):

                        derivative_filename = get_derivative_filename_template(
                            parameter_p_or_m, redshift, realisation_folder_number, R_i
                        )

                        bins, pdf = np.loadtxt(derivative_filename).T

                        dbins = bins[1:] - bins[:-1]

                        # Cut each pdf by its own CDF
                        cdf, cdf_cut_idx = get_cdf_of_pdf(pdf, dbins, cdf_cut_lims)

                        cut_pdf = pdf[cdf_cut_idx]
                        cut_bins = bins[cdf_cut_idx]
                        cut_dbins = dbins[cdf_cut_idx] 

                        cumulants_R_z = cut_pdf_to_cumulants(
                            cut_pdf, 
                            cut_bins, 
                            cut_dbins, 
                            prob_norm=cdf_cut_lims[1] - cdf_cut_lims[0]
                        )
                        assert not np.all(cumulants_R_z == 0)

                        dummy[realisation, i_z, R_i, :, pm] = cumulants_R_z

        # Store the finite difference gradients for each parameter
        derivatives[:, :, i_p, :, :] = (dummy[..., 0] - dummy[..., 1]) / DPARAMS[i_p]

    if FIDUCIAL_REDUCE:
        derivatives = derivatives / fiducial_vars[:, :, jnp.newaxis, ...]

    return fiducial_cumulants, cumulants, derivatives, parameters


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
        _,
        _,
        _,
        alpha,
        lower,
        upper,
        parameter_strings,
        _,
        _,
        _,
        _,
        _, 
        _ 
    ) = get_quijote_parameters() # These are set to SOBOLs

    p_value_min                  = config.p_value_min # Independent of choosing rho/delta for random variable of PDF
    p_value_max                  = config.p_value_max 

    use_means                    = use_means               # Use <delta> in calculation of cumulants from moments 
    stack_means                  = stack_means             # Stack bulk mean do bulk datavector For full shape <delta> is very close to zero but <rho> approximately one
    use_normalisations           = use_normalisations      # Stack M_0 normalisation of pdf into datavector ahead of mean M_1 
    central_moments              = True                    # Calculate central moments or not (NOTE: 4th cumulant not the same as 4th central moment, but Bernardeau formulae use non-central moments)

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
    pdf_dataset_filename = os.path.join(
        data_dir, "datasets/bulk_pdf_dataset{}.npz".format(dataset_identifier_str)
    )

    def generate_dataset() -> list[np.ndarray]:

        tqdm_desc_str = config.dataset_name
        if pdfs:
            tqdm_desc_str += " pdfs"
        if full_shape:
            tqdm_desc_str += " full-shape"

        (
            fiducial_moments_z_R, 
            latin_moments_z_R, 
            derivative_moments_z_R,
            latin_parameters
        ) = get_fiducials_latins_derivatives_cumulants(
            bulk_or_tails, 
            redshift=config.redshift,
            cdf_cut_lims=(p_value_max, p_value_min)
        )

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
            try:
                logger.info("Loading PDF dataset:\n\t{}".format(pdf_dataset_filename))

                dataset_dict = np.load(dataset_filename, allow_pickle=True) 

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
                    data=jnp.asarray(dataset_dict["data"]), # This will be zeros
                    parameters=jnp.asarray(dataset_dict["parameters"]),
                    derivatives=jnp.asarray(dataset_dict["derivatives"]),
                )
            except FileNotFoundError:
                # Bulk PDF dataset (no latin hypercube data, but this is not needed)
                (
                    fiducial_pdfs_z_R, 
                    derivative_pdfs_z_R
                ) = get_fiducials_latins_derivatives_bulk_pdf(
                    redshift=config.redshift,
                    cdf_cut_lims=(p_value_max, p_value_min)
                )

                # Fisher information in bulk of the PDF
                _, data_dim_pdfs = fiducial_pdfs_z_R.shape 
                C_pdf = np.cov(fiducial_pdfs_z_R, rowvar=False) 
                H = hartlap(n_s=fiducial_pdfs_z_R.shape[0], n_d=data_dim_pdfs) 
                Cinv_pdf = H * np.linalg.inv(C_pdf)
                dmu_pdfs = np.mean(derivative_pdfs_z_R, axis=0)
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
                    fiducial_data=jnp.asarray(fiducial_pdfs_z_R),
                    data=jnp.zeros((2000, data_dim_pdfs)), # NOTE: Dummy array
                    parameters=jnp.asarray(latin_parameters),
                    derivatives=jnp.asarray(derivative_pdfs_z_R)  
                )
                
                corr_pdf = np.corrcoef(fiducial_pdfs_z_R, rowvar=False) 

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

                # Save PDF dataset to ensure loading (not creating) next time around
                np.savez(pdf_dataset_filename, **asdict(return_dataset))

        # Save _return_ dataset to ensure loading (not creating) next time around
        np.savez(dataset_filename, **asdict(return_dataset))

        logger.info("Saved dataset:\n\t{}".format(dataset_filename))

        # Return PDF or cumulants dataset
        return return_dataset 

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

    if config.linearised:
        logger.info("Using linearised dataset [replacing only hypercube]...")

        D, Y = get_linearised_data(config, return_dataset) 

        return_dataset = replace(return_dataset, data=D, parameters=Y)

    if NON_GAUSSIAN_TEST:
        logger.info("Using non-Gaussian linear model dataset [replacing only hypercube]...")

        D, Y = get_linearised_data(config, return_dataset) 

        return_dataset = replace(return_dataset, data=D, parameters=Y)

    return return_dataset 



"""
    Dataset
"""


@dataclass
class SobolBulkCumulantsDataset:
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
class SobolTailsCumulantsDataset:
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


class SobolBulkPDFsDataset(SobolBulkCumulantsDataset):
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

    dataset = SobolBulkCumulantsDataset(config, pdfs=pdfs)

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

        dataset = SobolBulkCumulantsDataset(config, pdfs=True)

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
    # Finv_bulk_pdfs_all_z = add_planck_information_to_Finv(
    #     Finv_bulk_pdfs_all_z, use_planck=args.use_planck
    # )

    logger.info("Finv bulk PDFs all z loaded from:\n\t{}".format(Finv_file_path))

    return Finv_bulk_pdfs_all_z