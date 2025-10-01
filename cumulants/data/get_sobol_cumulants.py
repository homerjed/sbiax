from typing import Literal, Optional, Callable, Union
import os
from dataclasses import dataclass, replace, asdict
import jax
import jax.numpy as jnp
import jax.random as jr
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
    get_scales, 
    get_sobol_scale_numbers, 
    get_save_and_load_dirs, 
    get_quijote_parameters,
    ALL_REDSHIFTS, 
    QUIJOTE_DIR,
    DPARAMS, 
    ALPHA,
    PARAMETER_STRINGS, 
    PARAMETER_DERIVATIVE_STRINGS
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
    and calculate the cumulants as well as the bulk PDF measurements
"""

SOBOL_DATA_DIR  = Path(QUIJOTE_DIR) # Set to Sobol when using what's in this file

LATINS_DIR      = SOBOL_DATA_DIR / "matterPDF_BSQ"
DERIVATIVES_DIR = SOBOL_DATA_DIR / "matterPDF_fiducial" / "derivatives"
FIDUCIALS_DIR   = SOBOL_DATA_DIR / "matterPDF_fiducial" / "fiducial"

fiducial_sorted_folders = sorted(
    [p for p in FIDUCIALS_DIR.iterdir() if p.is_dir() and p.name.isdigit()],
    key=lambda p: int(p.name)
)

derivative_sorted_folders = sorted(
    [p for p in DERIVATIVES_DIR.iterdir() if p.is_dir() and p.name.isdigit()],
    key=lambda p: int(p.name)
)

latin_sorted_folders = sorted(
    [p for p in LATINS_DIR.iterdir() if p.is_dir() and p.name.isdigit()],
    key=lambda p: int(p.name)
)

N_FIDUCIAL_REALISATIONS = len(fiducial_sorted_folders)
N_LATIN_REALISATIONS = len(latin_sorted_folders)
N_DERIVATIVE_REALISATIONS = len(derivative_sorted_folders)

ALL_LATIN_PARAMETERS = np.loadtxt(LATINS_DIR / "BSQ_params.txt")

N_CUMULANTS = 5 # m_0, m_1, k_2, k_3, k_4

ALL_R_NUMBERS = get_sobol_scale_numbers()

CUTS = dict(bulk=(0.03, 0.90), tails=(0.0, 0.999999)) # tails=(0.01, 0.999))

if 1:
    bins = range(0, 33000, 1000)

    ids = [int(_.name) for _ in latin_sorted_folders]

    plt.figure(figsize=(8., 3.))
    plt.title("Sobol hypercube")
    plt.hist(ids, bins=bins)
    plt.xticks(bins[::2], rotation=90)
    plt.savefig("SOBOL_REALISATIONS.png")
    plt.close()

    try:
        print(
            "latins",
            len([_ for _ in range(32768) if _ not in ids]), 
            min([_ for _ in range(32768) if _ not in ids]), 
            max([_ for _ in range(32768) if _ not in ids])
        )
    except ValueError as e:
        print(e)

    bins = range(0, 16_000, 1000)

    ids = [int(_.name) for _ in fiducial_sorted_folders]

    plt.figure(figsize=(8., 3.))
    plt.title("Fiducials (Sobol-like)")
    plt.hist(ids, bins=bins)
    plt.xticks(bins, rotation=90)
    plt.savefig("SOBOL_REALISATIONS_FIDUCIAL.png")
    plt.close()

    try:
        print(
            "fiducials:",
            len([_ for _ in range(15000) if _ not in ids]), 
            min([_ for _ in range(15000) if _ not in ids]), 
            max([_ for _ in range(15000) if _ not in ids])
        )
    except ValueError as e:
        print(e)

PDF_FILENAME_TEMPLATE = 'matter_PDF_linbins_BSQ_z_{:.2f}_extraRind_{}_{}.dat'


def get_fiducial_pdf_filename_template(redshift, radius_index, realisation):
    assert ALL_R_NUMBERS[radius_index] in ALL_R_NUMBERS
    filename = PDF_FILENAME_TEMPLATE.format(redshift, radius_index, str(realisation))
    return FIDUCIALS_DIR / str(realisation) / filename


def get_latin_pdf_filename_template(redshift, radius_index, realisation):
    assert ALL_R_NUMBERS[radius_index] in ALL_R_NUMBERS
    filename = PDF_FILENAME_TEMPLATE.format(redshift, radius_index, str(realisation))
    return LATINS_DIR / str(realisation) / filename


def get_pdf_derivative_filename_template(redshift, radius_index, realisation, derivative_p_m_name):
    assert ALL_R_NUMBERS[radius_index] in ALL_R_NUMBERS
    filename = PDF_FILENAME_TEMPLATE.format(redshift, radius_index, str(realisation))
    return DERIVATIVES_DIR / str(realisation) / derivative_p_m_name / filename


# @typecheck
def get_cdf_of_pdf(
    pdf: Float[np.ndarray, "d"], 
    dbins: Float[np.ndarray, "d"], 
    cdf_cut_lims: tuple[float, float]
) -> tuple[Float[np.ndarray, "d"], Int[np.ndarray, "c"]]:

    n_bins_pdf = pdf.size

    p_value_min, p_value_max = cdf_cut_lims

    assert p_value_max > p_value_min, (
        "min/max={}/{}".format(p_value_min, p_value_max)
    )

    cdf = np.zeros_like(pdf)
    normalisation = 0.
    for i in range(1, n_bins_pdf):
        p_delta_i = pdf[i - 1]

        dp_i = p_delta_i * dbins[i - 1] 

        cdf[i] = cdf[i - 1] + dp_i

        normalisation += dp_i

    # Check CDF bounds
    # assert np.isclose(cdf.min(), 0.) and np.isclose(cdf.max(), 1.), (
    #     "CDF: min={} max={}".format(cdf.min(), cdf.max())
    # )

    # # Check PDF is normalised
    # assert np.isclose(normalisation, 1.0)

    cut_idx = np.where((cdf >= p_value_min) & (cdf <= p_value_max))[0] 

    return cdf, cut_idx


@typecheck
def cut_pdf_to_cumulants(
    cut_pdf: Float[np.ndarray, "d"], # Divide by cut-norm
    deltas: Float[np.ndarray, "d"],
    ddeltas: Float[np.ndarray, "d"],
    prob_norm: float # Max prob - min_prob in CDF cut
) -> Float[np.ndarray, "5"]:
    # Bernardeau 2002 Eq. 130

    # ddeltas = drhos ...

    cut_pdf = cut_pdf / prob_norm 

    m_0 = np.sum(cut_pdf * ddeltas)
    m_1 = np.sum(cut_pdf * ddeltas * deltas) 

    deltamod = deltas - m_1

    # Assuming <delta>=0? see Bernardeau eq (130)
    k_2 = np.sum(deltamod ** 2. * cut_pdf * ddeltas)
    k_3 = np.sum(deltamod ** 3. * cut_pdf * ddeltas)
    k_4 = np.sum(deltamod ** 4. * cut_pdf * ddeltas) - (3. * k_2 ** 2.)

    k_n = np.asarray([m_0, m_1, k_2, k_3, k_4]) 

    return k_n 


def get_fiducial_pdfs_lengths(
    redshift: float, 
    R_numbers: Float[np.ndarray, "R"]
) -> tuple[
    Float[np.ndarray, "..."],
    Float[np.ndarray, "..."],
    int
]:
    # Store all fiducial PDFs in a single array, padded to the maximum length
    # of the fiducial PDFs 

    pdfs = np.zeros((5229, len(R_numbers), N_FIDUCIAL_REALISATIONS))

    max_length = 0

    for realisation_idx, realisation_path_folder_number in zip(
        trange(
            N_FIDUCIAL_REALISATIONS, 
            desc="Fiducial PDFs",
            colour="green"
        ),
        fiducial_sorted_folders
    ):

        realisation_folder_number = int(realisation_path_folder_number.name)

        for i_r, R in enumerate(R_numbers):

            # Filename of FIDUCIAL PDF
            realisation_path = get_fiducial_pdf_filename_template(
                redshift=redshift, 
                radius_index=np.squeeze(np.argwhere(ALL_R_NUMBERS == R)),
                realisation=realisation_folder_number
            ) 

            try:
                # Load PDF bin centres and PDF in bins
                _bins, pdf = np.loadtxt(realisation_path).T

                # If loaded PDF is shorter than current array_len → safe insert,
                # else → resize pdfs to accommodate
                if pdf.size < len(pdfs):
                    pdfs[:pdf.size, i_r, realisation_idx] = pdf
                else:
                    pdfs = np.resize(pdfs, (pdf.size,) + pdfs.shape[1:])

                    pdfs[:, i_r, realisation_idx] = pdf

                    print("realisation", realisation_idx, "extended length to", pdf.size)

                # Track maximum length and bins
                if pdf.size > max_length:
                    max_length = pdf.size
                    bins = _bins

                # Plot
                # if realisation_idx < 2:
                #     plot_pdf(bins, None, pdf, cdf_cut_idx, cut_name=cut_name)

            except FileNotFoundError as e:
                print(e)

    bins = bins - 1. # NOTE: rho -> delta
    print("Rho -> delta in `get_fiducial_pdf_lengths`")

    return pdfs, bins, max_length


def get_pdf_dim_bulk_or_tails(cuts_cdf_cut_idx_z_R: dict[str, int]) -> int:
    # Get lengths of PDFs summed over scale for a fixed redshift

    lengths = jax.tree.map(lambda a: a.size, cuts_cdf_cut_idx_z_R)

    dimension = sum(jax.tree.leaves(lengths))

    return dimension


@typecheck
def get_fiducials_latins_derivatives_bulk_pdf(
    redshift: float, 
    cdf_cut_lims: tuple[float, float],
    *,
    get_latins: bool = False
) -> Union[
    tuple[
        Float[np.ndarray, "nf d"],
        Float[np.ndarray, "nd p d"]
    ],
    tuple[
        Float[np.ndarray, "nf d"],
        Float[np.ndarray, "nl d"],
        Float[np.ndarray, "nd p d"]
    ]
]:
    assert cdf_cut_lims[1] > cdf_cut_lims[0]

    scale_numbers = get_sobol_scale_numbers()

    pdfs, bins, max_length = get_fiducial_pdfs_lengths(redshift, scale_numbers)

    mean_pdfs = np.mean(pdfs, axis=-1)[:max_length]

    dbins = bins[1:] - bins[:-1]

    """
        FIDUCIALS
    """

    def get_cut_indices(cut_name):
        # Get indices into mean of fiducial PDF 

        # Get CDF cut indices for each scale and redshift 
        cuts_cdf_z_R = dict()
        cuts_cdf_cut_idx_z_R = dict() # NOTE: how does tails USE ALL BINS?

        for i_R, scale_index in enumerate(scale_numbers):

            _cdf_z_R, _cdf_cut_idx_z_R = get_cdf_of_pdf(
                mean_pdfs[:max_length, i_R], 
                dbins=dbins, 
                cdf_cut_lims=CUTS[cut_name]
            )

            cuts_cdf_z_R[str(scale_index)] = _cdf_z_R
            cuts_cdf_cut_idx_z_R[str(scale_index)] = _cdf_cut_idx_z_R

        return cuts_cdf_cut_idx_z_R, cuts_cdf_z_R


    cuts_cdf_cut_idx_z_R, cuts_cdf_z_R = get_cut_indices(cut_name="bulk")


    def get_pdf_fiducials(redshift, cut_name):

        all_scales_for_z_dim_ = get_pdf_dim_bulk_or_tails(cuts_cdf_cut_idx_z_R)

        fiducial_pdfs = np.zeros((N_FIDUCIAL_REALISATIONS, all_scales_for_z_dim_))

        bad_realisations = []

        for realisation_idx, realisation_path_folder_number in zip(
            trange(
                0, N_FIDUCIAL_REALISATIONS, 
                desc="Fiducial PDFs [{}]".format(cut_name),
                colour="blue" if cut_name == "bulk" else "red"
            ),
            fiducial_sorted_folders
        ):

            realisation_folder_number = int(realisation_path_folder_number.name)

            # Container for all scales, to concatenate
            _pdfs = []     

            bad = False # if any of the PDFs are not loaded skip the realisation
            for radius_index, R in enumerate(scale_numbers):

                try:
                    pdf_filename = get_fiducial_pdf_filename_template(
                        redshift=redshift, 
                        realisation=realisation_folder_number, 
                        radius_index=np.squeeze(np.argwhere(ALL_R_NUMBERS == R))
                    )

                    _, pdf = np.loadtxt(pdf_filename).T

                    cdf_cut_idx = cuts_cdf_cut_idx_z_R[str(R)]

                    cut_pdf = pdf[cdf_cut_idx]

                    _pdfs.append(cut_pdf)

                except FileNotFoundError as e:
                    print(e)
                    bad = True
                    bad_realisations.append(realisation_idx)
                    continue

            # If a good PDF, stack over scale for this redshift and realisation_idx
            if not bad:
                # assert np.concatenate(_pdfs).shape[0] == dummy_pdfs.shape[2]
                fiducial_pdfs[realisation_idx, :] = np.concatenate(_pdfs) 

        # Delete bad realisations in returned array (zero rows)
        bad_realisations = list(set(bad_realisations))
        for bad_idx in sorted(bad_realisations, reverse=True):
            fiducial_pdfs = np.delete(fiducial_pdfs, bad_idx, axis=0)

        print("bad:", bad_realisations)

        return fiducial_pdfs


    fiducial_pdfs = get_pdf_fiducials(redshift, cut_name="bulk")

    """
        LATINS
    """


    def get_pdf_latins(redshift, cut_name):

        all_scales_for_z_dim_ = get_pdf_dim_bulk_or_tails(cuts_cdf_cut_idx_z_R)

        latin_pdfs = np.zeros((N_LATIN_REALISATIONS, all_scales_for_z_dim_))

        bad_realisations = []

        for realisation_idx, realisation_path_folder_number in zip(
            trange(
                0, N_LATIN_REALISATIONS, 
                desc="Latin PDFs [{}]".format(cut_name),
                colour="blue" if cut_name == "bulk" else "red"
            ),
            latin_sorted_folders
        ):

            realisation_folder_number = int(realisation_path_folder_number.name)

            # Container for all scales, to concatenate
            _pdfs = []     

            bad = False # if any of the PDFs are not loaded skip the realisation
            for radius_index, R in enumerate(scale_numbers):

                try:
                    pdf_filename = get_latin_pdf_filename_template(
                        redshift=redshift, 
                        realisation=realisation_folder_number, 
                        radius_index=np.squeeze(np.argwhere(ALL_R_NUMBERS == R))
                    )

                    _, pdf = np.loadtxt(pdf_filename).T

                    cdf_cut_idx = cuts_cdf_cut_idx_z_R[str(R)]

                    cut_pdf = pdf[cdf_cut_idx]

                    _pdfs.append(cut_pdf)

                except FileNotFoundError as e:
                    print(e)
                    bad = True
                    bad_realisations.append(realisation_idx)
                    continue

            # If a good PDF, stack over scale for this redshift and realisation_idx
            if not bad:
                # assert np.concatenate(_pdfs).shape[0] == dummy_pdfs.shape[2]
                latin_pdfs[realisation_idx, :] = np.concatenate(_pdfs) 

        # Delete bad realisations in returned array (zero rows)
        bad_realisations = list(set(bad_realisations))
        for bad_idx in sorted(bad_realisations, reverse=True):
            latin_pdfs = np.delete(latin_pdfs, bad_idx, axis=0)

        print("bad:", bad_realisations)

        return latin_pdfs


    if get_latins:
        latin_pdfs = get_pdf_latins(redshift, cut_name="bulk")

    """
        DERIVATIVES
    """

    def get_pdf_derivatives(redshift, cut_name=None):

        all_scales_for_z_dim_ = get_pdf_dim_bulk_or_tails(cuts_cdf_cut_idx_z_R)

        derivatives_pdf = np.zeros((N_DERIVATIVE_REALISATIONS, ALPHA.size, all_scales_for_z_dim_))

        bad_realisations = []

        for i_p in range(ALPHA.size):

            PARAMETERS_PM = PARAMETER_DERIVATIVE_STRINGS[i_p]

            # For each parameter, calculate cumulants for all realisations, plus and minus
            dummy_pdfs = np.zeros((N_DERIVATIVE_REALISATIONS, all_scales_for_z_dim_, 2))

            for realisation_idx, realisation_path_folder_number in zip(
                trange(
                    0, N_DERIVATIVE_REALISATIONS, 
                    desc="Derivatives PDFs [bulk] " + PARAMETER_STRINGS[i_p][1:-1],
                    colour="green"
                ),
                derivative_sorted_folders
            ):

                realisation_folder_number = int(realisation_path_folder_number.name)

                for pm, parameter_p_or_m in enumerate(PARAMETERS_PM): # Plus then minus 

                    # Container for all scales, to concatenate
                    _pdfs = []     

                    bad = False # if any of the PDFs are not loaded skip the realisation
                    for radius_index, R in enumerate(scale_numbers):

                        try:
                            derivative_filename = get_pdf_derivative_filename_template(
                                derivative_p_m_name=parameter_p_or_m, 
                                redshift=redshift, 
                                realisation=realisation_folder_number, 
                                radius_index=np.squeeze(np.argwhere(ALL_R_NUMBERS == R))
                            )

                            _, pdf = np.loadtxt(derivative_filename).T

                            # Cut each pdf by its own CDF
                            cdf_cut_idx = cuts_cdf_cut_idx_z_R[str(R)]

                            cut_pdf = pdf[cdf_cut_idx]

                            _pdfs.append(cut_pdf)

                        except FileNotFoundError as e:
                            # print(e)
                            bad = True
                            bad_realisations.append((realisation_idx, realisation_folder_number))
                            continue

                    # If a good PDF, stack over scale for this redshift and realisation_idx
                    if not bad:
                        # assert np.concatenate(_pdfs).shape[0] == dummy_pdfs.shape[2]
                        dummy_pdfs[realisation_idx, :, pm] = np.concatenate(_pdfs) 
                        pass

            # Store the finite difference gradients for each parameter
            derivatives_pdf[:, i_p, :] = (dummy_pdfs[..., 0] - dummy_pdfs[..., 1]) / DPARAMS[i_p]

        # Delete bad realisations in returned array (zero rows)
        bad_realisations = list(set(bad_realisations))
        for bad_idx, _ in sorted(bad_realisations, reverse=True):
            derivatives_pdf = np.delete(derivatives_pdf, bad_idx, axis=0)

        print("bad:", bad_realisations)

        return derivatives_pdf


    derivatives_pdf = get_pdf_derivatives(redshift, cut_name="bulk")

    # Return Fisher ingredients for bulk PDF and remove redshift axis,
    # optionally return hypercube PDFs 
    if get_latins:
        return fiducial_pdfs, latin_pdfs, derivatives_pdf 
    else:
        return fiducial_pdfs, derivatives_pdf


@typecheck
def get_fiducials_latins_derivatives_cumulants(
    bulk_or_tails: Literal["bulk", "tails"], 
    redshift: float,
    cdf_cut_lims: tuple[float, float]
) -> tuple[
    Float[np.ndarray, "nf d"],
    Float[np.ndarray, "nl d"],
    Float[np.ndarray, "nd p d"],
    Float[np.ndarray, "nl p"]
]:
    # return fiducial_cumulants, cumulants, derivatives, parameters

    scale_numbers = get_sobol_scale_numbers()

    pdfs, bins, max_length = get_fiducial_pdfs_lengths(redshift, scale_numbers)

    mean_pdfs = np.mean(pdfs, axis=-1)[:max_length] 

    dbins = bins[1:] - bins[:-1]

    """
        Fiducials
    """


    @typecheck
    def reduce(
        k_n: Float[np.ndarray, "n r k"], 
        vars: Float[np.ndarray, "1 r 1"]
    ) -> Float[np.ndarray, "n r k"]:
        # Reduce skewness and kurtosis of each scale
        for r in range(k_n.shape[-2]):
            for n in range(k_n.shape[-1]):
                if n < 3:
                    continue
                else:
                    k_n[:, r, n] = k_n[:, r, n] / (vars[:, r, :] ** (n - 1))
        return k_n


    @typecheck
    def get_cut_indices(
        cut_name: Literal["bulk", "tails"]
    ) -> tuple[
        dict[str, Int[np.ndarray, "..."]], 
        dict[str, Float[np.ndarray, "..."]]
    ]:
        # Get bins of fiducial PDF mean within cut on CDF for all scales and a single redshift

        # Get CDF cut indices for each scale and redshift 
        cuts_cdf_z_R = dict()
        cuts_cdf_cut_idx_z_R = dict() 
        for i_R, scale_index in enumerate(scale_numbers):

            _cdf_z_R, _cdf_cut_idx_z_R = get_cdf_of_pdf(
                mean_pdfs[:max_length, i_R], 
                dbins, 
                cdf_cut_lims=CUTS[cut_name]
            )

            cuts_cdf_z_R[str(scale_index)] = _cdf_z_R
            cuts_cdf_cut_idx_z_R[str(scale_index)] = _cdf_cut_idx_z_R

        return cuts_cdf_cut_idx_z_R, cuts_cdf_z_R


    cuts_cdf_cut_idx_z_R, cuts_cdf_z_R = get_cut_indices(cut_name=bulk_or_tails)


    @typecheck
    def get_k_n_fiducials(
        redshift: float, 
        cut_name: Literal["bulk", "tails"]
    ) -> tuple[
        Float[np.ndarray, "n r k"], 
        Optional[Float[np.ndarray, "1 r 1"]]
    ]:

        fiducials_k_n = np.zeros((N_FIDUCIAL_REALISATIONS, len(scale_numbers), N_CUMULANTS))

        bad_realisations = []
        for realisation_idx, realisation_path_folder_number in zip(
            trange(
                0, N_FIDUCIAL_REALISATIONS, 
                desc="Fiducial k_n [{}]".format(cut_name),
                colour="blue" if cut_name == "bulk" else "red"
            ),
            fiducial_sorted_folders
        ):

            realisation_folder_number = int(realisation_path_folder_number.name)

            bad = False # if any of the PDFs are not loaded skip the realisation
            for radius_index, R in enumerate(scale_numbers):

                try:
                    pdf_filename = get_fiducial_pdf_filename_template(
                        redshift=redshift, 
                        realisation=realisation_folder_number, 
                        radius_index=np.squeeze(np.argwhere(ALL_R_NUMBERS == R))
                    )

                    _, pdf = np.loadtxt(pdf_filename).T

                    cdf_cut_idx = cuts_cdf_cut_idx_z_R[str(R)]

                    if cut_name == "bulk":
                        p_min, p_max = cdf_cut_lims
                        prob_norm = p_max - p_min
                    else:
                        prob_norm = 1.0

                    k_n_R_z = cut_pdf_to_cumulants(
                        pdf[cdf_cut_idx],
                        bins[cdf_cut_idx],
                        dbins[cdf_cut_idx],
                        prob_norm=prob_norm # NOTE: set to 1 for tails!
                    )

                    fiducials_k_n[realisation_idx, radius_index, :] = k_n_R_z

                except FileNotFoundError as e:
                    print(e)
                    bad = True
                    bad_realisations.append(realisation_idx)
                    continue

        # Delete bad realisations in returned array (zero rows)
        bad_realisations = list(set(bad_realisations))
        for bad_idx in sorted(bad_realisations, reverse=True):
            fiducials_k_n = np.delete(fiducials_k_n, bad_idx, axis=0)

        print("bad (fiducial):", bad_realisations)

        if FIDUCIAL_REDUCE:
            # Select variances for each scale and redshift
            # fiducial_vars = jax.tree.map(
            #     lambda a: jnp.mean(a[..., 2], axis=0), # Select variance
            #     fiducials_k_n
            # )[jnp.newaxis, ..., jnp.newaxis]
            # fiducial_vars = np.asarray(fiducial_vars)

            fiducial_vars = np.mean(fiducials_k_n, axis=0)[..., 2]
            fiducial_vars = fiducial_vars[np.newaxis, ..., np.newaxis]

            fiducials_k_n = reduce(fiducials_k_n, fiducial_vars)
        else:
            fiducial_vars = None

        return fiducials_k_n, fiducial_vars


    fiducials_k_n, fiducial_vars = get_k_n_fiducials(redshift, cut_name=bulk_or_tails)

    """
        Latins
    """

    @typecheck
    def get_k_n_latins(
        redshift: float, 
        fiducial_vars: Optional[Float[np.ndarray, "1 r 1"]], 
        cut_name: Literal["bulk", "tails"]
    ) -> tuple[Float[np.ndarray, "n r k"], Float[np.ndarray, "n p"]]:

        latins_k_n = np.zeros((N_LATIN_REALISATIONS, len(scale_numbers), N_CUMULANTS))

        latin_parameters = np.zeros((N_LATIN_REALISATIONS, ALPHA.size))

        bad_realisations = []

        for realisation_idx, realisation_path_folder_number in zip(
            trange(
                0, N_LATIN_REALISATIONS, 
                desc="Latin k_n [{}]".format(cut_name),
                colour="blue" if cut_name == "bulk" else "red"
            ),
            latin_sorted_folders
        ):

            realisation_folder_number = int(realisation_path_folder_number.name)

            latin_parameters[realisation_idx] = ALL_LATIN_PARAMETERS[realisation_folder_number]

            bad = False # if any of the PDFs are not loaded skip the realisation
            for radius_index, R in enumerate(scale_numbers):

                try:
                    pdf_filename = get_latin_pdf_filename_template(
                        redshift=redshift, 
                        realisation=realisation_folder_number, 
                        radius_index=np.squeeze(np.argwhere(ALL_R_NUMBERS == R))
                    )

                    _, pdf = np.loadtxt(pdf_filename).T

                    cdf_cut_idx = cuts_cdf_cut_idx_z_R[str(R)]

                    if cut_name == "bulk":
                        p_min, p_max = CUTS[cut_name]
                        prob_norm = p_max - p_min
                    else:
                        prob_norm = 1.0

                    k_n_R_z = cut_pdf_to_cumulants(
                        pdf[cdf_cut_idx],
                        bins[cdf_cut_idx],
                        dbins[cdf_cut_idx],
                        prob_norm=prob_norm # NOTE: set to 1 for tails!
                    )

                    latins_k_n[realisation_idx, radius_index, :] = k_n_R_z

                except FileNotFoundError as e:
                    print(e)
                    bad = True
                    bad_realisations.append(realisation_idx)
                    continue

        # Delete bad realisations in returned array (zero rows)
        bad_realisations = list(set(bad_realisations))
        for bad_idx in sorted(bad_realisations, reverse=True):
            latins_k_n = np.delete(latins_k_n, bad_idx, axis=0)
            latin_parameters = np.delete(latin_parameters, bad_idx, axis=0)

        print("bad (latin):", bad_realisations)

        if FIDUCIAL_REDUCE:
            assert fiducial_vars is not None

            latins_k_n = reduce(latins_k_n, fiducial_vars)

        return latins_k_n, latin_parameters


    latins_k_n, latin_parameters = get_k_n_latins(
        redshift, fiducial_vars=fiducial_vars, cut_name=bulk_or_tails
    )

    """
        Derivatives
    """


    @typecheck
    def get_k_n_derivatives(
        redshift: float, 
        fiducial_vars: Optional[Float[np.ndarray, "1 r 1"]], 
        cut_name: Literal["bulk", "tails"]
    ) -> Float[np.ndarray, "n p r k"]:

        derivatives_k_n = np.zeros((N_DERIVATIVE_REALISATIONS, ALPHA.size, len(scale_numbers), N_CUMULANTS))

        bad_realisations = []

        for i_p in range(ALPHA.size):

            PARAMETERS_PM = PARAMETER_DERIVATIVE_STRINGS[i_p]

            # For each parameter, calculate cumulants for all realisations, plus and minus
            dummy_k_n = np.zeros((N_DERIVATIVE_REALISATIONS, len(scale_numbers), N_CUMULANTS, 2))

            for realisation_idx, realisation_path_folder_number in zip(
                trange(
                    0, N_DERIVATIVE_REALISATIONS, 
                    desc="Derivatives k_n [{}] ".format(cut_name) + PARAMETER_STRINGS[i_p][1:-1],
                    colour="blue" if cut_name == "bulk" else "red"
                ),
                derivative_sorted_folders
            ):

                realisation_folder_number = int(realisation_path_folder_number.name)

                for pm, parameter_p_or_m in enumerate(PARAMETERS_PM): # Plus then minus 

                    bad = False # if any of the PDFs are not loaded skip the realisation
                    for radius_index, R in enumerate(scale_numbers):

                        try:
                            derivative_filename = get_pdf_derivative_filename_template(
                                derivative_p_m_name=parameter_p_or_m, 
                                redshift=redshift, 
                                realisation=realisation_folder_number, 
                                radius_index=np.squeeze(np.argwhere(ALL_R_NUMBERS == R))
                            )

                            _, pdf = np.loadtxt(derivative_filename).T

                            cdf_cut_idx = cuts_cdf_cut_idx_z_R[str(R)]

                            if cut_name == "bulk":
                                p_min, p_max = CUTS[cut_name]
                                prob_norm = p_max - p_min
                            else:
                                prob_norm = 1.0

                            k_n_R_z = cut_pdf_to_cumulants(
                                pdf[cdf_cut_idx],
                                bins[cdf_cut_idx],
                                dbins[cdf_cut_idx],
                                prob_norm=prob_norm # NOTE: set to 1 for tails!
                            )

                            dummy_k_n[realisation_idx, radius_index, :, pm] = k_n_R_z

                        except FileNotFoundError as e:
                            # print(e)
                            bad = True
                            bad_realisations.append((realisation_idx, realisation_folder_number))
                            continue

            # Store the finite difference gradients for each parameter
            derivatives_k_n[:, i_p, :, :] = (dummy_k_n[..., 0] - dummy_k_n[..., 1]) / DPARAMS[i_p]

        # Delete bad realisations in returned array (zero rows)
        bad_realisations = list(set(bad_realisations))
        for bad_idx, _ in sorted(bad_realisations, reverse=True):
            derivatives_k_n = np.delete(derivatives_k_n, bad_idx, axis=0)

        print("bad (derivative):", bad_realisations)

        if FIDUCIAL_REDUCE:
            assert fiducial_vars is not None

            for i_p in range(ALPHA.size):
                derivatives_k_n[:, i_p, :, :] = reduce(
                    derivatives_k_n[:, i_p, :, :], fiducial_vars
                )

        return derivatives_k_n


    _derivatives_k_n = get_k_n_derivatives(
        redshift, fiducial_vars=fiducial_vars, cut_name=bulk_or_tails
    )

    """
        Postprocess
    """

    # Flatten for fisher forecasting
    fiducials_k_n = np.reshape(fiducials_k_n, (fiducials_k_n.shape[0], -1))
    latins_k_n = np.reshape(latins_k_n, (latins_k_n.shape[0], -1))

    # derivatives_k_n = np.reshape(derivatives_k_n, (derivatives_k_n.shape[0], ALPHA.size, -1))

    derivatives_k_n = np.zeros((_derivatives_k_n.shape[0], ALPHA.size, len(scale_numbers) * N_CUMULANTS))
    for i_p in range(ALPHA.size):
        derivatives_k_n[:, i_p, :] = _derivatives_k_n[:, i_p, :, :].reshape(_derivatives_k_n.shape[0], -1)

    print("DATA:", jax.tree.map(lambda a: a.shape, (fiducials_k_n, latins_k_n, derivatives_k_n, latin_parameters)))

    return fiducials_k_n, latins_k_n, derivatives_k_n, latin_parameters


def get_calculated_cumulants_data(
    config: ConfigDict, 
    *, 
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
        *_
    ) = get_quijote_parameters() # These are set to SOBOLs

    # p_value_min                  = config.p_value_min # Independent of choosing rho/delta for random variable of PDF
    # p_value_max                  = config.p_value_max 

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
            "_sobol",
            "_R" + "".join(map(str, config.scales)),
            "_m" + "".join(map(str, config.order_idx)),
            "_z" + str(config.redshift),
            "_f" if config.freeze_parameters else "_nf",
            "_reduced" if FIDUCIAL_REDUCE else "", # Reduction k_n -> S_n with fiducial variance
            # Bulk calcuations 
            "_with_means" if use_means else "",
            "_central" if central_moments else "",
            "_with_norms" if use_normalisations else "",
            "_with_means_stacked" if stack_means else ""
        ]
    )

    # Try loading dataset instead of deriving it again and again NOTE: careful not to load PDFs when yhou need cumulants etc
    dataset_filename = os.path.join(
        data_dir, "datasets/{}_cumulants_dataset{}.npz".format(bulk_or_tails, dataset_identifier_str)
    )


    def generate_dataset() -> Dataset:

        (
            fiducial_moments_z_R, 
            latin_moments_z_R, 
            derivative_moments_z_R,
            latin_parameters
        ) = get_fiducials_latins_derivatives_cumulants(
            bulk_or_tails, 
            redshift=config.redshift,
            cdf_cut_lims=CUTS[bulk_or_tails]
        )

        """
            Bulk Cumulants
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
        cumulants_dataset = Dataset(
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
            # NOTE: for PDFs also?
            cumulants_dataset = freeze_out_parameters_dataset(cumulants_dataset)

        # Save _return_ dataset to ensure loading (not creating) next time around
        np.savez(dataset_filename, **asdict(cumulants_dataset))

        logger.info("Saved dataset:\n\t{}".format(dataset_filename))

        return cumulants_dataset


    """ 
        Forced recomputation or not
    """

    # Create a fresh dataset if required, or generate one if it does not exist
    if not FORCE_RECOMPUTE_DATASET:
        try:
            logger.info("Loading dataset:\n\t{}".format(dataset_filename))

            dataset_dict = np.load(dataset_filename, allow_pickle=True) 

            dataset_name = "bulk" if not full_shape else "tails"

            return_dataset = Dataset.from_dict(dataset_dict, name=dataset_name)

            logger.info("Loaded dataset:\n\t{}".format(dataset_filename))

        except FileNotFoundError:
            logger.info("Generating dataset [NotFound]:\n\t{}".format(dataset_filename))

            return_dataset = generate_dataset()

            logger.info("Generated dataset:\n\t{}".format(dataset_filename))
    else:
        logger.info("Generating dataset [Force]:\n\t{}".format(dataset_filename))

        return_dataset = generate_dataset()

        logger.info("Generated dataset:\n\t{}".format(dataset_filename))

    """ 
        Linearise or Non-Gaussianise; whether dataset is loaded or not
    """

    if config.linearised:
        logger.info("Using linearised dataset [replacing only hypercube]...")

        D, Y = get_linearised_data(config, return_dataset) 

        return_dataset = replace(return_dataset, data=D, parameters=Y)

    if NON_GAUSSIAN_TEST:
        logger.info("Using non-Gaussian linear model dataset [replacing only hypercube]...")

        key = jr.key(config.seed)

        D, Y = get_non_gaussian_linear_model_data(config, return_dataset, key=key)

        return_dataset = replace(return_dataset, data=D, parameters=Y)

    return return_dataset 


def get_calculated_pdfs_data(
    config: ConfigDict, 
    *, 
    full_shape: bool = False,
    results_dir: Optional[str] = None
) -> Dataset:
    """
        Get dataset for SBI experiments with the cumulants.
        - Cut the PDFs according to a p_min, p_max cut into the CDF which 
          indexes the bins of the PDF.
        - Return PDFs of cumulants of the bulk
    """

    logger.info("Getting PDFs for dataset={}".format(config.dataset_name))

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

    if full_shape:
        bulk_or_tails = "tails"
    else:
        bulk_or_tails = "bulk"

    # Name for dataset to load / save once created
    dataset_identifier_str = "".join(
        [
            # Datavector, model and specification
            "_sobol",
            "_R" + "".join(map(str, config.scales)),
            "_z" + str(config.redshift),
            "_f" if config.freeze_parameters else "_nf",
        ]
    )

    # Try loading dataset instead of deriving it again and again NOTE: careful not to load PDFs when yhou need cumulants etc
    dataset_filename = os.path.join(
        data_dir, "datasets/{}_pdf_dataset{}.npz".format(bulk_or_tails, dataset_identifier_str)
    )


    def generate_dataset() -> Dataset:

        """
            Bulk PDFs 
        """

        # If PDFs chosen, try and load PDF dataset else generate it
        try:
            logger.info("Loading PDF dataset:\n\t{}".format(dataset_filename))

            dataset_dict = np.load(dataset_filename, allow_pickle=True) 

            pdf_dataset = Dataset(
                name="bulk_pdf",
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
                cdf_cut_lims=CUTS[bulk_or_tails],
                get_latins=False
            )

            # Fisher information in bulk of the PDF
            n_fiducial_pdfs, data_dim_pdfs = fiducial_pdfs_z_R.shape 
            C_pdf = np.cov(fiducial_pdfs_z_R, rowvar=False) 
            H = hartlap(n_s=n_fiducial_pdfs, n_d=data_dim_pdfs) 
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
                data=jnp.zeros((config.n_linear_sims if config.linearised else 32768, data_dim_pdfs)), # NOTE: Dummy array
                parameters=jnp.zeros((config.n_linear_sims if config.linearised else 32768, 5)), # NOTE: dummy array
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

            logger.info("Returning PDFs as dataset...")

            # Save PDF dataset to ensure loading (not creating) next time around
            np.savez(dataset_filename, **asdict(pdf_dataset))

            logger.info("Saved dataset:\n\t{}".format(dataset_filename))

        # NOTE: this return dataset is OVERWRITING THE CUMULANTS DATASET
        return pdf_dataset


    # Create a fresh dataset if required, or generate one if it does not exist
    if not FORCE_RECOMPUTE_DATASET:
        try:
            logger.info("Loading dataset:\n\t{}".format(dataset_filename))

            dataset_dict = np.load(dataset_filename, allow_pickle=True) 

            dataset_name = "{}_pdf".format(bulk_or_tails)

            return_dataset = Dataset.from_dict(dataset_dict, name=dataset_name)

            logger.info("Loaded dataset:\n\t{}".format(dataset_filename))

        except FileNotFoundError:
            logger.info("Generating dataset [NotFound]:\n\t{}".format(dataset_filename))

            return_dataset = generate_dataset()

            logger.info("Generated dataset:\n\t{}".format(dataset_filename))
    else:
        logger.info("Generating dataset [Force]:\n\t{}".format(dataset_filename))

        return_dataset = generate_dataset()

        logger.info("Generated dataset:\n\t{}".format(dataset_filename))

    """ 
        Linearise or Non-Gaussianise; whether dataset is loaded or not
    """
    
    if config.linearised:
        logger.info("Using linearised dataset [replacing only hypercube]...")

        D, Y = get_linearised_data(config, return_dataset) 

        return_dataset = replace(return_dataset, data=D, parameters=Y)

    if NON_GAUSSIAN_TEST:
        logger.info("Using non-Gaussian linear model dataset [replacing only hypercube]...")

        key = jr.key(config.seed)

        D, Y = get_non_gaussian_linear_model_data(config, return_dataset, key=key)

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
    results_dir: Optional[str]

    def __init__(
        self, 
        config: ConfigDict, 
        *, 
        pdfs: bool = False,
        results_dir: Optional[str] = None
    ):
        self.config = config

        if pdfs:
            self.data = get_calculated_pdfs_data(
                config, full_shape=False, results_dir=results_dir
            )
        else:
            self.data = get_calculated_cumulants_data(
                config, 
                use_means=config.use_means,
                use_normalisations=config.use_normalisations,
                stack_means=config.stack_means,
                full_shape=False,
                results_dir=results_dir
            )

        self.prior = get_prior(config, self.data) # Possibly not equal to Quijote prior

        self.compression_fn = None

        self.results_dir = results_dir

        logger.info("BULK CUMULANT DATASET {}".format("pdfs" if pdfs else ""))
        logger.info(
            ">DATA:\n\t {}".format(
                ["{:.3E} {:.3E}".format(_.min(), _.max()) 
                 for _ in (self.data.fiducial_data, self.data.data)]
            )
        )
        logger.info(
            ">DATA / PARAMETERS:\n\t {}".format(
                [_.shape for _ in (self.data.data, self.data.parameters)]
            )
        )

    def get_parameter_strings(self):
        return get_parameter_strings()

    def sample_prior(
        self, 
        key: PRNGKeyArray, 
        n: int, 
        *, 
        hypercube: bool = True
    ) -> Float[Array, "n p"]:
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

    def get_compression_fn(self, train: bool = True):
        if self.compression_fn is None:
            key = jr.key(self.config.seed)

            fn = get_compression_fn(
                key, self.config, self.data, train=train, results_dir=self.results_dir
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
        Dataset for Simulation-Based Inference with cumulants of the bulk + tails of the matter PDF 
    """

    config: ConfigDict
    data: Dataset
    prior: tfd.Distribution
    compression_fn: Optional[Callable[[Array, Array], Array]]
    results_dir: Optional[str]

    def __init__(
        self, 
        config: ConfigDict, 
        *, 
        pdfs: bool = False,
        results_dir: Optional[str] = None
    ):
        self.config = config

        if pdfs:
            self.data = get_calculated_pdfs_data(
                config, full_shape=True, results_dir=results_dir
            )
        else:
            self.data = get_calculated_cumulants_data(
                config, 
                use_means=config.use_means,
                use_normalisations=config.use_normalisations,
                stack_means=config.stack_means,
                full_shape=True, # Implies full-shape calculation
                results_dir=results_dir
            )

        self.prior = get_prior(config, self.data) # Possibly not equal to Quijote prior

        self.compression_fn = None # Don't recalculate it every time

        self.results_dir = results_dir

        logger.info("TAILS CUMULANT DATASET")
        logger.info(
            ">DATA:\n\t {}".format(
                ["{:.3E} {:.3E}".format(_.min(), _.max()) 
                 for _ in (self.data.fiducial_data, self.data.data)]
            )
        )
        logger.info(
            ">DATA / PARAMETERS:\n\t {}".format(
                [_.shape for _ in (self.data.data, self.data.parameters)]
            )
        )

    def get_parameter_strings(self):
        return get_parameter_strings()

    def sample_prior(
        self, 
        key: PRNGKeyArray, 
        n: int, 
        *, 
        hypercube: bool = True
    ) -> Float[Array, "n p"]:
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

    def get_compression_fn(self, train: bool = True):
        if self.compression_fn is None:

            key = jr.key(self.config.seed)

            fn = get_compression_fn(
                key, self.config, self.data, train=train, results_dir=self.results_dir
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


            # return_dataset = Dataset(
            #     name=dataset_name,
            #     alpha=jnp.asarray(dataset_dict["alpha"]),
            #     lower=jnp.asarray(dataset_dict["lower"]),
            #     upper=jnp.asarray(dataset_dict["upper"]),
            #     parameter_strings=list(dataset_dict["parameter_strings"]),
            #     Finv=jnp.asarray(dataset_dict["Finv"]),
            #     Cinv=jnp.asarray(dataset_dict["Cinv"]),
            #     C=jnp.asarray(dataset_dict["C"]),
            #     fiducial_data=jnp.asarray(dataset_dict["fiducial_data"]),
            #     data=jnp.asarray(dataset_dict["data"]),
            #     parameters=jnp.asarray(dataset_dict["parameters"]),
            #     derivatives=jnp.asarray(dataset_dict["derivatives"]),
            # )