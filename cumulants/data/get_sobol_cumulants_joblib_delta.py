from typing import Literal, Optional, Callable, Union, Any
import time
import sys
import math
import os
from collections import Counter
from pathlib import Path
from dataclasses import dataclass, replace, asdict

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Int, Array, PRNGKeyArray, jaxtyped
from beartype import beartype as typechecker
import numpy as np
import matplotlib.pyplot as plt
from ml_collections import ConfigDict

# ---- NEW: joblib parallelism knobs ----
from joblib import Parallel, delayed
import multiprocessing

# Prevent BLAS oversubscription when using processes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

N_JOBS = int(os.environ.get("N_JOBS", multiprocessing.cpu_count()))

from configs.log import setup_module_logger, get_log_level
from data.constants import (
    get_sobol_scale_numbers, 
    get_save_and_load_dirs, 
    get_cumulant_names,
    ALL_REDSHIFTS, 
    QUIJOTE_DIR,
    DPARAMS, 
    ALPHA,
    LOWER,
    UPPER,
    PARAMETER_STRINGS, 
    PARAMETER_DERIVATIVE_STRINGS,
    REDSHIFT_STRINGS
)
from data.common import (
    Dataset,
    get_prior,
    sample_prior,
    get_linearised_data,
    get_non_gaussian_linear_model_data,
    get_datavector,
    hartlap
)
from compression import get_compression_fn
from configs.cumulants_configs import bulk_cumulants_config

TYPECHECK = True if os.environ.get("TYPECHECK", "").lower() in ("1", "true") else False
if TYPECHECK:
    typecheck = jaxtyped(typechecker=typechecker)
else:
    typecheck = lambda x: x

Distribution = Any

logger, log_figs_dir = setup_module_logger(__name__, level=get_log_level())

FORCE_RECOMPUTE_DATASET = True if os.environ.get("FORCE_RECOMPUTE_DATASET", "").lower() in ("1", "true") else False 
FIDUCIAL_REDUCE = True if os.environ.get("FIDUCIAL_REDUCE", "").lower() in ("1", "true") else False
NON_GAUSSIAN_TEST = True if os.environ.get("NON_GAUSSIAN_TEST", "").lower() in ("1", "true") else False
DATASET_TEST = True if os.environ.get("DATASET_TEST", "").lower() in ("1", "true") else False

DELTAS_CUT = True if os.environ.get("DELTAS_CUT", "").lower() in ("1", "true") else False

# Use smaller dataset to not have to gather entire dataset
if DATASET_TEST:
    N_TEST = 1000
    N_TEST_DD = min(N_TEST // 2, 500) # NOTE: use half as many derivatives, for speed
else:
    N_TEST = N_TEST_DD = None

"""
    Load Sobol PDFs for 
    - fiducial
    - latin
    - derivatives
    and calculate the cumulants as well as the bulk PDF measurements
"""

JOBLIB_VERBOSE = 10

SOBOL_DATA_DIR  = Path(QUIJOTE_DIR) # Set to Sobol when using what's in this file

LATINS_DIR      = SOBOL_DATA_DIR / "matterPDF_BSQ"
DERIVATIVES_DIR = SOBOL_DATA_DIR / "matterPDF_fiducial" / "derivatives"
FIDUCIALS_DIR   = SOBOL_DATA_DIR / "matterPDF_fiducial" / "fiducial"

fiducial_sorted_folders = sorted(
    [p for p in FIDUCIALS_DIR.iterdir() if p.is_dir() and p.name.isdigit()],
    key=lambda p: int(p.name)
)[slice(None, N_TEST)]

derivative_sorted_folders = sorted(
    [p for p in DERIVATIVES_DIR.iterdir() if p.is_dir() and p.name.isdigit()],
    key=lambda p: int(p.name)
)[slice(None, N_TEST_DD)] 

latin_sorted_folders = sorted(
    [p for p in LATINS_DIR.iterdir() if p.is_dir() and p.name.isdigit()],
    key=lambda p: int(p.name)
)[slice(None, N_TEST)]

# print(fiducial_sorted_folders[:100], derivative_sorted_folders[:100], latin_sorted_folders[:100])

N_FIDUCIAL_REALISATIONS = len(fiducial_sorted_folders)
N_LATIN_REALISATIONS = len(latin_sorted_folders)
N_DERIVATIVE_REALISATIONS = len(derivative_sorted_folders)

ALL_LATIN_PARAMETERS = np.loadtxt(LATINS_DIR / "BSQ_params.txt")

N_CUMULANTS = 5 # m_0, m_1, k_2, k_3, k_4

ALL_R_NUMBERS = get_sobol_scale_numbers()

CUTS = dict(bulk=(0.03, 0.90), tails=(0.0, 0.999999)) # tails=(0.01, 0.999))

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


def plot_cumulants_single_figure_grid(
    *,
    fiducials: np.ndarray,       # (Nf, R*K) or (Nf, R, K)
    latins: np.ndarray,          # (Nl, R*K) or (Nl, R, K)
    scale_numbers: np.ndarray,   # shape (R,)
    out_dir: str,
    redshift: float,
    bulk_or_tails: str,
    order_labels=get_cumulant_names(),
    max_show: int = 500,         # cap realizations per histogram for speed
    bins: int = 30               # histogram bins
):
    """
    Produces TWO images (one for fiducials, one for latins), each a single figure:
    rows = scales, columns = cumulant orders. Each panel is a histogram of realizations.
    """

    def _ensure_3d(arr, n_scales, n_cumulants=N_CUMULANTS):
        """Accept (N, R*K) or (N, R, K); return (N, R, K)."""
        if arr.ndim == 3:
            N, R, K = arr.shape
            if R != n_scales or K != n_cumulants:
                raise ValueError(f"Unexpected shape {arr.shape}; expected (N,{n_scales},{n_cumulants}).")
            return arr
        elif arr.ndim == 2:
            N, D = arr.shape
            expected = n_scales * n_cumulants
            if D != expected:
                raise ValueError(f"Cannot reshape: got (N,{D}) but n_scales*n_cumulants={expected}.")
            return arr.reshape(N, n_scales, n_cumulants)
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}.")

    os.makedirs(out_dir, exist_ok=True)
    R = len(scale_numbers)
    K = N_CUMULANTS

    def _one(name: str, A: np.ndarray):
        if A.size == 0:
            return
        A3 = _ensure_3d(A, R, K)  # (N, R, K)
        N = A3.shape[0]
        sel = np.arange(N) if N <= max_show else np.random.choice(N, max_show, replace=False)

        # fig, axes = plt.subplots(R, K, figsize=(3.8*K, 2.8*R), squeeze=False)
        fig, axes = plt.subplots(
            R, 
            K, 
            figsize=(15., 27.), 
            dpi=200, 
            sharex=False, 
            sharey=False
        )
        for i_R, Rval in enumerate(scale_numbers):
            for k in range(K):
                ax = axes[i_R][k]
                data = A3[sel, i_R, k]
                ax.hist(data, bins=bins, alpha=0.8, color="firebrick" if bulk_or_tails == "tails" else "royalblue")
                mu = np.mean(data)
                ax.axvline(mu, linestyle="--", linewidth=1, color="k")
                ax.set_title(f"R={Rval:.1f}, {order_labels[k]}", fontsize=10)
                ax.tick_params(labelsize=8)
        # plt.suptitle(f"{name} cumulants — z={redshift}, {bulk_or_tails}", fontsize=12)
        # plt.tight_layout(rect=[0, 0, 1, 0.96])
        fout = os.path.join(out_dir, f"{name}_cumulants_grid_z{redshift}_{bulk_or_tails}.png")
        plt.savefig(fout, bbox_inches="tight")
        plt.close()

    _one("fiducials", fiducials)
    _one("latins", latins)


def _plot_pdfs_per_scale_panel(
    fiducial_pdfs: np.ndarray,  # shape (N_fid, D)
    latin_pdfs: np.ndarray,     # shape (N_lat, D)
    bins: np.ndarray,           # 1D array of bin centers
    scale_numbers: np.ndarray,  # list of R values
    cuts_cdf_cut_idx_z_R: dict[str, np.ndarray],  # maps R→cut_idx
    out_dir: str,
    *,
    redshift: float,
    bulk_or_tails: str,
    max_show: int = 100,
):
    """
    Make multi-panel plots of PDFs (fiducials, latins) for each scale.
    Each panel shows up to `max_show` realizations as semi-transparent lines.
    Saves two PNGs: one for fiducials, one for latins.
    """

    os.makedirs(out_dir, exist_ok=True)

    n_scales = len(scale_numbers)
    n_cols = math.ceil(math.sqrt(n_scales))
    n_rows = math.ceil(n_scales / n_cols)

    def _plot_dataset(dataset_name: str, pdfs: np.ndarray, color: str):
        if pdfs.size == 0:
            return

        plt.figure(figsize=(3.5 * n_cols, 3.0 * n_rows))
        for i_R, R in enumerate(scale_numbers):
            plt.subplot(n_rows, n_cols, i_R + 1)
            cidx = cuts_cdf_cut_idx_z_R[str(R)]
            nbins = cidx.size
            D = pdfs.shape[1]
            if D % nbins != 0:
                # if flattened across R, we slice accordingly
                stride = D // n_scales
                seg = pdfs[:, i_R * stride : (i_R + 1) * stride]
            else:
                seg = pdfs[:, cidx]

            n_show = min(max_show, seg.shape[0])
            sel = np.random.choice(seg.shape[0], size=n_show, replace=False)

            for j in sel:
                plt.plot(bins[cidx], seg[j, :], color=color, alpha=0.2, lw=0.7)

            plt.title(f"R={R:.1f}")
            plt.xlabel(r"$\delta$")
            plt.ylabel("PDF")
            plt.xlim(bins[cidx].min(), bins[cidx].max())

        plt.tight_layout()
        fname = os.path.join(
            out_dir,
            f"{dataset_name}_pdfs_z{redshift}_{bulk_or_tails}.png"
        )
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved {dataset_name} PDF panel → {fname}")

    # Fiducials
    _plot_dataset("fiducials", fiducial_pdfs, color="black")
    # Latins
    _plot_dataset("latins", latin_pdfs, color="tab:blue")


def infer_pdf_type(pdf, widths, tol=1e-3):
    sum1 = np.sum(pdf)
    sum2 = np.sum(pdf * widths)
    if np.isclose(sum1, 1.0, rtol=tol):
        return "mass"  # already normalized per-bin probability
    elif np.isclose(sum2, 1.0, rtol=tol):
        return "height"  # density values (need *width for probability)
    else:
        return "unknown"


def get_cdf_of_pdf(
    pdf: Float[np.ndarray, "d"],
    dbins: Float[np.ndarray, "d"],
    cdf_cut_lims: tuple[float, float],
) -> tuple[Float[np.ndarray, "d"], Int[np.ndarray, "c"]]:
    """
    Build a *normalised* CDF from a binned PDF and return the indices that keep the
    percentile range [p_min, p_max].
    """
    d = pdf.size
    if dbins.size == d - 1:
        widths = np.concatenate([dbins, dbins[-1:]])
    elif dbins.size == d:
        widths = dbins
    else:
        raise ValueError(
            f"dbins must have length d or d-1; got d={d}, len(dbins)={dbins.size}"
        )

    # print("PDF TYPE GOING INTO CDF-CUTTER FN IS: {}".format(infer_pdf_type(pdf, widths, tol=0.1)))

    p_min, p_max = cdf_cut_lims
    if not (0. <= p_min < p_max <= 1.):
        raise ValueError(f"cdf_cut_lims must be within [0,1] and p_min<p_max; got {cdf_cut_lims}")

    dp = pdf * widths
    Z = np.sum(dp)
    if not np.isfinite(Z) or Z <= 0.:
        raise ValueError(f"PDF total mass must be positive/finite; got Z={Z}")

    dp_norm = dp / Z
    if not np.isclose(np.sum(dp_norm), 1., rtol=1e-6, atol=1e-7):
        raise ValueError("Normalized PDF does not sum to 1.")

    cdf = np.cumsum(dp_norm)
    cdf = np.clip(cdf, 0., 1.)

    cut_idx = np.flatnonzero((cdf >= p_min) & (cdf <= p_max))

    return cdf, cut_idx


@typecheck
def cut_pdf_to_cumulants(
    cut_pdf: Float[np.ndarray, "d"],
    deltas: Float[np.ndarray, "d"],
    ddeltas: Float[np.ndarray, "d"],
    prob_norm: float,
    *,
    dtype: np.typing.DTypeLike = np.float64
) -> Float[np.ndarray, "5"]:

    if (
        not np.isfinite(cut_pdf).all() 
        or not np.isfinite(deltas).all() 
        or not np.isfinite(ddeltas).all()
    ):
        raise ValueError("Inputs contain NaN/Inf.")
    if (ddeltas <= 0).any():
        raise ValueError("All bin widths (ddeltas) must be strictly positive.")
    if (cut_pdf < 0).any():
        raise ValueError("cut_pdf contains negative mass.")
    if not np.all(np.diff(deltas) > 0.):
        raise ValueError("`deltas` must be strictly increasing (bin ordering).")
    if not (0. < prob_norm <= 1.):
        raise ValueError(f"prob_norm must be in (0, 1], got {prob_norm}")

    # print("PDF TYPE GOING INTO K_N CALC. FN IS: {}".format(infer_pdf_type(cut_pdf, ddeltas)))

    cut_pdf, deltas, ddeltas, prob_norm = map(
        lambda a: np.asarray(a, dtype=dtype), 
        (cut_pdf, deltas, ddeltas, prob_norm)
    )

    dp = cut_pdf * ddeltas
    m_0 = np.sum(dp, dtype=dtype)
    p = dp / m_0 # max(m_0, 1e-12)

    # print(dp, m_0, p)

    if not np.isclose(p.sum(), 1.0, rtol=1e-3, atol=1e-6):
        # print("~" * 50)
        # print("P\n", p)
        # print("DP\n", dp)
        # print("M_0\n", m_0)
        # print("DELTAS\n", deltas.min(), deltas.max())
        # sys.exit()
        raise ValueError(f"Normalisation of PDF `p`={p.sum()}")

    m_1 = np.sum(p * deltas, dtype=dtype)
    deltamod = deltas - m_1

    k_2 = np.sum(deltamod ** 2. * p, dtype=dtype)
    k_3 = np.sum(deltamod ** 3. * p, dtype=dtype)
    k_4 = np.sum(deltamod ** 4. * p, dtype=dtype) - (3. * k_2 ** 2.)

    m_1, k_2, k_3, k_4 = map(lambda k_n: k_n * m_0, (m_1, k_2, k_3, k_4))

    if k_2 < -1e-12:
        raise ValueError(f"Computed k2 (variance) < 0: {k_2}")
    if m_0 <= 0.:
        raise ValueError(f"Computed m0 <= 0: {m_0}")

    k_n = np.asarray([m_0, m_1, k_2, k_3, k_4], dtype=np.float32)

    return k_n


def _argwhere_idx_for_R(R):
    return int(np.squeeze(np.argwhere(ALL_R_NUMBERS == R)))


# ---------- Fiducial PDF lengths (parallel over realizations) ----------

def _load_fiducial_realisation_for_lengths(realisation_path_folder_number, R_numbers, redshift):
    """Worker: load all R for one fiducial realization. Returns (id, list_of_pdfs, bins, max_len) or None on missing file."""
    realisation_folder_number = int(realisation_path_folder_number.name)
    pdfs_per_R = []
    bins = None
    max_len = 0
    for R in R_numbers:
        try:
            path = get_fiducial_pdf_filename_template(
                redshift=redshift,
                radius_index=_argwhere_idx_for_R(R),
                realisation=realisation_folder_number
            )
            _bins, pdf = np.loadtxt(path).T
            bins = _bins  # keep last seen; matches original behavior
            pdfs_per_R.append(pdf)
            max_len = max(max_len, pdf.size)
        except Exception as e:
            print("EXCEPTION FIDUCIAL REALISATION LENGTHS:\n", e)
            return None
    return (realisation_folder_number, pdfs_per_R, bins, max_len)

# ---------- Row builders for bulk PDF (fiducials/latins/derivatives) ----------

def _load_fiducial_row(
    realisation_path_folder_number: Path, 
    scale_numbers: list[int], 
    redshift: float, 
    cuts_cdf_cut_idx_z_R: dict[str, np.ndarray],
    deltas_cdf_z_R: dict[str, tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    realisation_folder_number = int(realisation_path_folder_number.name)
    parts = []
    for R in scale_numbers:
        path = get_fiducial_pdf_filename_template(
            redshift=redshift,
            radius_index=_argwhere_idx_for_R(R),
            realisation=realisation_folder_number
        )
        # bins, pdf = np.loadtxt(path).T
        # cut_idx = cuts_cdf_cut_idx_z_R[str(R)]
        # # cut_idx = (bins >= deltas_cdf_z_R[str(R)][0]) & (bins <= deltas_cdf_z_R[str(R)][1])
        # # print("FIDUCIAL CUT_IDX:", cut_idx.shape)
        # parts.append(pdf[cut_idx])
        try:
            bins, pdf = np.loadtxt(path).T
            # cut_idx = cuts_cdf_cut_idx_z_R[str(R)]
            delta_min = deltas_cdf_z_R[str(R)][0]
            delta_max = deltas_cdf_z_R[str(R)][1]
            cut_idx = (bins >= delta_min) & (bins <= delta_max)
            parts.append(pdf[cut_idx])
        except Exception as e:
            print("EXCEPTION FIDUCIAL ROWS:\n", e)
            return None
    return np.concatenate(parts)


def _load_latin_row_and_params(
    realisation_path_folder_number: Path, 
    scale_numbers: list[int], 
    redshift: float, 
    cuts_cdf_cut_idx_z_R: dict[str, np.ndarray],
    deltas_cdf_z_R: dict[str, tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    realisation_folder_number = int(realisation_path_folder_number.name)
    parts = []
    for R in scale_numbers:
        path = get_latin_pdf_filename_template(
            redshift=redshift,
            radius_index=_argwhere_idx_for_R(R),
            realisation=realisation_folder_number
        )
        try:
            bins, pdf = np.loadtxt(path).T
            # cut_idx = cuts_cdf_cut_idx_z_R[str(R)]
            delta_min = deltas_cdf_z_R[str(R)][0]
            delta_max = deltas_cdf_z_R[str(R)][1]
            cut_idx = (bins >= delta_min) & (bins <= delta_max)
            parts.append(pdf[cut_idx])
        except Exception as e:
            print("EXCEPTION LATIN ROWS:\n", e)
            return None
    return np.concatenate(parts), ALL_LATIN_PARAMETERS[realisation_folder_number]


def _load_derivative_row(
    realisation_path_folder_number: Path, 
    scale_numbers: list[int], 
    redshift: float, 
    cuts_cdf_cut_idx_z_R: dict[str, np.ndarray],
    deltas_cdf_z_R: dict[str, tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    realisation_folder_number = int(realisation_path_folder_number.name)
    rows = []
    for i_p in range(ALPHA.size):
        PARAMETERS_PM = PARAMETER_DERIVATIVE_STRINGS[i_p]
        plus_minus = []
        for param_p_m in PARAMETERS_PM:  # + then -
            parts = []
            for R in scale_numbers:
                path = get_pdf_derivative_filename_template(
                    derivative_p_m_name=param_p_m,
                    redshift=redshift,
                    realisation=realisation_folder_number,
                    radius_index=_argwhere_idx_for_R(R),
                )
                try:
                    bins, pdf = np.loadtxt(path).T
                    # cut_idx = cuts_cdf_cut_idx_z_R[str(R)]
                    delta_min = deltas_cdf_z_R[str(R)][0]
                    delta_max = deltas_cdf_z_R[str(R)][1]
                    cut_idx = (bins >= delta_min) & (bins <= delta_max)
                    parts.append(pdf[cut_idx])
                except FileNotFoundError as e:
                    print("EXCEPTION DERIVATIVES ROWS:\n", e)
                    return None
            plus_minus.append(np.concatenate(parts))
        diff = (plus_minus[0] - plus_minus[1]) / DPARAMS[i_p]
        rows.append(diff)
    return np.asarray(rows)  # shape (P, D)

# ---------- Row builders for cumulants (fiducials/latins/derivatives) ----------

def _fiducial_cumulants_row(
    realisation_path_folder_number: Path, 
    redshift: float, 
    scale_numbers: list[float], 
    bins: np.ndarray, 
    cuts_cdf_cut_idx_z_R: dict[str, np.ndarray], 
    deltas_cdf_z_R: dict[str, tuple[np.ndarray, np.ndarray]],
    cut_name: str
) -> np.ndarray:
    realisation_folder_number = int(realisation_path_folder_number.name)
    row = []
    p_min, p_max = CUTS[cut_name]
    prob_norm = p_max - p_min
    for R in scale_numbers:
        try:
            path = get_fiducial_pdf_filename_template(
                redshift=redshift, 
                realisation=realisation_folder_number, 
                radius_index=_argwhere_idx_for_R(R)
            )
            bins, pdf = np.loadtxt(path).T

            bins = bins - 1. # NOTE: assuming bins are delta'd here

            # bins, pdf = map(lambda a: a[:1000], (bins, pdf))

            # CDF cutting 
            # cidx = cuts_cdf_cut_idx_z_R[str(R)]

            delta_min = deltas_cdf_z_R[str(R)][0]
            delta_max = deltas_cdf_z_R[str(R)][1]
            cidx = (bins >= delta_min) & (bins <= delta_max)

            # Delta cutting
            # delta_min = deltas_cdf_z_R[str(R)][0]
            # delta_max = deltas_cdf_z_R[str(R)][1]
            # assert delta_min < delta_max, ("delta_min / delta_max = {} / {}".format(delta_min, delta_max))
            # cidx = (bins >= delta_min) & (bins <= delta_max)
            # assert cidx.size == pdf.size, "cidx, pdf.size = {}, {}".format(cidx.shape, pdf.shape)
            # assert cidx.size == bins.size, "cidx, bins.size = {}, {}".format(cidx.shape, bins.shape)

            dbins = (bins[1] - bins[0]) * np.ones_like(bins) # Fixed bin width...
            # assert cidx.size == dbins.size, "cidx, dbins.size = {}, {}".format(cidx.shape, dbins.shape)
            k_n_R_z = cut_pdf_to_cumulants(
                pdf[cidx], 
                bins[cidx], 
                dbins[cidx], 
                prob_norm=prob_norm
            )
            row.append(k_n_R_z)
        except Exception as e:
            print("EXCEPTION FIDUCIAL K_N ROWS:\n", e)
            return None
    return np.stack(row, axis=0)  # (R, K)


def _latin_cumulants_row_and_params(
    realisation_path_folder_number: Path, 
    redshift: float, 
    scale_numbers: list[float], 
    bins: np.ndarray, 
    cuts_cdf_cut_idx_z_R: dict[str, np.ndarray], 
    deltas_cdf_z_R: dict[str, tuple[np.ndarray, np.ndarray]],
    cut_name: str
) -> tuple[np.ndarray, np.ndarray]:
    realisation_folder_number = int(realisation_path_folder_number.name)
    row = []
    p_min, p_max = CUTS[cut_name]
    prob_norm = p_max - p_min
    for R in scale_numbers:
        try:
            path = get_latin_pdf_filename_template(
                redshift=redshift, 
                realisation=realisation_folder_number, 
                radius_index=_argwhere_idx_for_R(R)
            )
            bins, pdf = np.loadtxt(path).T

            bins = bins - 1. # NOTE: rho -> delta, is this consistent with fiducials?

            # cidx = cuts_cdf_cut_idx_z_R[str(R)]
            delta_min = deltas_cdf_z_R[str(R)][0]
            delta_max = deltas_cdf_z_R[str(R)][1]

            assert delta_min < delta_max, (
                "delta_min / delta_max = {} / {}".format(delta_min, delta_max)
            )
            assert not np.isclose(delta_min, delta_max, rtol=1e-8, atol=1e-12), (
                f"delta_min ({delta_min}) and delta_max ({delta_max}) are effectively equal"
            )

            cidx = (bins >= delta_min) & (bins <= delta_max)
            dbins = (bins[1] - bins[0]) * np.ones_like(bins) # bin widths in rho/delta are equal

            assert cidx.size == pdf.size
            assert cidx.size == bins.size
            assert cidx.size == dbins.size

            # plt.figure()
            # plt.plot(bins, pdf)
            # plt.plot(bins[cidx], pdf[cidx])
            # plt.xlim(-1., 10.)
            # plt.savefig(f"pdf_plots/pdf_{int(time.time())}.png")
            # plt.close()

            # print(pdf[cidx])

            k_n_R_z = cut_pdf_to_cumulants(
                pdf[cidx], 
                bins[cidx], 
                dbins[cidx], 
                prob_norm=prob_norm
            )
            row.append(k_n_R_z)
        except Exception as e:
            print("EXCEPTION LATIN K_N ROWS:\n", e)
            return None
    row = np.stack(row, axis=0)  # (R, K)
    params = ALL_LATIN_PARAMETERS[realisation_folder_number]
    return row, params


def _derivative_cumulants_row(    
    realisation_path_folder_number: Path, 
    redshift: float, 
    scale_numbers: list[float], 
    bins: np.ndarray, 
    cuts_cdf_cut_idx_z_R: dict[str, np.ndarray], 
    deltas_cdf_z_R: dict[str, tuple[np.ndarray, np.ndarray]],
    cut_name: str
) -> np.ndarray:    
    realisation_folder_number = int(realisation_path_folder_number.name)
    p_min, p_max = CUTS[cut_name]
    prob_norm = p_max - p_min
    P = ALPHA.size
    Rn = len(scale_numbers)
    K = N_CUMULANTS
    dummy = np.zeros((P, Rn, K, 2), dtype=np.float64)
    for i_p in range(P):
        PARAMETERS_PM = PARAMETER_DERIVATIVE_STRINGS[i_p]
        for pm, param_p_m in enumerate(PARAMETERS_PM):
            for r_idx, R in enumerate(scale_numbers):
                try:
                    path = get_pdf_derivative_filename_template(
                        derivative_p_m_name=param_p_m, redshift=redshift,
                        realisation=realisation_folder_number, radius_index=_argwhere_idx_for_R(R)
                    )
                    bins, pdf = np.loadtxt(path).T

                    bins = bins - 1.

                    cidx = cuts_cdf_cut_idx_z_R[str(R)]

                    delta_min = deltas_cdf_z_R[str(R)][0]
                    delta_max = deltas_cdf_z_R[str(R)][1]
                    cidx = (bins >= delta_min) & (bins <= delta_max)

                    # delta_min = deltas_cdf_z_R[str(R)][0]
                    # delta_max = deltas_cdf_z_R[str(R)][1]
                    # assert delta_min < delta_max, ("delta_min / delta_max = {} / {}".format(delta_min, delta_max))
                    # cidx = (bins >= delta_min) & (bins <= delta_max)
                    # assert cidx.size == pdf.size
                    # assert cidx.size == bins.size

                    dbins = (bins[1] - bins[0]) * np.ones_like(bins)
                    # assert cidx.size == dbins.size
                    k_n_R_z = cut_pdf_to_cumulants(
                        pdf[cidx], bins[cidx], dbins[cidx], prob_norm=prob_norm
                    )
                    dummy[i_p, r_idx, :, pm] = k_n_R_z
                except Exception as e:
                    print("EXCEPTION DERIVATIVE K_N ROWS:\n", e)
                    return None
    # finite difference per parameter
    out = np.zeros((P, Rn, K), dtype=np.float64)
    for i_p in range(P):
        out[i_p, :, :] = (dummy[i_p, :, :, 0] - dummy[i_p, :, :, 1]) / DPARAMS[i_p]
    return out  # (P, R, K)

# ------------------------------------------------------------------------------------
# ORIGINAL LOGIC WITH PARALLELIZED LOOPS
# ------------------------------------------------------------------------------------

def get_fiducial_pdfs_lengths(
    redshift: float, 
    R_numbers: Float[np.ndarray, "R"]
) -> tuple[
    Float[np.ndarray, "..."],
    Float[np.ndarray, "..."],
    int
]:
    # Parallel load all fiducial realizations (all R)
    results = Parallel(n_jobs=N_JOBS, verbose=JOBLIB_VERBOSE)(
        delayed(_load_fiducial_realisation_for_lengths)(folder, R_numbers, redshift)
        for folder in fiducial_sorted_folders
    )
    results = [r for r in results if r is not None]
    if not results:
        raise RuntimeError("No fiducial PDFs loaded!")

    max_length = max(r[3] for r in results)
    # Use last-seen bins consistent with original (bins updated when a longer PDF was found)
    bins = results[np.argmax([r[3] for r in results])][2]

    pdfs = np.zeros((max_length, len(R_numbers), len(results)))
    unique_lengths = [max_length]
    for i, (_, pdfs_per_R, _bins, mlen) in enumerate(results):
        unique_lengths.append(mlen)
        for j, pdf in enumerate(pdfs_per_R):
            pdfs[:pdf.size, j, i] = pdf

    bins = bins - 1. # NOTE: rho -> delta
    print("Rho -> delta in `get_fiducial_pdf_lengths`")

    counts = dict(Counter(unique_lengths))
    # print("UNIQUE LENGTHS IN FIDUCIAL PDFS: {}".format(counts))
    # logger.info("UNIQUE LENGTHS IN FIDUCIAL PDFS: {}".format(counts))

    plt.figure()
    plt.hist(unique_lengths)
    plt.xlabel("Number")
    plt.ylabel("Frequency")
    plt.title("Frequencies from PDF lengths")
    plt.tight_layout()
    plt.savefig("fiducial_pdf_lengths.png")
    plt.close()

    plt.figure()
    for i in range(pdfs.shape[-1]):
        plt.plot(bins, pdfs[:, 0, i], alpha=0.1, color="k")
    plt.xlim(-1., 10.)
    plt.savefig("FIDUCIALS_david.png")
    plt.close()

    return pdfs, bins, max_length


def get_pdf_dim_bulk_or_tails(cuts_cdf_cut_idx_z_R: dict[str, np.ndarray]) -> int:
    if not all([cut.ndim == 1 for cut in cuts_cdf_cut_idx_z_R.values()]):
        raise ValueError(
            f"cuts_cdf_cut_idx_z_R are not one dimensional arrays, {[cut.shape for cut in cuts_cdf_cut_idx_z_R.values()]}"
        )
    lengths = jax.tree.map(lambda a: a.size, cuts_cdf_cut_idx_z_R)
    dimension = sum(jax.tree.leaves(lengths))
    return dimension


@typecheck
def get_fiducials_latins_derivatives_bulk_pdf(
    redshift: float, 
    cdf_cut_lims: tuple[float, float],
    cut_name: Literal["bulk", "tails"],
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
        Float[np.ndarray, "nl p"],
        Float[np.ndarray, "nd p d"]
    ]
]:
    assert cdf_cut_lims[1] > cdf_cut_lims[0]

    scale_numbers = get_sobol_scale_numbers()

    pdfs, bins, max_length = get_fiducial_pdfs_lengths(redshift, scale_numbers)
    mean_pdfs = np.mean(pdfs, axis=-1)[:max_length]
    dbins = bins[1:] - bins[:-1]

    # ---- cuts from fiducial mean (original behavior) ----
    def get_cut_indices(cut_name: str) -> tuple[
        dict[str, Float[Array, "..."]], 
        dict[str, Int[Array, "..."]],
        dict[str, tuple[Float[Array, "..."], Float[Array, "..."]]]
    ]:
        cuts_cdf_z_R = dict()
        cuts_cdf_cut_idx_z_R = dict()
        deltas_cdf_z_R = dict()
        for i_R, scale_index in enumerate(scale_numbers):
            _cdf_z_R, _cdf_cut_idx_z_R = get_cdf_of_pdf(
                mean_pdfs[:max_length, i_R], 
                dbins=dbins, 
                cdf_cut_lims=CUTS[cut_name]
            )
            # CDF 
            cuts_cdf_z_R[str(scale_index)] = _cdf_z_R
            # CDF-cut indices
            cuts_cdf_cut_idx_z_R[str(scale_index)] = _cdf_cut_idx_z_R
            # Delta values for min and max of cut
            delta_min = bins[_cdf_cut_idx_z_R.min()] 
            delta_max = bins[_cdf_cut_idx_z_R.max()]
            assert -1. <= delta_min # It is rho here
            deltas_cdf_z_R[str(scale_index)] = (delta_min, delta_max)

            # print("DELTA MIN MAX R", scale_index, delta_min, delta_max)

        return cuts_cdf_cut_idx_z_R, cuts_cdf_z_R, deltas_cdf_z_R 

    cuts_cdf_cut_idx_z_R, cuts_cdf_z_R, deltas_cdf_z_R = get_cut_indices(cut_name=cut_name)

    # ---- FIDUCIALS (parallel) ----
    def get_pdf_fiducials(redshift, cut_name):
        all_scales_for_z_dim_ = get_pdf_dim_bulk_or_tails(cuts_cdf_cut_idx_z_R)

        rows = Parallel(n_jobs=N_JOBS, verbose=JOBLIB_VERBOSE)(
            delayed(_load_fiducial_row)(
                folder, scale_numbers, redshift, cuts_cdf_cut_idx_z_R, deltas_cdf_z_R
            )
            for folder in fiducial_sorted_folders
        )
        good = [r for r in rows if r is not None]
        fiducial_pdfs = np.stack(good) if len(good) else np.zeros((0, all_scales_for_z_dim_))
        # Derive bad indices (zeros in original were removed)
        bad_realisations = [i for i, r in enumerate(rows) if r is None]
        print("bad:", bad_realisations)
        return fiducial_pdfs

    fiducial_pdfs = get_pdf_fiducials(redshift, cut_name=cut_name)

    # ---- LATINS (parallel) ----
    def get_pdf_latins(redshift, cut_name):
        all_scales_for_z_dim_ = get_pdf_dim_bulk_or_tails(cuts_cdf_cut_idx_z_R)

        rows = Parallel(n_jobs=N_JOBS, verbose=JOBLIB_VERBOSE)(
            delayed(_load_latin_row_and_params)(
                folder, scale_numbers, redshift, cuts_cdf_cut_idx_z_R, deltas_cdf_z_R
            )
            for folder in latin_sorted_folders
        )
        good = []
        bad_realisations = []
        for i, rp in enumerate(rows):
            if isinstance(rp, tuple) and rp[0] is not None:
                good.append(rp)
            else:
                bad_realisations.append(i)

        if good:
            latin_pdfs = np.stack([r for (r, _) in good], axis=0)
            latin_parameters = np.stack([p for (_, p) in good], axis=0)
        else:
            latin_pdfs = np.zeros((0, all_scales_for_z_dim_))
            latin_parameters = np.zeros((0, ALPHA.size))

        print("bad:", bad_realisations)
        return latin_pdfs, latin_parameters

    if get_latins:
        latin_pdfs, latin_parameters = get_pdf_latins(redshift, cut_name=cut_name)

    # ---- DERIVATIVES (parallel) ----
    def get_pdf_derivatives(redshift, cut_name=None):
        all_scales_for_z_dim_ = get_pdf_dim_bulk_or_tails(cuts_cdf_cut_idx_z_R)

        rows = Parallel(n_jobs=N_JOBS, verbose=JOBLIB_VERBOSE)(
            delayed(_load_derivative_row)(
                folder, scale_numbers, redshift, cuts_cdf_cut_idx_z_R, deltas_cdf_z_R
            )
            for folder in derivative_sorted_folders
        )

        good = [r for r in rows if r is not None]

        if len(good):
            derivatives_pdf = np.stack(good, axis=0)
        else:
            derivatives_pdf = np.zeros((0, ALPHA.size, all_scales_for_z_dim_))
        bad_realisations = [i for i, r in enumerate(rows) if r is None]

        print("bad:", bad_realisations)
        return derivatives_pdf

    derivatives_pdf = get_pdf_derivatives(redshift, cut_name=cut_name)

    try:
        out_dir = os.path.join(
            log_figs_dir,
            f"pdfs_z{redshift}_{cut_name}"
        )
        _plot_pdfs_per_scale_panel(
            fiducial_pdfs=fiducial_pdfs,
            latin_pdfs=latin_pdfs if get_latins else np.empty((0,)),
            bins=bins,
            scale_numbers=np.asarray(scale_numbers),
            cuts_cdf_cut_idx_z_R=cuts_cdf_cut_idx_z_R,
            out_dir=out_dir,
            redshift=redshift,
            bulk_or_tails=cut_name,
            max_show=100,  # adjust as desired
        )
        logger.info(f"Saved PDF panels to {out_dir}")
    except Exception as e:
        logger.warning(f"PDF panel plotting failed: {e}")

    if get_latins:
        return fiducial_pdfs, latin_pdfs, latin_parameters, derivatives_pdf 
    else:
        return fiducial_pdfs, derivatives_pdf


@typecheck
def get_fiducials_latins_derivatives_cumulants(
    bulk_or_tails: Literal["bulk", "tails"], 
    redshift: float
) -> tuple[
    Float[np.ndarray, "nf d"],
    Float[np.ndarray, "nl d"],
    Float[np.ndarray, "nd p d"],
    Float[np.ndarray, "nl p"]
]:
    scale_numbers = get_sobol_scale_numbers()

    pdfs, bins, max_length = get_fiducial_pdfs_lengths(redshift, scale_numbers)
    mean_pdfs = np.mean(pdfs, axis=-1)[:max_length]
    dbins = bins[1:] - bins[:-1]

    @typecheck
    def reduce(
        k_n: Float[np.ndarray, "n r k"], 
        vars: Float[np.ndarray, "1 r 1"]
    ) -> Float[np.ndarray, "n r k"]:
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
        dict[str, Float[np.ndarray, "..."]],
        dict[str, tuple[Float[np.ndarray, "..."], Float[np.ndarray, "..."]]]
    ]:
        cuts_cdf_z_R = dict()
        cuts_cdf_cut_idx_z_R = dict() 
        deltas_cdf_z_R = dict() 
        for i_R, scale_index in enumerate(scale_numbers):
            _cdf_z_R, _cdf_cut_idx_z_R = get_cdf_of_pdf(
                mean_pdfs[:max_length, i_R], 
                dbins=dbins, 
                cdf_cut_lims=CUTS[cut_name]
            )
            cuts_cdf_z_R[str(scale_index)] = _cdf_z_R
            cuts_cdf_cut_idx_z_R[str(scale_index)] = _cdf_cut_idx_z_R
            delta_min = bins[_cdf_cut_idx_z_R.min()]
            delta_max = bins[_cdf_cut_idx_z_R.max()]
            assert delta_min < delta_max, "delta_min, delta_max={}, {}".format(delta_min, delta_max)
            deltas_cdf_z_R[str(scale_index)] = (delta_min, delta_max) 
        return cuts_cdf_cut_idx_z_R, cuts_cdf_z_R, deltas_cdf_z_R

    cuts_cdf_cut_idx_z_R, cuts_cdf_z_R, deltas_cdf_z_R = get_cut_indices(cut_name=bulk_or_tails)

    # ---- Fiducials (parallel) ----
    @typecheck
    def get_k_n_fiducials(
        redshift: float, 
        cut_name: Literal["bulk", "tails"]
    ) -> tuple[
        Float[np.ndarray, "n r k"], 
        Optional[Float[np.ndarray, "1 r 1"]]
    ]:
        rows = Parallel(n_jobs=N_JOBS, verbose=JOBLIB_VERBOSE)(
            delayed(_fiducial_cumulants_row)(
                folder, redshift, scale_numbers, bins, cuts_cdf_cut_idx_z_R, deltas_cdf_z_R, cut_name
            )
            for folder in fiducial_sorted_folders
        )
        good = [r for r in rows if r is not None]
        if len(good):
            fiducials_k_n = np.stack(good, axis=0)  # (N, R, K)
        else:
            fiducials_k_n = np.zeros((0, len(scale_numbers), N_CUMULANTS))
        bad_realisations = [i for i, r in enumerate(rows) if r is None]
        print("bad (fiducial):", bad_realisations)

        if FIDUCIAL_REDUCE and len(good):
            fiducial_vars = np.mean(fiducials_k_n, axis=0)[..., 2]
            fiducial_vars = fiducial_vars[np.newaxis, ..., np.newaxis]
            fiducials_k_n = reduce(fiducials_k_n, fiducial_vars)
        else:
            fiducial_vars = None

        return fiducials_k_n, fiducial_vars

    fiducials_k_n, fiducial_vars = get_k_n_fiducials(redshift, cut_name=bulk_or_tails)

    # ---- Latins (parallel) ----
    @typecheck
    def get_k_n_latins(
        redshift: float, 
        fiducial_vars: Optional[Float[np.ndarray, "1 r 1"]], 
        cut_name: Literal["bulk", "tails"]
    ) -> tuple[Float[np.ndarray, "n r k"], Float[np.ndarray, "n p"]]:
        rows = Parallel(n_jobs=N_JOBS, verbose=JOBLIB_VERBOSE)(
            delayed(_latin_cumulants_row_and_params)(
                folder, redshift, scale_numbers, bins, cuts_cdf_cut_idx_z_R, deltas_cdf_z_R, cut_name
            )
            for folder in latin_sorted_folders
        )
        # print("ROWS:", set(jax.tree.map(lambda a: a.shape, rows)) if rows is not None else rows)

        good = []
        bad_realisations = []
        for i, rp in enumerate(rows):
            if isinstance(rp, tuple) and rp[0] is not None:
                good.append(rp)
            else:
                bad_realisations.append(i)

        if good:
            latins_k_n = np.stack([r for (r, _) in good], axis=0)
            latin_parameters = np.stack([p for (_, p) in good], axis=0)
        else:
            latins_k_n = np.zeros((0, len(scale_numbers), N_CUMULANTS))
            latin_parameters = np.zeros((0, ALPHA.size))

        if FIDUCIAL_REDUCE and (fiducial_vars is not None) and len(good):
            latins_k_n = reduce(latins_k_n, fiducial_vars)

        return latins_k_n, latin_parameters

    latins_k_n, latin_parameters = get_k_n_latins(
        redshift, fiducial_vars=fiducial_vars, cut_name=bulk_or_tails
    )

    # ---- Derivatives (parallel) ----
    @typecheck
    def get_k_n_derivatives(
        redshift: float, 
        fiducial_vars: Optional[Float[np.ndarray, "1 r 1"]], 
        cut_name: Literal["bulk", "tails"]
    ) -> Float[np.ndarray, "n p r k"]:
        rows = Parallel(n_jobs=N_JOBS, verbose=JOBLIB_VERBOSE)(
            delayed(_derivative_cumulants_row)(
                folder, redshift, scale_numbers, bins, cuts_cdf_cut_idx_z_R, deltas_cdf_z_R, cut_name
            )
            for folder in derivative_sorted_folders
        )
        good = [r for r in rows if r is not None]
        if len(good):
            derivatives_k_n = np.stack(good, axis=0)  # (N, P, R, K)
        else:
            derivatives_k_n = np.zeros((0, ALPHA.size, len(scale_numbers), N_CUMULANTS))
        bad_realisations = [i for i, r in enumerate(rows) if r is None]
        print("bad (derivative):", bad_realisations)

        if FIDUCIAL_REDUCE and (fiducial_vars is not None) and len(good):
            # for i_p in range(ALPHA.size):
            #     _variances = (fiducial_vars ** (np.arange(N_CUMULANTS)[np.newaxis, np.newaxis, :] >= 3))
            #     derivatives_k_n[:, i_p, :, :] = derivatives_k_n[:, i_p, :, :] / _variances

            # The above line preserves your original intent via `reduce`;
            # but we reuse the same reduce() below for exact equivalence:
            for i_p in range(ALPHA.size):
                derivatives_k_n[:, i_p, :, :] = reduce(derivatives_k_n[:, i_p, :, :], fiducial_vars)

        return derivatives_k_n

    _derivatives_k_n = get_k_n_derivatives(
        redshift, fiducial_vars=fiducial_vars, cut_name=bulk_or_tails
    )

    # Plot
    try:
        out_dir = os.path.join(log_figs_dir, f"cumulants_grid_z{redshift}_{bulk_or_tails}")
        plot_cumulants_single_figure_grid(
            fiducials=fiducials_k_n,          # shape (Nf, R*K) or (Nf, R, K)
            latins=latins_k_n,                # shape (Nl, R*K) or (Nl, R, K)
            scale_numbers=np.asarray(scale_numbers),
            out_dir=out_dir,
            redshift=redshift,
            bulk_or_tails=bulk_or_tails,
            max_show=500,     # adjust to taste
            bins=30
        )

        logger.info(f"Saved separate cumulant plots to: {out_dir}")
    except Exception as e:
        logger.warning(f"Plotting cumulants failed: {e}")

    # ---- Postprocess (unchanged) ----
    fiducials_k_n = np.reshape(fiducials_k_n, (fiducials_k_n.shape[0], -1))
    latins_k_n = np.reshape(latins_k_n, (latins_k_n.shape[0], -1))

    derivatives_k_n = np.zeros((_derivatives_k_n.shape[0], ALPHA.size, len(scale_numbers) * N_CUMULANTS))
    for i_p in range(ALPHA.size):
        derivatives_k_n[:, i_p, :] = _derivatives_k_n[:, i_p, :, :].reshape(_derivatives_k_n.shape[0], -1)

    print("DATA:", jax.tree.map(lambda a: a.shape, (fiducials_k_n, latins_k_n, derivatives_k_n, latin_parameters)))

    return fiducials_k_n, latins_k_n, derivatives_k_n, latin_parameters


def get_calculated_cumulants_data(
    config: ConfigDict, 
    *, 
    use_means: bool = False,
    use_normalisations: bool = True,
    stack_means: bool = True,
    full_shape: bool = False,
    results_dir: Optional[str] = None
) -> Dataset:
    logger.info("Getting calculated cumulants for dataset={}".format(config.dataset_name))
    logger.info("Using bulk means..." if use_means else "Not using bulk means...")

    data_dir, *_ = get_save_and_load_dirs()

    use_means                    = use_means
    stack_means                  = stack_means
    use_normalisations           = use_normalisations

    bulk_or_tails = "tails" if full_shape else "bulk"

    dataset_identifier_str = "".join(
        [
            "_sobol",
            "_R" + "".join(map(str, config.scales)),
            "_m" + "".join(map(str, config.order_idx)),
            "_z" + str(config.redshift),
            "_reduced" if FIDUCIAL_REDUCE else "",
            "_with_means" if use_means else "",
            "_central",
            "_with_norms" if use_normalisations else "",
            "_with_means_stacked" if stack_means else "",
            "_deltacut" if DELTAS_CUT else ""
        ]
    )

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
            redshift=config.redshift
        )

        n_fiducial_moments, data_dim_moments = fiducial_moments_z_R.shape
        C_moments = np.cov(fiducial_moments_z_R, rowvar=False)

        corr_moments = jnp.corrcoef(fiducial_moments_z_R, rowvar=False)
        filename = os.path.join(log_figs_dir, "corr_coeff_moments_{}_z{}.png".format(bulk_or_tails, config.redshift))
        plt.figure()
        plt.title("Correlation matrix (moments) [{}]".format(bulk_or_tails))
        plt.imshow(corr_moments, cmap="coolwarm", vmin=-1., vmax=1.)
        plt.colorbar()
        plt.savefig(filename)
        plt.close()
        logger.debug("Saved correlation matrix (moments) figure at: \n\t{}".format(filename))

        H = hartlap(n_s=n_fiducial_moments, n_d=data_dim_moments)
        Cinv_moments = H * np.linalg.inv(C_moments)
        dmu_moments = np.mean(derivative_moments_z_R, axis=0)
        F_moments = np.linalg.multi_dot([dmu_moments, Cinv_moments, dmu_moments.T])
        Finv_moments = np.linalg.inv(F_moments)

        cumulants_dataset = Dataset(
            name=bulk_or_tails,
            alpha=jnp.asarray(ALPHA),
            lower=jnp.asarray(LOWER),
            upper=jnp.asarray(UPPER),
            parameter_strings=PARAMETER_STRINGS,
            Finv=jnp.asarray(Finv_moments),
            Cinv=jnp.asarray(Cinv_moments),
            C=jnp.asarray(C_moments),
            fiducial_data=jnp.asarray(fiducial_moments_z_R),
            data=jnp.asarray(latin_moments_z_R),
            parameters=jnp.asarray(latin_parameters),
            derivatives=jnp.asarray(derivative_moments_z_R)  
        )

        if not DATASET_TEST:
            np.savez(dataset_filename, **asdict(cumulants_dataset))
        logger.info("Saved dataset:\n\t{}".format(dataset_filename))
        return cumulants_dataset

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

    if config.linearised:
        logger.info("Using linearised dataset [replacing only hypercube]...")
        D0, D, Y = get_linearised_data(config, return_dataset) 
        return_dataset = replace(return_dataset, fiducial_data=D0, data=D, parameters=Y)

    if NON_GAUSSIAN_TEST:
        logger.info("Using non-Gaussian linear model dataset [replacing only hypercube]...")
        key = jr.key(config.seed)
        D, Y = get_non_gaussian_linear_model_data(config, return_dataset, key=key)
        return_dataset = replace(return_dataset, data=D, parameters=Y)

    if not config.linearised and not NON_GAUSSIAN_TEST:
        logger.info("Using Quijote dataset...")

    return return_dataset 


def get_calculated_pdfs_data(
    config: ConfigDict, 
    *, 
    full_shape: bool = False,
    results_dir: Optional[str] = None
) -> Dataset:

    logger.info("Getting PDFs for dataset={}".format(config.dataset_name))

    data_dir, *_ = get_save_and_load_dirs()
    bulk_or_tails = "tails" if full_shape else "bulk"

    dataset_identifier_str = "".join(
        [
            "_sobol",
            "_R" + "".join(map(str, config.scales)),
            "_z" + str(config.redshift),
            "_deltacut" if DELTAS_CUT else ""
        ]
    )

    dataset_filename = os.path.join(
        data_dir, "datasets/{}_pdf_dataset{}.npz".format(bulk_or_tails, dataset_identifier_str)
    )

    def generate_dataset() -> Dataset:
        # try:
        #     logger.info("Loading PDF dataset:\n\t{}".format(dataset_filename))

        #     dataset_dict = np.load(dataset_filename, allow_pickle=True) 

        #     pdf_dataset = Dataset(
        #         name="bulk_pdf",
        #         alpha=jnp.asarray(dataset_dict["alpha"]),
        #         lower=jnp.asarray(dataset_dict["lower"]),
        #         upper=jnp.asarray(dataset_dict["upper"]),
        #         parameter_strings=list(dataset_dict["parameter_strings"]),
        #         Finv=jnp.asarray(dataset_dict["Finv"]),
        #         Cinv=jnp.asarray(dataset_dict["Cinv"]),
        #         C=jnp.asarray(dataset_dict["C"]),
        #         fiducial_data=jnp.asarray(dataset_dict["fiducial_data"]),
        #         data=jnp.asarray(dataset_dict["data"]),
        #         parameters=jnp.asarray(dataset_dict["parameters"]),
        #         derivatives=jnp.asarray(dataset_dict["derivatives"]),
        #     )
        # except FileNotFoundError:
        #     (
        #         fiducial_pdfs_z_R, 
        #         latin_pdfs,
        #         latin_parameters,
        #         derivative_pdfs_z_R
        #     ) = get_fiducials_latins_derivatives_bulk_pdf(
        #         redshift=config.redshift,
        #         cdf_cut_lims=CUTS[bulk_or_tails],
        #         cut_name=bulk_or_tails,
        #         get_latins=True
        #     )

        #     n_fiducial_pdfs, data_dim_pdfs = fiducial_pdfs_z_R.shape 
        #     C_pdf = np.cov(fiducial_pdfs_z_R, rowvar=False) 
        #     H = hartlap(n_s=n_fiducial_pdfs, n_d=data_dim_pdfs) 
        #     Cinv_pdf = H * np.linalg.inv(C_pdf)
        #     dmu_pdfs = np.mean(derivative_pdfs_z_R, axis=0)
        #     F_pdf = jnp.linalg.multi_dot([dmu_pdfs, Cinv_pdf, dmu_pdfs.T])
        #     Finv_pdf = np.linalg.inv(F_pdf)

        #     pdf_dataset = Dataset(
        #         name="bulk_pdf",
        #         alpha=jnp.asarray(ALPHA),
        #         lower=jnp.asarray(LOWER),
        #         upper=jnp.asarray(UPPER),
        #         parameter_strings=PARAMETER_STRINGS,
        #         Finv=jnp.asarray(Finv_pdf),
        #         Cinv=jnp.asarray(Cinv_pdf),
        #         C=jnp.asarray(C_pdf),
        #         fiducial_data=jnp.asarray(fiducial_pdfs_z_R),
        #         data=latin_pdfs,
        #         parameters=latin_parameters,
        #         derivatives=jnp.asarray(derivative_pdfs_z_R)  
        #     )
            
        #     corr_pdf = np.corrcoef(fiducial_pdfs_z_R, rowvar=False) 
        #     filename = os.path.join(log_figs_dir, "corr_coeff_pdfs_{}.png".format(config.redshift))
        #     plt.figure()
        #     plt.title("Correlation matrix (PDFs) [{}]".format(bulk_or_tails))
        #     plt.imshow(corr_pdf, cmap="coolwarm", vmin=-1., vmax=1.)
        #     plt.colorbar()
        #     plt.savefig(filename)
        #     plt.close()
        #     logger.debug("Saved correlation matrix (PDFs) figure at: \n\t{}".format(filename))

        #     logger.info("Returning PDFs as dataset...")
        #     if not DATASET_TEST:
        #         np.savez(dataset_filename, **asdict(pdf_dataset))
        #     logger.info("Saved dataset:\n\t{}".format(dataset_filename))

        (
            fiducial_pdfs_z_R, 
            latin_pdfs,
            latin_parameters,
            derivative_pdfs_z_R
        ) = get_fiducials_latins_derivatives_bulk_pdf(
            redshift=config.redshift,
            cdf_cut_lims=CUTS[bulk_or_tails],
            cut_name=bulk_or_tails,
            get_latins=True
        )

        # print("BT: ", bulk_or_tails)

        n_fiducial_pdfs, data_dim_pdfs = fiducial_pdfs_z_R.shape 
        C_pdf = np.cov(fiducial_pdfs_z_R, rowvar=False) 
        H = hartlap(n_s=n_fiducial_pdfs, n_d=data_dim_pdfs) 
        Cinv_pdf = H * np.linalg.inv(C_pdf)
        dmu_pdfs = np.mean(derivative_pdfs_z_R, axis=0)
        F_pdf = jnp.linalg.multi_dot([dmu_pdfs, Cinv_pdf, dmu_pdfs.T])
        Finv_pdf = np.linalg.inv(F_pdf)

        pdf_dataset = Dataset(
            name="bulk_pdf",
            alpha=jnp.asarray(ALPHA),
            lower=jnp.asarray(LOWER),
            upper=jnp.asarray(UPPER),
            parameter_strings=PARAMETER_STRINGS,
            Finv=jnp.asarray(Finv_pdf),
            Cinv=jnp.asarray(Cinv_pdf),
            C=jnp.asarray(C_pdf),
            fiducial_data=jnp.asarray(fiducial_pdfs_z_R),
            data=latin_pdfs,
            parameters=latin_parameters,
            derivatives=jnp.asarray(derivative_pdfs_z_R)  
        )
        
        corr_pdf = np.corrcoef(fiducial_pdfs_z_R, rowvar=False) 
        filename = os.path.join(log_figs_dir, "corr_coeff_pdfs_{}.png".format(config.redshift))
        plt.figure()
        plt.title("Correlation matrix (PDFs) [{}]".format(bulk_or_tails))
        plt.imshow(corr_pdf, cmap="coolwarm", vmin=-1., vmax=1.)
        plt.colorbar()
        plt.savefig(filename)
        plt.close()
        logger.debug("Saved correlation matrix (PDFs) figure at: \n\t{}".format(filename))

        logger.info("Returning PDFs as dataset...")
        if not DATASET_TEST:
            np.savez(dataset_filename, **asdict(pdf_dataset))
        logger.info("Saved dataset:\n\t{}".format(dataset_filename))

        return pdf_dataset

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

    if config.linearised:
        logger.info("Using linearised dataset [replacing only hypercube]...")
        D0, D, Y = get_linearised_data(config, return_dataset) 
        return_dataset = replace(return_dataset, fiducial_data=D0, data=D, parameters=Y)

    if NON_GAUSSIAN_TEST:
        logger.info("Using non-Gaussian linear model dataset [replacing only hypercube]...")
        key = jr.key(config.seed)
        D, Y = get_non_gaussian_linear_model_data(config, return_dataset, key=key)
        return_dataset = replace(return_dataset, data=D, parameters=Y)

    if not config.linearised and not NON_GAUSSIAN_TEST:
        logger.info("Using Quijote dataset [PDFs]...")

    return return_dataset 


"""
    Dataset
"""

@dataclass
class SobolBulkCumulantsDataset:
    config: ConfigDict
    data: Dataset
    prior: Distribution
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
                config, 
                full_shape=False, 
                results_dir=results_dir
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

        self.prior = get_prior()
        self.compression_fn = None
        self.results_dir = results_dir

        logger.info("BULK CUMULANT DATASET {}".format("pdfs" if pdfs else ""))
        logger.info(">LINEARISED: {}".format(config.linearised))
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
        return PARAMETER_STRINGS

    def sample_prior(self, key: PRNGKeyArray, n: int) -> Float[Array, "n p"]:
        P = sample_prior(
            key, 
            n, 
            alpha=self.data.alpha, 
            lower=self.data.lower, 
            upper=self.data.upper
        )
        return P

    def get_compression_fn(self, train: bool = True):
        if self.compression_fn is None:
            key = jr.key(self.config.seed)
            fn = get_compression_fn(
                key, 
                self.config, 
                self.data, 
                train=train, 
                results_dir=self.results_dir
            )
            assert callable(fn), "Compression function returned is not callable"
            self.compression_fn = fn
        assert self.compression_fn is not None
        return self.compression_fn

    def get_datavector(self, key: PRNGKeyArray, n: int = 1) -> Float[Array, "... d"]:
        d = get_datavector(key, config=self.config, dataset=self.data, n=n)
        return d


@dataclass
class SobolTailsCumulantsDataset:
    config: ConfigDict
    data: Dataset
    prior: Distribution
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
                config, 
                full_shape=True, 
                results_dir=results_dir
            )
        else:
            self.data = get_calculated_cumulants_data(
                config, 
                use_means=config.use_means,
                use_normalisations=config.use_normalisations,
                stack_means=config.stack_means,
                full_shape=True,
                results_dir=results_dir
            )

        self.prior = get_prior()
        self.compression_fn = None
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
        return PARAMETER_STRINGS 

    def sample_prior(self, key: PRNGKeyArray, n: int) -> Float[Array, "n p"]:
        P = sample_prior(
            key, 
            n, 
            alpha=self.data.alpha, 
            lower=self.data.lower, 
            upper=self.data.upper
        )
        return P

    def get_compression_fn(self, train: bool = True):
        if self.compression_fn is None:
            key = jr.key(self.config.seed)
            fn = get_compression_fn(
                key, 
                self.config, 
                self.data, 
                train=train, 
                results_dir=self.results_dir
            )
            assert callable(fn), "Compression function returned is not callable"
            self.compression_fn = fn
        return self.compression_fn

    def get_datavector(self, key: PRNGKeyArray, n: int = 1) -> Float[Array, "... d"]:
        d = get_datavector(key, config=self.config, dataset=self.data, n=n)
        return d


class SobolBulkPDFsDataset(SobolBulkCumulantsDataset):
    def __init__(
        self,
        config: ConfigDict,
        *,
        results_dir: Optional[str] = None
    ):
        super().__init__(config, pdfs=True, results_dir=results_dir)


def load_multi_z_bulk_pdf_fisher_forecast(data_dir, args):
    def get_multi_z_bulk_pdf_fisher_forecast(args):
        F = np.zeros(())
        for redshift in args.redshifts: 
            config = bulk_cumulants_config(
                seed=args.seed, 
                redshift=redshift,
                linearised=args.linearised, 
                compression=args.compression,
                order_idx=args.order_idx,
                scales=args.scales,
                n_linear_sims=args.n_linear_sims,
                pre_train=args.pre_train,
            )

            logger.info("Using PDF dataset for bulk dataset. z={}".format(redshift))
            dataset = SobolBulkCumulantsDataset(config, pdfs=True)
            F_z = np.linalg.inv(dataset.data.Finv)
            F = F + F_z

        Finv = np.linalg.inv(F)
        return Finv

    identifier_str = "".join(
        [
            "_R" + "".join(map(str, args.scales)),
            "_z" + "".join(map(str, args.redshifts)),
            "_pdfs",
            "_deltacut" if DELTAS_CUT else ""
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
            if not DATASET_TEST:
                np.save(Finv_file_path, Finv_bulk_pdfs_all_z)
    else:
        Finv_bulk_pdfs_all_z = get_multi_z_bulk_pdf_fisher_forecast(args)

    logger.info("Finv bulk PDFs all z loaded from:\n\t{}".format(Finv_file_path))

    return Finv_bulk_pdfs_all_z
