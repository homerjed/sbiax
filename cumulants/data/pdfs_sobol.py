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

PRINT_FREQ = 500


"""
    Get cumulants of the bulk of the 3D matter PDF
    from the Sobol Latin Hypercube sequence
"""


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
    for i in range(1, n_bins_pdf):
        p_delta_i = pdf[i - 1]

        cdf[i] = cdf[i - 1] + p_delta_i * dbins[i - 1] 

    # Check CDF bounds
    assert np.isclose(cdf.min(), 0.) and np.isclose(cdf.max(), 1.), (
        "CDF: min={} max={}".format(cdf.min(), cdf.max())
    )

    cut_idx = np.where(
        (cdf >= p_value_min) & (cdf <= p_value_max)
    )[0] 

    return cdf, cut_idx

def get_pdf_filename_template(redshift, radius_index, realisation):
    filename = 'matter_PDF_linbins_BSQ_z_{:.2f}_extraRind_{}_{}.dat'.format(
        redshift, radius_index, str(realisation)
    )
    return filename


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

    import jax
    from tqdm.auto import trange

    p_value_min                  = config.p_value_min # Independent of choosing rho/delta for random variable of PDF
    p_value_max                  = config.p_value_max 

    n_cumulants = 5 # m_0, m_1, k_2, k_3, k_4

    n_plot = 2

    n_available_realisations = 20_000 #len(available_idx_realisations)

    cumulants = dict(
        bulk=np.zeros((n_available_realisations, len(config.redshifts), len(config.scales), n_cumulants)),
        tails=np.zeros((n_available_realisations, len(config.redshifts), len(config.scales), n_cumulants))
    )

    def iprint(i, string):
        if i < n_plot:
            print(string)

    bad_realisations = []

    for realisation_idx, realisation in zip(
        bar := trange(
            n_available_realisations, 
            colour="red" if cut_name == "tails" else "blue"
        ),
        np.arange(n_available_realisations) # available_idx_realisations
    ):

        if realisation_idx < n_plot:
            fig, axs = plt.subplots(2, 3, figsize=(8., 12.)[::-1])

        for i_z, z in enumerate(redshifts): 

            if realisation_idx < n_plot:
                ax_pdf, ax_cdf = axs[0, i_z], axs[1, i_z]

            for i_r, radius in enumerate(config.scales):

                # Grab physical scale index
                idx = np.squeeze(np.argwhere(scale_numbers == radius))

                # Filename of Sobol sequence PDF
                realisation_path = os.path.join(
                    data_dir, 
                    str(realisation) + "/", 
                    get_pdf_filename_template(
                        redshift=z, radius_index=idx, realisation=realisation
                    )
                ) 

                try:
                    # Load PDF bin centres and PDF in bins
                    bins, pdf = np.loadtxt(realisation_path).T

                    assert bins.size == pdf.size

                    dbins = bins[1:] - bins[:-1]

                    # Cut each pdf by its own CDF
                    cdf, cdf_cut_idx = get_cdf_of_pdf(pdf, dbins, cdf_cut_lim)

                    iprint(realisation_idx, "PDF sum / size: {} / {}".format(pdf.sum(), pdf.size))
                    iprint(realisation_idx, "CDF: {:.1f} {:.1f}".format(cdf.min(), cdf.max()))
                    iprint(realisation_idx, "CDF cut size: {}, dbins size: {}".format(cdf_cut_idx.size, dbins.size))

                    cut_pdf_normalised = pdf[cdf_cut_idx]
                    cut_bins = bins[cdf_cut_idx]
                    cut_dbins = dbins[cdf_cut_idx] 

                    iprint(realisation_idx, "{}".format(
                        jax.tree.map(lambda a: a.shape, (cut_pdf_normalised, cut_bins, cut_dbins)))
                    )

                    cumulants_R_z = cut_pdf_to_cumulants(
                        cut_pdf_normalised, 
                        cut_bins, 
                        cut_dbins, 
                        prob_norm=cdf_cut_lim[1] - cdf_cut_lim[0]
                    )
                    iprint(realisation_idx, "cumulants: {}".format(cumulants_R_z))

                    cumulants[cut_name][realisation_idx, i_z, i_r] = cumulants_R_z

                    if realisation_idx < n_plot:
                        ax_pdf.set_title("Redshift={}".format(z))

                        ax_pdf.loglog(
                            bins, 
                            pdf, 
                            marker=".", 
                            linestyle="", 
                            color="k",
                            zorder=0
                        )
                        ax_pdf.loglog(
                            cut_bins, 
                            cut_pdf_normalised, 
                            marker=".", 
                            linestyle="", 
                            zorder=1,
                            label="R={:.1f} Mpc/h".format(config.scales[i_r])
                        )
                        ax_pdf.legend(frameon=False)

                        # ax_cdf.set_title("Redshift={}".format(z))
                        ax_cdf.semilogx(
                            bins, 
                            cdf, 
                            marker=".", 
                            color="k",
                            linestyle="", 
                            zorder=0
                            # label="R={:.1f} Mpc/h".format(scales[i_r])
                        )
                        ax_cdf.semilogx(
                            cut_bins, 
                            cdf[cdf_cut_idx], 
                            marker=".", 
                            linestyle="", 
                            label="R={:.1f} Mpc/h".format(config.scales[i_r]),
                            zorder=1
                        )
                        ax_cdf.legend(frameon=False)

                        # print(cut_pdf_normalised.shape) # Take min of shape as default shape for the redshift

                except FileNotFoundError as e:
                    print(e)
                    bad_realisations.append(realisation_idx)

        bar.set_description("cut={}".format(cut_name))

        if realisation_idx < n_plot:
            plt.savefig(
                os.path.join(log_figs_dir, "LH_PDF_{}_{}.png".format(realisation_idx, cut_name)), 
                bbox_inches="tight"
            )
            plt.close()

    # print("Number of bad realisations:", len(bad_realisations))