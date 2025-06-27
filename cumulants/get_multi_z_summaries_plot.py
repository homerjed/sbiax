from typing import Optional, Literal, Callable
import time
import os

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import PRNGKeyArray, Float, Array, Scalar, jaxtyped

from beartype import beartype as typechecker
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from tqdm.auto import trange
from chainconsumer import ChainConsumer, Chain
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.distributions import Distribution

from sbiax.inference import nuts_sample
from sbiax.utils import make_df, marker

from configs.log import setup_module_logger, get_log_level
from configs.cumulants_configs import default_posterior_sampling
from configs.configs import (
    get_results_dir, 
    get_multi_z_posterior_filename,
    get_ndes_from_config
)
from configs.args import get_cumulants_sbi_args, get_cumulants_multi_z_args
from data.common import linearised_model, add_planck_information_to_Finv
from data.constants import get_base_posteriors_dir, get_save_and_load_dirs, get_target_idx, get_F_planck
from data.cumulants import get_parameter_strings
from data.pdfs import load_multi_z_bulk_pdf_fisher_forecast
from cumulants_ensemble import Ensemble, MultiEnsemble
from affine import affine_sample
from utils.utils import finite_samples_log_prob, get_datasets

typecheck = jaxtyped(typechecker=typechecker)

logger, log_figs_dir = setup_module_logger(__name__, level=get_log_level())

CompressionFn = Callable[[Float[Array, "d"], Float[Array, "p"]], Float[Array, "p"]]

jax.clear_caches()

cumulant_names = [
    r"$\langle \delta^2 \rangle_c$", 
    r"$\langle \delta^3 \rangle_c$",
    r"$\langle \delta^4 \rangle_c$"
] # ["var.", "skew.", "kurt."]

ix = get_target_idx()

PLOT_SUMMARIES = False

def default(v, d):
    return v if v is not None else d


"""
    Sample a posterior with a uniform physics-parameter prior
    and a likelihood function made from separate flows (ensembles) 
    trained on data from different redshifts. 

    This script takes a seed and loads the flows for each seed
    for each redshift.
    
    Datavector is made of one measurement at each redshift, 
    assumed to be independent between redshifts. 
    - Can use more than one datavector now, for scaling as a survey.
    - Fisher adds across redshifts

    Ensure scaling is switched on / off and EVERYTHING matches
    training configs.
    - Load configs for each flow based on redshift and experiment directory

    Meta all-redshift config `ensembles_bulk_pdfs_config` tells how to 
    sample the posterior made of the separate flows.
"""


@typecheck
def get_z_config_and_datavector(
    key: PRNGKeyArray, 
    seed: int,
    redshift: float, 
    linearised: bool = True, 
    order_idx: list[int] = [0, 1, 2],
    scales: list[float] = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
    freeze_parameters: bool = False,
    compression: Literal["linear", "nn"] = "linear",
    n_linear_sims: int = 10_000,
    pre_train: bool = False, 
    n_datavectors: int = 1,
    bulk_or_tails: Literal["tails", "bulk", "bulk_pdf"] = "tails",
    seed_datavector: Optional[int] = None, # Use fixed seed for config (ensemble, ...) and new seed for datavector
    *,
    verbose: bool = False
) -> tuple[
    Float[Array, "n p"],
    Float[Array, "n d"],
    Distribution,
    Float[Array, "p"],
    Float[Array, "p p"],
    Float[Array, "p p"],
    Float[Array, "p p"],
    Float[Array, "d d"],
    Float[Array, "d"],
    Float[Array, "p d"]
]:
    """ 
        Get config and datavector associated with a redshift z to load the experiment 
        for use in an ensemble of SBI likelihoods at different redshifts, for the bulk
        or tails datasets.
        - get config at redshift
        - load datasets, compression function and NDE ensemble. 
    """

    assert redshift in [0.0, 0.5, 1.0], "Redshift {} not in [0.0, 0.5, 1.0]".format(redshift)

    key_datavector, key_model = jr.split(key)

    # Change seed for datavector sampling without changing any other seed
    key_datavector = jr.key(int(time.time()))

    # Arguments for default SBI experiment
    sbi_args = get_cumulants_sbi_args(multi_z=True)

    sbi_args.seed              = seed
    sbi_args.redshift          = redshift # Set the sbi_args redshift to get datasets
    sbi_args.linearised        = linearised
    sbi_args.order_idx         = order_idx
    sbi_args.scales            = scales
    sbi_args.freeze_parameters = freeze_parameters
    sbi_args.compression       = compression
    sbi_args.n_linear_sims     = n_linear_sims
    sbi_args.pre_train         = pre_train

    # SBI configuration, main dataset and all datasets for given redshift
    config_z, cumulants_dataset, datasets = get_datasets(sbi_args) # Config and cumulants_dataset can be bulk ... etc

    logger.info("BULK/TAILS:".format(bulk_or_tails))
    logger.info("CONFIG:\n{}".format(config_z))

    # Sample datavector(s) at the fiducial parameters
    datavectors = cumulants_dataset.get_datavector(key_datavector, n=n_datavectors) # Generates linearised (or not) datavector 

    if datavectors.ndim == 1:
        datavectors = datavectors[jnp.newaxis, ...] # Add axis for vmapping...

    # Compressed datavectors at fiducial parameters
    x_ = jax.vmap(cumulants_dataset.compression_fn, in_axes=(0, None))(datavectors, cumulants_dataset.data.alpha) 

    # Debugging plots
    get_filename = lambda name: os.path.join(log_figs_dir, name)

    plt.figure()
    plt.title(
        "Cumulants covariance \n z={}, \n R={}, \n m={}]".format(
            redshift, 
            "".join(map(str, scales)),
            "".join(map(str, order_idx))
        )
    )
    print("cov max:", jnp.max(cumulants_dataset.data.C))
    plt.imshow(jnp.log(cumulants_dataset.data.C), cmap="coolwarm")
    plt.colorbar()
    plt.savefig(get_filename("multi_z_covariance_{}_{}.png".format(redshift, bulk_or_tails)))
    plt.close()

    plt.figure()
    plt.title(
        "Cumulants precision \n z={}, \n R={}, \n m={}]".format(
            redshift, 
            "".join(map(str, scales)),
            "".join(map(str, order_idx))
        )
    )
    _precision_ = jnp.linalg.inv(cumulants_dataset.data.C)
    assert jnp.all(jnp.isfinite(_precision_))
    plt.imshow(_precision_, cmap="coolwarm")
    plt.colorbar()
    plt.savefig(get_filename("multi_z_precision_{}_{}.png".format(redshift, bulk_or_tails)))
    plt.close()

    plt.figure()
    plt.title("Histogram; kurtoses")
    for R, _ in enumerate(scales):
        if R > 0:
            continue
        print(cumulants_dataset.data.fiducial_data.shape)
        ks = cumulants_dataset.data.fiducial_data[:, R * len(order_idx) : (R + 1) * len(order_idx)]
        kurts = ks[:, -1]
        print("var kurts:", np.var(kurts))
        print(kurts.shape, kurts.min(), kurts.max())
        plt.hist(
            kurts,
            bins="auto",
            histtype="step",
            density=True
        )
    # plt.xscale("log")
    # plt.yscale("log")
    plt.savefig(get_filename("kurtoses_hist_{}_{}.png".format(redshift, bulk_or_tails)))
    plt.close()

    return (
        x_, 
        datavectors,
        cumulants_dataset.prior, # Quijote prior (same for all z, only applied once with combined z-likelihoods)
        jnp.asarray(cumulants_dataset.data.alpha), 
        # Scale data and parameter covariances by number of measurements (these are not used in MLEs)
        jnp.asarray(datasets["bulk"].data.Finv) / n_datavectors, 
        jnp.asarray(datasets["tails"].data.Finv) / n_datavectors, 
        jnp.asarray(cumulants_dataset.data.Finv) / n_datavectors, 
        jnp.asarray(cumulants_dataset.data.C) / n_datavectors, 
        jnp.mean(cumulants_dataset.data.fiducial_data, axis=0), 
        jnp.mean(cumulants_dataset.data.derivatives, axis=0)
    )


def get_figs_dir(multi_z_args):
    # Save location for posterior plots

    # Where SBI's are saved (add on suffix for experiment details)
    posteriors_dir = get_base_posteriors_dir()

    parts = [
        "figs",
        "frozen" if multi_z_args.freeze_parameters else "nonfrozen",
        multi_z_args.bulk_or_tails,
        "linearised" if multi_z_args.linearised else "nonlinearised",
        multi_z_args.compression,
        "pretrain" if multi_z_args.pre_train else "nopretrain",
        "z={}_m={}".format( 
            "".join(map(str, multi_z_args.redshifts)),
            "".join(map(str, multi_z_args.order_idx))
        )
    ]
    path_str = "/".join(filter(None, parts)) + "/"

    figs_dir = os.path.join(posteriors_dir, path_str)
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir, exist_ok=True)

    logger.info("MULTI-Z FIGS_DIR:\n\t{}".format(figs_dir))

    return figs_dir 


if __name__ == "__main__":
    import time

    key = jr.key(int(time.time())) # Only for datavectors, split for each redshift, datavector and separate posterior

    multi_z_args = get_cumulants_multi_z_args()

    # Multi-z inference concerning the bulk or bulk + tails (just sampling args)
    sampling_config = default_posterior_sampling(config=None, no_config=True)

    # Get the bulk Fisher forecast for all redshifts 
    # but easier to load frozen or not since it autosaves...
    data_dir, _, _ = get_save_and_load_dirs()

    # Planck information added here if required
    Finv_bulk_pdfs_all_z = load_multi_z_bulk_pdf_fisher_forecast(data_dir, multi_z_args)
    Finv_bulk_pdfs_all_z = Finv_bulk_pdfs_all_z / multi_z_args.n_datavectors

    parameter_strings = get_parameter_strings()
    parameter_strings_ = [parameter_strings[_] for _ in ix]

    linear_str = "linearised" if multi_z_args.linearised else "nonlinearised"
    pretrain_str = "pretrain" if multi_z_args.pre_train else "nopretrain"

    figs_dir = get_figs_dir(multi_z_args)

    parameter_dim = 2 if multi_z_args.freeze_parameters else 5

    # Sample multiple posteriors across multiple redshifts
    for n_posterior in range(multi_z_args.n_posteriors_sample):

        key_n = jr.fold_in(key, n_posterior)

        # Loop over redshifts; loading ensembles and datavectors
        datavectors  = [] # I.i.d. datavectors, tuple'd for each redshift (plural measurements in tuple)
        x_s          = [] # Summaries of these datavectors
        ensembles    = [] # Ensembles of NDEs trained on simulations at each redshift
        covariances  = [] # Covariance matrices of simulations at each redshift
        derivatives_ = [] # Derivatives of theory model at each redshift 
        mus          = [] # Expectation model at each redshift 
        Finvs        = [] # Fisher parameter covariances at each redshift
        bulk_Finvs   = [] # Fisher parameter covariances at each redshift
        tails_Finvs  = [] # Fisher parameter covariances at each redshift
        F            = jnp.zeros((parameter_dim, parameter_dim)) # Add independent information from data at each redshift 
        F_bulk       = jnp.zeros((parameter_dim, parameter_dim)) # Add independent information from data at each redshift (for bulk)
        F_tails      = jnp.zeros((parameter_dim, parameter_dim)) # Add independent information from data at each redshift (for bulk)

        with trange(len(multi_z_args.redshifts), desc="Multi-z", colour="magenta") as bar:
            for _, (z, redshift) in zip(bar, enumerate(multi_z_args.redshifts)):

                print("@" * 80)
                print("Getting datavector(s) for redshift={}".format(redshift))

                key_z = jr.fold_in(key_n, z)

                # Load ensemble configuration, datavector/summary, prior, covariance, Fisher, derivatives
                (
                    x_z, # MLE[datavectors] at this redshift
                    datavector, # list[Array["n d"]]
                    prior, # Same prior for each z, only needed / used once
                    alpha, # Datavectors generated at these parameters for each redshift
                    bulk_Finv_z, # NOTE: This Finv not used for compression (scaled by n_datavectors)
                    tails_Finv_z, # NOTE: This Finv not used for compression (scaled by n_datavectors)
                    Finv_z, # NOTE: This Finv not used for compression (scaled by n_datavectors)
                    C, # NOTE: Scaled by n_datavectors
                    mu, 
                    derivatives
                ) = get_z_config_and_datavector(
                    key_z, 
                    seed=multi_z_args.seed,
                    order_idx=multi_z_args.order_idx,
                    scales=multi_z_args.scales,
                    linearised=multi_z_args.linearised, # NOTE: pre-train or not also...
                    compression=multi_z_args.compression,
                    redshift=redshift, 
                    n_datavectors=multi_z_args.n_datavectors,
                    pre_train=multi_z_args.pre_train,
                    bulk_or_tails=multi_z_args.bulk_or_tails,
                    freeze_parameters=multi_z_args.freeze_parameters,
                    seed_datavector=multi_z_args.seed_datavector,
                    verbose=multi_z_args.verbose
                ) 

                # Add Fisher information from redshift (independent; Limber)
                F += jnp.linalg.inv(Finv_z)
                F_bulk += jnp.linalg.inv(bulk_Finv_z)
                F_tails += jnp.linalg.inv(tails_Finv_z)

                derivatives_.append(derivatives)
                mus.append(mu)
                covariances.append(C)
                x_s.append(x_z)
                datavectors.append(datavector)
                Finvs.append(Finv_z)
                bulk_Finvs.append(bulk_Finv_z)
                tails_Finvs.append(tails_Finv_z)
                
                bar.set_postfix_str("z={}, n_posterior={}".format(redshift, n_posterior))

        assert len(x_s) == len(multi_z_args.redshifts)
        assert all([len(x_s[i]) == multi_z_args.n_datavectors for i in range(len(x_s))])

        # Combined Fisher information over all redshifts
        if multi_z_args.use_planck:
            F_planck = get_F_planck()

            # Only add Fisher from Planck once, to information accrued over all redshifts
            F = F + F_planck
            F_bulk = F_bulk + F_planck
            F_tails = F_tails + F_planck

            bulk_Finvs = [
               add_planck_information_to_Finv(Finv, use_planck=multi_z_args.use_planck)
               for Finv in bulk_Finvs
            ]
            tails_Finvs = [
               add_planck_information_to_Finv(Finv, use_planck=multi_z_args.use_planck)
               for Finv in bulk_Finvs
            ]

        Finv_all_z = jnp.linalg.inv(F) 
        bulk_Finv_all_z = jnp.linalg.inv(F_bulk) 
        tails_Finv_all_z = jnp.linalg.inv(F_tails) 

        if multi_z_args.verbose:
            # Plot Fisher forecasts
            for z, bulk_Finv_z, tails_Finv_z in zip(multi_z_args.redshifts, bulk_Finvs, tails_Finvs):
                c = ChainConsumer()
                c.add_chain(
                    Chain.from_covariance(
                        alpha,
                        tails_Finv_z,
                        columns=parameter_strings,
                        name=r"$F_{\Sigma^{-1}}$ z=" + str(z),
                        color="r",
                        shade_alpha=0.
                    )
                )
                c.add_chain(
                    Chain.from_covariance(
                        alpha,
                        bulk_Finv_z,
                        columns=parameter_strings,
                        name=r"$F_{\Sigma^{-1}}$ z=" + str(z) + " [bulk]",
                        color="b",
                        linestyle=":",
                        shade_alpha=0.
                    )
                )
                fig = c.plotter.plot()

                forecast_filename = os.path.join(
                        figs_dir if figs_dir is not None else "fisher_forecasts/", 
                        "fisher_forecast_z={}_R={}_m={}.png".format(
                            z,
                            "".join(map(str, multi_z_args.order_idx)),
                            "".join(map(str, multi_z_args.scales))
                        )
                    )

                plt.savefig(forecast_filename)
                plt.close()

                print("FISHER FORECAST SAVED AT: \n", forecast_filename)

            # Plot Fisher forecasts bulk / tails
            c = ChainConsumer()
            for z, bulk_Finv_z, tails_Finv_z in zip(multi_z_args.redshifts, bulk_Finvs, tails_Finvs):
                c.add_chain(
                    Chain.from_covariance(
                        alpha,
                        tails_Finv_z,
                        columns=parameter_strings,
                        name=r"$F_{\Sigma^{-1}}$ z=" + str(z),
                        color="r",
                        shade_alpha=0.
                    )
                )
            c.add_chain(
                Chain.from_covariance(
                    alpha,
                    tails_Finv_all_z,
                    columns=parameter_strings,
                    name=r"$F_{\Sigma^{-1}}$ z=all",
                    color="k",
                    shade_alpha=0.
                )
            )
            fig = c.plotter.plot()

            forecast_filename = os.path.join(
                    figs_dir if figs_dir is not None else "fisher_forecasts/", 
                    "fisher_forecast_tails_R={}_m={}.png".format(
                        z,
                        "".join(map(str, multi_z_args.order_idx)),
                        "".join(map(str, multi_z_args.scales))
                    )
                )

            plt.savefig(forecast_filename)
            plt.close()
            
            c = ChainConsumer()
            for z, bulk_Finv_z, tails_Finv_z in zip(multi_z_args.redshifts, bulk_Finvs, tails_Finvs):
                c.add_chain(
                    Chain.from_covariance(
                        alpha,
                        bulk_Finv_z,
                        columns=parameter_strings,
                        name=r"$F_{\Sigma^{-1}}$ z=" + str(z) + " [bulk]",
                        color="b",
                        shade_alpha=0.
                    )
                )
            c.add_chain(
                Chain.from_covariance(
                    alpha,
                    bulk_Finv_all_z,
                    columns=parameter_strings,
                    name=r"$F_{\Sigma^{-1}}$ z=all",
                    color="k",
                    shade_alpha=0.
                )
            )
            fig = c.plotter.plot()

            forecast_filename = os.path.join(
                    figs_dir if figs_dir is not None else "fisher_forecasts/", 
                    "fisher_forecast_bulk_R={}_m={}.png".format(
                        z,
                        "".join(map(str, multi_z_args.order_idx)),
                        "".join(map(str, multi_z_args.scales))
                    )
                )

            plt.savefig(forecast_filename)
            plt.close()

            print("FISHER FORECAST SAVED AT: \n", forecast_filename)


            # Multi z Fisher
            c = ChainConsumer()

            c.add_chain(
                Chain.from_covariance(
                    alpha,
                    tails_Finv_all_z,
                    columns=parameter_strings,
                    name=r"$F_{\Sigma^{-1}}$ (all z) [tails]",
                    color="r",
                    shade_alpha=0.
                )
            )
            c.add_chain(
                Chain.from_covariance(
                    alpha,
                    bulk_Finv_all_z,
                    columns=parameter_strings,
                    name=r"$F_{\Sigma^{-1}}$ (all z) [bulk]",
                    color="b",
                    shade_alpha=0.
                )
            )
            c.add_marker(
                location=marker(alpha, parameter_strings=parameter_strings),
                name=r"$\alpha$", 
                color="#7000b1"
            )
            fig = c.plotter.plot()

            forecast_filename = os.path.join(
                    figs_dir if figs_dir is not None else "fisher_forecasts/", 
                    "fisher_forecast_all_z_R={}_m={}.png".format(
                        # "".join(map(str, multi_z_args.redshifts)),
                        "".join(map(str, multi_z_args.scales)),
                        "".join(map(str, multi_z_args.order_idx))
                    )
                )

            plt.savefig(forecast_filename)
            plt.close()

            print("FISHER FORECAST SAVED AT: \n", forecast_filename)

            # Block diagonal covariance plot
            plt.figure()
            plt.imshow(block_diag(*covariances))
            plt.savefig("block_diag_covariance.png")
            plt.close()

        """
            Plot all summaries
        """
        c = ChainConsumer()

        colors = ["#7FFF00", "#FF8C00", "#C71585"]
        for i, redshift in enumerate([0.0, 0.5, 1.0]): 
            c.add_chain(
                Chain.from_covariance(
                    alpha[ix],
                    Finvs[i][ix, :][:, ix] * multi_z_args.n_datavectors, # Bulk or tails?
                    columns=parameter_strings_,
                    name=r"$F_{\Sigma^{-1}}$" + " (z={})".format(redshift),
                    color=colors[i],
                    linestyle=":",
                    shade_alpha=0.
                )
            )

            summaries = x_s[i]
            for s, summary in enumerate(summaries):
                c.add_marker(
                    location=marker(summary[ix], parameter_strings_), 
                    name="$x_z={}, {}$".format(redshift, s), 
                    color=colors[i]
                )

        c.add_chain(
            Chain.from_covariance(
                alpha[ix],
                Finv_all_z[ix, :][:, ix] * multi_z_args.n_datavectors, # Bulk or tails?
                columns=parameter_strings_,
                name=r"$F_{\Sigma^{-1}}$" + " (z=all)",
                color="k",
                linestyle=":",
                shade_alpha=0.
            )
        )

        fig = c.plotter.plot()
        fig.suptitle(
            r"{} SBI & $F_{{\Sigma}}^{{-1}}$".format("$k_n$") + "\n" +
            "{} z={},\n $n_s$={}, (pre-train $n_s$={}),\n R={} Mpc,\n $k_n$={}".format(
                    ("linearised" if multi_z_args.linearised else "non-linear") + "\n",
                    "[{}]".format(", ".join(map(str, multi_z_args.redshifts))),
                    multi_z_args.n_linear_sims if multi_z_args.linearised else 2000, 
                    multi_z_args.n_linear_sims if multi_z_args.pre_train else None,
                    "[{}]".format(", ".join(map(str, multi_z_args.scales))),
                    "[{}]".format(", ".join(map(str, [cumulant_names[_] for _ in multi_z_args.order_idx])))
                ),
            multialignment='center'
        )
        posterior_plot_filename = os.path.join(
            figs_dir, 
            "summaries_and_Finvs_{}_{}_{}_{}{}_mcmc.pdf".format(
                multi_z_args.seed, linear_str, pretrain_str, n_posterior,
                ("_" + str(multi_z_args.seed_datavector)) if multi_z_args.seed_datavector is not None else ""
            )
        )
        plt.savefig(posterior_plot_filename)
        plt.close()

        print("Summaries plot saved at:\n\t", posterior_plot_filename)
