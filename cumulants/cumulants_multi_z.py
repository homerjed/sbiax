from typing import Optional, Literal, Callable
import logging
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
from data.common import linearised_model, add_planck_information_to_Finv, get_prior_from_args
from data.constants import get_base_posteriors_dir, get_save_and_load_dirs, get_target_idx, get_F_planck, get_alpha_and_parameter_strings, LOWER, UPPER
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
    Ensemble,
    Float[Array, "n p"],
    Float[Array, "n d"],
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
    # if seed_datavector is not None:
    #     key_datavector = jr.fold_in(key_datavector, seed_datavector)
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
    # for key, value in hyperparameters.items():
    #     if hasattr(config, key):
    #         setattr(config, key, value)

    # SBI configuration, main dataset and all datasets for given redshift
    config_z, cumulants_dataset, datasets = get_datasets(sbi_args) # Config and cumulants_dataset can be bulk ... etc

    logger.info("BULK/TAILS: {}".format(bulk_or_tails))
    logger.info("CONFIG:\n{}".format(config_z))

    # Sample datavector(s) at the fiducial parameters
    datavectors = cumulants_dataset.get_datavector(key_datavector, n=n_datavectors) # Generates linearised (or not) datavector 

    if n_datavectors == 1:
        datavectors = datavectors[jnp.newaxis, :] # Add axis for vmapping...

    # Compressed datavectors at fiducial parameters
    x_ = jax.vmap(cumulants_dataset.compression_fn, in_axes=(0, None))(datavectors, cumulants_dataset.data.alpha) 

    # Get NDEs
    ndes = get_ndes_from_config(
        config_z, 
        cumulants_dataset,
        event_dim=cumulants_dataset.data.alpha.size, 
        use_scalers=config_z.use_scalers,
        key=key_model
    )

    # Ensemble of NDEs
    ensemble = Ensemble(ndes)

    # Load ensemble
    ensemble_path = os.path.join(get_results_dir(config_z, args=sbi_args), "ensemble.eqx")
    ensemble = eqx.tree_deserialise_leaves(ensemble_path, ensemble)

    logger.info("Loaded ensemble from:\n\t{}".format(ensemble_path))
    logger.info("Ensemble weights:\n\t{}".format(ensemble.weights))

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Debugging plots
    if 1:
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
        stddev = np.sqrt(np.diag(cumulants_dataset.data.C))
        plt.imshow(cumulants_dataset.data.C / np.outer(stddev, stddev), cmap="coolwarm")
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

        datavector = cumulants_dataset.get_datavector(key_datavector, n=n_datavectors)

        if n_datavectors == 1:
            datavector = datavector[jnp.newaxis, :]

        logger.debug("datavector {} \n {}".format(datavector.shape, datavector))

        _x_ = jax.vmap(cumulants_dataset.compression_fn, in_axes=(0, None))(datavector, cumulants_dataset.data.alpha)

        logger.debug("compressed datavector {} \n {} {}".format(x_.shape, _x_, cumulants_dataset.data.alpha))

        prior = get_prior_from_args(sbi_args)

        log_prob_fn = ensemble.ensemble_log_prob_fn(_x_, prior)

        key_state, key_sample = jr.split(jr.key(int(time.time())))
        state = jr.multivariate_normal(
            key_state, 
            cumulants_dataset.data.alpha, 
            add_planck_information_to_Finv(
                cumulants_dataset.data.Finv / n_datavectors, use_planck=False
            ), 
            (2 * config_z.n_walkers,)
        )
        state = jnp.clip(state, LOWER, UPPER)
        # state = parameter_prior.sample(seed=key_state, sample_shape=(2 * config.n_walkers,))

        parameter_strings = cumulants_dataset.get_parameter_strings()

        samples, weights = affine_sample(
            key_sample, 
            log_prob=log_prob_fn,
            n_walkers=config_z.n_walkers, 
            n_steps=config_z.n_steps + config_z.burn, 
            burn=config_z.burn, 
            current_state=state,
            description="Sampling",
            show_tqdm=True
        )

        samples_log_prob = jax.vmap(log_prob_fn)(samples)
        samples_log_prob = finite_samples_log_prob(samples_log_prob) 

        # samples, samples_log_prob = nuts_sample(
        #     key_sample, log_prob_fn=log_prob_fn, prior=prior
        # )
        # samples_log_prob = finite_samples_log_prob(samples_log_prob) 

        posterior_df = make_df(
            samples.squeeze(), 
            samples_log_prob.squeeze(), 
            parameter_strings=parameter_strings
        )

        c = ChainConsumer()
        c.add_chain(
            Chain.from_covariance(
                cumulants_dataset.data.alpha,
                add_planck_information_to_Finv(
                    cumulants_dataset.data.Finv / n_datavectors, use_planck=False
                ), 
                columns=parameter_strings,
                name=r"$F_{\Sigma^{-1}}$",
                color="k",
                linestyle=":",
                shade_alpha=0.
            )
        )
        c.add_chain(
            Chain.from_covariance(
                cumulants_dataset.data.alpha,
                add_planck_information_to_Finv(
                    cumulants_dataset.data.Finv, use_planck=False
                ), 
                columns=parameter_strings,
                name=r"$F_{\Sigma^{-1}}$ (unscaled)",
                color="k",
                linestyle="--",
                shade_alpha=0.
            )
        )
        c.add_chain(
            Chain.from_covariance(
                cumulants_dataset.data.alpha,
                add_planck_information_to_Finv(
                    datasets["bulk"].data.Finv / n_datavectors, use_planck=False
                ), 
                columns=parameter_strings,
                name=r"$F_{\Sigma^{-1}}$" + " {}".format("$k_n$[bulk]"),
                color="b",
                linestyle=":",
                shade_alpha=0.
            )
        )
        c.add_chain(
            Chain.from_covariance(
                cumulants_dataset.data.alpha,
                add_planck_information_to_Finv(
                    datasets["bulk_pdf"].data.Finv / n_datavectors, use_planck=False
                ), 
                columns=parameter_strings,
                name=r"$F_{\Sigma^{-1}}$" + " {}".format("PDF[bulk]"),
                color="g",
                linestyle=":",
                shade_alpha=0.
            )
        )
        c.add_chain(
            Chain(
                samples=posterior_df, 
                name="SBI[{}]".format(bulk_or_tails), 
                color="r" if bulk_or_tails == "tails" else "b"
            )
        )
        for i, __x__ in enumerate(_x_):
            c.add_marker(
                location=marker(__x__, parameter_strings=parameter_strings),
                name=str(i), 
                color="r" if bulk_or_tails == "tails" else "b"
            )
        c.add_marker(
            location=marker(cumulants_dataset.data.alpha, parameter_strings=parameter_strings),
            name=r"$\alpha$", 
            color="#7600bc"
        )
        fig = c.plotter.plot()

        filename = os.path.join(log_figs_dir, "posterior_affine_z={}_{}.pdf".format(redshift, bulk_or_tails))

        plt.savefig(filename)
        plt.close()

        print("DEBUG POSTERIOR SAVED AT:\n\t", filename)

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    return (
        ensemble, 
        x_, 
        datavectors,
        # Scale data and parameter covariances by number of measurements (these are not used in MLEs)
        jnp.asarray(datasets["bulk"].data.Finv) / n_datavectors, 
        jnp.asarray(datasets["tails"].data.Finv) / n_datavectors, 
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

    data_dir, _, _ = get_save_and_load_dirs()

    multi_z_args = get_cumulants_multi_z_args()

    alpha, parameter_strings = get_alpha_and_parameter_strings()
    alpha = jnp.asarray(alpha)
    parameter_strings_ = [parameter_strings[_] for _ in ix]

    prior = get_prior_from_args(multi_z_args)

    # Multi-z inference concerning the bulk or bulk + tails (just sampling args)
    sampling_config = default_posterior_sampling(config=None, no_config=True)

    # Get the bulk Fisher forecast for all redshifts 
    # but easier to load frozen or not since it autosaves...
    # Planck information added here if required
    Finv_bulk_pdfs_all_z = load_multi_z_bulk_pdf_fisher_forecast(data_dir, multi_z_args)
    Finv_bulk_pdfs_all_z = Finv_bulk_pdfs_all_z / multi_z_args.n_datavectors

    linear_str = "linearised" if multi_z_args.linearised else "nonlinearised"
    pretrain_str = "pretrain" if multi_z_args.pre_train else "nopretrain"

    figs_dir = get_figs_dir(multi_z_args)

    parameter_dim = 2 if multi_z_args.freeze_parameters else 5

    # Sample multiple posteriors across multiple redshifts
    for n_posterior in range(multi_z_args.n_posteriors_sample):

        key_n = jr.fold_in(key, n_posterior)

        # Loop over redshifts; loading ensembles and datavectors
        # datavectors  = [] # I.i.d. datavectors, tuple'd for each redshift (plural measurements in tuple)
        x_s          = [] # Summaries of these datavectors
        ensembles    = [] # Ensembles of NDEs trained on simulations at each redshift
        covariances  = [] # Covariance matrices of simulations at each redshift
        derivatives_ = [] # Derivatives of theory model at each redshift 
        mus          = [] # Expectation model at each redshift 
        bulk_Finvs   = [] # Fisher parameter covariances at each redshift
        tails_Finvs  = [] # Fisher parameter covariances at each redshift
        F_bulk       = jnp.zeros((parameter_dim, parameter_dim)) # Add independent information from data at each redshift (for bulk)
        F_tails      = jnp.zeros((parameter_dim, parameter_dim)) # Add independent information from data at each redshift (for bulk)

        with trange(len(multi_z_args.redshifts), desc="Multi-z", colour="magenta") as bar:
            for _, (z, redshift) in zip(bar, enumerate(multi_z_args.redshifts)):

                print("@" * 80)
                print("Getting n={} datavector(s) for redshift={}".format(multi_z_args.n_datavectors, redshift))

                key_z = jr.fold_in(key_n, z)

                # Load ensemble configuration, datavector/summary, prior, covariance, Fisher, derivatives
                (
                    ensemble, 
                    x_z, # MLE[datavectors] at this redshift
                    datavector, # list[Array["n d"]]
                    bulk_Finv_z, # NOTE: This Finv not used for compression (scaled by n_datavectors)
                    tails_Finv_z, # NOTE: This Finv not used for compression (scaled by n_datavectors)
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
                F_bulk += jnp.linalg.inv(bulk_Finv_z)
                F_tails += jnp.linalg.inv(tails_Finv_z)

                derivatives_.append(derivatives)
                mus.append(mu)
                covariances.append(C)
                x_s.append(x_z)
                # datavectors.append(datavector)
                bulk_Finvs.append(bulk_Finv_z)
                tails_Finvs.append(tails_Finv_z)
                ensembles.append(ensemble)      
                
                bar.set_postfix_str("z={}, n_posterior={}".format(redshift, n_posterior))

        assert len(x_s) == len(multi_z_args.redshifts)
        assert all([len(x_s[i]) == multi_z_args.n_datavectors for i in range(len(x_s))])

        # Multi-redshift ensemble of individual ensembles at each redshift
        multi_ensemble = MultiEnsemble(ensembles, prior=prior) 

        # Combined Fisher information over all redshifts
        if multi_z_args.use_planck:
            F_planck = get_F_planck()

            # Only add Fisher from Planck once, to information accrued over all redshifts
            F_bulk = F_bulk + F_planck
            F_tails = F_tails + F_planck

            bulk_Finvs = [
               add_planck_information_to_Finv(_Finv, use_planck=multi_z_args.use_planck)
               for _Finv in bulk_Finvs
            ]
            tails_Finvs = [
               add_planck_information_to_Finv(_Finv, use_planck=multi_z_args.use_planck)
               for _Finv in tails_Finvs
            ]

        bulk_Finv_all_z = jnp.linalg.inv(F_bulk) 
        tails_Finv_all_z = jnp.linalg.inv(F_tails) 

        # Choose 'main' Finv for sampling
        if multi_z_args.bulk_or_tails == "bulk":
            Finv_all_z = bulk_Finv_all_z
            Finvs = bulk_Finvs
        else:
            Finv_all_z = tails_Finv_all_z
            Finvs = tails_Finvs

        if 1:
            # Plot Fisher forecasts over all redshifts
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
                        log_figs_dir, 
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
                    log_figs_dir, 
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
                    log_figs_dir, 
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
                    log_figs_dir, 
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
            plt.imshow(block_diag(*covariances), cmap="coolwarm")
            plt.colorbar()
            plt.savefig(os.path.join(log_figs_dir, "block_diag_covariance.png"))
            plt.close()

        print("Sampling posterior {} (all redshifts, datavectors)".format(n_posterior))

        # Sample the multiple-redshift-ensemble posterior
        key_sample, key_state = jr.split(jr.fold_in(key, n_posterior))

        # Sample posterior across multiple redshifts
        log_prob_fn = multi_ensemble.get_multi_ensemble_log_prob_fn(x_s)

        # samples, samples_log_prob = nuts_sample(
        #     key_sample, 
        #     log_prob_fn=log_prob_fn, # NOTE: is it right to pass list of datavectors not the MLE above?
        #     prior=prior
        # )

        state = jr.multivariate_normal(
            key_state, alpha, Finv_all_z, (2 * sampling_config.n_walkers,) 
        )

        samples, weights = affine_sample(
            key_sample, 
            log_prob=log_prob_fn,
            n_walkers=sampling_config.n_walkers, 
            n_steps=sampling_config.n_steps + sampling_config.burn, 
            burn=sampling_config.burn, 
            current_state=state,
            description="Sampling",
            show_tqdm=True # multi_z_args.use_tqdm
        )

        alpha_log_prob = log_prob_fn(jnp.asarray(alpha))
        samples_log_prob = jax.vmap(log_prob_fn)(samples)
        samples_log_prob = finite_samples_log_prob(samples_log_prob)

        summaries_all_z = np.stack(x_s, axis=0) # NOTE: (n_z, n_datavectors, n_x) ?
        logger.info("SUMMARIES ALL Z:{}".format(summaries_all_z.shape))

        # Save posterior, Fisher and summary
        posterior_filename = get_multi_z_posterior_filename(multi_z_args)
        np.savez(
            posterior_filename,
            samples=samples, 
            samples_log_prob=samples_log_prob,
            Finv=Finv_all_z,
            # datavectors=datavectors,
            summaries=summaries_all_z, 
        )

        print("MEAN FISHER VARIANCE SIGMA_8 BULK/TAILS:", np.sqrt(np.diag(Finv_all_z))[4])
        print("MEAN FISHER VARIANCE SIGMA_8 BULK:", np.sqrt(np.diag(bulk_Finv_all_z))[4])
        print("MEAN FISHER VARIANCE SIGMA_8 TAILS:", np.sqrt(np.diag(bulk_Finv_all_z))[4])

        print("MULTI-Z POSTERIOR FILENAME:\n", posterior_filename)

        """
            MCMC sample with linear model
        """

        # Don't rescale by Finv for MCMC
        covariances_mcmc = [_C * multi_z_args.n_datavectors for _C in covariances]
        precisions_mcmc = [jnp.linalg.inv(_C) for _C in covariances_mcmc]
        Finvs_mcmc = [_Finv * multi_z_args.n_datavectors for _Finv in Finvs]


        @typecheck
        @eqx.filter_jit
        def mcmc_log_prob_fn_compressed(pi: Float[Array, "p"]) -> Scalar: 
            # Compressed data likelihood

            @typecheck
            def _log_prob_fn_z(
                pi_: Float[Array, "n d"], Finv: Float[Array, "p p"]
            ) -> Float[Array, "n"]:

                # Assumes all datavectors drawn at alpha
                def _posterior(_pi_, pi):
                    p = tfd.MultivariateNormalFullCovariance(loc=pi, covariance_matrix=Finv) 
                    return p.log_prob(_pi_)

                return jax.vmap(_posterior, in_axes=(0, None))(pi_, pi) # Vmap over multiple summaries

            # Tree map over lists of ingredients for each redshift
            Ls = jax.tree.map(lambda d, Finv: _log_prob_fn_z(d, Finv), x_s, Finvs_mcmc)

            return jnp.sum(jnp.asarray(Ls)) # NOTE: Correct sum?


        state = jr.multivariate_normal(
            key_state, alpha, Finv_all_z, (2 * sampling_config.n_walkers,) 
        )

        mcmc_samples, mcmc_weights = affine_sample(
            key_sample, 
            log_prob=lambda theta: mcmc_log_prob_fn_compressed(pi=theta),
            n_walkers=sampling_config.n_walkers, 
            n_steps=sampling_config.n_steps + sampling_config.burn, 
            burn=sampling_config.burn, 
            current_state=state,
            description="Sampling (MCMC)",
            show_tqdm=True # multi_z_args.use_tqdm
        )
        mcmc_samples_log_prob = jax.vmap(mcmc_log_prob_fn_compressed)(mcmc_samples)

        """
            Full posterior
        """

        if not multi_z_args.freeze_parameters:
            c = ChainConsumer() 
            c.add_chain(
                Chain.from_covariance(
                    alpha,
                    tails_Finv_all_z, # NOTE: Get multi redshift Fisher matrix, use a multi-inference config
                    columns=parameter_strings,
                    name=r"$F_{\Sigma^{-1}}$ (all z) $k_n$[tails]",
                    color="r",
                    linestyle=":",
                    shade_alpha=0.
                )
            )
            c.add_chain(
                Chain.from_covariance(
                    alpha,
                    bulk_Finv_all_z,
                    columns=parameter_strings,
                    name=r"$F_{\Sigma^{-1}}$ (all z) $k_n$[bulk]",
                    color="b",
                    linestyle=":",
                    shade_alpha=0.
                )
            )
            c.add_chain(
                Chain.from_covariance(
                    alpha,
                    Finv_bulk_pdfs_all_z,
                    columns=parameter_strings,
                    name=r"$F_{\Sigma^{-1}}$ (all z) PDF[bulk]",
                    color="g",
                    linestyle=":",
                    shade_alpha=0.
                )
            )
            posterior_df = make_df(
                samples, samples_log_prob, parameter_strings=parameter_strings
            )
            c.add_chain(
                Chain(
                    samples=posterior_df, 
                    name="SBI[{}]".format(multi_z_args.bulk_or_tails), 
                    color="r" if multi_z_args.bulk_or_tails == "tails" else "b"
                )
            )
            if PLOT_SUMMARIES:
                # If using multiple datavectors, plot them individually
                # if x_.ndim > 1:
                #     for i, _x_ in enumerate(x_):
                #         c.add_marker(
                #             location=marker(_x_, parameter_strings), 
                #             name=r"$\hat{x}$ " + str(i), 
                #             color="r" if multi_z_args.bulk_or_tails == "tails" else "b"
                #         )
                # else:
                #     c.add_marker(
                #         location=marker(x_, parameter_strings), 
                #         name=r"$\hat{x}$", 
                #         color="r" if multi_z_args.bulk_or_tails == "tails" else "b"
                #     )

                # c.add_marker(
                #     location=marker(np.mean(x_, axis=0) if x_.ndim > 1 else x_, parameter_strings), 
                #     name=r"$\bar{x}$", 
                #     color="r" if multi_z_args.bulk_or_tails == "tails" else "b",
                #     marker_style="x"
                # )


                # c.add_marker(
                #     location=marker(
                #         np.mean(
                #             np.asarray([np.mean(x_, axis=0) for x_ in x_s]), 
                #             axis=0
                #         ), 
                #         parameter_strings
                #     ), 
                #     name=r"$\bar{x}$ fresh", 
                #     color="r" if multi_z_args.bulk_or_tails == "tails" else "b",
                #     marker_style="x"
                # )

                mu_x_ = np.mean(np.mean(summaries_all_z, axis=0), axis=0)[ix]
                print("MU_X_", mu_x_.shape) # (3, 5) ; 3 redshifts
                c.add_marker(
                    location=marker(mu_x_, parameter_strings_), 
                    name=r"$\bar{x}$ fresh", 
                    color="r" if multi_z_args.bulk_or_tails == "tails" else "b",
                    marker_style="s"
                )
            c.add_marker(
                location=marker(alpha, parameter_strings), 
                name=r"$\alpha$", 
                color="#7600bc"
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
            plt.savefig(
                os.path.join(
                    figs_dir, 
                    "multi_ensemble_posterior_cumulants_{}_{}_{}_{}{}.pdf".format(
                        multi_z_args.seed, linear_str, pretrain_str, n_posterior,
                        ("_" + str(multi_z_args.seed_datavector)) if multi_z_args.seed_datavector is not None else ""
                    )
                )
            )
            plt.close()

        """
            Marginalised posterior with MCMC
            - Marginalise over all but Om, s8
        """

        c = ChainConsumer()
        c.add_chain(
            Chain.from_covariance(
                alpha[ix],
                Finv_all_z[ix, :][:, ix],
                columns=parameter_strings_,
                name=r"$F_{\Sigma^{-1}}$ $k_n$[tails]",
                color="r",
                linestyle=":",
                shade_alpha=0.
            )
        )
        c.add_chain(
            Chain.from_covariance(
                alpha[ix],
                bulk_Finv_all_z[ix, :][:, ix],
                columns=parameter_strings_,
                name=r"$F_{\Sigma^{-1}}$ (all z) $k_n$[bulk]",
                color="b",
                linestyle=":",
                shade_alpha=0.
            )
        )
        c.add_chain(
            Chain.from_covariance(
                alpha[ix],
                Finv_bulk_pdfs_all_z[ix, :][:, ix],
                columns=parameter_strings_,
                name=r"$F_{\Sigma^{-1}}$ (all z) PDF[bulk]",
                color="g",
                linestyle=":",
                shade_alpha=0.
            )
        )

        posterior_df = make_df(
            samples[:, ix], samples_log_prob, parameter_strings=parameter_strings_
        )
        c.add_chain(
            Chain(
                samples=posterior_df, 
                name="SBI[{}]".format(multi_z_args.bulk_or_tails), 
                color="r" if multi_z_args.bulk_or_tails == "tails" else "b"
            )
        )
        posterior_df = make_df(
            mcmc_samples[:, ix], mcmc_samples_log_prob, parameter_strings=parameter_strings_
        )
        c.add_chain(
            Chain(
                samples=posterior_df, 
                name="MCMC[{}]".format(multi_z_args.bulk_or_tails), 
                color="#9867C5"
            )
        )
        # if x_.ndim > 1:
        #     for i, _x_ in enumerate(x_):
        #         c.add_marker(
        #             location=marker(_x_[ix], parameter_strings_), 
        #             name=r"$\hat{x}$ " + str(i), 
        #             color="r" if multi_z_args.bulk_or_tails == "tails" else "b"
        #         )
        # else:
        #     c.add_marker(
        #         location=marker(x_[ix], parameter_strings_), 
        #         name=r"$\hat{x}$", 
        #         color="r" if multi_z_args.bulk_or_tails == "tails" else "b"
        # # )
        # c.add_marker(
        #     location=marker(np.mean(x_, axis=0)[ix] if x_.ndim > 1 else x_[ix], parameter_strings_), 
        #     name=r"$\bar{x}$", 
        #     color="r" if multi_z_args.bulk_or_tails == "tails" else "b",
        #     marker_style="x"
        # )

        if PLOT_SUMMARIES:
            # mu_x_ = np.mean(np.mean(summaries_all_z, axis=0), axis=0)[ix]
            # print("MU_X_", mu_x_.shape) # (3, 5) ; 3 redshifts
            # c.add_marker(
            #     location=marker(mu_x_, parameter_strings_), 
            #     name=r"$\bar{x}$ fresh", 
            #     color="r" if multi_z_args.bulk_or_tails == "tails" else "b",
            #     marker_style="s"
            # )
            for z in range(len(multi_z_args.redshifts)):
                x_s_z = np.mean(summaries_all_z[:, z, :], axis=0)
                c.add_marker(
                    location=marker(x_s_z, parameter_strings_), 
                    name=r"$\bar{x}$ z={}".format(multi_z_args.redshifts[z]), 
                    color="r" if multi_z_args.bulk_or_tails == "tails" else "b",
                    marker_style="x"
                )
        c.add_marker(
            location=marker(alpha[ix], parameter_strings_), 
            name=r"$\alpha$", 
            color="#7600bc"
        )

        # Plot summaries
        # colors = ["#7FFF00", "#FF8C00", "#C71585"]
        # for i, redshift in enumerate([0.0, 0.5, 1.0]): 
        #     summaries = x_s[i]
        #     for s, summary in enumerate(summaries):
        #         c.add_marker(
        #             location=marker(summary[ix], parameter_strings_), 
        #             name="$x_z={}, {}$".format(redshift, s), 
        #             color=colors[i]
        #         )

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
            "multi_ensemble_posterior_marginalised_cumulants_{}_{}_{}_{}{}_mcmc.pdf".format(
                multi_z_args.seed, 
                linear_str, 
                pretrain_str, 
                n_posterior,
                ("_" + str(multi_z_args.seed_datavector)) if multi_z_args.seed_datavector is not None else ""
            )
        )
        plt.savefig(posterior_plot_filename)
        plt.close()

        print("MULTI-Z POSTERIOR PLOT FILENAME (MCMC):\n", posterior_plot_filename)

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

# Precision of weights bug
# ensemble = eqx.tree_at(
#     lambda e: e.weights, 
#     ensemble, 
#     ensemble.weights.squeeze().astype(jnp.int32)
# )

# @typecheck
# @eqx.filter_jit
# def mcmc_log_prob_fn(pi: Float[Array, "p"]) -> Scalar: 
#     # Full uncompressed data likelihood

#     def _log_prob_fn_z(        
#         d: Float[Array, "n d"], 
#         pi: Float[Array, "p"], 
#         mu: Float[Array, "d"], 
#         dmu: Float[Array, "p d"], 
#         C: Float[Array, "d d"],         
#     ) -> Float[Array, "n"]:
#         mu_pi = linearised_model(alpha, pi, mu, dmu)
#         p = tfd.MultivariateNormalFullCovariance(loc=mu_pi, covariance_matrix=C)
#         return jax.vmap(p.log_prob)(d) # Vmap over multiple datavectors for each z

#     Ls = jax.tree.map(
#         lambda d, mu, dmu, C: _log_prob_fn_z(d, pi, mu, dmu, C),
#         datavectors,
#         mus,
#         derivatives_,
#         covariances_mcmc
#     )

#     return jnp.sum(jnp.asarray(Ls))



# @typecheck
# def get_multi_redshift_mle(
#     pi: Float[Array, "p"], 
#     d: Float[Array, "... d"], 
#     Finv: Float[Array, "p p"], 
#     mus: list[Float[Array, "d"]], 
#     covariances: list[Float[Array, "d d"]], 
#     derivatives: list[Float[Array, "p d"]],
#     *,
#     verbose: bool = False
# ) -> Float[Array, "p"]:
#     """ Chi2 minimisation using block-diagonalised simulation-estimated data covariance """

#     # Covariances, derivatives for all z datas
#     C = block_diag(*covariances)
#     Cinv = jnp.linalg.inv(C) # Hartlap? Individual covariances are corrected?

#     # Concatenate objects across redshift to match block-diagonal covariance
#     derivatives = jnp.concatenate(derivatives, axis=1) # Stack on data axis, having averaged over realisations
#     mu = jnp.concatenate(mus)
#     d = jnp.concatenate(d)

#     if verbose:
#         print("D, mu, C, dmu:", d.shape, mu.shape, C.shape, derivatives.shape)

#     return pi + jnp.linalg.multi_dot([Finv, derivatives, Cinv, d - mu]) # d is z-concatenated datavector


# @typecheck
# def maybe_vmap_multi_redshift_mle(
#     pi: Float[Array, "p"], 
#     datavectors: list[Float[Array, "n d"]], 
#     Finv: Float[Array, "p p"], 
#     mus: list[Float[Array, "d"]], 
#     covariances: list[Float[Array, "d d"]], 
#     derivatives: list[Float[Array, "p d"]],
#     *,
#     verbose: bool = False
# ) -> Float[Array, "n p"]:

#     # Vmap MLE function over datavectors if plural
#     # datavectors multiple per redshift, covariances are one per redshift...
#     fn = lambda d: get_multi_redshift_mle(
#         pi, d, Finv, mus, covariances, derivatives, verbose=verbose
#     )

#     if verbose:
#         print("DATAVECTORS", [_.shape for _ in datavectors])

#     # Shape: (n, z, d)
#     datavectors = jnp.stack(datavectors, axis=1) # Stack list of datavectors ... NOTE: may be wrongly shaped...

#     assert datavectors.ndim == 3 # (n, n_cumulants, n_scales)

#     if verbose:
#         print("DATAVECTORS", datavectors.shape)

#     # Vmaps over first axis, concatenates them inside 'fn'
#     x = jax.vmap(fn)(datavectors) 

#     if verbose:
#         print("SUMMARIES[DATAVECTORS]", x.shape)

#     return x

# Compress datavectors concatenated over redshift, using block-diagonal covariance
# x_ = maybe_vmap_multi_redshift_mle( 
#     alpha, 
#     datavectors, 
#     Finv=Finv_all_z,
#     mus=mus, 
#     covariances=covariances, # Block-diagonalised in this function
#     derivatives=derivatives_
# )


        # @typecheck
        # @eqx.filter_jit
        # def mcmc_log_prob_fn_compressed(pi: Float[Array, "p"]) -> Scalar: 
        #     # Compressed data likelihood

        #     @typecheck
        #     def _log_prob_fn_z(
        #         d: Float[Array, "n d"], 
        #         pi: Float[Array, "p"], 
        #         mu: Float[Array, "d"], 
        #         dmu: Float[Array, "p d"], 
        #         Cinv: Float[Array, "d d"], 
        #         Finv: Float[Array, "p p"]
        #     ) -> Float[Array, "n"]:

        #         # mu_pi = linearised_model(alpha, pi, mu, dmu) # Model[cosmology]

        #         pi_ = jax.vmap(_mle)(d) # Compress multiple realisations

        #         p = tfd.MultivariateNormalFullCovariance(loc=alpha, covariance_matrix=Finv) # NOTE: + prior?

        #         return jax.vmap(p.log_prob)(pi_) # Vmap over multiple summaries

        #     # Tree map over lists of ingredients for each redshift
        #     Ls = jax.tree.map(
        #         lambda d, mu, dmu, Cinv, Finv: _log_prob_fn_z(d, pi, mu, dmu, Cinv, Finv),
        #         x_s, #datavectors,
        #         mus,
        #         derivatives_,
        #         precisions_mcmc,
        #         Finvs_mcmc 
        #     )

        #     return jnp.sum(jnp.asarray(Ls)) # Correct sum?