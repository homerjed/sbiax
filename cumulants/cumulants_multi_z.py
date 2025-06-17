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
from ml_collections import ConfigDict
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from tqdm.auto import trange
from chainconsumer import ChainConsumer, Chain
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.distributions import Distribution

from sbiax.inference import nuts_sample
from sbiax.utils import make_df, marker

from configs.cumulants_configs import default_posterior_sampling
from configs.ensembles_configs import (
    ensembles_cumulants_config, ensembles_bulk_cumulants_config
)
from configs.configs import (
    get_results_dir, 
    get_multi_z_posterior_dir, 
    get_ndes_from_config
)
from configs.args import get_cumulants_multi_z_args
from data.common import linearised_model
from data.constants import get_base_posteriors_dir, get_save_and_load_dirs, get_target_idx
from data.cumulants import get_parameter_strings
from data.pdfs import load_multi_z_bulk_pdf_fisher_forecast
from cumulants_ensemble import Ensemble, MultiEnsemble
from affine import affine_sample
from utils.utils import finite_samples_log_prob, get_datasets

typecheck = jaxtyped(typechecker=typechecker)

CompressionFn = Callable[[Float[Array, "d"], Float[Array, "p"]], Float[Array, "p"]]

jax.clear_caches()

cumulant_names = [r"$\langle \delta^2 \rangle_c$", r"$\langle \delta^3 \rangle_c$", r"$\langle \delta^4 \rangle_c$"] # ["var.", "skew.", "kurt."]

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
    freeze_parameters: bool = False,
    compression: Literal["linear", "nn"] = "linear",
    n_linear_sims: int = 10_000,
    pre_train: bool = False, 
    sbi_type: str = "nle",
    n_datavectors: int = 1,
    bulk_or_tails: Literal["tails", "bulk", "bulk_pdf"] = "tails",
    seed_datavector: Optional[int] = None, # Use fixed seed for config (ensemble, ...) and new seed for datavector
    *,
    verbose: bool = False
) -> tuple[
    Ensemble,
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
    # if seed_datavector is not None:
    #     key_datavector = jr.fold_in(key_datavector, seed_datavector)
    key_datavector = jr.key(int(time.time()))

    args.redshift          = redshift # Set the args redshift to get datasets
    args.linearised        = linearised
    args.order_idx         = order_idx
    args.freeze_parameters = freeze_parameters
    args.compression       = compression
    args.n_linear_sims     = n_linear_sims
    args.pre_train         = pre_train
    args.sbi_type          = sbi_type

    # SBI configuration, main dataset and all datasets for given redshift
    config_z, cumulants_dataset, datasets = get_datasets(args) # Config and cumulants_dataset can be bulk ... etc

    config_z.seed = seed

    if verbose: 
        print("bulk or tails", bulk_or_tails)
        print(config_z)

    # Sample datavector(s) at the fiducial parameters
    datavectors = cumulants_dataset.get_datavector(key_datavector, n=n_datavectors) # Generates linearised (or not) datavector 

    if datavectors.ndim == 1:
        datavectors = datavectors[jnp.newaxis, ...] # Add axis for vmapping...

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
    ensemble = Ensemble(ndes, sbi_type=config_z.sbi_type)

    # Load ensemble
    ensemble_path = os.path.join(get_results_dir(config_z, args=args), "ensemble.eqx")
    ensemble = eqx.tree_deserialise_leaves(ensemble_path, ensemble)

    print("Loaded ensemble from:\n\t", ensemble_path)
    print("Ensemble weights", ensemble.weights)

    return (
        ensemble, 
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


if __name__ == "__main__":
    import time

    key = jr.key(int(time.time())) # Only for datavectors, split for each redshift, datavector and separate posterior

    args = get_cumulants_multi_z_args()

    # Multi-z inference concerning the bulk or bulk + tails
    # if args.bulk_or_tails == "tails":
    #     ensembles_config = ensembles_cumulants_config
    # if args.bulk_or_tails == "bulk" or args.bulk_or_tails == "bulk_pdf":
    #     ensembles_config = ensembles_bulk_cumulants_config

    # # config = ensembles_config(
    # #     seed=args.seed, # Defaults if run without argparse args
    # #     sbi_type=args.sbi_type, 
    # #     linearised=args.linearised,
    # #     n_linear_sims=args.n_linear_sims,
    # #     compression=args.compression,
    # #     redshifts=args.redshifts,
    # #     order_idx=args.order_idx,
    # #     scales=args.scales,
    # #     pre_train=args.pre_train,
    # #     freeze_parameters=args.freeze_parameters
    # # )
    sampling_config = default_posterior_sampling(config=None, no_config=True)

    # Get the bulk Fisher forecast for all redshifts 
    # but easier to load frozen or not since it autosaves...
    data_dir, _, _ = get_save_and_load_dirs()

    Finv_bulk_pdfs_all_z = load_multi_z_bulk_pdf_fisher_forecast(data_dir, args)
    Finv_bulk_pdfs_all_z = Finv_bulk_pdfs_all_z / args.n_datavectors

    parameter_strings = get_parameter_strings()
    parameter_strings_ = [parameter_strings[_] for _ in ix]

    linear_str = "linearised" if args.linearised else "nonlinearised"
    pretrain_str = "pretrain" if args.pre_train else "nopretrain"

    # Where SBI's are saved (add on suffix for experiment details)
    posteriors_dir = get_base_posteriors_dir()

    # Save location for posterior plots
    parts = [
        "figs",
        "frozen" if args.freeze_parameters else "nonfrozen",
        args.bulk_or_tails,
        args.sbi_type,
        "linearised" if args.linearised else "nonlinearised",
        args.compression,
        "pretrain" if args.pre_train else "nopretrain",
        "z={}_m={}".format( 
            "".join(map(str, args.redshifts)),
            "".join(map(str, args.order_idx))
        )
    ]
    path_str = "/".join(filter(None, parts)) + "/"

    figs_dir = os.path.join(posteriors_dir, path_str)
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir, exist_ok=True)

    print("MULTI-Z FIGS_DIR:\n\t", figs_dir)

    parameter_dim = 2 if args.freeze_parameters else 5

    # Sample multiple posteriors across multiple redshifts
    for n_posterior in range(args.n_posteriors_sample):

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

        with trange(len(args.redshifts), desc="Multi-z", colour="magenta") as bar:
            for _, (z, redshift) in zip(bar, enumerate(args.redshifts)):

                print("@" * 80)
                print("Getting datavector(s) for redshift={}".format(redshift))

                key_z = jr.fold_in(key_n, z)

                # Load ensemble configuration, datavector/summary, prior, covariance, Fisher, derivatives
                (
                    ensemble, 
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
                    seed=args.seed,
                    order_idx=args.order_idx,
                    linearised=args.linearised, # NOTE: pre-train or not also...
                    compression=args.compression,
                    redshift=redshift, 
                    n_datavectors=args.n_datavectors,
                    pre_train=args.pre_train,
                    bulk_or_tails=args.bulk_or_tails,
                    freeze_parameters=args.freeze_parameters,
                    seed_datavector=args.seed_datavector,
                    verbose=args.verbose
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
                ensembles.append(ensemble)      
                
                bar.set_postfix_str("z={}, n_posterior={}".format(redshift, n_posterior))

        assert len(x_s) == len(args.redshifts)
        assert all([len(x_s[i]) == args.n_datavectors for i in range(len(x_s))])

        # Multi-redshift ensemble of individual ensembles at each redshift
        multi_ensemble = MultiEnsemble(
            ensembles, prior=prior, sbi_type=args.sbi_type
        ) 

        Finv_all_z = jnp.linalg.inv(F) # Combined Fisher information over all redshifts
        bulk_Finv_all_z = jnp.linalg.inv(F_bulk) # Combined Fisher information over all redshifts
        tails_Finv_all_z = jnp.linalg.inv(F_tails) # Combined Fisher information over all redshifts

        if args.verbose:
            # Plot Fisher forecasts
            for z, bulk_Finv_z, tails_Finv_z in zip(args.redshifts, bulk_Finvs, tails_Finvs):
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
                            "".join(map(str, args.order_idx)),
                            "".join(map(str, args.scales))
                        )
                    )

                plt.savefig(forecast_filename)
                plt.close()

                print("FISHER FORECAST SAVED AT: \n", forecast_filename)

            # Plot Fisher forecasts bulk / tails
            c = ChainConsumer()
            for z, bulk_Finv_z, tails_Finv_z in zip(args.redshifts, bulk_Finvs, tails_Finvs):
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
                        "".join(map(str, args.order_idx)),
                        "".join(map(str, args.scales))
                    )
                )

            plt.savefig(forecast_filename)
            plt.close()
            c = ChainConsumer()
            for z, bulk_Finv_z, tails_Finv_z in zip(args.redshifts, bulk_Finvs, tails_Finvs):
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
                        "".join(map(str, args.order_idx)),
                        "".join(map(str, args.scales))
                    )
                )

            plt.savefig(forecast_filename)
            plt.close()

            print("FISHER FORECAST SAVED AT: \n", forecast_filename)


            # Multi z Fisher
            c = ChainConsumer()
            # for z, Finv_z in zip(config.redshifts, Finvs):
            #     c.add_chain(
            #         Chain.from_covariance(
            #             alpha,
            #             Finv_z,
            #             columns=parameter_strings,
            #             name=r"$F_{\Sigma^{-1}}$ z=" + str(z),
            #             linestyle=":",
            #             shade_alpha=0.
            #         )
            #     )
            # for z, bulk_Finv_z in zip(config.redshifts, bulk_Finvs):
            #     c.add_chain(
            #         Chain.from_covariance(
            #             alpha,
            #             bulk_Finv_z,
            #             columns=parameter_strings,
            #             name=r"$F_{\Sigma^{-1}}$ z=" + str(z) + " [bulk]",
            #             linestyle=":",
            #             color="g",
            #             shade_alpha=0.
            #         )
            #     )
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
                color="#7600bc"
            )
            fig = c.plotter.plot()

            forecast_filename = os.path.join(
                    figs_dir if figs_dir is not None else "fisher_forecasts/", 
                    "fisher_forecast_all_z_R={}_m={}.png".format(
                        # "".join(map(str, args.redshifts)),
                        "".join(map(str, args.scales)),
                        "".join(map(str, args.order_idx))
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
            show_tqdm=True # args.use_tqdm
        )

        alpha_log_prob = log_prob_fn(jnp.asarray(alpha))
        samples_log_prob = jax.vmap(log_prob_fn)(samples)
        samples_log_prob = finite_samples_log_prob(samples_log_prob)

        # Save posterior, Fisher and summary
        posterior_save_dir = get_multi_z_posterior_dir(args)
        if not os.path.exists(posterior_save_dir):
            os.makedirs(posterior_save_dir, exist_ok=True)

        print("Multi-z posterior save dir:\n\t", posterior_save_dir)
        
        # NOTE: Additional seed added if provided
        posterior_filename = os.path.join(
            posterior_save_dir, 
            "multi_z_posterior_{}{}.npz".format( # NOTE: was just 'posterior_...' before
                args.seed, 
                ("_" + str(args.seed_datavector)) if args.seed_datavector is not None else ""
            ) 
        )
        np.savez(
            posterior_filename,
            samples=samples, 
            samples_log_prob=samples_log_prob,
            Finv=Finv_all_z,
            datavectors=datavectors,
            summaries=np.stack(x_s, axis=0), # NOTE: (n_datavectors, n_z, n_x) ?
            # summary=x_ # Is this correct one to save?
        )

        print("MULTI-Z POSTERIOR FILENAME:\n", posterior_filename)

        """
            MCMC sample with linear model
        """
        C_all_z = block_diag(*covariances) # Block-diagonal covariance of all redshifts
        mu_all_z = jnp.concatenate(mus) # Concatenated expectation model across all redshifts
        dmu_all_z = jnp.concatenate(derivatives_) # Concatenated derivatives across all redshifts

        # Don't rescale by Finv for MCMC
        covariances_mcmc = [_C * args.n_datavectors for _C in covariances]
        precisions_mcmc = [jnp.linalg.inv(_C) for _C in covariances_mcmc]
        Finvs_mcmc = [_Finv * args.n_datavectors for _Finv in Finvs]


        @typecheck
        @eqx.filter_jit
        def mcmc_log_prob_fn_compressed(pi: Float[Array, "p"]) -> Scalar: 
            # Compressed data likelihood

            @typecheck
            def _log_prob_fn_z(
                d: Float[Array, "n d"], 
                pi: Float[Array, "p"], 
                mu: Float[Array, "d"], 
                dmu: Float[Array, "p d"], 
                Cinv: Float[Array, "d d"], 
                Finv: Float[Array, "p p"]
            ) -> Float[Array, "n"]:

                mu_pi = linearised_model(alpha, pi, mu, dmu) # Model[cosmology]

                _mle = lambda d: alpha + jnp.linalg.multi_dot([Finv, dmu, Cinv, d - mu_pi])

                pi_ = jax.vmap(_mle)(d) # Compress multiple realisations

                p = tfd.MultivariateNormalFullCovariance(loc=alpha, covariance_matrix=Finv) # NOTE: + prior?

                return jax.vmap(p.log_prob)(pi_) # Vmap over multiple summaries

            # Tree map over lists of ingredients for each redshift
            Ls = jax.tree.map(
                lambda d, mu, dmu, Cinv, Finv: _log_prob_fn_z(d, pi, mu, dmu, Cinv, Finv),
                datavectors,
                mus,
                derivatives_,
                precisions_mcmc,
                Finvs_mcmc 
            )

            return jnp.sum(jnp.asarray(Ls)) # Correct sum?


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
            show_tqdm=True # args.use_tqdm
        )
        mcmc_samples_log_prob = jax.vmap(mcmc_log_prob_fn_compressed)(mcmc_samples)

        """
            Full posterior
        """

        if not args.freeze_parameters:
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
                    name="SBI[{}]".format(args.bulk_or_tails), 
                    color="r" if args.bulk_or_tails == "tails" else "b"
                )
            )
            if PLOT_SUMMARIES:
                # If using multiple datavectors, plot them individually
                # if x_.ndim > 1:
                #     for i, _x_ in enumerate(x_):
                #         c.add_marker(
                #             location=marker(_x_, parameter_strings), 
                #             name=r"$\hat{x}$ " + str(i), 
                #             color="r" if args.bulk_or_tails == "tails" else "b"
                #         )
                # else:
                #     c.add_marker(
                #         location=marker(x_, parameter_strings), 
                #         name=r"$\hat{x}$", 
                #         color="r" if args.bulk_or_tails == "tails" else "b"
                #     )

                # c.add_marker(
                #     location=marker(np.mean(x_, axis=0) if x_.ndim > 1 else x_, parameter_strings), 
                #     name=r"$\bar{x}$", 
                #     color="r" if args.bulk_or_tails == "tails" else "b",
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
                #     color="r" if args.bulk_or_tails == "tails" else "b",
                #     marker_style="x"
                # )

                print([_.shape for _ in x_s])
                mu_x_ = np.asarray([np.mean(x_, axis=0) for x_ in x_s])
                print("MU_X_", mu_x_.shape) # (3, 5) ; 3 redshifts
                mu_x_ = np.mean(mu_x_, axis=0)[ix]
                print("MU_X_", mu_x_.shape) # (3, 5) ; 3 redshifts
                c.add_marker(
                    location=marker(mu_x_, parameter_strings_), 
                    name=r"$\bar{x}$ fresh", 
                    color="r" if args.bulk_or_tails == "tails" else "b",
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
                        ("linearised" if args.linearised else "non-linear") + "\n",
                        "[{}]".format(", ".join(map(str, args.redshifts))),
                        args.n_linear_sims if args.linearised else 2000, 
                        args.n_linear_sims if args.pre_train else None,
                        "[{}]".format(", ".join(map(str, args.scales))),
                        "[{}]".format(", ".join(map(str, [cumulant_names[_] for _ in args.order_idx])))
                    ),
                multialignment='center'
            )
            plt.savefig(
                os.path.join(
                    figs_dir, 
                    "multi_ensemble_posterior_cumulants_{}_{}_{}_{}{}.pdf".format(
                        args.seed, linear_str, pretrain_str, n_posterior,
                        ("_" + str(args.seed_datavector)) if args.seed_datavector is not None else ""
                    )
                )
            )
            plt.close()

        """
            Marginalised posterior
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
                name="SBI[{}]".format(args.bulk_or_tails), 
                color="r" if args.bulk_or_tails == "tails" else "b"
            )
        )
        if PLOT_SUMMARIES:
            # if x_.ndim > 1:
            #     for i, _x_ in enumerate(x_):
            #         c.add_marker(
            #             location=marker(_x_[ix], parameter_strings_), 
            #             name=r"$\hat{x}$ " + str(i), 
            #             color="r" if args.bulk_or_tails == "tails" else "b"
            #         )
            # else:
            #     c.add_marker(
            #         location=marker(x_[ix], parameter_strings_), 
            #         name=r"$\hat{x}$", 
            #         color="r" if args.bulk_or_tails == "tails" else "b"
            #     )
            # c.add_marker(
            #     location=marker(np.mean(x_, axis=0) if x_.ndim > 1 else x_, parameter_strings_), 
            #     name=r"$\bar{x}$", 
            #     color="r" if args.bulk_or_tails == "tails" else "b",
            #     marker_style="x"
            # )
            # c.add_marker(
            #     location=marker(
            #         np.mean(
            #             np.asarray([np.mean(x_, axis=0) for x_ in x_s]), 
            #             axis=0
            #         ), 
            #         parameter_strings_
            #     ), 
            #     name=r"$\bar{x}$ fresh", 
            #     color="r" if args.bulk_or_tails == "tails" else "b",
            #     marker_style="x"
            # )

            print([_.shape for _ in x_s])
            mu_x_ = np.asarray([np.mean(x_, axis=0) for x_ in x_s]) # Average over realisations
            print("MU_X_", mu_x_.shape) # (3, 5) ; 3 redshifts
            mu_x_ = np.mean(mu_x_, axis=0)[ix] # Average over redshifts
            print("MU_X_", mu_x_.shape) # (3, 5) ; 3 redshifts
            c.add_marker(
                location=marker(mu_x_, parameter_strings_), 
                name=r"$\bar{x}$ fresh", 
                color="r" if args.bulk_or_tails == "tails" else "b",
                marker_style="s"
            )
        c.add_marker(
            location=marker(alpha[ix], parameter_strings_), 
            name=r"$\alpha$", 
            color="#7600bc"
        )
        fig = c.plotter.plot()
        fig.suptitle(
            r"{} SBI & $F_{{\Sigma}}^{{-1}}$".format("$k_n$") + "\n" +
            "{} z={},\n $n_s$={}, (pre-train $n_s$={}),\n R={} Mpc,\n $k_n$={}".format(
                    ("linearised" if args.linearised else "non-linear") + "\n",
                    "[{}]".format(", ".join(map(str, args.redshifts))),
                    args.n_linear_sims if args.linearised else 2000, 
                    args.n_linear_sims if args.pre_train else None,
                    "[{}]".format(", ".join(map(str, args.scales))),
                    "[{}]".format(", ".join(map(str, [cumulant_names[_] for _ in args.order_idx])))
                ),
            multialignment='center'
        )
        posterior_plot_filename = os.path.join(
            figs_dir, 
            "multi_ensemble_posterior_marginalised_cumulants_{}_{}_{}_{}{}.pdf".format(
                args.seed, linear_str, pretrain_str, n_posterior,
                ("_" + str(args.seed_datavector)) if args.seed_datavector is not None else ""
            )
        )
        plt.savefig(posterior_plot_filename)
        plt.close()

        print("MULTI-Z POSTERIOR PLOT FILENAME:\n", posterior_plot_filename)

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
                name="SBI[{}]".format(args.bulk_or_tails), 
                color="r" if args.bulk_or_tails == "tails" else "b"
            )
        )
        posterior_df = make_df(
            mcmc_samples[:, ix], mcmc_samples_log_prob, parameter_strings=parameter_strings_
        )
        c.add_chain(
            Chain(
                samples=posterior_df, 
                name="MCMC[{}]".format(args.bulk_or_tails), 
                color="#9867C5"
            )
        )
        # if x_.ndim > 1:
        #     for i, _x_ in enumerate(x_):
        #         c.add_marker(
        #             location=marker(_x_[ix], parameter_strings_), 
        #             name=r"$\hat{x}$ " + str(i), 
        #             color="r" if args.bulk_or_tails == "tails" else "b"
        #         )
        # else:
        #     c.add_marker(
        #         location=marker(x_[ix], parameter_strings_), 
        #         name=r"$\hat{x}$", 
        #         color="r" if args.bulk_or_tails == "tails" else "b"
        # # )
        # c.add_marker(
        #     location=marker(np.mean(x_, axis=0)[ix] if x_.ndim > 1 else x_[ix], parameter_strings_), 
        #     name=r"$\bar{x}$", 
        #     color="r" if args.bulk_or_tails == "tails" else "b",
        #     marker_style="x"
        # )

        if PLOT_SUMMARIES:
            print([_.shape for _ in x_s])
            mu_x_ = np.asarray([np.mean(x_, axis=0) for x_ in x_s])
            print("MU_X_", mu_x_.shape) # (3, 5) ; 3 redshifts
            mu_x_ = np.mean(mu_x_, axis=0)[ix]
            print("MU_X_", mu_x_.shape) # (3, 5) ; 3 redshifts
            c.add_marker(
                location=marker(mu_x_, parameter_strings_), 
                name=r"$\bar{x}$ fresh", 
                color="r" if args.bulk_or_tails == "tails" else "b",
                marker_style="s"
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
                    ("linearised" if args.linearised else "non-linear") + "\n",
                    "[{}]".format(", ".join(map(str, args.redshifts))),
                    args.n_linear_sims if args.linearised else 2000, 
                    args.n_linear_sims if args.pre_train else None,
                    "[{}]".format(", ".join(map(str, args.scales))),
                    "[{}]".format(", ".join(map(str, [cumulant_names[_] for _ in args.order_idx])))
                ),
            multialignment='center'
        )
        posterior_plot_filename = os.path.join(
            figs_dir, 
            "multi_ensemble_posterior_marginalised_cumulants_{}_{}_{}_{}{}_mcmc.pdf".format(
                args.seed, linear_str, pretrain_str, n_posterior,
                ("_" + str(args.seed_datavector)) if args.seed_datavector is not None else ""
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
                    Finvs[i][ix, :][:, ix], # Bulk or tails?
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
                Finv_all_z[ix, :][:, ix], # Bulk or tails?
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
                    ("linearised" if args.linearised else "non-linear") + "\n",
                    "[{}]".format(", ".join(map(str, args.redshifts))),
                    args.n_linear_sims if args.linearised else 2000, 
                    args.n_linear_sims if args.pre_train else None,
                    "[{}]".format(", ".join(map(str, args.scales))),
                    "[{}]".format(", ".join(map(str, [cumulant_names[_] for _ in args.order_idx])))
                ),
            multialignment='center'
        )
        posterior_plot_filename = os.path.join(
            figs_dir, 
            "summaries_and_Finvs_{}_{}_{}_{}{}_mcmc.pdf".format(
                args.seed, linear_str, pretrain_str, n_posterior,
                ("_" + str(args.seed_datavector)) if args.seed_datavector is not None else ""
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