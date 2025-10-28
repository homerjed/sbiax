from typing import Optional, Literal, Callable, Any
import time
import os
import operator

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
from chainconsumer import ChainConsumer, Chain, Truth

from sbiax.inference import nuts_sample
from sbiax.utils import make_df, marker

from configs.log import setup_module_logger, get_log_level
from configs.configs import (
    get_results_dir, 
    get_multi_z_posterior_filename,
    get_ndes_from_config
)
from configs.args import get_cumulants_sbi_args, get_cumulants_multi_z_args
from compression.nn import wrap_ensemble_ndes_with_processors, Processor
from data.common import get_prior, linearised_model
from data.constants import (
    get_scales,
    get_base_posteriors_dir, 
    get_save_and_load_dirs, 
    get_target_idx, 
    get_cumulant_names,
    ALPHA,
    PARAMETER_STRINGS,
    LOWER, 
    UPPER
)
from sbiax.ndes import Ensemble, MultiEnsemble
from utils import (
    finite_samples_log_prob, 
    get_datasets, 
    get_fisher_chain_df, 
    plot_summaries_fiducial,
    overlay_bounds_on_corner,
    customize_plot
)

USE_SOBOL = True if os.environ.get("USE_SOBOL", "").lower() in ("1", "true") else False 

# Plot individual redshift posteriors for individual redshifts
DEBUG_POSTERIOR_SAMPLE = True if os.environ.get("USE_SOBOL", "").lower() in ("1", "true") else False 

N_ENSEMBLE_NETS = int(os.environ.get("N_ENSEMBLE_NETS", 10))

N_LINEAR_SIMS = 32768 if USE_SOBOL else 2000

N_NUTS_SAMPLES = 10_000
N_FISHER_SAMPLES = 800_000

def cut_samples(samples, lower, upper):
    return samples[np.all((samples >= lower) & (samples <= upper), axis=1)]

if USE_SOBOL:
    from data.get_sobol_cumulants import load_multi_z_bulk_pdf_fisher_forecast
else:
    from data.pdfs import load_multi_z_bulk_pdf_fisher_forecast

TYPECHECK = True if os.environ.get("TYPECHECK", "").lower() in ("1", "true") else False
if TYPECHECK:
    typecheck = jaxtyped(typechecker=typechecker)
else:
    typecheck = lambda x: x

logger, log_figs_dir = setup_module_logger(__name__, level=get_log_level())

CompressionFn = Callable[[Float[Array, "d"], Float[Array, "p"]], Float[Array, "p"]]

jax.clear_caches()

target_idx = get_target_idx()

PLOT_SUMMARIES = False

SUMMARIES_PLOT_COLOURS = dict(
    bulk=["#3b82f6", "#60a5fa", "#bfdbfe"], 
    tails=["#ef4444", "#f87171", "#fca5a5"]
)

def default(v, d):
    return v if v is not None else d


"""
    Sample a posterior with a uniform physics-parameter prior
    and a likelihood function made from separate flows (or ensembles
    of flows) that have been trained on data from different redshifts. 

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
    seed_datavector: int, # Use fixed seed for config (ensemble, ...) and new seed for datavector
    redshift: float, 
    bulk_or_tails: Literal["tails", "bulk", "bulk_pdf"],
    compression: Literal["linear", "nn", "imnn", "ensemble-nn"],
    linearised: bool = True, 
    order_idx: list[int] = [0, 1, 2],
    scales: list[float] = get_scales(),
    n_linear_sims: int = N_LINEAR_SIMS,
    pre_train: bool = False, 
    n_datavectors: int = 1,
    *,
    figs_dir: Optional[str] = None
) -> tuple[
    Ensemble,
    Float[Array, "n p"],
    Float[Array, "n d"],
    Float[Array, "p p"],
    Float[Array, "p p"],
    Float[Array, "d d"],
    Float[Array, "d"],
    Float[Array, "p d"],
    Float[Array, "..."], # X0
    Callable
]:
    """ 
        Get config and datavector associated with a redshift z to load the experiment 
        for use in an ensemble of SBI likelihoods at different redshifts, for the bulk
        or tails datasets.
        - get config at redshift
        - load datasets, compression function and NDE ensemble. 
    """

    assert redshift in [0.0, 0.5, 1.0], "Redshift {} not in [0.0, 0.5, 1.0]".format(redshift) 

    # Arguments for default SBI experiment
    sbi_args = get_cumulants_sbi_args(multi_z=True)

    sbi_args.seed          = seed
    sbi_args.redshift      = redshift # Set the redshift of sbi_args to get datasets
    sbi_args.bulk_or_tails = bulk_or_tails
    sbi_args.linearised    = linearised
    sbi_args.order_idx     = order_idx
    sbi_args.scales        = scales
    sbi_args.compression   = compression
    sbi_args.n_linear_sims = n_linear_sims
    sbi_args.pre_train     = pre_train

    # SBI configuration, main dataset and all datasets for given redshift
    config_z, cumulants_dataset, datasets = get_datasets(sbi_args) # Config and cumulants_dataset can be bulk ... etc

    results_dir_z = get_results_dir(config_z, args=sbi_args)

    assert config_z.redshift == redshift, (
        "Mismatch in config_z.redshift {} == redshift {}".format(config_z.redshift, redshift)
    )

    assert compression in ["linear", "nn", "ensemble-nn", "imnn"], (
        "{} is an invalid compression.".format(compression)
    )

    logger.info("BULK/TAILS: {}".format(bulk_or_tails))
    logger.info("CONFIG:\n{}".format(config_z))
    logger.info("RESULTS_DIR z:\n{}".format(results_dir_z))

    # Load summaries from SBI experiment at seed and redshift
    all_summaries = np.load(os.path.join(results_dir_z, "all_summaries.npz"))
    datavectors = all_summaries["datavectors"]
    summaries_z = all_summaries["summaries"]
    X = all_summaries["latins"]
    X0 = all_summaries["fiducials"]

    # Summaries is now a huge array so pick `n_datavectors` measurements
    key_datavector = jr.key(seed_datavector)
    ix = jr.choice(key_datavector, len(summaries_z), (n_datavectors,))
    datavectors = datavectors[ix]
    summaries_z = summaries_z[ix]

    # Get NDEs for this redshift (NOTE: dummy key)
    ensemble = get_ndes_from_config(config_z, key=key)

    print("ensemble ndes pre-loading", type(ensemble.ndes), len(ensemble.ndes))
    print("ensemble weights pre-loading", ensemble.weights.shape)

    # Load ensemble
    ensemble_path = os.path.join(results_dir_z, "ensemble.eqx")
    ensemble = eqx.tree_deserialise_leaves(ensemble_path, ensemble)

    logger.info("Loaded ensemble from:\n\t{}".format(ensemble_path))
    logger.info("Ensemble weights:\n\t{}".format(ensemble.weights))

    # Wrap after loading because all we will do is sample here
    # NOTE: load NDE processor used in cumulants_sbi
    processor = Processor(jnp.ones_like(X), jnp.ones_like(X)) # Mock hypercube set, massive compression
    processor_path = os.path.join(results_dir_z, "processor_nde.eqx") # NOTE: same processor for all NDEs in ensemble-z
    processor = eqx.tree_deserialise_leaves(processor_path, processor)

    logger.info("Loaded processor from:\n\t{}".format(processor_path))
    logger.info("Processor:\n\t{}".format(processor))

    ensemble = wrap_ensemble_ndes_with_processors(ensemble, processor)

    # Get linear compressor
    mu = jnp.mean(cumulants_dataset.data.fiducial_data, axis=0)
    dmu = jnp.mean(cumulants_dataset.data.derivatives, axis=0)
    Finv = cumulants_dataset.data.Finv
    precision = cumulants_dataset.data.Cinv

    def linear_compressor(d, pi):
        mu_p = linearised_model(jnp.asarray(ALPHA), pi, mu, dmu)
        return pi + jnp.linalg.multi_dot([Finv, dmu, precision, d - mu_p])

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Debugging plots

    """
        Summaries
    """

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

    if DEBUG_POSTERIOR_SAMPLE and (seed % 2 == 0):
            
        def get_mcmc_log_prob_fn(
            datavectors: Float[Array, "n d"] | Float[Array, "d"], 
            linear_compressor: Callable, 
            Finv: Float[Array, "p p"],
            prior: Any
        ) -> Callable:

            if datavectors.ndim == 1:
                datavectors = datavectors[jnp.newaxis, :]

            @typecheck
            @eqx.filter_jit
            def mcmc_log_prob_fn_compressed(pi: Float[Array, "p"]) -> Scalar: 
                # Compressed data likelihood, equivalent to multi-ensemble-SBI likelihood

                @typecheck
                def _log_prob_fn_z(
                    d: Float[Array, "n d"], 
                    Finv: Float[Array, "p p"], 
                    compressor: Callable
                ) -> Float[Array, "n"]:

                    # Assumes all datavectors drawn at alpha
                    def _likelihood(_pi_, pi):
                        return jax.scipy.stats.multivariate_normal.logpdf(_pi_, pi, Finv)

                    pi_ = jax.vmap(compressor, in_axes=(0, None))(d, pi)

                    return jax.vmap(_likelihood, in_axes=(0, None))(pi_, pi) # Vmap over multiple summaries

                # Tree map over lists of ingredients for each redshift
                Ls = jax.tree.map(
                    lambda d, Finv, c: _log_prob_fn_z(d, Finv, c), 
                    [datavectors], 
                    [Finv],
                    [linear_compressor]
                )

                prior_log_prob = prior.log_prob(pi)

                return jnp.sum(jnp.asarray(Ls)) + prior_log_prob # NOTE: Correct sum? LogSumExp?

            return mcmc_log_prob_fn_compressed

        logger.debug("compressed datavector {}, alpha {}".format(summaries_z.shape, cumulants_dataset.data.alpha))

        prior = get_prior()

        log_prob_fn = ensemble.ensemble_log_prob_fn(summaries_z, prior)

        key_state, key_sample = jr.split(jr.key(int(time.time())))

        # SBI sample
        samples, samples_log_prob = nuts_sample(
            key_sample, 
            log_prob_fn, 
            initial_state=jnp.asarray(ALPHA[jnp.newaxis, :]), 
            n_samples=N_NUTS_SAMPLES
        )
        samples = jnp.squeeze(samples) # NOTE: if n_chains != 1 ...
        samples_log_prob = jnp.squeeze(samples_log_prob)
        samples_log_prob = finite_samples_log_prob(samples_log_prob) # all 

        posterior_df = make_df(
            samples.squeeze(), 
            samples_log_prob.squeeze(), 
            parameter_strings=PARAMETER_STRINGS
        )

        # MCMC sample
        mcmc_samples, mcmc_samples_log_prob = nuts_sample(
            key_sample, 
            get_mcmc_log_prob_fn(datavectors, linear_compressor, Finv, prior), 
            initial_state=jnp.asarray(ALPHA[jnp.newaxis, :]), 
            n_samples=N_NUTS_SAMPLES
        )
        mcmc_samples = jnp.squeeze(mcmc_samples) # NOTE: if n_chains != 1 ...
        mcmc_samples_log_prob = jnp.squeeze(mcmc_samples_log_prob)
        mcmc_samples_log_prob = finite_samples_log_prob(mcmc_samples_log_prob) # all 

        mcmc_posterior_df = make_df(
            mcmc_samples.squeeze(), 
            mcmc_samples_log_prob.squeeze(), 
            parameter_strings=PARAMETER_STRINGS
        )

        # CLIPPED
        fishers = [
            cumulants_dataset.data.Finv / n_datavectors,
            datasets["bulk"].data.Finv / n_datavectors,
            datasets["tails"].data.Finv / n_datavectors,
            datasets["bulk_pdf"].data.Finv / n_datavectors
        ]

        fisher_dfs = []
        for fisher in fishers:

            fisher_samples = np.random.multivariate_normal(
                ALPHA, fisher, (N_FISHER_SAMPLES,) 
            ) 

            fisher_df = make_df(
                cut_samples(fisher_samples, LOWER, UPPER),
                parameter_strings=PARAMETER_STRINGS
            )

            fisher_dfs.append(fisher_df)

        c = ChainConsumer()
        c.add_chain(
            Chain(
                samples=fisher_dfs[1],
                name=r"$F_{\Sigma^{-1}}$" + " {}".format("$k_n$[bulk]"),
                color="b",
                linestyle=":",
                shade_alpha=0.
            )
        )
        c.add_chain(
            Chain(
                samples=fisher_dfs[2],
                name=r"$F_{\Sigma^{-1}}$" + " {}".format("$k_n$[tails]"),
                color="r",
                linestyle=":",
                shade_alpha=0.
            )
        )
        c.add_chain(
            Chain(
                samples=fisher_dfs[3],
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
        c.add_chain(
            Chain(
                samples=mcmc_posterior_df, 
                name="MCMC[{}]".format(bulk_or_tails), 
                color="purple"
            )
        )

        for i, _test_summary in enumerate(summaries_z):
            c.add_marker(
                location=marker(_test_summary, parameter_strings=PARAMETER_STRINGS),
                name=str(i), 
                color="r" if bulk_or_tails == "tails" else "b"
            )
        c.add_truth(
            Truth(location=dict(zip(PARAMETER_STRINGS, ALPHA)), name=r"$\pi^0$")
        )
        # c.add_marker(
        #     location=marker(cumulants_dataset.data.alpha, parameter_strings=PARAMETER_STRINGS),
        #     name=r"$\alpha$", 
        #     color="#7600bc"
        # )

        fig = c.plotter.plot()
        overlay_bounds_on_corner(fig, LOWER, UPPER)
        fig = customize_plot(fig)

        filename = os.path.join(
            figs_dir if figs_dir is not None else log_figs_dir, 
            "posterior_{}_clipped_z={}_{}_{}_{}.pdf".format(
                "blackjax", redshift, bulk_or_tails, seed, seed_datavector
            )
        )

        plt.savefig(filename)
        plt.close()

        print("CLIPPED DEBUG POSTERIOR SAVED AT:\n\t", filename)


        # Summaries plot
        # - These summaries are used 
        # - X0 and summaries not necessarily the same?

        summaries_df = make_df(
            X0[::4], parameter_strings=PARAMETER_STRINGS
        )

        fishers = [
            cumulants_dataset.data.Finv / n_datavectors,
            datasets["bulk"].data.Finv / n_datavectors,
            datasets["tails"].data.Finv / n_datavectors,
            datasets["bulk_pdf"].data.Finv / n_datavectors
        ]

        fisher_dfs = []
        for fisher in fishers:

            fisher_samples = np.random.multivariate_normal(
                ALPHA, fisher, (N_FISHER_SAMPLES,) 
            ) 

            fisher_df = make_df(
                cut_samples(fisher_samples, LOWER, UPPER),
                parameter_strings=PARAMETER_STRINGS
            )

            fisher_dfs.append(fisher_df)


        c = ChainConsumer()
        c.add_chain(
            Chain(
                samples=fisher_dfs[1],
                name=r"$F_{\Sigma^{-1}}$" + " {}".format("$k_n$[bulk]"),
                color="b",
                linestyle=":",
                shade_alpha=0.
            )
        )
        c.add_chain(
            Chain(
                samples=fisher_dfs[2],
                name=r"$F_{\Sigma^{-1}}$" + " {}".format("$k_n$[tails]"),
                color="r",
                linestyle=":",
                shade_alpha=0.
            )
        )
        c.add_chain(
            Chain(
                samples=fisher_dfs[3],
                name=r"$F_{\Sigma^{-1}}$" + " {}".format("PDF[bulk]"),
                color="g",
                linestyle=":",
                shade_alpha=0.
            )
        )
        c.add_chain(
            Chain(
                samples=summaries_df, 
                name="NN[{}]".format(bulk_or_tails), 
                color="r" if bulk_or_tails == "tails" else "b",
                shade_alpha=0.,
                plot_contour=False,
                plot_cloud=True
            )
        )
        c.add_chain(
            Chain(
                samples=posterior_df, 
                name="SBI[{}]".format(bulk_or_tails), 
                color="r" if bulk_or_tails == "tails" else "b"
            )
        )
        for i, _test_summary in enumerate(summaries_z):
            c.add_marker(
                location=marker(_test_summary, parameter_strings=PARAMETER_STRINGS),
                name=str(i), 
                color="r" if bulk_or_tails == "tails" else "b"
            )
        c.add_truth(
            Truth(location=dict(zip(PARAMETER_STRINGS, ALPHA)), name=r"$\pi^0$")
        )

        fig = c.plotter.plot()
        overlay_bounds_on_corner(fig, LOWER, UPPER)
        fig = customize_plot(fig)

        filename = os.path.join(
            figs_dir if figs_dir is not None else log_figs_dir, 
            "posterior_with_summaries_{}_clipped_z={}_{}_{}_{}.pdf".format(
                "blackjax", redshift, bulk_or_tails, seed, seed_datavector
            )
        )

        plt.savefig(filename)
        plt.close()

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    return (
        # Ensemble of flows trained at redshift z
        ensemble, 
        # Summaries at redshift z, x = postprocess_p[NN(preprocess_d[d])]
        summaries_z, 
        # Datavectors at redshift z
        datavectors,
        # Scale data and parameter covariances by number of measurements (these are not used in MLEs)
        jnp.asarray(datasets["bulk"].data.Finv) / n_datavectors, 
        jnp.asarray(datasets["tails"].data.Finv) / n_datavectors, 
        jnp.asarray(cumulants_dataset.data.C) / n_datavectors, 
        mu,
        dmu,
        X0,
        linear_compressor
    )


def get_figs_dir(multi_z_args):
    # Save location for posterior plots

    # Where SBI's are saved (add on suffix for experiment details)
    posteriors_dir = get_base_posteriors_dir()

    parts = [
        # "figs",
        "multi_z", # NOTE: ADDED 
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

    prior = get_prior()

    # Multi-z inference concerning the bulk or bulk + tails (just sampling args)

    # Get the bulk Fisher forecast for all redshifts 
    # but easier to load frozen or not since it autosaves...
    Finv_bulk_pdfs_all_z = load_multi_z_bulk_pdf_fisher_forecast(data_dir, multi_z_args) 
    Finv_bulk_pdfs_all_z = Finv_bulk_pdfs_all_z / multi_z_args.n_datavectors

    linear_str = "linearised" if multi_z_args.linearised else "nonlinearised"
    pretrain_str = "pretrain" if multi_z_args.pre_train else "nopretrain"

    figs_dir = get_figs_dir(multi_z_args)

    # parameter_dim = get_target_idx().size if multi_z_args.freeze_parameters else ALPHA.size
    parameter_dim = ALPHA.size

    # Loop over redshifts; loading ensembles and datavectors
    datavectors  = [] # I.i.d. datavectors, tuple'd for each redshift (plural measurements in tuple)
    summaries    = [] # Summaries of these datavectors
    ensembles    = [] # Ensembles of NDEs trained on simulations at each redshift
    covariances  = [] # Covariance matrices of simulations at each redshift
    bulk_Finvs   = [] # Fisher parameter covariances at each redshift
    tails_Finvs  = [] # Fisher parameter covariances at each redshift
    F_bulk       = jnp.zeros((parameter_dim, parameter_dim)) # Add independent information from data at each redshift (for bulk)
    F_tails      = jnp.zeros((parameter_dim, parameter_dim)) # Add independent information from data at each redshift (for bulk)
    X0s          = []
    linear_compression_fns = []

    with trange(len(multi_z_args.redshifts), desc="Multi-z", colour="magenta") as bar:
        for _, (z, redshift) in zip(bar, enumerate(multi_z_args.redshifts)):

            print("@" * 80)
            print("Getting n={} datavector(s) for redshift={}".format(multi_z_args.n_datavectors, redshift))

            key_z = jr.fold_in(key, z)

            # Load ensemble configuration, datavector/summary, prior, covariance, Fisher, derivatives
            (
                ensemble, 
                _summaries, # MLE[datavectors] at this redshift
                _datavectors, # list[Array["n d"]]
                bulk_Finv_z, # NOTE: This Finv not used for compression (scaled by n_datavectors)
                tails_Finv_z, # NOTE: This Finv not used for compression (scaled by n_datavectors)
                C_z, # NOTE: Scaled by n_datavectors
                mu, 
                derivatives,
                X0_z,
                linear_compression_fn_z
            ) = get_z_config_and_datavector(
                key_z, 
                seed=multi_z_args.seed,                       # NOTE: based on SBI run seed
                order_idx=multi_z_args.order_idx,
                scales=multi_z_args.scales,
                linearised=multi_z_args.linearised,           # NOTE: pre-train or not also...
                compression=multi_z_args.compression,
                redshift=redshift, 
                n_datavectors=multi_z_args.n_datavectors,
                pre_train=multi_z_args.pre_train,
                bulk_or_tails=multi_z_args.bulk_or_tails,
                seed_datavector=multi_z_args.seed_datavector, # NOTE: based on job-array
                figs_dir=figs_dir
            ) 

            # Add Fisher information from redshift (independent; Limber)
            F_bulk += jnp.linalg.inv(bulk_Finv_z)
            F_tails += jnp.linalg.inv(tails_Finv_z)

            ensembles.append(ensemble)      
            summaries.append(_summaries)
            datavectors.append(_datavectors)
            bulk_Finvs.append(bulk_Finv_z)
            tails_Finvs.append(tails_Finv_z)
            covariances.append(C_z)
            X0s.append(X0_z)

            # _Finv = bulk_Finv_z if multi_z_args.bulk_or_tails == "bulk" else tails_Finv_z
            # precision_z = jnp.linalg.inv(C_z * multi_z_args.n_datavectors) # NOTE: unscale

            # def linear_compression_fn_z(d, p):
            #     return p + jnp.linalg.multi_dot([_Finv, derivatives, precision_z, d - mu])

            linear_compression_fns.append(linear_compression_fn_z)
            
            bar.set_postfix_str("z={}".format(redshift))


    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # DEBUG

    c = ChainConsumer()

    for i, (_Finv_plot, dataset_type, name) in enumerate(zip(
        [
            Finv_bulk_pdfs_all_z * multi_z_args.n_datavectors, 
            np.linalg.inv(F_bulk) * multi_z_args.n_datavectors, 
            np.linalg.inv(F_tails) * multi_z_args.n_datavectors
        ],
        ["bulk_pdf", "bulk", "tails"],
        [" PDF[bulk]", " $k_n$[bulk]", " $k_n$[tails]"],
    )):
        c.add_chain(
            Chain.from_covariance(
                ALPHA,
                _Finv_plot,
                columns=PARAMETER_STRINGS,
                name=r"$F_{\Sigma^{-1}}$" + name,
                shade_alpha=0.
            )
        )

    # c.add_marker(
    #     location=marker(ALPHA, parameter_strings=PARAMETER_STRINGS),
    #     name=r"$\alpha$", 
    #     color="#7600bc"
    # )
    c.add_truth(
        Truth(location=dict(zip(PARAMETER_STRINGS, ALPHA)), name=r"$\pi^0$")
    )
    fig = c.plotter.plot()
    plt.savefig(os.path.join(log_figs_dir, "Fisher_tests.pdf"))
    plt.close()

    print("FISHER TESTS FILENAME:\n\t", os.path.join(log_figs_dir, "Fisher_tests.pdf"))

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ DEBUG

    assert len(summaries) == len(multi_z_args.redshifts)
    assert all([len(summaries[i]) == multi_z_args.n_datavectors for i in range(len(summaries))])

    # Multi-redshift ensemble of individual ensembles at each redshift
    multi_ensemble = MultiEnsemble(ensembles, prior=prior) 

    logger.info("MULTI-ENSEMBLE:{}".format(multi_ensemble))

    bulk_Finv_all_z = jnp.linalg.inv(F_bulk) 
    tails_Finv_all_z = jnp.linalg.inv(F_tails) 

    # Choose 'main' Finv for sampling
    assert multi_z_args.bulk_or_tails in ["bulk", "tails"]

    if multi_z_args.bulk_or_tails == "bulk":
        Finv_all_z = bulk_Finv_all_z # All of these are scaled by n_datavectors
        Finvs = bulk_Finvs
    if multi_z_args.bulk_or_tails == "tails":
        Finv_all_z = tails_Finv_all_z
        Finvs = tails_Finvs

    if 1:
        # Plot Fisher forecasts over all redshifts
        for i, (z, bulk_Finv_z, tails_Finv_z) in enumerate(zip(multi_z_args.redshifts, bulk_Finvs, tails_Finvs)):
            c = ChainConsumer()
            c.add_chain(
                Chain.from_covariance(
                    ALPHA,
                    tails_Finv_z,
                    columns=PARAMETER_STRINGS,
                    name=r"$F_{\Sigma^{-1}}$ z=" + str(z),
                    color="r",
                    shade_alpha=0.
                )
            )
            c.add_chain(
                Chain.from_covariance(
                    ALPHA,
                    bulk_Finv_z,
                    columns=PARAMETER_STRINGS,
                    name=r"$F_{\Sigma^{-1}}$ z=" + str(z) + " [bulk]",
                    color="b",
                    linestyle=":",
                    shade_alpha=0.
                )
            )
            c.add_chain(
                Chain(
                    samples=make_df(X0s[i], parameter_strings=PARAMETER_STRINGS),
                    name="NN summaries"
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
                    ALPHA,
                    tails_Finv_z,
                    columns=PARAMETER_STRINGS,
                    name=r"$F_{\Sigma^{-1}}$ z=" + str(z),
                    color="r",
                    shade_alpha=0.
                )
            )
        c.add_chain(
            Chain.from_covariance(
                ALPHA,
                tails_Finv_all_z,
                columns=PARAMETER_STRINGS,
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
                    ALPHA,
                    bulk_Finv_z,
                    columns=PARAMETER_STRINGS,
                    name=r"$F_{\Sigma^{-1}}$ z=" + str(z) + " [bulk]",
                    color="b",
                    shade_alpha=0.
                )
            )
        c.add_chain(
            Chain.from_covariance(
                ALPHA,
                bulk_Finv_all_z,
                columns=PARAMETER_STRINGS,
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
                ALPHA,
                tails_Finv_all_z,
                columns=PARAMETER_STRINGS,
                name=r"$F_{\Sigma^{-1}}$ (all z) [tails]",
                color="r",
                shade_alpha=0.
            )
        )
        c.add_chain(
            Chain.from_covariance(
                ALPHA,
                bulk_Finv_all_z,
                columns=PARAMETER_STRINGS,
                name=r"$F_{\Sigma^{-1}}$ (all z) [bulk]",
                color="b",
                shade_alpha=0.
            )
        )
        c.add_marker(
            location=marker(ALPHA, parameter_strings=PARAMETER_STRINGS),
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

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ DEBUG

    """
        SBI sample 
    """

    print("Sampling posterior (all redshifts, datavectors)")

    # Sample the multiple-redshift-ensemble posterior
    key_sample, key_mcmc_sample = jr.split(jr.key(int(time.time())))

    # Sample posterior across multiple redshifts (NOTE: prior defined above)
    log_prob_fn = multi_ensemble.get_multi_ensemble_log_prob_fn(summaries)

    samples, samples_log_prob = nuts_sample(
        key_sample, 
        log_prob_fn, # NOTE: implement with an NN?, 
        initial_state=jnp.asarray(ALPHA[jnp.newaxis, :]), 
        n_samples=2 * N_NUTS_SAMPLES
    )
    samples = jnp.squeeze(samples) # NOTE: if n_chains != 1 ...
    samples_log_prob = jnp.squeeze(samples_log_prob)
    samples_log_prob = finite_samples_log_prob(samples_log_prob) # all 
    alpha_log_prob = log_prob_fn(jnp.asarray(ALPHA))

    summaries_all_z = np.stack(summaries, axis=0) # NOTE: (n_z, n_datavectors, n_x) ?
    logger.info("SUMMARIES ALL Z:{}".format(summaries_all_z.shape)) #  >> SUMMARIES ALL Z:(3, 10, 5)

    # Save posterior, Fisher and summary
    posterior_filename = get_multi_z_posterior_filename(multi_z_args)

    np.savez(
        posterior_filename,
        samples=samples, 
        samples_log_prob=samples_log_prob,
        Finv=Finv_all_z,
        datavectors=datavectors,
        summaries=summaries_all_z,
        alpha_log_prob=alpha_log_prob
    )

    print("SAVED SBI MULTI-Z POSTERIOR AT:\n\t{}".format(posterior_filename))

    print("MEAN FISHER VARIANCE SIGMA_8 BULK/TAILS | SAMPLES:", np.sqrt(np.diag(Finv_all_z))[4], np.std(samples, axis=0)[4])
    print("MEAN FISHER VARIANCE SIGMA_8 BULK:", np.sqrt(np.diag(bulk_Finv_all_z))[4])
    print("MEAN FISHER VARIANCE SIGMA_8 TAILS:", np.sqrt(np.diag(bulk_Finv_all_z))[4])

    """
        MCMC sample with linear model
    """

    # Don't rescale by Finv for MCMC
    covariances_mcmc = [_C * multi_z_args.n_datavectors for _C in covariances]
    precisions_mcmc = [jnp.linalg.inv(_C) for _C in covariances_mcmc]
    Finvs_mcmc = [_Finv * multi_z_args.n_datavectors for _Finv in Finvs]


    @typecheck
    @eqx.filter_jit
    def mcmc_log_prob_fn_compressed_linearised(pi: Float[Array, "p"]) -> Scalar: 
        # Compressed data likelihood, assuming a Gaussian linear model, 
        # equivalent in function to the multi-ensemble-SBI likelihood.

        @typecheck
        def _log_prob_fn_z(
            d: Float[Array, "n d"], 
            Finv: Float[Array, "p p"], 
            compressor: Callable
        ) -> Float[Array, "n"]:

            # Assumes all datavectors drawn at alpha
            def _posterior(_pi_, pi):
                return jax.scipy.stats.multivariate_normal.logpdf(_pi_, pi, Finv)

            pi_ = jax.vmap(compressor, in_axes=(0, None))(d, pi)

            return jax.vmap(_posterior, in_axes=(0, None))(pi_, pi) # Vmap over multiple summaries

        # Tree map over lists of ingredients for each redshift
        Ls = jax.tree.map(
            lambda d, Finv, compressor: _log_prob_fn_z(d, Finv, compressor), 
            datavectors, 
            Finvs_mcmc,
            linear_compression_fns
        )

        prior_log_prob = prior.log_prob(pi)

        # return jnp.sum(jnp.asarray(Ls)) + prior_log_prob # NOTE: Correct sum? LogSumExp?
        # return sum(Ls) + prior_log_prob # NOTE: Correct sum? LogSumExp?
        return jnp.squeeze(jax.tree.reduce(operator.add, Ls) + prior_log_prob)


    mcmc_samples, mcmc_samples_log_prob = nuts_sample(
        key_mcmc_sample, 
        log_prob_fn=lambda theta: mcmc_log_prob_fn_compressed_linearised(pi=theta), # NOTE: implement with an NN?, 
        initial_state=jnp.asarray(ALPHA[jnp.newaxis, :]), 
    )
    mcmc_samples = jnp.squeeze(mcmc_samples) # NOTE: if n_chains != 1 ...
    mcmc_samples_log_prob = jnp.squeeze(mcmc_samples_log_prob)
    mcmc_samples_log_prob = finite_samples_log_prob(mcmc_samples_log_prob) # all 
    mcmc_alpha_log_prob = mcmc_log_prob_fn_compressed_linearised(jnp.asarray(ALPHA))

    # Save posterior, Fisher and summary
    mcmc_posterior_filename = get_multi_z_posterior_filename(multi_z_args, mcmc=True)

    np.savez(
        mcmc_posterior_filename,
        samples=mcmc_samples, 
        samples_log_prob=mcmc_samples_log_prob,
        Finv=Finv_all_z,
        datavectors=datavectors,
        summaries=summaries_all_z,
        alpha_log_prob=mcmc_alpha_log_prob
    )

    print("SAVED MCMC MULTI-Z POSTERIOR AT:\n\t{}".format(mcmc_posterior_filename))

    """
        Full posterior CLIPPED
    """

    c = ChainConsumer() 

    c.add_chain(
        Chain(
            samples=get_fisher_chain_df(
                ALPHA, 
                tails_Finv_all_z, 
                prior_clip=True
            ), 
            name=r"$F_{\Sigma^{-1}}$ (all z) $k_n$[tails]", 
            color="r",
            linestyle=":",
            shade_alpha=0.
        )
    )
    c.add_chain(
        Chain(
            samples=get_fisher_chain_df(
                ALPHA, 
                bulk_Finv_all_z,
                prior_clip=True
            ), 
            name=r"$F_{\Sigma^{-1}}$ (all z) $k_n$[bulk]", 
            color="b",
            linestyle=":",
            shade_alpha=0.
        )
    )
    c.add_chain(
        Chain(
            samples=get_fisher_chain_df(
                ALPHA, 
                Finv_bulk_pdfs_all_z,
                prior_clip=True
            ), 
            name=r"$F_{\Sigma^{-1}}$ (all z) PDF[bulk]", 
            color="g",
            linestyle=":",
            shade_alpha=0.
        )
    )

    posterior_df = make_df(
        samples, samples_log_prob, parameter_strings=PARAMETER_STRINGS
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
        for z, _summary_z in zip(multi_z_args.redshifts, summaries_all_z.mean(axis=1)):
            c.add_marker(
                location=marker(_summary_z, PARAMETER_STRINGS), 
                name=r"$\hat{x}$ " + str(z), 
                color="r" if multi_z_args.bulk_or_tails == "tails" else "b"
            )

        # summaries_all_z.shape = (3, 10, 5)
        mean_summaries_all_z = np.mean(np.mean(summaries_all_z, axis=1), axis=0)
        print("mean_summaries_all_z", mean_summaries_all_z.shape) # (3, 5) ; 3 redshifts
        c.add_marker(
            location=marker(mean_summaries_all_z, PARAMETER_STRINGS), 
            name=r"$\bar{x}$ fresh", 
            color="r" if multi_z_args.bulk_or_tails == "tails" else "b",
            marker_style="s"
        )

    # True parameters
    # c.add_marker(
    #     location=marker(ALPHA, PARAMETER_STRINGS), 
    #     name=r"$\alpha$", 
    #     color="#7600bc"
    # )
    c.add_truth(
        Truth(location=dict(zip(PARAMETER_STRINGS, ALPHA)), name=r"$\pi^0$")
    )

    fig = c.plotter.plot()
    fig = customize_plot(fig)
    fig.suptitle(
        r"{} SBI & $F_{{\Sigma}}^{{-1}}$".format("$k_n$") + "\n" +
        "{} z={},\n $n_s$={}, (pre-train $n_s$={}),\n R={} Mpc,\n $k_n$={}".format(
                ("linearised" if multi_z_args.linearised else "non-linear") + "\n",
                "[{}]".format(", ".join(map(str, multi_z_args.redshifts))),
                multi_z_args.n_linear_sims if multi_z_args.linearised else N_LINEAR_SIMS, 
                multi_z_args.n_linear_sims if multi_z_args.pre_train else None,
                "[{}]".format(", ".join(map(str, multi_z_args.scales))),
                "[{}]".format(", ".join(map(str, [get_cumulant_names()[_] for _ in multi_z_args.order_idx])))
            ),
        multialignment='center'
    )
    fig.savefig(
        os.path.join(
            figs_dir, 
            "multi_ensemble_posterior_cumulants_{}_{}_{}{}_{}.pdf".format(
                multi_z_args.seed, 
                linear_str, 
                pretrain_str, 
                ("_" + str(multi_z_args.seed_datavector)) if multi_z_args.seed_datavector is not None else "",
                "_blackjax"
            )
        )
    )
    plt.close()



    # TEST LINEARISED MCMC

    def test_linearised_mcmc():

        try:
            @typecheck
            @eqx.filter_jit
            def mcmc_log_prob_fn_compressed_linearised(pi: Float[Array, "p"]) -> Scalar: 
                # Compressed data likelihood, equivalent to multi-ensemble-SBI likelihood

                @typecheck
                def _log_prob_fn_z(
                    d: Float[Array, "n d"], 
                    Finv: Float[Array, "p p"], 
                    compressor: Callable
                ) -> Float[Array, "n"]:

                    # Assumes all datavectors drawn at alpha
                    def _posterior(_pi_, pi):
                        return jax.scipy.stats.multivariate_normal.logpdf(_pi_, pi, Finv)

                    pi_ = jax.vmap(compressor, in_axes=(0, None))(d, pi)

                    return jax.vmap(_posterior, in_axes=(0, None))(pi_, pi) # Vmap over multiple summaries

                # Tree map over lists of ingredients for each redshift
                Ls = jax.tree.map(
                    lambda d, Finv, compressor: _log_prob_fn_z(d, Finv, compressor), 
                    datavectors, 
                    Finvs_mcmc,
                    linear_compression_fns
                )

                prior_log_prob = prior.log_prob(pi)

                # return jnp.sum(jnp.asarray(Ls)) + prior_log_prob # NOTE: Correct sum? LogSumExp?
                # return sum(Ls) + prior_log_prob # NOTE: Correct sum? LogSumExp?
                return jnp.squeeze(jax.tree.reduce(operator.add, Ls) + prior_log_prob)


            mcmc_samples, mcmc_samples_log_prob = nuts_sample(
                key_sample, 
                log_prob_fn=lambda theta: mcmc_log_prob_fn_compressed_linearised(pi=theta), # NOTE: implement with an NN?, 
                initial_state=jnp.asarray(ALPHA[jnp.newaxis, :]), 
            )
            mcmc_samples = jnp.squeeze(mcmc_samples) # NOTE: if n_chains != 1 ...
            mcmc_samples_log_prob = jnp.squeeze(mcmc_samples_log_prob)
            mcmc_samples_log_prob = finite_samples_log_prob(mcmc_samples_log_prob) # all 
            mcmc_alpha_log_prob = mcmc_log_prob_fn_compressed(jnp.asarray(ALPHA))

            c = ChainConsumer() 

            c.add_chain(
                Chain(
                    samples=get_fisher_chain_df(
                        ALPHA, 
                        tails_Finv_all_z, 
                        prior_clip=True
                    ), 
                    name=r"$F_{\Sigma^{-1}}$ (all z) $k_n$[tails]", 
                    color="r",
                    linestyle=":",
                    shade_alpha=0.
                )
            )
            c.add_chain(
                Chain(
                    samples=get_fisher_chain_df(
                        ALPHA, 
                        bulk_Finv_all_z,
                        prior_clip=True
                    ), 
                    name=r"$F_{\Sigma^{-1}}$ (all z) $k_n$[bulk]", 
                    color="b",
                    linestyle=":",
                    shade_alpha=0.
                )
            )
            c.add_chain(
                Chain(
                    samples=get_fisher_chain_df(
                        ALPHA, 
                        Finv_bulk_pdfs_all_z,
                        prior_clip=True
                    ), 
                    name=r"$F_{\Sigma^{-1}}$ (all z) PDF[bulk]", 
                    color="g",
                    linestyle=":",
                    shade_alpha=0.
                )
            )
            posterior_df = make_df(
                mcmc_samples, mcmc_samples_log_prob, parameter_strings=PARAMETER_STRINGS
            )
            c.add_chain(
                Chain(
                    samples=posterior_df, 
                    name="MCMC[{}]".format(multi_z_args.bulk_or_tails), 
                    color="r" if multi_z_args.bulk_or_tails == "tails" else "b"
                )
            )

            if PLOT_SUMMARIES:
                # If using multiple datavectors, plot them individually
                for z, _summary_z in zip(multi_z_args.redshifts, summaries_all_z.mean(axis=1)):
                    c.add_marker(
                        location=marker(_summary_z, PARAMETER_STRINGS), 
                        name=r"$\hat{x}$ " + str(z), 
                        color="r" if multi_z_args.bulk_or_tails == "tails" else "b"
                    )

                # summaries_all_z.shape = (3, 10, 5)
                mean_summaries_all_z = np.mean(np.mean(summaries_all_z, axis=1), axis=0)
                print("mean_summaries_all_z", mean_summaries_all_z.shape) # (3, 5) ; 3 redshifts
                c.add_marker(
                    location=marker(mean_summaries_all_z, PARAMETER_STRINGS), 
                    name=r"$\bar{x}$ fresh", 
                    color="r" if multi_z_args.bulk_or_tails == "tails" else "b",
                    marker_style="s"
                )

            # True parameters
            c.add_truth(
                Truth(location=dict(zip(PARAMETER_STRINGS, ALPHA)), name=r"$\pi^0$")
            )

            fig = c.plotter.plot()
            fig = customize_plot(fig)
            fig.suptitle(
                r"{} SBI & $F_{{\Sigma}}^{{-1}}$".format("$k_n$") + "\n" +
                "{} z={},\n $n_s$={}, (pre-train $n_s$={}),\n R={} Mpc,\n $k_n$={}".format(
                        ("linearised" if multi_z_args.linearised else "non-linear") + "\n",
                        "[{}]".format(", ".join(map(str, multi_z_args.redshifts))),
                        multi_z_args.n_linear_sims if multi_z_args.linearised else N_LINEAR_SIMS, 
                        multi_z_args.n_linear_sims if multi_z_args.pre_train else None,
                        "[{}]".format(", ".join(map(str, multi_z_args.scales))),
                        "[{}]".format(", ".join(map(str, [get_cumulant_names()[_] for _ in multi_z_args.order_idx])))
                    ),
                multialignment='center'
            )
            fig.savefig(
                os.path.join(
                    figs_dir, 
                    "multi_linearised_test_posterior_cumulants_{}_{}_{}{}_{}.pdf".format(
                        multi_z_args.seed, 
                        linear_str, 
                        pretrain_str, 
                        ("_" + str(multi_z_args.seed_datavector)) if multi_z_args.seed_datavector is not None else "",
                        "_blackjax"
                    )
                )
            )
        except Exception as e:
            print("TEST MCMC ERROR:", e)


    test_linearised_mcmc()



    mean_summaries_all_z = np.mean(np.mean(summaries_all_z, axis=1), axis=0)
    c.add_marker(
        location=marker(mean_summaries_all_z, PARAMETER_STRINGS), 
        name=r"$\hat{\pi}$", 
        marker_style="x",
        color="r" if multi_z_args.bulk_or_tails == "tails" else "b",
    )

    fig = c.plotter.plot()
    fig = customize_plot(fig)
    fig.suptitle(
        r"{} SBI & $F_{{\Sigma}}^{{-1}}$".format("$k_n$") + "\n" +
        "{} z={},\n $n_s$={}, (pre-train $n_s$={}),\n R={} Mpc,\n $k_n$={}".format(
                ("linearised" if multi_z_args.linearised else "non-linear") + "\n",
                "[{}]".format(", ".join(map(str, multi_z_args.redshifts))),
                multi_z_args.n_linear_sims if multi_z_args.linearised else N_LINEAR_SIMS, 
                multi_z_args.n_linear_sims if multi_z_args.pre_train else None,
                "[{}]".format(", ".join(map(str, multi_z_args.scales))),
                "[{}]".format(", ".join(map(str, [get_cumulant_names()[_] for _ in multi_z_args.order_idx])))
            ),
        multialignment='center'
    )
    fig.savefig(
        os.path.join(
            figs_dir, 
            "multi_ensemble_posterior_cumulants_{}_{}_{}{}_{}.pdf".format(
                multi_z_args.seed, 
                linear_str, 
                pretrain_str, 
                ("_" + str(multi_z_args.seed_datavector)) if multi_z_args.seed_datavector is not None else "",
                "_blackjax",
                "_summary"
            )
        )
    )

    plt.close()


    """
        Full posterior (clipped fisher)
    """

    c = ChainConsumer() 

    # Tails 
    fisher_samples = np.random.multivariate_normal(
        ALPHA, tails_Finv_all_z, (N_FISHER_SAMPLES,) 
    ) 
    fisher_samples_log_prob = jax.scipy.stats.multivariate_normal.logpdf(
        fisher_samples, ALPHA, tails_Finv_all_z
    )
    fisher_df = make_df(
        cut_samples(fisher_samples, LOWER, UPPER),
        parameter_strings=PARAMETER_STRINGS
    )
    c.add_chain(
        Chain(
            samples=fisher_df, 
            name=r"$F_{\Sigma^{-1}}$ (all z) $k_n$[tails]", 
            color="r",
            linestyle=":",
            shade_alpha=0.
        )
    )

    # Bulk
    fisher_samples = np.random.multivariate_normal(
        ALPHA, bulk_Finv_all_z, (N_FISHER_SAMPLES,) 
    ) 
    fisher_samples_log_prob = jax.scipy.stats.multivariate_normal.logpdf(
        fisher_samples, ALPHA, bulk_Finv_all_z
    )
    fisher_df = make_df(
        cut_samples(fisher_samples, LOWER, UPPER),
        parameter_strings=PARAMETER_STRINGS
    )
    c.add_chain(
        Chain(
            samples=fisher_df, 
            name=r"$F_{\Sigma^{-1}}$ (all z) $k_n$[bulk]", 
            color="b",
            linestyle=":",
            shade_alpha=0.
        )
    )

    # PDF
    fisher_samples = np.random.multivariate_normal(
        ALPHA, Finv_bulk_pdfs_all_z, (N_FISHER_SAMPLES,) 
    ) 
    fisher_samples_log_prob = jax.scipy.stats.multivariate_normal.logpdf(
        fisher_samples, ALPHA, Finv_bulk_pdfs_all_z
    )
    fisher_df = make_df(
        cut_samples(fisher_samples, LOWER, UPPER),
        parameter_strings=PARAMETER_STRINGS
    )
    c.add_chain(
        Chain(
            samples=fisher_df, 
            name=r"$F_{\Sigma^{-1}}$ (all z) PDF[bulk]", 
            color="g",
            linestyle=":",
            shade_alpha=0.
        )
    )

    posterior_df = make_df(
        samples, samples_log_prob, parameter_strings=PARAMETER_STRINGS
    )
    c.add_chain(
        Chain(
            samples=posterior_df, 
            name="SBI[{}]".format(multi_z_args.bulk_or_tails), 
            color="r" if multi_z_args.bulk_or_tails == "tails" else "b"
        )
    )
    mcmc_posterior_df = make_df(
        mcmc_samples, mcmc_samples_log_prob, parameter_strings=PARAMETER_STRINGS
    )
    c.add_chain(
        Chain(
            samples=mcmc_posterior_df, 
            name="MCMC[{}]".format(multi_z_args.bulk_or_tails), 
            color="#9867C5"
        )
    )

    if PLOT_SUMMARIES:

        mean_summaries_all_z = np.mean(np.mean(summaries_all_z, axis=1), axis=0)

        print("mean_summaries_all_z", mean_summaries_all_z.shape) # (3, 5) ; 3 redshifts

        c.add_marker(
            location=marker(mean_summaries_all_z, PARAMETER_STRINGS), 
            name=r"$\bar{x}$ fresh", 
            color="r" if multi_z_args.bulk_or_tails == "tails" else "b",
            marker_style="s"
        )

    c.add_truth(
        Truth(location=dict(zip(PARAMETER_STRINGS, ALPHA)), name=r"$\pi^0$")
    )
    # c.add_marker(
    #     location=marker(ALPHA, PARAMETER_STRINGS), 
    #     name=r"$\alpha$", 
    #     color="#7600bc"
    # )

    fig = c.plotter.plot()
    overlay_bounds_on_corner(fig, LOWER, UPPER)
    fig = customize_plot(fig)
    fig.suptitle(
        r"{} SBI & $F_{{\Sigma}}^{{-1}}$".format("$k_n$") + "\n" +
        "{} z={},\n $n_s$={}, (pre-train $n_s$={}),\n R={} Mpc,\n $k_n$={}".format(
                ("linearised" if multi_z_args.linearised else "non-linear") + "\n",
                "[{}]".format(", ".join(map(str, multi_z_args.redshifts))),
                multi_z_args.n_linear_sims if multi_z_args.linearised else N_LINEAR_SIMS, 
                multi_z_args.n_linear_sims if multi_z_args.pre_train else None,
                "[{}]".format(", ".join(map(str, multi_z_args.scales))),
                "[{}]".format(", ".join(map(str, [get_cumulant_names()[_] for _ in multi_z_args.order_idx])))
            ),
        multialignment='center'
    )
    plt.savefig(
        os.path.join(
            figs_dir, 
            "multi_ensemble_posterior_cumulants_clipped_fisher_blackjax_{}_{}_{}_{}.pdf".format(
                multi_z_args.seed, 
                linear_str, 
                pretrain_str, 
                str(multi_z_args.seed_datavector) if multi_z_args.seed_datavector is not None else "",
                "blackjax"
            )
        )
    )
    plt.close()

    """
        Plot all summaries
    """

    if 1:
        c = ChainConsumer()

        colors = SUMMARIES_PLOT_COLOURS[multi_z_args.bulk_or_tails] #["#7FFF00", "#FF8C00", "#C71585"]
        for i, redshift in enumerate(multi_z_args.redshifts): 
            c.add_chain(
                Chain.from_covariance(
                    ALPHA,
                    # NOTE: plotting scaled ones now
                    Finvs[i] * multi_z_args.n_datavectors, # Bulk or tails?
                    columns=PARAMETER_STRINGS,
                    name=r"$F_{\Sigma^{-1}}$" + " (z={})".format(redshift),
                    color=colors[i],
                    linestyle="-",
                    shade_alpha=0.
                )
            )

            summaries_z = summaries[i] # List over redshift of stacked summaries
            assert len(summaries[i]) == multi_z_args.n_datavectors
            for s, summary in enumerate(summaries_z):
                c.add_marker(
                    location=marker(summary, PARAMETER_STRINGS), 
                    name="$x_z={}, {}$".format(redshift, s), 
                    color=colors[i]
                )

        c.add_chain(
            Chain.from_covariance(
                ALPHA,
                Finv_all_z,# * multi_z_args.n_datavectors, # Bulk or tails?
                columns=PARAMETER_STRINGS,
                name=r"$F_{\Sigma^{-1}}$" + " (z=all)",
                color="k",
                linestyle="-",
                shade_alpha=0.
            )
        )

        c.add_truth(
            Truth(location=dict(zip(PARAMETER_STRINGS, ALPHA)), name=r"$\pi^0$")
        )
        # c.add_marker(
        #     location=marker(ALPHA, PARAMETER_STRINGS), 
        #     name=r"$\alpha$", 
        #     color="#7600bc"
        # )

        fig = c.plotter.plot()
        overlay_bounds_on_corner(fig, LOWER, UPPER)
        fig = customize_plot(fig)
        fig.suptitle(
            r"{} SBI & $F_{{\Sigma}}^{{-1}}$".format("$k_n$") + "\n" +
            "{} z={},\n $n_s$={}, (pre-train $n_s$={}),\n R={} Mpc,\n $k_n$={}".format(
                    ("linearised" if multi_z_args.linearised else "non-linear") + "\n",
                    "[{}]".format(", ".join(map(str, multi_z_args.redshifts))),
                    multi_z_args.n_linear_sims if multi_z_args.linearised else N_LINEAR_SIMS, 
                    multi_z_args.n_linear_sims if multi_z_args.pre_train else None,
                    "[{}]".format(", ".join(map(str, multi_z_args.scales))),
                    "[{}]".format(", ".join(map(str, [get_cumulant_names()[_] for _ in multi_z_args.order_idx])))
                ),
            multialignment='center'
        )
        posterior_plot_filename = os.path.join(
            figs_dir, 
            "summaries_and_Finvs_unscaled_{}_{}_{}{}_mcmc.pdf".format(
                multi_z_args.seed, 
                linear_str, 
                pretrain_str,
                ("_" + str(multi_z_args.seed_datavector)) if multi_z_args.seed_datavector is not None else ""
            )
        )
        plt.savefig(posterior_plot_filename)
        plt.close()


        c = ChainConsumer()

        colors = SUMMARIES_PLOT_COLOURS[multi_z_args.bulk_or_tails] #["#7FFF00", "#FF8C00", "#C71585"]
        for i, redshift in enumerate(multi_z_args.redshifts): 
            fisher_samples = np.random.multivariate_normal(
                ALPHA, Finvs[i], (N_FISHER_SAMPLES,) 
            ) 
            fisher_df = make_df(
                cut_samples(fisher_samples, LOWER, UPPER),
                parameter_strings=PARAMETER_STRINGS
            )
            c.add_chain(
                Chain(
                    samples=fisher_df, 
                    name=r"$F_{\Sigma^{-1}}$" + " (z={})".format(redshift),
                    color=colors[i],
                    linestyle="-",
                    shade_alpha=0.
                )
            )

            summaries_z = summaries[i] # List over redshift of stacked summaries
            assert len(summaries[i]) == multi_z_args.n_datavectors
            for s, summary in enumerate(summaries_z):
                c.add_marker(
                    location=marker(summary, PARAMETER_STRINGS), 
                    name="$x_z={}, {}$".format(redshift, s), 
                    color=colors[i]
                )

        fisher_samples = np.random.multivariate_normal(
            ALPHA, Finv_all_z, (N_FISHER_SAMPLES,)
        ) 
        fisher_df = make_df(
            cut_samples(fisher_samples, LOWER, UPPER), 
            parameter_strings=PARAMETER_STRINGS
        )
        c.add_chain(
            Chain(
                samples=fisher_df, 
                name=r"$F_{\Sigma^{-1}}$" + " (z=all)",
                color="k",
                linestyle="-",
                shade_alpha=0.
            )
        )

        c.add_truth(
            Truth(location=dict(zip(PARAMETER_STRINGS, ALPHA)), name=r"$\pi^0$")
        )
        # c.add_marker(
        #     location=marker(ALPHA, PARAMETER_STRINGS), 
        #     name=r"$\alpha$", 
        #     color="#7600bc"
        # )

        fig = c.plotter.plot()
        overlay_bounds_on_corner(fig, LOWER, UPPER)
        fig = customize_plot(fig)
        fig.suptitle(
            r"{} SBI & $F_{{\Sigma}}^{{-1}}$".format("$k_n$") + "\n" +
            "{} z={},\n $n_s$={}, (pre-train $n_s$={}),\n R={} Mpc,\n $k_n$={}".format(
                ("linearised" if multi_z_args.linearised else "non-linear") + "\n",
                "[{}]".format(", ".join(map(str, multi_z_args.redshifts))),
                multi_z_args.n_linear_sims if multi_z_args.linearised else N_LINEAR_SIMS, 
                multi_z_args.n_linear_sims if multi_z_args.pre_train else None,
                "[{}]".format(", ".join(map(str, multi_z_args.scales))),
                "[{}]".format(", ".join(map(str, [get_cumulant_names()[_] for _ in multi_z_args.order_idx])))
            ),
            multialignment='center'
        )
        posterior_plot_filename = os.path.join(
            figs_dir, 
            "summaries_and_Finvs_clipped_{}_{}_{}{}_mcmc.pdf".format(
                multi_z_args.seed, 
                linear_str, 
                pretrain_str,
                ("_" + str(multi_z_args.seed_datavector)) if multi_z_args.seed_datavector is not None else ""
            )
        )
        plt.savefig(posterior_plot_filename)
        plt.close()

    print("Summaries plot saved at:\n\t", posterior_plot_filename)

print("Done.")


# OLD COMPRESSION LOADING
    # if compression == "linear":
    #     compression_fn = cumulants_dataset.get_compression_fn(train=False)

    #     x_ = jax.vmap(compression_fn, in_axes=(0, None))(datavectors, cumulants_dataset.data.alpha) 

    # if compression == "nn":
       
    #     # Grab same architecture from nn.py
    #     net = get_default_nn(cumulants_dataset.data, config_z, key)

    #     # assert DATA_PROCESS_TYPE_NN in ["d", "p", "dp", None], (
    #     #     "data_process_type_nn={}".format(DATA_PROCESS_TYPE_NN)
    #     # )
    #     # logger.info("DATA_PROCESS_TYPE_NN: {}".format(DATA_PROCESS_TYPE_NN))

    #     # Assuming same form of data/parameter pre/post-processing functions
    #     # preprocess_fn_d, preprocess_fn_p, postprocess_fn_p = get_nn_process_fns(
    #     #     cumulants_dataset.data, 
    #     #     data_process_type_nn=DATA_PROCESS_TYPE_NN # How to normalise data, parameters
    #     # )

    #     nn_path = os.path.join(results_dir_z, "nn.eqx")

    #     net = eqx.tree_deserialise_leaves(nn_path, net)

    #     logger.info("Loaded compression net from:\n\t{}".format(nn_path))

    #     # compression_fn = lambda d, p: postprocess_fn_p(net(preprocess_fn_d(d)))
    #     def compression_fn(d, p): 
    #         # return postprocess_fn_p(net(preprocess_fn_d(d))) # Ignore parameter kwarg for NN
    #         return net(d) # Ignore parameter kwarg for NN

    #     x_ = jax.vmap(compression_fn, in_axes=(0, None))(datavectors, cumulants_dataset.data.alpha)
        
    # if compression == "ensemble-nn":

    #     net = get_default_nn(cumulants_dataset.data, config_z, key)

    #     # assert DATA_PROCESS_TYPE_NN in ["d", "p", "dp", None], (
    #     #     "data_process_type_nn={}".format(DATA_PROCESS_TYPE_NN)
    #     # )
    #     # logger.info("DATA_PROCESS_TYPE_NN: {}".format(DATA_PROCESS_TYPE_NN))

    #     # # Assuming same form of data/parameter pre/post-processing functions
    #     # preprocess_fn_d, preprocess_fn_p, postprocess_fn_p = get_nn_process_fns(
    #     #     cumulants_dataset.data, 
    #     #     data_process_type_nn=DATA_PROCESS_TYPE_NN # How to normalise data, parameters
    #     # )

    #     nets = []
    #     for n in range(N_ENSEMBLE_NETS):
    #         nn_path = os.path.join(results_dir_z, "nn_{}.eqx".format(n))

    #         net_n = eqx.tree_deserialise_leaves(nn_path, net)

    #         net_n = eqx.nn.inference_mode(net_n, True)

    #         nets.append(net_n)

    #         logger.info("Loaded compression net from:\n\t{}".format(nn_path))

    #     class EnsembleNet(eqx.Module):
    #         nets: list[eqx.Module]
    #         def __init__(self, nets):
    #             self.nets = nets
    #         def __call__(self, d):
    #             xs = jax.tree.map(
    #                 lambda d, net: net(d), [d] * len(self.nets), self.nets
    #             )
    #             return jnp.mean(jnp.asarray(xs), axis=0)

    #     ensemble_net = EnsembleNet(nets)

    #     logger.info("Loaded compression net ensemble.")

    #     # compression_fn = lambda d, p: postprocess_fn_p(ensemble_net(preprocess_fn_d(d)))
    #     def compression_fn(d, p): 
    #         # return postprocess_fn_p(ensemble_net(preprocess_fn_d(d))) # Ignore parameter kwarg for NN
    #         return ensemble_net(d) # Ignore parameter kwarg for NN

    #     x_ = jax.vmap(compression_fn, in_axes=(0, None))(datavectors, cumulants_dataset.data.alpha)





    # """
    #     Marginalised posterior with MCMC
    #     - Marginalise over all but Om, s8
    # """
    # PARAMETER_STRINGS_ = [PARAMETER_STRINGS[_] for _ in target_idx]

    # if BLACKJAX_SAMPL_E:

    #     c = ChainConsumer()
    #     c.add_chain(
    #         Chain(
    #             samples=get_fisher_chain_df(
    #                 ALPHA[target_idx], 
    #                 tails_Finv_all_z[target_idx, :][:, target_idx], 
    #                 parameter_strings=PARAMETER_STRINGS_,
    #                 prior_clip=True # multi_z_args.linearised
    #             ), 
    #             name=r"$F_{\Sigma^{-1}}$ (all z) $k_n$[tails]", 
    #             color="r",
    #             linestyle=":",
    #             shade_alpha=0.
    #         )
    #     )
    #     c.add_chain(
    #         Chain(
    #             samples=get_fisher_chain_df(
    #                 ALPHA[target_idx], 
    #                 bulk_Finv_all_z[target_idx, :][:, target_idx], 
    #                 parameter_strings=PARAMETER_STRINGS_,
    #                 prior_clip=True # multi_z_args.linearised
    #             ), 
    #             name=r"$F_{\Sigma^{-1}}$ (all z) $k_n$[bulk]", 
    #             color="b",
    #             linestyle=":",
    #             shade_alpha=0.
    #         )
    #     )
    #     c.add_chain(
    #         Chain(
    #             samples=get_fisher_chain_df(
    #                 ALPHA[target_idx], 
    #                 Finv_bulk_pdfs_all_z[target_idx, :][:, target_idx], 
    #                 parameter_strings=PARAMETER_STRINGS_,
    #                 prior_clip=True #multi_z_args.linearised
    #             ), 
    #             name=r"$F_{\Sigma^{-1}}$ (all z) PDF[bulk]", 
    #             color="g",
    #             linestyle=":",
    #             shade_alpha=0.
    #         )
    #     )

    #     posterior_df = make_df(
    #         samples[:, target_idx], samples_log_prob, parameter_strings=PARAMETER_STRINGS_
    #     )
    #     c.add_chain(
    #         Chain(
    #             samples=posterior_df, 
    #             name="SBI[{}]".format(multi_z_args.bulk_or_tails), 
    #             color="r" if multi_z_args.bulk_or_tails == "tails" else "b"
    #         )
    #     )
    #     mcmc_posterior_df = make_df(
    #         mcmc_samples[:, target_idx], mcmc_samples_log_prob, parameter_strings=PARAMETER_STRINGS_
    #     )
    #     c.add_chain(
    #         Chain(
    #             samples=mcmc_posterior_df, 
    #             name="MCMC[{}]".format(multi_z_args.bulk_or_tails), 
    #             color="#9867C5"
    #         )
    #     )

    #     if PLOT_SUMMARIES:
    #         colors = ["#7FFF00", "#FF8C00", "#C71585"]
    #         for z in range(len(multi_z_args.redshifts)):
    #             x_s_z = np.mean(summaries_all_z[z, :, :], axis=0) # summaries_all_z.shape = (3, 10, 5)
    #             c.add_marker(
    #                 location=marker(x_s_z, PARAMETER_STRINGS_), 
    #                 name=r"$\bar{{x}}$ z={}".format(multi_z_args.redshifts[z]), 
    #                 color=colors[z], #"r" if multi_z_args.bulk_or_tails == "tails" else "b",
    #                 marker_style="x"
    #             )
    #     c.add_marker(
    #         location=marker(ALPHA[target_idx], PARAMETER_STRINGS_), 
    #         name=r"$\alpha$", 
    #         color="#7600bc"
    #     )

    #     fig = c.plotter.plot()
    #     overlay_bounds_on_corner(fig, LOWER, UPPER)
    #     fig.suptitle(
    #         r"{} SBI & $F_{{\Sigma}}^{{-1}}$".format("$k_n$") + "\n" +
    #         "{} z={},\n $n_s$={}, (pre-train $n_s$={}),\n R={} Mpc,\n $k_n$={}".format(
    #             ("linearised" if multi_z_args.linearised else "non-linear") + "\n",
    #             "[{}]".format(", ".join(map(str, multi_z_args.redshifts))),
    #             multi_z_args.n_linear_sims if multi_z_args.linearised else N_LINEAR_SIMS, 
    #             multi_z_args.n_linear_sims if multi_z_args.pre_train else None,
    #             "[{}]".format(", ".join(map(str, multi_z_args.scales))),
    #             "[{}]".format(", ".join(map(str, [get_cumulant_names()[_] for _ in multi_z_args.order_idx])))
    #         ),
    #         multialignment='center'
    #     )
    #     posterior_plot_filename = os.path.join(
    #         figs_dir, 
    #         "multi_ensemble_posterior_marginalised_cumulants_blackjax_{}_{}_{}{}_{}_mcmc.pdf".format(
    #             multi_z_args.seed, 
    #             linear_str, 
    #             pretrain_str, 
    #             ("_" + str(multi_z_args.seed_datavector)) if multi_z_args.seed_datavector is not None else "",
    #             "blackjax"
    #         )
    #     )
    #     plt.savefig(posterior_plot_filename)
    #     plt.close()

    #     print("MULTI-Z POSTERIOR PLOT FILENAME (MCMC):\n", posterior_plot_filename)





    # @typecheck
    # @eqx.filter_jit
    # def mcmc_log_prob_fn_compressed(pi: Float[Array, "p"]) -> Scalar: 
    #     # Compressed data likelihood, equivalent to multi-ensemble-SBI likelihood

    #     @typecheck
    #     def _log_prob_fn_z(
    #         pi_: Float[Array, "n d"], Finv: Float[Array, "p p"]
    #     ) -> Float[Array, "n"]:

    #         # Assumes all datavectors drawn at alpha
    #         def _posterior(_pi_, pi):
    #             # p = tfd.MultivariateNormalFullCovariance(loc=pi, covariance_matrix=Finv)  
    #             # return p.log_prob(_pi_)
    #             return jax.scipy.stats.multivariate_normal.logpdf(_pi_, pi, Finv)

    #         return jax.vmap(_posterior, in_axes=(0, None))(pi_, pi) # Vmap over multiple summaries

    #     assert (
    #         jax.tree.structure(summaries) == jax.tree.structure(Finvs_mcmc)
    #     ), (
    #         "Structure mismatch: summaries / Finvs_mcmc: {}, {}".format(
    #             jax.tree.structure(summaries), jax.tree.structure(Finvs_mcmc)
    #         )
    #     )

    #     # Tree map over lists of ingredients for each redshift
    #     Ls = jax.tree.map(lambda d, Finv: _log_prob_fn_z(d, Finv), summaries, Finvs_mcmc)

    #     prior_log_prob = prior.log_prob(pi)

    #     # Sum of log-likelihoods and prior probability
    #     return jnp.sum(jnp.asarray(Ls)) + prior_log_prob # NOTE: Correct sum? LogSumExp?



    # @typecheck
    # @eqx.filter_jit
    # def mcmc_log_prob_fn_compressed(pi: Float[Array, "p"]) -> Scalar: 
    #     # Compressed data likelihood, equivalent to multi-ensemble-SBI likelihood

    #     @typecheck
    #     def _log_prob_fn_z(
    #         d: Float[Array, "n d"], 
    #         Finv: Float[Array, "p p"], 
    #         compressor: Callable
    #     ) -> Float[Array, "n"]:

    #         # Assumes all datavectors drawn at alpha
    #         def _posterior(_pi_, pi):
    #             return jax.scipy.stats.multivariate_normal.logpdf(_pi_, pi, Finv)

    #         pi_ = jax.vmap(compressor, in_axes=(0, None))(d, pi)

    #         return jax.vmap(_posterior, in_axes=(0, None))(pi_, pi) # Vmap over multiple summaries

    #     # Tree map over lists of ingredients for each redshift
    #     Ls = jax.tree.map(
    #         lambda d, Finv, compressor: _log_prob_fn_z(d, Finv, compressor), 
    #         datavectors, 
    #         Finvs_mcmc,
    #         linear_compression_fns
    #     )

    #     prior_log_prob = prior.log_prob(pi)

    #     return jnp.sum(jnp.asarray(Ls)) + prior_log_prob # NOTE: Correct sum? LogSumExp?

