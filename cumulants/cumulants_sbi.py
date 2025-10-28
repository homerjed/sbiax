import os
import sys
import time
import datetime
from copy import deepcopy

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

import numpy as np 
import matplotlib.pyplot as plt
from chainconsumer import Chain, ChainConsumer, Truth

from sbiax.train import train_ensemble
from sbiax.utils import make_df, marker
from sbiax.inference import nuts_sample

from configs import (
    get_results_dir, 
    get_posteriors_dir, 
    get_ndes_from_config
)
from configs.log import setup_module_logger, get_log_level
from configs.args import get_cumulants_sbi_args
from data.constants import (
    get_cumulant_names, 
    N_S_HYPERCUBE, 
    ALPHA, 
    LOWER, 
    UPPER, 
    PARAMETER_STRINGS
)
from data.common import Dataset
from compression.nn import Processor, wrap_ensemble_ndes_with_processors
from utils import (
    get_datasets,
    finite_samples_log_prob,
    overlay_bounds_on_corner,
    customize_plot,
    cut_samples
)


jax.clear_caches()

logger, log_figs_dir = setup_module_logger(__name__, level=get_log_level())

NON_GAUSSIAN_TEST = True if os.environ.get("NON_GAUSSIAN_TEST", "").lower() in ("1", "true") else False

N_NUTS_SAMPLES = 10_000
N_FISHER_SAMPLES = 800_000
SAMPLE_POSTERIORS = True # Just leave it to multi-z, where debugging plots sample the posteriors anyway

""" 
    Run NLE or NPE SBI with the moments of the 1pt matter PDF.

    - diagonal of covariance for compression?
    - freezing 'nuisance parameters'
    - covariance conditioning?
    - remove outliers in latins?
""" 

t0 = time.time()

args = get_cumulants_sbi_args()

print("TIME:", datetime.datetime.now().strftime("%H:%M %d-%m-%y"))
print("SEED:", args.seed)
print("MOMENTS:", args.order_idx)
print("LINEARISED:", args.linearised)

"""
    Config
"""

config, cumulants_dataset, datasets = get_datasets(args) # Config and cumulants_dataset can be bulk ... etc

key = jr.key(config.seed)

(
    model_key, train_key, key_prior, 
    key_datavector, key_state, key_sample
) = jr.split(key, 6)

_seed = int(time.time()) + int(os.getpid())  # or + int(os.getenv("SLURM_JOB_ID", 0))
key_datavector = jr.key(_seed)

results_dir = get_results_dir(config, args)

posteriors_dir = get_posteriors_dir(args)

dataset: Dataset = cumulants_dataset.data

if args.compression != "linear":
    sbi_dataset = np.load(os.path.join(results_dir, "sbi_dataset.npz"))
    X = sbi_dataset["latins"]
    Y = sbi_dataset["parameters"]
    X0 = sbi_dataset["fiducials"]
    datavector = sbi_dataset["datavector"]
    x_ = sbi_dataset["summary"]
    x_noiseless = sbi_dataset["summary_noiseless"]
else:
    fn = cumulants_dataset.get_compression_fn(train=False)
    X = jax.vmap(fn)(dataset.data, dataset.parameters)
    Y = dataset.parameters
    X0 = jax.vmap(fn, in_axes=(0, None))(dataset.fiducial_data, dataset.alpha)
    datavector = dataset.fiducial_data[-1]
    x_ = X0[-1]
    x_noiseless = fn(dataset.fiducial_data.mean(axis=0), dataset.alpha)

"""
    Build NDEs
"""

ensemble = get_ndes_from_config(config, key=model_key)

"""
    Train NDE on data
"""

processor = Processor(X, Y)

# If ensemble exists, don't train it, load it
try:
    ensemble = eqx.tree_deserialise_leaves(
        os.path.join(results_dir, "ensemble.eqx"), ensemble
    )

    processor = eqx.tree_deserialise_leaves(
        os.path.join(results_dir, "processor_nde.eqx"), processor
    )

    print("LOADED ENSEMBLE")
except Exception as e:
    print("Exception (cumulants_sbi.py ensemble load): \n\t{}".format(e))

    opt = getattr(optax, config.train.opt)(config.train.lr)

    ensemble_copy = deepcopy(ensemble) # Just to check saving/loading multi-NDE ensembles works

    eqx.tree_serialise_leaves(
        os.path.join(results_dir, "processor_nde.eqx"), processor
    )

    train_data = (processor.forward_x(X), processor.forward_y(Y))

    ensemble, stats = train_ensemble(
        train_key, 
        ensemble,
        train_mode="nle",
        train_data=train_data, # Pre-processing done in NDEs `Scaler`
        opt=opt,
        n_batch=config.train.n_batch,
        patience=config.train.patience,
        n_epochs=config.train.n_epochs,
        valid_fraction=config.valid_fraction,
        tqdm_description="Training (data)",
        show_tqdm=args.use_tqdm,
        results_dir=results_dir
    )

    # Save and reload back into original pytree
    eqx.tree_serialise_leaves(
        os.path.join(results_dir, "ensemble.eqx"), ensemble
    )
    ensemble = eqx.tree_deserialise_leaves(
        os.path.join(results_dir, "ensemble.eqx"), ensemble_copy
    )

    print("SAVED AND LOADED ENSEMBLE")

# Wrap now because all we will do is sample here
ensemble = wrap_ensemble_ndes_with_processors(ensemble, processor)

""" 
    Sample and plot posterior for NDE with noisy datavectors
"""

logger.debug("datavector {} \n {}".format(datavector.shape, datavector))
logger.debug("compressed datavector {} \n {} {}".format(x_.shape, x_, ALPHA))

log_prob_fn = ensemble.ensemble_log_prob_fn(x_, cumulants_dataset.prior)

if SAMPLE_POSTERIORS:

    print("BLACKJAX SAMPLING...")
    
    samples, samples_log_prob = nuts_sample(
        key_sample, 
        log_prob_fn, 
        initial_state=jnp.asarray(ALPHA[jnp.newaxis, :]), 
        n_samples=N_NUTS_SAMPLES
    )
    samples = jnp.squeeze(samples) # NOTE: if n_chains != 1 ...
    samples_log_prob = jnp.squeeze(samples_log_prob)
    samples_log_prob = finite_samples_log_prob(samples_log_prob) 

    print("samples:", samples.min(), samples.max())
    print("probs:", samples_log_prob.min(), samples_log_prob.max())

    posterior_df = make_df(
        samples, 
        samples_log_prob, 
        parameter_strings=PARAMETER_STRINGS
    )

    np.savez(
        os.path.join(results_dir, "posterior_blackjax.npz"), 
        alpha=ALPHA,
        samples=samples,
        samples_log_prob=samples_log_prob,
        datavector=datavector,
        summary=x_
    )

    c = ChainConsumer()

    fisher_samples = np.random.multivariate_normal(
        ALPHA, datasets["tails"].data.Finv, (N_FISHER_SAMPLES,) 
    ) 
    fisher_samples_log_prob = jax.scipy.stats.multivariate_normal.logpdf(
        fisher_samples, ALPHA, datasets["tails"].data.Finv
    )
    fisher_df = make_df(
        cut_samples(fisher_samples, LOWER, UPPER),
        # samples_log_prob, 
        parameter_strings=PARAMETER_STRINGS
    )
    c.add_chain(
        Chain(
            samples=fisher_df,
            name=r"$F_{\Sigma^{-1}}$" + " {}".format("$k_n$[tails]"),
            color="r",
            linestyle=":",
            shade_alpha=0.
        )
    )

    fisher_samples = np.random.multivariate_normal(
        ALPHA, datasets["bulk"].data.Finv, (N_FISHER_SAMPLES,) 
    ) 
    fisher_samples_log_prob = jax.scipy.stats.multivariate_normal.logpdf(
        fisher_samples, ALPHA, datasets["bulk"].data.Finv
    )
    fisher_df = make_df(
        cut_samples(fisher_samples, LOWER, UPPER),
        # samples_log_prob, 
        parameter_strings=PARAMETER_STRINGS
    )
    c.add_chain(
        Chain(
            samples=fisher_df,
            name=r"$F_{\Sigma^{-1}}$" + " {}".format("$k_n$[bulk]"),
            color="b",
            linestyle=":",
            shade_alpha=0.
        )
    )

    fisher_samples = np.random.multivariate_normal(
        ALPHA, datasets["bulk_pdf"].data.Finv, (N_FISHER_SAMPLES,) 
    ) 
    fisher_samples_log_prob = jax.scipy.stats.multivariate_normal.logpdf(
        fisher_samples, ALPHA, datasets["bulk_pdf"].data.Finv
    )
    fisher_df = make_df(
        cut_samples(fisher_samples, LOWER, UPPER),
        # samples_log_prob, 
        parameter_strings=PARAMETER_STRINGS
    )
    c.add_chain(
        Chain(
            samples=fisher_df,
            name=r"$F_{\Sigma^{-1}}$" + " {}".format("$k_n$[bulk PDF]"),
            color="g",
            linestyle=":",
            shade_alpha=0.
        )
    )

    c.add_chain(
        Chain(
            samples=posterior_df, 
            name="SBI[{}]".format(args.bulk_or_tails), 
            color="r" if args.bulk_or_tails == "tails" else "b"
        )
    )
    c.add_marker(
        location=marker(x_, parameter_strings=PARAMETER_STRINGS),
        name=r"$\hat{x}$", 
        color="r" if args.bulk_or_tails == "tails" else "b",
        marker_style="x"
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
    overlay_bounds_on_corner(fig, LOWER, UPPER)
    fig = customize_plot(fig)
    fig.suptitle(
        (
            r"$k_n$ SBI & $F_{{\Sigma}}^{{-1}}$"
            + " z={}".format(config.redshift) + "\n"
            + (" linearised" if config.linearised else " Quijote") + ("[bulk]" if args.bulk_or_tails == "bulk" else "[tails]") + "\n"
            # + (r"[non-Gaussian $\xi_L[\pi]$ test]\n" if NON_GAUSSIAN_TEST else "") + 
            + r"$n_s$ = {}".format(config.n_linear_sims if config.linearised else N_S_HYPERCUBE) + "\n"
            + r"$R$ = [{}] Mpc".format(", ".join(map(str, config.scales))) + "\n"
            + r"$k_n$ = [{}]".format(
                ", ".join([get_cumulant_names()[_] for _ in config.order_idx])
            )
        ),
        multialignment='center'
    )
    plt.savefig(os.path.join(results_dir, "posterior_blackjax.pdf"))
    plt.savefig(os.path.join(posteriors_dir, "posterior_blackjax.pdf"))
    plt.close()

    # except Exception as e:
    #     print("~" * 50)
    #     print(f"Exception:\n\t{e}")
    #     print("~" * 50)




    # Per NDE sampling
    if ensemble.n_ndes > 1:
        posterior_dfs = []
        for n, nde in enumerate(ensemble.ndes):

            log_prob_fn_n = ensemble.nde_log_prob_fn(nde, cumulants_dataset.prior, x_)

            samples, samples_log_prob = nuts_sample(
                key_sample, 
                log_prob_fn_n, 
                initial_state=jnp.asarray(ALPHA[jnp.newaxis, :]), 
                n_samples=N_NUTS_SAMPLES
            )
            samples = jnp.squeeze(samples) # NOTE: if n_chains != 1 ...
            samples_log_prob = jnp.squeeze(samples_log_prob)
            samples_log_prob = finite_samples_log_prob(samples_log_prob) # all 

            print("samples:", samples.min(), samples.max())
            print("probs:", samples_log_prob.min(), samples_log_prob.max())

            posterior_df = make_df(
                samples, 
                samples_log_prob, 
                parameter_strings=PARAMETER_STRINGS
            )
            posterior_dfs.append(posterior_df)

            np.savez(
                os.path.join(results_dir, "posterior_blackjax_nde{}.npz".format(n)), 
                alpha=ALPHA,
                samples=samples,
                samples_log_prob=samples_log_prob,
                datavector=datavector,
                summary=x_
            )


        c = ChainConsumer()

        for n, posterior_df_n in enumerate(posterior_dfs):
            c.add_chain(
                Chain(
                    samples=posterior_df_n, 
                    name="SBI[{}] (NDE {})".format(args.bulk_or_tails, n), 
                    color="r" if args.bulk_or_tails == "tails" else "b"
                )
            )

        fisher_samples = np.random.multivariate_normal(
            ALPHA, datasets["tails"].data.Finv, (N_FISHER_SAMPLES,) 
        ) 
        fisher_samples_log_prob = jax.scipy.stats.multivariate_normal.logpdf(
            fisher_samples, ALPHA, datasets["tails"].data.Finv
        )
        fisher_df = make_df(
            cut_samples(fisher_samples, LOWER, UPPER),
            # samples_log_prob, 
            parameter_strings=PARAMETER_STRINGS
        )
        c.add_chain(
            Chain(
                samples=fisher_df,
                name=r"$F_{\Sigma^{-1}}$" + " {}".format("$k_n$[tails]"),
                color="r",
                linestyle=":",
                shade_alpha=0.
            )
        )

        fisher_samples = np.random.multivariate_normal(
            ALPHA, datasets["bulk"].data.Finv, (N_FISHER_SAMPLES,) 
        ) 
        fisher_samples_log_prob = jax.scipy.stats.multivariate_normal.logpdf(
            fisher_samples, ALPHA, datasets["bulk"].data.Finv
        )
        fisher_df = make_df(
            cut_samples(fisher_samples, LOWER, UPPER),
            # samples_log_prob, 
            parameter_strings=PARAMETER_STRINGS
        )
        c.add_chain(
            Chain(
                samples=fisher_df,
                name=r"$F_{\Sigma^{-1}}$" + " {}".format("$k_n$[bulk]"),
                color="b",
                linestyle=":",
                shade_alpha=0.
            )
        )

        fisher_samples = np.random.multivariate_normal(
            ALPHA, datasets["bulk_pdf"].data.Finv, (N_FISHER_SAMPLES,) 
        ) 
        fisher_samples_log_prob = jax.scipy.stats.multivariate_normal.logpdf(
            fisher_samples, ALPHA, datasets["bulk_pdf"].data.Finv
        )
        fisher_df = make_df(
            cut_samples(fisher_samples, LOWER, UPPER),
            # samples_log_prob, 
            parameter_strings=PARAMETER_STRINGS
        )
        c.add_chain(
            Chain(
                samples=fisher_df,
                name=r"$F_{\Sigma^{-1}}$" + " {}".format("$k_n$[bulk PDF]"),
                color="g",
                linestyle=":",
                shade_alpha=0.
            )
        )

        c.add_marker(
            location=marker(x_, parameter_strings=PARAMETER_STRINGS),
            name=r"$\hat{x}$", 
            color="r" if args.bulk_or_tails == "tails" else "b",
            marker_style="x"
        )
        c.add_truth(
            Truth(location=dict(zip(PARAMETER_STRINGS, ALPHA)), name=r"$\pi^0$")
        )

        fig = c.plotter.plot()
        overlay_bounds_on_corner(fig, LOWER, UPPER)
        fig = customize_plot(fig)
        fig.suptitle(
            (
                r"$k_n$ SBI & $F_{{\Sigma}}^{{-1}}$"
                + " z={}".format(config.redshift) + "\n"
                + (" linearised" if config.linearised else " Quijote") + ("[bulk]" if args.bulk_or_tails == "bulk" else "[tails]") + "\n"
                + (r"[non-Gaussian $\xi_L[\pi]$ test]" if NON_GAUSSIAN_TEST else "") + "\n"
                + r"$n_s$ = {}".format(config.n_linear_sims if config.linearised else N_S_HYPERCUBE) + "\n"
                + r"$R$ = [{}] Mpc".format(", ".join(map(str, config.scales))) + "\n"
                + r"$k_n$ = [{}]".format(
                    ", ".join([get_cumulant_names()[_] for _ in config.order_idx])
                )
            ),
            multialignment='center'
        )
        plt.savefig(os.path.join(results_dir, "posterior_blackjax_ndes.pdf"))
        plt.savefig(os.path.join(posteriors_dir, "posterior_blackjax_ndes.pdf"))
        plt.close()


    log_prob_fn = ensemble.ensemble_log_prob_fn(x_noiseless, cumulants_dataset.prior)

    samples, samples_log_prob = nuts_sample(
        key_sample, 
        log_prob_fn, 
        initial_state=jnp.asarray(ALPHA[jnp.newaxis, :]), 
        n_samples=N_NUTS_SAMPLES
    )
    samples = jnp.squeeze(samples) # NOTE: if n_chains != 1 ...
    samples_log_prob = jnp.squeeze(samples_log_prob)
    samples_log_prob = finite_samples_log_prob(samples_log_prob) 

    print("samples:", samples.min(), samples.max())
    print("probs:", samples_log_prob.min(), samples_log_prob.max())

    posterior_df = make_df(
        samples, 
        samples_log_prob, 
        parameter_strings=PARAMETER_STRINGS
    )

    c = ChainConsumer()

    fisher_samples = np.random.multivariate_normal(
        ALPHA, datasets["tails"].data.Finv, (N_FISHER_SAMPLES,) 
    ) 
    fisher_samples_log_prob = jax.scipy.stats.multivariate_normal.logpdf(
        fisher_samples, ALPHA, datasets["tails"].data.Finv
    )
    fisher_df = make_df(
        cut_samples(fisher_samples, LOWER, UPPER),
        # samples_log_prob, 
        parameter_strings=PARAMETER_STRINGS
    )
    c.add_chain(
        Chain(
            samples=fisher_df,
            name=r"$F_{\Sigma^{-1}}$" + " {}".format("$k_n$[tails]"),
            color="r",
            linestyle=":",
            shade_alpha=0.
        )
    )

    fisher_samples = np.random.multivariate_normal(
        ALPHA, datasets["bulk"].data.Finv, (N_FISHER_SAMPLES,) 
    ) 
    fisher_samples_log_prob = jax.scipy.stats.multivariate_normal.logpdf(
        fisher_samples, ALPHA, datasets["bulk"].data.Finv
    )
    fisher_df = make_df(
        cut_samples(fisher_samples, LOWER, UPPER),
        # samples_log_prob, 
        parameter_strings=PARAMETER_STRINGS
    )
    c.add_chain(
        Chain(
            samples=fisher_df,
            name=r"$F_{\Sigma^{-1}}$" + " {}".format("$k_n$[bulk]"),
            color="b",
            linestyle=":",
            shade_alpha=0.
        )
    )

    fisher_samples = np.random.multivariate_normal(
        ALPHA, datasets["bulk_pdf"].data.Finv, (N_FISHER_SAMPLES,) 
    ) 
    fisher_samples_log_prob = jax.scipy.stats.multivariate_normal.logpdf(
        fisher_samples, ALPHA, datasets["bulk_pdf"].data.Finv
    )
    fisher_df = make_df(
        cut_samples(fisher_samples, LOWER, UPPER),
        # samples_log_prob, 
        parameter_strings=PARAMETER_STRINGS
    )
    c.add_chain(
        Chain(
            samples=fisher_df,
            name=r"$F_{\Sigma^{-1}}$" + " {}".format("$k_n$[bulk PDF]"),
            color="g",
            linestyle=":",
            shade_alpha=0.
        )
    )

    c.add_chain(
        Chain(
            samples=posterior_df, 
            name="SBI[{}]".format(args.bulk_or_tails), 
            color="r" if args.bulk_or_tails == "tails" else "b"
        )
    )
    c.add_marker(
        location=marker(x_noiseless, parameter_strings=PARAMETER_STRINGS),
        name=r"$\hat{x}$", 
        color="r" if args.bulk_or_tails == "tails" else "b",
        marker_style="x"
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
    overlay_bounds_on_corner(fig, LOWER, UPPER)
    fig = customize_plot(fig)
    fig.suptitle(
        (
            r"$k_n$ SBI & $F_{{\Sigma}}^{{-1}}$"
            + " z={}".format(config.redshift) + "\n"
            + (" linearised" if config.linearised else " Quijote") + ("[bulk]" if args.bulk_or_tails == "bulk" else "[tails]") + "\n"
            # + (r"[non-Gaussian $\xi_L[\pi]$ test]\n" if NON_GAUSSIAN_TEST else "") + 
            + r"$n_s$ = {}".format(config.n_linear_sims if config.linearised else N_S_HYPERCUBE) + "\n"
            + r"$R$ = [{}] Mpc".format(", ".join(map(str, config.scales))) + "\n"
            + r"$k_n$ = [{}]".format(
                ", ".join([get_cumulant_names()[_] for _ in config.order_idx])
            )
        ),
        multialignment='center'
    )
    plt.savefig(os.path.join(results_dir, "posterior_noiseless_blackjax.pdf"))
    plt.savefig(os.path.join(posteriors_dir, "posterior_noiseless_blackjax.pdf"))
    plt.close()


print("Time={:.1} mins.".format((time.time() - t0) / 60.))


# c.add_chain(
#     Chain.from_covariance(
#         ALPHA,
#         dataset.Finv,
#         columns=PARAMETER_STRINGS,
#         name=r"$F_{\Sigma^{-1}}$" + " {}".format("$k_n$[tails]"),
#         color="k",
#         linestyle=":",
#         shade_alpha=0.
#     )
# )
# c.add_chain(
#     Chain.from_covariance(
#         ALPHA,
#         datasets["bulk"].data.Finv,
#         columns=PARAMETER_STRINGS,
#         name=r"$F_{\Sigma^{-1}}$" + " {}".format("$k_n$[bulk]"),
#         color="b",
#         linestyle=":",
#         shade_alpha=0.
#     )
# )
# c.add_chain(
#     Chain.from_covariance(
#         ALPHA,
#         datasets["bulk_pdf"].data.Finv,
#         columns=PARAMETER_STRINGS,
#         name=r"$F_{\Sigma^{-1}}$" + " {}".format("PDF[bulk]"),
#         color="g",
#         linestyle=":",
#         shade_alpha=0.
#     )
# )