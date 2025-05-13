import os
import time
import datetime
import argparse
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

import numpy as np 
import matplotlib.pyplot as plt
from chainconsumer import Chain, ChainConsumer, Truth
from tensorflow_probability.substrates.jax.distributions import Distribution
import tensorflow_probability.substrates.jax.distributions as tfd

from sbiax.ndes import Scaler, CNF, MAF 
from sbiax.train import train_ensemble
from sbiax.inference import nuts_sample
from sbiax.utils import make_df, marker

from configs import (
    cumulants_config, 
    bulk_cumulants_config, 
    get_results_dir, 
    get_posteriors_dir, 
    get_ndes_from_config
)
from configs.args import get_cumulants_sbi_args
from data.cumulants import CumulantsDataset
from data.common import (
    Dataset, 
    get_datavector, 
    get_linearised_data
)
from data.pdfs import BulkCumulantsDataset, get_bulk_dataset
from cumulants_ensemble import Ensemble
from affine import affine_sample
from utils.utils import (
    get_datasets,
    plot_cumulants,
    plot_moments, 
    plot_latin_moments, 
    plot_summaries, 
    plot_fisher_summaries, 
    replace_scalers,
    get_dataset_and_config,
    finite_samples_log_prob
)

jax.clear_caches()


""" 
    Run NLE or NPE SBI with the moments of the 1pt matter PDF.

    - diagonal of covariance for compression?
    - freezing 'nuisance parameters'
    - scaling of inputs? summaries seem to be high magnitude ... PCA whitening?
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

config, cumulants_dataset, datasets = get_datasets(args)

key = jr.key(config.seed)

( 
    model_key, train_key, key_prior, 
    key_datavector, key_state, key_sample
) = jr.split(key, 6)

results_dir = get_results_dir(config, args)

posteriors_dir = get_posteriors_dir(config, args)

dataset: Dataset = cumulants_dataset.data

parameter_prior: Distribution = cumulants_dataset.prior

plot_cumulants(args, config, dataset.fiducial_data, results_dir=results_dir)

################################ Check fisher forecasts

if args.seed == 0:

    c = ChainConsumer()

    for i, (dataset_type, name) in enumerate(zip(
        ["bulk_pdf", "bulk", "tails"],
        [" PDF[bulk]", " $k_n$[bulk]", " $k_n$[tails]"],
    )):
        c.add_chain(
            Chain.from_covariance(
                datasets[dataset_type].data.alpha,
                datasets[dataset_type].data.Finv,
                columns=dataset.parameter_strings,
                name=r"$F_{\Sigma^{-1}}$" + name,
                shade_alpha=0.
            )
        )
    c.add_marker(
        location=marker(dataset.alpha, parameter_strings=dataset.parameter_strings),
        name=r"$\alpha$", 
        color="#7600bc"
    )
    fig = c.plotter.plot()
    plt.savefig(os.path.join(results_dir, "Fisher_tests.pdf"))
    plt.close()

    if not config.freeze_parameters:
        c = ChainConsumer()

        target_idx = np.array([0, 4])
        for i, (dataset_type, name) in enumerate(zip(
            ["bulk_pdf", "bulk", "tails"],
            [" PDF[bulk]", " $k_n$[bulk]", " $k_n$[tails]"],
        )):
            c.add_chain(
                Chain.from_covariance(
                    datasets[dataset_type].data.alpha[target_idx],
                    datasets[dataset_type].data.Finv[:, target_idx][target_idx, :],
                    columns=[dataset.parameter_strings[_] for _ in target_idx],
                    name=r"$F_{\Sigma^{-1}}$" + name,
                    shade_alpha=0.
                )
            )
        c.add_marker(
            location=marker(dataset.alpha[target_idx], parameter_strings=[dataset.parameter_strings[_] for _ in target_idx]),
            name=r"$\alpha$", 
            color="#7600bc"
        )
        fig = c.plotter.plot()
        plt.savefig(os.path.join(results_dir, "Fisher_tests_marginalised.pdf"))
        plt.close()

    plt.figure()
    corr = jnp.corrcoef(dataset.fiducial_data, rowvar=False)
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1., vmax=1.)
    plt.colorbar(im)
    plt.savefig(
        os.path.join(results_dir, "correlation_matrix_cumulants_{}.png".format(args.bulk_or_tails))
    )
    plt.close()

    plt.figure()
    im = plt.imshow(dataset.Cinv)
    plt.colorbar(im)
    plt.savefig(
        os.path.join(results_dir, "precision_matrix_cumulants_{}.png".format(args.bulk_or_tails))
    )
    plt.close()

    print("Covariance condition number: {:.3E}".format(jnp.linalg.cond(dataset.C)))

################################

"""
    Compression
"""

# Compress simulations
compression_fn = cumulants_dataset.compression_fn

X = jax.vmap(compression_fn)(dataset.data, dataset.parameters)

# Plot summaries
plot_summaries(X, dataset.parameters, dataset, results_dir)

plot_moments(dataset.fiducial_data, config, results_dir)

plot_latin_moments(dataset.data, config, results_dir)

"""
    Build NDEs
"""

scaler = Scaler(
    X, dataset.parameters, use_scaling=config.use_scalers
)

ndes = get_ndes_from_config(
    config, 
    event_dim=dataset.alpha.size, 
    scalers=scaler, # Same scaler for all NDEs 
    use_scalers=config.use_scalers, # NOTE: not to be trusted
    key=model_key
)

print("scaler:", ndes[0].scaler.mu_x if ndes[0].scaler is not None else None) # Check scaler mu, std are not changed by gradient

ensemble = Ensemble(ndes, sbi_type=config.sbi_type)

data_preprocess_fn = lambda x: x #2.0 * (x - X.min()) / (X.max() - X.min()) - 1.0 #/ jnp.max(dataset.fiducial_data, axis=0) #jnp.log(jnp.clip(x, min=1e-10))

"""
    Pre-train NDEs on linearised data
"""

# Only pre-train if required and not inferring from linear simulations
if ((not config.linearised) and config.pre_train and (config.n_linear_sims is not None)):
    print("Linearised pre-training...")

    pre_train_key, summaries_key = jr.split(key)

    # Pre-train data = linearised simulations
    D_l, Y_l = cumulants_dataset.get_linearised_data()

    X_l = jax.vmap(compression_fn)(D_l, Y_l)

    print("Pre-training with", D_l.shape, X_l.shape, Y_l.shape)

    plot_fisher_summaries(X_l, Y_l, dataset, results_dir)

    if config.use_scalers:
        ensemble = replace_scalers(
            ensemble, config=config, X=data_preprocess_fn(X_l), P=Y_l
        )

    opt = getattr(optax, config.pretrain.opt)(config.pretrain.lr)

    ensemble, stats = train_ensemble(
        pre_train_key, 
        ensemble,
        train_mode=config.sbi_type,
        train_data=(data_preprocess_fn(X_l), Y_l), 
        opt=opt,
        use_ema=config.use_ema,
        ema_rate=config.ema_rate,
        n_batch=config.pretrain.n_batch,
        patience=config.pretrain.patience,
        n_epochs=config.pretrain.n_epochs,
        valid_fraction=config.valid_fraction,
        tqdm_description="Training (pre-train)",
        show_tqdm=args.use_tqdm,
        results_dir=results_dir
    )

    datavector = cumulants_dataset.get_datavector(key_datavector)

    x_ = compression_fn(datavector, dataset.alpha)

    log_prob_fn = ensemble.ensemble_log_prob_fn(data_preprocess_fn(x_), parameter_prior)

    state = jr.multivariate_normal(
        key_state, x_, dataset.Finv, (2 * config.n_walkers,)
    )

    samples, weights = affine_sample(
        key_sample, 
        log_prob=log_prob_fn,
        n_walkers=config.n_walkers, 
        n_steps=config.n_steps + config.burn, 
        burn=config.burn, 
        current_state=state,
        description="Sampling",
        show_tqdm=args.use_tqdm
    )

    alpha_log_prob = log_prob_fn(jnp.asarray(dataset.alpha))
    samples_log_prob = jax.vmap(log_prob_fn)(samples)
    samples_log_prob = finite_samples_log_prob(samples_log_prob)

    posterior_df = make_df(
        samples, 
        samples_log_prob, 
        parameter_strings=dataset.parameter_strings
    )

    np.savez(
        os.path.join(results_dir, "posterior.npz"), 
        alpha=dataset.alpha,
        samples=samples,
        samples_log_prob=samples_log_prob,
        datavector=datavector,
        summary=x_
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
        Chain.from_covariance(
            dataset.alpha,
            datasets["bulk"].data.Finv,
            columns=dataset.parameter_strings,
            name=r"$F_{\Sigma^{-1}}$" + " {}".format("$k_n$[bulk]"),
            color="b",
            linestyle=":",
            shade_alpha=0.
        )
    )
    c.add_chain(
        Chain.from_covariance(
            dataset.alpha,
            datasets["bulk_pdf"].data.Finv,
            columns=dataset.parameter_strings,
            name=r"$F_{\Sigma^{-1}}$" + " {}".format("PDF[bulk]"),
            color="g",
            linestyle=":",
            shade_alpha=0.
        )
    )
    c.add_chain(
        Chain(samples=posterior_df, name="SBI[{}]".format(args.bulk_or_tails), color="r")
    )
    c.add_marker(
        location=marker(x_, parameter_strings=dataset.parameter_strings),
        name=r"$\hat{x}$", 
        color="b"
    )
    c.add_marker(
        location=marker(dataset.alpha, parameter_strings=dataset.parameter_strings),
        name=r"$\alpha$", 
        color="#7600bc"
    )
    fig = c.plotter.plot()
    plt.savefig(os.path.join(results_dir, "posterior_affine_pretrain.pdf"))
    plt.savefig(os.path.join(posteriors_dir, "posterior_affine_pretrain.pdf"))
    plt.close()

"""
    Train NDE on data
"""

opt = getattr(optax, config.train.opt)(config.train.lr)

if config.use_scalers:
    ensemble = replace_scalers(
        ensemble, config=config, X=data_preprocess_fn(X), P=dataset.parameters
    )

ensemble, stats = train_ensemble(
    train_key, 
    ensemble,
    train_mode=config.sbi_type,
    train_data=(data_preprocess_fn(X), dataset.parameters), 
    opt=opt,
    use_ema=config.use_ema,
    ema_rate=config.ema_rate,
    n_batch=config.train.n_batch,
    patience=config.train.patience,
    n_epochs=config.train.n_epochs,
    valid_fraction=config.valid_fraction,
    tqdm_description="Training (data)",
    show_tqdm=args.use_tqdm,
    results_dir=results_dir
)

print("scaler mu:", ndes[0].scaler.mu_x if ndes[0].scaler is not None else None)
print("scaler std:", ndes[0].scaler.std_x if ndes[0].scaler is not None else None)

""" 
    Sample and plot posterior for NDE with noisy datavectors
"""

# Generates linearised (or not) datavector at fiducial parameters
datavector = cumulants_dataset.get_datavector(key_datavector)

print("datavector {} \n {}".format(datavector.shape, datavector))

x_ = compression_fn(datavector, dataset.alpha)

print("compressed datavector {} \n {} {}".format(x_.shape, x_, dataset.alpha))

log_prob_fn = ensemble.ensemble_log_prob_fn(data_preprocess_fn(x_), parameter_prior)

print(log_prob_fn(dataset.alpha)) # Scaler works how here?

if 1:
    try:
        state = jr.multivariate_normal(
            key_state, x_, dataset.Finv, (2 * config.n_walkers,)
        )
        # state = parameter_prior.sample(seed=key_state, sample_shape=(2 * config.n_walkers,))

        samples, weights = affine_sample(
            key_sample, 
            log_prob=log_prob_fn,
            n_walkers=config.n_walkers, 
            n_steps=config.n_steps + config.burn, 
            burn=config.burn, 
            current_state=state,
            description="Sampling",
            show_tqdm=args.use_tqdm
        )

        alpha_log_prob = log_prob_fn(dataset.alpha)
        samples_log_prob = jax.vmap(log_prob_fn)(samples)
        samples_log_prob = finite_samples_log_prob(samples_log_prob) 

        print("samples:", samples.min(), samples.max())
        print("probs:", samples_log_prob.min(), samples_log_prob.max())

        posterior_df = make_df(
            samples, 
            samples_log_prob, 
            parameter_strings=dataset.parameter_strings
        )

        np.savez(
            os.path.join(results_dir, "posterior.npz"), 
            alpha=dataset.alpha,
            samples=samples,
            samples_log_prob=samples_log_prob,
            datavector=datavector,
            summary=x_
        )

        c = ChainConsumer()
        c.add_chain(
            Chain.from_covariance(
                dataset.alpha,
                datasets["tails"].data.Finv,
                columns=dataset.parameter_strings,
                name=r"$F_{\Sigma^{-1}}$" + " {}".format("$k_n$[tails]"),
                color="k",
                linestyle=":",
                shade_alpha=0.
            )
        )
        c.add_chain(
            Chain.from_covariance(
                dataset.alpha,
                datasets["bulk"].data.Finv,
                columns=dataset.parameter_strings,
                name=r"$F_{\Sigma^{-1}}$" + " {}".format("$k_n$[bulk]"),
                color="b",
                linestyle=":",
                shade_alpha=0.
            )
        )
        c.add_chain(
            Chain.from_covariance(
                dataset.alpha,
                datasets["bulk_pdf"].data.Finv,
                columns=dataset.parameter_strings,
                name=r"$F_{\Sigma^{-1}}$" + " {}".format("PDF[bulk]"),
                color="g",
                linestyle=":",
                shade_alpha=0.
            )
        )
        c.add_chain(
            Chain(
                samples=posterior_df, 
                name="SBI[{}]".format(args.bulk_or_tails), 
                color="r"
            )
        )
        c.add_marker(
            location=marker(x_, parameter_strings=dataset.parameter_strings),
            name=r"$\hat{x}$", 
            color="b"
        )
        c.add_marker(
            location=marker(dataset.alpha, parameter_strings=dataset.parameter_strings),
            name=r"$\alpha$", 
            color="k"
        )
        fig = c.plotter.plot()
        fig.suptitle(
            (
                r"$k_n$ SBI & $F_{{\Sigma}}^{{-1}}$"
                + " z={}".format(config.redshift) + "\n"
                + (" linearised" if config.linearised else " Quijote") + ("[bulk]" if args.bulk_or_tails == "bulk" else "[tails]") + "\n"
                + r"$n_s$ = {}".format(config.n_linear_sims if config.linearised else 2000) + "\n"
                + r"$R$ = [{}] Mpc".format(", ".join(map(str, config.scales))) + "\n"
                + r"$k_n$ = [{}]".format(
                    ", ".join([["var.", "skew.", "kurt."][_] for _ in config.order_idx])
                )
            ),
            multialignment='center'
        )
        plt.savefig(os.path.join(results_dir, "posterior_affine.pdf"))
        plt.savefig(os.path.join(posteriors_dir, "posterior_affine.pdf"))
        plt.close()
    except Exception as e:
        print("~" * 50)
        print(f"Exception:\n\t{e}")
        print("~" * 50)

if 0:
    try:
        nuts_prior = tfd.MultivariateNormalFullCovariance(
            loc=dataset.alpha, covariance_matrix=dataset.Finv
        )
        samples, samples_log_prob = nuts_sample(
            key_sample, 
            log_prob_fn, 
            initial_state=dataset.alpha[jnp.newaxis, :], 
            prior=nuts_prior, #parameter_prior
            # n_chains=1000,
            # n_samples=2000
        )
        samples = jnp.squeeze(samples) # NOTE: if n_chains != 1 ...
        samples_log_prob = jnp.squeeze(samples_log_prob)
        # samples = jnp.concatenate(samples,) 
        # samples_log_prob = jnp.squeeze(samples_log_prob)
        samples_log_prob = finite_samples_log_prob(samples_log_prob) # all 

        print("samples:", samples.min(), samples.max())
        print("probs:", samples_log_prob.min(), samples_log_prob.max())

        posterior_df = make_df(
            samples, 
            samples_log_prob, 
            parameter_strings=dataset.parameter_strings
        )

        np.savez(
            os.path.join(results_dir, "posterior_blackjax.npz"), 
            alpha=dataset.alpha,
            samples=samples,
            samples_log_prob=samples_log_prob,
            datavector=datavector,
            summary=x_
        )

        c = ChainConsumer()
        c.add_chain(
            Chain.from_covariance(
                dataset.alpha,
                dataset.Finv,
                columns=dataset.parameter_strings,
                name=r"$F_{\Sigma^{-1}}$" + " {}".format(
                    "$S_n$[tails]" if config.reduced_cumulants else "$k_n$[tails]"
                ),
                color="k",
                linestyle=":",
                shade_alpha=0.
            )
        )
        c.add_chain(
            Chain.from_covariance(
                dataset.alpha,
                datasets["bulk"].data.Finv,
                columns=dataset.parameter_strings,
                name=r"$F_{\Sigma^{-1}}$" + " {}".format("$k_n$[bulk]"),
                color="b",
                linestyle=":",
                shade_alpha=0.
            )
        )
        c.add_chain(
            Chain.from_covariance(
                dataset.alpha,
                datasets["bulk_pdf"].data.Finv,
                columns=dataset.parameter_strings,
                name=r"$F_{\Sigma^{-1}}$" + " {}".format("PDF[bulk]"),
                color="g",
                linestyle=":",
                shade_alpha=0.
            )
        )
        c.add_chain(Chain(samples=posterior_df, name="SBI[{}]".format(args.bulk_or_tails), color="r"))
        c.add_marker(
            location=marker(x_, parameter_strings=dataset.parameter_strings),
            name=r"$\hat{x}$", 
            color="b"
        )
        c.add_marker(
            location=marker(dataset.alpha, parameter_strings=dataset.parameter_strings),
            name=r"$\alpha$", 
            color="#7600bc"
        )
        fig = c.plotter.plot()
        fig.suptitle(
            (
                r"$k_n$ SBI & $F_{{\Sigma}}^{{-1}}$"
                + " z={}".format(config.redshift) + "\n"
                + (" linearised" if config.linearised else " Quijote") + ("[bulk]" if args.bulk_or_tails == "bulk" else "[tails]") + "\n"
                + r"$n_s$ = {}".format(config.n_linear_sims if config.linearised else 2000) + "\n"
                + r"$R$ = [{}] Mpc".format(", ".join(map(str, config.scales))) + "\n"
                + r"$k_n$ = [{}]".format(
                    ", ".join([["var.", "skew.", "kurt."][_] for _ in config.order_idx])
                )
            ),
            multialignment='center'
        )
        plt.savefig(os.path.join(results_dir, "posterior_blackjax.pdf"))
        plt.savefig(os.path.join(posteriors_dir, "posterior_blackjax.pdf"))
        plt.close()
    except Exception as e:
        print("~" * 50)
        print(f"Exception:\n\t{e}")
        print("~" * 50)

print("Time={:.1} mins.".format((time.time() - t0) / 60.))

# ensemble = Ensemble(ndes, sbi_type=config.sbi_type)
# eqx.tree_deserialise_leaves(os.path.join(results_dir, "ensemble.eqx"), ensemble)

################################ Save / load ensemble test

# print(">Saved ensemble")
# ensemble = Ensemble(ndes, sbi_type=config.sbi_type)
# ensemble = eqx.tree_deserialise_leaves(os.path.join(results_dir, "ensemble.eqx"), ensemble)
# print(">Loaded ensemble")

# print(jax.tree.map(lambda x: x.shape, eqx.filter(ensemble, eqx.is_array)))

################################