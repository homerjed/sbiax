import argparse
import os
import time
import datetime

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key
import optax

import numpy as np 
import matplotlib.pyplot as plt
from chainconsumer import Chain, ChainConsumer, Truth
from tensorflow_probability.substrates.jax.distributions import Distribution
from ml_collections import ConfigDict

from configs import cumulants_config, get_results_dir, get_posteriors_dir, get_cumulants_sbi_args, get_ndes_from_config
from cumulants import Dataset, get_data, get_prior, get_linear_compressor, get_datavector, get_linearised_data

from sbiax.utils import make_df, marker
from sbiax.ndes import Scaler, Ensemble, CNF, MAF 
from sbiax.train import train_ensemble
from sbiax.inference import nuts_sample

from affine import affine_sample
from utils import plot_moments, plot_latin_moments, plot_summaries, plot_fisher_summaries
from pca import PCA
from nn import fit_nn

""" 
    Run NLE or NPE SBI with the moments of the 1pt matter PDF.

    - Run with just variance of PDF (a test)? 
    - diagonal of covariance for compression?

    - freezing 'nuisance parameters'

    - cumulants? What are Quijote conventions again? 
        - Default quijote is cumulants (k-stats are sample cumulants, unbiased estimators)?

    - scaling of inputs? summaries seem to be high magnitude ... PCA whitening?
    - covariance c1onditioning?
    
    - pretraining with linear sims vs fisher summaries is the same thing?
""" 

def get_nn_compressor(key, D, P, data_preprocess_fn, results_dir):
    net_key, train_key = jr.split(key)

    net = eqx.nn.MLP(
        dataset.data.shape[-1], 
        dataset.parameters.shape[-1], 
        width_size=32, 
        depth=1, 
        activation=jax.nn.tanh,
        key=net_key
    )

    def preprocess_fn(x): 
        # Preprocess with covariance?
        return (jnp.asarray(x) - jnp.mean(dataset.data, axis=0)) / jnp.std(dataset.data, axis=0)

    net, losses = fit_nn(
        train_key, 
        net, 
        (preprocess_fn(data_preprocess_fn(dataset.data)), dataset.parameters), 
        opt=optax.adam(1e-3), 
        precision=jnp.linalg.inv(dataset.Finv),
        n_batch=500, 
        patience=1000,
        n_steps=50_000
    )

    plt.figure()
    plt.loglog(losses)
    plt.savefig(os.path.join(results_dir, "losses_nn.png"))
    plt.close()

    return net, preprocess_fn

# jax.config.update("jax_enable_x64", True) # Moments require this?

args = get_cumulants_sbi_args()

print("TIME:", datetime.datetime.now().strftime("%H:%M %d-%m-%y"))
print("SEED:", args.seed)
print("MOMENTS:", args.order_idx)
print("LINEARISED:", args.linearised)

t0 = time.time()

config = cumulants_config(
    seed=args.seed, 
    redshift=args.redshift, 
    reduced_cumulants=args.reduced_cumulants,
    sbi_type=args.sbi_type,
    linearised=args.linearised, 
    compression=args.compression,
    order_idx=args.order_idx,
    n_linear_sims=args.n_linear_sims,
    pre_train=args.pre_train
)

key = jr.key(config.seed)

( 
    model_key, train_key, key_prior, 
    key_datavector, key_state, key_sample
) = jr.split(key, 6)

results_dir: str = get_results_dir(config, args)

posteriors_dir: str = get_posteriors_dir(config)

# Dataset of simulations, parameters, covariance, ...
dataset: Dataset = get_data(config, verbose=args.verbose)

print("DATA:", ["{:.3E} {:.3E}".format(_.min(), _.max()) for _ in (dataset.fiducial_data, dataset.data)])
print("DATA:", [_.shape for _ in (dataset.fiducial_data, dataset.data)])

parameter_prior: Distribution = get_prior(config)

# Get linear or neural network compressor
if config.compression == "nn":

    net, preprocess_fn = get_nn_compressor(
        key, dataset.data, dataset.parameters, results_dir=results_dir
    )

    compressor = lambda d, p: net(preprocess_fn(d)) # Ignore parameter kwarg!

if config.compression == "linear":
    compressor = get_linear_compressor(config)

# Fit PCA transform to simulated data and apply after compressing
if config.use_pca:

    # Compress simulations as usual 
    X = jax.vmap(compressor)(dataset.data, dataset.parameters)

    # Standardise before PCA (don't get tricked by high variance due to units)
    X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)

    # Fit whitening-PCA to compressed simulations
    pca = PCA(num_components=dataset.alpha.size) 
    pca.fit(X) # Fit on fiducial data?
    
    # Reparameterize compression with both transforms
    compression_fn = lambda d, p: pca.transform(compressor(d, p))
else:
    compression_fn = lambda d, p: compressor(d, p)

# Compress simulations
X = jax.vmap(compression_fn)(dataset.data, dataset.parameters)

plot_summaries(X, dataset.parameters, dataset, results_dir)

plot_moments(dataset.fiducial_data, config, results_dir)

plot_latin_moments(dataset.data, config, results_dir)

"""
    Build NDEs
"""

scaler = Scaler(
    X, dataset.parameters, use_scaling=config.maf.use_scaling
)

ndes = get_ndes_from_config(
    config, 
    event_dim=dataset.alpha.size, 
    scalers=scaler, 
    use_scalers=config.ndes[0].use_scaling, # NOTE: not to be trusted
    key=model_key
)

print("scaler:", ndes[0].scaler.mu_x if ndes[0].scaler is not None else None)

ensemble = Ensemble(ndes, sbi_type=config.sbi_type)

data_preprocess_fn = lambda x: x #jnp.log(jnp.clip(x, min=1e-10))

"""
    Pre-train NDEs on linearised data
"""

# Only pre-train if required and not inferring from linear simulations
if (
    (not config.linearised) 
    and config.pre_train 
    and (config.n_linear_sims is not None)
):
    print("Linearised pre-training...")

    pre_train_key, summaries_key = jr.split(key)

    # Pre-train data = linearised simulations
    X_l, Y_l = get_linearised_data(config)
    X_l = jax.vmap(compression_fn)(X_l, Y_l)

    print("Pre-training with", X_l.shape, Y_l.shape)

    plot_fisher_summaries(X_l, Y_l, dataset, results_dir)

    opt = getattr(optax, config.opt)(config.lr)

    ensemble, stats = train_ensemble(
        pre_train_key, 
        ensemble,
        train_mode=config.sbi_type,
        train_data=(data_preprocess_fn(X_l), Y_l), 
        opt=opt,
        n_batch=config.n_batch,
        patience=config.patience,
        n_epochs=config.n_epochs,
        tqdm_description="Training (pre-train)",
        show_tqdm=args.use_tqdm,
        results_dir=results_dir
    )

    datavector = get_datavector(key_datavector, config)

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

    samples_log_prob = jax.vmap(log_prob_fn)(samples)
    alpha_log_prob = log_prob_fn(jnp.asarray(dataset.alpha))

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
    c.add_chain(Chain(samples=posterior_df, name="SBI", color="r"))
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

opt = getattr(optax, config.opt)(config.lr)

print("Data / Parameters", [_.shape for _ in (X, dataset.parameters)])

ensemble, stats = train_ensemble(
    train_key, 
    ensemble,
    train_mode=config.sbi_type,
    train_data=(data_preprocess_fn(X), dataset.parameters), 
    opt=opt,
    n_batch=config.n_batch,
    patience=config.patience,
    n_epochs=config.n_epochs,
    tqdm_description="Training (data)",
    show_tqdm=args.use_tqdm,
    results_dir=results_dir
)

print("scaler:", ndes[0].scaler.mu_x if ndes[0].scaler is not None else None)

eqx.tree_serialise_leaves(os.path.join(results_dir, "ensemble.eqx"), ensemble)

""" 
    Sample and plot posterior for NDE with noisy datavectors
"""

ensemble = eqx.nn.inference_mode(ensemble)

# Generates linearised (or not) datavector at fiducial parameters
datavector = get_datavector(key_datavector, config)

x_ = compression_fn(datavector, dataset.alpha)

print("datavector", x_, dataset.alpha)

log_prob_fn = ensemble.ensemble_log_prob_fn(data_preprocess_fn(x_), parameter_prior)

if 1:
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

    samples_log_prob = jax.vmap(log_prob_fn)(samples)
    alpha_log_prob = log_prob_fn(jnp.asarray(dataset.alpha))

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

    try:
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
        c.add_chain(Chain(samples=posterior_df, name="SBI", color="r"))
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
            r"$k_n/k_2^{n-1}$ SBI & $F_{{\Sigma}}^{{-1}}$" + "\n" +
            r"(z={}, R={} Mpc, $k_n$={})".format(
                config.redshift, 
                "[{}]".format(", ".join(map(str, config.scales))),
                ", ".join(map(str, [["var.", "skew.", "kurt."][_] for _ in config.order_idx]))
                ),
            multialignment='center'
        )
        plt.savefig(os.path.join(results_dir, "posterior_affine.pdf"))
        plt.savefig(os.path.join(posteriors_dir, "posterior_affine.pdf"))
        plt.close()
    except Exception as e:
        print(e)
        print("(!) Shit posterior")
        pass

if 1:
    try:
        # BLACKJAX SAMPLE IS FUNNY WITH float64
        samples, samples_log_prob = nuts_sample(
            key_sample, log_prob_fn, prior=parameter_prior
        )
        samples = samples.squeeze()
        samples_log_prob = samples_log_prob.squeeze()

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
        c.add_chain(Chain(samples=posterior_df, name="SBI", color="r"))
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
            r"$k_n/k_2^{n-1}$ SBI & $F_{{\Sigma}}^{{-1}}$" + "\n" +
            r"(z={}, R={} Mpc, $k_n$={})".format(
                config.redshift, 
                "[{}]".format(", ".join(map(str, config.scales))),
                ", ".join(map(str, [["var.", "skew.", "kurt."][_] for _ in config.order_idx]))
                ),
            multialignment='center'
        )
        plt.savefig(os.path.join(results_dir, "posterior.pdf"))
        plt.close()
    except Exception as e:
        print("(!) Shit posterior")
        pass

print(f"Time={(time.time() - t0) / 60.:.1} mins.")


# # Generates datavector d ~ G[d|xi[pi], Sigma]
# mu = fiducial_pdfs.mean(axis=0)
# datavector = jr.multivariate_normal(key, mean=mu, cov=C)
# # datavector = fiducial_pdfs[0]

# Chi^2 min would be performed for each different universe where alpha is sampled
# X_ = _mle(
#     datavector,
#     pi=alpha,
#     Finv=Finv, 
#     mu=mu, 
#     dmu=derivatives.mean(axis=0), 
#     precision=Cinv
# )