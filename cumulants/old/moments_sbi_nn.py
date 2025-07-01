import argparse
import os
import time
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import numpy as np 
import matplotlib.pyplot as plt
from chainconsumer import Chain, ChainConsumer, Truth
from tensorflow_probability.substrates.jax.distributions import Distribution

from configs import moments_config, get_results_dir, get_moments_sbi_args, get_ndes_from_config
from moments import Dataset, get_data, get_prior, get_linear_compressor, get_datavector

from sbiax.utils import make_df, marker
from sbiax.ndes import Scaler, Ensemble, CNF, MAF 
from sbiax.train import train_ensemble
from sbiax.inference import nuts_sample
from sbiax.compression.nn import fit_nn

from affine import affine_sample

""" 
    Run NLE or NPE SBI with the moments of the 1pt matter PDF.

    - Run with just variance of PDF (a test)? 
    - diagonal of covariance for compression?

    - freezing 'nuisance parameters'

    - cumulants? What are Quijote conventions again? Default quijote is cumulants (k-stats are sample cumulants, unbiased estimators)?

    - scaling of inputs? summaries seem to be high magnitude 
    - covariance c1onditioning?
    
    - pretraining with linear sims vs fisher summaries is the same thing?
""" 

args = get_moments_sbi_args()

print("SEED", args.seed)

t0 = time.time()

config = moments_config(
    seed=args.seed + 1, 
    redshift=args.redshift, 
    linearised=args.linearised, 
    n_linear_sims=args.n_linear_sims,
    pre_train=args.pre_train
)

key = jr.key(config.seed)

(
    key, model_key, train_key, key_prior, 
    key_datavector, key_state, key_sample
) = jr.split(key, 7)

results_dir: str = get_results_dir(config)

# Dataset of simulations, parameters, covariance, ...
dataset: Dataset = get_data(config)

parameter_prior: Distribution = get_prior(config)

net_key, train_key = jr.split(key)

net = eqx.nn.MLP(
    dataset.data.shape[-1], 
    dataset.parameters.shape[-1], 
    width_size=32, 
    depth=2, 
    activation=jax.nn.tanh,
    key=net_key
)

preprocess_fn = lambda x: (x - dataset.data.mean(axis=0)) / dataset.data.std(axis=0)

net, losses = fit_nn(
    train_key, 
    net, 
    (jnp.asarray(preprocess_fn(dataset.data)), jnp.asarray(dataset.parameters)), 
    opt=optax.adam(1e-3), 
    n_batch=500, 
    patience=100
)

net_fn = lambda d: net(preprocess_fn(d))

X = jax.vmap(net_fn)(dataset.data)

plt.figure()
plt.loglog(losses)
plt.savefig(os.path.join(results_dir, "loss_nn.png"))
plt.close()
c = ChainConsumer()
c.add_chain(
    Chain(
        samples=make_df(dataset.parameters, parameter_strings=dataset.parameter_strings), 
        name="Params", 
        color="blue", 
        plot_cloud=True, 
        plot_contour=False
    )
)
c.add_chain(
    Chain(
        samples=make_df(X, parameter_strings=dataset.parameter_strings), 
        name="Summaries", 
        color="red", 
        plot_cloud=True, 
        plot_contour=False
    )
)
c.add_truth(
    Truth(location=dict(zip(dataset.parameter_strings, dataset.alpha)), name=r"$\pi^0$")
)
fig = c.plotter.plot()
plt.savefig(os.path.join(results_dir, "params.pdf"))
plt.close()

fig, axs = plt.subplots(1, dataset.alpha.size, figsize=(2. * dataset.alpha.size, 2.))
for p, ax in enumerate(axs):
    ax.scatter(dataset.parameters[:, p], X[:, p])
    ax.axline((0, 0), slope=1., color="k", linestyle="--")
    ax.set_xlim(dataset.lower[p], dataset.upper[p])
    ax.set_ylim(dataset.lower[p], dataset.upper[p])
plt.savefig(os.path.join(results_dir, "scatter.png"))
plt.show()

"""
    Build NDEs
"""
scaler = Scaler(
    X, dataset.parameters, use_scaling=config.maf.use_scaling
)

# NOTE: make per-NDE scalers...
ndes = get_ndes_from_config(
    config, event_dim=dataset.alpha.size, scalers=scaler, key=model_key
)
ensemble = Ensemble(ndes, sbi_type=config.sbi_type)

"""
    Pre-train NDEs on linearised data
"""

# Only pre-train if required and not inferring from linear simulations
if (
    (not config.linearised) and config.pre_train and (config.n_linear_sims is not None)
):
    pre_train_key, summaries_key = jr.split(key)

    # Pre-train data = linearised simulations
    # X_l, Y_l = get_linearised_data(config)
    # X_l = jax.vmap(s)(X_l, Y_l)

    # Pre-train data = Fisher summaries
    X_l, Y_l = get_fisher_summaries(
        summaries_key, 
        n=config.n_linear_sims, 
        parameter_prior=parameter_prior, 
        Finv=dataset.Finv
    )

    c = ChainConsumer()
    c.add_chain(
        Chain(
            samples=make_df(X_l, parameter_strings=dataset.parameter_strings), 
            name="Summaries: linearised data", 
            color="blue", 
            plot_cloud=True, 
            plot_contour=False
        )
    )
    c.add_truth(
        Truth(location=marker(dataset.alpha, dataset.parameter_strings), name=r"$\pi^0$")
    )
    fig = c.plotter.plot()
    plt.savefig(os.path.join(results_dir, "fisher_x.pdf"))
    plt.close()

    opt = getattr(optax, config.opt)(config.lr)

    ensemble, stats = train_ensemble(
        pre_train_key, 
        ensemble,
        train_mode=config.sbi_type,
        train_data=(X_l, Y_l), 
        opt=opt,
        n_batch=config.n_batch,
        patience=config.patience,
        n_epochs=config.n_epochs,
        tqdm_description="Training (pre-train)",
        show_tqdm=args.use_tqdm,
        results_dir=results_dir
    )

"""
    Train NDE on data
"""
opt = getattr(optax, config.opt)(config.lr)

ensemble, stats = train_ensemble(
    train_key, 
    ensemble,
    train_mode=config.sbi_type,
    train_data=(X, dataset.parameters), 
    opt=opt,
    n_batch=config.n_batch,
    patience=config.patience,
    n_epochs=config.n_epochs,
    tqdm_description="Training (data)",
    show_tqdm=args.use_tqdm,
    results_dir=results_dir
)

eqx.tree_serialise_leaves(os.path.join(results_dir, "ensemble.eqx"), ensemble)

""" 
    Sample and plot posterior for NDE with noisy datavectors
"""
ensemble = eqx.nn.inference_mode(ensemble)

# Generates linearised (or not) datavector at fiducial parameters
datavector = get_datavector(key_datavector, config)

x_ = net_fn(datavector)

log_prob_fn = ensemble.ensemble_log_prob_fn(x_, parameter_prior)

# state = jr.multivariate_normal(
#     key_state, x_, dataset.Finv, (2 * config.n_walkers,)
# )

# samples, weights = affine_sample(
#     key_sample, 
#     log_prob=log_prob_fn,
#     n_walkers=config.n_walkers, 
#     n_steps=config.n_steps + config.burn, 
#     burn=config.burn, 
#     current_state=state,
#     description="Sampling" 
# )

# samples_log_prob = jax.vmap(log_prob_fn)(samples)
# alpha_log_prob = log_prob_fn(jnp.asarray(dataset.alpha))


samples, samples_log_prob = nuts_sample(
    key_sample, log_prob_fn, prior=parameter_prior
)

posterior_df = make_df(samples, samples_log_prob, parameter_strings=dataset.parameter_strings)

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
plt.savefig(os.path.join(results_dir, "posterior.pdf"))
plt.close()


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
    description="Sampling" 
)

samples_log_prob = jax.vmap(log_prob_fn)(samples)
alpha_log_prob = log_prob_fn(jnp.asarray(dataset.alpha))

posterior_df = make_df(samples, samples_log_prob, parameter_strings=dataset.parameter_strings)

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
plt.savefig(os.path.join(results_dir, "posterior_affine.pdf"))
plt.close()

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