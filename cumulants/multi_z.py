import argparse
from typing import Tuple
import os
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key
import numpy as np
from ml_collections import ConfigDict
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer, Chain, Truth

from configs import get_base_results_dir, get_results_dir, get_multi_z_posterior_dir
from configs.moments import moments_config, ensembles_moments_config 
from data.moments import Dataset, get_data, get_linear_compressor, get_datavector, get_prior, get_parameter_strings
from sbiax.ndes import Ensemble, MultiEnsemble, get_ndes_from_config
from sbiax.ndes import CNF, MAF, Scaler
from sbiax.compression.linear import _mle
from sbiax.inference import nuts_sample
from sbiax.inference.nle import affine_sample
from sbiax.utils import make_df, marker


def default(v, d):
    return v if v is not None else d


"""
    Sample a posterior with a uniform physics-parameter prior
    and a likelihood function made from separate flows trained
    on data from different redshifts. 

    This script takes a seed and loads the flows for each seed
    for each redshift.
    
    Datavector is made of one measurement at each redshift, 
    assumed to be independent between redshifts. 
    - Can use more than one datavector now, for scaling as a survey.

    Ensure scaling is switched on / off and EVERYTHING matches
    training configs.
    - Load configs for each flow based on redshift and experiment directory

    Meta all-redshift config `ensembles_bulk_pdfs_config` tells how to 
    sample the posterior made of the separate flows.

    Save posteriors at each z, to see what adding them together does
"""

parser = argparse.ArgumentParser(
    description="Run SBI experiment with moments of the matter PDF."
)
parser.add_argument(
    "-s", 
    "--seed", 
    type=int, 
    default=0,
    help="Seed for random number generation."
)
parser.add_argument(
    "-l",
    "--linearised", 
    action=argparse.BooleanOptionalAction, 
    default=True,
    help="Linearised model for datavector."
)
parser.add_argument(
    "-n",
    "--n_linear_sims", 
    type=int,
    default=100_000,
    action=argparse.BooleanOptionalAction, 
    help="Number of linearised simulations (used for pre-training if non-linear simulations and requested)."
)
parser.add_argument(
    "-p",
    "--pre-train", 
    action=argparse.BooleanOptionalAction, 
    help="Pre-train (only) when using non-linearised model for datavector. Pre-train on linearised simulations."
)
parser.add_argument(
    "-z",
    "--redshift", 
    type=float,
    choices=[0.0, 0.5, 1.0],
    default=0.,
    help="Redshift of simulations."
)
parser.add_argument(
    "-t",
    "--sbi_type", 
    type=str,
    choices=["nle", "npe"],
    default="nle",
    help="Method of SBI: neural likelihood (NLE) or posterior (NPE)."
)
args = parser.parse_args()


def get_z_config_and_datavector(
    key: PRNGKeyArray, 
    seed: int,
    config: ConfigDict, 
    redshift: float, 
    linearised: bool = False, 
    pre_train: bool = False, 
    sbi_type: str = "nle",
    exp_name_format: str = "z={}",
    n_datavectors: int = 1
) -> Tuple:
    """ Get config and datavector associated with a redshift z """

    key_datavector, key_model = jr.split(key)

    # Get config and change redshift to load each ensemble and datavector
    config_z = moments_config(seed=seed)

    # Set to current redshift
    config_z.exp_name = exp_name_format.format(redshift)
    config_z.redshift = float(redshift)

    # Ensure matching NLE or NPE and linearisation
    config_z.sbi_type = config.sbi_type
    config_z.linearised = config.linearised

    # Get datas, compressor
    dataset: Dataset = get_data(config_z)

    s = get_linear_compressor(config_z)

    # Generates linearised (or not) datavector 
    datavectors = get_datavector(key_datavector, config_z, n=n_datavectors)

    if datavectors.ndim == 1:
        datavectors = datavectors[jnp.newaxis, ...]

    # Compressed datavector
    x_ = jax.vmap(s, in_axes=(0, None))(datavectors, dataset.alpha) 

    # Compress whole simulation dataset 
    X = jax.vmap(s)(dataset.data, dataset.parameters)

    # Input scaler functions for individual NDEs
    scalers = [
        Scaler(X, dataset.parameters, use_scaling=nde.use_scaling)
        for nde in config_z.ndes
    ]

    # Get NDEs
    ndes = get_ndes_from_config(
        config_z, event_dim=dataset.alpha.size, scalers=scalers, key=key_model
    )

    # Ensemble of NDEs
    ensemble = Ensemble(ndes, sbi_type=config_z.sbi_type)

    # Load Ensemble
    ensemble_path = os.path.join(get_results_dir(config_z), "ensemble.eqx")
    ensemble = eqx.tree_deserialise_leaves(ensemble_path, ensemble)

    # Quijote prior (same for all z)
    prior = get_prior(config_z) 

    return (
        ensemble, 
        x_, 
        datavectors,
        prior, 
        dataset.alpha, 
        dataset.Finv, 
        dataset.C, 
        dataset.fiducial_data.mean(axis=0), 
        dataset.derivatives.mean(axis=0)
    )


def maybe_vmap_multi_redshift_mle(pi, datavectors, Finv, mus, covariances, derivatives):
    # Vmap MLE function over datavectors if plural
    # datavectors multiple per redshift, covariances is one per redshift...
    fn = lambda d: get_multi_redshift_mle(
        pi, d, Finv, mus, covariances, derivatives
    )
    datavectors = jnp.stack(datavectors, axis=1)
    _, n_realisations, _ = datavectors.shape
    if n_realisations > 1:
        # Stack on 'realisations' axis (not redshifts axis)
        x = jax.vmap(fn)(datavectors) # vmaps over first axis, concatenates them inside 'fn'
    else:
        x = fn(datavectors)
    return x


def get_multi_redshift_mle(pi, d, Finv, mus, covariances, derivatives):
    """ Chi2 minimisation using block-diagonalised simulation-estimated data covariance """
    # Covariances, derivatives for all z datas
    C = block_diag(*covariances)
    Cinv = jnp.linalg.inv(C) # Hartlap?
    derivatives = jnp.concatenate(derivatives, axis=1)
    mu = jnp.concatenate(mus)
    d = jnp.concatenate(d)
    print(d.shape, mu.shape, C.shape, derivatives.shape)
    return pi + jnp.linalg.multi_dot([Finv, derivatives, Cinv, d - mu]) # d is z-concatenated datavector


key = jr.key(0)

config = ensembles_moments_config(
    seed=default(args.seed, 0), # Defaults if run without argparse args
    sbi_type=default(args.sbi_type, "nle"), 
    linearised=default(args.linearised, True),
    pre_train=default(args.pre_train, False)
)

parameter_strings = get_parameter_strings()

linear_str = "linear" if config.linearised else ""

# Where SBI's are saved (add on suffix for experiment details)
results_dir = get_base_results_dir()
exps_dir = "{}moments/".format(results_dir) # Import this from a constants file
figs_dir = "{}figs/".format(results_dir)

n_posteriors_sample = 1   # Number of posteriors to sample (repeated datavectors?)
n_datavectors       = 100 # Number of i.i.d. datavectors per redshift

# Loop over redshifts; loading ensembles and datavectors
datavectors = []  # I.i.d. datavectors, tuple'd for each redshift (plural measurements in tuple)
x_s = []          # Summaries of these datavectors
ensembles = []    # Ensembles of NDEs trained on simulations at each redshift
covariances = []  # Covariance matrices of simulations at each redshift
derivatives_ = [] # Derivatives of theory model at each redshift 
mus = []          # Expectation model at each redshift 
F = 0.            # Add independent information from data at each redshift 
for z, redshift in enumerate(config.redshifts):
    print("Getting datavector(s) for redshift={}".format(redshift))

    key_z = jr.fold_in(key, z)

    # Load ensemble configuration, datavector/summary, prior, covariance, Fisher, derivatives
    (
        ensemble, 
        x_z, 
        datavector, 
        prior, 
        alpha, 
        Finv_z, 
        C, 
        mu, 
        derivatives
    ) = get_z_config_and_datavector(
        key_z, 
        args.seed,
        config, 
        redshift=redshift, 
        n_datavectors=n_datavectors
    ) # x_ norm'd in model

    # Add Fisher information from redshift (independent)
    F += jnp.linalg.inv(Finv_z)

    derivatives_.append(derivatives)
    mus.append(mu)
    covariances.append(C)
    x_s.append(x_z)
    datavectors.append(datavector)
    ensembles.append(ensemble)      

multi_ensemble = MultiEnsemble(ensembles, prior=prior) # Same prior for each z

Finv = jnp.linalg.inv(F) # Combined Fisher information over all redshifts

# Sample the multiple-redshift-ensemble posterior
for n_posterior in range(n_posteriors_sample):
    print("Sampling posterior {} (all redshifts, datavectors)".format(n_posterior))

    key_sample, key_state = jr.split(jr.fold_in(key, n_posterior))

    # Compress datavectors concatenated over redshift, using block-diagonal covariance
    x_ = maybe_vmap_multi_redshift_mle( 
        alpha, 
        datavectors, 
        Finv=Finv,
        mus=mus, 
        covariances=covariances, 
        derivatives=derivatives_
    )
    print("x_", x_.shape)

    # Sample posterior across multiple redshifts
    log_prob_fn = multi_ensemble.get_multi_ensemble_log_prob_fn(x_s)

    samples, samples_log_prob = nuts_sample(
        key_sample, 
        log_prob_fn=log_prob_fn, # NOTE: is it right to pass list of datavectors not the MLE above?
        prior=prior
    )
    
    # Remove NaNs
    ix = jnp.isfinite(samples_log_prob)
    samples_log_prob = samples_log_prob[ix]
    samples = samples[ix]
    assert jnp.all(jnp.isfinite(samples_log_prob))
    assert jnp.all(jnp.isfinite(samples))

    # Save posterior, Fisher and summary
    posterior_save_dir = get_multi_z_posterior_dir(config, default(args.sbi_type, "nle"))
    if not os.path.exists(posterior_save_dir):
        os.makedirs(posterior_save_dir, exist_ok=True)

    posterior_filename = os.path.join(
        posterior_save_dir, "posterior_{}.npz".format(args.seed)
    )
    np.savez(
        posterior_filename,
        samples=samples, 
        samples_log_prob=samples_log_prob,
        Finv=Finv,
        summary=x_ # Is this correct one to save?
    )
    print("POSTERIOR FILENAME", posterior_filename)

    c = ChainConsumer() 
    c.add_chain(
        Chain.from_covariance(
            alpha,
            Finv, # NOTE: Get multi redshift Fisher matrix, use a multi-inference config
            columns=parameter_strings,
            name=r"$F_{\Sigma^{-1}}$",
            color="k",
            linestyle=":",
            shade_alpha=0.
        )
    )
    posterior_df = make_df(samples, samples_log_prob, parameter_strings)
    c.add_chain(Chain(samples=posterior_df, name="SBI", color="r"))
    if x_.ndim > 1:
        for i, _x_ in enumerate(x_):
            c.add_marker(
                location=marker(_x_, parameter_strings), 
                name=r"$\hat{x}$ " + str(i), 
                color="b"
            )
    else:
        c.add_marker(
            location=marker(x_, parameter_strings), 
            name=r"$\hat{x}$", 
            color="b"
        )
    c.add_marker(
        location=marker(alpha, parameter_strings), 
        name=r"$\alpha$", 
        color="#7600bc"
    )
    fig = c.plotter.plot()
    plt.savefig(
        os.path.join(
            figs_dir, 
            "multi_ensemble_posterior_moments_{}_{}.pdf".format(n_posterior, linear_str))
    )
    plt.close()

    # Marginalise over all but Om, s8
    ix = np.array([0, -1]) # Indices for Om, s8
    parameter_names_ = [parameter_strings[_] for _ in ix]

    c = ChainConsumer()
    # NOTE: Get multi redshift Fisher matrix, use a multi-inference config
    c.add_chain(
        Chain.from_covariance(
            alpha[ix],
            Finv[ix, :][:, ix],
            columns=parameter_names_,
            name=r"$F_{\Sigma^{-1}}$",
            color="k",
            linestyle=":",
            shade_alpha=0.
        )
    )
    posterior_df = make_df(samples[:, ix], samples_log_prob, parameter_names_)
    c.add_chain(Chain(samples=posterior_df, name="SBI", color="r"))
    if x_.ndim > 1:
        for i, _x_ in enumerate(x_):
            c.add_marker(
                location=marker(_x_, parameter_names_), 
                name=r"$\hat{x}$ " + str(i), 
                color="b"
            )
    else:
        c.add_marker(
            location=marker(x_[ix], parameter_names_), 
            name=r"$\hat{x}$", 
            color="b"
        )
    c.add_marker(
        location=marker(alpha[ix], parameter_names_), 
        name=r"$\alpha$", 
        color="#7600bc"
    )
    fig = c.plotter.plot()
    plt.savefig(
        os.path.join(
            figs_dir, 
            "multi_ensemble_posterior_marginalised_moments_{}_{}.pdf".format(n_posterior, linear_str)
        )
    )
    plt.close()



# AFFINE SAMPLING
# n_walkers = 1000
# n_steps = 200
# burn = int(0.1 * n_steps)

# Multi-ensemble likelihood (only applies prior to ensembles once)
# log_prob_fn = multi_ensemble.get_multi_ensemble_log_prob_fn(x_s)

# state = jr.multivariate_normal(key_state, alpha, Finv, (2 * n_walkers,)) # prior.sample((2 * n_walkers,), seed=key_state)
# assert jnp.all(jnp.isfinite(state))

# samples, weights = affine_sample(
#     key_sample, 
#     log_prob=log_prob_fn,
#     n_walkers=n_walkers, 
#     n_steps=n_steps + burn, 
#     burn=burn, 
#     current_state=state,
#     description="Sampling" 
# )
# samples_log_prob = jax.vmap(log_prob_fn)(samples)