import argparse
from typing import Tuple, Optional, Literal
import os
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import PRNGKeyArray, Float, Array, jaxtyped
from beartype import beartype as typechecker
import numpy as np
from ml_collections import ConfigDict
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer, Chain, Truth
from tensorflow_probability.substrates.jax.distributions import Distribution

typecheck = jaxtyped(typechecker=typechecker)

from configs import (
    cumulants_config, 
    ensembles_cumulants_config,
    get_base_results_dir, 
    get_base_posteriors_dir,
    get_results_dir, 
    get_multi_z_posterior_dir, 
    get_posteriors_dir, 
    get_cumulants_sbi_args, 
    get_cumulants_multi_z_args,
    get_ndes_from_config
)
from cumulants import (
    Dataset, 
    get_data, 
    get_prior, 
    get_linear_compressor, 
    get_datavector, 
    get_linearised_data,
    get_parameter_strings
)

from sbiax.ndes import CNF, MAF, Scaler
from sbiax.compression.linear import mle
from sbiax.inference import nuts_sample
from sbiax.utils import make_df, marker

from ensemble import Ensemble, MultiEnsemble
from affine import affine_sample


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


@typecheck
def get_z_config_and_datavector(
    key: PRNGKeyArray, 
    seed: int,
    config: ConfigDict, 
    redshift: float, 
    linearised: bool = True, 
    order_idx: list[int] = [0, 1, 2],
    compression: Literal["linear", "nn"] = "linear",
    reduced_cumulants: bool = True,
    pre_train: bool = False, 
    sbi_type: str = "nle",
    exp_name_format: str = "z={}_m={}",
    n_datavectors: int = 1
) -> tuple[
    Ensemble,
    Float[Array, "n p"],
    Float[Array, "n d"],
    Distribution,
    Float[Array, "p"],
    Float[Array, "p p"],
    Float[Array, "d d"],
    Float[Array, "d"],
    Float[Array, "p d"]
]:
    """ 
        Get config and datavector associated with a redshift z to load the experiment 
    """

    key_datavector, key_model = jr.split(key)

    # Get config and change redshift to load each ensemble and datavector
    # config_z = cumulants_config(seed=seed,)
    config_z = cumulants_config(
        seed=args.seed, 
        redshift=redshift, 
        linearised=linearised, 
        compression=compression,
        reduced_cumulants=reduced_cumulants,
        order_idx=order_idx,
        pre_train=pre_train
    )

    # Set to current redshift
    config_z.exp_name = exp_name_format.format(redshift, "".join(map(str, config.order_idx)))
    config_z.redshift = redshift

    # Ensure matching NLE or NPE and linearisation
    config_z.sbi_type = config.sbi_type
    config_z.linearised = config.linearised

    config_z.reduced_cumulants = reduced_cumulants
    config_z.compression = compression

    # Get datas, compressor
    dataset: Dataset = get_data(config_z)

    compressor = get_linear_compressor(config_z)

    # NOTE: not supported for NN compression
    if config_z.use_pca:
        # Compress simulations as usual 
        X = jax.vmap(compressor)(dataset.data, dataset.parameters)

        # Standardise before PCA (don't get tricked by high variance due to units)
        X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)

        # Fit whitening-PCA to compressed simulations
        pca = PCA(num_components=dataset.alpha.size)
        pca.fit(X)
        
        # Reparameterize compression with both transforms
        compression_fn = lambda d, p: pca.transform(compressor(d, p))
    else:
        compression_fn = lambda d, p: compressor(d, p)

    # Generates linearised (or not) datavector 
    datavectors = get_datavector(key_datavector, config_z, n=n_datavectors)

    if datavectors.ndim == 1:
        datavectors = datavectors[jnp.newaxis, ...]

    # Compressed datavector
    x_ = jax.vmap(s, in_axes=(0, None))(datavectors, dataset.alpha) # NOTE: at true parameters

    # Compress whole simulation dataset 
    X = jax.vmap(compression_fn)(dataset.data, dataset.parameters)

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
    ensemble_path = os.path.join(
        get_results_dir(config_z, args=args), "ensemble.eqx"
    )
    ensemble = eqx.tree_deserialise_leaves(ensemble_path, ensemble)

    # Quijote prior (same for all z)
    prior = get_prior(config_z) 

    return (
        ensemble, 
        x_, 
        datavectors,
        prior, 
        jnp.asarray(dataset.alpha), 
        jnp.asarray(dataset.Finv), 
        jnp.asarray(dataset.C), 
        jnp.mean(dataset.fiducial_data, axis=0), 
        jnp.mean(dataset.derivatives, axis=0)
    )


@typecheck
def get_multi_redshift_mle(
    pi: Float[Array, "p"], 
    d: Float[Array, "... d"], 
    Finv: Float[Array, "p p"], 
    mus: list[Float[Array, "d"]], 
    covariances: list[Float[Array, "d d"]], 
    derivatives: list[Float[Array, "p d"]]
) -> Float[Array, "p"]:
    """ Chi2 minimisation using block-diagonalised simulation-estimated data covariance """

    # Covariances, derivatives for all z datas
    C = block_diag(*covariances)
    Cinv = jnp.linalg.inv(C) # Hartlap? Individual covariances are corrected?

    # Concatenate objects across redshift to match block-diagonal covariance
    derivatives = jnp.concatenate(derivatives, axis=1) # Stack on data axis
    mu = jnp.concatenate(mus)
    d = jnp.concatenate(d)

    print("D, mu, C, dmu:", d.shape, mu.shape, C.shape, derivatives.shape)

    return pi + jnp.linalg.multi_dot([Finv, derivatives, Cinv, d - mu]) # d is z-concatenated datavector


@typecheck
def maybe_vmap_multi_redshift_mle(
    pi: Float[Array, "p"], 
    datavectors: list[Float[Array, "n d"]], 
    Finv: Float[Array, "p p"], 
    mus: list[Float[Array, "d"]], 
    covariances: list[Float[Array, "d d"]], 
    derivatives: list[Float[Array, "p d"]]
) -> Float[Array, "n p"]:

    # Vmap MLE function over datavectors if plural
    # datavectors multiple per redshift, covariances are one per redshift...
    fn = lambda d: get_multi_redshift_mle(
        pi, d, Finv, mus, covariances, derivatives
    )

    print("DATAVECTORS", [_.shape for _ in datavectors])

    # Shape: (n, z, d)
    datavectors = jnp.stack(datavectors, axis=1) # Stack list of datavectors ... NOTE: may be wrongly shaped...
    assert datavectors.shape.ndim == 3
    n_realisations, *_ = datavectors.shape

    print("DATAVECTORS", datavectors.shape)

    # Vmaps over first axis, concatenates them inside 'fn'
    x = jax.vmap(fn)(datavectors) 

    return x


if __name__ == "__main__":

    key = jr.key(0)

    args = get_cumulants_multi_z_args()

    config = ensembles_cumulants_config(
        seed=args.seed, # Defaults if run without argparse args
        sbi_type=args.sbi_type, 
        linearised=args.linearised,
        reduced_cumulants=args.reduced_cumulants,
        order_idx=args.order_idx,
        compression=args.compression,
        pre_train=args.pre_train
    )

    parameter_strings = get_parameter_strings()

    linear_str = "linear" if config.linearised else ""

    # Where SBI's are saved (add on suffix for experiment details)
    posteriors_dir = get_base_posteriors_dir()
    if config.reduced_cumulants:
        exps_dir = "{}reduced_cumulants_multi_z/".format(posteriors_dir) # Import this from a constants file
    else:
        exps_dir = "{}cumulants_multi_z/".format(posteriors_dir) # Import this from a constants file
    figs_dir = "{}figs/".format(posteriors_dir)

    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir, exist_ok=True)

    # Loop over redshifts; loading ensembles and datavectors
    datavectors  = [] # I.i.d. datavectors, tuple'd for each redshift (plural measurements in tuple)
    x_s          = [] # Summaries of these datavectors
    ensembles    = [] # Ensembles of NDEs trained on simulations at each redshift
    covariances  = [] # Covariance matrices of simulations at each redshift
    derivatives_ = [] # Derivatives of theory model at each redshift 
    mus          = [] # Expectation model at each redshift 
    F            = 0. # Add independent information from data at each redshift 
    for z, redshift in enumerate(config.redshifts):
        print("@" * 80)
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
            seed=args.seed,
            config=config, 
            order_idx=args.order_idx,
            compression=args.compression,
            reduced_cumulants=args.reduced_cumulants,
            redshift=redshift, 
            n_datavectors=args.n_datavectors,
            pre_train=args.pre_train
        ) # x_ norm'd in model

        # Add Fisher information from redshift (independent)
        F += jnp.linalg.inv(Finv_z)

        derivatives_.append(derivatives)
        mus.append(mu)
        covariances.append(C)
        x_s.append(x_z)
        datavectors.append(datavector)
        ensembles.append(ensemble)      

    multi_ensemble = MultiEnsemble(
        ensembles, prior=prior, sbi_type=config.sbi_type
    ) # Same prior for each z

    Finv = jnp.linalg.inv(F) # Combined Fisher information over all redshifts

    print("Finv", Finv)

    # Sample the multiple-redshift-ensemble posterior
    for n_posterior in range(args.n_posteriors_sample):

        print("Sampling posterior {} (all redshifts, datavectors)".format(n_posterior))

        key_sample, key_state = jr.split(jr.fold_in(key, n_posterior))

        if args.verbose:
            plt.figure()
            plt.imshow(block_diag(*covariances))
            plt.savefig("block_diag_covariance.png")
            plt.close()

        # Compress datavectors concatenated over redshift, using block-diagonal covariance
        x_ = maybe_vmap_multi_redshift_mle( 
            alpha, 
            datavectors, 
            Finv=Finv,
            mus=mus, 
            covariances=covariances, 
            derivatives=derivatives_
        )

        if args.verbose:
            print("x_", x_.shape, x_)

        # Sample posterior across multiple redshifts
        log_prob_fn = multi_ensemble.get_multi_ensemble_log_prob_fn(x_s)

        # samples, samples_log_prob = nuts_sample(
        #     key_sample, 
        #     log_prob_fn=log_prob_fn, # NOTE: is it right to pass list of datavectors not the MLE above?
        #     prior=prior
        # )

        state = jr.multivariate_normal(
            key_state, alpha, Finv, (2 * config.n_walkers,) # x_
        )

        if args.verbose:
            print("State:", state)

        samples, weights = affine_sample(
            key_sample, 
            log_prob=log_prob_fn,
            n_walkers=config.n_walkers, 
            n_steps=config.n_steps + config.burn, 
            burn=config.burn, 
            current_state=state,
            description="Sampling",
            show_tqdm=True # args.use_tqdm
        )

        samples_log_prob = jax.vmap(log_prob_fn)(samples)
        alpha_log_prob = log_prob_fn(jnp.asarray(alpha))
        
        # Remove NaNs
        # ix = jnp.isfinite(samples_log_prob)
        # samples_log_prob = samples_log_prob[ix]
        # samples = samples[ix]
        # assert jnp.all(jnp.isfinite(samples_log_prob))
        # assert jnp.all(jnp.isfinite(samples))

        # Save posterior, Fisher and summary
        posterior_save_dir = get_multi_z_posterior_dir(
            config, default(args.sbi_type, "nle")
        )
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
        posterior_df = make_df(
            samples, samples_log_prob, parameter_strings=parameter_strings
        )
        c.add_chain(Chain(samples=posterior_df, name="SBI", color="r"))
        # If using multiple datavectors, plot them individually
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
                "multi_ensemble_posterior_cumulants_{}_{}.pdf".format(n_posterior, linear_str))
        )
        plt.close()

        # Marginalise over all but Om, s8
        ix = np.array([0, -1]) # Indices for Om, s8
        parameter_names_ = [parameter_strings[_] for _ in ix]

        c = ChainConsumer()
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
        posterior_df = make_df(
            samples[:, ix], samples_log_prob, parameter_strings=parameter_names_
        )
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
                "multi_ensemble_posterior_marginalised_cumulants_{}_{}.pdf".format(n_posterior, linear_str)
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