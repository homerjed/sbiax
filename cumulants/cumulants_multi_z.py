from typing import Tuple, Optional, Literal, Callable
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
from tqdm.auto import trange
from chainconsumer import ChainConsumer, Chain, Truth
from tensorflow_probability.substrates.jax.distributions import Distribution

from sbiax.ndes import CNF, MAF, Scaler
from sbiax.compression.linear import mle
from sbiax.inference import nuts_sample
from sbiax.utils import make_df, marker

from configs import cumulants_config, bulk_cumulants_config
from configs.ensembles_configs import ensembles_cumulants_config, ensembles_bulk_cumulants_config
from configs.configs import (
    get_results_dir, 
    get_multi_z_posterior_dir, 
    get_posteriors_dir, 
    get_ndes_from_config
)
from configs.args import get_cumulants_sbi_args, get_cumulants_multi_z_args
from data.constants import get_base_posteriors_dir, get_save_and_load_dirs
from data.cumulants import (
    CumulantsDataset,
    Dataset, 
    get_data, 
    get_prior, 
    get_linear_compressor, 
    get_datavector, 
    get_linearised_data,
    get_parameter_strings
)
from data.pdfs import (
    BulkCumulantsDataset, 
    get_bulk_dataset, 
    get_multi_z_bulk_pdf_fisher_forecast
)
from cumulants_ensemble import Ensemble, MultiEnsemble
from compression.pca import PCA
from affine import affine_sample
from utils.utils import get_dataset_and_config, finite_samples_log_prob

typecheck = jaxtyped(typechecker=typechecker)

CompressionFn = Callable[[Float[Array, "d"], Float[Array, "p"]], Float[Array, "p"]]

jax.clear_caches()


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
    config: ConfigDict, 
    redshift: float, 
    linearised: bool = True, 
    order_idx: list[int] = [0, 1, 2],
    freeze_parameters: bool = False,
    compression: Literal["linear", "nn"] = "linear",
    reduced_cumulants: bool = True,
    n_linear_sims: int = 10_000,
    pre_train: bool = False, 
    sbi_type: str = "nle",
    exp_name_format: str = "z={}_m={}",
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
    Float[Array, "d d"],
    Float[Array, "d"],
    Float[Array, "p d"]
]:
    """ 
        Get config and datavector associated with a redshift z to load the experiment 
        for use in an ensemble of SBI likelihoods at different redshifts, for the bulk
        or tails datasets.
        - get config at redshift
        - load datasets, compression function and  
    """

    key_datavector, key_model = jr.split(key)

    # Change seed for datavector sampling without changing any other seed
    if seed_datavector is not None:
        key_datavector = jr.fold_in(key_datavector, seed_datavector)

    _dataset, _config = get_dataset_and_config(bulk_or_tails) 

    # Get config and change redshift to load each ensemble and datavector
    config_z = _config(
        seed=seed, 
        redshift=redshift, 
        reduced_cumulants=reduced_cumulants,
        sbi_type=sbi_type,
        linearised=linearised, 
        compression=compression,
        order_idx=order_idx,
        freeze_parameters=freeze_parameters,
        n_linear_sims=n_linear_sims,
        pre_train=pre_train
    )

    bulk_config_z = bulk_cumulants_config(
        seed=seed, 
        redshift=redshift, 
        reduced_cumulants=reduced_cumulants,
        sbi_type=sbi_type,
        linearised=linearised, 
        compression=compression,
        n_linear_sims=n_linear_sims,
        order_idx=order_idx,
        freeze_parameters=freeze_parameters,
        pre_train=pre_train
    )

    if verbose: 
        print("bulk or tails", bulk_or_tails)
        print(config_z)

    # Set to current redshift, ensure matching NLE or NPE and linearisation
    config_z.exp_name = exp_name_format.format(redshift, "".join(map(str, config.order_idx)))

    # Cumulants bulk and tails datasets
    cumulants_dataset = _dataset(
        config_z, results_dir=None, verbose=verbose
    )
    dataset: Dataset = cumulants_dataset.data

    bulk_cumulants_dataset = BulkCumulantsDataset(
        bulk_config_z, pdfs=("pdf" in bulk_or_tails), results_dir=None, verbose=verbose
    )
    bulk_dataset: Dataset = bulk_cumulants_dataset.data # For Fisher forecast comparisons

    # Parameter prior
    parameter_prior: Distribution = cumulants_dataset.prior # Quijote prior (same for all z, only applied once with combined z-likelihoods)

    # Compression function (neural network or linear)
    compression_fn: CompressionFn = cumulants_dataset.get_compression_fn() 
 
    # Sample datavector(s) at the fiducial parameters
    datavectors = cumulants_dataset.get_datavector(key_datavector, n=n_datavectors) # Generates linearised (or not) datavector 

    if datavectors.ndim == 1:
        datavectors = datavectors[jnp.newaxis, ...] # Add axis for vmapping...

    # Compressed datavectors at fiducial parameters
    x_ = jax.vmap(compression_fn, in_axes=(0, None))(datavectors, dataset.alpha) 

    # Compressed hypercube simulations (defines mu/std of NDE scalers)
    X = jax.vmap(compression_fn)(dataset.data, dataset.parameters)

    # Input scaler functions for individual NDEs (NOTE: scaling on X which may need to be linearised or not) 
    scaler = Scaler(
        X, dataset.parameters, use_scaling=config_z.use_scalers # NOTE: data_preprocess_fn ...
    )

    # Get NDEs
    ndes = get_ndes_from_config(
        config_z, 
        event_dim=dataset.alpha.size, 
        scalers=scaler, 
        use_scalers=config_z.use_scalers,
        key=key_model
    )

    # Ensemble of NDEs
    ensemble = Ensemble(ndes, sbi_type=config_z.sbi_type)

    # Load ensemble
    ensemble_path = os.path.join(get_results_dir(config_z, args=args), "ensemble.eqx")
    ensemble = eqx.tree_deserialise_leaves(ensemble_path, ensemble)

    print("Loaded ensemble from:\n\t", ensemble_path)

    if verbose:
        print("Datavectors", datavectors.shape)

    return (
        ensemble, 
        x_, 
        datavectors,
        parameter_prior, 
        jnp.asarray(dataset.alpha), 
        # Scale data and parameter covariances by number of measurements (these are not used in MLEs)
        jnp.asarray(bulk_dataset.Finv) / n_datavectors, 
        jnp.asarray(dataset.Finv) / n_datavectors, 
        jnp.asarray(dataset.C) / n_datavectors, 
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
    derivatives: list[Float[Array, "p d"]],
    *,
    verbose: bool = False
) -> Float[Array, "p"]:
    """ Chi2 minimisation using block-diagonalised simulation-estimated data covariance """

    # Covariances, derivatives for all z datas
    C = block_diag(*covariances)
    Cinv = jnp.linalg.inv(C) # Hartlap? Individual covariances are corrected?

    # Concatenate objects across redshift to match block-diagonal covariance
    derivatives = jnp.concatenate(derivatives, axis=1) # Stack on data axis, having averaged over realisations
    mu = jnp.concatenate(mus)
    d = jnp.concatenate(d)

    if verbose:
        print("D, mu, C, dmu:", d.shape, mu.shape, C.shape, derivatives.shape)

    return pi + jnp.linalg.multi_dot([Finv, derivatives, Cinv, d - mu]) # d is z-concatenated datavector


@typecheck
def maybe_vmap_multi_redshift_mle(
    pi: Float[Array, "p"], 
    datavectors: list[Float[Array, "n d"]], 
    Finv: Float[Array, "p p"], 
    mus: list[Float[Array, "d"]], 
    covariances: list[Float[Array, "d d"]], 
    derivatives: list[Float[Array, "p d"]],
    *,
    verbose: bool = False
) -> Float[Array, "n p"]:

    # Vmap MLE function over datavectors if plural
    # datavectors multiple per redshift, covariances are one per redshift...
    fn = lambda d: get_multi_redshift_mle(
        pi, d, Finv, mus, covariances, derivatives, verbose=verbose
    )

    if verbose:
        print("DATAVECTORS", [_.shape for _ in datavectors])

    # Shape: (n, z, d)
    datavectors = jnp.stack(datavectors, axis=1) # Stack list of datavectors ... NOTE: may be wrongly shaped...

    assert datavectors.ndim == 3 # (n, n_cumulants, n_scales)

    if verbose:
        print("DATAVECTORS", datavectors.shape)

    # Vmaps over first axis, concatenates them inside 'fn'
    x = jax.vmap(fn)(datavectors) 

    return x


if __name__ == "__main__":

    key = jr.key(0) # Only for datavectors, split for each redshift, datavector and separate posterior

    args = get_cumulants_multi_z_args()

    # Multi-z inference concerning the bulk or bulk + tails
    if args.bulk_or_tails == "tails":
        ensembles_config = ensembles_cumulants_config
    if args.bulk_or_tails == "bulk" or args.bulk_or_tails == "bulk_pdf":
        ensembles_config = ensembles_bulk_cumulants_config

    config = ensembles_config(
        seed=args.seed, # Defaults if run without argparse args
        sbi_type=args.sbi_type, 
        linearised=args.linearised,
        n_linear_sims=args.n_linear_sims,
        compression=args.compression,
        reduced_cumulants=args.reduced_cumulants,
        redshifts=args.redshifts,
        order_idx=args.order_idx,
        pre_train=args.pre_train,
        freeze_parameters=args.freeze_parameters
    )

    # Get the bulk Fisher forecast for all redshifts 
    # but easier to load frozen or not since it autosaves...
    data_dir, _, _ = get_save_and_load_dirs()
    try:
        Finv_bulk_pdfs_all_z = np.load(
            os.path.join(
                data_dir, 
                "Finv_bulk_pdfs_all_z_{}.npy".format(
                    # "".join(map(str, args.order_idx)), NOTE: no cumulants associated with bulk pdf?!
                    "f" if args.freeze_parameters else "nf"
                )
            )
        )
    except:
        Finv_bulk_pdfs_all_z = get_multi_z_bulk_pdf_fisher_forecast(args)

        np.save(
            os.path.join(
                data_dir, 
                "Finv_bulk_pdfs_all_z_{}.npy".format(
                    "f" if args.freeze_parameters else "nf"
                )
            ),
            Finv_bulk_pdfs_all_z
        )

    parameter_strings = get_parameter_strings()

    linear_str = "linearised" if config.linearised else "nonlinearised"
    pretrain_str = "pretrain" if config.pre_train else "nopretrain"

    # Where SBI's are saved (add on suffix for experiment details)
    posteriors_dir = get_base_posteriors_dir()

    # Save location for posterior plots
    figs_dir = "{}figs/{}{}{}{}{}{}{}{}".format(
        posteriors_dir, 
        "frozen/" if config.freeze_parameters else "nonfrozen/",
        "{}/".format(args.bulk_or_tails),
        "reduced_cumulants/" if config.reduced_cumulants else "cumulants/",
        config.sbi_type + "/", 
        config.compression + "/",
        "linearised/" if config.linearised else "nonlinearised/",
        "pretrain/" if config.pre_train else "nopretrain/",
        "z={}_m={}".format( 
            "".join(map(str, config.redshifts)),
            "".join(map(str, config.order_idx))
        )
    )

    parts = [
        "figs",
        "frozen" if config.freeze_parameters else "nonfrozen",
        args.bulk_or_tails,
        "reduced_cumulants" if config.reduced_cumulants else "cumulants",
        config.sbi_type,
        "linearised" if config.linearised else "nonlinearised",
        config.compression,
        "pretrain" if config.pre_train else "nopretrain",
        # config.exp_name if include_exp and config.exp_name else None,
        "z={}_m={}".format( 
            "".join(map(str, config.redshifts)),
            "".join(map(str, config.order_idx))
        )
    ]
    path_str = "/".join(filter(None, parts)) + "/"

    figs_dir = os.path.join(posteriors_dir, path_str)
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir, exist_ok=True)

    print("MULTI-Z FIGS_DIR:\n\t", figs_dir)

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
        F            = jnp.zeros(()) # Add independent information from data at each redshift 
        F_bulk       = jnp.zeros(())

        with trange(len(config.redshifts), desc="Multi-z", colour="magenta") as bar:
            for _, (z, redshift) in zip(bar, enumerate(config.redshifts)):

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
                    bulk_Finv_z,
                    Finv_z, 
                    C, 
                    mu, 
                    derivatives
                ) = get_z_config_and_datavector(
                    key_z, 
                    seed=args.seed,
                    config=config, 
                    order_idx=args.order_idx,
                    linearised=args.linearised, # NOTE: pre-train or not also...
                    compression=args.compression,
                    reduced_cumulants=args.reduced_cumulants,
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

                derivatives_.append(derivatives)
                mus.append(mu)
                covariances.append(C)
                x_s.append(x_z)
                datavectors.append(datavector)
                Finvs.append(Finv_z)
                bulk_Finvs.append(bulk_Finv_z)
                ensembles.append(ensemble)      
                
                bar.set_postfix_str("z={}, n_posterior={}".format(redshift, n_posterior))

        # Multi-redshift ensemble of individual ensembles at each redshift
        multi_ensemble = MultiEnsemble(
            ensembles, prior=prior, sbi_type=config.sbi_type
        ) 

        Finv_all_z = jnp.linalg.inv(F) # Combined Fisher information over all redshifts
        bulk_Finv_all_z = jnp.linalg.inv(F_bulk) # Combined Fisher information over all redshifts

        if args.verbose:
            # Plot Fisher forecast
            c = ChainConsumer()
            for z, Finv_z in zip(config.redshifts, Finvs):
                c.add_chain(
                    Chain.from_covariance(
                        alpha,
                        Finv_z,
                        columns=parameter_strings,
                        name=r"$F_{\Sigma^{-1}}$ z=" + str(z),
                        linestyle=":",
                        shade_alpha=0.
                    )
                )
            for z, bulk_Finv_z in zip(config.redshifts, bulk_Finvs):
                c.add_chain(
                    Chain.from_covariance(
                        alpha,
                        bulk_Finv_z,
                        columns=parameter_strings,
                        name=r"$F_{\Sigma^{-1}}$ z=" + str(z) + " [bulk]",
                        linestyle=":",
                        color="g",
                        shade_alpha=0.
                    )
                )
            c.add_chain(
                Chain.from_covariance(
                    alpha,
                    Finv_all_z,
                    columns=parameter_strings,
                    name=r"$F_{\Sigma^{-1}}$ (all z)",
                    color="k",
                    shade_alpha=0.
                )
            )
            c.add_chain(
                Chain.from_covariance(
                    alpha,
                    bulk_Finv_all_z,
                    columns=parameter_strings,
                    name=r"$F_{\Sigma^{-1}}$ (all z) [bulk]",
                    color="g",
                    shade_alpha=0.
                )
            )
            c.add_marker(
                location=marker(alpha, parameter_strings=parameter_strings),
                name=r"$\alpha$", 
                color="#7600bc"
            )
            fig = c.plotter.plot()
            plt.savefig(
                os.path.join(
                    figs_dir if figs_dir is not None else "fisher_forecasts/", 
                    "fisher_forecast_{}_z={}_R={}_m={}.png".format(
                        config.linearised,  
                        "".join(map(str, config.redshifts)),
                        "".join(map(str, config.order_idx)),
                        "".join(map(str, config.scales))
                    )
                ), 
            )
            plt.close()

            # Block diagonal covariance plot
            plt.figure()
            plt.imshow(block_diag(*covariances))
            plt.savefig("block_diag_covariance.png")
            plt.close()

        print("Sampling posterior {} (all redshifts, datavectors)".format(n_posterior))

        # Sample the multiple-redshift-ensemble posterior
        key_sample, key_state = jr.split(jr.fold_in(key, n_posterior))

        # Compress datavectors concatenated over redshift, using block-diagonal covariance
        x_ = maybe_vmap_multi_redshift_mle( 
            alpha, 
            datavectors, 
            Finv=Finv_all_z,
            mus=mus, 
            covariances=covariances, # Block-diagonalised in this function
            derivatives=derivatives_
        )

        if args.verbose:
            print("x_ (compressed)", x_.shape, x_)

        # Sample posterior across multiple redshifts
        log_prob_fn = multi_ensemble.get_multi_ensemble_log_prob_fn(x_s)

        # samples, samples_log_prob = nuts_sample(
        #     key_sample, 
        #     log_prob_fn=log_prob_fn, # NOTE: is it right to pass list of datavectors not the MLE above?
        #     prior=prior
        # )

        state = jr.multivariate_normal(
            key_state, alpha, Finv_all_z, (2 * config.n_walkers,) # x_
        )

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

        alpha_log_prob = log_prob_fn(jnp.asarray(alpha))
        samples_log_prob = jax.vmap(log_prob_fn)(samples)
        samples_log_prob = finite_samples_log_prob(samples_log_prob)

        # Save posterior, Fisher and summary
        posterior_save_dir = get_multi_z_posterior_dir(config, args)
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
            summary=x_ # Is this correct one to save?
        )

        print("MULTI-Z POSTERIOR FILENAME:\n", posterior_filename)

        """
            Full posterior
        """
        # assert not np.allclose(Finv_all_z, bulk_Finv_all_z)

        if not config.freeze_parameters:
            c = ChainConsumer() 
            c.add_chain(
                Chain.from_covariance(
                    alpha,
                    Finv_all_z, # NOTE: Get multi redshift Fisher matrix, use a multi-inference config
                    columns=parameter_strings,
                    name=r"$F_{\Sigma^{-1}}$ (all z) $k_n$[tails]",
                    color="k",
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
                    color="r"
                )
            )
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
            fig.suptitle(
                r"{} SBI & $F_{{\Sigma}}^{{-1}}$".format(
                    "$k_n/k_2^{n-1}$" if config.reduced_cumulants else "$k_n$"
                ) + "\n" +
                "{} z={},\n $n_s$={}, (pre-train $n_s$={}),\n R={} Mpc,\n $k_n$={}".format(
                        ("linearised" if config.linearised else "non-linear") + "\n",
                        "[{}]".format(", ".join(map(str, config.redshifts))),
                        config.n_linear_sims if config.linearised else 2000, 
                        config.n_linear_sims if config.pre_train else None,
                        "[{}]".format(", ".join(map(str, config.scales))),
                        "[{}]".format(", ".join(map(str, [["var.", "skew.", "kurt."][_] for _ in config.order_idx])))
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
        """

        # Marginalise over all but Om, s8
        ix = np.array([0, -1]) # Indices for Om, s8
        parameter_names_ = [parameter_strings[_] for _ in ix]

        c = ChainConsumer()
        c.add_chain(
            Chain.from_covariance(
                alpha[ix],
                Finv_all_z[ix, :][:, ix],
                columns=parameter_names_,
                name=r"$F_{\Sigma^{-1}}$ $k_n$[tails]",
                color="k",
                linestyle=":",
                shade_alpha=0.
            )
        )
        c.add_chain(
            Chain.from_covariance(
                alpha[ix],
                bulk_Finv_all_z[ix, :][:, ix],
                columns=parameter_names_,
                name=r"$F_{\Sigma^{-1}}$ (all z) $k_n$[bulk]",
                color="b",
                shade_alpha=0.
            )
        )
        c.add_chain(
            Chain.from_covariance(
                alpha[ix],
                Finv_bulk_pdfs_all_z[ix, :][:, ix],
                columns=parameter_names_,
                name=r"$F_{\Sigma^{-1}}$ (all z) PDF[bulk]",
                color="g",
                linestyle=":",
                shade_alpha=0.
            )
        )

        posterior_df = make_df(
            samples[:, ix], samples_log_prob, parameter_strings=parameter_names_
        )
        c.add_chain(
            Chain(
                samples=posterior_df, 
                name="SBI[{}]".format(args.bulk_or_tails), 
                color="r"
            )
        )
        if x_.ndim > 1:
            for i, _x_ in enumerate(x_):
                c.add_marker(
                    location=marker(_x_[ix], parameter_names_), 
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
        fig.suptitle(
            r"{} SBI & $F_{{\Sigma}}^{{-1}}$".format("$k_n/k_2^{n-1}$" if config.reduced_cumulants else "$k_n$") + "\n" +
            "{} z={},\n $n_s$={}, (pre-train $n_s$={}),\n R={} Mpc,\n $k_n$={}".format(
                    ("linearised" if config.linearised else "non-linear") + "\n",
                    "[{}]".format(", ".join(map(str, config.redshifts))),
                    config.n_linear_sims if config.linearised else 2000, 
                    config.n_linear_sims if config.pre_train else None,
                    "[{}]".format(", ".join(map(str, config.scales))),
                    "[{}]".format(", ".join(map(str, [["var.", "skew.", "kurt."][_] for _ in config.order_idx])))
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

# Precision of weights bug
# ensemble = eqx.tree_at(
#     lambda e: e.weights, 
#     ensemble, 
#     ensemble.weights.squeeze().astype(jnp.int32)
# )