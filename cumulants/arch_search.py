import warnings
import os
import time
import pickle
import gc
from datetime import datetime
import multiprocessing as mp
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Array
import optax
import numpy as np 
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
from chainconsumer import Chain, ChainConsumer, Truth
from tensorflow_probability.substrates.jax.distributions import Distribution
import optuna

from configs import get_results_dir
from configs.moments import moments_config 
from data.moments import Dataset, get_data, get_prior, get_linear_compressor, get_datavector
from sbiax.utils import make_df, marker, get_fisher_summaries
from sbiax.ndes import Scaler, Ensemble, get_ndes_from_config
from sbiax.train import train_ensemble
from sbiax.inference import nuts_sample, affine_sample
from sbiax.meta import get_trial_hyperparameters, callback, get_args

"""
    - Integrate optuna into training function       x
    - get meta functions above                      x
"""


def date_stamp():
    now = datetime.now()
    return "{}_{}_{}_{}".format(now.hour, now.day, now.month, now.year)


def assign_trial_parameters_to_config(
    hyperparameters: dict, config: ConfigDict
) -> ConfigDict:
    """ Experiments run on configs, optuna uses trials, so assign trials to configs """

    # Assign non-NDE hyperparameters e.g. lr, opt, patience...
    for key, value in hyperparameters.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Assign NDE hyperparameters
    for nde_type in ["maf", "cnf"]:
        if hasattr(config, nde_type):
            config_obj = getattr(config, nde_type) # Get NDE of config
            for key, value in hyperparameters.items():
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value) 
            setattr(config, nde_type, config_obj) # Set NDE of config back 
    return config


def objective(
    trial: optuna.trial.Trial, 
    model_type: str, 
    arch_search_dir: str, 
    linearised: bool = False,
    show_tqdm: bool = False
) -> Array:
    """
        Run a SBI experiment with n_sims simulations and parameters.
        - Metric is either multi-posterior KL loss or validation log prob on an unseen test set
    """
    # This draws the trial hyperparameters
    hyperparameters = get_trial_hyperparameters(trial, model_type)

    print("Hyperparameters:\n", hyperparameters)

    t0 = time.time()

    config = moments_config(
        seed=0, 
        redshift=0., 
        linearised=linearised, 
        pre_train=False,
        n_linear_sims=10_000
    )

    assert config.n_ndes == 1, "One architecture for one NDE during hyperparameter estimation."

    config = assign_trial_parameters_to_config(hyperparameters, config)
    
    print("Config:\n", config)

    key = jr.key(config.seed)

    (
        key, model_key, train_key, key_prior, 
        key_datavector, key_state, key_sample
    ) = jr.split(key, 7)

    # n_s fixed with Quijote (maybe linearised?) # results_dir: str = get_results_dir(config)
    results_dir  = os.path.join(
        arch_search_dir, "{}/{}/{}/{}".format(
            model_type, config.seed, trial.number, "linear/" if linearised else ""
        )
    ) 
    posterior_figs_dir = os.path.join(
        arch_search_dir, "{}/posterior_figs/".format(model_type)
    )
    for _dir in [results_dir, posterior_figs_dir]:
        if not os.path.exists(_dir):
            os.makedirs(_dir, exist_ok=True)
   
    # Dataset of simulations, parameters, covariance, ...
    dataset: Dataset = get_data(config)

    parameter_prior: Distribution = get_prior(config)

    # Get linear compressor
    s = get_linear_compressor(config)

    # Compress simulations
    X = jax.vmap(s)(dataset.data, dataset.parameters)

    """
        Build NDEs
    """

    scaler = Scaler(
        X, dataset.parameters, use_scaling=config.maf.use_scaling
    )

    ndes = get_ndes_from_config(
        config, event_dim=dataset.alpha.size, scalers=scaler, key=model_key # Fix scaler passing to ndes...
    )
    ensemble = Ensemble(ndes, sbi_type=config.sbi_type)

    """
        Train NDE on data
    """
    opt = getattr(optax, config.opt)(config.lr)

    ensemble, stats = train_ensemble(
        train_key, 
        ensemble,
        train_mode=config.sbi_type,
        train_data=[X, dataset.parameters], 
        opt=opt,
        n_batch=config.n_batch,
        patience=config.patience,
        n_epochs=config.n_epochs,
        tqdm_description="Training (data)",
        show_tqdm=show_tqdm,
        trial=trial,
        results_dir=results_dir
    )

    eqx.tree_serialise_leaves(os.path.join(results_dir, "ensemble.eqx"), ensemble)

    """ 
        Sample and plot posterior for NDE with noisy datavectors
    """
    ensemble = eqx.nn.inference_mode(ensemble)

    # Generates linearised (or not) datavector at fiducial parameters
    datavector = get_datavector(key_datavector, config)

    x_ = s(datavector, dataset.alpha)

    log_prob_fn = ensemble.ensemble_log_prob_fn(x_, parameter_prior)

    # samples, samples_log_prob = nuts_sample(
    #     key_sample, log_prob_fn, prior=parameter_prior
    # )

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

    posterior_df = make_df(samples, samples_log_prob, dataset.parameter_strings)

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
            location=marker(x_, dataset.parameter_strings),
            name=r"$\hat{x}$", 
            color="b"
        )
        c.add_marker(
            location=marker(dataset.alpha, dataset.parameter_strings),
            name=r"$\alpha$", 
            color="#7600bc"
        )
        fig = c.plotter.plot()
        plt.savefig(os.path.join(results_dir, "posterior.pdf"))
        plt.savefig(os.path.join(posterior_figs_dir, "posterior_{}.pdf".format(trial.number)))
        plt.close()
    except Exception as e:
        print("EXCEPTION\n\t", e)

    np.savez(
        os.path.join(results_dir, "posterior.npz"), 
        alpha=dataset.alpha,
        samples=samples,
        samples_log_prob=samples_log_prob,
        datavector=datavector,
        summary=x_
    )

    print(f"Time={(time.time() - t0) / 60.:.1} mins.")

    # Free memory
    del ensemble, X, Q, dataset
    gc.collect()
    jax.clear_backends()
    jax.clear_caches()

    return stats[0]["all_valid_loss"] # Assuming one NDE


if __name__ == "__main__":

    args = get_args()

    show_results_dataframe = False # Simply show best trial from existing dataframe

    arch_search_dir = "/project/ls-gruen/users/jed.homer/sbipdf/results/arch_search/nle/" 

    arch_search_dir = os.path.join(
        arch_search_dir, 
        (args.exp_name + "/") if args.exp_name is not None else ""
    )

    arch_search_dir = "/project/ls-gruen/users/jed.homer/sbipdf/results/arch_search/nle/nle_maf/" 
    print((args.exp_name + "/") if args.exp_name is not None else "")
    print(arch_search_dir)

    n_trials = 500        # Number of trials in hyperparameter optimisation (per process)
    n_startup_trials = 50 # Number of warmup trials in hyperparameter optimisation

    if show_results_dataframe:
        # df.to_pickle(os.path.join(arch_search_dir, "arch_search_df.pkl")) # Where to save it, usually as a .pkl

        with open(os.path.join(arch_search_dir, "arch_search_df.pkl"), "rb") as f:
            loaded_df = pickle.load(f)

        # Find the trial with the best value (minimum objective value)
        best_trial_row = loaded_df.loc[loaded_df['value'].idxmin()]
        print("Best Trial Parameters:")
        print(best_trial_row.filter(like="params_")) 
        print(best_trial_row)
    else:
        assert args.multiprocess and args.n_processes, (
        )
        assert args.model_type in ["cnf", "maf"], "Model doesn't exist or string not lower case."

        journal_name = "1pt_arch_search_{}.log".format(date_stamp())
        study_name = "1pt_nle_{}".format(date_stamp())
        df_name = "arch_search_df_{}.pkl".format(date_stamp())

        arch_search_figs_dir = os.path.join(
            arch_search_dir, f"figs/{args.model_type}/"
        )

        for _dir in [arch_search_dir, arch_search_figs_dir]:
            if not os.path.exists(_dir):
                os.makedirs(_dir, exist_ok=True)

        # Journal storage allows independent process optimisation
        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(
                os.path.join(arch_search_dir, journal_name)
            )
        )

        # Minimise negative log-likelihood
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize", 
            storage=storage,
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=n_startup_trials, multivariate=True
            ),
            load_if_exists=True
        ) 

        # study.enqueue_trial(good_hyperparams) 

        trial_fn = lambda trial: objective(
            trial, 
            model_type=args.model_type, 
            arch_search_dir=arch_search_dir,
            linearised=args.linearised
        )

        _callback = partial(
            callback, 
            figs_dir=arch_search_figs_dir, 
            arch_search_dir=arch_search_dir,
            df_name=df_name
        )

        # Run multiprocessed architecture search or single process
        if args.multiprocess:

            def mp_optimize(process, study): 
                # Function links processes to same study (lambdas not allowed)
                study.optimize(trial_fn, n_trials=n_trials, callbacks=[_callback])

            with mp.Pool(processes=args.n_processes) as pool:
                pool.map(
                    partial(mp_optimize, study=study), 
                    [*range(args.n_parallel)] # Number of parallel jobs per process in args.n_processes?
                )
        else:
            study.optimize(trial_fn, n_trials=n_trials, callbacks=[callback])

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print(">Value: {}".format(trial.value))
        print(">Number: {}".format(trial.number))
        print(">Params: ")
        for key, value in trial.params.items():
            print("\t{} : {}".format(key, value))

        df = study.trials_dataframe()
        df.to_pickle(os.path.join(arch_search_dir, df_name)) # Where to save it, usually as a .pkl