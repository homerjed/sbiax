import warnings
import os
import time
import yaml
import json
import pickle
import gc
import argparse
from datetime import datetime
from typing import Callable, Optional
import multiprocessing as mp
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import PRNGKeyArray, Array
import optax

import numpy as np 
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
from chainconsumer import Chain, ChainConsumer, Truth
from tensorflow_probability.substrates.jax.distributions import Distribution
import optuna

from sbiax.utils import make_df, marker
from sbiax.ndes import Scaler, CNF, MAF 
from cumulants_ensemble import Ensemble
from sbiax.train import train_ensemble
from sbiax.inference import nuts_sample

from configs import (
    arch_search_config, 
    cumulants_config, 
    arch_search_cumulants_config, 
    get_results_dir, 
    get_posteriors_dir, 
    get_ndes_from_config
)
from configs.args import get_arch_search_args, get_cumulants_sbi_args
from data.constants import get_base_results_dir
from data.cumulants import (
    Dataset, 
    CumulantsDataset, 
    get_data, 
    get_prior, 
    get_compression_fn, 
    get_datavector, 
    get_linearised_data
)
from affine import affine_sample
from utils.utils import (
    get_datasets,
    plot_cumulants,
    plot_moments, 
    plot_latin_moments, 
    plot_summaries, 
    plot_fisher_summaries, 
    replace_scalers,
    finite_samples_log_prob
)

"""
    - Integrate optuna into training function       x
    - get meta functions above                      x
"""

# jax.config.update("jax_debug_nans", True)

TEST = True if os.environ.get('TEST', '').lower() in ('1', 'true') else False

# Implies no datestamping for dirs (so everything writes so same storage)
RUNNING_MULTIPLE_SLURM_JOBS = True if os.environ.get('MULTI_SLURM', '').lower() in ('1', 'true') else False 


def date_stamp():
    if RUNNING_MULTIPLE_SLURM_JOBS:
        stamp = ""
    else:
        now = datetime.now()
        stamp = "{}_{}_{}_{}".format(now.hour, now.day, now.month, now.year)
    return stamp


def assign_trial_parameters_to_config(
    hyperparameters: dict, config: ConfigDict
) -> ConfigDict:
    """ 
        Experiments run on configs, optuna uses trials, so assign trials to configs 
    """

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
    args: argparse.Namespace, # Use same args for all trials
    config: ConfigDict, # Use same config for all trials
    random_seeds: bool,
    arch_search_dir: str, 
    n_repeats: Optional[int] = None,
    show_tqdm: bool = False
) -> Array:
    
    jax.clear_caches()

    t0 = time.time()

    print("TIME:", datetime.now().strftime("%H:%M %d-%m-%y"))
    print("SEED:", args.seed)
    print("MOMENTS:", args.order_idx)
    print("LINEARISED:", args.linearised)
    print("TRIAL NUMBER", trial.number)

    """
        Config
    """
    _, cumulants_dataset, datasets = get_datasets(args) # Ignore config

    # Seed overwritten in config
    seed = args.seed + trial.number if random_seeds else args.seed
    config.seed = seed

    # Set config attributes based on these hyperparameters
    config = get_trial_hyperparameters(trial, config)

    key = jr.key(int(trial.number)) 

    ( 
        model_key, train_key, key_prior, 
        key_datavector, key_state, key_sample
    ) = jr.split(key, 6)

    results_dir = get_results_dir(config, args, arch_search=True)
    posteriors_dir = get_posteriors_dir(config, args, arch_search=True)
    for _dir in [posteriors_dir, results_dir]:
        if not os.path.exists(_dir):
            os.makedirs(_dir, exist_ok=True)

    # Save command line arguments and config together
    with open(os.path.join(results_dir, "config.yml"), "w") as f:
        yaml.dump({"args": ""}, f, default_flow_style=False)
        yaml.dump(vars(args), f, default_flow_style=False)
        yaml.dump({"config": ""}, f, default_flow_style=False)
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    dataset: Dataset = cumulants_dataset.data

    parameter_prior: Distribution = cumulants_dataset.prior

    if n_repeats is None:
        n_repeats = 1

    # def train_validation_split(X, Y):
    #     """ 
    #         Different train/validation split for repeated cross-validation run 
    #         - shuffle dataset which implies different split
    #         - just give a different training seed? simplest...
    #     """
    #     X

    # Container for cross-validation scores 
    scores, losses_lengths = [], []
    for i_repeat in range(n_repeats):
        jax.clear_caches()

        # Keys passed to pre-train and train functions (implies different dataset splits / training)
        keys_train = jr.split(jr.fold_in(key, i_repeat))

        """
            Compression
        """

        compression_fn = cumulants_dataset.compression_fn

        X = jax.vmap(compression_fn)(dataset.data, dataset.parameters)

        # Plot summaries
        # plot_summaries(X, dataset.parameters, dataset, results_dir=results_dir)

        # plot_moments(dataset.fiducial_data, config, results_dir=results_dir)

        # plot_latin_moments(dataset.data, config, results_dir=results_dir)

        """
            Build NDEs
        """
        # cov_sqrt_inv = fractional_matrix_power(cov_X, -0.5)
        # X_whitened = (X - mean_X) @ cov_sqrt_inv

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

            pre_train_key, summaries_key = jr.split(keys_train[0])

            # Pre-train data = linearised simulations
            D_l, Y_l = cumulants_dataset.get_linearised_data()

            X_l = jax.vmap(compression_fn)(D_l, Y_l)

            print("Pre-training with", D_l.shape, X_l.shape, Y_l.shape)

            plot_fisher_summaries(X_l, Y_l, dataset, results_dir)

            opt = getattr(optax, config.pretrain.opt)(config.pretrain.lr)

            if config.use_scalers:
                ensemble = replace_scalers(
                    ensemble, X=data_preprocess_fn(X_l), P=Y_l, config=config
                )

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
                trial=None, # Pretraining! #trial if n_repeats > 1 else None, # Don't allow trial to report within inner repetition of objective 
                results_dir=results_dir
            )

            # Test pre-training on a linearised datavector...
            mu = jnp.mean(dataset.fiducial_data, axis=0)
            datavector = jr.multivariate_normal(key, mu, dataset.C)

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
            samples_log_prob = jax.vmap(log_prob_fn)(samples) # No pre-processing on parameters
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
            plt.savefig(os.path.join(results_dir, "posterior_affine_pretrain.pdf"))
            plt.savefig(os.path.join(posteriors_dir, "posterior_affine_pretrain.pdf"))
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
                    samples=make_df(X_l, parameter_strings=dataset.parameter_strings), 
                    name="Summaries (linearised)", 
                    color="red", 
                    plot_cloud=True, 
                    plot_contour=False
                )
            )
            c.add_chain(
                Chain(
                    samples=make_df(X, parameter_strings=dataset.parameter_strings),
                    name="Summaries", 
                    color="green", 
                    plot_cloud=True, 
                    plot_contour=False
                )
            )
            c.add_truth(
                Truth(location=dict(zip(dataset.parameter_strings, dataset.alpha)), name=r"$\pi^0$")
            )
            fig = c.plotter.plot()
            plt.close()

            fig, axs = plt.subplots(1, dataset.alpha.size, figsize=(2. + 2. * dataset.alpha.size, 2.5))
            for p, ax in enumerate(axs):
                ax.scatter(Y_l[:, p], X_l[:, p], s=0.1)
                ax.scatter(dataset.parameters[:, p], X[:, p], s=0.1)
                ax.axline((0, 0), slope=1., color="k", linestyle="--")
                ax.set_xlim(dataset.lower[p], dataset.upper[p])
                ax.set_ylim(dataset.lower[p], dataset.upper[p])
                ax.set_xlabel(dataset.parameter_strings[p])
                ax.set_ylabel(dataset.parameter_strings[p] + "'")
            plt.savefig(os.path.join(results_dir, "Xl.png"))
            plt.close()

        """
            Train NDE on data
        """

        opt = getattr(optax, config.train.opt)(config.train.lr)

        if config.use_scalers:
            ensemble = replace_scalers(
                ensemble, X=data_preprocess_fn(X), P=dataset.parameters, config=config
            )

        ensemble, stats = train_ensemble(
            keys_train[1], 
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
            # trial=trial,
            trial=None if n_repeats > 1 else trial, # Don't allow trial to report within inner repetition of objective 
            show_tqdm=args.use_tqdm,
            # results_dir=results_dir
        )

        """ 
            Sample and plot posterior for NDE with noisy datavectors
        """

        # Generates linearised (or not) datavector at fiducial parameters
        datavector = cumulants_dataset.get_datavector(key_datavector)

        x_ = compression_fn(datavector, dataset.alpha)

        print("datavector", x_, dataset.alpha)

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

        np.savez(
            os.path.join(results_dir, "posterior.npz"), 
            alpha=dataset.alpha,
            samples=samples,
            samples_log_prob=samples_log_prob,
            datavector=datavector,
            summary=x_
        )

        posterior_df = make_df(
            samples, 
            samples_log_prob, 
            parameter_strings=dataset.parameter_strings
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
            r"{} SBI & $F_{{\Sigma}}^{{-1}}$".format("$k_n/k_2^{n-1}$" if config.reduced_cumulants else "$k_n$") + "\n" +
            "{} z={},\n $n_s$={}, (pre-train $n_s$={}),\n R={} Mpc,\n $k_n$={}".format(
                    ("linearised" if config.linearised else "non-linear") + "\n",
                    config.redshift, 
                    len(X), 
                    config.n_linear_sims if config.pre_train else None,
                    "[{}]".format(", ".join(map(str, config.scales))),
                    "[{}]".format(", ".join(map(str, [["var.", "skew.", "kurt."][_] for _ in config.order_idx])))
                ),
            multialignment='center'
        )
        plt.savefig(os.path.join(results_dir, "posterior_affine.pdf"))
        plt.savefig(os.path.join(posteriors_dir, "posterior_affine.pdf"))
        plt.close()

        # %%
        # X.min(), X.max()
        # jnp.log(jnp.clip(X, a=1e-5)).min(), jnp.log(jnp.clip(X, a=1e-5)).max()

        # %%
        Om_s8_idx = np.array([0, -1])
        posterior_df = make_df(
            samples[:, Om_s8_idx], 
            samples_log_prob, 
            parameter_strings=[dataset.parameter_strings[p] for p in Om_s8_idx]
        )

        c = ChainConsumer()
        c.add_chain(
            Chain.from_covariance(
                dataset.alpha[Om_s8_idx],
                dataset.Finv[:, Om_s8_idx][Om_s8_idx, :],
                columns=[dataset.parameter_strings[p] for p in Om_s8_idx],
                name=r"$F_{\Sigma^{-1}}$",
                color="k",
                linestyle=":",
                shade_alpha=0.
            )
        )
        c.add_chain(Chain(samples=posterior_df, name="SBI", color="r"))
        c.add_marker(
            location=marker(x_[Om_s8_idx], parameter_strings=[dataset.parameter_strings[p] for p in Om_s8_idx]),
            name=r"$\hat{x}$", 
            color="b"
        )
        c.add_marker(
            location=marker(dataset.alpha[Om_s8_idx], parameter_strings=[dataset.parameter_strings[p] for p in Om_s8_idx]),
            name=r"$\alpha$", 
            color="#7600bc"
        )
        fig = c.plotter.plot()
        fig.suptitle(
            r"{} SBI & $F_{{\Sigma}}^{{-1}}$".format("$k_n/k_2^{n-1}$" if config.reduced_cumulants else "$k_n$") + "\n" +
            "{} z={},\n $n_s$={}, (pre-train $n_s$={}),\n R={} Mpc,\n $k_n$={}".format(
                    ("linearised" if config.linearised else "non-linear") + "\n",
                    config.redshift, 
                    len(X), 
                    config.n_linear_sims if config.pre_train else None,
                    "[{}]".format(", ".join(map(str, config.scales))),
                    "[{}]".format(", ".join(map(str, [["var.", "skew.", "kurt."][_] for _ in config.order_idx])))
                ),
            multialignment='center'
        )
        plt.savefig(os.path.join(results_dir, "posterior_affine_Oms8.pdf"))
        plt.savefig(os.path.join(posteriors_dir, "posterior_affine_Oms8.pdf"))
        plt.close()

        # Free memory
        # del ensemble, X, dataset
        gc.collect()
        jax.clear_caches()

        scores.append(stats[0]["all_valid_loss"]) # Assuming one NDE
        losses_lengths.append(len(stats[0]["valid_losses"]))

    # Delete results directories of useless trials
    # if trial.state != optuna.trial.TrialState.COMPLETE:
    #     rmtree(results_dir, ignore_errors=True)

    mean_score = np.mean(scores)

    # Trial reports this metric if repeating the training, not the individual validation losses from training 
    if n_repeats > 1:
        trial.report(mean_score, int(np.mean(losses_lengths))) # Mean score and average length of trainings

    return mean_score # Assuming one NDE!


def callback(
    study: optuna.Study, 
    trial: optuna.Trial, 
    df_name: str,
    figs_dir: str, 
    arch_search_dir: str
) -> None:

    def json_best_trial():
        def json_serial(obj):
            """ JSON serializer for objects not serializable by default json """
            if isinstance(obj, (datetime)):
                return obj.isoformat()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, set):
                return list(obj)
            return str(obj)

        # Filter for outliers
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        values = np.array([t.value for t in trials]) # Compute validation losses
        low, high = np.percentile(values, [5, 95])
        filtered_trials = [t for t in trials if low <= t.value <= high]
        best_filtered = min(filtered_trials, key=lambda t: t.value)
        print("Best trial after trimming outliers:\n", best_filtered.params)

        trial_dict = best_filtered.__dict__ # study.best_trial.__dict__
        filtered = {k: v for k, v in trial_dict.items() if k != "intermediate_values"} # Remove long list of losses

        print("@" * 80 + datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
        print("Best values so far:\n\t{}\n\t{}".format(study.best_trial.params, study.best_trial.value))
        print("Best trial so far:\n\t{}".format(filtered))
        print("Optuna figures saved at:\n\t{}".format(figs_dir))
        print("@" * 80 + "n_trials=" + str(len(study.trials)))

        # Write best trial to json
        trial_json_path = os.path.join(figs_dir, "best_trial.json")
        # with open(trial_json_path, "w") as f:
        #     json.dump(filtered, f, indent=2, default=json_serial)

        # Append new entry to json file (load previous json and append to it)
        if os.path.exists(trial_json_path):
            with open(trial_json_path, "r") as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list): # Corrupted or wrong format
                        data = []
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        data.append(filtered)

        with open(trial_json_path, "w") as f:
            json.dump(data, f, indent=2, default=json_serial)

    json_best_trial()

    # def delete_bad_trials():
    #     if trial.state != optuna.trial.TrialState.COMPLETE:


    layout_kwargs = dict(template="simple_white", title=dict(text=None))

    fig = optuna.visualization.plot_param_importances(study)
    fig.update_layout(**layout_kwargs)
    fig.write_image(os.path.join(figs_dir, "importances.pdf"))

    fig = optuna.visualization.plot_optimization_history(study)
    fig.update_layout(**layout_kwargs)
    fig.write_image(os.path.join(figs_dir, "history.pdf"))

    fig = optuna.visualization.plot_contour(study)
    fig.update_layout(**layout_kwargs)
    fig.write_image(os.path.join(figs_dir, "contour.pdf"))

    fig = optuna.visualization.plot_intermediate_values(study)
    fig.update_layout(**layout_kwargs)
    fig.write_image(os.path.join(figs_dir, "intermediates.pdf"))

    fig = optuna.visualization.plot_timeline(study)
    fig.update_layout(**layout_kwargs)
    fig.write_image(os.path.join(figs_dir, "timeline.pdf"))

    df = study.trials_dataframe()
    df.to_pickle(os.path.join(arch_search_dir, df_name)) 
    # except Exception as e:
    #     print("HYPERPARAMETER PLOT ISSUE:\n\t", e) # Not enough trials to plot yet


def get_trial_hyperparameters(trial: optuna.Trial, config: ConfigDict) -> ConfigDict:
    """
        Trial hyperparameters for CNF/MAF and training
    """

    model_type = config.ndes[0].model_type # NOTE: important; get the model type being used 

    assert model_type in ["cnf", "maf"], ("Model type {} not allowable".format(model_type))

    # Arrange hyperparameters to optimise for and return to the experiment
    if model_type == "cnf":
        model_hyperparameters = {
            "width" : trial.suggest_int(name="width", low=2, high=6, step=1), # NN width (NOTE: base 2!)
            "depth" : trial.suggest_int(name="depth", low=0, high=4, step=1), # NN depth
            "dt" : trial.suggest_float(name="dt", low=0.01, high=0.15, step=0.01), # ODE solver timestep
            "solver" : trial.suggest_categorical(name="solver", choices=["Euler", "Heun", "Tsit5"]), # ODE solver
            "activation" : trial.suggest_categorical(name="activation", choices=["tanh", "gelu", "leaky_relu", "swish"])
        }
        config.ndes[0].width_size = 2 ** model_hyperparameters["width"]
        config.ndes[0].depth = model_hyperparameters["depth"]
        config.ndes[0].dt = model_hyperparameters["dt"]
        config.ndes[0].solver = model_hyperparameters["solver"]

    if model_type == "maf":
        model_hyperparameters = {
            "width" : trial.suggest_int(name="width", low=3, high=8, step=1), # Hidden units in NNs (NOTE: base 2!)
            "depth" : trial.suggest_int(name="depth", low=1, high=10, step=1), # Flow depth
            "layers" : trial.suggest_int(name="layers", low=1, high=3, step=1), # NN layers
            "activation" : trial.suggest_categorical(name="activation", choices=["tanh", "gelu", "leaky_relu", "swish"])
        }
        config.ndes[0].width_size = 2 ** model_hyperparameters["width"] 
        config.ndes[0].n_layers = model_hyperparameters["depth"]
        config.ndes[0].nn_depth = model_hyperparameters["layers"]

    # Training
    training_hyperparameters = {
        "n_batch" : trial.suggest_int(name="n_batch", low=40, high=100, step=10), 
        "lr" : trial.suggest_float(name="lr", low=1e-5, high=1e-3, log=True), 
        "patience" : trial.suggest_int(name="p", low=10, high=200, step=10),
        "opt" : trial.suggest_categorical(name="opt", choices=["adam", "adamw", "adabelief", "lion"])
    }

    # Set config parameters explicitly
    config.train.n_batch = training_hyperparameters["n_batch"]
    config.train.lr = training_hyperparameters["lr"]
    config.train.patience = training_hyperparameters["patience"]
    config.train.opt = training_hyperparameters["opt"]

    hyperparameters = {**model_hyperparameters, **training_hyperparameters} 
    print("Hyperparameters:\n", hyperparameters)

    return config


if __name__ == "__main__":

    search_args = get_arch_search_args() # Specification for architecture search
    
    args = get_cumulants_sbi_args() # Don't conflate with arch_search_args 

    config = arch_search_cumulants_config(
        seed=0, # Gets replaced in objective!
        redshift=args.redshift, 
        reduced_cumulants=args.reduced_cumulants,
        sbi_type=args.sbi_type,
        linearised=args.linearised, 
        compression=args.compression,
        order_idx=args.order_idx,
        n_linear_sims=args.n_linear_sims,
        freeze_parameters=args.freeze_parameters,
        pre_train=args.pre_train
    )

    # Identify this arch search run later on (unique among different dataset/training types)
    identifier_str = "arch_search_{}_{}_{}_{}_m{}".format(
        "l" if args.linearised else "nl", 
        "f" if args.freeze_parameters else "nf", 
        "pt" if args.pre_train else "npt", 
        args.bulk_or_tails,
        "".join(map(str, args.order_idx))
    )

    assert search_args.multiprocess and search_args.n_processes

    journal_name = "1pt_arch_search_{}_{}.log".format(identifier_str, date_stamp())
    study_name = "1pt_nle_{}_{}".format(identifier_str, date_stamp())
    df_name = "arch_search_df_{}_{}.pkl".format(identifier_str, date_stamp())

    arch_search_dir = os.path.join(get_base_results_dir(), "arch_search/")
    arch_search_figs_dir = os.path.join(arch_search_dir, "figs_{}/".format(identifier_str))

    for _dir in [arch_search_dir, arch_search_figs_dir]:
        if not os.path.exists(_dir):
            os.makedirs(_dir, exist_ok=True)

    # Journal storage allows independent process optimisation
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(
            os.path.join(arch_search_dir, journal_name)
        )
    )
    # storage = optuna.storages.RDBStorage(
    #     url="sqlite:///optuna_study.db",  # Creates optuna_study.db in current dir
    #     engine_kwargs={"connect_args": {"timeout": 10}},  # Prevents DB lock issues
    # )

    # Optuna's default pruner is the MedianPruner(), don't prune if cross validating...
    if (search_args.n_repeats == 0) or (search_args.n_repeats is None):
        pruner = optuna.pruners.MedianPruner()
    else:
        pruner = optuna.pruners.NopPruner()  

    # Minimise negative log-likelihood
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize", 
        storage=storage,
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=search_args.n_startup_trials, multivariate=True
        ),
        pruner=pruner,
        load_if_exists=True
    ) 

    # study.enqueue_trial(good_hyperparams) 

    trial_fn = lambda trial: objective(
        trial, 
        args=args,
        config=config,
        random_seeds=search_args.random_seeds,
        arch_search_dir=arch_search_dir, 
        n_repeats=search_args.n_repeats, # 'Cross validation' of trials... doesn't work with pruning
        show_tqdm=False
    )

    callback_fn = partial(
        callback, 
        figs_dir=arch_search_figs_dir, 
        arch_search_dir=arch_search_dir,
        df_name=df_name
    )

    # Only run one trial per worker...
    study.optimize(trial_fn, n_trials=search_args.n_trials, callbacks=[callback_fn])

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