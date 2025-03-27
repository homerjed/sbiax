import warnings
import os
import time
import yaml
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

from configs import (
    arch_search_config, get_arch_search_args, cumulants_config, get_results_dir, get_posteriors_dir, 
    get_cumulants_sbi_args, get_ndes_from_config, get_base_results_dir
)
from cumulants import (
    CumulantsDataset, Dataset, get_data, get_prior, 
    get_compression_fn, get_datavector, get_linearised_data
)

from sbiax.utils import make_df, marker
from sbiax.ndes import Scaler, CNF, MAF 
from cumulants_ensemble import Ensemble
from sbiax.train import train_ensemble
from sbiax.inference import nuts_sample

from affine import affine_sample
from utils import plot_moments, plot_latin_moments, plot_summaries, plot_fisher_summaries

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
    arch_search_dir: str, 
    show_tqdm: bool = False
) -> Array:
    t0 = time.time()

    args = get_cumulants_sbi_args() # Don't conflate with arch_search_args

    print("TIME:", datetime.now().strftime("%H:%M %d-%m-%y"))
    print("SEED:", args.seed)
    print("MOMENTS:", args.order_idx)
    print("LINEARISED:", args.linearised)
    print("TRIAL NUMBER", trial.number)

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

    # Set config attributes based on these hyperparameters
    config = get_trial_hyperparameters(trial, config)

    key = jr.key(int(trial.number)) # config.seed

    ( 
        model_key, train_key, key_prior, 
        key_datavector, key_state, key_sample
    ) = jr.split(key, 6)

    results_dir = os.path.join(
        get_results_dir(config, args, arch_search=True), "{}/".format(trial.number)
    )

    posteriors_dir = os.path.join(
        get_posteriors_dir(config, arch_search=True), "{}/".format(trial.number)
    )

    for _dir in [posteriors_dir, results_dir]:
        if not os.path.exists(_dir):
            os.makedirs(_dir, exist_ok=True)

    # Save command line arguments and config together
    with open(os.path.join(results_dir, "config.yml"), "w") as f:
        yaml.dump({"args": ""}, f, default_flow_style=False)
        yaml.dump(vars(args), f, default_flow_style=False)
        yaml.dump({"config": ""}, f, default_flow_style=False)
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    cumulants_dataset: CumulantsDataset = CumulantsDataset(config, results_dir=results_dir)

    dataset: Dataset = cumulants_dataset.data
    parameter_prior: Distribution = cumulants_dataset.prior

    print("DATA:", ["{:.3E} {:.3E}".format(_.min(), _.max()) for _ in (dataset.fiducial_data, dataset.data)])
    print("DATA:", [_.shape for _ in (dataset.fiducial_data, dataset.data)])

    compression_fn = cumulants_dataset.compression_fn

    X = jax.vmap(compression_fn)(dataset.data, dataset.parameters)

    # Fiducial
    X0 = jax.vmap(compression_fn, in_axes=(0, None))(dataset.fiducial_data, dataset.alpha)

    if 1:
        c = ChainConsumer()
        c.add_chain(
            Chain(
                samples=make_df(X0, parameter_strings=dataset.parameter_strings), 
                name="X (fiducial)", 
                color="b", 
                plot_contour=False, 
                plot_cloud=True
            )
        )
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
        # c.add_chain(
        #     Chain(
        #         samples=make_df(dataset.parameters, parameter_strings=dataset.parameter_strings), 
        #         plot_contour=False, 
        #         plot_cloud=True, 
        #         name="P", 
        #         color="r"
        #     )
        # )
        c.add_marker(
            location=marker(dataset.alpha, parameter_strings=dataset.parameter_strings),
            name=r"$\alpha$", 
            color="#7600bc"
        )
        fig = c.plotter.plot()
        fig.suptitle(
            r"$k_n/k_2^{n-1}$ SBI & $F_{{\Sigma}}^{{-1}}$" + "\n" +
            "z={},\n $n_s$={}, (pre-train $n_s$={}),\n R={} Mpc,\n $k_n$={}".format(
                    config.redshift, 
                    len(X), 
                    config.n_linear_sims if config.pre_train else None,
                    "[{}]".format(", ".join(map(str, config.scales))),
                    "[{}]".format(",".join(map(str, [["var.", "skew.", "kurt."][_] for _ in config.order_idx])))
                ),
            multialignment='center'
        )
        plt.savefig(os.path.join(results_dir, "X0.png"))
        plt.close()

    # Plot summaries
    plot_summaries(X, dataset.parameters, dataset, results_dir=results_dir)

    plot_moments(dataset.fiducial_data, config, results_dir=results_dir)

    plot_latin_moments(dataset.data, config, results_dir=results_dir)

    def replace_scalers(ensemble, *, X, P):
        is_scaler = lambda x: isinstance(x, Scaler)
        get_scalers = lambda m: [
            x
            for x in jax.tree.leaves(m, is_leaf=is_scaler)
            if is_scaler(x)
        ]
        ensemble = eqx.tree_at(
            get_scalers, 
            ensemble, 
            [Scaler(X, P)] * sum(int(nde.use_scaling) for nde in config.ndes) 
        )
        return ensemble

    scaler = Scaler(X, dataset.parameters, use_scaling=config.use_scalers)

    ndes = get_ndes_from_config(
        config, 
        event_dim=dataset.alpha.size, 
        scalers=scaler, # Same scaler for all NDEs 
        use_scalers=config.use_scalers,
        key=model_key
    )

    ensemble = Ensemble(ndes, sbi_type=config.sbi_type)

    data_preprocess_fn = lambda x: 2.0 * (x - X.min()) / (X.max() - X.min()) - 1.0 # jnp.log(jnp.clip(x, min=1e-10)) 

    # Only pre-train if required and not inferring from linear simulations
    if (
        (not config.linearised) 
        and config.pre_train 
        and (config.n_linear_sims is not None)
    ):
        print("Linearised pre-training...")

        pre_train_key, summaries_key = jr.split(key)

        # Pre-train data = linearised simulations
        D_l, Y_l = cumulants_dataset.get_linearised_data()

        X_l = jax.vmap(compression_fn)(D_l, Y_l)

        print("Pre-training with", D_l.shape, X_l.shape, Y_l.shape)

        plot_fisher_summaries(X_l, Y_l, dataset, results_dir)

        opt = getattr(optax, config.pretrain.opt)(config.pretrain.lr)

        if config.use_scalers:
            ensemble = replace_scalers(
                ensemble, X=data_preprocess_fn(X_l), P=dataset.parameters
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
            trial=trial,
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
        plt.show()

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

    opt = getattr(optax, config.train.opt)(config.train.lr)

    print("Data / Parameters", [_.shape for _ in (X, dataset.parameters)])
    print("Data / Parameters", [(jnp.min(X).item(), jnp.max(X).item()), (jnp.min(dataset.parameters).item(), jnp.max(dataset.parameters).item())])

    if config.use_scalers:
        ensemble = replace_scalers(
            ensemble, X=data_preprocess_fn(X), P=dataset.parameters
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
        trial=trial,
        show_tqdm=args.use_tqdm,
        # results_dir=results_dir
    )

    ensemble = eqx.nn.inference_mode(ensemble)

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

    samples_log_prob = jax.vmap(log_prob_fn)(samples)
    alpha_log_prob = log_prob_fn(jnp.asarray(dataset.alpha))

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
    # plt.show()
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
    # plt.show()
    plt.savefig(os.path.join(results_dir, "posterior_affine_Oms8.pdf"))
    plt.savefig(os.path.join(posteriors_dir, "posterior_affine_Oms8.pdf"))
    plt.close()

    # Free memory
    del ensemble, X, dataset
    gc.collect()
    jax.clear_backends()
    jax.clear_caches()

    return stats[0]["all_valid_loss"] # Assuming one NDE


def callback(
    study: optuna.Study, 
    trial: optuna.Trial, 
    df_name: str,
    figs_dir: str, 
    arch_search_dir: str
) -> None:
    try:
        print("@" * 80 + datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
        print("Best values so far:\n\t{}\n\t{}".format(study.best_trial, study.best_trial.params))
        print("@" * 80 + "n_trials=" + str(len(study.trials)))

        layout_kwargs = dict(template="simple_white", title=dict(text=None))
        fig = optuna.visualization.plot_param_importances(study)
        fig.update_layout(**layout_kwargs)
        fig.show()
        fig.write_image(os.path.join(figs_dir, "importances.pdf"))

        fig = optuna.visualization.plot_optimization_history(study)
        fig.update_layout(**layout_kwargs)
        fig.show()
        fig.write_image(os.path.join(figs_dir, "history.pdf"))

        fig = optuna.visualization.plot_contour(study)
        fig.update_layout(**layout_kwargs)
        fig.show()
        fig.write_image(os.path.join(figs_dir, "contour.pdf"))

        fig = optuna.visualization.plot_intermediate_values(study)
        fig.update_layout(**layout_kwargs)
        fig.show()
        fig.write_image(os.path.join(figs_dir, "intermediates.pdf"))

        fig = optuna.visualization.plot_timeline(study)
        fig.update_layout(**layout_kwargs)
        fig.show()
        fig.write_image(os.path.join(figs_dir, "timeline.pdf"))

        df = study.trials_dataframe()
        df.to_pickle(os.path.join(arch_search_dir, df_name)) 
    except ValueError:
        pass # Not enough trials to plot yet


def get_trial_hyperparameters(trial: optuna.Trial, config: ConfigDict) -> ConfigDict:

    model_type = config.ndes[0].model_type

    # Arrange hyperparameters to optimise for and return to the experiment
    if model_type == "cnf":
        model_hyperparameters = {
            "width" : trial.suggest_int(name="width", low=2, high=6, step=1), # NN width
            "depth" : trial.suggest_int(name="depth", low=0, high=4, step=1), # NN depth
            "dt" : trial.suggest_float(name="dt", low=0.01, high=0.15, step=0.01), # ODE solver timestep
            "solver" : trial.suggest_categorical(name="solver", choices=["Euler", "Heun", "Tsit5"]), # ODE solver
        }
        config.ndes[0].width_size = 2 ** model_hyperparameters["width"]
        config.ndes[0].depth = model_hyperparameters["depth"]
        config.ndes[0].dt = model_hyperparameters["dt"]
        config.ndes[0].solver = model_hyperparameters["solver"]

    if model_type == "maf":
        model_hyperparameters = {
            "width" : trial.suggest_int(name="width", low=3, high=7, step=1), # Hidden units in NNs
            "depth" : trial.suggest_int(name="depth", low=1, high=5, step=1), # Flow depth
            "layers" : trial.suggest_int(name="layers", low=1, high=3, step=1), # NN layers
        }
        config.ndes[0].width = 2 ** model_hyperparameters["width"]
        config.ndes[0].n_layers = model_hyperparameters["depth"]
        config.ndes[0].nn_depth = model_hyperparameters["layers"]

    training_hyperparameters = {
        # Training
        "n_batch" : trial.suggest_int(name="n_batch", low=40, high=100, step=10), 
        "lr" : trial.suggest_float(name="lr", low=1e-5, high=1e-3, log=True), 
        "patience" : trial.suggest_int(name="p", low=10, high=100, step=10),
    }

    config.train.n_batch = training_hyperparameters["n_batch"]
    config.train.lr = training_hyperparameters["lr"]
    config.train.patience = training_hyperparameters["patience"]

    hyperparameters = {**model_hyperparameters, **training_hyperparameters} 
    print("Hyperparameters:\n", hyperparameters)

    return config


if __name__ == "__main__":

    # args = arch_search_config()
    args = get_arch_search_args()

    show_results_dataframe = False # Simply show best trial from existing dataframe

    arch_search_dir = os.path.join(get_base_results_dir(), "arch_search/")

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
        assert args.multiprocess and args.n_processes

        journal_name = "1pt_arch_search_{}.log".format(date_stamp())
        study_name = "1pt_nle_{}".format(date_stamp())
        df_name = "arch_search_df_{}.pkl".format(date_stamp())

        arch_search_figs_dir = os.path.join(arch_search_dir, "figs/")

        for _dir in [arch_search_dir, arch_search_figs_dir]:
            if not os.path.exists(_dir):
                os.makedirs(_dir, exist_ok=True)

        # Journal storage allows independent process optimisation
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(
                os.path.join(arch_search_dir, journal_name)
            )
        )

        # Minimise negative log-likelihood
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize", 
            storage=storage,
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=args.n_startup_trials, multivariate=False
            ),
            load_if_exists=True
        ) 

        # study.enqueue_trial(good_hyperparams) 

        trial_fn = lambda trial: objective(
            trial, arch_search_dir=arch_search_dir, show_tqdm=False
        )

        callback_fn = partial(
            callback, 
            figs_dir=arch_search_figs_dir, 
            arch_search_dir=arch_search_dir,
            df_name=df_name
        )

        # Run multiprocessed architecture search or single process
        if args.multiprocess:

            def mp_optimize(process, study): 
                # Function links processes to same study (lambdas not allowed)
                study.optimize(
                    trial_fn, n_trials=args.n_trials, callbacks=[callback_fn]
                )

            with mp.Pool(processes=args.n_processes) as pool:
                pool.map(
                    partial(mp_optimize, study=study), 
                    [*range(args.n_parallel)] # Number of parallel jobs per process in args.n_processes?
                )
        else:
            study.optimize(trial_fn, n_trials=args.n_trials, callbacks=[callback])

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