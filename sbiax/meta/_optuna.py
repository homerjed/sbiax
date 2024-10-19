import warnings
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import equinox as eqx
import optuna
from optuna.pruners import BasePruner
import plotly.graph_objects as go


class DuplicateIterationPruner(BasePruner):
    """
    DuplicatePruner

    https://github.com/optuna/optuna/issues/2021#issuecomment-1702539660

    Pruner to detect duplicate trials based on the parameters.

    This pruner is used to identify and prune trials that have the same set of parameters
    as a previously completed trial.
    """

    def prune(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> bool:
        completed_trials = study.get_trials(
            states=[optuna.trial.TrialState.COMPLETE]
        )

        for completed_trial in completed_trials:
            if completed_trial.params == trial.params:
                return True

        print("Pruned duplicate...")

        return False


def get_args():
    parser = argparse.ArgumentParser(
        description="Run architecture search for NPE or NLE."
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="Name of arch search experiment."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="CNF",
        help="Density estimator model type."
    )
    parser.add_argument(
        "--multiprocess", 
        action=argparse.BooleanOptionalAction, 
        help="Run in parallel or not."
    )
    parser.add_argument(
        "--n_processes", 
        type=int,
        default=0,
        help="Run in parallel or not."
    )
    args = parser.parse_args()
    return args


def get_trial_hyperparameters(trial, model_type):
    # Arrange hyperparameters to optimise for and return to the experiment
    if model_type == "CNF":
        model_hyperparameters = {
            "width" : trial.suggest_int(name="width", low=3, high=7, step=1), # NN width
            "depth" : trial.suggest_int(name="depth", low=0, high=2, step=1), # NN depth
            "dt" : trial.suggest_float(name="dt", low=0.01, high=0.15, step=0.01), # ODE solver timestep
            "solver" : trial.suggest_categorical(name="solver", choices=["Euler", "Heun", "Tsit5"]),
            "dropout" : 0.
        }
    if model_type == "MAF":
        model_hyperparameters = {
            "width" : trial.suggest_int(name="width", low=3, high=7, step=1), # Hidden units in NNs
            "depth" : trial.suggest_int(name="depth", low=1, high=5, step=1), # Flow depth
            "layers" : trial.suggest_int(name="layers", low=1, high=3, step=1), # NN layers
            "dropout" : 0.
        }
    training_hyperparameters = {
        # Training
        "bs" : trial.suggest_int(name="bs", low=40, high=100, step=10), 
        "lr": trial.suggest_float(name="lr", low=1e-5, high=1e-3, log=True), 
        "p" : trial.suggest_int(name="p", low=10, high=100, step=10),
        "opt" : "adamw" 
    }
    return {**model_hyperparameters, **training_hyperparameters} 


def callback(study: optuna.Study, trial: optuna.Trial, figs_dir: str, arch_search_dir: str) -> None:
    try:
        print("@" * 80 + datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
        print(f"Best values so far:\n\t{study.best_trial}\n\t{study.best_trial.params}")
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
        df.to_pickle(os.path.join(arch_search_dir, "arch_search_df.pkl")) # Where to save it, usually as a .pkl
    except ValueError:
        pass # Not enough trials to plot yet


# @eqx.filter_jit
# def kl_loss(key, model, X, Y, Finv, preprocess_fn, n_samples_per_test):
#     """ KL divergence between posterior samples and true posterior """
#     model = eqx.nn.inference_mode(model, True)

#     # Gaussian posterior pdf at each point in parameter space (NOTE: flow returns log probs!)
#     gaussian = lambda y, x, Finv: jss.multivariate_normal.logpdf(y, mean=x, cov=Finv) # Posterior around MAP
#     model_log_prob = lambda y, x: model.log_prob(y, x)

#     def _kl(x, y):
#         # KL divergence between p and q for one x and many posterior samples y
#         q_pi_x = jax.vmap(gaussian, in_axes=(0, None, None))(y, x, Finv)
#         p_pi_x = jax.vmap(model_log_prob, in_axes=(0, None))(y, x) # NPE
#         # return 0.5 * (
#         #     jnp.log(jnp.exp(p_pi_x) / jnp.exp(q_pi_x)) + jnp.log(jnp.exp(q_pi_x) / jnp.exp(p_pi_x)) 
#         # )
#         # return jnp.log(jnp.exp(p_pi_x) / jnp.exp(q_pi_x)) 
#         return jnp.log(jnp.exp(q_pi_x) / jnp.exp(p_pi_x)) # Same "direction" of KL as training
#         # return 0.5 * jnp.sum(
#         #     jsp.kl_div(jnp.exp(p_pi_x), jnp.exp(q_pi_x)), 
#         #     jsp.kl_div(jnp.exp(q_pi_x), jnp.exp(p_pi_x))
#         # )

#     def _sample_posterior(key, x):
#         # Get posterior samples given datavector
#         samples, _ = sample_nde(
#             model,
#             key, 
#             x, 
#             preprocess_fn=preprocess_fn,
#             n_samples=(n_samples_per_test,)
#         )
#         return samples

#     # Get all posterior samples from all datavectors (n_test_data, n_samples_per_test)
#     keys = jr.split(key, len(X))
#     Y = jax.vmap(_sample_posterior)(keys, X)

#     # Apply KL loss to each posterior set of samples and the datavector used to derive them
#     kl_loss = jax.vmap(_kl)(X, Y).mean() # NOTE: not negative: minimise it.

#     print("kl loss", kl_loss)
#     return kl_loss


# @eqx.filter_jit
# def kl_loss_nle(model, simulations, parameters, Finv, parameter_prior, preprocess_fn):
#     model = eqx.nn.inference_mode(model, True)

#     def posterior_log_prob(x, y):
#         return model.log_prob(x, y) + parameter_prior.log_prob(y)

#     # Log posterior probs given true posterior
#     q_pi_x = jss.multivariate_normal(
#         parameters, mean=simulations, cov=Finv
#     )
#     # Log posterior probs given sbi posterior (no preprocess_fn used here...)
#     p_pi_x = jax.vmap(posterior_log_prob)(simulations, parameters)
#     return -jnp.log(q_pi_x / p_pi_x).mean() # NOTE: negative?
