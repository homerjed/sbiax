import os
from datetime import datetime
import optuna
# import plotly.graph_objects as go


def get_trial_hyperparameters(trial, model_type):
    # Arrange hyperparameters to optimise for and return to the experiment
    if model_type == "cnf":
        model_hyperparameters = {
            "width" : trial.suggest_int(name="width", low=3, high=7, step=1), # NN width
            "depth" : trial.suggest_int(name="depth", low=0, high=2, step=1), # NN depth
            "dt" : trial.suggest_float(name="dt", low=0.01, high=0.15, step=0.01), # ODE solver timestep
            "solver" : trial.suggest_categorical(name="solver", choices=["Euler", "Heun", "Tsit5"]), # ODE solver
        }
    if model_type == "maf":
        model_hyperparameters = {
            "width" : trial.suggest_int(name="width", low=3, high=7, step=1), # Hidden units in NNs
            "depth" : trial.suggest_int(name="depth", low=1, high=5, step=1), # Flow depth
            "layers" : trial.suggest_int(name="layers", low=1, high=3, step=1), # NN layers
        }
    training_hyperparameters = {
        # Training
        "n_batch" : trial.suggest_int(name="n_batch", low=40, high=100, step=10), 
        "lr": trial.suggest_float(name="lr", low=1e-5, high=1e-3, log=True), 
        "p" : trial.suggest_int(name="p", low=10, high=100, step=10),
    }
    return {**model_hyperparameters, **training_hyperparameters} 


def callback(
    study: optuna.Study, 
    trial: optuna.Trial, 
    figs_dir: str, 
    arch_search_dir: str
) -> None:
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