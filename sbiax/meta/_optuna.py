import os
from datetime import datetime
import optuna
import plotly.graph_objects as go


def get_trial_hyperparameters(
    trial: optuna.Trial, 
    model_type: str
) -> dict:
    """
    This function uses Optuna's `trial` object to suggest hyperparameter values for optimization. 

    Returns a dictionary of hyperparameters. 

    The hyperparameters returned are specific to the type of model being optimized (`CNF`, `MAF`, or `GMM`), 
    along with common training hyperparameters.

    Args:
        trial (optuna.Trial): The current trial object, used to suggest hyperparameter values.
        model_type (str): The type of model for which hyperparameters are being optimized. 
            Must be one of the following:
            - `"CNF"`: Continuous Normalizing Flow
            - `"MAF"`: Masked Autoregressive Flow
            - `"GMM"`: Gaussian Mixture Model

    Returns:
        dict: A dictionary containing the model-specific and training hyperparameters.

        Model-specific hyperparameters:
        - For `"CNF"`:
            - `width` (int): Neural network width (2 to 5).
            - `depth` (int): Neural network depth (0 to 2).
            - `dt` (float): ODE solver timestep (0.01 to 0.15, step 0.01).
            - `solver` (str): ODE solver choice (`"Euler"`, `"Heun"`, `"Tsit5"`).
        - For `"MAF"`:
            - `width` (int): Number of hidden units in neural networks (3 to 7).
            - `depth` (int): Flow depth (1 to 5).
            - `layers` (int): Number of layers in neural networks (1 to 3).
        - For `"GMM"`:
            - `width` (int): Number of hidden units in neural networks (3 to 7).
            - `depth` (int): Number of hidden layers in neural networks (1 to 5).
            - `n_components` (int): Number of Gaussian mixture components (1 to 5).

        Common training hyperparameters:
            - `n_batch` (int): Batch size (40 to 100, step 10).
            - `lr` (float): Learning rate (log scale, 1e-5 to 1e-3).
            - `p` (int): Additional parameter `p` (10 to 100, step 10).

    Raises:
        ValueError: If `model_type` is not one of `"CNF"`, `"MAF"`, or `"GMM"`.

    Example:
        >>> def objective(trial):
        >>>     model_type = "CNF"
        >>>     hyperparameters = get_trial_hyperparameters(trial, model_type)
        >>>     # Use hyperparameters to train and evaluate the model.
        >>>     return score  # Optimization target.
    """
    # Arrange hyperparameters to optimise for and return to the experiment
    if model_type == "CNF":
        model_hyperparameters = {
            "width" : trial.suggest_int(name="width", low=2, high=5, step=1), # NN width
            "depth" : trial.suggest_int(name="depth", low=0, high=2, step=1), # NN depth
            "dt" : trial.suggest_float(name="dt", low=0.01, high=0.15, step=0.01), # ODE solver timestep
            "solver" : trial.suggest_categorical(name="solver", choices=["Euler", "Heun", "Tsit5"]), # ODE solver
        }
    if model_type == "MAF":
        model_hyperparameters = {
            "width" : trial.suggest_int(name="width", low=3, high=7, step=1), # Hidden units in NNs
            "depth" : trial.suggest_int(name="depth", low=1, high=5, step=1), # Flow depth
            "layers" : trial.suggest_int(name="layers", low=1, high=3, step=1), # NN layers
        }
    if model_type == "GMM":
        model_hyperparameters = {
            "width" : trial.suggest_int(name="width", low=3, high=7, step=1), # Hidden units in NNs
            "depth" : trial.suggest_int(name="depth", low=1, high=5, step=1), # Hidden layers 
            "n_components" : trial.suggest_int(name="n_components", low=1, high=5, step=1), # Mixture components
        }

    training_hyperparameters = {
        # Training
        "n_batch" : trial.suggest_int(name="n_batch", low=40, high=100, step=10), 
        "lr" : trial.suggest_float(name="lr", low=1e-5, high=1e-3, log=True), 
        "p" : trial.suggest_int(name="p", low=10, high=100, step=10),
    }
    return {**model_hyperparameters, **training_hyperparameters} 


def callback(
    study: optuna.Study, 
    trial: optuna.Trial, 
    figs_dir: str, 
    arch_search_dir: str
) -> None:
    """
    Callback function for Optuna study to log and visualize progress during optimization.

    This function is executed at the end of each trial, saving visualizations of the study's
    progress, parameter importances, and other insights. 

    Args:
        study (`optuna.Study`): The Optuna study instance, which contains the trials and results of 
            the optimization process.
        trial (`optuna.Trial`): The current trial instance being evaluated.
        figs_dir (str): Directory path to save visualization figures, such as parameter importances, 
            optimization history, contour plots, etc.
        arch_search_dir (`str`): Directory path to save a DataFrame containing trial details.

    Workflow:
        - Logs the best trial and its parameters after each trial.
        - Generates and displays the following visualizations:
            1. Parameter importances
            2. Optimization history
            3. Contour plot
            4. Intermediate values (if applicable)
            5. Timeline of the trials
        - Saves the generated visualizations as PDF files in the `figs_dir`.
        - Saves a DataFrame of the trial results as a pickle file in `arch_search_dir`.

    Exceptions:
        - Catches `ValueError` if there are not enough trials to generate certain visualizations (e.g., 
          parameter importance or contour plots).

    Notes:
        - Requires `plotly` for visualization.
        - Directories specified by `figs_dir` and `arch_search_dir` must exist before running the callback.

    Example:
        >>> study = optuna.create_study(direction="minimize")
        >>> study.optimize(objective_function, callbacks=[callback])

    """
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
        df.to_pickle(os.path.join(arch_search_dir, "arch_search_df.pkl")) 
    except ValueError:
        pass # Not enough trials to plot yet