import os
import yaml
from ml_collections import ConfigDict


def save_config(config: ConfigDict, filepath: str):
    """ Save a config to a yaml file. """
    with open(filepath, 'w') as f:
        yaml.dump(config.to_dict(), f)


def load_config(filepath: str) -> ConfigDict:
    """ Load a config to a yaml file. """
    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)
    return ConfigDict(config_dict)


def make_dirs(results_dir: str) -> None:
    """ Create directories for saving experimental results. """
    print("RESULTS_DIR:\n", results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    dirs = [
        "data/", "posteriors/", "models/", "figs/"
    ]
    for _dir in dirs:
        os.makedirs(os.path.join(results_dir, _dir), exist_ok=True)


def get_results_dir(config: ConfigDict, base_dir="./") -> str:
    """ Format the results directory based on parameters of a config. """
    results_dir = os.path.join(
        base_dir, "results/{}/{}/{}/".format(
            config.sbi_type, config.exp_name, config.seed
        )
    )
    make_dirs(results_dir)
    return results_dir