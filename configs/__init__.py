import os
import yaml
from ml_collections import ConfigDict

from ..cumulants.data.constants import get_base_results_dir


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