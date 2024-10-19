import os
import yaml
from ml_collections import ConfigDict


def save_config(config: ConfigDict, filepath: str):
    with open(filepath, 'w') as f:
        yaml.dump(config.to_dict(), f)


def load_config(filepath: str) -> ConfigDict:
    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)
    return ConfigDict(config_dict)


def make_dirs(results_dir):

    print("RESULTS_DIR:\n", results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    dirs = [
        "data/", "posteriors/", "models/", "figs/"
    ]
    for _dir in dirs:
        os.makedirs(
            os.path.join(results_dir, _dir), 
            exist_ok=True
        )


def get_results_dir(config, base_dir="./"):
    results_dir = (
        "results/" +  
        f"{config.sbi_type}/" + 
        f"{config.exp_name}/" + 
        f"{config.seed}/"
    )
    make_dirs(results_dir)
    return results_dir