import os
from typing import Literal, Optional
import argparse
import yaml
import jax.random as jr 
from equinox import Module
from jaxtyping import PRNGKeyArray, jaxtyped
from beartype import beartype as typechecker
from ml_collections import ConfigDict

from data.constants import get_base_results_dir, get_base_posteriors_dir
from sbiax.ndes import CNF, MAF

typecheck = jaxtyped(typechecker=typechecker)


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def save_config(config: ConfigDict, filepath: str):
    with open(filepath, 'w') as f:
        yaml.dump(config.to_dict(), f)


def load_config(filepath: str) -> ConfigDict:
    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)
    return ConfigDict(config_dict)


"""
    Save & load directories
"""


def make_dirs(results_dir: str) -> None:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    print("RESULTS_DIR:\n", results_dir)


def get_results_dir(config: ConfigDict, args: argparse.Namespace, *, arch_search: bool = False) -> str:

    results_dir = "{}{}{}{}{}{}{}{}{}{}{}".format( # Import this from a constants file
        get_base_results_dir(),
        "arch_search/" if arch_search else "",
        "frozen/" if config.freeze_parameters else "nonfrozen/",
        "{}/".format(args.bulk_or_tails),
        "reduced_cumulants/" if config.reduced_cumulants else "cumulants/",
        (config.sbi_type + "/") if config.sbi_type else "" ,
        "linearised/" if config.linearised else "nonlinearised/",
        (config.compression + "/") if config.compression else "",
        "pretrain/" if config.pre_train else "nopretrain/",
        (config.exp_name + "/") if config.exp_name else "",
        (str(config.seed) + "/") if (config.seed is not None) else ""
    )

    make_dirs(results_dir)

    # Save command line arguments and config together
    with open(os.path.join(results_dir, "config.yml"), "w") as f:
        yaml.dump({"args": ""}, f, default_flow_style=False)
        yaml.dump(vars(args), f, default_flow_style=False)
        yaml.dump({"config": ""}, f, default_flow_style=False)
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    return results_dir


def get_posteriors_dir(config: ConfigDict, args: argparse.Namespace, *, arch_search: bool = False) -> str:

    results_dir = "{}{}{}{}{}{}{}{}{}{}{}".format( 
        get_base_posteriors_dir(),
        "arch_search/" if arch_search else "",
        "frozen/" if config.freeze_parameters else "nonfrozen/",
        "{}/".format(args.bulk_or_tails),
        "reduced_cumulants/" if config.reduced_cumulants else "cumulants/",
        (config.sbi_type + "/") if config.sbi_type else "" ,
        "linearised/" if config.linearised else "nonlinearised/",
        (config.compression + "/") if config.compression else "",
        "pretrain/" if config.pre_train else "nopretrain/",
        (config.exp_name + "/") if config.exp_name else "",
        (str(config.seed) + "/") if (config.seed is not None) else ""
    )

    make_dirs(results_dir)

    return results_dir


def get_multi_z_posterior_dir(config: ConfigDict, args: argparse.Namespace) -> str:

    posterior_save_dir = "{}{}{}{}{}{}{}{}{}/multi_z/".format(
        get_base_posteriors_dir(),
        "frozen/" if config.freeze_parameters else "nonfrozen/",
        "{}/".format(args.bulk_or_tails),
        "reduced_cumulants" if config.reduced_cumulants else "cumulants",
        (config.sbi_type + "/") if config.sbi_type else "" , 
        (config.compression + "/") if config.compression else "",
        "linearised/" if config.linearised else "nonlinearised/",
        "pretrain/" if config.pre_train else "nopretrain/",
        (str(config.seed) + "/") if (config.seed is not None) else "", # No exp name here...
    )

    print("Multi-z posterior dir:\n", posterior_save_dir)

    return posterior_save_dir


"""
    Configs
"""


DEFAULT_MAF_ARCH = dict(
    width_size       = 32,
    n_layers         = 2,
    nn_depth         = 2,
    activation       = "tanh",
    use_scaling      = True
)

DEFAULT_CNF_ARCH = dict(
    model_type       = "cnf",
    width_size       = 8,
    depth            = 0,
    activation       = "tanh",
    dropout_rate     = 0.,
    dt               = 0.1,
    t1               = 1.,
    solver           = "Euler",
    exact_log_prob   = True,
    use_scaling      = True # Defaults  
)


@typecheck
def get_ndes_from_config(
    config: ConfigDict, 
    event_dim: int, 
    context_dim: Optional[int] = None, 
    scalers: Optional[list[Module] | Module] = None, 
    *, 
    use_scalers: bool = False,
    key: PRNGKeyArray 
) -> list[Module]:

    keys = jr.split(key, len(config.ndes))

    # Scaler for each nde...
    if not isinstance(scalers, list):
        # Pack the single scaler for each NDE
        scalers = [scalers] * len(config.ndes)

    ndes = []
    for nde, scaler, key in zip(config.ndes, scalers, keys):

        if nde.model_type == "maf":
            nde_arch = MAF
        if nde.model_type == "cnf":
            nde_arch = CNF

        # Required to remove / add some arguments to specify NDEs
        nde_dict = dict(
            event_dim=event_dim, 
            context_dim=context_dim if (context_dim is not None) else event_dim, 
            key=key,
            scaler=scaler if (nde.use_scaling and use_scalers) else None,
            **dict(nde)
        )
        nde_dict.pop("model_type")
        nde_dict.pop("use_scaling")

        ndes.append(nde_arch(**nde_dict))

    return ndes