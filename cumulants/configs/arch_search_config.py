import os
from typing import Literal, Optional
import argparse
import yaml
import jax.random as jr 
from equinox import Module
from jaxtyping import PRNGKeyArray, jaxtyped
from beartype import beartype as typechecker
from ml_collections import ConfigDict

typecheck = jaxtyped(typechecker=typechecker)

""" 
    Architecture search
""" 

@typecheck
def arch_search_config(
    seed: int = 0, 
    exp_name: str = "",
    sbi_type: str = "nle", 
    linearised: bool = True, 
    n_linear_sims: int = 10_000,
    compression: Literal["linear", "nn", "nn-lbfgs"] = "linear",
    reduced_cumulants: bool = False,
    redshifts: list[float] = [0.0, 0.5, 1.0],
    order_idx: list[int] = [0, 1, 2],
    freeze_parameters: bool = False,
    pre_train: bool = False
) -> ConfigDict:

    config = ConfigDict()

    config.seed              = seed # NOTE: seed that ensemble configs run at also!
    config.sbi_type          = sbi_type
    config.exp_name_format   = "z={}_m={}" #.format(config.redshift, "".join(map(str, config.order_idx)))

    config.compression       = compression

    # Data
    config.dataset_name      = "cumulants" 
    config.redshifts         = redshifts # Redshifts to combine
    config.scales            = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
    config.order_idx         = order_idx
    config.n_linear_sims     = n_linear_sims
    config.compression       = compression
    config.reduced_cumulants = reduced_cumulants
    config.linearised        = linearised 
    config.pre_train         = pre_train and (not linearised) # Load linearised or pre-trained models
    config.freeze_parameters = freeze_parameters

    # Trials
    config.n_trials         = 100 # Number of trials in hyperparameter optimisation (per process)
    config.n_startup_trials = 20 # Number of warmup trials in hyperparameter optimisation
    config.n_processes      = 10

    return config