from typing import Literal
from jaxtyping import jaxtyped
from beartype import beartype as typechecker
from ml_collections import ConfigDict

from .cumulants_configs import default_posterior_sampling
from data.constants import ALL_RADII

typecheck = jaxtyped(typechecker=typechecker)

"""
    Ensembles
"""


@typecheck
def ensembles_cumulants_config(
    seed: int = 0, 
    sbi_type: str = "nle", 
    linearised: bool = True, 
    n_linear_sims: int = 10_000,
    compression: Literal["linear", "nn", "nn-lbfgs"] = "linear",
    redshifts: list[float] = [0.0, 0.5, 1.0],
    order_idx: list[int] = [0, 1, 2],
    scales: list[float] = ALL_RADII,
    freeze_parameters: bool = False,
    pre_train: bool = False
) -> ConfigDict:

    config = ConfigDict()

    config.seed              = seed # NOTE: seed that ensemble configs run at also!
    config.sbi_type          = sbi_type

    config.compression       = compression

    # Data
    config.dataset_name      = "cumulants" 
    config.redshifts         = redshifts # Redshifts to combine
    config.scales            = scales
    config.order_idx         = order_idx
    config.n_linear_sims     = n_linear_sims
    config.compression       = compression
    config.linearised        = linearised 
    config.pre_train         = pre_train and (not linearised) # Load linearised or pre-trained models
    config.freeze_parameters = freeze_parameters

    # Posterior sampling
    config                   = default_posterior_sampling(config)

    return config


@typecheck
def ensembles_bulk_cumulants_config(
    seed: int = 0, 
    sbi_type: str = "nle", 
    linearised: bool = True, 
    n_linear_sims: int = 10_000,
    compression: Literal["linear", "nn", "nn-lbfgs"] = "linear",
    redshifts: list[float] = [0.0, 0.5, 1.0],
    order_idx: list[int] = [0, 1, 2],
    scales: list[float] = ALL_RADII,
    freeze_parameters: bool = False,
    pre_train: bool = False
) -> ConfigDict:

    config = ConfigDict()

    config.seed              = seed # NOTE: seed that ensemble configs run at also!
    config.sbi_type          = sbi_type

    config.compression       = compression

    # Data
    config.dataset_name      = "bulk cumulants" 
    config.redshifts         = redshifts # Redshifts to combine
    config.scales            = scales
    config.order_idx         = order_idx
    config.n_linear_sims     = n_linear_sims
    config.compression       = compression
    config.linearised        = linearised 
    config.pre_train         = pre_train and (not linearised) # Load linearised or pre-trained models
    config.freeze_parameters = freeze_parameters 

    # Posterior sampling
    config                   = default_posterior_sampling(config)

    return config