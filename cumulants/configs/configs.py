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

USE_SCALERS = True
FREEZE_PARAMETERS = False

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


def get_results_dir(config: ConfigDict, args: dict, *, arch_search: bool = False) -> str:

    results_dir = "{}{}{}{}{}{}{}{}{}{}{}".format( # Import this from a constants file
        get_base_results_dir(),
        "arch_search/" if arch_search else "",
        "frozen/" if config.freeze_parameters else "nonfrozen/",
        "{}/".format(args.bulk_or_tails),
        "reduced_cumulants/" if config.reduced_cumulants else "cumulants/",
        (config.sbi_type + "/") if config.sbi_type else "" ,
        "linearised/" if config.linearised else "nonlinearised/",
        (config.compression + "/") if config.compression else "",
        "pretrain/" if config.pre_train else "",
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


def get_posteriors_dir(config: ConfigDict, args, *, arch_search: bool = False) -> str:

    results_dir = "{}{}{}{}{}{}{}{}{}{}{}".format( 
        get_base_posteriors_dir(),
        "arch_search/" if arch_search else "",
        "frozen/" if config.freeze_parameters else "nonfrozen/",
        "{}/".format(args.bulk_or_tails),
        "reduced_cumulants/" if config.reduced_cumulants else "cumulants/",
        (config.sbi_type + "/") if config.sbi_type else "" ,
        "linearised/" if config.linearised else "nonlinearised/",
        (config.compression + "/") if config.compression else "",
        "pretrain/" if config.pre_train else "",
        (config.exp_name + "/") if config.exp_name else "",
        (str(config.seed) + "/") if (config.seed is not None) else ""
    )

    make_dirs(results_dir)

    return results_dir


def get_multi_z_posterior_dir(config: ConfigDict, args) -> str:
    posterior_save_dir = "{}{}{}{}/multi_z/{}/{}{}".format(
        get_base_posteriors_dir(),
        "frozen/" if config.freeze_parameters else "nonfrozen/",
        "{}/".format(args.bulk_or_tails),
        "reduced_cumulants" if config.reduced_cumulants else "cumulants",
        config.sbi_type if (hasattr(config, "sbi_type") and (config.sbi_type is not None)) else args.sbi_type,
        "linearised/" if config.linearised else "nonlinearised/",
        "pretrain/" if config.pre_train else ""
    )
    print("Multi-z posterior dir:\n", posterior_save_dir)
    return posterior_save_dir


"""
    Configs
"""


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


@typecheck
def cumulants_config(
    seed: int = 0, 
    redshift: float = 0., 
    reduced_cumulants: bool = False,
    sbi_type: Literal["nle", "npe"] = "nle", 
    linearised: bool = True, 
    compression: Literal["linear", "nn", "nn-lbfgs"] = "linear",
    order_idx: list[int] = [0, 1, 2],
    # nonlinearised: bool = True, 
    freeze_parameters: bool = False,
    n_linear_sims: Optional[int] = None,
    pre_train: bool = False
) -> ConfigDict:

    config = ConfigDict()

    config.seed               = seed # For argparse script running without args!

    # Data
    config.dataset_name       = "reduced cumulants" if reduced_cumulants else "cumulants" 
    config.redshift           = redshift
    config.scales             = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
    config.order_idx          = order_idx # Maximum index is 2
    config.compression        = compression
    config.linearised         = linearised
    config.covariance_epsilon = None # 1e-6
    config.reduced_cumulants  = reduced_cumulants
    config.pre_train          = pre_train and (not linearised)
    config.n_linear_sims      = n_linear_sims # This is for pre-train or linearised simulations 
    config.use_expectation    = False # Noiseless datavector
    config.valid_fraction     = 0.1
    config.freeze_parameters  = freeze_parameters

    # Miscallaneous
    config.use_scalers        = USE_SCALERS # Input scalers for (xi, pi) in NDEs (NOTE: checked that scalings aren't optimised!)
    config.use_pca            = False # Need to add this into other scripts...
    config.ema_rate           = 0.995
    config.use_ema            = False # Use it and sample with it

    # SBI
    config.sbi_type           = sbi_type

    # Experiments
    config.exp_name           = "z={}_m={}".format(config.redshift, "".join(map(str, config.order_idx)))

    # Posterior sampling
    config.n_steps            = 200
    config.n_walkers          = 1000
    config.burn               = int(0.1 * config.n_steps)

    if config.linearised:
        # NDEs
        config.cnf = cnf = ConfigDict()
        cnf.model_type       = "cnf"
        cnf.width_size       = 8
        cnf.depth            = 0
        cnf.activation       = "tanh"
        cnf.dropout_rate     = 0.
        cnf.dt               = 0.1
        cnf.t1               = 1.
        cnf.solver           = "Euler" 
        cnf.exact_log_prob   = True
        cnf.use_scaling      = True # Defaults to (mu, std) of (x, y)

        config.maf = maf = ConfigDict()
        maf.model_type       = "maf" # = model.__class__.__name__
        maf.width_size       = 32
        maf.n_layers         = 2
        maf.nn_depth         = 2
        maf.activation       = "tanh"
        maf.use_scaling      = True # Defaults to (mu, std) of (x, y)

        config.ndes          = [cnf]#, cnf, cnf]  
        config.n_ndes        = len(config.ndes)

        # Optimisation hyperparameters (same for all NDEs...)
        config.train = train = ConfigDict()
        train.start_step     = 0
        train.n_epochs       = 10_000
        train.n_batch        = 100 
        train.patience       = 10
        train.lr             = 1e-3
        train.opt            = "adam" 
        train.opt_kwargs     = {}
    else:
        # NDEs
        config.cnf = cnf = ConfigDict()
        cnf.model_type       = "cnf"
        cnf.width_size       = 32 # 8
        cnf.depth            = 2 # 0
        cnf.activation       = "tanh"
        cnf.dropout_rate     = 0.
        cnf.dt               = 0.05
        cnf.t1               = 1.
        cnf.solver           = "Heun"
        cnf.exact_log_prob   = True
        cnf.use_scaling      = True # Defaults to (mu, std) of (x, y)

        config.maf = maf = ConfigDict()
        maf.model_type       = "maf" # = model.__class__.__name__
        maf.width_size       = 32
        maf.n_layers         = 2
        maf.nn_depth         = 2
        maf.activation       = "tanh"
        maf.use_scaling      = True # Defaults to (mu, std) of (x, y)

        config.ndes          = [maf] #cnf]#, cnf, cnf] 
        config.n_ndes        = len(config.ndes)

        # Optimisation hyperparameters (same for all NDEs...)
        config.pretrain = pretrain = ConfigDict()
        pretrain.start_step  = 0
        pretrain.n_epochs    = 10_000
        pretrain.n_batch     = 100 
        pretrain.patience    = 10
        pretrain.lr          = 1e-3
        pretrain.opt         = "adam" 
        pretrain.opt_kwargs  = {}

        config.train = train = ConfigDict()
        train.start_step     = 0
        train.n_epochs       = 10_000
        train.n_batch        = 100 
        train.patience       = 200
        train.lr             = 1e-3
        train.opt            = "adam" 
        train.opt_kwargs     = {}

    return config


@typecheck
def bulk_cumulants_config(
    seed: int = 0, 
    redshift: float = 0., 
    reduced_cumulants: bool = False,
    sbi_type: Literal["nle", "npe"] = "nle", 
    linearised: bool = True, 
    compression: Literal["linear", "nn", "nn-lbfgs"] = "linear",
    order_idx: list[int] = [0, 1, 2],
    # nonlinearised: bool = True, 
    freeze_parameters: bool = False,
    n_linear_sims: Optional[int] = None,
    pre_train: bool = False
) -> ConfigDict:

    config = ConfigDict()

    config.seed               = seed # For argparse script running without args!

    # Data
    config.dataset_name       = "reduced bulk cumulants" if reduced_cumulants else "bulk cumulants" 
    config.redshift           = redshift
    config.p_value_min        = 0.03
    config.p_value_max        = 0.90
    config.scales             = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
    config.order_idx          = order_idx # Maximum index is 2
    config.compression        = compression
    config.linearised         = linearised
    config.covariance_epsilon = None # 1e-6
    config.reduced_cumulants  = reduced_cumulants
    config.pre_train          = pre_train and (not linearised)
    config.n_linear_sims      = n_linear_sims # This is for pre-train or linearised simulations 
    config.use_expectation    = False # Noiseless datavector
    config.valid_fraction     = 0.1
    config.freeze_parameters  = freeze_parameters

    # Miscallaneous
    config.use_scalers        = USE_SCALERS # Input scalers for (xi, pi) in NDEs (NOTE: checked that scalings aren't optimised!)
    config.use_pca            = False # Need to add this into other scripts...
    config.ema_rate           = 0.995
    config.use_ema            = False # Use it and sample with it

    # SBI
    config.sbi_type           = sbi_type

    # Experiments
    config.exp_name           = "z={}_m={}".format(config.redshift, "".join(map(str, config.order_idx)))

    # Posterior sampling
    config.n_steps            = 200
    config.n_walkers          = 1000
    config.burn               = int(0.1 * config.n_steps)

    if config.linearised:
        # NDEs
        config.cnf = cnf = ConfigDict()
        cnf.model_type       = "cnf"
        cnf.width_size       = 8
        cnf.depth            = 0
        cnf.activation       = "tanh"
        cnf.dropout_rate     = 0.
        cnf.dt               = 0.1
        cnf.t1               = 1.
        cnf.solver           = "Euler" 
        cnf.exact_log_prob   = True
        cnf.use_scaling      = True # Defaults to (mu, std) of (x, y)

        config.maf = maf = ConfigDict()
        maf.model_type       = "maf" # = model.__class__.__name__
        maf.width_size       = 32
        maf.n_layers         = 2
        maf.nn_depth         = 2
        maf.activation       = "tanh"
        maf.use_scaling      = True # Defaults to (mu, std) of (x, y)

        config.ndes          = [cnf]#, cnf, cnf]  
        config.n_ndes        = len(config.ndes)

        # Optimisation hyperparameters (same for all NDEs...)
        config.train = train = ConfigDict()
        train.start_step     = 0
        train.n_epochs       = 10_000
        train.n_batch        = 100 
        train.patience       = 10
        train.lr             = 1e-3
        train.opt            = "adam" 
        train.opt_kwargs     = {}
    else:
        # NDEs
        config.cnf = cnf = ConfigDict()
        cnf.model_type       = "cnf"
        cnf.width_size       = 32 # 8
        cnf.depth            = 2 # 0
        cnf.activation       = "tanh"
        cnf.dropout_rate     = 0.
        cnf.dt               = 0.01
        cnf.t1               = 1.
        cnf.solver           = "Euler"
        cnf.exact_log_prob   = True
        cnf.use_scaling      = True # Defaults to (mu, std) of (x, y)

        config.maf = maf = ConfigDict()
        maf.model_type       = "maf" # = model.__class__.__name__
        maf.width_size       = 32
        maf.n_layers         = 8
        maf.nn_depth         = 2
        maf.activation       = "tanh"
        maf.use_scaling      = True # Defaults to (mu, std) of (x, y)

        config.ndes          = [cnf]#, cnf, cnf] #maf, maf, maf]  
        config.n_ndes        = len(config.ndes)

        # Optimisation hyperparameters (same for all NDEs...)
        config.pretrain = pretrain = ConfigDict()
        pretrain.start_step  = 0
        pretrain.n_epochs    = 10_000
        pretrain.n_batch     = 100 
        pretrain.patience    = 10
        pretrain.lr          = 1e-3
        pretrain.opt         = "adam" 
        pretrain.opt_kwargs  = {}

        config.train = train = ConfigDict()
        train.start_step     = 0
        train.n_epochs       = 10_000
        train.n_batch        = 100 
        train.patience       = 100
        train.lr             = 1e-3
        train.opt            = "adam" 
        train.opt_kwargs     = {}

    return config
    

"""
    Ensembles
"""


@typecheck
def ensembles_cumulants_config(
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
    config.dataset_name      = "reduced cumulants" if reduced_cumulants else "cumulants" 
    config.redshifts         = redshifts # Redshifts to combine
    config.scales            = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
    config.order_idx         = order_idx
    config.n_linear_sims     = n_linear_sims
    config.compression       = compression
    config.reduced_cumulants = reduced_cumulants
    config.linearised        = linearised 
    config.pre_train         = pre_train and (not linearised) # Load linearised or pre-trained models
    config.freeze_parameters = freeze_parameters

    config.use_ema           = False # Use it and sample with it

    # Posterior sampling
    config.n_steps           = 200
    config.n_walkers         = 1000
    config.burn              = int(0.1 * config.n_steps)

    return config


@typecheck
def ensembles_bulk_cumulants_config(
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
    config.dataset_name      = "reduced bulk cumulants" if reduced_cumulants else "bulk cumulants" 
    config.redshifts         = redshifts # Redshifts to combine
    config.scales            = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
    config.order_idx         = order_idx
    config.n_linear_sims     = n_linear_sims
    config.compression       = compression
    config.reduced_cumulants = reduced_cumulants
    config.linearised        = linearised 
    config.pre_train         = pre_train and (not linearised) # Load linearised or pre-trained models
    config.freeze_parameters = freeze_parameters 

    config.use_ema           = False # Use it and sample with it

    # Posterior sampling
    config.n_steps           = 200
    config.n_walkers         = 1000
    config.burn              = int(0.1 * config.n_steps)

    return config


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
    config.freeze_parameters = FREEZE_PARAMETERS

    # Trials
    config.n_trials         = 500 # Number of trials in hyperparameter optimisation (per process)
    config.n_startup_trials = 50 # Number of warmup trials in hyperparameter optimisation
    config.n_processes      = 10

    return config
