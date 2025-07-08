import os
from typing import Literal, Optional
from jaxtyping import jaxtyped
from beartype import beartype as typechecker
from ml_collections import ConfigDict

from data.constants import ALL_RADII

typecheck = jaxtyped(typechecker=typechecker)

USE_SCALERS = True #if os.environ.get("USE_SCALERS", "").lower() in ("1", "true") else False 
DEFAULT_NDE_TYPE = os.environ.get("DEFAULT_NDE_TYPE", None)
DEFAULT_N_NDES = int(os.environ.get("DEFAULT_N_NDES", 1))

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# Architecture hyperparameters (default and found with arch search)
DEFAULT_MAF_ARCH = dict(
    width_size       = 32,
    n_layers         = 2,
    nn_depth         = 2,
    activation       = "tanh",
    use_scaling      = True
)

HP_OPT_MAF_ARCH = dict(
    width_size       = 2 ** 6, #32,
    n_layers         = 6,
    nn_depth         = 2,
    activation       = "swish",
    use_scaling      = True
)

DEFAULT_CNF_ARCH = dict(
    model_type       = "cnf",
    width_size       = 8, #8 # 32
    depth            = 0, #0, # 2
    activation       = "tanh",
    dropout_rate     = 0.,
    dt               = 0.05,
    t1               = 1.,
    solver           = "Euler", # Heun
    exact_log_prob   = True,
    use_scaling      = True # Defaults  
)

HP_OPT_CNF_ARCH = dict(
    model_type       = "cnf",
    width_size       = 32,
    depth            = 0,
    activation       = "tanh",
    dropout_rate     = 0.,
    dt               = 0.12,
    t1               = 1.,
    solver           = "Heun",
    exact_log_prob   = True,
    use_scaling      = True # Defaults  
)

# Optimisation hyperparameters (default and found with arch search)
DEFAULT_OPT = dict(
    start_step       = 0,
    n_epochs         = 10_000,
    n_batch          = 100,
    patience         = 20, #20,
    lr               = 1e-3,
    opt              = "adam",
    opt_kwargs       = {}
)

HP_OPT_OPT_MAF = dict(
    start_step       = 0,
    n_epochs         = 10_000,
    n_batch          = 80, #100,
    patience         = 300, #70,
    lr               = 0.000489390761268084, #1e-3,
    opt              = "adam",
    opt_kwargs       = {}
)

HP_OPT_OPT_CNF = dict(
    start_step       = 0,
    n_epochs         = 10_000,
    n_batch          = 70, 
    patience         = 250, #190, 
    lr               = 0.00013408396337403455,
    opt              = "lion",
    opt_kwargs       = {}
)

# Set the default training and architectures
DEFAULT_MAF_ARCH = HP_OPT_MAF_ARCH 
DEFAULT_OPT_MAF = HP_OPT_OPT_MAF 

DEFAULT_CNF_ARCH = HP_OPT_CNF_ARCH #DEFAULT_CNF_ARCH # HP_OPT_CNF_ARCH 
DEFAULT_OPT_CNF = HP_OPT_OPT_CNF # HP_OPT_OPT_CNF 

# Number of density estimators in the ensemble
N_NDES = default(int(DEFAULT_N_NDES), 1)


def get_default_nde(cnf, maf):
    _default = {"CNF": cnf, "MAF": maf}[DEFAULT_NDE_TYPE] if exists(DEFAULT_NDE_TYPE) else None
    return default(_default, maf)


def get_default_nde_opt():
    _default = {"CNF": DEFAULT_OPT_CNF, "MAF": DEFAULT_OPT_MAF}[DEFAULT_NDE_TYPE] if exists(DEFAULT_NDE_TYPE) else DEFAULT_OPT 
    return _default


def get_config_ndes(config):
    # Set the NDE architecture and training parameters for a config

    # CNF
    cnf = ConfigDict()
    cnf.model_type       = "cnf"
    cnf.width_size       = DEFAULT_CNF_ARCH["width_size"]
    cnf.depth            = DEFAULT_CNF_ARCH["depth"]
    cnf.activation       = DEFAULT_CNF_ARCH["activation"]
    cnf.dropout_rate     = DEFAULT_CNF_ARCH["dropout_rate"]
    cnf.dt               = DEFAULT_CNF_ARCH["dt"]
    cnf.t1               = DEFAULT_CNF_ARCH["t1"]
    cnf.solver           = DEFAULT_CNF_ARCH["solver"]
    cnf.exact_log_prob   = DEFAULT_CNF_ARCH["exact_log_prob"]
    cnf.use_scaling      = DEFAULT_CNF_ARCH["use_scaling"] # Defaults to (mu, std) of (x, y)

    # MAF
    maf = ConfigDict()
    maf.model_type       = "maf" # = model.__class__.__name__
    maf.width_size       = DEFAULT_MAF_ARCH["width_size"]
    maf.n_layers         = DEFAULT_MAF_ARCH["n_layers"]
    maf.nn_depth         = DEFAULT_MAF_ARCH["nn_depth"]
    maf.activation       = DEFAULT_MAF_ARCH["activation"]
    maf.use_scaling      = DEFAULT_MAF_ARCH["use_scaling"] # Defaults to (mu, std) of (x, y)

    # Ensemble
    config.ndes          = [get_default_nde(cnf, maf)] * N_NDES
    config.n_ndes        = len(config.ndes)

    _CONFIG_DEFAULT_OPT = get_default_nde_opt()

    # Optimisation (pre-train) hyperparameters (same for all NDEs...)
    if config.pre_train: # (Assumes config.ndes defined after default config setup)
        config.pretrain = pretrain = ConfigDict()
        pretrain.start_step  = 0
        pretrain.n_epochs    = 10_000
        pretrain.n_batch     = _CONFIG_DEFAULT_OPT["n_batch"] #100 
        pretrain.patience    = _CONFIG_DEFAULT_OPT["patience"] #10
        pretrain.lr          = _CONFIG_DEFAULT_OPT["lr"] #1e-3
        pretrain.opt         = _CONFIG_DEFAULT_OPT["opt"] #"adam" 
        pretrain.opt_kwargs  = {}

    # Optimisation hyperparameters (same for all NDEs...)
    config.train = train = ConfigDict()
    train.start_step     = 0
    train.n_epochs       = 10_000
    train.n_batch        = _CONFIG_DEFAULT_OPT["n_batch"]
    train.patience       = _CONFIG_DEFAULT_OPT["patience"]
    train.lr             = _CONFIG_DEFAULT_OPT["lr"]
    train.opt            = _CONFIG_DEFAULT_OPT["opt"]
    train.opt_kwargs     = {}

    return config


def default_posterior_sampling(config, no_config=False):

    # If not supplying config, return just the sampling parameters
    if no_config:
        config = ConfigDict()
        linearised = True
    else:
        linearised = config.linearised

    # Posterior sampling
    if linearised:
        config.n_steps        = 100
        config.n_walkers      = 2000
    else:
        config.n_steps        = 100
        config.n_walkers      = 2000
    config.burn               = int(0.1 * config.n_steps)

    return config


def default_cumulants_configuration(
    config,
    redshift: float = 0., 
    linearised: bool = True, 
    compression: Literal["linear", "nn", "nn-lbfgs"] = "linear",
    order_idx: list[int] = [0, 1, 2],
    scales: list[float] = ALL_RADII,
    freeze_parameters: bool = False,
    n_linear_sims: Optional[int] = None,
    pre_train: bool = False,
    use_planck: bool = False
) -> ConfigDict:

    config.redshift           = redshift
    config.scales             = scales
    config.order_idx          = order_idx # Maximum index is 2
    config.compression        = compression
    config.linearised         = linearised
    config.covariance_epsilon = None # 1e-6
    config.pre_train          = pre_train and (not linearised)
    config.n_linear_sims      = n_linear_sims # This is for pre-train or linearised simulations 
    config.use_expectation    = False # Noiseless datavector
    config.valid_fraction     = 0.1
    config.freeze_parameters  = freeze_parameters

    config.use_planck         = use_planck

    return config
    

def default_cut_configuration(config, bulk_or_tails):

    if bulk_or_tails == "tails":
        config.p_value_min    = 0.01
        config.p_value_max    = 0.99
    if bulk_or_tails == "bulk":
        config.p_value_min    = 0.03
        config.p_value_max    = 0.90

    config.use_means            = False # Calculate central moments of the bulk or not
    config.stack_means          = True # Stack means of bulk of the PDF at each scale with the other cumulants
    config.use_normalisations   = True # Stack norms of bulk of the PDF at each scale with the other cumulants
    config.fiducial_based_normalisation = config.p_value_max - config.p_value_min

    return config 


@typecheck
def cumulants_config(
    seed: int = 0, 
    redshift: float = 0., 
    linearised: bool = True, 
    compression: Literal["linear", "nn", "nn-lbfgs"] = "linear",
    order_idx: list[int] = [0, 1, 2],
    scales: list[float] = ALL_RADII,
    freeze_parameters: bool = False,
    n_linear_sims: Optional[int] = None,
    pre_train: bool = False,
    use_planck: bool = False
) -> ConfigDict:

    config = ConfigDict()

    config.seed               = seed # For argparse script running without args!

    # Data
    config.dataset_name       = "cumulants" 

    config = default_cumulants_configuration(
        config, 
        redshift=redshift, 
        linearised=linearised,
        compression=compression,
        order_idx=order_idx,
        scales=scales,
        freeze_parameters=freeze_parameters,
        n_linear_sims=n_linear_sims,
        pre_train=pre_train
    )

    config = default_cut_configuration(config, bulk_or_tails="tails")

    config.use_planck         = use_planck

    # Miscallaneous
    config.use_scalers        = USE_SCALERS # Input scalers for (xi, pi) in NDEs (NOTE: checked that scalings aren't optimised!)

    # SBI
    config.sbi_type           = "nle"

    # Posterior sampling
    config = default_posterior_sampling(config)

    # NDEs
    config = get_config_ndes(config)

    return config


@typecheck
def arch_search_cumulants_config( # Copy of the above config for architecture search
    seed: int = 0, 
    redshift: float = 0., 
    linearised: bool = True, 
    compression: Literal["linear", "nn", "nn-lbfgs"] = "linear",
    order_idx: list[int] = [0, 1, 2],
    scales: list[float] = ALL_RADII,
    freeze_parameters: bool = False,
    n_linear_sims: Optional[int] = None,
    pre_train: bool = False,
    use_planck: bool = False
) -> ConfigDict:

    config = ConfigDict()

    config.seed               = seed # For argparse script running without args!

    # Data
    config.dataset_name       = "cumulants"

    config = default_cumulants_configuration(
        config, 
        redshift=redshift, 
        linearised=linearised,
        compression=compression,
        order_idx=order_idx,
        scales=scales,
        freeze_parameters=freeze_parameters,
        n_linear_sims=n_linear_sims,
        pre_train=pre_train
    )

    config = default_cut_configuration(config, bulk_or_tails="tails")

    config.use_planck         = use_planck

    # Miscallaneous
    config.use_scalers        = USE_SCALERS # Input scalers for (xi, pi) in NDEs (NOTE: checked that scalings aren't optimised!)

    # SBI
    config.sbi_type           = "nle" 

    # Posterior sampling
    config = default_posterior_sampling(config)

    # NDEs
    config = get_config_ndes(config)

    return config


@typecheck
def bulk_cumulants_config(
    seed: int = 0, 
    redshift: float = 0., 
    linearised: bool = True, 
    compression: Literal["linear", "nn", "nn-lbfgs"] = "linear",
    order_idx: list[int] = [0, 1, 2],
    scales: list[float] = ALL_RADII,
    freeze_parameters: bool = False,
    n_linear_sims: Optional[int] = None,
    pre_train: bool = False,
    use_planck: bool = False
) -> ConfigDict:

    config = ConfigDict()

    config.seed               = seed # For argparse script running without args!

    # Data
    config.dataset_name       = "bulk cumulants" 

    config = default_cumulants_configuration(
        config, 
        redshift=redshift, 
        linearised=linearised,
        compression=compression,
        order_idx=order_idx,
        scales=scales,
        freeze_parameters=freeze_parameters,
        n_linear_sims=n_linear_sims,
        pre_train=pre_train
    )

    config = default_cut_configuration(config, bulk_or_tails="bulk")

    config.use_planck         = use_planck

    # Miscallaneous
    config.use_scalers        = USE_SCALERS 

    # SBI
    config.sbi_type           = "nle" 

    # Posterior sampling
    config = default_posterior_sampling(config)

    # NDEs
    config = get_config_ndes(config)

    return config


@typecheck
def bulk_pdf_config(
    seed: int = 0, 
    redshift: float = 0., 
    linearised: bool = True, 
    compression: Literal["linear", "nn", "nn-lbfgs"] = "linear",
    order_idx: list[int] = [0, 1, 2],
    scales: list[float] = ALL_RADII,
    freeze_parameters: bool = False,
    n_linear_sims: Optional[int] = None,
    pre_train: bool = False,
    use_planck: bool = False
) -> ConfigDict:

    config = bulk_cumulants_config(
        seed=seed,
        redshift=redshift,
        linearised=linearised,
        compression=compression,
        order_idx=order_idx,
        scales=scales,
        freeze_parameters=freeze_parameters,
        n_linear_sims=n_linear_sims,
        pre_train=pre_train,
        use_planck=use_planck
    )

    return config