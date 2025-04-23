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

USE_SCALERS = True

DEFAULT_MAF_ARCH = dict(
    width_size       = 32,
    n_layers         = 2,
    nn_depth         = 2,
    activation       = "tanh",
    use_scaling      = True
)

DEFAULT_CNF_ARCH = dict(
    model_type       = "cnf",
    width_size       = 8, # 32
    depth            = 0, # 2
    activation       = "tanh",
    dropout_rate     = 0.,
    dt               = 0.1,
    t1               = 1.,
    solver           = "Euler", # Heun
    exact_log_prob   = True,
    use_scaling      = True # Defaults  
)

DEFAULT_OPT = dict(
    start_step     = 0,
    n_epochs       = 10_000,
    n_batch        = 100,
    patience       = 200,
    lr             = 1e-3,
    opt            = "adam",
    opt_kwargs     = {}
)


@typecheck
def cumulants_config(
    seed: int = 0, 
    redshift: float = 0., 
    reduced_cumulants: bool = False,
    sbi_type: Literal["nle", "npe"] = "nle", 
    linearised: bool = True, 
    compression: Literal["linear", "nn", "nn-lbfgs"] = "linear",
    order_idx: list[int] = [0, 1, 2],
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
        cnf.width_size       = DEFAULT_CNF_ARCH["width_size"]
        cnf.depth            = DEFAULT_CNF_ARCH["depth"]
        cnf.activation       = DEFAULT_CNF_ARCH["activation"]
        cnf.dropout_rate     = DEFAULT_CNF_ARCH["dropout_rate"]
        cnf.dt               = DEFAULT_CNF_ARCH["dt"]
        cnf.t1               = DEFAULT_CNF_ARCH["t1"]
        cnf.solver           = DEFAULT_CNF_ARCH["solver"]
        cnf.exact_log_prob   = DEFAULT_CNF_ARCH["exact_log_prob"]
        cnf.use_scaling      = DEFAULT_CNF_ARCH["use_scaling"] # Defaults to (mu, std) of (x, y)

        config.maf = maf = ConfigDict()
        maf.model_type       = "maf" # = model.__class__.__name__
        maf.width_size       = DEFAULT_MAF_ARCH["width_size"]
        maf.n_layers         = DEFAULT_MAF_ARCH["n_layers"]
        maf.nn_depth         = DEFAULT_MAF_ARCH["nn_depth"]
        maf.activation       = DEFAULT_MAF_ARCH["activation"]
        maf.use_scaling      = DEFAULT_MAF_ARCH["use_scaling"] # Defaults to (mu, std) of (x, y)

        config.ndes          = [maf]#, cnf, cnf]  
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
        cnf.width_size       = DEFAULT_CNF_ARCH["width_size"]
        cnf.depth            = DEFAULT_CNF_ARCH["depth"]
        cnf.activation       = DEFAULT_CNF_ARCH["activation"]
        cnf.dropout_rate     = DEFAULT_CNF_ARCH["dropout_rate"]
        cnf.dt               = DEFAULT_CNF_ARCH["dt"]
        cnf.t1               = DEFAULT_CNF_ARCH["t1"]
        cnf.solver           = DEFAULT_CNF_ARCH["solver"]
        cnf.exact_log_prob   = DEFAULT_CNF_ARCH["exact_log_prob"]
        cnf.use_scaling      = DEFAULT_CNF_ARCH["use_scaling"] # Defaults to (mu, std) of (x, y)

        config.maf = maf = ConfigDict()
        maf.model_type       = "maf" # = model.__class__.__name__
        maf.width_size       = DEFAULT_MAF_ARCH["width_size"]
        maf.n_layers         = DEFAULT_MAF_ARCH["n_layers"]
        maf.nn_depth         = DEFAULT_MAF_ARCH["nn_depth"]
        maf.activation       = DEFAULT_MAF_ARCH["activation"]
        maf.use_scaling      = DEFAULT_MAF_ARCH["use_scaling"] # Defaults to (mu, std) of (x, y)

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
        cnf.width_size       = DEFAULT_CNF_ARCH["width_size"]
        cnf.depth            = DEFAULT_CNF_ARCH["depth"]
        cnf.activation       = DEFAULT_CNF_ARCH["activation"]
        cnf.dropout_rate     = DEFAULT_CNF_ARCH["dropout_rate"]
        cnf.dt               = DEFAULT_CNF_ARCH["dt"]
        cnf.t1               = DEFAULT_CNF_ARCH["t1"]
        cnf.solver           = DEFAULT_CNF_ARCH["solver"]
        cnf.exact_log_prob   = DEFAULT_CNF_ARCH["exact_log_prob"]
        cnf.use_scaling      = DEFAULT_CNF_ARCH["use_scaling"] # Defaults to (mu, std) of (x, y)

        config.maf = maf = ConfigDict()
        maf.model_type       = "maf" # = model.__class__.__name__
        maf.width_size       = DEFAULT_MAF_ARCH["width_size"]
        maf.n_layers         = DEFAULT_MAF_ARCH["n_layers"]
        maf.nn_depth         = DEFAULT_MAF_ARCH["nn_depth"]
        maf.activation       = DEFAULT_MAF_ARCH["activation"]
        maf.use_scaling      = DEFAULT_MAF_ARCH["use_scaling"] # Defaults to (mu, std) of (x, y)

        config.ndes          = [maf]#, cnf, cnf]  
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
        cnf.width_size       = DEFAULT_CNF_ARCH["width_size"]
        cnf.depth            = DEFAULT_CNF_ARCH["depth"]
        cnf.activation       = DEFAULT_CNF_ARCH["activation"]
        cnf.dropout_rate     = DEFAULT_CNF_ARCH["dropout_rate"]
        cnf.dt               = DEFAULT_CNF_ARCH["dt"]
        cnf.t1               = DEFAULT_CNF_ARCH["t1"]
        cnf.solver           = DEFAULT_CNF_ARCH["solver"]
        cnf.exact_log_prob   = DEFAULT_CNF_ARCH["exact_log_prob"]
        cnf.use_scaling      = DEFAULT_CNF_ARCH["use_scaling"] # Defaults to (mu, std) of (x, y)

        config.maf = maf = ConfigDict()
        maf.model_type       = "maf" # = model.__class__.__name__
        maf.width_size       = DEFAULT_MAF_ARCH["width_size"]
        maf.n_layers         = DEFAULT_MAF_ARCH["n_layers"]
        maf.nn_depth         = DEFAULT_MAF_ARCH["nn_depth"]
        maf.activation       = DEFAULT_MAF_ARCH["activation"]
        maf.use_scaling      = DEFAULT_MAF_ARCH["use_scaling"] # Defaults to (mu, std) of (x, y)

        config.ndes          = [maf]#, cnf, cnf] #maf, maf, maf]  
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