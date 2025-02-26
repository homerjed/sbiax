import os
# from shutil import rmtree
from typing import Literal
import argparse
import yaml
import jax.random as jr 
from equinox import Module
from jaxtyping import Key, jaxtyped
from beartype import beartype as typechecker
from ml_collections import ConfigDict
from sbiax.ndes import CNF, MAF

RESULTS_DIR = "/project/ls-gruen/users/jed.homer/sbiaxpdf/results/"

POSTERIORS_DIR = "/project/ls-gruen/users/jed.homer/sbiaxpdf/results/posteriors/"

typecheck = jaxtyped(typechecker=typechecker)


def default(v, d):
    return v if v is not None else d


def get_base_results_dir():
    return RESULTS_DIR


def save_config(config: ConfigDict, filepath: str):
    with open(filepath, 'w') as f:
        yaml.dump(config.to_dict(), f)


def load_config(filepath: str) -> ConfigDict:
    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)
    return ConfigDict(config_dict)


def make_dirs(results_dir: str):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    print("RESULTS_DIR:\n", results_dir)


def get_results_dir(config: ConfigDict, args) -> str:

    if "cumulants" in config.dataset_name:
        results_dir = "{}{}{}{}{}{}{}{}".format( # Import this from a constants file
            RESULTS_DIR,
            "cumulants/",
            (config.sbi_type + "/") if config.sbi_type else "" ,
            "linear/" if config.linearised else "",
            (config.compression + "/") if config.compression else "",
            "pretrain/" if config.pre_train else "",
            (config.exp_name + "/") if config.exp_name else "",
            (str(config.seed) + "/") if (config.seed is not None) else ""
        )
    if ("cumulants" in config.dataset_name) and ("reduced" in config.dataset_name):
        results_dir = "{}{}{}{}{}{}{}{}".format( # Import this from a constants file
            RESULTS_DIR,
            "reduced_cumulants/",
            (config.sbi_type + "/") if config.sbi_type else "" ,
            "linear/" if config.linearised else "",
            (config.compression + "/") if config.compression else "",
            "pretrain/" if config.pre_train else "",
            (config.exp_name + "/") if config.exp_name else "",
            (str(config.seed) + "/") if (config.seed is not None) else ""
        )

    make_dirs(results_dir)

    with open(os.path.join(results_dir, "config.yml"), "w") as f:
        yaml.dump({"args": ""}, f, default_flow_style=False)
        yaml.dump(vars(args), f, default_flow_style=False)
        yaml.dump({"config": ""}, f, default_flow_style=False)
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    return results_dir


def get_posteriors_dir(config: ConfigDict) -> str:

    if "cumulants" in config.dataset_name:
        results_dir = "{}{}{}{}{}{}{}{}".format( # Import this from a constants file
            POSTERIORS_DIR,
            "cumulants/",
            (config.sbi_type + "/") if config.sbi_type else "" ,
            "linear/" if config.linearised else "",
            (config.compression + "/") if config.compression else "",
            "pretrain/" if config.pre_train else "",
            (config.exp_name + "/") if config.exp_name else "",
            (str(config.seed) + "/") if (config.seed is not None) else ""
        )
    if ("cumulants" in config.dataset_name) and ("reduced" in config.dataset_name):
        results_dir = "{}{}{}{}{}{}{}{}".format( # Import this from a constants file
            RESULTS_DIR,
            "reduced_cumulants/",
            (config.sbi_type + "/") if config.sbi_type else "" ,
            "linear/" if config.linearised else "",
            (config.compression + "/") if config.compression else "",
            "pretrain/" if config.pre_train else "",
            (config.exp_name + "/") if config.exp_name else "",
            (str(config.seed) + "/") if (config.seed is not None) else ""
        )

    make_dirs(results_dir)

    return results_dir


def get_multi_z_posterior_dir(config: ConfigDict, sbi_type: str = None) -> str:
    posterior_save_dir = "{}cumulants/multi_z/{}/{}{}".format(
        POSTERIORS_DIR, #RESULTS_DIR,
        config.sbi_type if (hasattr(config, sbi_type) and (config.sbi_type is not None)) else sbi_type,
        "linear/" if config.linearised else "",
        "pretrain/" if config.pre_train else ""
    )
    return posterior_save_dir


def get_cumulants_sbi_args():
    parser = argparse.ArgumentParser(
        description="Run SBI experiment with cumulants of the matter PDF."
    )
    parser.add_argument(
        "-s", 
        "--seed", 
        type=int, 
        help="Seed for random number generation.", 
        default=0
    )
    parser.add_argument(
        "-l",
        "--linearised", 
        default=True,
        action=argparse.BooleanOptionalAction, 
        help="Linearised model for datavector."
    )
    parser.add_argument(
        "-r",
        "--reduced-cumulants", 
        default=True,
        action=argparse.BooleanOptionalAction, 
        help="Reduced cumulants or not."
    )
    parser.add_argument(
        "-c",
        "--compression", 
        default="linear",
        choices=["linear", "nn"],
        type=str,
        help="Compression with neural network or MOPED."
    )
    parser.add_argument(
        "-n",
        "--n_linear_sims", 
        default=100_000,
        type=int,
        help="Number of linearised simulations (used for pre-training if non-linear simulations and requested)."
    )
    parser.add_argument(
        "-p",
        "--pre-train", 
        default=False,
        action=argparse.BooleanOptionalAction, 
        help="Pre-train (only) when using non-linearised model for datavector. Pre-train on linearised simulations."
    )
    parser.add_argument(
        "-z",
        "--redshift", 
        default=0.0,
        choices=[0.0, 0.5, 1.0],
        type=float,
        help="Redshift of simulations."
    )
    parser.add_argument(
        "-o", 
        "--order_idx",
        default=[0, 1, 2],
        nargs="+", 
        type=int,
        help="Indices of variance, skewness and kurtosis sample cumulants."
    )
    parser.add_argument(
        "-t",
        "--sbi_type", 
        default="nle",
        choices=["nle", "npe"],
        type=str,
        help="Method of SBI: neural likelihood (NLE) or posterior (NPE)."
    )
    parser.add_argument(
        "-u",
        "--use-tqdm", 
        default=True,
        action=argparse.BooleanOptionalAction, 
        help="Show loading bar. Useful to turn off during architecture search."
    )
    parser.add_argument(
        "-v",
        "--verbose", 
        default=False,
        action=argparse.BooleanOptionalAction, 
        help="Say what's going on."
    )
    args = parser.parse_args()
    return args


def get_cumulants_multi_z_args():
    parser = argparse.ArgumentParser(
        description="Run SBI experiment with moments of the matter PDF."
    )
    parser.add_argument(
        "-s", 
        "--seed", 
        type=int, 
        help="Seed for random number generation.", 
        default=0
    )
    parser.add_argument(
        "-l",
        "--linearised", 
        default=True,
        action=argparse.BooleanOptionalAction, 
        help="Linearised model for datavector."
    )
    parser.add_argument(
        "-r",
        "--reduced-cumulants", 
        default=True,
        action=argparse.BooleanOptionalAction, 
        help="Reduced cumulants or not."
    )
    parser.add_argument(
        "-c",
        "--compression", 
        default="linear",
        choices=["linear", "nn"],
        type=str,
        help="Compression with neural network or MOPED."
    )
    parser.add_argument(
        "-n",
        "--n_linear_sims", 
        type=int,
        action=argparse.BooleanOptionalAction, 
        help="Number of linearised simulations (used for pre-training if non-linear simulations and requested)."
    )
    parser.add_argument(
        "-p",
        "--pre-train", 
        action=argparse.BooleanOptionalAction, 
        help="Pre-train (only) when using non-linearised model for datavector. Pre-train on linearised simulations."
    )
    parser.add_argument(
        "-z",
        "--redshift", 
        choices=[0.0, 0.5, 1.0],
        type=float,
        help="Redshift of simulations."
    )
    parser.add_argument(
        "-o", 
        "--order_idx",
        default=[0, 1, 2],
        nargs="+", 
        type=int,
        help="Indices of variance, skewness and kurtosis sample cumulants."
    )
    parser.add_argument(
        "-t",
        "--sbi_type", 
        choices=["nle", "npe"],
        type=str,
        help="Method of SBI: neural likelihood (NLE) or posterior (NPE)."
    )
    parser.add_argument(
        "-v",
        "--verbose", 
        default=False,
        action=argparse.BooleanOptionalAction, 
        help="Say what's going on."
    )
    args = parser.parse_args()
    return args


def get_ndes_from_config(
    config: ConfigDict, 
    event_dim: int, 
    context_dim: int = None, 
    scalers: list[Module] = None, 
    *, 
    key: Key 
) -> list[Module]:
    # Scaler for each nde...
    keys = jr.split(key, len(config.ndes))

    if not isinstance(scalers, list):
        scalers = [scalers]

    ndes = []
    for nde, scaler, key in zip(config.ndes, scalers, keys):

        if nde.model_type == "maf":
            nde_arch = MAF
        if nde.model_type == "cnf":
            nde_arch = CNF

        # Required to remove / add some arguments to specify NDEs
        nde_dict = dict(
            event_dim=event_dim, 
            context_dim=context_dim if context_dim is not None else event_dim, 
            key=key,
            scaler=scaler,
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
    compression: Literal["linear", "nn"] = "linear",
    order_idx: list[int] = [0, 1, 2],
    # nonlinearised: bool = True, 
    n_linear_sims: int = 100_000,
    pre_train: bool = False
) -> ConfigDict:

    config = ConfigDict()

    config.seed               = default(seed, 0) # For argparse script running without args!

    # Data
    # config.bulk               = False
    config.dataset_name       = "cumulants" if not reduced_cumulants else "reduced cumulants"
    # config.cumulants          = False
    config.redshift           = default(redshift, 0.0)
    config.scales             = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
    config.order_idx          = default(order_idx, [0, 1, 2]) # Maximum index is 2
    config.compression        = default(compression, "linear")
    config.linearised         = default(linearised, True)
    config.covariance_epsilon = None #1e-6
    config.reduced_cumulants  = reduced_cumulants
    # config.nonlinearised = default(nonlinearised, False)
    config.pre_train          = default(pre_train, False) and (not linearised)
    config.n_linear_sims      = default(n_linear_sims, 10_000) # This is for pre-train or linearised simulations 

    # assert not (not config.linearised and config.pre_train)
    print(config.linearised, config.pre_train)

    # SBI
    config.sbi_type      = sbi_type
    config.exp_name      = "z={}_m={}".format(config.redshift, "".join(map(str, config.order_idx)))

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
        cnf.solver           = "Euler" #dfx.Euler() 
        cnf.exact_log_prob   = True
        cnf.use_scaling      = False # Defaults to (mu, std) of (x, y)

        config.maf = maf = ConfigDict()
        maf.model_type       = "maf" # = model.__class__.__name__
        maf.width_size       = 32
        maf.n_layers         = 8
        maf.nn_depth         = 2
        maf.activation       = "tanh"
        maf.use_scaling      = False # Defaults to (mu, std) of (x, y)

        config.ndes          = [maf]
        config.n_ndes        = len(config.ndes)

        # Posterior sampling
        config.use_ema       = True # Use it and sample with it
        config.n_steps       = 100
        config.n_walkers     = 1000
        config.burn          = int(0.1 * config.n_steps)

        # Optimisation hyperparameters (same for all NDEs...)
        config.start_step    = 0
        config.n_epochs      = 10_000
        config.n_batch       = 50 
        config.patience      = 20
        config.lr            = 1e-3
        config.opt           = "adamw" 
        config.opt_kwargs    = {}
    else:
        # NDEs
        config.cnf = cnf = ConfigDict()
        cnf.model_type       = "cnf"
        cnf.width_size       = 32
        cnf.depth            = 2
        cnf.activation       = "tanh"
        cnf.dropout_rate     = 0.
        cnf.dt               = 0.1
        cnf.t1               = 1.
        cnf.solver           = "Euler" #dfx.Euler() 
        cnf.exact_log_prob   = True
        cnf.use_scaling      = False # Defaults to (mu, std) of (x, y)

        config.maf = maf = ConfigDict()
        maf.model_type       = "maf" # = model.__class__.__name__
        maf.width_size       = 32
        maf.n_layers         = 8
        maf.nn_depth         = 2
        maf.activation       = "tanh"
        maf.use_scaling      = False # Defaults to (mu, std) of (x, y)

        config.ndes          = [maf]  
        config.n_ndes        = len(config.ndes)

        # Posterior sampling
        config.use_ema       = True # Use it and sample with it
        config.n_steps       = 100
        config.n_walkers     = 1000
        config.burn          = int(0.1 * config.n_steps)

        # Optimisation hyperparameters (same for all NDEs...)
        config.start_step    = 0
        config.n_epochs      = 10_000
        config.n_batch       = 50 
        config.patience      = 40
        config.lr            = 1e-3
        config.opt           = "adamw" 
        config.opt_kwargs    = {}

    return config


def ensembles_cumulants_config(
    seed: int = 0, 
    exp_name: str = "",
    sbi_type: str = "nle", 
    linearised: bool = True, 
    compression: Literal["linear", "nn"] = "linear",
    reduced_cumulants: bool = True,
    order_idx: list[int] = [0, 1, 2],
    pre_train: bool = False
) -> ConfigDict:

    config = ConfigDict()

    config.seed              = seed # Note: seed that ensemble configs run at also!
    config.sbi_type          = sbi_type
    config.exp_name_format   = "z={}_m={}".format(config.redshift, "".join(map(str, config.order_idx)))

    config.compression       = compression

    # Data
    config.bulk              = False
    config.dataset_name      = "cumulants" 
    config.redshifts         = [0.0, 0.5, 1.0] # Redshifts to combine
    config.scales            = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
    config.order_idx         = order_idx
    config.compression       = compression
    config.reduced_cumulants = reduced_cumulants
    config.linearised        = linearised 
    config.pre_train         = pre_train and (not linearised) # Load linearised or pre-trained models

    return config