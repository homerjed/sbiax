import os
import argparse

from configs.log import setup_module_logger, get_log_level
from data.constants import get_scales 

"""
    CLI args 
"""

USE_SOBOL = True if os.environ.get("USE_SOBOL", "").lower() in ("1", "true") else False 

logger, log_figs_dir = setup_module_logger(__name__, level=get_log_level())

DEFAULT_N_LINEAR_SIMS = 32768 if USE_SOBOL else 2000


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "-l", 
        "--linearised", 
        action=argparse.BooleanOptionalAction, 
        help="Linearised model for datavector."
    )
    parser.add_argument(
        "-c", 
        "--compression", 
        # default="linear", 
        choices=["linear", "nn", "nn-lbfgs", "imnn", "ensemble-nn"], 
        type=str, 
        help="Compression with neural network or MOPED."
    )
    parser.add_argument(
        "-bt", 
        "--bulk_or_tails", 
        # default="tails", 
        choices=["bulk", "tails", "bulk_pdf"], 
        type=str, 
        help="Use cumulants from bulk or tails of PDF"
    )
    parser.add_argument(
        "-n", 
        "--n_linear_sims", 
        default=DEFAULT_N_LINEAR_SIMS, 
        type=int, 
        help="Number of linearised simulations."
    )
    parser.add_argument(
        "-p", 
        "--pre-train", 
        action=argparse.BooleanOptionalAction, 
        help="Pre-train using linearised simulations."
    )
    parser.add_argument(
        "-o", 
        "--order_idx", 
        default=[0, 1, 2], 
        nargs="+", 
        type=int, 
        help="Indices of variance, skewness and kurtosis."
    )
    parser.add_argument(
        "-r", 
        "--scales", 
        default=get_scales(), 
        nargs="+", 
        type=float, 
        help="Physical scales."
    )
    parser.add_argument(
        "-ut", 
        "--use-tqdm", 
        action=argparse.BooleanOptionalAction, 
        help="Show loading bar."
    )
    return parser


def get_cumulants_sbi_args(multi_z: bool = False) -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Run SBI experiment with cumulants of the matter PDF."
    )

    parser = add_common_args(parser)

    parser.add_argument("-s", "--seed", type=int, help="Seed for random number generation.")
    parser.add_argument("-z", "--redshift", choices=[0.0, 0.5, 1.0], type=float, help="Redshift of simulations.")
    parser.add_argument("-n_d", "--n_datavectors", type=int, help="Number of datavectors per redshift.")

    if multi_z:
        args, _ = parser.parse_known_args() 
    else:
        args = parser.parse_args()

    logger.info("CUMULANTS SBI ARGS (multi_z={}):".format(multi_z))
    for k, v in vars(args).items():
        logger.info("%-12s : %s", k, v)

    return args


def get_cumulants_multi_z_args(figure_one: bool = False) -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Run posterior sampling over multi-redshift SBI experiments with moments of the matter PDF."
    )

    parser = add_common_args(parser)

    parser.add_argument("-s", "--seed", type=int, help="Seed for random number generation.")
    parser.add_argument("-sd", "--seed_datavector", type=int, help="Seed for datavector.")
    parser.add_argument("-n_d", "--n_datavectors", type=int, help="Number of datavectors per redshift.")
    parser.add_argument("-z", "--redshifts", default=[0.0, 0.5, 1.0], nargs="+", type=float, help="Redshifts.")

    if figure_one:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    logger.info("MULTI-Z ARGS (figure_one={}):".format(figure_one))
    for k, v in vars(args).items():
        logger.info("%-12s : %s", k, v)

    return args


def get_figure_one_args():
    parser = argparse.ArgumentParser(
        description="Plot multi-redshift posteriors from datavector types (bulk, tails) together."
    )

    parser = add_common_args(parser)

    parser.add_argument("-s", "--seed", type=int, help="Seed for random number generation.")
    parser.add_argument("-sd", "--seed_datavector", type=int, help="Seed for datavector.")
    parser.add_argument("-n_d", "--n_datavectors", type=int, help="Number of datavectors per redshift.")
    parser.add_argument("-z", "--redshifts", default=[0.0, 0.5, 1.0], nargs="+", type=float, help="Redshifts.")

    args = parser.parse_args()

    logger.info("FIGURE_ONE ARGS:")
    for k, v in vars(args).items():
        logger.info("%-12s : %s", k, v)

    return args


def get_figure_two_args():
    parser = argparse.ArgumentParser(
        description="Plot measured marginal posterior widths over independent datavector and SBI parameters."
    )

    parser = add_common_args(parser)

    parser.add_argument("-n_d", "--n_datavectors", type=int, help="Number of datavectors per redshift.")
    parser.add_argument("-z", "--redshifts", default=[0.0, 0.5, 1.0], nargs="+", type=float, help="Redshifts.")

    args = parser.parse_args()

    logger.info("FIGURE_TWO ARGS:")
    for k, v in vars(args).items():
        logger.info("%-12s : %s", k, v)

    return args


def get_arch_search_args():
    # Fixed args "tuple" to not interfere with CLI parsed args

    class Args:
        n_parallel = 10
        n_processes = 10
        multiprocess = True
        n_trials = 500
        n_startup_trials = 100
        random_seeds = True
        n_repeats = 3
        n_test_sims = 20_000
        use_independent_test_set = True

    return Args()