import argparse
from collections import namedtuple
from typing import NamedTuple, Type

ArgsTuple: Type[tuple] = None

"""
    CLI args 
"""


def args_to_namedtuple(args: argparse.Namespace) -> tuple:
    ArgsTuple = namedtuple("ArgsTuple", vars(args).keys()) # Create namedtuple type
    args = ArgsTuple(**vars(args)) # Convert Namespace to namedtuple
    return args


def get_cumulants_sbi_args(using_notebook: bool = False) -> argparse.Namespace | ArgsTuple:
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
        default=False,
        action=argparse.BooleanOptionalAction, 
        help="Reduced cumulants or not."
    )
    parser.add_argument(
        "-c",
        "--compression", 
        default="linear",
        choices=["linear", "nn", "nn-lbfgs"],
        type=str,
        help="Compression with neural network or MOPED."
    )
    parser.add_argument(
        "-bt",
        "--bulk_or_tails", 
        default="tails",
        choices=["bulk", "tails", "bulk_pdf"],
        type=str,
        help="Use cumulants from bulk or tails of PDF"
    )
    parser.add_argument(
        "-n",
        "--n_linear_sims", 
        default=10_000,
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
        "-f",
        "--freeze-parameters", 
        default=False,
        action=argparse.BooleanOptionalAction, 
        help="Freeze parameters not in [Om, s8] to their fixed values, in hypercube simulations."
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

    if using_notebook:
        args = args_to_namedtuple(args)

    return args


def get_cumulants_multi_z_args(using_notebook: bool = False) -> argparse.Namespace | ArgsTuple:
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
        default=False,
        action=argparse.BooleanOptionalAction, 
        help="Reduced cumulants or not."
    )
    parser.add_argument(
        "-c",
        "--compression", 
        default="linear",
        choices=["linear", "nn", "nn-lbfgs"],
        type=str,
        help="Compression with neural network or MOPED."
    )
    parser.add_argument(
        "-bt",
        "--bulk_or_tails", 
        default="tails",
        choices=["bulk", "tails", "bulk_pdf"], # Choose bulk cumulants/tails or bulk pdf
        type=str,
        help="Use cumulants from bulk or tails of PDF, or bulk of the PDF."
    )
    parser.add_argument(
        "-n",
        "--n_linear_sims", 
        type=int,
        default=10_000,
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
        "--redshifts",
        default=[0.0, 0.5, 1.0],
        nargs="+", 
        type=float,
        help="Redshifts."
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
        "-f",
        "--freeze-parameters", 
        default=False,
        action=argparse.BooleanOptionalAction, 
        help="Freeze parameters not in [Om, s8] to their fixed values, in hypercube simulations."
    )
    parser.add_argument(
        "-t",
        "--sbi_type", 
        choices=["nle", "npe"],
        default="nle",
        type=str,
        help="Method of SBI: neural likelihood (NLE) or posterior (NPE)."
    )
    parser.add_argument(
        "-n_d",
        "--n_datavectors", 
        type=int,
        default=2,
        help="Number of independent datavectors to measure at each redshift." # NOTE: possibly make this depend on redshift, a list of ints
    )
    parser.add_argument(
        "-n_p",
        "--n_posteriors_sample", 
        type=int,
        default=1,
        help="Number of posteriors to sample (using different measurements for each)." 
    )
    parser.add_argument(
        "-v",
        "--verbose", 
        default=False,
        action=argparse.BooleanOptionalAction, 
        help="Say what's going on."
    )

    if using_notebook:
        args = parser.parse_args([])
        args = args_to_namedtuple(args)
    else:
        args = parser.parse_args()

    return args


def get_arch_search_args(using_notebook: bool = False):
    # parser = argparse.ArgumentParser(
    #     description="Run architecture and hyperparameters search for cumulants SBI."
    # )
    # parser.add_argument(
    #     "-s", 
    #     "--seed", 
    #     type=int, 
    #     help="Seed for random number generation.", 
    #     default=0
    # )
    # parser.add_argument(
    #     "-l",
    #     "--linearised", 
    #     default=True,
    #     action=argparse.BooleanOptionalAction, 
    #     help="Linearised model for datavector."
    # )
    # parser.add_argument(
    #     "-r",
    #     "--reduced-cumulants", 
    #     default=True,
    #     action=argparse.BooleanOptionalAction, 
    #     help="Reduced cumulants or not."
    # )
    # parser.add_argument(
    #     "-c",
    #     "--compression", 
    #     default="linear",
    #     choices=["linear", "nn", "nn-lbfgs"],
    #     type=str,
    #     help="Compression with neural network or MOPED."
    # )
    # parser.add_argument(
    #     "-n",
    #     "--n_linear_sims", 
    #     type=int,
    #     default=10_000,
    #     help="Number of linearised simulations (used for pre-training if non-linear simulations and requested)."
    # )
    # parser.add_argument(
    #     "-p",
    #     "--pre-train", 
    #     default=False,
    #     action=argparse.BooleanOptionalAction, 
    #     help="Pre-train (only) when using non-linearised model for datavector. Pre-train on linearised simulations."
    # )
    # parser.add_argument(
    #     "-z", 
    #     "--redshifts",
    #     default=[0.0, 0.5, 1.0],
    #     nargs="+", 
    #     type=float,
    #     help="Redshifts."
    # )
    # parser.add_argument(
    #     "-o", 
    #     "--order_idx",
    #     default=[0, 1, 2],
    #     nargs="+", 
    #     type=int,
    #     help="Indices of variance, skewness and kurtosis sample cumulants."
    # )
    # parser.add_argument(
    #     "-t",
    #     "--sbi_type", 
    #     choices=["nle", "npe"],
    #     default="nle",
    #     type=str,
    #     help="Method of SBI: neural likelihood (NLE) or posterior (NPE)."
    # )
    # parser.add_argument(
    #     "-n_t",
    #     "--n_trials", 
    #     type=int,
    #     default=100,
    #     help="" 
    # )
    # parser.add_argument(
    #     "-n_st",
    #     "--n_startup_trials", 
    #     type=int,
    #     default=10,
    #     help="" 
    # )
    # parser.add_argument(
    #     "-m",
    #     "--multiprocess", 
    #     default=True,
    #     action=argparse.BooleanOptionalAction, 
    #     help="Reduced cumulants or not."
    # )
    # parser.add_argument(
    #     "-n_pro",
    #     "--n_processes", 
    #     type=int,
    #     default=10,
    #     help="" 
    # )
    # parser.add_argument(
    #     "-n_par",
    #     "--n_parallel", 
    #     type=int,
    #     default=10,
    #     help="" 
    # )
    # parser.add_argument(
    #     "-v",
    #     "--verbose", 
    #     default=False,
    #     action=argparse.BooleanOptionalAction, 
    #     help="Say what's going on."
    # )

    # if using_notebook:
    #     args = parser.parse_args([])
    #     args = args_to_namedtuple(args)
    # else:
    #     args = parser.parse_args()

    class Args:
        n_parallel = 10
        n_processes = 10
        multiprocess = True
        n_trials = 1000
        n_startup_trials = 100
        random_seeds = True

    args = Args()

    return args