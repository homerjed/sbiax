import os
from typing import Optional
import argparse
import yaml
import jax.numpy as jnp
import jax.random as jr 
from equinox import Module
from jaxtyping import PRNGKeyArray, jaxtyped
from beartype import beartype as typechecker
from ml_collections import ConfigDict

from data.constants import get_base_results_dir, get_base_posteriors_dir
from data.common import Dataset
from data.cumulants import CumulantsDataset
from data.pdfs import BulkCumulantsDataset, TailsCumulantsDataset, BulkPDFsDataset
from sbiax.ndes import CNF, MAF, Scaler

typecheck = jaxtyped(typechecker=typechecker)

DatasetClass = BulkCumulantsDataset | TailsCumulantsDataset | BulkPDFsDataset | CumulantsDataset


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def save_config(config: ConfigDict, filepath: str) -> None:
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


def dump_args_and_config(args: argparse.Namespace, config: ConfigDict, results_dir: str) -> None:
    """ Save command line arguments and config together """
    with open(os.path.join(results_dir, "config.yml"), "w") as f:
        yaml.dump({">ARGS": ""}, f, default_flow_style=False)
        yaml.dump(vars(args), f, default_flow_style=False)
        yaml.dump({">CONFIG": ""}, f, default_flow_style=False)
        yaml.dump(config.to_dict(), f, default_flow_style=False)


def get_config_subdir(
    args: argparse.Namespace, 
    *, 
    arch_search: bool = False, 
    multi_z: bool = False
) -> str:
    parts = [
        "arch_search" if arch_search else None,
        "frozen" if args.freeze_parameters else "nonfrozen",
        args.bulk_or_tails,
        "linearised" if args.linearised else "nonlinearised",
        args.compression,
        "pretrain" if args.pre_train else "nopretrain",
        "z={}".format(
            "".join(map(str, args.redshifts)) if multi_z else args.redshift
        ),
        "".join(map(str, args.order_idx)) if multi_z else None, # NOTE: Multi-z posteriors marked by cumulants in datavector, not redshift!
        # "".join(map(str, args.scales)) if multi_z else None, # NOTE: Multi-z posteriors marked by cumulants in datavector, not redshift!
        str(args.seed),
        "multi_z" if multi_z else None
    ]
    return "/".join(filter(None, parts)) + "/"


def get_results_dir(
    config: ConfigDict,
    args: argparse.Namespace, 
    *, 
    arch_search: bool = False
) -> str:
    """ General results directory format for individual SBI experiments """

    results_dir = os.path.join(
        get_base_results_dir(), 
        get_config_subdir(args, arch_search=arch_search)
    )

    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    dump_args_and_config(args, config, results_dir=results_dir) # Save conifg and args in run dir

    print("RESULTS_DIR:\n", results_dir)

    return results_dir


def get_posteriors_dir(
    args: argparse.Namespace, 
    *, 
    arch_search: bool = False
) -> str:
    """ General results directory format for posteriors from individual SBI experiments """

    posteriors_dir = os.path.join(
        get_base_posteriors_dir(), 
        get_config_subdir(args, arch_search=arch_search)
    )

    if not os.path.exists(posteriors_dir):
        os.makedirs(posteriors_dir, exist_ok=True)

    print("POSTERIORS_DIR:\n", posteriors_dir)
    
    return posteriors_dir


def get_multi_z_posterior_dir(args: argparse.Namespace) -> str:
    """ General results directory format for posteriors from bulk or tails multi-redshift SBI sampling """

    multi_z_dir = os.path.join(
        get_base_posteriors_dir(), 
        get_config_subdir(args, multi_z=True) 
    )

    print("Multi-z posterior dir:\n", multi_z_dir)

    return multi_z_dir


def get_multi_z_posterior_filename(args: argparse.Namespace) -> str:
    # Save posterior, Fisher and summary

    posterior_save_dir = get_multi_z_posterior_dir(args)

    if not os.path.exists(posterior_save_dir):
        os.makedirs(posterior_save_dir, exist_ok=True)

    print("Multi-z posterior save dir:\n\t", posterior_save_dir)
    
    # Posterior depends on the seed of the SBI experiment and the seed used to generate the datavector
    posterior_filename = os.path.join(
        posterior_save_dir, 
        "multi_z_posterior_{}{}.npz".format( # NOTE: was just 'posterior_...' before
            args.seed, 
            ("_" + str(args.seed_datavector)) if args.seed_datavector is not None else ""
        ) 
    )
    return posterior_filename


"""
    Configs
"""


@typecheck
def get_ndes_from_config(
    config: ConfigDict, 
    dataset: DatasetClass,
    event_dim: int, 
    context_dim: Optional[int] = None, 
    *, 
    use_scalers: bool = False,
    key: PRNGKeyArray 
) -> list[Module]:

    fisher_mu_std = (dataset.data.alpha, jnp.sqrt(jnp.diag(dataset.data.Finv)))
    # X = jax.vmap(dataset.compression_fn)(dataset.data, dataset.parameters)

    # Pack the single scaler for each NDE
    scaler = Scaler(
        # X, dataset.parameters,
        x_mu_std=fisher_mu_std,
        q_mu_std=fisher_mu_std,
        use_scaling=use_scalers
    )

    keys = jr.split(key, len(config.ndes))

    ndes = []
    for nde, key in zip(config.ndes, keys):

        assert nde.model_type in ["maf", "cnf"], (
            "Invalid NDE model type (={})".format(nde.model_type)
        )

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

    assert len(config.ndes) == len(ndes)

    return ndes