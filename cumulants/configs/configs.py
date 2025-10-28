import os
from typing import Optional, Union, Type
import argparse
import yaml
import jax.numpy as jnp
import jax.random as jr 
from equinox import Module
from jaxtyping import PRNGKeyArray, Float, Array, jaxtyped
from beartype import beartype as typechecker
from ml_collections import ConfigDict

from .log import setup_module_logger, get_log_level
from data.constants import get_base_results_dir, get_base_posteriors_dir, get_scales, ALPHA
from data.cumulants import CumulantsDataset
from sbiax.ndes import CNF, MAF, GMM, Ensemble

import os
TYPECHECK = True if os.environ.get("TYPECHECK", "").lower() in ("1", "true") else False
if TYPECHECK:
    typecheck = jaxtyped(typechecker=typechecker)
else:
    typecheck = lambda x: x


logger, log_figs_dir = setup_module_logger(__name__, level=get_log_level())

USE_SOBOL = True if os.environ.get("USE_SOBOL", "True").lower() in ("1", "true") else False
NON_GAUSSIAN_TEST = True if os.environ.get("NON_GAUSSIAN_TEST", "").lower() in ("1", "true") else False

if USE_SOBOL:
    from data.get_sobol_cumulants import (
        SobolBulkCumulantsDataset, SobolTailsCumulantsDataset, SobolBulkPDFsDataset
    )
    DatasetClass: Type = Union[ 
        SobolBulkCumulantsDataset, 
        SobolTailsCumulantsDataset, 
        SobolBulkPDFsDataset
    ]
else:
    from data.pdfs import (
        BulkCumulantsDataset, TailsCumulantsDataset, BulkPDFsDataset
    )
    DatasetClass: Type = Union[ 
        BulkCumulantsDataset, 
        TailsCumulantsDataset, 
        BulkPDFsDataset, 
        CumulantsDataset,
    ]

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
    # NOTE: Multi-z posteriors marked by cumulants in datavector, not redshift!

    # Scales marker
    if args.scales == get_scales():
        R_str = "R_all" 
    else:
        R_str = "R_{}".format("".join(map(str, args.scales)))

    # Datavector cumulant marker
    m_str = "m_{}".format("".join(map(str, args.order_idx)))

    # Seed str (make sure not same seed for each)
    s_str = "s={}".format(str(args.seed))

    # Redshift marker (NOTE: multi-z over all z, so no redshift str)
    if multi_z:
        z_str = "z={}".format("".join(map(str, args.redshifts)))
    else:
        z_str = "z={}".format(str(args.redshift))

    parts = [
        "sobol" if USE_SOBOL else None,
        "arch_search" if arch_search else None,
        "NON_GAUSSIAN_TEST" if NON_GAUSSIAN_TEST else None,
        args.bulk_or_tails,
        "linearised" if args.linearised else "nonlinearised",
        args.compression,
        "pretrain" if args.pre_train else "nopretrain",
        z_str,
        m_str, 
        R_str,
        s_str, # SBI seed
        "multi_z" if multi_z else None
    ]

    config_subdir = "/".join(filter(None, parts)) + "/"

    return config_subdir


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

    print("RESULTS_DIR:\n\t", results_dir)

    logger.info("RESULTS DIR: {}".format(results_dir))

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

    print("POSTERIORS DIR:\n\t", posteriors_dir)

    logger.info("POSTERIORS DIR: {}".format(posteriors_dir))
    
    return posteriors_dir


def get_multi_z_posterior_dir(args: argparse.Namespace) -> str:
    """ General results directory format for posteriors from bulk or tails multi-redshift SBI sampling """

    multi_z_dir = os.path.join(
        get_base_posteriors_dir(), 
        get_config_subdir(args, multi_z=True) 
    )

    logger.info("MULTI-Z POSTERIOR DIR: {}".format(multi_z_dir))

    return multi_z_dir


def get_multi_z_posterior_filename(
    args: argparse.Namespace, 
    mcmc: bool = False, 
    blackjax: bool = True
) -> str:
    # Save posterior, Fisher and summary

    posterior_save_dir = get_multi_z_posterior_dir(args)

    if not os.path.exists(posterior_save_dir):
        os.makedirs(posterior_save_dir, exist_ok=True)

    # Posterior depends on the seed of the SBI experiment and the seed used to generate the datavector
    posterior_filename = os.path.join(
        posterior_save_dir, 
        "multi_z_posterior_{}{}{}{}.npz".format( # NOTE: was just 'posterior_...' before
            args.seed, 
            ("_" + str(args.seed_datavector)) if args.seed_datavector is not None else "",
            "_MCMC" if mcmc else "",
            "_blackjax" if blackjax else ""
        ) 
    )

    logger.info("MULTI-Z POSTERIOR FILENAME: {}".format(posterior_filename))

    return posterior_filename


"""
    Configs
"""


@typecheck
def get_ndes_from_config(
    config: ConfigDict, 
    *, 
    event_dim: Optional[int] = None, 
    context_dim: Optional[int] = None, 
    key: PRNGKeyArray 
) -> Ensemble:

    # Default is compressed data => parameter-set shaped
    if event_dim is None:
        event_dim = ALPHA.size

    keys = jr.split(key, len(config.ndes))

    ndes = []
    for nde, key in zip(config.ndes, keys):

        logger.info("Using NDE of type '{}'".format(nde.model_type))

        assert nde.model_type in ["maf", "cnf"], (
            "Invalid NDE model type (={})".format(nde.model_type)
        )

        if nde.model_type == "maf":
            nde_arch = MAF
        if nde.model_type == "cnf":
            nde_arch = CNF
        if nde.model_type == "gmm":
            nde_arch = GMM

        context_dim = context_dim if exists(context_dim) else event_dim

        # Required to remove / add some arguments to specify NDEs
        nde_dict = dict(
            key=key,
            event_dim=event_dim, 
            context_dim=context_dim,
            **dict(nde)
        )

        logger.info("NDE DICT: {}".format(nde_dict))

        nde_dict.pop("model_type")

        ndes.append(nde_arch(**nde_dict))

    assert len(config.ndes) == len(ndes)

    ensemble = Ensemble(ndes)

    return ensemble