import os
import argparse
from typing import Tuple
from collections import namedtuple
from itertools import product

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

import numpy as np
from ml_collections import ConfigDict
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer, Chain, Truth
from tqdm.auto import trange
from tensorflow_probability.substrates.jax.distributions import Distribution

from sbiax.ndes import CNF, MAF, Scaler
from sbiax.inference import nuts_sample
from sbiax.utils import make_df, marker

from configs import (
    cumulants_config, 
    bulk_cumulants_config, 
    get_posteriors_dir
)
from configs.args import (
    get_cumulants_sbi_args, get_cumulants_multi_z_args
)
from configs.configs import (
    get_base_results_dir, 
    get_results_dir, 
    get_multi_z_posterior_dir, 
    get_ndes_from_config
)
from configs.ensembles_configs import (
    ensembles_cumulants_config, ensembles_bulk_cumulants_config 
)
from data.constants import (
    get_quijote_parameters, 
    get_base_posteriors_dir,
    get_save_and_load_dirs,
    get_target_idx
)
from data.cumulants import (
    Dataset, 
    get_data, 
    get_linear_compressor, 
    get_datavector, 
    get_prior, 
    get_parameter_strings
)
from data.pdfs import get_multi_z_bulk_pdf_fisher_forecast
from cumulants_ensemble import Ensemble, MultiEnsemble
from affine import affine_sample

jax.clear_caches()

# General constants
data_dir, _, _ = get_save_and_load_dirs()

(
    _, 
    _, 
    _, 
    alpha, 
    lower, 
    upper, 
    parameter_strings, 
    *_
) = get_quijote_parameters()

target_idx = get_target_idx()

global_seed = 0 # Set to None to load different ensembles for each seed (NOTE: iterate over this if testing linearised!)
n_seeds = 50 # NOTE: decide if seeds are for experiments or datavectors 

scale_by_fisher = True # Scale parameter constraints by bulk-PDF Fisher widths
use_consistent_binning = False # Same bins for bulk / tails posterior widths

keys = ["linearised", "freeze_parameters", "pretrain"]
exp_dicts = [
    dict(zip(keys, values)) 
    for values in list(product([True, False], repeat=len(keys)))
]

# for exp_dict in map(dict, itertools.product(["linearised"])):
for exp_dict in exp_dicts:

    print("EXP_DICT:\n", exp_dict)

    # Ignore this setup!
    if exp_dict["linearised"] and exp_dict["pretrain"]:
        continue

    # Arguments for given multi-z posterior
    args = get_cumulants_multi_z_args()

    # Set args in multi_z_configuration
    for key in exp_dict:
        setattr(args, key, exp_dict[key])

    n_p = target_idx.size if args.freeze_parameters else alpha.size

    # Load all posteriors from multi-z, calculating widths, for bulk and tails (over all redshifts)
    posterior_widths = dict(
        bulk=np.zeros((n_seeds, n_p)) if not exp_dict["linearised"] else np.zeros((n_seeds, 10, n_p)), 
        tails=np.zeros((n_seeds, n_p)) if not exp_dict["linearised"] else np.zeros((n_seeds, 10, n_p)), # Extra seed
    )
    for bulk_or_tails in ["bulk", "tails"]:

        # Loop over datavector seeds?
        for s in trange(n_seeds, desc="Posterior widths"):

            args.seed = global_seed if global_seed is not None else s # Fixed ensemble, diffferent datavectors
            args.bulk_or_tails = bulk_or_tails

            # Load Bulk PDF Fisher matrix just once
            if s == 0:
                try:
                    Finv_bulk_pdfs_all_z = np.load(
                        os.path.join(
                            data_dir, 
                            "Finv_bulk_pdfs_all_z_{}.npy".format(
                                "f" if args.freeze_parameters else "nf"
                            )
                        )
                    )
                except:
                    Finv_bulk_pdfs_all_z = get_multi_z_bulk_pdf_fisher_forecast(args)

                    np.save(
                        os.path.join(
                            data_dir, 
                            "Finv_bulk_pdfs_all_z_{}.npy".format(
                                "f" if args.freeze_parameters else "nf"
                            )
                        ),
                        Finv_bulk_pdfs_all_z
                    )

            def get_widths(run_seed, datavector_seed):

                # Multi-z inference concerning the bulk or bulk + tails
                if args.bulk_or_tails == "tails":
                    ensembles_config = ensembles_cumulants_config
                if args.bulk_or_tails == "bulk" or args.bulk_or_tails == "bulk_pdf":
                    ensembles_config = ensembles_bulk_cumulants_config

                config = ensembles_config(
                    seed=run_seed, # Defaults if run without argparse args
                    sbi_type=args.sbi_type, 
                    linearised=args.linearised,
                    n_linear_sims=args.n_linear_sims,
                    compression=args.compression,
                    reduced_cumulants=args.reduced_cumulants,
                    redshifts=args.redshifts,
                    order_idx=args.order_idx,
                    pre_train=args.pre_train,
                    freeze_parameters=args.freeze_parameters
                )

                # Load posterior for seed and experiment
                posterior_save_dir = get_multi_z_posterior_dir(config, args)
                posterior_filename = os.path.join(
                    posterior_save_dir, 
                    "multi_z_posterior_{}{}.npz".format(
                        run_seed, # NOTE: when running for many linearised experiments, 
                        ("_" + str(datavector_seed)) if global_seed is not None else "" # Check this loads posterior from independent datavector with fixed NDE training seed
                    ) 
                )

                posterior = np.load(posterior_filename)

                widths = np.var(posterior["samples"], axis=0)

                if scale_by_fisher:
                    widths = widths / np.diag(Finv_bulk_pdfs_all_z) - 1.

                return widths

            if exp_dict["linearised"]:
                # Multiple run-seeds for linearised experiments
                for _run_seed in range(10):
                    widths = get_widths(run_seed=_run_seed, datavector_seed=s)
                    posterior_widths[bulk_or_tails][_run_seed, s] = widths
            else:
                # One fixed run-seed for non-linearised experiments
                widths = get_widths(run_seed=args.seed, datavector_seed=s)
                posterior_widths[bulk_or_tails][s] = widths

    # Plot histogram of posterior widths across all seeds for all multi-z posteriors
    landscape = False

    vertical_lines = jnp.diag(Finv_bulk_pdfs_all_z) # Variances (widths) for Bulk PDF Gaussian posterior

    plotting_dict = dict(
        bulk=dict(color="b"),
        tails=dict(color="r")
    )

    n_bins = 10

    if use_consistent_binning:
        bins = np.histogram_bin_edges(
            np.concatenate([posterior_widths["bulk"], posterior_widths["tails"]]), bins=n_bins
        )
    else:
        bins = n_bins

    fig_dim = (16. / 5.) * n_p
    if landscape:
        fig, axes = plt.subplots(
            1, n_p, figsize=(fig_dim, 4.), sharey=True
        )
    else:
        fig, axes = plt.subplots(
            n_p, 1, figsize=(5., fig_dim), sharex=False
        )
    axes = np.atleast_1d(axes)

    for i in range(n_p):

        ax = axes.ravel()[i]

        if args.linearised:
            for _run_seed in range(10):
                _ = ax.hist(
                    posterior_widths["bulk"][:, _run_seed, i], 
                    bins=bins, 
                    color=plotting_dict["bulk"]["color"], 
                    edgecolor="none", 
                    alpha=0.4, 
                    label="SBI[bulk]" if _run_seed == 0 else ""
                )
                _ = ax.hist(
                    posterior_widths["bulk"][:, _run_seed, i], 
                    bins=bins, 
                    color='k', 
                    histtype="step", 
                    alpha=0.4
                )

                _ = ax.hist(
                    posterior_widths["tails"][:, _run_seed, i], 
                    bins=bins, 
                    color=plotting_dict["tails"]["color"], 
                    edgecolor="none", 
                    alpha=0.4, 
                    label="SBI[tails]" if _run_seed == 0 else ""
                )
                _ = ax.hist(
                    posterior_widths["tails"][:, _run_seed, i], 
                    bins=bins, 
                    color='k', 
                    histtype="step", 
                    alpha=0.4
                )
        else:
            _ = ax.hist(
                posterior_widths["bulk"][:, i], 
                bins=bins, 
                color=plotting_dict["bulk"]["color"], 
                edgecolor="none", 
                alpha=0.7, 
                label="SBI[bulk]"
            )
            _ = ax.hist(
                posterior_widths["bulk"][:, i], 
                bins=bins, 
                color='k', 
                histtype="step", 
                alpha=0.7
            )

            _ = ax.hist(
                posterior_widths["tails"][:, i], 
                bins=bins, 
                color=plotting_dict["tails"]["color"], 
                edgecolor="none", 
                alpha=0.7, 
                label="SBI[tails]"
            )
            _ = ax.hist(
                posterior_widths["tails"][:, i], 
                bins=bins, 
                color='k', 
                histtype="step", 
                alpha=0.7
            )

        # Bulk Fisher information line
        ax.axvline(
            vertical_lines[i], 
            color="green", 
            linestyle="--", 
            linewidth=2, 
            label=r"$F^{{-1}}[{}]$ (PDF[bulk])".format(parameter_strings[i][1:-1])
        )

        if scale_by_fisher:
            fisher_width_str = r"/F^{{-1}}_{{PDF[bulk]}}[{}] - 1".format(
                parameter_strings[i][1:-1] # Trim '$' from parameter strings
            )
        else:
            fisher_width_str = ""

        ax.set_xlabel(
            r"$\sigma^2[{}]{}$".format(
                parameter_strings[i][1:-1], 
                fisher_width_str # Trim '$' from parameter strings
            )
        ) 

        ax.legend(frameon=False)

    fig.tight_layout()

    figs_dir = os.path.join(get_base_results_dir(), "figure_two/")
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir, exist_ok=True)

    parts = [
        "frozen" if config.freeze_parameters else "nonfrozen",
        # "reduced_cumulants" if config.reduced_cumulants else "cumulants",
        # config.sbi_type,
        "linearised" if config.linearised else "nonlinearised",
        config.compression,
        "pretrain" if config.pre_train else "nopretrain",
        # config.exp_name if include_exp and config.exp_name else None, # NOTE: This is ignored for multi_z!
        "".join(map(str, args.order_idx)),
        str(config.seed)
    ]
    identifier_str = "_".join(filter(None, parts))

    filename = os.path.join(figs_dir, "figure_two_{}.pdf".format(identifier_str))

    print("Figure two saved at:\n\t", filename)

    plt.savefig(filename, bbox_inches="tight")
    plt.close()