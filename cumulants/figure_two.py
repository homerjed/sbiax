import argparse
from typing import Tuple
from collections import namedtuple
import os
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key
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
    _, _, _, alpha, lower, upper, parameter_strings, *_
) = get_quijote_parameters()

n_seeds = 2

# Load all posteriors from multi-z, calculating widths
posterior_widths = dict(
    bulk=np.zeros((n_seeds, alpha.size)), 
    tails=np.zeros((n_seeds, alpha.size))
)
for bulk_or_tails in ["bulk", "tails"]:
    for s in trange(n_seeds, desc="Posterior widths"):

        # Arguments for given multi-z posterior
        args = get_cumulants_multi_z_args()

        args.seed = s
        args.bulk_or_tails = bulk_or_tails

        # Load Bulk PDF Fisher matrix just once
        if s == 0:
            try:
                Finv_bulk_pdfs_all_z = np.load(
                    os.path.join(
                        data_dir, 
                        "Finv_bulk_pdfs_all_z_{}.npy".format(
                            # "".join(map(str, args.order_idx)), NOTE: no cumulants associated with bulk pdf?!
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
                            # "".join(map(str, args.order_idx)),
                            "f" if args.freeze_parameters else "nf"
                        )
                    ),
                    Finv_bulk_pdfs_all_z
                )

        # Multi-z inference concerning the bulk or bulk + tails
        if args.bulk_or_tails == "tails":
            ensembles_config = ensembles_cumulants_config
        if args.bulk_or_tails == "bulk" or args.bulk_or_tails == "bulk_pdf":
            ensembles_config = ensembles_bulk_cumulants_config

        config = ensembles_config(
            seed=args.seed, # Defaults if run without argparse args
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
            posterior_save_dir, "posterior_{}.npz".format(args.seed)
        )
        posterior = np.load(posterior_filename)

        posterior_widths[bulk_or_tails][s] = np.var(posterior["samples"], axis=0)

# Plot histogram of posterior widths across all seeds for all multi-z posteriors
landscape = False

vertical_lines = jnp.diag(Finv_bulk_pdfs_all_z)

plotting_dict = dict(
    bulk=dict(color="b"),
    tails=dict(color="r")
)

n_bins = 10

if landscape:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
else:
    fig, axes = plt.subplots(4, 1, figsize=(5, 16), sharex=True)
axes = np.atleast_1d(axes)

for i, (ax, posterior_vars_bulk, posterior_vars_tails, vline) in enumerate(
    zip(
        axes, 
        posterior_widths["bulk"],
        posterior_widths["tails"],
        vertical_lines  
    )
):
    _ = ax.hist(
        posterior_vars_bulk[i], bins=n_bins, color=plotting_dict["bulk"]["color"], edgecolor='none', alpha=0.7, label='Bulk'
    )
    _ = ax.hist(
        posterior_vars_bulk[i], bins=n_bins, color='k', histtype="step", alpha=0.7
    )

    _ = ax.hist(
        posterior_vars_tails[i], bins=n_bins, color=plotting_dict["tails"]["color"], edgecolor='none', alpha=0.7, label='Tails'
    )
    _ = ax.hist(
        posterior_vars_tails[i], bins=n_bins, color='k', histtype="step", alpha=0.7
    )

    # Bulk Fisher information line
    ax.axvline(vertical_lines[i], color='black', linestyle='--', linewidth=2)

    ax.set_xlabel(r"$\sigma^2[{}]$".format(parameter_strings[i][1:-1])) # Trim '$' from parameter strings

    # ax.set_title(f"Histogram Set {i}")
    ax.legend()

fig.tight_layout()

figs_dir = os.path.join(get_base_results_dir(), "figure_two/")
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir, exist_ok=True)

filename = os.path.join(figs_dir, "figure_two.pdf")

print("Figure two saved at:\n\t", filename)

plt.savefig(filename, bbox_inches="tight")
plt.close()