import os
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange

from configs.args import get_cumulants_multi_z_args
from configs.configs import (
    get_base_results_dir, 
    get_multi_z_posterior_filename
)
from data.constants import (
    get_quijote_parameters, 
    get_save_and_load_dirs,
    get_target_idx,
    get_Finv_planck
)
from data.pdfs import load_multi_z_bulk_pdf_fisher_forecast

jax.clear_caches()

N_DATAVECTOR_SEEDS = int(os.environ.get("N_DATAVECTOR_SEEDS", 200))
N_REPEATED_SBI_SEEDS = int(os.environ.get("N_REPEATED_SBI_SEEDS", 10))

"""
    Same as figure_two.py except that we calculate histograms
    for repeated experiments (sbi training) and repeated posteriors 
    (for independent datavectors) for both linearised and non-linearised
    experiments.
"""

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

scale_by_fisher = False # Scale parameter constraints by bulk-PDF Fisher widths
use_consistent_binning = False # Same bins for bulk / tails posterior widths
y_axis_off = True # Turn off y-axis for histograms, "density" label

# Experiment setups for which to load posteriors (both bulk and tails)
no_frozen = True
keys = ["linearised", "pretrain", "freeze_parameters"]
exp_dicts = [
    dict(zip(keys, values)) 
    for values in list(product([True, False], repeat=len(keys)))
]

print(exp_dicts)

"""
    Figure two for global-seed repeated LINEARISED experiments
"""

print("LINEARISED EXPERIMENTS PLOT")
for exp_dict in exp_dicts:

    # Ignore this setup!
    if exp_dict["linearised"] and exp_dict["pretrain"]:
        continue

    # Ignore if not requested before
    if exp_dict["freeze_parameters"]:
        continue

    print("EXP_DICT:\n", exp_dict)

    # Arguments for given multi-z posterior (edited for experimental setup being loaded)
    multi_z_args = get_cumulants_multi_z_args()

    # Set args in multi_z_configuration (linearised, pre-train, freeze)
    for key in exp_dict:
        setattr(multi_z_args, key, exp_dict[key])

    n_p = target_idx.size if multi_z_args.freeze_parameters else alpha.size

    # Load all posteriors from multi-z, calculating widths, for bulk and tails (over all redshifts)
    posterior_widths = dict(
        bulk=np.zeros((N_DATAVECTOR_SEEDS, N_REPEATED_SBI_SEEDS, n_p)), 
        tails=np.zeros((N_DATAVECTOR_SEEDS, N_REPEATED_SBI_SEEDS, n_p))
    )
    Finvs = dict(
        bulk=dict(frozen=None, nonfrozen=None),
        tails=dict(frozen=None, nonfrozen=None)
    )
    for bulk_or_tails in ["bulk", "tails"]:

        for _global_seed in range(N_REPEATED_SBI_SEEDS):
            # Loop over datavector seeds?
            for s in trange(N_DATAVECTOR_SEEDS, desc="Posterior widths"):

                # Attempt to load posterior 
                try:
                    # NOTE: what is the seed here should be reset?
                    multi_z_args.seed = _global_seed # Seed for SBI experiment
                    multi_z_args.seed_datavector = s # Seed for datavector 
                    multi_z_args.bulk_or_tails = bulk_or_tails

                    # Load Bulk PDF Fisher matrix just once NOTE: replace this with PDFs dataset
                    if s == 0:
                        # NOTE: Scale PDF Fisher information by number of datavectors!
                        Finv_bulk_pdfs_all_z = load_multi_z_bulk_pdf_fisher_forecast(data_dir, multi_z_args)
                        Finv_bulk_pdfs_all_z = Finv_bulk_pdfs_all_z / multi_z_args.n_datavectors 

                    # Load posterior for seed and experiment
                    posterior_filename = get_multi_z_posterior_filename(multi_z_args)
                    posterior = np.load(posterior_filename)

                    widths = np.var(posterior["samples"], axis=0)

                    if scale_by_fisher:
                        widths = widths / np.diag(Finv_bulk_pdfs_all_z) - 1.

                    posterior_widths[bulk_or_tails][s, _global_seed, :] = widths

                    if exp_dict["freeze_parameters"]:
                        Finvs[bulk_or_tails]["frozen"] = posterior["Finv"]
                    else:
                        Finvs[bulk_or_tails]["nonfrozen"] = posterior["Finv"]
                except:
                    print("BAD POSTERIOR:\n\t SBI_SEED={}, DATAVECTOR_SEED={}")

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
            np.concatenate(
                [
                    posterior_widths["bulk"][:, _global_seed, :], 
                    posterior_widths["tails"][:, _global_seed, :]
                ]
            ), 
            bins=n_bins
        )
    else:
        bins = n_bins

    fig_dim = (16. / 5.) * n_p
    if landscape:
        fig, axes = plt.subplots(1, n_p, figsize=(fig_dim, 4.), sharey=True)
    else:
        fig, axes = plt.subplots(n_p, 1, figsize=(5., fig_dim), sharex=False)
    axes = np.atleast_1d(axes)

    for i in range(n_p):

        ax = axes.ravel()[i]

        for _global_seed in range(N_REPEATED_SBI_SEEDS):

            if exp_dict["linearised"]:
                tag = " (linearised)" 
            else:
                tag = ""

            _ = ax.hist(
                posterior_widths["bulk"][:, _global_seed, i], 
                bins=bins, 
                color=plotting_dict["bulk"]["color"], 
                edgecolor="none", 
                alpha=0.3, 
                label=("SBI[bulk]" + tag) if _global_seed == 0 else None, # Legend entry only for first plot
                density=True
            )
            _ = ax.hist(
                posterior_widths["bulk"][:, _global_seed, i], 
                bins=bins, 
                color='k', 
                histtype="step", 
                alpha=0.7, 
                density=True
            )

            _ = ax.hist(
                posterior_widths["tails"][:, _global_seed, i], 
                bins=bins, 
                color=plotting_dict["tails"]["color"], 
                edgecolor="none", 
                alpha=0.3, 
                label=("SBI[tails]" + tag) if _global_seed == 0 else None, 
                density=True
            )
            _ = ax.hist(
                posterior_widths["tails"][:, _global_seed, i], 
                bins=bins, 
                color='k', 
                histtype="step", 
                alpha=0.7,
                density=True
            )

        # Bulk Fisher information line
        ax.axvline(
            vertical_lines[i], 
            color="green", 
            linestyle="--", 
            linewidth=2, 
            label=r"$F^{{-1}}[{}]$ (PDF[bulk])".format(parameter_strings[i][1:-1])
        )
        ax.axvline(
            np.diag(Finvs["bulk"]["frozen"] if exp_dict["freeze_parameters"] else Finvs["bulk"]["nonfrozen"])[i], 
            color="blue", 
            linestyle="--", 
            linewidth=2, 
            label=r"$F^{{-1}}[{}]$ ($k_n$[bulk])".format(parameter_strings[i][1:-1])
        )
        ax.axvline(
            np.diag(Finvs["tails"]["frozen"] if exp_dict["freeze_parameters"] else Finvs["tails"]["nonfrozen"])[i], 
            color="red", 
            linestyle="--", 
            linewidth=2, 
            label=r"$F^{{-1}}[{}]$ ($k_n$[tails])".format(parameter_strings[i][1:-1])
        )
        # if multi_z_args.use_planck:
        #     ax.axvline(
        #         np.diag(get_Finv_planck())[i], 
        #         color="k", 
        #         linestyle="--", 
        #         linewidth=2, 
        #         label=r"$F^{{-1}}[{}]_{{Planck}}$".format(parameter_strings[i][1:-1])
        #     )


        # Set xlims for this marginalised plot only, for given parameter i 
        # if i == 0:
        #     ax.set_xlim(
        #         0.0005 * vertical_lines[i], 1.05e-4
        #     )
        # if i == 4:
        #     ax.set_xlim(
        #         0.0005 * vertical_lines[i], 4.2e-6
        #     )

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

        if y_axis_off:
            ax.set_yticks([]) # Density=True for histograms, they have arbitrary units
            ax.set_ylabel("Density")

    fig.tight_layout()

    figs_dir = os.path.join(get_base_results_dir(), "figure_two/")
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir, exist_ok=True)

    parts = [
        "frozen" if multi_z_args.freeze_parameters else "nonfrozen",
        "linearised" if multi_z_args.linearised else "nonlinearised",
        multi_z_args.compression,
        "pretrain" if multi_z_args.pre_train else "nopretrain",
        "".join(map(str, multi_z_args.order_idx)),
        # str(multi_z_args.seed)
    ]
    identifier_str = "_".join(filter(None, parts))

    filename = os.path.join(figs_dir, "figure_two_repeated_{}.pdf".format(identifier_str))

    print("Figure two saved at:\n\t", filename)

    plt.savefig(filename, bbox_inches="tight")
    plt.close() 

    """ 
        Linearised marginalised posterior widths (non-frozen only)
    """

    # Plot marginalised posterior widths for the target parameters
    if not exp_dict["freeze_parameters"]: 

        n_p = target_idx.size 

        fig_dim = (16. / 5.) * n_p

        if landscape:
            fig, axes = plt.subplots(1, n_p, figsize=(fig_dim, 4.), sharey=True)
        else:
            fig, axes = plt.subplots(n_p, 1, figsize=(5., fig_dim), sharex=False)
        axes = np.atleast_1d(axes)

        for _i, i in enumerate(target_idx):

            ax = axes.ravel()[_i]

            for _global_seed in range(N_REPEATED_SBI_SEEDS):

                if _global_seed == 0 and exp_dict["linearised"]:
                    tag = " (linearised)" 
                else:
                    tag = ""

                _ = ax.hist(
                    posterior_widths["bulk"][:, _global_seed, i], 
                    bins=bins, 
                    color=plotting_dict["bulk"]["color"], 
                    edgecolor="none", 
                    alpha=0.3, 
                    label=("SBI[bulk]" + tag) if _global_seed == 0 else None, 
                    density=True
                )
                _ = ax.hist(
                    posterior_widths["bulk"][:, _global_seed, i], 
                    bins=bins, 
                    color='k', 
                    histtype="step", 
                    alpha=0.7,
                    density=True
                )

                _ = ax.hist(
                    posterior_widths["tails"][:, _global_seed, i], 
                    bins=bins, 
                    color=plotting_dict["tails"]["color"], 
                    edgecolor="none", 
                    alpha=0.3, 
                    label=("SBI[tails]" + tag) if _global_seed == 0 else None, 
                    density=True
                )
                _ = ax.hist(
                    posterior_widths["tails"][:, _global_seed, i], 
                    bins=bins, 
                    color='k', 
                    histtype="step", 
                    alpha=0.7,
                    density=True
                )

            # Bulk Fisher information line
            ax.axvline(
                vertical_lines[i], 
                color="green", 
                linestyle="--", 
                linewidth=2, 
                label=r"$F^{{-1}}[{}]$ (PDF[bulk])".format(parameter_strings[i][1:-1])
            )
            ax.axvline(
                np.diag(Finvs["bulk"]["frozen"] if exp_dict["freeze_parameters"] else Finvs["bulk"]["nonfrozen"])[i], 
                color="blue", 
                linestyle="--", 
                linewidth=2, 
                label=r"$F^{{-1}}[{}]$ ($k_n$[bulk])".format(parameter_strings[i][1:-1])
            )
            ax.axvline(
                np.diag(Finvs["tails"]["frozen"] if exp_dict["freeze_parameters"] else Finvs["tails"]["nonfrozen"])[i], 
                color="red", 
                linestyle="--", 
                linewidth=2, 
                label=r"$F^{{-1}}[{}]$ ($k_n$[tails])".format(parameter_strings[i][1:-1])
            )
            # if multi_z_args.use_planck:
            #     ax.axvline(
            #         np.diag(get_Finv_planck())[i], 
            #         color="k", 
            #         linestyle="--", 
            #         linewidth=2, 
            #         label=r"$F^{{-1}}[{}]_{{Planck}}$".format(parameter_strings[i][1:-1])
            #     )

            # Set xlims for this marginalised plot only, for given parameter i 
            # if _i == 0:
            #     ax.set_xlim(
            #         vertical_lines[i] - 0.05 * (1.05e-4 - vertical_lines[i]), 1.05e-4
            #     )
            # if _i == 1:
            #     ax.set_xlim(
            #         vertical_lines[i] - 0.05 * (4.2e-6 - vertical_lines[i]), 4.2e-6
            #     )

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

            if y_axis_off:
                ax.set_yticks([]) # Density=True for histograms, they have arbitrary units
                ax.set_ylabel("Density")

        fig.tight_layout()

        figs_dir = os.path.join(get_base_results_dir(), "figure_two/")
        if not os.path.exists(figs_dir):
            os.makedirs(figs_dir, exist_ok=True)

        parts = [
            "frozen" if multi_z_args.freeze_parameters else "nonfrozen",
            "linearised" if multi_z_args.linearised else "nonlinearised",
            multi_z_args.compression,
            "pretrain" if multi_z_args.pre_train else "nopretrain",
            "".join(map(str, multi_z_args.order_idx)),
            # str(multi_z_args.seed),
            "marginalised"
        ]
        identifier_str = "_".join(filter(None, parts))

        filename = os.path.join(figs_dir, "figure_two_repeated_{}.pdf".format(identifier_str))

        print("Figure two (marginalised) saved at:\n\t", filename)

        plt.savefig(filename, bbox_inches="tight")
        plt.close()