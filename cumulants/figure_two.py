import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange

from configs.log import setup_module_logger, get_log_level
from configs.args import get_cumulants_multi_z_args, get_figure_two_args
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

logger, log_figs_dir = setup_module_logger(__name__, level=get_log_level())

jax.clear_caches()

N_DATAVECTOR_SEEDS = int(os.environ.get("N_DATAVECTOR_SEEDS", 200))
N_REPEATED_SBI_SEEDS = int(os.environ.get("N_REPEATED_SBI_SEEDS", 10))

"""
    Same as figure_two.py except that we calculate histograms
    for repeated experiments (sbi training) and repeated posteriors 
    (for independent datavectors) for both linearised and non-linearised
    experiments.

    -> figure_two.py: plotting histograms for one exp_dict instead of 
    all of them together
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

# Plotting stuff
scale_by_fisher = False # Scale parameter constraints by bulk-PDF Fisher widths
use_consistent_binning = False # Same bins for bulk / tails posterior widths
y_axis_off = False # Turn off y-axis for histograms, "density" label

"""
    Figure two for global-seed repeated experiments
"""

figure_two_args = get_figure_two_args()

# Experiment setup for which to load posteriors (both bulk and tails)
exp_dict = dict(
    linearised=figure_two_args.linearised,
    pretrain=figure_two_args.pre_train,
    freeze_parameters=figure_two_args.freeze_parameters
)

# Arguments for given multi-z posterior (edited for experimental setup being loaded)
multi_z_args = get_cumulants_multi_z_args(figure_one=True)

# Set args in multi_z_configuration (linearised, pre-train, freeze)
for key in exp_dict:
    setattr(multi_z_args, key, exp_dict[key])

print("EXP_DICT:\n", exp_dict)
logger.info("EXP DICT: {}".format(exp_dict))

n_p = target_idx.size if multi_z_args.freeze_parameters else alpha.size

print("MULTI-Z ARGS:{}".format(vars(multi_z_args)))
logger.info("MULTI-Z ARGS:{}".format(vars(multi_z_args)))

print("FIGURE TWO ARGS:{}".format(vars(figure_two_args)))
logger.info("FIGURE TWO ARGS:{}".format(vars(figure_two_args)))


# Load Bulk PDF Fisher matrix just once NOTE: replace this with PDFs dataset NOTE: Scale PDF Fisher information by number of datavectors!
Finv_bulk_pdfs_all_z = load_multi_z_bulk_pdf_fisher_forecast(data_dir, multi_z_args)
Finv_bulk_pdfs_all_z = Finv_bulk_pdfs_all_z / multi_z_args.n_datavectors 

"""
    Get posterior widths; marginalise-index these for marginals histogram plot
"""

# Load all posteriors from multi-z, calculating widths, for bulk and tails (over all redshifts)
posterior_widths = dict(
    bulk=np.zeros((N_DATAVECTOR_SEEDS, N_REPEATED_SBI_SEEDS, n_p)), 
    tails=np.zeros((N_DATAVECTOR_SEEDS, N_REPEATED_SBI_SEEDS, n_p))
)
Finvs = dict(bulk=None, tails=None)
for bulk_or_tails in ["bulk", "tails"]:

    for _global_seed in range(N_REPEATED_SBI_SEEDS):
        # Loop over datavector seeds?
        for s in trange(
            N_DATAVECTOR_SEEDS, 
            colour="red" if bulk_or_tails == "tails" else "blue",
            desc="Posterior widths (SBI seed={})".format(_global_seed)
        ):
            # Attempt to load posterior 
            try:
                # NOTE: what is the seed here should be reset?
                multi_z_args.seed = _global_seed # Seed for SBI experiment
                multi_z_args.seed_datavector = s # Seed for datavector 
                multi_z_args.bulk_or_tails = bulk_or_tails
                multi_z_args.redshifts = figure_two_args.redshifts

                # Load posterior for seed and experiment
                posterior_filename = get_multi_z_posterior_filename(multi_z_args)
                print("POSTERIOR_FILENAME:", posterior_filename)

                posterior = np.load(posterior_filename)
                    
                widths = np.var(posterior["samples"], axis=0) # Shape (n_samples, parameters)

                print(widths.shape)

                # print(
                #     "SBI seed: {}\ndatavector: {}\nvar (sigma8): {}".format(
                #         _global_seed, bulk_or_tails, widths[-1]
                #     )
                # )

                if scale_by_fisher:
                    widths = widths / np.diag(Finv_bulk_pdfs_all_z) - 1.

                posterior_widths[bulk_or_tails][s, _global_seed, :] = widths

                # Grab multi-z Fisher forecast, scaled by n_datavectors
                Finvs[bulk_or_tails] = posterior["Finv"] # This may be multi-z when it shouldn't be...

            except Exception as e:
                print(
                    "BAD POSTERIOR:\n\tEXCEPTION: {}\n\t SBI_SEED={}, DATAVECTOR_SEED={} \n\t FILENAME={}".format(
                        e, _global_seed, s, posterior_filename
                    )
                )

for _global_seed in range(N_REPEATED_SBI_SEEDS):
    print("SEED", _global_seed)
    print("MEAN VARIANCE SIGMA_8 BULK:", np.sqrt(posterior_widths["bulk"][:, _global_seed, 4].mean(axis=0)))
    print("MEAN VARIANCE SIGMA_8 TAILS:", np.sqrt(posterior_widths["tails"][:, _global_seed, 4].mean(axis=0)))
    # print("MEAN FISHER VARIANCE SIGMA_8 BULK:", np.sqrt(np.diag(Finvs["bulk"]))[4])
    # print("MEAN FISHER VARIANCE SIGMA_8 TAILS:", np.sqrt(np.diag(Finvs["tails"]))[4])
 
"""
    Plotting
"""

# Plot histogram of posterior widths across all seeds for all multi-z posteriors
landscape = False

vertical_lines = np.diag(Finv_bulk_pdfs_all_z) # Variances (widths) for Bulk PDF Gaussian posterior

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
    fig, axes = plt.subplots(1, n_p, figsize=(fig_dim, 4.), sharey=False)
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

        # Don't plot bad runs
        if (
            jnp.all(posterior_widths["bulk"][:, _global_seed, i] == 0.)
            or
            jnp.all(posterior_widths["tails"][:, _global_seed, i] == 0.)
        ): 
            continue

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
        np.diag(Finv_bulk_pdfs_all_z)[i], 
        color="green", 
        linestyle=":", 
        linewidth=2, 
        label=r"$F^{{-1}}[{}]$ (PDF[bulk])".format(parameter_strings[i][1:-1])
    )
    ax.axvline(
        np.diag(Finvs["bulk"])[i], 
        color="blue", 
        linestyle="--", 
        linewidth=2, 
        label=r"$F^{{-1}}[{}]$ ($k_n$[bulk])".format(parameter_strings[i][1:-1])
    )
    ax.axvline(
        np.diag(Finvs["tails"])[i], 
        color="red", 
        linestyle="--", 
        linewidth=2, 
        label=r"$F^{{-1}}[{}]$ ($k_n$[tails])".format(parameter_strings[i][1:-1])
    )

    if multi_z_args.use_planck:
        ax.axvline(
            np.diag(get_Finv_planck())[i], 
            color="k", 
            linestyle="--", 
            linewidth=2, 
            label=r"$F^{{-1}}[{}]_{{Planck}}$".format(parameter_strings[i][1:-1])
        )


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
    "".join(map(str, multi_z_args.scales)),
    # str(multi_z_args.seed)
]
identifier_str = "_".join(filter(None, parts))

filename = os.path.join(figs_dir, "figure_two_repeated_{}.pdf".format(identifier_str))

print("Figure two saved at:\n\t", filename)

plt.savefig(filename, bbox_inches="tight")
plt.close() 

"""
    Plotting (marginalised, non-frozen only)
"""

# Plot marginalised posterior widths for the target parameters
if not exp_dict["freeze_parameters"]: 

    n_p = target_idx.size 

    fig_dim = (16. / 5.) * n_p

    if landscape:
        fig, axes = plt.subplots(1, n_p, figsize=(fig_dim, 4.), sharey=False)
    else:
        fig, axes = plt.subplots(n_p, 1, figsize=(5., fig_dim), sharex=False)
    axes = np.atleast_1d(axes)

    for _i, i in enumerate(target_idx):
        print("TARGET PARAMETER i: ", parameter_strings[i], i)

        ax = axes.ravel()[_i]

        for _global_seed in range(N_REPEATED_SBI_SEEDS):

            if _global_seed == 0 and exp_dict["linearised"]:
                tag = " (linearised)" 
            else:
                tag = ""

            _bulk_widths = posterior_widths["bulk"][:, _global_seed, i]
            _tails_widths = posterior_widths["tails"][:, _global_seed, i]

            # Don't plot bad runs
            if (
                jnp.all(_bulk_widths == 0.)
                or
                jnp.all(_tails_widths == 0.)
            ): 
                continue

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

        print(
            "bulk", parameter_strings[i], np.diag(Finvs["bulk"])[i], posterior_widths["bulk"][:, _global_seed, i].mean(),
            "tails", parameter_strings[i], np.diag(Finvs["tails"])[i], posterior_widths["tails"][:, _global_seed, i].mean(),
            "pdf", parameter_strings[i], np.diag(Finv_bulk_pdfs_all_z)[i]
        )

        # Bulk PDFs Fisher information line
        ax.axvline(
            np.diag(Finv_bulk_pdfs_all_z)[i], 
            color="green", 
            linestyle=":", 
            linewidth=2, 
            label=r"$F^{{-1}}[{}]$ (PDF[bulk])".format(parameter_strings[i][1:-1])
        )
        # Bulk Fisher information line
        ax.axvline(
            np.diag(Finvs["bulk"])[i], 
            color="blue", 
            linestyle="--", 
            linewidth=2, 
            label=r"$F^{{-1}}[{}]$ ($k_n$[bulk])".format(parameter_strings[i][1:-1])
        )
        # Tails Fisher information line
        ax.axvline(
            np.diag(Finvs["tails"])[i], 
            color="red", 
            linestyle="--", 
            linewidth=2, 
            label=r"$F^{{-1}}[{}]$ ($k_n$[tails])".format(parameter_strings[i][1:-1])
        )

        if multi_z_args.use_planck:
            ax.axvline(
                np.diag(get_Finv_planck())[i], 
                color="k", 
                linestyle="--", 
                linewidth=2, 
                label=r"$F^{{-1}}[{}]_{{Planck}}$".format(parameter_strings[i][1:-1])
            )

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
        "".join(map(str, multi_z_args.scales)),
        # str(multi_z_args.seed),
        "marginalised"
    ]
    identifier_str = "_".join(filter(None, parts))

    filename = os.path.join(figs_dir, "figure_two_repeated_{}.pdf".format(identifier_str))

    print("Figure two (marginalised) saved at:\n\t", filename)

    plt.savefig(filename, bbox_inches="tight")
    plt.close()