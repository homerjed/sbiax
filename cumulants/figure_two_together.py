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
from data.common import get_prior
from data.constants import (
    ALPHA,
    LOWER,
    UPPER,
    get_quijote_parameters, 
    get_save_and_load_dirs,
    get_target_idx
)

USE_SOBOL = True if os.environ.get("USE_SOBOL", "").lower() in ("1", "true") else False 

if USE_SOBOL:
    from data.get_sobol_cumulants import load_multi_z_bulk_pdf_fisher_forecast
else:
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

BLUE_HEX, RED_HEX = '#3b82f6', '#ef4444' 


def calculate_coverages(posterior):
    # Calculate one and two sigma coverages for 
    # SBI posteriors. Assume all samples drawn at 
    # alpha point in parameter space given measurements.
    # https://github.com/homerjed/dodelsonsbi/blob/398c9ffab6e79147f3bf4d8bc5eba1d57a2b4c83/analysis/get_posterior_coverages.py#L79
    samples_log_prob = posterior["samples_log_prob"]
    alpha_log_prob = posterior["alpha_log_prob"]

    coverage = jnp.mean(alpha_log_prob > samples_log_prob)

    truth_inside_68 = coverage > 1. - 0.68
    truth_inside_95 = coverage > 1. - 0.95

    return truth_inside_68, truth_inside_95


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

# Arguments for given multi-z posterior (edited for experimental setup being loaded)
multi_z_args = get_cumulants_multi_z_args(figure_one=True)

prior = get_prior()

n_p = alpha.size

print("MULTI-Z ARGS:{}".format(vars(multi_z_args)))
logger.info("MULTI-Z ARGS:{}".format(vars(multi_z_args)))

print("FIGURE TWO ARGS:{}".format(vars(figure_two_args)))
logger.info("FIGURE TWO ARGS:{}".format(vars(figure_two_args)))

# Dummy seed, reset later
if multi_z_args.seed is None:
    multi_z_args.seed = 0

# Load Bulk PDF Fisher matrix just once NOTE: replace this with PDFs dataset NOTE: Scale PDF Fisher information by number of datavectors!
Finv_bulk_pdfs_all_z = load_multi_z_bulk_pdf_fisher_forecast(data_dir, multi_z_args)
Finv_bulk_pdfs_all_z = Finv_bulk_pdfs_all_z / multi_z_args.n_datavectors 

"""
    Get posterior widths; marginalise-index these for marginals histogram plot
"""

figs_dir = os.path.join(get_base_results_dir(), "figure_two/")
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir, exist_ok=True)

# LOAD COVERAGES, WIDTHS (SBI AND MCMC)

"""
    Plotting
"""

# Plot histogram of posterior widths across all seeds for all multi-z posteriors
landscape = False

# Clip multi-z forecast to prior boundary
#vertical_lines = np.std(fisher_samples, axis=0)

print("\nCLIPPING CHAINS\n")

n_fisher_samples = 800_000

def cut_samples(samples, lower, upper):
    return samples[np.all((samples >= lower) & (samples <= upper), axis=1)]


def load_grouped_npz(path):
    groups = {}
    with np.load(path) as z:
        for name in z.files:
            grp, key = name.split("/", 1)
            groups.setdefault(grp, {})[key] = z[name]
    return groups


plotting_dict = dict(
    bulk=dict(color=BLUE_HEX),
    tails=dict(color=RED_HEX)
)

n_bins = 10
bins = n_bins

fig_dim = (16. / 5.) * n_p

# Two axes_widths for linear / non-linear
if landscape:
    fig_widths, axes_widths = plt.subplots(2, n_p, figsize=(fig_dim, 4. * 2.), sharey=False)
else:
    fig_widths, axes_widths = plt.subplots(n_p, 2, figsize=(5. * 2., fig_dim), sharex=False)

fig_coverages, axes_coverages = plt.subplots(ncols=1, nrows=2, figsize=(6, 10), dpi=200)

for l, lin_or_nonlin in enumerate(["linearised", "nonlinearised"]):

    # Load widths, coverages
    _path = os.path.join(figs_dir, "posterior_statistics_{}.npz".format(lin_or_nonlin))

    loaded = load_grouped_npz(_path)    
    
    # These are linear or non-linear depending on the iteration
    posterior_widths = loaded["posterior_widths"]
    mcmc_posterior_widths = loaded["mcmc_posterior_widths"]
    posterior_coverages = loaded["posterior_coverages"]
    mcmc_posterior_coverages = loaded["mcmc_posterior_coverages"]
    Finvs = loaded["Finvs"]

    # Clipped
    fisher_widths = dict()
    for name in ["bulk_pdf", "bulk", "tails"]:

        print("FINV SHAPE", name, Finvs[name].shape)

        fisher_samples = np.random.multivariate_normal(
            ALPHA, Finvs[name], (n_fisher_samples,) 
        ) 

        fisher_samples = cut_samples(fisher_samples, LOWER, UPPER)

        # Marginal variances
        fisher_widths[name] = np.var(fisher_samples, axis=0) # fisher_samples

    # Calculate prior widths for histogram plots
    fisher_widths["prior"] = prior.variance()


    # ~~~~~~~~~ Coverages
    ax = axes_coverages[l]
    for _global_seed in range(N_REPEATED_SBI_SEEDS):

        hist_kwargs = dict(bins=20, range=[0, 1], density=True)

        for ax in axes_coverages.ravel():
            # SBI bulk
            ax.hist(
                posterior_coverages["bulk"][:, _global_seed, 0], 
                **hist_kwargs,
                alpha=0.7, 
                color="blue", 
                label="SBI [bulk]"
            )
            ax.hist(
                posterior_coverages["bulk"][:, _global_seed, 1], 
                **hist_kwargs,
                alpha=0.7, 
                color="blue"
            )

            # SBI tails
            ax.hist(
                posterior_coverages["tails"][:, _global_seed, 0], 
                **hist_kwargs,
                alpha=0.7, 
                color="red", 
                label="SBI [tails]"
            )
            ax.hist(
                posterior_coverages["tails"][:, _global_seed, 1], 
                **hist_kwargs,
                alpha=0.7, 
                color="red"
            )

            # MCMC bulk
            ax.hist(
                mcmc_posterior_coverages["bulk"][:, _global_seed, 0], 
                **hist_kwargs,
                alpha=1.0, 
                facecolor='gray', 
                edgecolor='red', 
                histtype="step", 
                label="MCMC [bulk]"
            )
            ax.hist(
                mcmc_posterior_coverages["mcmc/bulk"][:, _global_seed, 1], 
                **hist_kwargs,
                alpha=1.0, 
                facecolor='gray', 
                edgecolor='red', 
                histtype="step"
            )

            # MCMC tails 
            ax.hist(
                mcmc_posterior_coverages["mcmc/tails"][:, _global_seed, 0], 
                **hist_kwargs,
                alpha=1.0, 
                facecolor='gray', 
                edgecolor='blue', 
                histtype="step", 
                label="MCMC [tails]"
            )
            ax.hist(
                mcmc_posterior_coverages["mcmc/tails"][:, _global_seed, 1], 
                **hist_kwargs,
                alpha=1.0, 
                facecolor='gray', 
                edgecolor='blue', 
                histtype="step"
            )

        for ax in axes_coverages:
            # Add dashed vertical lines at 0.68 and 0.95, each drawn twice
            ax.axvline(0.68, linestyle="--", linewidth=2, color="k")
            ax.axvline(0.95, linestyle="--", linewidth=2, color="k")

            ax.set_xlim(-0.05, 1.05)

            ax.set_xlabel(r"$F_{\omega}$")
            ax.set_ylabel("Density")

            ax.legend(loc="upper left", fontsize=8, frameon=False)
    # ~~~~~~~~~



    _axs = axes_widths[l] if landscape else axes_widths[:, l]

    for i in range(n_p):

        ax = _axs.ravel()[i]

        for _global_seed in range(N_REPEATED_SBI_SEEDS):

            tag = "({})".format("quijote" if lin_or_nonlin == "nonlinearised" else "linearised")

            # Don't plot bad runs
            if (
                jnp.all(posterior_widths["bulk"][:, _global_seed, i] == 0.)
                or
                jnp.all(posterior_widths["tails"][:, _global_seed, i] == 0.)
            ): 
                continue

            """
                SBI
            """
            _ = ax.hist(
                posterior_widths["bulk"][:, _global_seed, i], # Parameter i, SBI run `_global_seed`, all posteriors
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

            """
                MCMCs 
            """
            _ = ax.hist(
                mcmc_posterior_widths["bulk"][:, _global_seed, i], # Parameter i, SBI run `_global_seed`, all posteriors
                bins=bins, 
                color="lightgray", 
                edgecolor="none", 
                alpha=0.5, 
                density=True
            )
            _ = ax.hist(
                mcmc_posterior_widths["bulk"][:, _global_seed, i], 
                bins=bins, 
                color=plotting_dict["bulk"]["color"], 
                histtype="step", 
                label=("MCMC[bulk]" + tag) if _global_seed == 0 else None, # Legend entry only for first plot
                alpha=0.7, 
                density=True
            )

            _ = ax.hist(
                mcmc_posterior_widths["tails"][:, _global_seed, i], 
                bins=bins, 
                color="lightgray", 
                edgecolor="none", 
                alpha=0.5, 
                density=True
            )
            _ = ax.hist(
                mcmc_posterior_widths["tails"][:, _global_seed, i], 
                bins=bins, 
                color=plotting_dict["tails"]["color"], 
                histtype="step", 
                label=("MCMC[tails]" + tag) if _global_seed == 0 else None, 
                alpha=0.7,
                density=True
            )

        # Bulk Fisher information line
        ax.axvline(
            fisher_widths["bulk_pdf"][i],
            color="green", 
            linestyle=":", 
            linewidth=2, 
            label=r"$F^{{-1}}[{}]$ (PDF[bulk])".format(parameter_strings[i][1:-1])
        )
        ax.axvline(
            fisher_widths["bulk"][i],
            color=BLUE_HEX, 
            linestyle="--", 
            linewidth=2, 
            label=r"$F^{{-1}}[{}]$ ($k_n$[bulk])".format(parameter_strings[i][1:-1])
        )
        ax.axvline(
            fisher_widths["tails"][i],
            color=RED_HEX, 
            linestyle="--", 
            linewidth=2, 
            label=r"$F^{{-1}}[{}]$ ($k_n$[tails])".format(parameter_strings[i][1:-1])
        )
        ax.axvline(
            fisher_widths["prior"][i],
            color="darkgray", 
            linestyle="--", 
            linewidth=2, 
            # label=r"$\sigma^2_{{\text{{prior}}}}[{}]$".format(parameter_strings[i][1:-1])
            label=r"$\sigma^2_{\text{{prior}}}[{}]$".format(parameter_strings[i][1:-1])
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

        if y_axis_off:
            ax.set_yticks([]) # Density=True for histograms, they have arbitrary units
            ax.set_ylabel("Density")

fig_widths.tight_layout()

figs_dir = os.path.join(get_base_results_dir(), "figure_two/")
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir, exist_ok=True)

parts = [
    multi_z_args.compression,
    "pretrain" if multi_z_args.pre_train else "nopretrain",
    "".join(map(str, multi_z_args.order_idx)),
    "".join(map(str, multi_z_args.scales))
]
identifier_str = "_".join(filter(None, parts))

filename = os.path.join(figs_dir, "figure_two_repeated_together_{}.pdf".format(identifier_str))

print("Figure two saved at:\n\t", filename)

fig_widths.savefig(filename, bbox_inches="tight")

filename = os.path.join(figs_dir, "figure_coverages_repeated_together_{}.pdf".format(identifier_str))

print("Figure coverages saved at:\n\t", filename)

fig_coverages.savefig(filename, bbox_inches="tight")

plt.close() 