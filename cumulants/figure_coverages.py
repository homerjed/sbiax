import os
import numpy as np
import matplotlib.pyplot as plt

from configs.log import setup_module_logger, get_log_level
from configs.configs import get_base_results_dir
from data.constants import (
    get_quijote_parameters, get_save_and_load_dirs
)

logger, log_figs_dir = setup_module_logger(__name__, level=get_log_level())

N_DATAVECTOR_SEEDS = int(os.environ.get("N_DATAVECTOR_SEEDS", 200))
N_REPEATED_SBI_SEEDS = int(os.environ.get("N_REPEATED_SBI_SEEDS", 10))

"""
    Plot coverages
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

"""
    Coverages
"""

figs_dir = os.path.join(get_base_results_dir(), "figure_two/")
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir, exist_ok=True)
 
posterior_coverages_linear = np.load(
    os.path.join(figs_dir, "posterior_coverages_linear.npz")
)
posterior_coverages_nonlinear = np.load(
    os.path.join(figs_dir, "posterior_coverages_nonlinear.npz")
)
posterior_coverages = dict(
    linear=posterior_coverages_linear,
    nonlinear=posterior_coverages_nonlinear,
)

for _global_seed in range(N_REPEATED_SBI_SEEDS):

    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(6, 10), dpi=200)

    hist_kwargs = dict(bins=20, range=[0, 1], density=True)

    for ax, _exp in zip(axs.ravel(), ["linear", "nonlinear"]):
        # SBI bulk
        ax.hist(
            posterior_coverages[_exp]["sbi/bulk"][:, _global_seed, 0], 
            **hist_kwargs,
            alpha=0.7, 
            color="blue", 
            label="SBI [bulk]"
        )
        ax.hist(
            posterior_coverages[_exp]["sbi/bulk"][:, _global_seed, 1], 
            **hist_kwargs,
            alpha=0.7, 
            color="blue", 
            # label="SBI [bulk]"
        )

        # SBI tails
        ax.hist(
            posterior_coverages[_exp]["sbi/tails"][:, _global_seed, 0], 
            **hist_kwargs,
            alpha=0.7, 
            color="red", 
            label="SBI [tails]"
        )
        ax.hist(
            posterior_coverages[_exp]["sbi/tails"][:, _global_seed, 1], 
            **hist_kwargs,
            alpha=0.7, 
            color="red", 
            # label="SBI [tails]"
        )

        # MCMC bulk
        ax.hist(
            mcmc_posterior_coverages[_exp]["mcmc/bulk"][:, _global_seed, 0], 
            **hist_kwargs,
            alpha=1.0, 
            facecolor='gray', 
            edgecolor='red', 
            histtype="step", 
            label="MCMC [bulk]"
        )
        ax.hist(
            mcmc_posterior_coverages[_exp]["mcmc/bulk"][:, _global_seed, 1], 
            **hist_kwargs,
            alpha=1.0, 
            facecolor='gray', 
            edgecolor='red', 
            histtype="step", 
            # label="MCMC [bulk]"
        )

        # MCMC tails 
        ax.hist(
            mcmc_posterior_coverages[_exp]["mcmc/tails"][:, _global_seed, 0], 
            **hist_kwargs,
            alpha=1.0, 
            facecolor='gray', 
            edgecolor='blue', 
            histtype="step", 
            label="MCMC [tails]"
        )
        ax.hist(
            mcmc_posterior_coverages[_exp]["mcmc/tails"][:, _global_seed, 1], 
            **hist_kwargs,
            alpha=1.0, 
            facecolor='gray', 
            edgecolor='blue', 
            histtype="step", 
            # label="MCMC [tails]"
        )

    for ax in axs:
        # Add dashed vertical lines at 0.68 and 0.95, each drawn twice
        ax.axvline(0.68, linestyle="--", linewidth=2, color="k")
        ax.axvline(0.95, linestyle="--", linewidth=2, color="k")

        ax.set_xlim(-0.05, 1.05)

        ax.set_xlabel(r"$F_{\omega}$")
        ax.set_ylabel("Density")

        ax.legend(loc="upper left", fontsize=8, frameon=False)

    plt.savefig(
        os.path.join(figs_dir, "posterior_coverages.pdf"), bbox_inches="tight"
    )
    plt.close()
