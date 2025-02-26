import argparse
import os
import time
import numpy as np 
import matplotlib.pyplot as plt
from chainconsumer import Chain, ChainConsumer, Truth

from configs import get_multi_z_posterior_dir 
from configs.moments import moments_config, ensembles_moments_config
from constants import get_alpha_and_parameter_strings
from sbiax.utils import make_df


def default(v, d):
    return v if v is not None else d


"""
    First plot: 
    multi-z plot of Finv_z, linearised and SBI together.
    - Setup so seeds are the same for each experiment 

    Compare multi-redshift posteriors from different SBI experiments
    - Fishe
    - linearised data
    - non-linearised data
"""

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
    "-n",
    "--n_linear_sims", 
    type=int,
    action=argparse.BooleanOptionalAction, 
    help="Number of linearised simulations (used for pre-training if non-linear simulations and requested).",
    default=10_000 #!
)
parser.add_argument(
    "-t",
    "--sbi_type", 
    choices=["nle", "npe"],
    type=str,
    help="Method of SBI: neural likelihood (NLE) or posterior (NPE).",
    default="nle"
)
args = parser.parse_args()

# Run with just variance of PDF? diagonal of covariance?

t0 = time.time()

# multi_z_save_dir = "/project/ls-gruen/users/jed.homer/sbipdf/results/moments/multi_z/"

figs_dir = "/project/ls-gruen/users/jed.homer/sbipdf/results/figs/"

alpha, parameter_strings = get_alpha_and_parameter_strings()

exp_setups = {
    "non-lin." : (False, False), 
    "non-lin. (pt)" : (False, True), 
    "lin." : (True, False),
}


# Loop through linear, non-linear, non-linear + pre-train
posteriors = []
for exp_setup, (linearised, pre_train) in exp_setups.items():
    print("Getting posterior", exp_setup, linearised, pre_train)

    if pre_train: #and (not linearised):
        continue

    multi_z_save_dir = "/project/ls-gruen/users/jed.homer/sbipdf/results/moments/multi_z/{}/{}{}".format(
        args.sbi_type,
        "linear/" if linearised else "",
        "pretrain/" if pre_train else ""
    )

    # Get config for linearised / pre-training / non-linearised config
    config = ensembles_moments_config(
        seed=args.seed, 
        sbi_type=args.sbi_type, 
        linearised=linearised, 
        pre_train=pre_train and (not linearised)
    )

    posterior_save_dir = get_multi_z_posterior_dir(config, default(args.sbi_type, "nle"))
    posterior_filename = os.path.join(posterior_save_dir, "posterior_{}.npz".format(args.seed))

    # posterior_save_file = os.path.join(
    #     multi_z_save_dir, 
    #     "{}/{}{}".format(
    #         config.sbi_type, 
    #         "linear/" if linearised else "", 
    #         "pretrain/" if pre_train else ""
    #     ),
    #     "posterior_{}.npz".format(args.seed)
    # )
    # print("POSTERIOR FILENAME", posterior_save_file)

    posterior_file = np.load(posterior_filename)
    samples, samples_log_prob, Finv_z, summary = posterior_file.values()

    print(summary) # Assuming summary the same for analyses, same for Finv_z

c = ChainConsumer()
c.add_chain(
    Chain.from_covariance(
        alpha, # Same truth (save this with posterior!)
        Finv_z,
        columns=parameter_strings,
        name=r"$F_{\Sigma^{-1}}$",
        color="k",
        linestyle=":",
        shade_alpha=0.
    )
)

# Plot posteriors
colors = ["red", "blue", "green"]

for posterior, color, (exp_setup, (linearised, pre_train)) in zip(
    posteriors, colors, exp_setups.items()
):
    print("Plotting posterior:", exp_setup, linearised, pre_train)

    samples, samples_log_probs = posterior 

    c.add_chain(
        Chain(
            samples=make_df(samples, samples_log_prob, parameter_strings), 
            name="SBI Posterior [{}]".format(exp_setup), 
            color=color, 
            shade_alpha=0.
        )
    )

c.add_truth(
    Truth(location=dict(zip(parameter_strings, alpha)), name=r"$\pi^0$")
)

# If summaries are from multiple datavectors just plot them as a cloud
if summary.ndim == 2 and summary.shape[0] > 1:
    c.add_chain(
        Chain(
            samples=make_df(summary, parameter_strings=parameter_strings), 
            name="Summaries".format(exp_setup), 
            color="k", 
            plot_contour=False,
            plot_cloud=True,
        )
    )
else:
    c.add_marker(
        location=dict(zip(parameter_strings, summary.squeeze())), name=r"$\hat{x}$", color="b"
    )
fig = c.plotter.plot()
plt.savefig(os.path.join(figs_dir, "initial_plot_{}.pdf".format(args.seed)))
plt.close()


# config = moments_config(
#     seed=args.seed, 
#     redshift=redshift, 
#     linearised=linearised, 
#     pre_train=pre_train
# )

# Dataset of simulations, parameters, covariance, ...
# dataset: Dataset = get_data(config)

# parameter_prior = get_prior(config)