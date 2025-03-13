import os 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from chainconsumer import Chain, ChainConsumer, Truth

from sbiax.utils import make_df, marker


def plot_moments(fiducial_moments_z_R, config, results_dir=None):

    moment_names = ["variance", "skewness", "kurtosis"]

    fiducial_moments_z_R = np.clip(fiducial_moments_z_R, a_min=0., a_max=fiducial_moments_z_R.max())
    bins = np.geomspace(fiducial_moments_z_R.min() + 1e-6, fiducial_moments_z_R.max(), 32)

    fig, axs = plt.subplots(
        1, len(config.order_idx), figsize=(1. + len(config.order_idx) * 3., 3.), sharex=True, sharey=True
    )

    if len(config.order_idx) == 1: axs = [axs]
    for i in config.order_idx:  
        for j in range(len(config.scales)):  
            ix = i + j * len(config.order_idx) #(j % len(config.scales))
            axs[i].hist(
                fiducial_moments_z_R[:, ix], 
                range=[np.min(fiducial_moments_z_R), np.max(fiducial_moments_z_R)], 
                bins=bins,
                alpha=0.7, 
                density=True,
                histtype="step",
                label="R={}".format(config.scales[j])
            )
        axs[i].set_title(moment_names[i])
        axs[i].legend(frameon=False)  
        axs[i].set_xscale("log")
        axs[i].set_yscale("log")
    plt.tight_layout()

    if results_dir is not None:
        plt.savefig(os.path.join(results_dir, "moments_histogram.png"), bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_latin_moments(latin_moments_z_R, config, results_dir=None):
    moment_names = ["variance", "skewness", "kurtosis"]

    latin_moments_z_R = np.clip(latin_moments_z_R, a_min=0., a_max=latin_moments_z_R.max())
    bins = np.geomspace(latin_moments_z_R.min() + 1e-6, latin_moments_z_R.max(), 8)

    fig, axs = plt.subplots(
        1, len(config.order_idx), figsize=(1. + len(config.order_idx) * 3., 3.), sharex=True, sharey=True
    )

    if len(config.order_idx) == 1: axs = [axs]
    for i in config.order_idx:  
        for j in range(len(config.scales)):  
            ix = i + j * len(config.order_idx) #(j % len(config.scales))
            axs[i].hist(
                latin_moments_z_R[:, ix], 
                range=[np.min(latin_moments_z_R), np.max(latin_moments_z_R)], 
                bins=bins,
                alpha=0.7, 
                density=True,
                histtype="step",
                label="R={}".format(config.scales[j])
            )
        axs[i].set_title(moment_names[i])
        axs[i].legend(frameon=False)  
        axs[i].set_xscale("log")
        axs[i].set_yscale("log")
    plt.tight_layout()

    if results_dir is not None:
        plt.savefig(os.path.join(results_dir, "moments_latin_histogram.png"), bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_summaries(X, P, dataset, results_dir=None):
    c = ChainConsumer()
    c.add_chain(
        Chain(
            samples=make_df(P, parameter_strings=dataset.parameter_strings), 
            name="Params", 
            color="blue", 
            plot_cloud=True, 
            plot_contour=False
        )
    )
    c.add_chain(
        Chain(
            samples=make_df(X, parameter_strings=dataset.parameter_strings), 
            name="Summaries", 
            color="red", 
            plot_cloud=True, 
            plot_contour=False
        )
    )
    c.add_truth(
        Truth(location=dict(zip(dataset.parameter_strings, dataset.alpha)), name=r"$\pi^0$")
    )
    fig = c.plotter.plot()
    if results_dir is not None:
        plt.savefig(os.path.join(results_dir, "params.pdf")) 
        plt.close()
    else:
        plt.show()

    fig, axs = plt.subplots(1, dataset.alpha.size, figsize=(2. * dataset.alpha.size, 2.))
    for p, ax in enumerate(axs):
        ax.scatter(P[:, p], X[:, p])
        ax.axline((0, 0), slope=1., color="k", linestyle="--")
        ax.set_xlim(dataset.lower[p], dataset.upper[p])
        ax.set_ylim(dataset.lower[p], dataset.upper[p])

    if results_dir is not None:
        plt.savefig(os.path.join(results_dir, "scatter.png"))
        plt.close()
    else:
        plt.show()


def plot_fisher_summaries(X, P, dataset, results_dir=None):
# def plot_fisher_summaries(X_l, dataset, results_dir):
    # c = ChainConsumer()

    # c.add_chain(
    #     Chain(
    #         samples=make_df(X_l, parameter_strings=dataset.parameter_strings), 
    #         name="Summaries: linearised data", 
    #         color="blue", 
    #         plot_cloud=True, 
    #         plot_contour=False
    #     )
    # )
    # c.add_truth(
    #     Truth(location=marker(dataset.alpha, dataset.parameter_strings), name=r"$\pi^0$")
    # )
    # fig = c.plotter.plot()
    # plt.savefig(os.path.join(results_dir, "fisher_x.pdf"))
    # plt.close()

    c = ChainConsumer()
    c.add_chain(
        Chain(
            samples=make_df(P, parameter_strings=dataset.parameter_strings), 
            name="Params", 
            color="blue", 
            plot_cloud=True, 
            plot_contour=False
        )
    )
    c.add_chain(
        Chain(
            samples=make_df(X, parameter_strings=dataset.parameter_strings), 
            name="Summaries", 
            color="red", 
            plot_cloud=True, 
            plot_contour=False
        )
    )
    c.add_truth(
        Truth(location=dict(zip(dataset.parameter_strings, dataset.alpha)), name=r"$\pi^0$")
    )
    fig = c.plotter.plot()
    plt.savefig(os.path.join(results_dir, "fisher_params.pdf"))
    plt.close()

    fig, axs = plt.subplots(1, dataset.alpha.size, figsize=(2. * dataset.alpha.size, 2.))
    for p, ax in enumerate(axs):
        ax.scatter(P[:, p], X[:, p])
        ax.axline((0, 0), slope=1., color="k", linestyle="--")
        ax.set_xlim(dataset.lower[p], dataset.upper[p])
        ax.set_ylim(dataset.lower[p], dataset.upper[p])

    if results_dir is not None:
        plt.savefig(os.path.join(results_dir, "fisher_scatter.png"))
        plt.close()
    else:
        plt.show()