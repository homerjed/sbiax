import os 
import argparse
from typing import Literal
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from ml_collections import ConfigDict
from chainconsumer import Chain, ChainConsumer, Truth, PlotConfig

from sbiax.utils import make_df, marker
from sbiax.ndes import Scaler

from configs import cumulants_config, bulk_cumulants_config, bulk_pdf_config, get_results_dir
from data.common import Dataset
from data.pdfs import BulkCumulantsDataset, BulkPDFsDataset
from data.cumulants import CumulantsDataset


def finite_samples_log_prob(samples_log_prob):
    samples_log_prob = jnp.where(
        jnp.logical_or(
            jnp.isnan(samples_log_prob), jnp.isneginf(samples_log_prob)
        ),
        # jnp.isnan(samples_log_prob), 
        -1e32,
        samples_log_prob
    )
    return samples_log_prob


def get_dataset_and_config(
    bulk_or_tails: Literal["bulk", "bulk_pdf", "tails"]
) -> tuple[Dataset, ConfigDict]:

    assert bulk_or_tails in ["bulk", "bulk_pdf", "tails"], (
        "bulk_or_tails == {}".format(bulk_or_tails)
    )

    if bulk_or_tails == "bulk": 
        dataset_constructor = BulkCumulantsDataset
        config = bulk_cumulants_config 
    if bulk_or_tails == "bulk_pdf":
        dataset_constructor = BulkPDFsDataset
        config = bulk_pdf_config 
    if bulk_or_tails == "tails":
        dataset_constructor = CumulantsDataset
        config = cumulants_config 

    return dataset_constructor, config


def get_datasets(args: argparse.Namespace) -> tuple[Dataset, dict[str, Dataset]]:
    # Get all configs and dataset objects for the dataset types here
    dataset_types = ["bulk", "bulk_pdf", "tails"]

    assert args.bulk_or_tails in dataset_types

    use_pdfs_or_cumulants = "pdf" in args.bulk_or_tails

    datasets, configs = dict(), dict()
    for dataset_type in dataset_types:
        # Dataset and config constructor for each type
        _dataset, _config = get_dataset_and_config(dataset_type) 

        config = _config(
            seed=args.seed, 
            redshift=args.redshift, 
            reduced_cumulants=args.reduced_cumulants,
            sbi_type=args.sbi_type,
            linearised=args.linearised, 
            compression=args.compression,
            order_idx=args.order_idx,
            freeze_parameters=args.freeze_parameters,
            n_linear_sims=args.n_linear_sims,
            pre_train=args.pre_train
        )

        results_dir = get_results_dir(config, args)

        configs[dataset_type] = config        

        datasets[dataset_type] = _dataset(
            configs[dataset_type], results_dir=results_dir
        )

    # Config and dataset being used in the experiment
    config = configs[args.bulk_or_tails]
    dataset = datasets[args.bulk_or_tails]

    return config, dataset, datasets


def plot_cumulants(args, config, cumulants, results_dir):
    cumulant_strings = [
        r"$\langle\delta^0\rangle$", 
        r"$\langle\delta\rangle$", 
        r"$\langle\delta^2\rangle$",
        r"$\langle\delta^3\rangle_c$",
        r"$\langle\delta^4\rangle_c$"
    ]

    n_scales = len(config.scales)
    n_cumulants_plot = 3
    if args.bulk_or_tails == "bulk":
        if config.stack_bulk_means:
            n_cumulants_plot += 1
        if config.stack_bulk_norms:
            n_cumulants_plot += 1

    fig, axs = plt.subplots(
        n_scales, 
        n_cumulants_plot, 
        figsize=(15., 27.), 
        dpi=200, 
        sharex=False, 
        sharey=False
    )
    if axs.ndim == 1:
        axs = axs[np.newaxis, :]

    for r in range(n_scales):
        for c in range(n_cumulants_plot):
            ax = axs[r, c]
            _cumulants = cumulants[
                :, r * n_cumulants_plot + c : (c + 1) + r * n_cumulants_plot
            ]
            mu = jnp.mean(_cumulants) 
            _cumulants = (_cumulants - mu) / jnp.std(_cumulants)
            ax.hist(
                _cumulants, 
                color="firebrick" if args.bulk_or_tails == "tails" else "royalblue",
                bins=32,
                density=True
            )
            x = np.linspace(-7., 7., 2000)
            gaussian_pdf = jax.scipy.stats.norm.pdf(x, loc=0., scale=1.)
            ax.plot(x, gaussian_pdf, color="k")
            ax.set_title(
                r"{}, R={}, $\mu$={:.2E}".format(
                    cumulant_strings[c], config.scales[r], mu
                )
            )
            ax.set_xlim(-7., 7.)
    plt.savefig(
        os.path.join(results_dir, "cumulants_test.png"), 
        bbox_inches="tight"
    )
    plt.close()


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
        # axs[i].set_xscale("log")
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

            if np.any(latin_moments_z_R[:, ix] < 0.):
                print("Warning: some latin moments less than zero (scale {})".format(config.scales[j]))

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
        # axs[i].set_xscale("log")
        axs[i].set_yscale("log")
    plt.tight_layout()

    if results_dir is not None:
        plt.savefig(os.path.join(results_dir, "moments_latin_histogram.png"), bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_summaries(X, P, dataset, results_dir=None):
    # Corner plot of summaries
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
    # plot_config = PlotConfig(
    #     extents=dict(
    #         zip(
    #             dataset.parameter_strings, 
    #             np.stack([dataset.lower, dataset.upper], axis=1)
    #         )
    #     )
    # )
    # c.set_plot_config(plot_config)
    fig = c.plotter.plot()
    if results_dir is not None:
        plt.savefig(os.path.join(results_dir, "params.pdf")) 
        plt.close()
    else:
        plt.show()

    # Scatter plot
    fig, axs = plt.subplots(1, dataset.alpha.size, figsize=(2. + 2. * dataset.alpha.size, 2.5))
    for p, ax in enumerate(axs):
        ax.scatter(dataset.parameters[:, p], X[:, p], s=0.1)
        ax.axline((0, 0), slope=1., color="k", linestyle="--")
        ax.set_xlim(dataset.lower[p], dataset.upper[p])
        ax.set_ylim(dataset.lower[p], dataset.upper[p])
        ax.set_xlabel(dataset.parameter_strings[p])
        ax.set_ylabel(dataset.parameter_strings[p] + "'")

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


def replace_scalers(ensemble, *, config, X, P):
    if config.use_scalers:
        is_scaler = lambda x: isinstance(x, Scaler)
        get_scalers = lambda m: [
            x
            for x in jax.tree.leaves(m, is_leaf=is_scaler)
            if is_scaler(x)
        ]
        ensemble = eqx.tree_at(
            get_scalers, 
            ensemble, 
            [Scaler(X, P)] * sum(int(nde.use_scaling) for nde in config.ndes) 
        )
    return ensemble