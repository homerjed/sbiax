import os 
import time
import argparse
from typing import Literal, Callable, Union
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np
from scipy.stats import chi2
from ml_collections import ConfigDict
from chainconsumer import Chain, ChainConsumer, Truth

from sbiax.utils import make_df
from sbiax.ndes import Scaler

from configs.log import setup_module_logger, get_log_level
from configs.configs import get_results_dir
from configs.cumulants_configs import cumulants_config, bulk_cumulants_config, bulk_pdf_config
from data.common import Dataset
from data.cumulants import CumulantsDataset
from data.constants import get_alpha_and_parameter_strings, LOWER, UPPER, get_target_idx

logger, log_figs_dir = setup_module_logger(__name__, level=get_log_level())


FORCE_RECOMPUTE_DATASET = True if os.environ.get("FORCE_RECOMPUTE_DATASET", "").lower() in ("1", "true") else False 
USE_QUIJOTE_TAILS = True if os.environ.get("USE_QUIJOTE_TAILS", "").lower() in ("1", "true") else False
USE_SOBOL = True if os.environ.get("USE_SOBOL", "").lower() in ("1", "true") else False 

if USE_SOBOL:
    from data.get_sobol_cumulants import (
        SobolBulkCumulantsDataset, 
        SobolTailsCumulantsDataset, 
        SobolBulkPDFsDataset
    )
    DatasetType = Union[
        SobolBulkCumulantsDataset, 
        SobolBulkPDFsDataset, 
        SobolTailsCumulantsDataset
    ]
else:
    from data.pdfs import (
        BulkCumulantsDataset, 
        BulkPDFsDataset, 
        TailsCumulantsDataset
    )
    DatasetType = Union[
        CumulantsDataset, 
        BulkCumulantsDataset, 
        BulkPDFsDataset, 
        TailsCumulantsDataset
    ]


import matplotlib.patches as patches

def overlay_bounds_on_corner(fig, lower, upper, *, ec="k", lw=1.5, alpha=0.12):
    lower = list(lower) 
    upper = list(upper)
    n = len(lower)

    # Map each Axes to its (row, col) index in the n x n grid
    ax_at = {}
    for ax in fig.axes:
        ss = ax.get_subplotspec()
        r, c = ss.rowspan.start, ss.colspan.start
        ax_at[(r, c)] = ax

    # Diagonals: 1D marginals (shade the allowed interval)
    for i in range(n):
        ax = ax_at.get((i, i))
        if ax is None: 
            continue
        ax.axvspan(lower[i], upper[i], color=ec, fc=None, alpha=alpha * 0.5, lw=0)

    # Off-diagonals (lower triangle): draw rectangles
    for i in range(1, n):          # rows (y = param i)
        for j in range(0, i):      # cols (x = param j)
            ax = ax_at.get((i, j))
            if ax is None: 
                continue
            rect = patches.Rectangle(
                (lower[j], upper[i]),                 # (x0, y0)
                (upper[j] - lower[j]),                  # width  in x (param j)
                -(upper[i] - lower[i]),                  # height in y (param i)
                fill=False, 
                ec=ec, 
                lw=lw, 
                linestyle=":"
            )
            ax.add_patch(rect)


def get_fisher_chain_df(alpha, Finv, parameter_strings=None, prior_clip=True):

    if parameter_strings is None:
        parameter_strings = get_alpha_and_parameter_strings()[1]

    # Use samples log probs or not?
    samples = np.random.multivariate_normal(alpha, Finv, size=(100_000,))

    if prior_clip:
        if alpha.size == 2:
            target_idx = get_target_idx()

            _lower, _upper = LOWER[target_idx], UPPER[target_idx]
        else:
            _lower, _upper = LOWER, UPPER

        # samples = np.clip(samples, _lower, _upper)
        mask = np.all((samples >= _lower) & (samples <= _upper), axis=1)
        samples = samples[mask]

    # samples_log_prob = np.random.multivariate_normal(alpha, Finv, size=(100_000,))
    df = make_df(samples, parameter_strings=parameter_strings)

    return df


def finite_samples_log_prob(samples_log_prob):
    n_bad = jnp.logical_or(
        jnp.isnan(samples_log_prob), jnp.isneginf(samples_log_prob)
    ).sum()

    print("CHAIN HAS {}/{} bad samples:".format(n_bad, samples_log_prob.size))

    samples_log_prob = jnp.where(
        jnp.logical_or(
            jnp.isnan(samples_log_prob), 
            jnp.isinf(samples_log_prob)
        ),
        -1e32,
        samples_log_prob
    )

    return samples_log_prob


def get_dataset_and_config(
    bulk_or_tails: Literal["bulk", "bulk_pdf", "tails"]
) -> tuple[
    Callable[[...], DatasetType], Callable[[...], ConfigDict]
]:

    assert bulk_or_tails in ["bulk", "bulk_pdf", "tails"], (
        "bulk_or_tails == {}".format(bulk_or_tails)
    )

    if bulk_or_tails == "bulk": 
        if USE_SOBOL:
            dataset_constructor = SobolBulkCumulantsDataset
        else:
            dataset_constructor = BulkCumulantsDataset

        config = bulk_cumulants_config

    if bulk_or_tails == "bulk_pdf":
        if USE_SOBOL:
            dataset_constructor = SobolBulkPDFsDataset
        else:
            dataset_constructor = BulkPDFsDataset

        config = bulk_pdf_config 

    if bulk_or_tails == "tails":
        if USE_QUIJOTE_TAILS:
            logger.info("NOTE:\n\tusing Quijote data for full-shape dataset.")
            print("NOTE:\n\tusing Quijote data for full-shape dataset.")

            if USE_SOBOL:
                raise NotImplementedError()

            dataset_constructor = CumulantsDataset
        else:
            logger.info("NOTE:\n\tusing calculations for full-shape dataset.")
            print("NOTE:\n\tusing calculations for full-shape dataset.")

            if USE_SOBOL:
                dataset_constructor = SobolTailsCumulantsDataset
            else:
                dataset_constructor = TailsCumulantsDataset 

        config = cumulants_config 

    return dataset_constructor, config


def get_datasets(args: argparse.Namespace) -> tuple[ConfigDict, Dataset, dict[str, Dataset]]:
    # Get all configs and dataset objects for the dataset types here

    dataset_types = ["bulk", "bulk_pdf", "tails"]

    assert args.bulk_or_tails in dataset_types

    datasets, configs = dict(), dict()
    for dataset_type in dataset_types:

        # Dataset and config constructor for each type
        _dataset, _config = get_dataset_and_config(dataset_type) 

        config = _config(
            seed=args.seed, 
            redshift=args.redshift, 
            linearised=args.linearised, 
            compression=args.compression,
            order_idx=args.order_idx,
            scales=args.scales,
            freeze_parameters=args.freeze_parameters,
            n_linear_sims=args.n_linear_sims,
            pre_train=args.pre_train,
            use_planck=args.use_planck
        )

        results_dir = get_results_dir(config, args)

        configs[dataset_type] = config        

        datasets[dataset_type] = _dataset(
            configs[dataset_type], results_dir=results_dir
        )

        assert args.redshift == configs[dataset_type].redshift == datasets[dataset_type].config.redshift, (
            "Mistmatch in redshifts for args / config / dataset = {} / {} / {}".format(
                args.redshift, configs[dataset_type].redshift, datasets[dataset_type].config.redshift
            )
        )

    # Config and dataset being used in the experiment
    config = configs[args.bulk_or_tails]
    dataset = datasets[args.bulk_or_tails]
    
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    fig, axs = plt.subplots(1, 3, figsize=(12., 4.))

    for ax, _dataset in zip(axs, datasets.values()):

        color = dict(bulk="b", tails="r", bulk_pdf="g")[_dataset.data.name]

        empirical_mean = np.mean(_dataset.data.fiducial_data, axis=0)

        k = _dataset.data.fiducial_data.shape[-1]

        # Compute Mahalanobis distances squared (chi2 values)
        mahalanobis_sq = np.array([
            (x - empirical_mean) @ _dataset.data.Cinv @ (x - empirical_mean)
            for x in _dataset.data.fiducial_data
        ])

        # Plot histogram of measured chi2 values
        bins = np.linspace(0., np.max(mahalanobis_sq), 50)
        ax.hist(
            mahalanobis_sq, 
            bins=bins, 
            density=True, 
            alpha=0.3, 
            # histtype="step",
            color=color,
            label=_dataset.data.name
        )

        key = jr.key(int(time.time()))
        linearised_data = jr.multivariate_normal(
            key, 
            empirical_mean, 
            _dataset.data.C, 
            shape=(len(_dataset.data.fiducial_data),)
        )
    
        # Compute Mahalanobis distances squared (chi2 values)
        mahalanobis_sq = np.array([
            (x - empirical_mean) @ _dataset.data.Cinv @ (x - empirical_mean)
            for x in linearised_data
        ])

        # Plot histogram of measured chi2 values
        bins = np.linspace(0., np.max(mahalanobis_sq), 50)
        ax.hist(
            mahalanobis_sq, 
            bins=bins, 
            density=True, 
            # alpha=0.3, 
            histtype="step",
            color="k",
            label=_dataset.data.name + " [linearised]"
        )

        ax.axvline(k, linestyle="--", color="k", label="d.o.f.={}".format(k))

        # Plot theoretical chi2 PDF
        x = np.linspace(0., np.max(mahalanobis_sq), 1000)
        ax.plot(
            x, 
            chi2.pdf(x, df=k),
            color + '--', 
            label=r"$\chi^2$ (d.o.f.={})".format(k)
        )

        ax.legend(frameon=False)

    linearised_str = "linearised" if config.linearised else "non-linear"

    plt.suptitle(r"$\chi^2$ [{}]".format(linearised_str))
    plt.tight_layout()
    plt.savefig(os.path.join(log_figs_dir, "chi2_tests_{}.png".format(linearised_str)))
    plt.close()

    print("SAVED Chi2 at:", os.path.join(log_figs_dir, "chi2_tests_{}.png".format(linearised_str)))

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    logger.info("MAIN DATASET:{} (requested: {})".format(dataset.data.name, args.bulk_or_tails))

    return config, dataset, datasets


def load_multi_z_cumulants_fisher_forecast(data_dir, args):
    """
        Load Fisher inverse matrix of cumulants dataset, over multiple redshifts, consistently with args
        - load bulk and tails dataset Finvs
    """

    if USE_QUIJOTE_TAILS:
        quijote_str = "_quijote"
    else:
        quijote_str = ""

    parts = [
        "_R" + "".join(map(str, args.scales)),
        "_m" + "".join(map(str, args.order_idx)),
        "_f" if args.freeze_parameters else "_nf"
    ]
    identifier_str = "".join(parts)

    Finv_bulk_file_path = os.path.join(
        data_dir, "Finv_bulk_all_z_{}.npy".format(identifier_str)
    )
    Finv_tails_file_path = os.path.join(
        data_dir, "Finv_tails_all_z_{}.npy".format(identifier_str + quijote_str)
    )

    if not FORCE_RECOMPUTE_DATASET:
        try:
            Finv_bulk_all_z = np.load(Finv_bulk_file_path)
            Finv_tails_all_z = np.load(Finv_tails_file_path)
        except:

            _, _, datasets = get_datasets(args) # Config and cumulants_dataset can be bulk ... etc
            Finv_bulk_all_z = datasets["bulk"].data.Finv
            Finv_tails_all_z = datasets["tails"].data.Finv

            np.save(Finv_bulk_file_path, Finv_bulk_all_z)
            np.save(Finv_tails_file_path, Finv_tails_all_z)
        else:
            _, _, datasets = get_datasets(args) # Config and cumulants_dataset can be bulk ... etc
            Finv_bulk_all_z = datasets["bulk"].data.Finv
            Finv_tails_all_z = datasets["tails"].data.Finv

    logger.info("Finv bulk all z loaded from:\n\t{}".format(Finv_bulk_file_path))
    logger.info("Finv tails all z loaded from:\n\t{}".format(Finv_tails_file_path))

    return Finv_bulk_all_z, Finv_tails_all_z


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
    if 1: #args.bulk_or_tails == "bulk":
        if config.stack_means:
            n_cumulants_plot += 1
        if config.use_normalisations:
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
            color="k", 
            plot_cloud=True, 
            plot_contour=False
        )
    )
    c.add_chain(
        Chain(
            samples=make_df(X, parameter_strings=dataset.parameter_strings), 
            name="Summaries", 
            color="red" if dataset.name == "tails" else "blue", 
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
        plt.savefig(os.path.join(results_dir, "params.png")) 
        plt.close()
    else:
        plt.show()

    # Scatter plot
    fig, axs = plt.subplots(1, dataset.alpha.size, figsize=(2. + 2. * dataset.alpha.size, 2.5))
    l = np.linspace(-2., 2., 1000)
    for p, ax in enumerate(axs):
        Finv_std = jnp.sqrt(dataset.Finv[p, p])

        ax.scatter(dataset.parameters[:, p], X[:, p], s=0.1, color="red" if dataset.name == "tails" else "blue")
        ax.axline((0, 0), slope=1., color="k", linestyle="--")
        ax.fill_between(l, l - Finv_std, l + Finv_std, color="gray", alpha=0.3, label=r"±$F^{-1}_{\pi}")

        ax.set_xlim(dataset.lower[p], dataset.upper[p])
        ax.set_ylim(dataset.lower[p], dataset.upper[p])

        ax.set_xlabel(dataset.parameter_strings[p])
        ax.set_ylabel(dataset.parameter_strings[p] + "'")

    if results_dir is not None:
        plt.savefig(os.path.join(results_dir, "scatter.png"))
        plt.close()
    else:
        plt.show()


def plot_summaries_fiducial(X, X_, alpha, dataset, results_dir=None, Finv=None, filename=None):

    P = jnp.tile(alpha[jnp.newaxis, :], (X.shape[0], 1))

    # Corner plot of summaries
    c = ChainConsumer()
    c.add_chain(
        Chain(
            samples=make_df(P, parameter_strings=dataset.parameter_strings), 
            name="Params", 
            color="k", 
            plot_cloud=True, 
            plot_contour=False
        )
    )
    c.add_chain(
        Chain(
            samples=make_df(X, parameter_strings=dataset.parameter_strings), 
            name="Summaries", 
            color="red" if dataset.name == "tails" else "blue", 
            plot_cloud=True, 
            plot_contour=True
        )
    )
    c.add_chain(
        Chain(
            samples=make_df(X_, parameter_strings=dataset.parameter_strings), 
            name="Summaries data", 
            color="red" if dataset.name == "tails" else "blue", 
            plot_cloud=True, 
            plot_contour=False
        )
    )

    n_fisher_samples = 800_000

    def cut_samples(samples, lower, upper):
        return samples[np.all((samples >= lower) & (samples <= upper), axis=1)]

    fisher_samples = cut_samples(
        np.random.multivariate_normal(
            dataset.alpha, 
            Finv if Finv is not None else dataset.Finv,
            size=(n_fisher_samples,)
        ), 
        LOWER, 
        UPPER
    )

    c.add_chain(
        Chain(
            samples=make_df(fisher_samples, parameter_strings=dataset.parameter_strings), 
            name=r"$F_{\Sigma^{-1}}$",
            color="k",
            linestyle=":",
            shade_alpha=0.
        )
        # Chain.from_covariance(
        #     dataset.alpha,
        #     Finv if Finv is not None else dataset.Finv,
        #     columns=dataset.parameter_strings,
        #     name=r"$F_{\Sigma^{-1}}$",
        #     color="k",
        #     linestyle=":",
        #     shade_alpha=0.
        # )
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
    overlay_bounds_on_corner(fig, dataset.lower, dataset.upper)

    if results_dir is not None:
        if filename is not None:
            plt.savefig(os.path.join(results_dir, filename)) 
        else:
            plt.savefig(os.path.join(results_dir, "fiducial_params.png")) 
        plt.close()
    else:
        plt.show()

    # Scatter plot
    fig, axs = plt.subplots(1, dataset.alpha.size, figsize=(2. + 2. * dataset.alpha.size, 2.5))
    tiled_alpha = np.tile(alpha[np.newaxis, :], (len(X), 1))
    l = np.linspace(-2., 2., 1000)
    for p, ax in enumerate(axs):
        Finv_std = jnp.sqrt(dataset.Finv[p, p])

        ax.scatter(tiled_alpha[:, p], X[:, p], s=0.1, color="red" if dataset.name == "tails" else "blue")
        ax.axline((0, 0), slope=1., color="k", linestyle="--")
        ax.fill_between(l, l - Finv_std, l + Finv_std, color="gray", alpha=0.3, label=r"±$F^{-1}_{\pi}")

        ax.set_xlim(dataset.lower[p], dataset.upper[p])
        ax.set_ylim(dataset.lower[p], dataset.upper[p])

        ax.set_xlabel(dataset.parameter_strings[p])
        ax.set_ylabel(dataset.parameter_strings[p] + "'")

    if results_dir is not None:
        plt.savefig(os.path.join(results_dir, "scatter_fiducial.png"))
        plt.close()
    else:
        plt.show()


def plot_fisher_summaries(X, P, dataset, results_dir=None):
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
    c.add_chain(
        Chain.from_covariance(
            dataset.alpha,
            dataset.Finv,
            columns=dataset.parameter_strings,
            name=r"$F_{\Sigma^{-1}}$",
            color="k",
            linestyle=":",
            shade_alpha=0.
        )
    )
    c.add_truth(
        Truth(location=dict(zip(dataset.parameter_strings, dataset.alpha)), name=r"$\pi^0$")
    )
    fig = c.plotter.plot()
    plt.savefig(os.path.join(results_dir, "fisher_params.png"))
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