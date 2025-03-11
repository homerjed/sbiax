import argparse
import os
import time
import datetime
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key
import optax
import numpy as np 
import matplotlib.pyplot as plt
from chainconsumer import Chain, ChainConsumer, Truth
from tensorflow_probability.substrates.jax.distributions import Distribution
from ml_collections import ConfigDict
from matplotlib.ticker import MaxNLocator

from configs import cumulants_config, get_results_dir, get_posteriors_dir, get_cumulants_sbi_args, get_ndes_from_config
from cumulants import Dataset, get_data, get_prior, get_linear_compressor, get_datavector, get_linearised_data

from sbiax.utils import make_df, marker
from sbiax.ndes import Scaler, Ensemble, CNF, MAF 
from sbiax.train import train_ensemble
from sbiax.inference import nuts_sample
from sbiax.compression.nn import fit_nn

from affine import affine_sample
from utils import plot_moments, plot_latin_moments, plot_summaries, plot_fisher_summaries
from pca import PCA


if __name__ == "__main__":

    scratch_dir = "/project/ls-gruen/users/jed.homer/sbiaxpdf/cumulants/scratch/"

    args = get_cumulants_sbi_args()

    args.linearised = True
    args.reduced_cumulants = False
    args.order_idx = [0]
    args.n_linear_sims = 10_000

    print("TIME:", datetime.datetime.now().strftime("%H:%M %d-%m-%y"))
    print("SEED:", args.seed)
    print("MOMENTS:", args.order_idx)
    print("LINEARISED:", args.linearised)

    t0 = time.time()

    config = cumulants_config(
        seed=args.seed, 
        redshift=args.redshift, 
        linearised=args.linearised, 
        compression=args.compression,
        reduced_cumulants=args.reduced_cumulants,
        order_idx=args.order_idx,
        pre_train=args.pre_train,
        n_linear_sims=args.n_linear_sims
    )

    key = jr.key(config.seed)

    ( 
        model_key, train_key, key_prior, 
        key_datavector, key_state, key_sample
    ) = jr.split(key, 6)

    results_dir: str = get_results_dir(config, args)

    posteriors_dir: str = get_posteriors_dir(config)

    dataset: Dataset = get_data(config, verbose=args.verbose)

    if 0:
        """
            Plot moments
        """

        fig, axs = plt.subplots(
            len(config.scales), 
            len(config.order_idx), 
            figsize=(1. + len(config.order_idx) * 3., len(config.scales) * 3.), 
            # sharex=True, 
            # sharey=True
        )

        moment_names = ["variance", "skewness", "kurtosis"]

        fiducial_moments_z_R = jnp.asarray(dataset.fiducial_data)
        latin_moments_z_R = jnp.asarray(dataset.data)

        print("EGG", fiducial_moments_z_R.shape)

        if len(config.order_idx) == 1: 
            axs = axs # [axs]

        for i in config.order_idx:  
            for j in range(len(config.scales)):  
                ax = axs[j, i] if (len(config.order_idx) > 1) else axs[j]

                egg = i + j * len(config.order_idx) #(j % len(config.scales))

                print(i, j, egg, fiducial_moments_z_R.shape)

                cumulants_n_R = fiducial_moments_z_R[:, egg]

                print(cumulants_n_R)

                bins_range = [np.min(cumulants_n_R), np.max(cumulants_n_R)]

                # bins = np.geomspace(bins_range[0] + 1e-6, bins_range[1], 100)
                bins = np.linspace(bins_range[0] + 1e-6, bins_range[1], 100)

                # Sample from a Gaussian distribution with mean/var of each 
                mean_n_R = np.mean(cumulants_n_R, axis=0)
                std_n_R = np.std(cumulants_n_R, axis=0)
                gaussian_samples = np.random.normal(loc=mean_n_R, scale=std_n_R, size=(100_000,))

                ax.hist(
                    gaussian_samples,
                    range=bins_range,
                    bins=bins,
                    alpha=0.3, 
                    color="royalblue",
                    density=True,
                    label="R={} (Gaussian)".format(config.scales[j])
                )
                ax.hist(
                    gaussian_samples, 
                    range=bins_range,
                    bins=bins,
                    density=True,
                    color="k",
                    histtype="step",
                )

                ax.hist(
                    cumulants_n_R,
                    range=bins_range,
                    bins=bins,
                    alpha=0.3, 
                    color="goldenrod",
                    density=True,
                    label="R={}".format(config.scales[j])
                )
                ax.hist(
                    cumulants_n_R, 
                    range=bins_range,
                    bins=bins,
                    density=True,
                    color="k",
                    histtype="step",
                )

                ax.set_title(moment_names[i])
                ax.legend(frameon=False)  
                ax.set_xticks([])
                ax.set_xticks([])
                ticks = np.linspace(bins_range[0], bins_range[1], 5)
                labels = ["{:.3E}".format(tick) for tick in ticks]
                ax.set_xticks(ticks, labels, rotation='vertical')
        plt.tight_layout()
        linear_str = "linear" if args.linearised else "nonlinear"
        plt.savefig(os.path.join(scratch_dir, "moments_histogram_nolog_{}_z={}.png".format(linear_str, args.redshift)), bbox_inches="tight")
        plt.close()



        fig, axs = plt.subplots(
            len(config.scales), 
            len(config.order_idx), 
            figsize=(1. + len(config.order_idx) * 3., len(config.scales) * 3.), 
            # sharex=True, 
            # sharey=True
        )

        for i in config.order_idx:  
            for j in range(len(config.scales)):  
                ax = axs[j, i] if (len(config.order_idx) > 1) else axs[j]

                egg = i + j * len(config.order_idx) #(j % len(config.scales))

                cumulants_n_R = latin_moments_z_R[:, egg]

                print(cumulants_n_R)

                bins_range = [np.min(cumulants_n_R), np.max(cumulants_n_R)]

                # bins = np.geomspace(bins_range[0] + 1e-6, bins_range[1], 100)
                bins = np.linspace(bins_range[0] + 1e-6, bins_range[1], 100)

                # Sample from a Gaussian distribution with mean/var of each 
                mean_n_R = np.mean(cumulants_n_R, axis=0)
                std_n_R = np.std(cumulants_n_R, axis=0)
                gaussian_samples = np.random.normal(loc=mean_n_R, scale=std_n_R, size=(100_000,))

                ax.hist(
                    gaussian_samples,
                    range=bins_range,
                    bins=bins,
                    alpha=0.3, 
                    color="royalblue",
                    density=True,
                    label="R={} (Gaussian)".format(config.scales[j])
                )
                ax.hist(
                    gaussian_samples, 
                    range=bins_range,
                    bins=bins,
                    density=True,
                    color="k",
                    histtype="step",
                )

                ax.hist(
                    cumulants_n_R,
                    range=bins_range,
                    bins=bins,
                    alpha=0.3, 
                    color="goldenrod",
                    density=True,
                    label="R={}".format(config.scales[j])
                )
                ax.hist(
                    cumulants_n_R, 
                    range=bins_range,
                    bins=bins,
                    density=True,
                    color="k",
                    histtype="step",
                )

                ax.set_title(moment_names[i])
                ax.legend(frameon=False)  
                ax.set_xticks([])
                ax.set_xticks([])
                ticks = np.linspace(bins_range[0], bins_range[1], 5)
                labels = ["{:.3E}".format(tick) for tick in ticks]
                ax.set_xticks(ticks, labels, rotation='vertical')
        plt.tight_layout()
        linear_str = "linear" if args.linearised else "nonlinear"
        plt.savefig(os.path.join(scratch_dir, "moments_histogram_nolog_latin_{}_z{}.png".format(linear_str, args.redshift)), bbox_inches="tight")
        plt.close()






        fig, axs = plt.subplots(
            len(config.scales), 
            len(config.order_idx), 
            figsize=(1. + len(config.order_idx) * 3., len(config.scales) * 3.), 
            # sharex=True, 
            # sharey=True
        )

        moment_names = ["variance", "skewness", "kurtosis"]

        fiducial_moments_z_R = dataset.fiducial_data 

        # Few very small negative values
        fiducial_moments_z_R = np.clip(
            fiducial_moments_z_R, a_min=0., a_max=fiducial_moments_z_R.max()
        )

        if len(config.order_idx) == 1: axs = [axs]
        for i in config.order_idx:  
            for j in range(len(config.scales)):  
                ax = axs[j, i]

                cumulants_n_R = fiducial_moments_z_R[:, i + (j % len(config.scales))]

                bins_range = [0.6 * np.min(cumulants_n_R), 1.2 * np.max(cumulants_n_R)]

                bins = np.geomspace(bins_range[0] + 1e-6, bins_range[1], 100)

                # Sample from a Gaussian distribution with mean/var of each 
                mean_n_R = np.mean(cumulants_n_R, axis=0)
                std_n_R = np.std(cumulants_n_R, axis=0)
                gaussian_samples = np.random.normal(loc=mean_n_R, scale=std_n_R, size=(100_000,))

                ax.hist(
                    gaussian_samples,
                    range=bins_range,
                    bins=bins,
                    alpha=0.3, 
                    color="royalblue",
                    density=True,
                    label="R={} (Gaussian)".format(config.scales[j])
                )
                ax.hist(
                    gaussian_samples, 
                    range=bins_range,
                    bins=bins,
                    density=True,
                    color="k",
                    histtype="step",
                )

                ax.hist(
                    cumulants_n_R,
                    range=bins_range,
                    bins=bins,
                    alpha=0.3, 
                    color="goldenrod",
                    density=True,
                    label="R={}".format(config.scales[j])
                )
                ax.hist(
                    cumulants_n_R, 
                    range=bins_range,
                    bins=bins,
                    density=True,
                    color="k",
                    histtype="step",
                )

                ax.set_title(moment_names[i])
                ax.legend(frameon=False)  
                # ax.set_xticks(np.linspace(bins_range[0], bins_range[1], 5))
                # ax.set_xticklabels(ax.get_xticks(), rotation=45)
                ax.set_xticks([])
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xticks([])
                ticks = np.linspace(bins_range[0], bins_range[1], 5)
                labels = ["{:.3E}".format(tick) for tick in ticks]
                ax.set_xticks(ticks, labels, rotation='vertical')
                # ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Show ~5 labels
        plt.tight_layout()
        plt.savefig(os.path.join(scratch_dir, "moments_histogram.png"), bbox_inches="tight")
        plt.close()

    """
        Test PCA
    """

    print("Testing PCA...")

    parameter_prior: Distribution = get_prior(config)

    # Get linear or neural network compressor
    if config.compression == "nn":
        net_key, train_key = jr.split(key)

        net = eqx.nn.MLP(
            dataset.data.shape[-1], 
            dataset.parameters.shape[-1], 
            width_size=32, 
            depth=2, 
            activation=jax.nn.tanh,
            key=net_key
        )

        def preprocess_fn(x): 
            return (jnp.asarray(x) - jnp.mean(dataset.data, axis=0)) / jnp.std(dataset.data, axis=0)

        net, losses = fit_nn(
            train_key, 
            net, 
            (preprocess_fn(dataset.data), dataset.parameters), 
            opt=optax.adam(1e-3), 
            n_batch=500, 
            patience=1000
        )

        plt.figure()
        plt.loglog(losses)
        plt.savefig(os.path.join(results_dir, "losses_nn.png"))
        plt.close()

        compressor = lambda d, p: net(preprocess_fn(d)) # Ignore parameter kwarg!
    if config.compression == "linear":
        compressor = get_linear_compressor(config)

    # Fit PCA transform to simulated data and apply after compressing
    use_pca = False 
    if use_pca:

        # Compress simulations as usual 
        X = jax.vmap(compressor)(dataset.data, dataset.parameters)

        # Standardise before PCA (don't get tricked by high variance due to units)
        X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)

        # Fit whitening-PCA to compressed simulations
        pca = PCA(num_components=dataset.alpha.size)
        pca.fit(X)

        print(X.shape)
        
        # Reparameterize compression with both transforms
        compression_fn = lambda d, p: pca.transform(compressor(d, p))
    else:
        compression_fn = lambda *args, **kwargs: compressor(*args, **kwargs)

    # Compress simulations
    X = jax.vmap(compression_fn)(dataset.data, dataset.parameters)

    print("Plotting...")

    c = ChainConsumer()
    c.add_chain(
        Chain(
            samples=make_df(
                dataset.parameters, 
                parameter_strings=dataset.parameter_strings
            ), 
            name="Params", 
            color="blue", 
            plot_cloud=True, 
            plot_contour=False
        )
    )
    c.add_chain(
        Chain(
            samples=make_df(
                X, 
                parameter_strings=dataset.parameter_strings
            ), 
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
    plt.savefig(os.path.join(scratch_dir, "params.pdf"))
    plt.close()

    print("Plotting...")

    fig, axs = plt.subplots(1, dataset.alpha.size, figsize=(2. * dataset.alpha.size, 2.))
    for p, ax in enumerate(axs):
        ax.scatter(dataset.parameters[:len(X), p], X[:, p])
        ax.axline((0, 0), slope=1., color="k", linestyle="--")
        ax.set_xlim(dataset.lower[p], dataset.upper[p])
        ax.set_ylim(dataset.lower[p], dataset.upper[p])
    plt.savefig(os.path.join(scratch_dir, "scatter.png"))
    plt.close()



    """
        Test PCA, fitting to fiducials
    """

    print("Testing PCA...")

    parameter_prior: Distribution = get_prior(config)

    # Get linear or neural network compressor
    if config.compression == "nn":
        net_key, train_key = jr.split(key)

        net = eqx.nn.MLP(
            dataset.data.shape[-1], 
            dataset.parameters.shape[-1], 
            width_size=32, 
            depth=2, 
            activation=jax.nn.tanh,
            key=net_key
        )

        def preprocess_fn(x): 
            return (jnp.asarray(x) - jnp.mean(dataset.data, axis=0)) / jnp.std(dataset.data, axis=0)

        net, losses = fit_nn(
            train_key, 
            net, 
            (preprocess_fn(dataset.data), dataset.parameters), 
            opt=optax.adam(1e-3), 
            n_batch=500, 
            patience=1000
        )

        plt.figure()
        plt.loglog(losses)
        plt.savefig(os.path.join(results_dir, "losses_nn.png"))
        plt.close()

        compressor = lambda d, p: net(preprocess_fn(d)) # Ignore parameter kwarg!
    if config.compression == "linear":
        compressor = get_linear_compressor(config)

    # Fit PCA transform to simulated data and apply after compressing
    use_pca = True 
    if use_pca:

        # Compress simulations as usual 
        X = jax.vmap(compressor, in_axes=(0, None))(dataset.fiducial_data, dataset.alpha)

        # Standardise before PCA (don't get tricked by high variance due to units)
        X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)

        # Fit whitening-PCA to compressed simulations
        pca = PCA(num_components=dataset.alpha.size)
        pca.fit(X) # Fit on fiducial data

        print(X.shape)
        
        # Reparameterize compression with both transforms
        compression_fn = lambda d, p: pca.transform(compressor(d, p))
    else:
        compression_fn = lambda *args, **kwargs: compressor(*args, **kwargs)

    # Compress simulations
    X = jax.vmap(compression_fn)(dataset.data, dataset.parameters)

    print("Plotting...")

    c = ChainConsumer()
    c.add_chain(
        Chain(
            samples=make_df(
                dataset.parameters, 
                parameter_strings=dataset.parameter_strings
            ), 
            name="Params", 
            color="blue", 
            plot_cloud=True, 
            plot_contour=False
        )
    )
    c.add_chain(
        Chain(
            samples=make_df(
                X, 
                parameter_strings=dataset.parameter_strings
            ), 
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
    plt.savefig(os.path.join(scratch_dir, "params_fiducial.pdf"))
    plt.close()

    print("Plotting...")

    fig, axs = plt.subplots(1, dataset.alpha.size, figsize=(2. * dataset.alpha.size, 2.))
    for p, ax in enumerate(axs):
        ax.scatter(dataset.parameters[:len(X), p], X[:, p])
        ax.axline((0, 0), slope=1., color="k", linestyle="--")
        ax.set_xlim(dataset.lower[p], dataset.upper[p])
        ax.set_ylim(dataset.lower[p], dataset.upper[p])
    plt.savefig(os.path.join(scratch_dir, "scatter_fiducial.png"))
    plt.close()
