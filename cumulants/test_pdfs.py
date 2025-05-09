# import jax
# jax.config.update("jax_debug_nans", True)

import jax
import matplotlib.pyplot as plt
import numpy as np
from chainconsumer import Chain, ChainConsumer, Truth

from configs.cumulants_configs import cumulants_config, bulk_cumulants_config
from data.pdfs import BulkCumulantsDataset
from data.cumulants import CumulantsDataset
from utils.utils import get_dataset_and_config
from sbiax.utils import marker

bulk_config = bulk_cumulants_config()
full_shape_config = cumulants_config()

# bulk_config.order_idx = [0, 1]

""" 
    Compare with Quijote
"""

# Calculate full shape cumulants from quijote pdfs, compare to pdf measured in quijote
if 0:
    use_mean = False
    verbose = True

    # Calculated cumulants for full shape of PDF, using my calculations
    calculated_cumulants_dataset = BulkCumulantsDataset(
        bulk_config, 
        pdfs=False, 
        # use_mean=use_mean, 
        check_cumulants_against_quijote=True, 
        verbose=verbose
    )

    # Bulk cumulants by my calculations
    bulk_cumulants_dataset = BulkCumulantsDataset(
        bulk_config, 
        pdfs=False, 
        # use_mean=use_mean, 
        check_cumulants_against_quijote=False, 
        verbose=verbose
    )

    # Quijote cumulants for full shape of PDF
    cumulants_dataset = CumulantsDataset(full_shape_config, verbose=verbose)

    # PDFs of bulk dataset
    pdfs_dataset = BulkCumulantsDataset(bulk_config, pdfs=True, verbose=verbose)






    # Histogram of cumulants I calculate against those measured in Quijote
    for cumulants, name in zip(
        [
            calculated_cumulants_dataset.data.fiducial_data, 
            cumulants_dataset.data.fiducial_data,
            bulk_cumulants_dataset.data.fiducial_data
        ],
        ["MY CALCULATION", "QUIJOTE", "MY CALCULATION BULK"]
    ):
        print("{} {:.3E} {:.3E}".format(name, cumulants.min(), cumulants.max()))

    fig, axes = plt.subplots(7, 3, figsize=(9., 14.))

    names = ["var", "skew", "kurt"] if not use_mean else ["mean", "var", "skew", "kurt"]
    scales = ["5.0", "10.0", "15.0", "20.0", "25.0", "30.0", "35.0"]
    for i in range(len(names)):
        for r in range(len(scales)):

            axes[r, i].hist(
                calculated_cumulants_dataset.data.fiducial_data[:, i + r * 3], # If using mean, shift i here by 1
                bins=50, 
                color="blue",
                alpha=0.1, 
                label='C'
            )
            axes[r, i].hist(
                cumulants_dataset.data.fiducial_data[:, i + r * 3], 
                bins=50, 
                histtype='step', 
                color='orange', 
                linewidth=1.5, 
                alpha=0.2, 
                label='Q'
            )
            axes[r, i].hist(
                bulk_cumulants_dataset.data.fiducial_data[:, i + r * 3], 
                bins=50, 
                histtype='step', 
                color='g', 
                linewidth=1.5, 
                alpha=0.2, 
                label='C[b]'
            )

            axes[r, i].set_title(names[i] + " R=" + scales[r])
            axes[r, i].legend(frameon=False)

    plt.tight_layout()
    plt.savefig("QUIJOTE_VS_FULLSHAPE_CUMULANTS.png", bbox_inches="tight")
    plt.close()









    # Compare Fisher forecasts from bulk of PDF to cumulants of bulk

    c = ChainConsumer()
    c.add_chain(
        Chain.from_covariance(
            cumulants_dataset.data.alpha,
            cumulants_dataset.data.Finv,
            columns=cumulants_dataset.data.parameter_strings,
            name=r"$F_{\Sigma^{-1}}$" + " {}".format("Cumulants[full shape]"),
            color="k",
            linestyle=":",
            shade_alpha=0.
        )
    )
    c.add_chain(
        Chain.from_covariance(
            calculated_cumulants_dataset.data.alpha,
            calculated_cumulants_dataset.data.Finv,
            columns=cumulants_dataset.data.parameter_strings,
            name=r"$F_{\Sigma^{-1}}$" + " {}".format("Cumulants[bulk]"),
            color="r",
            linestyle=":",
            shade_alpha=0.
        )
    )
    c.add_chain(
        Chain.from_covariance(
            cumulants_dataset.data.alpha,
            pdfs_dataset.data.Finv,
            columns=cumulants_dataset.data.parameter_strings,
            name=r"$F_{\Sigma^{-1}}$" + " {}".format("PDF[bulk]"),
            color="g",
            linestyle=":",
            shade_alpha=0.
        )
    )
    c.add_marker(
        location=marker(cumulants_dataset.data.alpha, parameter_strings=cumulants_dataset.data.parameter_strings),
        name=r"$\alpha$", 
        color="#7600bc"
    )
    fig = c.plotter.plot()

    filename = "pdfs_cumulants_Finv_bulk.png"

    print("Finv pdfs_dataset:\n", pdfs_dataset.data.Finv)
    print("Finv cumulants_dataset:\n", cumulants_dataset.data.Finv)
    print("Finv bulk cumulants_dataset:\n", bulk_cumulants_dataset.data.Finv)
    print("Finv calculated cumulants_dataset:\n", calculated_cumulants_dataset.data.Finv)

    print(f"Saved figure at: {filename}")

    plt.savefig(filename)
    plt.close()


""" 
    Compare bulk cumulants with means and without means 
"""
verbose = True
if 0:

    bulk_config = bulk_cumulants_config()
    bulk_config.use_bulk_means = False # Override manually in BulkCumulantsDataset call

    full_shape_config = cumulants_config()
    cumulants_dataset = CumulantsDataset(full_shape_config, verbose=verbose)

    # Calculated cumulants for full shape of PDF, using my calculations
    bulk_cumulants_dataset_means = BulkCumulantsDataset(
        bulk_config, 
        pdfs=False, 
        use_mean=True, 
        check_cumulants_against_quijote=False, 
        verbose=verbose
    )

    # Bulk cumulants by my calculations
    bulk_cumulants_dataset_no_means = BulkCumulantsDataset(
        bulk_config, 
        pdfs=False, 
        use_mean=False, 
        check_cumulants_against_quijote=False, 
        verbose=verbose
    )

    parameter_strings = bulk_cumulants_dataset_means.data.parameter_strings

    c = ChainConsumer()
    c.add_chain(
        Chain.from_covariance(
            cumulants_dataset.data.alpha,
            cumulants_dataset.data.Finv,
            columns=parameter_strings,
            name=r"$F_{\Sigma^{-1}}$" + " {}".format("Cumulants[full shape]"),
            color="b",
            shade_alpha=0.1
        )
    )
    c.add_chain(
        Chain.from_covariance(
            bulk_cumulants_dataset_means.data.alpha,
            bulk_cumulants_dataset_means.data.Finv,
            columns=parameter_strings,
            name=r"$F_{\Sigma^{-1}}$" + " {}".format("Cumulants[bulk] with means"),
            color="k",
            shade_alpha=0.
        )
    )
    c.add_chain(
        Chain.from_covariance(
            bulk_cumulants_dataset_no_means.data.alpha,
            bulk_cumulants_dataset_no_means.data.Finv,
            columns=parameter_strings,
            name=r"$F_{\Sigma^{-1}}$" + " {}".format("Cumulants[bulk] no means"),
            color="r",
            shade_alpha=0.1
        )
    )

    c.add_marker(
        location=marker(
            bulk_cumulants_dataset_means.data.alpha, 
            parameter_strings=parameter_strings
        ),
        name=r"$\alpha$", 
        color="#7600bc"
    )
    fig = c.plotter.plot()

    filename = "pdfs_cumulants_Finv_bulk_means.png"

    print("Finv bulk cumulants_dataset (no means):\n", bulk_cumulants_dataset_no_means.data.Finv)
    print("Finv bulk cumulants_dataset (with means):\n", bulk_cumulants_dataset_means.data.Finv)

    print(f"Saved figure at: {filename}")

    plt.savefig(filename)
    plt.close()




# Fisher forecasts
if 0:
   # NOTE: if running on bulk, won't get tails cumulants Finv!

    config_kwargs = dict(
        seed=0, 
        redshift=0.0, 
        reduced_cumulants=False,
        sbi_type="nle",
        linearised=False, 
        compression="linear",
        order_idx=[0,],# 1, 2],
        n_linear_sims=10_000,
        freeze_parameters=False,
        pre_train=False
    )

    pdf_dataset, config = get_dataset_and_config("bulk_pdf") 
    pdf_dataset = pdf_dataset(config(**config_kwargs), pdfs=True)

    bulk_dataset, config = get_dataset_and_config("bulk") 
    bulk_dataset = bulk_dataset(config(**config_kwargs))

    tails_dataset, config = get_dataset_and_config("tails") 
    tails_dataset = tails_dataset(config(**config_kwargs))

    c = ChainConsumer()

    for i, (name, Finv) in enumerate(
        zip(
            [" PDF[bulk]", " $k_n$[bulk]", " $k_n$[tails]"],
            [pdf_dataset.data.Finv, bulk_dataset.data.Finv, tails_dataset.data.Finv]
        )
    ):
        c.add_chain(
            Chain.from_covariance(
                tails_dataset.data.alpha,
                Finv,
                columns=tails_dataset.data.parameter_strings,
                name=r"$F_{\Sigma^{-1}}$" + name,
                shade_alpha=0.
            )
        )
    c.add_marker(
        location=marker(tails_dataset.data.alpha, parameter_strings=tails_dataset.data.parameter_strings),
        name=r"$\alpha$", 
        color="#7600bc"
    )
    fig = c.plotter.plot()
    plt.savefig("Fisher_tests.pdf")
    plt.close()

    c = ChainConsumer()

    target_idx = np.array([0, 4])
    for i, (name, Finv) in enumerate(zip(
        [" PDF[bulk]", " $k_n$[bulk]", " $k_n$[tails]"],
        [pdf_dataset.data.Finv, bulk_dataset.data.Finv, tails_dataset.data.Finv]
    )):
        c.add_chain(
            Chain.from_covariance(
                tails_dataset.data.alpha[target_idx],
                Finv[:, target_idx][target_idx, :],
                columns=[tails_dataset.data.parameter_strings[_] for _ in target_idx],
                name=r"$F_{\Sigma^{-1}}$" + name,
                shade_alpha=0.
            )
        )
    c.add_marker(
        location=marker(
            tails_dataset.data.alpha[target_idx], 
            parameter_strings=[tails_dataset.data.parameter_strings[_] for _ in target_idx]
        ),
        name=r"$\alpha$", 
        color="#7600bc"
    )
    fig = c.plotter.plot()
    plt.savefig("Fisher_tests_marginalised.pdf")
    plt.close()



# Plot bulk and tails cumulants
if 0:
    config_kwargs = dict(
        seed=0, 
        redshift=0.0, 
        reduced_cumulants=False,
        sbi_type="nle",
        linearised=False, 
        compression="linear",
        order_idx=[0, 1, 2],
        n_linear_sims=10_000,
        freeze_parameters=False,
        pre_train=False
    )

    bulk_dataset, config = get_dataset_and_config("bulk") 
    config = config(**config_kwargs)
    bulk_dataset = bulk_dataset(config)

    tails_dataset, config = get_dataset_and_config("tails") 
    config = config(**config_kwargs)
    tails_dataset = tails_dataset(config)

    n_cumulants = 3
    n_scales = len(config.scales)
    n_bins = 32

    if 0:
        # Plot fiducial cumulants (bulk and tails) SEPARATELY
        for bulk_or_tails, dataset in zip(
            ["bulk", "tails"], [bulk_dataset, tails_dataset]
        ):

            _n_cumulants = 5 if bulk_or_tails == "bulk" else 3

            cumulant_strings = [
                r"$\langle\delta^0\rangle$", 
                r"$\langle\delta\rangle$", 
                r"$\langle\delta^2\rangle$",
                r"$\langle\delta^3\rangle_c$",
                r"$\langle\delta^4\rangle_c$"
            ]

            cumulant_strings = cumulant_strings[-_n_cumulants:]

            fig, axs = plt.subplots(n_scales, _n_cumulants, figsize=(10., 27.), dpi=200)
            for r in range(n_scales):
                for c in range(_n_cumulants):
                    ax = axs[r, c]
                    ax.set_title("{}, R={}".format(cumulant_strings[c], config.scales[r]))
                    _cumulants_r = dataset.data.fiducial_data[:, r : (r + 1) * n_scales]
                    _cumulants = _cumulants_r[:, c]
                    _cumulants = (_cumulants - np.mean(_cumulants, axis=0)) / np.std(_cumulants, axis=0)
                    ax.hist(_cumulants)
            plt.savefig("fiducial_cumulants_{}.png".format(bulk_or_tails), bbox_inches="tight")
            plt.close()

    # Plot fiducial cumulants TOGETHER
    fig, axs = plt.subplots(n_scales, 5, figsize=(15., 25.), dpi=200, sharex=True, sharey=True)

    cumulant_strings = [
        r"$\langle\delta^0\rangle$", 
        r"$\langle\delta\rangle$", 
        r"$\langle\delta^2\rangle$",
        r"$\langle\delta^3\rangle_c$",
        r"$\langle\delta^4\rangle_c$"
    ]
    
    lim = 7. # Limit for x-axis

    def gaussian(mu, std):
        x = np.linspace(-lim, +lim, 10000)
        return x, jax.scipy.stats.norm.pdf(x, loc=mu, scale=std)

    # Bulk
    for r in range(n_scales):
        for c in range(5):
            ax = axs[r, c]
            ax.set_title("{}, R={} Mpc/h".format(cumulant_strings[c], config.scales[r]))
            _cumulants = bulk_dataset.data.fiducial_data[:, r * n_cumulants + c : (c + 1) + r * n_cumulants]
            mu = np.mean(_cumulants, axis=0)
            std = np.std(_cumulants, axis=0)
            _cumulants = (_cumulants - mu) / std
            ax.hist(_cumulants, color="royalblue", bins=n_bins, density=True, label="Bulk $k_n$")
            ax.plot(*gaussian(0., 1.), color="k", label="Gaussian")

    # Tails
    for r in range(n_scales):
        for c in range(3): # Tails don't use M_0 or M_1
            ax = axs[r, 2 + c]
            ax.set_title("{}, R={} Mpc/h".format(cumulant_strings[2 + c], config.scales[r]))
            _cumulants = tails_dataset.data.fiducial_data[:, r * n_cumulants + c : r * n_cumulants + (c + 1)]
            mu = np.mean(_cumulants, axis=0)
            std = np.std(_cumulants, axis=0)
            _cumulants = (_cumulants - mu) / std
            ax.hist(_cumulants, color="firebrick", histtype="step", bins=n_bins, density=True, label="Tails $k_n$")

    axs[0, 3].legend(frameon=False, loc="upper left")

    for ax in axs.ravel():
        ax.set_xlim(-lim, lim)

    plt.savefig("fiducial_cumulants_bulk_and_tails.png", bbox_inches="tight")
    plt.close()




""" 
    Compare bulk cumulants with means/norms and without 
"""
verbose = False
if 1:

    bulk_config = bulk_cumulants_config()

    bulk_config.use_bulk_means = True 
    bulk_config.stack_bulk_means = True
    bulk_config.stack_bulk_norms = True 

    # Calculated cumulants for full shape of PDF, using my calculations
    bulk_cumulants_dataset_means = BulkCumulantsDataset(
        bulk_config, 
        pdfs=False, 
        check_cumulants_against_quijote=False, 
        verbose=verbose
    )

    bulk_config = bulk_cumulants_config()

    bulk_config.stack_bulk_means = False
    bulk_config.stack_bulk_norms = False

    # Bulk cumulants by my calculations
    bulk_cumulants_dataset_no_means = BulkCumulantsDataset(
        bulk_config, 
        pdfs=False, 
        check_cumulants_against_quijote=False, 
        verbose=verbose
    )

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axs = plt.subplots(1, 2, figsize=(13., 5.))
    datasets = [bulk_cumulants_dataset_means, bulk_cumulants_dataset_no_means]
    for i, ax in enumerate(axs):
        fiducial_data = datasets[i].data.fiducial_data
        corr = np.corrcoef(fiducial_data, rowvar=False)
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1., vmax=1.)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        n = corr.shape[0]
        for pos in range(n):
            if i == 0:
                if (pos - 2) % 3 == 0: # Plot green lines for same as second plot
                    ax.axhline(pos - 0.5, color='green', linewidth=1.)
                    ax.axvline(pos - 0.5, color='green', linewidth=1.)
                if pos % 5 == 0:
                    ax.axhline(pos - 0.5, color='black', linewidth=1.)
                    ax.axvline(pos - 0.5, color='black', linewidth=1.)
            elif i == 1:
                if pos % 3 == 0:
                    ax.axhline(pos - 0.5, color='green', linewidth=1.)
                    ax.axvline(pos - 0.5, color='green', linewidth=1.)

    plt.savefig("corr_matrices.png", bbox_inches="tight")
    plt.close()