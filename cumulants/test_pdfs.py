# import jax
# jax.config.update("jax_debug_nans", True)

import matplotlib.pyplot as plt
import numpy as np
from chainconsumer import Chain, ChainConsumer, Truth

from configs.cumulants_configs import cumulants_config, bulk_cumulants_config
from data.pdfs import BulkCumulantsDataset
from data.cumulants import CumulantsDataset
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
        use_mean=use_mean, 
        check_cumulants_against_quijote=True, 
        verbose=verbose
    )

    # Bulk cumulants by my calculations
    bulk_cumulants_dataset = BulkCumulantsDataset(
        bulk_config, 
        pdfs=False, 
        use_mean=use_mean, 
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
if 1:

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