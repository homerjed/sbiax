import argparse
from collections import namedtuple
import os
import jax
import numpy as np
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer, Chain

from sbiax.utils import make_df, marker

from configs.args import (
    get_cumulants_multi_z_args,
    get_figure_one_args
)
from configs.configs import (
    get_base_results_dir, 
    get_multi_z_posterior_dir, 
)
from configs.ensembles_configs import (
    ensembles_cumulants_config, ensembles_bulk_cumulants_config 
)
from data.constants import (
    get_quijote_parameters, 
    get_save_and_load_dirs,
    get_target_idx
)
from data.pdfs import load_multi_z_bulk_pdf_fisher_forecast

jax.clear_caches()

PLOT_SUMMARIES = False

target_idx = get_target_idx()

def customize_plot(fig, lw=1.5, fs=16):
    fig.set_size_inches(6., 6.)
    fig.set_dpi(200)

    # Loop over axes to customize them
    for ax in fig.axes:
        # Change axis label font sizes
        ax.xaxis.label.set_size(fs)
        ax.yaxis.label.set_size(fs)

        # Change tick label font sizes
        ax.tick_params(axis='both', labelsize=fs - 2)

        # Change spline (axis spine) linewidths
        for spine in ax.spines.values():
            spine.set_linewidth(lw)

        # Change contour line widths (if any exist)
        for coll in ax.collections:
            if hasattr(coll, 'get_linewidths'):
                coll.set_linewidths([lw])  # or another desired width

        # Identify diagonal axes
        for line in ax.lines:
            line.set_linewidth(lw)  # set your desired linewidth here

        # Legend fontsize
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(fs) 

        # Marker sizes
        for coll in ax.collections:
            if hasattr(coll, 'get_sizes'):  # Check if this is a PathCollection (e.g. scatter/marker)
                sizes = coll.get_sizes()
                if len(sizes) > 0:
                    # Set new marker size (squared points); e.g., 50 means ~7 px
                    coll.set_sizes([50] * len(sizes))
    return fig

"""
    Loop through seeds, getting...
    - configs for ensembles for 
        - bulk and bulk + tails
        -over all redshifts, 
    ...loading posteriors from them.
    Then plot posteriors together with the bulk PDF Fisher forecast.
"""

def get_posterior_object(posterior_file):
    # Create posterior object from .npz posterior file that contains samples, log prob, Finv, summary, ...
    PosteriorTuple = namedtuple("PosteriorTuple", posterior_file.files)
    posterior_tuple = PosteriorTuple(*(posterior_file[key] for key in posterior_file.files))
    return posterior_tuple

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

figs_dir = os.path.join(get_base_results_dir(), "figure_one/")
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir, exist_ok=True)

ARGS = get_figure_one_args() # Args for figure one

args = get_cumulants_multi_z_args() # Blueprint args for analysis

# Plotting properties for bulk / tails
plotting_dict = dict(
    bulk=dict(color="b", linestyle="-", shade_alpha=0.5),
    tails=dict(color="r", linestyle="-", shade_alpha=0.5)
)

# Args that are shared between bulk and tails SBI analyses/posteriors
args.seed = ARGS.seed
args.linearised = ARGS.linearised 
args.pre_train = ARGS.pre_train
args.order_idx = ARGS.order_idx
args.freeze_parameters = ARGS.freeze_parameters
args.n_linear_sims = ARGS.n_linear_sims

posterior_objects = dict(bulk=None, tails=None)

# Loop through bulk / tails (just grab PDF Fisher forecast, no posterior for PDFs)
for bulk_or_tails in ["bulk", "tails"]:

    # Force args for posterior to be bulk or tails (for posterior save dir)
    args.bulk_or_tails = bulk_or_tails 

    # Posterior for bulk/tails for a given seed
    posterior_save_dir = get_multi_z_posterior_dir(args)
    posterior_filename = os.path.join(
        posterior_save_dir, 
        "multi_z_posterior_{}{}.npz".format(
            args.seed, 
            ("_" + str(args.seed_datavector)) if args.seed_datavector is not None else ""
        ) 
    )
    posterior_file = np.load(posterior_filename)
    posterior_object = get_posterior_object(posterior_file)

    posterior_objects[bulk_or_tails] = posterior_object

    print("POSTERIOR OBJECT", jax.tree.map(lambda x: x.shape, posterior_object))

# Get the bulk PDF Fisher forecast for all redshifts 
# (easier to load frozen or not since it autosaves...)
Finv_bulk_pdfs_all_z = load_multi_z_bulk_pdf_fisher_forecast(data_dir, args)

""" 
    Plot the posteriors for SBI on the bulk and tails, bulk PDF Fisher 
"""

def maybe_marginalise(posterior_object, alpha, parameter_strings, Finv_bulk_pdfs_all_z, marginalise):
    # Marginalise posterior object if required
    if marginalise:
        posterior_object = posterior_object._replace(
            Finv=posterior_object.Finv[target_idx, :][:, target_idx]
        )
        posterior_object = posterior_object._replace(
            samples=posterior_object.samples[:, target_idx]
        )
        alpha = alpha[target_idx] 
        parameter_strings = [parameter_strings[t] for t in target_idx]
        Finv_bulk_pdfs_all_z = Finv_bulk_pdfs_all_z[target_idx, :][:, target_idx] 
    return posterior_object, alpha, parameter_strings, Finv_bulk_pdfs_all_z

# Load posteriors from bulk / tails for marginalised and non-marginalised cases
for marginalised in [True, False]:

    # Don't plot marginalised posterior if freezing parameters, 'same' effect...
    if marginalised and args.freeze_parameters:
        continue

    # Plot 
    c = ChainConsumer() 

    for bulk_or_tails in ["bulk", "tails"]:

        title = "$k_n$[{}]".format(bulk_or_tails) 

        _posterior_object = posterior_objects[bulk_or_tails]

        # Marginalise relevant posterior objects if required
        (
            _posterior_object, 
            _alpha,
            _parameter_strings,
            _Finv_bulk_pdfs_all_z,
        ) = maybe_marginalise(
            _posterior_object, 
            alpha, 
            parameter_strings, 
            Finv_bulk_pdfs_all_z, 
            marginalise=marginalised
        )

        if args.freeze_parameters: 
            _alpha = alpha[target_idx]
            _parameter_strings = [parameter_strings[t] for t in target_idx]
        
        print(
            "_alpha", _alpha.shape, 
            "_posterior_object", jax.tree.map(lambda x: x.shape, _posterior_object), 
            "_Finv_bulk", _Finv_bulk_pdfs_all_z.shape
        )

        # Fisher forecast for bulk or tails
        c.add_chain(
            Chain.from_covariance(
                _alpha,
                _posterior_object.Finv, # NOTE: Get multi redshift Fisher matrix, use a multi-inference config
                columns=_parameter_strings,
                name=r"$F_{\Sigma^{-1}}$ " + title,
                color=plotting_dict[bulk_or_tails]["color"],
                linestyle=":",
                shade_alpha=0.
            )
        )

        # Posterior from SBI on bulk or tails
        posterior_df = make_df(
            _posterior_object.samples, 
            _posterior_object.samples_log_prob, 
            parameter_strings=_parameter_strings
        )
        c.add_chain(
            Chain(
                samples=posterior_df, name="SBI " + title, 
                color=plotting_dict[bulk_or_tails]["color"],
                linestyle=plotting_dict[bulk_or_tails]["linestyle"],
                shade_alpha=plotting_dict[bulk_or_tails]["shade_alpha"],
            )
        )

        print("POSTERIOR OBJECT SUMMARIES SHAPE", _posterior_object.summaries.shape)

        # Compressed datavectors (assuming more than one of them)
        if PLOT_SUMMARIES:
            # for n_z, _summaries in enumerate(_posterior_object.summaries):
            #     for n, _summary in enumerate(_summaries):
            #         c.add_marker(
            #             location=marker(_summary, _parameter_strings), 
            #             name=r"$\hat{\pi}[\hat{\xi}]$ " + "z={}, n={}".format(args.redshifts[n_z], n) + title + n * " ", # Whitespace for unique name?
            #             color=plotting_dict[bulk_or_tails]["color"],
            #             show_label_in_legend=False if n > 0 else True
            #         )
            c.add_marker(
                location=marker(np.mean(np.mean(_posterior_object.summaries, axis=0), axis=0), _parameter_strings), 
                name=r"$\bar{\pi}[\hat{\xi}_i,...]$ " + title, # Whitespace for unique name?
                color=plotting_dict[bulk_or_tails]["color"]
            )

    # Scale Fisher matrix for bulk PDF by number of datavectors (already done for other Finvs)
    # POSTERIOR OBJECT SUMMARIES SHAPE (3, 10, 5)
    _, n_datavectors, _ = _posterior_object.summaries.shape
    _Finv_bulk_pdfs_all_z = _Finv_bulk_pdfs_all_z / n_datavectors 

    # Fisher forecast for bulk of PDF over all redshifts
    c.add_chain(
        Chain.from_covariance(
            _alpha,
            _Finv_bulk_pdfs_all_z, 
            columns=_parameter_strings,
            name=r"$F_{\Sigma^{-1}}$ PDF[bulk]",
            color="g",
            linestyle=":",
            shade_alpha=0.
        )
    )

    # True parameters
    c.add_marker(
        location=marker(_alpha, _parameter_strings), 
        name=r"$\alpha$", 
        color="#7600bc",
        marker_style="x"
    )

    fig = c.plotter.plot()
    fig = customize_plot(fig)
    fig.suptitle(
        r"{} SBI (bulk & tails) & $F_{{\Sigma}}^{{-1}}$".format("$k_n$") + "\n" +
        "{} z={},\n $n_s$={}, (pre-train $n_s$={}),\n R={} Mpc,\n $k_n$={}".format(
                ("linearised" if args.linearised else "non-linear") + "\n",
                "[{}]".format(", ".join(map(str, args.redshifts))),
                args.n_linear_sims if args.linearised else 2000, 
                args.n_linear_sims if args.pre_train else None,
                "[{}]".format(", ".join(map(str, args.scales))),
                "[{}]".format(", ".join(map(str, [["var.", "skew.", "kurt."][_] for _ in args.order_idx])))
            ),
        multialignment='center'
    )

    # Naming convention for figure one
    sub_figs_dir = os.path.join(
        figs_dir, 
        "frozen/" if args.freeze_parameters else "nofrozen/", 
        "linearised/" if args.linearised else "nonlinearised/", 
        "pretrain/" if args.pre_train else "nopretrain/", 
        "m{}/".format("".join(map(str, args.order_idx)))
    )
    if not os.path.exists(sub_figs_dir):
        os.makedirs(sub_figs_dir, exist_ok=True)

    filename = os.path.join(
        sub_figs_dir, 
        "figure_one_{}{}{}.pdf".format(
            args.seed, 
            "_marginalised" if marginalised else "",
            ("_" + str(args.seed_datavector)) if args.seed_datavector is not None else ""
        )
    )

    plt.savefig(filename)
    plt.close()

    print("Saved figure one to {}".format(filename))