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
    get_multi_z_posterior_filename 
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

def maybe_marginalise(
    posterior_object, 
    alpha, 
    parameter_strings, 
    Finv_bulk_pdfs_all_z, 
    *,
    marginalise, 
    target_idx
):
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

figure_one_args = get_figure_one_args() # Args for figure one

multi_z_args = get_cumulants_multi_z_args(figure_one=True) # Blueprint multi_z_args for analysis

# Plotting properties for bulk / tails
plotting_dict = dict(
    bulk=dict(color="b", linestyle="-", shade_alpha=0.5),
    tails=dict(color="r", linestyle="-", shade_alpha=0.5)
)

# Args that are shared between bulk and tails SBI analyses/posteriors
multi_z_args.seed              = figure_one_args.seed
multi_z_args.seed_datavector   = figure_one_args.seed_datavector
multi_z_args.n_datavectors     = figure_one_args.n_datavectors
multi_z_args.scales            = figure_one_args.scales
multi_z_args.linearised        = figure_one_args.linearised 
multi_z_args.pre_train         = figure_one_args.pre_train
multi_z_args.order_idx         = figure_one_args.order_idx
multi_z_args.freeze_parameters = figure_one_args.freeze_parameters
multi_z_args.n_linear_sims     = figure_one_args.n_linear_sims

# Loop through bulk / tails (just grab PDF Fisher forecast, no posterior for PDFs)
posterior_objects = dict(bulk=None, tails=None)
for bulk_or_tails in ["bulk", "tails"]:

    # Force multi_z_args for posterior to be bulk or tails (for posterior save dir)
    multi_z_args.bulk_or_tails = bulk_or_tails 

    # Posterior for bulk/tails for a given seed
    posterior_filename = get_multi_z_posterior_filename(multi_z_args)
    posterior_file = np.load(posterior_filename)
    posterior_object = get_posterior_object(posterior_file)

    posterior_objects[bulk_or_tails] = posterior_object

    print("MULTI-Z POSTERIOR FILENAME:\n", posterior_filename)
    print("POSTERIOR OBJECT", jax.tree.map(lambda x: x.shape, posterior_object))

# Get the bulk PDF Fisher forecast for all redshifts 
# (easier to load frozen or not since it autosaves...)
Finv_bulk_pdfs_all_z = load_multi_z_bulk_pdf_fisher_forecast(data_dir, multi_z_args)
Finv_bulk_pdfs_all_z = Finv_bulk_pdfs_all_z / multi_z_args.n_datavectors 

""" 
    Plot the posteriors for SBI on the bulk and tails, bulk PDF Fisher 
"""

# Load posteriors from bulk / tails for marginalised and non-marginalised cases
for marginalised in [True, False]:

    # Don't plot marginalised posterior if freezing parameters, 'same' effect...
    if marginalised and multi_z_args.freeze_parameters:
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
            target_idx=target_idx,
            marginalise=marginalised
        )

        if multi_z_args.freeze_parameters: 
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
            #             name=r"$\hat{\pi}[\hat{\xi}]$ " + "z={}, n={}".format(multi_z_args.redshifts[n_z], n) + title + n * " ", # Whitespace for unique name?
            #             color=plotting_dict[bulk_or_tails]["color"],
            #             show_label_in_legend=False if n > 0 else True
            #         )
            c.add_marker(
                location=marker(np.mean(np.mean(_posterior_object.summaries, axis=0), axis=0), _parameter_strings), 
                name=r"$\bar{\pi}[\hat{\xi}_i,...]$ " + title, # Whitespace for unique name?
                color=plotting_dict[bulk_or_tails]["color"]
            )

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
                ("linearised" if multi_z_args.linearised else "non-linear") + "\n",
                "[{}]".format(", ".join(map(str, multi_z_args.redshifts))),
                multi_z_args.n_linear_sims if multi_z_args.linearised else 2000, 
                multi_z_args.n_linear_sims if multi_z_args.pre_train else None,
                "[{}]".format(", ".join(map(str, multi_z_args.scales))),
                "[{}]".format(", ".join(map(str, [["var.", "skew.", "kurt."][_] for _ in multi_z_args.order_idx])))
            ),
        multialignment='center'
    )

    # Naming convention for figure one
    sub_figs_dir = os.path.join(
        figs_dir, 
        "frozen/" if multi_z_args.freeze_parameters else "nofrozen/", 
        "linearised/" if multi_z_args.linearised else "nonlinearised/", 
        "pretrain/" if multi_z_args.pre_train else "nopretrain/", 
        "m{}/".format("".join(map(str, multi_z_args.order_idx)))
    )
    if not os.path.exists(sub_figs_dir):
        os.makedirs(sub_figs_dir, exist_ok=True)

    filename = os.path.join(
        sub_figs_dir, 
        "figure_one_{}{}{}.pdf".format(
            multi_z_args.seed, 
            "_marginalised" if marginalised else "",
            ("_" + str(multi_z_args.seed_datavector)) if multi_z_args.seed_datavector is not None else ""
        )
    )

    plt.savefig(filename)
    plt.close()

    print("Saved figure one to {}".format(filename))