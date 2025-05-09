import argparse
from typing import Tuple
from collections import namedtuple
import os
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key
import numpy as np
from ml_collections import ConfigDict
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer, Chain, Truth
from tensorflow_probability.substrates.jax.distributions import Distribution

from sbiax.ndes import CNF, MAF, Scaler
from sbiax.inference import nuts_sample
from sbiax.utils import make_df, marker

from configs import (
    cumulants_config, 
    bulk_cumulants_config, 
    get_posteriors_dir
)
from configs.args import (
    get_cumulants_sbi_args, get_cumulants_multi_z_args
)
from configs.configs import (
    get_base_results_dir, 
    get_results_dir, 
    get_multi_z_posterior_dir, 
    get_ndes_from_config
)
from configs.ensembles_configs import (
    ensembles_cumulants_config, ensembles_bulk_cumulants_config 
)
from data.constants import (
    get_quijote_parameters, 
    get_base_posteriors_dir,
    get_save_and_load_dirs,
    get_target_idx
)
from data.cumulants import (
    Dataset, 
    get_data, 
    get_linear_compressor, 
    get_datavector, 
    get_prior, 
    get_parameter_strings
)
from data.pdfs import get_multi_z_bulk_pdf_fisher_forecast
from cumulants_ensemble import Ensemble, MultiEnsemble
from affine import affine_sample

jax.clear_caches()

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

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s", 
    "--seed", 
    type=int, 
    help="Seed for random number generation.", 
    default=0
)
parser.add_argument(
    "-s_d", 
    "--seed_datavector", 
    type=int, 
    help="Seed for random number generation.", 
    default=0
)
parser.add_argument(
    "-l",
    "--linearised", 
    default=True,
    action=argparse.BooleanOptionalAction, 
    help="Linearised model for datavector."
)
parser.add_argument(
    "-c",
    "--compression", 
    default="linear",
    choices=["linear", "nn", "nn-lbfgs"],
    type=str,
    help="Compression with neural network or MOPED."
)
parser.add_argument(
    "-p",
    "--pre-train", 
    default=False,
    action=argparse.BooleanOptionalAction, 
    help="Pre-train (only) when using non-linearised model for datavector. Pre-train on linearised simulations."
)
parser.add_argument(
    "-o", 
    "--order_idx",
    default=[0, 1, 2],
    nargs="+", 
    type=int,
    help="Indices of variance, skewness and kurtosis sample cumulants."
)
parser.add_argument(
    "-t",
    "--sbi_type", 
    default="nle",
    choices=["nle", "npe"],
    type=str,
    help="Method of SBI: neural likelihood (NLE) or posterior (NPE)."
)
parser.add_argument(
    "-f",
    "--freeze-parameters", 
    default=False,
    action=argparse.BooleanOptionalAction, 
    help="Freeze parameters not in [Om, s8] to their fixed values, in hypercube simulations."
)
ARGS = parser.parse_args()

# General constants
data_dir, _, _ = get_save_and_load_dirs()

(
    _, _, _, alpha, lower, upper, parameter_strings, *_
) = get_quijote_parameters()

figs_dir = os.path.join(get_base_results_dir(), "figure_one/")
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir, exist_ok=True)

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

# Get the bulk PDF Fisher forecast for all redshifts 
# (easier to load frozen or not since it autosaves...)
try:
    Finv_bulk_pdfs_all_z = np.load(
        os.path.join(
            data_dir, 
            "Finv_bulk_pdfs_all_z_{}.npy".format(
                "f" if args.freeze_parameters else "nf"
            )
        )
    )
except:
    Finv_bulk_pdfs_all_z = get_multi_z_bulk_pdf_fisher_forecast(args)

    np.save(
        os.path.join(
            data_dir, 
            "Finv_bulk_pdfs_all_z_{}.npy".format(
                "f" if args.freeze_parameters else "nf"
            )
        ),
        Finv_bulk_pdfs_all_z
    )

print("FINV_BULK_PDFS_ALL_Z", Finv_bulk_pdfs_all_z.shape)

posterior_objects = dict(bulk=None, tails=None)

# Loop through bulk / tails (just grab PDF Fisher forecast, no posterior)
for bulk_or_tails in ["bulk", "tails"]:

    # Multi-z inference concerning the bulk or bulk + tails
    if bulk_or_tails == "tails":
        ensembles_config = ensembles_cumulants_config
    if bulk_or_tails == "bulk": # or bulk_or_tails == "bulk_pdf":
        ensembles_config = ensembles_bulk_cumulants_config

    # Force args for posterior to be bulk or tails (for posterior save dir)
    args.bulk_or_tails = bulk_or_tails 

    config = ensembles_config(
        seed=args.seed, # Defaults if run without argparse args
        sbi_type=args.sbi_type, 
        linearised=args.linearised,
        reduced_cumulants=args.reduced_cumulants,
        order_idx=args.order_idx,
        redshifts=args.redshifts,
        compression=args.compression,
        n_linear_sims=args.n_linear_sims,
        freeze_parameters=args.freeze_parameters,
        pre_train=args.pre_train
    )

    # Posterior for bulk/tails for a given seed
    posterior_save_dir = get_multi_z_posterior_dir(config, args)
    posterior_filename = os.path.join(
        posterior_save_dir, 
        "posterior_{}{}.npz".format(
            args.seed, 
            ("_" + str(args.seed_datavector)) if args.seed_datavector is not None else ""
        ) 
    )
    posterior_file = np.load(posterior_filename)
    posterior_object = get_posterior_object(posterior_file)

    posterior_objects[bulk_or_tails] = posterior_object

    print("POSTERIOR OBJECT", jax.tree.map(lambda x: x.shape, posterior_object))

# Plot the posteriors for SBI on the bulk and tails, bulk PDF Fisher 

PLOT_SUMMARIES = False

target_idx = get_target_idx()

def maybe_marginalise(posterior_object, alpha, parameter_strings, Finv_bulk_pdfs_all_z, marginalise):
    if marginalise:
        posterior_object = posterior_object._replace(
            Finv=posterior_object.Finv[target_idx, :][:, target_idx]
        )
        posterior_object = posterior_object._replace(
            samples=posterior_object.samples[:, target_idx]
        )
        posterior_object = posterior_object._replace(
            summary=posterior_object.summary[:, target_idx] # NOTE: check shape... (n, 5)
        ) 
        alpha = alpha[target_idx] 
        parameter_strings = [parameter_strings[t] for t in target_idx]
        Finv_bulk_pdfs_all_z = Finv_bulk_pdfs_all_z[target_idx, :][:, target_idx] 
    return posterior_object, alpha, parameter_strings, Finv_bulk_pdfs_all_z

for marginalised in [True, False]:

    # Don't plot marginalised posterior if freezing parameters, same effect...
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

        # Compressed datavectors (assuming more than one of them)
        if PLOT_SUMMARIES:
            for n, _summary in enumerate(_posterior_object.summary):
                c.add_marker(
                    location=marker(_summary, _parameter_strings), 
                    name=r"$\hat{\pi}[\hat{\xi}]$ " + str(n) + title, 
                    color=plotting_dict[bulk_or_tails]["color"],
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
        color="#7600bc"
    )

    fig = c.plotter.plot()
    fig.suptitle(
        r"{} SBI (bulk & tails) & $F_{{\Sigma}}^{{-1}}$".format(
            "$k_n/k_2^{n-1}$" if config.reduced_cumulants else "$k_n$"
        ) + "\n" +
        "{} z={},\n $n_s$={}, (pre-train $n_s$={}),\n R={} Mpc,\n $k_n$={}".format(
                ("linearised" if config.linearised else "non-linear") + "\n",
                "[{}]".format(", ".join(map(str, config.redshifts))),
                config.n_linear_sims if config.linearised else 2000, 
                config.n_linear_sims if config.pre_train else None,
                "[{}]".format(", ".join(map(str, config.scales))),
                "[{}]".format(", ".join(map(str, [["var.", "skew.", "kurt."][_] for _ in config.order_idx])))
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
        sub_figs_dir, "figure_one_{}{}.pdf".format(args.seed, "_marginalised" if marginalised else "")
    )

    plt.savefig(filename)
    plt.close()

    print("Saved figure one to {}".format(filename))