import os
import time
import datetime

import jax
import jax.numpy as jnp
import jax.random as jr

import numpy as np 
import matplotlib.pyplot as plt
from chainconsumer import Chain, ChainConsumer, Truth

from sbiax.utils import make_df, marker

from configs import get_results_dir
from configs.log import setup_module_logger, get_log_level
from configs.args import get_cumulants_sbi_args
from data.common import Dataset
from data.constants import ALPHA, LOWER, UPPER, PARAMETER_STRINGS
from compression import get_compression_fn
from utils import (
    get_datasets,
    plot_cumulants,
    plot_moments, 
    plot_latin_moments, 
    plot_summaries, 
    plot_summaries_fiducial,
    overlay_bounds_on_corner,
    customize_plot
)


jax.clear_caches()

logger, log_figs_dir = setup_module_logger(__name__, level=get_log_level())

N_NUTS_SAMPLES = 10_000
N_FISHER_SAMPLES = 800_000
SAMPLE_POSTERIORS = True # Just leave it to multi-z, where debugging plots sample the posteriors anyway

""" 
    Run NLE or NPE SBI with the moments of the 1pt matter PDF.

    - diagonal of covariance for compression?
    - freezing 'nuisance parameters'
    - covariance conditioning?
    - remove outliers in latins?
""" 

t0 = time.time()

args = get_cumulants_sbi_args()

print("TIME:", datetime.datetime.now().strftime("%H:%M %d-%m-%y"))
print("SEED:", args.seed)
print("MOMENTS:", args.order_idx)
print("LINEARISED:", args.linearised)

"""
    Config
"""

config, cumulants_dataset, datasets = get_datasets(args) # Config and cumulants_dataset can be bulk ... etc

key = jr.key(config.seed)

(
    model_key, train_key, key_prior, 
    key_datavector, key_state, key_sample
) = jr.split(key, 6)

_seed = int(time.time()) + int(os.getpid())  # or + int(os.getenv("SLURM_JOB_ID", 0))
key_datavector = jr.key(_seed)

results_dir = get_results_dir(config, args)

dataset: Dataset = cumulants_dataset.data

plot_cumulants(args, config, dataset.fiducial_data, results_dir=results_dir)

################################ Check fisher forecasts

if 1:

    c = ChainConsumer()

    for i, (dataset_type, name) in enumerate(zip(
        ["bulk_pdf", "bulk", "tails"],
        [" PDF[bulk]", " $k_n$[bulk]", " $k_n$[tails]"],
    )):
        c.add_chain(
            Chain.from_covariance(
                datasets[dataset_type].data.alpha,
                datasets[dataset_type].data.Finv,
                columns=PARAMETER_STRINGS,
                name=r"$F_{\Sigma^{-1}}$" + name,
                shade_alpha=0.
            )
        )
    c.add_marker(
        location=marker(ALPHA, parameter_strings=PARAMETER_STRINGS),
        name=r"$\alpha$", 
        color="#7600bc"
    )
    fig = c.plotter.plot()
    overlay_bounds_on_corner(fig, LOWER, UPPER)
    fig = customize_plot(fig)
    plt.savefig(os.path.join(log_figs_dir, "Fisher_tests_z{}.pdf".format(args.redshift)))
    plt.close()

    plt.figure()
    corr = jnp.corrcoef(dataset.fiducial_data, rowvar=False)
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1., vmax=1.)
    plt.colorbar(im)
    plt.savefig(
        os.path.join(log_figs_dir, "correlation_matrix_cumulants_{}.png".format(args.bulk_or_tails))
    )
    plt.close()

    plt.figure()
    im = plt.imshow(dataset.C)
    plt.colorbar(im)
    plt.savefig(
        os.path.join(log_figs_dir, "covariance_matrix_cumulants_{}.png".format(args.bulk_or_tails))
    )
    plt.close()

    plt.figure()
    im = plt.imshow(dataset.Cinv)
    plt.colorbar(im)
    plt.savefig(
        os.path.join(log_figs_dir, "precision_matrix_cumulants_{}.png".format(args.bulk_or_tails))
    )
    plt.close()

    logger.info("Covariance condition number: {:.3E}".format(jnp.linalg.cond(dataset.C)))

################################

"""
    Compression
"""

# Compress simulations
try:
    compression_fn = get_compression_fn(
        key, 
        config, 
        dataset=cumulants_dataset.data, 
        train=False, 
        results_dir=results_dir
    )

    print("Loaded NN...")
except FileNotFoundError as e:
    print("Exception: \n\t{}\n\tTraining NN...".format(e))

    compression_fn = get_compression_fn(
        key, 
        config, 
        dataset=cumulants_dataset.data, 
        train=True, 
        results_dir=results_dir
    )

    print("Training NN...")

X = jax.vmap(compression_fn)(dataset.data, dataset.parameters)

X0 = jax.vmap(compression_fn, in_axes=(0, None))(dataset.fiducial_data, dataset.alpha)
datavectors = cumulants_dataset.get_datavector(key, n=10_000)
summaries = jax.vmap(compression_fn, in_axes=(0, None))(datavectors, dataset.alpha)

print("SHAPES OF DATAVECTORS:")
print(jax.tree.map(lambda a: a.shape, (X, X0, datavectors, summaries)))

np.savez(
    os.path.join(results_dir, "all_summaries.npz"), 
    latins=X, 
    parameters=dataset.parameters,
    fiducials=X0, 
    summaries=summaries, # Save a very large number of datavectors, use them in multi-z
    datavectors=datavectors
)

np.savez(
    os.path.join(results_dir, "sbi_dataset.npz"),
    latins=X, 
    parameters=dataset.parameters,
    fiducials=X0,
    datavector=datavectors[-1],
    summary=summaries[-1] # Choose random last datavector for SBI runs
)

# Plot summaries
plot_summaries(X, dataset.parameters, dataset, results_dir)

X_ = jax.vmap(compression_fn, in_axes=(0, None))(cumulants_dataset.get_datavector(key, n=1000), dataset.alpha)
plot_summaries_fiducial(
    X0, 
    X_,
    dataset.alpha, 
    dataset, 
    results_dir,
    Finv=dataset.Finv
)

plot_moments(dataset.fiducial_data, config, results_dir)

plot_latin_moments(dataset.data, config, results_dir)