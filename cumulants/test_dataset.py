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
from utils import (
    get_datasets,
    get_dataset_and_config,
    plot_cumulants,
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

"""
    Config
"""
# # Run just PDFs
# if 1:
#     _dataset, _config = get_dataset_and_config("bulk_pdf") 

#     config = _config(
#         seed=args.seed, 
#         redshift=args.redshift, 
#         linearised=args.linearised, 
#         compression=args.compression,
#         order_idx=args.order_idx,
#         scales=args.scales,
#         n_linear_sims=args.n_linear_sims,
#         pre_train=args.pre_train
#     )

#     results_dir = get_results_dir(config, args)

#     _dataset(config, results_dir=results_dir)

#     import sys
#     sys.exit()

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
    os.path.join(log_figs_dir, "correlation_matrix_cumulants_{}_{}.png".format(args.bulk_or_tails, args.redshift))
)
plt.close()

plt.figure()
im = plt.imshow(dataset.C)
plt.colorbar(im)
plt.savefig(
    os.path.join(log_figs_dir, "covariance_matrix_cumulants_{}_{}.png".format(args.bulk_or_tails, args.redshift))
)
plt.close()

plt.figure()
im = plt.imshow(dataset.Cinv)
plt.colorbar(im)
plt.savefig(
    os.path.join(log_figs_dir, "precision_matrix_cumulants_{}_{}.png".format(args.bulk_or_tails, args.redshift))
)
plt.close()

logger.info("Covariance condition number: {:.3E}".format(jnp.linalg.cond(dataset.C)))