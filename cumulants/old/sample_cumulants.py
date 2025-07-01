# import jax
# jax.config.update("jax_debug_nans", True)

import os

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np
from chainconsumer import Chain, ChainConsumer, Truth
from tqdm.auto import trange

from configs import get_results_dir
from configs.configs import get_ndes_from_config
from configs.args import get_cumulants_sbi_args
from configs.cumulants_configs import cumulants_config, bulk_cumulants_config
from data.pdfs import BulkCumulantsDataset
from data.cumulants import CumulantsDataset
from cumulants_ensemble import Ensemble
from sbiax.ndes import CNF, MAF, Scaler
from sbiax.utils import make_df, marker

""" 
    Sample cumulants from the generative models at fiducial parameters, e.g. redshift zero; 
    and compare with Quijote.
    - Ensembles => plot a histogram per model in the ensemble
    - do it for bulk and tails separately?

    - for loop over redshift? just redshift zero is the most interesting
"""

key = jr.key(0)

key, key_model = jr.split(key)

args = get_cumulants_sbi_args()

config = cumulants_config()
# bulk_config = bulk_cumulants_config()

# Calculate full shape cumulants from quijote pdfs, compare to pdf measured in quijote
verbose = True

# Quijote cumulants for full shape of PDF
cumulants_dataset = CumulantsDataset(config, verbose=verbose)

n_fiducial_cumulants = cumulants_dataset.data.fiducial_data.shape[0]

alpha = cumulants_dataset.data.alpha
alphas = jnp.tile(alpha, (n_fiducial_cumulants, 1))

# Get compressed dataset for scaler...
compression_fn = cumulants_dataset.get_compression_fn() 
X = jax.vmap(compression_fn)(cumulants_dataset.data.data, cumulants_dataset.data.parameters)

X_fiducial = jax.vmap(compression_fn)(cumulants_dataset.data.fiducial_data, alphas)

# Scaler for flow model
scaler = Scaler(
    X, 
    cumulants_dataset.data.parameters, 
    use_scaling=config.use_scalers # NOTE: data_preprocess_fn ...
)

# Get NDEs
ndes = get_ndes_from_config(
    config, 
    event_dim=cumulants_dataset.data.alpha.size, 
    scalers=scaler, 
    use_scalers=config.use_scalers,
    key=key_model
)

# Ensemble of NDEs
ensemble = Ensemble(ndes, sbi_type=config.sbi_type)

# Load ensemble
ensemble_path = os.path.join(get_results_dir(config, args=args), "ensemble.eqx")
ensemble = eqx.tree_deserialise_leaves(ensemble_path, ensemble)

sampled_cumulants = []
for i, nde in zip(
    trange(len(ensemble.ndes), desc="Sampling ensemble"), ensemble.ndes
):
    key = jr.fold_in(key, i)

    _, _alpha = scaler.forward(jnp.ones_like(alpha), alpha)

    _sampled_cumulants, _ = nde.sample_and_log_prob_n(
        key, y=_alpha, n_samples=n_fiducial_cumulants
    )

    _sampled_cumulants, _ = jax.vmap(scaler.reverse)(_sampled_cumulants, alphas)

    sampled_cumulants.append(_sampled_cumulants)

sampled_cumulants = jnp.squeeze(jnp.asarray(sampled_cumulants))

c = ChainConsumer()
c.add_chain(
    Chain(
        samples=make_df(X_fiducial, parameter_strings=cumulants_dataset.data.parameter_strings), 
        name=r"Quijote $\hat{\pi}$", 
        color="blue", 
        plot_cloud=True, 
        plot_contour=False
    )
)
c.add_chain(
    Chain(
        samples=make_df(sampled_cumulants, parameter_strings=cumulants_dataset.data.parameter_strings), 
        name=r"Sampled $\hat{\pi}$", 
        color="red", 
        plot_cloud=True, 
        plot_contour=False
    )
)
c.add_truth(
    Truth(location=dict(zip(cumulants_dataset.data.parameter_strings, cumulants_dataset.data.alpha)), name=r"$\pi^0$")
)
# plot_config = PlotConfig(
#     extents=dict(
#         zip(
#             cumulants_dataset.data.parameter_strings, 
#             np.stack([cumulants_dataset.data.lower, cumulants_dataset.data.upper], axis=1)
#         )
#     )
# )
# c.set_plot_config(plot_config)
fig = c.plotter.plot()
plt.savefig("quijote_vs_sampled_summaries.pdf") 
plt.close()



# # Scatter plot
# fig, axs = plt.subplots(1, dataset.alpha.size, figsize=(2. + 2. * dataset.alpha.size, 2.5))
# for p, ax in enumerate(axs):
#     ax.scatter(dataset.parameters[:, p], X[:, p], s=0.1, label="Quijote summaries")
#     ax.scatter(dataset.parameters[:, p], sampled_fiducial_cumulants[:, p], s=0.1, label="Sampled summaries")
#     ax.axline((0, 0), slope=1., color="k", linestyle="--")
#     ax.set_xlim(dataset.lower[p], dataset.upper[p])
#     ax.set_ylim(dataset.lower[p], dataset.upper[p])
#     ax.set_xlabel(dataset.parameter_strings[p])
#     ax.set_ylabel(dataset.parameter_strings[p] + "'")


# Histogram of sampled cumulants against those measured in Quijote
# for cumulants, name in zip(
#     [
#         cumulants_dataset.data.fiducial_data,
#         sampled_fiducial_cumulants
#     ],
#     ["Quijote", "Sampled"]
# ):
#     print("{} {:.3E} {:.3E}".format(name, cumulants.min(), cumulants.max()))

# names = [
#     r"$\langle\delta\rangle$", 
#     r"$\langle\delta^2\rangle$",
#     r"$\langle\delta^3\rangle_c$",
#     r"$\langle\delta^4\rangle_c$"
# ]
# names = names[1:] if not use_mean else names
# scales = ["5.0", "10.0", "15.0", "20.0", "25.0", "30.0", "35.0"]

# fig, axes = plt.subplots(7, 3, figsize=(9., 14.))
# for i in range(len(names)):
#     for r in range(len(scales)):

#         axes[r, i].hist(
#             calculated_cumulants_dataset.data.fiducial_data[:, i + r * 3], # If using mean, shift i here by 1
#             bins=50, 
#             color="blue",
#             alpha=0.1, 
#             label='C'
#         )
#         axes[r, i].hist(
#             cumulants_dataset.data.fiducial_data[:, i + r * 3], 
#             bins=50, 
#             histtype='step', 
#             color='orange', 
#             linewidth=1.5, 
#             alpha=0.2, 
#             label='Q'
#         )
#         axes[r, i].hist(
#             bulk_cumulants_dataset.data.fiducial_data[:, i + r * 3], 
#             bins=50, 
#             histtype='step', 
#             color='g', 
#             linewidth=1.5, 
#             alpha=0.2, 
#             label='C[b]'
#         )

#         axes[r, i].set_title(names[i] + " R=" + scales[r] + " Mpc/h")
#         axes[r, i].legend(frameon=False)

# plt.tight_layout()
# plt.savefig("QUIJOTE_VS_SAMPLED_CUMULANTS.png", bbox_inches="tight")
# plt.close()