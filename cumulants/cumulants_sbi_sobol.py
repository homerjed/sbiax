import os
import time
import datetime

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

import numpy as np 
import matplotlib.pyplot as plt
from chainconsumer import Chain, ChainConsumer
from tensorflow_probability.substrates.jax.distributions import Distribution
import tensorflow_probability.substrates.jax.distributions as tfd
from tqdm.auto import trange
from jaxtyping import Float, Array

from sbiax.train import train_ensemble
from sbiax.ndes import CNF, MAF, Scaler
from sbiax.inference import nuts_sample
from sbiax.utils import make_df, marker

from configs import (
    get_results_dir, 
    get_posteriors_dir, 
    get_ndes_from_config
)
from configs.log import setup_module_logger, get_log_level
from configs.args import get_cumulants_sbi_args
from data.constants import get_cumulant_names, get_sobol_ingredients
from data.common import Dataset, add_planck_information_to_Finv, fit_nn
from cumulants_ensemble import Ensemble
from affine import affine_sample
from utils import (
    get_datasets,
    plot_cumulants,
    plot_moments, 
    plot_latin_moments, 
    plot_summaries, 
    plot_summaries_fiducial,
    plot_fisher_summaries, 
    finite_samples_log_prob
)

jax.clear_caches()

logger, log_figs_dir = setup_module_logger(__name__, level=get_log_level())


def get_fiducial_and_latin_data(bulk_or_tails, redshift=0.):
    BoxSize = 1000.0 #Mpc/h
    grid = 256
    d = BoxSize / grid

    scale_numbers = np.array([3., 5., 7., 9., 11., 13., 15., 17.])
    scales = scale_numbers * d / 2 # Mpc/h, for accurate results the mesh is 1/10 of this

    redshifts = [redshift]

    cdf_cut_lims = dict(bulk=(0.03, 0.90), tails=(0.001, 0.999))

    datasets_dir = os.path.join("/project/ls-gruen/users/jed.homer/quijote_pdfs_later/sobol/", "datasets/")

    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir, exist_ok=True)

    try:
        dataset = np.load(
            os.path.join(datasets_dir, "dataset.npz")
        )
        return (
            jnp.asarray(x)
            for x in [
                dataset["fiducial_cumulants"], dataset["cumulants"], dataset["parameters"]
            ]
        )
    except FileNotFoundError:
        pass

    def get_pdf_filename_template(redshift, radius_index, realisation):
        filename = 'matter_PDF_linbins_BSQ_z_{:.2f}_extraRind_{}_{}.dat'.format(
            redshift, radius_index, str(realisation)
        )
        return filename

    def cut_pdf_to_cumulants(
        cut_pdf: Float[np.ndarray, "d"], # Divide by cut-norm
        deltas: Float[np.ndarray, "d"],
        ddeltas: Float[np.ndarray, "d"],
        prob_norm: float # Max prob - min_prob in CDF cut
    ) -> Float[np.ndarray, "5"]:
        # Bernardeau 2002 Eq. 130

        # Renormalise PDF?
        cut_pdf = cut_pdf / prob_norm # Was 'fiducial_based_normalisation'

        _delta_ = np.sum(ddeltas * cut_pdf * deltas)

        deltamod = deltas - _delta_

        m_0 = np.sum(cut_pdf * ddeltas)
        m_1 = np.sum(cut_pdf * ddeltas * deltas) 

        # Assuming <delta>=0? see Bernardeau eq (130)
        cumulant_2 = np.sum(deltamod ** 2 * cut_pdf * ddeltas)
        cumulant_3 = np.sum(deltamod ** 3 * cut_pdf * ddeltas)
        cumulant_4 = np.sum(deltamod ** 4 * cut_pdf * ddeltas) - (3. * np.sum(deltamod ** 2 * cut_pdf * ddeltas) ** 2)

        cumulants = np.asarray([m_0, m_1, cumulant_2, cumulant_3, cumulant_4]) 

        return cumulants

    def get_cdf_of_pdf(pdf, dbins, cdf_cut_lims):
        n_bins_pdf = pdf.size

        p_value_min, p_value_max = cdf_cut_lims

        cdf = np.zeros_like(pdf)
        normalisation = 0.
        for i in range(1, n_bins_pdf):
            p_delta_i = pdf[i - 1]

            dp_i = p_delta_i * dbins[i - 1] 

            cdf[i] = cdf[i - 1] + dp_i

            normalisation += dp_i

        # Check CDF bounds
        assert np.isclose(cdf.min(), 0.) and np.isclose(cdf.max(), 1.), (
            "CDF: min={} max={}".format(cdf.min(), cdf.max())
        )

        # Check PDF is normalised
        assert np.isclose(normalisation, 1.0)

        cut_idx = np.where(
            (cdf >= p_value_min) & (cdf <= p_value_max)
        )[0] 

        return cdf, cut_idx


    fiducial_data_dir = "/project/ls-gruen/users/jed.homer/quijote_pdfs_later/sobol/matter_pdf_fiducial/"

    n_cumulants = 5 # m_0, m_1, k_2, k_3, k_4

    n_fiducial_pdfs = 100

    fiducial_cumulants = dict(
        bulk=np.zeros((n_fiducial_pdfs, len(redshifts), len(scales), n_cumulants)),
        tails=np.zeros((n_fiducial_pdfs, len(redshifts), len(scales), n_cumulants))
    )

    bad_realisations = []
    for cut_name, cdf_cut_lim in cdf_cut_lims.items():

        if cut_name != bulk_or_tails:
            continue

        for realisation_idx, realisation in zip(
            bar := trange(
                n_fiducial_pdfs, 
                desc="Fiducial PDFs",
                colour="red" if cut_name == "tails" else "blue"
            ),
            np.arange(n_fiducial_pdfs) # Assume 0-N are all available
        ):

            for i_z, z in enumerate(redshifts): 

                for i_r, radius in enumerate(scale_numbers):

                    # Grab physical scale index
                    idx = np.squeeze(np.argwhere(scale_numbers == radius))

                    # Filename of FIDUCIAL PDF
                    realisation_path = os.path.join(
                        fiducial_data_dir, 
                        str(realisation), 
                        get_pdf_filename_template(
                            redshift=z, radius_index=idx, realisation=realisation
                        ) 
                    )

                    try:
                        # Load PDF bin centres and PDF in bins
                        bins, pdf = np.loadtxt(realisation_path).T

                        # bins = bins - 1. # Rho -> delta

                        assert bins.size == pdf.size

                        dbins = bins[1:] - bins[:-1]

                        # Cut each pdf by its own CDF
                        cdf, cdf_cut_idx = get_cdf_of_pdf(pdf, dbins, cdf_cut_lim)

                        cut_pdf = pdf[cdf_cut_idx]
                        cut_bins = bins[cdf_cut_idx]
                        cut_dbins = dbins[cdf_cut_idx] 

                        cumulants_R_z = cut_pdf_to_cumulants(
                            cut_pdf, 
                            cut_bins, 
                            cut_dbins, 
                            prob_norm=cdf_cut_lim[1] - cdf_cut_lim[0]
                        )

                        fiducial_cumulants[cut_name][realisation_idx, i_z, i_r] = cumulants_R_z

                    except FileNotFoundError as e:
                        print(e)
                        bad_realisations.append(realisation_idx)

            bar.set_description("cut={}".format(cut_name))

        # print("Number of bad realisations:", len(bad_realisations))

    from pathlib import Path

    data_dir = Path("/project/ls-gruen/users/jed.homer/quijote_pdfs_later/sobol/matterPDF_BSQ/")

    parameters = np.loadtxt(Path(data_dir) / "BSQ_params.txt")

    # Get file numbers that exist in Sobol LH directory
    available_idx_realisations = []
    for _ in data_dir.iterdir():
        try:
            number = int(str(_).split("/")[-1])
        except ValueError:
            continue
        available_idx_realisations.append(number)

    available_idx_realisations = np.asarray(available_idx_realisations)

    n_available_realisations = len(available_idx_realisations)

    cumulants = dict(
        bulk=np.zeros((n_available_realisations, len(redshifts), len(scales), n_cumulants)),
        tails=np.zeros((n_available_realisations, len(redshifts), len(scales), n_cumulants))
    )

    bad_realisations = []
    for cut_name, cdf_cut_lim in cdf_cut_lims.items():

        if cut_name != bulk_or_tails:
            continue

        for realisation_idx, realisation in zip(
            bar := trange(
                n_available_realisations, 
                colour="red" if cut_name == "tails" else "blue"
            ),
            available_idx_realisations
        ):

            for i_z, z in enumerate(redshifts): 

                for i_r, radius in enumerate(scale_numbers):

                    # Grab physical scale index
                    idx = np.squeeze(np.argwhere(scale_numbers == radius))

                    # Filename of Sobol sequence PDF
                    realisation_path = data_dir / str(realisation) / get_pdf_filename_template(
                        redshift=z, radius_index=idx, realisation=realisation
                    ) 

                    try:
                        # Load PDF bin centres and PDF in bins
                        bins, pdf = np.loadtxt(realisation_path).T

                        # bins = bins - 1. # Rho -> delta

                        assert bins.size == pdf.size

                        dbins = bins[1:] - bins[:-1]

                        # Cut each pdf by its own CDF
                        cdf, cdf_cut_idx = get_cdf_of_pdf(pdf, dbins, cdf_cut_lim)

                        cut_pdf = pdf[cdf_cut_idx]
                        cut_bins = bins[cdf_cut_idx]
                        cut_dbins = dbins[cdf_cut_idx] 

                        cumulants_R_z = cut_pdf_to_cumulants(
                            cut_pdf, 
                            cut_bins, 
                            cut_dbins, 
                            prob_norm=cdf_cut_lim[1] - cdf_cut_lim[0]
                        )

                        cumulants[cut_name][realisation_idx, i_z, i_r] = cumulants_R_z

                    except FileNotFoundError as e:
                        print(e)
                        bad_realisations.append(realisation_idx)

            bar.set_description("cut={}".format(cut_name))

    fiducial_cumulants = np.concatenate(
        [fiducial_cumulants[bulk_or_tails][:, 0, r, :] for r in range(len(scales))], axis=-1
    )
    cumulants = np.concatenate(
        [cumulants[bulk_or_tails][:, 0, r, :] for r in range(len(scales))], axis=-1
    )

    np.savez(
        os.path.join(datasets_dir, "dataset.npz"), 
        fiducial_cumulants=fiducial_cumulants, 
        cumulants=cumulants, 
        parameters=parameters[available_idx_realisations]
    )

    return fiducial_cumulants, cumulants, parameters[available_idx_realisations]


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

results_dir = get_results_dir(config, args)

posteriors_dir = get_posteriors_dir(args)

parameter_strings, alpha, (lower, upper) = get_sobol_ingredients()

parameter_prior = tfd.Blockwise([tfd.Uniform(l, u) for l, u in zip(lower, upper)])

# FLATTEN
fiducial_data, latin_data, parameters = get_fiducial_and_latin_data(args.bulk_or_tails, redshift=config.redshift)

"""
    Compression
"""
import equinox as eqx

@eqx.filter_vmap
def make_ensemble(key):
    return eqx.nn.MLP(
    latin_data.shape[-1], 
    parameters.shape[-1], 
    width_size=config.nn.width_size, 
    depth=config.nn.depth, 
    use_final_bias=config.nn.use_final_bias,
    activation=getattr(jax.nn, config.nn.activation),
    key=key
)

@eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
def evaluate_ensemble(model, x):
    return model(x)

class Ensemble(eqx.Module):
    ensemble: eqx.Module

    def __init__(self, ensemble):
        self.ensemble = ensemble

    def __call__(self, x):
        x = evaluate_ensemble(self.ensemble, x)
        return jnp.mean(x, axis=0)

def get_preprocess_fn(D):
    mu_D = latin_data.mean(axis=0) #jnp.mean(dataset.fiducial_data, axis=0)
    std_D = latin_data.mean(axis=0) #jnp.std(dataset.fiducial_data, axis=0)
    D = (D - mu_D) / std_D

    def preprocess_fn(d):
        d = jnp.asarray(d)
        return (d - mu_D) / std_D

    return D, preprocess_fn

description = "Fitting NN"

net_key, train_key = jr.split(key)

keys = jr.split(net_key, config.nn.n_ensemble)

net = Ensemble(make_ensemble(keys))

D, preprocess_fn = get_preprocess_fn(latin_data)

def preprocess_fn_p(p):
    # Scale parameters into loss / out of net
    # return (p - dataset.alpha) / np.diag(dataset.Finv)
    return (p - parameters.mean(axis=0)) / parameters.std(axis=0)

def postprocess_fn_p(p):
    # Scale parameters into loss / out of net
    # return (p - dataset.alpha) / np.diag(dataset.Finv)
    return p * parameters.std(axis=0) + parameters.mean(axis=0)

train_data = (D, preprocess_fn_p(parameters))

precision = None #jnp.linalg.inv(dataset.Finv) # In reality this varies with parameters

opt = getattr(optax, config.nn.train.opt)(config.nn.train.lr)

net, losses = fit_nn(
    train_key, 
    net, 
    opt=opt, 
    train_data=train_data,
    precision=precision, 
    n_batch=config.nn.train.n_batch,
    n_steps=config.nn.train.n_steps,
    patience=config.nn.train.patience,
    valid_fraction=config.nn.train.valid_fraction,
    description=description
)

plt.figure()
plt.loglog(losses, color="red" if args.bulk_or_tails == "tails" else "blue")
plt.savefig(os.path.join(results_dir, "losses_nn.png"))
plt.close()

eqx.tree_serialise_leaves(
    os.path.join(results_dir, "nn.eqx"), net
)

compression_fn = lambda d, p: postprocess_fn_p(net(preprocess_fn(d)))

# Compress simulations

X = jax.vmap(compression_fn)(latin_data, parameters)

"""
    Build NDEs
"""

use_scalers = True

scaler = Scaler(X, parameters, use_scaling=use_scalers)

keys = jr.split(key, len(config.ndes))

ndes = []
for nde, key in zip(config.ndes, keys):

    logger.info("Using NDE of type '{}'".format(nde.model_type))

    assert nde.model_type in ["maf", "cnf"], (
        "Invalid NDE model type (={})".format(nde.model_type)
    )

    if nde.model_type == "maf":
        nde_arch = MAF
    if nde.model_type == "cnf":
        nde_arch = CNF

    # Required to remove / add some arguments to specify NDEs
    nde_dict = dict(
        event_dim=alpha.size, 
        context_dim=alpha.size, 
        key=key,
        scaler=scaler if use_scalers else None,
        **dict(nde)
    )

    logger.info("NDE DICT: {}".format(nde_dict))

    nde_dict.pop("model_type")
    nde_dict.pop("use_scaling")

    ndes.append(nde_arch(**nde_dict))

ensemble = Ensemble(ndes)

"""
    Train NDE on data
"""

opt = getattr(optax, config.train.opt)(config.train.lr)

ensemble, stats = train_ensemble(
    train_key, 
    ensemble,
    train_mode="nle",
    train_data=(X, parameters), 
    opt=opt,
    n_batch=config.train.n_batch,
    patience=config.train.patience,
    n_epochs=config.train.n_epochs,
    valid_fraction=config.valid_fraction,
    tqdm_description="Training (data)",
    show_tqdm=args.use_tqdm,
    results_dir=results_dir
)

""" 
    Sample and plot posterior for NDE with noisy datavectors
"""

# Generates linearised (or not) datavector at fiducial parameters
datavector = fiducial_data[0]

logger.debug("datavector {} \n {}".format(datavector.shape, datavector))

x_ = compression_fn(datavector, alpha)

logger.debug("compressed datavector {} \n {} {}".format(x_.shape, x_, alpha))

log_prob_fn = ensemble.ensemble_log_prob_fn(x_, parameter_prior)

if 1:
    try:
        state = parameter_prior.sample(sample_shape=(2 * config.n_walkers,), seed=key_state)

        samples, weights = affine_sample(
            key_sample, 
            log_prob=log_prob_fn,
            n_walkers=config.n_walkers, 
            n_steps=config.n_steps + config.burn, 
            burn=config.burn, 
            current_state=state,
            description="Sampling",
            show_tqdm=args.use_tqdm
        )

        alpha_log_prob = log_prob_fn(dataset.alpha)
        samples_log_prob = jax.vmap(log_prob_fn)(samples)
        samples_log_prob = finite_samples_log_prob(samples_log_prob) 

        logger.debug("samples: {} {}".format(samples.min(), samples.max()))
        logger.debug("probs: {} {}".format(samples_log_prob.min(), samples_log_prob.max()))

        posterior_df = make_df(
            samples, 
            samples_log_prob, 
            parameter_strings=dataset.parameter_strings
        )

        np.savez(
            os.path.join(results_dir, "posterior.npz"), 
            alpha=dataset.alpha,
            samples=samples,
            samples_log_prob=samples_log_prob,
            datavector=datavector,
            summary=x_
        )

        c = ChainConsumer()
        c.add_chain(
            Chain(
                samples=posterior_df, 
                name="SBI[{}]".format(args.bulk_or_tails), 
                color="r" if args.bulk_or_tails == "tails" else "b"
            )
        )
        c.add_marker(
            location=marker(x_, parameter_strings=parameter_strings),
            name=r"$\hat{x}$", 
            color="r" if args.bulk_or_tails == "tails" else "b"
        )
        c.add_marker(
            location=marker(alpha, parameter_strings=parameter_strings),
            name=r"$\alpha$", 
            color="k"
        )
        fig = c.plotter.plot()
        fig.suptitle(
            (
                r"$k_n$ SBI & $F_{{\Sigma}}^{{-1}}$"
                + " z={}".format(config.redshift) + "\n"
                + (" linearised" if config.linearised else " Quijote") + ("[bulk]" if args.bulk_or_tails == "bulk" else "[tails]") + "\n"
                + r"$n_s$ = {}".format(config.n_linear_sims if config.linearised else 2000) + "\n"
                + r"$R$ = [{}] Mpc".format(", ".join(map(str, config.scales))) + "\n"
                + r"$k_n$ = [{}]".format(
                    ", ".join([get_cumulant_names()[_] for _ in config.order_idx])
                )
            ),
            multialignment='center'
        )
        plt.savefig(os.path.join(results_dir, "posterior_affine.png"))
        plt.savefig(os.path.join(posteriors_dir, "posterior_affine.pdf"))
        plt.close()

        target_idx = np.array([0, 4])
        _parameter_strings = [parameter_strings[p] for p in target_idx]
        posterior_df = make_df(
            samples[:, target_idx], 
            samples_log_prob, 
            parameter_strings=_parameter_strings
        )

        c = ChainConsumer()
        c.add_chain(
            Chain.from_covariance(
                alpha[target_idx],
                add_planck_information_to_Finv(
                    datasets["tails"].data.Finv, use_planck=args.use_planck
                )[:, target_idx][target_idx, :],
                columns=_parameter_strings,
                name=r"$F_{\Sigma^{-1}}$" + " {}".format("$k_n$[tails]"),
                color="r",
                linestyle=":",
                shade_alpha=0.
            )
        )
        c.add_chain(
            Chain.from_covariance(
                alpha[target_idx],
                add_planck_information_to_Finv(
                    datasets["bulk"].data.Finv, use_planck=args.use_planck
                )[:, target_idx][target_idx, :],
                columns=_parameter_strings,
                name=r"$F_{\Sigma^{-1}}$" + " {}".format("$k_n$[bulk]"),
                color="b",
                linestyle=":",
                shade_alpha=0.
            )
        )
        c.add_chain(
            Chain.from_covariance(
                alpha[target_idx],
                add_planck_information_to_Finv(
                    datasets["bulk_pdf"].data.Finv, use_planck=args.use_planck
                )[:, target_idx][target_idx, :],
                columns=_parameter_strings,
                name=r"$F_{\Sigma^{-1}}$" + " {}".format("PDF[bulk]"),
                color="g",
                linestyle=":",
                shade_alpha=0.
            )
        )
        c.add_chain(
            Chain(
                samples=posterior_df, 
                name="SBI[{}]".format(args.bulk_or_tails), 
                color="r" if args.bulk_or_tails == "tails" else "b"
            )
        )
        c.add_marker(
            location=marker(x_[target_idx], parameter_strings=_parameter_strings),
            name=r"$\hat{x}$", 
            color="r" if args.bulk_or_tails == "tails" else "b"
        )
        c.add_marker(
            location=marker(alpha[target_idx], parameter_strings=_parameter_strings),
            name=r"$\alpha$", 
            color="k"
        )
        fig = c.plotter.plot()
        fig.suptitle(
            (
                r"$k_n$ SBI & $F_{{\Sigma}}^{{-1}}$"
                + " z={}".format(config.redshift) + "\n"
                + (" linearised" if config.linearised else " Quijote") + ("[bulk]" if args.bulk_or_tails == "bulk" else "[tails]") + "\n"
                + r"$n_s$ = {}".format(config.n_linear_sims if config.linearised else 2000) + "\n"
                + r"$R$ = [{}] Mpc".format(", ".join(map(str, config.scales))) + "\n"
                + r"$k_n$ = [{}]".format(
                    ", ".join([get_cumulant_names()[_] for _ in config.order_idx])
                )
            ),
            multialignment='center'
        )
        plt.savefig(os.path.join(results_dir, "posterior_affine_marginalised.png"))
        plt.savefig(os.path.join(posteriors_dir, "posterior_affine_marginalised.pdf"))
        plt.close()
    except Exception as e:
        print("~" * 50)
        print(f"Exception:\n\t{e}")
        print("~" * 50)

print("Time={:.1} mins.".format((time.time() - t0) / 60.))