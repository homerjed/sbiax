import os
from dataclasses import dataclass
from functools import partial
from typing import Tuple, Callable, Optional
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Key, PRNGKeyArray, Array, Float, jaxtyped
from beartype import beartype as typechecker
import numpy as np
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
from chainconsumer import Chain, ChainConsumer, Truth
import tensorflow_probability.substrates.jax.distributions as tfd
from tqdm.auto import trange

from constants import get_quijote_parameters, get_save_and_load_dirs
from sbiax.utils import make_df, marker
from sbiax.compression.linear import mle


typecheck = jaxtyped(typechecker=typechecker)


@typecheck
@dataclass
class Dataset:
    alpha: Float[Array, "p"]
    lower: Float[Array, "p"]
    upper: Float[Array, "p"]
    parameter_strings: list[str]
    Finv: Float[Array, "p p"]
    Cinv: Float[Array, "d d"]
    C: Float[Array, "d d"]
    fiducial_data: Float[Array, "nf d"]
    data: Float[Array, "nl d"]
    parameters: Float[Array, "nl p"]
    derivatives: Float[Array, "nd p d"]


def convert_dataset_to_jax(dataset: Dataset) -> Dataset:
    def convert_to_jax_array(a):
        if isinstance(a, np.ndarray):
            a = jnp.asarray(a)
        return a
    return jax.tree.map(convert_to_jax_array, dataset)


@typecheck
def get_raw_data(
    data_dir: str, 
    cumulants: bool = False,
    verbose: bool = False
) -> Tuple[
    Float[np.ndarray, "z 15000 R d"],
    Float[np.ndarray, "z 2000 R d"],
    Float[np.ndarray, "2000 p"],
    Float[np.ndarray, "500 p z R 2 d"]
]:
    """
        Load fiducial and latin PDFs
    """
    cumulants_or_moments_str = "CUMULANTS" if cumulants else "MOMENTS" 

    # (z, n, d, R)
    fiducial_pdfs = np.load(
        os.path.join(data_dir, "ALL_FIDUCIAL_{}.npy".format(cumulants_or_moments_str)) # (z, n, d, R)
    )
    latin_pdfs = np.load(
        os.path.join(data_dir, "ALL_LATIN_{}.npy".format(cumulants_or_moments_str)) # (z, n, d, R)
    ) 
    latin_pdfs_parameters = np.load(
        os.path.join(data_dir, "ALL_LATIN_PDFS_PARAMETERS.npy") # (n, p)
    )
    derivatives = np.load(
        os.path.join(data_dir, "{}_derivatives_plus_minus.npy".format(cumulants_or_moments_str.lower())) # (n, p, z, R, pm, d)
    )

    if verbose:
        print("Raw data shapes:", [_.shape for _ in [fiducial_pdfs, latin_pdfs, latin_pdfs_parameters, derivatives]])

    return fiducial_pdfs, latin_pdfs, latin_pdfs_parameters, derivatives


@typecheck
def get_moments_of_order(
    fiducial_pdfs: Float[np.ndarray, "z 15000 R d"], 
    latin_pdfs: Float[np.ndarray, "z 2000 R d"], 
    derivatives: Float[np.ndarray, "500 p z R d"],
    *, 
    order_idx: Optional[list[int]] = None,
    verbose: bool = False
) -> Tuple[
    Float[np.ndarray, "z 15000 R _"],
    Float[np.ndarray, "z 2000 R _"],
    Float[np.ndarray, "500 p z R _"]
]:
    # Change the size of 'd' in arguments by selecting moments
    assert max(order_idx) <= 2, "Maximum index is 2."

    order_idx = np.asarray(order_idx, dtype="int")

    if verbose:
        print("order_idx", order_idx)

    return (
        fiducial_pdfs[:, :, :, order_idx],
        latin_pdfs[:, :, :, order_idx],
        derivatives[:, :, :, :, order_idx]
    )


@typecheck
def get_reduced_cumulants(
    fiducial_moments: Float[np.ndarray, "15000 R d"], 
    latin_moments: Float[np.ndarray, "2000 R d"], 
    derivatives: Float[np.ndarray, "500 p R d"]
) -> Tuple[
    Float[np.ndarray, "15000 R d"], 
    Float[np.ndarray, "2000 R d"], 
    Float[np.ndarray, "500 p R d"]
]:
    """ 
        Scale derivatives like this, use autodiff for jacobian: dc/dp = dc/dm * dm/dp 
        - Quijote calculates: (smoothing, var, mom2, mom3, mom4) where mom's are sample cumulants (kstats)
        => CALCULATE JACOBIAN
        - is this assuming that what I think are cumulants are actually cumulants here?
    """

    # Index is zero for moments loaded with first 2 entries in Quijote skipped
    _, n_R, _ = fiducial_moments.shape

    def _reduce(moment_n, var, n):
        return moment_n / (var ** (n - 2))

    fiducial_cumulants = jnp.zeros_like(fiducial_moments)
    latin_cumulants = jnp.zeros_like(latin_moments)
    derivatives = jnp.asarray(derivatives)
    for n in range(2, 5): # Order of moment
        for R in range(n_R):

            print("n={}, R={}".format(n, R))

            if n == 2:
                print("Skipping reduction for variance")
                continue # Don't divide variance by itself

            reduce = partial(_reduce, n=n) # Order n moments
            
            fiducial_cumulants = fiducial_cumulants.at[:, R, n - 2].set(
                jax.vmap(reduce)(
                    fiducial_moments[:, R, n - 2], fiducial_moments[:, R, 0]
                )
            )
            latin_cumulants = latin_cumulants.at[:, R, n - 2].set(
                jax.vmap(reduce)(
                    latin_moments[:, R, n - 2], latin_moments[:, R, 0]
                )
            )

            dcdm = jax.vmap(jax.jacfwd(reduce))(
                fiducial_moments[:500, R, n - 2], fiducial_moments[:500, R, 0]
            ) # (500, 6, 6)
            # reduced_derivatives_n = jnp.einsum(
            #     "n p, n c m -> n p c", derivatives[:, :, R, n - 2], dcdm
            # )
            reduced_derivatives_n = derivatives[:, :, R, n - 2] * dcdm[:, jnp.newaxis]
            derivatives = derivatives.at[:, :, R, n - 2].set(reduced_derivatives_n)


    assert all(
        jnp.all(jnp.isfinite(a)) #and jnp.all(a > 0.)
        for a in (fiducial_cumulants, latin_cumulants, derivatives)
    )

    return tuple(map(np.asarray, (fiducial_cumulants, latin_cumulants, derivatives)))


@typecheck
def get_R_and_z_moments(
    z_idx: int, 
    R_idx: list[int], 
    fiducial_pdfs: Float[np.ndarray, "z 15000 R d"], 
    latin_pdfs: Float[np.ndarray, "z 2000 R d"], 
    derivatives: Float[np.ndarray, "500 p z R d"],
    *, 
    order_idx: Optional[list[int]] = None,
    reduced_cumulants: bool = False,
    verbose: bool = False
) -> Tuple[
    Float[np.ndarray, "15000 zRd"],
    Float[np.ndarray, "2000 zRd"],
    Float[np.ndarray, "500 p zRd"]
]:
    """ 
        Get and stack moments for smoothing scales and redshift. 
        - select for moment order (e.g. var, skewness, kurtosis ... before final reshape)
    """

    if order_idx is not None:
        n_cumulants = len(order_idx)

        (
            fiducial_pdfs, 
            latin_pdfs,
            derivatives
        ) = get_moments_of_order(
            fiducial_pdfs, 
            latin_pdfs, 
            derivatives, 
            order_idx=order_idx,
            verbose=verbose
        )
    else:
        n_cumulants = 3

    def get_R_z(
        simulations: Array, 
        z_idx: int, 
        R_idx: list[int], 
        *,
        reduced_cumulants: bool = False,
        are_derivatives: bool = False
    ) -> Array:
        # Obtain fiducial/latin/derivative cumulants at chosen redshift and scales

        @typecheck
        def _maybe_reduce(
            cumulants: Float[np.ndarray, "... c"] | Float[np.ndarray, "5 c"], 
            reduce: bool = False
        ) -> Float[np.ndarray, "... c"] | Float[np.ndarray, "5 c"]:
            """ Derivatives; linear operation on two +/- parameter nudged sims => still just divide? """

            cumulant_orders = [2, 3, 4] # Order of input cumulants (e.g. var, skew, kurt)

            var = cumulants[..., 0]

            for cumulant_index in range(cumulants.shape[-1]): # Derivatives or sims => choose last axis
                # Only reduce cumulants of higher order than variance
                cumulant_n = cumulants[..., cumulant_index]

                if reduce and (cumulant_index > 0):
                    # E.g. for skewness; skewness_reduced = skewness / (var ** 2)
                    order = cumulant_orders[cumulant_index]
                    cumulant_n = cumulants[..., cumulant_index] / (var ** (order - 1)) 

                cumulants[..., cumulant_index] = cumulant_n

            return cumulants # Reduced

        if isinstance(z_idx, int):
            z_idx = [z_idx]

        n_scales = len(R_idx)
        n_redshifts = len(z_idx)

        # n_s is number of derivatives or number of simulations (latin / fiducial)
        if are_derivatives:
            n_s, *_ = simulations.shape

            R_z_simulations = np.zeros((n_s, 5, n_scales * n_redshifts * n_cumulants))
        else:
            _, n_s, *_ = simulations.shape

            R_z_simulations = np.zeros((n_s, n_scales * n_redshifts * n_cumulants))

        for n in trange(n_s):
            for z, z_i in enumerate(z_idx):
                for r, r_i in enumerate(R_idx):
                    # _slice = z_i * n_scales + r_i
                    _slice = z * n_scales + r # NOTE: These must be positions in new array
                    if are_derivatives:
                        # Shape (5, 3)
                        simulation = _maybe_reduce(
                            simulations[n, :, z_i, r_i, :], reduce=reduced_cumulants
                        )
                        R_z_simulations[n, :, _slice * n_cumulants : (_slice + 1) * n_cumulants] = simulation
                    else:
                        # Shape (3,)
                        simulation = _maybe_reduce(
                            simulations[z_i, n, r_i, :], reduce=reduced_cumulants
                        )
                        R_z_simulations[n, _slice * n_cumulants : (_slice + 1) * n_cumulants] = simulation

        return R_z_simulations

    fiducial_pdfs_z_R = get_R_z(
        fiducial_pdfs, z_idx, R_idx, reduced_cumulants=reduced_cumulants
    )
    latin_pdfs_z_R = get_R_z(
        latin_pdfs, z_idx, R_idx, reduced_cumulants=reduced_cumulants
    )
    derivatives_z_R = get_R_z(
        derivatives, z_idx, R_idx, reduced_cumulants=reduced_cumulants, are_derivatives=True
    )

    # # Choose R-values and redshift from all PDFs
    # fiducial_pdfs_z_R = np.stack([
    #     fiducial_pdfs[z_idx, :, R, :] for R in R_idx], axis=-2
    # )
    # latin_pdfs_z_R = np.stack([
    #     latin_pdfs[z_idx, :, R, :] for R in R_idx], axis=-2
    # )
    # derivatives_z_R = np.stack([
    #     derivatives[:, :, z_idx, R, :] for R in R_idx], axis=2
    # )

    # if reduced_cumulants:
    #     (
    #         fiducial_pdfs_z_R, 
    #         latin_pdfs_z_R,
    #         derivatives
    #     ) = get_reduced_cumulants(
    #         fiducial_pdfs_z_R, 
    #         latin_pdfs_z_R,
    #         derivatives_z_R
    #     )

    print("Processed data shapes:", [_.shape for _ in [fiducial_pdfs_z_R, latin_pdfs_z_R, derivatives]])
    # Flatten moments / scales axes NOTE: TRANSPOSE instead? NOTE: must reflect covariance ordering of datavector...
    return fiducial_pdfs_z_R, latin_pdfs_z_R, derivatives_z_R
    # return (
    #     fiducial_pdfs_z_R.reshape(fiducial_pdfs_z_R.shape[0], -1), 
    #     latin_pdfs_z_R.reshape(latin_pdfs_z_R.shape[0], -1), 
    #     derivatives_z_R.reshape(derivatives_z_R.shape[0], 5, -1)
    # )


def hartlap(n_s: int, n_d: int) -> float: 
    return (n_s - n_d - 2) / (n_s - 1)


def get_parameter_strings() -> list[str]:
    (_, _, _, _, _, _, parameter_strings, *_) = get_quijote_parameters()
    return parameter_strings


def remove_nuisances(dataset: Dataset) -> Dataset:
    # Remove 'nuisance' parameters not constrained by the moments/cumulants

    return dataset


def get_moment_data(config: ConfigDict, *, verbose: bool = False) -> Dataset:

    data_dir, *_ = get_save_and_load_dirs()

    (
        all_R_values,
        all_redshifts,
        resolution,
        alpha,
        lower,
        upper,
        parameter_strings,
        redshift_strings,
        parameter_derivative_names,
        dparams,
        deltas,
        delta_bin_edges,
        D_deltas 
    ) = get_quijote_parameters()

    R_values = config.scales
    redshift = config.redshift
    R_idx    = [all_R_values.index(R) for R in R_values]
    z_idx    = all_redshifts.index(redshift)

    # Raw moments
    (
        fiducial_moments,
        latin_moments, 
        latin_moments_parameters,
        derivatives
    ) = get_raw_data(
        data_dir, 
        cumulants=config.cumulants, 
        verbose=verbose
    )

    # Euler derivative from plus minus statistics
    derivatives = derivatives[..., 1, :] - derivatives[..., 0, :] # NOTE: derivatives: Float[np.ndarray, "500 p z R 2 d"]
    for p in range(alpha.size):
        if verbose:
            print(parameter_strings[p], alpha[p], dparams[p], parameter_derivative_names[p])
        derivatives[:, p, ...] = derivatives[:, p, ...] / dparams[p]

    # Grab and stack by redshift and scales
    (
        fiducial_moments_z_R,
        latin_moments_z_R,
        derivatives
    ) = get_R_and_z_moments(
        z_idx, 
        R_idx, 
        fiducial_moments, 
        latin_moments, 
        derivatives,
        order_idx=config.order_idx,
        reduced_cumulants=config.reduced_cumulants,
        verbose=verbose
    )

    n_fiducial_pdfs, data_dim = fiducial_moments_z_R.shape 

    # Calculate covariance for datavector of each fiducial pdfs for all chosen scales 
    C = np.cov(fiducial_moments_z_R, rowvar=False) 

    # Condition number regularisation
    if config.covariance_epsilon is not None:
        L = jnp.trace(C) / data_dim * config.covariance_epsilon

        # U, S, Vt = jnp.linalg.svd(C)
        # L = 0.01 * S.min()
        # L = S.max() / 1000

        C = jnp.identity(data_dim) * L + C

    assert np.isfinite(C).all(), "Bad covariance."
    assert derivatives.ndim == 3, "Do derivatives have batch axis? Required."

    H = hartlap(n_fiducial_pdfs, data_dim)
    Cinv = H * np.linalg.inv(C)
    # Cinv = jnp.linalg.svd(C) * H

    # Fisher information matrix, all scales, one redshift
    dmu = derivatives.mean(axis=0)
    F = np.linalg.multi_dot([dmu, Cinv, dmu.T])
    Finv = np.linalg.inv(F)

    if verbose:
        print("Covariance condition number: {:.3E}".format(jnp.linalg.cond(C)))
        print("Dtype of moments", fiducial_moments_z_R.dtype)

    corr_matrix = np.corrcoef(fiducial_moments_z_R, rowvar=False) + 1e-6 # Log colouring

    # plt.figure()
    # norm = mcolors.LogNorm(vmin=np.min(corr_matrix[corr_matrix > 0]), vmax=np.max(corr_matrix)) 
    # plt.imshow(corr_matrix, norm=norm)
    # plt.colorbar()
    # plt.axis("off")
    # plt.savefig("moments_corr_matrix.png", bbox_inches="tight")
    # plt.close()
    
    print("DATA:", ["{:.3E} {:.3E}".format(_.min(), _.max()) for _ in (fiducial_moments_z_R, latin_moments_z_R)])
    print("DATA:", [_.shape for _ in (fiducial_moments_z_R, latin_moments_z_R)])

    dataset = Dataset(
        jnp.asarray(alpha),
        jnp.asarray(lower),
        jnp.asarray(upper),
        parameter_strings,
        jnp.asarray(Finv),
        jnp.asarray(Cinv),
        jnp.asarray(C),
        jnp.asarray(fiducial_moments_z_R),
        jnp.asarray(latin_moments_z_R),
        jnp.asarray(latin_moments_parameters),
        jnp.asarray(derivatives)
    )

    # Plot Fisher forecast
    c = ChainConsumer()
    c.add_chain(
        Chain.from_covariance(
            dataset.alpha,
            dataset.Finv,
            columns=dataset.parameter_strings,
            name=r"$F_{\Sigma^{-1}}$",
            color="k",
            linestyle=":",
            shade_alpha=0.
        )
    )
    c.add_marker(
        location=marker(dataset.alpha, parameter_strings=dataset.parameter_strings),
        name=r"$\alpha$", 
        color="#7600bc"
    )
    fig = c.plotter.plot()
    plt.savefig(
        "fisher_forecast_{}_z={}_R={}_m={}.png".format(
            config.linearised, 
            config.redshift, 
            "".join(map(str, config.order_idx)),
            "".join(map(str, config.scales))
        )
    )
    plt.close()

    return dataset


def get_prior(config: ConfigDict) -> tfd.Distribution:
    dataset: Dataset = get_data(config)

    lower = jnp.asarray(dataset.lower)
    upper = jnp.asarray(dataset.upper)

    assert jnp.all(upper - lower > 0.)

    # if config.linearised:
    #     print("Expanding prior...")
    #     prior = tfd.Blockwise(
    #         [tfd.Uniform(-1e3, 1e3) for p in range(dataset.alpha.size)]
    #     )
    # else:
    #     prior = tfd.Blockwise(
    #         [tfd.Uniform(lower[p], upper[p]) for p in range(dataset.alpha.size)]
    #     )

    prior = tfd.Blockwise(
        [tfd.Uniform(lower[p], upper[p]) for p in range(dataset.alpha.size)]
    )
    return prior


@typecheck
def linearised_model(
    alpha: Float[Array, "p"], 
    alpha_: Float[Array, "p"], 
    mu: Float[Array, "d"], 
    dmu: Float[Array, "p d"]
) -> Float[Array, "d"]:
    return mu + jnp.dot(alpha_ - alpha, dmu)


@typecheck
def sample_prior(
    key: PRNGKeyArray, 
    n_linear_sims: int, 
    alpha: Float[Array, "p"], 
    lower: Float[Array, "p"], 
    upper: Float[Array, "p"]
) -> Float[Array, "n p"]:
    # Avoid tfp warning
    lower = lower.astype(jnp.float32)
    upper = upper.astype(jnp.float32)

    keys_p = jr.split(key, alpha.size)

    Y = jnp.stack(
        [
            jr.uniform(
                key_p, 
                (n_linear_sims,), 
                minval=lower[p], 
                maxval=upper[p]
            )
            for p, key_p in enumerate(keys_p)
        ], 
        axis=1
    )

    return Y


@typecheck
def get_linearised_data(config: ConfigDict) -> Tuple[Float[Array, "n d"], Float[Array, "n p"]]:
    """
        Get linearised PDFs and get their MLEs 
    """
    key = jr.key(config.seed)

    dataset: Dataset = get_moment_data(config)

    if config.n_linear_sims is not None:
        Y = sample_prior(
            key, config.n_linear_sims, dataset.alpha, dataset.lower, dataset.upper
        )

    assert dataset.derivatives.ndim == 3, (
        "Do derivatives [{}] have batch axis? Required.".format(dataset.derivatives.shape)
    )

    dmu = dataset.derivatives.mean(axis=0)
    mu = dataset.fiducial_data.mean(axis=0)

    def _simulator(key, pi):
        # Data model for linearised expectation
        _mu = linearised_model(alpha=dataset.alpha, alpha_=pi, mu=mu, dmu=dmu)
        return jr.multivariate_normal(key, mean=_mu, cov=dataset.C)

    keys = jr.split(key, len(Y))
    D = jax.vmap(_simulator)(keys, Y) 

    return D, Y 


@typecheck
def get_nonlinearised_data(config: ConfigDict) -> Tuple[Float[Array, "n d"], Float[Array, "n p"]]:
    """
        Get non-linearised PDFs. 
        - use linearised model at a random fiducial pdf noise realisation
    """
    key = jr.key(config.seed)

    assert config.n_linear_sims <= 15_000, "Must use n_linear_sims <= 15,000. Quijote limit."

    # This is not simply returning fiducials, latins from Quijote...
    if 0:
        dataset: Dataset = get_moment_data(config)

        if config.n_linear_sims is not None:
            Y = sample_prior(
                key, config.n_linear_sims, dataset.alpha, dataset.lower, dataset.upper
            )

        assert dataset.derivatives.ndim == 3, "Do derivatives have batch axis? Required."

        dmu = dataset.derivatives.mean(axis=0)

        @typecheck
        def simulator(
            key: PRNGKeyArray, xi_0_i: Float[Array, "d"], pi: Float[Array, "p"]
        ) -> Float[Array, "d"]:
            # Data model for non-linear expectation
            _mu = linearised_model(dataset.alpha, pi, mu=xi_0_i, dmu=dmu)
            return jr.multivariate_normal(key, mean=_mu, cov=dataset.C)

        keys = jr.split(key, len(Y))
        D = jax.vmap(simulator)(keys, dataset.fiducial_data[:len(Y)], Y) 

    dataset: Dataset = get_moment_data(config)
    D = dataset.data
    Y = dataset.parameters

    return D, Y 


def get_data(config: ConfigDict, *, verbose: bool = False) -> Dataset:
    """ 
        Get data for linearised-model data or full simulation data. 
        - Start with Quijote default data; linearise or nonlinearise 
          if required
    """

    dataset: Dataset = get_moment_data(config, verbose=verbose)

    if hasattr(config, "linearised"):
        if config.linearised:
            print("Using linearised model, Gaussian noise.")
            D, Y = get_linearised_data(config) 

            dataset.data = D # NOTE: does this actually work?
            dataset.parameters = Y

    if hasattr(config, "nonlinearised"):
        if config.nonlinearised:
            print("Using linearised model, non-Gaussian noise.") 
            # NOTE: does `get_datavector()` do this also?
            D, Y = get_nonlinearised_data(config)

            dataset.data = D
            dataset.parameters = Y

    return dataset


@typecheck
def get_linear_compressor(
    config: ConfigDict
) -> Callable[[Float[Array, "d"], Float[Array, "p"]], Float[Array, "p"]]:
    """ Get Chi^2 minimisation function; compressing datavector at estimated parameters to summary """

    dataset: Dataset = get_data(config)

    mu = dataset.fiducial_data.mean(axis=0)
    dmu = dataset.derivatives.mean(axis=0)

    compressor = lambda d, p: mle(
        d,
        pi=p,
        Finv=dataset.Finv, 
        mu=linearised_model(dataset.alpha, p, mu=mu, dmu=dmu), 
        dmu=dmu, 
        precision=dataset.Cinv
    )

    return compressor


@typecheck
def get_datavector(
    key: PRNGKeyArray, 
    config: ConfigDict, 
    n: int = 1
) -> Float[Array, "d"]:
    """ Measurement: either Gaussian linear model or not """

    # NOTE: must be working with fiducial parameters!
    dataset: Dataset = get_data(config)
    
    mu = dataset.fiducial_data.mean(axis=0)

    # Choose a linearised model datavector or simply one of the Quijote realisations
    # which corresponds to a non-linearised datavector with Gaussian noise
    if config.linearised:
        print("Using linearised datavector")
        datavector = jr.multivariate_normal(key, mean=mu, cov=dataset.C, shape=(n,))
    else:
        print("Using non-linearised datavector")
        datavector = jr.choice(key, dataset.fiducial_data, shape=(n,))

        # print("USING DATA EXPECTATION")
        # datavector = jnp.mean(dataset.fiducial_data, axis=0, keepdims=True)

    return jnp.squeeze(datavector, axis=0) # Remove batch axis by default