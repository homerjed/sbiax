import os
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray, Array, Float, Int, jaxtyped

import equinox as eqx
import optax
from beartype import beartype as typechecker 
from scipy.stats import qmc
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
import tensorflow_probability.substrates.jax.distributions as tfd

from configs.log import setup_module_logger, get_log_level
from data.constants import get_quijote_parameters, get_target_idx, get_F_planck, get_Finv_planck, LOWER, UPPER, ALPHA

typecheck = jaxtyped(typechecker=typechecker)

logger, log_figs_dir = setup_module_logger(__name__, level=get_log_level())

FORCE_NOISELESS_DATAVECTOR = True if os.environ.get("FORCE_NOISELESS_DATAVECTOR", "").lower() in ("1", "true") else False

"""
    Objects common to the PDF and cumulant datasets
"""

def add_planck_information_to_Finv(Finv, use_planck=False):
    # Add Fisher information from Planck to any Finv matrix
    if use_planck:
        F_planck = get_F_planck()
        Finv = jnp.linalg.inv(jnp.linalg.inv(Finv) + F_planck)
    return Finv


def hartlap(n_s: int, n_d: int) -> float: 
    return (n_s - n_d - 2) / (n_s - 1)


def get_parameter_strings() -> list[str]:
    (_, _, _, _, _, _, parameter_strings, *_) = get_quijote_parameters()
    return parameter_strings


@typecheck
@dataclass
class Dataset:
    name: Literal["bulk_pdf", "bulk", "tails"]
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
    derivatives: Float[Array, "500 p d"]


def freeze_out_parameters_dataset(dataset: Dataset) -> Dataset:

    @typecheck
    def _process_latins(
        latins: Float[Array, "n d"], 
        parameters: Float[Array, "n 5"], 
        alpha: Float[Array, "5"], 
        mu: Float[Array, "d"], 
        dmu: Float[Array, "5 d"],
        p_idx: Int[Array, "_"]
    ) -> Float[Array, "n d"]:
        """ 
            Freeze out non-target parameters in latin hypercube simulations
            by removing linear response and adding fiducial linear response.
        """ 

        @typecheck
        def _freeze_parameters(
            pdf_or_cumulant: Float[Array, "d"], 
            p: Float[Array, "p"]
        ) -> Float[Array, "d"]:
            # Freeze the nuisance parameters of the latin hypercube realisations

            # Om, s8 and nuisances at fiducial values
            p0 = jnp.array(
                [_p if (i_p in p_idx) else alpha[i_p] for i_p, _p in enumerate(p)]
            ) 

            mu_p_nu = linearised_model(alpha, alpha_=p, mu=mu, dmu=dmu)
            mu_p_nu_0 = linearised_model(alpha, alpha_=p0, mu=mu, dmu=dmu)

            return pdf_or_cumulant - mu_p_nu + mu_p_nu_0

        return jax.vmap(_freeze_parameters)(latins, parameters)

    p_idx = get_target_idx()

    # Recalculate Fisher information (not marginalising out nuisances, they are known)
    derivatives = dataset.derivatives[:, p_idx, :]  
    _derivatives = jnp.mean(derivatives, axis=0)
    F = jnp.linalg.multi_dot([_derivatives, dataset.Cinv, _derivatives.T])
    Finv = jnp.linalg.inv(F)

    # Remove influence of nuisances from latins by linearisation
    latins_data = _process_latins(
        dataset.data, 
        dataset.parameters, 
        alpha=dataset.alpha, 
        mu=jnp.mean(dataset.fiducial_data, axis=0), 
        dmu=jnp.mean(dataset.derivatives, axis=0), # Must be derivatives for all parameters!
        p_idx=p_idx
    )

    frozen_dataset = Dataset(
        name=dataset.name,
        alpha=dataset.alpha[p_idx],
        lower=dataset.lower[p_idx],
        upper=dataset.upper[p_idx],
        parameter_strings=[
            dataset.parameter_strings[p] for p in p_idx
        ],
        Finv=Finv,
        Cinv=dataset.Cinv,
        C=dataset.C,
        fiducial_data=dataset.fiducial_data,
        data=latins_data,
        parameters=dataset.parameters[:, p_idx],
        derivatives=derivatives # Target-indexed derivatives
    )

    return frozen_dataset 


@typecheck
def get_prior(config: ConfigDict, dataset: Dataset) -> tfd.Distribution:

    if config.linearised:
        logger.info("Using flat prior")

        flat_limit = 1e4
        lower = jnp.ones((dataset.alpha.size,)) * -flat_limit
        upper = jnp.ones((dataset.alpha.size,)) * flat_limit
    else:
        logger.info("Using Quijote uniform prior")
        lower = jnp.asarray(dataset.lower) # Avoid tfp warning
        upper = jnp.asarray(dataset.upper)

    assert jnp.all((upper - lower) > 0.)

    # print("FORCING FLAT PRIOR")
    # lower = jnp.ones((dataset.alpha.size,)) * -1e4
    # upper = jnp.ones((dataset.alpha.size,)) * 1e4

    # print("FORCING QUIJOTE PRIOR")
    # lower = jnp.asarray(dataset.lower) # Avoid tfp warning
    # upper = jnp.asarray(dataset.upper)

    # parameter_distributions = []
    # for p in range(5):
    #     if p in [1, 3]: # O_b and n_s
    #         if p == 1:
    #             dist = tfd.Normal(dataset.alpha[1], 0.052 / 100. / (dataset.alpha[2] ** 2.))
    #         if p == 3:
    #             dist = tfd.Normal(dataset.alpha[3], 0.0041)
    #         parameter_distributions.append(dist)
    #     else:
    #         parameter_distributions.append(
    #             tfd.Uniform(dataset.lower[p], dataset.upper[p])
    #         )
    # prior = tfd.Blockwise(parameter_distributions)

    if config.use_planck:
        prior = tfd.MultivariateNormalFullCovariance(
            dataset.alpha, covariance_matrix=get_Finv_planck()
        )
    else:
        prior = tfd.Blockwise(
            [tfd.Uniform(l, u) for l, u in zip(lower, upper)]
        )

    return prior


@typecheck
def get_prior_from_args(args) -> tfd.Distribution:

    if args.linearised:
        logger.info("Using flat prior")

        flat_limit = 1e4
        lower = jnp.ones((5,)) * -flat_limit
        upper = jnp.ones((5,)) * flat_limit
    else:
        logger.info("Using Quijote uniform prior")
        lower = jnp.asarray(LOWER) # Avoid tfp warning
        upper = jnp.asarray(UPPER)

    assert jnp.all((upper - lower) > 0.)

    if args.use_planck:
        prior = tfd.MultivariateNormalFullCovariance(
            ALPHA, covariance_matrix=get_Finv_planck()
        )
    else:
        # flat_limit = 1e4
        # lower = jnp.ones((5,)) * -flat_limit
        # upper = jnp.ones((5,)) * flat_limit
        prior = tfd.Blockwise(
            [tfd.Uniform(l, u) for l, u in zip(lower, upper)]
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
    upper: Float[Array, "p"],
    *,
    hypercube: bool = True
) -> Float[Array, "n p"]:
    # Forcing Quijote prior for simulating, this prior for inference

    lower = lower.astype(jnp.float32) # Avoid tfp warning
    upper = upper.astype(jnp.float32)

    assert jnp.all((upper - lower) > 0.)

    keys_p = jr.split(key, alpha.size)

    if hypercube:
        logger.info("Hypercube sampling...")

        sampler = qmc.LatinHypercube(d=alpha.size)
        samples = sampler.random(n=n_linear_sims)
        Y = jnp.asarray(qmc.scale(samples, lower, upper))
    else:
        logger.info("Uniform box sampling...")

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
def get_linearised_data(
    config: ConfigDict, 
    dataset: Dataset,
    *,
    n_linear_sims: Optional[int] = None
) -> tuple[Float[Array, "n d"], Float[Array, "n p"]]:
    """
        Get linearised PDFs and get their MLEs 

        # Pre-train data = Fisher summaries
        X_l, Y_l = get_fisher_summaries(
            summaries_key, 
            n=config.n_linear_sims, 
            parameter_prior=parameter_prior, 
            Finv=dataset.Finv
        )
    """
    
    logger.info("Linearising data...")

    key = jr.key(config.seed)

    key_parameters, key_simulations = jr.split(key)

    if config.n_linear_sims is not None:
        Y = sample_prior(
            key_parameters, 
            config.n_linear_sims if n_linear_sims is None else n_linear_sims, 
            dataset.alpha, 
            dataset.lower, 
            dataset.upper, 
            hypercube=True
        )
    else:
        Y = dataset.parameters

    assert dataset.derivatives.ndim == 3, (
        "Do derivatives [{}] have batch axis? Required.".format(dataset.derivatives.shape)
    )

    dmu = jnp.mean(dataset.derivatives, axis=0)
    mu = jnp.mean(dataset.fiducial_data, axis=0)

    def _simulator(key: PRNGKeyArray, pi: Float[Array, "p"]) -> Float[Array, "d"]:
        # Data model with linearised expectation
        _mu = linearised_model(alpha=dataset.alpha, alpha_=pi, mu=mu, dmu=dmu)
        return jr.multivariate_normal(key, mean=_mu, cov=dataset.C)

    keys = jr.split(key_simulations, len(Y))
    D = jax.vmap(_simulator)(keys, Y) 

    logger.info("... linearised data {} {}".format(D.shape, Y.shape))

    return D, Y 


@typecheck
def get_datavector(
    key: PRNGKeyArray, 
    config: ConfigDict, 
    dataset: Dataset, 
    n: int = 1, 
    *, 
    use_expectation: bool = False
) -> Float[Array, "... d"]:
    """ Measurement: either Gaussian linear model or not """

    # Choose a linearised model datavector or simply one of the Quijote realisations
    # which corresponds to a non-linearised datavector with Gaussian noise
    if config.use_expectation or use_expectation or FORCE_NOISELESS_DATAVECTOR:
        logger.info("Using expectation (noiseless datavector)...")

        datavector = jnp.mean(dataset.fiducial_data, axis=0, keepdims=True)
    else:
        if config.linearised:
            logger.info("Using linearised datavector...")

            mu = jnp.mean(dataset.fiducial_data, axis=0)

            datavector = jr.multivariate_normal(key, mean=mu, cov=dataset.C, shape=(n,))
        else:
            logger.info("Using non-linearised datavector...")

            # datavector = jr.choice(key, dataset.fiducial_data, shape=(n,))
            ix = jr.choice(key, jnp.arange(len(dataset.fiducial_data)), shape=(n,))
            datavector = dataset.fiducial_data[ix]

    if not (n > 1):
        datavector = jnp.squeeze(datavector, axis=0) # Remove batch axis by default

    return datavector 


"""
    Compression
"""


@typecheck
def get_linear_compressor(
    config: ConfigDict, 
    dataset: Dataset
) -> Callable[[Float[Array, "d"], Float[Array, "p"]], Float[Array, "p"]]:
    """ 
        Get Chi^2 minimisation function; compressing datavector 
        at estimated parameters to summary 
    """

    alpha = dataset.alpha
    F_planck = get_F_planck()

    logger.info("Using MAP linear compression" if config.use_planck else "Using MLE linear compression")

    @typecheck
    def mle(
        d: Float[Array, "d"], 
        pi: Float[Array, "p"], 
        Finv: Float[Array, "p p"], 
        mu: Float[Array, "d"], 
        dmu: Float[Array, "p d"], 
        precision: Float[Array, "d d"]
    ) -> Float[Array, "p"]:
        return pi + jnp.linalg.multi_dot([Finv, dmu, precision, d - mu])

    @typecheck
    def map(
        d: Float[Array, "d"], 
        pi: Float[Array, "p"], 
        Finv: Float[Array, "p p"], 
        mu: Float[Array, "d"], 
        dmu: Float[Array, "p d"], 
        precision: Float[Array, "d d"]
    ) -> Float[Array, "p"]:
        F = jnp.linalg.inv(Finv)
        _Finv = jnp.linalg.inv(F + F_planck)
        return pi + jnp.linalg.multi_dot([_Finv, dmu, precision, d - mu]) + jnp.linalg.multi_dot([_Finv, F_planck, alpha - pi])

    @typecheck
    def compressor(
        d: Float[Array, "d"], 
        p: Float[Array, "p"],
        mu: Float[Array, "d"], 
        dmu: Float[Array, "p d"]
    ) -> Float[Array, "p"]: 

        mu_p = linearised_model(
            alpha=alpha, alpha_=p, mu=mu, dmu=dmu
        )

        _estimator_fn = map if config.use_planck else mle

        p_ = _estimator_fn(
            d,
            pi=p,
            Finv=dataset.Finv, 
            mu=mu_p,            
            dmu=dmu, 
            precision=dataset.Cinv
        )

        return p_

    mu = jnp.mean(dataset.fiducial_data, axis=0)
    dmu = jnp.mean(dataset.derivatives, axis=0)

    return partial(compressor, mu=mu, dmu=dmu)


if 0:
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import svd


    class PCA:
        """
        Principal Component Analysis (PCA) for dimensionality reduction.

        Attributes:
            num_components (int): Number of principal components to keep.
            mean (jax.Array, optional): Mean of each feature in the training data.
            principal_components (jax.Array, optional): Principal components (eigenvectors) of the training data.
            explained_variance (jax.Array, optional): Explained variance of each principal component.
        """

        def __init__(self, num_components: int):
            self.num_components = num_components
            self.mean = None
            self.principal_components = None
            self.explained_variance = None

        def fit(self, X: jax.Array):
            n, m = X.shape

            if self.mean is None:
                self.mean = X.mean(axis=0)

            X_centred = X - self.mean
            S, self.principal_components = svd(X_centred, full_matrices=True)[1:]

            self.explained_variance = jnp.square(S) / jnp.sum(jnp.square(S))

        def transform(self, X: jax.Array):
            if self.principal_components is None:
                raise RuntimeError("Must fit before transforming.")

            X_centred = X - X.mean(axis=0)
            return jnp.dot(X_centred, self.principal_components[: self.num_components].T)

        def fit_transform(self, X: jax.Array):
            if self.mean is None:
                self.mean = X.mean(axis=0)

            X_centred = X - self.mean

            self.principal_components = svd(X_centred, full_matrices=True)[2]

            return jnp.dot(X_centred, self.principal_components[: self.num_components].T)

        def inverse_transform(self, X_transformed: jax.Array):
            if self.principal_components is None:
                raise RuntimeError("Must fit before transforming.")

            return (
                jnp.dot(X_transformed, self.principal_components[: self.num_components])
                + self.mean
            )


    from typing import Tuple, Optional, Sequence
    from functools import partial
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    from jax.sharding import NamedSharding, PositionalSharding
    import equinox as eqx
    from optimistix import minimise, BFGS, LevenbergMarquardt, rms_norm
    from jaxtyping import Key, Array, Float, Scalar, jaxtyped
    from beartype import beartype as typechecker
    import optax
    import numpy as np 
    from tqdm.auto import trange

    """
        Tools for compression with neural networks.
        - train a user-defined `eqx.Module` network that compresses a datavector
        to a model-dimensional summary, by minimising a MSE loss.
    """

    typecheck = jaxtyped(typechecker=typechecker)


    def loss(
        model: eqx.Module, 
        x: Float[Array, "b x"], 
        y: Float[Array, "b y"], 
        *,
        precision: Optional[Float[Array, "y y"]] = None
    ) -> Scalar:
        def fn(x, y):
            y_ = model(x)
            dy = jnp.subtract(y_, y)
            if precision is not None:
                l = jnp.linalg.multi_dot([dy, precision, dy.T]) # NOTE: which transpose?!
            else:
                l = jnp.square(dy)
            return l
        return jnp.mean(jax.vmap(fn)(x, y))


    @eqx.filter_jit
    def evaluate(
        model: eqx.Module, 
        x: Float[Array, "b x"], 
        y: Float[Array, "b y"],
        *, 
        precision: Optional[Float[Array, "y y"]] = None,
        replicated_sharding: Optional[PositionalSharding] = None
    ) -> Scalar:
        if replicated_sharding is not None:
            model = eqx.filter_shard(model, replicated_sharding)
        return loss(model, x, y, precision=precision)


    @typecheck
    @eqx.filter_jit
    def make_step(
        model: eqx.Module, 
        opt_state: optax.OptState,
        x: Float[Array, "b x"], 
        y: Float[Array, "b y"],
        opt: optax.GradientTransformation, 
        *, 
        precision: Optional[Float[Array, "y y"]] = None,
        replicated_sharding: Optional[PositionalSharding]
    ) -> Tuple[eqx.Module, optax.OptState, Scalar]:

        if replicated_sharding is not None:
            model, opt_state = eqx.filter_shard(
                (model, opt_state), replicated_sharding
            )

        grad_fn = eqx.filter_value_and_grad(partial(loss, precision=precision))

        loss_value, grads = grad_fn(model, x, y)

        updates, opt_state = opt.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        if replicated_sharding is not None:
            model, opt_state = eqx.filter_shard(
                (model, opt_state), replicated_sharding
            )

        return model, opt_state, loss_value


    def get_batch(
        D: Float[Array, "n x"], 
        Y: Float[Array, "n y"], 
        n: int, 
        key: Key
    ) -> Tuple[Float[Array, "b x"], Float[Array, "b y"]]:
        idx = jr.choice(key, jnp.arange(D.shape[0]), (n,))
        return D[idx], Y[idx]


    @typecheck
    def fit_nn(
        key: Key[jnp.ndarray, "..."], 
        model: eqx.Module, 
        train_data: Tuple[Float[Array, "n x"], Float[Array, "n y"]], 
        opt: optax.GradientTransformation, 
        n_batch: int, 
        patience: Optional[int], 
        n_steps: int = 10_000, 
        valid_fraction: int = 0.9, 
        valid_data: Sequence[Array] = None,
        batch_dataset: bool = True,
        use_tqdm: bool = False,
        *,
        precision: Optional[Float[Array, "y y"]] = None,
        sharding: Optional[NamedSharding] = None,
        replicated_sharding: Optional[PositionalSharding] = None,
    ) -> Tuple[eqx.Module, Float[np.ndarray, "l 2"]]:
        """
        Trains a neural network model with early stopping.

        Args:
            key: A `PRNGKeyArray`.
            model: The neural network model to be trained, represented as an `eqx.Module`.
            D: The input data matrix (`Array`), where rows are data points and columns are features.
            Y: The target values (`Array`) corresponding to the input data.
            opt: The optimizer to be used for gradient updates, defined as an `optax.GradientTransformation`.
            n_batch: The number of data points per mini-batch for each training step (`int`).
            patience: The number of steps to continue without improvement on the validation loss 
                before early stopping is triggered (`int`).
            n_steps: The maximum number of training steps to perform (`int`, optional). Default is 100,000.
            valid_fraction: The fraction of the data to use for training, with the remainder
                used for validation (`float`, optional). Default is 0.9 (90% training, 10% validation).

        Returns:
            Tuple[`eqx.Module`, `Array`]: 
                - The trained `model` after the optimization process.
                - A 2D array of shape (n_steps, 2), where the first column contains the training loss at each 
                step, and the second column contains the validation loss.
        
        Notes:
            1. The data `D` and targets `Y` are split into training and validation sets based on the 
            `valid_fraction` parameter.
            4. Early stopping occurs if the validation loss does not improve within a specified 
            number of steps (`patience`).
            5. The function returns the trained model and the recorded training/validation loss history.
        """
        D, Y = train_data

        n_s, _ = D.shape

        opt_state = opt.init(eqx.filter(model, eqx.is_array))

        if valid_data is not None:
            Xt, Yt = train_data
            Xv, Yv = valid_data
        else:
            Xt, Xv = jnp.split(D, [int(valid_fraction * n_s)]) 
            Yt, Yv = jnp.split(Y, [int(valid_fraction * n_s)])

        if use_tqdm: 
            steps = trange(n_steps, desc="Training NN", colour="blue")
        else: 
            steps = trange(n_steps)

        L = np.zeros((n_steps, 2))
        for step in steps:
            key_t, key_v = jr.split(jr.fold_in(key, step))

            if batch_dataset:
                x, y = get_batch(Xt, Yt, n=n_batch, key=key_t) # Xt, Yt
            else:
                x, y = Xt, Yt
            
            if sharding is not None:
                x, y = eqx.filter_shard((x, y), sharding)

            model, opt_state, train_loss = make_step(
                model, 
                opt_state, 
                x, 
                y, 
                opt=opt, 
                precision=precision, 
                replicated_sharding=replicated_sharding
            )

            if batch_dataset:
                x, y = get_batch(Xv, Yv, n=n_batch, key=key_v)
            else:
                x, y = Xv, Yv

            if sharding is not None:
                x, y = eqx.filter_shard((x, y), sharding)

            valid_loss = evaluate(
                model, x, y, precision=precision, replicated_sharding=replicated_sharding
            )

            L[step] = train_loss, valid_loss
            steps.set_postfix_str(
                "train={:.3E}, valid={:.3E}".format(train_loss.item(), valid_loss.item())
            )

            if patience is not None:
                if (step > 0) and (step - np.argmin(L[:step, 1]) > patience):
                    steps.set_description_str("Stopped at {}".format(step))
                    break

        return model, L[:step]


    """
        L-BFGS
    """

    @typecheck
    @eqx.filter_jit(donate="all-except-first")
    def make_step_lbfgs(
        net: eqx.Module, 
        opt_state: optax.OptState, 
        X: Float[Array, "n d"], 
        P: Float[Array, "n p"], 
        *,
        opt: optax.GradientTransformation,
        precision: Optional[Float[Array, "p p"]] = None, 
        replicated_sharding: Optional[jax.sharding.NamedSharding] = None
    ) -> tuple[eqx.Module, optax.OptState, Scalar]:
        f = partial(loss, x=X, y=P, precision=precision)
        value_and_grad_fn = optax.value_and_grad_from_state(f)
        l, grad = value_and_grad_fn(net, state=opt_state)
        updates, opt_state = opt.update(
            grad, opt_state, net, value=l, grad=grad, value_fn=f 
        )
        net = eqx.apply_updates(net, updates)
        return net, opt_state, l 


    def fit_nn_lbfgs(
        key: Key[jnp.ndarray, "..."], 
        model: eqx.Module, 
        train_data: Tuple[Float[Array, "n x"], Float[Array, "n y"]], 
        valid_fraction: int = 0.9, 
        valid_data: Sequence[Array] = None,
        batch_dataset: bool = True,
        *,
        precision: Optional[Float[Array, "y y"]] = None,
        sharding: Optional[NamedSharding] = None,
        replicated_sharding: Optional[PositionalSharding] = None,
    ) -> Tuple[eqx.Module, Float[np.ndarray, "l 2"]]:

        D, Y = train_data

        y0, static = eqx.partition(model, eqx.is_array)

        # Standardise before PCA (don't get tricked by high variance due to units)
        # X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0) # NOTE: already standardised 

        # Fit whitening-PCA to compressed simulations
        # pca = PCA(num_components=D.shape[-1]) 
        # D = (D - jnp.mean(D, axis=0)) / jnp.std(D, axis=0)
        # pca.fit(D) # Fit on fiducial data?
        # D = pca.transform(D)

        @eqx.filter_jit
        def f(y, args):
            # Dataset to fit network to
            D, Y = args
            # Combine iteration parameters and architecture
            model = eqx.combine(y, static)
            return loss(model, D, Y, precision=precision)

        # BFGS optimisation
        res = minimise(
            f,
            BFGS(rtol=1e-6, atol=1e-6, norm=rms_norm),
            y0=y0,
            max_steps=1_000_000,
            args=(D, Y)
        )

        # Put solution parameters into model
        model = eqx.combine(res.value, static)

        L = np.zeros((1, 2))

        # _model = lambda d: pca.transform(model(d))

        return _model, L


    @typecheck
    def get_nn_compressor(
        key: PRNGKeyArray, 
        dataset: Dataset, 
        data_preprocess_fn: Optional[Callable] = None, 
        *, 
        lbfgs: bool = False, 
        results_dir: str, 
        net: Optional[eqx.Module] = None
    ) -> tuple[eqx.Module, Callable]:
        """
            Train neural network compression function
            - Optionally use parameter covariance for chi2 loss
        """
        net_key, train_key = jr.split(key)

        if data_preprocess_fn is None:
            data_preprocess_fn = lambda x: x

        if net is None:
            net = eqx.nn.MLP(
                dataset.data.shape[-1], 
                dataset.parameters.shape[-1], 
                width_size=32, 
                depth=3, 
                activation=jax.nn.tanh,
                key=net_key
            )

        def preprocess_fn(x): 
            # Preprocess with covariance?
            return (jnp.asarray(x) - jnp.mean(dataset.data, axis=0)) / jnp.std(dataset.data, axis=0)

        if lbfgs:
            net, losses = fit_nn_lbfgs(
                train_key, 
                net, 
                (preprocess_fn(data_preprocess_fn(dataset.data)), dataset.parameters), 
                # precision=jnp.linalg.inv(dataset.Finv) # In reality this varies with parameters
            )
        else:
            net, losses = fit_nn(
                train_key, 
                net, 
                (preprocess_fn(data_preprocess_fn(dataset.data)), dataset.parameters), 
                opt=optax.adam(1e-3), 
                precision=jnp.linalg.inv(dataset.Finv), # In reality this varies with parameters
                n_batch=500, 
                patience=1000,
                n_steps=50_000
            )

        plt.figure()
        plt.loglog(losses)
        plt.savefig(os.path.join(results_dir, "losses_nn.png"))
        plt.close()

        return net, preprocess_fn


def get_compression_fn(key, config, dataset, *, results_dir):
    """ 
        Get linear or neural network compressor
    """ 

    assert config.compression in ["linear", "nn", "nn-lbfgs"]

    # if config.compression == "nn" or config.compression == "nn-lbfgs":

    #     net, preprocess_fn = get_nn_compressor(
    #         key, 
    #         dataset, 
    #         lbfgs=config.compression == "nn-lbfgs", 
    #         results_dir=results_dir
    #     )

    #     compressor = lambda d, p: net(preprocess_fn(d)) # Ignore parameter kwarg!

    if config.compression == "linear":
        compressor = get_linear_compressor(config, dataset)

    compression_fn = lambda d, p: compressor(d, p)

    return compression_fn 