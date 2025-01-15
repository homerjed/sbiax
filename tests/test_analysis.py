import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax as dfx
import optax
import tensorflow_probability.substrates.jax.distributions as tfd

from sbiax.ndes import CNF, Ensemble, Scaler
from sbiax.train import train_ensemble
from sbiax.inference import nuts_sample

"""
    Test a run-through of code with NPE / NLE on a toy problem.
"""

def get_data(key):

    def _simulator(key, p):
        mu, sigma = p
        return mu + jr.normal(key, (2,)) * sigma

    keys = jr.split(key, 100)
    Y = jnp.stack(
        [jnp.linspace(-1., 1., 100), jnp.linspace(0.5, 2.0, 100)], axis=1
    )
    X = jax.vmap(_simulator)(keys, Y)
    return X, Y


def test_nle():
    key = jr.key(0)

    model_key, train_key, sample_key = jr.split(key, 3)

    X, Y = get_data(key)

    _, data_dim = X.shape
    _, parameter_dim = Y.shape

    parameter_prior = tfd.Blockwise(
        [tfd.Uniform(-1., 1.) for p in range(parameter_dim)]
    )

    model_keys = jr.split(model_key, 2)

    scaler = Scaler(X, Y, use_scaling=False)

    solver = dfx.Euler()

    ndes = [
        CNF(
            event_dim=parameter_dim, 
            context_dim=parameter_dim, 
            width_size=8,
            depth=0,
            solver=solver,
            activation=jax.nn.tanh,
            dt=0.1, 
            t1=1.0, 
            dropout_rate=0.,
            exact_log_prob=True,
            scaler=scaler,
            key=key
        )
        for key in model_keys
    ]

    ensemble = Ensemble(ndes, sbi_type="nle")

    opt = optax.adamw(1e-2)

    ensemble, stats = train_ensemble(
        train_key, 
        ensemble,
        train_mode="nle",
        train_data=(X, Y), 
        opt=opt,
        n_batch=10,
        patience=2,
        n_epochs=5,
        results_dir=None
    )

    key_data, key_sample = jr.split(sample_key)

    mu = jnp.zeros((2,))
    covariance = jnp.eye(2)

    X_ = jr.multivariate_normal(key_data, mu, covariance)

    ensemble = eqx.nn.inference_mode(ensemble)

    log_prob_fn = ensemble.ensemble_log_prob_fn(X_, parameter_prior)

    samples, samples_log_prob = nuts_sample(
        key_sample, log_prob_fn, prior=parameter_prior, n_samples=10
    )

    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all(jnp.isfinite(samples_log_prob))
    assert samples.shape[-1] == parameter_dim


def test_npe():
    key = jr.key(0)

    model_key, train_key, sample_key = jr.split(key, 3)

    X, Y = get_data(key)
    
    _, data_dim = X.shape
    _, parameter_dim = Y.shape

    parameter_prior = tfd.Blockwise(
        [tfd.Uniform(-1., 1.) for p in range(parameter_dim)]
    )

    model_keys = jr.split(model_key, 2)

    scaler = Scaler(X, Y, use_scaling=False)

    solver = dfx.Euler()

    ndes = [
        CNF(
            event_dim=parameter_dim, 
            context_dim=parameter_dim, 
            width_size=8,
            depth=0,
            solver=solver,
            activation=jax.nn.tanh,
            dt=0.1, 
            t1=1.0, 
            dropout_rate=0.,
            exact_log_prob=True,
            scaler=scaler,
            key=key
        )
        for key in model_keys
    ]

    ensemble = Ensemble(ndes, sbi_type="npe")

    opt = optax.adamw(1e-2)

    ensemble, stats = train_ensemble(
        train_key, 
        ensemble,
        train_mode="npe",
        train_data=(X, Y), 
        opt=opt,
        n_batch=10,
        patience=2,
        n_epochs=5,
        results_dir=None
    )

    key_data, key_sample = jr.split(sample_key)

    mu = jnp.zeros((2,))
    covariance = jnp.eye(2)

    X_ = jr.multivariate_normal(key_data, mu, covariance)

    ensemble = eqx.nn.inference_mode(ensemble)

    log_prob_fn = ensemble.ensemble_log_prob_fn(X_, parameter_prior)

    samples, samples_log_prob = nuts_sample(
        key_sample, log_prob_fn, prior=parameter_prior, n_samples=10
    )

    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all(jnp.isfinite(samples_log_prob))
    assert samples.shape[-1] == parameter_dim