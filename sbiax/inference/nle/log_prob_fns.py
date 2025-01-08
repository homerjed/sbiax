from functools import partial
from typing import Callable, Sequence, Any, Tuple
import jax
import jax.numpy as jnp
import jax.random as jr 
import equinox as eqx
from jaxtyping import Key, Array
from tensorflow_probability.substrates.jax.distributions import Distribution 

Prior = Distribution 


def nde_log_prob(
    nde: eqx.Module, data: Array, theta: Array
) -> Array:
    """ NDE log-likelihood log_prob(data|theta) """
    nde = eqx.nn.inference_mode(nde, True)
    return nde.log_prob(x=data, y=theta) 


def nde_log_prob_posterior(
    nde: eqx.Module, data: Array, theta: Array, prior: Prior
) -> Array:
    """ NDE log-poserior logp(data|theta) + logprior(theta) """
    log_likelihood = nde_log_prob(nde, data, theta) 
    log_prior = prior.log_prob(theta)
    log_posterior = log_likelihood + log_prior
    return log_posterior


def get_nde_log_prob_fn(
    nde: eqx.Module, data: Array, prior: Prior
) -> Callable:
    def nde_log_prob_fn(theta):
        return nde_log_prob_posterior(
            nde=nde, 
            data=data, 
            theta=theta, 
            prior=prior
        )
    return eqx.filter_jit(nde_log_prob_fn)


def sample_state(
    key: Key, 
    asymptotic: Distribution | Any, 
    n_walkers: int, 
    chain_length: int
) -> Array:
    key_sample, key_choice = jr.split(key)
    n_samples = 2 * n_walkers * chain_length

    posterior_samples = asymptotic.sample_n(key_sample, n_samples)
    posterior_weights = jnp.full((n_samples,), 1. / n_samples)

    key, _ = jr.split(key)
    p = posterior_weights / posterior_weights.sum()
    ix = jr.choice(
        key_choice,
        jnp.arange(len(posterior_samples)), 
        p=p,
        replace=False, 
        shape=(2 * n_walkers,)
    )

    state = posterior_samples[ix]
    return state


def _get_ix(
    key: Key, 
    samples: Array, 
    weights: Array, 
    n_walkers: int
) -> Array:
    p = weights / weights.sum()
    ix = jr.choice(
        key,
        jnp.arange(len(samples)), 
        p=p,
        replace=False, 
        shape=(2 * n_walkers,)
    )
    return ix 


def _sample_initial_state(
    key: Key, 
    asymptotic: Any, 
    n_walkers: int, 
    chain_length: int
) -> Tuple[Array, Array]:
    n_samples = 2 * n_walkers * chain_length
    samples = asymptotic.sample_n(key, n_samples)
    weights = jnp.full((n_samples,), 1. / n_samples)
    return samples, weights


def sample_nde_state(
    key: Key, 
    nde: eqx.Module, 
    asymptotic: Any, 
    n_walkers: int, 
    chain_length: int
) -> Array:
    """ 
        Sample posterior samples / weights for state from Fisher results
        or a new walkers state
    """
    if nde.fisher_posterior is not None:
        samples = nde.fisher_posterior.samples 
        weights = nde.fisher_posterior.weights
    else:
        key, _ = jr.split(key)
        samples, weights = _sample_initial_state(
            key, asymptotic, n_walkers, chain_length
        )
    key, _ = jr.split(key)
    ix = _get_ix(key, samples, weights, n_walkers)
    state = samples[ix]
    return state