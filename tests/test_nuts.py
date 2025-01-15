def test_nuts():

    from functools import partial
    
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import tensorflow_probability.substrates.jax.distributions as tfd

    from sbiax.inference import nuts_sample

    # Sample a posterior G[p|0, 1] given a datavector d ~ G[d|[p, p], I]

    key = jr.key(0)

    dim = 2
    n_samples = 100_000

    mu = jnp.zeros((dim,))
    cov = jnp.identity(dim)

    parameter_prior = tfd.Normal(loc=jnp.zeros((1,)), scale=jnp.ones((1,)))

    @jax.jit
    def log_prob_fn(d, p): 
        log_likelihood = jax.scipy.stats.multivariate_normal.logpdf(d, mu * p, cov) 
        log_prior = parameter_prior.log_prob(p)
        log_posterior = log_likelihood + log_prior
        return jnp.squeeze(log_posterior) # Grads of this function needed

    key_sample, key_data = jr.split(key)

    data = jr.multivariate_normal(key_data, mu, cov)

    samples_a, samples_log_prob_a = nuts_sample(
        key_sample, 
        log_prob_fn=partial(log_prob_fn, data), 
        n_samples=n_samples,
        prior=parameter_prior
    )

    samples_b, samples_log_prob_b = nuts_sample(
        key_sample, 
        log_prob_fn=partial(log_prob_fn, data), 
        n_samples=n_samples,
        prior=parameter_prior
    )

    # Remove chains axes
    samples_a = jnp.squeeze(samples_a)
    samples_b = jnp.squeeze(samples_a)
    samples_log_prob_a = jnp.squeeze(samples_log_prob_a)
    samples_log_prob_b = jnp.squeeze(samples_log_prob_b)

    assert jnp.all(jnp.isfinite(samples_a))
    assert jnp.all(jnp.isfinite(samples_log_prob_a))

    assert jnp.allclose(samples_a, samples_b)  
    assert jnp.allclose(samples_log_prob_a, samples_log_prob_b)  

    assert jnp.allclose(jnp.mean(samples_a, axis=0), jnp.zeros((dim,)), atol=1e-2)
    assert jnp.allclose(jnp.var(samples_a, axis=0), jnp.ones((dim,)), atol=1e-2)