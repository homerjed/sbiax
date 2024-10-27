import jax.numpy as jnp
import jax.random as jr 

"""
    Tools for linear compression.
    - derive an MLE given a datavector and the data covariance, expectation,
      theory derivatives and parameters.
"""


def linearized_model(_alpha, mu, alpha, derivatives):
    return mu + jnp.dot(_alpha - alpha, derivatives)


def simulator(key, parameters, alpha, mu, derivatives, covariance):
    """
        Simulate from a Gaussian likelihood.
    """
    d = jr.multivariate_normal(
        key=key, 
        mean=linearized_model(
            _alpha=parameters, 
            mu=mu, 
            alpha=alpha, 
            derivatives=derivatives
        ),
        cov=covariance
    ) 
    return d


def mle(d, pi, Finv, mu, dmu, precision):
    return pi + jnp.linalg.multi_dot([Finv, dmu, precision, d - mu])