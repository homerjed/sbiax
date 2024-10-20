import jax.numpy as jnp
import jax.random as jr 


def linearized_model(_alpha, mu, alpha, derivatives):
    return mu + jnp.dot(_alpha - alpha, derivatives)


def simulator(key, parameters, alpha, mu, derivatives, covariance):
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


def _mle(d, pi, Finv, mu, dmu, precision):
    return pi + jnp.linalg.multi_dot([Finv, dmu, precision, d - mu])