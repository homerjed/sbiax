import jax.numpy as jnp
import jax.random as jr 


def linearized_model(_alpha, fiducial_dv, alpha, derivatives):
    return fiducial_dv + jnp.dot(_alpha - alpha, derivatives)


def simulator(key, parameters, alpha, fiducial_dv, derivatives, covariance):
    d = jr.multivariate_normal(
        key=key, 
        mean=linearized_model(
            _alpha=parameters, 
            fiducial_dv=fiducial_dv, 
            alpha=alpha, 
            derivatives=derivatives
        ),
        cov=covariance
    ) 
    return d


def _mle(d, pi, Finv, mu, dmu, precision):
    return pi + jnp.linalg.multi_dot([Finv, dmu, precision, d - mu])