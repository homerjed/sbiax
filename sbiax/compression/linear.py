import jax.numpy as jnp
import jax.random as jr 
from jaxtyping import Key, Array, Float, jaxtyped
from beartype import beartype as typechecker


@jaxtyped(typechecker=typechecker)
def linearized_model(
    pi_: Float[Array, "p"], 
    mu: Float[Array, "d"], 
    pi: Float[Array, "p"], 
    dmu: Float[Array, "p d"]
) -> Float[Array, "p"]:
    """
        Calculate a linearised model prediction given a set of parameters.

        Args:
            pi_ (`Array`): The parameters of the model to calculate.
            mu (`Array`): The model evaluated at the estimated set of parameters `pi`.
            pi (`Array`): A fiducial set of parameters (e.g. those the `covariance` is calculated at)
            dmu (`Array`): The first-order theory derivatives (for the implicitly assumed linear model, 
                these are parameter independent!)

        Returns:
            `Array`: a model prediction at the parameters `pi_`
    """
    return mu + jnp.dot(pi_ - pi, dmu)


@jaxtyped(typechecker=typechecker)
def simulator(
    key: Key[jnp.ndarray, "..."], 
    pi_: Float[Array, "p"], 
    pi: Float[Array, "p"], 
    mu: Float[Array, "d"], 
    dmu: Float[Array, "p d"], 
    covariance: Float[Array, "d d"]
) -> Float[Array, "d"]:
    """
        Simulate from a Gaussian likelihood defined by a model `mu` and a data covariance `covariance`.

        Args:
            key (`Key`): A JAX random key.
            pi (`Array`): The estimated parameters of the datavector.
            alpha (`Array`): A fiducial set of parameters (e.g. those the `covariance` is calculated at)
            mu (`Array`): The model evaluated at the estimated set of parameters `pi`.
            dmu (`Array`): The first-order theory derivatives (for the implicitly assumed linear model, 
                these are parameter independent!)
            covariance (`Array`): The data covariance matrix.

        Returns:
            `Array`: a datavector drawn from the Gaussian likelihood.
    """
    d = jr.multivariate_normal(
        key=key, 
        mean=linearized_model(
            pi_=pi_, 
            mu=mu, 
            pi=pi, 
            dmu=dmu 
        ),
        cov=covariance
    ) 
    return d


@jaxtyped(typechecker=typechecker)
def mle(
    d: Float[Array, "d"], 
    pi: Float[Array, "p"], 
    Finv: Float[Array, "p p"], 
    mu: Float[Array, "d"], 
    dmu: Float[Array, "p d"], 
    precision: Float[Array, "d d"]
) -> Float[Array, "p"]:
    """
        Calculates a maximum likelihood estimator (MLE) from a datavector by
        assuming a linear model `mu` in parameters `pi` and using

        Args:
            d (`Array`): The datavector to compress.
            p (`Array`): The estimated parameters of the datavector (e.g. a fiducial set).
            Finv (`Array`): The Fisher matrix. Calculated with a precision matrix (e.g. `precision`) and 
                theory derivatives.
            mu (`Array`): The model evaluated at the estimated set of parameters `pi`.
            dmu (`Array`): The first-order theory derivatives (for the implicitly assumed linear model, 
                these are parameter independent!)
            precision (`Array`): The precision matrix - defined as the inverse of the data covariance matrix.

        Returns:
            `Array`: the MLE.
    """
    return pi + jnp.linalg.multi_dot([Finv, dmu, precision, d - mu])