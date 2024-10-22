from typing import Any, Optional, Tuple, Callable
import jax
import jax.numpy as jnp
import jax.random as jr  
import equinox as eqx
from jaxtyping import PRNGKeyArray, Array, Float, jaxtyped
from beartype import beartype as typechecker
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd


class GMM(eqx.Module):
    event_dim: int 
    context_dim: int 
    n_components: int 
    covariance_eps: float 
    covariance_init: float 

    net: eqx.Module
    activation: Callable 
    sigma_tri_dim: int
    sigmas_out_shape: Tuple[int]

    _alpha: eqx.Module
    _mean: eqx.Module
    _sigma: eqx.Module

    x_dim: int
    y_dim: int

    def __init__(
        self,
        event_dim: int,
        context_dim: int,
        n_components: int,
        width_size: int,
        depth: int,
        activation: Callable = jax.nn.tanh,
        covariance_init: float = 1e-8,
        covariance_eps: float = 1e-8, 
        *,
        key: PRNGKeyArray
    ):
        self.event_dim = event_dim
        self.context_dim = context_dim
        self.n_components = n_components
        self.covariance_eps = covariance_eps
        self.covariance_init = covariance_init
        self.activation = activation

        self.sigma_tri_dim = (self.context_dim * (self.context_dim + 1)) // 2

        key_net, key_alpha, key_mean, key_sigma = jr.split(key, 4)
        self.net = eqx.nn.MLP(
            self.context_dim,
            width_size,
            width_size=width_size, 
            depth=depth, 
            final_activation=self.activation, 
            key=key_net
        )

        if n_components == 1:
            self._alpha = lambda x: jnp.ones((x.shape[0], 1))
        else:
            key, _key = jr.split(key)
            self._alpha = eqx.nn.Linear(
                width_size, n_components, key=key_alpha
            )

        self._mean = eqx.nn.Linear(
            width_size, 
            n_components * self.context_dim, 
            key=key_mean
        )
        self._sigma = eqx.nn.Linear(
            width_size,
            n_components * self.sigma_tri_dim, 
            key=key_sigma
        )
        self.sigmas_out_shape = (self.n_components,) + ((self.context_dim * (self.context_dim + 1)) // 2,)

        self.x_dim = event_dim
        self.y_dim = context_dim

    def regularise_diagonal(self, x: Array) -> Array:
        """ Subtract diagonal and replace it with a positively activated one. """
        diag = jnp.diag(jnp.exp(jnp.diag(x))) # Positive activation on diagonal
        regularize = jnp.eye(x.shape[-1]) * self.covariance_eps # Avoid overfitting (factor depends on Finv?)
        x = x - jnp.diag(jnp.diag(x)) 
        x = x + diag + regularize
        return x 

    def __call__(self, parameters: Array) -> tfd.Distribution:
        net_out = self.net(parameters)
        alphas = self._alpha(net_out)
        means = self._mean(net_out)
        means = means.reshape(self.n_components, self.context_dim)

        sigmas = self._sigma(net_out)
        sigmas = sigmas.reshape(self.sigmas_out_shape)
        sigmas = tfp.math.fill_triangular(sigmas) 

        cov_shape = (self.event_dim, self.event_dim)
        covariance = jax.vmap(self.regularise_diagonal)(sigmas.reshape((-1, *cov_shape)))
        covariance = covariance.reshape((self.n_components, *cov_shape))

        # GMM defined as a distribution. Mixture of neurally parameterised Gaussians.
        _alpha = jax.nn.softmax(alphas, axis=0) # sum(alpha) = 1 for each in batch
        weights_dist = tfd.Categorical(probs=_alpha)

        # Full covariance distribution for components
        components_dist = tfd.MultivariateNormalTriL(loc=means, scale_tril=sigmas)
        
        gmm = tfd.MixtureSameFamily(
            mixture_distribution=weights_dist, components_distribution=components_dist)
        return gmm
    
    def get_parameters(self, parameters: Array) -> Tuple[Array, Array, Array]:
        net_out = self.net(parameters)
        alphas = self._alpha(net_out)
        mean = self._mean(net_out)
        sigmas = self._sigma(net_out)

        sigmas = sigmas.reshape(self.sigmas_out_shape)
        sigmas = tfp.math.fill_triangular(sigmas) 

        cov_shape = (self.event_dim, self.event_dim)
        covariance = jax.vmap(self.regularise_diagonal)(sigmas.reshape((-1, *cov_shape)))
        covariance = covariance.reshape((self.n_components, *cov_shape))

        alpha = jax.nn.softmax(alphas, axis=0) 
        return mean, alpha, covariance

    @jaxtyped(typechecker=typechecker)
    def loss(
        self, 
        x: Float[Array, "{self.x_dim}"], 
        y: Float[Array, "{self.y_dim}"], 
        key: Optional[PRNGKeyArray]
    ) -> Float[Array, ""]:
        return self.log_prob(x, y)

    @jaxtyped(typechecker=typechecker)
    def log_prob(
        self,
        x: Float[Array, "{self.x_dim}"], 
        y: Float[Array, "{self.y_dim}"], 
        key: Optional[PRNGKeyArray]
    ) -> Float[Array, ""]:
        x = jnp.atleast_1d(y)
        y = jnp.atleast_1d(y)
        return self.__call__(y).log_prob(x) 
