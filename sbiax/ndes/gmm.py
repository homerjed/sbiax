from typing import Any, Optional, Tuple, Callable
import jax
import jax.numpy as jnp
import jax.random as jr  
import equinox as eqx
from jaxtyping import Key, Array, Float, jaxtyped
from beartype import beartype as typechecker
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd


class GMM(eqx.Module):
    """
    Implements a Gaussian Mixture Model (GMM) with neural network parameterization for density 
    estimation, supporting custom covariance regularization and multiple mixture components.

    Attributes:
        event_dim (`int`): Dimensionality of the input event (e.g., data).
        context_dim (`int`): Dimensionality of the conditioning context (e.g., parameters).
        n_components (`int`): Number of Gaussian components in the mixture.
        covariance_eps (`float`): Small positive value to add to the covariance diagonals for stability.
        covariance_init (`float`): Initial value for the covariance matrix components.
        net (`eqx.Module`): Neural network module used to parameterize GMM weights, means, and covariances.
        activation (`Callable`): Activation function applied to the network outputs.
        sigma_tri_dim (`int`): Dimension of the triangular portion of the covariance matrix.
        sigmas_out_shape (`Tuple[int]`): Shape of the output tensor for covariance parameters.
        _alpha (`eqx.Module`): Network component predicting mixture weights (alphas).
        _mean (`eqx.Module`): Network component predicting the means of each Gaussian component.
        _sigma (`eqx.Module`): Network component predicting the covariances of each component.
        x_dim (`int`): Dimensionality of the data.
        y_dim (`int`): Dimensionality of the conditioning parameters.

    Methods:
        __init__(event_dim: `int`, context_dim: `int`, n_components: `int`, 
                width_size: `int`, depth: `int`, activation: `Callable` = jax.nn.tanh, 
                covariance_init: `float` = 1e-8, covariance_eps: `float` = 1e-8, key: `PRNGKeyArray`):
            Initializes the GMM model, including setting up the neural network components.

        regularise_diagonal(x: `Array`) -> `Array`:
            Applies positive activation on the diagonal of the covariance matrix to ensure stability.

        __call__(parameters: `Array`) -> `tfd.Distribution`:
            Generates a GMM distribution instance using neural network outputs.

        get_parameters(parameters: `Array`) -> `Tuple[Array, Array, Array]`:
            Returns the alphas, means, and covariance matrices for each Gaussian component.

        loss(x: `Float[Array, "{self.x_dim}"]`, y: `Float[Array, "{self.y_dim}"]`, key: `Optional[PRNGKeyArray]`) -> `Float[Array, ""]`:
            Computes the negative log-probability for given data and conditioning parameters.

        log_prob(x: `Float[Array, "{self.x_dim}"]`, y: `Float[Array, "{self.y_dim}"]`, key: `Optional[PRNGKeyArray]`) -> `Float[Array, ""]`:
            Returns the log-probability for the given data and parameters.
    """
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

    scaler: eqx.Module
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
        scaler: eqx.Module = None,
        *,
        key: Key 
    ):
        """
        Initializes the Gaussian Mixture Model with specified event and context dimensions, 
        number of mixture components, neural network width, depth, and activation.

        Args:
            event_dim (`int`): Dimensionality of the data.
            context_dim (`int`): Dimensionality of the conditioning parameters.
            n_components (`int`): Number of Gaussian components in the mixture.
            width_size (`int`): Width of each hidden layer in the neural network.
            depth (`int`): Number of layers in the neural network.
            activation (`Callable`, optional): Activation function for neural network layers. 
                Defaults to `jax.nn.tanh`.
            covariance_init (`float`, optional): Initial value for covariance entries. Defaults to `1e-8`.
            covariance_eps (`float`, optional): Small positive value for covariance regularization. 
                Defaults to `1e-8`.
            key (`PRNGKeyArray`): Random key for parameter initialization.
        """
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
            self._alpha = lambda x: jnp.ones((1,))
        else:
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

        self.scaler = scaler
        self.x_dim = event_dim
        self.y_dim = context_dim

    def regularise_diagonal(
        self, 
        x: Float[Array, "{self.x_dim} {self.x_dim}"]
    ) -> Float[Array, "{self.x_dim} {self.x_dim}"]:
        """
        Applies positive activation to the diagonal of the covariance matrix to ensure non-negativity.

        Args:
            x (`Array`): A covariance matrix input.

        Returns:
            `Array`: The covariance matrix with a positive diagonal and regularization applied.
        """
        diag = jnp.diag(jnp.exp(jnp.diag(x))) # Positive activation on diagonal
        regularize = jnp.eye(x.shape[-1]) * self.covariance_eps # Avoid overfitting
        x = x - jnp.diag(jnp.diag(x)) 
        x = x + diag + regularize
        return x 

    def __call__(self, parameters: Float[Array, "{self.y_dim}"]) -> tfd.Distribution:
        """
        Generates a `tfd.MixtureSameFamily` distribution representing the Gaussian mixture model.

        Args:
            parameters (`Array`): Context (conditioning parameters)

        Returns:
            `tfd.Distribution`: A Gaussian mixture distribution with component weights, means, and 
                covariances predicted by the neural network.
        """

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

        weights_dist = tfd.Categorical(probs=jax.nn.softmax(alphas))

        # Full covariance distribution for components
        components_dist = tfd.MultivariateNormalTriL(loc=means, scale_tril=sigmas)
        
        gmm = tfd.MixtureSameFamily(
            mixture_distribution=weights_dist, 
            components_distribution=components_dist
        )
        return gmm
    
    def get_parameters(
        self, parameters: Float[Array, "{self.y_dim}"]
    ) -> Tuple[
        Float[Array, "{self.n_components} {self.x_dim}"], 
        Float[Array, "{self.n_components}"], 
        Float[Array, "{self.n_components} {self.x_dim} {self.x_dim}"]
    ]:
        """
        Gets the mixture component weights, means, and covariance matrices for the given parameters.

        Args:
            parameters (`Array`): Input parameter vector for the model.

        Returns:
            `Tuple[Array, Array, Array]`: Returns a tuple containing mixture weights (alphas), 
            means, and covariance matrices for each component in the GMM.
        """
        net_out = self.net(parameters)
        alphas = self._alpha(net_out)
        mean = self._mean(net_out)
        sigmas = self._sigma(net_out)

        sigmas = sigmas.reshape(self.sigmas_out_shape)
        sigmas = tfp.math.fill_triangular(sigmas) 

        cov_shape = (self.event_dim, self.event_dim)
        covariance = jax.vmap(self.regularise_diagonal)(sigmas.reshape((-1, *cov_shape)))
        covariance = covariance.reshape((self.n_components, *cov_shape))

        alpha = jax.nn.softmax(alphas) 
        return mean, alpha, covariance

    @jaxtyped(typechecker=typechecker)
    def loss(
        self, 
        x: Float[Array, "{self.x_dim}"], 
        y: Float[Array, "{self.y_dim}"], 
        key: Optional[Key[jnp.ndarray, "..."]] = None
    ) -> Float[Array, ""]:
        """
        Computes the loss as the negative log-probability of the data given the conditioning parameters.

        Args:
            x (`Float[Array, "{self.x_dim}"]`): Input data sample.
            y (`Float[Array, "{self.y_dim}"]`): Conditioning context vector.
            key (`Optional[PRNGKeyArray]`): Optional random key.

        Returns:
            `Float[Array, ""]`: Negative log-probability value as the loss.
        """
        return -self.log_prob(x, y)

    @jaxtyped(typechecker=typechecker)
    def sample_and_log_prob(
        self, 
        key: Key[jnp.ndarray, "..."], 
        y: Float[Array, "{self.y_dim}"]
    ) -> Tuple[Float[Array, "{self.x_dim}"], Float[Array, ""]]:
        x = self.__call__(y).sample(seed=key)
        log_prob = self.log_prob(x, y)
        return x, log_prob

    @jaxtyped(typechecker=typechecker)
    def log_prob(
        self,
        x: Float[Array, "{self.x_dim}"], 
        y: Float[Array, "{self.y_dim}"], 
        key: Optional[Key[jnp.ndarray, "..."]] = None
    ) -> Float[Array, ""]:
        """
        Computes the log-probability of the data given the conditioning parameters.

        Args:
            x (`Float[Array, "{self.x_dim}"]`): Input data sample.
            y (`Float[Array, "{self.y_dim}"]`): Conditioning context vector.
            key (`Optional[PRNGKeyArray]`): Optional random key.

        Returns:
            `Float[Array, ""]`: Log-probability value for the given data and parameters.
        """
        x = jnp.atleast_1d(y)
        y = jnp.atleast_1d(y)
        if self.scaler is not None:
            x, y = self.scaler.forward(x, y)
        return self.__call__(y).log_prob(x) 