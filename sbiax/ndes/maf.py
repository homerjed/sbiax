from typing import Tuple, Optional, Callable
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Key, Array, Float, jaxtyped
from beartype import beartype as typechecker
from flowjax.flows import masked_autoregressive_flow
from flowjax.distributions import Normal


class MAF(eqx.Module):
    """
    Implements a Masked Autoregressive Flow (MAF, Papamakarios++ 2017) model for density 
    estimation and sampling, with support for conditional inputs and an optional data scaler.

    Attributes:
        flow (`eqx.Module`): The MAF flow model implementing the density transformation.
        base_dist (`eqx.Module`): The base normal distribution used as the input for the flow.
        scaler (`Optional[eqx.Module]`): A scaler for input and context data, if provided.
        x_dim (`int`): Dimensionality of the data (`x`).
        y_dim (`int`): Dimensionality of the parameters (e.g. conditioning) (`y`).

    Methods:
        __init__(
            event_dim: `int`, 
            context_dim: `int`, 
            width_size: `int`, 
            n_layers: `int`, 
            nn_depth: `int`, 
            activation: `Callable` = jax.nn.tanh, 
            scaler: `Optional[eqx.Module]` = None, 
            key: `PRNGKeyArray`
        ):
            Initializes the MAF model with a specified flow structure and configuration parameters.
        
        log_prob(
            x: `Float[Array, "{self.x_dim}"]`, 
            y: `Float[Array, "{self.y_dim}"]`, 
            key: `Optional[PRNGKeyArray]` = None
        ) -> `Float[Array, ""]`:
            Computes the log-probability of `x` conditioned on `y` using the flow model.

        loss(x: `Array`, y: `Array`, **kwargs) -> `Float[Array, ""]`:
            Computes the negative log-probability for given `x` and `y` as a loss function.

        sample_and_log_prob(
            key: `PRNGKeyArray`, 
            y: `Float[Array, "{self.y_dim}"]`
        ) -> `Tuple[Float[Array, "{self.x_dim}"], Float[Array, ""]]`:
            Samples from the flow model conditioned on `y`, returning both the sample and its 
            log-probability.

        sample_and_log_prob_n(
            key: `PRNGKeyArray`, 
            y: `Float[Array, "{self.y_dim}"]`, 
            n_samples: `int`
        ) -> `Tuple[Array, Array]`:
            Generates `n_samples` samples conditioned on `y`, returning the samples and their 
            corresponding log-probabilities.

    Example:
        ```python
        import jax
        import jax.random as jr
        import equinox as eqx
        from sbiax.ndes import MAF

        key = jr.key(0)

        event_dim = 2
        context_dim = 3
        maf = MAF(
            event_dim=event_dim, 
            context_dim=context_dim, 
            width_size=64, 
            n_layers=5, 
            nn_depth=2, 
            key=key
        )
        
        x = jr.normal(key, (event_dim,))
        y = jr.normal(key, (context_dim,))
        
        log_prob = maf.log_prob(x, y)
        sample, sample_log_prob = maf.sample_and_log_prob(key, y)
        ```
    """
    flow: eqx.Module
    base_dist: eqx.Module
    scaler: eqx.Module
    x_dim: int
    y_dim: int

    def __init__(
        self, 
        event_dim: int, 
        context_dim: int, 
        width_size: int, 
        n_layers: int, 
        nn_depth: int, 
        activation: Callable = jax.nn.tanh, 
        scaler: eqx.Module = None,
        *, 
        key: Key
    ):
        """
        Initializes a Masked Autoregressive Flow (MAF) model for conditional density estimation.

        Args:
            event_dim (int): Dimensionality of the data space (target variable).
            context_dim (int): Dimensionality of the conditioning space.
            width_size (int): Width (number of neurons) of each hidden layer in the neural network.
            n_layers (int): Number of flow layers in the MAF.
            nn_depth (int): Depth (number of layers) of each neural network in the flow.
            activation (callable, optional): Activation function for the neural network layers. 
                Defaults to `jax.nn.tanh`.
            scaler (optional): Optional scaler for data preprocessing. Defaults to `None`.
            key: Random key for initializing the MAF layers.

        Attributes:
            base_dist (Distribution): The base distribution, set to a standard Gaussian for MAF sampling.
            flow (Distribution): The main masked autoregressive flow model for conditional sampling and density estimation.
            x_dim (int): Dimensionality of the target variable.
            y_dim (int): Dimensionality of the conditioning variable.
        """
        self.base_dist = Normal(
            loc=jnp.zeros(event_dim), scale=jnp.ones(event_dim)
        )
        self.flow = masked_autoregressive_flow(
            key,
            base_dist=self.base_dist,
            cond_dim=context_dim,
            flow_layers=n_layers,
            nn_width=width_size,
            nn_depth=nn_depth,
            nn_activation=activation
        )
        self.scaler = scaler
        self.x_dim = event_dim
        self.y_dim = context_dim

    @jaxtyped(typechecker=typechecker)
    def log_prob(
        self, 
        x: Float[Array, "{self.x_dim}"], 
        y: Float[Array, "{self.y_dim}"],
        key: Optional[Key[jnp.ndarray, ""]] = None
    ) -> Float[Array, ""]:
        """
        Computes the log-probability of the data `x` given the conditioning `y`.

        Args:
            x (Float[Array, "{self.x_dim}"]): Input data for which the log-probability is calculated.
            y (Float[Array, "{self.y_dim}"]): Conditioning context used for the transformation.
            key (Optional[PRNGKeyArray], optional): Optional random key for sampling.

        Returns:
            Float[Array, ""]: The computed log-probability of the input data `x` given `y`.
        """
        if self.scaler is not None:
            x, y = self.scaler.forward(x, y)
        return self.flow.log_prob(x, y)

    def loss(self, x, y, **kwargs):
        """
        Computes the loss for training the Masked Autoregressive Flow model, which is defined as 
        the negative log-probability of the data `x` given the conditioning `y`.

        Args:
            x (Array): Input data for which the loss is calculated.
            y (Array): Conditioning used for the transformation.
            **kwargs: Additional arguments for the log-prob calculation.

        Returns:
            Array: Computed loss for the input data `x` given the context `y`.
        """
        return -self.log_prob(x, y, **kwargs)

    @jaxtyped(typechecker=typechecker)
    def sample_and_log_prob(
        self,
        key: Key[jnp.ndarray, ""],
        y: Float[Array, "{self.y_dim}"]
    ) -> Tuple[Float[Array, "{self.x_dim}"], Float[Array, ""]]:
        """
        Samples from the MAF model and computes the log-probability of the generated sample 
        given the conditioning `y`.

        Args:
            key (PRNGKeyArray): Random key used for sampling.
            y (Float[Array, "{self.y_dim}"]): Conditioning used for generating the sample.

        Returns:
            Tuple[Float[Array, "{self.x_dim}"], Float[Array, ""]]: A tuple containing:
                - The sampled data `x` from the MAF model.
                - The log-probability of the sampled data `x` given `y`.
        """
        sample = self.flow.sample(key, (), condition=y)
        log_prob = self.flow.log_prob(sample, y)
        return sample, log_prob

    @jaxtyped(typechecker=typechecker)
    def sample_and_log_prob_n(
        self, 
        key: Key[jnp.ndarray, ""], 
        y: Float[Array, "#n {self.y_dim}"], 
        n_samples: int
    ) -> Tuple[Float[Array, "#n {self.x_dim}"], Float[Array, "#n"]]:
        """
        Generates `n_samples` samples from the MAF model and computes the log-probabilities 
        for each sample given the conditioning context `y`.

        Args:
            key (Key): Random key used for generating samples.
            y (Array): Conditioning context used for generating the samples.
            n_samples (int): Number of samples to generate.

        Returns:
            Tuple[Array, Array]: A tuple containing:
                - An array of generated samples of shape `(n_samples, self.x_dim)`.
                - An array of log-probabilities for each sample of shape `(n_samples,)`.
        """
        samples = self.flow.sample(key, (n_samples,), condition=y)
        log_probs = self.flow.log_prob(samples, y)
        return samples, log_probs