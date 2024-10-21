from typing import Tuple, Optional
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Key, jaxtyped, PRNGKeyArray
from beartype import beartype as typechecker
from flowjax.flows import masked_autoregressive_flow
from flowjax.distributions import Normal


class MAF(eqx.Module):
    flow: eqx.Module
    base_dist: eqx.Module
    scaler: eqx.Module
    x_dim: int
    y_dim: int

    def __init__(
        self, 
        event_dim, 
        context_dim, 
        width_size, 
        n_layers, 
        nn_depth, 
        activation=jax.nn.tanh, 
        scaler=None,
        *, 
        key
    ):
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
        key: Optional[PRNGKeyArray] = None
    ) -> Float[Array, ""]:
        if self.scaler is not None:
            x, y = self.scaler.forward(x, y)
        return self.flow.log_prob(x, y)

    def loss(self, x, y, **kwargs):
        return -self.log_prob(x, y, **kwargs)

    @jaxtyped(typechecker=typechecker)
    def sample_and_log_prob(
        self,
        key: PRNGKeyArray,
        y: Float[Array, "{self.y_dim}"]
    ) -> Tuple[Float[Array, "{self.x_dim}"], Float[Array, ""]]:
        sample = self.flow.sample(key, (), condition=y)
        log_prob = self.flow.log_prob(sample, y)
        return sample, log_prob

    def sample_and_log_prob_n(
        self, 
        key: Key, 
        y: Array, 
        n_samples: int
    ):
        samples = self.flow.sample(key, (n_samples,), condition=y)
        log_probs = self.flow.log_prob(samples, y)
        return samples, log_probs