from typing import Optional, Tuple
import jax
import equinox as eqx
from jaxtyping import Float, Array, jaxtyped
from beartype import beartype as typechecker


def stop_grad(a):
    return jax.lax.stop_gradient(a)


class Scaler(eqx.Module):
    x_dim: int
    q_dim: Optional[int] = None
    mu_x: Array
    std_x: Array
    mu_q: Array
    std_q: Array
    use_scaling: bool

    def __init__(
        self, 
        X=None, 
        Q=None, 
        *,
        x_mu_std=None,
        q_mu_std=None,
        use_scaling=True
    ):
        assert not (X is not None and x_mu_std)
        if X is not None:
            self.x_dim = X.shape[-1]
            self.mu_x = X.mean(axis=0)
            self.std_x = X.std(axis=0)
        if x_mu_std is not None:
            self.mu_x, self.std_x = x_mu_std
            self.x_dim = self.mu_x.size

        assert not (Q is not None and q_mu_std)
        if Q is not None:
            self.q_dim = Q.shape[-1] 
            self.mu_q = Q.mean(axis=0)
            self.std_q = Q.std(axis=0)
        if x_mu_std is not None:
            self.mu_q, self.std_q = q_mu_std
            self.q_dim = self.mu_q.size

        self.use_scaling = use_scaling

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, 
        x: Float[Array, "{self.x_dim}"], 
        q: Optional[Float[Array, "{self.q_dim}"]] = None
    ) -> Tuple[Float[Array, "{self.x_dim}"], Float[Array, "{self.q_dim}"]]: 
        if self.use_scaling:
            x = (x - stop_grad(self.mu_x)) / stop_grad(self.std_x)
            q = (q - stop_grad(self.mu_q)) / stop_grad(self.std_q)
        return x, q