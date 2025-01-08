from typing import Optional, Tuple
import jax
import equinox as eqx
from jaxtyping import Float, Array, jaxtyped
from beartype import beartype as typechecker


def stop_grad(a):
    return jax.lax.stop_gradient(a)


class Scaler(eqx.Module):
    """
    A scaling module for inputs to NDE models. Easier to keep track of scaling/rescaling
    this way.

    Attributes:
        x_dim (`int`): Size of flattened data input array (excluding any batch axes).
        q_dim (`int`): Size of flattened parameter input array (excluding any batch axes).
        mu_x (`Array`): Mean of input data.
        std_x (`Array`): Standard deviation of input data.
        mu_q (`Array`): Mean of input parameters.
        std_q (`Array`): Standard deviation of input parameters.
        use_scaling (`bool`): Whether to scale inputs / ouputs or not.
    """
    x_dim: int
    q_dim: Optional[int] = None
    mu_x: Array
    std_x: Array
    mu_q: Array
    std_q: Array
    use_scaling: bool

    def __init__(
        self, 
        X: Float[Array, "n x"] = None, 
        Q: Float[Array, "n q"] = None, 
        *,
        x_mu_std: Tuple[Float[Array, "x"], Float[Array, "x"]] = None,
        q_mu_std: Tuple[Float[Array, "q"], Float[Array, "q"]] = None,
        use_scaling=True
    ):
        """
        Initializes the Scaler module.

        This method computes the mean and standard deviation of the data and parameter inputs, 
        or uses pre-computed values if provided. It also determines whether to apply scaling.

        Args:
            X (`Array`, optional): Data input array of shape `(n, x)` used to compute 
                mean and standard deviation. Mutually exclusive with `x_mu_std`.
            Q (`Array`, optional): Parameter input array of shape `(n, q)` used to compute 
                mean and standard deviation. Mutually exclusive with `q_mu_std`.
            x_mu_std (`Tuple[Array, Array]`, optional): Pre-computed mean and standard deviation 
                for the data input. Mutually exclusive with `X`.
            q_mu_std (`Tuple[Array, Array]`, optional): Pre-computed mean and standard deviation 
                for the parameter input. Mutually exclusive with `Q`.
            use_scaling (`bool`, optional): Whether to scale inputs and outputs. Defaults to `True`.

        Raises:
            AssertionError: If both `X` and `x_mu_std` are provided.
            AssertionError: If both `Q` and `q_mu_std` are provided.
        """
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
        """ 
        Implements forward scaling of input data and parameters.

        Args:
            x (`Array`): Data to scale.
            q (`Array`): Parameters to scale.

        Returns:
            (`Tuple[Array, Array]`): Scaled data and parameters.
        """
        if self.use_scaling:
            x = (x - stop_grad(self.mu_x)) / stop_grad(self.std_x)
            q = (q - stop_grad(self.mu_q)) / stop_grad(self.std_q)
        return x, q

    @jaxtyped(typechecker=typechecker)
    def reverse(
        self, 
        x: Float[Array, "{self.x_dim}"], 
        q: Optional[Float[Array, "{self.q_dim}"]] = None
    ) -> Tuple[Float[Array, "{self.x_dim}"], Float[Array, "{self.q_dim}"]]: 
        """ 
        Implements reverse scaling of input data and parameters.

        Args:
            x (`Array`): Data to rescale.
            q (`Array`): Parameters to rescale.

        Returns:
            (`Tuple[Array, Array]`): Rescaled data and parameters.
        """
        if self.use_scaling:
            x = x * stop_grad(self.std_x) + stop_grad(self.mu_x)
            q = q * stop_grad(self.std_q) + stop_grad(self.mu_q)
        return x, q