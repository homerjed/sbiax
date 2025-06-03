from typing import Callable
import equinox as eqx
from jaxtyping import Float, Array, Scalar, jaxtyped
from beartype import beartype as typechecker

typecheck = jaxtyped(typechecker=typechecker)


class VMIMDensityEstimator(eqx.Module):
    density_estimator: eqx.Module
    summary_network: eqx.Module | Callable[[Float[Array, "d"]], Float[Array, "p"]]

    @typecheck
    def __init__(
        self, 
        density_estimator: eqx.Module, 
        summary_network: eqx.Module | Callable[[Float[Array, "d"]], Float[Array, "p"]]
    ):
        self.density_estimator = density_estimator
        self.summary_network = summary_network

    @typecheck
    def __call__(self, x: Float[Array, "d"], y: Float[Array, "p"], **kwargs) -> Scalar:
        y_ = self.summary_network(y)
        return self.density_estimator.log_prob(x, y_)

    @typecheck
    def log_prob(self, x: Float[Array, "d"], y: Float[Array, "p"], **kwargs) -> Scalar:
        return self.__call__(x, y)

    @typecheck
    def loss(self, x: Float[Array, "d"], y: Float[Array, "p"], **kwargs) -> Scalar:
        return -self.log_prob(x, y)


VMIM = VMIMDensityEstimator # Alternative name