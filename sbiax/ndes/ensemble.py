from typing import Sequence, List, Callable, Optional, Literal
import jax
import jax.numpy as jnp
import jax.random as jr 
import equinox as eqx
from jaxtyping import Key, Array, Float
from tensorflow_probability.substrates.jax.distributions import Distribution

from .vmim import VMIM


def default_weights(weights, ndes):
    return weights if weights is not None else jnp.ones((len(ndes))) / len(ndes)


class Ensemble(eqx.Module):
    """
    An `eqx.Module` representing an ensemble of neural density estimators (NDEs) with methods to 
    compute the ensemble log-probability function and 'stacking weights' for this function.

    This `Ensemble` supports different types of density estimation SBI techniques, such as neural 
    likelihood estimation (NLE) and neural posterior estimation (NPE). It also supports saving and 
    loading of the ensemble's state.

    Attributes:
        sbi_type (`str`): Specifies the type of SBI (`"nle"` or `"npe"`).
        ndes (`List[eqx.Module]`): A listof NDE models that make up the ensemble.
        weights (`Array`): Weights assigned to each NDE in the ensemble, which are used for 
            calculating the ensemble likelihood.

    Methods:
        __init__(ndes: `Sequence[eqx.Module]`, sbi_type: `str` = "nle", weights: `Array` = None):
            Initializes the ensemble with a list of NDEs, SBI type, and optional weights.
        
        nde_log_prob_fn(nde: `eqx.Module`, data: `Array`, prior: `Distribution`) -> `Callable`:
            Returns a log-probability function for a single NDE with respect to the data and prior.

        ensemble_log_prob_fn(data: `Array`, prior: `Optional[Distribution]` = None) -> `Callable`:
            Returns the ensemble log-probability function that combines all NDEs at the given 
            observation, adjusted based on `sbi_type`.

        _ensemble_log_prob_fn(datavectors: `Union[Array, List[Array]]`, prior: `Optional[Distribution]` = None) -> `Callable`:
            Internal method for generating the log-probability function for the ensemble 
            across batched data vectors.

        ensemble_likelihood(data: `Array`) -> `Callable`:
            Returns the ensemble likelihood without the prior term, evaluated at `data`.

        calculate_stacking_weights(losses: `List[float]`) -> `Array`:
            Calculates the weights for each NDE in the ensemble using the final-epoch validation losses.

        save_ensemble(path: `str`) -> None:
            Saves the ensemble model to the specified path.

        load_ensemble(path: `str`) -> `Ensemble`:
            Loads and returns the ensemble model from the specified path.

    Example:
        ```python
        import equinox as eqx
        import jax.random as jr
        from tensorflow_probability.substrates.jax.distributions import Normal
        from sbiax.ndes import CNF

        # Define some NDE models
        ndes = [CNF(...), CNF(...)]
        prior = Normal(0, 1)

        ensemble = Ensemble(ndes=ndes, sbi_type="nle")
        log_prob_fn = ensemble.ensemble_log_prob_fn(data, prior=prior)
        ```
    """
    sbi_type: str
    ndes: List[eqx.Module]
    weights: Array

    def __init__(
        self, 
        ndes: Sequence[eqx.Module], 
        sbi_type: Literal["nle", "npe"] = "nle", 
        weights: Optional[Float[Array, "l"]] = None
    ):
        """
        Initializes the ensemble with a list of neural density estimators (NDEs), 
        an SBI type (`"nle"` or `"npe"`), and optional stacking weights for the
        ensemble log-likelihood (default is uniform).

        Args:
            ndes (`Sequence[eqx.Module]`): A sequence of NDE models in the ensemble.
            sbi_type (`str`): Specifies the type of SBI, either `"nle"` (neural likelihood estimation) 
                or `"npe"` (neural posterior estimation).
            weights (`Array`, optional): Optional weights for each NDE in the ensemble. If not 
                provided, weights are assigned equally.
        """
        self.ndes = ndes
        self.sbi_type = sbi_type
        self.weights = default_weights(weights, ndes)

        assert not (any([isinstance(nde, VMIM) for nde in ndes]) and sbi_type == "nle"), (
            "Note: VMIM NDEs cannot be used with Neural Likelihood Estimation."
        )

    def nde_log_prob_fn(
        self, 
        nde: eqx.Module, 
        data: Float[Array, "x"], 
        prior: Distribution
    ) -> Callable[
        [Float[Array, "y"], Optional[Key[jnp.ndarray, "..."]]], Float[Array, ""]
    ]:
        """ 
        Returns a posterior log-probability function for a single NDE model parameterised by 
        a datavector `data` and prior `prior`.

        Args:
            nde (`eqx.Module`): The NDE model for which the log-probability function is generated.
            data (`Array`): The observed data vector.
            prior (`Distribution`): The prior distribution to apply on the parameters.

        Returns:
            `Callable`: A function that computes the log-probability of the NDE 
                given `data` and `prior`.
        """
        _nle = self.sbi_type == "nle"

        def _nde_log_prob_fn(theta, **kwargs): 
            if _nle:
                x, y = data, theta
                nde_likelihood = nde.log_prob(x=x, y=y, **kwargs) 
                nde_posterior = nde_likelihood + prior.log_prob(theta)
            else:
                x, y = theta, data
                nde_likelihood = nde.log_prob(x=x, y=y, **kwargs) 
                nde_posterior = nde_likelihood 
            return nde_posterior
        return _nde_log_prob_fn

    def ensemble_log_prob_fn(
        self, 
        data: Float[Array, "x"], 
        prior: Optional[Distribution]
    ) -> Callable[
        [Float[Array, "y"], Optional[Key[jnp.ndarray, "..."]]], Float[Array, ""]
    ]:
        """ 
        Returns the ensemble log-probability function that combines all NDEs with the 
        given observation, depending on `self.sbi_type`.

        Args:
            data (`Array`): The observed data vector.
            prior (`Optional[Distribution]`): Optional prior distribution for conditioning 
                the ensemble.

        Returns:
            `Callable`: A function that computes the log-probability for the ensemble, 
                conditioned on `data` and adjusted by the `sbi_type` ("nle" or "npe").
        """

        _nle = self.sbi_type == "nle"

        def _joint_log_prob_fn(
            theta: Float[Array, "y"], 
            key: Optional[Key[jnp.ndarray, "..."]] = None
        ) -> Float[Array, ""]:
            L = jnp.zeros(())
            for n, (nde, weight) in enumerate(zip(self.ndes, self.weights)):
                if key is not None:
                    key = jr.fold_in(key, n)
                nde_log_L = nde.log_prob(
                    x=data if _nle else theta, 
                    y=theta if _nle else data, 
                    key=key
                )
                L = L + weight * jnp.exp(nde_log_L)
            L = jnp.log(L) 
            if prior is not None and _nle:
                L = L + prior.log_prob(theta)
            return L 

        return _joint_log_prob_fn

    def ensemble_likelihood(self, data: Float[Array, "x"]) -> Float[Array, ""]:
        """
        Returns the ensemble likelihood (no prior term), evaluated at `data`.
        Useful for using multiple ensembles, as independent likelihoods, together.

        Args:
            data (`Array`): The observed data vector.

        Returns:
            `Array`: The likelihood of the ensemble at the given `data`.
        """
        return self.ensemble_log_prob_fn(data, prior=None)

    def calculate_stacking_weights(self, losses: Sequence[Float[Array, "..."]]) -> Float[Array, "l"]:
        """
        Calculates the weights for each NDE in the ensemble using the final-epoch 
        validation losses.

        Args:
            losses (`Sequence[Array]`): A list of validation losses for each NDE model 
                in the ensemble.

        Returns:
            `Array`: Calculated weights for each NDE in the ensemble based on their 
                validation losses.
        """
        Ls = jnp.array([-losses[n] for n, _ in enumerate(self.ndes)])
        nde_weights = jnp.exp(Ls) / jnp.sum(jnp.exp(Ls)) 
        return nde_weights

    def get_nde_names(self) -> List[str]:
        """
        Gets names of NDEs in the ensemble


        Returns:
            `List[str]`: List of names of NDEs in the ensemble.
        """
        return [nde.__class__.__name__ for nde in self.ndes]

    def save_ensemble(self, path: str) -> None:
        """
        Saves the ensemble model's state to the specified path.

        Args:
            path (`str`): The file path where the ensemble's state will be saved.
        """
        eqx.tree_serialise_leaves(path, self)

    def load_ensemble(self, path: str) -> eqx.Module:
        """
        Loads and returns the ensemble model's state from the specified path.

        Args:
            path (`str`): The file path from which the ensemble's state will be loaded.

        Returns:
            `Ensemble`: The deserialized ensemble model with the saved state.
        """
        return eqx.tree_deserialise_leaves(path, self)