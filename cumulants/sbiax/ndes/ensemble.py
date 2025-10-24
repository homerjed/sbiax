import os
from typing import Tuple, Literal, Sequence, Optional, Callable, Self, Any
import operator
import jax
import jax.numpy as jnp
import jax.random as jr 
import equinox as eqx
from jaxtyping import PRNGKeyArray, Array, Float, Scalar, jaxtyped
from beartype import beartype as typechecker

from .cnf import CNF

TYPECHECK = True if os.environ.get("TYPECHECK", "").lower() in ("1", "true") else False
if TYPECHECK:
    typecheck = jaxtyped(typechecker=typechecker)
else:
    typecheck = lambda x: x

Distribution = Any

LogProbFn = Callable[[Float[Array, "p"]], Scalar]

is_eqx_module = lambda l: isinstance(l, eqx.Module)


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def default_weights(
    weights: Float[Array, "n"], 
    ndes: list[eqx.Module]
) -> Float[Array, "n"]:

    assert len(ndes) > 0

    return weights if exists(weights) else jnp.ones((len(ndes),)) / len(ndes)


class Ensemble(eqx.Module):
    """
        Ensemble of NDEs to be fit to simulations at a fixed redshift
    """

    sbi_type: Literal["nle", "npe"]
    ndes: Sequence[eqx.Module]
    weights: Float[Array, "n"]

    @typecheck
    def __init__(
        self, 
        ndes: Sequence[eqx.Module], 
        sbi_type: Literal["nle", "npe"] = "nle", 
        weights: Optional[Float[Array, "{self.n_ndes}"]] = None
    ):
        self.ndes = ndes
        self.sbi_type = sbi_type
        self.weights = default_weights(weights, ndes)

    @property
    @typecheck
    def n_ndes(self) -> int:
        return len(self.ndes)

    @typecheck
    def nde_log_prob_fn(
        self, 
        nde: eqx.Module, 
        prior: Distribution, 
        data: Float[Array, "d"]
    ) -> LogProbFn:
        """ 
            Get log-probability function for NDE at given observation.
        """

        def _nde_log_prob_fn(
            theta: Float[Array, "p"], key: Optional[PRNGKeyArray] = None
        ) -> Scalar: 

            if self.sbi_type == "nle":
                l = nde.log_prob(x=data, y=theta, key=key) + prior.log_prob(theta)
            else:
                l = nde.log_prob(x=theta, y=data, key=key)

            return l

        return _nde_log_prob_fn

    @typecheck
    def ensemble_log_prob_fn(
        self, 
        data: Float[Array, "d"] | Float[Array, "n d"], 
        prior: Optional[Distribution] = None
    ) -> LogProbFn:
        """ 
            Get log-probability function for NDE at given observation 
            for whole ensemble of NDEs.
            - some NDEs may have a probabilistic estimate of the likelihood
              so a key is provided, the ndes are set to inference mode to 
              imply this key is not used for dropout etc.
        """
        assert self.n_ndes == len(self.weights), (
            "ndes={}, weights={}".format(self.n_ndes), len(self.weights)
        )

        @typecheck
        def _maybe_vmap_nde_log_L(
            nde: eqx.Module, 
            data: Float[Array, "d"] | Float[Array, "n d"], 
            theta: Float[Array, "p"], 
            *,
            key: Optional[PRNGKeyArray] = None
        ) -> Scalar:
            """ 
                Add log-likelihoods of datavectors together 
                > Assumptions about datavector shape and batch axis...
            """

            # Force exact log-prob for CNFs during inference
            if isinstance(nde, CNF):
                nde = eqx.tree_at(lambda nde: nde.exact_log_prob, nde, True)
                nde_kwargs = dict(exact_log_prob=True)
            else:
                nde_kwargs = dict()

            # Log-prob function with fixed parameters, NLE, scaling done in here
            log_prob_fn = lambda data, key: nde.log_prob(x=data, y=theta, key=key, **nde_kwargs)

            # If stacked datavectors, split keys and vmap, else just calculate
            if data.ndim > 1:

                if exists(key):
                    keys = jr.split(key, data.shape[0])
                    in_axes = (0, 0)
                else:
                    keys = None
                    in_axes = (0, None)

                L = jnp.sum(jax.vmap(log_prob_fn, in_axes=in_axes)(data, keys)) # Independent => sum
            else:
                L = log_prob_fn(data, key)

            return L

        @typecheck
        def _joint_log_prob_fn(
            theta: Float[Array, "p"], 
            key: Optional[PRNGKeyArray] = None
        ) -> Scalar:
            """ 
                Joint log-probability function for ensemble of NDEs 
            """

            if key is not None:
                keys = list(jr.split(key, self.n_ndes)) # List for tree map
            else: 
                keys = [None] * self.n_ndes

            assert (
                jax.tree.structure(keys, is_leaf=lambda x: x is None) 
                == jax.tree.structure(self.ndes, is_leaf=is_eqx_module)
            ), (
                "Structure mismatch: keys / self.ndes: {}, {}".format(
                    jax.tree.structure(keys, is_leaf=lambda x: x is None), 
                    jax.tree.structure(self.ndes, is_leaf=is_eqx_module)
                )
            )

            # Possibly vmap the NDE over the data, given a parameter set,
            # with a key for each NDE
            nde_log_Ls = jax.tree.map(
                lambda key, nde: _maybe_vmap_nde_log_L(
                    nde=nde, data=data, theta=theta, key=key
                ),
                keys,
                self.ndes,
                is_leaf=lambda x: x is None # Allow keys=[None, None, ...]
            )

            # Weighted sum of log-likelihoods
            L = jax.scipy.special.logsumexp(
                jnp.asarray(nde_log_Ls), b=jnp.atleast_1d(self.weights)
            )

            if exists(prior) and self.sbi_type == "nle":
                L = L + prior.log_prob(theta) # NOTE: just adding prior is the difference between NPE and NLE?

            return L

        return _joint_log_prob_fn

    @typecheck
    def ensemble_likelihood(
        self, 
        data: Float[Array, "d"] | Float[Array, "n d"]
    ) -> LogProbFn:
        return self.ensemble_log_prob_fn(data, prior=None)

    @typecheck
    def calculate_stacking_weights(
        self, 
        losses: list[Scalar]
    ) -> Float[Array, "{self.n_ndes}"]:
        """
            Calculate weightings of NDEs in ensemble
            - losses is a list of final-epoch validation losses
            - never used in gradient calculations
        """

        assert len(losses) == self.n_ndes, (
            "Mismatch: len(losses)={} / self.n_ndes={}".format(losses, self.n_ndes)
        )

        nde_Ls = jnp.array([-losses[n] for n, _ in enumerate(self.ndes)])

        nde_Ls = jnp.exp(nde_Ls - jnp.max(nde_Ls)) # Numerical stability 

        nde_weights = nde_Ls / jnp.sum(nde_Ls) # jax.nn.softmax(Ls)

        assert nde_weights.shape == (self.n_ndes,)

        nde_weights = nde_weights.astype(jnp.float32)

        return nde_weights

    def save_ensemble(self, path: str) -> None:
        eqx.tree_serialise_leaves(path, self)

    def load_ensemble(self, path: str) -> eqx.Module:
        return eqx.tree_deserialise_leaves(path, self)


# fn = lambda nde, key: _maybe_vmap_nde_log_L(
#     nde=nde, data=data, theta=theta, key=key
# )
# nde_log_Ls = jax.tree.map(
#     lambda weight, key, nde: weight * jnp.exp(fn(nde, key)),
#     list(jnp.atleast_1d(self.weights)),
#     keys,
#     self.ndes
# )
# L = jnp.log(sum(nde_log_Ls)) 


class MultiEnsemble(eqx.Module):
    """
        Ensemble for bringing together ensembles of NDEs fit at
        individual redshifts
    """

    ensembles: list[Ensemble]
    prior: Optional[Distribution]
    sbi_type: Literal["nle", "npe"] = "nle"

    @typecheck
    def __init__(
        self, 
        ensembles: list[Ensemble], 
        prior: Optional[Distribution],
        *,
        sbi_type: Literal["nle", "npe"] = "nle"
    ):
        self.ensembles = ensembles
        self.prior = prior # Allow to be overwritten in inference call
        self.sbi_type = sbi_type

        assert all([ensemble.sbi_type == "nle" for ensemble in self.ensembles]), (
            "Mismatch of ensemble SBI types: {}".format(
                [ensemble.sbi_type for ensemble in self.ensembles]
            )
        )

    @typecheck
    def get_multi_ensemble_log_prob_fn(
        self, 
        datavectors: list[Float[Array, "n d"]],
        prior: Optional[Distribution] = None
    ) -> LogProbFn:
        
        # Prioritise prior specified in args of this function
        _prior = default(prior, self.prior)

        if isinstance(datavectors, jax.Array):
            datavectors = [datavectors]

        assert len(self.ensembles) == len(datavectors), (
            "Ensembles={}, datavectors={}".format(len(self.ensembles), len(datavectors))
        )

        assert all([len(_datavectors) == len(datavectors[0]) for _datavectors in datavectors]), (
            "Non-equal shapes between datavectors in list[datavectors] {}".format(
                [len(_datavectors) for _datavectors in datavectors]
            )
        )

        # This will fail
        assert (
            jax.tree.structure(datavectors) 
            == jax.tree.structure(self.ensembles, is_leaf=is_eqx_module)
        ), (
            "Mismatch in datavectors / self.ensembles structures: {}, {}".format(
                jax.tree.structure(datavectors),
                jax.tree.structure(self.ensembles, is_leaf=is_eqx_module)
            )
        )

        @typecheck
        def _multi_ensemble_log_prob_fn(theta: Float[Array, "p"]) -> Scalar:

            L = jax.tree.map(
                lambda d, e: e.ensemble_likelihood(d)(theta),
                datavectors,
                self.ensembles
            )
            
            L = jax.tree.reduce(operator.add, L) # L = jnp.sum(jnp.asarray(L))

            # Force prior here since `ensemble_likelihood` doesn't use it by definition
            if self.sbi_type == "nle":
                L = L + _prior.log_prob(theta) 

            return jnp.squeeze(L)

        return eqx.filter_jit(_multi_ensemble_log_prob_fn)

    def load_ensembles(self, paths: list[str], ensembles: list[Ensemble]) -> None:
        # Load sub-ensembles
        self.ensembles = [
            eqx.tree_deserialise_leaves(path, ensemble)
            for path, ensemble in zip(paths, ensembles)
        ]

    def save_ensemble(self, path: str) -> None:
        eqx.tree_serialise_leaves(path, self)

    def load_ensemble(self, path: str) -> Self:
        return eqx.tree_deserialise_leaves(path, self)


# def default_weights(weights, ndes):
#     return weights if weights is not None else jnp.ones((len(ndes))) / len(ndes)


# class Ensemble(eqx.Module):
#     """
#     An `eqx.Module` representing an ensemble of neural density estimators (NDEs) with methods to 
#     compute the ensemble log-probability function and 'stacking weights' for this function.

#     This `Ensemble` supports different types of density estimation SBI techniques, such as neural 
#     likelihood estimation (NLE) and neural posterior estimation (NPE). It also supports saving and 
#     loading of the ensemble's state.

#     Attributes:
#         sbi_type (`str`): Specifies the type of SBI (`"nle"` or `"npe"`).
#         ndes (`List[eqx.Module]`): A listof NDE models that make up the ensemble.
#         weights (`Array`): Weights assigned to each NDE in the ensemble, which are used for 
#             calculating the ensemble likelihood.

#     Methods:
#         __init__(ndes: `Sequence[eqx.Module]`, sbi_type: `str` = "nle", weights: `Array` = None):
#             Initializes the ensemble with a list of NDEs, SBI type, and optional weights.
        
#         nde_log_prob_fn(nde: `eqx.Module`, data: `Array`, prior: `Distribution`) -> `Callable`:
#             Returns a log-probability function for a single NDE with respect to the data and prior.

#         ensemble_log_prob_fn(data: `Array`, prior: `Optional[Distribution]` = None) -> `Callable`:
#             Returns the ensemble log-probability function that combines all NDEs at the given 
#             observation, adjusted based on `sbi_type`.

#         _ensemble_log_prob_fn(datavectors: `Union[Array, List[Array]]`, prior: `Optional[Distribution]` = None) -> `Callable`:
#             Internal method for generating the log-probability function for the ensemble 
#             across batched data vectors.

#         ensemble_likelihood(data: `Array`) -> `Callable`:
#             Returns the ensemble likelihood without the prior term, evaluated at `data`.

#         calculate_stacking_weights(losses: `List[float]`) -> `Array`:
#             Calculates the weights for each NDE in the ensemble using the final-epoch validation losses.

#         save_ensemble(path: `str`) -> None:
#             Saves the ensemble model to the specified path.

#         load_ensemble(path: `str`) -> `Ensemble`:
#             Loads and returns the ensemble model from the specified path.

#     Example:
#         ```python
#         import equinox as eqx
#         import jax.random as jr
#         from tensorflow_probability.substrates.jax.distributions import Normal
#         from sbiax.ndes import CNF

#         # Define some NDE models
#         ndes = [CNF(...), CNF(...)]
#         prior = Normal(0, 1)

#         ensemble = Ensemble(ndes=ndes, sbi_type="nle")
#         log_prob_fn = ensemble.ensemble_log_prob_fn(data, prior=prior)
#         ```
#     """
#     sbi_type: str
#     ndes: List[eqx.Module]
#     weights: Array

#     def __init__(
#         self, 
#         ndes: Sequence[eqx.Module], 
#         sbi_type: Literal["nle", "npe"] = "nle", 
#         weights: Optional[Float[Array, "l"]] = None
#     ):
#         """
#         Initializes the ensemble with a list of neural density estimators (NDEs), 
#         an SBI type (`"nle"` or `"npe"`), and optional stacking weights for the
#         ensemble log-likelihood (default is uniform).

#         Args:
#             ndes (`Sequence[eqx.Module]`): A sequence of NDE models in the ensemble.
#             sbi_type (`str`): Specifies the type of SBI, either `"nle"` (neural likelihood estimation) 
#                 or `"npe"` (neural posterior estimation).
#             weights (`Array`, optional): Optional weights for each NDE in the ensemble. If not 
#                 provided, weights are assigned equally.
#         """
#         self.ndes = ndes
#         self.sbi_type = sbi_type
#         self.weights = default_weights(weights, ndes)

#     def nde_log_prob_fn(
#         self, 
#         nde: eqx.Module, 
#         data: Float[Array, "x"], 
#         prior: Distribution
#     ) -> Callable[
#         [Float[Array, "y"], Optional[Key[jnp.ndarray, "..."]]], Float[Array, ""]
#     ]:
#         """ 
#         Returns a posterior log-probability function for a single NDE model parameterised by 
#         a datavector `data` and prior `prior`.

#         Args:
#             nde (`eqx.Module`): The NDE model for which the log-probability function is generated.
#             data (`Array`): The observed data vector.
#             prior (`Distribution`): The prior distribution to apply on the parameters.

#         Returns:
#             `Callable`: A function that computes the log-probability of the NDE 
#                 given `data` and `prior`.
#         """
#         _nle = self.sbi_type == "nle"

#         def _nde_log_prob_fn(theta, **kwargs): 
#             nde_likelihood = nde.log_prob(x=data, y=theta, **kwargs) 
#             if _nle:
#                 nde_posterior = nde_likelihood + prior.log_prob(theta)
#             else:
#                 nde_posterior = nde_likelihood 
#             return nde_posterior
#         return _nde_log_prob_fn

#     def ensemble_log_prob_fn(
#         self, 
#         data: Float[Array, "x"], 
#         prior: Optional[Distribution]
#     ) -> Callable[
#         [Float[Array, "y"], Optional[Key[jnp.ndarray, "..."]]], Float[Array, ""]
#     ]:
#         """ 
#         Returns the ensemble log-probability function that combines all NDEs with the 
#         given observation, depending on `self.sbi_type`.

#         Args:
#             data (`Array`): The observed data vector.
#             prior (`Optional[Distribution]`): Optional prior distribution for conditioning 
#                 the ensemble.

#         Returns:
#             `Callable`: A function that computes the log-probability for the ensemble, 
#                 conditioned on `data` and adjusted by the `sbi_type` ("nle" or "npe").
#         """

#         _nle = self.sbi_type == "nle"

#         def _joint_log_prob_fn(
#             theta: Float[Array, "y"], 
#             key: Optional[Key[jnp.ndarray, "..."]] = None
#         ) -> Float[Array, ""]:
#             L = jnp.zeros(())
#             for n, (nde, weight) in enumerate(zip(self.ndes, self.weights)):
#                 if key is not None:
#                     key = jr.fold_in(key, n)
#                 nde_log_L = nde.log_prob(
#                     x=data if _nle else theta, 
#                     y=theta if _nle else data, 
#                     key=key
#                 )
#                 L = L + weight * jnp.exp(nde_log_L)
#             L = jnp.log(L) 
#             if prior is not None and _nle:
#                 L = L + prior.log_prob(theta)
#             return L 

#         return _joint_log_prob_fn

#     def ensemble_likelihood(self, data: Float[Array, "x"]) -> Float[Array, ""]:
#         """
#         Returns the ensemble likelihood (no prior term), evaluated at `data`.
#         Useful for using multiple ensembles, as independent likelihoods, together.

#         Args:
#             data (`Array`): The observed data vector.

#         Returns:
#             `Array`: The likelihood of the ensemble at the given `data`.
#         """
#         return self.ensemble_log_prob_fn(data, prior=None)

#     def calculate_stacking_weights(self, losses: Sequence[Float[Array, "..."]]) -> Float[Array, "l"]:
#         """
#         Calculates the weights for each NDE in the ensemble using the final-epoch 
#         validation losses.

#         Args:
#             losses (`Sequence[Array]`): A list of validation losses for each NDE model 
#                 in the ensemble.

#         Returns:
#             `Array`: Calculated weights for each NDE in the ensemble based on their 
#                 validation losses.
#         """
#         Ls = jnp.array([-losses[n] for n, _ in enumerate(self.ndes)])
#         Ls = Ls - jnp.max(Ls)
#         nde_weights = jnp.exp(Ls) / jnp.sum(jnp.exp(Ls)) 
#         return nde_weights

#     def get_nde_names(self) -> List[str]:
#         """
#         Gets names of NDEs in the ensemble


#         Returns:
#             `List[str]`: List of names of NDEs in the ensemble.
#         """
#         return [nde.__class__.__name__ for nde in self.ndes]

#     def save_ensemble(self, path: str) -> None:
#         """
#         Saves the ensemble model's state to the specified path.

#         Args:
#             path (`str`): The file path where the ensemble's state will be saved.
#         """
#         eqx.tree_serialise_leaves(path, self)

#     def load_ensemble(self, path: str) -> eqx.Module:
#         """
#         Loads and returns the ensemble model's state from the specified path.

#         Args:
#             path (`str`): The file path from which the ensemble's state will be loaded.

#         Returns:
#             `Ensemble`: The deserialized ensemble model with the saved state.
#         """
#         return eqx.tree_deserialise_leaves(path, self)