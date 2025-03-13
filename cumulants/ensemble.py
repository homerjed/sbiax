from typing import Tuple, Literal, Sequence, Optional, Callable, Self
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr 
import equinox as eqx
from jaxtyping import PRNGKeyArray, Array, Float, Scalar, jaxtyped
from beartype import beartype as typechecker
from tensorflow_probability.substrates.jax.distributions import Distribution

typecheck = jaxtyped(typechecker=typechecker)

LogProbFn = Callable[[Float[Array, "p"]], Scalar]


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def default_weights(weights: Float[Array, "n"], ndes: list[eqx.Module]) -> Float[Array, "n"]:
    assert len(ndes) > 0
    return weights if exists(weights) else jnp.ones((len(ndes),)) / len(ndes)


class Ensemble(eqx.Module):
    sbi_type: str
    ndes: Tuple[eqx.Module]
    weights: Float[Array, "{self.n_ndes}"]

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
    def n_ndes(self):
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
                l = nde.log_prob(x=data, y=theta, key=key) + prior.prior.log_prob(theta)
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

        def _maybe_vmap_nde_log_L(
            nde: eqx.Module, 
            data: Float[Array, "d"] | Float[Array, "n d"], 
            theta: Float[Array, "p"], 
            *,
            key: Optional[PRNGKeyArray] = None
        ) -> Scalar:
            # Add log-likelihoods of datavectors together 
            # > Assumptions about datavector shape and batch axis...

            # Log-prob function with fixed parameters
            fn = lambda data, key: nde.log_prob(x=data, y=theta, key=key)

            # If stacked datavectors, split keys and vmap
            if data.ndim > 1:
                if exists(key):
                    keys = jr.split(key, data.shape[0])
                else:
                    keys = None
                L = jnp.sum(jax.vmap(fn)(data, keys)) # Independent => sum
            else:
                L = fn(data, key)

            return L

        if self.sbi_type == "nle":
            @typecheck
            def _joint_log_prob_fn(
                theta: Float[Array, "p"], key: Optional[PRNGKeyArray] = None
            ) -> Scalar:
                L = jnp.zeros(())
                for n, (nde, weight) in enumerate(zip(self.ndes, self.weights)): # NOTE: lax.scan

                    if exists(key):
                        key = jr.fold_in(key, n)

                    nde_log_L = _maybe_vmap_nde_log_L(nde, data, theta, key=key) 

                    # Add likelihoods together for ensemble ndes
                    L_nde = weight * jnp.exp(nde_log_L) # NOTE: weight inside log-prob fun?! weighting doesn't distribute over batches of datavecotrs?

                    L = L + L_nde

                L = jnp.log(L) 

                if exists(prior):
                    L = L + prior.log_prob(theta) # NOTE: just adding prior is the difference between NPE and NLE?

                return L

        if self.sbi_type == "npe":
            @typecheck
            def _joint_log_prob_fn(
                theta: Float[Array, "p"], key: Optional[PRNGKeyArray] = None
            ) -> Scalar:
                L = jnp.zeros(())
                for n, (nde, weight) in enumerate(zip(self.ndes, self.weights)):

                    if exists(key):
                        key = jr.fold_in(key, n)

                    nde_log_L = nde.log_prob(x=theta, y=data, key=key)

                    L_nde = weight * jnp.exp(nde_log_L)

                    L = L + L_nde

                return jnp.log(L) 

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
        losses: list[Float[Array, "_"]]
    ) -> Float[Array, "{self.n_ndes}"]:
        """
            Calculate weightings of NDEs in ensemble
            - losses is a list of final-epoch validation losses
        """
        nde_Ls = jnp.array([-losses[n] for n, _ in enumerate(self.ndes)])
        nde_weights = jnp.exp(nde_Ls) / jnp.sum(jnp.exp(nde_Ls)) #jax.nn.softmax(Ls)
        assert nde_weights.shape == (self.n_ndes,)
        return jnp.atleast_1d(nde_weights)

    def save_ensemble(self, path: str) -> None:
        eqx.tree_serialise_leaves(path, self)

    def load_ensemble(self, path: str) -> eqx.Module:
        return eqx.tree_deserialise_leaves(path, self)


class MultiEnsemble(eqx.Module):
    ensembles: list[Ensemble]
    prior: Optional[Distribution]
    sbi_type: Literal["nle", "npe"] 

    @typecheck
    def __init___(
        self, 
        ensembles: list[Ensemble], 
        prior: Optional[Distribution],
        *,
        sbi_type: Literal["nle", "npe"] = "nle"
    ):
        self.ensembles = ensembles
        self.prior = prior # Allow to be overwritten in inference call
        self.sbi_type = sbi_type

    @typecheck
    def get_multi_ensemble_log_prob_fn(
        self, 
        datavectors: list[Float[Array, "n d"]] | Float[Array, "n d"], 
        prior: Optional[Distribution] = None
    ) -> LogProbFn:
        # Prioritise prior specified in args
        _prior = default(prior, self.prior)

        if isinstance(datavectors, jax.Array):
            datavectors = [datavectors]

        def _multi_ensemble_log_prob_fn(theta: Float[Array, "p"]) -> Scalar:
            # Loop over matched ensembles / datavectors NOTE: vmap over datavectors (when have multiple per redshift)?
            L = jnp.zeros(())

            # Don't need this loop? tree map or something?
            for ensemble, _datavectors in zip(self.ensembles, datavectors):
                ensemble_log_L = ensemble.ensemble_likelihood(_datavectors)(theta) 
                L = L + ensemble_log_L

            if self.sbi_type == "nle":
                L = L + _prior.log_prob(theta) # NOTE: only if NLE!

            return L 

        return _multi_ensemble_log_prob_fn

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