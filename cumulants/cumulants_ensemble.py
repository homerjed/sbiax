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
    """
        Ensemble of NDEs to be fit to simulations at a fixed redshift
    """

    sbi_type: str
    ndes: Tuple[eqx.Module]
    n_ndes: int
    weights: list[float]

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
        self.n_ndes = len(ndes)

    # @property
    # def n_ndes(self):
    #     return len(self.ndes)

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
            """ 
                Add log-likelihoods of datavectors together 
                > Assumptions about datavector shape and batch axis...
            """

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

        @typecheck
        def _joint_log_prob_fn(
            theta: Float[Array, "p"], key: Optional[PRNGKeyArray] = None
        ) -> Scalar:
            """ Joint log-probability function for ensemble of NDEs """

            L = jnp.zeros(())
            for n, (nde, weight) in enumerate(
                zip(self.ndes, jnp.atleast_1d(self.weights)) # NOTE: lax.scan
            ): 

                if exists(key):
                    key = jr.fold_in(key, n)

                nde_log_L = _maybe_vmap_nde_log_L(nde=nde, data=data, theta=theta, key=key) 

                # Add likelihoods together for ensemble ndes
                L_nde = weight * jnp.exp(nde_log_L) # NOTE: weight inside log-prob fun?! weighting doesn't distribute over batches of datavecotrs?

                L = L + L_nde

            L = jnp.log(L) 

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
        nde_Ls = jnp.array([-losses[n] for n, _ in enumerate(self.ndes)])

        nde_weights = jnp.exp(nde_Ls) / jnp.sum(jnp.exp(nde_Ls)) # jax.nn.softmax(Ls)

        assert nde_weights.shape == (self.n_ndes,)

        nde_weights = jnp.atleast_1d(nde_weights.astype(jnp.float32))

        return nde_weights

    def save_ensemble(self, path: str) -> None:
        eqx.tree_serialise_leaves(path, self)

    def load_ensemble(self, path: str) -> eqx.Module:
        return eqx.tree_deserialise_leaves(path, self)


class MultiEnsemble(eqx.Module):
    """
        Ensemble for bringing together ensembles of NDEs fit at
        individual redshifts
    """

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

        assert len(self.ensembles) == len(datavectors), (
            "Ensembles={}, datavectors={}".format(len(self.ensembles), len(datavectors))
        )

        assert all([len(_datavectors) for _datavectors in datavectors]), (
            "Non-equal shapes between datavectors {}".format(
                [len(_datavectors) for _datavectors in datavectors]
            )
        )

        @typecheck
        def _multi_ensemble_log_prob_fn(theta: Float[Array, "p"]) -> Scalar:
            # Loop over matched ensembles / datavectors NOTE: vmap over datavectors (when have multiple per redshift)?
            L = jnp.zeros(())

            # Don't need this loop? tree map or something? jax.tree.map(lambda f, x: jax.vmap(f)(x), f, x, is_leaf=lambda l: isinstance(l, ...)) x is stack of datavectors, f is ensemble nde
            for ensemble, _datavectors in zip(self.ensembles, datavectors):

                ensemble_log_L = ensemble.ensemble_likelihood(_datavectors)(theta) # No use of prior

                L = L + ensemble_log_L

            if self.sbi_type == "nle":
                L = L + _prior.log_prob(theta) # NOTE: only if NLE!

            return L 

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