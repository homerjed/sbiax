from typing import Tuple, Literal, Sequence, Optional, Callable, Self
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


@typecheck
def reduce_weighted_logsumexp(
    x: Float[Array, "n"], weights: Float[Array, "n"], axis: Optional[int] = None
) -> Scalar:
    """ Stable computation of log(sum(weights * exp(x))) """

    assert jnp.all(weights > 0.), "All weights must be positive: weights={}".format(weights)

    # Shift for numerical stability
    x_max = jnp.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max

    weighted_exp = weights * jnp.exp(x_shifted)
    sum_weighted_exp = jnp.sum(weighted_exp, axis=axis)

    return jnp.log(sum_weighted_exp) + jnp.squeeze(x_max, axis=axis)


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
    n_ndes: int
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
        self.n_ndes = len(ndes)

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

            if key is not None:
                keys = jr.split(key, self.n_ndes) 
            else: 
                keys = [None] * self.n_ndes

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

            nde_log_Ls = jax.tree.map(
                lambda key, nde: _maybe_vmap_nde_log_L(
                    nde=nde, data=data, theta=theta, key=key
                ),
                keys,
                self.ndes,
                is_leaf=lambda x: x is None # Allow keys=[None, ...]
            )
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
        nde_Ls = jnp.array([-losses[n] for n, _ in enumerate(self.ndes)])

        nde_Ls = jnp.exp(nde_Ls - jnp.max(nde_Ls))

        nde_weights = nde_Ls / jnp.sum(nde_Ls) # jax.nn.softmax(Ls)

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
    sbi_type: Literal["nle", "npe"] = "nle"

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
        datavectors: list[Float[Array, "n d"]],# | Float[Array, "n d"], 
        prior: Optional[Distribution] = None
    ) -> LogProbFn:
        
        # Prioritise prior specified in args of this function
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

            L = jax.tree.map(
                lambda d, e: e.ensemble_likelihood(d)(theta),
                datavectors,
                self.ensembles
            )
            L = jnp.sum(jnp.asarray(L))

            if self.sbi_type == "nle":
                L = L + _prior.log_prob(theta) 

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