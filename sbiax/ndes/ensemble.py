from typing import Tuple, Sequence
import jax
import jax.numpy as jnp
import jax.random as jr 
import equinox as eqx
from jaxtyping import Key, Array


def default_weights(weights, ndes):
    return weights if weights is not None else jnp.ones((len(ndes))) / len(ndes)


class Ensemble(eqx.Module):
    sbi_type: str
    ndes: Tuple[eqx.Module]
    weights: Array

    def __init__(
        self, ndes: Sequence[eqx.Module], sbi_type: str = "nle", weights: Array = None
    ):
        self.ndes = ndes
        self.sbi_type = sbi_type
        self.weights = default_weights(weights, ndes)

    def nde_log_prob_fn(self, nde, prior, data):
        """ 
            Get log-probability function for NDE at given observation.
        """
        def _nde_log_prob_fn(theta, **kwargs): 
            return (
                nde.log_prob(x=data, y=theta, **kwargs) 
                + prior.prior.log_prob(theta)
            )
        return _nde_log_prob_fn

    def ensemble_log_prob_fn(self, data, prior=None):
        """ 
            Get log-probability function for NDE at given observation 
            for whole ensemble of NDEs.
            - some NDEs may have a probabilistic estimate of the likelihood
              so a key is provided, the ndes are set to inference mode to 
              imply this key is not used for dropout etc.
        """
        if self.sbi_type == "nle":
            def _joint_log_prob_fn(theta, key=None):
                L = 0.
                for n, (nde, weight) in enumerate(zip(self.ndes, self.weights)):
                    if key is not None:
                        key = jr.fold_in(key, n)
                    nde_log_L = nde.log_prob(x=data, y=theta, key=key)
                    L_nde = weight * jnp.exp(nde_log_L)
                    L = L + L_nde
                L = jnp.log(L) 
                if prior is not None:
                    L = L + prior.log_prob(theta)
                return L

        if self.sbi_type == "npe":
            def _joint_log_prob_fn(theta, key=None):
                L = 0.
                for n, (nde, weight) in enumerate(zip(self.ndes, self.weights)):
                    if key is not None:
                        key = jr.fold_in(key, n)
                    nde_log_L = nde.log_prob(x=theta, y=data, key=key)
                    L_nde = weight * jnp.exp(nde_log_L)
                    L = L + L_nde
                return jnp.log(L) 

        return _joint_log_prob_fn

    def _ensemble_log_prob_fn(self, datavectors, prior=None):
        """ 
            Get log-probability function for NDE at given observation 
            for whole ensemble of NDEs.
            - some NDEs may have a probabilistic estimate of the likelihood
              so a key is provided, the ndes are set to inference mode to 
              imply this key is not used for dropout etc.
        """
        # if datavectors is a list, jax.tree_map(lambda x: nde.log_prob(x=x, y=theta, key=key))
        # > assert isinstance(datavectors, List[Array, ...])
        # if its an array, check if it has a 'batch dim' 
        # 

        if self.sbi_type == "nle":

            # def _joint_log_prob_fn(theta, key=None):
            #     Ls = jax.tree_map(
            #         lambda nde, x: jnp.exp(nde.log_prob(x=x, y=theta, key=key)), self.ndes
            #     )
            #     return Ls

            def _joint_log_prob_fn(theta, key=None):
                L = 0.
                for n, (nde, weight) in enumerate(zip(self.ndes, self.weights)):
                    if key is not None:
                        key = jr.fold_in(key, n)
                    nde_log_L = nde.log_prob(x=data, y=theta, key=key)
                    L_nde = weight * jnp.exp(nde_log_L)
                    L = L + L_nde
                L = jnp.log(L) 
                if prior is not None:
                    L = L + prior.log_prob(theta)
                return L

        if self.sbi_type == "npe":
            def _joint_log_prob_fn(theta, key=None):
                L = 0.
                for n, (nde, weight) in enumerate(zip(self.ndes, self.weights)):
                    if key is not None:
                        key = jr.fold_in(key, n)
                    nde_log_L = nde.log_prob(x=theta, y=data, key=key)
                    L_nde = weight * jnp.exp(nde_log_L)
                    L = L + L_nde
                return jnp.log(L) 

        return _joint_log_prob_fn

    def ensemble_likelihood(self, data):
        return self.ensemble_log_prob_fn(data, prior=None)

    def calculate_stacking_weights(self, losses):
        """
            Calculate weightings of NDEs in ensemble
            - losses is a list of final-epoch validation losses
        """
        Ls = jnp.array(
            [-losses[n] for n, _ in enumerate(self.ndes)]
        )
        nde_weights = jnp.exp(Ls) / jnp.exp(Ls).sum() #jax.nn.softmax(Ls)
        return nde_weights

    def save_ensemble(self, path):
        eqx.tree_serialise_leaves(path, self)

    def load_ensemble(self, path):
        return eqx.tree_deserialise_leaves(path, self)
