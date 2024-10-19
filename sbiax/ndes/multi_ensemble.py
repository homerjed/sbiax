from typing import List
import equinox as eqx
from tensorflow_probability.substrates.jax.distributions import Distribution

from .ensemble import Ensemble


def default(v, d):
    return v if v is not None else d


class MultiEnsemble(eqx.Module):
    ensembles: List[Ensemble]
    prior: Distribution

    def __init___(self, ensembles, prior):
        self.ensembles = ensembles
        self.prior = prior # Allow to be overwritten in inference call

    def get_multi_ensemble_log_prob_fn(self, datavectors, prior=None):
        _prior = default(prior, self.prior)

        def _multi_ensemble_log_prob_fn(theta):
            # Loop over matched ensembles / datavectors
            L = 0.
            for ensemble, datavector in zip(self.ensembles, datavectors):
                ensemble_log_L = ensemble.ensemble_likelihood(datavector)(theta)
                L = L + ensemble_log_L
            L = L + _prior.log_prob(theta)
            return L 

        return _multi_ensemble_log_prob_fn

    def load_ensembles(self, paths, ensembles):
        # Load sub-ensembles
        self.ensembles = [
            eqx.tree_deserialise_leaves(path, ensemble)
            for path, ensemble in zip(paths, ensembles)
        ]

    def save_ensemble(self, path):
        eqx.tree_serialise_leaves(path, self)

    def load_ensemble(self, path):
        return eqx.tree_deserialise_leaves(path, self)