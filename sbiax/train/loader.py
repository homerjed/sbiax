import abc
from typing import NamedTuple
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key, Array


class Sample(NamedTuple):
    x: Array 
    y: Array 


def sort_sample(train_mode, simulations, parameters):
    # Sort simulations and parameters according to NPE or NLE
    _nle = train_mode.lower() == "nle"
    return Sample(
        x=simulations if _nle else parameters,
        y=parameters if _nle else simulations 
    )


class _AbstractDataLoader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, data, targets, *, key):
        pass

    def __iter__(self):
        raise RuntimeError("Use `.loop` to iterate over the data loader.")

    @abc.abstractmethod
    def loop(self, batch_size):
        pass


class _InMemoryDataLoader(_AbstractDataLoader):
    def __init__(
        self, 
        simulations: Array, 
        parameters: Array, 
        train_mode: str, 
        *, 
        key: Key
    ): 
        self.simulations = simulations 
        self.parameters = parameters 
        self.train_mode = train_mode.lower()
        self.key = key
        assert self.train_mode.lower() in ["nle", "npe"]

    @property 
    def n_batches(self, batch_size):
        return max(int(self.simulations.shape[0] / batch_size), 1)

    def loop(self, batch_size: int):
        # Loop through dataset, batching, while organising data for NPE or NLE
        dataset_size = self.simulations.shape[0]
        one_batch = batch_size >= dataset_size
        key = self.key
        indices = jnp.arange(dataset_size)
        while True:
            # Yield whole dataset if batch size is larger than dataset size
            if one_batch:
                yield sort_sample(
                    self.train_mode, 
                    self.simulations, 
                    self.parameters
                )
            else:
                key, subkey = jr.split(key)
                perm = jr.permutation(subkey, indices)
                start = 0
                end = batch_size
                while end < dataset_size:
                    batch_perm = perm[start:end]
                    yield sort_sample(
                        self.train_mode, 
                        self.simulations[batch_perm], 
                        self.parameters[batch_perm] 
                    )
                    start = end
                    end = start + batch_size


class DataLoader(eqx.Module):
    arrays: tuple[Array, ...]
    batch_size: int
    key: Key

    def __check_init__(self):
        dataset_size = self.arrays[0].shape[0]
        assert all(array.shape[0] == dataset_size for array in self.arrays)

    def __call__(self, step):
        dataset_size = self.arrays[0].shape[0]
        num_batches = dataset_size // self.batch_size
        epoch = step // num_batches
        key = jr.fold_in(self.key, epoch)
        perm = jr.permutation(key, jnp.arange(dataset_size))
        start = (step % num_batches) * self.batch_size
        slice_size = self.batch_size
        batch_indices = jax.lax.dynamic_slice_in_dim(perm, start, slice_size)
        return tuple(array[batch_indices] for array in self.arrays)