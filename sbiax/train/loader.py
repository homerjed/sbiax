from typing import NamedTuple, Literal, Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key, Array, Float


class Sample(NamedTuple):
    x: Array 
    y: Array 


def sort_sample(
    train_mode: Literal["npe", "nle"], 
    simulations: Float[Array, "b x"],
    parameters: Float[Array, "b y"]
) -> Sample:
    """
        Sort simulations and parameters according to NPE or NLE
        
        Args:
            train_mode (`str`): NPE or NLE mode of SBI.
            simulations (`Array`): Simulations array.
            parameters (`Array`): Parameters array.
        
        Returns:
            (`Sample`): Ordered sample of simulations and parameters.
    """
    _nle = train_mode.lower() == "nle"
    return Sample(
        x=simulations if _nle else parameters,
        y=parameters if _nle else simulations 
    )


class DataLoader(eqx.Module):
    """
        Ultra simple and jit compilable dataloader.
    """
    arrays: tuple[Array, ...]
    batch_size: int
    key: Key

    def __check_init__(self):
        dataset_size = self.arrays[0].shape[0]
        assert all(array.shape[0] == dataset_size for array in self.arrays)

    def __call__(self, step: int) -> Tuple[Float[Array, "b x"], Float[Array, "b y"]]:
        """
            Return a batch of simulations and parameters given the step.

            Args:
                step (`int`): Training iteration.

            Returns:
                (`Tuple[Array, Array]`): Tuple of simulations and parameter arrays.
        """
        dataset_size = self.arrays[0].shape[0]
        num_batches = dataset_size // self.batch_size
        epoch = step // num_batches
        key = jr.fold_in(self.key, epoch)
        perm = jr.permutation(key, jnp.arange(dataset_size))
        start = (step % num_batches) * self.batch_size
        slice_size = self.batch_size
        batch_indices = jax.lax.dynamic_slice_in_dim(perm, start, slice_size)
        return tuple(array[batch_indices] for array in self.arrays)