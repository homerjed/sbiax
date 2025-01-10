import abc
from typing import NamedTuple, Literal, Tuple, Generator
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
    """
    An in-memory data loader designed to support neural likelihood estimation (NLE) 
    and neural posterior estimation (NPE). This loader organizes simulations and 
    parameters into batches for training, and manages data shuffling and batching.

    Attributes:
        simulations (Array): The simulation data, with `n` samples 
            and `x` features per sample.
        parameters (Array): The corresponding parameter data, with `n` 
            samples and `y` features per sample.
        train_mode (Literal["nle", "npe"]): The training mode, either "nle" or "npe".
        key (Key): The random key for data shuffling and reproducibility.

    Methods:
        n_batches(batch_size: int) -> int:
            Computes the number of batches given a batch size.

        loop(batch_size: int) -> Generator:
            Generates batches of data, shuffling if necessary, while organizing 
            data for NLE or NPE based on the training mode.
    """

    def __init__(
        self, 
        simulations: Float[Array, "n x"], 
        parameters: Float[Array, "n y"], 
        train_mode: Literal["nle", "npe"],
        *, 
        key: Key
    ): 
        self.simulations = simulations 
        self.parameters = parameters 
        self.train_mode = train_mode.lower()
        self.key = key
        assert self.train_mode.lower() in ["nle", "npe"]

    @property 
    def n_batches(self, batch_size: int) -> int:
        return max(int(self.simulations.shape[0] / batch_size), 1)

    def loop(
        self, batch_size: int
    ) -> Generator[Tuple[Float[Array, "batch x"], Float[Array, "batch y"]], None, None]:
        """
        Generates batches of simulation and parameter data for training.

        This function handles data shuffling and batching for training neural likelihood 
        estimation (NLE) or neural posterior estimation (NPE). Depending on the batch size 
        and the dataset size, it either yields the entire dataset or divides it into smaller 
        batches. Shuffling is applied for reproducibility and to prevent overfitting.

        Args:
            batch_size: The number of samples in each batch. If the batch size is larger 
                than the dataset size, the entire dataset is returned as a single batch.

        Yields:
            A tuple containing:
                - A batch of simulation data with shape `(batch, x)`.
                - A batch of parameter data with shape `(batch, y)`.

        Notes:
            - The batching and data preparation align with the training mode, which is either 
            "nle" (neural likelihood estimation) or "npe" (neural posterior estimation).
            - This function operates as an infinite generator, repeatedly cycling through 
            the data.

        Example:
            ```python
            loader = _InMemoryDataLoader(simulations, parameters, "nle", key=jr.PRNGKey(0))
            batch_size = 64

            for batch_x, batch_y in loader.loop(batch_size):
                # Perform training or evaluation with batch_x and batch_y
                ...
            ```
        """
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
    """
        Ultra simple and jit compilable dataloader.
    """
    arrays: tuple[Float[Array, "n x"], Float[Array, "n y"]]
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