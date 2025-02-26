from typing import Tuple, List, Optional
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PositionalSharding, Mesh, PartitionSpec
from jax.experimental import mesh_utils
from jaxtyping import Array, Float
import numpy as np
import pandas as pd


def make_df(
    samples: Float[Array, "..."], 
    log_probs: Optional[Float[Array, "..."]] = None, 
    *,
    parameter_strings: List[str]
) -> pd.DataFrame:
    """
        Chainconsumer requires pd.Dataframe for chains.
    """
    if log_probs is None:
        log_probs = jnp.ones((samples.shape[0],))
    df = pd.DataFrame(samples, columns=parameter_strings).assign(log_posterior=log_probs)
    return df


def nan_to_value(
    samples: Float[Array, "..."], 
    log_probs: Float[Array, "..."]
) -> Tuple[Float[Array, "..."], Float[Array, "..."]]:
    """
        Set any bad samples in an MCMC to very low probability.
    """
    log_probs = log_probs.at[~jnp.isfinite(log_probs)].set(-1e-100)
    return samples, log_probs


def get_shardings() -> Tuple[NamedSharding, PositionalSharding]:
    """
        Obtain array shardings for batches and models.
    """
    devices = jax.local_devices()
    n_devices = len(devices)
    print("Running on {} local devices: \n\t{}".format(n_devices, devices))

    if n_devices > 1:
        mesh = Mesh(devices, ("x",))
        sharding = NamedSharding(mesh, PartitionSpec("x"))

        devices = mesh_utils.create_device_mesh((n_devices, 1))
        replicated = PositionalSharding(devices).replicate()
    else:
        sharding = replicated = None

    return sharding, replicated


def marker(x, parameter_strings=None):
    x = np.asarray(x)
    if parameter_strings is None:
        parameter_strings = [str(n) for n in jnp.arange(x.size)]
    return dict(zip(parameter_strings, x))