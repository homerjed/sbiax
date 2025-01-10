from typing import Tuple, List
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PositionalSharding, Mesh, PartitionSpec
from jax.experimental import mesh_utils
from jaxtyping import Array, Float
import pandas as pd


def make_df(
    samples: Float[Array, "..."], 
    log_probs: Float[Array, "..."], 
    param_names: List[str]
) -> pd.DataFrame:
    """
        Chainconsumer requires pd.Dataframe for chains.
    """
    df = pd.DataFrame(samples, columns=param_names).assign(log_posterior=log_probs)
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