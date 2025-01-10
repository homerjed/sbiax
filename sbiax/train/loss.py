from typing import Optional
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.sharding import PositionalSharding 
import equinox as eqx
from jaxtyping import Key, Array, Float


def pdf_mse_loss(
    nde: eqx.Module, 
    x: Float[Array, "..."], 
    y: Float[Array, "..."], 
    pdf: Float[Array, "..."],
    key: Optional[Key[jnp.ndarray, "..."]] = None
) -> Float[Array, ""]:
    p_x_y = nde.loss(x=x, y=y, key=key) 
    return jnp.square(jnp.subtract(p_x_y, pdf))


@eqx.filter_jit
def batch_loss_fn(
    nde: eqx.Module, 
    x: Float[Array, "..."], 
    y: Float[Array, "..."], 
    pdfs: Optional[Float[Array, "..."]] = None, 
    key: Optional[Key[jnp.ndarray, "..."]] = None
) -> Float[Array, ""]:
    nde = eqx.nn.inference_mode(nde, False)
    keys = jr.split(key, len(x))
    loss = jax.vmap(nde.loss)(x=x, y=y, key=keys).mean()
    return loss


@eqx.filter_jit
def batch_eval_fn(
    nde: eqx.Module, 
    x: Float[Array, "..."], 
    y: Float[Array, "..."], 
    pdfs: Optional[Float[Array, "..."]] = None, 
    key: Optional[Key[jnp.ndarray, "..."]] = None,
    replicated_sharding: Optional[PositionalSharding] = None
) -> Float[Array, ""]:
    if replicated_sharding is not None:
        nde = eqx.filter_shard(nde, replicated_sharding)
    nde = eqx.nn.inference_mode(nde, True)
    keys = jr.split(key, len(x))
    loss = jax.vmap(nde.loss)(x=x, y=y, key=keys).mean()
    return loss