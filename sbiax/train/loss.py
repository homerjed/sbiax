from typing import Optional
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.sharding import NamedSharding
import equinox as eqx
from jaxtyping import Key, Array, PyTree


def pdf_mse_loss(
    nde: eqx.Module, 
    x: Array, 
    y: Array, 
    pdf: Array,
    key: Key
) -> Array:
    p_x_y = nde.loss(x=x, y=y, key=key) 
    return jnp.square(jnp.subtract(p_x_y, pdf))


@eqx.filter_jit
def batch_loss_fn(
    nde: eqx.Module, 
    x: Array, 
    y: Array,
    pdfs: Optional[Array] = None, 
    key: Key = None
) -> Array:
    nde = eqx.nn.inference_mode(nde, False)
    keys = jr.split(key, len(x))
    loss = jax.vmap(nde.loss)(x=x, y=y, key=keys).mean()
    return loss


@eqx.filter_jit
def batch_eval_fn(
    nde: eqx.Module, 
    x: Array, 
    y: Array,  
    pdfs: Optional[Array] = None, 
    key: Key = None
) -> Array:
    nde = eqx.nn.inference_mode(nde, True)
    keys = jr.split(key, len(x))
    loss = jax.vmap(nde.loss)(x=x, y=y, key=keys).mean()
    return loss