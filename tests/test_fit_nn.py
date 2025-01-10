import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

from sbiax.compression.nn import fit_nn

"""
    Test fitting neural network compressors.
"""


def test_fit_nn():
    key = jr.key(0)

    net_key, net_train_key = jr.split(key)

    net = eqx.nn.MLP(
        3, 
        2, 
        width_size=8, 
        depth=2, 
        activation=jax.nn.tanh,
        key=net_key
    )

    opt = optax.adamw(1e-3)

    D = jnp.ones((100, 3))
    Y = jnp.ones((100, 2))

    model, losses = fit_nn(
        net_train_key, 
        net, 
        train_data=(D, Y), 
        opt=opt, 
        n_batch=8, 
        patience=10
    )

    X = jax.vmap(model)(D)