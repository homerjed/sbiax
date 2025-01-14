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

    in_size = 3
    out_size = 2

    net = eqx.nn.MLP(
        in_size, 
        out_size, 
        width_size=8, 
        depth=2, 
        activation=jax.nn.tanh,
        key=net_key
    )

    opt = optax.adamw(1e-3)

    D = jnp.ones((100, in_size))
    Y = jnp.ones((100, out_size))

    model, losses = fit_nn(
        net_train_key, 
        net, 
        train_data=(D, Y), 
        opt=opt, 
        n_batch=8, 
        patience=10
    )

    X = jax.vmap(model)(D)

    assert jnp.all(jnp.isfinite(X))
    assert jnp.all(jnp.isfinite(losses))
    assert X.shape == (D.shape[0], out_size)