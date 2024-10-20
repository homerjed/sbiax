import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np 
from tqdm.auto import trange


def loss(model, x, y):
    def fn(x, y):
        y_ = model(x)
        return jnp.square(jnp.subtract(y_, y))
    return jnp.mean(jax.vmap(fn)(x, y))


@eqx.filter_jit
def evaluate(model, x, y):
    return loss(model, x, y)


@eqx.filter_jit
def make_step(model, opt_state, x, y, opt):
    loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


def get_batch(D, Y, n, key):
    idx = jr.choice(key, jnp.arange(D.shape[0]), (n,))
    return D[idx], Y[idx]


def fit_nn(
    key, 
    model, 
    D, 
    Y, 
    opt, 
    n_batch, 
    patience, 
    n_steps=100_000, 
    valid_fraction=0.9, 
):
    n_s, _ = D.shape

    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    Xt, Xv = jnp.split(D, [int(valid_fraction * n_s)])
    Yt, Yv = jnp.split(Y, [int(valid_fraction * n_s)])

    L = np.zeros((n_steps, 2))
    with trange(
        n_steps, desc="Training NN", colour="blue"
    ) as steps:
        for step in steps:
            key_t, key_v = jr.split(jr.fold_in(key, step))

            x, y = get_batch(Xt, Yt, n=n_batch, key=key_t)
            model, opt_state, train_loss = make_step(model, opt_state, x, y, opt)

            x, y = get_batch(Xv, Yv, n=n_batch, key=key_v)
            valid_loss = evaluate(model, x, y)

            L[step] = train_loss, valid_loss
            steps.set_postfix_str(
                (
                    f"train={train_loss.item():.3E}, " + 
                    f"valid={valid_loss.item():.3E}"
                )
            )

            if (step > 0) and (step - np.argmin(L[:step, 1]) > patience):
                steps.set_description_str(f"Stopped at {step=}")
                break

    return model, L[:step]