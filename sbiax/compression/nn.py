from typing import Tuple, Optional
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import PRNGKeyArray, Array
import optax
import numpy as np 
from tqdm.auto import trange

"""
    Tools for compression with neural networks.
    - train a user-defined `eqx.Module` network that compresses a datavector
      to a model-dimensional summary, by minimising a MSE loss.
"""


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
    key: PRNGKeyArray, 
    model: eqx.Module, 
    D: Array, 
    Y: Array, 
    opt: optax.GradientTransformation, 
    n_batch: int, 
    patience: Optional[int], 
    n_steps: int = 100_000, 
    valid_fraction: int = 0.9, 
) -> Tuple[eqx.Module, Array]:
    """
        Trains a neural network model with early stopping.

        Args:
            key: A `PRNGKeyArray`.
            model: The neural network model to be trained, represented as an `eqx.Module`.
            D: The input data matrix (`Array`), where rows are data points and columns are features.
            Y: The target values (`Array`) corresponding to the input data.
            opt: The optimizer to be used for gradient updates, defined as an `optax.GradientTransformation`.
            n_batch: The number of data points per mini-batch for each training step (`int`).
            patience: The number of steps to continue without improvement on the validation loss 
                before early stopping is triggered (`int`).
            n_steps: The maximum number of training steps to perform (`int`, optional). Default is 100,000.
            valid_fraction: The fraction of the data to use for training, with the remainder
                used for validation (`float`, optional). Default is 0.9 (90% training, 10% validation).

        Returns:
            Tuple[`eqx.Module`, `Array`]: 
                - The trained `model` after the optimization process.
                - A 2D array of shape (n_steps, 2), where the first column contains the training loss at each 
                step, and the second column contains the validation loss.
        
        Notes:
            1. The data `D` and targets `Y` are split into training and validation sets based on the 
            `valid_fraction` parameter.
            4. Early stopping occurs if the validation loss does not improve within a specified 
            number of steps (`patience`).
            5. The function returns the trained model and the recorded training/validation loss history.

    """
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

            if patience is not None:
                if (step > 0) and (step - np.argmin(L[:step, 1]) > patience):
                    steps.set_description_str(f"Stopped at {step=}")
                    break

    return model, L[:step]