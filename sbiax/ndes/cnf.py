from typing import Callable, Tuple, Optional
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax as dfx
from jaxtyping import Array, Float, Key, jaxtyped, PRNGKeyArray
from beartype import beartype as typechecker

TimeArray = Float[Array, ""]


def get_timestep_embedding(timesteps, embedding_dim: int):
    # Convert scalar timesteps to an array
    assert embedding_dim % 2 == 0
    if jnp.isscalar(timesteps):
        timesteps = jnp.array(timesteps)
    timesteps *= 1000.
    half_dim = embedding_dim // 2
    emb = jnp.log(10_000.) / (half_dim - 1.)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = timesteps * emb
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)])
    return emb


def get_time_embedder(embedding_dim):
    return (
        partial(get_timestep_embedding, embedding_dim=embedding_dim)
        if embedding_dim is not None 
        else lambda t: t 
    )


def _log_prob_exact(
    t: float | Float[Array, "1"], 
    y: Float[Array, "y"], 
    args: Tuple[None, Float[Array, "q"], eqx.Module, Key]
) -> Tuple[Float[Array, "y"], Float[Array, "1"]]:
    """ Compute trace directly. Use this for low dimensions. """
    y, _ = y
    _, q, func, key, t_embed = args
    t = jnp.atleast_1d(t)

    # fn = lambda y: func(jnp.concatenate([y, t, q]))  
    fn = lambda y: func(y, t_embed(t), q, key=key)
    f, f_vjp = jax.vjp(fn, y) 

    # Compute trace of Jacobian
    (size,) = y.shape
    (dfdy,) = jax.vmap(f_vjp)(jnp.eye(size))
    logp = jnp.trace(dfdy)
    return f, logp


def _log_prob_approx(
    t: TimeArray, 
    y: Float[Array, "y"], 
    args: Tuple[None, Float[Array, "q"], eqx.Module, Key]
) -> Tuple[Float[Array, "y"], Float[Array, "1"]]:
    """ Approx. trace using Hutchinson's trace estimator. """
    y, _ = y
    z, q, func, key, t_embed = args
    t = jnp.atleast_1d(t)
    
    fn = lambda y: func(y, t_embed(t), q, key=key)
    f, f_vjp = jax.vjp(fn, y) 
    
    # Trace estimator
    (z_dfdy,) = f_vjp(z)
    logp = jnp.sum(z_dfdy * z)
    return f, logp


def _get_solver() -> dfx.AbstractSolver:
    return dfx.Heun()


class Linear(eqx.Module):
    weight: Array
    bias: Optional[Array] = None
    use_bias: bool = True

    def __init__(
        self, 
        in_size: int, 
        out_size: int, 
        use_bias: bool = True, 
        *, 
        key: Key
    ):
        lim = jnp.sqrt(1. / (in_size + 1.))
        self.weight = jr.truncated_normal(
            key, shape=(out_size, in_size), lower=-2., upper=2.
        ) * lim
        if use_bias:
            self.bias = jnp.zeros((out_size,))
        self.use_bias = use_bias

    def __call__(self, x: Float[Array, "self.in_size"]) -> Float[Array, "self.out_size"]:
        y = self.weight @ x 
        if self.use_bias:
            y = y + self.bias
        return y



class ResidualNetwork(eqx.Module):
    _in: Linear
    layers: Tuple[Linear]
    dropouts: Tuple[eqx.nn.Dropout]
    _out: Linear
    activation: Callable
    dropout_rate: float
    y_dim: int

    def __init__(
        self, 
        in_size: int, 
        out_size: int, 
        width_size: int, 
        depth: int, 
        y_dim: int, 
        activation: Callable,
        dropout_rate: float = 0.,
        *, 
        key: Key
    ):
        """ Time-embedding may be necessary """
        in_key, *net_keys, out_key = jr.split(key, 2 + depth)
        # NOTE: was in_size + y_dim + 1 but t_emb_dim added to y_dim here
        self._in = Linear(in_size + y_dim, width_size, key=in_key)
        layers = [
            Linear(
                width_size + y_dim, width_size, key=_key
            )
            for _key in net_keys 
        ]
        self._out = Linear(width_size, out_size, key=out_key)
        # NOTE: make this optional
        dropouts = [
            eqx.nn.Dropout(p=dropout_rate) for _ in layers
        ]
        self.layers = tuple(layers)
        self.dropouts = tuple(dropouts)
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.y_dim = y_dim
    
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        x: Float[Array, "{self.in_size}"], 
        t: TimeArray, 
        y: Float[Array, "{self.y_dim}"],
        *, 
        key: PRNGKeyArray
    ) -> Float[Array, "{self.out_size}"]:
        t = jnp.atleast_1d(t)
        xyt = jnp.concatenate([x, y, t])
        h0 = self.activation(self._in(xyt))
        h = h0
        for l, d in zip(self.layers, self.dropouts):
            # Condition on time at each layer
            hyt = jnp.concatenate([h, y, t])
            h = l(hyt)
            h = d(h, key=key)
            h = self.activation(h)
            h = h0 + h
        o = self._out(h)
        return o


class MLP(eqx.nn.MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self, 
        x: Array, 
        t: float | Array, 
        y: Array, 
        *, 
        key: Key
    ) -> Array:
        return super().__call__(jnp.concatenate([x, t, y]))


class SquashMLP(eqx.Module):
    layers: list[eqx.nn.Linear]
    norms: list[eqx.nn.LayerNorm]
    activation: Callable

    def __init__(
        self, 
        in_size: int, 
        width_size: int, 
        depth: int, 
        y_dim: int, 
        dropout_rate: float = 0.,
        activation: Callable = jax.nn.tanh,
        *, 
        key: Key
    ):
        keys = jr.split(key, depth + 1)
        layers = []
        norms = []
        if depth == 0:
            layers.append(
                ConcatSquash(
                    in_size=in_size, 
                    out_size=in_size, 
                    y_dim=y_dim, 
                    dropout_rate=dropout_rate, 
                    key=keys[0]
                )
            )
        else:
            layers.append(
                ConcatSquash(
                    in_size=in_size, 
                    out_size=width_size, 
                    y_dim=y_dim, 
                    dropout_rate=dropout_rate,
                    key=keys[0]
                )
            )
            for i in range(depth - 1):
                layers.append(
                    ConcatSquash(
                        in_size=width_size, 
                        out_size=width_size, 
                        y_dim=y_dim, 
                        dropout_rate=dropout_rate,
                        key=keys[i + 1]
                    )
                )
            layers.append(
                ConcatSquash(
                    in_size=width_size, 
                    out_size=in_size, 
                    y_dim=y_dim, 
                    dropout_rate=dropout_rate,
                    key=keys[-1]
                )
            )
        self.layers = layers
        self.norms = norms 
        self.activation = activation

    def __call__(self, x, t, y, key=None):
        t = jnp.atleast_1d(t)
        for layer in self.layers[:-1]:
            if key is not None:
                key, _ = jr.split(key)
            x = layer(x, t, y, key=key)
            x = self.activation(x)
        if key is not None:
            key, _ = jr.split(key)
        x = self.layers[-1](x, t, y, key=key)
        return x


class ConcatSquash(eqx.Module):
    lin1: Linear
    lin2: Linear
    lin3: Linear
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout

    def __init__(self, *, in_size, out_size, y_dim, dropout_rate, key):
        key1, key2, key3 = jr.split(key, 3)
        self.lin1 = Linear(in_size, out_size, key=key1)
        self.lin2 = Linear(y_dim, out_size, key=key2)
        self.lin3 = Linear(y_dim, out_size, use_bias=False, key=key3)
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)

    def __call__(self, x, t, y, *, key=None):
        if key is not None:
            keys = jr.split(key)
        else:
            keys = [None, None]
        ty = jnp.concatenate([t, y])
        v = self.dropout1(self.lin1(x) * jax.nn.sigmoid(self.lin2(ty)), key=keys[0])
        u = self.dropout2(self.lin3(ty), key=keys[1])
        return v + u


class CNF(eqx.Module):
    net: eqx.Module
    x_dim: int
    y_dim: int
    dt: float
    t1: float
    exact_log_prob: bool 
    solver: dfx.AbstractSolver
    time_embedder: Callable
    scaler: eqx.Module

    def __init__(
        self,
        event_dim: int,
        context_dim: int, 
        width_size: int,
        depth: int,
        activation: Callable,
        dt: float,  
        t1: float,  
        t_emb_dim: int = None,
        exact_log_prob: bool = True, 
        dropout_rate: float = 0.,
        solver: Optional[dfx.AbstractSolver] = None,
        scaler: eqx.Module = None,
        *,
        key: Key
    ):
        if t_emb_dim is not None:
            y_dim = context_dim + t_emb_dim
        else:
            y_dim = context_dim + 1
  
        self.net = SquashMLP(
            in_size=event_dim,
            width_size=width_size,
            depth=depth,
            y_dim=y_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            key=key
        )
        self.x_dim = event_dim
        self.y_dim = context_dim 
        self.dt = dt
        self.t1 = t1
        self.exact_log_prob = exact_log_prob
        self.solver = solver
        self.time_embedder = get_time_embedder(t_emb_dim)
        self.scaler = scaler
    
    @jaxtyped(typechecker=typechecker)
    def log_prob(
        self, 
        x: Float[Array, "{self.x_dim}"], 
        y: Float[Array, "{self.y_dim}"], 
        key: Optional[PRNGKeyArray] = None,
        exact_log_prob: Optional[bool] = False 
    ) -> Float[Array, ""]:
        solver = _get_solver() if self.solver is None else self.solver

        if self.scaler is not None:
            x, y = self.scaler.forward(x, y)

        if key is not None:
            key, key_eps = jr.split(key)

        args = (y, self.net, key, self.time_embedder)

        # Train or evaluate with exact or estimated Jacobian 
        if exact_log_prob:
            _use_exact_log_prob = exact_log_prob
        else:
            _use_exact_log_prob = self.exact_log_prob

        term = dfx.ODETerm(
            _log_prob_exact if _use_exact_log_prob else _log_prob_approx
        ) 

        if _use_exact_log_prob:
            eps = None
        else:
            eps = jr.normal(key_eps, (self.x_dim,))

        # Add Hutchinson estimator noise samples
        args = (eps,) + args

        x0 = (x, 0.)
        soln = dfx.diffeqsolve(term, solver, 0., self.t1, self.dt, x0, args)
        (z,), (delta_log_likelihood,) = soln.ys
        log_prob = delta_log_likelihood + self.prior_log_prob(z)
        return log_prob

    @jaxtyped(typechecker=typechecker)
    def sample_and_log_prob(
        self, 
        key: PRNGKeyArray, 
        y: Float[Array, "{self.y_dim}"], 
        exact_log_prob: Optional[bool] = None
    ) -> Tuple[Float[Array, "{self.x_dim}"], Float[Array, ""]]:
        """ Sample many samples given a single condition """
        key_eps, key_z, key_sample = jr.split(key, 3)

        args = (y, self.net, key_sample, self.time_embedder)

        # Sample with exact or estimated Jacobian 
        if exact_log_prob is not None:
            _use_exact_log_prob = exact_log_prob
        else:
            _use_exact_log_prob = self.exact_log_prob

        term = dfx.ODETerm(
            _log_prob_exact if _use_exact_log_prob else _log_prob_approx
        ) 

        if _use_exact_log_prob or self.exact_log_prob:
            eps = None
        else:
            eps = jr.normal(key_eps, (self.x_dim,))

        # Add Hutchinson estimator noise samples
        args = (eps,) + args

        # Latent sample
        z = jr.normal(key_z, (self.x_dim,))

        solver = _get_solver() if self.solver is None else self.solver
        delta_log_likelihood = 0.
        x1 = (z, delta_log_likelihood)
        soln = dfx.diffeqsolve(term, solver, self.t1, 0., -self.dt, x1, args)
        (x,), (delta_log_likelihood,) = soln.ys

        log_prob = delta_log_likelihood + self.prior_log_prob(z)
        return x, log_prob

    def sample_and_log_prob_n(
        self, 
        key: Key, 
        y: Array, 
        n_samples: int, 
        exact_log_prob: Optional[bool] = None
    ) -> Tuple[Array, Array]:
        """ Sample x ~ p(x|y) for a fixed y. """

        if exact_log_prob is not None:
            _use_exact_log_prob = exact_log_prob
        else:
            _use_exact_log_prob = self.exact_log_prob

        _sampler = jax.vmap(
            partial(
                self.sample_and_log_prob, 
                exact_log_prob=_use_exact_log_prob
            ), 
            in_axes=(0, None)
        )
        keys = jr.split(key, n_samples)
        samples, log_probs = _sampler(keys, y)
        return samples, log_probs 

    def prior_log_prob(self, z: Array) -> Array:
        return jax.scipy.stats.multivariate_normal.logpdf(
            z, jnp.zeros(self.x_dim), jnp.eye(self.x_dim)
        )

    def loss(self, x: Array, y: Array, key: Optional[Key] = None) -> Array:
        return -self.log_prob(x, y, key=key)