from dataclasses import dataclass
from typing import Optional, Literal, Sequence, Tuple, Callable
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import equinox as eqx
from jaxtyping import Float, Array, PRNGKeyArray, Scalar


class GMMParams(eqx.Module):
    logits: Array
    means: Array
    var_diag: Optional[Array] = None   # diagonal covariance
    L_tril: Optional[Array]  = None    # Cholesky factor

    def __post_init__(self):
        # exactly one of var_diag or L_tril must be set
        if (self.var_diag is None) == (self.L_tril is None):
            raise ValueError("Specify exactly one of var_diag or L_tril.")

        # simple shape checks (optional)
        K, Dy = self.means.shape
        if self.logits.shape[-1] != K:
            raise ValueError("logits last dim must equal K.")
        if self.var_diag is not None and self.var_diag.shape != (K, Dy):
            raise ValueError("var_diag must be (K, Dy).")
        if self.L_tril is not None and self.L_tril.shape != (K, Dy, Dy):
            raise ValueError("L_tril must be (K, Dy, Dy).")


class GaussianMixtureMath:
    def __init__(self, cov_type: Literal["diag", "full"] = "diag", eps: float = 1e-6):
        self.cov_type = cov_type
        self.eps = eps

    # per-component log N(y | μ_k, Σ_k) -> (K,)
    def _log_normal_diag(self, y: Array, mean: Array, var: Array) -> Array:
        Dy = y.shape[0]

        var = jnp.clip(var, 1e-12, None)

        log_det = jnp.sum(jnp.log(var), axis=-1)                     # (K,)

        quad = jnp.sum((y[jnp.newaxis, :] - mean) ** 2. / var, axis=-1)      # (K,)

        return -0.5 * (Dy * jnp.log(2. * jnp.pi) + log_det + quad)

    def _log_normal_full(self, y: Array, mean: Array, L: Array) -> Array:
        # y: (Dy,), mean: (K, Dy), L: (K, Dy, Dy) lower-tri

        Dy = y.shape[0]

        delta = mean - y[jnp.newaxis, :]                                     # (K, Dy)

        def one(Lk, dk):
            z = jsp.linalg.solve_triangular(Lk, dk, lower=True)       # (Dy,)
            return jnp.sum(z * z)                                     # scalar

        quad = jax.vmap(one)(L, delta)                                # (K,)

        log_det = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)  # (K,)

        return -0.5 * (Dy * jnp.log(2.0 * jnp.pi) + log_det + quad)

    def log_prob(self, y: Array, p: GMMParams) -> Array:

        if self.cov_type == "diag":
            log_comp = self._log_normal_diag(y, p.means, p.var_diag)     # (K,)
        else:
            log_comp = self._log_normal_full(y, p.means, p.L_tril)       # (K,)

        log_mix = jax.nn.log_softmax(p.logits)                            # (K,)

        return jsp.special.logsumexp(log_mix + log_comp, axis=-1)         # scalar ()

    def responsibilities(self, y: Array, p: GMMParams) -> Array:

        if self.cov_type == "diag":
            log_comp = self._log_normal_diag(y, p.means, p.var_diag)
        else:
            log_comp = self._log_normal_full(y, p.means, p.L_tril)

        log_mix = jax.nn.log_softmax(p.logits)
        log_num = log_mix + log_comp

        return jnp.exp(log_num - jsp.special.logsumexp(log_num, axis=-1, keepdims=True))  # (K,)

    def sample(self, p: GMMParams, n: int, *, key: jax.Array) -> Array:
        """Return (n, Dy) samples."""
        K, Dy = p.means.shape
        k1, k2 = jr.split(key, 2)

        comp = jr.categorical(k1, p.logits, shape=(n,))         # (n,)
        mu = p.means[comp]                                              # (n, Dy)

        if self.cov_type == "diag":
            std = jnp.sqrt(p.var_diag[comp])                            # (n, Dy)
            eps = jr.normal(k2, shape=(n, Dy), dtype=mu.dtype)
            x = mu + std * eps
        else:
            L = p.L_tril[comp]                                          # (n, Dy, Dy)
            eps = jr.normal(k2, shape=(n, Dy), dtype=mu.dtype)
            x = mu + eps @ jnp.swapaxes(L, -1, -2)                   # (n, Dy)

        return x


def tril_from_vector(v: jnp.ndarray, D: int) -> jnp.ndarray:
    """(K * D(D+1)//2,) -> (K, D, D) lower-triangular."""
    t = D * (D + 1) // 2
    v = v.reshape(-1, t)             # (K, t)
    L = jnp.zeros((v.shape[0], D, D), dtype=v.dtype)
    ii, jj = jnp.tril_indices(D)
    return L.at[:, ii, jj].set(v)


class GMM(eqx.Module):
    x_dim: int
    y_dim: int
    layers: tuple[eqx.nn.Linear, ...]
    act: Callable = eqx.field(static=True)
    head_logits: eqx.nn.Linear
    head_means: eqx.nn.Linear
    head_scale: eqx.nn.Linear | None
    head_tril: eqx.nn.Linear | None
    context_dim: int = eqx.field(static=True)
    K: int = eqx.field(static=True)
    cov_type: str = eqx.field(static=True)
    min_scale: float = eqx.field(static=True)

    def __init__(
        self,
        event_dim: int,
        context_dim: int,
        K: int,
        *,
        hidden: Sequence[int] = (128, 128),
        cov_type: Literal["diag", "full"] = "diag",
        min_scale: float = 1e-7,
        key: PRNGKeyArray,
    ):
        self.x_dim = event_dim
        self.y_dim = context_dim
        self.context_dim, self.K = context_dim, K
        self.cov_type = cov_type
        self.min_scale = float(min_scale)
        self.act = jax.nn.tanh

        keys = jr.split(key, len(hidden) + 3)
        layers = []
        in_dim = event_dim
        for i, h in enumerate(hidden):
            layers.append(eqx.nn.Linear(in_dim, h, key=keys[i]))
            in_dim = h
        self.layers = tuple(layers)

        self.head_logits = eqx.nn.Linear(in_dim, K, key=keys[-3])
        self.head_means  = eqx.nn.Linear(in_dim, K * context_dim, key=keys[-2])
        if cov_type == "diag":
            self.head_scale = eqx.nn.Linear(in_dim, K * context_dim, key=keys[-1])
            self.head_tril  = None
        else:
            t = context_dim * (context_dim + 1) // 2
            self.head_tril  = eqx.nn.Linear(in_dim, K * t, key=keys[-1])
            self.head_scale = None

    def get_params(self, x: Array) -> GMMParams:
        """x: (event_dim,)  ->  per-sample mixture params (NamedTuple)."""

        h = x
        for lin in self.layers:
            h = self.act(lin(h))

        logits = self.head_logits(h)                                  # (K,)
        means  = self.head_means(h).reshape(self.K, self.context_dim) # (K, Dy)

        if self.cov_type == "diag":
            raw = self.head_scale(h).reshape(self.K, self.context_dim)
            var = jax.nn.softplus(raw) ** 2. + self.min_scale

            params = GMMParams(logits, means, var_diag=var, L_tril=None)
        else:
            t = self.context_dim * (self.context_dim + 1) // 2
            raw = self.head_tril(h)                                # (K*t,)
            L = tril_from_vector(raw, self.context_dim)            # (K, Dy, Dy)
            diag = jnp.diagonal(L, axis1=-2, axis2=-1)
            diag = jax.nn.softplus(diag) + self.min_scale
            L = L.at[..., jnp.arange(self.context_dim), jnp.arange(self.context_dim)].set(diag)

            params = GMMParams(logits, means, var_diag=None, L_tril=L)

        return params

    def log_prob(self, x: Array, y: Array, *args, **kwargs) -> Scalar:
        # Ignoring key here...
        return GaussianMixtureMath(cov_type=self.cov_type).log_prob(y, self.get_params(x))

    def loss(
        self, 
        x: Float[Array, "{self.x_dim}"], 
        y: Float[Array, "{self.y_dim}"], 
        key: Optional[PRNGKeyArray] = None
    ) -> Scalar:
        """
        Computes the loss for training the continuous normalizing flow model, which is 
        defined as the negative log-probability of the data `x` given the conditioning 
        context `y`.

        Args:
            x (Array): Input data for which the loss is calculated.
            y (Array): Conditioning used for the transformation.
            key (Optional[Key], optional): Optional random key for sampling if using approximate log-prob. 

        Returns:
            Array: Computed loss for the input data `x` given the context `y`.
        """
        return -self.log_prob(x, y, key=key)