import time
import os
from typing import Tuple, Optional, Sequence, Callable, Any, Literal, List, Optional, Tuple

from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jss
from jax.sharding import NamedSharding, Sharding
import equinox as eqx
from optimistix import minimise, BFGS, LevenbergMarquardt, rms_norm
import optax
from jaxtyping import Key, Array, Float, Scalar, PRNGKeyArray, jaxtyped
import einops
from beartype import beartype as typechecker
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
import numpy as np 
from tqdm.auto import trange

from configs.log import setup_module_logger, get_log_level

DATA_PROCESS_TYPE_NN = os.environ.get("DATA_PROCESS_TYPE_NN", None) # Inputs/outputs to compression NN
USE_PRECISION_NN = True if os.environ.get("USE_PRECISION_NN", "").lower() in ("1", "true") else False
COVARIANCE_NN = True if os.environ.get("COVARIANCE_NN", "").lower() in ("1", "true") else False
NN_CLIP_NORM = True if os.environ.get("NN_CLIP_NORM", "").lower() in ("1", "true") else False

assert DATA_PROCESS_TYPE_NN in ["d", "p", "dp", None], (
    "DATA_PROCESS_TYPE_NN={}".format(DATA_PROCESS_TYPE_NN)
)

if USE_PRECISION_NN:
    assert DATA_PROCESS_TYPE_NN in ["d", None], (
        "CANNOT USE DATA_PROCESS_TYPE_NN={} with USE_PRECISION_NN=True".format(DATA_PROCESS_TYPE_NN)
    )

"""
    Tools for compression with neural networks.
    - train a user-defined `eqx.Module` network that compresses a datavector
    to a model-dimensional summary, by minimising a MSE loss.
"""

TYPECHECK = True if os.environ.get("TYPECHECK", "").lower() in ("1", "true") else False
if TYPECHECK:
    typecheck = jaxtyped(typechecker=typechecker)
else:
    typecheck = lambda x: x


logger, log_figs_dir = setup_module_logger(__name__, level=get_log_level())


def wrap_ensemble_ndes_with_processors(ensemble, processor):

    class NDEWrapper(eqx.Module):
        # Wrap a density estimator to have scaled inputs
        nde: Callable
        processor: "Processor"

        def __init__(self, nde, processor):
            self.nde = nde 
            self.processor = processor

        def log_prob(self, x, y, *args, **kwargs):
            if self.processor is not None:
                x = self.processor.forward_x(x)
                y = self.processor.forward_y(y)
            l = self.nde.log_prob(x, y, *args, **kwargs)
            # if self.processor is not None:
            #     y_ = self.processor.reverse_y(y_)
            return l

    if isinstance(processor, list):
        # Wrap after training
        for n, (nde, _processor) in enumerate(zip(ensemble.ndes, processor)):
            ensemble = eqx.tree_at(
                lambda e: e.ndes[n], 
                ensemble, 
                replace=NDEWrapper(nde, _processor)
            )
    else:
        # Wrap after training
        for n, nde in enumerate(ensemble.ndes):
            ensemble = eqx.tree_at(
                lambda e: e.ndes[n], 
                ensemble, 
                replace=NDEWrapper(nde, processor)
            )

    return ensemble


from typing import Optional
from jaxtyping import Array

class Processor(eqx.Module):
    mu_x: Optional[Array]
    std_x: Optional[Array]
    mu_y: Optional[Array]
    std_y: Optional[Array]

    eps: float = 1e-12  # scalar

    def __init__(
        self,
        x: Optional[Array],
        y: Optional[Array],
        *,
        axis: Optional[int] = 0,
        keepdims: bool = False,
        eps: float = 1e-12,
    ):
        if x is not None:
            self.mu_x = jnp.mean(x, axis=axis, keepdims=keepdims)
            self.std_x = jnp.std(x, axis=axis, keepdims=keepdims)
        else:
            self.mu_x = self.std_x = None

        if y is not None:
            self.mu_y = jnp.mean(y, axis=axis, keepdims=keepdims)
            self.std_y = jnp.std(y, axis=axis, keepdims=keepdims)
        else:
            self.mu_y = self.std_y = None

        self.eps = eps
        
    def forward_x(self, x: Array) -> Array:
        if (self.std_x is not None) and (self.mu_x is not None):
            std_x = jax.lax.stop_gradient(self.std_x)
            mu_x = jax.lax.stop_gradient(self.mu_x)
            x = (x - mu_x) / (std_x + self.eps)
        return x

    def forward_y(self, y: Array) -> Array:
        if (self.std_y is not None) and (self.mu_y is not None):
            std_y = jax.lax.stop_gradient(self.std_y)
            mu_y = jax.lax.stop_gradient(self.mu_y)
            y = (y - mu_y) / (std_y + self.eps)
        return y

    def reverse_x(self, x: Array) -> Array:
        if (self.std_x is not None) and (self.mu_x is not None):
            std_x = jax.lax.stop_gradient(self.std_x)
            mu_x = jax.lax.stop_gradient(self.mu_x)
            x = x * (std_x + self.eps) + mu_x
        return x

    def reverse_y(self, y: Array) -> Array:
        if (self.std_y is not None) and (self.mu_y is not None):
            std_y = jax.lax.stop_gradient(self.std_y)
            mu_y = jax.lax.stop_gradient(self.mu_y)
            y = y * (std_y + self.eps) + mu_y
        return y


def exists(v):
    return v is not None

from jax.scipy.linalg import solve_triangular


def split_head(raw_out, d: int, min_diag: float = 1e-8):
    """
    raw_out: shape (d + d*(d+1)//2,)
      first d are means, remaining are lower-triangular entries row-wise.
    returns:
      mu: shape (d,)
      L : shape (d, d) lower-triangular with positive diagonal
    """
    n_tril = d * (d + 1) // 2

    mu        = raw_out[:d]
    tril_flat = raw_out[d:d + n_tril]

    assert raw_out.shape == (d + n_tril,), (
        f"raw_out has shape {raw_out.shape}, but expected {(d + n_tril,)} "
        f"for d={d} (means + lower-tri entries)"
    )

    L = jnp.zeros((d, d), dtype=raw_out.dtype)
    i, j = jnp.tril_indices(d)
    L = L.at[i, j].set(tril_flat)

    # strictly positive diagonal: softplus + small jitter
    diag = jnp.arange(d)
    diag_pos = jax.nn.softplus(L[diag, diag]) + min_diag
    L = L.at[diag, diag].set(diag_pos)
    return mu, L


def nll_mvn_cholesky(y, mu, L):
    """
    Negative log-likelihood for N(y | mu, Sigma), Sigma = L L^T.
    y, mu: (d,)
    L    : (d, d) lower-triangular with positive diagonal
    returns: scalar nll
    """
    diff = y - mu                      # (d,)
    z = solve_triangular(L, diff[:, jnp.newaxis], lower=True)[:, 0]  # (d,)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    quad   = jnp.dot(z, z)
    return 0.5 * (logdet + quad)


def gaussian_nll_from_raw(y, raw_out, min_diag: float = 1e-8):
    """
    raw_out: (d + d*(d+1)//2,)
    y      : (d,)
    returns scalar nll
    """
    mu, L = split_head(raw_out, d=y.size, min_diag=min_diag)
    return nll_mvn_cholesky(y, mu, L)


# Optional: get Sigma back if you need it
def covariance_from_raw(raw_out, d: int, min_diag: float = 1e-5):
    mu, L = split_head(raw_out, d=d, min_diag=min_diag)
    Sigma = L @ L.T
    return mu, Sigma


class ExtraMLP(eqx.Module):
    layers: list[eqx.nn.Linear]
    dropouts: list[eqx.nn.Dropout | eqx.nn.Identity]
    layernorms: list[eqx.nn.LayerNorm | eqx.nn.Identity]
    activation: Callable
    final_activation: Optional[Callable]
    p: float
    parameter_dim: int
    covariance_nn: bool 

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_sizes: list[int],
        *,
        p: float = 0.0,
        activation: Callable = jax.nn.gelu,
        final_activation: Optional[Callable] = None,
        use_final_bias: bool = True,
        key: PRNGKeyArray,
        use_bias: bool = True,
        covariance_nn: bool = False
    ):
        """Arbitrary-width MLP with per-layer LayerNorm + Dropout (optional).

        Args:
            in_size: input feature dimension
            out_size: output feature dimension
            hidden_sizes: list of hidden layer widths, e.g. [128, 64, 32]
            p: dropout probability (applied to each layer output; 0 disables)
            activation: activation between hidden layers (not after last)
            final_activation: optional activation after final layer
            key: PRNG key
            use_bias: whether Linear layers include bias
        """
        self.p = p
        self.activation = activation
        self.final_activation = final_activation

        # If not training with summary covariance, output parameters only
        self.parameter_dim = out_size
        if covariance_nn:
            # Cholesky dimension of the output covariance
            n_tril = out_size * (out_size + 1) // 2
            out_size = out_size + n_tril
            self.covariance_nn = covariance_nn

        # Build Linear stack according to sizes
        if (
            hidden_sizes == [0] 
            or len(list(hidden_sizes)) == 0 
            or hidden_sizes == [None]
        ):
            sizes = [in_size, out_size]
        else:
            sizes = [in_size, *hidden_sizes, out_size]
        n_layers = len(sizes) - 1
        k_lin = jr.split(key, n_layers)

        layers = []
        for i in range(n_layers):

            if sizes[i + 1] == out_size:
                use_bias = use_final_bias

            layers.append(
                eqx.nn.Linear(sizes[i], sizes[i + 1], use_bias=use_bias, key=k_lin[i])
            )
        self.layers = layers

        # Per-layer LayerNorms on the OUTPUT dimension of each layer
        lns = []
        for lyr in self.layers[:-1]:
            out_dim = lyr.weight.shape[0]  # (out, in)
            # lns.append(eqx.nn.Identity()) # eqx.nn.LayerNorm(out_dim))
            lns.append(eqx.nn.LayerNorm(out_dim))
        lns.append(eqx.nn.Identity())  # no LN on final layer output (can change if you want)
        self.layernorms = lns

        # Per-layer Dropouts mirroring LayerNorms
        drops = []
        if self.p > 0.0:
            for _ in self.layers[:-1]:
                drops.append(eqx.nn.Dropout(p=self.p))
            drops.append(eqx.nn.Identity())  # no dropout on final layer output
        else:
            drops = [eqx.nn.Identity() for _ in self.layers]
        self.dropouts = drops

    def __call__(self, x, *, key: Optional[PRNGKeyArray] = None, inference: bool = False):
        # If we have real Dropout modules and a key, split it so masks differ per layer

        if self.p > 0. and key is not None:
            k_list = list(jr.split(key, len(self.layers)))
        else:
            k_list = [None] * len(self.layers)

        for i, (
            layer, 
            norm, 
            drop, 
            k_i
        ) in enumerate(
            zip(
                self.layers, 
                self.layernorms, 
                self.dropouts, 
                k_list
            )
        ):
            x = layer(x)
            x = norm(x)
            x = drop(x, key=k_i)
            if i != len(self.layers) - 1:
                x = self.activation(x)  # no hidden activation after last layer

        if self.final_activation is not None:
            x = self.final_activation(x)

        # If using covariance network and running in 
        # inference mode after training, ignore covariance
        if self.covariance_nn and inference:
            x, _ = jnp.split(x, [self.parameter_dim])

        return x


class LayerNorm1D(eqx.Module):
    gamma: jnp.ndarray
    beta: jnp.ndarray
    eps: float = eqx.static_field()

    def __init__(self, dim: int, *, eps: float = 1e-5):
        self.gamma = jnp.ones((dim,))
        self.beta = jnp.zeros((dim,))
        self.eps = eps

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        m = x.mean(axis=-1, keepdims=True)
        v = jnp.mean((x - m) ** 2, axis=-1, keepdims=True)
        xhat = (x - m) / jnp.sqrt(v + self.eps)
        return self.gamma * xhat + self.beta


# --- Residual block with LN + Dropout ---
class BasicBlockMLP(eqx.Module):
    ln1: LayerNorm1D
    ln2: LayerNorm1D
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    proj: eqx.nn.Linear | None
    drop1: eqx.nn.Dropout
    drop2: eqx.nn.Dropout
    act: Callable = eqx.static_field()

    def __init__(self, in_dim: int, out_dim: int, *, p_dropout: float, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.ln1 = LayerNorm1D(in_dim)
        self.fc1 = eqx.nn.Linear(in_dim, out_dim, key=k1)
        self.ln2 = LayerNorm1D(out_dim)
        self.fc2 = eqx.nn.Linear(out_dim, out_dim, key=k2)
        self.proj = None if in_dim == out_dim else eqx.nn.Linear(in_dim, out_dim, key=k3)
        self.drop1 = eqx.nn.Dropout(p_dropout)
        self.drop2 = eqx.nn.Dropout(p_dropout)
        self.act = jax.nn.tanh

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        # We’ll split keys for the two dropout sites inside the block.
        k1, k2 = (None, None) if key is None else jax.random.split(key, 2)

        y = self.fc1(self.ln1(x))
        y = self.act(y)
        y = self.drop1(y, key=k1)

        y = self.fc2(self.ln2(y))
        y = self.drop2(y, key=k2)

        res = x if self.proj is None else self.proj(x)
        return self.act(y + res)


# --- ResNet MLP with LN + Dropout ---
class ResNetMLP(eqx.Module):
    stem_ln: LayerNorm1D
    stem: eqx.nn.Linear
    stem_drop: eqx.nn.Dropout
    stages: tuple[tuple[BasicBlockMLP, ...], ...]
    head_ln: LayerNorm1D
    head: eqx.nn.Linear
    act: Callable = eqx.static_field()

    def __init__(
        self,
        in_features: int,            # 40
        num_outputs: int,            # e.g. classes or regression dims
        *,
        widths: Sequence[int] = (64, 128, 128),
        blocks_per_stage: Sequence[int] = (2, 2, 2),
        p_dropout: float = 0.1,      # dropout prob inside blocks & stem
        key: PRNGKeyArray,
    ):
        assert len(widths) == len(blocks_per_stage)
        n_blocks = sum(blocks_per_stage)
        keys = jax.random.split(key, 3 + n_blocks)  # stem, head, stemdrop, blocks...
        k_stem, k_head, k_dummy, *block_keys = keys  # k_dummy just keeps count simple

        self.stem_ln = LayerNorm1D(in_features)
        self.stem = eqx.nn.Linear(in_features, widths[0], key=k_stem)
        self.stem_drop = eqx.nn.Dropout(p_dropout)
        self.act = jax.nn.tanh

        stages = []
        in_dim = widths[0]
        k_iter = iter(block_keys)
        for out_dim, n in zip(widths, blocks_per_stage):
            stage = []
            # First block may change width
            stage.append(BasicBlockMLP(in_dim, out_dim, p_dropout=p_dropout, key=next(k_iter)))
            in_dim = out_dim
            # Remaining blocks keep width
            for _ in range(n - 1):
                stage.append(BasicBlockMLP(in_dim, in_dim, p_dropout=p_dropout, key=next(k_iter)))
            stages.append(tuple(stage))
        self.stages = tuple(stages)

        self.head_ln = LayerNorm1D(in_dim)
        self.head = eqx.nn.Linear(in_dim, num_outputs, key=k_head)

    def __call__(self, x: jnp.ndarray, key=None) -> jnp.ndarray:
        # x: (in_features,) e.g. (40,)
        # Split a key per block (+ stem dropout). If key=None, dropout raises only if train=True & p>0.
        if key is None:
            k_stem = None
            block_keys = [None] * sum(len(s) for s in self.stages)
        else:
            n_needed = 1 + sum(len(s) for s in self.stages)  # 1 for stem, rest per block
            keys = jax.random.split(key, n_needed)
            k_stem, *block_keys = keys

        y = self.stem(self.stem_ln(x))
        y = self.act(y)
        y = self.stem_drop(y, key=k_stem)

        for stage, k in zip(self.stages, block_keys):
            # Share the same k across blocks in a stage? Better: split per block.
            if k is None:
                subkeys = [None] * len(stage)
            else:
                subkeys = list(jax.random.split(k, len(stage)))
            for blk, bk in zip(stage, subkeys):
                y = blk(y, key=bk)

        y = self.head(self.head_ln(y))
        return y  # (num_outputs,)


class PatchEmbedding1D(eqx.Module):
    linear: eqx.nn.Linear | eqx.nn.Identity
    patch_size: int

    def __init__(
        self, 
        patch_size: int, 
        embed_dim: int, 
        *,
        key: PRNGKeyArray
    ):
        self.patch_size = patch_size
        self.linear = (
            eqx.nn.Linear(patch_size, embed_dim, key=key)
            if patch_size > 1
            else eqx.nn.Identity()
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (L,)  ->  tokens: (N, E)
        Pads to a multiple of patch_size if needed.
        """
        L = x.shape[0]
        pad = (-L) % self.patch_size

        if pad > 0:
            x = jnp.pad(x, (0, pad))

        # reshape into patches of length patch_size
        patches = einops.rearrange(
            x, "(n p) -> n p", p=self.patch_size
        )  # (N, patch_size)
        tokens = jax.vmap(self.linear)(patches)  # (N, E)

        return tokens


class AttentionBlock(eqx.Module):
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    drop1: eqx.nn.Dropout
    drop2: eqx.nn.Dropout

    def __init__(
        self, 
        embed_dim: int, 
        mlp_hidden: int, 
        num_heads: int, 
        dropout: float, 
        *,
        key: PRNGKeyArray
    ):
        k1, k2, k3 = jr.split(key, 3)
        self.ln1 = eqx.nn.LayerNorm(embed_dim)
        self.ln2 = eqx.nn.LayerNorm(embed_dim)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads, query_size=embed_dim, key=k1
        )
        self.fc1 = eqx.nn.Linear(embed_dim, mlp_hidden, key=k2)
        self.fc2 = eqx.nn.Linear(mlp_hidden, embed_dim, key=k3)
        self.drop1 = eqx.nn.Dropout(dropout)
        self.drop2 = eqx.nn.Dropout(dropout)

    def __call__(self, x: jnp.ndarray, *, key: PRNGKeyArray) -> jnp.ndarray:
        # x: (N, E)
        k1, k2 = jr.split(key, 2)

        y = jax.vmap(self.ln1)(x)
        
        x = x + self.attn(y, y, y)

        y = jax.vmap(self.ln2)(x)
        y = jax.vmap(self.fc1)(y)
        y = jax.nn.gelu(y)
        y = self.drop1(y, key=k1)
        y = jax.vmap(self.fc2)(y)
        y = self.drop2(y, key=k2)

        return x + y


class ViT(eqx.Module):
    patch: PatchEmbedding1D
    pos_emb: jnp.ndarray         # (N_max, E)
    blocks: List[AttentionBlock]
    drop: eqx.nn.Dropout
    head: eqx.nn.Sequential
    embed_dim: int
    parameter_dim: int
    covariance_nn: bool 

    def __init__(
        self,
        input_len: int = 40,
        output_dim: int = 5,
        patch_size: int = 1,          # divides 40 -> N=8 tokens
        embed_dim: int = 4,
        mlp_hidden: int = 16,
        num_heads: int = 2,
        num_layers: int = 1,
        dropout: float = 0.1,
        covariance_nn: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        k_patch, k_pos, k_blocks, k_head = jr.split(key, 4)
        
        self.patch = PatchEmbedding1D(patch_size, embed_dim, key=k_patch)

        # Maximum tokens after padding
        n_tokens = (input_len + (-input_len) % patch_size) // patch_size

        self.pos_emb = jr.normal(k_pos, (n_tokens, embed_dim)) * 0.02

        self.blocks = [
            AttentionBlock(
                embed_dim, mlp_hidden, num_heads, dropout, key=_key
            )
            for _key in jr.split(k_blocks, num_layers)
        ]
        self.drop = eqx.nn.Dropout(dropout)

        self.parameter_dim = output_dim
        if covariance_nn:
            # Cholesky dimension of the output covariance
            n_tril = output_dim * (output_dim + 1) // 2
            output_dim = output_dim + n_tril
            self.covariance_nn = covariance_nn
        else:
            self.covariance_nn = False

        self.head = eqx.nn.Sequential(
            [
                eqx.nn.LayerNorm(embed_dim), 
                eqx.nn.Linear(embed_dim, output_dim, key=k_head)
            ]
        )

        self.embed_dim = embed_dim

    def __call__(self, x: jnp.ndarray, *, key: PRNGKeyArray, inference: bool = False) -> jnp.ndarray:
        """x: (40,) -> y: (5,)"""

        # Patchify & embed
        tokens = self.patch(x) # (N, E)
        N = tokens.shape[0]
        tokens = tokens + self.pos_emb[:N] 
        tokens = self.drop(tokens, key=key)

        # Transformer layers
        keys = jr.split(key, len(self.blocks))
        for blk, k in zip(self.blocks, keys):
            tokens = blk(tokens, key=k)

        # Mean pool tokens (simple & stable for regression)
        rep = tokens.mean(axis=0) # (E,)

        x = self.head(rep)        # (5,)

        if self.covariance_nn and inference:
            x, _ = jnp.split(x, [self.parameter_dim])

        return x 


def trunc_init(weight: jax.Array, key: PRNGKeyArray) -> jax.Array:

    out, in_ = weight.shape
    stddev = jnp.sqrt(1. / in_)

    return stddev * jr.truncated_normal(key, shape=(out, in_), lower=-2., upper=2.)


def init_linear_weight(model, init_fn, key):

    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [
        x.weight
        for x in jax.tree.leaves(m, is_leaf=is_linear)
        if is_linear(x)
    ]
    weights = get_weights(model)

    new_weights = [
        init_fn(weight, subkey)
        for weight, subkey in zip(weights, jr.split(key, len(weights)))
    ]
    new_model = eqx.tree_at(get_weights, model, new_weights)

    return new_model


def loss(
    model: eqx.Module | eqx.nn.MLP, 
    x: Float[Array, "b x"], 
    y: Float[Array, "b y"], 
    key: PRNGKeyArray,
    *,
    precision: Optional[Float[Array, "y y"]] = None,
    covariance_nn: bool = False
) -> Scalar:

    if precision is None:
        precision = jnp.eye(y.shape[-1])

    if covariance_nn: 
        def fn(x, y, key):
            logits = model(x, key=key)
            L = gaussian_nll_from_raw(y, logits)
            return L
    else:
        def fn(x, y, key):
            y_ = model(x, key=key)

            dy = jnp.subtract(y_, y)

            l = jnp.linalg.multi_dot([dy, precision, dy.T])

            return l

    keys = jr.split(key, len(x))

    return jnp.mean(jax.vmap(fn)(x, y, keys))


def get_batch(
    D: Float[Array, "n x"], 
    Y: Float[Array, "n y"], 
    n: Optional[int], 
    key: Optional[PRNGKeyArray]
) -> Tuple[Float[Array, "b x"], Float[Array, "b y"]]:
    if n is not None and key is not None:
        idx = jr.choice(key, jnp.arange(D.shape[0]), (n,))
        return D[idx], Y[idx]
    else:
        return D, Y


@typecheck
@eqx.filter_jit
def evaluate(
    model: eqx.Module, 
    Xv: Float[Array, "n x"], 
    Yv: Float[Array, "n y"],
    key: PRNGKeyArray,
    *, 
    n_batch: Optional[int],
    precision: Optional[Float[Array, "y y"]] = None,
    sharding: Optional[jax.sharding.NamedSharding] = None,
    replicated_sharding: Optional[NamedSharding] = None,
    covariance_nn: bool = False
) -> Scalar:

    model = eqx.nn.inference_mode(model, True)

    x, y = get_batch(Xv, Yv, n=n_batch, key=key)

    if sharding is not None:
        x, y = eqx.filter_shard((x, y), sharding)

    if replicated_sharding is not None:
        model = eqx.filter_shard(model, replicated_sharding)

    l = loss(
        model, 
        x, y, 
        key=key, 
        precision=precision,
        covariance_nn=covariance_nn
    )

    return l


@typecheck
@eqx.filter_jit
def make_step(
    model: eqx.Module, 
    opt_state: optax.OptState,
    Xt: Float[Array, "n x"], 
    Yt: Float[Array, "n y"],
    key: PRNGKeyArray,
    opt: optax.GradientTransformation, 
    *, 
    n_batch: Optional[int] = None,
    precision: Optional[Float[Array, "y y"]] = None,
    sharding: Optional[jax.sharding.NamedSharding],
    replicated_sharding: Optional[NamedSharding],
    covariance_nn: bool = False
) -> Tuple[eqx.Module, optax.OptState, Scalar]:

    model = eqx.nn.inference_mode(model, False)

    x, y = get_batch(Xt, Yt, n=n_batch, key=key) 
    
    if sharding is not None:
        x, y = eqx.filter_shard((x, y), sharding)

    if replicated_sharding is not None:
        model, opt_state = eqx.filter_shard(
            (model, opt_state), replicated_sharding
        )

    grad_fn = eqx.filter_value_and_grad(
        partial(
            loss, 
            precision=precision,
            covariance_nn=covariance_nn
        )
    )

    loss_value, grads = grad_fn(model, x, y, key)

    # if l2_regularisation:
    #     l2 = sum(
    #         jax.tree.map(
    #             lambda w: jnp.sum(jnp.square(w)),
    #             jax.tree.leaves(eqx.filter(model, eqx.is_array))
    #         )
    #     )
    #     loss_value = l

    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    if replicated_sharding is not None:
        model, opt_state = eqx.filter_shard(
            (model, opt_state), replicated_sharding
        )

    return model, opt_state, loss_value


@typecheck
def fit_nn(
    key: PRNGKeyArray,
    model: eqx.Module, 
    train_data: Tuple[Float[Array, "n x"], Float[Array, "n y"]], 
    valid_data: Tuple[Float[Array, "n x"], Float[Array, "n y"]],
    opt: optax.GradientTransformation, 
    n_batch: Optional[int], 
    patience: Optional[int], 
    n_steps: int = 100_000, 
    valid_fraction: float = 0.1, 
    use_tqdm: bool = True,
    *,
    description: str = "Training NN",
    precision: Optional[Float[Array, "y y"]] = None,
    sharding: Optional[NamedSharding] = None,
    replicated_sharding: Optional[NamedSharding] = None,
) -> Tuple[eqx.Module, Float[np.ndarray, "l 2"]]:

    if exists(precision) and USE_PRECISION_NN:
        logger.info("Using Fisher precision for NN.")

    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    Xt, Yt = train_data
    Xv, Yv = valid_data

    if use_tqdm: 
        steps = trange(n_steps, desc=description, colour="magenta")
    else: 
        steps = range(n_steps)

    L = np.zeros((n_steps, 2))
    for step in steps:

        key_t, key_v = jr.split(jr.fold_in(key, step))

        model, opt_state, train_loss = make_step(
            model, 
            opt_state, 
            Xt, 
            Yt, 
            key_t,
            opt=opt, 
            n_batch=n_batch,
            precision=precision, 
            sharding=sharding,
            replicated_sharding=replicated_sharding,
            covariance_nn=COVARIANCE_NN
        )

        valid_loss = evaluate(
            model, 
            Xv, 
            Yv, 
            key_v, 
            n_batch=n_batch,
            precision=precision, 
            sharding=sharding,
            replicated_sharding=replicated_sharding,
            covariance_nn=COVARIANCE_NN
        )

        L[step] = train_loss, valid_loss

        if use_tqdm:
            if step > 0:
                steps.set_description_str(
                    "t={:.3E}, v={:.3E}, p={:04d}".format(
                        train_loss.item(), 
                        valid_loss.item(), 
                        int(step - np.argmin(L[:step, 1]))
                    )
                )

        if patience is not None:
            if (step > 0) and (step - np.argmin(L[:step, 1]) > patience):
                if use_tqdm:
                    steps.set_description_str("Stopped at {}".format(step))
                break

    return model, L[:step]


"""
    L-BFGS
"""

@typecheck
@eqx.filter_jit(donate="all-except-first")
def make_step_lbfgs(
    net: eqx.Module, 
    opt_state: optax.OptState, 
    X: Float[Array, "n d"], 
    P: Float[Array, "n p"], 
    *,
    opt: optax.GradientTransformation,
    precision: Optional[Float[Array, "p p"]] = None, 
    replicated_sharding: Optional[jax.sharding.NamedSharding] = None
) -> tuple[eqx.Module, optax.OptState, Scalar]:
    f = partial(loss, x=X, y=P, precision=precision)
    value_and_grad_fn = optax.value_and_grad_from_state(f)
    l, grad = value_and_grad_fn(net, state=opt_state)
    updates, opt_state = opt.update(
        grad, opt_state, net, value=l, grad=grad, value_fn=f 
    )
    net = eqx.apply_updates(net, updates)
    return net, opt_state, l 


def fit_nn_lbfgs(
    key: Key[jnp.ndarray, "..."], 
    model: eqx.Module, 
    train_data: Tuple[Float[Array, "n x"], Float[Array, "n y"]], 
    valid_fraction: float = 0.1, 
    valid_data: Sequence[Array] = None,
    batch_dataset: bool = True,
    *,
    precision: Optional[Float[Array, "y y"]] = None,
    sharding: Optional[NamedSharding] = None,
    replicated_sharding: Optional[NamedSharding] = None,
) -> Tuple[eqx.Module, Float[np.ndarray, "l 2"]]:

    D, Y = train_data

    y0, static = eqx.partition(model, eqx.is_array)
    
    # Standardise before PCA (don't get tricked by high variance due to units)
    # X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0) # NOTE: already standardised 

    # Fit whitening-PCA to compressed simulations
    # pca = PCA(num_components=D.shape[-1]) 
    # D = (D - jnp.mean(D, axis=0)) / jnp.std(D, axis=0)
    # pca.fit(D) # Fit on fiducial data?
    # D = pca.transform(D)

    # D, preprocess_fn = get_preprocess_fn(D, use_pca=config.use_pca)

    @eqx.filter_jit
    def f(y, args):
        # Dataset to fit network to
        D, Y = args
        # Combine iteration parameters and architecture
        model = eqx.combine(y, static)
        return loss(model, D, Y, precision=precision)

    # BFGS optimisation
    res = minimise(
        f,
        BFGS(rtol=1e-6, atol=1e-6, norm=rms_norm),
        y0=y0,
        max_steps=1_000_000,
        args=(D, Y)
    )

    # Put solution parameters into model
    model = eqx.combine(res.value, static)

    L = np.zeros((1, 2))

    # model = lambda d: pca.transform(model(d))

    return model, L


def get_nn_process_fns(
    X: Float[Array, "n d"], 
    Y: Float[Array, "n p"],
    data_process_type_nn: Literal["d", "p", "dp"] | None
) -> Processor:

    assert data_process_type_nn in ["d", "p", "dp", None], (
        "data_process_type_nn={}".format(data_process_type_nn)
    )

    if data_process_type_nn == "d":
        processor = Processor(X, None)

    if data_process_type_nn == "p":
        processor = Processor(None, Y)

    if data_process_type_nn == "dp":
        processor = Processor(X, Y)
    
    return processor


@typecheck
def get_nn_compressor(
    key: PRNGKeyArray, 
    net: eqx.nn.MLP | eqx.Module,
    config: ConfigDict,
    dataset: Any, #Dataset, 
    *, 
    train: bool = True,
    lbfgs: bool = False, 
    results_dir: str, 
    filename: str = "nn"
) -> eqx.Module:
    """
        Train neural network compression function
        - Optionally use parameter covariance for chi2 loss
    """

    class CompressorWrapper(eqx.Module):
        # Wrap a neural network to accept, but disregard, a parameter input
        fn: Callable
        processor: Optional[Processor]
        def __init__(self, fn, processor):
            self.fn = fn
            self.processor = processor
        def __call__(self, d, *args, **kwargs):
            if self.processor is not None:
                d = self.processor.forward_x(d)
            # Disregard physics-parameter args / kwargs,
            # ensure that covariance_nn's are inference mode
            y_ = self.fn(d, inference=True) 
            if self.processor is not None:
                y_ = self.processor.reverse_y(y_)
            return y_
 

    description = "Fitting NN [{}]".format(dataset.name)

    # NOTE: must use same seed for shuffling -> parametrising `Processor`
    data_key = jr.key(config.seed) 
    train_key = jr.key(int(time.time())) # Force training to be random

    if USE_PRECISION_NN:
        precision = jnp.linalg.inv(dataset.Finv) # In reality this varies with parameters

        assert DATA_PROCESS_TYPE_NN not in ["dp", "p"], (
            "CANNOT USE DATA_PROCESS_TYPE_NN={} with USE_PRECISION_NN=True".format(DATA_PROCESS_TYPE_NN)
        )

        print("USING PRECISION FOR NN.")
    else:
        precision = None 

        print("NOT USING PRECISION FOR NN.")

    logger.info("DATA_PROCESS_TYPE_NN: {}".format(DATA_PROCESS_TYPE_NN))

    print("D, Y:", dataset.data.shape, dataset.parameters.shape)

    nn_path = os.path.join(results_dir, filename + ".eqx")
    processor_path = os.path.join(results_dir, filename + "_processor" + ".eqx")

    if train:
        # Preprocess
        # valid_fraction = 0.2
        n_s = dataset.data.shape[0]

        X, Y = dataset.data, dataset.parameters

        ix = jr.permutation(data_key, jnp.arange(n_s))
        n_valid = int(config.nn.train.valid_fraction * n_s) 
        Xt, Xv = jnp.split(X[ix], [n_s - n_valid]) # Shuffle and split
        Yt, Yv = jnp.split(Y[ix], [n_s - n_valid]) # Shuffle and split, same ix

        processor = get_nn_process_fns(Xt, Yt, data_process_type_nn=DATA_PROCESS_TYPE_NN) 

        train_data = (processor.forward_x(Xt), processor.forward_y(Yt)) 
        valid_data = (processor.forward_x(Xv), processor.forward_y(Yv)) 

        # Train or reload the neural network
        if lbfgs:
            net, losses = fit_nn_lbfgs(
                train_key, 
                net, 
                train_data=train_data,
                precision=precision
            )
        else:
            if config.nn.train.opt == "adamw":
                # filter_spec = jax.tree.map(lambda x: x.ndim != 1, eqx.filter(net, eqx.is_array))

                def finder(x):
                    if isinstance(x, jax.Array):
                        if x.ndim > 1: # No scalars or biases, just weights
                            return True
                    return False

                filter_spec = jax.tree.map(lambda x: finder(x), net)

                opt = optax.adamw(
                    config.nn.train.lr, 
                    mask=filter_spec, 
                    weight_decay=config.nn.train.weight_decay
                )
            else:
                opt = getattr(optax, config.nn.train.opt)(config.nn.train.lr)

            # NOTE: problematic for layernorm?
            if NN_CLIP_NORM:
                opt = optax.chain(optax.clip_by_global_norm(1.), opt)

            net, losses = fit_nn(
                train_key, 
                net, 
                opt=opt, 
                train_data=train_data,
                valid_data=valid_data,
                precision=precision, 
                n_batch=config.nn.train.n_batch,
                n_steps=config.nn.train.n_steps,
                patience=config.nn.train.patience,
                valid_fraction=config.nn.train.valid_fraction,
                description=description
            )

        plt.figure()
        plt.title("Min. Lt/Lv: {:.3E}/{:.3E}".format(np.max(-losses[:, 0]), np.max(-losses[:, 1])))
        plt.loglog(-losses, color="red" if dataset.name == "tails" else "blue")
        plt.legend(frameon=False)
        plt.savefig(os.path.join(results_dir, "losses_{}.png".format(filename)))
        plt.close()

        net = eqx.nn.inference_mode(net, True)

        # Always ensure it is this network that is saved!
        net = CompressorWrapper(net, processor)

        eqx.tree_serialise_leaves(nn_path, net)

        eqx.tree_serialise_leaves(processor_path, processor)
    else:
        # Processors for NNs always work on uncompressed data shapes
        Xt = jnp.ones_like(dataset.data)
        Yt = jnp.ones_like(dataset.parameters)

        processor = get_nn_process_fns(Xt, Yt, data_process_type_nn=DATA_PROCESS_TYPE_NN) 

        net = eqx.tree_deserialise_leaves(nn_path, net)

        processor = eqx.tree_deserialise_leaves(processor_path, processor)

        net = CompressorWrapper(net, processor)

        net = eqx.nn.inference_mode(net, True)

    return net


"""
    Run NNs
"""

from typing import Optional, Callable
import jax
import jax.random as jr
import equinox as eqx
from jaxtyping import Array, jaxtyped
from beartype import beartype as typechecker
import numpy as np

import os
TYPECHECK = True if os.environ.get("TYPECHECK", "").lower() in ("1", "true") else False
if TYPECHECK:
    typecheck = jaxtyped(typechecker=typechecker)
else:
    typecheck = lambda x: x

LossFunction = Optional[Callable[[eqx.Module, Array, Array], Array]]

"""
    Tools for compression with neural networks.
    - train a user-defined `eqx.Module` network that compresses a datavector
      to a model-dimensional summary, by minimising a MSE loss.
"""

if __name__ == "__main__":
    import os
    import time
    import datetime
    import matplotlib.pyplot as plt
    from chainconsumer import Chain, ChainConsumer, Truth

    from configs.args import get_cumulants_sbi_args
    from . import get_compression_fn
    from utils import get_datasets, get_results_dir, make_df

    t0 = time.time()

    args = get_cumulants_sbi_args()

    print("TIME:", datetime.datetime.now().strftime("%H:%M %d-%m-%y"))
    print("SEED:", args.seed)
    print("MOMENTS:", args.order_idx)
    print("LINEARISED:", args.linearised)

    """
        Config
    """

    config, cumulants_dataset, datasets = get_datasets(args) 

    results_dir = get_results_dir(config, args)

    key = jr.key(config.seed)

    (
        model_key, train_key, key_prior, 
        key_datavector, key_state, key_sample
    ) = jr.split(key, 6)

    assert config.compression in ["nn", "nn-lbfgs", "imnn", "ensemble-nn"]

    compression_fn = get_compression_fn(
        key, config, cumulants_dataset.data, results_dir=results_dir
    )

    # NOTE: parameters ignored here
    X = jax.vmap(compression_fn)(
        cumulants_dataset.data.data, 
        cumulants_dataset.data.parameters
    )

    # Corner plot of summaries
    c = ChainConsumer()
    c.add_chain(
        Chain(
            samples=make_df(
                cumulants_dataset.data.parameters, 
                parameter_strings=cumulants_dataset.get_parameter_strings()
            ), 
            name="Params", 
            color="k", 
            plot_cloud=True, 
            plot_contour=False
        )
    )
    c.add_chain(
        Chain(
            samples=make_df(X, parameter_strings=cumulants_dataset.get_parameter_strings()), 
            name="Summaries", 
            color="red" if cumulants_dataset.data.name == "tails" else "blue", 
            plot_cloud=True, 
            plot_contour=False
        )
    )
    c.add_truth(
        Truth(location=dict(zip(cumulants_dataset.get_parameter_strings(), cumulants_dataset.data.alpha)), name=r"$\pi^0$")
    )

    fig = c.plotter.plot()
    plt.savefig(os.path.join(results_dir, "nn_params.png")) 
    plt.close()

    # Scatter plot
    fig, axs = plt.subplots(1, cumulants_dataset.data.alpha.size, figsize=(2. + 2. * cumulants_dataset.data.alpha.size, 2.5))
    for p, ax in enumerate(axs):
        ax.scatter(
            cumulants_dataset.data.parameters[:, p], 
            X[:, p], 
            s=0.1, 
            color="red" if cumulants_dataset.data.name == "tails" else "blue"
        )
        ax.axline((0, 0), slope=1., color="k", linestyle="--")
        ax.set_xlim(cumulants_dataset.data.lower[p], cumulants_dataset.data.upper[p])
        ax.set_ylim(cumulants_dataset.data.lower[p], cumulants_dataset.data.upper[p])
        ax.set_xlabel(cumulants_dataset.get_parameter_strings()[p])
        ax.set_ylabel(cumulants_dataset.get_parameter_strings()[p] + "'")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "nn_scatter.png"), bbox_inches="tight")
    plt.close()

    # NOTE: parameters ignored here
    X = jax.vmap(compression_fn, in_axes=(0, None))(
        cumulants_dataset.data.fiducial_data, 
        cumulants_dataset.data.alpha
    )

    # Corner plot of summaries
    c = ChainConsumer()
    # c.add_chain(
    #     Chain(
    #         samples=make_df(
    #             cumulants_dataset.data.parameters, 
    #             parameter_strings=cumulants_dataset.get_parameter_strings()
    #         ), 
    #         name="Params", 
    #         color="blue", 
    #         plot_cloud=True, 
    #         plot_contour=False
    #     )
    # )
    c.add_chain(
        Chain(
            samples=make_df(X, parameter_strings=cumulants_dataset.get_parameter_strings()), 
            name="Summaries", 
            # color="red", 
            color="red" if cumulants_dataset.data.name == "tails" else "blue",
            plot_cloud=True, 
            plot_contour=True
        )
    )
    # c.add_chain(
    #     Chain.from_covariance(
    #         cumulants_dataset.data.alpha,
    #         cumulants_dataset.data.Finv,
    #         columns=cumulants_dataset.get_parameter_strings(),
    #         name=r"$F_{\Sigma^{-1}}$",
    #         color="k",
    #         linestyle=":",
    #         shade_alpha=0.
    #     )
    # )
    c.add_truth(
        Truth(
            location=dict(
                zip(cumulants_dataset.get_parameter_strings(), 
                    cumulants_dataset.data.alpha)),
            name=r"$\pi^0$"
        )
    )
    fisher_samples = np.random.multivariate_normal(
        cumulants_dataset.data.alpha, cumulants_dataset.data.Finv, (800_000,) 
    ) 
    fisher_samples_log_prob = jax.scipy.stats.multivariate_normal.logpdf(
        fisher_samples, 
        cumulants_dataset.data.alpha, 
        cumulants_dataset.data.Finv
    )

    def cut_samples(samples, lower, upper):
        return samples[np.all((samples >= lower) & (samples <= upper), axis=1)]

    fisher_df = make_df(
        cut_samples(fisher_samples, cumulants_dataset.data.lower, cumulants_dataset.data.upper),
        # samples_log_prob, 
        parameter_strings=cumulants_dataset.get_parameter_strings()
    )
    c.add_chain(
        Chain(
            samples=fisher_df,
            name=r"$F_{\Sigma^{-1}}$" + " {}".format("clipped"),
            color="k",
            linestyle=":",
            shade_alpha=0.
        )
    )

    fig = c.plotter.plot()
    plt.savefig(os.path.join(results_dir, "nn_params_fiducial.png")) 
    plt.close()

    results_dir = get_results_dir(config, args)

   # def stop_grad(a):
    #     return jax.lax.stop_gradient(a)

    # class ScalerWrapper(eqx.Module):
    #     # Module to wrap input and output of neural network, this gets saved and loaded with it
    #     # - wraps whatever network e.g. ensemble
    #     net: eqx.Module
    #     forward_scaler_d: Optional[Callable]
    #     forward_scaler_p: Optional[Callable]
    #     reverse_scaler_p: Optional[Callable]

    #     def __init__(self, net, forward_scaler_d=None, forward_scaler_p=None, reverse_scaler_p=None):
    #         self.net = net
    #         self.forward_scaler_d = forward_scaler_d
    #         self.forward_scaler_p = forward_scaler_p
    #         self.reverse_scaler_p = reverse_scaler_p

    #     def __call__(self, d, *args, **kwargs):

    #         if self.forward_scaler_d is not None:
    #             d_ = self.forward_scaler_d(d)
    #         else:
    #             d_ = d
            
    #         y_ = self.net(d_, *args, **kwargs)

    #         if self.reverse_scaler_p is not None:
    #             assert self.forward_scaler_p is not None
    #             y_ = self.reverse_scaler_p(y_)

    #         return y_