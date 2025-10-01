from typing import Tuple, Callable, Optional, Sequence
import jax
import jax.numpy as jnp
import jax.random as jr 
import equinox as eqx
import optax
from jaxtyping import Array, PyTree, Float, Scalar, jaxtyped, PRNGKeyArray
from beartype import beartype as typechecker

# from ._imnn import get_F

typecheck = jaxtyped(typechecker=typechecker)

OptState = PyTree
GradientTransformation = optax.GradientTransformation


def regularise_C_f(C_f: Float[Array, "p p"], C_f_inv: Float[Array, "p p"]) -> Scalar:
    """
        The Frobenius Norm of a matrix is defined as the square 
        root of the sum of the squares of the elements of the matrix. 
        ||A||_F = tr(sqrt(AA^T))
    """
    I = jnp.eye(C_f.shape[-1])
    # return jnp.linalg.norm(C_f - I) + jnp.linalg.norm(C_f_inv - I)
    # a, b = jnp.linalg.slogdet(C_f)
    # return a * b

    # return 0.5 * jnp.add(
    #     jnp.linalg.norm(jnp.subtract(C_f, I), ord="fro"),
    #     jnp.linalg.norm(jnp.subtract(C_f_inv, I), ord="fro")
    # )

    return jnp.linalg.norm(jnp.subtract(C_f, I), ord="fro")


def logdet(F: Float[Array, "p p"]) -> Scalar:
    # Numpy docs => det = logdet[0] * exp(logdet[1])
    logdet = jnp.linalg.slogdet(F)
    return logdet[0] * logdet[1]
     

def get_alpha(f: float, eps: float) -> Scalar:
    # eps = how close to 1 the det. of C_f, C_f_inv should be
    # f = scalar for covariance regularisation strength
    return -jnp.log(eps * (f - 1.) + eps ** 2. / (1. + eps)) / eps


def get_r(covariance_reg: Scalar, f: float = 10., eps: float = 0.1) -> Scalar:
    alpha = get_alpha(f, eps)
    return f * covariance_reg / (covariance_reg + jnp.exp(-alpha * covariance_reg))




def filter_value_and_jacfwd(f: eqx.Module, x: Array) -> Tuple[Array, Array]:
    basis = jnp.eye(x.size, dtype=x.dtype)
    y, J = eqx.filter_vmap(
        partial(eqx.filter_jvp, f, (x,)), out_axes=(None, 1)
    )((basis,))
    return y, J 


@typecheck
def get_f_d_alpha(
    net: eqx.Module, 
    d_0: Float[Array, "d"], 
    d_0_derivative: Float[Array, "p d"]
) -> Tuple[
    Float[Array, "p"], 
    Float[Array, "p p"]
]:

    x, J_x_d = filter_value_and_jacfwd(net, d_0)

    # Chain rule d(summaries)/d(alpha) = d(summaries)/d(data) * d(data)/d(alpha)
    # (n_summaries, n_data) * (n_parameters, n_data) -> (n_summaries, n_parameters)
    mu_f_alpha = jnp.einsum("ij, kj -> ik", J_x_d, d_0_derivative) # Eq. 4.3

    return x, mu_f_alpha # = mu_f(x),alpha


@typecheck
def get_summaries_covariance(
    x: Float[Array, "n p"]
) -> Tuple[
    Float[Array, "p p"], 
    Float[Array, "p p"]
]:
    n_data, data_dim = x.shape
    C_f = jnp.cov(x, rowvar=False)
    if x.shape[-1] == 1:
        C_f = jnp.array([[C_f]]) # Deal with single-summary compressions
    C_f_inv = jnp.linalg.inv(C_f)
    H = (n_data - 2. - data_dim) / (n_data - 1.) # Hartlap factor
    return C_f, H * C_f_inv


@typecheck
@eqx.filter_jit
def get_F(
    fiducials: Float[Array, "n d"], 
    net: eqx.Module, 
    fiducials_and_derivatives: Tuple[Float[Array, "n_d d"], Float[Array, "n_d p d"]],
) -> Tuple[
    Float[Array, "p p"], 
    Tuple[
        Float[Array, "n p"], 
        Float[Array, "p p"], 
        Float[Array, "p p"], 
        Float[Array, "p p"]
    ]
]:
    d_0, dd_0 = fiducials_and_derivatives

    # Get the derivatives of the fiducial summaries w.r.t. summaries
    _get_mu_f_alpha = eqx.filter_vmap(partial(get_f_d_alpha, net))

    x0, mu_f_alpha = _get_mu_f_alpha(d_0, dd_0) # same x, just less of them?

    mu_f_alpha_mean = jnp.mean(mu_f_alpha, axis=0)

    x = jax.vmap(net)(fiducials) # x0 are first 500 fiducial summaries

    # Stack all the summaries we have to calculate F (this function NEVER calculates via latins)
    C_f, C_f_inv = get_summaries_covariance(jnp.concatenate([x, x0]))

    # (n_summaries, n_parameters) * (n_summaries, n_summaries) * (n_summaries, n_parameters) -> (n_parameters, n_parameters)
    F = jnp.einsum(
        "pq, pr, rs -> qs", 
        mu_f_alpha_mean, 
        C_f_inv, # Already Hartlap'd
        mu_f_alpha_mean
    )
    return F, (x, mu_f_alpha_mean, C_f, C_f_inv)


@typecheck
@eqx.filter_jit
def get_x(
    alpha: Float[Array, "p"], 
    fiducials: Float[Array, "n d"], 
    d: Float[Array, "#N p"], # Data to compress
    net: eqx.Module,
    fiducials_and_derivatives: Tuple[Float[Array, "n_d d"], Float[Array, "n_d p d"]],
) -> Tuple[Float[Array, "n p"], Float[Array, "n p"]]:

    F, (x0, mu_f_alpha_mean, C_f, C_f_inv) = get_F(
        fiducials, net, fiducials_and_derivatives
    )
    mu = x0.mean(axis=0)
    Finv = jnp.linalg.inv(F) # survey scaling

    # Summaries of data (not necessarily fiducials) with validation network  
    x_d = jax.vmap(net)(d)

    # Derive MLE of arbitrary d by score compression of Gaussian IMNN likelihood
    mle = lambda x: jnp.einsum(
        "pq, rq, rs, s -> p", 
        Finv,
        mu_f_alpha_mean, 
        C_f_inv, 
        x - mu
    )
    s_x_d = alpha + jax.vmap(mle)(x_d)

    # qMLEs and summaries
    return s_x_d, x_d # Score compressed summaries of IMNN Gaussian likelihood and IMNN outputs


@typecheck
@eqx.filter_jit
def loss_fn(
    net: eqx.Module, 
    d0: Float[Array, "n d"], 
    fiducials_and_derivatives: Tuple[
        Float[Array, "n_d d"], 
        Float[Array, "n_d p d"]
    ], 
    f: float = 10., 
    eps: float = 0.1,
    *,
    weight_regularise: bool = False
) -> Tuple[
    Scalar, 
    Tuple[
        Scalar, 
        Scalar, 
        Float[Array, "p p"], 
        Float[Array, "p p"], 
    ]
]:
    """
        Given network params and physics parameters, simulate
        a dataset {d}, compress to {x} and calculate the Fisher
        and covariance matrices, the sum of determinants of which 
        are the loss function.
    """
    # Survey scale Finv later
    F, (_, _, C_f, C_f_inv) = get_F(
        d0, net, fiducials_and_derivatives
    )

    # Maximise Fisher information, summaries covariance -> identity
    L_F = logdet(F) 
    L_C = regularise_C_f(C_f, C_f_inv)
    
    # Secondary loss term to scale 'x', Fisher info alone being invariant to linear rescaling of x
    L = -L_F + get_r(L_C, f=f, eps=eps) * L_C 

    if weight_regularise:
        w2 = sum(
            jax.tree.map(
                lambda w: jnp.sum(jnp.square(w)), 
                jax.tree.leaves(eqx.filter(net, eqx.is_array))
            )
        )
        L = L + w2

    # Return det(C) for plotting here?
    return L, (L_F, L_C, C_f, C_f_inv)


@eqx.filter_jit
def update(
    net: eqx.Module, 
    grads: eqx.Module, 
    opt_state: OptState, 
    optimizer: GradientTransformation
) -> Tuple[eqx.Module, OptState]:
    updates, opt_state = optimizer.update(grads, opt_state, net) 
    return eqx.apply_updates(net, updates), opt_state 



import os, json
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
import equinox as eqx
import numpy as np 


def get_sharding():
    n_devices = len(jax.local_devices())
    print(f"Running on {n_devices} devices: \n\t{jax.local_devices()}")

    use_sharding = n_devices > 1
    # Sharding mesh: speed and allow training on high resolution?
    if use_sharding:
        # Split array evenly across data dimensions, this reshapes automatically
        mesh = Mesh(jax.devices(), ('x',))
        sharding = NamedSharding(mesh, P('x'))
        print(f"Sharding:\n {sharding}")
    else:
        sharding = None


def count_params(net):
    return sum(
        x.size for x in jax.tree_util.tree_leaves(net)
        if eqx.is_array(x)
    )


def scale_fn(x): 
    x = 1. / x 
    y = jnp.log(x + jnp.sqrt(jnp.square(x) + 1.))
    return y


def log_scale_pdfs(scale_fn, fiducials, latin_pdfs, derivatives):
    """ Log-scale data, returning adjusted derivatives. """
    print("nans?", ~jnp.all(jnp.isfinite(fiducials)), ~jnp.all(jnp.isfinite(latin_pdfs)))
    
    # Derivatives of log(d_fiducial) w.r.t. d_fiducial:
    # > d(log(pdf))/d(alpha) = d(log(pdf))/d(pdf) * d(pdf)/d(alpha)
    # Assuming the derivatives supplied by Quijote are run at the first 500 fiducials  
    dfdd = jax.vmap(jax.jacfwd(scale_fn))(fiducials[:len(derivatives)])

    print("log d's / d's", dfdd.shape, derivatives.shape)

    adjusted_derivatives = jnp.einsum("nij, nkj -> nki", dfdd, derivatives)

    print("derivatives", derivatives.shape)
    return scale_fn(fiducials), scale_fn(latin_pdfs), adjusted_derivatives


import abc
import jax.random as jr
from jaxtyping import Key, Array

class _AbstractDataLoader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, data, targets, *, key):
        pass

    def __iter__(self):
        raise RuntimeError("Use `.loop` to iterate over the data loader.")

    @abc.abstractmethod
    def loop(self, batch_size):
        pass


class _InMemoryDataLoader(_AbstractDataLoader):
    def __init__(
        self, 
        simulations: Array, 
        parameters: Array = None, 
        *, 
        key: Key
    ): 
        self.simulations = simulations 
        self.parameters = parameters 
        self.key = key

    def n_batches(self, batch_size):
        return max(int(self.simulations.shape[0] / batch_size), 1)

    def loop(self, batch_size: int):
        # Loop through dataset, batching, while organising data for NPE or NLE
        dataset_size = self.simulations.shape[0]
        one_batch = batch_size >= dataset_size
        key = self.key
        indices = jnp.arange(dataset_size)
        while True:
            # Yield whole dataset if batch size is larger than dataset size
            if one_batch:
                out = self.simulations
                if self.parameters is not None:
                    out = (out, self.parameters)
                yield out
            else:
                key, subkey = jr.split(key)
                perm = jr.permutation(subkey, indices)
                start = 0
                end = batch_size
                while end < dataset_size:
                    batch_perm = perm[start:end]
                    out = self.simulations[batch_perm]
                    if self.parameters is not None:
                        out = (out, self.parameters[batch_perm])
                    yield out
                    start = end
                    end = start + batch_size


def get_dataloaders(
    key, data_tuple_train, data_tuple_valid
):
    Xt = data_tuple_train
    Xv = data_tuple_valid
    return (
        _InMemoryDataLoader(Xt, key=key), 
        _InMemoryDataLoader(Xv, key=key)
    )


def get_fiducials_derivatives_dataloaders(
    key, fids_and_dd_train, fids_and_dd_valid
):
    ft, ddt = fids_and_dd_train 
    fv, ddv = fids_and_dd_valid 
    return (
        _InMemoryDataLoader(ft, ddt, key=key), 
        _InMemoryDataLoader(fv, ddv, key=key)
    )


def split_fiducials_and_derivatives(fiducials, derivatives, split):
    """
        Returns matched fids/derivatives in first two tuples
        and train/valid fiducial pdfs.
    """
    n_pdfs, *_ = fiducials.shape
    n_derivatives, *_ = derivatives.shape

    # Need dataloader to return (d, d(data)/dalpha, alpha)
    # though during training alpha is always alpha^0
    n_train = int(split * n_pdfs) - n_derivatives # first 500 pdfs removed 
    n_valid = n_pdfs - n_train

    # Split derivatives (same as in dataloaders)
    # and PDFs (0 -> 500), where first 500 given to derivatives dataloaders
    # Dataloaders of fiducial pdfs for epoch, don't use first 500 pdfs belonging to derivatives
    dd_train, dd_valid = jnp.split(
        derivatives, [int(split * n_derivatives)]
    )
    fiducials_train, fiducials_valid = jnp.split(
        fiducials[:n_derivatives], [int(split * n_derivatives)]
    )
    _d0_train, _d0_valid = jnp.split(
        fiducials[n_derivatives:], [split * (n_pdfs - n_derivatives)]
    )
    return (
        (dd_train, dd_valid),
        (fiducials_train, fiducials_valid),
        (_d0_train, _d0_valid),
    )


from functools import partial
from tqdm.auto import trange


def train_imnn(
    key, 
    net,
    dataset, 
    n_epochs=100_000, 
    batch_size=1000, 
    min_train_epochs=100, 
    split=0.8,
    patience=100,
    weight_regularise=True
):

    key, key_data, key_dd = jr.split(key, 3)

    n_derivatives = dataset.derivatives.shape[0]
    n_s = dataset.fiducial_data.shape[0] - 500 # Test set size

    batch_size_dd = batch_size // 2

    optimizer = optax.adam(1e-3)

    # Initialise model + optimiser
    opt_state = optimizer.init(eqx.filter(net, eqx.is_inexact_array))

    grad_fn = eqx.filter_value_and_grad(
        partial(loss_fn, weight_regularise=weight_regularise), has_aux=True
    ) # NOTE: changed covariance norm to just covaraince (not precision too)

    _d0_train, _d0_valid = jnp.split(
        dataset.fiducial_data[:n_s][n_derivatives:], [int(split * (n_s - n_derivatives))]
    )

    fiducials_train, fiducials_valid = jnp.split(
        dataset.fiducial_data[:n_s][:n_derivatives], [int(split * n_derivatives)]
    )

    dd_train, dd_valid = jnp.split(
        dataset.derivatives, [int(split * n_derivatives)]
    )
    fids_dd_dl_train, fids_dd_dl_valid = get_fiducials_derivatives_dataloaders(
        key_dd, (fiducials_train, dd_train), (fiducials_valid, dd_valid)
    )

    # Dataloaders of fiducials for epoch, not using first N belonging to derivatives
    train_dl, valid_dl = get_dataloaders(key_data, _d0_train, _d0_valid)

    # Fisher determinant + summary covariance metric
    metrics = np.zeros((n_epochs, 2))

    # Early stopping stuff
    best_F = -jnp.inf 

    # Patience counter
    counter = 0

    with trange(n_epochs) as epochs:
        for e in epochs:
            # Epoch metrics
            L_steps_train, L_steps_valid = [], [] 

            # Train
            for d0_train, fdd_train, n in zip(
                train_dl.loop(batch_size), 
                fids_dd_dl_train.loop(batch_size_dd), 
                range(train_dl.n_batches(batch_size))
            ):
                # d(-|F(x)| + |C(x)|)/d(phi)
                (L, (L_F, L_C, *_)), dLdp = grad_fn(
                    net, 
                    d0=d0_train, 
                    fiducials_and_derivatives=fdd_train
                )
                net, opt_state = update(net, dLdp, opt_state, optimizer)

                L_steps_train.append((L_F, L_C))

            # Validate
            for d0_valid, fdd_valid, n in zip(
                valid_dl.loop(batch_size), 
                fids_dd_dl_valid.loop(batch_size_dd), 
                range(valid_dl.n_batches(batch_size))
            ):
                L_valid, (L_valid_F, L_valid_C, *_) = loss_fn(
                    net, 
                    d0=d0_valid, 
                    fiducials_and_derivatives=fdd_valid, 
                    weight_regularise=weight_regularise
                )

                L_steps_valid.append((L_valid_F, L_valid_C))

            # Store metrics
            epoch_F, epoch_C = jnp.asarray(L_steps_train).mean(axis=0)
            epoch_valid_F, epoch_valid_C = jnp.asarray(L_steps_valid).mean(axis=0)

            metrics[e] = (epoch_F, epoch_valid_F)

            epochs.set_description(
                f"\re={e:04d}" +
                f" |F|={epoch_F:.3E}" + 
                f" |F_v|={epoch_valid_F:.3E}" + 
                f" |F_b|={best_F:.3E}" + 
                f" c={patience - counter}"
            )

            # Test patience: note minimising -|F| so compare |F| from validation
            if e > min_train_epochs:
                if epoch_valid_F > best_F:
                    best_F = epoch_valid_F 
                    counter = 0
                else:
                    counter = counter + 1
                    if counter >= patience:
                        epochs.set_description(
                            f"\nStopping training at epoch {e}, |F_t|={epoch_F:.3E}, |F_v|={epoch_valid_F:.3E}."
                        )
                        break

    F_imnn, _ = get_F(d0_valid, net, fdd_valid)

    return net, F_imnn, metrics[:e]


class ExtraMLP(eqx.Module):
    layers: list[eqx.nn.Linear]
    dropouts: list[eqx.nn.Dropout | eqx.nn.Identity]
    layernorms: list[eqx.nn.LayerNorm | eqx.nn.Identity]
    activation: Callable
    final_activation: Optional[Callable]
    p: float

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_sizes: Sequence[int],
        *,
        p: float = 0.0,
        activation: Callable = jax.nn.gelu,
        final_activation: Optional[Callable] = None,
        key: PRNGKeyArray,
        use_bias: bool = True,
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
        self.p = float(p)
        self.activation = activation
        self.final_activation = final_activation

        # Build Linear stack according to sizes
        if hidden_sizes == [0] or len(list(hidden_sizes)) == 0:
            sizes = [in_size, out_size]
        else:
            sizes = [in_size, *hidden_sizes, out_size]
        n_layers = len(sizes) - 1
        k_lin = jr.split(key, n_layers)

        layers = []
        for i in range(n_layers):
            layers.append(
                eqx.nn.Linear(
                    sizes[i], sizes[i + 1], use_bias=use_bias, key=k_lin[i]
                )
            )
        self.layers = layers

        # Per-layer LayerNorms on the OUTPUT dimension of each layer
        lns = []
        for lyr in self.layers[:-1]:
            out_dim = lyr.weight.shape[0]  # (out, in)
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

    def __call__(self, x, *, key: Optional[PRNGKeyArray] = None):
        # If we have real Dropout modules and a key, split it so masks differ per layer
        if self.p > 0.0 and key is not None:
            k_list = list(jr.split(key, len(self.layers)))
        else:
            k_list = [None] * len(self.layers)

        for i, (layer, norm, drop, k_i) in enumerate(zip(self.layers, self.layernorms, self.dropouts, k_list)):
            x = layer(x)
            x = norm(x)
            x = drop(x, key=k_i)
            if i != len(self.layers) - 1:
                x = self.activation(x)  # no hidden activation after last layer

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x


if __name__ == "__main__":
    import os
    import time
    import datetime
    import matplotlib.pyplot as plt
    from chainconsumer import Chain, ChainConsumer, Truth

    from configs.args import get_cumulants_sbi_args
    from data.common import get_compression_fn
    from utils import get_datasets, get_results_dir, make_df, plot_cumulants

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

    plot_cumulants(
        args, 
        config, 
        cumulants_dataset.data.fiducial_data, 
        results_dir=results_dir
    )

    key = jr.key(config.seed)

    (
        model_key, train_key, key_prior, 
        key_datavector, key_state, key_sample
    ) = jr.split(key, 6)

    assert config.compression in ["nn", "nn-lbfgs"]

    net = ExtraMLP(
        40,
        5,
        [64, 32, 8],
        activation=jax.nn.tanh,
        key=key
    )

    net, F_imnn, metrics = train_imnn(
        key,
        net,
        cumulants_dataset.data,
        n_epochs=10_000,
        min_train_epochs=50,
        patience=100,
    )

    log_F_true = jnp.linalg.slogdet(jnp.linalg.inv(cumulants_dataset.data.Finv))
    log_F_true = log_F_true[0] * log_F_true[1]
    F_t, F_v = metrics.T
    plt.figure()
    plt.semilogy(F_t)
    plt.semilogy(F_v)
    plt.axhline(log_F_true, linestyle="--", color="k")
    plt.savefig(os.path.join(results_dir, "losses.png"), bbox_inches="tight")
    plt.close()

    Finv_imnn = jnp.linalg.inv(F_imnn)

    def compression_fn(d, p):
        # dmu = jnp.mean(
        #     get_f_d_alpha(net, cumulants_dataset.data.fiducial_data, ), 
        #     axis=0
        # )
        return p + Finv_imnn @ net(d)
    # def compression_fn(d, p):
    #     return net(d)

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