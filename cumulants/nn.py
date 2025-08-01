from typing import Optional, Callable
import jax
import jax.random as jr
import equinox as eqx
from jaxtyping import Array, jaxtyped
from beartype import beartype as typechecker
import numpy as np

typecheck = jaxtyped(typechecker=typechecker)

LossFunction = Optional[Callable[[eqx.Module, Array, Array], Array]]

"""
    Tools for compression with neural networks.
    - train a user-defined `eqx.Module` network that compresses a datavector
      to a model-dimensional summary, by minimising a MSE loss.
"""


# def loss(
#     model: eqx.Module, 
#     x: Float[Array, "b x"], 
#     y: Float[Array, "b y"]
# ) -> Float[Array, ""]:
#     def fn(x, y):
#         y_ = model(x)
#         return jnp.square(jnp.subtract(y_, y))
#     return jnp.mean(jax.vmap(fn)(x, y))


# @eqx.filter_jit
# def evaluate(
#     model: eqx.Module, 
#     x: Float[Array, "b x"], 
#     y: Float[Array, "b y"],
#     *, 
#     loss_fn: LossFunction = None,
#     replicated_sharding: Optional[PositionalSharding] = None
# ) -> Float[Array, ""]:
#     if replicated_sharding is not None:
#         model = eqx.filter_shard(model, replicated_sharding)
#     _loss_fn = loss_fn if loss_fn is not None else loss 
#     return _loss_fn(model, x, y)


# @typecheck
# @eqx.filter_jit
# def make_step(
#     model: eqx.Module, 
#     opt_state: optax.OptState,
#     x: Float[Array, "b x"], 
#     y: Float[Array, "b y"],
#     opt: optax.GradientTransformation, 
#     *, 
#     loss_fn: LossFunction = None,
#     replicated_sharding: Optional[PositionalSharding]
# ) -> Tuple[eqx.Module, optax.OptState, Float[Array, ""]]:
#     if replicated_sharding is not None:
#         model, opt_state = eqx.filter_shard(
#             (model, opt_state), replicated_sharding
#         )
#     _loss_fn = loss_fn if loss_fn is not None else loss
#     loss_value, grads = eqx.filter_value_and_grad(_loss_fn)(model, x, y)
#     updates, opt_state = opt.update(grads, opt_state, model)
#     model = eqx.apply_updates(model, updates)
#     if replicated_sharding is not None:
#         model, opt_state = eqx.filter_shard(
#             (model, opt_state), replicated_sharding
#         )
#     return model, opt_state, loss_value


# def get_batch(
#     D: Float[Array, "n x"], 
#     Y: Float[Array, "n y"], 
#     n: int, 
#     key: PRNGKeyArray 
# ) -> Tuple[Float[Array, "b x"], Float[Array, "b y"]]:
#     idx = jr.choice(key, jnp.arange(D.shape[0]), (n,))
#     return D[idx], Y[idx]


# @typecheck
# def fit_nn(
#     key: PRNGKeyArray,
#     model: eqx.Module, 
#     dataset: DatasetClass, 
#     opt: optax.GradientTransformation, 
#     n_batch: int, 
#     patience: Optional[int], 
#     n_steps: int = 10_000, 
#     valid_fraction: float = 0.9, 
#     valid_data: Optional[Sequence[Array]] = None,
#     batch_dataset: bool = True,
#     *,
#     loss_fn: LossFunction = None,
#     preprocess_fn: Callable[[Array], Array],
#     sharding: Optional[NamedSharding] = None,
#     replicated_sharding: Optional[PositionalSharding] = None,
# ) -> Tuple[eqx.Module, Float[np.ndarray, "l 2"]]:
#     """
#         Trains a neural network model with early stopping.
#     """

#     n_s, _ = dataset.data.data.shape

#     opt_state = opt.init(eqx.filter(model, eqx.is_array))

#     if valid_data is not None:
#         raise NotImplementedError()
#         # Xt, Yt = train_data
#         # Xv, Yv = valid_data
#     else:
#         Xt, Xv = jnp.split(preprocess_fn(dataset.data.data), [int(valid_fraction * n_s)]) 
#         Yt, Yv = jnp.split(dataset.data.parameters, [int(valid_fraction * n_s)])

#     L = np.zeros((n_steps, 2))
#     with trange(n_steps, desc="Training NN", colour="blue") as steps:
#         for step in steps:
#             key_t, key_v = jr.split(jr.fold_in(key, step))

#             if batch_dataset:
#                 x, y = get_batch(Xt, Yt, n=n_batch, key=key_t) # Xt, Yt
#             else:
#                 x, y = Xt, Yt
            
#             if sharding is not None:
#                 x, y = eqx.filter_shard((x, y), sharding)

#             model, opt_state, train_loss = make_step(
#                 model, opt_state, x, y, opt, loss_fn=loss_fn, replicated_sharding=replicated_sharding
#             )

#             if batch_dataset:
#                 x, y = get_batch(Xv, Yv, n=n_batch, key=key_v)
#             else:
#                 x, y = Xv, Yv

#             if sharding is not None:
#                 x, y = eqx.filter_shard((x, y), sharding)

#             valid_loss = evaluate(
#                 model, x, y, loss_fn=loss_fn, replicated_sharding=replicated_sharding
#             )

#             L[step] = train_loss, valid_loss
#             steps.set_postfix_str(
#                 "train={:.3E}, valid={:.3E}".format(train_loss.item(), valid_loss.item())
#             )

#             if patience is not None:
#                 if (step > 0) and (step - np.argmin(L[:step, 1]) > patience):
#                     steps.set_description_str("Stopped at {}".format(step))
#                     break

#     return model, L[:step]


if __name__ == "__main__":
    import os
    import time
    import datetime
    import matplotlib.pyplot as plt
    from chainconsumer import Chain, ChainConsumer, Truth

    from configs.args import get_cumulants_sbi_args
    from data.common import get_compression_fn
    from utils import get_datasets, get_results_dir, make_df

    t0 = time.time()

    args = get_cumulants_sbi_args()

    config, cumulants_dataset, datasets = get_datasets(args) # Config and cumulants_dataset can be bulk ... etc

    print("TIME:", datetime.datetime.now().strftime("%H:%M %d-%m-%y"))
    print("SEED:", args.seed)
    print("MOMENTS:", args.order_idx)
    print("LINEARISED:", args.linearised)

    """
        Config
    """

    config, cumulants_dataset, datasets = get_datasets(args) # Config and cumulants_dataset can be bulk ... etc

    results_dir = get_results_dir(config, args)

    key = jr.key(config.seed)

    (
        model_key, train_key, key_prior, 
        key_datavector, key_state, key_sample
    ) = jr.split(key, 6)

    results_dir = get_results_dir(config, args)

    key = jr.key(config.seed)

    assert config.compression in ["nn", "nn-lbfgs"]

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
            color="blue", 
            plot_cloud=True, 
            plot_contour=False
        )
    )
    c.add_chain(
        Chain(
            samples=make_df(X, parameter_strings=cumulants_dataset.get_parameter_strings()), 
            name="Summaries", 
            color="red", 
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
        ax.scatter(cumulants_dataset.data.parameters[:, p], X[:, p], s=0.1)
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
    c.add_chain(
        Chain(
            samples=make_df(
                cumulants_dataset.data.parameters, 
                parameter_strings=cumulants_dataset.get_parameter_strings()
            ), 
            name="Params", 
            color="blue", 
            plot_cloud=True, 
            plot_contour=False
        )
    )
    c.add_chain(
        Chain(
            samples=make_df(X, parameter_strings=cumulants_dataset.get_parameter_strings()), 
            name="Summaries", 
            color="red", 
            plot_cloud=True, 
            plot_contour=False
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
        cumulants_dataset.data.alpha, cumulants_dataset.data.Finv, (20_000,) 
    ) 
    fisher_samples_log_prob = jax.scipy.stats.multivariate_normal.logpdf(
        fisher_samples, 
        cumulants_dataset.data.alpha, 
        cumulants_dataset.data.Finv
    )
    fisher_df = make_df(
        np.clip(fisher_samples, cumulants_dataset.data.lower, cumulants_dataset.data.upper),
        # samples_log_prob, 
        parameter_strings=cumulants_dataset.get_parameter_strings()
    )
    c.add_chain(
        Chain(
            samples=fisher_df,
            name=r"$F_{\Sigma^{-1}}$" + " {}".format("clipped"),
            color="g",
            linestyle=":",
            shade_alpha=0.
        )
    )

    fig = c.plotter.plot()
    plt.savefig(os.path.join(results_dir, "nn_params_fiducial.png")) 
    plt.close()