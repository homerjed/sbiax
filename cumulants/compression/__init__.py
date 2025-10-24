import time
import os
from functools import partial
from typing import Callable, Any

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import PRNGKeyArray, Array, Float, jaxtyped

from beartype import beartype as typechecker 
import scipy
from ml_collections import ConfigDict
import matplotlib.pyplot as plt

from configs.log import setup_module_logger, get_log_level
from data.common import Dataset, linearised_model

from .nn import ExtraMLP, ResNetMLP, ViT, init_linear_weight, trunc_init, get_nn_compressor
from .imnn import get_imnn_compressor

TYPECHECK = True if os.environ.get("TYPECHECK", "").lower() in ("1", "true") else False
if TYPECHECK:
    typecheck = jaxtyped(typechecker=typechecker)
else:
    typecheck = lambda x: x

Distribution = Any

logger, log_figs_dir = setup_module_logger(__name__, level=get_log_level())

FORCE_NOISELESS_DATAVECTOR = True if os.environ.get("FORCE_NOISELESS_DATAVECTOR", "").lower() in ("1", "true") else False
N_ENSEMBLE_NETS = int(os.environ.get("N_ENSEMBLE_NETS", 10))
NN_TYPE = os.environ.get("NN_TYPE", "NN")
COVARIANCE_NN = True if os.environ.get("COVARIANCE_NN", "").lower() in ("1", "true") else False

COMPRESSION_TYPES = [
    "linear", "nn", "nn-lbfgs", 
    "imnn", "ensemble-nn", "vit", 
    "ensemble-vit", "none"
]

"""
    Objects common to the PDF and cumulant datasets
"""


def exists(v):
    return v is not None


"""
    Compression
"""


@typecheck
def get_linear_compressor(
    config: ConfigDict, 
    dataset: Dataset
) -> Callable[[Float[Array, "d"], Float[Array, "p"]], Float[Array, "p"]]:
    """ 
        Get Chi^2 minimisation function; compressing datavector 
        at estimated parameters to summary 
    """

    alpha = dataset.alpha

    # logger.info("Using MAP linear compression" if config.use_planck else "Using MLE linear compression")

    @typecheck
    def mle(
        d: Float[Array, "d"], 
        pi: Float[Array, "p"], 
        Finv: Float[Array, "p p"], 
        mu: Float[Array, "d"], 
        dmu: Float[Array, "p d"], 
        precision: Float[Array, "d d"]
    ) -> Float[Array, "p"]:
        return pi + jnp.linalg.multi_dot([Finv, dmu, precision, d - mu])

    # @typecheck
    # def map(
    #     d: Float[Array, "d"], 
    #     pi: Float[Array, "p"], 
    #     Finv: Float[Array, "p p"], 
    #     mu: Float[Array, "d"], 
    #     dmu: Float[Array, "p d"], 
    #     precision: Float[Array, "d d"]
    # ) -> Float[Array, "p"]:
    #     F = jnp.linalg.inv(Finv)
    #     _Finv = jnp.linalg.inv(F + F_planck)
    #     return pi + jnp.linalg.multi_dot([_Finv, dmu, precision, d - mu]) + jnp.linalg.multi_dot([_Finv, F_planck, alpha - pi])

    @typecheck
    def compressor(
        d: Float[Array, "d"], 
        p: Float[Array, "p"],
        mu: Float[Array, "d"], 
        dmu: Float[Array, "p d"]
    ) -> Float[Array, "p"]: 

        mu_p = linearised_model(
            alpha=alpha, alpha_=p, mu=mu, dmu=dmu
        )

        _estimator_fn = mle 

        p_ = _estimator_fn(
            d,
            pi=p,
            Finv=dataset.Finv, 
            mu=mu_p,            
            dmu=dmu, 
            precision=dataset.Cinv
        )

        return p_

    mu = jnp.mean(dataset.fiducial_data, axis=0)
    dmu = jnp.mean(dataset.derivatives, axis=0)

    return partial(compressor, mu=mu, dmu=dmu)


def get_default_nn(dataset: Dataset, config: ConfigDict, net_key: PRNGKeyArray) -> eqx.Module:

    if NN_TYPE == "NN":
        net = ExtraMLP(
            dataset.data.shape[-1], 
            dataset.parameters.shape[-1], 
            hidden_sizes=config.nn.width_size, 
            use_final_bias=config.nn.use_final_bias,
            activation=(
                getattr(jax.nn, config.nn.activation)
                if config.nn.depth > 0 else lambda x: x
            ),
            p=config.nn.p,
            covariance_nn=COVARIANCE_NN,
            key=net_key
        )
    if NN_TYPE == "RESNET":
        net = ResNetMLP(
            in_features=dataset.data.shape[-1], 
            num_outputs=dataset.parameters.shape[-1], 
            # widths=(128, 128),
            # blocks_per_stage=(2, 2),
            p_dropout=config.nn.p,
            key=net_key
        )
    if NN_TYPE == "VIT":
        net = ViT(
            input_len=dataset.data.shape[-1],
            output_dim=dataset.parameters.shape[-1], 
            patch_size=1, 
            embed_dim=4,
            mlp_hidden=32,
            num_heads=2,
            num_layers=1,
            dropout=0.1,
            covariance_nn=COVARIANCE_NN,
            key=net_key,
        )

    net = init_linear_weight(net, trunc_init, net_key)

    return net


class EnsembleNet(eqx.Module):
    # Create an ensemble of identical networks applied to a single input
    nets: list[eqx.Module]
    def __init__(self, nets):
        self.nets = nets
    def __call__(self, d, *args, **kwargs): # Disregard parameter input for NN
        xs = jax.tree.map(lambda d, net: net(d), [d] * len(self.nets), self.nets)
        return jnp.mean(jnp.asarray(xs), axis=0)


def get_nn_ensemble_compressor(
    net_key: PRNGKeyArray,
    train_key: PRNGKeyArray,
    config: ConfigDict,
    dataset: Dataset,
    *,
    results_dir: str,
    train: bool = True
) -> EnsembleNet:

    nets = []
    for n in range(N_ENSEMBLE_NETS):
        train_key_n = jr.fold_in(train_key, n)
        net_key_n = jr.fold_in(net_key, n)

        net_n = get_default_nn(dataset, config, net_key_n)

        print("MLP COMPRESSOR (linearised={}):\n".format(config.linearised), net_n)

        net_n = get_nn_compressor(
            train_key_n, 
            net_n,
            config,
            dataset, 
            lbfgs=(config.compression == "nn-lbfgs"), 
            results_dir=results_dir,
            train=train, # If not training, return initialised net and processing fns
            filename="nn_{}".format(n)
        )

        nets.append(net_n)

    compressor = EnsembleNet(nets)

    return compressor


@typecheck
def get_compression_fn(
    key: PRNGKeyArray, 
    config: ConfigDict, 
    dataset: Dataset, 
    *, 
    train: bool = True,
    results_dir: str
) -> Callable[[Float[Array, "d"], Float[Array, "p"]], Float[Array, "p"]]:
    """ 
        Get linear or neural network compressor
    """ 

    logger.info("Getting compression fn ({}) for dataset={}".format(config.compression, dataset.name))

    assert config.compression in COMPRESSION_TYPES, (
        'config.compression={} which is not in {}'.format(
            config.compression, COMPRESSION_TYPES 
        )
    )

    if config.compression == "nn" or config.compression == "nn-lbfgs":

        net_key, train_key = jr.split(key)

        net = get_default_nn(dataset, config, net_key)

        print("MLP COMPRESSOR (linearised={}):\n".format(config.linearised), net)

        compressor = get_nn_compressor(
            train_key, 
            net,
            config,
            dataset, 
            lbfgs=(config.compression == "nn-lbfgs"), 
            results_dir=results_dir,
            train=train, # If not training, return initialised net and processing fns
            filename="nn"
        )

        print("NET AFTER TRAINING OR LOADING", net)

        logger.info("Using NN compression function.")
        
    if config.compression == "ensemble-nn":

        net_key, train_key = jr.split(key)

        compressor = get_nn_ensemble_compressor(
            net_key, 
            train_key, 
            config, 
            dataset, 
            results_dir=results_dir,
            train=train
        )

        logger.info("Using Ensemble-NN compression function.")

    if config.compression == "imnn":

        net_key, train_key = jr.split(key)

        # net = get_default_nn(dataset, config, net_key)

        net = ExtraMLP(
            dataset.data.shape[-1], 
            dataset.parameters.shape[-1], 
            hidden_sizes=[32, 32], 
            use_final_bias=True,
            activation=jax.nn.tanh,
            p=0.,
            key=net_key
        )

        compressor = get_imnn_compressor(
            train_key,
            net,
            config,
            dataset,
            results_dir=results_dir,
            train=train
        )

        # def compression_fn_nn(d, p): 
        #     return net(d) #p + dataset.Finv @ net(d) # Ignore parameter kwarg for NN

        logger.info("Using IMNN compression function.")

    if config.compression == "linear":
        compressor = get_linear_compressor(config, dataset)

        logger.info("Using linear compression function.")
        
    if config.compression == "cca":
        # Compute the sampled parameter auto covariance, simulated data vector auto covariance
        # and the parameter-data vector cross covariance

        cov = jnp.cov(dataset.parameters.T, dataset.data.T)

        n_p = dataset.alpha.size
        cp = cov[:n_p, :n_p]
        cd = cov[n_p:, n_p:]
        cpd = cov[:n_p, n_p:]

        # This 'cl' can be understood as the projection of 'cp' to data vector space
        cl = cpd.T @ jnp.linalg.inv(cp) @ cpd

        # As seen in the paper, this generalized eigenvalue problem is equivalent to CCA
        # but is more numerical stable as 'cd' and 'cd-cl' are both invertible.
        # This problem is motivated as mutual information maximization under Gaussian linear model assumptions
        evals, evecs = map(jnp.asarray, scipy.linalg.eigh(cd, cd - cl))

        # In the context of the CCA, only min( dim(param), dim(data vector) ) components are real and the rest are noise. 
        evals = evals[::-1][:n_p]
        evecs = evecs[:, ::-1][:, :n_p]

        plt.title("Mutual Information Plot for CCA")
        plt.xlabel("Compressed Data Vector (CDV) Index")
        plt.ylabel("Mutual Information Between Parameter and CDV")
        plt.plot(jnp.log(evals)/2)
        plt.savefig("CCA_MI_plot.png")
        plt.close()

        # compressed_dv = dv_LFI@evecs

        # PCC = np.abs(np.corrcoef( compressed_dv,pars_LFI, rowvar = False))

        def compression_fn_cca(d, p):
            return d @ evecs

    if config.compression == "none":
        compressor = lambda d, p: d

    return compressor


class PCA:
    """
    Principal Component Analysis (PCA) for dimensionality reduction.

    Attributes:
        num_components (int): Number of principal components to keep.
        mean (jax.Array, optional): Mean of each feature in the training data.
        principal_components (jax.Array, optional): Principal components (eigenvectors) of the training data.
        explained_variance (jax.Array, optional): Explained variance of each principal component.
    """

    def __init__(self, num_components: int):
        self.num_components = num_components
        self.mean = None
        self.principal_components = None
        self.explained_variance = None

    def fit(self, X: jax.Array):
        n, m = X.shape

        if self.mean is None:
            self.mean = X.mean(axis=0)

        X_centred = X - self.mean
        S, self.principal_components = jax.scipy.linalg.svd(X_centred, full_matrices=True)[1:]

        self.explained_variance = jnp.square(S) / jnp.sum(jnp.square(S))

    def transform(self, X: jax.Array):
        if self.principal_components is None:
            raise RuntimeError("Must fit before transforming.")

        X_centred = X - X.mean(axis=0)
        return jnp.dot(X_centred, self.principal_components[: self.num_components].T)

    def fit_transform(self, X: jax.Array):
        if self.mean is None:
            self.mean = X.mean(axis=0)

        X_centred = X - self.mean

        self.principal_components = jax.scipy.linalg.svd(X_centred, full_matrices=True)[2]

        return jnp.dot(X_centred, self.principal_components[: self.num_components].T)

    def inverse_transform(self, X_transformed: jax.Array):
        if self.principal_components is None:
            raise RuntimeError("Must fit before transforming.")

        return (
            jnp.dot(X_transformed, self.principal_components[: self.num_components])
            + self.mean
        )