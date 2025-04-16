import jax
import jax.numpy as jnp
from jax.scipy.linalg import svd


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
        S, self.principal_components = svd(X_centred, full_matrices=True)[1:]

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

        self.principal_components = svd(X_centred, full_matrices=True)[2]

        return jnp.dot(X_centred, self.principal_components[: self.num_components].T)

    def inverse_transform(self, X_transformed: jax.Array):
        if self.principal_components is None:
            raise RuntimeError("Must fit before transforming.")

        return (
            jnp.dot(X_transformed, self.principal_components[: self.num_components])
            + self.mean
        )