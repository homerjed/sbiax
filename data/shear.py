from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd


def get_shear_experiment():
    data_dir = "/Users/Jed.Homer/phd/sbiax/data/shear/"

    covariance   = np.loadtxt(data_dir + "covariance_cosmic_shear_PMEpaper.dat")
    precision    = np.linalg.inv(np.matrix(covariance))
    mu           = np.loadtxt(data_dir + "DES_shear-shear_a1.0_b0.5_data_vector")[:, 1]
    derivatives  = np.loadtxt(data_dir + "derivatives.dat").T
    param_names  = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$"]
    alpha        = np.array([0.3156, 0.831, -1.0]) # prior parameters (DES prior) # Om, s8, w0
    lower        = np.array([0.05, 0.45, -1.40])
    upper        = np.array([0.55, 1.00, -0.10])
    F            = np.linalg.multi_dot([derivatives, precision, derivatives.T])
    Finv         = np.linalg.inv(F)

    returns = (
        alpha,
        param_names,
        mu,
        covariance,
        precision,
        derivatives,
        F,
        Finv, 
        lower,
        upper
    )
    return [
        jnp.asarray(item) if isinstance(item, np.ndarray) else item 
        for item in returns 
    ]


def linearized_model(_alpha, mu, alpha, derivatives):
    """ Linearised model always uses true mu_0, C """
    return mu + jnp.dot(_alpha - alpha, derivatives)


def simulator(key, parameters, alpha, mu, derivatives, covariance):
    d = jr.multivariate_normal(
        key=key, 
        mean=linearized_model(
            _alpha=parameters, 
            mu=mu, 
            alpha=alpha, 
            derivatives=derivatives
        ),
        cov=covariance
    ) 
    return d


def get_estimated_objects(key, n_sims):
    """ 
        Get estimated covariance, Fisher information and precision. 
        Note that these objects are all independent of the model parameters.
    """

    # Get constants for experiment
    (
        alpha,
        param_names,
        mu,
        covariance,
        precision,
        derivatives,
        F,
        Finv, 
        lower,
        upper
    ) = get_shear_experiment()

    # Draw from true data-generating likelihood
    fiducials = jr.multivariate_normal(
        key, mean=mu, cov=covariance, shape=(n_sims,)
    )

    # Estimated expectation and covariance
    S = jnp.cov(fiducials, rowvar=False)

    # Calculate estimated precision matrix
    H = (n_sims - mu.size - 2.) / (n_sims - 1.) # Hartlap de-bias for data covariance (=h^-1)
    S_ = H * jnp.linalg.inv(S) # Hartlap on Cov. in Fisher + Cov.

    # Calculate estimated Fisher information 
    F_ = jnp.linalg.multi_dot([derivatives, S_, derivatives.T]) # Includes Hartlap
    Finv_ = jnp.linalg.inv(F_) 

    return S, Finv_, S_ 


def _mle(d, pi, Finv, mu, dmu, precision):
    return pi + jnp.linalg.multi_dot([Finv, dmu, precision, d - mu])


def get_experiment_data(key, good, n_sims, n_obs, *, results_dir):

    key, key_prior, key_simulate, key_obs = jr.split(key, 4)

    # Get constants for experiment
    (
        alpha,
        param_names,
        mu,
        covariance,
        precision,
        derivatives,
        F,
        Finv, 
        lower,
        upper
    ) = get_shear_experiment()

    # Estimate covariance, Fisher information and precision given n_sims 
    if not good:
        (
            covariance_est, Finv_est, precision_est
        ) = get_estimated_objects(key, n_sims) # NOTE: no cosmology needed here?
    else:
        covariance_est = Finv_est = precision_est = None

    # Data-generating likelihood 
    _simulator = partial(
        simulator, 
        alpha=alpha, 
        mu=mu, 
        derivatives=derivatives,
        covariance=covariance
    )

    # Sample from eig-prior or DES box prior
    prior = tfd.Blockwise(
        [
            tfd.Uniform(low=lower[p], high=upper[p]) 
            for p in range(alpha.size)
        ]
    )
    Y = prior.sample((n_sims,), seed=key_prior)

    # Simulate training set
    keys = jr.split(key_simulate, n_sims)
    D = jax.vmap(_simulator)(keys, Y)

    # Compress latins, observations and expectation
    mus = jax.vmap(linearized_model, in_axes=(0, None, None, None))(
        Y, mu, alpha, derivatives
    )
    X = jax.vmap(_mle, in_axes=(0, 0, None, 0, None, None))(
        D, 
        Y, 
        Finv if good else Finv_est,
        mus, # Models at each parameter set 
        derivatives,
        precision if good else precision_est
    )

    # Save all if the array exists for this experiment
    returns = (
        # Dataset for fitting likelihood/posterior model
        ("simulations.npy", X),
        ("parameters.npy", Y),
        # Estimated covariance, derivatives and Fisher information
        ("covariance.npy", covariance_est),
        # ("derivatives.npy", derivatives_est),
        ("Finv_est.npy", Finv_est),
        ("precision_est.npy", precision_est)
    )
    return [arr for _, arr in returns] + [prior]