def test_toy_nle():

    import os
    import multiprocessing

    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
        multiprocessing.cpu_count()
    )

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx
    import diffrax as dfx
    import optax
    import tensorflow_probability.substrates.jax.distributions as tfd

    from configs import make_dirs

    from sbiax.utils import make_df, get_shardings
    from sbiax.ndes import Ensemble, MAF, CNF, Scaler
    from sbiax.train import train_ensemble
    from sbiax.inference import nuts_sample
    from sbiax.compression.nn import fit_nn

    key = jr.key(0)

    key, model_key, train_key, sample_key, data_key = jr.split(key, 5)

    results_dir = "results/toy/"

    make_dirs(results_dir)

    n_sims = 10_000
    data_dim = 100
    parameter_dim = 2

    alpha = jnp.array([0.5, 0.5]) # True values of mu and sigma
    lower = jnp.array([0., 0.1])
    upper = jnp.array([1., 1.])
    parameter_names = [r"$\mu$", r"$\sigma$"]

    parameter_prior = tfd.Blockwise(
        [tfd.Uniform(lower[0], upper[0]), tfd.Uniform(lower[1], upper[1])]
    )

    def simulator(key, mu, sigma):
        # Draw data d ~ G[d|mu, I * sigma]
        return jr.multivariate_normal(key, jnp.ones(data_dim) * mu, jnp.eye(data_dim) * sigma)

    key_data, key_sims, key_prior = jr.split(data_key, 3)

    d = simulator(key_data, alpha[0], alpha[1])

    Y = parameter_prior.sample((n_sims,), seed=key_prior)

    keys = jr.split(key_sims, n_sims)
    D = jax.vmap(simulator)(keys, Y[:, 0], Y[:, 1])

    sharding, replicated_sharding = get_shardings()

    net_key, net_train_key = jr.split(key)

    net = eqx.nn.MLP(
        D.shape[-1], 
        Y.shape[-1], 
        width_size=32, 
        depth=2, 
        activation=jax.nn.tanh,
        key=net_key
    )

    opt = optax.adamw(1e-3)

    preprocess_fn = lambda x: (x - D.mean(axis=0)) / D.std(axis=0)

    model, losses = fit_nn(
        net_train_key, 
        net, 
        train_data=(preprocess_fn(D), Y), 
        opt=opt, 
        n_batch=500, 
        patience=1000,
        sharding=sharding,
        replicated_sharding=replicated_sharding
    )

    X = jax.vmap(model)(preprocess_fn(D))

    X_ = model(preprocess_fn(d)) 

    assert jnp.all(jnp.isfinite(X))
    assert jnp.all(jnp.isfinite(losses))
    assert X.shape[-1] == parameter_dim

    model_keys = jr.split(model_key, 2)

    sbi_type = "nle"

    scaler = Scaler(X, Y, use_scaling=True)

    solver = dfx.Heun()

    ndes = [
        MAF(
            event_dim=alpha.size, 
            context_dim=alpha.size, 
            width_size=32,
            nn_depth=2,
            n_layers=5,
            scaler=scaler,
            key=model_keys[0]
        ),
        CNF(
            event_dim=alpha.size, 
            context_dim=alpha.size, 
            solver=solver,
            dt=0.1,
            t1=1.0,
            width_size=8,
            depth=0,
            activation=jax.nn.tanh,
            scaler=scaler,
            key=model_keys[1]
        ) 
    ]

    ensemble = Ensemble(ndes, sbi_type=sbi_type)

    opt = optax.adamw(1e-3)

    ensemble, stats = train_ensemble(
        train_key, 
        ensemble,
        train_mode=sbi_type,
        train_data=(X, Y), 
        opt=opt,
        n_batch=50,
        patience=20,
        n_epochs=1000,
        sharding=sharding,
        replicated_sharding=replicated_sharding,
        results_dir=results_dir
    )

    # Check losses of NDEs
    assert all(
        [
            jnp.all(jnp.isfinite(jnp.asarray(stats[n]["train_losses"])))
            for n in range(len(ensemble.ndes))
        ]
    )
    assert all(
        [
            jnp.all(jnp.isfinite(jnp.asarray(stats[n]["train_losses"])))
            for n in range(len(ensemble.ndes))
        ]
    )
    assert all(
        [
            jnp.all(jnp.isfinite(jnp.asarray(stats[n]["all_valid_loss"])))
            for n in range(len(ensemble.ndes))
        ]
    )

    key_data, key_sample = jr.split(sample_key)

    ensemble = eqx.nn.inference_mode(ensemble)

    log_prob_fn = ensemble.ensemble_log_prob_fn(X_, parameter_prior)

    n_chains = 2 # Sample multiple chains for this posterior

    samples, samples_log_prob = nuts_sample(
        key_sample, 
        log_prob_fn, 
        n_chains=n_chains, 
        prior=parameter_prior
    )

    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all(jnp.isfinite(samples_log_prob))
    assert samples.shape[-1] == parameter_dim

    n_chains = 1

    posteriors = []
    for nde in ensemble.ndes:
        log_prob_fn = ensemble.nde_log_prob_fn(nde, X_, parameter_prior)

        nde_posterior = nuts_sample(
            key_sample, log_prob_fn, n_chains=n_chains, prior=parameter_prior
        )
        posteriors.append(nde_posterior)

    # Check each posterior
    assert all(
        [jnp.all(jnp.isfinite(jnp.squeeze(samples))) for samples, _ in posteriors]
    )
    assert all(
        [jnp.all(jnp.isfinite(jnp.squeeze(samples_log_probs))) for _, samples_log_probs in posteriors]
    )
    assert all([samples.shape[-1] == parameter_dim for samples, _ in posteriors])


# Test takes a long time so only run NLE for now

# def test_toy_npe():

#     import os
#     import multiprocessing

#     os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
#         multiprocessing.cpu_count()
#     )

#     import jax
#     import jax.numpy as jnp
#     import jax.random as jr
#     import equinox as eqx
#     import diffrax as dfx
#     import optax
#     import tensorflow_probability.substrates.jax.distributions as tfd

#     from configs import make_dirs

#     from sbiax.utils import make_df, get_shardings
#     from sbiax.ndes import Ensemble, MAF, CNF, Scaler
#     from sbiax.train import train_ensemble
#     from sbiax.inference import nuts_sample
#     from sbiax.compression.nn import fit_nn

#     key = jr.key(0)

#     key, model_key, train_key, sample_key, data_key = jr.split(key, 5)

#     results_dir = "results/toy/"

#     make_dirs(results_dir)

#     n_sims = 10_000
#     data_dim = 100
#     parameter_dim = 2

#     alpha = jnp.array([0.5, 0.5]) # True values of mu and sigma
#     lower = jnp.array([0., 0.1])
#     upper = jnp.array([1., 1.])
#     parameter_names = [r"$\mu$", r"$\sigma$"]

#     parameter_prior = tfd.Blockwise(
#         [tfd.Uniform(lower[0], upper[0]), tfd.Uniform(lower[1], upper[1])]
#     )

#     def simulator(key, mu, sigma):
#         # Draw data d ~ G[d|mu, I * sigma]
#         return jr.multivariate_normal(key, jnp.ones(data_dim) * mu, jnp.eye(data_dim) * sigma)

#     key_data, key_sims, key_prior = jr.split(data_key, 3)

#     d = simulator(key_data, alpha[0], alpha[1])

#     Y = parameter_prior.sample((n_sims,), seed=key_prior)

#     keys = jr.split(key_sims, n_sims)
#     D = jax.vmap(simulator)(keys, Y[:, 0], Y[:, 1])

#     sharding, replicated_sharding = get_shardings()

#     net_key, net_train_key = jr.split(key)

#     net = eqx.nn.MLP(
#         D.shape[-1], 
#         Y.shape[-1], 
#         width_size=32, 
#         depth=2, 
#         activation=jax.nn.tanh,
#         key=net_key
#     )

#     opt = optax.adamw(1e-3)

#     preprocess_fn = lambda x: (x - D.mean(axis=0)) / D.std(axis=0)

#     model, losses = fit_nn(
#         net_train_key, 
#         net, 
#         train_data=(preprocess_fn(D), Y), 
#         opt=opt, 
#         n_batch=500, 
#         patience=1000,
#         sharding=sharding,
#         replicated_sharding=replicated_sharding
#     )

#     X = jax.vmap(model)(preprocess_fn(D))

#     X_ = model(preprocess_fn(d)) 

#     assert jnp.all(jnp.isfinite(X))
#     assert jnp.all(jnp.isfinite(losses))
#     assert X.shape[-1] == parameter_dim

#     model_keys = jr.split(model_key, 2)

#     sbi_type = "npe"

#     scaler = Scaler(X, Y, use_scaling=True)

#     solver = dfx.Heun()

#     ndes = [
#         MAF(
#             event_dim=alpha.size, 
#             context_dim=alpha.size, 
#             width_size=32,
#             nn_depth=2,
#             n_layers=5,
#             scaler=scaler,
#             key=model_keys[0]
#         ),
#         CNF(
#             event_dim=alpha.size, 
#             context_dim=alpha.size, 
#             solver=solver,
#             dt=0.1,
#             t1=1.0,
#             width_size=8,
#             depth=0,
#             activation=jax.nn.tanh,
#             scaler=scaler,
#             key=model_keys[1]
#         ) 
#     ]

#     ensemble = Ensemble(ndes, sbi_type=sbi_type)

#     opt = optax.adamw(1e-3)

#     ensemble, stats = train_ensemble(
#         train_key, 
#         ensemble,
#         train_mode=sbi_type,
#         train_data=(X, Y), 
#         opt=opt,
#         n_batch=50,
#         patience=20,
#         n_epochs=1000,
#         sharding=sharding,
#         replicated_sharding=replicated_sharding,
#         results_dir=results_dir
#     )

#     # Check losses of NDEs
#     assert all(
#         [
#             jnp.all(jnp.isfinite(jnp.asarray(stats[n]["train_losses"])))
#             for n in range(len(ensemble.ndes))
#         ]
#     )
#     assert all(
#         [
#             jnp.all(jnp.isfinite(jnp.asarray(stats[n]["train_losses"])))
#             for n in range(len(ensemble.ndes))
#         ]
#     )
#     assert all(
#         [
#             jnp.all(jnp.isfinite(jnp.asarray(stats[n]["all_valid_loss"])))
#             for n in range(len(ensemble.ndes))
#         ]
#     )

#     key_data, key_sample = jr.split(sample_key)

#     ensemble = eqx.nn.inference_mode(ensemble)

#     log_prob_fn = ensemble.ensemble_log_prob_fn(X_, parameter_prior)

#     n_chains = 2 # Sample multiple chains for this posterior

#     samples, samples_log_prob = nuts_sample(
#         key_sample, 
#         log_prob_fn, 
#         n_chains=n_chains, 
#         prior=parameter_prior
#     )

#     assert jnp.all(jnp.isfinite(samples))
#     assert jnp.all(jnp.isfinite(samples_log_prob))
#     assert samples.shape[-1] == parameter_dim

#     n_chains = 1

#     posteriors = []
#     for nde in ensemble.ndes:
#         log_prob_fn = ensemble.nde_log_prob_fn(nde, X_, parameter_prior)

#         nde_posterior = nuts_sample(
#             key_sample, log_prob_fn, n_chains=n_chains, prior=parameter_prior
#         )
#         posteriors.append(nde_posterior)

#     # Check each posterior
#     assert all(
#         [jnp.all(jnp.isfinite(jnp.squeeze(samples))) for samples, _ in posteriors]
#     )
#     assert all(
#         [jnp.all(jnp.isfinite(jnp.squeeze(samples_log_probs))) for _, samples_log_probs in posteriors]
#     )
#     assert all([samples.shape[-1] == parameter_dim for samples, _ in posteriors])