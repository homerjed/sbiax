import blackjax.progress_bar
import jax
import jax.random as jr
import jax.numpy as jnp
import blackjax
from blackjax.util import run_inference_algorithm
from blackjax.progress_bar import progress_bar_scan

# def blackjax_sample(key, log_prob_fn, initial_parameters):
#     alg = blackjax.nuts(
#         logdensity_fn=logdensity_fn,
#         inverse_mass_matrix=jnp.eye(2),
#         step_size=1e-3
#     )

#     initial_position = jnp.ones((2,))

#     _ = run_inference_algorithm(
#         rng_key=key,
#         initial_position=initial_parameters,
#         inference_algorithm=alg,
#         num_steps=100,
#         progress_bar=True,
#     )


# def blackjax_sample(key, log_prob_fn, initial_parameters, n_chains=1, n_samples=10_000):

#     key, init_key, warmup_key, sample_key = jr.split(key, 4)

#     warmup = blackjax.window_adaptation(blackjax.nuts, log_prob_fn)

#     def call_warmup(seed, param):
#         (initial_states, tuned_params), _ = warmup.run(seed, param, 1_000)
#         return initial_states, tuned_params

#     warmup_keys = jr.split(warmup_key, n_chains)
#     initial_states, tuned_params = jax.jit(jax.vmap(call_warmup))(warmup_keys, initial_parameters)

#     def inference_loop_multiple_chains(
#         key, initial_states, tuned_params, log_prob_fn, n_samples, n_chains
#     ):
#         kernel = blackjax.nuts.build_kernel()

#         def _step_fn(key, state, **params):
#             return kernel(key, state, log_prob_fn, **params)

#         def _one_step(states, rng_key):
#             keys = jr.split(rng_key, n_chains)
#             states, infos = jax.vmap(_step_fn)(keys, states, **tuned_params)
#             return states, (states, infos)

#         keys = jr.split(key, n_samples)
#         _, (states, infos) = jax.lax.scan(_one_step, initial_states, keys)

#         return (states, infos)

#     states, infos = inference_loop_multiple_chains(
#         sample_key, initial_states, tuned_params, log_prob_fn, n_samples, n_chains
#     )

#     positions = jnp.concatenate(
#         [states.position[:, n, :] for n in range(n_chains)]
#     )
#     densities = jnp.concatenate(
#         [states.logdensity[:, n] for n in range(n_chains)]
#     )
#     return positions, densities


def blackjax_sample(key, log_prob_fn, prior=None, n_chains=1, n_samples=100_000):

    def init_param_fn(seed):
        # Get initial parameters
        return prior.sample(seed=seed)

    warmup = blackjax.window_adaptation(blackjax.nuts, log_prob_fn)

    # we use 4 chains for sampling
    key, init_key, warmup_key, sample_key = jr.split(key, 4)
    init_keys = jr.split(init_key, n_chains)
    init_params = jax.vmap(init_param_fn)(init_keys)

    @jax.vmap
    def call_warmup(seed, param):
        (initial_states, tuned_params), _ = warmup.run(seed, param, 1000)
        return initial_states, tuned_params

    warmup_keys = jr.split(warmup_key, n_chains)
    initial_states, tuned_params = jax.jit(call_warmup)(warmup_keys, init_params)

    def inference_loop_multiple_chains(
        rng_key, initial_states, tuned_params, log_prob_fn, n_samples, num_chains
    ):
        """
            Does this just step EACH sample once? 
            Need to run this for multiple steps?!
        """
        kernel = blackjax.nuts.build_kernel()

        def step_fn(key, state, **params):
            return kernel(key, state, log_prob_fn, **params)

        def one_step(states, i):
            keys = jr.split(jr.fold_in(rng_key, i), num_chains)
            states, infos = jax.vmap(step_fn)(keys, states, **tuned_params)
            return states, (states, infos)

        _, (states, infos) = jax.lax.scan(
            one_step, initial_states, jnp.arange(n_samples)
        )
        return states, infos


    key, sample_key = jr.split(key)
    states, infos = inference_loop_multiple_chains(
        sample_key, initial_states, tuned_params, log_prob_fn, n_samples, n_chains
    )

    return states.position[:, 0], states.logdensity[:, 0]