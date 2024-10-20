import blackjax.progress_bar
import jax
import jax.random as jr
import jax.numpy as jnp
import blackjax


def nuts_sample(key, log_prob_fn, prior=None, n_samples=100_000):

    def init_param_fn(seed):
        return prior.sample(seed=seed)

    warmup = blackjax.window_adaptation(blackjax.nuts, log_prob_fn)

    n_chains = 1

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