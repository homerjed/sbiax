from typing import Tuple, Callable, Optional, Any
import blackjax.progress_bar
import jax
import jax.random as jr
import jax.numpy as jnp
from jaxtyping import Key, Array, Float, jaxtyped
from beartype import beartype as typechecker
import blackjax

Distribution = Any

@jaxtyped(typechecker=typechecker)
def nuts_sample(
    key: Key[jnp.ndarray, "..."], 
    log_prob_fn: Callable[[Float[Array, "..."]], Float[Array, ""]], 
    n_samples: int = 100_000, 
    n_chains: int = 1,
    n_warmup_steps: int = 1000,
    prior: Optional[Distribution] = None, 
    initial_state: Optional[Float[Array, "#i p"]] = None,
    sampling_kwargs: Optional[dict] = None
) -> Tuple[Float[Array, "#c #n p"], Float[Array, "#c #n"]]:
    """
    Runs NUTS (No-U-Turn Sampler) to sample from a posterior distribution using JAX.

    This function performs sampling using the NUTS algorithm, implemented via BlackJAX, 
    with an initial warm-up phase for tuning parameters. It uses the window adaptation 
    process to adjust the parameters during warm-up and runs the sampler in parallel 
    for multiple chains.

    Args:
        key: A JAX `PRNGKeyArray`.
        log_prob_fn: A callable representing the log probability function of the 
            posterior distribution. This function should take a set of parameters 
            as input and return their log probability (`Callable`).
        prior: A `tensorflow_probability` `Distribution` object representing the 
            prior distribution from which the initial parameter values are sampled (`Distribution`).
        n_samples: The number of posterior samples to generate (`int`). Default is 100,000.

    Returns:
        Tuple[`Array`, `Array`]:
            - The first array contains the sampled parameter positions from the NUTS algorithm 
            with shape `(n_samples,)` for one chain.
            - The second array contains the log densities (log posterior probabilities) corresponding 
            to each sampled position with shape `(n_samples,)`.

    Process:
        1. The prior distribution is used to sample initial parameter values.
        2. The NUTS sampler is tuned and adapted during the warm-up phase using `blackjax.window_adaptation`.
        3. After warm-up, the function performs sampling over `n_samples` using the NUTS kernel 
        provided by `blackjax.nuts.build_kernel`, running the sampler for one or more chains.
        4. The function returns the positions (sampled parameter values) and their associated 
        log densities from the posterior distribution.

    Example:
        ```python
        import jax
        import jax.random as jr
        from tensorflow_probability.substrates.jax.distributions import Normal
        from sbiax.inference import nuts_sample

        def log_prob_fn(params):
            # Typically this takes in a datavector of some kind!
            return -0.5 * jnp.sum(params ** 2)  

        key = jr.key(0)
        prior = Normal(0, 1)
        samples, log_densities = nuts_sample(key, log_prob_fn, prior)
        ```
    """
    if prior is not None:
        assert isinstance(prior, Distribution), (
            "Only tfp distributions are compatible currently."
        )

    key, init_key, warmup_key, sample_key = jr.split(key, 4)

    def init_param_fn(seed: Key[jnp.ndarray, "..."]) -> Float[Array, "..."]:
        """
        Samples initial parameters from the provided prior distribution.

        Args:
            seed: A JAX `PRNGKeyArray` used for random sampling.

        Returns:
            An array of sampled parameter values from the prior distribution.
        """
        return prior.sample(seed=seed)

    warmup = blackjax.window_adaptation(blackjax.nuts, log_prob_fn)

    init_keys = jr.split(init_key, n_chains)

    if initial_state is not None:
        initial_params = initial_state
    else:
        initial_params = jax.vmap(init_param_fn)(init_keys)

    @jax.vmap
    def call_warmup(seed: Key[jnp.ndarray, "..."], param: Float[Array, "..."]):
        """
        Performs the warm-up phase of NUTS to adapt parameters and obtain initial states.

        Args:
            seed: A JAX `PRNGKeyArray` used for warm-up.
            param: Initial parameter values for the sampler.

        Returns:
            Tuple containing:
                - Initial states after warm-up.
                - Tuned parameters obtained during warm-up.
        """
        (initial_states, tuned_params), _ = warmup.run(seed, param, n_warmup_steps)
        return initial_states, tuned_params

    warmup_keys = jr.split(warmup_key, n_chains)
    initial_states, tuned_params = jax.jit(call_warmup)(warmup_keys, initial_params)

    def inference_loop_multiple_chains(
        key: Key[jnp.ndarray, "..."], 
        initial_states: Float[Array, "..."], 
        tuned_params: Float[Array, "..."], 
        log_prob_fn: Callable[[Float[Array, "..."]], Float[Array, "..."]], 
        n_samples: int, 
        num_chains: int
    ) -> Tuple[blackjax._hmc.HMCState, blackjax._nuts.NUTSInfo]:
        """
        Runs the NUTS sampler for multiple chains to obtain posterior samples.

        Args:
            key: A JAX `PRNGKeyArray` for random sampling.
            initial_states: Initial states for the sampler, obtained after warm-up.
            tuned_params: Parameters tuned during the warm-up phase.
            log_prob_fn: The log probability function of the posterior distribution.
            n_samples: Number of samples to generate for each chain.
            num_chains: The number of parallel chains to run.

        Returns:
            Tuple containing:
                - An array of sampled states for all chains.
                - An array of additional information about the sampling process.
        """
        kernel = blackjax.nuts.build_kernel()

        def step_fn(
            key: Key[jnp.ndarray, "..."], state: blackjax._hmc.HMCState, **params
        ) -> Callable:
            """
            Performs a single step of the NUTS algorithm.

            Args:
                key: A JAX `PRNGKeyArray` for random sampling.
                state: The current state of the sampler.
                **params: Additional parameters for the NUTS kernel.

            Returns:
                The next state of the sampler and associated information.
            """
            return kernel(key, state, log_prob_fn, **params)

        def one_step(
            states: blackjax._hmc.HMCState, i: int
        ) -> Tuple[blackjax._hmc.HMCState, blackjax._nuts.NUTSInfo]: 
            """
            Executes one step of sampling across all chains.

            Args:
                states: The current states of all chains.
                i: The iteration index for tracking progress.

            Returns:
                Updated states and a tuple of new states and additional information.
            """
            keys = jr.split(jr.fold_in(key, i), num_chains)
            states, infos = jax.vmap(step_fn)(keys, states, **tuned_params)
            return states, (states, infos)

        _, (states, infos) = jax.lax.scan(
            one_step, initial_states, jnp.arange(n_samples)
        )
        return states, infos

    states, infos = inference_loop_multiple_chains(
        sample_key, 
        initial_states, 
        tuned_params, 
        log_prob_fn, 
        n_samples, 
        n_chains
    )
    
    return (
        states.position.transpose(1, 0, 2), 
        states.logdensity.transpose(1, 0)
    )

from typing import Callable, Optional, Tuple
import jax
import jax.numpy as jnp
import jax.random as jr 
from jax.sharding import NamedSharding
from jaxtyping import Key, Array
import numpy as np 
from tqdm import trange 
import matplotlib.pyplot as plt



def get_n_split_keys(key: Key, n: int) -> Tuple[Array, Array]:
    key, *keys = jr.split(key, n + 1)
    return key, jnp.asarray(keys)


def affine_sample(
    key: Key,
    log_prob: Callable,           
    n_walkers: int,          
    n_steps: int,
    burn: int,
    current_state: Array,      
    sharding: Optional[NamedSharding] = None,
    description: str = "Sampling",
    show_tqdm: bool = False
) -> Array:

    _log_prob = jax.vmap(log_prob)

    # Split the current state
    state_split = jnp.split(current_state, 2)
    if sharding is not None:
        state_split = jax.device_put(state_split, sharding)
    current_state1, current_state2 = state_split

    # Pull out the number of parameters and walkers
    n_state, n_params = current_state.shape

    key, keys = get_n_split_keys(key, len(current_state1))
    logp_current1 = _log_prob(theta=current_state1) #, key=keys
    key, keys = get_n_split_keys(key, len(current_state2))
    logp_current2 = _log_prob(theta=current_state2) #, key=keys

    logp_current1 = jnp.where(
        jnp.isnan(logp_current1),
        jnp.ones_like(logp_current1) * jnp.log(0.), # = -inf
        logp_current1
    )
    logp_current2 = jnp.where(
        jnp.isnan(logp_current2), 
        jnp.ones_like(logp_current2) * jnp.log(0.), 
        logp_current2
    )

    chain = [
        jnp.expand_dims(jnp.concatenate([current_state1, current_state2]), axis=0)
    ]
    
    # Kwargs for sampling random numbers
    uniform_kwargs = dict(minval=0., maxval=1.)
    randint_kwargs = dict(minval=0, maxval=n_walkers)
    
    bar = trange(n_steps, desc=description, colour="red")

    with bar as steps:
        for step in steps: 
            """ First set of walkers """

            # Proposals
            key, key_ix, key_u1, key_a1 = jr.split(key, 4)
            ix = jr.randint(key_ix, shape=(n_walkers,), **randint_kwargs)
            partners1 = current_state2[ix]
            u1 = jr.uniform(key_u1, shape=(n_walkers,), **uniform_kwargs)
            z1 = 0.5 * (u1 + 1.) ** 2.
            proposed_state1 = partners1 + (z1 * (current_state1 - partners1).T).T

            # Target log prob at proposed points
            if sharding is not None:
                proposed_state1 = jax.device_put(proposed_state1, sharding)

            key, keys = get_n_split_keys(key, len(proposed_state1))
            logp_proposed1 = _log_prob(theta=proposed_state1)#, key=keys)
            logp_proposed1 = jnp.where(
                jnp.isnan(logp_proposed1), 
                jnp.ones_like(logp_proposed1) * jnp.log(0.), 
                logp_proposed1
            )

            # Acceptance probability
            p_accept1 = jnp.minimum(
                jnp.ones(n_walkers), 
                z1 ** (n_params - 1.) * jnp.exp(logp_proposed1 - logp_current1)
            )

            # Accept or not
            accept1 = jr.uniform(key_a1, shape=(n_walkers,), **uniform_kwargs) <= p_accept1

            # Update the state
            current_state1 = (current_state1.T * (1. - accept1) + proposed_state1.T * accept1).T
            logp_current1 = jnp.where(accept1, logp_proposed1, logp_current1)

            """ Second set of walkers """

            # Proposals
            key, key_ix, key_u2, key_a2 = jr.split(key, 4)
            ix = jr.randint(key_ix, shape=(n_walkers,), **randint_kwargs)
            partners2 = current_state1[ix]

            u2 = jr.uniform(key_u2, shape=(n_walkers,), **uniform_kwargs)
            z2 = 0.5 * (u2 + 1.) ** 2.
            proposed_state2 = partners2 + (z2 * (current_state2 - partners2).T).T

            # Target log prob at proposed points
            if sharding is not None:
                proposed_state2 = jax.device_put(proposed_state2, sharding)

            key, keys = get_n_split_keys(key, len(proposed_state1))
            logp_proposed2 = _log_prob(theta=proposed_state2)#, key=keys)
            logp_proposed2 = jnp.where(
                jnp.isnan(logp_proposed2), 
                jnp.ones_like(logp_proposed2) * jnp.log(0.), 
                logp_proposed2
            )

            # Acceptance probability
            p_accept2 = jnp.minimum(
                jnp.ones(n_walkers), 
                z2 ** (n_params - 1.) * jnp.exp(logp_proposed2 - logp_current2)
            )

            # Accept or not
            accept2 = jr.uniform(key_a2, (n_walkers,), **uniform_kwargs) <= p_accept2

            # Update the state
            current_state2 = (current_state2.T * (1. - accept2) + proposed_state2.T * accept2).T
            logp_current2 = jnp.where(accept2, logp_proposed2, logp_current2)

            # Append to chain
            chain.append(
                # Stack?
                # jnp.stack([current_state1, current_state2])
                jnp.expand_dims(jnp.concatenate([current_state1, current_state2]), axis=0)
            )

    # def gelman_rubin(chains):
    #     n_samples, n_params = chains.shape
    #     Rhat = np.zeros(n_params)
    #     for p in range(n_params):
    #         chain_means = np.mean(chains[:, p], axis=1)
    #         chain_vars = np.var(chains[:, p], axis=1, ddof=1)

    #         # Between-chain variance
    #         B = n_samples * np.var(chain_means, ddof=1)
    #         # Within-chain variance
    #         W = np.mean(chain_vars)
    #         # Estimate of marginal posterior variance
    #         var_hat = (1. - 1. / n_samples) * W + B / n_samples
    #         Rhat[p] = np.sqrt(var_hat / W)
    #     return Rhat

    # def plot_traces(chains):
    #     n_samples, n_params = chains.shape
    #     fig, axes = plt.subplots(n_params, 1, figsize=(10, 2 * n_params), sharex=True)
    #     if n_params == 1:
    #         axes = [axes]
    #     for p in range(n_params):
    #         axes[p].plot(chains[:, p], alpha=0.6)
    #         axes[p].set_title(f'Trace for parameter {p}')
    #     plt.xlabel("Sample")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig("traces.png")


    chain = jnp.concatenate(chain)
    chain = chain[burn:].reshape(-1, chain.shape[-1])

    # plot_traces(chain)
    # print("GELMAN-RUBIN:", gelman_rubin(chain))

    return jnp.unique(chain, axis=0, return_counts=True)