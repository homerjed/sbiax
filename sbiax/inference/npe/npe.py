import jax
import equinox as eqx


def sample_nde(model, key, x_, preprocess_fn, n_samples):
    # Sample model conditioned on x_ observation
    model = eqx.nn.inference_mode(model, True) # No dropout
    if preprocess_fn is not None:
        _, x_ = preprocess_fn((None, x_))
    print("x_, n_samples, key", x_, n_samples, key)
    samples, log_probs = model.sample_and_log_prob_n(key, x_, n_samples)
    return samples, log_probs 


def posterior_log_prob(model, data, parameters, preprocess_fn):
    model = eqx.nn.inference_mode(model, True) # No dropout
    data, parameters = preprocess_fn((data, parameters))
    return jax.vmap(model.log_prob, in_axes=(0, None))(parameters, data) # Switch y for x (log_prob(x, y))