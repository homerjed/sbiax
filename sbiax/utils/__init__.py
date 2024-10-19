import jax.numpy as jnp
import pandas as pd


def make_df(samples, log_probs, param_names):
    df = pd.DataFrame(samples, columns=param_names).assign(log_posterior=log_probs)
    return df

def nan_to_value(samples, log_probs):
    # Set any bad samples to very low probability
    log_probs = log_probs.at[~jnp.isfinite(log_probs)].set(-1e-100)
    return samples, log_probs