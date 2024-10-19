import os
import numpy as np 
import matplotlib.pyplot as plt
from chainconsumer import Chain, ChainConfig, ChainConsumer, PlotConfig, Truth

from utils import make_df

"""
    Posterior plotting etc
"""

def plot_parameters_and_summaries(parameters, summaries, param_names, filename):
    c = ChainConsumer()
    c.add_chain(
        Chain(
            samples=make_df(parameters, np.ones((len(parameters),)), param_names), 
            name="Params", 
            color="blue", 
            plot_cloud=True, 
            plot_contour=False
        )
    )
    c.add_chain(
        Chain(
            samples=make_df(summaries, np.ones((len(summaries),)), param_names), 
            name="Summaries", 
            color="red", 
            plot_cloud=True, 
            plot_contour=False
        )
    )
    c.add_truth(Truth(location=dict(zip(param_names, alpha)), name=r"$\pi^0$"))
    fig = c.plotter.plot()
    plt.savefig(filename)#os.path.join(results_dir, "params.pdf"))
    plt.close()


def plot_posterior(
    samples,
    samples_log_prob,
    alpha,
    X_,
    Finv,
    alpha_=None,
    Finv_est=None,
    param_names=None,
    filename=None
):
    posterior_df = make_df(samples, samples_log_prob, param_names)

    c = ChainConsumer()

    # Fisher truth
    fisher_chain = Chain.from_covariance(
        alpha,
        Finv,
        columns=param_names,
        name="Fisher",
        color="k"
    )
    c.add_chain(fisher_chain)

    # Fisher estimated
    c.add_chain(
        Chain.from_covariance(
            X_,
            Finv_est if Finv_est is not None else Finv,
            columns=param_names,
            name=r"$F_{\Sigma^{-1}}$",
            color="k",
            linestyle=":"
        )
    )

    # SBI posterior
    c.add_chain(Chain(samples=posterior_df, name="SBI", color="r"))

    # True parameters
    c.add_truth(Truth(location=dict(zip(param_names, alpha)), name=r"$\pi$"))

    # MAP
    if alpha_ is not None:
        c.add_truth(Truth(location=dict(zip(param_names, alpha_)), name=r"$\pi$'"))

    # Summary
    c.add_marker(location=dict(zip(param_names, np.asarray(X_))), name=r"$\hat{x}$", color="b")

    # Truth (fiducial)
    c.add_marker(location=dict(zip(param_names, np.asarray(alpha))), name=r"$\alpha$", color="#7600bc")

    # Truth sampled
    c.add_marker(location=dict(zip(param_names, np.asarray(alpha_))), name=r"$\alpha'$", color="g")

    fig = c.plotter.plot()
    plt.savefig(filename)
    plt.close()