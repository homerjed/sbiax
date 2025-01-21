---
title: 'SBIAX: Density-estimation simulation-based inference in JAX'
tags:
  - Python
  - Machine learning 
  - Generative models 
  - Bayesian Inference
  - Simulation based inference
authors:
  - name: Jed Homer
    orcid: 0009-0002-0985-1437 
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Oliver Friedrich
    orcid: 0000-0001-6120-4988
    equal-contrib: False
    affiliation: "1, 2, 3" # (Multiple affiliations must be quoted)
affiliations:
 - name: University Observatory, Faculty for Physics, Ludwig-Maximilians-Universität München, Scheinerstrasse 1, München, Deustchland.
   index: 1
   ror: 00hx57361
 - name: Munich Center for Machine Learning.
   index: 2
   ror: 00hx57361
 - name: Excellence Cluster ORIGINS, Boltzmannstr. 2, 85748 Garching, Deutschland.
   index: 3
   ror: 00hx57361
date: 10 January 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

In a typical Bayesian inference problem, the data likelihood is not known. However, in recent years, machine learning methods for density estimation can allow for inference using an estimator of the data likelihood. This likelihood estimator is fit with neural networks that are trained on simulations to maximise the likelihood of the simulation-parameter pairs - one of the many available tools for Simulation Based Inference (SBI), [@sbi]. In such analyses, density-estimation simulation-based inference methods can derive a posterior, which typically involves 

* simulating a set of data and model parameters $\{(\boldsymbol{\xi}, \boldsymbol{\pi})_0, \ldots, (\boldsymbol{\xi}, \boldsymbol{\pi})_N\}$,
* obtaining a measurement $\hat{\boldsymbol{\xi}}$,
* compressing the simulations and the measurements - usually with a neural network or linear compression - to a set of summaries $\{(\boldsymbol{x}, \boldsymbol{\pi})_0, \ldots, (\boldsymbol{x}, \boldsymbol{\pi})_N\}$ and $\hat{\boldsymbol{x}}$, 
* fitting an ensemble of normalising flow or similar density estimation algorithms (e.g. a Gaussian mixture model),
* the optional optimisation of the parameters for the architecture and fitting hyper-parameters of the algorithms,
- * sampling the ensemble posterior (using an MCMC-sampler if the likelihood is fit directly), conditioned on the data-vector, to obtain parameter constraints on the parameters of a physical model, $\boldsymbol{\pi}$.

`sbiax` is a software package that implements each of these steps. The code allows for Neural Likelihood Estimation [@papamakarios; @delfi], and Neural Posterior Estimation [@npe].

As shown in @homersbi, SBI can successfully obtain the correct posterior widths and coverages given enough simulations which agree with the analytic solution - this software was used in the research for this publication. 

# Statement of need

Simulation Based Inference (SBI) covers a broad class of statistical techniques such as Approximate Bayesian Computation (ABC) [@ABC], Neural Ratio Estimation (NRE) [@NRE], Neural Likelihood Estimation (NLE), and Neural Posterior Estimation (NPE). These techniques can derive posterior distributions, conditioned of noisy data vectors, in a rigorous and efficient manner with assumptions on the data likelihood. In particular, density-estimation methods have emerged as a promising method, given their efficiency, in which generative models are used to fit likelihoods or posteriors directly using simulations.

In the field of cosmology, SBI is of particular interest due to complexity and non-linearity of models for the expectations of non-standard summary statistics of the large-scale structure, as well as the non-Gaussian noise distributions for these statistics. The assumptions required for the complex analytic modelling of these statistics - as well as the increasing dimensionality of data returned by spectroscopic and photometric galaxy surveys, limit the amount of information that can be obtained on fundamental physical parameters. Therefore, the study and research into current and future statistical methods for Bayesian inference is of paramount importance for cosmology, especially in light of current and next-generation survey missions such as DES [@Euclid], DESI [@DESI], and Euclid [@Euclid].

The software we present, `sbiax`, is designed to be used by machine learning and physics researchers for running Bayesian inferences using density-estimation SBI techniques. These models can be fit easily with multi-accelerator training and inference within the code. This software - written in `jax` [@jax] - allows for seemless integration of cutting edge generative models to SBI, including continuous normalising flows [@ffjord], matched flows [@flowmatching], masked autoregressive flows [@mafs; @flowjax], and Gaussian mixture models - all of which are implemented in the code. The code features integration with the `optuna` [@optuna] hyper-parameter optimisation framework which would be used to ensure consistent analyses, `blackjax` [@blackjax] for fast MCMC sampling, and `equinox` [@equinox] for neural network methods. The design of `sbiax` allows for new density estimation algorithms to be trained and sampled from, as long as they conform to a simple and typical design pattern demonstrated in `sbiax`. 

Whilst excellent software packages already exist for conducting simulation-based inference (e.g. `sbi` [@sbimacke], `sbijax` [@sbidirmeier]) for some applications it is useful to have a lightweight implementation that focuses on speed, ensembling of density estimators and easily integrated MCMC sampling (e.g. for ensembles of likelihoods) - all of which is based on a lightweight and regularly maintained `jax` machine learning library such as `equinox` [@equinox]. `sbiax` depends on density estimators and compression modules - as long as log-probability and callable methods exists for these, they can be integrated seemlessly.

# Density estimation with normalising flows 

The use of density-estimation in SBI has been accelerated by the advent of normalising flows. These models parameterise a change-of-variables $\boldsymbol{y}=f_\phi(\boldsymbol{x};\boldsymbol{\pi})$ between a simple base distribution (e.g. a multivariate unit Gaussian $\mathcal{G}[\boldsymbol{z}|\mathbf{0}, \mathbf{I}]$) and an unknown distribution $q(\boldsymbol{x}|\boldsymbol{\pi})$ (from which we have simulated samples $\boldsymbol{x}$). Naturally, this is of particular importance for inference problems in which the likelihood is not known. The change-of-variables is fit from data by training neural networks to model the transformation in order to maximise the log-likelihood of the simulated data $\boldsymbol{x}$ conditioned on the parameters $\boldsymbol{\pi}$ of a simulator model. The mapping is expressed as

$$
    \boldsymbol{y} = f_\phi(\boldsymbol{x};\boldsymbol{\pi}),
$$

where $\phi$ are the parameters of the neural network. The log-likelihood of the flow is expressed as 

$$
    \log p_\phi(\boldsymbol{x}|\boldsymbol{\pi}) = \log \mathcal{G}[f_\phi(\boldsymbol{x};\boldsymbol{\pi})|\boldsymbol{0}, \mathbb{I}] + \log \big | \mathbf{J}_{f_\phi}(\boldsymbol{x};\boldsymbol{\pi})\big |,
$$

This density estimator is fit to a set of $N$ simulation-parameter pairs $\{(\boldsymbol{\xi}, \boldsymbol{\pi})_0, ..., (\boldsymbol{\xi}, \boldsymbol{\pi})_N\}$ by minimising a Monte-Carlo estimate of the KL-divergence 

\begin{align}
    \langle D_{KL}(q||p_\phi) \rangle_{\boldsymbol{\pi} \sim p(\boldsymbol{\pi})} &= \int \text{d}\boldsymbol{\pi} \; p(\boldsymbol{\pi}) \int \text{d}\boldsymbol{x} \; q(\boldsymbol{x}|\boldsymbol{\pi}) \log \frac{q(\boldsymbol{x}|\boldsymbol{\pi})}{p_\phi(\boldsymbol{x}|\boldsymbol{\pi})}, \nonumber \\
    &= \int \text{d}\boldsymbol{\pi} \int \text{d}\boldsymbol{x} \; p(\boldsymbol{\pi}, \boldsymbol{x})[\log q(\boldsymbol{x}|\boldsymbol{\pi}) - \log p_\phi(\boldsymbol{x}|\boldsymbol{\pi})], \nonumber \\
    &\geq -\int \text{d}\boldsymbol{\pi} \int \text{d}\boldsymbol{x} \; p(\boldsymbol{\pi},\boldsymbol{x}) \log p_\phi(\boldsymbol{x}|\boldsymbol{\pi}), \nonumber \\
    &\approx -\frac{1}{N}\sum_{i=1}^N \log p_\phi(\boldsymbol{x}_i|\boldsymbol{\pi}_i),
\end{align}

where $q(\boldsymbol{x}|\boldsymbol{\pi})$ is the unknown likelihood from which the simulations $\boldsymbol{x}$ are drawn. This applies similarly for an estimator of the posterior (instead of the likelihood as shown here) and is the basis of being able to estimate the likelihood or posterior directly when an analytic form is not available. If the likelihood is fit from simulations, a prior is required and the posterior is sampled via an MCMC-sampler given some measurement. This is implemented within the code.

An ensemble of density estimators (with parameters - e.g. the weights and biases of the networks - denoted by $\{ \phi_0, ..., \phi_J\}$) has a likelihood which is written as

$$
    p_{\text{ensemble}}(\boldsymbol{\xi}|\boldsymbol{\pi}) = \sum_{j=1}^J \alpha_j p_{\phi_j}(\hat{\boldsymbol{\xi}}|\boldsymbol{\pi})
$$

where

$$
    \alpha_i = \frac{\exp(p_{\phi_i}(\hat{\boldsymbol{\xi}}|\boldsymbol{\pi}))}{\sum_{j=1}^J\exp(p_{\phi_j}(\hat{\boldsymbol{\xi}}|\boldsymbol{\pi}))}
$$

are the weights of each density estimator in the ensemble. This ensemble likelihood can be easily sampled with an MCMC-sampler. In Figure 
\ref{fig:sbi_example} we show an example posterior from applying SBI, with our software, using two compression methods separately. 

![An example of posteriors derived with `sbiax`. We fit an ensemble of two continuous normalising flows to a set of simulations of cosmic shear two-point functions. The expectation $\xi[\pi]$ is linearised with respect to $\pi$ and a theoretical data covariance model $\Sigma$ (in this example) allows for easy sampling of many simulations - an ideal test arena for SBI methods. We derive two posteriors, from separate experiments, where a linear (red) or neural network compression (blue) is used. In black, the true analytic posterior is shown. Note that for a finite set of simulations the blue posterior will not overlap completely with the black and red posteriors - we explore this effect upon the posteriors from SBI methods, due to an unknown data covariance, in @homersbi.\label{fig:sbi_example}](sbi_example.png)


<!-- # Compression

Maximum likelihood estimators for the parameters of a model $\xi[\pi]$ for the data $\hat{\xi}$ are derived by $\chi^2$ minimisation with respect to $\pi$

\begin{align}
    \hat{\pi} &= \text{min}_\pi \chi^2(\pi, \hat{\xi}, \xi[\pi], \Sigma) \nonumber \\
    & = \pi + F_{\Sigma}^{-1}E^T\Sigma^{-1}(\hat{\xi} - \xi[\pi]).
\end{align}

where $\Sigma$ is the data covariance, $E$ is a $\text{dim}(\pi) \times \text{\dim}(\xi)$ dimensional matrix and $F_{\Sigma}^{-1}$ is the Fisher information matrix.

Estimators for the model parameters can also be derived using neural networks that minimise the mean-squared error loss $ -->

<!-- # Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)" -->

# Acknowledgements

We thank the developers of the packages `jax` [@jax], `blackjax` [@blackjax], `optax` [@optax], `equinox` [@equinox], `diffrax` [@diffrax] and `flowjax` [@flowjax] for their work and for making their code available to the community.

# References
