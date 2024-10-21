---
title: 'SBIAX: Density-estimation simulation-based inference in JAX.'
tags:
  - Python
  - Machine learning 
  - Generative models 
  - Bayesian Inference
  - Simulation based inference
authors:
  - name: Jed Homer
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: Ludwig-Maximilians-Universität München, Faculty for Physics, University Observatory, Scheinerstrasse 1, München, Deustchland.
   index: 1
   ror: 00hx57361
date: 1 October 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Bayesian analyses where the likelihood function is unknown can proceed with density-estimation simulation-based inference methods, which typically involve

* synthesising a set of simulated data and model parameters $\{(\boldsymbol{\xi}, \boldsymbol{\pi})_0, ..., (\boldsymbol{\xi}, \boldsymbol{\pi})_N\}$,
* obtaining a measurement $\hat{\boldsymbol{\xi}}$,
* compressing the simulations and the measurements - usually with a neural network or linear compression - to a set of summaries $\{(\boldsymbol{x}, \boldsymbol{\pi})_0, ..., (\boldsymbol{x}, \boldsymbol{\pi})_N\}$ and $\hat{\boldsymbol{x}}$, 
* fitting an ensemble of normalising flow or similar density estimation algorithms (e.g. a Gaussian mixture model),
* the optional optimisation of the parameters for the architecture and fitting hyperparameters of the algorithms,
* sampling the ensemble posterior (using an MCMC sampler if the likelihood is fit directly) conditioned on the datavector to obtain parameter constraints on the parameters of a physical model, $\boldsymbol{\pi}$.

`sbiax` is a code for implementing each of these steps.

<!--  
    What the code does
    - Large / parallel training of big diffusion models on multiple accelerators
    - Speeding up MCMC, LFI, field-level, inverse problems
-->

# Statement of need

<!--  
    - Diffusion models are theoretically complex generative models. 
      Need fast sampling and likleihood methods built on GPU-parallel
      ODE solvers (diffrax). Subclass of energy-based generative models.

    - Given this dataset, the goal of generative modeling is to fit a model 
      to the data distribution such that we can synthesize new data points 
      at will by sampling from the distribution.

    - Significant limitations of implicit and likelihood-based ML models
      e.g. modelling normalised probability distributions, likelihood calculations
      and sampling speed. Score matching avoids this. Diffusion scales to large
      datasets of high dimension better than other approaches.

    - Score-based models have achieved SOTA results on many tasks and applications
      e.g. LDMs, ...

    - Given the new avenues of research fast and large generative models offer,
      a code that carefully implements them is valuable.

    - Memory efficiency compared to normalising flows for the same tasks (one network conditioned on 't' compared to many sub-flows + faster than CNFs)

    - implemented in JAX, equinox and diffrax
-->

Simulation-based inference (SBI) covers a broad class of statistical techniques such as Approximate Bayesian Computation (ABC), Neural Ratio Estimation (NRE), Neural Likelihood Estimation (NLE) and Neural Posterior Estimation (NPE). These techniques can derive posterior distributions conditioned of noisy data vectors in a rigorous and efficient manner. In particular, density-estimation methods have emerged as a promising method, given their efficiency, using generative models to fit likelihoods or posteriors directly using simulations.

In the field of cosmology, SBI is of particular interest due to complexity and non-linearity of models for the expectations of non-standard summary statistics of the large-scale structure, as well as the non-Gaussian noise distributions for these statistics. The assumptions required for the complex analytic modelling of these statistics as well as the increasing dimensionality of data returned by spectroscopic and photometric galaxy surveys limits the amount of information that can be obtained on fundamental physical parameters. Therefore, the study and research into current and future statistical methods for Bayesian inference is of paramount importance for the field of cosmology.

The software we present, `sbiax`, is designed to be used by machine learning and physics researchers for running Bayesian inferences using density-estimation SBI techniques. These models can be fit easily with multi-accelerator training and inference within the code. This code - written in `jax` [@jax] - allows for seemless integration of cutting edge generative models to SBI, including continuous normalising flows [@ffjord], matched flows [@flowmatching] and masked autoregressive flows [@mafs; @flowjax]. The code features integration with the `optuna` [@optuna] hyperparameter optimisation framework which would be used to ensure consistent analyses, `blackjax` [@blackjax] for fast MCMC sampling and neural network compression methods with `equinox` [@equinox].  

<!-- BlackJAX integrated for MCMC sampling -->

# Density estimation with normalising flows 

<!-- What is SBI -->

<!-- What is a normalising flow -->

The use of density-estimation in SBI has been accelerated by the advent of normalising flows. These models parameterise a change-of-variables $\boldsymbol{y}=f_\phi(\boldsymbol{x};\boldsymbol{\pi})$ between a simple base distribution (e.g. a multivariate unit Gaussian $\mathcal{G}[\boldsymbol{z}|\mathbf{0}, \mathbf{I}]$) and an unknown distribution $q(\boldsymbol{x}|\boldsymbol{\pi})$ (from which we have simulated samples $\boldsymbol{x}$). Naturally, this is of particular importance in inference problems in which the likelihood is not known. The change-of-variables is fit from data by training neural networks to model the transformation in order to maximise the log-likelihood of the simulated data $\boldsymbol{x}$ conditioned on the parameters $\boldsymbol{\pi}$ of a simulator model. The mapping is expressed as

$$
    \boldsymbol{y} = f_\phi(\boldsymbol{x};\boldsymbol{\pi}),
$$

where $\phi$ are the parameters of the neural network. The log-likelihood of the flow is expressed as 

$$
    \log p_\phi(\boldsymbol{x}|\boldsymbol{\pi}) = \log \mathcal{G}[f_\phi(\boldsymbol{x};\boldsymbol{\pi})|\boldsymbol{0}, \mathbb{I}] + \log \big | \mathbf{J}_{f_\phi}(\boldsymbol{x};\boldsymbol{\pi})\big |,
$$

This density estimate is fit to a set of $N$ simulation-parameter pairs $\{(\boldsymbol{x}_i, \boldsymbol{\pi}_i)\}_{i=1}^N$ by minimising a Monte-Carlo estimate of the KL-divergence 

$$
\begin{align}
    \langle D_{KL}(q||p_\phi) \rangle_{\boldsymbol{\pi} \sim p(\boldsymbol{\pi})} &= \int \text{d}\boldsymbol{\pi} \; p(\boldsymbol{\pi}) \int \text{d}\boldsymbol{x} \; q(\boldsymbol{x}|\boldsymbol{\pi}) \log \frac{q(\boldsymbol{x}|\boldsymbol{\pi})}{p_\phi(\boldsymbol{x}|\boldsymbol{\pi})}, \nonumber \\
    &= \int \text{d}\boldsymbol{\pi} \int \text{d}\boldsymbol{x} \; p(\boldsymbol{\pi}, \boldsymbol{x})[\log q(\boldsymbol{x}|\boldsymbol{\pi}) - \log p_\phi(\boldsymbol{x}|\boldsymbol{\pi})], \nonumber \\
    &\geq -\int \text{d}\boldsymbol{\pi} \int \text{d}\boldsymbol{x} \; p(\boldsymbol{\pi},\boldsymbol{x}) \log p_\phi(\boldsymbol{x}|\boldsymbol{\pi}), \nonumber \\
    &\approx -\frac{1}{N}\sum_i^N \log p_\phi(\boldsymbol{x}_i|\boldsymbol{\pi}_i),
\end{align}
$$

where $q(\boldsymbol{x}|\boldsymbol{\pi})$ is the unknown likelihood from which the simulations $\boldsymbol{x}$ are drawn. This applies similarly for an estimator of the posterior (instead of the likelihood as shown here) and is the basis of being able to estimate the likelihood or posterior directly when an analytic form is not available. If the likelihood is fit from simulations, a prior is required and the posterior is sampled via an MCMC given some measurement. This is implemented within the code.



# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We thank the developers of these packages for their work and for making their code available to the community.

# References