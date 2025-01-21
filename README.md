<h1 align='center'>sbiax</h1>
<h2 align='center'>Fast, lightweight and parallel simulation-based inference.</h2>

[![DOI](https://joss.theoj.org/papers/10.21105/joss.07606/status.svg)](https://doi.org/10.21105/joss.07606) [![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) [![arXiv](https://img.shields.io/badge/arXiv-2412.02311-b31b1b.svg)](https://arxiv.org/abs/2412.02311)

<!-- 
<picture>
  <source srcset="https://github.com/homerjed/sbipdf/blob/main/assets/cover_dark.png" media="(prefers-color-scheme: dark)">
  <source srcset="https://github.com/homerjed/sbipdf/blob/main/assets/cover.png" media="(prefers-color-scheme: light)">
  <img src="https://github.com/homerjed/sbipdf/blob/main/assets/cover.png" alt="Your image description">
</picture> -->
<p align="center">
  <picture>
    <source srcset="https://github.com/homerjed/sbiax/blob/main/assets/cover_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="https://github.com/homerjed/sbiax/blob/main/assets/cover.png" media="(prefers-color-scheme: light)">
    <img src="https://github.com/homerjed/sbiax/blob/main/assets/cover.png" alt="Your image description">
  </picture>
</p>

<!-- <p align="center">
    <picture>
        <source srcset="assets/cover_dark.png" media="(prefers-color-scheme: dark)">
        <source srcset="assets/cover.png" media="(prefers-color-scheme: light)">
        <img src="assets/cover.png" alt="Your image description">
    </picture>
</p> -->

`sbiax` is a lightweight library for simulation-based inference (SBI) with a fixed grid of simulations. 

<!-- The design puts the neural density estimator (NDE) models at the centre of the code, allowing for flexible combinations of different models.  -->

> [!WARNING]
> :building_construction: Note this repository is under construction, expect changes. :building_construction:

-----

### Design

<!-- A typical inference with SBI occurs with  

* fitting a density estimator to a set of simulations and parameters $(\xi, \pi)$ that may be compressed to summary statistics,
* the measurement of a datavector $\hat{\xi}$,
* the sampling of a posterior $p(\pi|\hat{\xi})$ conditioned on the measurement $\hat{\xi}$.

`sbiax` is designed to perform such an inference.  -->

In a typical inference problem the data likelihood is unknown. Using density-estimation SBI, we can proceed by

<!-- Bayesian analyses where the likelihood function is unknown can proceed with density-estimation simulation-based inference methods, which typically involve -->

* simulating a set of data and model parameters $\{(\boldsymbol{\xi}, \boldsymbol{\pi})_0, ..., (\boldsymbol{\xi}, \boldsymbol{\pi})_N\}$,
* obtaining a measurement $\hat{\boldsymbol{\xi}}$,
* compressing the simulations and the measurements - usually with a neural network or linear compression - to a set of summaries $\{(\boldsymbol{x}, \boldsymbol{\pi})_0, ..., (\boldsymbol{x}, \boldsymbol{\pi})_N\}$ and $\hat{\boldsymbol{x}}$, 
* fitting an ensemble of normalising flow or similar density estimation algorithms (e.g. a Gaussian mixture model),
* the optional optimisation of the parameters for the architecture and fitting hyperparameters of the algorithms,
* sampling the ensemble posterior (using an MCMC sampler if the likelihood is fit directly) conditioned on the datavector to obtain parameter constraints on the parameters of a physical model, $\boldsymbol{\pi}$.

`sbiax` is a code for implementing each of these steps.


<!-- #### a) Configuration

An inference is defined by a `config` file. This is a dictionary that includes

* the architecture(s) of the NDEs,
* how to train these models,
* how to sample these models (e.g. MCMC, ...),
* where to save models, posteriors and figures,
* and generally any other information for your experiments.

NDEs are grouped in an ensemble that defines its own ensemble-likelihood function given an observation.

#### b) Density estimation

A posterior or likelihood is derived from a set of simulations and parameters by fitting a generative model with some loss - this may be a diffusion model or a normalising flow. 

`sbiax` is designed to be centred around these algorithms and to adopt the latest innovations from the machine learning literature.

#### c) Compression 

Density estimation is one of the oldest problems in machine learning. To avoid the difficulties of fitting high-dimensional models to data it is common to compress the data. 

`sbiax` gives you common compression methods that use linear methods or neural networks.  -->

-----

### Usage

Install via

```pip install sbiax```

and have a look at [examples](https://github.com/homerjed/sbiax/tree/main/examples).

-----

### Contributing

Want to add something? See `CONTRIBUTING.md`.

-----

### Citation

If you found this library to be useful in academic work, please cite:  <!--([arXiv link](https://arxiv.org/abs/2111.00254)) -->

```bibtex
@misc{homer2024simulationbasedinferencedodelsonschneidereffect,
      title={Simulation-based inference has its own Dodelson-Schneider effect (but it knows that it does)}, 
      author={Jed Homer and Oliver Friedrich and Daniel Gruen},
      year={2024},
      eprint={2412.02311},
      archivePrefix={arXiv},
      primaryClass={astro-ph.CO},
      url={https://arxiv.org/abs/2412.02311}, 
}
```

```bibtex
@article{
    Homer2025, 
    doi = {10.21105/joss.07606}, 
    url = {https://doi.org/10.21105/joss.07606}, 
    year = {2025}, 
    publisher = {The Open Journal}, 
    volume = {10}, 
    number = {105}, 
    pages = {7606}, 
    author = {Jed Homer and Oliver Friedrich}, 
    title = {SBIAX: Density-estimation simulation-based inference in JAX}, 
    journal = {Journal of Open Source Software} 
} 
```