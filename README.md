<h1 align='center'>sbiax</h1>
<h2 align='center'>Fast, lightweight and parallel simulation-based inference.</h2>

<p align="center">
  <img style="max-width: 128px; max-height: 128px" src="https://github.com/homerjed/sbipdf/blob/main/assets/cover.png" />
</p>

`sbiax` is a lightweight library for simulation-based inference (SBI) with a fixed-grid of simulations. 

The design puts the neural density estimator (NDE) models at the centre of the code, allowing for flexible combinations of different models. 

-----

> [!WARNING]
> :building_construction: Note this repository is under construction, expect changes. :building_construction:

### Design

A typical inference with SBI occurs with  

* fitting a density estimator to a set of simulations and parameters $\{\xi, \pi\}$ that may be compressed to summary statistics,
* the measurement of a datavector $\hat{\xi}$,
* the sampling of a posterior $p(\pi|\hat{\xi})$ conditioned on the measurement $\hat{\xi}$.

`sbiax` is designed to perform such an inference. 

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