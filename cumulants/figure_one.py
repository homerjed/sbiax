# import argparse
# from typing import Tuple
# import os
# import jax
# import jax.numpy as jnp
# import jax.random as jr
# import equinox as eqx
# from jaxtyping import Key
# import numpy as np
# from ml_collections import ConfigDict
# from scipy.linalg import block_diag
# import matplotlib.pyplot as plt
# from chainconsumer import ChainConsumer, Chain, Truth
# from tensorflow_probability.substrates.jax.distributions import Distribution

# from configs.configs import get_base_results_dir, get_results_dir, get_multi_z_posterior_dir
# from configs.configs import ensembles_cumulants_config, ensembles_bulk_cumulants_config 
# from data.moments import Dataset, get_data, get_linear_compressor, get_datavector, get_prior, get_parameter_strings
# from sbiax.ndes import Ensemble, MultiEnsemble, get_ndes_from_config
# from sbiax.ndes import CNF, MAF, Scaler
# from sbiax.compression.linear import _mle
# from sbiax.inference import nuts_sample
# from sbiax.inference.nle import affine_sample
# from sbiax.utils import make_df, marker

# from multi_z import (
#     get_multi_z_args, get_z_config_and_datavector, 
#     maybe_vmap_multi_redshift_mle, get_multi_redshift_mle
# )

# """
#     Loop through seeds
#     - loading posteriors sampled over all redshifts, 
#     - loading bulk[cumulants/pdf] and tails[cumulants] posteriors
#     to plot together
# """

# def default(v, d):
#     return v if v is not None else d

# def posterior_object(posterior_file):

#     PosteriorTuple = namedtuple("PosteriorTuple", posterior_file.files)

#     # Instantiate the namedtuple with the corresponding arrays
#     posterior_tuple = DataTuple(
#         *(posterior_file[key] for key in posterior_file.files)
#     )

#     return posterior_tuple

# if __name__ == "__main__":

#     args = get_multi_z_args()

#     key = jr.key(0)

#     config = ensembles_moments_config(
#         seed=default(args.seed, 0), # Defaults if run without argparse args
#         sbi_type=default(args.sbi_type, "nle"), 
#         linearised=default(args.linearised, True),
#         pre_train=default(args.pre_train, False)
#     )

#     parameter_strings = get_parameter_strings()

#     linear_str = "linear" if config.linearised else ""

#     # Where SBI's are saved (add on suffix for experiment details)
#     results_dir = get_base_results_dir()
#     figs_dir = "{}figs/".format(results_dir)

#     POSTERIOR_OBJECTS = [] # Bulk, tails objects for given seed

#     # Sample the multiple-redshift-ensemble posterior
#     for :

#         posterior_save_dir = get_multi_z_posterior_dir(config, default(args.sbi_type, "nle"))

#         posterior_filename = os.path.join(
#             posterior_save_dir, "posterior_{}.npz".format(args.seed)
#         )

#         posterior_file = np.load(posterior_filename)
#         posterior_file = posterior_object(posterior_file)

#         POSTERIOR_OBJECTS.append(posterior_file)

#     for posterior_object, title in zip(
#         POSTERIOR_OBJECTS, ["bulk", "tails"]
#     ):

#         # posterior_object keys:
#         # np.savez(
#         #     posterior_filename,
#         #     samples=samples, 
#         #     samples_log_prob=samples_log_prob,
#         #     Finv=Finv,
#         #     summary=x_ # Is this correct one to save?
#         # )

#         c = ChainConsumer() 

#         c.add_chain(
#             Chain.from_covariance(
#                 alpha,
#                 posterior_object.Finv, # NOTE: Get multi redshift Fisher matrix, use a multi-inference config
#                 columns=parameter_strings,
#                 name=r"$F_{\Sigma^{-1}}$ " + title,
#                 color="k",
#                 linestyle=":",
#                 shade_alpha=0.
#             )
#         )

#         posterior_df = make_df(
#             posterior_object.samples, 
#             posterior_object.samples_log_prob, 
#             parameter_strings
#         )
#         c.add_chain(Chain(samples=posterior_df, name="SBI " + title, color="r"))

#         c.add_marker(
#             location=marker(posterior_object.x_, parameter_strings), 
#             name=r"$\hat{x}$ " + title, 
#             color="b"
#         )

#     c.add_marker(
#         location=marker(alpha, parameter_strings), 
#         name=r"$\alpha$", 
#         color="#7600bc"
#     )

#     fig = c.plotter.plot()
#     plt.savefig(os.path.join(figs_dir, "figure_one_{}.pdf".format(args.seed)))
#     plt.close()









#     # def get_dataset_and_config(bulk_or_tails):
#     #     if bulk_or_tails == "bulk" or bulk_or_tails == "bulk_pdf":
#     #         dataset_constructor = BulkCumulantsDataset
#     #         config = bulk_cumulants_config 
#     #     if bulk_or_tails == "tails":
#     #         dataset_constructor = CumulantsDataset
#     #         config = cumulants_config 
#     #     return dataset_constructor, config

#     # t0 = time.time()

#     # args = get_cumulants_sbi_args()

#     # print("TIME:", datetime.datetime.now().strftime("%H:%M %d-%m-%y"))
#     # print("SEED:", args.seed)
#     # print("MOMENTS:", args.order_idx)
#     # print("LINEARISED:", args.linearised)

#     # # Loop through bulk[cumulants / pdf], tails[cumulants] posteriors from given seeds
#     # for s in range(123):
#     #     """
#     #         Config
#     #     """

#     #     # Bulk / tails constructors for dataset / config
#     #     _dataset, _config = get_dataset_and_config(args.bulk_or_tails) 

#     #     config = _config(
#     #         seed=args.seed, 
#     #         redshift=args.redshift, 
#     #         reduced_cumulants=args.reduced_cumulants,
#     #         sbi_type=args.sbi_type,
#     #         linearised=args.linearised, 
#     #         compression=args.compression,
#     #         order_idx=args.order_idx,
#     #         n_linear_sims=args.n_linear_sims,
#     #         pre_train=args.pre_train
#     #     )

#     #     key = jr.key(config.seed)

#     #     ( 
#     #         model_key, train_key, key_prior, 
#     #         key_datavector, key_state, key_sample
#     #     ) = jr.split(key, 6)

#     #     results_dir = get_results_dir(config, args)

#     #     posteriors_dir = get_posteriors_dir(config, args)

#     #     # Dataset of simulations, parameters, covariance, ...
#     #     cumulants_dataset = _dataset(
#     #         config, pdfs=("pdf" in args.bulk_or_tails), results_dir=results_dir
#     #     )

#     #     dataset: Dataset = cumulants_dataset.data

#     #     parameter_prior: Distribution = cumulants_dataset.prior

#     #     bulk_pdfs = True # Use PDFs for Finv not cumulants
#     #     bulk_dataset: Dataset = get_bulk_dataset(args, pdfs=bulk_pdfs) # For Fisher forecast comparisons

#     #     posterior_filename = os.path.join(results_dir, "posterior.npz"), 


#     #     posterior_file = np.load(posterior_filename)

#     #     PosteriorTuple = namedtuple("PosteriorTuple", posterior_file.files)

#     #     # Instantiate the namedtuple with the corresponding arrays
#     #     data_named = DataTuple(
#     #         *(posterior_file[key] for key in posterior_file.files)
#     #     )

#     #     # np.savez(
#     #     #     alpha=dataset.alpha,
#     #     #     samples=samples,
#     #     #     samples_log_prob=samples_log_prob,
#     #     #     datavector=datavector,
#     #     #     summary=x_
#     #     # )













if 1:
    import argparse
    from typing import Tuple
    import os
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx
    from jaxtyping import Key
    import numpy as np
    from ml_collections import ConfigDict
    from scipy.linalg import block_diag
    import matplotlib.pyplot as plt
    from chainconsumer import ChainConsumer, Chain, Truth
    from tensorflow_probability.substrates.jax.distributions import Distribution

    from configs.configs import get_base_results_dir, get_results_dir, get_multi_z_posterior_dir
    from configs.configs import ensembles_cumulants_config, ensembles_bulk_cumulants_config 
    from data.moments import Dataset, get_data, get_linear_compressor, get_datavector, get_prior, get_parameter_strings
    from sbiax.ndes import Ensemble, MultiEnsemble, get_ndes_from_config
    from sbiax.ndes import CNF, MAF, Scaler
    from sbiax.compression.linear import _mle
    from sbiax.inference import nuts_sample
    from sbiax.inference.nle import affine_sample
    from sbiax.utils import make_df, marker

    from cumulants_multi_z import 
    from configs.configs import (
        cumulants_config, 
        ensembles_cumulants_config,
        ensembles_bulk_cumulants_config,
        get_multi_z_posterior_dir
    )
    from configs.args import (
        get_cumulants_sbi_args, 
        get_cumulants_multi_z_args
    )

    """
        Loop through seeds, getting configs for ensembles for bulk or tails
        over all redshifts, loading posteriors from them 
    """

    def posterior_object(posterior_file):
        PosteriorTuple = namedtuple("PosteriorTuple", posterior_file.files)
        posterior_tuple = PosteriorTuple(
            *(posterior_file[key] for key in posterior_file.files)
        )
        return posterior_tuple

    args = get_cumulants_multi_z_args()

    c = ChainConsumer() 

    for bulk_or_tails in ["bulk", "tails"]:

        # Multi-z inference concerning the bulk or bulk + tails
        if bulk_or_tails == "tails":
            ensembles_config = ensembles_cumulants_config
        if bulk_or_tails == "bulk" or args.bulk_or_tails == "bulk_pdf":
            ensembles_config = ensembles_bulk_cumulants_config

        config = ensembles_config(
            seed=args.seed, # Defaults if run without argparse args
            sbi_type=args.sbi_type, 
            linearised=args.linearised,
            reduced_cumulants=args.reduced_cumulants,
            order_idx=args.order_idx,
            redshifts=args.redshifts,
            compression=args.compression,
            pre_train=args.pre_train
        )

        # Save posterior, Fisher and summary
        posterior_save_dir = get_multi_z_posterior_dir(config, args)
        if not os.path.exists(posterior_save_dir):
            os.makedirs(posterior_save_dir, exist_ok=True)

        # Posterior for bulk/tails for a given seed
        posterior_filename = os.path.join(
            posterior_save_dir, "posterior_{}.npz".format(args.seed)
        )
        posterior_file = np.load(
            posterior_filename,
        )
        posterior_file = posterior_object(posterior_file)

        c.add_chain(
            Chain.from_covariance(
                alpha,
                posterior_object.Finv, # NOTE: Get multi redshift Fisher matrix, use a multi-inference config
                columns=parameter_strings,
                name=r"$F_{\Sigma^{-1}}$ " + title,
                color="k",
                linestyle=":",
                shade_alpha=0.
            )
        )

        posterior_df = make_df(
            posterior_object.samples, 
            posterior_object.samples_log_prob, 
            parameter_strings
        )
        c.add_chain(Chain(samples=posterior_df, name="SBI " + title, color="r"))

        c.add_marker(
            location=marker(posterior_object.summary, parameter_strings), 
            name=r"$\hat{x}$ " + title, 
            color="b"
        )

    c.add_marker(
        location=marker(alpha, parameter_strings), 
        name=r"$\alpha$", 
        color="#7600bc"
    )

    fig = c.plotter.plot()
    plt.savefig(os.path.join(figs_dir, "figure_one_{}.pdf".format(args.seed)))
    plt.close()