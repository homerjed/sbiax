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
    from collections import namedtuple
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

    from sbiax.ndes import CNF, MAF, Scaler
    from sbiax.inference import nuts_sample
    from sbiax.utils import make_df, marker

    from configs import (
        cumulants_config, 
        bulk_cumulants_config, 
        get_results_dir, 
        get_posteriors_dir, 
        get_ndes_from_config
    )
    from configs.configs import (
        get_base_results_dir, 
        get_results_dir, 
        get_multi_z_posterior_dir, 
        get_ndes_from_config
    )
    from configs.ensembles_configs import (
        ensembles_cumulants_config, ensembles_bulk_cumulants_config 
    )
    from cumulants_ensemble import Ensemble, MultiEnsemble
    from data.constants import (
        get_quijote_parameters, 
        get_base_posteriors_dir,
        get_save_and_load_dirs,
        get_target_idx
    )
    from data.cumulants import (
        Dataset, 
        get_data, 
        get_linear_compressor, 
        get_datavector, 
        get_prior, 
        get_parameter_strings
    )
    from data.pdfs import get_multi_z_bulk_pdf_fisher_forecast
    from configs.args import (
        get_cumulants_sbi_args, 
        get_cumulants_multi_z_args
    )
    from affine import affine_sample

    jax.clear_caches()

    """
        Loop through seeds, getting configs for ensembles for bulk and bulk + tails
        over all redshifts, loading posteriors from them 
    """
    
    def get_posterior_object(posterior_file):
        # Posterior object contains samples, log prob, Finv, summary, ...
        PosteriorTuple = namedtuple("PosteriorTuple", posterior_file.files)
        print(posterior_file.files) 
        posterior_tuple = PosteriorTuple(
            *(posterior_file[key] for key in posterior_file.files)
        )
        return posterior_tuple

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", 
        "--seed", 
        type=int, 
        help="Seed for random number generation.", 
        default=0
    )
    parser.add_argument(
        "-l",
        "--linearised", 
        default=True,
        action=argparse.BooleanOptionalAction, 
        help="Linearised model for datavector."
    )
    parser.add_argument(
        "-c",
        "--compression", 
        default="linear",
        choices=["linear", "nn", "nn-lbfgs"],
        type=str,
        help="Compression with neural network or MOPED."
    )
    parser.add_argument(
        "-p",
        "--pre-train", 
        default=False,
        action=argparse.BooleanOptionalAction, 
        help="Pre-train (only) when using non-linearised model for datavector. Pre-train on linearised simulations."
    )
    parser.add_argument(
        "-o", 
        "--order_idx",
        default=[0, 1, 2],
        nargs="+", 
        type=int,
        help="Indices of variance, skewness and kurtosis sample cumulants."
    )
    parser.add_argument(
        "-t",
        "--sbi_type", 
        default="nle",
        choices=["nle", "npe"],
        type=str,
        help="Method of SBI: neural likelihood (NLE) or posterior (NPE)."
    )
    parser.add_argument(
        "-f",
        "--freeze-parameters", 
        default=False,
        action=argparse.BooleanOptionalAction, 
        help="Freeze parameters not in [Om, s8] to their fixed values, in hypercube simulations."
    )
    ARGS = parser.parse_args()

    # General constants
    data_dir, _, _ = get_save_and_load_dirs()

    (
        _, _, _, alpha, lower, upper, parameter_strings, *_
    ) = get_quijote_parameters()

    figs_dir = os.path.join(get_base_results_dir(), "figure_one/")
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir, exist_ok=True)

    args = get_cumulants_multi_z_args() # Blueprint args for analysis

    # Plot
    c = ChainConsumer() 

    # Plotting properties for bulk / tails
    plotting_dict = dict(
        bulk=dict(color="b", linestyle="-", shade_alpha=0.5),
        tails=dict(color="r", linestyle="-", shade_alpha=0.5)
    )

    # Args that are shared between bulk and tails SBI analyses/posteriors
    args.seed = ARGS.seed
    args.linearised = ARGS.linearised 
    args.pre_train = ARGS.pre_train
    args.order_idx = ARGS.order_idx
    args.freeze_parameters = ARGS.freeze_parameters

    # Get the bulk Fisher forecast for all redshifts (same whether linearised or not)
    try:
        Finv_bulk_pdfs_all_z = np.load(os.path.join(data_dir, "Finv_bulk_pdfs_all_z.npy"))
    except:
        Finv_bulk_pdfs_all_z = get_multi_z_bulk_pdf_fisher_forecast(args)
        np.save(os.path.join(data_dir, "Finv_bulk_pdfs_all_z.npy"), Finv_bulk_pdfs_all_z)

    if args.freeze_parameters:
        target_idx = get_target_idx()
        Finv_bulk_pdfs_all_z = Finv_bulk_pdfs_all_z[:target_idx, :][:, :target_idx]

    # Loop through bulk / tails (just grab PDF Fisher forecast, no posterior)
    for bulk_or_tails in ["bulk", "tails"]:

        # Multi-z inference concerning the bulk or bulk + tails
        if bulk_or_tails == "tails":
            ensembles_config = ensembles_cumulants_config
        if bulk_or_tails == "bulk" or bulk_or_tails == "bulk_pdf":
            ensembles_config = ensembles_bulk_cumulants_config

        # Force args for posterior to be bulk or tails
        args.bulk_or_tails = bulk_or_tails 

        config = ensembles_config(
            seed=args.seed, # Defaults if run without argparse args
            sbi_type=args.sbi_type, 
            linearised=args.linearised,
            reduced_cumulants=args.reduced_cumulants,
            order_idx=args.order_idx,
            redshifts=args.redshifts,
            compression=args.compression,
            n_linear_sims=args.n_linear_sims,
            freeze_parameters=args.freeze_parameters,
            pre_train=args.pre_train
        )

        # Posterior for bulk/tails for a given seed
        posterior_save_dir = get_multi_z_posterior_dir(config, args)
        print(posterior_save_dir)
        posterior_filename = os.path.join(
            posterior_save_dir, "posterior_{}.npz".format(args.seed)
        )
        posterior_file = np.load(posterior_filename)
        posterior_object = get_posterior_object(posterior_file)

        print(jax.tree.map(lambda x: x.shape, posterior_object))

        title = " " + bulk_or_tails

        # Fisher forecast
        c.add_chain(
            Chain.from_covariance(
                alpha,
                posterior_object.Finv, # NOTE: Get multi redshift Fisher matrix, use a multi-inference config
                columns=parameter_strings,
                name=r"$F_{\Sigma^{-1}}$ " + title,
                color=plotting_dict[bulk_or_tails]["color"],
                linestyle=":",
                shade_alpha=0.
            )
        )

        # Posterior from SBI
        posterior_df = make_df(
            posterior_object.samples, 
            posterior_object.samples_log_prob, 
            parameter_strings=parameter_strings
        )
        c.add_chain(
            Chain(
                samples=posterior_df, name="SBI " + title, 
                color=plotting_dict[bulk_or_tails]["color"],
                linestyle=plotting_dict[bulk_or_tails]["linestyle"],
                shade_alpha=plotting_dict[bulk_or_tails]["shade_alpha"],
            )
        )

        # Compressed datavectors (assuming more than one of them)
        for n, _summary in enumerate(posterior_object.summary):
            c.add_marker(
                location=marker(_summary, parameter_strings), 
                name=r"$\hat{\pi}[\hat{\xi}]$ " + str(n) + title, 
                color=plotting_dict[bulk_or_tails]["color"],
            )

    print(alpha.shape, Finv_bulk_pdfs_all_z.shape) 

    # Fisher forecast for bulk of PDF over all redshifts
    c.add_chain(
        Chain.from_covariance(
            alpha,
            Finv_bulk_pdfs_all_z, 
            columns=parameter_strings,
            name=r"$F_{\Sigma^{-1}}$ " + "bulk pdf",
            color="g",
            linestyle=":",
            shade_alpha=0.
        )
    )

    # True parameters
    c.add_marker(
        location=marker(alpha, parameter_strings), 
        name=r"$\alpha$", 
        color="#7600bc"
    )

    fig = c.plotter.plot()
    plt.savefig(os.path.join(figs_dir, "figure_one_{}.pdf".format(args.seed)))
    plt.close()