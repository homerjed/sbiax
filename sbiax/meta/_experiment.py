import argparse


def get_n_sims_values():
    return [
        500, 600, 700, 800, 900, 1000, 1200, 1400, 
        1600, 1800, 2000, 3000, 4000, 6000, 8000, 
        10_000, 20_000 
    ]


def get_sbi_types():
    return "NPE", "NLE"


def get_nde_types():
    return "CNF", "MAF"


def get_args():
    parser = argparse.ArgumentParser(
        description="Run Dodelson experiment for linearised model, either NLE or NPE."
    )
    parser.add_argument(
        "-n", '--n_sims', type=int, help="Number of sims.", default=1000
    )
    parser.add_argument(
        "-e", "--n_ndes", type=int, help="Number of flows.", default=1
    )
    parser.add_argument(
        "-o", "--n_obs", type=int, help="Number of mock-data realisations to sample.", default=1
    )
    parser.add_argument(
        "-m", "--model_type", type=str, help="Type of model density estimator", default="CNF"
    )
    parser.add_argument(
        "-s", "--seed", type=int, help="Seed for random number generation.", default=0
    )
    parser.add_argument(
        "-d", "--exp_name", type=str, help="Directory to save runs in.", default=""
    )
    parser.add_argument(
        "--perfect-score", 
        action=argparse.BooleanOptionalAction, 
        help="Score compressor uses as many n_sims as --n_sims."
    )
    parser.add_argument(
        "--eig-sampling", 
        action=argparse.BooleanOptionalAction, 
        help="Sample simulation parameters parallel to Fisher eigendirections or not."
    )
    args = parser.parse_args()
    return args


def save_posterior_objects(
    samples,
    samples_log_prob,
    alpha__log_prob,
    alpha_,
    X_,
    *,
    results_dir
):
    np.save(
        os.path.join(
            results_dir, "posteriors/alpha_samples.npy"
        ), 
        samples
    )
    np.save(
        os.path.join(
            results_dir, "posteriors/alpha_samples_log_prob.npy"
        ), 
        samples_log_prob
    )
    np.save(
        os.path.join(
            results_dir, "posteriors/alpha__log_prob.npy"
        ), 
        alpha__log_prob
    )
    np.save(
        os.path.join(
            results_dir, "data/alpha_.npy"
        ), 
        alpha_
    )
    np.save(
        os.path.join(
            results_dir, "data/alpha_datavector.npy"
        ), 
        X_ 
    )