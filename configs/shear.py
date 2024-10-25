import jax 
import ml_collections


def shear():
    config = ml_collections.ConfigDict()

    config.seed            = 0
    config.sbi_type        = "nle"
    config.exp_name        = "shear"

    # Data
    config.dataset_name    = "shear" 
    config.compression     = "linear"
    config.n_s             = 2_000

    # NDEs
    config.model = model = ml_collections.ConfigDict()
    model.model_type       = "cnf"
    model.width_size       = 5#8
    model.depth            = 0
    model.activation       = jax.nn.tanh
    model.dropout_rate     = 0.
    model.dt               = 0.08#0.1
    model.t1               = 1.
    model.solver           = "Heun"
    model.exact_log_prob   = True
    model.use_scaling      = False

    # Posterior sampling
    config.use_ema         = True
    config.n_steps         = 200
    config.n_walkers       = 1000
    config.burn            = int(0.1 * config.n_steps)

    # Optimisation hyperparameters
    config.start_step      = 0
    config.n_epochs        = 10_000
    config.n_batch         = 40 #50
    config.patience        = 60 #20
    config.lr              = 1e-4
    config.opt             = "adamw" 
    config.opt_kwargs      = {}

    return config