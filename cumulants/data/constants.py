import os
from pathlib import Path
import jax.numpy as jnp
import numpy as np

_RESULTS_DIR_ = os.environ.get("RESULTS_DIR", "results") + "/"
DEFAULT_RESOLUTION = int(os.environ.get("DEFAULT_RESOLUTION", 1024))

assert _RESULTS_DIR_ != "/", "RESULTS_DIR={} IS NOT ALLOWED.".format(_RESULTS_DIR_)

# This file is in repo/subfolder/ w.r.t. .git root
ROOT_DIR = str(Path(__file__).resolve().parent.parent.parent) 

# Results and plots directories
RESULTS_DIR = os.path.join(ROOT_DIR, _RESULTS_DIR_) 
POSTERIORS_DIR = os.path.join(ROOT_DIR, _RESULTS_DIR_ + "posteriors/") 
FIGS_DIR = os.path.join(ROOT_DIR, _RESULTS_DIR_) 

# Save and load directories for quijote data
DATA_DIR = os.path.join(ROOT_DIR, "quijote_data/") 
OUT_DIR = DATA_DIR
# QUIJOTE_DIR = "/project/ls-gruen/users/jed.homer/quijote_pdfs/" # Cluster only!
QUIJOTE_DIR = "/project/ls-gruen/users/jed.homer/quijote_pdfs_later/" # Cluster only!
DERIVATIVES_DIR = os.path.join(QUIJOTE_DIR, "derivatives/")


def get_raw_quijote_dir():
    return QUIJOTE_DIR # Directory containing Quijote simulation data


def get_save_and_load_dirs():
    return DATA_DIR, DERIVATIVES_DIR, FIGS_DIR


def get_base_results_dir():
    return RESULTS_DIR


def get_base_posteriors_dir():
    return POSTERIORS_DIR


def get_cumulant_names(include_m0_m1=False):
    cumulant_names = [
        r"$\langle \delta^2 \rangle_c$", 
        r"$\langle \delta^3 \rangle_c$",
        r"$\langle \delta^4 \rangle_c$"
    ]
    if include_m0_m1:
        cumulant_names = [
            r"$\langle \delta^0 \rangle_c$",
            r"$\langle \delta^1 \rangle_c$"
        ] + cumulant_names
    return cumulant_names


if DEFAULT_RESOLUTION == 1024:
    ALL_RADII = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
else:
    ALL_RADII = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
ALL_REDSHIFTS = [0., 0.5, 1., 2., 3.]
REDSHIFT_STRINGS = ["0", "0.5", "1", "2", "3"] # Quijote filename strings

RESOLUTION = DEFAULT_RESOLUTION

PARAMETER_STRINGS = [
    r"$\Omega_m$", r"$\Omega_b$", r"$h_m$", r"$n_s$", r"$\sigma_8$"
]

ALPHA = np.array([0.3175, 0.049, 0.6711, 0.9624, 0.834])

# Prior bounds
LOWER = np.array([0.10, 0.03, 0.50, 0.80, 0.60])
UPPER = np.array([0.50, 0.07, 0.90, 1.20, 1.00])

# Minus derivative is first, then plus derivative
PARAMETER_DERIVATIVE_STRINGS = [
  ["Om_m", "Om_p"], 
  ["Ob2_m", "Ob2_p"], # Larger dtheta for Ob
  ["h_m",  "h_p"], 
  ["ns_m", "ns_p"], 
  ["s8_m", "s8_p"]
]

# Derivative: dS/dp = (S(p + dp) - S(p - dp)) / 2dp
# > below are 2dp values for dOm, dOb, dh, dn_s, ds8
DPARAMS = np.array(
    [
        0.3275 - 0.3075, 
        0.051 - 0.047, 
        0.6911 - 0.6511, 
        0.9824 - 0.9424, 
        0.849 - 0.819
    ]
)

# Bins of PDF; edge and middle values, 100 log-spaced bins in rho
DELTA_BIN_EDGES = np.geomspace(1e-2, 1e2, num=100)
D_DELTAS = DELTA_BIN_EDGES[1:] - DELTA_BIN_EDGES[:-1]


def get_target_idx():
    idx = jnp.array([0, 4]) # Om, s8; ignoring h, n_s, Ob
    print("TARGET_IDX:", [PARAMETER_STRINGS[_] for _ in idx])
    return idx


def get_quijote_parameters():
    return (
        ALL_RADII,
        ALL_REDSHIFTS,
        RESOLUTION,                   # Quijote mesh high-resolution=1024
        ALPHA,                        # Fiducial cosmology in Quijote
        LOWER,
        UPPER,
        PARAMETER_STRINGS,            # Quijote cosmology parameter strings
        REDSHIFT_STRINGS,             # Quijote redshift strings "0.", "0.5", ...
        PARAMETER_DERIVATIVE_STRINGS, # Quijote derivatives e.g. "Ob2_m" or "Ob2_p"
        DPARAMS,                      # Changes in parameters for derivatives
        None,                         # Bin centres for PDF density 1+delta
        DELTA_BIN_EDGES,              # Bin edges
        D_DELTAS                      # Bin width
    )


def get_alpha_and_parameter_strings():
    return ALPHA, PARAMETER_STRINGS


def get_delta_bin_widths():
    return D_DELTAS


F_PLANCK = jnp.array(
    [
        [ 2.13080592e+05, -1.20573100e+06,  1.48016560e+05, 2.93458548e+04, -2.06713944e+04, -1.65766154e+03],
        [-1.20573100e+06,  1.35133806e+07, -2.18303421e+05, -1.26270926e+04, -1.61514959e+04, -5.92496230e+04],
        [ 1.48016560e+05, -2.18303421e+05,  2.03038428e+05, -1.38685185e+04, -1.61497519e+04, -1.55300001e+03],
        [ 2.93458548e+04, -1.26270926e+04, -1.38685185e+04, 1.02172866e+05, -6.36387231e+03, -5.65461481e+03],
        [-2.06713944e+04, -1.61514959e+04, -1.61497519e+04, -6.36387231e+03,  2.29958884e+04,  6.30418193e+03],
        [-1.65766154e+03, -5.92496230e+04, -1.55300001e+03, -5.65461481e+03,  6.30418193e+03,  2.27796421e+03]
    ]
)


def get_F_planck():
    return F_PLANCK[:-1, :-1] # Drop M_nu


def get_Finv_planck():
    return jnp.linalg.inv(get_F_planck())


def get_sobol_prior_limits():
    # Not quite the same as previous hypercube
    lower = np.array([0.10, 0.02, 0.50, 0.80, 0.60])
    upper = np.array([0.50, 0.08, 0.90, 1.20, 1.00])
    return lower, upper


def get_sobol_scales():
    box_size = 1000.0 # Mpc/h
    grid = 256
    d = box_size / grid

    scale_numbers = np.array([3., 5., 7., 9., 11., 13., 15., 17.])
    scales = scale_numbers * d / 2 # Mpc/h, for accurate results the mesh is 1/10 of this
    return scales


def get_sobol_ingredients():
    parameter_strings = [
        r"$\Omega_m$", r"$\Omega_b$", r"$h_m$", r"$n_s$", r"$\sigma_8$"
    ]

    alpha = np.array([0.3175, 0.049, 0.6711, 0.9624, 0.834])

    lower, upper = get_sobol_prior_limits()

    return parameter_strings, alpha, (lower, upper)