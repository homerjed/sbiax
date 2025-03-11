import os 
import numpy as np

ALL_RADII = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
ALL_REDSHIFTS = [0., 0.5, 1., 2., 3.]
REDSHIFT_STRINGS = ["0", "0.5", "1", "2", "3"] # Quijote filename strings

RESOLUTION = 1024 # NOTE: are hypercube realisations high resolution also?

PARAMETER_STRINGS = [
    r"$\Omega_m$", r"$\Omega_b$", r"$h_m$", r"$n_s$", r"$\sigma_8$"
]

ALPHA = np.array([0.3175, 0.049, 0.6711, 0.9624, 0.834])

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

# dOm, dOb, dh, dn_s, ds8
DPARAMS = np.array(
    [
        0.3275 - 0.3075, 
        0.051 - 0.047, 
        0.6911 - 0.6511, 
        0.9824 - 0.9424, 
        0.849 - 0.819
    ]
)

DELTA_BIN_EDGES = np.geomspace(1e-2, 1e2, num=100) 
D_DELTAS = DELTA_BIN_EDGES[1:] - DELTA_BIN_EDGES[:-1]

# Save and load directories for quijote data
DERIVATIVES_DIR = "/project/ls-gruen/users/jed.homer/quijote_pdfs/derivatives/"
OUT_DIR = "/project/ls-gruen/users/jed.homer/quijote_pdfs/data/" 
DATA_DIR = "/project/ls-gruen/users/jed.homer/quijote_pdfs/data/"
FIGS_DIR = "/project/ls-gruen/users/jed.homer/sbipdf/results/figs/" # General plots

# Common to all moments (values of delta in the middle of the bins?)
DELTAS = np.load(os.path.join(DATA_DIR, "deltas.npy"))


def get_save_and_load_dirs():
    return DATA_DIR, DERIVATIVES_DIR, FIGS_DIR


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
        DELTAS,                       # Bin centres for PDF density 1+delta
        DELTA_BIN_EDGES,              # Bin edges
        D_DELTAS                      # Bin widths
    )


def get_alpha_and_parameter_strings():
    return ALPHA, PARAMETER_STRINGS