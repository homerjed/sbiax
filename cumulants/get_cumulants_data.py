import time
import os
import numpy as np 
from tqdm.auto import trange

from constants import get_quijote_parameters, get_save_and_load_dirs, get_raw_quijote_dir

"""
    Get fiducial and latin moments, plus/minus derivatives
    from raw Quijote files.
    -> raw files on cluster only
"""

quijote_dir = get_raw_quijote_dir()

(
    data_dir, 
    derivatives_dir, 
    figs_dir 
) = get_save_and_load_dirs()

(
    R_values,
    redshifts,
    resolution,
    alpha,
    *_,
    parameter_strings,
    redshift_strings,
    parameter_derivative_names,
    dparams,
    deltas,
    delta_bin_edges,
    D_deltas 
) = get_quijote_parameters()

n_fiducials         = 15_000
n_latins            = 2000
n_derivatives       = 500
n_bins_pdf          = 99
n_params            = alpha.size
n_redshifts         = len(redshifts)
n_scales            = len(R_values)
n_moments_start     = 2 # Skip first 2 of 5
n_moments_calculate = 3 # Take last 3 of 5
z_idx               = [redshifts.index(z) for z in redshifts] # Chosen scales/redshifts

"""
    Get fiducials
"""

ALL_FIDUCIAL_CUMULANTS = np.zeros((n_redshifts, n_fiducials, n_scales, n_moments_calculate))

t0 = time.time()
# Loop through integer -> string realisation folders to avoid other realisation types
for n in trange(n_fiducials, desc="Fiducials"):
    for n_z, z in enumerate(redshifts):

        z_string = redshift_strings[n_z]

        try:
            cumulants_n = np.loadtxt(
                os.path.join(
                    quijote_dir, 
                    "fiducial/", 
                    str(n) + "/", 
                    f"moments_PDF_m_z={z_string}.txt"
                )
            )
            assert cumulants_n.shape == (7, 5)

            # Print in a table format
            headers = ["R", "var1", "var2", "skew", "kurt"]
            print(f"{headers[0]:<6} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10} {headers[4]:<10}")  # Print header
            print("-" * 50)  # Separator
            for row in cumulants_n:
                print(f"{row[0]:<6.2f} {row[1]:<10.3E} {row[2]:<10.3E} {row[3]:<10.3E} {row[4]:<10.3E}")

            print(cumulants_n[:, n_moments_start:])
            assert cumulants_n[:, n_moments_start:].shape == (7, 3)

        except FileNotFoundError:
            print("bad", n)
            pass

        ALL_FIDUCIAL_CUMULANTS[n_z, n, :, :] = cumulants_n[:, n_moments_start:]
        
np.save(
    os.path.join(data_dir, "ALL_FIDUCIAL_CUMULANTS.npy"), 
    ALL_FIDUCIAL_CUMULANTS
)
print("\nDONE.")

"""
    Get latins
"""

ALL_LATIN_CUMULANTS = np.zeros((n_redshifts, n_latins, n_scales, n_moments_calculate))

t0 = time.time()
# Loop through integer -> string realisation folders to avoid other realisation types
for n in trange(n_latins, desc="Latins"):
    for n_z, z in enumerate(redshifts):

        z_string = redshift_strings[n_z]

        try:
            cumulants_n = np.loadtxt(
                os.path.join(
                    quijote_dir, 
                    "latin_hypercube/", 
                    str(n) + "/", 
                    f"moments_PDF_m_z={z_string}.txt"
                )
            )
        except FileNotFoundError:
            print("bad", n)
            pass

        assert cumulants_n.shape == (7, 5)

        ALL_LATIN_CUMULANTS[n_z, n, :, :] = cumulants_n[:, n_moments_start:]

np.save(
    os.path.join(data_dir, "ALL_LATIN_CUMULANTS.npy"), 
    ALL_LATIN_CUMULANTS
)
print("\nDONE.")

"""
    Get derivatives
"""

# Derivatives at each R, for all parameters, at redshift z
derivatives = np.zeros((n_derivatives, n_redshifts, n_params, n_scales, 2, n_moments_calculate))
bad_idx = []

# Each realisation's derivative
for n_d in trange(n_derivatives, desc="Derivatives"):

    for n_z, z in enumerate(redshifts):

        z_string = redshift_strings[n_z]

        # Each parameter
        for p, parameter_derivative in enumerate(parameter_derivative_names):

            # Each plus/minus derivative (minus derivative is first, then plus derivative)
            for pm, p_or_m in enumerate(parameter_derivative):

                # Directory of a realisation with all parameter derivatives in
                derivative_dir = os.path.join(derivatives_dir, p_or_m, str(n_d) + "/")

                for R, R_value in enumerate(R_values):
                    # Derivative of pdf w.r.t. param
                    try:
                        d_cumulant_dparam = np.loadtxt(
                            os.path.join(derivative_dir, f"moments_PDF_m_z={z_string}.txt"), 
                            unpack=True
                        ).T

                        assert d_cumulant_dparam.shape == (7, 5)

                        # n-th derivative of parameter p, pdf at scale R, +/- derivative
                        derivatives[n_d, n_z, p, :, pm, :] = d_cumulant_dparam[:, n_moments_start:] # Assuming ordered by moment n

                    except ValueError:
                        bad_idx.append(n_d)
                        pass

# Once acquired plus/minus derivatives
if len(bad_idx) > 0:
    derivatives = np.delete(derivatives, bad_idx, axis=0)

# Derivatives: plus and minus, all scales and redshifts
np.save(
    os.path.join(data_dir, "cumulants_derivatives_plus_minus.npy"), 
    derivatives
)
print("Derivatives:", derivatives.shape)
print("Done.")