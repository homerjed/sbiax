import time
import os
import numpy as np 
from tqdm.auto import trange

from constants import get_quijote_parameters, get_save_and_load_dirs, get_raw_quijote_dir

"""
    Get fiducial and latin pdfs, plus/minus derivatives
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


# Chosen scales/redshifts
resolution          = 1024
n_bins_pdf          = 99
n_fiducials         = 15_000
n_latins            = 2000
n_derivatives       = 500
n_bins_pdf          = 99
n_params            = alpha.size
n_redshifts         = len(redshifts)
n_scales            = len(R_values)
z_idx               = [redshifts.index(z) for z in redshifts] # Chosen scales/redshifts

"""
    Get fiducials
"""

ALL_FIDUCIAL_PDFS = np.zeros((n_redshifts, n_fiducials, n_scales, n_bins_pdf))

t0 = time.time()
for n in trange(n_fiducials, desc="Fiducials"):
    for n_z, z in enumerate(redshifts):
        for n_R, R in enumerate(R_values):

            if resolution == 1024:
                filename_ = f"PDF_m_{resolution}_{R}_z={redshift_strings[n_z]}.txt"
            else:
                filename_ = f"PDF_m_{R}_z={z}.txt"

            deltas, pdfs_z_R = np.loadtxt(
                os.path.join(quijote_dir, "fiducial/", f"{n}/", filename_),
                unpack=True
            )

            ALL_FIDUCIAL_PDFS[n_z, n, n_R, :] = pdfs_z_R / D_deltas

            print(
                f"\r fiducials: n={n:05d} " + 
                f"z={z} R={R} " + 
                f"t={(time.time() - t0) / 60.:.2f} mins", 
                end=""
            )

np.save(
    os.path.join(data_dir, f"ALL_FIDUCIAL_PDFS.npy"), 
    ALL_FIDUCIAL_PDFS
)

print("\nDONE FIDUCIALS.")

"""
    Get latins 
"""

ALL_LATIN_PDFS = np.zeros((n_redshifts, n_latins, n_scales, n_bins_pdf))

t0 = time.time()
for n in trange(n_latins, desc="Latins"):
    for n_z, z in enumerate(redshifts):
        for n_R, R in enumerate(R_values):

            if resolution == 1024:
                filename_ = f"PDF_m_{resolution}_{R}_z={redshift_strings[n_z]}.txt"
            else:
                filename_ = f"PDF_m_{R}_z={z}.txt"

            delta, pdfs_z_R = np.loadtxt(
                os.path.join(quijote_dir, "latin_hypercube/", f"{n}/", filename_),
                unpack=True
            )

            # Normalise PDFs (they measure p(delta_i) * D_delta_i)
            ALL_LATIN_PDFS[n_z, n, n_R, :] = pdfs_z_R / D_deltas

            t_mins = (time.time() - t0) / 60.
            print(f"\r latins: n={n:05d} z={z} R={R} t={t_mins:.2f} mins", end="")

np.save(
    os.path.join(data_dir, f"ALL_LATIN_PDFS.npy"), 
    ALL_LATIN_PDFS
)

print("\nDONE LATINS.")

"""
    Get derivatives
"""

# Derivatives at each R, for all parameters, at redshift z
derivatives = np.zeros((n_derivatives, n_redshifts, n_params, n_scales, 2, n_bins_pdf))
bad_idx = []

# Each realisation's derivative
for n_d in trange(n_derivatives, desc="Derivatives"):

    for n_z, z in enumerate(redshifts):

        z_string = redshift_strings[n_z]

        # Each parameter
        for p, parameter_derivative in enumerate(parameter_derivative_names):

            # Each plus/minus derivative
            for pm, p_or_m in enumerate(parameter_derivative):

                # Directory of a realisation with all parameter derivatives in
                derivative_dir = os.path.join(derivatives_dir, p_or_m, str(n_d) + "/")

                for R, R_value in enumerate(R_values):
                    # Derivative of pdf w.r.t. param
                    msg = f"\rz={redshifts[0]:.1f} ({z_string}) R={R_value} n_d={n_d:04d}"
                    try:
                        if resolution == 1024:
                            _filename = f"PDF_m_{resolution}_{R_value}_z={z_string}.txt"
                        else: 
                            _filename = f"PDF_m_{R_value}_z={z_string}.txt"

                        _filename = os.path.join(derivative_dir, _filename)

                        # correctly unpacked 
                        deltas, dpdf_dparam = np.loadtxt(_filename, unpack=True)

                        # n-th derivative of parameter p, pdf at scale R, +/- derivative
                        derivatives[n_d, n_z, p, R, pm, :] = dpdf_dparam  / D_deltas

                    except ValueError:
                        msg += f" (BAD PDF {n_d:04d})"
                        bad_idx.append(n_d)
                        pass

                print(msg, end="")

# Once acquired plus/minus derivatives
if len(bad_idx) > 0:
    derivatives = np.delete(derivatives, bad_idx, axis=0)

# Derivatives: plus and minus, all scales and redshifts
np.save(
    os.path.join(data_dir, f"pdfs_derivatives_plus_minus.npy"), 
    derivatives
)
print("Derivatives:", derivatives.shape)
print("Done.")