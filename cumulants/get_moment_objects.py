import time, os
import numpy as np 

from constants import get_quijote_parameters, get_save_and_load_dirs

"""
    Get fiducial and latin moments, plus/minus derivatives
    from raw Quijote files.
"""

    """ 
        Scale derivatives like this, use autodiff for jacobian: dc/dp = dc/dm * dm/dp 
        - Quijote calculates: (smoothing, var, mom2, mom3, mom4) where mom's are sample cumulants (kstats)
    """

def get_reduced_cumulants(cumulants):
    var = moments[:, 2]
    reduced_cumulants = np.zeros_like(cumulants)
    for n in range(2, 5):
        reduced_cumulants[:, n] = cumulants[:, n] / (var ** (n - 2))
    return cumulants 


quijote_dir = "/project/ls-gruen/users/jed.homer/quijote_pdfs/" 

(
    data_dir, 
    out_dir, 
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

ALL_FIDUCIAL_MOMENTS = np.zeros(
    (n_redshifts, n_fiducials, n_scales, n_moments_calculate)
)
ALL_FIDUCIAL_CUMULANTS = np.zeros_like(ALL_FIDUCIAL_MOMENTS)

t0 = time.time()
# Loop through integer -> string realisation folders to avoid other realisation types
for n in range(n_fiducials):
    for n_z, z in enumerate(redshifts):
        z_string = redshift_strings[n_z]
        try:
            moments_n = np.loadtxt(
                os.path.join(
                    quijote_dir, 
                    "fiducial/", 
                    str(n) + "/", 
                    f"moments_PDF_m_z={z_string}.txt"
                )
            )
            cumulants_n = get_reduced_cumulants(moments_n)
            print("\n", moments_n)
        except FileNotFoundError:
            print("bad", n)
            pass

        ALL_FIDUCIAL_MOMENTS[n_z, n, :] = moments_n[:, n_moments_start:]
        ALL_FIDUCIAL_CUMULANTS[n_z, n, :] = cumulants_n[:, n_moments_start:]
        
        print(
            f"\rfiducials: n={n:05d} z={z} t={(time.time() - t0) / 60.:.2f} mins", end=""
        )

np.save(
    os.path.join(out_dir, f"ALL_FIDUCIAL_MOMENTS.npy"), 
    ALL_FIDUCIAL_MOMENTS
)
np.save(
    os.path.join(out_dir, f"ALL_FIDUCIAL_CUMULANTS.npy"), 
    ALL_FIDUCIAL_CUMULANTS
)
print("\nDONE.")

"""
    Get latins
"""

ALL_LATIN_MOMENTS = np.zeros(
    (n_redshifts, n_latins, n_scales, n_moments_calculate)
)
ALL_LATIN_CUMULANTS = np.zeros_like(ALL_LATIN_MOMENTS)

t0 = time.time()
# Loop through integer -> string realisation folders to avoid other realisation types
for n in range(n_latins):
    for n_z, z in enumerate(redshifts):
        z_string = redshift_strings[n_z]
        try:
            moments_n = np.loadtxt(
                os.path.join(
                    quijote_dir, 
                    "latin_hypercube/", 
                    str(n) + "/", 
                    f"moments_PDF_m_z={z_string}.txt"
                )
            )
            cumulants_n = get_reduced_cumulants(moments_n)
        except FileNotFoundError:
            print("bad", n)
            pass

        ALL_LATIN_MOMENTS[n_z, n, :] = moments_n[:, n_moments_start:]
        ALL_LATIN_CUMULANTS[n_z, n, :] = cumulants_n[:, n_moments_start:]
        
        print(
            f"\r latins: n={n:05d} z={z} t={(time.time() - t0) / 60.:.2f} mins", end=""
        )

np.save(
    os.path.join(out_dir, f"ALL_LATIN_MOMENTS.npy"), 
    ALL_LATIN_MOMENTS
)
np.save(
    os.path.join(out_dir, f"ALL_LATIN_CUMULANTS.npy"), 
    ALL_LATIN_CUMULANTS
)
print("\nDONE.")

"""
    Get derivatives
"""

# Derivatives at each R, for all parameters, at redshift z
derivatives = np.zeros(
    (n_derivatives, n_params, n_redshifts, n_scales, 2, n_moments_calculate)
)
derivatives_cumulants = np.zeros_like(derivatives)
bad_idx = []

# Each realisation's derivative
for n_d in range(n_derivatives):

    for n_z, z in enumerate(redshifts):
        # Each parameter
        z_string = redshift_strings[n_z]

        for p, parameter_derivative in enumerate(parameter_derivative_names):

            # Each plus/minus derivative (minus derivative is first, then plus derivative)
            for pm, p_or_m in enumerate(parameter_derivative):

                # Directory of a realisation with all parameter derivatives in
                derivative_dir = os.path.join(derivatives_dir, p_or_m, str(n_d) + "/")

                for R, R_value in enumerate(R_values):
                    # Derivative of pdf w.r.t. param
                    msg = f"\rderivatives: z={redshifts[0]:.1f} ({z_string}) R={R_value} n_d={n_d:04d}"
                    try:
                        d_cumulant_dparam = np.loadtxt(
                            os.path.join(
                                derivative_dir, f"moments_PDF_m_z={z_string}.txt"
                            ), 
                            unpack=True
                        ).T
                        d_reducedcumulant_dparam = get_reduced_cumulants(d_cumulant_dparam) # NOTE: why are cumulant/moment derivatives the same?

                        # n-th derivative of parameter p, pdf at scale R, +/- derivative
                        derivatives[n_d, p, n_z, :, pm, :] = d_cumulant_dparam[:, 2:] # Assuming ordered by moment n
                        derivatives_cumulants[n_d, p, n_z, :, pm, :] = d_reducedcumulant_dparam[:, 2:] 
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
    os.path.join(out_dir, f"moments_derivatives_plus_minus.npy"), 
    derivatives
)
np.save(
    os.path.join(out_dir, f"cumulants_derivatives_plus_minus.npy"), 
    derivatives_cumulants
)
print("Derivatives:", derivatives.shape)
print("Done.")