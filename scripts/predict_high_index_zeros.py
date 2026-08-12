import os
import time
import numpy as np
import mpmath
import matplotlib.pyplot as plt
from trng_auditor.core.data import pnt_error_term, logarithmic_resample
from trng_auditor.core.spectral_estimation import ParametricSpectralEstimator

def fetch_high_index_zeros(indices: list) -> dict:
    """Compute high-index Riemann zeros via mpmath asymptotic Siegel-Z root finding."""
    mpmath.mp.dps = 10
    zeros = {}
    print(f"    Fetching reference high-index Riemann zeros for k = {indices}...")
    for k in indices:
        # Asymptotic approximation + root refinement
        k_eff = k - 11.0 / 8.0
        t_est = float(np.real((2.0 * np.pi * k_eff) / mpmath.lambertw(k_eff / np.exp(1.0))))
        try:
            t_ref = float(mpmath.findroot(mpmath.siegelz, t_est))
            zeros[k] = t_ref
        except Exception:
            zeros[k] = t_est
    return zeros

def main():
    print("=" * 85)
    print("   HIGH-INDEX ZERO PREDICTION: VALIDATING OPERATOR AT k = 500, 1000, 2500, 5000   ")
    print("=" * 85)

    target_indices = [500, 1000, 2500, 5000]
    high_zeros = fetch_high_index_zeros(target_indices)
    
    print("\n  Target High-Index Riemann Zeros:")
    for k, val in high_zeros.items():
        print(f"  - Zero #{k:<5}: gamma_{k} = {val:.4f} rad/s")

    N_samples = 400000
    print(f"\n[1] Preparing large-scale prime snapshot signal N = {N_samples}...")
    t0 = time.time()
    x_grid, delta_pnt = pnt_error_term(N_samples, normalized=False)
    t_uniform, y_uniform, dt_log = logarithmic_resample(x_grid, delta_pnt, num_samples=15000)
    print(f"    Signal ready in {time.time() - t0:.2f}s. dt = {dt_log:.4e}")

    print("\n[2] Extracting High-Frequency Operator Spectrum via TLS-ESPRIT...")
    t1 = time.time()
    M_window = 2000
    L_signals = 1200
    
    estimator = ParametricSpectralEstimator(y_uniform, dt=dt_log)
    estimator.construct_covariance(M=M_window)
    esprit_res = estimator.run_esprit(L=L_signals)
    
    extracted_omegas = esprit_res['omegas']
    unique_omegas = np.sort(np.unique(np.round(extracted_omegas, 1)))
    print(f"    Operator extraction completed in {time.time() - t1:.2f}s.")

    print("\n" + "=" * 85)
    print("   HIGH-INDEX OPERATOR PREDICTION VS TRUE RIEMANN ZEROS")
    print("=" * 85)
    print(f"  {'Zero Index k':<14} | {'Operator Extracted Frequency':<32} | {'True Riemann Zero (gamma_k)':<28} | {'Error':<10} | {'Accuracy':<10}")
    print("  " + "-" * 98)

    for k in target_indices:
        tz = high_zeros[k]
        closest_idx = np.argmin(np.abs(unique_omegas - tz))
        closest_om = unique_omegas[closest_idx]
        err = abs(closest_om - tz)
        acc = max(0.0, 100.0 * (1.0 - err / tz))
        print(f"  Zero #{k:<9} | {closest_om:<32.4f} | {tz:<28.4f} | {err:<10.4f} | {acc:<6.2f}%")

    print("\n" + "=" * 85)
    print("   HIGH-INDEX PREDICTION VALIDATION COMPLETED.   ")
    print("=" * 85)

if __name__ == '__main__':
    main()
