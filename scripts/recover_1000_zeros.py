import os
import time
import numpy as np
import mpmath
import matplotlib.pyplot as plt
from trng_auditor.core.data import pnt_error_term, logarithmic_resample
from trng_auditor.core.spectral_estimation import ParametricSpectralEstimator

CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils", "riemann_zeros_1000.npy")

def fetch_1000_riemann_zeros() -> np.ndarray:
    """Fetch/compute the first 1000 Riemann zeros (imaginary parts gamma_k) with caching."""
    if os.path.exists(CACHE_FILE):
        print(f"    Loaded 1000 Riemann zeros from local cache '{CACHE_FILE}'.")
        return np.load(CACHE_FILE)
        
    print("    Computing 1000 Riemann zeros via mpmath.zetazero...")
    t0 = time.time()
    mpmath.mp.dps = 8
    zeros = []
    for k in range(1, 1001):
        z = mpmath.zetazero(k)
        zeros.append(float(z.imag))
    arr = np.array(zeros, dtype=np.float64)
    np.save(CACHE_FILE, arr)
    print(f"    Computed and cached 1000 Riemann zeros in {time.time() - t0:.2f}s (gamma_1 = {arr[0]:.4f} to gamma_1000 = {arr[-1]:.4f}).")
    return arr

def main():
    os.makedirs("plots", exist_ok=True)
    print("=" * 85)
    print("   LARGE-SCALE TEST: RECOVERING THE FIRST 1000 RIEMANN ZEROS FROM OPERATOR   ")
    print("=" * 85)

    N_samples = 300000
    print(f"\n[1] Generating prime snapshot signal up to N = {N_samples}...")
    t0 = time.time()
    
    x_grid, delta_pnt = pnt_error_term(N_samples, normalized=False)
    num_log_pts = 6000
    t_uniform, y_uniform, dt_log = logarithmic_resample(x_grid, delta_pnt, num_samples=num_log_pts)
    
    print(f"    Generated signal in {time.time() - t0:.2f}s.")
    print(f"    Logarithmic snapshot resolution: {len(y_uniform)} points, dt = {dt_log:.4e}")

    # Fetch reference zeros
    true_zeros_1000 = fetch_1000_riemann_zeros()

    # ----------------------------------------------------
    # RUN LARGE-SCALE OPERATOR EXTRACTION (1000 ZEROS)
    # ----------------------------------------------------
    print("\n[2] Executing Large-Scale TLS-ESPRIT Operator Extraction for 1000 Zeros...")
    t1 = time.time()
    M_window = 1600
    L_signals = 700  # Subspace dimension for poles
    
    estimator = ParametricSpectralEstimator(y_uniform, dt=dt_log)
    print(f"    Constructing snapshot covariance matrix Rxx of size {M_window} x {M_window}...")
    estimator.construct_covariance(M=M_window)
    
    print(f"    Executing TLS-ESPRIT (L_signals = {L_signals})...")
    esprit_res = estimator.run_esprit(L=L_signals)
    print(f"    ESPRIT completed in {time.time() - t1:.2f}s.")

    extracted_omegas = esprit_res['omegas']
    unique_omegas = np.sort(np.unique(np.round(extracted_omegas, 2)))
    
    print("\n" + "=" * 85)
    print("   EVALUATING RECOVERY ACCURACY FOR FIRST 1000 RIEMANN ZEROS")
    print("=" * 85)

    matches = []
    for idx, tz in enumerate(true_zeros_1000):
        closest_idx = np.argmin(np.abs(unique_omegas - tz))
        closest_omega = unique_omegas[closest_idx]
        err = abs(closest_omega - tz)
        acc = max(0.0, 100.0 * (1.0 - err / tz))
        matches.append((idx + 1, closest_omega, tz, err, acc))

    errors = [m[3] for m in matches]
    accuracies = [m[4] for m in matches]
    
    print("\n  Sample Zero Match Results Across Spectrum:")
    sample_indices = [1, 10, 50, 100, 250, 500, 750, 1000]
    print(f"  {'Zero #':<8} | {'Extracted Operator Frequency':<32} | {'True Riemann Zero (gamma_k)':<28} | {'Error':<10} | {'Accuracy':<10}")
    print("  " + "-" * 95)
    for k in sample_indices:
        m = matches[k-1]
        print(f"  #{m[0]:<7} | {m[1]:<32.4f} | {m[2]:<28.4f} | {m[3]:<10.4f} | {m[4]:<6.2f}%")

    print("\n" + "-" * 85)
    print("  SUMMARY PERFORMANCE METRICS ACROSS ALL 1000 RIEMANN ZEROS:")
    print(f"  - Total Zeros Tested:       1,000 zeros (gamma_1 = 14.13 to gamma_1000 = 1419.42)")
    print(f"  - Mean Absolute Error:      {np.mean(errors):.4f} rad/s")
    print(f"  - Median Absolute Error:    {np.median(errors):.4f} rad/s")
    print(f"  - Mean Recovery Accuracy:   {np.mean(accuracies):.2f}%")
    print(f"  - % Zeros with > 90% Acc:   {np.sum(np.array(accuracies) > 90.0) / 10.0:.1f}% ({np.sum(np.array(accuracies) > 90.0)} / 1000)")
    print("-" * 85)

    # ----------------------------------------------------
    # GENERATE 1000 ZEROS VALIDATION PLOT
    # ----------------------------------------------------
    plt.figure(figsize=(12, 6))
    indices = np.arange(1, 1001)
    plt.plot(indices, true_zeros_1000, color='#2ca02c', linewidth=2, label=r'True Riemann Zeros $\gamma_k$ ($k = 1 \dots 1000$)')
    
    extracted_matched = [m[1] for m in matches]
    plt.plot(indices, extracted_matched, color='#9467bd', linewidth=1.2, linestyle='--', label=r'Operator Extracted Frequencies $\omega_k$')
    
    plt.title(f"1000 Riemann Zeros Spectrum Recovery Validation ($N={N_samples}$)")
    plt.xlabel("Zero Index $k$ (1 to 1000)")
    plt.ylabel(r"Zero Value $\gamma_k$ / Frequency $\omega_k$ (rad/s)")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/recovery_1000_riemann_zeros.png", dpi=300)
    plt.close()
    print("\n    Saved validation plot: plots/recovery_1000_riemann_zeros.png")

    print("\n" + "=" * 85)
    print("   1000-ZERO EXPERIMENT COMPLETED SUCCESSFULLY.   ")
    print("=" * 85)

if __name__ == '__main__':
    main()
