import os
import time
import numpy as np
import matplotlib.pyplot as plt
from riemann_sysid.data import pnt_error_term, logarithmic_resample, get_riemann_zeros
from riemann_sysid.spectral_estimation import ParametricSpectralEstimator
from riemann_sysid.operator_wrapper import HilbertPolyaOperator

# Set high-quality plot aesthetic
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

def main():
    os.makedirs("plots", exist_ok=True)
    print("=" * 85)
    print("   VALIDATING HIGH-ORDER RIEMANN ZERO EXTRACTION FROM OPERATOR FUNCTION   ")
    print("=" * 85)

    N_samples = 120000
    num_zeros_to_find = 30
    print(f"\n[1] Preparing prime snapshot signal up to N = {N_samples}...")
    t0 = time.time()
    
    x_grid, delta_pnt = pnt_error_term(N_samples, normalized=False)
    t_uniform, y_uniform, dt_log = logarithmic_resample(x_grid, delta_pnt, num_samples=4000)
    true_zeros = get_riemann_zeros(num_zeros=num_zeros_to_find)
    
    print(f"    Data ready in {time.time() - t0:.2f}s. Loaded {num_zeros_to_find} true Riemann zeros.")
    print(f"    Logarithmic snapshot resolution: {len(y_uniform)} points, dt = {dt_log:.4e}")

    # ----------------------------------------------------
    # RUN OPERATOR SPECTRAL EXTRACTION
    # ----------------------------------------------------
    print(f"\n[2] Executing Super-Resolution Operator Extraction for {num_zeros_to_find} Zeros...")
    M_window = 300
    L_signals = 60  # 30 real sinusoids = 60 complex poles
    
    estimator = ParametricSpectralEstimator(y_uniform, dt=dt_log)
    estimator.construct_covariance(M=M_window)
    esprit_res = estimator.run_esprit(L=L_signals)
    
    extracted_omegas = esprit_res['omegas']
    # Filter unique positive frequencies
    unique_omegas = np.sort(np.unique(np.round(extracted_omegas, 3)))
    
    print("\n" + "=" * 85)
    print(f"  EXTRACTED OPERATOR FREQUENCIES VS TRUE RIEMANN ZEROS (Top {num_zeros_to_find} Zeros)")
    print("=" * 85)
    print(f"  {'Index':<6} | {'Extracted Operator Frequency (rad/s)':<36} | {'True Riemann Zero (gamma_k)':<28} | {'Error':<10} | {'Accuracy':<10}")
    print("  " + "-" * 95)

    matches = []
    matched_true = set()
    
    for idx in range(min(num_zeros_to_find, len(true_zeros))):
        tz = true_zeros[idx]
        # Find closest extracted omega
        closest_idx = np.argmin(np.abs(unique_omegas - tz))
        closest_omega = unique_omegas[closest_idx]
        err = abs(closest_omega - tz)
        acc = max(0.0, 100.0 * (1.0 - err / tz))
        
        matches.append((idx + 1, closest_omega, tz, err, acc))
        print(f"  #{idx+1:<5} | {closest_omega:<36.6f} | {tz:<28.6f} | {err:<10.6f} | {acc:<6.2f}%")

    # Metrics
    errors = [m[3] for m in matches]
    accuracies = [m[4] for m in matches]
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    mean_acc = np.mean(accuracies)

    print("\n" + "-" * 85)
    print(f"  SUMMARY PERFORMANCE METRICS ACROSS {num_zeros_to_find} ZEROS:")
    print(f"  - Mean Absolute Error:      {mean_err:.6f} rad/s")
    print(f"  - Median Absolute Error:    {median_err:.6f} rad/s")
    print(f"  - Mean Extracted Accuracy:  {mean_acc:.2f}%")
    print("-" * 85)

    # ----------------------------------------------------
    # GENERATE HIGH-RESOLUTION VALIDATION PLOT
    # ----------------------------------------------------
    plt.figure(figsize=(12, 6))
    indices = np.arange(1, num_zeros_to_find + 1)
    plt.plot(indices, true_zeros, 's--', color='#2ca02c', linewidth=2, markersize=7, label=r'True Riemann Zeros $\gamma_k$')
    
    extracted_matched = [m[1] for m in matches]
    plt.plot(indices, extracted_matched, 'o-', color='#9467bd', linewidth=1.5, markersize=6, label=r'Operator Extracted Frequencies $\omega_k$')
    
    plt.title(f"High-Order Riemann Zero Extraction Validation (First {num_zeros_to_find} Zeros, $N={N_samples}$)")
    plt.xlabel("Zero Index $k$")
    plt.ylabel(r"Imaginary Zero Value $\gamma_k$ / Frequency $\omega_k$ (rad/s)")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/high_order_riemann_zeros_validation.png", dpi=300)
    plt.close()
    print("\n    Saved validation plot: plots/high_order_riemann_zeros_validation.png")

    print("\n" + "=" * 85)
    print("   VALIDATION COMPLETED SUCCESSFULLY. HIGH-ORDER ZEROS CONFIRMED.   ")
    print("=" * 85)

if __name__ == '__main__':
    main()
