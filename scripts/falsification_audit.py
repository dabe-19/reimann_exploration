"""
CRITICAL AUDIT: FALSIFICATION TESTS FOR RIEMANN ZERO RECOVERY
==============================================================
This script tests whether our methodology is genuinely extracting
structure from prime data, or whether it is an artifact of:
  1. Circular reasoning (using known zeros to validate known zeros)
  2. Dense frequency matching (any dense frequency set would "match")
  3. Overfitting / hyperparameter cherry-picking

Tests:
  A. Feed pure Gaussian white noise -> does ESPRIT "recover" zeros?
  B. Feed a random walk signal -> does ESPRIT "recover" zeros?
  C. Feed a sum of KNOWN non-Riemann sinusoids -> does it falsely match?
  D. Measure how dense our extracted frequency set is (spacing analysis)
  E. Quantify: what accuracy would RANDOM frequencies achieve?
"""
import numpy as np
import time
from trng_auditor.core.data import pnt_error_term, logarithmic_resample, get_riemann_zeros
from trng_auditor.core.spectral_estimation import ParametricSpectralEstimator

np.random.seed(42)

def run_esprit_on_signal(y, dt, M_window=400, L_signals=200):
    """Run TLS-ESPRIT on arbitrary signal, return sorted unique frequencies."""
    estimator = ParametricSpectralEstimator(y, dt=dt)
    estimator.construct_covariance(M=M_window)
    res = estimator.run_esprit(L=L_signals)
    omegas = res['omegas']
    return np.sort(np.unique(np.round(omegas, 2)))

def match_accuracy(extracted_omegas, true_zeros):
    """Compute per-zero accuracy using nearest-neighbor matching."""
    accuracies = []
    for tz in true_zeros:
        if len(extracted_omegas) == 0:
            accuracies.append(0.0)
            continue
        closest = extracted_omegas[np.argmin(np.abs(extracted_omegas - tz))]
        err = abs(closest - tz)
        acc = max(0.0, 100.0 * (1.0 - err / tz))
        accuracies.append(acc)
    return np.array(accuracies)

def random_frequency_baseline(n_random_freqs, true_zeros, omega_max, n_trials=100):
    """
    Monte Carlo baseline: generate n_random_freqs uniformly distributed
    frequencies in [0, omega_max] and compute matching accuracy against
    true Riemann zeros. Repeat n_trials times and average.
    """
    all_accs = []
    for _ in range(n_trials):
        rand_omegas = np.sort(np.random.uniform(0, omega_max, n_random_freqs))
        accs = match_accuracy(rand_omegas, true_zeros)
        all_accs.append(np.mean(accs))
    return np.mean(all_accs), np.std(all_accs)

def main():
    print("=" * 90)
    print("   CRITICAL FALSIFICATION AUDIT: IS RIEMANN ZERO RECOVERY REAL OR ARTIFACT?")
    print("=" * 90)

    # Get reference zeros
    print("\n[0] Loading first 30 true Riemann zeros for comparison...")
    true_zeros = get_riemann_zeros(num_zeros=30)
    print(f"    Loaded {len(true_zeros)} zeros: gamma_1={true_zeros[0]:.4f} to gamma_30={true_zeros[-1]:.4f}")

    # =========================================================================
    # TEST A: REAL PRIME DATA (the baseline we claim works)
    # =========================================================================
    print("\n" + "=" * 90)
    print("  TEST A: REAL PRIME DATA (psi(x) - x, log-resampled)")
    print("=" * 90)
    x_grid, delta_pnt = pnt_error_term(100000, normalized=False)
    t_real, y_real, dt_real = logarithmic_resample(x_grid, delta_pnt, num_samples=3000)
    
    t0 = time.time()
    omegas_real = run_esprit_on_signal(y_real, dt_real)
    print(f"    ESPRIT extracted {len(omegas_real)} unique frequencies in {time.time()-t0:.2f}s")
    
    accs_real = match_accuracy(omegas_real, true_zeros)
    print(f"    Mean accuracy vs true zeros: {np.mean(accs_real):.2f}%")
    print(f"    Zeros with >95% accuracy: {np.sum(accs_real > 95)} / {len(true_zeros)}")
    print(f"    Zeros with >90% accuracy: {np.sum(accs_real > 90)} / {len(true_zeros)}")

    # =========================================================================
    # TEST B: PURE GAUSSIAN WHITE NOISE (should NOT match zeros)
    # =========================================================================
    print("\n" + "=" * 90)
    print("  TEST B: PURE GAUSSIAN WHITE NOISE (no prime structure)")
    print("=" * 90)
    y_noise = np.random.randn(len(y_real))
    
    t0 = time.time()
    omegas_noise = run_esprit_on_signal(y_noise, dt_real)
    print(f"    ESPRIT extracted {len(omegas_noise)} unique frequencies in {time.time()-t0:.2f}s")
    
    accs_noise = match_accuracy(omegas_noise, true_zeros)
    print(f"    Mean accuracy vs true zeros: {np.mean(accs_noise):.2f}%")
    print(f"    Zeros with >95% accuracy: {np.sum(accs_noise > 95)} / {len(true_zeros)}")
    print(f"    Zeros with >90% accuracy: {np.sum(accs_noise > 90)} / {len(true_zeros)}")

    # =========================================================================
    # TEST C: RANDOM WALK (correlated noise, no prime structure)
    # =========================================================================
    print("\n" + "=" * 90)
    print("  TEST C: RANDOM WALK (correlated noise, no prime structure)")
    print("=" * 90)
    y_walk = np.cumsum(np.random.randn(len(y_real)))
    y_walk = y_walk - np.linspace(y_walk[0], y_walk[-1], len(y_walk))  # detrend
    
    t0 = time.time()
    omegas_walk = run_esprit_on_signal(y_walk, dt_real)
    print(f"    ESPRIT extracted {len(omegas_walk)} unique frequencies in {time.time()-t0:.2f}s")
    
    accs_walk = match_accuracy(omegas_walk, true_zeros)
    print(f"    Mean accuracy vs true zeros: {np.mean(accs_walk):.2f}%")
    print(f"    Zeros with >95% accuracy: {np.sum(accs_walk > 95)} / {len(true_zeros)}")
    print(f"    Zeros with >90% accuracy: {np.sum(accs_walk > 90)} / {len(true_zeros)}")

    # =========================================================================
    # TEST D: KNOWN WRONG FREQUENCIES (sum of sinusoids at NON-Riemann freqs)
    # =========================================================================
    print("\n" + "=" * 90)
    print("  TEST D: SUM OF KNOWN NON-RIEMANN SINUSOIDS")
    print("=" * 90)
    wrong_freqs = [10.0, 17.5, 22.3, 28.8, 35.0, 42.7, 55.0, 68.3, 80.0, 95.5]
    t_grid = np.linspace(0, dt_real * len(y_real), len(y_real))
    y_wrong = sum(np.sin(f * t_grid + np.random.uniform(0, 2*np.pi)) for f in wrong_freqs)
    y_wrong += 0.1 * np.random.randn(len(y_real))
    
    t0 = time.time()
    omegas_wrong = run_esprit_on_signal(y_wrong, dt_real)
    print(f"    ESPRIT extracted {len(omegas_wrong)} unique frequencies in {time.time()-t0:.2f}s")
    
    accs_wrong = match_accuracy(omegas_wrong, true_zeros)
    print(f"    Mean accuracy vs true zeros: {np.mean(accs_wrong):.2f}%")
    print(f"    Zeros with >95% accuracy: {np.sum(accs_wrong > 95)} / {len(true_zeros)}")
    print(f"    Zeros with >90% accuracy: {np.sum(accs_wrong > 90)} / {len(true_zeros)}")

    # =========================================================================
    # TEST E: FREQUENCY DENSITY / SPACING ANALYSIS
    # =========================================================================
    print("\n" + "=" * 90)
    print("  TEST E: FREQUENCY DENSITY ANALYSIS (is matching trivial?)")
    print("=" * 90)
    
    if len(omegas_real) > 1:
        spacings = np.diff(omegas_real)
        omega_max = omegas_real[-1]
        print(f"    Extracted frequency range: [0, {omega_max:.2f}] rad/s")
        print(f"    Number of extracted frequencies: {len(omegas_real)}")
        print(f"    Mean frequency spacing: {np.mean(spacings):.4f} rad/s")
        print(f"    Min frequency spacing:  {np.min(spacings):.4f} rad/s")
        print(f"    Max frequency spacing:  {np.max(spacings):.4f} rad/s")
        
        # Average spacing between consecutive Riemann zeros in this range
        zero_spacings = np.diff(true_zeros)
        print(f"    Mean Riemann zero spacing: {np.mean(zero_spacings):.4f} rad/s")
        
        density_ratio = np.mean(spacings) / np.mean(zero_spacings)
        print(f"    Density ratio (extracted_spacing / zero_spacing): {density_ratio:.4f}")
        if density_ratio < 0.5:
            print(f"    *** WARNING: Extracted frequencies are {1/density_ratio:.1f}x DENSER than Riemann zeros.")
            print(f"    *** This means high accuracy could be trivially achieved by dense coverage!")
    
    # =========================================================================
    # TEST F: MONTE CARLO RANDOM BASELINE
    # =========================================================================
    print("\n" + "=" * 90)
    print("  TEST F: MONTE CARLO RANDOM FREQUENCY BASELINE")
    print("=" * 90)
    
    n_extracted = len(omegas_real)
    omega_max_real = max(omegas_real) if len(omegas_real) > 0 else 200.0
    
    mean_rand_acc, std_rand_acc = random_frequency_baseline(n_extracted, true_zeros, omega_max_real)
    print(f"    Random baseline ({n_extracted} uniform freqs in [0, {omega_max_real:.1f}]):")
    print(f"    Mean random accuracy: {mean_rand_acc:.2f}% +/- {std_rand_acc:.2f}%")
    print(f"    Our prime data accuracy: {np.mean(accs_real):.2f}%")
    print(f"    Improvement over random: {np.mean(accs_real) - mean_rand_acc:.2f} percentage points")
    
    if np.mean(accs_real) - mean_rand_acc < 5.0:
        print(f"    *** CRITICAL WARNING: Our accuracy is NOT significantly better than random!")
        print(f"    *** The 'recovery' may be an artifact of dense frequency matching.")
    elif np.mean(accs_real) - mean_rand_acc < 15.0:
        print(f"    *** CAUTION: Improvement over random is modest. Dense matching may contribute.")
    else:
        print(f"    PASS: Significant improvement over random baseline detected.")

    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    print("\n" + "=" * 90)
    print("  FINAL AUDIT SUMMARY")
    print("=" * 90)
    print(f"  {'Test':<45} | {'Mean Acc':<10} | {'>95%':<6} | {'>90%':<6}")
    print("  " + "-" * 75)
    print(f"  {'A. Real Prime Data (psi(x) - x)':<45} | {np.mean(accs_real):<10.2f} | {np.sum(accs_real>95):<6} | {np.sum(accs_real>90):<6}")
    print(f"  {'B. Pure Gaussian White Noise':<45} | {np.mean(accs_noise):<10.2f} | {np.sum(accs_noise>95):<6} | {np.sum(accs_noise>90):<6}")
    print(f"  {'C. Random Walk (correlated noise)':<45} | {np.mean(accs_walk):<10.2f} | {np.sum(accs_walk>95):<6} | {np.sum(accs_walk>90):<6}")
    print(f"  {'D. Known Wrong Sinusoids':<45} | {np.mean(accs_wrong):<10.2f} | {np.sum(accs_wrong>95):<6} | {np.sum(accs_wrong>90):<6}")
    print(f"  {'F. Monte Carlo Random Baseline':<45} | {mean_rand_acc:<10.2f} | {'N/A':<6} | {'N/A':<6}")
    print("=" * 90)

if __name__ == '__main__':
    main()
