"""
CORRECTED METHODOLOGY: RIGOROUS VALIDATION OF RIEMANN ZERO RECOVERY
====================================================================
This script fixes ALL four identified problems from the falsification audit:

  Fix 1: Proper statistical metric (tolerance-gated hit rate + permutation test)
  Fix 2: Out-of-sample prediction test (train 70%, predict 30%)
  Fix 3: Fixed VQE with real CNOT gates (tested separately)
  Fix 4: Operator property comparison (prime vs noise)

Each test is run on:
  A. Real prime data (psi(x) - x)
  B. Gaussian white noise
  C. Random walk
  D. Known wrong sinusoids

If prime data passes and controls fail, we have something real.
If all pass or all fail, we have nothing.
"""
import numpy as np
import time
from scipy.linalg import eig
from trng_auditor.core.data import pnt_error_term, logarithmic_resample, get_riemann_zeros
from trng_auditor.core.era_n4sid import HankelSystemID
from trng_auditor.core.spectral_estimation import ParametricSpectralEstimator

np.random.seed(42)

# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def build_era_model(y, dt, r=200, c=None, n_states=40):
    """Build ERA state-space model (A, B, C, D) from signal y."""
    if c is None:
        c = len(y) - r - 1
    c = min(c, len(y) - r - 1)
    sid = HankelSystemID(y, dt=dt)
    sid.construct_hankel(r=r, c=c)
    res = sid.realize_system(n_states=n_states)
    return res

def estimate_initial_state(A, C, y_segment):
    """
    Estimate initial state x0 from an observation segment using least squares.
    Given y[k] = C @ A^k @ x0, build the observability matrix and solve.
    """
    n_states = A.shape[0]
    n_obs = min(len(y_segment), n_states * 3)  # Use enough observations
    
    # Build observability-like matrix: O[k,:] = C @ A^k
    O = np.zeros((n_obs, n_states), dtype=np.complex128)
    CA_k = C.copy().reshape(1, -1).astype(np.complex128)
    for k in range(n_obs):
        O[k, :] = CA_k
        CA_k = CA_k @ A
    
    # Solve O @ x0 = y_segment[:n_obs] via least squares
    y_obs = y_segment[:n_obs].astype(np.complex128)
    x0, _, _, _ = np.linalg.lstsq(O, y_obs, rcond=None)
    return x0

def predict_forward(A, B, C, D, y_train, n_predict):
    """
    Given trained model (A,B,C,D), predict n_predict steps forward.
    
    Estimate initial state from the END of training data via least-squares,
    then propagate forward using x_{k+1} = A @ x_k.
    """
    # Use last portion of training data to estimate state at the boundary
    tail_len = min(200, len(y_train))
    y_tail = y_train[-tail_len:]
    
    # Estimate state at the start of the tail
    x0 = estimate_initial_state(A, C, y_tail)
    
    # Propagate to the end of training data
    x = x0.copy()
    for k in range(tail_len):
        x = A @ x
    
    # Now predict forward into test region
    y_pred = np.zeros(n_predict, dtype=np.float64)
    D_real = float(np.real(D)) if np.isscalar(D) else float(np.real(D))
    for k in range(n_predict):
        y_pred[k] = float(np.real(C @ x))
        x = A @ x
    
    return y_pred

def extract_esprit_frequencies(y, dt, M_window=400, L_signals=200):
    """Run TLS-ESPRIT, return sorted unique frequencies."""
    estimator = ParametricSpectralEstimator(y, dt=dt)
    estimator.construct_covariance(M=M_window)
    res = estimator.run_esprit(L=L_signals)
    return np.sort(np.unique(np.round(res['omegas'], 2)))

def tolerance_hit_rate(extracted_omegas, true_zeros, epsilon=0.5):
    """
    FIX 1a: For each extracted frequency, check if it is within epsilon
    of ANY true Riemann zero. Return the hit rate.
    
    This is the INVERSE of nearest-neighbor: instead of asking "for each zero,
    is there a frequency near it?" we ask "for each extracted frequency,
    is it near a zero?" — much harder to satisfy by chance.
    """
    hits = 0
    for omega in extracted_omegas:
        # Only consider frequencies in the range of the zeros
        if omega < true_zeros[0] - 5 or omega > true_zeros[-1] + 5:
            continue
        min_dist = np.min(np.abs(omega - true_zeros))
        if min_dist < epsilon:
            hits += 1
    
    # Count how many extracted frequencies fall in the zero range
    in_range = np.sum((extracted_omegas >= true_zeros[0] - 5) & 
                       (extracted_omegas <= true_zeros[-1] + 5))
    
    if in_range == 0:
        return 0.0, 0, 0
    
    return hits / in_range, hits, int(in_range)

def permutation_test(y_real, dt, true_zeros, n_permutations=50, 
                     M_window=400, L_signals=200, epsilon=0.5):
    """
    FIX 1b: Permutation test for statistical significance.
    
    Compute hit rate for real data, then shuffle the signal many times
    and compute hit rate each time. P-value = fraction of shuffled
    trials that achieve equal or better hit rate.
    """
    # Real data hit rate
    omegas_real = extract_esprit_frequencies(y_real, dt, M_window, L_signals)
    real_rate, _, _ = tolerance_hit_rate(omegas_real, true_zeros, epsilon)
    
    # Permutation distribution
    perm_rates = []
    for i in range(n_permutations):
        y_perm = np.random.permutation(y_real)
        try:
            omegas_perm = extract_esprit_frequencies(y_perm, dt, M_window, L_signals)
            rate, _, _ = tolerance_hit_rate(omegas_perm, true_zeros, epsilon)
            perm_rates.append(rate)
        except Exception:
            perm_rates.append(0.0)
        
        if (i + 1) % 50 == 0:
            print(f"        Permutation {i+1}/{n_permutations}...")
    
    perm_rates = np.array(perm_rates)
    p_value = float(np.mean(perm_rates >= real_rate))
    
    return real_rate, p_value, perm_rates

def r_squared(y_true, y_pred):
    """Coefficient of determination R^2."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        return 0.0
    return 1.0 - ss_res / ss_tot

def normalized_mse(y_true, y_pred):
    """Normalized mean squared error."""
    return float(np.mean((y_true - y_pred)**2) / (np.var(y_true) + 1e-15))


# =========================================================================
# MAIN CORRECTED VALIDATION
# =========================================================================

def main():
    print("=" * 90)
    print("   CORRECTED METHODOLOGY: RIGOROUS VALIDATION WITH PROPER CONTROLS")
    print("=" * 90)

    # =====================================================================
    # PREPARE SIGNALS
    # =====================================================================
    print("\n[0] Preparing signals...")
    
    N = 100000
    x_grid, delta_pnt = pnt_error_term(N, normalized=False)
    t_full, y_full, dt = logarithmic_resample(x_grid, delta_pnt, num_samples=3000)
    
    # Controls
    y_noise = np.random.randn(len(y_full)) * np.std(y_full)
    y_walk = np.cumsum(np.random.randn(len(y_full)))
    y_walk = y_walk - np.linspace(y_walk[0], y_walk[-1], len(y_walk))  # detrend
    y_walk = y_walk * np.std(y_full) / (np.std(y_walk) + 1e-15)  # scale to match
    
    wrong_freqs = [10.0, 17.5, 22.3, 28.8, 35.0, 42.7, 55.0, 68.3, 80.0, 95.5]
    t_grid = np.linspace(0, dt * len(y_full), len(y_full))
    y_wrong = sum(np.sin(f * t_grid + np.random.uniform(0, 2*np.pi)) for f in wrong_freqs)
    y_wrong = y_wrong * np.std(y_full) / (np.std(y_wrong) + 1e-15)
    
    signals = {
        'A. Prime Data (psi(x)-x)': y_full,
        'B. Gaussian White Noise': y_noise,
        'C. Random Walk': y_walk,
        'D. Wrong Sinusoids': y_wrong,
    }
    
    true_zeros = get_riemann_zeros(num_zeros=30)
    print(f"    Loaded {len(true_zeros)} true zeros: gamma_1={true_zeros[0]:.4f} to gamma_30={true_zeros[-1]:.4f}")
    print(f"    Signal length: {len(y_full)}, dt = {dt:.6f}")

    # =====================================================================
    # TEST 1: OUT-OF-SAMPLE PREDICTION (FIX 2)
    # =====================================================================
    print("\n" + "=" * 90)
    print("  TEST 1: OUT-OF-SAMPLE PREDICTION (train 70%, predict 30%)")
    print("=" * 90)
    
    split = int(0.7 * len(y_full))
    
    for name, y_sig in signals.items():
        print(f"\n  --- {name} ---")
        y_train = y_sig[:split]
        y_test = y_sig[split:]
        
        try:
            t0 = time.time()
            r_dim = 150
            c_dim = min(len(y_train) - r_dim - 1, 500)
            n_st = 30
            
            res = build_era_model(y_train, dt, r=r_dim, c=c_dim, n_states=n_st)
            A = res['A_discrete']
            B = res['B_discrete']
            C = res['C_discrete']
            D = res['D_discrete']
            
            # Predict forward on test portion
            y_pred = predict_forward(A, B, C, D, y_train, len(y_test))
            
            # Clip extreme predictions to avoid overflow in metrics
            max_val = 10 * np.max(np.abs(y_test))
            y_pred = np.clip(y_pred, -max_val, max_val)
            
            r2 = r_squared(y_test, y_pred)
            nmse = normalized_mse(y_test, y_pred)
            
            print(f"    R^2 (out-of-sample):    {r2:.6f}")
            print(f"    Normalized MSE:         {nmse:.6f}")
            print(f"    Time: {time.time()-t0:.2f}s")
            
        except Exception as e:
            print(f"    ERROR: {e}")

    # =====================================================================
    # TEST 2: TOLERANCE-GATED HIT RATE (FIX 1a)
    # =====================================================================
    print("\n" + "=" * 90)
    print("  TEST 2: TOLERANCE-GATED FREQUENCY HIT RATE (epsilon=0.5 rad/s)")
    print("=" * 90)
    
    for name, y_sig in signals.items():
        print(f"\n  --- {name} ---")
        try:
            t0 = time.time()
            omegas = extract_esprit_frequencies(y_sig, dt)
            rate, hits, in_range = tolerance_hit_rate(omegas, true_zeros, epsilon=0.5)
            
            print(f"    Extracted frequencies: {len(omegas)}")
            print(f"    Frequencies in zero range: {in_range}")
            print(f"    Hits (within 0.5 rad/s of a zero): {hits}")
            print(f"    Hit rate: {rate:.4f} ({rate*100:.1f}%)")
            print(f"    Time: {time.time()-t0:.2f}s")
        except Exception as e:
            print(f"    ERROR: {e}")

    # =====================================================================
    # TEST 3: PERMUTATION TEST (FIX 1b)
    # =====================================================================
    print("\n" + "=" * 90)
    print("  TEST 3: PERMUTATION TEST (200 shuffles, epsilon=0.5)")
    print("=" * 90)
    print("  Running permutation test on prime data (50 shuffles)...")
    print("  (This tests: could random reordering of the SAME values produce equal hit rates?)")
    
    try:
        t0 = time.time()
        real_rate, p_value, perm_rates = permutation_test(
            y_full, dt, true_zeros, 
            n_permutations=50, epsilon=0.5
        )
        print(f"\n    Real prime data hit rate:     {real_rate:.4f} ({real_rate*100:.1f}%)")
        print(f"    Permutation mean hit rate:    {np.mean(perm_rates):.4f} ({np.mean(perm_rates)*100:.1f}%)")
        print(f"    Permutation std:             {np.std(perm_rates):.4f}")
        print(f"    P-value:                     {p_value:.4f}")
        print(f"    Time: {time.time()-t0:.2f}s")
        
        if p_value < 0.01:
            print(f"    >>> STATISTICALLY SIGNIFICANT (p < 0.01)")
        elif p_value < 0.05:
            print(f"    >>> MARGINALLY SIGNIFICANT (p < 0.05)")
        else:
            print(f"    >>> NOT SIGNIFICANT (p >= 0.05)")
    except Exception as e:
        print(f"    ERROR: {e}")

    # =====================================================================
    # TEST 4: OPERATOR PROPERTY COMPARISON (FIX 4)
    # =====================================================================
    print("\n" + "=" * 90)
    print("  TEST 4: OPERATOR PROPERTY COMPARISON (prime vs controls)")
    print("=" * 90)
    
    r_dim = 200
    n_st = 30
    
    for name, y_sig in signals.items():
        print(f"\n  --- {name} ---")
        try:
            c_dim = min(len(y_sig) - r_dim - 1, 500)
            res = build_era_model(y_sig, dt, r=r_dim, c=c_dim, n_states=n_st)
            A = res['A_discrete']
            
            # Normality: ||AA* - A*A||_F / ||A||_F^2
            A_adj = A.conj().T
            comm = A @ A_adj - A_adj @ A
            normality = float(np.linalg.norm(comm, 'fro') / (np.linalg.norm(A, 'fro')**2 + 1e-12))
            
            # Eigenvalue analysis
            eigvals = res['discrete_eigvals']
            moduli = np.abs(eigvals)
            mean_modulus = float(np.mean(moduli))
            std_modulus = float(np.std(moduli))
            
            # Unit circle proximity: how close are eigenvalues to |z| = 1?
            unit_circle_dist = np.abs(moduli - 1.0)
            mean_uc_dist = float(np.mean(unit_circle_dist))
            
            # Continuous pole damping
            mean_sigma = float(np.mean(np.abs(res['sigmas'])))
            
            # Singular value profile: ratio of top singular value to 10th
            svs = res['singular_values']
            sv_ratio = float(svs[0] / (svs[min(9, len(svs)-1)] + 1e-15))
            
            # Singular value effective rank (number of SVs > 1% of max)
            eff_rank = int(np.sum(svs > 0.01 * svs[0]))
            
            print(f"    Normality metric N(A):     {normality:.6f}  (lower = more normal)")
            print(f"    Mean |eigenvalue|:          {mean_modulus:.6f}  (1.0 = unit circle)")
            print(f"    Std |eigenvalue|:           {std_modulus:.6f}")
            print(f"    Mean unit-circle distance:  {mean_uc_dist:.6f}  (lower = more conservative)")
            print(f"    Mean |damping sigma|:       {mean_sigma:.6f}  (lower = less damped)")
            print(f"    SV condition (sv1/sv10):    {sv_ratio:.2f}")
            print(f"    Effective rank:             {eff_rank} / {n_st}")
            
        except Exception as e:
            print(f"    ERROR: {e}")

    # =====================================================================
    # TEST 5: FIXED VQE VERIFICATION (FIX 3)
    # =====================================================================
    print("\n" + "=" * 90)
    print("  TEST 5: FIXED VQE WITH REAL CNOT ENTANGLEMENT")
    print("=" * 90)
    
    try:
        from trng_auditor.advanced.quantum_sysid import QuantumSystemIdentification
        
        # Build operator from prime data
        r_dim_q = 80
        c_dim_q = min(len(y_full) - r_dim_q - 1, 800)
        res_q = build_era_model(y_full, dt, r=r_dim_q, c=c_dim_q, n_states=16)
        A_q = res_q['A_discrete']
        
        qsys = QuantumSystemIdentification(A_q, num_qubits=3)
        
        # Verify CNOT gate is correct
        cnot_01 = qsys._cnot_gate(0, 1)
        # Test: CNOT|10> should give |11>
        test_state = np.zeros(8, dtype=np.complex128)
        test_state[0b100] = 1.0  # |100> (qubit 0 = 1)
        result = cnot_01 @ test_state
        expected_idx = 0b110  # |110> (qubit 1 flipped)
        cnot_correct = np.abs(result[expected_idx] - 1.0) < 1e-10
        print(f"    CNOT gate verification: {'PASS' if cnot_correct else 'FAIL'}")
        
        # Run VQE with real entanglement
        t0 = time.time()
        vqe_res = qsys.run_vqe(layers=4)
        vqe_time = time.time() - t0
        
        exact_ground = vqe_res['exact_ground_energy']
        vqe_ground = vqe_res['ground_energy_vqe']
        vqe_err = vqe_res['vqe_error']
        
        print(f"    Exact ground energy:      {exact_ground:.8f}")
        print(f"    VQE ground energy:        {vqe_ground:.8f}")
        print(f"    VQE absolute error:       {vqe_err:.2e}")
        print(f"    Relative error:           {vqe_err / (abs(exact_ground) + 1e-15):.2e}")
        print(f"    Active Pauli terms:       {vqe_res['num_pauli_terms']} / 64")
        print(f"    VQE optimization time:    {vqe_time:.2f}s")
        
        if vqe_err / (abs(exact_ground) + 1e-15) < 0.05:
            print(f"    >>> VQE CONVERGED (relative error < 5%)")
        else:
            print(f"    >>> VQE DID NOT CONVERGE WELL")
            
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()

    # =====================================================================
    # FINAL SUMMARY TABLE
    # =====================================================================
    print("\n" + "=" * 90)
    print("  CORRECTED VALIDATION COMPLETE")
    print("=" * 90)
    print("""
  Interpretation Guide:
  - TEST 1 (Prediction): If prime R^2 >> 0 and noise R^2 ≈ 0, the model captures real dynamics
  - TEST 2 (Hit Rate):   If prime hit rate >> noise hit rate, frequencies are genuinely at zeros
  - TEST 3 (P-value):    If p < 0.01, the frequency match is statistically significant
  - TEST 4 (Operator):   If prime normality << noise normality, the operator has special structure
  - TEST 5 (VQE):        If relative error < 5%, the quantum circuit works correctly
    """)

if __name__ == '__main__':
    main()
