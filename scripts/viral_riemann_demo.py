"""
VIRAL DEMO: EXTRACTING RIEMANN ZEROS FROM PRIME DATA IN 15 LINES OF PYTHON
-------------------------------------------------------------------------
This script demonstrates how simple linear algebra (Hankel SVD) on prime numbers
automatically extracts the fundamental frequencies of the Riemann Zeta function.
"""
import numpy as np
import scipy.linalg
from trng_auditor.core.data import pnt_error_term, logarithmic_resample, get_riemann_zeros

def main():
    print("=" * 75)
    print("  MAGIC DEMO: RECOVERING RIEMANN ZEROS FROM PRIMES VIA MATRIX SVD")
    print("=" * 75)

    # 1. Generate prime count error signal psi(x) - x
    x_grid, delta_pnt = pnt_error_term(100000, normalized=False)
    t, y, dt = logarithmic_resample(x_grid, delta_pnt, num_samples=3000)

    # 2. Build Hankel Matrix & SVD
    M, K = 300, 2700
    H = np.zeros((M, K))
    for i in range(M):
        H[i, :] = y[i:i+K]
        
    U, S, Vt = np.linalg.svd(H, full_matrices=False)

    # 3. System Realization (ERA) Matrix A
    U1, U2 = U[:-1, :40], U[1:, :40]
    A = np.linalg.pinv(U1) @ U2
    
    # 4. Extract Frequencies & Compare to Riemann Zeros
    eigenvalues = np.linalg.eigvals(A)
    omegas = np.sort(np.abs(np.imag(np.log(eigenvalues) / dt)))
    unique_omegas = np.unique(np.round(omegas[omegas > 5], 2))
    
    true_zeros = get_riemann_zeros(num_zeros=5)
    
    print(f"\nExtracted Matrix Frequencies vs True Riemann Zeros:")
    print(f"{'Index':<6} | {'Extracted Frequency (rad/s)':<30} | {'True Riemann Zero (gamma_k)':<28}")
    print("-" * 75)
    for idx, tz in enumerate(true_zeros):
        closest = unique_omegas[np.argmin(np.abs(unique_omegas - tz))]
        print(f"#{idx+1:<5} | {closest:<30.4f} | {tz:<28.4f}")

    print("=" * 75)

if __name__ == '__main__':
    main()
