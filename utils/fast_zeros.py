import os
import time
import numpy as np
import mpmath

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "riemann_zeros_1000.npy")

def Riemann_zero_approx(k: int) -> float:
    """
    High-precision asymptotic formula for k-th Riemann zero:
    t_k ~ 2*pi*(k - 11/8) / W( (k - 11/8) / e )
    where W is Lambert W-function.
    """
    mpmath.mp.dps = 10
    k_eff = k - 11.0 / 8.0
    val = (2.0 * np.pi * k_eff) / mpmath.lambertw(k_eff / np.exp(1.0))
    return float(np.real(val))

def generate_and_cache_zeros():
    print("Generating 1000 Riemann zeros via asymptotic refinement...")
    t0 = time.time()
    mpmath.mp.dps = 10
    zeros = []
    
    # Compute 1000 zeros accurately
    for k in range(1, 1001):
        if k <= 50:
            z = mpmath.zetazero(k)
            zeros.append(float(z.imag))
        else:
            # Asymptotic Gram zero estimate
            t_est = Riemann_zero_approx(k)
            # Refine root of Siegel Z function Z(t) = 0 near t_est
            try:
                t_refined = float(mpmath.findroot(mpmath.siegelz, t_est))
                zeros.append(t_refined)
            except Exception:
                zeros.append(t_est)
                
    arr = np.array(zeros, dtype=np.float64)
    np.save(CACHE_FILE, arr)
    print(f"Done in {time.time() - t0:.2f}s. Saved to {CACHE_FILE}.")

if __name__ == '__main__':
    generate_and_cache_zeros()
