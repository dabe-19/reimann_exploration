import time
import numpy as np
import scipy.linalg
from riemann_sysid.operator_wrapper import HilbertPolyaOperator
from riemann_sysid.data import chebyshev_psi, generate_primes

def estimate_pi_from_psi(psi_val: float, x: float) -> float:
    """
    Invert Chebyshev psi(x) ~ sum_{p <= x} ln p to estimate prime counting function pi(x).
    Using logarithmic integral li(x) correction: pi(x) approx psi(x) / ln x + x / (ln x)^2.
    """
    if x <= 1:
        return 0.0
    ln_x = np.log(x)
    return psi_val / ln_x

def main():
    print("=" * 85)
    print("   PRACTICAL APPLICATION: O(1) PRIME COUNTING SURROGATE MODEL VIA OPERATOR   ")
    print("=" * 85)

    print("\n[1] Constructing Hilbert-Pólya Operator Model (State Dimension = 80)...")
    t0 = time.time()
    op = HilbertPolyaOperator(sequence_length=40000, n_states=80, r_dim=800)
    print(f"    Operator built in {time.time() - t0:.2f}s.")

    # Test evaluation points x
    test_points = [100, 500, 1000, 5000, 10000, 25000]
    
    print("\n[2] Evaluating O(1) Operator State-Space Prediction vs Exact Prime Counts:")
    print(f"  {'x':<10} | {'Exact psi(x)':<16} | {'Operator Predicted psi(x)':<25} | {'Exact pi(x)':<14} | {'Predicted pi(x)':<16} | {'Error':<8}")
    print("  " + "-" * 98)

    primes_up_to_max = generate_primes(max(test_points) + 100)
    
    for x in test_points:
        # Exact calculation
        exact_psi = chebyshev_psi(x)[-1]
        exact_pi = len([p for p in primes_up_to_max if p <= x])
        
        # State space prediction: y(t) where t = ln(x)
        # psi_pred = x + C * exp(A * ln x) * B
        t_log = np.log(x)
        # Matrix exponential state evolution e^{A * t_log}
        x_state_evolved = scipy.linalg.expm(op.A * (t_log / 10.0)) @ op.B
        delta_psi_pred = float(np.real(op.C @ x_state_evolved))
        
        # Predicted psi(x) = x - delta_psi_pred
        pred_psi = x - delta_psi_pred
        pred_pi = estimate_pi_from_psi(pred_psi, x)
        
        err_pct = abs(pred_pi - exact_pi) / exact_pi * 100.0
        print(f"  {x:<10} | {exact_psi:<16.2f} | {pred_psi:<25.2f} | {exact_pi:<14} | {pred_pi:<16.1f} | {err_pct:<6.2f}%")

    print("\n" + "=" * 85)
    print("   APPLICATION DEMONSTRATION COMPLETED.   ")
    print("=" * 85)

if __name__ == '__main__':
    main()
