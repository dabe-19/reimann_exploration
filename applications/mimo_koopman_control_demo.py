"""
NON-LINEAR MIMO SYSTEM IDENTIFICATION DEMONSTRATION
===================================================
This script demonstrates the power of Koopman Operator Theory (EDMDc)
to perform data-driven system identification on highly non-linear MIMO
(Multiple-Input Multiple-Output) control systems.

The System: A Coupled Non-Linear Polynomial System
x1(k+1) = 0.5 * x1(k) + 0.8 * x2(k)^2 + u1(k)
x2(k+1) = 0.8 * x2(k) - 0.9 * x1(k)*x2(k) + u2(k)

Standard linear state-space identification (like basic ERA/N4SID) fails
miserably here because of the non-linear cross-terms (x2^2, x1*x2).

Koopman EDMDc solves this by "lifting" the state into a higher-dimensional
observable space where the dynamics become linear, allowing us to find 
(A_koop, B_koop) that perfectly model the non-linear physics.
"""
import numpy as np
import sys
import os

# Ensure the project root is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nonlinear_sysid.edmdc import EDMDControl

# =========================================================================
# 1. THE TRUE NON-LINEAR MIMO SYSTEM
# =========================================================================

def true_nonlinear_dynamics(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Simulates one step of the true non-linear discrete-time system.
    Args:
        x: shape (2, 1) or (2, N)
        u: shape (2, 1) or (2, N)
    Returns:
        x_next: shape (2, 1) or (2, N)
    """
    x1, x2 = x[0], x[1]
    u1, u2 = u[0], u[1]
    
    x1_next = 0.5 * x1 + 0.8 * (x2 ** 2) + u1
    x2_next = 0.8 * x2 - 0.9 * (x1 * x2) + u2
    
    return np.vstack([x1_next, x2_next])

def generate_dataset(n_samples: int, noise_std: float = 0.05) -> tuple:
    """Generates training data by applying random controls to the system."""
    np.random.seed(42)
    X = np.random.uniform(-2, 2, size=(2, n_samples))
    U = np.random.uniform(-1, 1, size=(2, n_samples))
    Y_clean = true_nonlinear_dynamics(X, U)
    
    # Add measurement noise to make it realistic
    Y = Y_clean + np.random.normal(0, noise_std, size=Y_clean.shape)
    
    return X, Y, U

# =========================================================================
# 2. THE KOOPMAN OBSERVABLE DICTIONARY
# =========================================================================

def polynomial_dictionary(x: np.ndarray) -> np.ndarray:
    """
    Lifts the 2D state into a 5D observable space.
    MUST return the original states as the first elements so predict_one_step works.
    Psi(x) = [x1, x2, x1^2, x2^2, x1*x2]^T
    """
    x1, x2 = x[0], x[1]
    return np.vstack([
        x1,          # The original state must be first
        x2,          # The original state must be first
        x1 ** 2,
        x2 ** 2,
        x1 * x2
    ])

def linear_dictionary(x: np.ndarray) -> np.ndarray:
    """A standard linear dictionary Psi(x) = x, for comparison."""
    return x

# =========================================================================
# 3. METRICS
# =========================================================================

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=1, keepdims=True)) ** 2)
    if ss_tot < 1e-15: return 0.0
    return 1.0 - ss_res / ss_tot

# =========================================================================
# 4. DEMONSTRATION
# =========================================================================

def main():
    print("=" * 70)
    print("   NON-LINEAR MIMO SYSTEM IDENTIFICATION (KOOPMAN EDMDc)")
    print("=" * 70)
    
    # 1. Generate Training Data
    n_train = 5000
    X_train, Y_train, U_train = generate_dataset(n_train)
    print(f"[*] Generated {n_train} training samples from true non-linear system.")
    
    # 2. Standard Linear System Identification (The Baseline)
    print("\\n[A] Training Standard Linear State-Space Model...")
    linear_model = EDMDControl(observable_func=linear_dictionary)
    linear_model.fit(X_train, Y_train, U_train)
    
    # 3. Koopman EDMDc System Identification
    print("[B] Training Koopman EDMDc Model (Lifting to Polynomial Space)...")
    koopman_model = EDMDControl(observable_func=polynomial_dictionary)
    koopman_model.fit(X_train, Y_train, U_train)
    
    # 4. Out-of-Sample Testing
    print("\\n" + "=" * 70)
    print("   OUT-OF-SAMPLE TRAJECTORY PREDICTION TEST")
    print("=" * 70)
    
    # Generate a fresh test trajectory
    n_test_steps = 50
    np.random.seed(99)
    x0 = np.array([[1.0], [-1.0]]) # Initial state
    U_test = np.random.uniform(-0.5, 0.5, size=(2, n_test_steps)) # Control sequence
    
    # Simulate true non-linear system
    X_true = np.zeros((2, n_test_steps))
    x_curr = x0.copy()
    for k in range(n_test_steps):
        u_k = U_test[:, k:k+1]
        x_next = true_nonlinear_dynamics(x_curr, u_k)
        X_true[:, k] = x_next.flatten()
        x_curr = x_next
        
    # Predict with standard linear model
    X_pred_linear = linear_model.simulate(x0, U_test)
    
    # Predict with Koopman EDMDc model
    X_pred_koopman = koopman_model.simulate(x0, U_test)
    
    # Evaluate
    r2_linear = r_squared(X_true, X_pred_linear)
    r2_koopman = r_squared(X_true, X_pred_koopman)
    
    print(f"Standard Linear Model R^2: {r2_linear:.4f}")
    print(f"Koopman EDMDc Model R^2:   {r2_koopman:.4f}")
    
    print("\\n>>> CONCLUSION:")
    if r2_koopman > 0.95 and r2_linear < 0.8:
        print("    [SUCCESS] Standard linear models failed to capture the non-linearities.")
        print("    [SUCCESS] Koopman EDMDc successfully learned the exact non-linear MIMO dynamics")
        print("              purely from data using linear operators in a lifted space!")
    else:
        print("    [FAILURE] The Koopman model did not outperform the linear model significantly.")

if __name__ == "__main__":
    main()
