"""
CONCEPTOR-TUNED ESN (MANIFOLD ENGINEERING)
==========================================
Standard ESNs fail on unbounded/unstable systems (like Cart-Pole) because the 
tanh activation physically cannot replicate exponential divergence. 

To "engineer good nodes", we introduce CONCEPTORS (Jaeger, 2014). 
A Conceptor is a geometric filter matrix (C) computed from the reservoir's 
correlation matrix. By applying C during the recurrent update 
( r = C * tanh(...) ), we actively force the chaotic, random reservoir nodes 
to strictly obey the topological manifold of the physical system, killing off 
useless dimensions and engineering the remaining nodes to hold prediction longer.
"""
import numpy as np
import scipy.stats
import sys

# =========================================================================
# 1. THE TRUE PHYSICAL SYSTEM (CART-POLE)
# =========================================================================

class CartPolePhysics:
    def __init__(self):
        self.g = 9.81
        self.mc = 1.0
        self.mp = 0.1
        self.l = 0.5
        self.dt = 0.02

    def step(self, state, force):
        x, x_dot, theta, theta_dot = state[0], state[1], state[2], state[3]
        total_mass = self.mc + self.mp
        pole_mass_length = self.mp * self.l
        costheta, sintheta = np.cos(theta), np.sin(theta)
        
        temp = (force + pole_mass_length * theta_dot**2 * sintheta) / total_mass
        thetaacc = (self.g * sintheta - costheta * temp) / (
            self.l * (4.0/3.0 - self.mp * costheta**2 / total_mass)
        )
        xacc = temp - pole_mass_length * thetaacc * costheta / total_mass
        
        return np.array([
            x + self.dt * x_dot,
            x_dot + self.dt * xacc,
            theta + self.dt * theta_dot,
            theta_dot + self.dt * thetaacc
        ])

def generate_dataset(n_samples: int) -> tuple:
    physics = CartPolePhysics()
    X_list, Y_list, U_list = [], [], []
    curr_state = np.random.uniform(-1.0, 1.0, size=4)
    curr_state[2] = np.random.uniform(-np.pi, np.pi)
    
    for _ in range(n_samples):
        force = np.random.uniform(-20.0, 20.0)
        next_state = physics.step(curr_state, force)
        if np.abs(next_state[3]) < 15.0 and np.abs(next_state[0]) < 10.0:
            X_list.append(curr_state)
            Y_list.append(next_state)
            U_list.append([force])
            curr_state = next_state
        else:
            curr_state = np.random.uniform(-1.0, 1.0, size=4)
            curr_state[2] = np.random.uniform(-np.pi, np.pi)
            
    return np.array(X_list).T, np.array(Y_list).T, np.array(U_list).T

# =========================================================================
# 2. HAAR-ORTHOGONAL ESN WITH CONCEPTOR FILTERING
# =========================================================================

class HaarOrthogonalESN:
    def __init__(self, input_dim: int, reservoir_dim: int, spectral_radius: float = 0.95):
        self.N = reservoir_dim
        np.random.seed(42)
        self.W_in = np.random.uniform(-1.0, 1.0, (self.N, input_dim))
        
        # Haar-distributed orthogonal matrix
        orthogonal_matrix = scipy.stats.ortho_group.rvs(dim=self.N, random_state=42)
        self.W_res = orthogonal_matrix * spectral_radius
        
        self.W_out = None
        self.Conceptor = np.eye(self.N) # Default to identity (no filtering)

    def rollout_features(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        n_samples = X.shape[1]
        Phi = np.zeros((self.N + 4, n_samples))
        r_curr = np.zeros((self.N, 1))
        
        for k in range(n_samples):
            x_k = X[:, k:k+1]
            u_k = U[:, k:k+1]
            xu = np.vstack([x_k, u_k])
            
            # ESN update with Conceptor filtering
            r_next = self.Conceptor @ np.tanh(self.W_in @ xu + self.W_res @ r_curr)
            
            Phi[:, k] = np.vstack([x_k, r_next]).flatten()
            r_curr = r_next
            
        return Phi
        
    def predict_sequence(self, x_start: np.ndarray, U_seq: np.ndarray) -> np.ndarray:
        horizon = U_seq.shape[1]
        Y_pred = np.zeros((4, horizon))
        
        x_curr = x_start
        r_curr = np.zeros((self.N, 1))
        
        for k in range(horizon):
            u_k = U_seq[:, k:k+1]
            xu = np.vstack([x_curr, u_k])
            
            # ESN update with Conceptor filtering
            r_next = self.Conceptor @ np.tanh(self.W_in @ xu + self.W_res @ r_curr)
            phi = np.vstack([x_curr, r_next])
            
            x_next = self.W_out @ phi
            
            Y_pred[:, k:k+1] = x_next
            x_curr = x_next
            r_curr = r_next
            
        return Y_pred

# =========================================================================
# 3. TRAINING ALGORITHMS
# =========================================================================

def stlsq(Phi: np.ndarray, Y: np.ndarray, threshold: float = 0.1, iterations: int = 10) -> np.ndarray:
    """SINDy Sparse Regression."""
    Phi_T = Phi.T
    Y_T = Y.T
    W_T = np.linalg.lstsq(Phi_T, Y_T, rcond=None)[0]
    
    for _ in range(iterations):
        small_inds = np.abs(W_T) < threshold
        W_T[small_inds] = 0.0
        for j in range(Y_T.shape[1]):
            big_inds = ~small_inds[:, j]
            if np.sum(big_inds) > 0:
                W_T[big_inds, j] = np.linalg.lstsq(Phi_T[:, big_inds], Y_T[:, j], rcond=None)[0]
    return W_T.T

def dense_ridge(Phi: np.ndarray, Y: np.ndarray, alpha: float = 1e-4) -> np.ndarray:
    """Standard Dense Ridge Regression."""
    Phi_PhiT = Phi @ Phi.T
    return Y @ Phi.T @ np.linalg.inv(Phi_PhiT + alpha * np.eye(Phi.shape[0]))

def compute_conceptor(R: np.ndarray, aperture: float = 10.0) -> np.ndarray:
    """Computes the Conceptor matrix to mathematically constrain the manifold."""
    # R is the raw reservoir states (without physical states appended)
    correlation = (R @ R.T) / R.shape[1]
    # C = R (R + alpha^-2 I)^-1
    # Aperture controls how tightly we squeeze the manifold.
    alpha_sq = aperture ** -2
    C = correlation @ np.linalg.inv(correlation + alpha_sq * np.eye(R.shape[0]))
    return C

# =========================================================================
# 4. METRICS
# =========================================================================

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=1, keepdims=True)) ** 2)
    if ss_tot < 1e-15: return 0.0
    return 1.0 - ss_res / ss_tot

# =========================================================================
# 5. DEMONSTRATION
# =========================================================================

def main():
    print("=" * 80)
    print("   GRAND BASELINE: ENGINEERING GOOD RESERVOIR NODES")
    print("=" * 80)
    
    n_train = 20000
    X_train, Y_train, U_train = generate_dataset(n_train)
    
    print(f"[*] Simulated {n_train} physical time-steps of the Cart-Pole.")
    
    # We include U in the ESN input dimension: [x; u] = 4 + 1 = 5
    res_dim = 1000
    esn_dense = HaarOrthogonalESN(input_dim=5, reservoir_dim=res_dim, spectral_radius=0.99)
    esn_sparse = HaarOrthogonalESN(input_dim=5, reservoir_dim=res_dim, spectral_radius=0.99)
    esn_conceptor = HaarOrthogonalESN(input_dim=5, reservoir_dim=res_dim, spectral_radius=0.99)
    
    # Sync all weights for a fair benchmark
    esn_sparse.W_in = esn_dense.W_in.copy()
    esn_sparse.W_res = esn_dense.W_res.copy()
    esn_conceptor.W_in = esn_dense.W_in.copy()
    esn_conceptor.W_res = esn_dense.W_res.copy()
    
    print("\n[*] Unrolling untuned reservoir to collect states...")
    Phi_train_raw = esn_dense.rollout_features(X_train, U_train)
    
    # ---------------------------------------------------------
    # [A] Train Standard Dense ESN (Black Box)
    # ---------------------------------------------------------
    print("\n[A] Training Standard Dense ESN (Ridge)...")
    esn_dense.W_out = dense_ridge(Phi_train_raw, Y_train, alpha=1e-2)
    
    # ---------------------------------------------------------
    # [B] Train SINDy-ESN (Sparse Pruning)
    # ---------------------------------------------------------
    print("[B] Training SINDy-ESN (Sparse Readout)...")
    esn_sparse.W_out = stlsq(Phi_train_raw, Y_train, threshold=0.15, iterations=10)
    
    # ---------------------------------------------------------
    # [C] Train Conceptor-Tuned ESN (Manifold Engineering)
    # ---------------------------------------------------------
    print("[C] Training Conceptor-Tuned ESN (Manifold Constrained)...")
    # 1. Compute the Conceptor from the raw reservoir states (excluding physical x_k nodes)
    R_raw = Phi_train_raw[4:, :] 
    C = compute_conceptor(R_raw, aperture=10.0)
    esn_conceptor.Conceptor = C
    
    # 2. Re-unroll the reservoir, this time actively applying the Conceptor filter
    Phi_train_conceptor = esn_conceptor.rollout_features(X_train, U_train)
    
    # 3. Train readout on the newly constrained manifold
    esn_conceptor.W_out = dense_ridge(Phi_train_conceptor, Y_train, alpha=1e-2)
    
    # ---------------------------------------------------------
    # [D] Out-of-Sample Evaluation
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("   OUT-OF-SAMPLE TRAJECTORY PREDICTION TEST")
    print("=" * 80)
    
    n_test = 2000
    X_test, _, U_test = generate_dataset(n_test)
    n_test_actual = X_test.shape[1]
    
    horizons = [1, 5, 10, 20]
    print(f"{'Horizon (steps)':<20} | {'Dense R^2':<15} | {'SINDy R^2':<15} | {'Conceptor R^2':<15}")
    print("-" * 75)
    
    for h in horizons:
        # Simulate true physics
        Y_test_h = np.zeros_like(X_test)
        physics = CartPolePhysics()
        for k in range(n_test_actual):
            curr = X_test[:, k]
            for _ in range(h):
                curr = physics.step(curr, U_test[0, k])
            Y_test_h[:, k] = curr
            
        # Predict 
        Y_pred_dense = np.zeros_like(Y_test_h)
        Y_pred_sparse = np.zeros_like(Y_test_h)
        Y_pred_conceptor = np.zeros_like(Y_test_h)
        
        for k in range(n_test_actual):
            Y_pred_dense[:, k:k+1] = esn_dense.predict_sequence(X_test[:, k:k+1], U_test[:, k:k+h])[:, -1:]
            Y_pred_sparse[:, k:k+1] = esn_sparse.predict_sequence(X_test[:, k:k+1], U_test[:, k:k+h])[:, -1:]
            Y_pred_conceptor[:, k:k+1] = esn_conceptor.predict_sequence(X_test[:, k:k+1], U_test[:, k:k+h])[:, -1:]
            
        r2_dense = r_squared(Y_test_h[3:4, :], Y_pred_dense[3:4, :])
        r2_sparse = r_squared(Y_test_h[3:4, :], Y_pred_sparse[3:4, :])
        r2_conceptor = r_squared(Y_test_h[3:4, :], Y_pred_conceptor[3:4, :])
        
        print(f"{h:<20} | {r2_dense:<15.4f} | {r2_sparse:<15.4f} | {r2_conceptor:<15.4f}")

if __name__ == "__main__":
    main()
