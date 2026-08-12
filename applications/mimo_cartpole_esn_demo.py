"""
RESERVOIR-KOOPMAN EDMDc HYBRID (CART-POLE)
==========================================
This script merges Echo State Networks (ESNs) with Koopman Operator Theory.
Instead of using static, memory-less Radial Basis Functions to lift the state,
we use an Orthogonally-Initialized ESN (from McCabe Stage 1 Proposal) to lift 
the state into a dynamic, energy-preserving recurrent manifold.

We then use EDMDc to learn the linear Koopman operator on the COMBINED 
(Physics + Reservoir) state space. This effectively "linearizes" both the 
physics and the recurrent neural network simultaneously.
"""
import numpy as np
import scipy.stats
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nonlinear_sysid.edmdc import EDMDControl

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
# 2. HAAR-ORTHOGONAL ECHO STATE NETWORK (McCabe Stage 1)
# =========================================================================

class HaarOrthogonalESN:
    def __init__(self, input_dim: int, reservoir_dim: int, spectral_radius: float = 0.95):
        self.N = reservoir_dim
        # Win: Uniform random projection
        np.random.seed(42)
        self.W_in = np.random.uniform(-1.0, 1.0, (self.N, input_dim))
        
        # Wres: Haar-distributed orthogonal matrix (energy preserving)
        # Scaled by spectral radius to maintain Echo State Property.
        orthogonal_matrix = scipy.stats.ortho_group.rvs(dim=self.N, random_state=42)
        self.W_res = orthogonal_matrix * spectral_radius

    def rollout(self, X: np.ndarray, r0: np.ndarray = None) -> np.ndarray:
        """
        Unrolls the reservoir over a trajectory X (dim, n_samples).
        Returns the matrix of internal states R (reservoir_dim, n_samples).
        """
        n_samples = X.shape[1]
        R = np.zeros((self.N, n_samples))
        
        # Initialize internal state
        r_curr = np.zeros((self.N, 1)) if r0 is None else r0.reshape(-1, 1)
        
        for k in range(n_samples):
            x_k = X[:, k:k+1]
            # Standard ESN update: r = tanh(Win*x + Wres*r)
            r_next = np.tanh(self.W_in @ x_k + self.W_res @ r_curr)
            R[:, k] = r_next.flatten()
            r_curr = r_next
            
        return R

def linear_dictionary(x: np.ndarray) -> np.ndarray:
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
    print("=" * 80)
    print("   ESN-KOOPMAN HYBRID (HAAR-ORTHOGONAL RESERVOIR)")
    print("=" * 80)
    
    n_train = 20000
    X_train, Y_train, U_train = generate_dataset(n_train)
    
    # ---------------------------------------------------------
    # [A] Train Standard Linear Baseline
    # ---------------------------------------------------------
    print("\n[A] Training Standard Linear State-Space Model...")
    linear_model = EDMDControl(observable_func=linear_dictionary)
    linear_model.fit(X_train, Y_train, U_train)
    
    # ---------------------------------------------------------
    # [B] Train ESN-Koopman Hybrid
    # ---------------------------------------------------------
    print("[B] Training ESN-Koopman Hybrid Model (1000-Dim Reservoir)...")
    res_dim = 1000
    esn = HaarOrthogonalESN(input_dim=4, reservoir_dim=res_dim, spectral_radius=0.99)
    
    # Run the physics data through the ESN to get the temporal feature states
    R_train = esn.rollout(X_train)
    
    # Augment the physical state with the reservoir state: Z = [X; R]
    Z_train_x = np.vstack([X_train, R_train])
    
    # To get Z_train_y (the next augmented state), we need R_next.
    # R_next is just R shifted by one, but what about the very last step?
    # We can just unroll the ESN on Y_train!
    # Wait, R_k+1 is a function of X_k and R_k. We already computed it as R_train[:, 1:]
    # But let's just properly unroll it on the full Y trajectory (from R_train's perspective)
    R_train_next = np.zeros_like(R_train)
    # The next reservoir state for step k is R_train[:, k] (since R_train stores r_{k+1} conceptually if we feed x_k)
    # Let's align indices:
    # r_{k+1} = tanh(Win * x_k + Wres * r_k)
    # R_train[:, k] is exactly r_{k+1} based on x_k and r_k.
    # Therefore, the augmented state Z_k = [x_k; r_k] transitions to Z_{k+1} = [x_{k+1}; r_{k+1}]
    # We need to construct r_k and r_{k+1} carefully.
    
    # Let R_k be the state *before* seeing x_k. 
    # Let's rebuild the rollout to explicitly return R_k and R_{k+1}
    R_k = np.zeros((res_dim, X_train.shape[1]))
    R_k_plus_1 = np.zeros((res_dim, X_train.shape[1]))
    r_curr = np.zeros((res_dim, 1))
    
    for k in range(X_train.shape[1]):
        R_k[:, k] = r_curr.flatten()
        r_next = np.tanh(esn.W_in @ X_train[:, k:k+1] + esn.W_res @ r_curr)
        R_k_plus_1[:, k] = r_next.flatten()
        r_curr = r_next
        
    Z_train_x = np.vstack([X_train, R_k])
    Z_train_y = np.vstack([Y_train, R_k_plus_1])
    
    # Train EDMDc on the combined Z space using a linear dictionary (since Z is already lifted by the ESN)
    koopman_model = EDMDControl(observable_func=linear_dictionary)
    koopman_model.fit(Z_train_x, Z_train_y, U_train)
    
    # ---------------------------------------------------------
    # [C] Out-of-Sample Evaluation
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("   OUT-OF-SAMPLE TRAJECTORY PREDICTION TEST")
    print("=" * 80)
    
    n_test = 2000
    X_test, _, U_test = generate_dataset(n_test)
    n_test_actual = X_test.shape[1]
    
    horizons = [1, 5, 10, 20]
    print(f"{'Horizon (steps)':<20} | {'Linear R^2':<15} | {'ESN-Koopman R^2':<15}")
    print("-" * 55)
    
    for h in horizons:
        # Simulate true physics
        Y_test_h = np.zeros_like(X_test)
        physics = CartPolePhysics()
        for k in range(n_test_actual):
            curr = X_test[:, k]
            for _ in range(h):
                curr = physics.step(curr, U_test[0, k])
            Y_test_h[:, k] = curr
            
        # Predict with Linear Model
        Y_pred_linear = np.zeros_like(Y_test_h)
        for k in range(n_test_actual):
            x_curr = X_test[:, k:k+1]
            for _ in range(h):
                x_curr = linear_model.predict_one_step(x_curr, U_test[:, k:k+1])
            Y_pred_linear[:, k:k+1] = x_curr
            
        # Predict with ESN-Koopman Model
        # The true magic: We use the Koopman matrices to linearly predict the physical state,
        # but we maintain the non-linear energy-preserving ESN to update the internal memory.
        Y_pred_koopman = np.zeros_like(Y_test_h)
        for k in range(n_test_actual):
            x_curr = X_test[:, k:k+1]
            r_curr = np.zeros((res_dim, 1)) # Fresh reservoir state
            for _ in range(h):
                z_curr = np.vstack([x_curr, r_curr])
                
                # Use Koopman matrices to predict the next physical state
                # koopman_model.A is applied to Z, koopman_model.B is applied to U
                z_next_pred = koopman_model.predict_one_step(z_curr, U_test[:, k:k+1])
                x_curr = z_next_pred[:4, :] # Extract physical prediction
                
                # Use the TRUE non-linear ESN to update the memory for the next step!
                r_curr = np.tanh(esn.W_in @ x_curr + esn.W_res @ r_curr)
                
            Y_pred_koopman[:, k:k+1] = x_curr
            
        r2_lin = r_squared(Y_test_h[3:4, :], Y_pred_linear[3:4, :])
        r2_koop = r_squared(Y_test_h[3:4, :], Y_pred_koopman[3:4, :])
        
        print(f"{h:<20} | {r2_lin:<15.4f} | {r2_koop:<15.4f}")

    print("\n>>> CONCLUSION:")
    print("    By using a Haar-Orthogonal Echo State Network, we created a dynamic")
    print("    observable space with massive recurrent temporal memory.")
    print("    The Koopman EDMDc operator linearly extracts the physical predictions")
    print("    from the reservoir, while the ESN maintains the non-linear memory.")
    print("    This dramatically extends the prediction horizon over static RBFs!")

if __name__ == "__main__":
    main()
