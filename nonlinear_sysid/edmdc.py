import numpy as np
from typing import Callable, Tuple

class EDMDControl:
    """
    Extended Dynamic Mode Decomposition with Control (EDMDc).
    
    A data-driven Koopman Operator approach for non-linear MIMO system identification.
    It lifts the non-linear state space x_k into a higher-dimensional observable space 
    Psi(x_k) where the dynamics behave linearly with respect to the control input u_k.
    
    Model: Psi(x_{k+1}) = A_koop * Psi(x_k) + B_koop * u_k
    """
    def __init__(self, observable_func: Callable[[np.ndarray], np.ndarray]):
        """
        Args:
            observable_func: A function that takes a state matrix X (n_states, n_samples)
                             and returns a lifted observable matrix Psi(X) (n_obs, n_samples).
        """
        self.observable_func = observable_func
        self.A_koop = None
        self.B_koop = None
        
    def fit(self, X: np.ndarray, Y: np.ndarray, U: np.ndarray) -> None:
        """
        Fits the Koopman matrices A_koop and B_koop using least-squares.
        
        Args:
            X: Current states, shape (n_states, n_samples)
            Y: Next states (one-step ahead), shape (n_states, n_samples)
            U: Control inputs applied at X, shape (n_inputs, n_samples)
        """
        # Lift states into the observable dictionary space
        Psi_X = self.observable_func(X) # (n_obs, n_samples)
        Psi_Y = self.observable_func(Y) # (n_obs, n_samples)
        
        # Concatenate observables and controls: Z = [Psi(X); U]
        # Shape: (n_obs + n_inputs, n_samples)
        Z = np.vstack([Psi_X, U])
        
        # Solve least squares: [A_koop, B_koop] = Psi_Y * Z^+
        # Using pseudo-inverse for stability
        AB_koop = Psi_Y @ np.linalg.pinv(Z, rcond=1e-10)
        
        n_obs = Psi_X.shape[0]
        self.A_koop = AB_koop[:, :n_obs]
        self.B_koop = AB_koop[:, n_obs:]
        
    def predict_one_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Predicts the next state x_{k+1} given current state x_k and control u_k.
        
        Args:
            x: Current state (n_states, 1) or (n_states, n_samples)
            u: Control input (n_inputs, 1) or (n_inputs, n_samples)
            
        Returns:
            The predicted next state (in original state space).
            Assumes the first n_states of the observable vector are the original states itself.
        """
        if self.A_koop is None or self.B_koop is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
            
        Psi_x = self.observable_func(x)
        Psi_next = self.A_koop @ Psi_x + self.B_koop @ u
        
        # Extract the original states from the lifted space.
        # This requires the observable function to output the original states as the first elements.
        n_states = x.shape[0]
        return Psi_next[:n_states, :]
        
    def simulate(self, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Simulates the system forward for multiple steps given an initial state and a control sequence.
        
        Args:
            x0: Initial state (n_states, 1)
            U: Control sequence (n_inputs, n_steps)
            
        Returns:
            X_sim: Simulated state trajectory (n_states, n_steps)
        """
        n_steps = U.shape[1]
        n_states = x0.shape[0]
        X_sim = np.zeros((n_states, n_steps))
        
        x_curr = x0.copy()
        for k in range(n_steps):
            u_k = U[:, k:k+1]
            x_next = self.predict_one_step(x_curr, u_k)
            X_sim[:, k] = x_next.flatten()
            x_curr = x_next
            
        return X_sim
