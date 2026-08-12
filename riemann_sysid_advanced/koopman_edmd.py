import numpy as np
from scipy.linalg import eig, inv
from typing import Dict, Any, Tuple, List, Optional
from riemann_sysid.data import generate_primes, von_mangoldt_sequence, get_riemann_zeros

class KoopmanAdeleEDMD:
    """
    Non-Commutative Dynamic Mode Decomposition (EDMD) over Connes' Adèle Space observables
    constructing the data-driven transfer operator K_koop.
    """
    def __init__(self, n_max: int = 20000):
        self.n_max = n_max
        self.primes = generate_primes(n_max)
        self.vm_seq = von_mangoldt_sequence(n_max)
        self.riemann_zeros = get_riemann_zeros(num_zeros=15)
        self.K_koop: Optional[np.ndarray] = None
        self.eigvals: Optional[np.ndarray] = None
        self.eigvecs: Optional[np.ndarray] = None

    def build_adele_dictionary(self, x: np.ndarray) -> np.ndarray:
        """
        Construct vector of dictionary observables Psi(x) for points x > 1:
        Psi(x) = [1, ln x, sqrt(x), sin(gamma_1 ln x), cos(gamma_1 ln x), ..., sin(gamma_k ln x), cos(gamma_k ln x)]^T
        """
        N = len(x)
        t = np.log(np.maximum(x, 1.0001))
        
        obs_list = [np.ones(N), t, np.sqrt(x) / np.sqrt(self.n_max)]
        
        for gamma in self.riemann_zeros:
            obs_list.append(np.sin(gamma * t))
            obs_list.append(np.cos(gamma * t))
            
        # Shape: (num_observables, N)
        return np.vstack(obs_list)

    def fit_koopman(self, step_size: int = 1) -> Dict[str, Any]:
        """
        Compute Koopman operator K_koop via Extended Dynamic Mode Decomposition (EDMD):
        K_koop = A_obs * G_obs^+
        where G_obs = (1/N) * sum Psi(x_k) Psi(x_k)^T, A_obs = (1/N) * sum Psi(x_{k+1}) Psi(x_k)^T
        """
        x_grid = np.arange(2, self.n_max + 1, dtype=np.float64)
        
        x_curr = x_grid[:-step_size]
        x_next = x_grid[step_size:]
        
        Psi_curr = self.build_adele_dictionary(x_curr)  # (M, N-1)
        Psi_next = self.build_adele_dictionary(x_next)  # (M, N-1)
        
        N_pts = Psi_curr.shape[1]
        
        # Gram matrices
        G_obs = (Psi_curr @ Psi_curr.T) / float(N_pts)
        A_obs = (Psi_next @ Psi_curr.T) / float(N_pts)
        
        # Koopman matrix K_koop = A_obs * G_obs^(-1)
        # Use pseudo-inverse for numerical stability
        self.K_koop = A_obs @ np.linalg.pinv(G_obs, rcond=1e-10)
        
        # Eigendecomposition of K_koop
        eigvals, eigvecs = eig(self.K_koop)
        idx = np.argsort(np.abs(eigvals))[::-1]
        self.eigvals = eigvals[idx]
        self.eigvecs = eigvecs[:, idx]
        
        # Unitarity metric ||K_koop^\dagger K_koop - I||_F / sqrt(M)
        K_adj = self.K_koop.conj().T
        unit_diff = K_adj @ self.K_koop - np.eye(self.K_koop.shape[0])
        unitarity_metric = float(np.linalg.norm(unit_diff, 'fro') / np.sqrt(self.K_koop.shape[0]))
        
        # Check eigenvalue unit circle proximity | |lambda| - 1 |
        magnitudes = np.abs(self.eigvals)
        mean_unit_error = float(np.mean(np.abs(magnitudes - 1.0)))

        return {
            'K_koop': self.K_koop,
            'eigvals': self.eigvals,
            'magnitudes': magnitudes,
            'unitarity_metric': unitarity_metric,
            'mean_unit_error': mean_unit_error,
            'num_observables': self.K_koop.shape[0]
        }
