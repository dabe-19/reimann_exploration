import numpy as np
from scipy.linalg import svd, eig, logm
from typing import Dict, Any, Tuple, Optional

class HankelSystemID:
    """
    Subspace System Identification using the Eigensystem Realization Algorithm (ERA) / N4SID
    for discrete impulse response sequence y(k).
    """
    def __init__(self, y: np.ndarray, dt: float = 1.0):
        self.y = np.asarray(y, dtype=np.float64)
        self.dt = dt
        self.r: Optional[int] = None
        self.c: Optional[int] = None
        self.H0: Optional[np.ndarray] = None
        self.H1: Optional[np.ndarray] = None
        self.U: Optional[np.ndarray] = None
        self.S: Optional[np.ndarray] = None
        self.Vt: Optional[np.ndarray] = None

    def construct_hankel(self, r: int, c: int) -> Tuple[np.ndarray, np.ndarray]:
        """Construct r x c block Hankel matrix H0 and shifted matrix H1."""
        if r + c > len(self.y):
            raise ValueError(f"Hankel dimensions r={r}, c={c} (r+c={r+c}) exceed signal length {len(self.y)}.")
            
        self.r = r
        self.c = c
        self.H0 = np.zeros((r, c), dtype=np.float64)
        self.H1 = np.zeros((r, c), dtype=np.float64)
        
        for i in range(r):
            self.H0[i, :] = self.y[i : i + c]
            self.H1[i, :] = self.y[i + 1 : i + c + 1]
            
        return self.H0, self.H1

    def compute_svd(self) -> np.ndarray:
        """Compute SVD of the Hankel matrix H0."""
        if self.H0 is None:
            raise RuntimeError("Hankel matrix H0 has not been constructed. Call construct_hankel first.")
        self.U, self.S, self.Vt = svd(self.H0, full_matrices=False)
        return self.S

    def realize_system(self, n_states: int) -> Dict[str, Any]:
        """
        Extract continuous state-space realization (A_cont, B_cont, C_cont, D_cont)
        and analyze properties of operator A.
        """
        if self.U is None or self.S is None or self.Vt is None:
            self.compute_svd()
            
        n = min(n_states, len(self.S))
        
        # Truncate SVD components
        U_n = self.U[:, :n]
        S_n = self.S[:n]
        V_n = self.Vt[:n, :].T  # c x n
        
        S_sqrt = np.diag(np.sqrt(S_n))
        S_sqrt_inv = np.diag(1.0 / np.sqrt(S_n))
        
        # Discrete-time realization A_d, B_d, C_d
        A_d = S_sqrt_inv @ U_n.T @ self.H1 @ V_n @ S_sqrt_inv
        B_d = S_sqrt @ V_n[0, :].T  # shape (n,)
        C_d = U_n[0, :] @ S_sqrt     # shape (n,)
        D_d = self.y[0]
        
        # Discrete eigenvalues and continuous-time pole conversion s = (1/dt) * ln(lambda)
        eigvals_d, eigvecs_d = eig(A_d)
        
        # Avoid log of zero by adding epsilon if any eigenvalue is exactly 0
        eps = 1e-15
        eigvals_d_safe = np.where(np.abs(eigvals_d) < eps, eps, eigvals_d)
        cont_poles = np.log(eigvals_d_safe.astype(np.complex128)) / self.dt
        
        sigmas = np.real(cont_poles)
        gammas = np.abs(np.imag(cont_poles))
        
        # Operator Normality check: ||A A^H - A^H A||_F / ||A||_F^2
        A_adj = A_d.conj().T
        comm = A_d @ A_adj - A_adj @ A_d
        normality_metric = float(np.linalg.norm(comm, 'fro') / (np.linalg.norm(A_d, 'fro')**2 + 1e-12))
        
        # Symmetry / Hermiticity metric of A_d
        hermitian_diff = np.linalg.norm(A_d - A_adj, 'fro') / (np.linalg.norm(A_d, 'fro') + 1e-12)
        skew_hermitian_diff = np.linalg.norm(A_d + A_adj, 'fro') / (np.linalg.norm(A_d, 'fro') + 1e-12)

        return {
            'n_states': n,
            'singular_values': self.S[:n],
            'A_discrete': A_d,
            'B_discrete': B_d,
            'C_discrete': C_d,
            'D_discrete': D_d,
            'discrete_eigvals': eigvals_d,
            'continuous_poles': cont_poles,
            'sigmas': sigmas,
            'gammas': gammas,
            'normality_metric': normality_metric,
            'hermitian_diff': float(hermitian_diff),
            'skew_hermitian_diff': float(skew_hermitian_diff)
        }
