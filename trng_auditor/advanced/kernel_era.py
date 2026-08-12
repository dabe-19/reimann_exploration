import numpy as np
from scipy.linalg import eigh, eig
from typing import Dict, Any, Tuple, Optional, Callable

class KernelHankelSystemID:
    """
    Kernelized Subspace System Identification (Kernel-ERA / Kernel-Hankel SVD)
    mapping prime snapshot signals into an infinite-dimensional RKHS.
    """
    def __init__(self, y: np.ndarray, dt: float = 1.0):
        self.y = np.asarray(y, dtype=np.float64)
        self.dt = dt
        self.r: Optional[int] = None
        self.c: Optional[int] = None
        self.Gram0: Optional[np.ndarray] = None
        self.Gram1: Optional[np.ndarray] = None
        self.eigvals: Optional[np.ndarray] = None
        self.eigvecs: Optional[np.ndarray] = None

    @staticmethod
    def dirichlet_prime_kernel(u: np.ndarray, v: np.ndarray, num_primes: int = 50) -> float:
        """
        Dirichlet Prime Mercer Kernel:
        k(u, v) = sum_{p <= P} cos(ln(p) * (u - v)) / sqrt(p)
        """
        diff = float(np.mean(u) - np.mean(v))
        # Simple prime list
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97][:num_primes]
        val = sum(np.cos(np.log(p) * diff) / np.sqrt(p) for p in primes)
        return float(val)

    @staticmethod
    def rbf_kernel(u: np.ndarray, v: np.ndarray, gamma: float = 0.05) -> float:
        """Gaussian RBF Mercer Kernel: k(u, v) = exp(-gamma * ||u - v||^2)"""
        dist_sq = float(np.sum((u - v)**2))
        return float(np.exp(-gamma * dist_sq))

    def construct_kernel_gram(self, r: int, c: int, kernel_type: str = 'rbf', gamma: float = 0.05) -> np.ndarray:
        """
        Construct r x r Gram matrix K0_ij = k(h_i, h_j) where h_i is the i-th Hankel row vector.
        """
        if r + c > len(self.y):
            raise ValueError(f"Hankel dimensions r={r}, c={c} exceed signal length {len(self.y)}.")
            
        self.r = r
        self.c = c
        
        # Build Hankel snapshot vectors h_i of length c
        H = np.zeros((r, c), dtype=np.float64)
        for i in range(r):
            H[i, :] = self.y[i : i + c]
            
        # Select kernel function
        if kernel_type == 'rbf':
            kernel_fn = lambda u, v: self.rbf_kernel(u, v, gamma=gamma)
        elif kernel_type == 'dirichlet':
            kernel_fn = lambda u, v: self.dirichlet_prime_kernel(u, v)
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")
            
        self.Gram0 = np.zeros((r, r), dtype=np.float64)
        self.Gram1 = np.zeros((r, r), dtype=np.float64)
        
        for i in range(r):
            for j in range(i, r):
                val0 = kernel_fn(H[i, :], H[j, :])
                self.Gram0[i, j] = val0
                self.Gram0[j, i] = val0
                
                # Shifted snapshot comparison for Gram1
                if j + 1 < r:
                    val1 = kernel_fn(H[i, :], H[j+1, :])
                    self.Gram1[i, j] = val1
                if i + 1 < r:
                    val1_sym = kernel_fn(H[i+1, :], H[j, :])
                    self.Gram1[j, i] = val1_sym
                    
        return self.Gram0

    def compute_rkhs_svd(self) -> np.ndarray:
        """Compute eigendecomposition of the RKHS Gram matrix Gram0."""
        if self.Gram0 is None:
            raise RuntimeError("Gram matrix has not been constructed. Call construct_kernel_gram first.")
            
        # Eigendecomposition of symmetric positive semi-definite Gram matrix
        w, v = eigh(self.Gram0)
        # Sort in descending order
        idx = np.argsort(w)[::-1]
        self.eigvals = np.maximum(w[idx], 1e-15)
        self.eigvecs = v[:, idx]
        
        # Equivalent singular values in RKHS are sqrt(eigenvalues)
        return np.sqrt(self.eigvals)

    def realize_kernel_system(self, n_states: int) -> Dict[str, Any]:
        """
        Realize continuous state-space operator A in RKHS subspace.
        """
        if self.eigvals is None or self.eigvecs is None:
            self.compute_rkhs_svd()
            
        n = min(n_states, len(self.eigvals))
        
        # Subspace basis in RKHS
        V_n = self.eigvecs[:, :n]
        S_n = np.sqrt(self.eigvals[:n])
        S_n_inv = 1.0 / np.maximum(S_n, 1e-12)
        
        # Reduced discrete operator A_r = S_n_inv * V_n^T * Gram1 * V_n * S_n_inv
        A_d = np.diag(S_n_inv) @ V_n.T @ self.Gram1 @ V_n @ np.diag(S_n_inv)
        
        # Spectrum analysis
        eigvals_d, _ = eig(A_d)
        eps = 1e-15
        eigvals_d_safe = np.where(np.abs(eigvals_d) < eps, eps, eigvals_d)
        cont_poles = np.log(eigvals_d_safe.astype(np.complex128)) / self.dt
        
        sigmas = np.real(cont_poles)
        gammas = np.abs(np.imag(cont_poles))
        
        # Normality in RKHS
        A_adj = A_d.conj().T
        comm = A_d @ A_adj - A_adj @ A_d
        normality_metric = float(np.linalg.norm(comm, 'fro') / (np.linalg.norm(A_d, 'fro')**2 + 1e-12))

        return {
            'n_states': n,
            'rkhs_singular_values': S_n,
            'A_discrete': A_d,
            'continuous_poles': cont_poles,
            'sigmas': sigmas,
            'gammas': gammas,
            'normality_metric': normality_metric
        }
