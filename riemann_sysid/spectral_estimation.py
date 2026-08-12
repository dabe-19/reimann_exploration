import numpy as np
from scipy.linalg import eig, svd
from typing import Dict, Any, Tuple, Optional

class ParametricSpectralEstimator:
    """
    Parametric Super-Resolution Spectral Estimation using TLS-ESPRIT and MUSIC
    applied to time-series snapshots (e.g. logarithmic PNT error term psi(e^t) - e^t).
    """
    def __init__(self, y: np.ndarray, dt: float):
        self.y = np.asarray(y, dtype=np.float64)
        self.dt = dt
        self.M: Optional[int] = None
        self.L: Optional[int] = None
        self.Rxx: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self.Us: Optional[np.ndarray] = None
        self.Un: Optional[np.ndarray] = None

    def construct_covariance(self, M: int) -> np.ndarray:
        """
        Form snapshot matrix and estimate M x M spatial autocorrelation covariance matrix Rxx.
        """
        N = len(self.y)
        if M >= N:
            raise ValueError(f"Window length M={M} must be strictly less than signal length N={N}.")
            
        K = N - M + 1
        X = np.zeros((M, K), dtype=np.complex128)
        for i in range(M):
            X[i, :] = self.y[i : i + K]
            
        self.M = M
        self.Rxx = (X @ X.conj().T) / float(K)
        
        # Eigendecomposition of Rxx
        eigvals, eigvecs = eig(self.Rxx)
        # Sort in descending order of eigenvalue magnitude
        idx = np.argsort(np.abs(eigvals))[::-1]
        self.eigenvalues = np.real(eigvals[idx])
        self.eigenvectors = eigvecs[:, idx]
        
        return self.Rxx

    def partition_subspaces(self, L: int) -> Tuple[np.ndarray, np.ndarray]:
        """Partition matrix into L-dimensional signal subspace Us and noise subspace Un."""
        if self.eigenvectors is None:
            raise RuntimeError("Covariance matrix Rxx has not been constructed. Call construct_covariance first.")
        if L >= self.M:
            raise ValueError(f"Signal subspace dimension L={L} must be smaller than window size M={self.M}.")
            
        self.L = L
        self.Us = self.eigenvectors[:, :L]
        self.Un = self.eigenvectors[:, L:]
        return self.Us, self.Un

    def run_esprit(self, L: int) -> Dict[str, Any]:
        """
        Execute Total Least Squares ESPRIT (TLS-ESPRIT) to extract complex poles s_k = sigma_k + i * omega_k.
        """
        if self.Us is None or self.L != L:
            self.partition_subspaces(L)
            
        Us1 = self.Us[:-1, :]  # (M-1) x L
        Us2 = self.Us[1:, :]   # (M-1) x L
        
        # Form combined matrix [Us1, Us2]
        Us12 = np.hstack([Us1, Us2])  # (M-1) x 2L
        
        # SVD of Us12
        _, _, Vt = svd(Us12, full_matrices=False)
        V = Vt.conj().T
        
        # Partition V into 4 (L x L) sub-matrices
        V12 = V[:L, L:2*L]
        V22 = V[L:2*L, L:2*L]
        
        # TLS rotation matrix Psi = - V12 * V22^\dagger
        Psi = - V12 @ np.linalg.pinv(V22)
        
        # Eigenvalues of Psi are the discrete complex poles z_k
        z_k, _ = eig(Psi)
        
        # Convert discrete poles z_k to continuous poles s_k = ln(z_k) / dt
        # Handle zero or negative numerical anomalies safely
        eps = 1e-15
        z_k_safe = np.where(np.abs(z_k) < eps, eps, z_k)
        cont_poles = np.log(z_k_safe.astype(np.complex128)) / self.dt
        
        sigmas = np.real(cont_poles)
        omegas = np.abs(np.imag(cont_poles))
        
        # Sort poles by imaginary frequency omega
        sort_idx = np.argsort(omegas)
        
        return {
            'L_signals': L,
            'discrete_poles': z_k[sort_idx],
            'continuous_poles': cont_poles[sort_idx],
            'sigmas': sigmas[sort_idx],
            'omegas': omegas[sort_idx],
            'mean_sigma': float(np.mean(sigmas)),
            'std_sigma': float(np.std(sigmas)),
            'eigenvalues': self.eigenvalues
        }

    def run_music(self, omega_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute MUSIC pseudo-spectrum over candidate frequencies omega_range.
        Returns: (omega_range, pseudospectrum)
        """
        if self.Un is None:
            raise RuntimeError("Noise subspace Un is not available. Run partition_subspaces first.")
            
        M = self.M
        pseudospectrum = np.zeros(len(omega_range), dtype=np.float64)
        
        # Precompute Un * Un^H
        UnUnH = self.Un @ self.Un.conj().T
        
        for idx, w in enumerate(omega_range):
            # Steering vector a(w)
            a = np.exp(1j * w * self.dt * np.arange(M))
            denom = np.real(a.conj().T @ UnUnH @ a)
            pseudospectrum[idx] = 1.0 / (denom + 1e-15)
            
        return omega_range, pseudospectrum
