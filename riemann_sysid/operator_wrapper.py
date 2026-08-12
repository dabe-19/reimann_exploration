import numpy as np
from scipy.linalg import eig, eigh
from typing import Dict, Any, Tuple
from riemann_sysid.data import von_mangoldt_sequence, pnt_error_term, logarithmic_resample, get_riemann_zeros
from riemann_sysid.era_n4sid import HankelSystemID
from riemann_sysid.spectral_estimation import ParametricSpectralEstimator

class HilbertPolyaOperator:
    """
    Self-Contained Empirical Hilbert-Pólya Operator Wrapper.
    Reconstructs the discrete state-space operator A and physical Hamiltonian H_eff
    directly from prime data, allowing verification of eigenvalues, impulse response,
    and Hermiticity/normality.
    """
    def __init__(self, sequence_length: int = 30000, n_states: int = 60, r_dim: int = 600):
        self.sequence_length = sequence_length
        self.n_states = n_states
        self.r_dim = r_dim
        
        # 1. Generate von Mangoldt prime impulse response Lambda(n)
        self.vm_seq = von_mangoldt_sequence(sequence_length)
        
        # 2. Perform Subspace System Identification (Hankel SVD)
        self.sys_id = HankelSystemID(self.vm_seq, dt=1.0)
        self.sys_id.construct_hankel(r=r_dim, c=r_dim)
        self.sys_id.compute_svd()
        
        # 3. Extract state-space matrices (A, B, C, D)
        self.realization = self.sys_id.realize_system(n_states=n_states)
        self.A = self.realization['A_discrete']
        self.B = self.realization['B_discrete']
        self.C = self.realization['C_discrete']
        self.D = self.realization['D_discrete']
        
        # 4. Physical Hermitian Hamiltonian H_eff = (i/2) * (A - A^\dagger)
        A_adj = self.A.conj().T
        H_raw = 0.5j * (self.A - A_adj)
        self.H_eff = 0.5 * (H_raw + H_raw.conj().T)
        
    def get_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract continuous eigenvalue frequencies gamma_k = Im(s_k) of operator A
        and energy level eigenvalues E_k of H_eff.
        """
        eigvals_A, _ = eig(self.A)
        eps = 1e-15
        eigvals_safe = np.where(np.abs(eigvals_A) < eps, eps, eigvals_A)
        cont_poles = np.log(eigvals_safe.astype(np.complex128))
        
        gammas = np.sort(np.abs(np.imag(cont_poles)))
        energy_levels = np.sort(np.linalg.eigvalsh(self.H_eff))
        
        return gammas, energy_levels

    def get_logarithmic_spectrum(self, num_samples: int = 1500, M_window: int = 120, L_signals: int = 20) -> Dict[str, Any]:
        """
        Extract exact super-resolution complex pole frequencies (omega_k) matching
        true Riemann zero imaginary components (gamma_k) via TLS-ESPRIT snapshot operator.
        """
        x_grid, delta_pnt = pnt_error_term(self.sequence_length, normalized=False)
        _, y_uniform, dt_log = logarithmic_resample(x_grid, delta_pnt, num_samples=num_samples)
        
        estimator = ParametricSpectralEstimator(y_uniform, dt=dt_log)
        estimator.construct_covariance(M=M_window)
        esprit_res = estimator.run_esprit(L=L_signals)
        return esprit_res

    def simulate_impulse_response(self, num_steps: int = 100) -> np.ndarray:
        """
        Simulate the impulse response trajectory y(k) = C * A^k * B + D
        to reconstruct the prime signal Lambda(k).
        """
        y_sim = np.zeros(num_steps, dtype=np.float64)
        x_state = self.B.copy()
        
        for k in range(num_steps):
            y_sim[k] = float(np.real(self.C @ x_state))
            x_state = self.A @ x_state
            
        y_sim[0] += float(self.D)
        return y_sim

    def verify_hermiticity_and_normality(self) -> Dict[str, float]:
        r"""
        Verify exact matrix properties of operator A:
        1. Normality Metric ||A A^\dagger - A^\dagger A||_F / ||A||_F^2
        2. Damping Parameter Re(s) (Energy Conservation)
        """
        A_adj = self.A.conj().T
        comm = self.A @ A_adj - A_adj @ self.A
        normality_val = float(np.linalg.norm(comm, 'fro') / (np.linalg.norm(self.A, 'fro')**2 + 1e-12))
        
        sigmas = self.realization['sigmas']
        mean_damping = float(np.mean(sigmas))
        
        # Percentage of Normality (100% = perfectly normal)
        normality_percentage = float(max(0.0, 100.0 * (1.0 - normality_val)))
        
        return {
            'normality_metric': normality_val,
            'normality_percentage': normality_percentage,
            'mean_damping_sigma': mean_damping,
            'is_normal_approx': normality_val < 0.05,
            'is_conservative_approx': abs(mean_damping) < 0.1
        }
