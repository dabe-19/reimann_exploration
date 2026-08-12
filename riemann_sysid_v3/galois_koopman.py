import numpy as np
from scipy.linalg import eig
from typing import Dict, Any, List, Tuple, Optional
from riemann_sysid.data import generate_primes, von_mangoldt_sequence

class GaloisKoopmanTomography:
    """
    Koopman Operator Tomography on Prime Fields F_p (Galois Fields)
    for finite-characteristic system identification.
    """
    def __init__(self, p: int, sequence: np.ndarray):
        self.p = p
        self.sequence = np.asarray(sequence, dtype=int)
        self.mod_seq = self.sequence % p
        self.P_mat: Optional[np.ndarray] = None
        self.K_p: Optional[np.ndarray] = None
        self.eigvals: Optional[np.ndarray] = None

    def construct_transition_matrix(self) -> np.ndarray:
        """
        Build p x p stochastic transition probability matrix P^{(p)} over F_p.
        P_{ij} = P( x_{n+1} = j | x_n = i )
        """
        p = self.p
        C = np.zeros((p, p), dtype=np.float64)
        
        s_curr = self.mod_seq[:-1]
        s_next = self.mod_seq[1:]
        
        for i in range(len(s_curr)):
            C[s_curr[i], s_next[i]] += 1.0
            
        # Normalize rows to form stochastic matrix
        row_sums = C.sum(axis=1, keepdims=True)
        # Avoid division by zero for unvisited states
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        self.P_mat = C / row_sums
        
        # Dual Koopman operator on observable vectors is K_p = P_mat^T
        self.K_p = self.P_mat.T
        return self.K_p

    def compute_galois_spectrum(self) -> Dict[str, Any]:
        """
        Compute eigendecomposition of Galois Koopman operator K_p.
        Verify roots-of-unity spectral property |lambda_k| = 1.
        """
        if self.K_p is None:
            self.construct_transition_matrix()
            
        eigvals, eigvecs = eig(self.K_p)
        idx = np.argsort(np.abs(eigvals))[::-1]
        self.eigvals = eigvals[idx]
        
        magnitudes = np.abs(self.eigvals)
        # Roots of unity have |lambda| = 1.0000
        unit_circle_mask = np.isclose(magnitudes, 1.0, atol=1e-3)
        num_roots_of_unity = int(np.sum(unit_circle_mask))

        return {
            'prime_p': self.p,
            'transition_matrix': self.P_mat,
            'koopman_matrix': self.K_p,
            'eigvals': self.eigvals,
            'magnitudes': magnitudes,
            'num_roots_of_unity': num_roots_of_unity,
            'is_stochastic': bool(np.allclose(self.P_mat.sum(axis=1), 1.0))
        }

class AdelicProductOperator:
    """
    Global Adèlic Product Operator K_A = K^{(p_1)} (x) K^{(p_2)} (x) ... (x) K^{(p_m)}
    unifying local Galois field transition operators across prime bases.
    """
    def __init__(self, primes: List[int], sequence: np.ndarray):
        self.primes = primes
        self.sequence = sequence
        self.local_operators: List[np.ndarray] = []
        self.K_adelic: Optional[np.ndarray] = None

    def build_adelic_operator(self, max_dim: int = 256) -> Dict[str, Any]:
        """
        Construct global adèlic Kronecker tensor product operator K_adelic.
        """
        self.local_operators = []
        for p in self.primes:
            tomog = GaloisKoopmanTomography(p, self.sequence)
            K_p = tomog.construct_transition_matrix()
            self.local_operators.append(K_p)
            
        # Form Kronecker product up to max_dim size
        K_prod = self.local_operators[0]
        for K_p in self.local_operators[1:]:
            if K_prod.shape[0] * K_p.shape[0] > max_dim:
                break
            K_prod = np.kron(K_prod, K_p)
            
        self.K_adelic = K_prod
        eigvals, _ = eig(self.K_adelic)
        idx = np.argsort(np.abs(eigvals))[::-1]
        
        return {
            'primes_used': self.primes[:len(self.local_operators)],
            'adelic_dim': self.K_adelic.shape[0],
            'K_adelic': self.K_adelic,
            'eigvals': eigvals[idx],
            'magnitudes': np.abs(eigvals[idx])
        }
