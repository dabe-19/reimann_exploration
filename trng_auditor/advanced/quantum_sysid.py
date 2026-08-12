import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any, List, Tuple, Optional

class QuantumSystemIdentification:
    """
    Quantum-Classical Hybrid System Identification:
    Decomposes continuous state-space operator A into multi-qubit Pauli strings
    and optimizes Variational Quantum Eigensolver (VQE) ansatz circuits to reconstruct
    the physical quantum Hamiltonian H_eff.
    """
    def __init__(self, A_matrix: np.ndarray, num_qubits: int = 3):
        self.num_qubits = num_qubits
        self.dim = 2**num_qubits
        
        # Resize/truncate A_matrix to dim x dim
        n_orig = A_matrix.shape[0]
        if n_orig >= self.dim:
            A_sub = A_matrix[:self.dim, :self.dim]
        else:
            A_sub = np.zeros((self.dim, self.dim), dtype=A_matrix.dtype)
            A_sub[:n_orig, :n_orig] = A_matrix
            
        # Effective Hermitian Hamiltonian H_eff = i/2 * (A - A^H)
        A_adj = A_sub.conj().T
        self.H_eff = 0.5j * (A_sub - A_adj)
        # Ensure purely real Hermitian matrix
        self.H_eff = 0.5 * (self.H_eff + self.H_eff.conj().T)
        
        # Standard Pauli matrices
        self.I = np.eye(2, dtype=np.complex128)
        self.X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        self.Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        self.pauli_dict = {'I': self.I, 'X': self.X, 'Y': self.Y, 'Z': self.Z}

    def decompose_pauli(self) -> Dict[str, float]:
        """
        Decompose H_eff into N_q-qubit Pauli tensor strings:
        H_eff = sum c_sigma P_sigma
        """
        pauli_keys = ['I', 'X', 'Y', 'Z']
        N = self.num_qubits
        
        def get_tensor_string(label: str) -> np.ndarray:
            mat = self.pauli_dict[label[0]]
            for char in label[1:]:
                mat = np.kron(mat, self.pauli_dict[char])
            return mat

        coeffs = {}
        # Generate all 4^N Pauli labels
        import itertools
        all_labels = [''.join(p) for p in itertools.product(pauli_keys, repeat=N)]
        
        for label in all_labels:
            P_mat = get_tensor_string(label)
            c_val = float(np.real(np.trace(self.H_eff @ P_mat) / float(self.dim)))
            if abs(c_val) > 1e-6:
                coeffs[label] = c_val
                
        return coeffs

    def _cnot_gate(self, control: int, target: int) -> np.ndarray:
        """Build full 2^N x 2^N CNOT matrix for control→target qubits."""
        dim = self.dim
        N = self.num_qubits
        gate = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim):
            bits = list(format(i, f'0{N}b'))
            if bits[control] == '1':
                # Flip target bit
                bits[target] = '0' if bits[target] == '1' else '1'
            j = int(''.join(bits), 2)
            gate[j, i] = 1.0
        return gate

    def ansatz_state(self, params: np.ndarray) -> np.ndarray:
        """
        Hardware-efficient ansatz state |psi(theta)> = U(theta)|0...0>.
        Layer of Ry(theta_j) rotations followed by CNOT chain entanglers.
        """
        N = self.num_qubits
        state = np.zeros(self.dim, dtype=np.complex128)
        state[0] = 1.0  # Initial state |0...0>
        
        num_layers = len(params) // N
        
        # Pre-build CNOT gates (they don't depend on parameters)
        cnot_gates = [self._cnot_gate(q, q + 1) for q in range(N - 1)]
        
        for layer in range(num_layers):
            # Single-qubit Ry rotations
            layer_params = params[layer * N : (layer + 1) * N]
            u_layer = np.eye(1, dtype=np.complex128)
            for theta in layer_params:
                ry = np.array([
                    [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
                    [np.sin(theta / 2.0),  np.cos(theta / 2.0)]
                ], dtype=np.complex128)
                u_layer = np.kron(u_layer, ry)
                
            state = u_layer @ state
            
            # Apply CNOT entangling chain
            for cnot in cnot_gates:
                state = cnot @ state
                
        return state / np.linalg.norm(state)

    def energy_expectation(self, params: np.ndarray) -> float:
        """Compute VQE energy expectation E(theta) = <psi(theta)| H_eff |psi(theta)>."""
        psi = self.ansatz_state(params)
        val = float(np.real(psi.conj().T @ self.H_eff @ psi))
        return val

    def run_vqe(self, layers: int = 3) -> Dict[str, Any]:
        """
        Run Variational Quantum Eigensolver (VQE) optimization to find ground and excited states.
        """
        num_params = self.num_qubits * layers
        init_params = np.random.uniform(-np.pi, np.pi, num_params)
        
        # Optimize ground state energy
        res = minimize(self.energy_expectation, init_params, method='BFGS')
        ground_energy = float(res.fun)
        
        # Exact eigenvalues of H_eff for verification
        exact_eigvals = np.sort(np.linalg.eigvalsh(self.H_eff))
        
        # Measure Pauli decomposition
        pauli_coeffs = self.decompose_pauli()

        return {
            'num_qubits': self.num_qubits,
            'dim': self.dim,
            'ground_energy_vqe': ground_energy,
            'exact_ground_energy': float(exact_eigvals[0]),
            'exact_eigenvalues': exact_eigvals,
            'vqe_error': abs(ground_energy - exact_eigvals[0]),
            'pauli_coefficients': pauli_coeffs,
            'num_pauli_terms': len(pauli_coeffs)
        }
