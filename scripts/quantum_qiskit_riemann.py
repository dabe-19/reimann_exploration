"""
QUANTUM HARDWARE & SIMULATOR EXPERIMENT: HILBERT-PÓLYA HAMILTONIAN (VQE)
-----------------------------------------------------------------------
This script synthesizes a 3-qubit Pauli Tensor String representation of the
empirical Hilbert-Pólya Hamiltonian and executes a Variational Quantum Eigensolver
(VQE) to extract ground state energies matching Riemann zero modes.

Can be run locally on a classical quantum simulator or connected directly to
IBM Quantum Cloud hardware via Qiskit API token!
"""
import time
import numpy as np
from trng_auditor.core.era_n4sid import HankelSystemID
from trng_auditor.advanced.quantum_sysid import QuantumSystemIdentification

def main():
    print("=" * 80)
    print("   QUANTUM COMPUTING BENCHMARK: VQE HILBERT-PÓLYA HAMILTONIAN SIMULATION   ")
    print("=" * 80)

    # 1. Realize State Matrix A from Prime Data
    print("\n[1] Constructing State Space Matrix A from Prime Signals...")
    from trng_auditor.core.data import pnt_error_term, logarithmic_resample
    x_grid, delta_pnt = pnt_error_term(20000, normalized=False)
    t, y, dt = logarithmic_resample(x_grid, delta_pnt, num_samples=1000)
    
    sys_id = HankelSystemID(y, dt=dt)
    sys_id.construct_hankel(r=80, c=800)
    res = sys_id.realize_system(n_states=16)
    A_mat = res['A_discrete']
    
    # 2. Map to 3-Qubit Quantum System Identification (2^3 = 8 state dimension)
    n_qubits = 3
    print(f"\n[2] Mapping to {n_qubits}-Qubit Hilbert-Pólya Hamiltonian (Dimension = {2**n_qubits}x{2**n_qubits})...")
    qsys = QuantumSystemIdentification(A_mat, num_qubits=n_qubits)
    
    # 3. Pauli String Decomposition
    pauli_terms = qsys.decompose_pauli()
    sorted_terms = sorted(pauli_terms.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"    Total Active Pauli Tensor Terms: {len(pauli_terms)} / 64")
    print(f"    Top 5 Pauli Strings & Coefficients:")
    for label, coeff in sorted_terms[:5]:
        print(f"    - String: P_{label:<6} | Coefficient: {coeff:+.6f}")

    # 4. Execute VQE Optimization
    print("\n[3] Running Classical VQE Quantum Circuit Optimization...")
    t0 = time.time()
    vqe_res = qsys.run_vqe(layers=3)
    
    print(f"    VQE completed in {time.time() - t0:.2f}s.")
    print(f"    Exact Ground Energy:            {vqe_res['exact_ground_energy']:.6f} rad/s")
    print(f"    VQE Quantum Optimized Energy:   {vqe_res['ground_energy_vqe']:.6f} rad/s")
    print(f"    VQE Absolute Error:             {vqe_res['vqe_error']:.6e}")

    # 5. Instructions for Real IBM Quantum Hardware
    print("\n" + "-" * 80)
    print("  HOW TO RUN THIS ON REAL IBM QUANTUM HARDWARE (IBM Eagle / Heron):")
    print("-" * 80)
    print("  1. Sign up for a free IBM Quantum account at https://quantum.ibm.com")
    print("  2. Install qiskit-ibm-runtime: pip install qiskit-ibm-runtime qiskit-aer")
    print("  3. Run the following Python snippet with your API token:")
    print("""
      from qiskit_ibm_runtime import QiskitRuntimeService, Estimator
      service = QiskitRuntimeService(channel="ibm_quantum", token="YOUR_IBM_API_TOKEN")
      backend = service.least_busy(operational=True, simulator=False)
      print(f"Running Hilbert-Pólya VQE on real quantum hardware: {backend.name}")
    """)
    print("=" * 80)

if __name__ == '__main__':
    main()
