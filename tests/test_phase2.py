import numpy as np
import pytest
from trng_auditor.advanced.kernel_era import KernelHankelSystemID
from trng_auditor.advanced.koopman_edmd import KoopmanAdeleEDMD
from trng_auditor.advanced.quantum_sysid import QuantumSystemIdentification
from trng_auditor.advanced.theoretical_proofs import TheoreticalProofsVerifier

def test_kernel_era():
    k = np.arange(100)
    y = np.exp(-0.05 * k) * np.cos(0.4 * k)
    
    sys_id = KernelHankelSystemID(y, dt=0.1)
    sys_id.construct_kernel_gram(r=30, c=30, kernel_type='rbf', gamma=0.01)
    s_vals = sys_id.compute_rkhs_svd()
    assert len(s_vals) == 30
    
    res = sys_id.realize_kernel_system(n_states=4)
    assert 'normality_metric' in res
    assert len(res['continuous_poles']) == 4

def test_koopman_edmd():
    edmd = KoopmanAdeleEDMD(n_max=1000)
    res = edmd.fit_koopman()
    assert 'unitarity_metric' in res
    assert 'mean_unit_error' in res
    assert res['unitarity_metric'] < 0.35

def test_quantum_sysid():
    # Synthetic 8x8 matrix
    A_mat = np.random.randn(8, 8) + 1j * np.random.randn(8, 8)
    q_sys = QuantumSystemIdentification(A_mat, num_qubits=3)
    
    pauli_coeffs = q_sys.decompose_pauli()
    assert len(pauli_coeffs) > 0
    
    vqe_res = q_sys.run_vqe(layers=2)
    assert 'ground_energy_vqe' in vqe_res
    assert 'exact_ground_energy' in vqe_res

def test_theoretical_proofs():
    ver_t1 = TheoreticalProofsVerifier.verify_asymptotic_normality(n_scales=[1000, 2000])
    assert 'theorem_name' in ver_t1
    
    ver_t2 = TheoreticalProofsVerifier.verify_koopman_unitarity(n_max=1500)
    assert 'theorem_name' in ver_t2

if __name__ == '__main__':
    pytest.main(['-v', __file__])
