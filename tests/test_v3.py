import numpy as np
import pytest
from riemann_sysid.data import generate_primes
from riemann_sysid_v3.galois_koopman import GaloisKoopmanTomography, AdelicProductOperator
from riemann_sysid_v3.rigorous_analysis import RigorousUnitarityAnalyzer

def test_galois_koopman():
    primes = generate_primes(1000)
    tomog = GaloisKoopmanTomography(p=5, sequence=primes)
    res = tomog.compute_galois_spectrum()
    
    assert res['prime_p'] == 5
    assert res['transition_matrix'].shape == (5, 5)
    assert res['is_stochastic'] is True
    # Maximum eigenvalue of stochastic transition matrix is 1.0
    assert np.isclose(np.max(res['magnitudes']), 1.0)

def test_adelic_product_operator():
    primes = generate_primes(1000)
    adelic = AdelicProductOperator(primes=[2, 3, 5], sequence=primes)
    res = adelic.build_adelic_operator(max_dim=100)
    
    assert 'adelic_dim' in res
    assert res['adelic_dim'] == 2 * 3 * 5  # 30
    assert len(res['eigvals']) == 30

def test_rigorous_analysis():
    res = RigorousUnitarityAnalyzer.analyze_window_asymptotics(n_max=5000, windows=[20, 40])
    assert 'window_results' in res
    
    layman = RigorousUnitarityAnalyzer.layman_reality_check()
    assert 'layman_explanation' in layman
    assert 'practical_utility' in layman
    assert 'disproof_risk_analysis' in layman

if __name__ == '__main__':
    pytest.main(['-v', __file__])
