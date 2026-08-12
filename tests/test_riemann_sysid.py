import numpy as np
import pytest
from trng_auditor.core.data import generate_primes, von_mangoldt_sequence, chebyshev_psi, pnt_error_term, logarithmic_resample, get_riemann_zeros
from trng_auditor.core.era_n4sid import HankelSystemID
from trng_auditor.core.spectral_estimation import ParametricSpectralEstimator
from trng_auditor.core.delay_differential import DelayDifferentialAnalyzer

def test_data_generation():
    primes = generate_primes(30)
    expected_primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
    np.testing.assert_array_equal(primes, expected_primes)

    vm = von_mangoldt_sequence(10)
    # Lambda(1)=0, Lambda(2)=ln 2, Lambda(3)=ln 3, Lambda(4)=ln 2, Lambda(5)=ln 5, Lambda(6)=0, Lambda(7)=ln 7, Lambda(8)=ln 2, Lambda(9)=ln 3, Lambda(10)=0
    assert vm[0] == 0.0
    assert np.isclose(vm[1], np.log(2))
    assert np.isclose(vm[2], np.log(3))
    assert np.isclose(vm[3], np.log(2))  # 4 = 2^2
    assert vm[5] == 0.0  # 6 is composite with 2 distinct prime factors

    psi = chebyshev_psi(10)
    assert np.isclose(psi[-1], np.sum(vm))

    x, delta = pnt_error_term(10)
    assert len(x) == 10
    assert len(delta) == 10

def test_era_n4sid_synthetic():
    # Synthetic damped sinusoid impulse response: y(k) = e^(-0.1 k) sin(0.5 k)
    dt = 0.1
    k = np.arange(100)
    y = np.exp(-0.1 * k) * np.sin(0.5 * k)
    
    sys_id = HankelSystemID(y, dt=dt)
    sys_id.construct_hankel(r=30, c=30)
    res = sys_id.realize_system(n_states=2)
    
    # Check that continuous poles are close to -0.1 +/- i * 0.5
    cont_poles = res['continuous_poles']
    sigmas = res['sigmas']
    gammas = res['gammas']
    
    # Expected sigma ~ -0.1 / dt = -1.0, gamma ~ 0.5 / dt = 5.0
    assert np.isclose(np.mean(sigmas), -1.0, atol=0.2)
    assert np.isclose(np.max(gammas), 5.0, atol=0.2)

def test_esprit_music_synthetic():
    # Synthetic signal with two known frequencies: f1 = 2.0 Hz, f2 = 5.0 Hz
    dt = 0.05
    t = np.arange(0, 10, dt)
    y = np.sin(2 * np.pi * 2.0 * t) + np.sin(2 * np.pi * 5.0 * t)
    
    estimator = ParametricSpectralEstimator(y, dt=dt)
    estimator.construct_covariance(M=40)
    res = estimator.run_esprit(L=4)  # 2 real sinusoids = 4 complex poles
    
    freqs = res['omegas'] / (2 * np.pi)
    # Check that top frequencies match 2.0 and 5.0 Hz
    assert any(np.isclose(f, 2.0, atol=0.1) for f in freqs)
    assert any(np.isclose(f, 5.0, atol=0.1) for f in freqs)

def test_delay_differential():
    primes = generate_primes(1000)
    analyzer = DelayDifferentialAnalyzer(primes)
    
    a, c, rmse_lti = analyzer.fit_lti_baseline()
    assert isinstance(rmse_lti, float)
    
    dde_res = analyzer.fit_variable_deadtime_dde(max_tau=5, alpha=1.5)
    assert 'dde_rmse' in dde_res
    assert 'rmse_improvement_pct' in dde_res

def test_riemann_zeros():
    zeros = get_riemann_zeros(num_zeros=5)
    # Known first zeros: 14.1347, 21.0220, 25.0109, 30.4249, 32.9351
    assert len(zeros) == 5
    assert np.isclose(zeros[0], 14.1347, atol=0.01)
    assert np.isclose(zeros[1], 21.0220, atol=0.01)

if __name__ == '__main__':
    pytest.main(['-v', __file__])
