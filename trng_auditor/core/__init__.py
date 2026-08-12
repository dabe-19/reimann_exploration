"""
trng_auditor.core: System Identification & Parametric Spectral Analysis of Prime Signals & Riemann Zeros
"""

from .data import generate_primes, von_mangoldt_sequence, chebyshev_psi, pnt_error_term, logarithmic_resample, get_riemann_zeros
from .era_n4sid import HankelSystemID
from .spectral_estimation import ParametricSpectralEstimator
from .delay_differential import DelayDifferentialAnalyzer

__all__ = [
    'generate_primes',
    'von_mangoldt_sequence',
    'chebyshev_psi',
    'pnt_error_term',
    'logarithmic_resample',
    'get_riemann_zeros',
    'HankelSystemID',
    'ParametricSpectralEstimator',
    'DelayDifferentialAnalyzer'
]
