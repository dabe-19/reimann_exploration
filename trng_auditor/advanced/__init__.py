"""
trng_auditor.advanced: Advanced Operator Theory, Kernelized Subspace ID, Koopman EDMD, and Quantum System Identification
"""

from .kernel_era import KernelHankelSystemID
from .koopman_edmd import KoopmanAdeleEDMD
from .quantum_sysid import QuantumSystemIdentification
from .theoretical_proofs import TheoreticalProofsVerifier

__all__ = [
    'KernelHankelSystemID',
    'KoopmanAdeleEDMD',
    'QuantumSystemIdentification',
    'TheoreticalProofsVerifier'
]
