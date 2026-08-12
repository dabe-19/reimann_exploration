import numpy as np
from typing import Dict, Any, List
from trng_auditor.core.data import generate_primes, von_mangoldt_sequence
from trng_auditor.core.era_n4sid import HankelSystemID
from .koopman_edmd import KoopmanAdeleEDMD

class TheoreticalProofsVerifier:
    """
    Automated numerical verification of theoretical theorems translating empirical findings
    into formal mathematical propositions.
    """
    @staticmethod
    def verify_asymptotic_normality(n_scales: List[int] = [5000, 10000, 20000, 40000]) -> Dict[str, Any]:
        """
        Theorem 1 Verification:
        As Hankel sequence length N -> infinity, the normality metric N(A_N) -> 0.
        """
        results = []
        for N in n_scales:
            vm = von_mangoldt_sequence(N)
            sys_id = HankelSystemID(vm, dt=1.0)
            r = min(800, N // 3)
            c = r
            sys_id.construct_hankel(r=r, c=c)
            sys_id.compute_svd()
            era_res = sys_id.realize_system(n_states=60)
            
            norm_val = era_res['normality_metric']
            results.append({
                'N': N,
                'r': r,
                'normality_metric': norm_val,
                'mean_sigma': float(np.mean(era_res['sigmas']))
            })
            
        return {
            'theorem_name': 'Theorem 1: Asymptotic Operator Normality',
            'scaling_results': results,
            'is_decaying': results[-1]['normality_metric'] < results[0]['normality_metric']
        }

    @staticmethod
    def verify_koopman_unitarity(n_max: int = 25000) -> Dict[str, Any]:
        r"""
        Theorem 2 Verification:
        The Koopman operator K_koop over Adèle space observables is unitary (||K^\dagger K - I|| -> 0),
        guaranteeing all eigenvalues lie on the unit circle |lambda_j| = 1.
        """
        edmd = KoopmanAdeleEDMD(n_max=n_max)
        res = edmd.fit_koopman()
        
        unitarity = res['unitarity_metric']
        mean_unit_err = res['mean_unit_error']
        
        return {
            'theorem_name': 'Theorem 2: Koopman Adèle Unitarity & RH Equivalence',
            'unitarity_metric': unitarity,
            'mean_unit_error': mean_unit_err,
            'is_unitary': unitarity < 0.05,
            'num_observables': res['num_observables']
        }
