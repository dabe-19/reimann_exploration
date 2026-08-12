import numpy as np
from typing import Dict, Any, Tuple

class DelayDifferentialAnalyzer:
    """
    Variable Dead-Time & State-Dependent Delay-Differential Modeling of Prime Gap Signals.
    """
    def __init__(self, primes: np.ndarray):
        self.primes = np.asarray(primes, dtype=np.float64)
        if len(self.primes) < 10:
            raise ValueError("Need at least 10 prime numbers for delay-differential modeling.")
            
        self.raw_gaps = np.diff(self.primes)
        # Normalized gaps d_n = (p_{n+1} - p_n) / ln(p_n)
        self.norm_gaps = self.raw_gaps / np.log(self.primes[:-1])

    def fit_lti_baseline(self) -> Tuple[float, float, float]:
        """
        Fit baseline AR(1) LTI model: x_{n+1} = a * x_n + c + eps.
        Returns: (a, c, rmse)
        """
        x_curr = self.norm_gaps[:-1]
        x_next = self.norm_gaps[1:]
        
        # Design matrix X = [x_curr, 1]
        X = np.vstack([x_curr, np.ones_like(x_curr)]).T
        params, residuals, rank, s = np.linalg.lstsq(X, x_next, rcond=None)
        
        a, c = params[0], params[1]
        pred = X @ params
        rmse = float(np.sqrt(np.mean((x_next - pred)**2)))
        return float(a), float(c), rmse

    def fit_variable_deadtime_dde(self, max_tau: int = 10, alpha: float = 2.0) -> Dict[str, Any]:
        """
        Fit variable dead-time model: x_{n+1} = A0 * x_n + A1 * x_{n - tau(x_n)} + c + eps,
        where tau(x_n) = clip(floor(alpha * x_n), 1, max_tau).
        """
        x = self.norm_gaps
        N = len(x)
        
        # Determine variable delay tau_n for each point n
        # tau_n in [1, max_tau]
        tau_seq = np.clip(np.floor(alpha * x).astype(int), 1, max_tau)
        
        start_idx = max_tau
        end_idx = N - 1
        
        y_vec = x[start_idx + 1 : end_idx + 1]
        x_curr = x[start_idx : end_idx]
        
        x_delayed = np.zeros(len(y_vec), dtype=np.float64)
        for i, idx in enumerate(range(start_idx, end_idx)):
            t_delay = tau_seq[idx]
            x_delayed[i] = x[idx - t_delay]
            
        # Design matrix X = [x_curr, x_delayed, 1]
        X = np.vstack([x_curr, x_delayed, np.ones_like(x_curr)]).T
        params, residuals, rank, s = np.linalg.lstsq(X, y_vec, rcond=None)
        
        A0, A1, c = params[0], params[1], params[2]
        pred = X @ params
        dde_rmse = float(np.sqrt(np.mean((y_vec - pred)**2)))
        
        # Compare with baseline LTI on the exact same sample range
        X_lti = np.vstack([x_curr, np.ones_like(x_curr)]).T
        params_lti, _, _, _ = np.linalg.lstsq(X_lti, y_vec, rcond=None)
        pred_lti = X_lti @ params_lti
        lti_rmse = float(np.sqrt(np.mean((y_vec - pred_lti)**2)))
        
        rmse_improvement_pct = float((lti_rmse - dde_rmse) / lti_rmse * 100.0)

        return {
            'A0': float(A0),
            'A1': float(A1),
            'c': float(c),
            'dde_rmse': dde_rmse,
            'lti_rmse': lti_rmse,
            'rmse_improvement_pct': rmse_improvement_pct,
            'mean_tau': float(np.mean(tau_seq[start_idx:end_idx])),
            'max_tau': max_tau
        }

    def phase_space_reconstruction(self, tau: int = 1) -> np.ndarray:
        """
        Create 3D phase-space embedding (x_n, x_{n-tau}, x_{n-2*tau}).
        """
        x = self.norm_gaps
        N = len(x)
        if N <= 2 * tau:
            raise ValueError(f"Signal length N={N} too small for lag tau={tau}.")
            
        emb = np.vstack([
            x[2 * tau : N],
            x[tau : N - tau],
            x[0 : N - 2 * tau]
        ]).T
        return emb
