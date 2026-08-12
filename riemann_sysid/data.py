import numpy as np
import math
import mpmath
from typing import Tuple, List

def generate_primes(n: int) -> np.ndarray:
    """Generate all prime numbers up to n using the Sieve of Eratosthenes."""
    if n < 2:
        return np.array([], dtype=int)
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.flatnonzero(sieve)

def von_mangoldt_sequence(n: int) -> np.ndarray:
    """
    Generate von Mangoldt sequence Lambda(k) for k = 1..n.
    Lambda(k) = ln(p) if k = p^m for prime p and integer m >= 1; 0 otherwise.
    """
    lambda_seq = np.zeros(n + 1, dtype=np.float64)
    primes = generate_primes(n)
    
    for p in primes:
        log_p = math.log(p)
        pk = p
        while pk <= n:
            lambda_seq[pk] = log_p
            pk *= p
            
    return lambda_seq[1:]  # 1-indexed: index 0 corresponds to k=1

def chebyshev_psi(n: int) -> np.ndarray:
    """
    Compute the Chebyshev psi function psi(x) = sum_{k <= x} Lambda(k) for x = 1..n.
    """
    lambda_seq = von_mangoldt_sequence(n)
    return np.cumsum(lambda_seq)

def pnt_error_term(n: int, normalized: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Prime Number Theorem error term Delta(x) = psi(x) - x for x = 1..n.
    If normalized is True, return Delta(x) / sqrt(x).
    """
    x = np.arange(1, n + 1, dtype=np.float64)
    psi = chebyshev_psi(n)
    delta = psi - x
    if normalized:
        delta = delta / np.sqrt(x)
    return x, delta

def logarithmic_resample(x: np.ndarray, y: np.ndarray, num_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Resample (x, y) uniformly in t = ln(x) for snapshot spectral estimation.
    Returns: t_uniform, y_uniform, dt
    """
    valid_mask = x > 1.0
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    t_valid = np.log(x_valid)
    t_uniform = np.linspace(t_valid[0], t_valid[-1], num_samples)
    dt = float(t_uniform[1] - t_uniform[0])
    
    y_uniform = np.interp(t_uniform, t_valid, y_valid)
    return t_uniform, y_uniform, dt

def get_riemann_zeros(num_zeros: int = 100) -> np.ndarray:
    """
    Retrieve the first `num_zeros` non-trivial imaginary parts gamma_k of Riemann zeta zeros.
    0.5 + i * gamma_k
    """
    mpmath.mp.dps = 25
    zeros = []
    for k in range(1, num_zeros + 1):
        z = mpmath.zetazero(k)
        zeros.append(float(z.imag))
    return np.array(zeros, dtype=np.float64)
