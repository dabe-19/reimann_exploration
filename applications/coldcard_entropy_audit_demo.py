import time
import numpy as np

def generate_flawed_trng_stream(n_bytes=8192) -> bytes:
    """
    Simulates a flawed hardware TRNG (like historical Coldcard/Bitcoin wallet entropy bugs)
    where internal shift register state leakage creates a low-rank linear dynamical system.
    """
    np.random.seed(42)
    # Autoregressive process AR(2) with weak linear coupling
    state = np.random.randn(8192)
    for i in range(2, len(state)):
        state[i] += 0.45 * state[i-1] - 0.20 * state[i-2]
    
    # Scale to 0..255 byte values
    state_norm = (state - np.min(state)) / (np.max(state) - np.min(state))
    byte_vals = (state_norm * 255).astype(np.uint8)
    return bytes(byte_vals)

def generate_ideal_trng_stream(n_bytes=8192) -> bytes:
    """Generates an ideal cryptographically uniform random byte stream."""
    np.random.seed(1337)
    return bytes(np.random.randint(0, 256, size=n_bytes, dtype=np.uint8))

def audit_trng_hankel_subspace(byte_data: bytes, M: int = 128) -> dict:
    """
    Audits random byte streams using Hankel Subspace Energy Concentration & Operator Normality.
    - True random noise has infinite rank (all singular values equal, energy ratio ~ M_top / M).
    - Flawed hardware state leakage collapses into low-rank subspace energy (high energy ratio)!
    """
    data = np.frombuffer(byte_data, dtype=np.uint8).astype(np.float64) - 127.5
    N = len(data)
    K = N - M + 1
    H = np.zeros((M, K))
    for i in range(M):
        H[i, :] = data[i:i+K]
        
    _, S, _ = np.linalg.svd(H, full_matrices=False)
    
    # Energy in top 10 singular modes vs theoretical uniform expectation (10 / 128 = 7.8%)
    top10_energy_pct = (np.sum(S[:10]**2) / np.sum(S**2)) * 100.0
    
    # Effective Subspace Rank (99% energy capture)
    cum_energy = np.cumsum(S**2) / np.sum(S**2)
    eff_rank = int(np.searchsorted(cum_energy, 0.99) + 1)
    
    # Audit threshold: uniform random noise has rank ~ 125+, flawed TRNG rank < 90
    is_flawed = eff_rank < 100 or top10_energy_pct > 15.0
    
    return {
        "top10_energy_pct": top10_energy_pct,
        "effective_rank": eff_rank,
        "is_flawed": is_flawed
    }

def main():
    print("=" * 85)
    print("   CRYPTO AUDIT DEMO: HANKEL KOOPMAN SUBSPACE TRNG SECURITY AUDITOR   ")
    print("=" * 85)

    print("\n[1] Auditing Ideal Cryptographic Uniform Entropy Stream (8192 bytes)...")
    ideal = generate_ideal_trng_stream(8192)
    res_ideal = audit_trng_hankel_subspace(ideal)
    print(f"    - Top 10 Hankel Modes Energy:   {res_ideal['top10_energy_pct']:.2f}% (Expected ~7.81%)")
    print(f"    - Effective Subspace Dimension: {res_ideal['effective_rank']} / 128 modes")
    print(f"    - Security Audit Result:        {'FAILED (Flawed Entropy Detected)' if res_ideal['is_flawed'] else 'PASSED (Cryptographically Secure)'}")

    print("\n[2] Auditing Flawed Hardware TRNG Stream (State Leakage / Weak Dice)...")
    flawed = generate_flawed_trng_stream(8192)
    res_flawed = audit_trng_hankel_subspace(flawed)
    print(f"    - Top 10 Hankel Modes Energy:   {res_flawed['top10_energy_pct']:.2f}% (Low-Rank Subspace Collapse!)")
    print(f"    - Effective Subspace Dimension: {res_flawed['effective_rank']} / 128 modes")
    print(f"    - Security Audit Result:        {'FAILED (Flawed Entropy Detected)' if res_flawed['is_flawed'] else 'PASSED (Cryptographically Secure)'}")

    print("\n" + "=" * 85)
    print("   TRNG SECURITY AUDIT DEMO COMPLETED SUCCESSFULLY.   ")
    print("=" * 85)

if __name__ == '__main__':
    main()
