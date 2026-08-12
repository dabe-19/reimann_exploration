import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from riemann_sysid import (
    generate_primes,
    von_mangoldt_sequence,
    chebyshev_psi,
    pnt_error_term,
    logarithmic_resample,
    get_riemann_zeros,
    HankelSystemID,
    ParametricSpectralEstimator,
    DelayDifferentialAnalyzer
)

# Set high-quality plot aesthetic
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

def main():
    os.makedirs("plots", exist_ok=True)
    print("=" * 80)
    print("      RIEMANN ZEROS & PRIME SIGNAL SYSTEM IDENTIFICATION EXPERIMENTS      ")
    print("=" * 80)

    # ----------------------------------------------------
    # 1. DATA GENERATION & PREPARATION
    # ----------------------------------------------------
    N = 40000
    print(f"\n[1] Generating prime data and von Mangoldt sequence up to N = {N}...")
    t0 = time.time()
    primes = generate_primes(N)
    vm_seq = von_mangoldt_sequence(N)
    x_grid, delta_pnt = pnt_error_term(N, normalized=False)
    riemann_ref_zeros = get_riemann_zeros(num_zeros=50)
    print(f"    Done in {time.time() - t0:.2f}s. Generated {len(primes)} primes.")
    print(f"    First 5 Riemann Zeros (gamma_k): {riemann_ref_zeros[:5]}")

    # Plot 1: PNT Error Signal & Logarithmic Resampling
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6))
    ax1.plot(x_grid, delta_pnt, color='#1f77b4', linewidth=0.8)
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title(r"Chebyshev Error Term $\Delta(x) = \psi(x) - x$")
    ax1.set_ylabel(r"$\Delta(x)$")
    ax1.grid(True, linestyle='--', alpha=0.5)

    t_uniform, y_uniform, dt_log = logarithmic_resample(x_grid, delta_pnt, num_samples=2000)
    ax2.plot(t_uniform, y_uniform, color='#ff7f0e', linewidth=1.0)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title(r"Logarithmic Resampled Snapshot $y(t) = \psi(e^t) - e^t$ ($t = \ln x$)")
    ax2.set_xlabel(r"Logarithmic Time $t = \ln x$")
    ax2.set_ylabel(r"$y(t)$")
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("plots/pnt_error_signal.png", dpi=300)
    plt.close()
    print("    Saved plot: plots/pnt_error_signal.png")

    # ----------------------------------------------------
    # 2. SUBSPACE SYSTEM IDENTIFICATION (N4SID / ERA)
    # ----------------------------------------------------
    print("\n[2] Executing Subspace System Identification (N4SID / ERA)...")
    sys_id = HankelSystemID(vm_seq, dt=1.0)
    r_dim, c_dim = 1000, 1000
    print(f"    Constructing Hankel matrix of size {r_dim} x {c_dim}...")
    sys_id.construct_hankel(r=r_dim, c=c_dim)
    
    print("    Computing Hankel Singular Value Decomposition (SVD)...")
    singular_vals = sys_id.compute_svd()
    
    # Plot 2: Hankel Singular Values
    plt.figure(figsize=(9, 5))
    plt.semilogy(singular_vals[:120], 'o-', color='#1f77b4', linewidth=2, markersize=4)
    plt.title(r"Hankel Matrix Singular Value Decay (Prime Impulse Sequence $\Lambda(n)$)")
    plt.xlabel(r"Singular Value Index $i$")
    plt.ylabel(r"Singular Value $\sigma_i$")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/hankel_svd_spectrum.png", dpi=300)
    plt.close()
    print("    Saved plot: plots/hankel_svd_spectrum.png")

    # Realize continuous state-space operator for n_states = 80
    n_states = 80
    era_results = sys_id.realize_system(n_states=n_states)
    
    print(f"\n    === ERA Realization Properties (n_states = {n_states}) ===")
    print(f"    Normality Metric ||A A^H - A^H A||_F / ||A||_F^2: {era_results['normality_metric']:.6e}")
    print(f"    Hermitian Diff ||A - A^H||_F / ||A||_F:            {era_results['hermitian_diff']:.6f}")
    print(f"    Skew-Hermitian Diff ||A + A^H||_F / ||A||_F:       {era_results['skew_hermitian_diff']:.6f}")
    print(f"    Mean Real Damping (Sigma):                         {np.mean(era_results['sigmas']):.4f}")
    print(f"    Max Extracted Frequency (Gamma):                   {np.max(era_results['gammas']):.4f}")

    # Plot 3: ERA State Space Poles vs True Zeros
    plt.figure(figsize=(10, 6))
    plt.scatter(era_results['sigmas'], era_results['gammas'], color='#d62728', alpha=0.7, s=50, label=f'ERA Poles ($n={n_states}$)')
    for idx, gz in enumerate(riemann_ref_zeros[:20]):
        label_text = r'True Riemann Zeros $\gamma_k$' if idx == 0 else ""
        plt.axhline(y=gz, color='#2ca02c', linestyle='--', alpha=0.5, label=label_text)
        
    plt.title(fr"Empirical State-Space $A$ Matrix Spectrum (ERA Realization, $N={N}$)")
    plt.xlabel(r"Real Damping Rate $\sigma = \text{Re}(s)$")
    plt.ylabel(r"Imaginary Frequency $\gamma = |\text{Im}(s)|$")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/era_state_space_poles.png", dpi=300)
    plt.close()
    print("    Saved plot: plots/era_state_space_poles.png")

    # Plot 4: Operator Matrix & Commutator Heatmap
    A_mat = era_results['A_discrete']
    A_adj = A_mat.conj().T
    comm_mat = np.abs(A_mat @ A_adj - A_adj @ A_mat)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(np.abs(A_mat), cmap='viridis', norm=LogNorm(vmin=1e-4, vmax=np.max(np.abs(A_mat))))
    ax1.set_title(r"State Operator Magnitude $|A_{ij}|$")
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    im2 = ax2.imshow(comm_mat, cmap='plasma', norm=LogNorm(vmin=1e-6, vmax=np.max(comm_mat)))
    ax2.set_title(r"Commutator Magnitude $|[A, A^\dagger]_{ij}|$")
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    plt.tight_layout()
    plt.savefig("plots/operator_hermiticity_matrix.png", dpi=300)
    plt.close()
    print("    Saved plot: plots/operator_hermiticity_matrix.png")

    # ----------------------------------------------------
    # 3. PARAMETRIC SPECTRAL ESTIMATION (ESPRIT & MUSIC)
    # ----------------------------------------------------
    print("\n[3] Executing Parametric High-Resolution Spectral Estimation (ESPRIT & MUSIC)...")
    estimator = ParametricSpectralEstimator(y_uniform, dt=dt_log)
    M_window = 150
    print(f"    Constructing snapshot covariance matrix Rxx with window M = {M_window}...")
    estimator.construct_covariance(M=M_window)
    
    L_signals = 30
    esprit_results = estimator.run_esprit(L=L_signals)
    print(f"\n    === TLS-ESPRIT Pole Extraction Results (L = {L_signals}) ===")
    print(f"    Mean Real Pole Damping Re(s): {esprit_results['mean_sigma']:.4f} (std = {esprit_results['std_sigma']:.4f})")
    
    print("    Top 10 ESPRIT Extracted Frequencies (gamma_k):")
    top_omegas = esprit_results['omegas'][:10]
    for idx, om in enumerate(top_omegas):
        closest_ref = riemann_ref_zeros[np.argmin(np.abs(riemann_ref_zeros - om))]
        err = np.abs(om - closest_ref)
        print(f"      Poles #{idx+1:02d}: omega = {om:8.4f} rad/s  | Closest Riemann Zero = {closest_ref:8.4f} (err = {err:.4f})")

    # Plot 5: ESPRIT Complex Pole Constellation
    plt.figure(figsize=(9, 6))
    plt.scatter(esprit_results['sigmas'], esprit_results['omegas'], color='#9467bd', s=60, edgecolors='black', label='ESPRIT Extracted Poles $s_k$')
    for idx, gz in enumerate(riemann_ref_zeros[:15]):
        lbl = r'Riemann Zeros $\gamma_k$' if idx == 0 else ""
        plt.axhline(y=gz, color='#2ca02c', linestyle='--', alpha=0.5, label=lbl)
    plt.title(r"TLS-ESPRIT Extracted Complex Poles $s_k = \sigma_k + i \omega_k$")
    plt.xlabel(r"Real Damping Parameter $\sigma = \text{Re}(s)$")
    plt.ylabel(r"Angular Frequency $\omega = \text{Im}(s)$ (rad/s)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/esprit_pole_constellation.png", dpi=300)
    plt.close()
    print("    Saved plot: plots/esprit_pole_constellation.png")

    # Plot 6: MUSIC Pseudospectrum vs FFT Spectrum
    omega_scan = np.linspace(10, 70, 1200)
    _, music_spectrum = estimator.run_music(omega_scan)
    
    fft_vals = np.abs(np.fft.rfft(y_uniform))
    fft_freqs = np.fft.rfftfreq(len(y_uniform), d=dt_log) * 2 * np.pi

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(omega_scan, 10 * np.log10(music_spectrum / np.max(music_spectrum)), color='#9467bd', linewidth=2, label='MUSIC Pseudospectrum (dB)')
    for idx, gz in enumerate(riemann_ref_zeros):
        if 10 <= gz <= 70:
            lbl = r'Riemann Zeros $\gamma_k$' if idx == 0 else ""
            ax1.axvline(x=gz, color='#2ca02c', linestyle='--', alpha=0.6, label=lbl)
    ax1.set_title("Super-Resolution Parametric Spectral Estimation (MUSIC vs Riemann Zeros)")
    ax1.set_ylabel("Power (dB)")
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)

    mask_fft = (fft_freqs >= 10) & (fft_freqs <= 70)
    ax2.plot(fft_freqs[mask_fft], 10 * np.log10(fft_vals[mask_fft] / np.max(fft_vals[mask_fft])), color='#ff7f0e', linewidth=1.5, label='Standard FFT Spectrum (dB)')
    for idx, gz in enumerate(riemann_ref_zeros):
        if 10 <= gz <= 70:
            ax2.axvline(x=gz, color='#2ca02c', linestyle='--', alpha=0.6)
    ax2.set_title("Standard Fourier Transform (FFT) Spectrum (Exhibiting Leakage & Window Limits)")
    ax2.set_xlabel(r"Angular Frequency $\omega$ (rad/s)")
    ax2.set_ylabel("Power (dB)")
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("plots/esprit_music_resolution.png", dpi=300)
    plt.close()
    print("    Saved plot: plots/esprit_music_resolution.png")

    # ----------------------------------------------------
    # 4. VARIABLE DEAD-TIME DELAY-DIFFERENTIAL MODELING
    # ----------------------------------------------------
    print("\n[4] Executing Variable Dead-Time & Delay-Differential System Identification...")
    dde_analyzer = DelayDifferentialAnalyzer(primes)
    lti_a, lti_c, lti_rmse = dde_analyzer.fit_lti_baseline()
    dde_res = dde_analyzer.fit_variable_deadtime_dde(max_tau=8, alpha=2.0)
    
    print("\n    === Delay-Differential Equation (DDE) Model Comparison ===")
    print(f"    LTI Baseline Model AR(1) RMSE:            {lti_rmse:.6f}")
    print(f"    Variable Dead-Time DDE Model RMSE:        {dde_res['dde_rmse']:.6f}")
    print(f"    RMSE Prediction Reduction:                {dde_res['rmse_improvement_pct']:+.2f}%")
    print(f"    Identified Coefficients:                  A0 = {dde_res['A0']:.4f}, A1 = {dde_res['A1']:.4f}, c = {dde_res['c']:.4f}")
    print(f"    Mean Dynamic Delay tau(x):                {dde_res['mean_tau']:.2f} steps")

    # Plot 7: 3D Phase-Space Embedding
    emb = dde_analyzer.phase_space_reconstruction(tau=2)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=emb[:, 0], cmap='plasma', alpha=0.4, s=10)
    ax.set_title(r"3D Delay Attractor Embedding of Normalized Prime Gaps $d_n$")
    ax.set_xlabel(r"$d_n$")
    ax.set_ylabel(r"$d_{n-\tau}$")
    ax.set_zlabel(r"$d_{n-2\tau}$")
    plt.tight_layout()
    plt.savefig("plots/delay_deadtime_phase.png", dpi=300)
    plt.close()
    print("    Saved plot: plots/delay_deadtime_phase.png")

    print("\n" + "=" * 80)
    print("      EXPERIMENTS COMPLETED SUCCESSFULLY. ALL RESULTS & PLOTS GENERATED.      ")
    print("=" * 80)

if __name__ == '__main__':
    main()
