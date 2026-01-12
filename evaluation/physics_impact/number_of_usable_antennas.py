import numpy as np
from scipy.signal import hilbert
from matplotlib import pyplot as plt

def _hilbert_envelope(traces: np.ndarray) -> np.ndarray:
    """
    Compute Hilbert-envelope amplitude for a batch of 1D traces.

    Parameters
    ----------
    traces : array, shape (N_traces, N_samples)
        Real-valued time series.

    Returns
    -------
    env : array, shape (N_traces, N_samples)
        Envelope amplitude |Hilbert(traces)|.
    """
    analytic = hilbert(traces, axis=-1)
    env = np.abs(analytic)
    return env


def evaluate_usable_antennas_snr4(
    traces_clean: np.ndarray,
    traces_noisy: np.ndarray,
    traces_denoised: np.ndarray,
    dt_ns: float,
    noise_slice: slice | None = None,
    noise_sigma: np.ndarray | float | None = None,
    snr_threshold: float = 4.0,
    t_tol_ns: float = 10.0,
    snr_bins: np.ndarray | None = None,
):
    """
    Evaluate number of antennas with SNR >= snr_threshold whose trigger time is correctly
    reconstructed, for both the standard GRAND-style filter and the ML denoiser.

    Parameters
    ----------
    traces_clean : array, shape (N, T)
        Noise-free E-field traces (per antenna).
    traces_noisy : array, shape (N, T)
        Noisy traces processed with the standard GRAND-style chain (before ML denoising).
    traces_denoised : array, shape (N, T)
        Output of the ML denoiser, same shape as input traces.
    dt_ns : float
        Sampling interval in nanoseconds.
    noise_slice : slice, optional
        Slice selecting a noise-only region of the trace (e.g. slice(0, 256)).
        Used to estimate noise_sigma from `traces_noisy` if `noise_sigma` is None.
    noise_sigma : float or array, optional
        RMS noise level (in same units as the traces). If None, it is estimated
        from `traces_noisy[..., noise_slice]`.
        If array, it must be broadcastable to shape (N,).
    snr_threshold : float, default 4.0
        Physics SNR threshold (e.g. 4 ~ 4σ) for deciding whether a true signal
        is strong enough to be considered "usable".
    t_tol_ns : float, default 10.0
        Timing tolerance for "correct trigger time", in ns. A trace is counted
        as correctly reconstructed if |Δt| <= t_tol_ns.
    snr_bins : array, optional
        Bin edges for SNR to compute per-bin usable fractions, e.g.
        np.arange(0., 11., 1.). If None, no per-bin stats are returned.

    Returns
    -------
    results : dict
        Dictionary with:
        - 'snr_true' : array, shape (N,)
            True SNR per trace (from clean envelope / noise_sigma).
        - 'delta_t_baseline_ns' : array, shape (N,)
        - 'delta_t_ml_ns' : array, shape (N,)
            Timing residuals w.r.t. clean peak.
        - 'mask_snr' : boolean array, SNR >= snr_threshold.
        - 'usable_baseline' : boolean array, usable antennas for baseline filter.
        - 'usable_ml' : boolean array, usable antennas for ML denoiser.
        - 'summary' : dict with global counts/fractions.
        - 'binned' : dict with per-SNR-bin stats (if snr_bins is not None).
    """
    traces_clean = np.asarray(traces_clean, dtype=np.float64)
    traces_noisy = np.asarray(traces_noisy, dtype=np.float64)
    traces_denoised = np.asarray(traces_denoised, dtype=np.float64)

    if traces_clean.shape != traces_noisy.shape or traces_clean.shape != traces_denoised.shape:
        raise ValueError("clean, noisy, and denoised traces must have the same shape (N, T).")

    n_traces, n_samples = traces_clean.shape

    # ------------------------------------------------------------------
    # 1. Noise RMS per trace
    # ------------------------------------------------------------------
    if noise_sigma is None:
        if noise_slice is None:
            raise ValueError(
                "Either provide `noise_sigma` or a `noise_slice` selecting a noise-only window."
            )
        noise_region = traces_noisy[:, noise_slice]
        noise_sigma = noise_region.std(axis=-1, ddof=0)  # shape (N,)
    else:
        noise_sigma = np.asarray(noise_sigma, dtype=np.float64)
        # Broadcast to (N,)
        if noise_sigma.shape == ():
            noise_sigma = np.full(n_traces, float(noise_sigma))
        elif noise_sigma.shape != (n_traces,):
            try:
                noise_sigma = np.broadcast_to(noise_sigma, (n_traces,))
            except ValueError as e:
                raise ValueError(
                    f"noise_sigma shape {noise_sigma.shape} not compatible with n_traces={n_traces}"
                ) from e

    # ------------------------------------------------------------------
    # 2. Hilbert envelopes and peak amplitudes/times
    # ------------------------------------------------------------------
    env_clean = _hilbert_envelope(traces_clean)
    env_noisy = _hilbert_envelope(traces_noisy)
    env_denoised = _hilbert_envelope(traces_denoised)

    # Peak amplitudes
    amp_clean = env_clean.max(axis=-1)        # shape (N,)
    amp_noisy = env_noisy.max(axis=-1)
    amp_denoised = env_denoised.max(axis=-1)

    # Peak indices and corresponding times
    idx_clean = env_clean.argmax(axis=-1)
    idx_noisy = env_noisy.argmax(axis=-1)
    idx_denoised = env_denoised.argmax(axis=-1)

    t_clean_ns = idx_clean * dt_ns
    t_noisy_ns = idx_noisy * dt_ns
    t_denoised_ns = idx_denoised * dt_ns

    # Timing residuals (baseline and ML w.r.t. clean)
    delta_t_baseline_ns = t_noisy_ns - t_clean_ns
    delta_t_ml_ns = t_denoised_ns - t_clean_ns

    # ------------------------------------------------------------------
    # 3. True SNR and "usable" antennas
    # ------------------------------------------------------------------
    snr_true = amp_clean / noise_sigma  # physics SNR from clean envelope

    mask_snr = snr_true >= snr_threshold
    amp_thresh = snr_threshold * noise_sigma  # amplitude threshold corresponding to SNR>=snr_threshold

    # Calculate standard deviation of clean envelope for each trace
    sigma_clean_env = env_clean.std(axis=-1)  # shape (N,) - std across time samples for each trace
    
    # relative differences between noisy and clean envelope max amplitudes
    rel_tol = 0.1 
    amp_within_tol_noisy = np.abs(amp_noisy - amp_clean) <= rel_tol * amp_clean
    amp_within_tol_denoised = np.abs(amp_denoised - amp_clean) <= rel_tol * amp_clean

    # "Usable" = true SNR high enough, reconstructed amplitude above threshold,
    # and timing within t_tol_ns.
    usable_baseline = (
        mask_snr
        & (amp_noisy >= amp_thresh)
        & (np.abs(delta_t_baseline_ns) <= t_tol_ns)
        & amp_within_tol_noisy
    )
    usable_ml = (
        mask_snr
        & (amp_denoised >= amp_thresh)
        & (np.abs(delta_t_ml_ns) <= t_tol_ns)
        & amp_within_tol_denoised
    )

    # Global counts and fractions
    n_total_snr = int(mask_snr.sum())
    n_use_baseline = int(usable_baseline.sum())
    n_use_ml = int(usable_ml.sum())

    frac_use_baseline = n_use_baseline / n_total_snr if n_total_snr > 0 else np.nan
    frac_use_ml = n_use_ml / n_total_snr if n_total_snr > 0 else np.nan

    summary = {
        "n_traces_total": int(n_traces),
        "n_traces_snr_ge_thresh": n_total_snr,
        "n_usable_baseline": n_use_baseline,
        "n_usable_ml": n_use_ml,
        "frac_usable_baseline": frac_use_baseline,
        "frac_usable_ml": frac_use_ml,
        "snr_threshold": float(snr_threshold),
        "t_tol_ns": float(t_tol_ns),
    }

    results = {
        "snr_true": snr_true,
        "delta_t_baseline_ns": delta_t_baseline_ns,
        "delta_t_ml_ns": delta_t_ml_ns,
        "mask_snr": mask_snr,
        "usable_baseline": usable_baseline,
        "usable_ml": usable_ml,
        "summary": summary,
    }

    # ------------------------------------------------------------------
    # 4. Optional SNR-binned stats for later plotting
    # ------------------------------------------------------------------
    if snr_bins is not None:
        snr_bins = np.asarray(snr_bins, dtype=float)
        if snr_bins.ndim != 1 or snr_bins.size < 2:
            raise ValueError("snr_bins must be a 1D array of bin edges, length >= 2.")

        n_bins = snr_bins.size - 1
        n_in_bin = np.zeros(n_bins, dtype=int)
        n_use_baseline_bin = np.zeros(n_bins, dtype=int)
        n_use_ml_bin = np.zeros(n_bins, dtype=int)

        for i in range(n_bins):
            lo, hi = snr_bins[i], snr_bins[i + 1]
            in_bin = (snr_true >= lo) & (snr_true < hi)
            n_in = int(in_bin.sum())
            n_in_bin[i] = n_in
            if n_in > 0:
                n_use_baseline_bin[i] = int((usable_baseline & in_bin).sum())
                n_use_ml_bin[i] = int((usable_ml & in_bin).sum())

        frac_use_baseline_bin = np.divide(
            n_use_baseline_bin,
            n_in_bin,
            out=np.full_like(n_in_bin, np.nan, dtype=float),
            where=n_in_bin > 0,
        )
        frac_use_ml_bin = np.divide(
            n_use_ml_bin,
            n_in_bin,
            out=np.full_like(n_in_bin, np.nan, dtype=float),
            where=n_in_bin > 0,
        )

        results["binned"] = {
            "snr_bin_edges": snr_bins,
            "n_in_bin": n_in_bin,
            "n_usable_baseline_bin": n_use_baseline_bin,
            "n_usable_ml_bin": n_use_ml_bin,
            "frac_usable_baseline_bin": frac_use_baseline_bin,
            "frac_usable_ml_bin": frac_use_ml_bin,
        }

    return results

def plot_usable_fraction_comparison(results, save_path=None, figname='usable_fraction_comparison.pdf'):
    """
    Plot comparison of usable antenna fractions: ML denoised vs Baseline.
    
    Creates two subplots:
    1. Usable fraction vs SNR for both methods
    2. Improvement (ML - Baseline) vs SNR
    
    Args:
        results: Dictionary returned by evaluate_usable_antennas_snr4
        save_path: Directory to save the figure (optional)
        figname: Filename for the saved figure
    """
    if "binned" not in results:
        raise ValueError("Results must contain 'binned' data. Set snr_bins when calling evaluate_usable_antennas_snr4.")
    
    binned = results["binned"]
    snr_bin_edges = binned["snr_bin_edges"]
    frac_baseline = binned["frac_usable_baseline_bin"]
    frac_ml = binned["frac_usable_ml_bin"]
    n_in_bin = binned["n_in_bin"]
    
    # Calculate bin centers for plotting
    snr_bin_centers = (snr_bin_edges[:-1] + snr_bin_edges[1:]) / 2
    
    # Calculate improvement: ML - Baseline
    improvement = frac_ml - frac_baseline
    
    # Calculate relative improvement: (ML - Baseline) / Baseline * 100
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_improvement = np.where(
            frac_baseline > 0,
            (frac_ml - frac_baseline) / frac_baseline * 100,
            np.nan
        )
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fontsize = 14
    
    # --- Subplot 1: Usable fraction vs SNR ---
    ax1 = axes[0]
    ax1.plot(snr_bin_centers, frac_baseline * 100, 'o-', color='gray', 
             linewidth=2, markersize=8, label='Baseline (Noisy)')
    ax1.plot(snr_bin_centers, frac_ml * 100, 's-', color='red', 
             linewidth=2, markersize=8, label='ML Denoised')
    ax1.fill_between(snr_bin_centers, frac_baseline * 100, frac_ml * 100, 
                     alpha=0.3, color='green', where=(frac_ml >= frac_baseline),
                     label='ML Improvement')
    ax1.fill_between(snr_bin_centers, frac_baseline * 100, frac_ml * 100, 
                     alpha=0.3, color='red', where=(frac_ml < frac_baseline))
    
    ax1.set_xlabel('SNR', fontsize=fontsize)
    ax1.set_ylabel('Usable Fraction [%]', fontsize=fontsize)
    ax1.set_title('Usable Antenna Fraction vs SNR', fontsize=fontsize+2)
    ax1.legend(fontsize=fontsize-2, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis='both', labelsize=fontsize-2)
    
    # --- Subplot 2: Absolute improvement (ML - Baseline) ---
    ax2 = axes[1]
    colors = ['green' if imp >= 0 else 'red' for imp in improvement]
    bars = ax2.bar(snr_bin_centers, improvement * 100, width=0.8, color=colors, 
                   edgecolor='black', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    ax2.set_xlabel('SNR', fontsize=fontsize)
    ax2.set_ylabel('Improvement [%]', fontsize=fontsize)
    ax2.set_title('Absolute Improvement (ML - Baseline)', fontsize=fontsize+2)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='both', labelsize=fontsize-2)
    
    # Add value labels on bars
    for bar, val in zip(bars, improvement * 100):
        if not np.isnan(val):
            height = bar.get_height()
            ax2.annotate(f'{val:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -12),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=fontsize-4)
    
    # --- Subplot 3: Sample count per SNR bin ---
    ax3 = axes[2]
    ax3.bar(snr_bin_centers, n_in_bin, width=0.8, color='steelblue', 
            edgecolor='black', alpha=0.7)
    ax3.set_xlabel('SNR', fontsize=fontsize)
    ax3.set_ylabel('Number of Traces', fontsize=fontsize)
    ax3.set_title('Sample Distribution by SNR', fontsize=fontsize+2)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='both', labelsize=fontsize-2)
    ax3.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, figname)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {filepath}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SNR-BINNED COMPARISON SUMMARY")
    print("="*60)
    print(f"{'SNR Range':<12} {'Baseline %':<12} {'ML %':<12} {'Improvement':<12} {'N Traces':<10}")
    print("-"*60)
    for i in range(len(snr_bin_centers)):
        snr_range = f"[{snr_bin_edges[i]:.0f}, {snr_bin_edges[i+1]:.0f})"
        baseline = f"{frac_baseline[i]*100:.1f}%" if not np.isnan(frac_baseline[i]) else "N/A"
        ml = f"{frac_ml[i]*100:.1f}%" if not np.isnan(frac_ml[i]) else "N/A"
        imp = f"{improvement[i]*100:+.1f}%" if not np.isnan(improvement[i]) else "N/A"
        print(f"{snr_range:<12} {baseline:<12} {ml:<12} {imp:<12} {n_in_bin[i]:<10}")
    print("="*60)
    
    # Overall improvement
    total_baseline = results["summary"]["n_usable_baseline"]
    total_ml = results["summary"]["n_usable_ml"]
    total_snr = results["summary"]["n_traces_snr_ge_thresh"]
    
    print(f"\nOVERALL (SNR >= {results['summary']['snr_threshold']}):")
    print(f"  Baseline usable: {total_baseline} / {total_snr} = {total_baseline/total_snr*100:.1f}%")
    print(f"  ML usable:       {total_ml} / {total_snr} = {total_ml/total_snr*100:.1f}%")
    print(f"  Improvement:     {(total_ml - total_baseline)/total_snr*100:+.1f}%")
    if total_baseline > 0:
        print(f"  Relative gain:   {(total_ml - total_baseline)/total_baseline*100:+.1f}%")


def plot_cumulative_improvement(results, save_path=None, figname='cumulative_improvement.pdf'):
    """
    Plot cumulative usable antenna counts and improvement vs SNR threshold.
    
    Shows how many antennas are usable at SNR >= threshold for different thresholds.
    
    Args:
        results: Dictionary returned by evaluate_usable_antennas_snr4
        save_path: Directory to save the figure (optional)
        figname: Filename for the saved figure
    """
    snr_true = results["snr_true"]
    usable_baseline = results["usable_baseline"]
    usable_ml = results["usable_ml"]
    
    # Create SNR thresholds to evaluate
    snr_thresholds = np.arange(0, 11, 0.5)
    
    # Calculate cumulative counts for each threshold
    n_baseline_cumulative = []
    n_ml_cumulative = []
    n_total_cumulative = []
    
    for thresh in snr_thresholds:
        mask = snr_true >= thresh
        n_total_cumulative.append(mask.sum())
        n_baseline_cumulative.append((usable_baseline & mask).sum())
        n_ml_cumulative.append((usable_ml & mask).sum())
    
    n_baseline_cumulative = np.array(n_baseline_cumulative)
    n_ml_cumulative = np.array(n_ml_cumulative)
    n_total_cumulative = np.array(n_total_cumulative)
    
    # Calculate fractions
    with np.errstate(divide='ignore', invalid='ignore'):
        frac_baseline = np.where(n_total_cumulative > 0, 
                                  n_baseline_cumulative / n_total_cumulative, np.nan)
        frac_ml = np.where(n_total_cumulative > 0, 
                           n_ml_cumulative / n_total_cumulative, np.nan)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fontsize = 14
    
    # --- Subplot 1: Usable fraction vs SNR threshold ---
    ax1 = axes[0]
    ax1.plot(snr_thresholds, frac_baseline * 100, 'o-', color='gray', 
             linewidth=2, markersize=6, label='Baseline')
    ax1.plot(snr_thresholds, frac_ml * 100, 's-', color='red', 
             linewidth=2, markersize=6, label='ML Denoised')
    ax1.fill_between(snr_thresholds, frac_baseline * 100, frac_ml * 100, 
                     alpha=0.3, color='green', where=(frac_ml >= frac_baseline))
    
    ax1.set_xlabel('SNR Threshold (≥)', fontsize=fontsize)
    ax1.set_ylabel('Usable Fraction [%]', fontsize=fontsize)
    ax1.set_title('Cumulative Usable Fraction vs SNR Threshold', fontsize=fontsize+2)
    ax1.legend(fontsize=fontsize-2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis='both', labelsize=fontsize-2)
    
    # --- Subplot 2: Improvement vs SNR threshold ---
    ax2 = axes[1]
    improvement = (frac_ml - frac_baseline) * 100
    ax2.plot(snr_thresholds, improvement, 'o-', color='green', linewidth=2, markersize=6)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.fill_between(snr_thresholds, 0, improvement, alpha=0.3, 
                     color='green', where=(improvement >= 0))
    ax2.fill_between(snr_thresholds, 0, improvement, alpha=0.3, 
                     color='red', where=(improvement < 0))
    
    ax2.set_xlabel('SNR Threshold (≥)', fontsize=fontsize)
    ax2.set_ylabel('Improvement (ML - Baseline) [%]', fontsize=fontsize)
    ax2.set_title('ML Improvement vs SNR Threshold', fontsize=fontsize+2)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=fontsize-2)
    
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, figname)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {filepath}")
    
    plt.show()

# if your arrays have shape (N_antennas, N_samples):
# clean_traces, noisy_traces, denoised_traces
# and you know your sampling interval (e.g. dt_ns = 1.0).

# dt_ns = 1.0
# noise_slice = slice(0, 256)  # example: first 256 samples are noise-only

# results = evaluate_usable_antennas_snr4(
#     traces_clean=clean_traces,
#     traces_noisy=noisy_traces,
#     traces_denoised=denoised_traces,
#     dt_ns=dt_ns,
#     noise_slice=noise_slice,
#     snr_threshold=4.0,
#     t_tol_ns=10.0,
#     snr_bins=np.arange(0., 11., 1.),  # optional, for later plotting
# )

# print(results["summary"])
# -> {'n_traces_total': ..., 'n_traces_snr_ge_thresh': ...,
#     'n_usable_baseline': ..., 'n_usable_ml': ...,
#     'frac_usable_baseline': ..., 'frac_usable_ml': ..., ...}

