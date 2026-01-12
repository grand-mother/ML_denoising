#!/usr/bin/env python3
"""
Appendix Check 1: Standard-method ROI-NMSE vs SNR (per channel)

This implements the same lessons learned from prior scripts:

  1) SNR axis = "paper-style" but using ROI-peak amplitude anchored on CLEAN peak time:
       A_noisy = max(|hilbert(noisy)|) within +/- snr_peak_half_width_ns around CLEAN peak
       sigma_env = robust MAD scale of noisy envelope excluding +/- snr_exclude_half_width_ns around CLEAN peak
       SNR = A_noisy / sigma_env

  2) NMSE = ROI-NMSE in time domain (optionally bandpassed to 50-200 MHz),
     with ROI centered on CLEAN peak time (default +/- 150 ns):
       NMSE = ||rec-clean||^2 / ||clean||^2  (within ROI, per channel)

  3) Truth-conditioned clean-power gate (per channel) to avoid ill-conditioned NMSE
     when true signal energy is tiny in that polarization.

Produces a 2x3 figure:
  Row 1: Scatter of ROI-NMSE vs SNR for the STANDARD method (per channel),
         with rolling median and q16..q84 band (population spread).
  Row 2: High-SNR (>= high_snr_min) distribution summary via boxplot of log10(NMSE).

Optional: overlay ML denoiser points/median for context.

Inputs:
  clean_waveforms:    (N, 3, T)
  noisy_waveforms:    (N, 3, T)
  standard_waveforms: (N, 3, T)
  ml_waveforms:       (N, 3, T) optional
  extra_base_mask:    (N, 3) optional boolean mask (e.g., your timing or antenna-quality pre-cuts)

Dependencies: numpy, matplotlib, scipy
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, sosfiltfilt

# ML mode support
try:
    from common_ml_utils import (
        add_model_arguments, check_model_args, 
        load_data_and_run_inference
    )
    HAS_ML_UTILS = True
except ImportError:
    HAS_ML_UTILS = False


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class Check1Config:
    # sampling
    dt_ns: float = 2.0
    eps: float = 1e-12

    # ROI around CLEAN peak for NMSE
    roi_half_width_ns: float = 150.0

    # SNR (ROI-peak around CLEAN peak time)
    snr_peak_half_width_ns: float = 150.0
    snr_exclude_half_width_ns: float = 150.0
    mad_scale_gaussian: float = 1.4826022185056

    # bandpass (in-band fidelity)
    apply_bandpass: bool = True
    f_lo_hz: float = 50e6
    f_hi_hz: float = 200e6
    butter_order: int = 4

    # apply bandpass to the SNR definition (True if your paper SNR is in-band)
    apply_bandpass_to_snr: bool = True

    # optional trigger-like base selection
    apply_trigger: bool = False
    trigger_k_sigma: float = 1.0

    # truth-conditioned clean-power gate (per channel)
    apply_clean_power_gate: bool = True
    clean_power_gate_quantile: float = 0.10
    clean_power_gate_min_count: int = 200
    clean_power_gate_abs_floor: float = 0.0

    # rolling summary for row-1
    rolling_window: int = 2000
    rolling_grid: int = 60
    rolling_min_points: int = 150
    band_quantiles: Tuple[float, float] = (0.16, 0.84)

    # high-SNR definition for row-2
    high_snr_min: float = 8.0

    # plot cosmetics
    channel_names: Tuple[str, str, str] = ("X", "Y", "Z")
    scatter_alpha: float = 0.10
    scatter_s: float = 6.0
    show_ml_overlay: bool = False  # set True to overlay ML

    xlim: Tuple[float, float] = (1.5, 15.0)

    savepath: Optional[str] = None
    dpi: int = 220


# -----------------------------
# Helpers
# -----------------------------
def run_in_notebook(
    clean: np.ndarray,
    noisy: np.ndarray,
    standard: np.ndarray,
    ml: Optional[np.ndarray] = None,
    savepath: Optional[str] = None,
    show_ml_overlay: bool = False,
    **config_overrides,
) -> dict:
    """
    Convenience wrapper for Jupyter notebook usage.
    
    Args:
        clean: Clean waveforms (N, 3, T)
        noisy: Noisy waveforms (N, 3, T)
        standard: Standard denoised waveforms (N, 3, T)
        ml: ML denoised waveforms (N, 3, T), optional
        savepath: Optional path to save figure
        show_ml_overlay: Whether to show ML overlay on plot
        **config_overrides: Override any Check1Config field
        
    Returns:
        Dictionary with diagnostic arrays
    """
    cfg_kwargs = {
        "savepath": savepath,
        "show_ml_overlay": show_ml_overlay,
    }
    cfg_kwargs.update(config_overrides)
    cfg = Check1Config(**cfg_kwargs)
    
    return plot_check1_nmse_std_vs_snr(
        clean_waveforms=clean,
        noisy_waveforms=noisy,
        standard_waveforms=standard,
        ml_waveforms=ml,
        extra_base_mask=None,
        cfg=cfg,
    )

def _validate_shapes(clean: np.ndarray, noisy: np.ndarray, std: np.ndarray, ml: Optional[np.ndarray]) -> None:
    for name, arr in [("clean", clean), ("noisy", noisy), ("standard", std)]:
        if not isinstance(arr, np.ndarray) or arr.ndim != 3:
            raise ValueError(f"{name} must have shape (N,3,T). Got {None if arr is None else arr.shape}")
    if clean.shape != noisy.shape or clean.shape != std.shape:
        raise ValueError(f"clean/noisy/standard shapes must match. Got {clean.shape}, {noisy.shape}, {std.shape}")
    if clean.shape[1] != 3:
        raise ValueError(f"Expected 3 channels, got {clean.shape[1]}")
    if ml is not None and (not isinstance(ml, np.ndarray) or ml.shape != clean.shape):
        raise ValueError(f"ml_waveforms must be None or match clean shape. Got {None if ml is None else ml.shape}")


def _analytic_envelope(x: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(x, axis=-1))


def _mad_sigma(x: np.ndarray, scale: float) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(scale * mad)


def _maybe_bandpass(x: np.ndarray, cfg: Check1Config) -> np.ndarray:
    if not cfg.apply_bandpass:
        return x
    dt_s = cfg.dt_ns * 1e-9
    fs = 1.0 / dt_s
    nyq = 0.5 * fs
    if cfg.f_hi_hz >= nyq:
        raise ValueError(
            f"Bandpass upper cutoff {cfg.f_hi_hz/1e6:.1f} MHz >= Nyquist {nyq/1e6:.1f} MHz. "
            f"Check dt_ns={cfg.dt_ns}."
        )
    sos = butter(
        cfg.butter_order,
        [cfg.f_lo_hz, cfg.f_hi_hz],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    return sosfiltfilt(sos, x, axis=-1)


def _roi_centered_on_clean_peak_indices(clean_1d: np.ndarray, dt_ns: float, half_width_ns: float) -> Tuple[int, int, int]:
    T = clean_1d.size
    k = int(np.argmax(np.abs(hilbert(clean_1d))))
    half = max(1, int(round(half_width_ns / dt_ns)))
    lo = max(0, k - half)
    hi = min(T, k + half + 1)
    return k, lo, hi


def compute_snr_roi_peak(clean: np.ndarray, noisy: np.ndarray, cfg: Check1Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SNR per trace+channel, anchored on CLEAN peak time.
    Returns snr, A_noisy, sigma_env (all shape (N,3)).
    """
    N, C, T = noisy.shape
    snr = np.full((N, C), np.nan, dtype=np.float64)
    A = np.full((N, C), np.nan, dtype=np.float64)
    sigma = np.full((N, C), np.nan, dtype=np.float64)

    if cfg.apply_bandpass_to_snr:
        clean_s = _maybe_bandpass(clean.astype(np.float64), cfg)
        noisy_s = _maybe_bandpass(noisy.astype(np.float64), cfg)
    else:
        clean_s = clean.astype(np.float64)
        noisy_s = noisy.astype(np.float64)

    peak_hw = max(1, int(round(cfg.snr_peak_half_width_ns / cfg.dt_ns)))
    excl_hw = max(1, int(round(cfg.snr_exclude_half_width_ns / cfg.dt_ns)))

    for ch in range(C):
        env_noisy = _analytic_envelope(noisy_s[:, ch, :])
        for i in range(N):
            k, _, _ = _roi_centered_on_clean_peak_indices(clean_s[i, ch, :], cfg.dt_ns, cfg.roi_half_width_ns)

            lo_pk = max(0, k - peak_hw)
            hi_pk = min(T, k + peak_hw + 1)
            A[i, ch] = float(np.max(env_noisy[i, lo_pk:hi_pk]))

            lo_ex = max(0, k - excl_hw)
            hi_ex = min(T, k + excl_hw + 1)
            mask = np.ones(T, dtype=bool)
            mask[lo_ex:hi_ex] = False
            env_noise = env_noisy[i, mask]
            if env_noise.size < 16:
                env_noise = env_noisy[i, :]

            s = max(_mad_sigma(env_noise, cfg.mad_scale_gaussian), cfg.eps)
            sigma[i, ch] = s
            snr[i, ch] = A[i, ch] / s

    return snr, A, sigma


def compute_roi_nmse_and_cleanpower(clean: np.ndarray, rec: np.ndarray, cfg: Check1Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    ROI-NMSE and CLEAN ROI power, per trace+channel, ROI centered on CLEAN peak.
    """
    N, C, _ = clean.shape
    clean_f = _maybe_bandpass(clean.astype(np.float64), cfg)
    rec_f = _maybe_bandpass(rec.astype(np.float64), cfg)

    nmse = np.full((N, C), np.nan, dtype=np.float64)
    cleanpow = np.full((N, C), np.nan, dtype=np.float64)

    for ch in range(C):
        for i in range(N):
            _, lo, hi = _roi_centered_on_clean_peak_indices(clean_f[i, ch, :], cfg.dt_ns, cfg.roi_half_width_ns)
            x = clean_f[i, ch, lo:hi]
            y = rec_f[i, ch, lo:hi]
            sig = float(np.sum(x * x))
            err = float(np.sum((y - x) * (y - x)))
            cleanpow[i, ch] = sig
            nmse[i, ch] = max(err, cfg.eps) / max(sig, cfg.eps)

    return nmse, cleanpow


def _rolling_quantiles_equal_count(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
    grid: int,
    q_lo: float,
    q_hi: float,
    min_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort by x; for grid points, take a local equal-count window and compute q_lo, median, q_hi.
    Returns x_grid, qlo, med, qhi.
    """
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m].astype(np.float64)
    y = y[m].astype(np.float64)
    n = x.size
    if n < min_points:
        return np.array([]), np.array([]), np.array([]), np.array([])

    order = np.argsort(x)
    xs = x[order]
    ys = y[order]

    window = int(min(window, n))
    half = max(1, window // 2)

    idxs = np.linspace(0, n - 1, int(grid)).astype(int)
    xg = np.empty_like(idxs, dtype=np.float64)
    qlo = np.empty_like(idxs, dtype=np.float64)
    med = np.empty_like(idxs, dtype=np.float64)
    qhi = np.empty_like(idxs, dtype=np.float64)

    for j, idx in enumerate(idxs):
        lo = max(0, idx - half)
        hi = min(n, idx + half)
        yy = ys[lo:hi]
        xg[j] = xs[idx]
        qlo[j], med[j], qhi[j] = np.quantile(yy, [q_lo, 0.50, q_hi])

    return xg, qlo, med, qhi


# -----------------------------
# Main plot (Check 1)
# -----------------------------
def plot_check1_nmse_std_vs_snr(
    clean_waveforms: np.ndarray,
    noisy_waveforms: np.ndarray,
    standard_waveforms: np.ndarray,
    ml_waveforms: Optional[np.ndarray] = None,
    extra_base_mask: Optional[np.ndarray] = None,  # (N,3) boolean
    cfg: Check1Config = Check1Config(),
) -> dict:
    """
    Returns a dict of arrays useful for debugging.
    """
    _validate_shapes(clean_waveforms, noisy_waveforms, standard_waveforms, ml_waveforms)
    N, C, _ = clean_waveforms.shape

    # SNR per trace+channel (anchored on clean peak time)
    snr, A_noisy, sigma_env = compute_snr_roi_peak(clean_waveforms, noisy_waveforms, cfg)

    # Trigger mask (optional)
    trigger_mask = np.ones((N, C), dtype=bool)
    if cfg.apply_trigger:
        trigger_mask = A_noisy >= (cfg.trigger_k_sigma * sigma_env)

    # Extra base mask (optional)
    if extra_base_mask is None:
        extra_base_mask = np.ones((N, C), dtype=bool)
    else:
        extra_base_mask = np.asarray(extra_base_mask, dtype=bool)
        if extra_base_mask.shape != (N, C):
            raise ValueError(f"extra_base_mask must have shape (N,3). Got {extra_base_mask.shape}")

    # NMSE + clean ROI power
    nmse_std, cleanpow = compute_roi_nmse_and_cleanpower(clean_waveforms, standard_waveforms, cfg)
    nmse_ml = None
    if ml_waveforms is not None:
        nmse_ml, _ = compute_roi_nmse_and_cleanpower(clean_waveforms, ml_waveforms, cfg)

    # Truth-conditioned clean-power gate
    power_mask = np.ones((N, C), dtype=bool)
    clean_power_floor = np.full((C,), np.nan, dtype=np.float64)
    if cfg.apply_clean_power_gate:
        for ch in range(C):
            base = trigger_mask[:, ch] if cfg.apply_trigger else np.ones(N, dtype=bool)
            vals = cleanpow[base & np.isfinite(cleanpow[:, ch]) & (cleanpow[:, ch] > 0.0), ch]
            if vals.size < cfg.clean_power_gate_min_count:
                raise RuntimeError(
                    f"Not enough positive CLEAN ROI-power samples for channel {ch} "
                    f"(found {vals.size}, need >= {cfg.clean_power_gate_min_count})."
                )
            floor = float(np.quantile(vals, cfg.clean_power_gate_quantile))
            floor = max(floor, float(cfg.clean_power_gate_abs_floor))
            clean_power_floor[ch] = floor
            power_mask[:, ch] = np.isfinite(cleanpow[:, ch]) & (cleanpow[:, ch] >= floor)

    # Candidate mask
    base_mask = trigger_mask & extra_base_mask & power_mask

    # Plot
    q_lo, q_hi = cfg.band_quantiles
    # Note: sharex=False because bottom row uses boxplots with different x-axis
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 7.8), sharex=False)

    for ch in range(3):
        m = base_mask[:, ch] & np.isfinite(snr[:, ch]) & np.isfinite(nmse_std[:, ch])
        x = snr[m, ch]
        y = nmse_std[m, ch]

        # Row 1: scatter + rolling median band (log y)
        ax = axes[0, ch]
        ax.set_title(f"{cfg.channel_names[ch]} channel")

        ax.scatter(x, y, s=cfg.scatter_s, alpha=cfg.scatter_alpha, linewidths=0)

        xg, ql, md, qh = _rolling_quantiles_equal_count(
            x, np.log10(y),
            window=cfg.rolling_window,
            grid=cfg.rolling_grid,
            q_lo=q_lo,
            q_hi=q_hi,
            min_points=cfg.rolling_min_points,
        )
        # We computed quantiles in log10-space (more stable visually); convert back
        if xg.size > 0:
            ax.fill_between(xg, 10**ql, 10**qh, alpha=0.20, linewidth=0)
            ax.plot(xg, 10**md, linewidth=2.4, label="Std median" if ch == 0 else None)

        # Optional ML overlay
        if cfg.show_ml_overlay and nmse_ml is not None:
            mm = base_mask[:, ch] & np.isfinite(snr[:, ch]) & np.isfinite(nmse_ml[:, ch])
            x2 = snr[mm, ch]
            y2 = nmse_ml[mm, ch]
            ax.scatter(x2, y2, s=cfg.scatter_s, alpha=0.07, linewidths=0)

            xg2, ql2, md2, qh2 = _rolling_quantiles_equal_count(
                x2, np.log10(y2),
                window=cfg.rolling_window,
                grid=cfg.rolling_grid,
                q_lo=q_lo,
                q_hi=q_hi,
                min_points=cfg.rolling_min_points,
            )
            if xg2.size > 0:
                ax.plot(xg2, 10**md2, linewidth=2.2, label="ML median" if ch == 0 else None)

        ax.set_yscale("log")
        ax.set_xlim(*cfg.xlim)
        ax.set_xlabel("SNR")
        ax.grid(True, alpha=0.25)
        if ch == 0:
            ax.set_ylabel("ROI-NMSE (log scale)")
        ax.text(
            0.02, 0.06,
            f"P_clean floor={clean_power_floor[ch]:.3g}\nN={int(np.sum(m))}",
            transform=ax.transAxes,
            fontsize=9.5,
            alpha=0.85,
            va="bottom",
        )

        # Row 2: high-SNR distribution (boxplot of log10 NMSE)
        ax2 = axes[1, ch]
        mh = m & (snr[:, ch] >= cfg.high_snr_min)
        yh = nmse_std[mh, ch]
        if yh.size > 0:
            # Fix: boxplot expects a sequence of sequences, convert 1D array to list of lists
            ax2.boxplot(
                [np.log10(yh).tolist()],
                vert=True,
                widths=0.35,
                showfliers=False,
            )
            ax2.axhline(np.median(np.log10(yh)), color="k", lw=1.0, alpha=0.75)
            ax2.set_xticks([1])
            ax2.set_xticklabels([fr"SNR$\geq${cfg.high_snr_min:g}"])
            ax2.set_xlim(0.5, 1.5)  # Ensure boxplot is visible
        else:
            ax2.text(0.5, 0.5, "No high-SNR points", ha="center", va="center", transform=ax2.transAxes)
            ax2.set_xticks([])
            ax2.set_xlim(0, 1)  # Default range for empty plot

        ax2.grid(True, alpha=0.25)
        if ch == 0:
            ax2.set_ylabel(r"$\log_{10}(\mathrm{NMSE})$ (high-SNR)")

        ax2.set_xlabel("SNR bin")

    # Always show legend for the first subplot (Std label is always present)
    axes[0, 0].legend(loc="upper right", frameon=False)

    fig.tight_layout()

    if cfg.savepath is not None:
        fig.savefig(cfg.savepath, dpi=cfg.dpi, bbox_inches="tight")
        print(f"Saved: {cfg.savepath}")

    plt.show()

    return {
        "snr": snr,
        "nmse_std": nmse_std,
        "nmse_ml": (np.array([]) if nmse_ml is None else nmse_ml),
        "base_mask": base_mask,
        "clean_power_floor": clean_power_floor,
    }


# -----------------------------
# Main entry point
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Appendix Check 1: Standard-method ROI-NMSE vs SNR (per channel)")
    ap.add_argument("--npz", default=None, help="NPZ with keys clean, noisy, standard, denoised (optional for ML overlay)")
    
    if HAS_ML_UTILS:
        add_model_arguments(ap)
    
    ap.add_argument("--standard-npz", default=None, help="NPZ file with 'standard' key for standard denoiser waveforms")
    ap.add_argument("--ml-npz", default=None, help="NPZ file with 'denoised' key for ML waveforms (for overlay)")
    
    ap.add_argument("--dt-ns", type=float, default=2.0, help="Sampling interval in ns.")
    
    # ROI parameters
    ap.add_argument("--roi-half-width-ns", type=float, default=150.0,
                    help="ROI half-width (ns) around CLEAN peak (default: 150.0)")
    
    # SNR estimation parameters (ROI-peak style)
    ap.add_argument("--snr-peak-half-width-ns", type=float, default=150.0,
                    help="Half-width (ns) for peak search window around CLEAN peak (default: 150.0)")
    ap.add_argument("--snr-exclude-half-width-ns", type=float, default=150.0,
                    help="Half-width (ns) to exclude around CLEAN peak for SNR sigma estimate (default: 150.0)")
    ap.add_argument("--mad-scale-gaussian", type=float, default=1.4826022185056,
                    help="MAD scale factor for Gaussian consistency (default: 1.4826...)")
    ap.add_argument("--eps", type=float, default=1e-12,
                    help="Epsilon for numerical stability (default: 1e-12)")
    
    # Trigger-like selection
    ap.add_argument("--apply-trigger", action="store_true", default=False,
                    help="Enable trigger-like selection (default: disabled)")
    ap.add_argument("--trigger-k-sigma", type=float, default=1.0,
                    help="Trigger threshold in sigma units (default: 1.0)")
    
    # Bandpass parameters
    ap.add_argument("--no-apply-bandpass", dest="apply_bandpass", action="store_false", default=True,
                    help="Disable bandpass filtering (default: enabled)")
    ap.add_argument("--f-lo-hz", type=float, default=50e6,
                    help="Bandpass lower frequency (Hz) (default: 50e6)")
    ap.add_argument("--f-hi-hz", type=float, default=200e6,
                    help="Bandpass upper frequency (Hz) (default: 200e6)")
    ap.add_argument("--butter-order", type=int, default=4,
                    help="Butterworth filter order (default: 4)")
    ap.add_argument("--no-apply-bandpass-to-snr", dest="apply_bandpass_to_snr", action="store_false", default=True,
                    help="Disable bandpass for SNR computation (default: enabled)")
    
    # Truth-conditioned clean power gate parameters
    ap.add_argument("--no-apply-clean-power-gate", dest="apply_clean_power_gate", action="store_false", default=True,
                    help="Disable clean power gate (default: enabled)")
    ap.add_argument("--clean-power-gate-quantile", type=float, default=0.10,
                    help="Quantile for clean power gate floor (default: 0.10)")
    ap.add_argument("--clean-power-gate-min-count", type=int, default=200,
                    help="Minimum positive samples to set clean power gate floor (default: 200)")
    ap.add_argument("--clean-power-gate-abs-floor", type=float, default=0.0,
                    help="Optional absolute lower bound on floor (default: 0.0)")
    
    # Rolling summary parameters
    ap.add_argument("--rolling-window", type=int, default=2000,
                    help="Rolling window size for quantiles (default: 2000)")
    ap.add_argument("--rolling-grid", type=int, default=60,
                    help="Number of grid points for rolling quantiles (default: 60)")
    ap.add_argument("--rolling-min-points", type=int, default=150,
                    help="Minimum points required for rolling quantiles (default: 150)")
    ap.add_argument("--band-quantiles", type=str, default="0.16,0.84",
                    help="Band quantiles: low,high (default: 0.16,0.84)")
    
    # High-SNR definition
    ap.add_argument("--high-snr-min", type=float, default=8.0,
                    help="Minimum SNR for high-SNR distribution (default: 8.0)")
    
    # Plot cosmetics
    ap.add_argument("--scatter-alpha", type=float, default=0.10,
                    help="Scatter plot alpha (default: 0.10)")
    ap.add_argument("--scatter-s", type=float, default=6.0,
                    help="Scatter plot size (default: 6.0)")
    ap.add_argument("--show-ml-overlay", action="store_true", default=False,
                    help="Show ML overlay on plot (default: False)")
    ap.add_argument("--xlim", type=str, default="1.5,15.0",
                    help="X-axis limits: low,high (default: 1.5,15.0)")
    ap.add_argument("--channel-names", default="X,Y,Z",
                    help="Comma-separated channel labels (default: X,Y,Z)")
    
    # Output
    ap.add_argument("--out", default="fig_appendix_check1_nmse_std_vs_snr.pdf", help="Output figure path.")
    ap.add_argument("--dpi", type=int, default=220, help="Figure DPI (default: 220)")
    
    args = ap.parse_args()
    
    # Parse channel names
    channel_names = tuple(name.strip() for name in args.channel_names.split(","))
    if len(channel_names) != 3:
        raise ValueError(f"Expected 3 channel names, got {len(channel_names)}")
    
    # Parse band quantiles
    quantiles_str = args.band_quantiles.split(",")
    if len(quantiles_str) != 2:
        raise ValueError(f"--band-quantiles must have 2 comma-separated values, got {len(quantiles_str)}")
    band_quantiles = (float(quantiles_str[0]), float(quantiles_str[1]))
    
    # Parse xlim
    xlim_str = args.xlim.split(",")
    if len(xlim_str) != 2:
        raise ValueError(f"--xlim must have 2 comma-separated values, got {len(xlim_str)}")
    xlim = (float(xlim_str[0]), float(xlim_str[1]))
    
    # Load data
    ml_waveforms = None
    standard_waveforms = None
    
    if args.npz is not None:
        d = np.load(args.npz, allow_pickle=False)
        clean = d["clean"]
        noisy = d["noisy"]
        # Check for standard waveforms in main NPZ
        if "standard" in d:
            standard_waveforms = d["standard"]
        # Check for ML waveforms in main NPZ (for overlay)
        if "denoised" in d:
            ml_waveforms = d["denoised"]
    elif HAS_ML_UTILS and check_model_args(args):
        if not all([args.model_path, args.metrics_json, args.config_json]):
            raise ValueError("ML mode requires --model-path, --metrics-json, --config-json")
        eval_pack = load_data_and_run_inference(
            model_path=args.model_path,
            metrics_json=args.metrics_json,
            config_json=args.config_json,
            data_path=args.data_path,
            device=args.device,
            batch_size=args.batch_size,
            max_samples=getattr(args, 'max_samples', None),
        )
        clean = eval_pack.clean
        noisy = eval_pack.noisy
        ml_waveforms = eval_pack.denoised  # Use ML denoised as overlay
    else:
        raise ValueError("Provide either --npz or ML model arguments (--model-path, etc.)")
    
    # Load standard denoiser waveforms if provided separately
    if args.standard_npz is not None:
        d_std = np.load(args.standard_npz, allow_pickle=False)
        if "standard" in d_std:
            standard_waveforms = d_std["standard"]
        elif "denoised" in d_std:
            standard_waveforms = d_std["denoised"]
        else:
            raise ValueError(f"--standard-npz must contain 'standard' or 'denoised' key. Found keys: {list(d_std.keys())}")
    
    # Load ML waveforms separately if provided (for overlay)
    if args.ml_npz is not None:
        d_ml = np.load(args.ml_npz, allow_pickle=False)
        if "denoised" in d_ml:
            ml_waveforms = d_ml["denoised"]
        else:
            raise ValueError(f"--ml-npz must contain 'denoised' key. Found keys: {list(d_ml.keys())}")
    
    # If standard_waveforms not provided, use noisy_waveforms as baseline
    if standard_waveforms is None:
        print("Warning: standard_waveforms not provided, using noisy_waveforms as baseline")
        standard_waveforms = noisy
    
    cfg = Check1Config(
        dt_ns=args.dt_ns,
        eps=args.eps,
        roi_half_width_ns=args.roi_half_width_ns,
        snr_peak_half_width_ns=args.snr_peak_half_width_ns,
        snr_exclude_half_width_ns=args.snr_exclude_half_width_ns,
        mad_scale_gaussian=args.mad_scale_gaussian,
        apply_bandpass=args.apply_bandpass,
        f_lo_hz=args.f_lo_hz,
        f_hi_hz=args.f_hi_hz,
        butter_order=args.butter_order,
        apply_bandpass_to_snr=args.apply_bandpass_to_snr,
        apply_trigger=args.apply_trigger,
        trigger_k_sigma=args.trigger_k_sigma,
        apply_clean_power_gate=args.apply_clean_power_gate,
        clean_power_gate_quantile=args.clean_power_gate_quantile,
        clean_power_gate_min_count=args.clean_power_gate_min_count,
        clean_power_gate_abs_floor=args.clean_power_gate_abs_floor,
        rolling_window=args.rolling_window,
        rolling_grid=args.rolling_grid,
        rolling_min_points=args.rolling_min_points,
        band_quantiles=band_quantiles,
        high_snr_min=args.high_snr_min,
        channel_names=channel_names,
        scatter_alpha=args.scatter_alpha,
        scatter_s=args.scatter_s,
        show_ml_overlay=args.show_ml_overlay,
        xlim=xlim,
        savepath=args.out,
        dpi=args.dpi,
    )
    
    plot_check1_nmse_std_vs_snr(
        clean_waveforms=clean,
        noisy_waveforms=noisy,
        standard_waveforms=standard_waveforms,
        ml_waveforms=ml_waveforms,
        extra_base_mask=None,
        cfg=cfg
    )


if __name__ == "__main__":
    main()