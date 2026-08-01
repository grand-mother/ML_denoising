#!/usr/bin/env python3
"""
make_fig_nmse_snr_gain_vs_snr__option1_truth_cleanpower_gate__ROI_SNR_q05q95.py

Drop-in replacement for the Option-1 truth-conditioned NMSE/SNR-gain figure.

NEW (requested):
  (1) SNR axis uses *ROI-peak SNR*:
        A_noisy = max(|hilbert(noisy)|) within a time window centered on the CLEAN peak time.
      sigma_env is still a robust MAD-based envelope scale computed *away from the pulse*
      (excluding a window centered on the CLEAN peak time).
      This avoids extreme-value inflation from taking a global max over the full trace.

  (2) Shaded band uses a wider *population spread band* q05..q95 (not a CI on the median).

Notes:
  - This shaded band is NOT a bootstrap/CI; it is the conditional distribution spread in each rolling SNR window.
  - The truth-conditioned clean-power gate (Option 1) remains, to avoid ill-conditioned NMSE in channels
    with negligible true signal energy in the ROI.

Inputs (aligned arrays):
  clean_waveforms:    (N, 3, T)
  noisy_waveforms:    (N, 3, T)
  denoised_waveforms: (N, 3, T)
  standard_waveforms: (N, 3, T) optional

Usage:
- Command line: python make_fig_nmse_snr_gain_vs_snr__option3_truth_cleanpower_gate.py --model-path ... --metrics-json ... --config-json ...
- Or import and call plot_nmse_and_snr_gain_vs_snr(clean, noisy, denoised, standard_waveforms=None, snr=None, cfg=...).

Dependencies: numpy, matplotlib, scipy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, sosfiltfilt

# Canonical paper SNR (task 3): one shared definition for every figure.
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# SNR here is the paper definition max(clean)/std(noisy) over the full trace,
# computed inline; no off-pulse helper is imported.

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
class FidelityPlotConfig:
    # Sampling interval
    dt_ns: float = 2.0

    # ROI for waveform-fidelity metrics, centered on CLEAN envelope peak
    roi_half_width_ns: float = 150.0

    # --- SNR axis (paper-style, but ROI-peak amplitude) ---
    # Window for *peak search* in the noisy envelope, centered on CLEAN peak time.
    # If you want to push lower SNR, shrink this (e.g. 50 ns). Default = same as ROI.
    snr_peak_half_width_ns: float = 150.0

    # Exclusion window (centered on CLEAN peak) for estimating sigma_env from the noisy envelope.
    # Usually >= snr_peak_half_width_ns.
    snr_exclude_half_width_ns: float = 150.0

    # Robust sigma for envelope fluctuations via MAD *with normal-consistency factor*
    mad_scale_gaussian: float = 1.4826022185056
    snr_eps: float = 1e-12

    # Trigger-like selection (turn OFF to see low-SNR behavior)
    apply_trigger: bool = False
    trigger_k_sigma: float = 1.0

    # In-band fidelity (time-domain, after bandpass)
    apply_bandpass: bool = True
    f_lo_hz: float = 50e6
    f_hi_hz: float = 200e6
    butter_order: int = 4

    # Apply the same bandpass before computing SNR envelope?
    # If your paper’s SNR is defined in-band, keep True. If it was broadband, set False.
    apply_bandpass_to_snr: bool = True

    # Option 1 truth-conditioning gate on CLEAN ROI power (per channel)
    apply_clean_power_gate: bool = True
    clean_power_gate_quantile: float = 0.10   # conservative floor: drop lowest 10% of positive clean ROI powers
    clean_power_gate_min_count: int = 200      # require this many positives to set stable floor
    clean_power_gate_abs_floor: float = 0.0

    # Rolling population bands (equal-count window, adaptive at edges)
    rolling_window: int = 400
    rolling_min_points: int = 80
    rolling_grid: int = 45  # denser grid helps show low-SNR structure

    # NEW: Wider population band for visual spread
    band_quantiles: Tuple[float, float] = (0.05, 0.95)  # q05..q95

    # Plot styling
    channel_names: Tuple[str, str, str] = ("X", "Y", "Z")
    labels: Tuple[str, str, str] = ("Noisy", "Standard", "ML")
    colors: Tuple[str, str, str] = ("tab:red", "tab:orange", "tab:blue")
    band_alpha: float = 0.22

    # Axes
    xlim: Tuple[float, float] = (1.5, 15.0)
    nmse_ylim: Optional[Tuple[float, float]] = None
    snrgain_ylim: Optional[Tuple[float, float]] = None

    # Output
    savepath: Optional[str] = None
    dpi: int = 220


# -----------------------------
# Validation
# -----------------------------
def _validate_shapes(clean: np.ndarray, noisy: np.ndarray, den: np.ndarray, std: Optional[np.ndarray]) -> None:
    for name, arr in [("clean", clean), ("noisy", noisy), ("denoised", den)]:
        if not isinstance(arr, np.ndarray) or arr.ndim != 3:
            raise ValueError(f"{name} must have shape (N,3,T). Got {None if arr is None else arr.shape}")
    if clean.shape != noisy.shape or clean.shape != den.shape:
        raise ValueError(f"clean/noisy/denoised shapes must match. Got {clean.shape}, {noisy.shape}, {den.shape}")
    if clean.shape[1] != 3:
        raise ValueError(f"Expected 3 channels. Got {clean.shape[1]}")
    if std is not None:
        if not isinstance(std, np.ndarray) or std.ndim != 3 or std.shape != clean.shape:
            raise ValueError(f"standard_waveforms must match clean shape (N,3,T). Got {std.shape}")


# -----------------------------
# Signal helpers
# -----------------------------
def _analytic_envelope(x: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(x, axis=-1))


def _mad_sigma(x: np.ndarray, scale: float) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(scale * mad)


def _check_bandpass(cfg: FidelityPlotConfig) -> None:
    if not cfg.apply_bandpass:
        return
    dt_s = cfg.dt_ns * 1e-9
    fs = 1.0 / dt_s
    nyq = 0.5 * fs
    if cfg.f_hi_hz >= nyq:
        raise ValueError(
            f"Bandpass upper cutoff {cfg.f_hi_hz/1e6:.1f} MHz >= Nyquist {nyq/1e6:.1f} MHz. "
            f"Check dt_ns={cfg.dt_ns}."
        )


def _maybe_bandpass(x: np.ndarray, cfg: FidelityPlotConfig) -> np.ndarray:
    if not cfg.apply_bandpass:
        return x
    _check_bandpass(cfg)
    dt_s = cfg.dt_ns * 1e-9
    fs = 1.0 / dt_s
    sos = butter(
        cfg.butter_order,
        [cfg.f_lo_hz, cfg.f_hi_hz],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    return sosfiltfilt(sos, x, axis=-1)


def _roi_indices_centered_on_peak_index(T: int, k: int, dt_ns: float, half_width_ns: float) -> Tuple[int, int]:
    half = max(1, int(round(half_width_ns / dt_ns)))
    lo = max(0, k - half)
    hi = min(T, k + half + 1)
    return lo, hi


def _clean_peak_index(clean_1d: np.ndarray) -> int:
    env = np.abs(hilbert(clean_1d))
    return int(np.argmax(env))


# -----------------------------
# NEW: ROI-peak SNR (centered on CLEAN peak time)
# -----------------------------
# DEPRECATED (task 3): the noisy-ROI-peak / envelope-MAD SNR below is NOT the
# paper SNR and is no longer used for the figure's x-axis (replaced by
# the paper definition). Retained only so existing imports keep resolving.
def _compute_snr_roi_peak_style(
    clean: np.ndarray,
    noisy: np.ndarray,
    cfg: FidelityPlotConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per trace, per channel:

      k_clean = argmax(|hilbert(clean)|)   (optionally in-band if cfg.apply_bandpass_to_snr)
      A_noisy = max(|hilbert(noisy)| over t in [k_clean +/- snr_peak_half_width_ns])
      sigma_env = MAD-scale(|hilbert(noisy)| over samples excluding [k_clean +/- snr_exclude_half_width_ns])
      SNR = A_noisy / sigma_env

    Returns:
      snr (N,3), A_noisy (N,3), sigma_env (N,3)
    """
    N, C, T = noisy.shape
    snr = np.full((N, C), np.nan, dtype=np.float64)
    A = np.full((N, C), np.nan, dtype=np.float64)
    sigma = np.full((N, C), np.nan, dtype=np.float64)

    # Optional in-band filtering for SNR definition
    if cfg.apply_bandpass_to_snr and cfg.apply_bandpass:
        clean_s = _maybe_bandpass(clean.astype(np.float64), cfg)
        noisy_s = _maybe_bandpass(noisy.astype(np.float64), cfg)
    else:
        clean_s = clean.astype(np.float64)
        noisy_s = noisy.astype(np.float64)

    peak_half = float(cfg.snr_peak_half_width_ns)
    excl_half = float(cfg.snr_exclude_half_width_ns)

    for ch in range(C):
        # Envelope for noisy (for A and sigma)
        env_noisy = _analytic_envelope(noisy_s[:, ch, :])

        # Peak time from CLEAN (per trace)
        k_clean_all = np.array([_clean_peak_index(clean_s[i, ch, :]) for i in range(N)], dtype=int)

        for i in range(N):
            k = int(k_clean_all[i])

            # A_noisy: max envelope inside a peak-search window around CLEAN peak time
            lo_p, hi_p = _roi_indices_centered_on_peak_index(T, k, cfg.dt_ns, peak_half)
            A_i = float(np.max(env_noisy[i, lo_p:hi_p]))
            A[i, ch] = A_i

            # sigma_env: robust MAD on envelope outside an exclusion window around CLEAN peak time
            lo_e, hi_e = _roi_indices_centered_on_peak_index(T, k, cfg.dt_ns, excl_half)
            mask = np.ones(T, dtype=bool)
            mask[lo_e:hi_e] = False
            env_noise = env_noisy[i, mask]
            if env_noise.size < 16:
                env_noise = env_noisy[i, :]

            s = _mad_sigma(env_noise, cfg.mad_scale_gaussian)
            s = max(s, cfg.snr_eps)
            sigma[i, ch] = s
            snr[i, ch] = A_i / s

    return snr, A, sigma

# -----------------------------
# Fidelity metrics + clean ROI power
# -----------------------------
def _nmse_snrout_and_cleanpower_in_roi(
    clean: np.ndarray, rec: np.ndarray, cfg: FidelityPlotConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute in a +/- ROI around CLEAN envelope peak (per trace, per channel):

      NMSE = ||rec-clean||^2 / ||clean||^2   (ROI)
      SNR_out(dB) = 10 log10( ||clean||^2 / ||rec-clean||^2 ) (ROI)
      clean_power = ||clean||^2 (ROI)

    Returns (N,3) arrays.
    """
    N, C, T = clean.shape

    clean_f = _maybe_bandpass(clean.astype(np.float64), cfg)
    rec_f = _maybe_bandpass(rec.astype(np.float64), cfg)

    nmse = np.full((N, C), np.nan, dtype=np.float64)
    snrout_db = np.full((N, C), np.nan, dtype=np.float64)
    clean_pow = np.full((N, C), np.nan, dtype=np.float64)

    for ch in range(C):
        for i in range(N):
            k = _clean_peak_index(clean_f[i, ch, :])
            lo, hi = _roi_indices_centered_on_peak_index(T, k, cfg.dt_ns, cfg.roi_half_width_ns)

            x = clean_f[i, ch, lo:hi]
            y = rec_f[i, ch, lo:hi]

            sig_pow = float(np.sum(x * x))
            err_pow = float(np.sum((y - x) * (y - x)))

            sig_pow_safe = max(sig_pow, cfg.snr_eps)
            err_pow_safe = max(err_pow, cfg.snr_eps)

            clean_pow[i, ch] = sig_pow
            nmse[i, ch] = err_pow_safe / sig_pow_safe
            snrout_db[i, ch] = 10.0 * np.log10(sig_pow_safe / err_pow_safe)

    return nmse, snrout_db, clean_pow


# -----------------------------
# Rolling quantiles (population spread), adaptive at edges
# -----------------------------
def _rolling_quantiles_adaptive(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
    grid: int,
    q_lo: float,
    q_hi: float,
    min_points: int,
) -> Dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    n = x.size
    if n < min_points:
        return {"x": np.array([]), "qlo": np.array([]), "median": np.array([]), "qhi": np.array([]), "n": np.array([])}

    order = np.argsort(x)
    xs = x[order]
    ys = y[order]

    grid = int(grid)
    idxs = np.linspace(0, n - 1, grid).astype(int)

    window_eff = int(min(window, n))
    half = max(1, window_eff // 2)

    xg = np.empty(grid, dtype=np.float64)
    qlo = np.empty(grid, dtype=np.float64)
    q50 = np.empty(grid, dtype=np.float64)
    qhi = np.empty(grid, dtype=np.float64)
    n_eff = np.empty(grid, dtype=np.int64)

    for j, idx in enumerate(idxs):
        lo = max(0, idx - half)
        hi = min(n, idx + half)
        yy = ys[lo:hi]

        xg[j] = xs[idx]
        qlo[j], q50[j], qhi[j] = np.quantile(yy, [q_lo, 0.50, q_hi])
        n_eff[j] = yy.size

    return {"x": xg, "qlo": qlo, "median": q50, "qhi": qhi, "n": n_eff}


# -----------------------------
# Main plotting function
# -----------------------------
def plot_nmse_and_snr_gain_vs_snr(
    clean_waveforms: np.ndarray,
    noisy_waveforms: np.ndarray,
    denoised_waveforms: np.ndarray,
    standard_waveforms: Optional[np.ndarray] = None,
    snr: Optional[np.ndarray] = None,
    cfg: FidelityPlotConfig = FidelityPlotConfig(),
) -> None:
    """
    2x3 panel figure:
      top row: ROI-NMSE vs SNR (log y)
      bottom row: ΔSNR_out(dB) vs SNR relative to NOISY baseline (NOISY = 0)

    Shaded bands are *population* q05..q95 in rolling SNR windows (adaptive at edges).
    """
    _validate_shapes(clean_waveforms, noisy_waveforms, denoised_waveforms, standard_waveforms)
    N, C, _ = clean_waveforms.shape

    # --- SNR for x-axis: the paper definition used throughout the analysis ---
    #     SNR = max(clean) / std(noisy)
    # with the standard deviation over the FULL trace, channel by channel.
    with np.errstate(divide="ignore", invalid="ignore"):
        _snr_paper = np.max(clean_waveforms, axis=-1) / np.std(noisy_waveforms, axis=-1)
    # Envelope quantities kept only for the optional trigger-style selection below.
    A_noisy = np.max(np.abs(hilbert(noisy_waveforms, axis=-1)), axis=-1)
    sigma_env = np.std(noisy_waveforms, axis=-1)
    if snr is None:
        snr_pc = _snr_paper
    else:
        snr = np.asarray(snr, dtype=np.float64)
        if snr.shape != (N, 3):
            raise ValueError(f"Provided snr must have shape (N,3). Got {snr.shape}")
        snr_pc = snr

    # Trigger mask (optional)
    trigger_mask = np.ones((N, C), dtype=bool)
    if cfg.apply_trigger:
        trigger_mask = A_noisy >= (cfg.trigger_k_sigma * sigma_env)

    # Metrics (and CLEAN ROI power)
    nmse_noisy, snrout_noisy, cleanpow = _nmse_snrout_and_cleanpower_in_roi(clean_waveforms, noisy_waveforms, cfg)
    nmse_den, snrout_den, _ = _nmse_snrout_and_cleanpower_in_roi(clean_waveforms, denoised_waveforms, cfg)

    nmse_std = snrout_std = None
    if standard_waveforms is not None:
        nmse_std, snrout_std, _ = _nmse_snrout_and_cleanpower_in_roi(clean_waveforms, standard_waveforms, cfg)

    # Option 1: truth-conditioned signal-presence gate on CLEAN ROI power (per channel)
    power_mask = np.ones((N, C), dtype=bool)
    clean_power_floor = np.full((C,), np.nan, dtype=np.float64)

    if cfg.apply_clean_power_gate:
        for ch in range(C):
            base = trigger_mask[:, ch] if cfg.apply_trigger else np.ones(N, dtype=bool)
            vals = cleanpow[base & np.isfinite(cleanpow[:, ch]) & (cleanpow[:, ch] > 0.0), ch]
            if vals.size < cfg.clean_power_gate_min_count:
                raise RuntimeError(
                    f"Not enough positive CLEAN ROI-power samples to set truth gate in channel {ch} "
                    f"(found {vals.size}, need >= {cfg.clean_power_gate_min_count}). "
                    f"Try lowering clean_power_gate_min_count or disabling the gate."
                )

            floor = float(np.quantile(vals, cfg.clean_power_gate_quantile))
            floor = max(floor, float(cfg.clean_power_gate_abs_floor))
            clean_power_floor[ch] = floor

            power_mask[:, ch] = np.isfinite(cleanpow[:, ch]) & (cleanpow[:, ch] >= floor)

    # Final mask applied to all series for fairness
    final_mask = trigger_mask & power_mask

    def mask_arr(a: np.ndarray) -> np.ndarray:
        return np.where(final_mask, a, np.nan)

    nmse_noisy = mask_arr(nmse_noisy)
    nmse_den = mask_arr(nmse_den)
    snrout_noisy = mask_arr(snrout_noisy)
    snrout_den = mask_arr(snrout_den)
    if nmse_std is not None:
        nmse_std = mask_arr(nmse_std)
        snrout_std = mask_arr(snrout_std)

    dsnr_den = snrout_den - snrout_noisy
    dsnr_std = None if snrout_std is None else (snrout_std - snrout_noisy)

    # Plot
    q_lo, q_hi = cfg.band_quantiles
    fontsize = 24
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex=True)
    handles: List[plt.Line2D] = []

    for ch in range(C):
        x = snr_pc[:, ch]

        # --- Top: NMSE ---
        ax = axes[0, ch]
        ax.set_title(cfg.channel_names[ch], fontsize=fontsize)

        series = [
            (nmse_noisy[:, ch], cfg.labels[0], cfg.colors[0], "--", "s"),
        ]
        if nmse_std is not None:
            series.append((nmse_std[:, ch], cfg.labels[1], cfg.colors[1], "-", "d"))
        series.append((nmse_den[:, ch], cfg.labels[2], cfg.colors[2], "-", "o"))

        n_eff_list = []
        for y, lab, col, ls, mk in series:
            s = _rolling_quantiles_adaptive(
                x, y,
                window=cfg.rolling_window,
                grid=cfg.rolling_grid,
                q_lo=q_lo,
                q_hi=q_hi,
                min_points=cfg.rolling_min_points,
            )
            if s["x"].size == 0:
                continue

            n_eff_list.append(np.nanmedian(s["n"]) if s["n"].size else np.nan)

            ax.fill_between(s["x"], s["qlo"], s["qhi"], color=col, alpha=cfg.band_alpha, linewidth=0)
            h = ax.plot(
                s["x"], s["median"],
                linestyle=ls, linewidth=2.2,
                marker=mk, markersize=4.5,
                color=col, label=lab
            )[0]
            if ch == 0:
                handles.append(h)

        ax.set_yscale("log")
        if cfg.nmse_ylim is not None:
            ax.set_ylim(*cfg.nmse_ylim)
        ax.set_xlim(*cfg.xlim)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        if ch == 0:
            ax.set_ylabel("NMSE", fontsize=fontsize)

    
        # --- Bottom: ΔSNR_out ---
        ax2 = axes[1, ch]
        ax2.axhline(0.0, color="k", linewidth=1.0, alpha=0.6, ls='--')
        ax2.set_ylim(-5, 35)
        ax2.tick_params(axis="both", which="major", labelsize=fontsize)

        series2 = []
        if dsnr_std is not None:
            series2.append((dsnr_std[:, ch], cfg.labels[1], cfg.colors[1], "-", "d"))
        series2.append((dsnr_den[:, ch], cfg.labels[2], cfg.colors[2], "-", "o"))

        n_eff_list2 = []
        for y, lab, col, ls, mk in series2:
            s = _rolling_quantiles_adaptive(
                x, y,
                window=cfg.rolling_window,
                grid=cfg.rolling_grid,
                q_lo=q_lo,
                q_hi=q_hi,
                min_points=cfg.rolling_min_points,
            )
            if s["x"].size == 0:
                continue

            n_eff_list2.append(np.nanmedian(s["n"]) if s["n"].size else np.nan)

            ax2.fill_between(s["x"], s["qlo"], s["qhi"], color=col, alpha=cfg.band_alpha, linewidth=0)
            ax2.plot(
                s["x"], s["median"],
                linestyle=ls, linewidth=2.2,
                marker=mk, markersize=4.5,
                color=col
            )

        if cfg.snrgain_ylim is not None:
            ax2.set_ylim(*cfg.snrgain_ylim)
        ax2.set_xlim(*cfg.xlim)
        ax2.grid(True, alpha=0.25)
        ax2.set_xlabel("SNR",fontsize=fontsize)
        if ch == 0:
            ax2.set_ylabel(r"$\mathrm{SNR}_{\rm out}^{\rm rec}-\mathrm{SNR}_{\rm out}^{\rm noisy}$ (dB)", fontsize=fontsize)
        
    if handles:
        axes[0, 0].legend(handles=handles, loc="upper right", frameon=False, borderaxespad=0.6, fontsize=fontsize)

    fig.tight_layout()
    if cfg.savepath is not None:
        fig.savefig(cfg.savepath, dpi=cfg.dpi, bbox_inches="tight")
        print(f"Saved: {cfg.savepath}")

    plt.show()


# -----------------------------
# Main entry point
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="NMSE and SNR gain vs SNR (Option 3: ROI-peak SNR, truth-conditioned clean power gate, q05-q95 bands)")
    ap.add_argument("--npz", default=None, help="NPZ with keys clean, noisy, denoised, snr (optional), standard (optional)")
    
    if HAS_ML_UTILS:
        add_model_arguments(ap)
    
    ap.add_argument("--standard-npz", default=None, help="NPZ file with 'standard' key for standard denoiser waveforms")
    
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
    ap.add_argument("--snr-eps", type=float, default=1e-12,
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
    
    # Option 3: truth-conditioned clean power gate parameters
    ap.add_argument("--no-apply-clean-power-gate", dest="apply_clean_power_gate", action="store_false", default=True,
                    help="Disable clean power gate (default: enabled)")
    ap.add_argument("--clean-power-gate-quantile", type=float, default=0.10,
                    help="Quantile for clean power gate floor (default: 0.10)")
    ap.add_argument("--clean-power-gate-min-count", type=int, default=200,
                    help="Minimum positive samples to set clean power gate floor (default: 200)")
    ap.add_argument("--clean-power-gate-abs-floor", type=float, default=0.0,
                    help="Optional absolute lower bound on floor (default: 0.0)")
    
    # Rolling window parameters (adaptive)
    ap.add_argument("--rolling-window", type=int, default=400,
                    help="Points per window for rolling quantiles (default: 400)")
    ap.add_argument("--rolling-min-points", type=int, default=80,
                    help="Minimum points required to draw anything (default: 80)")
    ap.add_argument("--rolling-grid", type=int, default=45,
                    help="Number of x locations to evaluate (default: 45)")
    
    # Band quantiles (wider for Option 3)
    ap.add_argument("--band-quantiles", type=str, default="0.05,0.95",
                    help="Band quantiles: low,high (default: 0.05,0.95)")
    
    # Styling
    ap.add_argument("--band-alpha", type=float, default=0.22,
                    help="Alpha for bands (default: 0.22)")
    ap.add_argument("--channel-names", default="X,Y,Z",
                    help="Comma-separated channel labels (default: X,Y,Z)")
    ap.add_argument("--noisy-label", default="Noisy", help="Label for noisy traces")
    ap.add_argument("--standard-label", default="Standard", help="Label for standard denoiser traces")
    ap.add_argument("--ml-label", default="ML", help="Label for ML denoiser traces")
    
    # Axes limits
    ap.add_argument("--xlim", type=str, default="1.5,15.0",
                    help="X-axis limits: low,high (default: 1.5,15.0)")
    ap.add_argument("--nmse-ylim", type=str, default=None,
                    help="NMSE y-axis limits: low,high (default: auto)")
    ap.add_argument("--snrgain-ylim", type=str, default=None,
                    help="SNR gain y-axis limits: low,high (default: auto)")
    
    # Output
    ap.add_argument("--out", default="fig_nmse_snrgain_vs_snr_option3.pdf", help="Output figure path.")
    ap.add_argument("--dpi", type=int, default=220, help="Figure DPI (default: 220)")
    
    args = ap.parse_args()
    
    # Parse channel names
    channel_names = tuple(name.strip() for name in args.channel_names.split(","))
    if len(channel_names) != 3:
        raise ValueError(f"Expected 3 channel names, got {len(channel_names)}")
    
    # Parse labels
    labels = (args.noisy_label, args.standard_label, args.ml_label)
    
    # Parse xlim
    xlim_str = args.xlim.split(",")
    if len(xlim_str) != 2:
        raise ValueError(f"--xlim must have 2 comma-separated values, got {len(xlim_str)}")
    xlim = (float(xlim_str[0]), float(xlim_str[1]))
    
    # Parse ylims
    nmse_ylim = None
    if args.nmse_ylim is not None:
        ylim_str = args.nmse_ylim.split(",")
        if len(ylim_str) != 2:
            raise ValueError(f"--nmse-ylim must have 2 comma-separated values, got {len(ylim_str)}")
        nmse_ylim = (float(ylim_str[0]), float(ylim_str[1]))
    
    snrgain_ylim = None
    if args.snrgain_ylim is not None:
        ylim_str = args.snrgain_ylim.split(",")
        if len(ylim_str) != 2:
            raise ValueError(f"--snrgain-ylim must have 2 comma-separated values, got {len(ylim_str)}")
        snrgain_ylim = (float(ylim_str[0]), float(ylim_str[1]))
    
    # Parse band quantiles
    quantiles_str = args.band_quantiles.split(",")
    if len(quantiles_str) != 2:
        raise ValueError(f"--band-quantiles must have 2 comma-separated values, got {len(quantiles_str)}")
    band_quantiles = (float(quantiles_str[0]), float(quantiles_str[1]))
    
    # Load data
    snr_external = None
    standard_waveforms = None
    
    if args.npz is not None:
        d = np.load(args.npz, allow_pickle=False)
        clean = d["clean"]
        noisy = d["noisy"]
        denoised = d["denoised"]
        # If snr key exists, use it (may be (N,) or (N,3))
        if "snr" in d:
            snr_external = d["snr"]
        # If standard key exists in main npz, use it
        if "standard" in d:
            standard_waveforms = d["standard"]
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
        denoised = eval_pack.denoised
        # Note: eval_pack.snr is (N,), but we compute per-channel SNR internally
        # So we set snr_external=None to trigger internal computation
        snr_external = None
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
    
    cfg = FidelityPlotConfig(
        dt_ns=args.dt_ns,
        roi_half_width_ns=args.roi_half_width_ns,
        snr_peak_half_width_ns=args.snr_peak_half_width_ns,
        snr_exclude_half_width_ns=args.snr_exclude_half_width_ns,
        mad_scale_gaussian=args.mad_scale_gaussian,
        snr_eps=args.snr_eps,
        apply_trigger=args.apply_trigger,
        trigger_k_sigma=args.trigger_k_sigma,
        apply_bandpass=args.apply_bandpass,
        f_lo_hz=args.f_lo_hz,
        f_hi_hz=args.f_hi_hz,
        butter_order=args.butter_order,
        apply_bandpass_to_snr=args.apply_bandpass_to_snr,
        apply_clean_power_gate=args.apply_clean_power_gate,
        clean_power_gate_quantile=args.clean_power_gate_quantile,
        clean_power_gate_min_count=args.clean_power_gate_min_count,
        clean_power_gate_abs_floor=args.clean_power_gate_abs_floor,
        rolling_window=args.rolling_window,
        rolling_min_points=args.rolling_min_points,
        rolling_grid=args.rolling_grid,
        band_quantiles=band_quantiles,
        channel_names=channel_names,
        labels=labels,
        band_alpha=args.band_alpha,
        xlim=xlim,
        nmse_ylim=nmse_ylim,
        snrgain_ylim=snrgain_ylim,
        savepath=args.out,
        dpi=args.dpi,
    )
    
    plot_nmse_and_snr_gain_vs_snr(
        clean_waveforms=clean,
        noisy_waveforms=noisy,
        denoised_waveforms=denoised,
        standard_waveforms=standard_waveforms,
        snr=snr_external,
        cfg=cfg
    )


if __name__ == "__main__":
    main()