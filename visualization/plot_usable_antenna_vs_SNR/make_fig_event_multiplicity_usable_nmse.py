#!/usr/bin/env python3
"""
make_fig_event_multiplicity_usable_nmse.py

Option (ii): "Multiplicity gain per event" figure using ROI-NMSE usability.

Key lessons incorporated (from prior scripts):
  (A) ROI-NMSE in time domain on +/- ROI around CLEAN peak (default 150 ns),
      optionally after 50-200 MHz bandpass (still time-domain).
  (B) SNR axis uses ROI-peak SNR anchored on the CLEAN peak time:
        A_noisy = max(|hilbert(noisy)|) within +/- snr_peak_half_width around CLEAN peak time.
      sigma_env from robust MAD of noisy envelope away from the CLEAN peak time.
      This avoids extreme-value inflation from scanning the full trace.
  (C) Truth-conditioned clean-power gate (per channel) to avoid ill-conditioned NMSE when
      true signal energy is tiny in that polarization.

Inputs (aligned):
  clean_waveforms:    (N_traces, 3, T)
  noisy_waveforms:    (N_traces, 3, T)
  denoised_waveforms: (N_traces, 3, T)   [ML]
  standard_waveforms: (N_traces, 3, T)   [baseline/standard]
  event_ids:          (N_traces,) integer event identifier per trace (antenna)

Outputs:
  A 2x3 panel figure:
    Row 1: median N_usable per event vs event-level SNR (per channel), ML vs Standard,
           with a population spread band (q_lo..q_hi).
    Row 2: median ΔN_usable per event vs event-level SNR, with spread band.

Usability definition (per trace, per channel):
  trace is "candidate" if base_mask = trigger_mask & extra_base_mask & power_mask
  usable(method) = base_mask & (NMSE_method <= NMSE_max[ch])

NMSE_max calibration:
  per channel, set NMSE_max[ch] as the q=calib_pass_rate quantile of ML NMSE among
  candidate traces with SNR >= snr_calib_min.

Usage:
- Command line: python make_fig_nmse_snr_gain_vs_snr.py --model-path ... --metrics-json ... --config-json ... --event-ids-path ...
- Or import and call plot_event_multiplicity_usable_nmse(clean, noisy, denoised, standard_waveforms, event_ids, cfg=...).

Dependencies: numpy, matplotlib, scipy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Literal

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
class EventMultiplicityConfig:
    # Sampling
    # Sampling interval of the simulated traces, in nanoseconds.
    #
    # CORRECTED: this defaulted to 2.0 ns, which does not match the 0.5 ns
    # sampling stated in the paper. dt_ns sets the sampling rate used to design
    # the band-pass (fs = 1/dt) and converts every *_ns window into samples, so a
    # wrong value silently shifts both the filtered band and the ROI width.
    dt_ns: float = 0.5
    eps: float = 1e-12

    # ROI for NMSE around CLEAN peak
    roi_half_width_ns: float = 150.0

    # --- SNR axis (paper-style, but ROI-peak anchored on CLEAN peak) ---
    snr_peak_half_width_ns: float = 150.0
    snr_exclude_half_width_ns: float = 150.0
    mad_scale_gaussian: float = 1.4826022185056  # normal-consistent MAD constant

    # Apply same bandpass before computing SNR envelope? (set True if your paper SNR is in-band)
    apply_bandpass_to_snr: bool = True

    # Trigger-like selection (optional)
    apply_trigger: bool = False
    trigger_k_sigma: float = 1.0

    # In-band fidelity (still time-domain, but after bandpass)
    apply_bandpass: bool = True
    f_lo_hz: float = 50e6
    f_hi_hz: float = 200e6
    butter_order: int = 4

    # Truth-conditioned gate: require real CLEAN signal energy in that channel
    apply_clean_power_gate: bool = True
    clean_power_gate_quantile: float = 0.10   # floor = q-quantile of *positive* clean ROI powers
    clean_power_gate_min_count: int = 200
    clean_power_gate_abs_floor: float = 0.0

    # NMSE threshold calibration (on ML outputs)
    snr_calib_min: float = 6.0
    calib_pass_rate: float = 0.95

    # Event-level SNR statistic (per channel) computed across candidate traces in the event
    event_snr_stat: Literal["median", "mean", "q80"] = "median"

    # Binning in event-level SNR
    snr_bins: Tuple[float, ...] = (1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14)

    # Spread band across events within each SNR bin (population spread, NOT CI)
    band_quantiles: Tuple[float, float] = (0.16, 0.84)

    # Plot styling
    channel_names: Tuple[str, str, str] = ("X Channel", "Y Channel", "Z Channel")
    labels: Tuple[str, str] = ("Standard", "ML")
    colors: Tuple[str, str] = ("C1", "k")
    band_alpha: float = 0.18

    # Axes
    ylim_counts: Optional[Tuple[float, float]] = None
    ylim_delta: Optional[Tuple[float, float]] = None

    # Output
    savepath: Optional[str] = None
    dpi: int = 220

def run_in_notebook(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    standard: np.ndarray,
    event_ids: np.ndarray,
    savepath: str = None,
    **config_overrides,
) -> Dict[str, np.ndarray]:
    """
    Convenience wrapper for Jupyter notebook usage.
    
    Args:
        clean: Clean waveforms (N, 3, T)
        noisy: Noisy waveforms (N, 3, T)
        denoised: ML denoised waveforms (N, 3, T)
        standard: Standard denoised waveforms (N, 3, T)
        event_ids: Event IDs per trace (N,)
        savepath: Optional path to save figure
        **config_overrides: Override any EventMultiplicityConfig field
        
    Returns:
        Dictionary with diagnostic arrays
    """
    cfg_kwargs = {"savepath": savepath}
    cfg_kwargs.update(config_overrides)
    cfg = EventMultiplicityConfig(**cfg_kwargs)
    
    return plot_event_multiplicity_usable_nmse(
        clean_waveforms=clean,
        noisy_waveforms=noisy,
        denoised_waveforms=denoised,
        standard_waveforms=standard,
        event_ids=event_ids,
        cfg=cfg,
    )
# -----------------------------
# Validation
# -----------------------------
def _validate_shapes(
    clean: np.ndarray,
    noisy: np.ndarray,
    ml: np.ndarray,
    std: np.ndarray,
    event_ids: np.ndarray,
    extra_base_mask: Optional[np.ndarray],
) -> None:
    for name, arr in [("clean", clean), ("noisy", noisy), ("denoised_waveforms", ml), ("standard_waveforms", std)]:
        if not isinstance(arr, np.ndarray) or arr.ndim != 3:
            raise ValueError(f"{name} must be numpy array with shape (N,3,T). Got {None if arr is None else arr.shape}")
    if clean.shape != noisy.shape or clean.shape != ml.shape or clean.shape != std.shape:
        raise ValueError(f"clean/noisy/ml/std shapes must match. Got {clean.shape}, {noisy.shape}, {ml.shape}, {std.shape}")
    if clean.shape[1] != 3:
        raise ValueError(f"Expected 3 channels. Got {clean.shape[1]}")

    event_ids = np.asarray(event_ids)
    if event_ids.ndim != 1 or event_ids.shape[0] != clean.shape[0]:
        raise ValueError(f"event_ids must have shape (N,). Got {event_ids.shape}, expected ({clean.shape[0]},)")

    if extra_base_mask is not None:
        m = np.asarray(extra_base_mask, dtype=bool)
        if m.shape != (clean.shape[0], 3):
            raise ValueError(f"extra_base_mask must have shape (N,3). Got {m.shape}")


# -----------------------------
# Signal helpers
# -----------------------------
def _analytic_envelope(x: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(x, axis=-1))


def _mad_sigma(x: np.ndarray, scale: float) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(scale * mad)


def _maybe_bandpass(x: np.ndarray, cfg: EventMultiplicityConfig) -> np.ndarray:
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
    return sosfiltfilt(sos, x, axis=-1)  # zero-phase forward-backward filtering


def _roi_indices_centered_on_clean_peak(clean_1d: np.ndarray, cfg: EventMultiplicityConfig) -> Tuple[int, int, int]:
    T = clean_1d.size
    env = np.abs(hilbert(clean_1d))
    k = int(np.argmax(env))
    half = max(1, int(round(cfg.roi_half_width_ns / cfg.dt_ns)))
    lo = max(0, k - half)
    hi = min(T, k + half + 1)
    return k, lo, hi


# -----------------------------
# SNR (paper style), but ROI-peak around CLEAN peak
# -----------------------------
def compute_snr_roi_peak_anchored_on_clean(
    clean: np.ndarray,
    noisy: np.ndarray,
    cfg: EventMultiplicityConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per trace, per channel:
      - find CLEAN peak time k_clean from CLEAN envelope
      - A_noisy = max(noisy envelope) within +/- snr_peak_half_width_ns around k_clean
      - sigma_env = MAD(noisy envelope) excluding +/- snr_exclude_half_width_ns around k_clean
      - SNR = A_noisy / sigma_env
    """
    N, C, T = noisy.shape
    snr = np.full((N, C), np.nan, dtype=np.float64)
    A = np.full((N, C), np.nan, dtype=np.float64)
    sigma = np.full((N, C), np.nan, dtype=np.float64)

    # optionally bandpass for the SNR definition
    if cfg.apply_bandpass_to_snr:
        clean_s = _maybe_bandpass(clean.astype(np.float64), cfg)
        noisy_s = _maybe_bandpass(noisy.astype(np.float64), cfg)
    else:
        clean_s = clean.astype(np.float64)
        noisy_s = noisy.astype(np.float64)

    peak_hw = max(1, int(round(cfg.snr_peak_half_width_ns / cfg.dt_ns)))
    excl_hw = max(1, int(round(cfg.snr_exclude_half_width_ns / cfg.dt_ns)))

    for ch in range(C):
        env_noisy = _analytic_envelope(noisy_s[:, ch, :])  # (N,T)
        for i in range(N):
            k_clean, _, _ = _roi_indices_centered_on_clean_peak(clean_s[i, ch, :], cfg)

            lo_pk = max(0, k_clean - peak_hw)
            hi_pk = min(T, k_clean + peak_hw + 1)
            A[i, ch] = float(np.max(env_noisy[i, lo_pk:hi_pk]))

            lo_ex = max(0, k_clean - excl_hw)
            hi_ex = min(T, k_clean + excl_hw + 1)
            mask = np.ones(T, dtype=bool)
            mask[lo_ex:hi_ex] = False
            env_noise = env_noisy[i, mask]
            if env_noise.size < 16:
                env_noise = env_noisy[i, :]

            s = _mad_sigma(env_noise, cfg.mad_scale_gaussian)
            s = max(s, cfg.eps)
            sigma[i, ch] = s
            snr[i, ch] = A[i, ch] / s

    return snr, A, sigma


# -----------------------------
# ROI-NMSE and CLEAN ROI power
# -----------------------------
def compute_roi_nmse_and_cleanpower(
    clean: np.ndarray,
    rec: np.ndarray,
    cfg: EventMultiplicityConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ROI is centered on CLEAN peak for each trace+channel.
    NMSE = ||rec-clean||^2 / ||clean||^2   (within ROI; after optional bandpass)
    Also returns CLEAN ROI power = ||clean||^2 (within ROI).
    """
    N, C, _ = clean.shape
    clean_f = _maybe_bandpass(clean.astype(np.float64), cfg)
    rec_f = _maybe_bandpass(rec.astype(np.float64), cfg)

    nmse = np.full((N, C), np.nan, dtype=np.float64)
    cleanpow = np.full((N, C), np.nan, dtype=np.float64)

    for ch in range(C):
        for i in range(N):
            _, lo, hi = _roi_indices_centered_on_clean_peak(clean_f[i, ch, :], cfg)
            x = clean_f[i, ch, lo:hi]
            y = rec_f[i, ch, lo:hi]
            sig = float(np.sum(x * x))
            err = float(np.sum((y - x) * (y - x)))
            cleanpow[i, ch] = sig
            nmse[i, ch] = max(err, cfg.eps) / max(sig, cfg.eps)

    return nmse, cleanpow


def calibrate_nmse_max(
    nmse_ml: np.ndarray,
    snr: np.ndarray,
    base_mask: np.ndarray,
    cfg: EventMultiplicityConfig,
) -> np.ndarray:
    """
    Per channel: NMSE_max[ch] = quantile(calib_pass_rate) of ML NMSE among candidate traces
    with SNR >= snr_calib_min.
    """
    nmse_max = np.full((3,), np.nan, dtype=np.float64)
    q = float(cfg.calib_pass_rate)

    for ch in range(3):
        m = (
            base_mask[:, ch]
            & np.isfinite(nmse_ml[:, ch])
            & np.isfinite(snr[:, ch])
            & (snr[:, ch] >= cfg.snr_calib_min)
        )
        vals = nmse_ml[m, ch]
        if vals.size < 20:
            raise RuntimeError(
                f"Not enough high-SNR candidate traces to calibrate NMSE_max in channel {ch}. "
                f"Found {vals.size}. Try lowering snr_calib_min, disabling trigger, or relaxing gates."
            )
        nmse_max[ch] = np.quantile(vals, q)

    return nmse_max


# -----------------------------
# Event-level aggregation
# -----------------------------
def _event_snr_from_trace_snrs(trace_snrs: np.ndarray, cfg: EventMultiplicityConfig) -> float:
    """trace_snrs: 1D finite array for an event in one channel"""
    if trace_snrs.size == 0:
        return np.nan
    if cfg.event_snr_stat == "median":
        return float(np.median(trace_snrs))
    if cfg.event_snr_stat == "mean":
        return float(np.mean(trace_snrs))
    if cfg.event_snr_stat == "q80":
        return float(np.quantile(trace_snrs, 0.80))
    raise ValueError(f"Unknown event_snr_stat={cfg.event_snr_stat}")


def compute_event_level_arrays(
    event_ids: np.ndarray,
    snr: np.ndarray,
    base_mask: np.ndarray,
    usable_ml: np.ndarray,
    usable_std: np.ndarray,
    cfg: EventMultiplicityConfig,
) -> Dict[str, np.ndarray]:
    """
    Returns per-event arrays (per channel):
      event_snr[ch, e]
      N_candidate[ch, e]
      N_usable_ml[ch, e]
      N_usable_std[ch, e]
    """
    event_ids = np.asarray(event_ids)
    uniq = np.unique(event_ids)
    E = uniq.size
    C = 3

    event_snr = np.full((C, E), np.nan, dtype=np.float64)
    n_cand = np.zeros((C, E), dtype=np.int64)
    n_ml = np.zeros((C, E), dtype=np.int64)
    n_std = np.zeros((C, E), dtype=np.int64)

    # map event id -> indices
    for ei, eid in enumerate(uniq):
        idx = np.where(event_ids == eid)[0]
        for ch in range(C):
            cand = base_mask[idx, ch] & np.isfinite(snr[idx, ch])
            n_cand[ch, ei] = int(np.sum(cand))

            # event-level SNR from candidate traces only (method-independent)
            event_snr[ch, ei] = _event_snr_from_trace_snrs(snr[idx, ch][cand], cfg)

            # usable counts (subset of candidates)
            n_ml[ch, ei] = int(np.sum(usable_ml[idx, ch] & base_mask[idx, ch]))
            n_std[ch, ei] = int(np.sum(usable_std[idx, ch] & base_mask[idx, ch]))

    return {
        "event_ids_unique": uniq,
        "event_snr": event_snr,
        "n_candidate": n_cand,
        "n_usable_ml": n_ml,
        "n_usable_std": n_std,
    }


def bin_and_summarize_events(
    x_event: np.ndarray,  # (E,)
    y_event: np.ndarray,  # (E,)
    edges: np.ndarray,
    q_lo: float,
    q_hi: float,
) -> Dict[str, np.ndarray]:
    """
    For each bin, compute median and quantile band of y among events in bin.
    """
    nb = edges.size - 1
    xc = np.full(nb, np.nan, dtype=np.float64)
    med = np.full(nb, np.nan, dtype=np.float64)
    lo = np.full(nb, np.nan, dtype=np.float64)
    hi = np.full(nb, np.nan, dtype=np.float64)
    n = np.zeros(nb, dtype=np.int64)

    for b in range(nb):
        m = np.isfinite(x_event) & (x_event >= edges[b]) & (x_event < edges[b + 1]) & np.isfinite(y_event)
        vals = y_event[m]
        n[b] = int(vals.size)
        if vals.size == 0:
            continue
        xc[b] = 0.5 * (edges[b] + edges[b + 1])
        med[b] = np.median(vals)
        lo[b] = np.quantile(vals, q_lo)
        hi[b] = np.quantile(vals, q_hi)

    return {"x": xc, "median": med, "qlo": lo, "qhi": hi, "n": n}


# -----------------------------
# Plotting
# -----------------------------
def plot_event_multiplicity_usable_nmse(
    clean_waveforms: np.ndarray,
    noisy_waveforms: np.ndarray,
    denoised_waveforms: np.ndarray,
    standard_waveforms: np.ndarray,
    event_ids: np.ndarray,
    extra_base_mask: Optional[np.ndarray] = None,  # (N,3) boolean, e.g. timing cut, amplitude cut, etc.
    cfg: EventMultiplicityConfig = EventMultiplicityConfig(),
) -> Dict[str, np.ndarray]:
    """
    Main entrypoint: produces the 2x3 multiplicity-gain figure.
    """
    _validate_shapes(clean_waveforms, noisy_waveforms, denoised_waveforms, standard_waveforms, event_ids, extra_base_mask)
    N, C, _ = clean_waveforms.shape

    # 1) SNR axis per trace (ROI-peak anchored on CLEAN peak)
    snr, A_noisy, sigma_env = compute_snr_roi_peak_anchored_on_clean(clean_waveforms, noisy_waveforms, cfg)

    # 2) Trigger mask (optional)
    trigger_mask = np.ones((N, C), dtype=bool)
    if cfg.apply_trigger:
        trigger_mask = A_noisy >= (cfg.trigger_k_sigma * sigma_env)

    # 3) Extra base mask (user-supplied)
    if extra_base_mask is None:
        extra_base_mask = np.ones((N, C), dtype=bool)
    else:
        extra_base_mask = np.asarray(extra_base_mask, dtype=bool)

    # 4) ROI-NMSE for ML and standard; CLEAN ROI power from CLEAN (method-independent)
    nmse_ml, cleanpow = compute_roi_nmse_and_cleanpower(clean_waveforms, denoised_waveforms, cfg)
    nmse_std, _ = compute_roi_nmse_and_cleanpower(clean_waveforms, standard_waveforms, cfg)

    # 5) Truth-conditioned clean-power gate (per channel)
    power_mask = np.ones((N, C), dtype=bool)
    clean_power_floor = np.full((C,), np.nan, dtype=np.float64)

    if cfg.apply_clean_power_gate:
        for ch in range(C):
            # Condition on trigger if enabled, but floor defined from CLEAN ROI power only.
            base = trigger_mask[:, ch] if cfg.apply_trigger else np.ones(N, dtype=bool)
            vals = cleanpow[base & np.isfinite(cleanpow[:, ch]) & (cleanpow[:, ch] > 0.0), ch]
            if vals.size < cfg.clean_power_gate_min_count:
                raise RuntimeError(
                    f"Not enough positive CLEAN ROI-power samples to set power gate in channel {ch} "
                    f"(found {vals.size}, need >= {cfg.clean_power_gate_min_count}). "
                    f"Try lowering clean_power_gate_min_count, disabling trigger, or disabling the gate."
                )
            floor = float(np.quantile(vals, cfg.clean_power_gate_quantile))
            floor = max(floor, float(cfg.clean_power_gate_abs_floor))
            clean_power_floor[ch] = floor
            power_mask[:, ch] = np.isfinite(cleanpow[:, ch]) & (cleanpow[:, ch] >= floor)

    # 6) Candidate mask applied to ALL methods for fair comparison
    base_mask = trigger_mask & extra_base_mask & power_mask

    # 7) Calibrate NMSE_max from high-SNR ML outputs
    nmse_max = calibrate_nmse_max(nmse_ml, snr, base_mask, cfg)

    # 8) Usable flags per trace+channel
    usable_ml = base_mask & np.isfinite(nmse_ml) & (nmse_ml <= nmse_max.reshape(1, -1))
    usable_std = base_mask & np.isfinite(nmse_std) & (nmse_std <= nmse_max.reshape(1, -1))

    # 9) Aggregate to event-level arrays
    ev = compute_event_level_arrays(event_ids, snr, base_mask, usable_ml, usable_std, cfg)
    edges = np.array(cfg.snr_bins, dtype=np.float64)
    q_lo, q_hi = cfg.band_quantiles

    # 10) Plot 2x3: counts + delta
    fig, axes = plt.subplots(1, 3, figsize=(30, 10), sharex=True)
    fontsize = 40
    for ch in range(3):
        x = ev["event_snr"][ch, :]              # (E,)
        y_std = ev["n_usable_std"][ch, :].astype(np.float64)
        y_ml = ev["n_usable_ml"][ch, :].astype(np.float64)
        y_d = y_ml - y_std

        s_std = bin_and_summarize_events(x, y_std, edges, q_lo, q_hi)
        s_ml = bin_and_summarize_events(x, y_ml, edges, q_lo, q_hi)
        s_d = bin_and_summarize_events(x, y_d, edges, q_lo, q_hi)

        # --- Top row: N_usable ---
        # ax = axes[0, ch]
        # ax.set_title(cfg.channel_names[ch],fontsize=fontsize)

        # for s, lab, col in [
        #     (s_std, cfg.labels[0], cfg.colors[0]),
        #     (s_ml, cfg.labels[1], cfg.colors[1]),
        # ]:
        #     m = np.isfinite(s["x"]) & np.isfinite(s["median"]) & np.isfinite(s["qlo"]) & np.isfinite(s["qhi"])
        #     ax.fill_between(s["x"][m], s["qlo"][m], s["qhi"][m], color=col, alpha=cfg.band_alpha, linewidth=0)
        #     ax.plot(s["x"][m], s["median"][m], marker="o", linewidth=2.4, color=col, label=lab)

        # if cfg.ylim_counts is not None:
        #     ax.set_ylim(*cfg.ylim_counts)
        # ax.grid(True, alpha=0.25)
        # ax.tick_params(axis='both', which='major', labelsize=fontsize)
        # if ch == 0:
        #     ax.set_ylabel(r"$N_{\rm usable}$",fontsize=fontsize)

        # annotate key thresholds (useful for paper reproducibility)
        # ax.text(
        #     0.02, 0.06,
        #     f"NMSE_max={nmse_max[ch]:.3g}\nP_clean floor={clean_power_floor[ch]:.3g}",
        #     transform=ax.transAxes, fontsize=9.5, alpha=0.85, va="bottom"
        # )

        # --- Bottom row: ΔN_usable ---
        ax2 = axes[ch]
        channel_label = cfg.channel_names[ch].split()[0] if " " in cfg.channel_names[ch] else cfg.channel_names[ch][0]
        ax2.set_title(channel_label, fontsize=fontsize)
        m2 = np.isfinite(s_d["x"]) & np.isfinite(s_d["median"]) & np.isfinite(s_d["qlo"]) & np.isfinite(s_d["qhi"])
        ax2.axhline(0.0, color="k", lw=1.0, alpha=0.7)
        ax2.fill_between(s_d["x"][m2], s_d["qlo"][m2], s_d["qhi"][m2], color="tab:blue", alpha=0.15, linewidth=0)
        ax2.plot(s_d["x"][m2], s_d["median"][m2], marker="o", linewidth=2.2, color="tab:blue")

        if cfg.ylim_delta is not None:
            ax2.set_ylim(*cfg.ylim_delta)
        ax2.grid(True, alpha=0.25)
        ax2.set_xlabel("SNR",fontsize=fontsize)
        ax2.tick_params(axis='both', which='major', labelsize=fontsize)
        if ch == 0:
            ax2.set_ylabel(r"$\Delta N_{\rm usable}$ (ML $-$ Std)",fontsize=fontsize)

    axes[1].legend(loc="upper left", frameon=False, fontsize=fontsize)
    fig.tight_layout()

    if cfg.savepath is not None:
        fig.savefig(cfg.savepath, dpi=cfg.dpi, bbox_inches="tight")
        print(f"Saved: {cfg.savepath}")

    plt.show()

    return {
        "snr_trace": snr,
        "A_noisy": A_noisy,
        "sigma_env": sigma_env,
        "trigger_mask": trigger_mask,
        "power_mask": power_mask,
        "base_mask": base_mask,
        "nmse_ml": nmse_ml,
        "nmse_std": nmse_std,
        "nmse_max": nmse_max,
        "clean_power_floor": clean_power_floor,
        **ev,
    }


# -----------------------------
# Main entry point
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Event multiplicity gain per event using ROI-NMSE usability")
    ap.add_argument("--npz", default=None, help="NPZ with keys clean, noisy, denoised, standard, event_ids")
    
    if HAS_ML_UTILS:
        add_model_arguments(ap)
    
    ap.add_argument("--standard-npz", default=None, help="NPZ file with 'standard' key for standard denoiser waveforms")
    ap.add_argument("--event-ids-path", default=None, help="Path to .npy file containing event_ids array (N,) or NPZ key name")
    ap.add_argument("--event-ids-npz-key", default="event_ids", help="Key name for event_ids in NPZ file (default: event_ids)")
    
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
    
    # NMSE threshold calibration
    ap.add_argument("--snr-calib-min", type=float, default=6.0,
                    help="Minimum SNR for calibration (default: 6.0)")
    ap.add_argument("--calib-pass-rate", type=float, default=0.95,
                    help="Calibration pass rate (default: 0.95)")
    
    # Event-level SNR statistic
    ap.add_argument("--event-snr-stat", choices=["median", "mean", "q80"], default="median",
                    help="Event-level SNR statistic (default: median)")
    
    # SNR bins
    ap.add_argument("--snr-bins", type=str, default="1.5,2,3,4,5,6,7,8,9,10,12,14",
                    help="Comma-separated SNR bin edges (default: 1.5,2,3,4,5,6,7,8,9,10,12,14)")
    
    # Band quantiles
    ap.add_argument("--band-quantiles", type=str, default="0.16,0.84",
                    help="Band quantiles: low,high (default: 0.16,0.84)")
    
    # Styling
    ap.add_argument("--band-alpha", type=float, default=0.18,
                    help="Alpha for bands (default: 0.18)")
    ap.add_argument("--channel-names", default="X Channel,Y Channel,Z Channel",
                    help="Comma-separated channel labels (default: X Channel,Y Channel,Z Channel)")
    ap.add_argument("--standard-label", default="Standard", help="Label for standard denoiser traces")
    ap.add_argument("--ml-label", default="ML", help="Label for ML denoiser traces")
    
    # Axes limits
    ap.add_argument("--ylim-counts", type=str, default=None,
                    help="Y-axis limits for counts: low,high (default: auto)")
    ap.add_argument("--ylim-delta", type=str, default=None,
                    help="Y-axis limits for delta: low,high (default: auto)")
    
    # Output
    ap.add_argument("--out", default="fig_event_multiplicity_usable_nmse.pdf", help="Output figure path.")
    ap.add_argument("--dpi", type=int, default=220, help="Figure DPI (default: 220)")
    
    args = ap.parse_args()
    
    # Parse channel names
    channel_names = tuple(name.strip() for name in args.channel_names.split(","))
    if len(channel_names) != 3:
        raise ValueError(f"Expected 3 channel names, got {len(channel_names)}")
    
    # Parse labels
    labels = (args.standard_label, args.ml_label)
    
    # Parse colors (default from config)
    colors = ("C1", "k")
    
    # Parse SNR bins
    snr_bins_str = args.snr_bins.split(",")
    if len(snr_bins_str) < 2:
        raise ValueError(f"--snr-bins must have at least 2 comma-separated values, got {len(snr_bins_str)}")
    snr_bins = tuple(float(x.strip()) for x in snr_bins_str)
    
    # Parse band quantiles
    quantiles_str = args.band_quantiles.split(",")
    if len(quantiles_str) != 2:
        raise ValueError(f"--band-quantiles must have 2 comma-separated values, got {len(quantiles_str)}")
    band_quantiles = (float(quantiles_str[0]), float(quantiles_str[1]))
    
    # Parse ylims
    ylim_counts = None
    if args.ylim_counts is not None:
        ylim_str = args.ylim_counts.split(",")
        if len(ylim_str) != 2:
            raise ValueError(f"--ylim-counts must have 2 comma-separated values, got {len(ylim_str)}")
        ylim_counts = (float(ylim_str[0]), float(ylim_str[1]))
    
    ylim_delta = None
    if args.ylim_delta is not None:
        ylim_str = args.ylim_delta.split(",")
        if len(ylim_str) != 2:
            raise ValueError(f"--ylim-delta must have 2 comma-separated values, got {len(ylim_str)}")
        ylim_delta = (float(ylim_str[0]), float(ylim_str[1]))
    
    # Load data
    standard_waveforms = None
    event_ids = None
    indices_used = None  # Track indices used in ML mode for event_ids filtering
    
    if args.npz is not None:
        d = np.load(args.npz, allow_pickle=False)
        clean = d["clean"]
        noisy = d["noisy"]
        denoised = d["denoised"]
        # Check for standard waveforms in main NPZ
        if "standard" in d:
            standard_waveforms = d["standard"]
        # Load event_ids from NPZ
        if args.event_ids_npz_key in d:
            event_ids = d[args.event_ids_npz_key]
        elif "event_ids" in d:
            event_ids = d["event_ids"]
    elif HAS_ML_UTILS and check_model_args(args):
        if not all([args.model_path, args.metrics_json, args.config_json]):
            raise ValueError("ML mode requires --model-path, --metrics-json, --config-json")
        
        # Import here to avoid circular dependency
        from raytune_lib_sept25.raytune_training_function import split_indices
        from raytune_lib_sept25.produce_noise_and_noiseless_data import produce_noise_and_noiseless_data
        
        # Get indices used (same logic as load_data_and_run_inference)
        _, clean_signals = produce_noise_and_noiseless_data(args.data_path)
        total_samples = clean_signals.shape[1]
        train_indices, valid_indices, test_indices = split_indices(
            total_samples, train_frac=0.8, valid_frac=0.1
        )
        indices_used = test_indices  # load_data_and_run_inference uses test_only=True by default
        if getattr(args, 'max_samples', None) is not None and getattr(args, 'max_samples', None) < len(indices_used):
            indices_used = indices_used[:getattr(args, 'max_samples', None)]
        
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
    
    # Load event_ids if provided separately or try to auto-load from data_path
    if event_ids is None:
        if args.event_ids_path is not None:
            # Load from explicit path
            if args.event_ids_path.endswith('.npy'):
                event_ids_full = np.load(args.event_ids_path, allow_pickle=False)
            elif args.event_ids_path.endswith('.npz'):
                d_ev = np.load(args.event_ids_path, allow_pickle=False)
                if args.event_ids_npz_key in d_ev:
                    event_ids_full = d_ev[args.event_ids_npz_key]
                elif "event_ids" in d_ev:
                    event_ids_full = d_ev["event_ids"]
                else:
                    raise ValueError(f"event_ids NPZ must contain '{args.event_ids_npz_key}' or 'event_ids' key. Found keys: {list(d_ev.keys())}")
            else:
                raise ValueError(f"event_ids_path must be .npy or .npz file, got {args.event_ids_path}")
            
            # If using ML mode and have indices_used, filter event_ids
            if indices_used is not None:
                event_ids = event_ids_full[indices_used]
            else:
                event_ids = event_ids_full
        elif HAS_ML_UTILS and check_model_args(args) and indices_used is not None:
            # Try to auto-load from data_path
            import os
            event_number_list_path = os.path.join(args.data_path, "event_number_list.npy")
            if os.path.exists(event_number_list_path):
                event_ids_full = np.load(event_number_list_path, allow_pickle=False)
                event_ids = event_ids_full[indices_used]
            else:
                raise ValueError(
                    f"event_ids is required for ML mode. "
                    f"Provide via --event-ids-path, or place event_number_list.npy at {event_number_list_path}"
                )
    
    # Validate event_ids
    if event_ids is None:
        raise ValueError("event_ids is required. Provide via --npz (with 'event_ids' key), --event-ids-path, or ensure event_number_list.npy exists in data_path for ML mode")
    
    event_ids = np.asarray(event_ids, dtype=np.int64)
    if event_ids.ndim != 1 or event_ids.shape[0] != clean.shape[0]:
        raise ValueError(f"event_ids must have shape (N,). Got {event_ids.shape}, expected ({clean.shape[0]},)")
    
    # If standard_waveforms not provided, use noisy_waveforms as baseline
    if standard_waveforms is None:
        print("Warning: standard_waveforms not provided, using noisy_waveforms as baseline")
        standard_waveforms = noisy
    
    cfg = EventMultiplicityConfig(
        dt_ns=args.dt_ns,
        eps=args.eps,
        roi_half_width_ns=args.roi_half_width_ns,
        snr_peak_half_width_ns=args.snr_peak_half_width_ns,
        snr_exclude_half_width_ns=args.snr_exclude_half_width_ns,
        mad_scale_gaussian=args.mad_scale_gaussian,
        apply_bandpass_to_snr=args.apply_bandpass_to_snr,
        apply_trigger=args.apply_trigger,
        trigger_k_sigma=args.trigger_k_sigma,
        apply_bandpass=args.apply_bandpass,
        f_lo_hz=args.f_lo_hz,
        f_hi_hz=args.f_hi_hz,
        butter_order=args.butter_order,
        apply_clean_power_gate=args.apply_clean_power_gate,
        clean_power_gate_quantile=args.clean_power_gate_quantile,
        clean_power_gate_min_count=args.clean_power_gate_min_count,
        clean_power_gate_abs_floor=args.clean_power_gate_abs_floor,
        snr_calib_min=args.snr_calib_min,
        calib_pass_rate=args.calib_pass_rate,
        event_snr_stat=args.event_snr_stat,
        snr_bins=snr_bins,
        band_quantiles=band_quantiles,
        channel_names=channel_names,
        labels=labels,
        colors=colors,
        band_alpha=args.band_alpha,
        ylim_counts=ylim_counts,
        ylim_delta=ylim_delta,
        savepath=args.out,
        dpi=args.dpi,
    )
    
    plot_event_multiplicity_usable_nmse(
        clean_waveforms=clean,
        noisy_waveforms=noisy,
        denoised_waveforms=denoised,
        standard_waveforms=standard_waveforms,
        event_ids=event_ids,
        extra_base_mask=None,
        cfg=cfg
    )


if __name__ == "__main__":
    main()