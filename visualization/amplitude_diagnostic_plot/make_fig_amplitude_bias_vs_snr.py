#!/usr/bin/env python3
"""
make_fig_amplitude_bias_vs_snr.py

Amplitude-bias diagnostic for referee question 5.

For each trace and polarization channel (X, Y, Z), and for each method
(noisy/baseline, denoised), compute the fractional peak-amplitude residual

    delta_A = A_rec / A_true - 1,

where

    A_true = max_t |Hilbert(clean)(t)|,
    A_rec  = max_t |Hilbert(rec)(t)|,

with rec equal to either the noisy trace or the denoised trace.

This script is intended to answer the referee's amplitude-bias concern without
introducing a new fluence observable. It reports the SNR-dependent bias and
scatter of peak-amplitude recovery.

Important SNR convention
------------------------
The SNR axis uses the paper definition used throughout the analysis:

    SNR = max(clean) / std(noisy),

with the standard deviation evaluated over the FULL trace (no off-pulse
exclusion, no band-limiting, raw signed clean maximum), computed per channel.
The identical SNR array is used for the noisy and the denoised curves.

Selection
---------
Traces are selected by input SNR: min_snr < SNR < max_snr (default 1 < SNR <
1000), applied identically to both curves through shared_selection_mask(). This
is NOT the noisy-input trigger used by the timing analysis, so this figure does
not share a sample with it. The plotted range is set by snr_edges (default
1 <= SNR < 10).

The --noise-estimator envelope_std / envelope_mad options select an alternative
off-pulse envelope noise scale; they are non-default cross-checks only and are
not the paper definition.

Data source
-----------
The script uses the shared evaluation pipeline

    visualization/common_ml_utils.py::load_data_and_run_inference

with test_only=True, so it evaluates the held-out test set and the trained
time+frequency DualBranchAutoencoder.

Outputs
-------
  - amplitude_bias_table.csv
  - amplitude_bias_vs_snr.pdf
  - amplitude_bias_per_trace.npz

Quantities reported per bin
---------------------------
  channel, method, snr_bin_low, snr_bin_high, snr_bin_center, n_traces,
  median_delta_A, q16_delta_A, q84_delta_A,
  sigma68_delta_A = 0.5 * (q84 - q16),
  mean_delta_A, std_delta_A, median_abs_delta_A

Recommended headline quantities
-------------------------------
  - median_delta_A: amplitude bias
  - sigma68_delta_A: robust scatter
  - median_abs_delta_A: typical absolute fractional amplitude error

Example
-------
python make_fig_amplitude_bias_vs_snr.py \
    --model-path ../../results/multi_l1_example/best_model.pth \
    --metrics-json ../../results/multi_l1_example/best_trial_metrics.json \
    --config-json ../../results/multi_l1_example/best_trial_config.json \
    --data-path /sps/grand/blevy/sims/sims_for_denoising_sept2025 \
    --device cuda \
    --out-dir .
"""

from __future__ import annotations

import os
import sys
import csv
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert


# ---------------------------------------------------------------------
# Make the project importable
# ---------------------------------------------------------------------
def find_project_root(start: Path) -> Path:
    """
    Find the ML_denoising project root by walking upward from this file.

    This is safer than assuming a fixed parents[k] depth, because Sam may place
    this script either in visualization/, visualization/referee_plots/, or a
    local scripts/ directory.
    """
    start = start.resolve()
    candidates = [start.parent] + list(start.parents)

    for p in candidates:
        if (p / "visualization").is_dir() and (p / "training").is_dir():
            return p

    raise RuntimeError(
        "Could not find project root. Place this script inside the ML_denoising "
        "repository, or update find_project_root() manually."
    )


ROOT_DIR = find_project_root(Path(__file__))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from visualization.common_ml_utils import load_data_and_run_inference  # noqa: E402
# SNR is the paper definition max(clean)/std(noisy) over the full trace, computed inline.


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class AmpBiasConfig:
    min_snr: float = 1.0
    max_snr: float = 1.0e3

    # Default bins: width-1 bins from 1 to 10, matching the broad style of
    # the current SNR-binned paper plots.
    snr_edges: Tuple[float, ...] = tuple(np.linspace(1.0, 10.0, 10).tolist())

    # Exclude +/- exclude_radius samples around the clean Hilbert-envelope peak
    # when estimating the off-pulse noise scale.
    exclude_radius: int = 64

    # Options:
    #   "raw_rms"      : std(noisy waveform off-pulse), recommended default.
    #   "envelope_std" : std(|Hilbert(noisy)| off-pulse).
    #   "envelope_mad" : robust MAD scale of |Hilbert(noisy)| off-pulse.
    noise_estimator: str = "raw_rms"

    min_count_plot: int = 20

    channel_names: Tuple[str, str, str] = ("X", "Y", "Z")
    color_noisy: str = "tab:red"
    color_denoised: str = "tab:blue"
    band_alpha: float = 0.22
    fontsize: int = 22
    ylim: Tuple[float, float] = (-1.0, 2.0)
    dpi: int = 220


# ---------------------------------------------------------------------
# Core quantities
# ---------------------------------------------------------------------
def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """
    Return the Hilbert-envelope amplitude along the last axis.

    x: (..., L)
    returns: (..., L)
    """
    return np.abs(hilbert(x, axis=-1))


def envelope_peak(x: np.ndarray) -> np.ndarray:
    """
    Global Hilbert-envelope peak along the last axis.

    x: (..., L)
    returns: (...,)
    """
    return np.max(hilbert_envelope(x), axis=-1)


def clean_peak_index(clean: np.ndarray) -> np.ndarray:
    """
    Index of the clean Hilbert-envelope peak for each trace/channel.

    clean: (N, C, L)
    returns: (N, C)
    """
    return np.argmax(hilbert_envelope(clean), axis=-1)


def robust_mad_sigma(x: np.ndarray) -> float:
    """
    Robust Gaussian-equivalent scale from the median absolute deviation.

    For Gaussian noise, 1.4826 * MAD estimates the standard deviation.
    """
    if x.size == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def offpulse_noise_sigma(
    noisy: np.ndarray,
    clean: np.ndarray,
    exclude_radius: int = 64,
    noise_estimator: str = "raw_rms",
) -> np.ndarray:
    """
    Estimate the per-trace, per-channel noise scale from off-pulse samples.

    Parameters
    ----------
    noisy, clean
        Arrays with shape (N, C, L).
    exclude_radius
        Number of samples excluded on each side of the clean Hilbert-envelope
        peak. For 0.5 ns sampling, exclude_radius=64 removes +/-32 ns.
        Increase this if the pulse or RF response is broader.
    noise_estimator
        "raw_rms":
            sigma = std(noisy waveform samples outside the excluded pulse window).
        "envelope_std":
            sigma = std(|Hilbert(noisy)| outside the excluded pulse window).
        "envelope_mad":
            sigma = 1.4826 * MAD(|Hilbert(noisy)| outside the excluded pulse window).

    Returns
    -------
    sigma
        Array with shape (N, C).
    """
    if noise_estimator not in {"raw_rms", "envelope_std", "envelope_mad"}:
        raise ValueError(
            "noise_estimator must be one of: raw_rms, envelope_std, envelope_mad"
        )

    if noisy.shape != clean.shape:
        raise ValueError(f"noisy and clean shapes differ: {noisy.shape} vs {clean.shape}")

    peak_idx = clean_peak_index(clean)

    if noise_estimator == "raw_rms":
        noise_array = noisy
    else:
        noise_array = hilbert_envelope(noisy)

    n_traces, n_channels, n_samples = noisy.shape
    sigma = np.full((n_traces, n_channels), np.nan, dtype=float)

    min_samples = max(16, int(0.1 * n_samples))

    for i in range(n_traces):
        for c in range(n_channels):
            k = int(peak_idx[i, c])
            mask = np.ones(n_samples, dtype=bool)

            lo = max(0, k - exclude_radius)
            hi = min(n_samples, k + exclude_radius + 1)
            mask[lo:hi] = False

            samples = noise_array[i, c, mask]

            # Fallback should rarely trigger, but protects against pathological
            # short traces or overly large exclusion windows.
            if samples.size < min_samples:
                samples = noise_array[i, c]

            samples = samples[np.isfinite(samples)]
            if samples.size == 0:
                sigma[i, c] = np.nan
                continue

            if noise_estimator in {"raw_rms", "envelope_std"}:
                sigma[i, c] = float(np.std(samples))
            else:
                sigma[i, c] = robust_mad_sigma(samples)

    return sigma


def injected_snr_offpulse(
    clean: np.ndarray,
    noisy: np.ndarray,
    exclude_radius: int = 64,
    noise_estimator: str = "raw_rms",
) -> np.ndarray:
    """
    Per-channel injected SNR used for the amplitude-bias diagnostic.

        SNR = A_true / sigma_noise_off,

    where

        A_true = max_t |Hilbert(clean)(t)|,

    and sigma_noise_off is estimated from off-pulse noisy samples after
    excluding a window around the clean pulse time.

    clean, noisy: (N, C, L)
    returns: (N, C)
    """
    a_true = envelope_peak(clean)
    sigma_noise = offpulse_noise_sigma(
        noisy=noisy,
        clean=clean,
        exclude_radius=exclude_radius,
        noise_estimator=noise_estimator,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(sigma_noise > 0.0, a_true / sigma_noise, np.inf)

    return snr


def compute_delta_A(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    cfg: AmpBiasConfig,
) -> Dict[str, np.ndarray]:
    """
    Return per-trace, per-channel SNR and delta_A for both methods.

    All returned arrays have shape (N, C). A_true must be positive for delta_A
    to be finite.
    """
    if clean.shape != noisy.shape or clean.shape != denoised.shape:
        raise ValueError(
            "clean, noisy, denoised must have the same shape. Got "
            f"clean={clean.shape}, noisy={noisy.shape}, denoised={denoised.shape}"
        )

    a_true = envelope_peak(clean)
    a_noisy = envelope_peak(noisy)
    a_den = envelope_peak(denoised)

    with np.errstate(divide="ignore", invalid="ignore"):
        delta_noisy = np.where(a_true > 0.0, a_noisy / a_true - 1.0, np.nan)
        delta_denoised = np.where(a_true > 0.0, a_den / a_true - 1.0, np.nan)

    # Paper SNR, the single definition used throughout the analysis:
    #
    #     SNR = max(clean) / std(noisy)
    #
    # with the standard deviation over the FULL trace (no off-pulse exclusion, no
    # band-limiting, raw signed clean maximum), computed channel by channel so it
    # matches the (N, C) shape of delta_A. Identical for the noisy and the
    # denoised curves, which share this one array.
    if cfg.noise_estimator == "raw_rms":
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.max(clean, axis=-1) / np.std(noisy, axis=-1)
    else:
        snr = injected_snr_offpulse(
            clean=clean,
            noisy=noisy,
            exclude_radius=cfg.exclude_radius,
            noise_estimator=cfg.noise_estimator,
        )

    return {
        "snr": snr,
        "A_true": a_true,
        "A_noisy": a_noisy,
        "A_denoised": a_den,
        "delta_A_noisy": delta_noisy,
        "delta_A_denoised": delta_denoised,
    }


# ---------------------------------------------------------------------
# Binning and statistics
# ---------------------------------------------------------------------
_STAT_COLUMNS = [
    "channel",
    "method",
    "snr_bin_low",
    "snr_bin_high",
    "snr_bin_center",
    "n_traces",
    "median_delta_A",
    "q16_delta_A",
    "q84_delta_A",
    "sigma68_delta_A",
    "mean_delta_A",
    "std_delta_A",
    "median_abs_delta_A",
]


def shared_selection_mask(
    snr: np.ndarray,
    delta_noisy: np.ndarray,
    delta_denoised: np.ndarray,
    cfg: AmpBiasConfig,
) -> np.ndarray:
    """
    One selection mask used by BOTH the noisy and the denoised series.

    The referee asked us to verify that the two curves are built from identical
    events. Rather than let each series drop its own non-finite entries, the
    selection is computed once here and requires a trace to be usable for BOTH
    methods. It is therefore impossible for the two curves to see different
    traces, by construction.

    Same shape as the inputs, (N, C).
    """
    return (
        np.isfinite(snr)
        & np.isfinite(delta_noisy)
        & np.isfinite(delta_denoised)
        & (snr > cfg.min_snr)
        & (snr < cfg.max_snr)
    )


def bin_statistics(
    snr: np.ndarray,
    delta: np.ndarray,
    edges: Tuple[float, ...],
    cfg: AmpBiasConfig,
    valid: Optional[np.ndarray] = None,
) -> List[dict]:
    """
    Compute per-bin statistics for one channel/method series.

    snr, delta: 1-D arrays.
    valid: optional pre-computed selection. When supplied (the normal path) it is
        the SHARED mask from shared_selection_mask(), so the noisy and denoised
        series are guaranteed to use identical events and identical SNR values.
    """
    if valid is None:
        valid = (
            np.isfinite(snr)
            & np.isfinite(delta)
            & (snr > cfg.min_snr)
            & (snr < cfg.max_snr)
        )
    snr = snr[valid]
    delta = delta[valid]

    rows: List[dict] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        center = 0.5 * (lo + hi)
        m = (snr >= lo) & (snr < hi)
        d = delta[m]
        n = int(d.size)

        if n == 0:
            stats = {
                "median_delta_A": np.nan,
                "q16_delta_A": np.nan,
                "q84_delta_A": np.nan,
                "sigma68_delta_A": np.nan,
                "mean_delta_A": np.nan,
                "std_delta_A": np.nan,
                "median_abs_delta_A": np.nan,
            }
        else:
            q16, q50, q84 = np.percentile(d, [16.0, 50.0, 84.0])
            stats = {
                "median_delta_A": float(q50),
                "q16_delta_A": float(q16),
                "q84_delta_A": float(q84),
                "sigma68_delta_A": float(0.5 * (q84 - q16)),
                "mean_delta_A": float(np.mean(d)),
                "std_delta_A": float(np.std(d)),
                "median_abs_delta_A": float(np.median(np.abs(d))),
            }

        rows.append(
            {
                "snr_bin_low": float(lo),
                "snr_bin_high": float(hi),
                "snr_bin_center": float(center),
                "n_traces": n,
                **stats,
            }
        )

    return rows


def build_table(quantities: Dict[str, np.ndarray], cfg: AmpBiasConfig) -> List[dict]:
    """
    Assemble the full channel x method x SNR-bin table.
    """
    methods = [
        ("noisy", quantities["delta_A_noisy"]),
        ("denoised", quantities["delta_A_denoised"]),
    ]

    # ONE selection for both methods (referee: identical event selections).
    shared_valid = shared_selection_mask(
        snr=quantities["snr"],
        delta_noisy=quantities["delta_A_noisy"],
        delta_denoised=quantities["delta_A_denoised"],
        cfg=cfg,
    )

    table: List[dict] = []
    for ch_idx, ch_name in enumerate(cfg.channel_names):
        snr_ch = quantities["snr"][:, ch_idx]
        valid_ch = shared_valid[:, ch_idx]

        for method_name, delta in methods:
            rows = bin_statistics(
                snr=snr_ch,
                delta=delta[:, ch_idx],
                edges=cfg.snr_edges,
                cfg=cfg,
                valid=valid_ch,
            )

            for r in rows:
                table.append({"channel": ch_name, "method": method_name, **r})

    return table


def write_counts_table(table: List[dict], cfg: AmpBiasConfig, out_dir: str) -> None:
    """
    Standalone table of SNR-selected traces per SNR bin, one row per bin.

    The referee asked for these counts "in the figure or an accompanying table".
    They crowded the figure panels, so they are reported here instead, as both a
    CSV and a LaTeX fragment ready to drop into the manuscript.

    The noisy and denoised series share one selection (shared_selection_mask), so
    a single count per bin/channel describes both curves; this is asserted below.
    """
    centers = sorted({r["snr_bin_center"] for r in table})
    lookup = {
        (r["channel"], r["method"], r["snr_bin_center"]): r for r in table
    }

    rows: List[dict] = []
    for c in centers:
        row = {"snr_bin_center": c}
        lo = hi = None
        for ch in cfg.channel_names:
            n_noisy = lookup[(ch, "noisy", c)]["n_traces"]
            n_den = lookup[(ch, "denoised", c)]["n_traces"]
            if n_noisy != n_den:
                raise AssertionError(
                    f"Selection mismatch in {ch} bin centred {c}: "
                    f"noisy={n_noisy} vs denoised={n_den}. The shared selection "
                    "mask should make this impossible."
                )
            row[f"n_traces_{ch}"] = n_noisy
            lo = lookup[(ch, "noisy", c)]["snr_bin_low"]
            hi = lookup[(ch, "noisy", c)]["snr_bin_high"]
        row["snr_bin_low"] = lo
        row["snr_bin_high"] = hi
        rows.append(row)

    fields = (["snr_bin_low", "snr_bin_high", "snr_bin_center"]
              + [f"n_traces_{ch}" for ch in cfg.channel_names])

    csv_path = os.path.join(out_dir, "snr_selected_counts.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fields})
    print(f"Saved counts table: {csv_path}")

    tex_path = os.path.join(out_dir, "snr_selected_counts.tex")
    with open(tex_path, "w") as f:
        f.write("% Number of SNR-selected traces per input-SNR bin (1 < SNR < 1000).\n")
        f.write("% Identical for the noisy and denoised curves by construction\n")
        f.write("% (both use one shared selection).\n")
        f.write("\\begin{tabular}{ccrrr}\n\\hline\n")
        f.write("SNR bin & centre & $N_X$ & $N_Y$ & $N_Z$ \\\\\n\\hline\n")
        for r in rows:
            f.write(
                f"$[{r['snr_bin_low']:.1f},{r['snr_bin_high']:.1f})$ & "
                f"{r['snr_bin_center']:.1f} & "
                + " & ".join(f"{r[f'n_traces_{ch}']:,}" for ch in cfg.channel_names)
                + " \\\\\n"
            )
        totals = [sum(r[f"n_traces_{ch}"] for r in rows) for ch in cfg.channel_names]
        f.write("\\hline\n")
        f.write("Total & & " + " & ".join(f"{t:,}" for t in totals) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    print(f"Saved counts table: {tex_path}")

    print("\nSNR-selected traces per SNR bin "
          "(identical for noisy and denoised):")
    header = f"  {'SNR bin':>14}" + "".join(f"{'N_' + ch:>10}" for ch in cfg.channel_names)
    print(header)
    for r in rows:
        line = f"  [{r['snr_bin_low']:5.1f},{r['snr_bin_high']:5.1f})"
        line += "".join(f"{r[f'n_traces_{ch}']:>10,}" for ch in cfg.channel_names)
        print(line)
    print(f"  {'Total':>14}" + "".join(f"{t:>10,}" for t in totals))


def write_csv(table: List[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_STAT_COLUMNS)
        writer.writeheader()
        for row in table:
            writer.writerow({k: row[k] for k in _STAT_COLUMNS})
    print(f"Saved table: {path}")


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def make_figure(table: List[dict], cfg: AmpBiasConfig, out_path: str) -> None:
    """
    3-panel X/Y/Z plot of median delta_A versus SNR with central-68% bands.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)

    method_style = {
        "noisy": {
            "color": cfg.color_noisy,
            "marker": "s",
            "ls": "--",
            "label": "Noisy",
        },
        "denoised": {
            "color": cfg.color_denoised,
            "marker": "o",
            "ls": "-",
            "label": "Denoised",
        },
    }

    def series(ch_name: str, method: str):
        rows = [
            r
            for r in table
            if r["channel"] == ch_name
            and r["method"] == method
            and r["n_traces"] >= cfg.min_count_plot
            and np.isfinite(r["median_delta_A"])
        ]
        rows.sort(key=lambda r: r["snr_bin_center"])

        x = np.array([r["snr_bin_center"] for r in rows])
        med = np.array([r["median_delta_A"] for r in rows])
        q16 = np.array([r["q16_delta_A"] for r in rows])
        q84 = np.array([r["q84_delta_A"] for r in rows])
        n = np.array([r["n_traces"] for r in rows])
        return x, med, q16, q84, n

    handles, labels = [], []
    for ch_idx, ch_name in enumerate(cfg.channel_names):
        ax = axes[ch_idx]
        ax.axhline(0.0, color="k", lw=1.0, ls="--", alpha=0.7)
        ax.set_title(f"{ch_name} channel", fontsize=cfg.fontsize)

        # Draw noisy first, denoised second.
        for method in ("noisy", "denoised"):
            st = method_style[method]
            x, med, q16, q84, n = series(ch_name, method)
            if x.size == 0:
                continue

            ax.fill_between(
                x,
                q16,
                q84,
                color=st["color"],
                alpha=cfg.band_alpha,
                lw=0,
            )
            h = ax.plot(
                x,
                med,
                color=st["color"],
                ls=st["ls"],
                lw=2.2,
                marker=st["marker"],
                markersize=6,
                label=st["label"] + " (median)",
            )[0]

            if ch_idx == 0:
                handles.append(h)
                labels.append(st["label"] + " (median)")

        # Per-bin trace counts are NOT drawn on the figure (they crowded the
        # panels); they are reported in the accompanying
        # snr_selected_counts.csv / .tex table instead.

        ax.set_xlabel("SNR", fontsize=cfg.fontsize)
        ax.set_ylim(*cfg.ylim)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", which="major", labelsize=cfg.fontsize)

        if ch_idx == 0:
            ax.set_ylabel(
                r"$\delta_A = A_{\rm rec}/A_{\rm true} - 1$",
                fontsize=cfg.fontsize,
            )

    if handles:
        # Spell the shaded interval out in the legend itself, so the figure is
        # self-describing and cannot be misread as a standard error or +/-1 sigma.
        from matplotlib.patches import Patch

        band_proxy = Patch(
            facecolor="0.5", alpha=cfg.band_alpha, edgecolor="none",
            label="shaded: 16th-84th pct (central 68%)",
        )
        axes[0].legend(
            handles + [band_proxy],
            labels + ["shaded: 16th-84th pct (central 68%)"],
            loc="upper right",
            frameon=False,
            fontsize=cfg.fontsize - 6,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    print(f"Saved figure: {out_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_edges(s: str | None) -> Tuple[float, ...]:
    if s is None:
        return tuple(np.linspace(1.0, 10.0, 10).tolist())

    edges = tuple(float(v.strip()) for v in s.split(",") if v.strip())
    if len(edges) < 2:
        raise ValueError("--snr-edges must contain at least two values.")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("--snr-edges must be strictly increasing.")
    return edges


def parse_ylim(s: str) -> Tuple[float, float]:
    vals = [float(v.strip()) for v in s.split(",")]
    if len(vals) != 2:
        raise ValueError("--ylim must be of the form low,high")
    if vals[0] >= vals[1]:
        raise ValueError("--ylim lower bound must be smaller than upper bound.")
    return vals[0], vals[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Amplitude-bias diagnostic versus SNR for referee Q5."
    )

    default_results = Path("/sps/grand/macias/Sam_Result/multi_l1_CNN_100epochs_24samples")

    parser.add_argument(
        "--model-path",
        default=str(default_results / "best_model.pth"),
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--metrics-json",
        default=str(default_results / "best_trial_metrics.json"),
        help="Path to best trial metrics JSON.",
    )
    parser.add_argument(
        "--config-json",
        default=str(default_results / "best_trial_config.json"),
        help="Path to best trial config JSON.",
    )
    parser.add_argument(
        "--data-path",
        default="/sps/grand/blevy/sims/sims_for_denoising_sept2025",
        help="Path to simulation dataset.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="Inference device.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on test traces for debugging.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent),
        help="Output directory.",
    )

    parser.add_argument(
        "--min-snr",
        type=float,
        default=1.0,
        help="Minimum SNR retained in table/plot statistics.",
    )
    parser.add_argument(
        "--max-snr",
        type=float,
        default=1.0e3,
        help="Maximum SNR retained in table/plot statistics.",
    )
    parser.add_argument(
        "--snr-edges",
        default=None,
        help=(
            "Comma-separated SNR bin edges. Default: 1,2,...,10. "
            "Example: '1,1.5,2,2.5,3,4,5,6,8,10'."
        ),
    )
    parser.add_argument(
        "--exclude-radius",
        type=int,
        default=64,
        help=(
            "Samples excluded on each side of the clean pulse peak when estimating "
            "off-pulse noise. With 0.5 ns sampling, 64 samples = 32 ns."
        ),
    )
    parser.add_argument(
        "--noise-estimator",
        default="raw_rms",
        choices=["raw_rms", "envelope_std", "envelope_mad"],
        help=(
            "Noise scale estimator for the SNR denominator. Recommended default: "
            "raw_rms. Use envelope_std or envelope_mad for envelope-space noise."
        ),
    )
    parser.add_argument(
        "--min-count-plot",
        type=int,
        default=20,
        help="Minimum number of traces per bin required for plotting.",
    )
    parser.add_argument(
        "--ylim",
        default="-1.0,2.0",
        help="Y-axis limits as low,high.",
    )
    parser.add_argument(
        "--eval-len",
        type=int,
        default=512,
        help=(
            "Deterministic evaluation trace length. Default 512 = the production "
            "training length, so the checkpoint is evaluated at the length it was "
            "trained on. Pass 0 to use the full 1024-sample trace."
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=12345,
        help="Seed for the deterministic 80/10/10 split, so the test set is recorded "
             "and reproducible.",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    edges = parse_edges(args.snr_edges)
    ylim = parse_ylim(args.ylim)

    cfg = AmpBiasConfig(
        min_snr=args.min_snr,
        max_snr=args.max_snr,
        snr_edges=edges,
        exclude_radius=args.exclude_radius,
        noise_estimator=args.noise_estimator,
        min_count_plot=args.min_count_plot,
        ylim=ylim,
    )

    print("Project root:", ROOT_DIR)
    print("Loading model + held-out test set and running inference...")
    print("SNR definition:")
    print("  numerator   = max |Hilbert(clean)|")
    print(f"  denominator = off-pulse noisy-trace scale ({cfg.noise_estimator})")
    print(f"  exclude +/- {cfg.exclude_radius} samples around clean pulse peak")

    eval_len = args.eval_len if args.eval_len and args.eval_len > 0 else None
    manifest_path = os.path.join(args.out_dir, "evaluation_manifest.npz")

    pack = load_data_and_run_inference(
        model_path=args.model_path,
        metrics_json=args.metrics_json,
        config_json=args.config_json,
        data_path=args.data_path,
        device=args.device,
        batch_size=args.batch_size,
        test_only=True,
        max_samples=args.max_samples,
        eval_len=eval_len,
        split_seed=args.split_seed,
        manifest_path=manifest_path,
    )

    clean = np.asarray(pack.clean)
    noisy = np.asarray(pack.noisy)
    denoised = np.asarray(pack.denoised)

    print(f"Shapes: clean={clean.shape}, noisy={noisy.shape}, denoised={denoised.shape}")

    if clean.ndim != 3:
        raise ValueError(f"Expected arrays with shape (N, C, L), got {clean.shape}")
    if clean.shape[1] != 3:
        raise ValueError(f"Expected three polarization channels, got shape {clean.shape}")

    quantities = compute_delta_A(clean=clean, noisy=noisy, denoised=denoised, cfg=cfg)
    table = build_table(quantities, cfg)

    # --- Provenance record: exactly which checkpoint produced this figure -----
    import json as _json
    try:
        with open(args.config_json) as _f:
            _ckpt_cfg = _json.load(_f)
    except Exception:
        _ckpt_cfg = {}
    try:
        with open(args.metrics_json) as _f:
            _ckpt_metrics = _json.load(_f)
    except Exception:
        _ckpt_metrics = {}

    provenance = {
        "checkpoint_path": os.path.abspath(args.model_path),
        "config_json": os.path.abspath(args.config_json),
        "metrics_json": os.path.abspath(args.metrics_json),
        "criterion": _ckpt_cfg.get("criterion"),
        "model_type": _ckpt_cfg.get("model_type"),
        "model_config": _ckpt_cfg.get("model_config"),
        "checkpoint_metrics": _ckpt_metrics,
        "input_length_samples": eval_len if eval_len is not None else int(clean.shape[-1]),
        "evaluated_trace_shape": list(clean.shape),
        "data_path": args.data_path,
        "test_split": {
            "seed": args.split_seed,
            "fractions": {"train": 0.8, "valid": 0.1, "test": 0.1},
            "manifest": os.path.abspath(manifest_path),
            "n_traces_evaluated": int(clean.shape[0]),
            "max_samples_cap": args.max_samples,
        },
        "snr_definition": "max(clean) / std(noisy), std over the full trace, per channel",
        "snr_selection": {
            "min_snr_exclusive": cfg.min_snr,
            "max_snr_exclusive": cfg.max_snr,
            "plotted_bin_edges": list(cfg.snr_edges),
            "min_traces_per_plotted_bin": cfg.min_count_plot,
            "note": "Truth-conditioned SNR window applied identically to the noisy "
                    "and denoised curves. This is NOT the noisy-input trigger used "
                    "by the timing analysis.",
        },
        "shaded_band": "16th-84th percentile of delta_A (central 68%)",
    }
    prov_path = os.path.join(args.out_dir, "figure_provenance.json")
    with open(prov_path, "w") as f:
        _json.dump(provenance, f, indent=2)
    print(f"Saved provenance: {prov_path}")
    print("  checkpoint      :", provenance["checkpoint_path"])
    print("  criterion       :", provenance["criterion"])
    print("  input length    :", provenance["input_length_samples"], "samples")
    print("  test split seed :", args.split_seed,
          f"({provenance['test_split']['n_traces_evaluated']} traces evaluated)")

    csv_path = os.path.join(args.out_dir, "amplitude_bias_table.csv")
    write_csv(table, csv_path)

    # Standalone counts table (kept out of the figure to avoid crowding it).
    write_counts_table(table, cfg, args.out_dir)

    npz_path = os.path.join(args.out_dir, "amplitude_bias_per_trace.npz")
    np.savez_compressed(
        npz_path,
        snr=quantities["snr"],
        A_true=quantities["A_true"],
        A_noisy=quantities["A_noisy"],
        A_denoised=quantities["A_denoised"],
        delta_A_noisy=quantities["delta_A_noisy"],
        delta_A_denoised=quantities["delta_A_denoised"],
        snr_edges=np.array(edges),
        channel_names=np.array(cfg.channel_names),
        exclude_radius=np.array([cfg.exclude_radius]),
        noise_estimator=np.array([cfg.noise_estimator]),
    )
    print(f"Saved per-trace arrays: {npz_path}")

    fig_path = os.path.join(args.out_dir, "amplitude_bias_vs_snr.pdf")
    make_figure(table, cfg, fig_path)

    print("\nHeadline summary over plotted SNR bins:")
    print("  median_delta_A = bias")
    print("  sigma68_delta_A = robust scatter")
    for ch in cfg.channel_names:
        for method in ("noisy", "denoised"):
            rows = [
                r
                for r in table
                if r["channel"] == ch
                and r["method"] == method
                and r["n_traces"] >= cfg.min_count_plot
                and np.isfinite(r["median_delta_A"])
            ]
            if not rows:
                continue

            med_bias = np.nanmedian([r["median_delta_A"] for r in rows])
            med_s68 = np.nanmedian([r["sigma68_delta_A"] for r in rows])
            med_abs = np.nanmedian([r["median_abs_delta_A"] for r in rows])
            print(
                f"  {ch} {method:9s}: "
                f"median(bias)={med_bias:+.3f}, "
                f"median(sigma68)={med_s68:.3f}, "
                f"median(|delta_A|)={med_abs:.3f}"
            )


if __name__ == "__main__":
    main()