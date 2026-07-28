"""Canonical losses, SNR, and binned diagnostics for the GRAND denoiser paper.

This module is deliberately small and side-effect free.  It centralizes the
quantities that must be identical across training and all publication figures.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import hilbert

Reduction = Literal["mean", "sum", "none"]
PhaseWeighting = Literal["none", "target_magnitude"]
Norm = Literal["l1", "mse"]


def wrapped_phase_difference(pred_phase: torch.Tensor, true_phase: torch.Tensor) -> torch.Tensor:
    """Return the signed circular residual in [-pi, pi]."""
    delta = pred_phase - true_phase
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def _reduce(x: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "mean":
        return x.mean()
    if reduction == "sum":
        return x.sum()
    if reduction == "none":
        return x
    raise ValueError(f"Unknown reduction: {reduction}")


def circular_phase_loss(
    pred_fft: torch.Tensor,
    true_fft: torch.Tensor,
    *,
    norm: Norm = "l1",
    weighting: PhaseWeighting = "none",
    min_relative_magnitude: float | None = None,
    eps: float = 1e-12,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Circular phase loss with optional target-magnitude weighting/masking.

    Parameters
    ----------
    pred_fft, true_fft
        Complex Fourier coefficients with identical shapes.
    norm
        ``l1`` uses |Delta phi|; ``mse`` uses (Delta phi)^2.
    weighting
        ``none`` reproduces the manuscript's unweighted wrapped loss.
        ``target_magnitude`` downweights bins whose reference magnitude is small.
    min_relative_magnitude
        Optional per-trace mask.  A bin is retained when
        |X_true(f)| >= min_relative_magnitude * max_f |X_true(f)|.
        Do not enable this without documenting the threshold in the paper.
    """
    if pred_fft.shape != true_fft.shape:
        raise ValueError(f"FFT shapes differ: {pred_fft.shape} vs {true_fft.shape}")
    if not torch.is_complex(pred_fft) or not torch.is_complex(true_fft):
        raise TypeError("pred_fft and true_fft must be complex tensors")

    delta = wrapped_phase_difference(torch.angle(pred_fft), torch.angle(true_fft))
    point_loss = delta.abs() if norm == "l1" else delta.square()
    if norm not in {"l1", "mse"}:
        raise ValueError(f"Unknown norm: {norm}")

    target_mag = true_fft.abs()
    weights = torch.ones_like(target_mag)

    if weighting == "target_magnitude":
        scale = target_mag.amax(dim=-1, keepdim=True).clamp_min(eps)
        weights = target_mag / scale
    elif weighting != "none":
        raise ValueError(f"Unknown phase weighting: {weighting}")

    if min_relative_magnitude is not None:
        if not 0.0 <= min_relative_magnitude <= 1.0:
            raise ValueError("min_relative_magnitude must lie in [0, 1]")
        scale = target_mag.amax(dim=-1, keepdim=True).clamp_min(eps)
        weights = weights * (target_mag >= min_relative_magnitude * scale)

    if reduction == "none":
        return point_loss * weights

    denom = weights.sum().clamp_min(eps)
    weighted = (point_loss * weights).sum() / denom
    return weighted if reduction == "mean" else weighted * denom


def multi_domain_loss(
    clean: torch.Tensor,
    pred: torch.Tensor,
    *,
    mag_weight: float,
    phase_weight: float,
    norm: Norm = "l1",
    phase_weighting: PhaseWeighting = "none",
    min_relative_magnitude: float | None = None,
) -> torch.Tensor:
    """Time + Fourier-magnitude + wrapped-Fourier-phase objective."""
    if clean.shape != pred.shape:
        raise ValueError(f"Signal shapes differ: {clean.shape} vs {pred.shape}")

    if norm == "l1":
        time_loss = F.l1_loss(pred, clean)
    elif norm == "mse":
        time_loss = F.mse_loss(pred, clean)
    else:
        raise ValueError(f"Unknown norm: {norm}")

    clean_fft = torch.fft.rfft(clean, dim=-1)
    pred_fft = torch.fft.rfft(pred, dim=-1)

    if norm == "l1":
        mag_loss = F.l1_loss(pred_fft.abs(), clean_fft.abs())
    else:
        mag_loss = F.mse_loss(pred_fft.abs(), clean_fft.abs())

    phase_loss = circular_phase_loss(
        pred_fft,
        clean_fft,
        norm=norm,
        weighting=phase_weighting,
        min_relative_magnitude=min_relative_magnitude,
    )
    return time_loss + float(mag_weight) * mag_loss + float(phase_weight) * phase_loss


def multi_domain_l1_loss(
    clean: torch.Tensor,
    pred: torch.Tensor,
    mag_weight: float,
    phase_weight: float,
    *,
    phase_weighting: PhaseWeighting = "none",
    min_relative_magnitude: float | None = None,
) -> torch.Tensor:
    return multi_domain_loss(
        clean,
        pred,
        mag_weight=mag_weight,
        phase_weight=phase_weight,
        norm="l1",
        phase_weighting=phase_weighting,
        min_relative_magnitude=min_relative_magnitude,
    )


def multi_domain_mse_loss(
    clean: torch.Tensor,
    pred: torch.Tensor,
    mag_weight: float,
    phase_weight: float,
    *,
    phase_weighting: PhaseWeighting = "none",
    min_relative_magnitude: float | None = None,
) -> torch.Tensor:
    return multi_domain_loss(
        clean,
        pred,
        mag_weight=mag_weight,
        phase_weight=phase_weight,
        norm="mse",
        phase_weighting=phase_weighting,
        min_relative_magnitude=min_relative_magnitude,
    )


@dataclass(frozen=True)
class PaperSNR:
    snr: np.ndarray
    clean_peak: np.ndarray
    sigma_off: np.ndarray
    clean_peak_index: np.ndarray


# ---------------------------------------------------------------------
# Canonical paper-SNR configuration (task 3).
# Every figure must use paper_input_snr with THIS exclusion half-width and
# estimator. The value is PROVISIONAL: recover the exact production value in
# task 4 and update it HERE only, so all figures stay consistent. Record it in
# each figure's metadata.
# ---------------------------------------------------------------------
PAPER_SNR_ESTIMATOR = "raw_rms_offpulse"          # std of the raw off-pulse noisy samples
# Off-pulse exclusion half-width and std ddof for the paper SNR.
# CONFIRMED (task-4, 2026-07-23): 64 samples (= +/-32 ns @ 0.5 ns/sample), std-based,
# ddof=0. Used by BOTH publication figures -- the amplitude-bias figure and the
# NMSE/SNR-gain (option3) figure, the latter regenerated after the task-3 refactor
# so it shares this exact estimator. Change HERE only so all figures stay consistent.
# (The ns interpretation still assumes dt = 0.5 ns/sample, pending the task-1 audit;
# the SNR itself uses the 64-sample count directly and is unaffected by that.)
PRODUCTION_OFFPULSE_EXCLUDE_HALF_WIDTH = 64       # samples
PRODUCTION_DDOF = 0                               # std degrees-of-freedom correction
# Backward-compatible alias (old name kept so existing imports keep resolving).
PROVISIONAL_OFFPULSE_EXCLUDE_HALF_WIDTH = PRODUCTION_OFFPULSE_EXCLUDE_HALF_WIDTH

def paper_input_snr(
    clean: np.ndarray,
    noisy: np.ndarray,
    *,
    exclude_half_width_samples: int,
    ddof: int = 0,
    eps: float = 1e-12,
) -> PaperSNR:
    """Compute the paper SNR channel by channel.

    SNR_in = max_t |Hilbert(clean)| / std(noisy off-pulse samples), where the
    off-pulse samples exclude a symmetric window around the clean-envelope peak.

    ``exclude_half_width_samples`` is mandatory so the production value cannot
    be silently guessed.  Inputs may have shape (..., T); outputs have shape (...).
    """
    clean = np.asarray(clean, dtype=np.float64)
    noisy = np.asarray(noisy, dtype=np.float64)
    if clean.shape != noisy.shape or clean.ndim < 1:
        raise ValueError(f"clean/noisy shapes must match and include time: {clean.shape}, {noisy.shape}")
    if exclude_half_width_samples < 0:
        raise ValueError("exclude_half_width_samples must be non-negative")

    leading = clean.shape[:-1]
    n_time = clean.shape[-1]
    flat_clean = clean.reshape(-1, n_time)
    flat_noisy = noisy.reshape(-1, n_time)

    env = np.abs(hilbert(flat_clean, axis=-1))
    peak_index = np.argmax(env, axis=-1)
    peak = np.max(env, axis=-1)
    sigma = np.empty(flat_clean.shape[0], dtype=np.float64)

    for i, k in enumerate(peak_index):
        lo = max(0, int(k) - exclude_half_width_samples)
        hi = min(n_time, int(k) + exclude_half_width_samples + 1)
        mask = np.ones(n_time, dtype=bool)
        mask[lo:hi] = False
        off = flat_noisy[i, mask]
        if off.size <= ddof:
            raise ValueError(
                f"Too few off-pulse samples for trace {i}: {off.size}; "
                "reduce exclude_half_width_samples"
            )
        sigma[i] = np.std(off, ddof=ddof)

    sigma = np.maximum(sigma, eps)
    return PaperSNR(
        snr=(peak / sigma).reshape(leading),
        clean_peak=peak.reshape(leading),
        sigma_off=sigma.reshape(leading),
        clean_peak_index=peak_index.reshape(leading),
    )


def hilbert_peak(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.max(np.abs(hilbert(x, axis=-1)), axis=-1)


def trigger_pass_mask(noisy: np.ndarray, sigma_off: np.ndarray, *, k_sigma: float = 1.0) -> np.ndarray:
    """Candidate preselection used in the revised manuscript."""
    return hilbert_peak(noisy) >= float(k_sigma) * np.asarray(sigma_off)


def signed_peak_amplitude_bias(clean: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    true_peak = hilbert_peak(clean)
    rec_peak = hilbert_peak(reconstructed)
    return rec_peak / np.maximum(true_peak, 1e-12) - 1.0


@dataclass(frozen=True)
class BinnedSummary:
    left: np.ndarray
    right: np.ndarray
    center: np.ndarray
    median: np.ndarray
    q_low: np.ndarray
    q_high: np.ndarray
    count: np.ndarray


def binned_summary(
    x: np.ndarray,
    y: np.ndarray,
    bins: np.ndarray,
    *,
    quantiles: tuple[float, float] = (0.05, 0.95),
) -> BinnedSummary:
    """Median, population interval, and count in fixed bins."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    bins = np.asarray(bins, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same number of entries")
    if bins.ndim != 1 or bins.size < 2 or np.any(np.diff(bins) <= 0):
        raise ValueError("bins must be a strictly increasing one-dimensional array")

    q0, q1 = quantiles
    if not 0 <= q0 < q1 <= 1:
        raise ValueError("quantiles must satisfy 0 <= q0 < q1 <= 1")

    med = np.full(bins.size - 1, np.nan)
    qlo = np.full_like(med, np.nan)
    qhi = np.full_like(med, np.nan)
    cnt = np.zeros(bins.size - 1, dtype=int)

    finite = np.isfinite(x) & np.isfinite(y)
    for i in range(bins.size - 1):
        mask = finite & (x >= bins[i]) & (x < bins[i + 1])
        vals = y[mask]
        cnt[i] = vals.size
        if vals.size:
            med[i] = np.median(vals)
            qlo[i], qhi[i] = np.quantile(vals, [q0, q1])

    return BinnedSummary(
        left=bins[:-1],
        right=bins[1:],
        center=0.5 * (bins[:-1] + bins[1:]),
        median=med,
        q_low=qlo,
        q_high=qhi,
        count=cnt,
    )


def deterministic_split_indices(
    n_total: int,
    *,
    train_fraction: float = 0.8,
    valid_fraction: float = 0.1,
    seed: int = 12345,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One deterministic, disjoint train/validation/test split."""
    if n <= 0:
        raise ValueError("n must be positive")
    if train_frac <= 0 or valid_frac <= 0 or train_frac + valid_frac >= 1:
        raise ValueError("fractions must be positive and sum to less than one")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    n_valid = int(n * valid_frac)
    return idx[:n_train], idx[n_train:n_train + n_valid], idx[n_train + n_valid:]


def save_split_manifest(
    path: str | Path,
    train: np.ndarray,
    valid: np.ndarray,
    test: np.ndarray,
    *,
    n_total: int,
    seed: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train=np.asarray(train, dtype=np.int64),
        valid=np.asarray(valid, dtype=np.int64),
        test=np.asarray(test, dtype=np.int64),
        n_total=np.int64(n_total),
        seed=np.int64(seed),
    )
