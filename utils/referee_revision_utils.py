"""Small utilities for the minimal GRAND denoiser referee revision.

The module supports three low-cost tasks:
1. audit the direct versus wrapped Fourier-phase residual on an existing model;
2. create or validate a common evaluation manifest;
3. compute the manuscript input SNR and binned population summaries.

It deliberately does not implement magnitude-weighted phase losses or a new
hyperparameter search.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import json

import numpy as np
import torch
from scipy.signal import hilbert


def wrapped_phase_difference(
    predicted_phase: torch.Tensor,
    target_phase: torch.Tensor,
) -> torch.Tensor:
    """Shortest signed angular difference in [-pi, pi]."""
    delta = predicted_phase - target_phase
    return torch.atan2(torch.sin(delta), torch.cos(delta))


@dataclass(frozen=True)
class PhaseAuditResult:
    coefficient_count: int
    boundary_count: int
    boundary_fraction: float
    direct_phase_l1: float
    wrapped_phase_l1: float
    absolute_phase_l1_change: float
    relative_phase_l1_change: float
    phase_weight: float
    weighted_objective_change: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class PhaseAuditAccumulator:
    """Accumulate phase-boundary diagnostics across validation batches.

    The audit reproduces the original unweighted phase term: every RFFT
    coefficient is counted equally, including DC and Nyquist bins when present.
    A Boolean RFFT-frequency mask may be supplied to restrict the audit to an
    auxiliary frequency band.
    """

    def __init__(self, *, phase_weight: float, frequency_mask: torch.Tensor | None = None):
        self.phase_weight = float(phase_weight)
        self.frequency_mask = frequency_mask
        self._count = 0
        self._boundary_count = 0
        self._direct_abs_sum = 0.0
        self._wrapped_abs_sum = 0.0

    @torch.no_grad()
    def update(self, clean: torch.Tensor, reconstructed: torch.Tensor) -> None:
        if clean.shape != reconstructed.shape:
            raise ValueError(f"Shape mismatch: {clean.shape} vs {reconstructed.shape}")
        if clean.ndim < 2:
            raise ValueError("Expected tensors with a final time dimension")

        clean_fft = torch.fft.rfft(clean, dim=-1)
        rec_fft = torch.fft.rfft(reconstructed, dim=-1)
        direct = torch.angle(rec_fft) - torch.angle(clean_fft)

        if self.frequency_mask is not None:
            mask = self.frequency_mask.to(device=direct.device, dtype=torch.bool)
            if mask.ndim != 1 or mask.numel() != direct.shape[-1]:
                raise ValueError(
                    f"Frequency mask has shape {mask.shape}; expected ({direct.shape[-1]},)"
                )
            direct = direct[..., mask]

        wrapped = torch.atan2(torch.sin(direct), torch.cos(direct))
        direct_abs = direct.abs()
        wrapped_abs = wrapped.abs()

        self._count += int(direct_abs.numel())
        self._boundary_count += int((direct_abs > torch.pi).sum().item())
        self._direct_abs_sum += float(direct_abs.sum().item())
        self._wrapped_abs_sum += float(wrapped_abs.sum().item())

    def finalize(self) -> PhaseAuditResult:
        if self._count == 0:
            raise RuntimeError("No coefficients were accumulated")

        direct_mean = self._direct_abs_sum / self._count
        wrapped_mean = self._wrapped_abs_sum / self._count
        change = direct_mean - wrapped_mean
        relative = change / direct_mean if direct_mean > 0 else 0.0

        return PhaseAuditResult(
            coefficient_count=self._count,
            boundary_count=self._boundary_count,
            boundary_fraction=self._boundary_count / self._count,
            direct_phase_l1=direct_mean,
            wrapped_phase_l1=wrapped_mean,
            absolute_phase_l1_change=change,
            relative_phase_l1_change=relative,
            phase_weight=self.phase_weight,
            weighted_objective_change=self.phase_weight * change,
        )


def rfft_frequency_mask(
    n_time: int,
    sample_spacing_seconds: float,
    *,
    f_min_hz: float,
    f_max_hz: float,
) -> torch.Tensor:
    """Boolean RFFT mask for a closed frequency interval."""
    if n_time <= 0:
        raise ValueError("n_time must be positive")
    if sample_spacing_seconds <= 0:
        raise ValueError("sample_spacing_seconds must be positive")
    if not 0 <= f_min_hz <= f_max_hz:
        raise ValueError("Require 0 <= f_min_hz <= f_max_hz")

    freq = torch.fft.rfftfreq(n_time, d=sample_spacing_seconds)
    return (freq >= f_min_hz) & (freq <= f_max_hz)


def save_phase_audit_json(
    path: str | Path,
    *,
    full_band: PhaseAuditResult,
    analysis_band: PhaseAuditResult | None = None,
    metadata: dict | None = None,
) -> None:
    payload: dict[str, object] = {"full_band": full_band.to_dict()}
    if analysis_band is not None:
        payload["analysis_band"] = analysis_band.to_dict()
    if metadata:
        payload["metadata"] = metadata

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def wrapped_phase_l1_loss(clean: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """Wrapped phase term for the conditional one-run retraining fallback."""
    clean_fft = torch.fft.rfft(clean, dim=-1)
    rec_fft = torch.fft.rfft(reconstructed, dim=-1)
    delta = wrapped_phase_difference(torch.angle(rec_fft), torch.angle(clean_fft))
    return delta.abs().mean()


@dataclass(frozen=True)
class EvaluationManifest:
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    n_total: int
    seed: int
    provenance: str


def deterministic_split_indices(
    n_total: int,
    *,
    train_fraction: float = 0.8,
    valid_fraction: float = 0.1,
    seed: int = 12345,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create one deterministic, disjoint train/validation/test split."""
    if n_total <= 0:
        raise ValueError("n_total must be positive")
    if train_fraction <= 0 or valid_fraction <= 0:
        raise ValueError("train_fraction and valid_fraction must be positive")
    if train_fraction + valid_fraction >= 1:
        raise ValueError("train_fraction + valid_fraction must be less than one")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_train = int(train_fraction * n_total)
    n_valid = int(valid_fraction * n_total)
    return (
        indices[:n_train],
        indices[n_train:n_train + n_valid],
        indices[n_train + n_valid:],
    )


def _validate_split(train: np.ndarray, valid: np.ndarray, test: np.ndarray, n_total: int) -> None:
    arrays = [np.asarray(x, dtype=np.int64).ravel() for x in (train, valid, test)]
    joined = np.concatenate(arrays)
    if joined.size != n_total:
        raise ValueError(f"Manifest has {joined.size} entries; expected {n_total}")
    if np.unique(joined).size != n_total:
        raise ValueError("Manifest indices overlap or contain duplicates")
    if joined.min(initial=0) < 0 or joined.max(initial=-1) >= n_total:
        raise ValueError("Manifest contains out-of-range indices")


def save_evaluation_manifest(
    path: str | Path,
    train: np.ndarray,
    valid: np.ndarray,
    test: np.ndarray,
    *,
    n_total: int,
    seed: int,
    provenance: str,
) -> None:
    _validate_split(train, valid, test, n_total)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train=np.asarray(train, dtype=np.int64),
        valid=np.asarray(valid, dtype=np.int64),
        test=np.asarray(test, dtype=np.int64),
        n_total=np.int64(n_total),
        seed=np.int64(seed),
        provenance=np.asarray(provenance),
    )


def load_evaluation_manifest(path: str | Path) -> EvaluationManifest:
    with np.load(Path(path), allow_pickle=False) as data:
        manifest = EvaluationManifest(
            train=np.asarray(data["train"], dtype=np.int64),
            valid=np.asarray(data["valid"], dtype=np.int64),
            test=np.asarray(data["test"], dtype=np.int64),
            n_total=int(data["n_total"]),
            seed=int(data["seed"]),
            provenance=str(data["provenance"].item()),
        )
    _validate_split(manifest.train, manifest.valid, manifest.test, manifest.n_total)
    return manifest


@dataclass(frozen=True)
class PaperSNR:
    snr: np.ndarray
    clean_peak: np.ndarray
    sigma_off: np.ndarray
    clean_peak_index: np.ndarray


def paper_input_snr(
    clean: np.ndarray,
    noisy: np.ndarray,
    *,
    exclude_half_width_samples: int,
    ddof: int = 0,
    eps: float = 1e-12,
) -> PaperSNR:
    """Input SNR used in the revised manuscript, channel by channel.

    SNR_in = max_t |Hilbert(clean)| / std(noisy off-pulse samples).
    The exclusion half-width is mandatory and must be recovered from the
    production analysis rather than guessed.
    """
    clean = np.asarray(clean, dtype=np.float64)
    noisy = np.asarray(noisy, dtype=np.float64)
    if clean.shape != noisy.shape or clean.ndim < 1:
        raise ValueError("clean and noisy must have the same shape and a time axis")
    if exclude_half_width_samples < 0:
        raise ValueError("exclude_half_width_samples must be non-negative")

    leading_shape = clean.shape[:-1]
    n_time = clean.shape[-1]
    flat_clean = clean.reshape(-1, n_time)
    flat_noisy = noisy.reshape(-1, n_time)

    envelope = np.abs(hilbert(flat_clean, axis=-1))
    peak_index = np.argmax(envelope, axis=-1)
    clean_peak = np.max(envelope, axis=-1)
    sigma_off = np.empty(flat_clean.shape[0], dtype=np.float64)

    for i, peak in enumerate(peak_index):
        lo = max(0, int(peak) - exclude_half_width_samples)
        hi = min(n_time, int(peak) + exclude_half_width_samples + 1)
        mask = np.ones(n_time, dtype=bool)
        mask[lo:hi] = False
        off_pulse = flat_noisy[i, mask]
        if off_pulse.size <= ddof:
            raise ValueError("The exclusion window leaves too few off-pulse samples")
        sigma_off[i] = np.std(off_pulse, ddof=ddof)

    sigma_off = np.maximum(sigma_off, eps)
    return PaperSNR(
        snr=(clean_peak / sigma_off).reshape(leading_shape),
        clean_peak=clean_peak.reshape(leading_shape),
        sigma_off=sigma_off.reshape(leading_shape),
        clean_peak_index=peak_index.reshape(leading_shape),
    )


@dataclass(frozen=True)
class BinnedSummary:
    center: np.ndarray
    median: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    count: np.ndarray


def binned_summary(
    x: np.ndarray,
    y: np.ndarray,
    bins: Iterable[float],
    *,
    interval: tuple[float, float] = (0.05, 0.95),
) -> BinnedSummary:
    """Median, named population interval, and count in fixed bins."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    bins = np.asarray(list(bins), dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if bins.ndim != 1 or bins.size < 2 or np.any(np.diff(bins) <= 0):
        raise ValueError("bins must be a strictly increasing one-dimensional sequence")

    q0, q1 = interval
    if not 0 <= q0 < q1 <= 1:
        raise ValueError("interval must satisfy 0 <= q0 < q1 <= 1")

    n_bin = bins.size - 1
    median = np.full(n_bin, np.nan)
    lower = np.full(n_bin, np.nan)
    upper = np.full(n_bin, np.nan)
    count = np.zeros(n_bin, dtype=np.int64)
    finite = np.isfinite(x) & np.isfinite(y)

    for i in range(n_bin):
        in_bin = finite & (x >= bins[i]) & (x < bins[i + 1])
        values = y[in_bin]
        count[i] = values.size
        if values.size:
            median[i] = np.median(values)
            lower[i], upper[i] = np.quantile(values, [q0, q1])

    return BinnedSummary(
        center=0.5 * (bins[:-1] + bins[1:]),
        median=median,
        lower=lower,
        upper=upper,
        count=count,
    )
