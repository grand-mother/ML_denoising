import json
from pathlib import Path

import numpy as np
import torch

from referee_revision_utils import (
    PhaseAuditAccumulator,
    binned_summary,
    deterministic_split_indices,
    load_evaluation_manifest,
    paper_input_snr,
    rfft_frequency_mask,
    save_evaluation_manifest,
    save_phase_audit_json,
    wrapped_phase_difference,
)


def test_wrapped_phase_boundary():
    eps = 1.0e-3
    predicted = torch.tensor([-torch.pi + eps])
    target = torch.tensor([torch.pi - eps])
    residual = wrapped_phase_difference(predicted, target)
    torch.testing.assert_close(residual, torch.tensor([2.0 * eps]), atol=1e-6, rtol=0.0)


def test_phase_audit_detects_branch_crossing(tmp_path: Path):
    # Two-bin synthetic traces are inconvenient in the time domain, so construct
    # real signals with one phase-shifted Fourier component.
    n = 32
    t = torch.arange(n, dtype=torch.float32)
    clean = torch.cos(2.0 * torch.pi * 3.0 * t / n).reshape(1, 1, -1)
    reconstructed = torch.cos(2.0 * torch.pi * 3.0 * t / n + 0.2).reshape(1, 1, -1)

    audit = PhaseAuditAccumulator(phase_weight=0.4)
    audit.update(clean, reconstructed)
    result = audit.finalize()
    assert result.coefficient_count == n // 2 + 1
    assert result.direct_phase_l1 >= result.wrapped_phase_l1

    path = tmp_path / "audit.json"
    save_phase_audit_json(path, full_band=result, metadata={"test": True})
    payload = json.loads(path.read_text())
    assert payload["metadata"]["test"] is True


def test_rfft_frequency_mask():
    mask = rfft_frequency_mask(
        1024,
        0.5e-9,
        f_min_hz=50e6,
        f_max_hz=200e6,
    )
    assert mask.dtype == torch.bool
    assert int(mask.sum()) > 0


def test_evaluation_manifest_round_trip(tmp_path: Path):
    train, valid, test = deterministic_split_indices(100, seed=31415)
    path = tmp_path / "evaluation_manifest.npz"
    save_evaluation_manifest(
        path,
        train,
        valid,
        test,
        n_total=100,
        seed=31415,
        provenance="unit test",
    )
    loaded = load_evaluation_manifest(path)
    assert loaded.provenance == "unit test"
    assert np.array_equal(loaded.test, test)
    assert np.unique(np.concatenate([loaded.train, loaded.valid, loaded.test])).size == 100


def test_snr_uses_only_off_pulse_samples():
    n = 256
    clean = np.zeros((1, n))
    clean[0, 128] = 10.0
    noisy = np.ones((1, n))
    noisy[0, 120:137] = 1.0e6
    result = paper_input_snr(clean, noisy, exclude_half_width_samples=10)
    assert result.sigma_off.shape == (1,)
    assert result.sigma_off[0] < 1e-9


def test_binned_summary_reports_interval_and_counts():
    result = binned_summary(
        np.array([0.2, 0.8, 1.2]),
        np.array([1.0, 3.0, 5.0]),
        [0.0, 1.0, 2.0],
        interval=(0.05, 0.95),
    )
    assert np.array_equal(result.count, np.array([2, 1]))
    assert np.allclose(result.median, np.array([2.0, 5.0]))
