import numpy as np
import torch

from paper_losses_metrics import (
    binned_summary,
    deterministic_split_indices,
    paper_input_snr,
    wrapped_phase_difference,
)


def test_wrapped_phase_boundary():
    true = torch.tensor([torch.pi - 0.01])
    pred = torch.tensor([-torch.pi + 0.01])
    delta = wrapped_phase_difference(pred, true)
    assert torch.allclose(delta.abs(), torch.tensor([0.02]), atol=1e-6)


def test_snr_uses_off_pulse_samples_only():
    n = 256
    clean = np.zeros((1, n))
    clean[0, 128] = 10.0
    noisy = np.ones((1, n))
    noisy[0, 120:137] = 1.0e6  # must be excluded from sigma_off
    out = paper_input_snr(clean, noisy, exclude_half_width_samples=10)
    assert out.sigma_off.shape == (1,)
    assert out.sigma_off[0] < 1e-9  # constant off-pulse baseline


def test_deterministic_splits_are_reproducible_and_disjoint():
    a = deterministic_split_indices(100, seed=12345)
    b = deterministic_split_indices(100, seed=12345)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    joined = np.concatenate(a)
    assert np.unique(joined).size == 100


def test_binned_summary_reports_counts():
    out = binned_summary(
        x=np.array([0.2, 0.8, 1.2]),
        y=np.array([1.0, 3.0, 5.0]),
        bins=np.array([0.0, 1.0, 2.0]),
    )
    assert np.array_equal(out.count, np.array([2, 1]))
    assert np.allclose(out.median, np.array([2.0, 5.0]))
