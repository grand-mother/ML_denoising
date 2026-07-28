# Minimal integration snippets

## A. Phase-boundary audit on the existing checkpoint

Use the production validation loader with deterministic evaluation settings. Do not modify the loss or retrain before this audit.

```python
import torch

from referee_revision_utils import (
    PhaseAuditAccumulator,
    rfft_frequency_mask,
    save_phase_audit_json,
)

phase_weight = 0.4351633  # replace with the value from the archived best trial
sample_spacing_seconds = 0.5e-9  # replace if the production value differs
trace_length = 1024               # replace with the production value

full_audit = PhaseAuditAccumulator(phase_weight=phase_weight)
band_mask = rfft_frequency_mask(
    trace_length,
    sample_spacing_seconds,
    f_min_hz=50e6,
    f_max_hz=200e6,
)
band_audit = PhaseAuditAccumulator(
    phase_weight=phase_weight,
    frequency_mask=band_mask,
)

model.eval()
with torch.no_grad():
    for noisy, clean in validation_loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        reconstructed = model(noisy)
        full_audit.update(clean, reconstructed)
        band_audit.update(clean, reconstructed)

full_result = full_audit.finalize()
band_result = band_audit.finalize()
print("Full band:", full_result)
print("50--200 MHz:", band_result)

save_phase_audit_json(
    "phase_boundary_audit.json",
    full_band=full_result,
    analysis_band=band_result,
    metadata={
        "checkpoint": str(checkpoint_path),
        "phase_weight": phase_weight,
        "trace_length": trace_length,
        "sample_spacing_seconds": sample_spacing_seconds,
        "validation_manifest": str(manifest_path),
    },
)
```

## B. Common deterministic evaluation sample

Prefer the test indices archived with the production run. Save those exact indices if they are available. Only create a new seeded manifest if the original indices cannot be recovered.

```python
from referee_revision_utils import (
    deterministic_split_indices,
    load_evaluation_manifest,
    save_evaluation_manifest,
)

manifest_path = output_dir / "evaluation_manifest.npz"

if manifest_path.exists():
    manifest = load_evaluation_manifest(manifest_path)
else:
    train_idx, valid_idx, test_idx = deterministic_split_indices(
        total_samples,
        train_fraction=0.8,
        valid_fraction=0.1,
        seed=12345,
    )
    save_evaluation_manifest(
        manifest_path,
        train_idx,
        valid_idx,
        test_idx,
        n_total=total_samples,
        seed=12345,
        provenance="post-hoc common evaluation; original indices unavailable",
    )
    manifest = load_evaluation_manifest(manifest_path)
```

Construct the evaluation dataset with no augmentation:

```python
test_dataset = CustomDataset(
    clean_signals,
    [noise_signals],
    traces_len=production_trace_length,
    indices=manifest.test,
    no_random=True,
    swap_prob=0.0,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=evaluation_batch_size,
    shuffle=False,
    pin_memory=True,
)
```

Evaluate both existing checkpoints through this same loader. Report common-test metrics. Do not quote best validation losses unless the validation indices were also identical.

## C. Conditional wrapped-loss replacement

Use this only after Oscar reviews the phase audit and authorizes one retraining.

In `training/raytune_training_function.py`, replace only the phase line in the fiducial L1 objective:

```python
from referee_revision_utils import wrapped_phase_difference

phase_delta = wrapped_phase_difference(
    torch.angle(pred_fft),
    torch.angle(clean_fft),
)
phase_loss = phase_delta.abs().mean()
```

Keep the archived phase weight and all other settings fixed. Do not add magnitude weighting and do not run a new Ray search.

## D. Paper input SNR

```python
from referee_revision_utils import paper_input_snr

snr_result = paper_input_snr(
    clean_array,
    noisy_array,
    exclude_half_width_samples=PRODUCTION_EXCLUSION_HALF_WIDTH,
    ddof=PRODUCTION_DDOF,
)
snr_in = snr_result.snr
```

The two capitalized values must come from the production plotting code or archived analysis notes. Do not infer them.
