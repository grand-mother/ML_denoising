# Minimal integration snippets

> **Status (2026-07-23): all four sections A–D are integrated in the repository.**
> The blocks below are the *original* target snippets (kept for reference of intent).
> Each section now carries an **`▶ Implementation`** note stating where the code
> actually lives, how it differs from the snippet, and what still needs a
> production value.
>
> Import path note: the snippets show `from referee_revision_utils import ...`
> (the top-level layout of the `update_loss/` delivery package). In the
> integrated tree the same functions are imported from
> **`utils.referee_revision_utils`** and **`utils.paper_losses_metrics`**.
> `update_loss/` retains a duplicate copy of these modules — `utils/` is the
> integrated source of truth.

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

> **▶ Implementation — DONE.**
> Realized as a standalone CLI script:
> - **Code:** [`evaluation/phase_boundary_audit.py`](../evaluation/phase_boundary_audit.py)
>   — imports `PhaseAuditAccumulator`, `rfft_frequency_mask`, `save_phase_audit_json`
>   from `utils.referee_revision_utils` (audit loop at lines 158–198). `phase_weight`
>   is read from the checkpoint's `best_trial_config.json`, not hard-coded.
> - **Runners:** [`evaluation/run_phase_audit.sh`](../evaluation/run_phase_audit.sh),
>   [`evaluation/run_phase_audit_multi_v3.sh`](../evaluation/run_phase_audit_multi_v3.sh).
> - **Outputs:** `results/phase_boundary_audit/phase_boundary_audit.json`,
>   `results/phase_boundary_audit_multi_v3/phase_boundary_audit.json` (full band +
>   50–200 MHz band, plus metadata).
> - **Job logs:** `evaluation/phase_audit_54261743.out`, `evaluation/phase_audit_v3_54898016.out`.
> - **Write-up:** [`docs/phase_boundary_audit_table.md`](phase_boundary_audit_table.md).
> - **Differs from snippet:** the script also recovers/creates the eval manifest
>   (Section B logic, `_get_manifest`) and evaluates on the *validation* indices.
> - **⚠ Still PROVISIONAL:** `--trace-length` (512) and `--sample-spacing-seconds`
>   (0.5e-9) are placeholders pending the task-1 production-config audit; the
>   50–200 MHz band result depends on the correct `dt`.

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

Evaluate both existing checkpoints through this same loader. Report common-test metrics. Do not quote best validation losses unless the validation indices were also identical.AC

> **▶ Implementation — DONE (two entry points).**
> - **Production training (frozen split created once, reused by every Ray trial):**
>   [`training/raytune_main_sept25.py`](../training/raytune_main_sept25.py) — calls
>   `deterministic_split_indices(...)` + `save_split_manifest(...)` from
>   `utils.paper_losses_metrics`, writes `<output_path>/split_manifest.npz`, and
>   builds the test loader with `CustomDataset(..., no_random=True, swap_prob=0.0)`
>   + `DataLoader(batch_size=1, shuffle=False, pin_memory=True)`.
> - **Post-hoc evaluation/audit:** [`evaluation/phase_boundary_audit.py`](../evaluation/phase_boundary_audit.py)
>   `_get_manifest()` (lines 74–106) — recovers an archived
>   `split_manifest.npz` / `evaluation_manifest.npz` if present, otherwise creates
>   one seeded post-hoc `evaluation_manifest.npz` via
>   `deterministic_split_indices` / `save_evaluation_manifest` /
>   `load_evaluation_manifest` from `utils.referee_revision_utils`, recording
>   `provenance`. Same pattern in [`training/scripts/compare_phase_loss.py`](../training/scripts/compare_phase_loss.py).
> - **Persisted manifests:** `results/phase_boundary_audit/evaluation_manifest.npz`,
>   `results/phase_loss_compare*/split_manifest.npz`.
> - **Signature note (unified 2026-07-23):** both `deterministic_split_indices`
>   implementations — `utils.referee_revision_utils` and `utils.paper_losses_metrics`
>   — now share the identical signature
>   `(n_total, *, train_fraction=0.8, valid_fraction=0.1, seed=12345)` and are
>   drop-in interchangeable. (The frozen `update_loss/` delivery snapshot still
>   carries the old `train_frac=`/`valid_frac=` names; it is not on the import path.)

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

> **▶ Implementation — DONE (centralized, superset of the snippet).**
> Rather than an inline one-line patch, the whole multi-domain objective was moved
> to a single canonical definition that uses the wrapped residual
> `atan2(sin Δφ, cos Δφ)`:
> - **Canonical loss:** [`utils/paper_losses_metrics.py`](../utils/paper_losses_metrics.py)
>   — `circular_phase_loss()` (wrapped residual) feeding `multi_domain_loss()` and its
>   `multi_domain_l1_loss()` / `multi_domain_mse_loss()` wrappers.
> - **Import site:** [`training/raytune_training_function.py`](../training/raytune_training_function.py)
>   (~lines 342–359) — the old in-file `torch.angle`-based phase terms were removed
>   and replaced by an import of the canonical functions; the header comment records
>   the referee-revision rationale.
> - **Wiring:** [`training/raytune_train_sept25.py`](../training/raytune_train_sept25.py)
>   `get_criterion()` (lines 147–161) — `multi_l1` / `multi_mse` dispatch to the
>   canonical losses, passing `phase_weighting=config.get("phase_weighting", "none")`.
> - **Reproduces manuscript:** default `phase_weighting="none"` gives the original
>   unweighted wrapped loss; `mag_weight`/`phase_weight` are still taken from the
>   archived config, no new Ray search.
> - **Superset:** the optional `"target_magnitude"` weighting + a comparison harness
>   ([`training/scripts/compare_phase_loss.py`](../training/scripts/compare_phase_loss.py),
>   [`training/scripts/eval_phase_loss_cells.py`](../training/scripts/eval_phase_loss_cells.py)
>   → `results/phase_loss_compare/`, `results/phase_loss_compare_1024/`) go beyond the
>   minimal snippet. Keep `phase_weighting="none"` for the fiducial retrain.

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

> **▶ Implementation — DONE (single shared SNR everywhere; task-4 values RESOLVED).**
> - **Canonical SNR:** [`utils/paper_losses_metrics.py`](../utils/paper_losses_metrics.py)
>   — `paper_input_snr()` returning a `PaperSNR` dataclass (`.snr`), plus the CONFIRMED
>   production constants `PRODUCTION_OFFPULSE_EXCLUDE_HALF_WIDTH = 64` and
>   `PRODUCTION_DDOF = 0` (old name `PROVISIONAL_OFFPULSE_EXCLUDE_HALF_WIDTH` retained
>   as a backward-compatible alias).
> - **Production plotting uses it directly:**
>   [`visualization/nmse_snr_gain_vs_snr/make_fig_nmse_snr_gain_vs_snr__option3_truth_cleanpower_gate.py`](../visualization/nmse_snr_gain_vs_snr/make_fig_nmse_snr_gain_vs_snr__option3_truth_cleanpower_gate.py)
>   and [`visualization/amplitude_diagnostic_plot/make_fig_amplitude_bias_vs_snr.py`](../visualization/amplitude_diagnostic_plot/make_fig_amplitude_bias_vs_snr.py).
> - **Legacy call sites now delegate (deprecated wrappers):**
>   [`visualization/common_ml_utils.py`](../visualization/common_ml_utils.py) `compute_snr()`
>   and [`training/raytune_training_function.py`](../training/raytune_training_function.py)
>   `calculate_snr()` both emit a `DeprecationWarning` and forward to `paper_input_snr`.
> - **Task-4 value recovery (audited + RESOLVED 2026-07-23):**
>   - **`ddof` → 0 (definitive).** Every live `np.std(...)` in the SNR path uses
>     `ddof=0`: both canonical modules default `ddof: int = 0`
>     ([`utils/paper_losses_metrics.py`](../utils/paper_losses_metrics.py) L206,
>     [`utils/referee_revision_utils.py`](../utils/referee_revision_utils.py) L258),
>     and the amplitude figure's inline estimator is `float(np.std(samples))`
>     (numpy default = 0,
>     [`make_fig_amplitude_bias_vs_snr.py`](../visualization/amplitude_diagnostic_plot/make_fig_amplitude_bias_vs_snr.py) L285).
>     No code passes `ddof=1`. → pinned as `PRODUCTION_DDOF = 0`.
>   - **`exclude_half_width_samples` → 64 samples (confirmed by the author).**
>     Both publication figures use **64 samples (±32 ns @ 0.5 ns/sample), std-based**:
>     - *Amplitude-bias figure:* `exclude_radius = 64` (script default; docstring
>       "for 0.5 ns sampling, exclude_radius=64 removes ±32 ns").
>     - *NMSE / SNR-gain (option3) figure:* **regenerated after the task-3 refactor**,
>       so its x-axis is the canonical `paper_input_snr(..., 64)` (std). The script's
>       old native `snr_exclude_half_width_ns = 150.0 ns` + MAD estimator is
>       **deprecated and no longer drives any published figure**
>       ([option3 script](../visualization/nmse_snr_gain_vs_snr/make_fig_nmse_snr_gain_vs_snr__option3_truth_cleanpower_gate.py) L206–208).
>     → pinned as `PRODUCTION_OFFPULSE_EXCLUDE_HALF_WIDTH = 64`; all live call sites
>     now import the confirmed constants and pass `ddof=PRODUCTION_DDOF` explicitly.
>   - **Residual `dt` caveat (task-1, non-blocking):** the *ns* label "64 samples =
>     ±32 ns" assumes dt = 0.5 ns/sample, still pending the task-1 production-config
>     audit. The SNR value uses the 64-sample count directly and is unaffected.
