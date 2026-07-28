# Phase-loss comparison figure — clarifications

Figure concerned: `results/phase_loss_compare_1024/phase_loss_val_curves.pdf`
(two-run A/B: validation-MSE curves for two training cells of the dual-branch network).

The figure was produced by the revision working tree (branch `raytune_lib_final_v1`),
scripts `training/scripts/compare_phase_loss.py` and `utils/paper_losses_metrics.py`;
all file:line references below are to that code.

---

**Q1. Does each run use the direct or the wrapped phase residual?**

**A.** Both runs use the **wrapped** residual,
`atan2(sin(Δφ), cos(Δφ))` with `Δφ = φ_rec − φ_true`. Neither run uses the direct
(unwrapped) residual. The two runs differ **only** in the per-frequency-bin weighting
applied inside the wrapped phase term: `none` versus `target_magnitude`.

*Evidence:* the wrapping is implemented in
`utils/paper_losses_metrics.py:22-25` (`wrapped_phase_difference`, the `atan2` form),
used by the training objective `multi_domain_l1_loss`
(`utils/paper_losses_metrics.py:136`). Each cell builds its criterion via
`get_criterion("multi_l1", cfg)` — `training/raytune_train_sept25.py:160-161` — which
passes `phase_weighting = cfg["phase_weighting"]`; the per-cell mode is set in
`train_one_cell` (`training/scripts/compare_phase_loss.py:136`, criterion built at
`:145`).

---

**Q2. What does "none" mean?**

**A.** "none" means **uniform (unweighted) per-bin weighting**: every RFFT coefficient
contributes equally to the wrapped phase loss. It does **not** mean a zero phase term,
and it does **not** mean the direct residual. ("target_magnitude" down-weights bins in
proportion to the clean spectrum magnitude, `w(f) = |X_true(f)| / max_f |X_true(f)|`.)

*Evidence:* `utils/paper_losses_metrics.py:38` (`circular_phase_loss`); the `none` path
sets `weights = torch.ones_like(target_mag)` (`:75`); the `target_magnitude` path sets
`weights = target_mag / scale` (`:77-79`).

---

**Q3. Do both runs use the same initialization and the same saved split?**

**A.** Yes to both.
- *Initialization:* the same seed is set immediately before model construction for each
  cell, so both models start from identical weights.
- *Split:* both cells use the single seeded 80/10/10 split saved as
  `split_manifest.npz` (seed 12345). Validation is deterministic: no trace swapping,
  no random cropping, `shuffle = False`.

*Evidence:* `set_seed` defined at `training/scripts/compare_phase_loss.py:97` and called
at `:139`, immediately before `DualBranchAutoencoder(...)` in `train_one_cell`. The
split is created once with `deterministic_split_indices(..., seed=12345)` and persisted
by `save_split_manifest` (`compare_phase_loss.py:278-281`). Determinism of the loaders:
`swap_prob=0.0, no_random=True` for both datasets (`:105-106`) and `shuffle=False` for
the validation loader (`:111`).

---

**Q4. Which training objective produced the plotted phase-independent MSE?**

**A.** The plotted curves are the **time-domain validation MSE**, computed with the same
metric code for every cell; it is used as a phase-independent *selection* metric
precisely because the training criteria differ between cells. The **training** objective
of each cell was the multi-domain L1 (time L1 + Fourier-magnitude L1 + **wrapped**
Fourier-phase L1) with that cell's weighting mode. The criterion value itself is not
plotted because it is not comparable across cells.

*Evidence:* the metric is `F.mse_loss(pred, clean)` inside `validate`
(`training/scripts/compare_phase_loss.py:116-125`, MSE at `:125`), identical for every
cell; `aggregate` plots that column (`:222-224`, y-label "validation MSE
(phase-independent)"). The training objective is built at `:145`
(`get_criterion("multi_l1", cfg)` → `multi_domain_l1_loss`,
`utils/paper_losses_metrics.py:136`).

---

**Q5. Why does the title say that only the phase weight varies, when the displayed
phase weights are identical?**

**A.** The title was **stale and incorrect** — it was written for an earlier grid
version of the script that scanned several phase-weight values. In the two-run version
the phase weight is **fixed** at the production value for both cells, and only the
per-bin weighting **mode** (`none` vs `target_magnitude`) varies. The title has been
corrected (`training/scripts/compare_phase_loss.py:226`) to:

> *"Wrapped phase loss: per-bin weighting mode A/B (fixed architecture, fixed phase
> weight, same init and saved split)."*

---

## Scope note

Per the latest instruction, the revision is limited to the referee's ±π wrapping
question; the magnitude-weighting comparison is out of scope. This figure can therefore
be **dropped from the revision**. The deliverable replacing it is the compact,
inference-only table comparing the **direct** and **wrapped** residuals on the existing
validation predictions (mean phase L1, fraction of coefficients with |Δφ_raw| > π, and
the weighted contribution to the total validation loss), reported for the full Fourier
range and for the 50–200 MHz band, computed by
`evaluation/phase_boundary_audit.py` with
`utils/referee_revision_utils.py::PhaseAuditAccumulator`. No phase-weight scan, no new
Ray search, no magnitude weighting.
