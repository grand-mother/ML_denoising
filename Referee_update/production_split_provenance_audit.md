# Audit: can the production trial's train/valid/test indices be recovered?

**Date:** 2026-08-08. **Question:** recover the exact split used by the selected
production Ray trial `ce8f7_00029`, save it as a manifest, and check whether the
seed-12345 evaluation split overlaps it. Nothing was rerun or retrained.

## Answer: the indices are NOT recoverable, and the overlap is ~80 %.

### What was searched

| Location | Result |
|---|---|
| `.../multi_v3_CNN_100epochs_36samples/` | `best_model.pth`, 3 JSONs, `training_params.json`, `Summary_Plots/`, `Test_Sample_plots/` — no indices |
| Archived Ray dir `train_validate_wrapper_2025-10-28_03-14-17/` (all 36 trials retained) | present, but see below |
| Trial `ce8f7_00029/`: `params.json`, `params.pkl`, `progress.csv`, `result.json`, `events.out.tfevents.*` | only hyper-parameters and per-epoch metrics; `result.json` keys are `loss, validation_loss, validation_psnr, training_iteration, …` — no split record |
| `experiment_state-*.json` | strings `indices`, `split`, `seed`, `manifest` occur **0** times |
| 100 × `checkpoint_0000NN/data.pkl` | keys are `epoch` and `net_state_dict` only; 25 MB ≈ model (8.4 MB) + Adam moments. `indices`/`split`/`seed` occur 0 times |
| Whole of `/sps/grand/macias/Sam_Result/` | one `split_manifest.npz`, but it belongs to `multi_l1_CNN_100epochs_24samples_1024` (a later run) and already uses seed 12345 — not the production run |
| Every directory matching `*multi_v3*` anywhere on `/sps/grand/macias`, `/sps/grand/blevy`, `/pbs/home/o/omacias/Sam_project` | two exist. The archive above has no split record of any kind. `results/phase_boundary_audit_multi_v3/evaluation_manifest.npz` is an *evaluation* manifest written after the fact, and its own `provenance` field already states: **"post-hoc common evaluation; original production indices unavailable"** (written 2026-07-22) |

### Why no multi_v3 run can have one

The frozen-split mechanism lives in `raytune_lib_final/training/raytune_main_sept25.py:64-75`
— the split is built **once** at the outer level with `deterministic_split_indices(seed=12345)`
and persisted with `save_split_manifest` before any trial starts. Exactly three
`split_manifest.npz` files exist anywhere, and their `train` **and** `test` index arrays are
byte-identical to one another (verified with `np.array_equal`):

| run | date | belongs to |
|---|---|---|
| `Sam_Result/multi_l1_CNN_100epochs_24samples_1024` | 2026-07-19 | `multi_l1`, 1024-sample traces, 24 Ray samples |
| `results/phase_loss_compare_1024` | 2026-07-19 | phase-loss comparison |
| `results/fixed_config_runs` | 2026-07-29 | the item-5 ablation |

None is `multi_v3`. The mechanism first took effect on **2026-07-19**; the production run is
from **2025-10-28** and was launched from `raytune_lib_sept25`, nine months earlier. So the
production indices were never written, by construction.

A checkpoint that *is* genuinely held out with respect to the seed-12345 split does exist —
`multi_l1_CNN_100epochs_24samples_1024`, trial `aa2ed_00014` — but it is a different model
(loss `multi_l1` rather than `multi_v3`, 1024-sample traces rather than 512, batch 512,
lr 2.16e-6, step_size 40000, 24 Ray samples). Substituting it would change the model the
paper reports, so it is not a drop-in fix for the figures.

### Why they cannot be reconstructed

Production training ran from `/pbs/home/o/omacias/Sam_project/raytune_lib_sept25/`
(per `training_params.json`, whose `training_config_path` points there), using
`raytune_main_sept25.py`. In that script the split is made **inside**
`train_validate_wrapper`, i.e. independently in every trial:

```python
train_indices, valid_indices, test_indices = split_indices(
    total_samples_train, train_frac=0.8, valid_frac=0.1)     # :136
```

and `split_indices` is unseeded:

```python
indices = np.arange(n)
np.random.shuffle(indices)        # no seed, no manifest written
```

There is no `np.random.seed`/`torch.manual_seed` anywhere in that path. So the
permutation depended on each Ray worker's numpy global RNG state at that moment,
which was never recorded. Each of the 36 trials therefore used a *different*
unrecorded split. The indices are gone.

### Overlap with the seed-12345 evaluation split

The production split and the seed-12345 split are statistically independent, so
each of the 41,068 evaluation traces had probability 0.8 of being in the
production training set:

| Quantity | Value |
|---|---|
| N (all traces) | 410,673 |
| seed-12345 test split | 41,068 |
| **Expected overlap with production training data** | **≈ 32,854 traces (80 %)** |
| Binomial s.d. | 81 traces (0.25 %) |
| P(no overlap) | 0.2^41068 ≈ 10^−28705, i.e. zero |

**The seed-12345 split is therefore not held out with respect to the production
`multi_v3` checkpoint.** About four fifths of it was seen during that training.

### What this does and does not affect

- **Affected — figures made with the production checkpoint on the seed-12345
  split:** peak-amplitude bias (item 3), timing efficiency (item 2/Fig. 7),
  appendix band-limited fidelity (item 4), and the example waveform figure. These
  are in-sample for ~80 % of the traces shown, so they should be described as
  evaluated on a *reproducible, recorded subset*, not as held-out generalisation
  estimates. Note this was equally true of the originally published versions of
  those figures — the published notebook also called the unseeded
  `split_indices()` — so this is a pre-existing description issue, not something
  introduced by the revision.
- **Not affected — the time-only ablation (item 5).** Both fixed-config cells were
  trained through `train_fixed_config.py`, which builds and writes the seeded
  manifest (`results/fixed_config_runs/split_manifest.npz`, seed 12345) before
  training, so their 41,068-trace test split is genuinely disjoint from their own
  training data. That comparison stands as an unbiased held-out result.

### Options (no work done on these yet)

1. Re-word the captions/reply for the affected figures: "evaluated on a recorded,
   reproducible subset of 41,068 traces (seed 12345)" rather than "held out".
2. Retrain the production configuration once against the seeded manifest so that
   every figure is genuinely out-of-sample. This is a real retraining run.
3. Leave as is and state the limitation explicitly.

---

# Second question: did production training use gradient clipping at norm 1.0?

**No.** The archived script in this repository does, but that is a later version
than the one that ran.

| File | Backward pass | mtime |
|---|---|---|
| `raytune_lib_sept25/raytune_train_sept25.py` — **the version that ran** | `loss.backward()` → `optimizer.step()`, **no clipping** (`grep clip` returns nothing) | **2025-10-28 03:04:08** |
| `raytune_lib_final/training/raytune_train_sept25.py` — archived copy | `loss.backward()` → `clip_grad_norm_(..., max_norm=1.0)` → `optimizer.step()` (`:270`) | later |

The selected trial's directory is `train_validate_wrapper_2025-10-28_03-14-17`,
i.e. the run started **10 minutes after** that file was last written, which ties
the un-clipped version to the production run.

Evidence chain (five independent points, all consistent):

1. `training_params.json` in the multi_v3 archive points `training_config_path` and
   `model_config_path` at `raytune_lib_sept25/params_for_training/`.
2. `raytune_lib_sept25/raytune_main_sept25.py:26` imports `train_validate` from
   `raytune_train_sept25`, and calls it at `:151`.
3. `grep -rn clip_grad` over the **entire** `raytune_lib_sept25` tree returns nothing.
4. That file's mtime is 2025-10-28 03:04:08; the trial directory is
   `..._2025-10-28_03-14-17`, ten minutes later.
5. This repository, which did contain the clipping call, has initial commit
   2026-01-12 — two and a half months after the production run.

Caveat: Ray's `params.json` records hyper-parameters only and has no field for
clipping, so this rests on code provenance, not on a logged value.

**Action taken (2026-08-08):** the `clip_grad_norm_(max_norm=1.0)` call was removed
from `training/raytune_train_sept25.py` so the archived script no longer
misrepresents the production run, with a comment recording why it must not be
reinstated. `grep -r clip_grad` over this repository now returns only that comment.

**Training loop realigned to production (2026-08-08).** Two further differences
were removed at the same time, so that `train_validate`'s inner loop is now
line-for-line what the production run executed (`zero_grad → forward → loss →
backward → step`):

- the three NaN/Inf batch guards (`continue` on non-finite input, output or loss),
  which the executed version did not have;
- the `valid_batches` denominator: the epoch loss is again
  `total_train_loss / len(train_loader)`, as in production, rather than being
  divided by the number of non-skipped batches.

**Legacy criterion names restored (2026-08-08).** The archived configs carry
`"criterion": "multi_v3"`, a name the revision had renamed to `multi_l1` — so
loading the production config raised `ValueError: Unknown criterion: multi_v3`
and the archived checkpoints could not be reproduced at all. The original losses
are restored in `training/legacy_losses.py` and wired into `get_criterion` under
their archived names (`multi_v2`, `multi_v3`), each marked as reproduction-only.

They are **not** the current losses: the legacy phase term applies L1/MSE directly
to `torch.angle`, which is discontinuous at ±π and mis-scores errors across the
wrap-around; the current `multi_l1`/`multi_mse` use the wrapped residual
`atan2(sin Δφ, cos Δφ)` from reply2. Both remain available, and new training
should use the corrected ones. (`multi_v4` is deliberately not restored — no
archived checkpoint used it.)

**Still outstanding — one remaining discrepancy, not changed:**

- `training/scripts/train_fixed_config.py:200-201` (the item-5 ablation) writes its
  own training loop — `loss.backward()` then `optimizer.step()` — and does not go
  through `train_validate`. It never clipped and has no NaN guards, so it already
  matches production behaviour, but the repository does contain two training loops
  and any methods text should say which one it describes.
