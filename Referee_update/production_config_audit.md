# Production configuration audit (referee revision, task 4)

**Purpose.** Resolve, against the archived production code/checkpoint rather than by
guessing, the five public-code / manuscript discrepancies raised in the revision
checklist. This document is the "recovered production configuration" to be
confirmed **before** the final production retraining.

**Sources**
- **Production run**: `/sps/grand/macias/Sam_Result/l1_CNN_100epochs_36samples`
  (l1 CNN, 100 epochs, 36 Ray Tune samples; trained 2025-10-26 → 10-28).
- **Production code**: `/pbs/home/o/omacias/Sam_project/raytune_lib_sept25/`
  (the archived run's config points its `training_config_path` /
  `model_config_path` here; model in `raytune_CNN.py` dated 2025-10-18, training
  loop `raytune_train_sept25.py` dated 2025-10-28 — i.e. inside the production window).
- **Public code**: this repository, `raytune_lib_final` (branch `raytune_lib_final_v1`).

All values below were read directly from source; no value was inferred.

---

## Summary table

| # | Item | Production run (recovered) | Public code (`raytune_lib_final`) | Manuscript | Verdict |
|---|------|----------------------------|-----------------------------------|-----------|---------|
| 1 | Trace length (samples) | **512** | 512 | 1024 | production = public = **512**; **manuscript (1024) is wrong** |
| 2 | Gradient clipping | **none** | clip `max_norm=1.0` | 5 | **three-way mismatch**: none / 1 / 5 |
| 3 | FFT taper | **none (direct rFFT)** | none (direct rFFT) | Hann taper | production = public = **no taper**; **manuscript (Hann) is wrong** |
| 4 | Max-pool ops per branch | **2** | 2 | 3 | production = public = **2**; **manuscript (3) is wrong** |
| 5 | `decoder_channels[0]` | **overwritten by `fusion_channels`** | overwritten | treated as searched | first decoder value is **not an active hyperparameter** |

**Bottom line:** items 1, 3, 4, 5 are identical between the production run and the
public code, so the **manuscript** is the side that is wrong. Item 2 (gradient
clipping) is the only place where the production run, the public code, and the
manuscript all disagree — the production model was trained with **no gradient
clipping at all**.

---

## Evidence, item by item

### 1. Trace length: 512 (not 1024)
- **Production**: `raytune_lib_sept25/raytune_training_function.py:26` —
  `CustomDataset(..., traces_len=512, ...)`; the trace is cropped to `traces_len`
  in `__getitem__` (`...[idstart:idstart+traces_len]`).
- **Public**: `training/raytune_training_function.py:29` — same `traces_len=512`
  default; crop at line 168. (Inference on the held-out set returns shape
  `(N, 3, 512)`.)
- **Manuscript**: 1024.
- **Note**: the raw simulation arrays are 1024 samples long
  (`(3, n_samples, 1024)`); the network sees a 512-sample crop.

### 2. Gradient clipping: none (not 1, not 5)
- **Production**: `raytune_lib_sept25/raytune_train_sept25.py:244–247` — the
  optimizer step is `loss.backward(); optimizer.step()` with **no**
  `clip_grad_norm_`; a repo-wide search for `clip_grad` in `raytune_lib_sept25`
  returns nothing. The production model was trained **without** gradient clipping.
- **Public**: `training/raytune_train_sept25.py:270` —
  `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` (added after
  production).
- **Manuscript**: 5.

### 3. FFT taper: none (direct rFFT, no Hann)
- **Production**: `raytune_lib_sept25/raytune_CNN.py:163` —
  `x_fft = torch.fft.rfft(x, dim=-1)`; no window function anywhere in the file.
- **Public**: `training/models/cnn.py:176` — `torch.fft.rfft(x, dim=-1)`; no Hann.
- **Manuscript**: a Hann taper is applied.

### 4. Max-pool operations per branch: 2 (not 3)
- **Production**: `raytune_lib_sept25/raytune_CNN.py` — `self.pool` is applied
  twice per branch (after `conv1` and after `res_block1`; lines 75, 78 for the
  time branch and 95, 98 for the frequency branch); `res_block2` is not followed
  by a pool.
- **Public**: `training/models/cnn.py:75, 78` (time) and `95, 98` (freq) — same
  two pools per branch.
- **Manuscript**: three max-pool operations.

### 5. `decoder_channels[0]` is overwritten (searched value is dead)
- **Production**: `raytune_lib_sept25/raytune_CNN.py:132` —
  `decoder_channels[0] = fusion_channels`, where
  `fusion_channels = total_channels_before_fusion // 2`. The first element of the
  searched `decoder_channels` list is therefore replaced and never used.
- **Public**: `training/models/cnn.py:139` — identical.
- **Search space**: `configs/model_config.json` (now `configs/search_spaces/model_config.json`
  after the working-tree `configs/` reorganisation) searches
  `decoder_channels ∈ {[128,64,32,3], [256,128,64,3], [64,32,16,3], [32,16,8,3]}`,
  but the first entry (128/256/64/32) has no effect on the built model.

---

## Additional finding (not one of the five, but affects reproducibility)

The public `training/models/cnn.py` fuses the frequency branch through an
**adaptive average pool** to a global context vector
(`F.adaptive_avg_pool1d(...)`, `cnn.py:187–188`), introduced in commit `f088f23`
(2026-06). The **production** model `raytune_lib_sept25/raytune_CNN.py` does **not**
contain this step. The current public architecture therefore differs from the
archived production model in how the frequency features are combined. The final
retraining must state explicitly which fusion is used.

---

## Recommended resolution (to confirm with Sam before the final run)

Per the checklist, each item is resolved by either correcting the code and
retraining, or correcting the manuscript.

| # | Item | Recommended action |
|---|------|--------------------|
| 1 | Trace length | Manuscript wrong → **correct manuscript to 512**, or adopt 1024 in code and retrain (science decision: does the pulse+RF response need the full 1024?). |
| 2 | Gradient clipping | Recovered production value = **none**. Decide the final value deliberately (none vs a documented `max_norm`) and record it; the "5" in the manuscript is unsupported. |
| 3 | FFT taper | Manuscript wrong → **correct manuscript to "no taper"**, or add a Hann taper in code and retrain if a taper is physically desired. |
| 4 | Max-pool count | Manuscript wrong → **correct manuscript to 2 per branch**. |
| 5 | `decoder_channels[0]` | Either **document that `decoder_channels[0]` is not searched** (it is overwritten by `fusion_channels`), or change the code to honor the searched value. |

**Open item (for 100% certainty on items 1 & 2):** the dataset file in
`raytune_lib_sept25` was last modified after the production run (2026-01-05), so
confirm `traces_len` and the absence of clipping directly from the archived
2025-10 training log in
`/sps/grand/macias/Sam_Result/l1_CNN_100epochs_36samples` before finalizing.
