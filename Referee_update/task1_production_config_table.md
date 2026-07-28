# Task 1 — Architecture and training description: manuscript vs production

Scope: the five architecture/training statements listed in the revision plan, resolved
against the archived production run. **The paper is corrected to describe the production
model; the code is not changed to match the draft.** No new calculation is required.

**Archived production code (what trained the published checkpoints):**
`/pbs/home/o/omacias/Sam_project/raytune_lib_sept25/`

**Public code (what a reader of the repository sees):**
`github.com/grand-mother/ML_denoising`, branch `raytune_lib_final`, commit `f088f237`
(2026-06-10). All "public" line numbers below are from this commit; the local working
tree carries uncommitted revision work and does not match it.

**Published checkpoints (100 epochs, 36 Ray Tune trials each), in `/sps/grand/macias/Sam_Result/`:**
`l1_CNN_100epochs_36samples`, `multi_v3_CNN_100epochs_36samples`,
`multi_v2_CNN_100epochs_36samples`.

Items 1–4 below are identical for all three runs (they are properties of the production
code). Item 5 is per-run and is tabulated separately.

| # | Manuscript statement | Production value | Evidence (file : line) | Paper correction |
|---|---|---|---|---|
| 1 | Trace length **1,024** samples | **512-sample window.** Simulations are stored at 1,024 samples; the dataset crops a 512-sample window containing the pulse. | production `raytune_lib_sept25/raytune_training_function.py:26` (`traces_len=512`) with the `[idstart:idstart+traces_len]` crop in `__getitem__`; public `training/raytune_training_function.py:29` | Replace 1,024 with **512** as the network input length; state that simulations are stored at 1,024 samples. |
| 2 | Gradient clipping at max norm **5** | **No gradient clipping.** The production step is `loss.backward(); optimizer.step()`; no `clip_grad` call exists in the production tree. | production `raytune_lib_sept25/raytune_train_sept25.py:244-247`; (public `training/raytune_train_sept25.py:267` adds `clip_grad_norm_(..., max_norm=1.0)`, introduced **after** production) | Replace "clipped at 5" with **no gradient clipping**. |
| 3 | **Hann taper** applied before the FFT | **No taper.** The real FFT is applied directly to the input window. | production `raytune_lib_sept25/raytune_CNN.py:163` (`torch.fft.rfft(x, dim=-1)`); no window function in the file; public `training/models/cnn.py:176` | **Remove the Hann-taper statement.** |
| 4 | **Three** max-pooling operations per branch | **Two** per branch (`conv → ReLU → pool → residual → pool → residual`); total temporal downsampling factor **4**. | production `raytune_lib_sept25/raytune_CNN.py:75, 78` (time) and `:95, 98` (frequency); public `training/models/cnn.py:75, 78` and `:95, 98` | State **two** max-pooling operations per branch, downsampling factor **4**. |
| 5 | First decoder width `decoder_channels[0]` reported as a searched hyperparameter | **Overwritten and inert.** `decoder_channels[0] = fusion_channels = (concatenated encoder width) // 2`. The nominal first entry is never used. | production `raytune_lib_sept25/raytune_CNN.py:132`; public `training/models/cnn.py:139`; archived `best_trial_config.json` of each run | **List the effective fusion and decoder widths** (table below), not the overwritten nominal first channel. State that only `decoder_channels[1:]` were effective search dimensions. |

## Item 5 — effective fusion and decoder widths per published run

Computation, per the reference model **`training/models/cnn.py`** (public branch,
`f088f237`): each branch outputs `res_channels[1]` channels (`cnn.py:70` time, `:90`
frequency; the frequency branch is instantiated twice, for magnitude and phase);
`total_channels_before_fusion = time_out + magnitude_out + phase_out` (`cnn.py:121-123`);
`fusion_channels = total_channels_before_fusion // 2` (`cnn.py:129`);
`decoder_channels[0] = fusion_channels` (`cnn.py:139`); the fusion layer is applied in the
forward pass (`cnn.py:199-200`). The archived production model implements the identical
channel arithmetic (`raytune_lib_sept25/raytune_CNN.py:117-132`).

**Hard verification — measured from the trained checkpoints** (state-dict weight shapes
read directly from each `best_model.pth`; `Conv1d` weight = (out, in, k),
`ConvTranspose1d` weight = (in, out, k)):

| Run | `fusion_layer.0.weight` | decoder `conv1` chain | ⇒ fusion / effective decoder |
|---|---|---|---|
| `l1_CNN_100epochs_36samples` | (384, 768, 1) | (384,16,3) → (16,8,3) → (8,3,5) | **384 / `[384, 16, 8, 3]`** |
| `multi_v3_CNN_100epochs_36samples` | (384, 768, 1) | (384,64,3) → (64,32,3) → (32,3,5) | **384 / `[384, 64, 32, 3]`** |
| `multi_v2_CNN_100epochs_36samples` | (256, 512, 1) | (256,64,3) → (64,32,3) → (32,3,5) | **256 / `[256, 64, 32, 3]`** |

| Run (`/sps/grand/macias/Sam_Result/`) | criterion | time out | mag out | phase out | concat | **fusion** | nominal `decoder_channels` | **effective decoder widths** |
|---|---|---|---|---|---|---|---|---|
| `l1_CNN_100epochs_36samples` | `l1` | 256 | 256 | 256 | 768 | **384** | `[32, 16, 8, 3]` | **`[384, 16, 8, 3]`** |
| `multi_v3_CNN_100epochs_36samples` | `multi_v3` (time+mag+phase, L1) | 256 | 256 | 256 | 768 | **384** | `[128, 64, 32, 3]` | **`[384, 64, 32, 3]`** |
| `multi_v2_CNN_100epochs_36samples` | `multi_v2` (time+mag+phase, MSE) | 256 | 128 | 128 | 512 | **256** | `[128, 64, 32, 3]` | **`[256, 64, 32, 3]`** |

(Branch configs from each run's archived `best_trial_config.json`. `l1` and `multi_v3`
share the same encoder widths and therefore the same fusion width, 384; only their
effective `decoder_channels[1:]` differ. `multi_v2` uses a narrower frequency branch, so
its fusion width is 256.)

## Fiducial checkpoint identification

The three runs above are all dual-branch (time + frequency) networks; they differ only in
the training objective. The **fiducial time-plus-frequency model** for the paper is:

> **`/sps/grand/macias/Sam_Result/l1_CNN_100epochs_36samples`** (criterion `l1`),
> effective decoder widths `[384, 16, 8, 3]`.

Basis: it is the checkpoint used for the paper's amplitude/PSNR/timing figures (it was
the "best freq+time model" designated for the referee-Q5 amplitude diagnostic), and the
author confirms most figures come from the `l1` run.

**Open reconciliation (needs the author, affects the phase reply).** The Methods section
describes a **multi-domain** objective with a Fourier-**phase** term. No `l1` figure uses
a phase term. That description corresponds to **`multi_v3`** (`multi_l1`: time + magnitude
+ phase, L1), not to the fiducial `l1` run. So one of the following must be made explicit
in the paper:

- (i) the headline/fiducial figures are `l1`, and the multi-domain objective is described
  only as a variant that was explored (state which, if any, figures use `multi_v3` /
  `multi_v2`); or
- (ii) the fiducial is `multi_v3` and the amplitude/PSNR/timing figures must be
  re-attributed to it.

This choice — not the widths above — is what determines whether the referee's ±π phase
question requires the wrapped-loss reply (template §2) or only a clarification (template
§1), and on which checkpoint the phase-boundary audit must be run (`multi_v3`, using the
**production** model class, not the public `cnn.py`).
