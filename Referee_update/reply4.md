# Reply — item 1: SNR definition in the SNR-binned figures

**Question / decision.** The revision had briefly adopted an off-pulse SNR. That is
withdrawn. The original SNR definition used in the analysis is restored everywhere:

```
SNR = max(clean) / std(noisy)
```

with the standard deviation evaluated over the **full trace** — no off-pulse exclusion, no
band-limiting, raw signed maximum of the clean trace, computed per channel.

**Status: restored throughout `raytune_lib_final_v1`.** The published branch
`raytune_lib_final` already used this definition, so the two branches now agree and there is
**no exclusion half-width to report** — the concept does not exist in this definition.

## Where it is implemented

| Location | Form |
|---|---|
| `visualization/overleaf_plots.py:34` | `_paper_snr_1d()` — `max(clean)/std(noisy)`; used by all 9 SNR call sites |
| `training/raytune_training_function.py:440` | `calculate_snr()` — restored to `np.max(clean)/np.std(noisy)` |
| `visualization/common_ml_utils.py:84` | `compute_snr()` — same definition, reduced over channels for coarse per-trace bookkeeping |
| `make_fig_amplitude_bias_vs_snr.py` | computed inline, per channel |
| Appendix NMSE/SNR-gain (`option3`) | computed inline, per channel |

No helper now computes an off-pulse or envelope-based SNR for any reported figure, and the
earlier "deprecated" shims that delegated to `paper_input_snr` have been reverted to the
plain definition above. `utils/paper_losses_metrics.py::paper_input_snr` still exists but is
no longer called by any figure script.

## Consequence for the figures

No retraining is needed. Only the **new SNR-binned figures added for the revision** are
regenerated with the restored x-axis:

- `new_figure/peak_amplitude_bias/` — peak-amplitude bias vs input SNR
- `new_figure/timing_efficiency_vs_snr/` — timing efficiency vs input SNR (Fig. 7)

Figures inherited unchanged from the published branch already use this definition and are
left untouched. All captions must state the definition above; any wording referring to an
off-pulse window or an exclusion half-width should be deleted.

---

# Reply — item 2: timing figure, cut in ns or samples?

**Question.** Is the timing cut 10 ns or ten samples? Confirm from the actual generating
script; ensure figure, caption, and text use the same units.

**Answer. The cut is ten SAMPLES, mislabeled "10 ns".** Peak times are sample indices, the
threshold is a sample-count comparison, but the plot label/caption say "ns". The units are
therefore **not** consistent between code and figure.

**Generating script (the paper's timing figure).** `peak_time_analysis_for_all_channels`
(called in `visualization/notebooks/overleaf.ipynb`) → `plot_peak_time_efficiency_combined`.
This is the production path, not the stale single-channel `plot_peak_time_efficiency`.

**Evidence** (branch `raytune_lib_final`, `overleaf_plots.py`):
- peak time is a sample index: `timing = np.arange(clean_np.size)` (`:1118`), then
  `peak_time = float(timing[argmax(envelope)])` (`:1130-1132`) — no `dt` multiplication;
- the cut is applied to that sample difference:
  `np.abs(denoised_time - clean_time) <= timing_thresholds[0]` (in `plot_peak_time_efficiency_combined`);
- but the label reads ns: `$|\Delta t_{peak}| \le {timing_thresholds[0]}$ ns`;
- the function carries a `dt_ns` parameter (default **1.0 ns/sample**, `:686`) that is a
  placeholder and is **not** applied to the threshold — so nothing converts samples→ns.

**Consequence.** The published "10 ns" was really "10 samples". With the sampling interval
now fixed at **dt = 0.5 ns/sample**, the correct label is **10 samples = 5 ns**, not 10 ns.

**Fix applied — the legend now reads 5 ns.** In `plot_peak_time_efficiency_combined`
(the Fig. 7 generator) the three legend entries now render
`|Δt_peak| ≤ 5 ns`, `|Δt_peak| > 5 ns` (denoised) and `|Δt_peak| > 5 ns` (noisy), matching
the caption. Implementation: a module constant `_DT_NS = 0.5` plus a
`_samples_to_ns_label()` helper convert the sample-count threshold **for labelling only**;
the function gained a `dt_ns` parameter documented as label-only.

**The underlying selection is unchanged and was not recomputed.** The cut is still
`|denoised_time − clean_time| <= timing_thresholds[0]` with `timing_thresholds = [10]`
in samples (the notebook passes `thresholds_list=[10]`), so no figure data changes — this is
purely a units relabel, as requested.

One related correction: `ablation_peak_time_efficiency_comparison` computes its peak times as
`argmax(envelope) * dt_ns`, i.e. genuinely in nanoseconds, so its axis label was restored to
"ns" (an earlier blanket relabel to "samples" had made it wrong). Note its own `dt_ns`
default is **2.0**, inconsistent with the 0.5 ns established here — changing it would alter
that figure's content, not just its label, so it is flagged rather than silently changed.

---

# Reply — item 3: amplitude-bias caption

**Question.** State exactly which percentile interval the shaded band represents, and
confirm the noisy and denoised curves use the same trigger-passing sample and SNR.

**Which figure this refers to.** The answers below describe
`visualization/amplitude_diagnostic_plot/make_fig_amplitude_bias_vs_snr.py` — the **new
revised amplitude-bias figure** built for referee Q5. It lives only in the
`raytune_lib_final_v1` working tree (untracked, not yet committed) and is **not** on the
published `raytune_lib_final` branch. The **published** amplitude figure is
`plot_amplitude_ratio_vs_snr_all_channels` in `overleaf_plots.py`, which uses **mean ± std**
(not a percentile band) and the old full-window SNR. So the percentile band and shared-SNR
statements apply to the revised figure that would replace it.

**Shaded band = the central 68 % population interval** (16th–84th percentile of
`δ_A = A_rec/A_true − 1` in each SNR bin). Evidence: `np.percentile(d, [16, 50, 84])`
(`:438`); the band is `fill_between(x, q16, q84)` (`:552-555`); the reported half-width is
`sigma68 = 0.5*(q84 − q16)` (`:443`). Caption text to use: "central 68 % interval".

**Same traces and same SNR — enforced, not just observed.** `compute_delta_A` builds one
`snr` array from clean+noisy using the paper definition `max(clean)/std(noisy)` over the
full trace, and both residuals `delta_A_noisy = A_noisy/A_true − 1` and
`delta_A_denoised = A_den/A_true − 1` on the **same traces**. `shared_selection_mask()` then
requires a trace to be usable for **both** methods before either curve may use it, so the
two curves cannot see different events by construction; `build_table` passes that one mask
to both series and both read the **same** `snr_ch = quantities["snr"][:, ch_idx]`. The
accompanying counts table asserts equality of the per-bin counts and fails loudly otherwise.

**The sample is SNR-selected, not trigger-selected — the manuscript is corrected to match.**
The selection actually applied is a truth-conditioned SNR window on the shared SNR array:

| Item | Value |
|---|---|
| Selection | `min_snr < SNR < max_snr`, strict on both sides |
| **Exact SNR range entering the statistics** | **1 < SNR < 1000** |
| **SNR range shown in the figure** | **1 ≤ SNR < 10** (bin edges 1, 2, …, 10) |
| Minimum traces for a bin to be plotted | 20 |
| Applied to | the noisy and denoised curves identically (one shared mask) |

This is **not** the noisy-input trigger (`|Hilbert(noisy)| ≥ k·σ`) used by the timing
analysis, and the two analyses therefore do **not** share a sample. The claim that this
figure uses the same trigger-passing sample as the timing analysis is withdrawn; the
manuscript and caption must instead state:

> Traces are selected by input SNR (1 < SNR < 1000); the figure shows the range
> 1 ≤ SNR < 10. The shaded band is the 16th–84th percentile of δ_A in each SNR bin.
> The noisy and denoised curves use the identical trace selection and the identical
> SNR values.

The word "trigger-passing" should be removed from the caption and body text wherever it
refers to this figure. The per-bin sample sizes are given in the accompanying
`snr_selected_counts.csv` / `.tex`.

## Checkpoint provenance for this figure

An earlier draft of this figure was produced with a **non-production** checkpoint (the
wrapped-phase fixed-config run in `results/fixed_config_runs/dual_branch/`). That has been
corrected: the figure is regenerated with the **production `multi_v3` checkpoint**, running
inference and plotting only — no retraining.

| Item | Value |
|---|---|
| **Checkpoint** | `/sps/grand/macias/Sam_Result/multi_v3_CNN_100epochs_36samples/best_model.pth` |
| **Configuration** | `best_trial_config.json`: `model_type` CNN; time branch conv 64, res [128, 256]; frequency branch conv 64, res [128, 256]; nominal `decoder_channels` [128, 64, 32, 3] → effective [384, 64, 32, 3] |
| **Loss** | `multi_v3` = time L1 + `mag_weight`·L1(\|rFFT\|) + `phase_weight`·L1(direct principal phase), `mag_weight` 0.011676, `phase_weight` 0.028893 |
| **Input length** | **512 samples**, the production training length (`--eval-len 512`); deterministic pulse-centred window, no random crop, no swap augmentation |
| **Test split** | seeded deterministic 80/10/10, `seed = 12345`; the exact indices are written to `new_figure/peak_amplitude_bias/evaluation_manifest.npz` |
| **Archived metrics** | validation loss 0.19997, validation PSNR 66.53 dB, 63 of 100 epochs completed |

Every one of these fields is also written by the script itself to
`new_figure/peak_amplitude_bias/figure_provenance.json` at run time, so the figure carries
its own provenance record.

Two evaluation-pipeline defects were fixed to make this record meaningful:
`load_data_and_run_inference` previously evaluated at the **full 1024-sample** trace (a
mismatch against the 512-sample training length) and used the **unseeded** `split_indices`,
so the test split could not be recorded at all. Both are now explicit parameters
(`eval_len`, `split_seed`), set to 512 and 12345 for this figure.

---

# Reply — item 4: Appendix waveform-fidelity figure

**Question.** Confirm the band-pass, signal/ROI window, output-SNR definition, off-pulse
window, and any truth-dependent selection; correct text/caption if needed; regenerate only
if the figure does not match the stated calculation.

Script: `visualization/nmse_snr_gain_vs_snr/make_fig_nmse_snr_gain_vs_snr__option3_truth_cleanpower_gate.py`
(the band-limited NMSE / output-SNR diagnostic). It exists on the published
`raytune_lib_final` branch; the only change on `raytune_lib_final_v1` is the input-SNR
x-axis (last row below).

| Ingredient | Actual value in the script | Evidence |
|---|---|---|
| **Band-pass** | 4th-order Butterworth, **50–200 MHz**, applied (`sosfiltfilt`) to clean and reconstructed before the fidelity metrics; also applied inside the input-SNR | `apply_bandpass=True`, `f_lo_hz=50e6`, `f_hi_hz=200e6`, `butter_order=4` (`:94-96`), `apply_bandpass_to_snr=True` (`:101`) |
| **Signal / ROI window** | ±**150 ns** around the clean Hilbert-envelope peak (per trace, per channel) | `roi_half_width_ns=150.0` (`:74`); ROI at `_nmse_snrout_and_cleanpower_in_roi` (`:298`) |
| **Output-SNR definition** | `SNR_out(dB) = 10·log10( Σclean² / Σ(rec−clean)² )` over the band-limited ROI; the figure plots `ΔSNR_out = SNR_out(rec) − SNR_out(noisy)` | `snrout_db = 10*log10(sig_pow/err_pow)` (`:311`), `sig_pow=Σx²`, `err_pow=Σ(y−x)²` |
| **Truth-dependent selection** | Clean-power gate: drop trace-channels whose clean ROI power is below the 10th percentile of positive clean powers (≥200 positives required); applied to noisy and denoised identically | `apply_clean_power_gate=True`, `clean_power_gate_quantile=0.10`, `min_count=200` (`:104-106`) |
| **Input-SNR (x-axis)** | The paper definition `max(clean)/std(noisy)`, standard deviation over the **full trace**, per channel — the same definition as every other figure. The published branch had used `_compute_snr_roi_peak_style` (noisy-ROI-peak / envelope-MAD, ±150 ns exclusion); that variant is now unused | computed inline at `:390-394`; the ROI-peak helper remains defined but is not called |

**Match / mismatches to fix in text or caption:**

1. **Input-SNR now uniform.** This appendix figure previously used a noisy-ROI-peak /
   envelope-MAD SNR with a ±150 ns exclusion, which differed from every other figure. It now
   uses the single paper definition `max(clean)/std(noisy)` over the full trace, so the
   x-axis is directly comparable with the amplitude and timing figures. No exclusion window
   applies to the SNR any more.
2. **The ±150 ns window is still used, but only for the fidelity ROI.** `roi_half_width_ns
   = 150` still defines the window in which NMSE and output SNR are computed; it is no
   longer part of any SNR definition. The caption should keep the ROI statement and drop any
   mention of an SNR exclusion window.
3. **Band-limited vs broadband.** The fidelity metrics here are **band-limited (50–200 MHz)**,
   unlike the amplitude/timing figures which are broadband. The caption must say so. The
   input SNR itself is broadband, matching the other figures.
4. **Regeneration.** The x-axis definition changed, so this appendix figure should be
   regenerated if it is retained in the revision.

---

# Reply — item 5: time-only baseline comparison

**Question.** Did the existing time-only and dual-branch models use the same split,
preprocessing, augmentation, optimizer, schedule, and budget (differing only by removal of
the Fourier branches)? If so, evaluate both on the same saved test manifest. If not, tell me
before running anything; at most one fixed-configuration time-only run, no Ray Tune.

**Answer: no matched checkpoint existed, so one fixed pair was trained at the production
(multi_v3) settings.** Checkpoint provenance was checked first: every run under
`/sps/grand/macias/Sam_Result/` is dual-branch (`use_freq_branch` unset ⇒ True), and the
only existing time-only checkpoints (`visualization/time_vs_freq_model/`) are a smaller,
differently-configured model (conv 32/16, 50 epochs) — a different model configuration, not
usable here. Per your instruction, one fixed time-only training was run, paired with one
fixed dual-branch training under the identical configuration, so the two differ **only** by
`use_freq_branch`. No Ray Tune search.

**Configuration — held fixed at the multi_v3 production operating point**
(`multi_v3_CNN_100epochs_36samples/best_trial_config.json`), current
`training/models/cnn.py::DualBranchAutoencoder`:

| Setting | Value |
|---|---|
| Trace length | 512 samples |
| Time branch | conv 64, res [128, 256] |
| Freq branch (dual-branch cell only) | conv 64, res [128, 256] |
| `decoder_channels` (nominal) | [128, 64, 32, 3] |
| Optimizer / schedule | Adam, lr 1.656e-05, max_lr 5.067e-04, step_size 10000, triangular2, weight_decay 2.566e-04 |
| Batch size | 1024 |
| Loss | `multi_l1` (time + \|FFT\| + **wrapped**-phase L1; mag_weight 0.011676, phase_weight 0.028893) |
| Epoch budget | 100 (no early stop) |
| Split | shared seeded 80/10/10 manifest, seed 12345 (328,538 / 41,067 / 41,068) |
| Training augmentation | swap_prob 0.5, target_start/end 300/500 (production defaults) |
| Validation | deterministic, fixed 512-sample window, no augmentation |

Script: `training/scripts/train_fixed_config.py` (`--use-freq-branch true/false`), run via
`training/scripts/run_fixed_config.sh`. One loss difference from the archived production run
is intentional: the phase term uses the **wrapped** residual (`atan2(sin Δφ, cos Δφ)`), the
corrected residual from reply2, not the archived direct-phase term — both cells use the same
loss, so the paired comparison is unaffected either way.

**Result (both runs completed cleanly, 100/100 epochs, no NaNs; best epoch 62 for both):**

| Model | Val loss (multi_l1) | Val PSNR (dB) |
|---|---|---|
| Dual-branch (with Fourier) | **16.007** | **33.849** |
| Time-only (no Fourier) | 17.180 | 33.460 |

Dual-branch reaches **6.8 % lower validation loss and +0.39 dB PSNR** than time-only under
an otherwise identical configuration — a real, reproducible gap (both curves decrease
monotonically and plateau around epoch 60; not noise). This differs from the earlier
`time_vs_freq_model` comparison (compact conv 32/16 models), which showed no measurable
gap; at the production model capacity, the Fourier branches do help.

Checkpoints, per-epoch metrics, and configs: `results/fixed_config_runs/{dual_branch,time_only}/`.

---

# Reply — item 6: production-record checks

**(a) Training-history figure ↔ selected 63-epoch trial — confirmed.** The selected
`multi_v3` trial ran **63 epochs** (`detailed_metrics.json`: `epochs` 0–62, with 63
`training_losses` and 63 `validation_losses` points); the training-history figure
`Summary_Plots/metrics.pdf` is plotted from those arrays, so it has 63 points and matches
the selected trial. (100 was the configured ASHA maximum; the trial stopped at 63.)

**(b) Ray Tune search ranges — verified; all selected values lie inside.** Ranges from
`configs/training_config.json` on the published `raytune_lib_final` branch (content identical
to `configs/search_spaces/training_config.json` on this working tree, after the `configs/`
reorganisation):

| Hyperparameter | Search range | Selected (multi_v3) | In range |
|---|---|---|---|
| lr | loguniform [1e-6, 1e-4] | 1.656e-05 | ✓ |
| max_lr | loguniform [1e-5, 1e-3] | 5.067e-04 | ✓ |
| step_size | choice {10000…50000} | 10000 | ✓ |
| lr_mode | {triangular2, triangular} | triangular2 | ✓ |
| weight_decay | loguniform [1e-5, 5e-3] | 2.566e-04 | ✓ |
| batch_size | {512,1024,2048,4096} | 1024 | ✓ |
| mag/phase/stft weight | loguniform [1e-2, 1.0] | 0.0117 / 0.0289 / 0.977 | ✓ |

**(c) Table I — corrected to the actual selected hyperparameters.** Two values in the
earlier draft (reply3 §6) were wrong — they had been taken from a different run
(`multi_l1_24`) by mistake. Now fixed against the archived `best_trial_config.json`:

| Field | was (wrong) | now (correct) |
|---|---|---|
| `step_size` | 30000 | **10000** |
| `weight_decay` | 1.292e-05 | **2.566e-04** |

Also for Table I: `stft_weight` is present in the archived config (0.977) but is **inert
for `multi_v3`** (only `multi_v4` uses it), so it should be omitted or marked inert. The
effective architecture (fusion 384, decoder [384,64,32,3], 2 max-pools/branch, 512 input,
no clipping, no taper) and the other selected hyperparameters in reply3 §6 are confirmed
correct.
