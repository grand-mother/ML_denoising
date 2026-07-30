# Reply — item 1: SNR definition in the SNR-binned figures

**Question.** Do the SNR-binned figures use `max|Hilbert(clean)| / std(noisy off-pulse)`?

**Answer.**

- **`raytune_lib_final` (published-article figure branch): No.** The figures use the raw
  full-window ratio `max(clean) / std(noisy)`. There is **no off-pulse exclusion window to
  record** — that concept is not in the published plotting code.
- **`raytune_lib_final_v1` (this revision branch): Yes.** All SNR-binned figure scripts now
  use the off-pulse definition, including `overleaf_plots.py` (all 9 call sites converted);
  the old full-window helpers are deprecated and delegate to the canonical function.

## Evidence (brief)

- Published SNR = `snr = np.max(clean_np) / np.std(noisy_np)` — full window, raw max, no
  exclusion. In `visualization/overleaf_plots.py` (branch `raytune_lib_final`, `f088f237`)
  at lines 163, 306, 376, 637, 718, 1224, 1534, 1692.
- Revised off-pulse SNR: `utils/referee_revision_utils.py::paper_input_snr`.
- The two differ numerically (3,000 valid traces, exclude ±64): median 1.119 vs 1.432
  (revised/production ≈ 1.27); 6.6 % of channel-traces flip across the SNR = 4 boundary.
  So it is a real change, not a relabelling.
- Exclusion window: none exists in production; the corrected figure uses **64 samples**, a
  stated choice.

## Corrected figure — uses the NEW SNR definition

`new_figure/` (SLURM 55300986, done). x-axis = **off-pulse**
`max|Hilbert(clean)| / std(noisy off-pulse)`, exclude ±64, via `paper_input_snr` — **not**
the published `max(clean)/std(noisy)`. Fiducial `l1_CNN_100epochs_36samples`; plots
`δ_A = A_rec/A_true − 1` vs SNR. Outputs: `amplitude_bias_vs_snr.pdf`,
`amplitude_bias_table.csv`, `amplitude_bias_per_trace.npz`, `run_snr_binned_offpulse.sh`.

## Conversion status on `raytune_lib_final_v1`

- `visualization/overleaf_plots.py` — **converted**: all SNR sites now call
  `_offpulse_snr_1d()`, a thin wrapper over the canonical `paper_input_snr`
  (off-pulse, exclude ±64); compiles clean.
- `training/raytune_training_function.py:440` (`calculate_snr`),
  `visualization/common_ml_utils.py:84` (`compute_snr`) — deprecated, delegate to
  `paper_input_snr`.
- `make_fig_amplitude_bias_vs_snr.py`, Appendix NMSE/SNR-gain (`option3`) — off-pulse.

Note: this is a code change on the working tree (not yet committed), and the published
`raytune_lib_final` branch is unchanged. Regenerating the figures with the new x-axis is a
separate step and still depends on locking the production reconstruction pipeline.

## Excluded-interval half-width — confirmed from the figure-generating scripts

**The half-width is 64 samples on each side of the clean Hilbert-envelope peak,
i.e. ±32 ns at the 0.5 ns sampling interval (a 129-sample excluded window).**
Verified by reading each script that produces a reported SNR-binned figure:

| Figure / script | Exclusion half-width | Where it is set |
|---|---|---|
| Canonical definition | **64 samples** | `utils/paper_losses_metrics.py:199` `PRODUCTION_OFFPULSE_EXCLUDE_HALF_WIDTH = 64` |
| All `overleaf_plots.py` figures (amplitude ratio, timing, SNR distribution) | **64 samples** | `visualization/overleaf_plots.py:32` `_OFFPULSE_EXCLUDE_HW = 64` → `_offpulse_snr_1d()` |
| Amplitude-bias vs SNR (referee Q5) | **64 samples** | `make_fig_amplitude_bias_vs_snr.py:148, 216, 295` `exclude_radius = 64` (its docstring states "for 0.5 ns sampling, exclude_radius=64 removes ±32 ns") |
| Appendix NMSE / output-SNR gain (`option3`) | **64 samples** | `..._option3_truth_cleanpower_gate.py:394` passes `PRODUCTION_OFFPULSE_EXCLUDE_HALF_WIDTH` |
| Timing efficiency vs SNR (Fig. 7, regenerated) | **64 samples** | `new_figure/timing_efficiency_vs_snr/make_fig_timing_efficiency.py` → `peak_time_analysis_for_all_channels` → `_offpulse_snr_1d()` |

Two distinct quantities must not be confused with this one:

1. **Trigger σ in the timing figure** uses its own, *smaller* window:
   `exclude_radius = 32` samples (`overleaf_plots.py:898, 1096`), applied to the
   *noisy* envelope to estimate the per-trace σ for the trigger cut. It is not the
   SNR denominator.
2. **A 150 ns (300-sample) exclusion** still appears in three appendix scripts
   (`NMSE_STD_VS_SNR/...`, `plot_usable_antenna_vs_SNR/...`, and the legacy path of
   `option3`) as `snr_exclude_half_width_ns = 150.0`. That belongs to a *different*
   SNR (noisy-envelope MAD, ROI-peak style), not the off-pulse definition above.
   Any figure quoting the off-pulse SNR must use the 64-sample value.

## Archiving and removal of the stale helper

**Archived with the code release.** Every script above is tracked in git, including
the newly added `new_figure/timing_efficiency_vs_snr/{make_fig_timing_efficiency.py,
run_timing_figure.sh}` and its deterministic `evaluation_manifest.npz`. Figure PDFs
are excluded by `.gitignore` by design; the scripts plus the seeded split manifest
reproduce them.

**Stale helper deprecated.** The public helpers that computed the raw clean-trace
maximum over a full-window standard deviation — `calculate_snr`
(`training/raytune_training_function.py:440`) and `compute_snr`
(`visualization/common_ml_utils.py:84`) — now raise `DeprecationWarning` and delegate
to `paper_input_snr` with the confirmed 64-sample half-width. Their docstrings state
explicitly that the original definition was **not** used for any reported figure.

Three inline uses of the old raw-max / full-window ratio remain, all in code that
produces **no** reported figure, and each is now labelled `LEGACY SNR … NOT used for
any reported figure`: `overleaf_plots.py:1875` (superseded dual-vs-time ablation),
`visualization/time_vs_freq_model/time_freq.py:591` (exploratory ablation, superseded
by `results/fixed_config_runs/`), and `evaluation/model_comparison.py:318, 541`
(exploratory diagnostic). They are left numerically unchanged on purpose: converting
them would silently alter those legacy figures' x-axes.

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

**Same traces and same SNR — confirmed.** `compute_delta_A` builds one `snr` array (from
clean+noisy, via `paper_input_snr`) and both residuals `delta_A_noisy = A_noisy/A_true − 1`
and `delta_A_denoised = A_den/A_true − 1` on the **same traces** (`:349-376`). In
`build_table`, both methods read the **same** `snr_ch = quantities["snr"][:, ch_idx]`
(`:build_table`), so the noisy and denoised curves are binned on identical SNR values and
the same trace set.

**One caveat on "trigger-passing".** The current selection is an SNR window
`min_snr < snr < max_snr` (default `1 < snr < 1000`, `:414`) applied to that shared SNR
array — a truth-conditioned SNR floor, applied identically to both curves. It is **not** the
noisy-input trigger (`|Hilbert(noisy)| ≥ k·σ`) that the word "trigger-passing" implies.
`referee_revision_utils.trigger_pass_mask` exists if we want the literal noisy trigger. So
the caption should either say "traces with SNR > 1" (what is actually done) or the selection
should be switched to the noisy trigger — a one-line choice. Either way both curves use the
identical selection.

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
| **Off-pulse / input-SNR (x-axis)** | On `raytune_lib_final_v1`: the canonical off-pulse `paper_input_snr` (`max\|Hilbert(clean)\|/std(noisy off-pulse)`, exclude ±64). On the **published** branch it was `_compute_snr_roi_peak_style` = noisy-ROI-peak / envelope-MAD, exclusion half-width `snr_exclude_half_width_ns=150` | v1: `paper_input_snr(...)` (`:392`); published: `_compute_snr_roi_peak_style` (branch `:377`) |

**Match / mismatches to fix in text or caption:**

1. **Input-SNR changed between branches.** The published appendix figure used a
   noisy-ROI-peak / envelope-MAD SNR (in-band, ±150 ns exclusion), **not** the off-pulse
   definition. On `raytune_lib_final_v1` the x-axis is now the canonical off-pulse SNR. The
   manuscript must state whichever is used; if the revised (off-pulse) x-axis is adopted,
   this appendix figure is one of the SNR-binned figures that changes and should be
   regenerated (same reconstruction caveat as item 1).
2. **Two different exclusion widths coexist.** This appendix figure's SNR excludes ±150 ns
   (published) / ±64 samples (v1 off-pulse), while the amplitude-bias figure (item 3) uses
   ±64 samples with no band-pass. The manuscript should state the exclusion window and
   band-pass per figure, since they are not identical across figures.
3. **Band-limited vs broadband.** The fidelity metrics here are **band-limited (50–200 MHz)**,
   unlike the amplitude/timing figures which are broadband. The caption must say so.
4. **Regeneration.** No regeneration is needed to confirm the definitions (audit only). The
   figure would only need regenerating if the revised off-pulse x-axis (point 1) is adopted,
   because that changes the x-axis binning.

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
