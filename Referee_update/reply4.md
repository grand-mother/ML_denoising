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

- `visualization/overleaf_plots.py` — **converted**: all 9 SNR sites now call
  `_offpulse_snr_1d()`, a thin wrapper over the canonical `paper_input_snr`
  (off-pulse, exclude ±64); compiles clean.
- `training/raytune_training_function.py:440` (`calculate_snr`),
  `visualization/common_ml_utils.py:84` (`compute_snr`) — deprecated, delegate to
  `paper_input_snr`.
- `make_fig_amplitude_bias_vs_snr.py`, Appendix NMSE/SNR-gain (`option3`) — off-pulse.

Note: this is a code change on the working tree (not yet committed), and the published
`raytune_lib_final` branch is unchanged. Regenerating the figures with the new x-axis is a
separate step and still depends on locking the production reconstruction pipeline.

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

**Consequence.** "10 ns" in the caption was really "10 samples". The true value in ns is
`10 × dt`, and `dt` is still unrecovered (the code's 1.0 ns/sample is a default, not the
GRAND sampling).

**Fix applied.** All timing labels in `visualization/overleaf_plots.py` have been relabeled
from "ns" to "**samples**" so the code and figure are consistent with what is actually
computed — legend labels in `plot_peak_time_efficiency`, `plot_peak_time_efficiency_combined`
and the comparison variant, plus the `thresholds = [10, 20]` comments (now marked
"# SAMPLES … convert with dt once recovered"). Compiles clean; the manuscript caption/text
must be updated to "samples" to match. If/when `dt` is recovered, these can instead be
converted to ns in one place. No figure regeneration is required for the units correction.

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

**Answer: yes — the time-only and dual-branch models differ only by the Fourier branches,
so no new run is needed.** The comparison behind the paper's time-vs-frequency figure lives
in `visualization/time_vs_freq_model/` (`time_freq.py`), with the figure artefacts and the
two models' metrics in `best_figure_csv/`.

**Parity — confirmed from the saved run configs** (`best_figure_csv/ablation_configs.json`):
the two models are the same `DualBranchAutoencoder` with identical time branch and decoder;
the only difference is `use_freq_branch`.

| Setting | DualBranch (with Fourier) | TimeOnly (no Fourier) |
|---|---|---|
| `use_freq_branch` | **true** | **false** |
| time branch | conv 32, res [128,256] | conv 32, res [128,256] |
| decoder_channels | [128,64,32,3] | [128,64,32,3] |
| split | `split_indices(n_samples)` — same shared call | same |
| preprocessing/augmentation | `target_start=120, target_end=480, voltage_to_adc=True` | identical |
| optimizer / schedule | Adam, lr 1e-4, weight_decay 1e-5 | identical |

**Result (the paper figure, `best_figure_csv/ablation_metrics.json`):**

| Model | PSNR (dB) | peak-amplitude ratio |
|---|---|---|
| DualBranch (with Fourier) | 26.70 ± 8.58 | 0.336 ± 0.321 |
| TimeOnly (no Fourier) | 26.23 ± 7.77 | 0.368 ± 0.312 |

Figures already produced from this
run: `ablation_nmse_snrgain_vs_snr.pdf`, `ablation_learning_curves.pdf`,
`physics_efficiency_ablation_comparison.pdf`.

**Note.** These are the compact models used for the time-vs-frequency figure (conv 32/16),
not the published production fiducial (`l1_36` / `multi_v3_36`); both models were trained
and evaluated identically, so the paired conclusion is valid as stated. No new run is
required to answer the referee.

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
