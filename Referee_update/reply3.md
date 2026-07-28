# Reply — manuscript-configuration table completed (multi_v3 objective, effective widths, training duration, SNR and ROI definitions)

**Request.** "In parallel, complete the manuscript-configuration table with the actual
multi_v3 objective, effective decoder widths, training duration, and exact SNR and
waveform-ROI definitions."

All four items are below, recovered from the archived run and the archived production code
(`/pbs/home/o/omacias/Sam_project/raytune_lib_sept25/`). Two of them produce a finding
that needs a decision, flagged in §5.

Run audited: `/sps/grand/macias/Sam_Result/multi_v3_CNN_100epochs_36samples`.

---

## 1. Actual multi_v3 objective

The published `multi_v3` criterion is a three-term multi-domain L1, with the phase term
taken as the **direct** difference of principal phases (no ±π wrapping):

```
L = L1(pred, clean)
  + mag_weight   * L1(|rFFT(pred)|,      |rFFT(clean)|)
  + phase_weight * L1(angle(rFFT(pred)), angle(rFFT(clean)))
```

| Item | Value / location |
|---|---|
| Definition | `raytune_lib_sept25/raytune_training_function.py:308` (`multi_domain_loss_v3`); phase line at `:319` |
| Selection | `raytune_lib_sept25/raytune_train_sept25.py:155-156` (`criterion == "multi_v3"`) |
| `mag_weight` | **0.011676** (archived best trial) |
| `phase_weight` | **0.028893** (archived best trial) |
| Reduction | element-wise mean (`F.l1_loss` default) over all channels and bins, DC and Nyquist included |
| No STFT term | `stft_weight` appears in the config but is used only by `multi_v4`; it is inert for `multi_v3` |

For the manuscript: `multi_v3` is the "multi-domain L1" objective, and its phase term is
the unwrapped principal-phase difference — the implementation the referee asked about.

## 2. Effective decoder widths

`decoder_channels[0]` is overwritten by the fusion width, so the configured first entry is
inert. Encoder outputs are concatenated as time + magnitude + phase, and the fusion layer
halves that width.

| Quantity | Value |
|---|---|
| time branch output | 256 |
| magnitude branch output | 256 |
| phase branch output | 256 |
| concatenated width | 768 |
| **fusion width** | **384** |
| nominal `decoder_channels` (config) | `[128, 64, 32, 3]` |
| **effective decoder widths** | **`[384, 64, 32, 3]`** |

Verified two ways — from the code (`raytune_lib_sept25/raytune_CNN.py:117-123` for the
concatenation and fusion width, `:132` for the overwrite) and directly from the trained
weights: `fusion_layer.0.weight = (384, 768, 1)`, `fusion_layer.2.weight = (384, 384, 3)`,
`decoder.0.conv1.weight = (384, 64, 3)`, `decoder.1.conv1.weight = (64, 32, 3)`,
`decoder.2.conv1.weight = (32, 3, 5)`.

Report `[384, 64, 32, 3]` in Table I, and state that only `decoder_channels[1:]` were
effective search dimensions.

## 3. Training duration

| Item | Value |
|---|---|
| Ray experiment start | **2025-10-28 03:14** (`train_validate_wrapper_2025-10-28_03-14-17`) |
| Best-trial artefacts written | **2025-10-30 00:19** |
| Wall-clock for the whole 36-trial search | **≈ 45 h** |
| Configured budget | 100 epochs, 36 Ray Tune trials |
| **Epochs actually completed by the selected trial** | **63** (`detailed_metrics.json`, epochs 0–62) |

Note for the manuscript: the selected trial ran **63 epochs, not 100**. The 100-epoch
figure is the configured maximum; the ASHA scheduler stopped the trial earlier. Table I
should quote the budget and the completed epochs separately, or quote 63.

## 4. Exact SNR and waveform-ROI definitions used for the published figures

Recovered from the production plotting code `raytune_lib_sept25/overleaf_plots.py`, which
generated the paper figures.

### SNR

```python
snr = np.max(clean_np) / np.std(noisy_np)      # overleaf_plots.py:160, :303, :373
```

| Property | Production value |
|---|---|
| Numerator | `max(clean)` — the **raw signed maximum** of the clean trace, *not* the Hilbert-envelope peak |
| Denominator | `std(noisy)` over the **entire 512-sample window** |
| Off-pulse exclusion | **none** — the pulse is included in the noise estimate |
| `ddof` | 0 (`np.std` default) |
| Band-limiting | **none** — no bandpass, no taper |
| Per channel | yes, computed independently for X, Y and Z |
| Trace selection | retained for `min_snr < snr < max_snr`, with `min_snr = 1`, `max_snr = 1e3` |

### Waveform ROI

There is **no ROI** in the production figure code. Amplitudes and peak times are taken as
the global extremum of the Hilbert envelope over the whole 512-sample window:

```python
envelope_clean = np.abs(hilbert(clean_np))     # overleaf_plots.py:166, :309, :378
peak_amp  = np.max(envelope_clean)
peak_time = timing[np.argmax(envelope_clean)]  # timing = np.arange(clean_np.size) → samples
```

| Property | Production value |
|---|---|
| ROI / gating window | **none**; full 512-sample window |
| Amplitude estimator | global maximum of \|Hilbert(trace)\| |
| Peak-time estimator | `argmax` of \|Hilbert(trace)\|, in **samples** |
| Timing-efficiency thresholds | 10 and 20, applied to a sample-index difference although labelled "ns" |
| Bandpass before the envelope | none |

## 5. Two findings that need a decision

**(a) The revised SNR definition is not the one used for the published figures.** The
revision plan specifies `max|Hilbert(clean)| / std(noisy off-pulse)` with an exclusion
window. The production figures used `max(clean) / std(noisy)` over the full window — raw
maximum rather than envelope peak, and no off-pulse exclusion. There is therefore **no
"production exclusion half-width" to recover**: the quantity does not exist in the
production code. Either the manuscript states the definition that was actually used, or
the retained SNR-binned figures are regenerated with the revised definition — the second
option changes the x-axis of those figures.

**(b) The timing-efficiency threshold is in samples, not nanoseconds.** `timing` is
`np.arange(n_samples)`, so the thresholds 10 and 20 are sample counts while the axis label
reads "ns". Converting them requires the sampling interval `dt`, which is still not
recovered from the simulation inputs. Until `dt` is fixed, the timing-efficiency threshold
cannot be quoted in nanoseconds.

## 6. Ready-to-use Table I entries (multi_v3)

| Field | Value |
|---|---|
| Objective | multi-domain L1: time + \|rFFT\| + unwrapped principal phase |
| `mag_weight` | 0.011676 |
| `phase_weight` | 0.028893 |
| Learning rate | 1.656e-05 |
| `max_lr` | 5.067e-04 |
| `step_size` | 10000 |
| `lr_mode` | triangular2 |
| `weight_decay` | 2.566e-04 |
| Batch size | 1024 |
| Time branch | conv 64, residual [128, 256] |
| Frequency branch | conv 64, residual [128, 256] (instantiated twice: magnitude, phase) |
| Fusion width | 384 |
| Effective decoder widths | [384, 64, 32, 3] |
| Max-pool per branch | 2 (downsampling factor 4) |
| Input length | 512-sample window (simulations stored at 1024) |
| Gradient clipping | none |
| FFT taper | none |
| Epochs completed / budget | 63 / 100 (36 Ray Tune trials) |
| Validation loss / PSNR (archived) | 0.19997 / 66.53 dB |
