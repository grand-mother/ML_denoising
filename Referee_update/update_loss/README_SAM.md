# Referee revision: minimal rigorous code plan

Please disregard the earlier referee-code checklist and use this version instead. The goal is to close the remaining referee points without reopening the entire analysis.

## What is already addressed

The revised paper already handles the following points through narrower claims and clearer caveats:

- the simplified RF-chain and stationary broadband-noise model;
- the distinction between antenna-level and event-level performance;
- the limited scope of the fixed-background false-positive test;
- the low-SNR peak-amplitude bias;
- removal of the composite usable-antenna and waveform-multiplicity results;
- removal of the numerical pointing-resolution extrapolation.

Do **not** add new realistic-noise simulations, an event-level direction fit, a fluence-to-energy reconstruction, matched filtering, wavelet denoising, or replacement multiplicity figures for this revision.

## 1. Recover the exact production configuration and correct the paper

Before changing the model, recover the configuration that produced the published checkpoint and figures from the archived Ray trial, checkpoint, JSON files, and logs. Resolve the following mismatches:

1. trace length: the public dataset defaults to 512 samples, whereas the current draft states 1,024;
2. gradient clipping: the public training loop uses a maximum norm of 1, whereas the current draft states 5;
3. Fourier preprocessing: the public model applies the real FFT directly; no Hann taper is present;
4. pooling: the public time and frequency branches each apply two max-pooling operations, not three;
5. decoder configuration: `decoder_channels[0]` is overwritten by the fusion width, so the first listed decoder value is not an independently active hyperparameter.

For each item, update the manuscript to describe the model that was actually trained. Do not modify the code merely to make it agree with the draft. Retraining is warranted only if the archived production code differs from the public branch in a way that changes the scientific result.

Please send me a one-page table with: manuscript statement, production value, evidence file/log, and proposed paper correction.

## 2. Put the time-only comparison on one common test sample

The referee asked for an additional baseline, and the time-only model already addresses this. We do not need another baseline or another hyperparameter search.

Evaluate the existing time-only and time-plus-frequency checkpoints on exactly the same deterministic test indices, with:

- `no_random=True`;
- `swap_prob=0.0`;
- the trace length used by the production checkpoint;
- `shuffle=False`;
- identical preprocessing and metric code.

First recover the original held-out indices from the archived run. If they exist, save them as `evaluation_manifest.npz` and use them. If they cannot be recovered, create one seeded manifest and state explicitly that it is a common post-hoc evaluation set.

Do not compare the two best validation losses unless both models used the same validation indices. The public Ray script draws a new unseeded split inside each trial, so those validation losses are not automatically paired. The paper can report the common-test PSNR and fractional amplitude error without quoting the validation-loss comparison.

No retraining is required for this step unless one of the checkpoints cannot be evaluated with the recovered production preprocessing.

## 3. Audit the phase boundary before deciding whether to retrain

The production loss compares the principal phases returned by `torch.angle` directly. It does not explicitly wrap the difference across the `-pi/pi` boundary. The referee asks us to clarify this point; they do not ask for a new phase-loss study.

Run the supplied post-hoc audit on the existing fiducial checkpoint and validation predictions. Report:

- the fraction of Fourier coefficients with `abs(delta_phi_raw) > pi`;
- the mean direct phase L1 term;
- the mean wrapped phase L1 term;
- the change in the total validation objective after multiplying by the selected phase weight;
- the same quantities in the 50--200 MHz band as a secondary diagnostic, using the production sampling interval.

Do **not** test magnitude weighting, phase masks, or a new phase-weight scan at this stage. Send me the audit table before changing the training loss.

### Decision after the audit

- If the boundary effect is negligible, retain the existing checkpoint. The paper and referee reply must state honestly that the original implementation used the direct principal-phase difference, and must quote the audit result.
- If the effect is not negligible, stop and send me the result. The fallback is one retraining of the selected model with the wrapped residual, using the same architecture, data manifest, optimizer settings, and phase weight. Do not rerun Ray Tune and do not introduce magnitude weighting unless we make a separate decision to do so.

## 4. Finalize the SNR and amplitude-bias diagnostics

Use one explicit input-SNR definition for every retained SNR-binned figure:

`SNR_in = max(abs(hilbert(clean))) / std(noisy off-pulse samples)`

The off-pulse samples must exclude the window around the known clean-pulse time. Recover the exact exclusion width and standard-deviation convention from the production plotting script; do not choose a new value. Record these choices in the figure metadata and manuscript.

A repository-wide refactor is not required before resubmission. It is sufficient that every retained paper figure uses the same reviewed helper and that incompatible legacy helpers are marked as legacy or are not called by the publication scripts.

For the amplitude-bias figure:

- keep the binned median;
- state exactly what the shaded population interval represents;
- state that the sample is conditioned on the noisy-input trigger selection;
- verify that the noisy and denoised curves use the same traces and SNR values.

The existing shaded interval is enough to answer the referee's request for scatter once it is defined. Counts per bin are useful but not mandatory if the bins are well populated and the counts are available in the analysis output.

## 5. Regenerate only what is necessary

Do not regenerate every model-dependent result automatically.

Regenerate figures only if:

1. the phase audit leads us to retrain;
2. a retained figure uses the wrong SNR or a mismatched event selection;
3. the common-test architecture comparison changes a reported number.

The retained results are:

- training history and selected hyperparameters;
- representative traces;
- timing efficiency;
- peak-amplitude bias;
- detection probability at fixed false-positive rate;
- Appendix band-limited NMSE/output-SNR diagnostic;
- time-only comparison on the common test sample.

Do not restore the composite usable-antenna or waveform-multiplicity figures, and do not quote an event-level pointing gain.

## 6. Deliverables

Please return:

1. the production-configuration audit table;
2. `evaluation_manifest.npz` and its provenance;
3. `phase_boundary_audit.json` for the full Fourier range and 50--200 MHz band;
4. the common-test time-only versus dual-branch table;
5. the exact SNR exclusion window and estimator;
6. any figures that genuinely required regeneration;
7. the commit hash and a short mapping from manuscript result to script and artifact;
8. the output of

```bash
pytest -q test_referee_revision_utils.py
```

Do not launch a production retraining until I have reviewed the phase audit and production-configuration table.
