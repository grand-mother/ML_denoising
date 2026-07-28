# Phase residual: direct vs wrapped — compact audit table

Inference-only calculation, as requested. No phase-weight scan, no Ray search, no
magnitude weighting, no retraining. Computed by `evaluation/phase_boundary_audit.py`
using `utils/referee_revision_utils.py::PhaseAuditAccumulator`; raw output in
`results/phase_boundary_audit_multi_v3/phase_boundary_audit.json`.

**Checkpoint:** `/sps/grand/macias/Sam_Result/multi_v3_CNN_100epochs_36samples/best_model.pth`
— the multi-domain L1 production run (`criterion = multi_v3`: time L1 + Fourier-magnitude
L1 + Fourier-phase L1), the only published objective containing a phase term.
**Phase weight:** `0.028893` (archived best-trial value).
**Validation predictions:** deterministic evaluation of 41,067 held-out traces
(512-sample windows, no augmentation, `shuffle=False`), split saved as
`evaluation_manifest.npz` (post-hoc seeded manifest; original production indices not
recoverable — stated in its provenance field).

## Result

| Quantity | Full Fourier range | 50–200 MHz band |
|---|---|---|
| Mean phase L1 — **direct** (production convention) | **2.0711** | 2.0294 |
| Mean phase L1 — **wrapped** | **1.5616** | 1.5079 |
| Fraction of coefficients with \|Δφ_raw\| > π | **24.5 %** | 23.4 % |
| Weighted contribution to validation loss — direct (pw × L1) | 0.0598 | 0.0586 |
| Weighted contribution — wrapped | 0.0451 | 0.0436 |
| Change from wrapping (direct − wrapped, weighted) | **0.0147** | 0.0151 |
| Change relative to the archived total validation loss (≈ 0.200) | **≈ 7.4 %** | ≈ 7.5 % |

Every audited statistic sits at the analytic **random-phase expectation**:

| Quantity | Measured | Random-phase expectation | Agreement |
|---|---|---|---|
| direct mean \|Δφ_raw\| | 2.0711 | 2π/3 ≈ 2.0944 | 98.9 % |
| wrapped mean \|Δφ\| | 1.5616 | π/2 ≈ 1.5708 | 99.4 % |
| fraction \|Δφ_raw\| > π | 24.5 % | 25 % | — |
| direct − wrapped | 0.5095 | π/6 ≈ 0.5236 | 97.3 % |