# Reply — complete direct vs wrapped-counterfactual objective on the audit sample

**Request.** "Add one final inference-only number: compute the complete direct and
wrapped-counterfactual objectives on the same 41,067 audit traces, rather than comparing
the phase change with the archived validation loss from a different split."

**Status: done.** The cross-split comparison has been withdrawn and replaced by a
same-sample counterfactual. One caveat below must be read before the number is quoted.

---

## 1. Result

Objective definition, matching the production `multi_domain_loss_v3`:

```
objective = time_l1 + mag_weight * mag_l1 + phase_weight * phase_l1
```

(element-wise-mean L1; `mag_weight = 0.011676`, `phase_weight = 0.028893`, both from the
archived multi_v3 best trial). Only the phase term differs between the two columns.

| Quantity | Value |
|---|---|
| `time_l1` | 259.5195 |
| `mag_l1` | 7401.3406 |
| `phase_l1` — direct | 2.071081 |
| `phase_l1` — wrapped | 1.561558 |
| **objective — direct** | **345.99429** |
| **objective — wrapped** | **345.97957** |
| **absolute change** | **0.0147216** |
| **relative change (same traces)** | **4.25 × 10⁻⁵ = 0.0043 %** |
| traces | 41,067 |

The phase term contributes ≈0.0598 (direct) of a total objective of ≈346, i.e. ≈0.017 %
of the objective; wrapping changes it by ≈0.0043 % of the objective. The previously
reported "≈7.4 %" was the phase change divided by the archived validation loss from a
different split and is withdrawn.

Inference only: no training, no Ray Tune, no phase-weight scan, no magnitude weighting.

## 2. Evidence

| Item | Location |
|---|---|
| Code | `evaluation/phase_boundary_audit.py` — `time_l1`/`mag_l1` accumulators inside the inference loop; `counterfactual` block after `finalize()` |
| Job | SLURM **54985187**, COMPLETED, elapsed 00:23:08, node `ccwgislurm0100` |
| Console log | `evaluation/phase_audit_v3_54985187.out`, section `=== COMPLETE OBJECTIVE ON THE SAME AUDIT TRACES ===` |
| Machine-readable | `results/phase_boundary_audit_multi_v3/phase_boundary_audit.json`, key `metadata.counterfactual_objective_same_traces` |
| Evaluation sample | `results/phase_boundary_audit_multi_v3/evaluation_manifest.npz` — 41,067 validation traces, `no_random=True`, `swap_prob=0.0`, `shuffle=False`; identical traces used for both columns |
| Checkpoint | `/sps/grand/macias/Sam_Result/multi_v3_CNN_100epochs_36samples/best_model.pth` |
