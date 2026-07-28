# Reply — status of the checkpoint labelled `none`, and the resulting decision

**Request.** "Before launching another run, please confirm whether the earlier checkpoint
labelled `none` used the unweighted wrapped residual and the saved 80/10/10 manifest. If
so, evaluate that checkpoint and do not retrain. Otherwise, train the selected multi_v3
configuration once with the wrapped residual, keeping the architecture, phase weight,
magnitude weight, optimizer, augmentation, and saved split fixed."

**Answer in one line.** Both conditions you asked about are satisfied — the `none`
checkpoint *does* use the unweighted wrapped residual and *does* use the saved seeded
80/10/10 manifest — **but it is not the selected multi_v3 configuration**: every
hyperparameter, the trace length, the training length and the architecture differ.
Evaluating it would therefore not answer the question about the published model, so the
**second branch applies: train multi_v3 once with the wrapped residual.**

---

## 1. Confirmation of the two conditions

### (a) Unweighted wrapped residual — **confirmed**

The cell was trained with `phase_weighting = "none"`, which is the *uniform per-bin*
weighting of the **wrapped** phase term — not the direct residual, and not a zero phase
term.

| Step | Location |
|---|---|
| Cell sets `phase_weighting = "none"` and builds its criterion | `training/scripts/compare_phase_loss.py:136`, criterion at `:145` (`get_criterion("multi_l1", cfg)`) |
| `get_criterion` forwards the mode | `training/raytune_train_sept25.py:160-161` → `multi_domain_l1_loss(..., phase_weighting=...)` |
| Phase term is wrapped | `utils/paper_losses_metrics.py:136` → `multi_domain_loss` → `circular_phase_loss` (`:38`), residual from `wrapped_phase_difference` (`:22-25`, `atan2(sin Δφ, cos Δφ)`) |
| `"none"` = uniform weights | `utils/paper_losses_metrics.py:75` (`weights = torch.ones_like(target_mag)`); the magnitude-weighted branch at `:77-79` is not taken |

### (b) Saved 80/10/10 manifest — **confirmed, and it is the same split as the audit**

| Item | Value |
|---|---|
| Manifest written by the comparison | `results/phase_loss_compare_1024/split_manifest.npz` (`compare_phase_loss.py:278-281`, `deterministic_split_indices(..., seed=12345)`) |
| Manifest used by the phase audit | `results/phase_boundary_audit_multi_v3/evaluation_manifest.npz` |
| seed / n_total | 12345 / 410,673 (both) |
| train / valid / test | 328,538 / 41,067 / 41,068 (both) |
| Index-by-index identity | `train`, `valid`, `test` all **identical** (verified with `np.array_equal`) |

Evaluation determinism in that run: `swap_prob = 0.0`, `no_random = True`
(`compare_phase_loss.py:105-106`), validation `shuffle = False` (`:111`), identical
initialization per cell (`set_seed` defined at `:97`, called at `:139`).

## 2. Why the checkpoint still cannot be used

The `none` cell was produced by the A/B harness, whose fixed configuration was taken from
a *different* archived run. It is not the multi_v3 configuration:

| Setting | `none` checkpoint | multi_v3 production | Match |
|---|---|---|---|
| phase weight | 0.023581 | **0.028893** | ✗ |
| magnitude weight | 0.019268 | **0.011676** | ✗ |
| time branch `conv_channels` | 128 | **64** | ✗ |
| freq branch `conv_channels` | 128 | **64** | ✗ |
| nominal `decoder_channels` | [256, 128, 64, 3] | **[128, 64, 32, 3]** | ✗ |
| effective decoder widths | [384, 128, 64, 3] | **[384, 64, 32, 3]** | ✗ |
| trace length | 1024 | **512** | ✗ |
| training length | 40 epochs, 1 trial | **100 epochs** | ✗ |
| lr / max_lr | A/B harness values | **1.656e-05 / 5.067e-04** | ✗ |
| model class | public `training/models/cnn.py` (with `adaptive_avg_pool1d` frequency fusion) | archived `raytune_lib_sept25/raytune_CNN.py` (direct concatenation) | ✗ |

Sources: `results/phase_loss_compare_1024/cell_none_pw0.023580860891914208/config.json`
versus `/sps/grand/macias/Sam_Result/multi_v3_CNN_100epochs_36samples/best_trial_config.json`.

The last row matters independently of the hyperparameters: the A/B cells were trained with
the public model class, whose frequency branch is combined through an adaptive-average-pool
global context that the archived production class does not contain. A checkpoint from that
class is not architecturally the published network.

## 3. Decision

Per your instruction the second branch applies: **one training run of the multi_v3
configuration with the wrapped residual**, holding fixed

- **architecture** — the archived production class `raytune_lib_sept25/raytune_CNN.py`
  with the multi_v3 `model_config` (time 64/[128,256], freq 64/[128,256], decoder
  [128,64,32,3] → effective widths [384,64,32,3]);
- **phase weight** 0.028893 and **magnitude weight** 0.011676;
- **optimizer and schedule** from the archived best trial (lr 1.656e-05,
  max_lr 5.067e-04, `step_size`, `lr_mode`, `weight_decay`, batch 1024), 100 epochs;
- **augmentation and trace length** as in production (512-sample random crop, swap as
  configured in production);
- the **saved seeded 80/10/10 manifest** above.

No Ray Tune, no phase-weight scan, no magnitude-weighting experiment. The only change
relative to production is the phase residual: `F.l1_loss(angle(pred), angle(clean))`
replaced by the wrapped residual `atan2(sin Δφ, cos Δφ)`.