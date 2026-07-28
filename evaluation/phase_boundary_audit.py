#!/usr/bin/env python3
"""
phase_boundary_audit.py

Referee-revision task 3 (minimal package): post-hoc audit of the wrapped-phase
boundary effect on the EXISTING fiducial checkpoint. No retraining, no loss
change. Reproduces the original unweighted phase term of the production
multi-domain loss and reports how much the +/-pi wrap-around matters.

For each RFFT coefficient it compares:
  * direct  : |angle(rec) - angle(clean)|                 (as in the production loss)
  * wrapped : |atan2(sin(delta), cos(delta))|             (the corrected residual)

Outputs (deliverable phase_boundary_audit.json):
  full_band     : all RFFT coefficients
  analysis_band : 50--200 MHz only (secondary diagnostic; needs the sampling dt)
  metadata      : checkpoint / phase_weight / trace_length / dt / manifest

Decision (per README_SAM.md, made by Oscar/Sam, NOT here):
  boundary effect negligible -> keep the checkpoint, state the direct-phase fact
  boundary effect not negligible -> one conditional retrain with wrapped residual.

Usage:
  python evaluation/phase_boundary_audit.py \
      --model-dir /sps/grand/macias/Sam_Result/multi_v3_CNN_100epochs_36samples \
      --device cuda
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.models.cnn import DualBranchAutoencoder
from training.raytune_training_function import CustomDataset
from utils.data_preprocessing import produce_noise_and_noiseless_data
from utils.referee_revision_utils import (
    PhaseAuditAccumulator,
    rfft_frequency_mask,
    save_phase_audit_json,
    deterministic_split_indices,
    save_evaluation_manifest,
    load_evaluation_manifest,
)


def _load_fiducial_model(model_dir: Path, device: str):
    with open(model_dir / "best_trial_config.json") as f:
        config = json.load(f)
    model_config = config["model_config"]
    phase_weight = float(config.get("phase_weight", float("nan")))
    mag_weight = float(config.get("mag_weight", float("nan")))
    criterion = config.get("criterion", "?")

    model = DualBranchAutoencoder(model_config).to(device)
    state = torch.load(model_dir / "best_model.pth", map_location=device, weights_only=True)
    # Strip a DataParallel "module." prefix if present.
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, phase_weight, mag_weight, criterion, model_config


def _get_manifest(out_dir: Path, model_dir: Path, total_samples: int, seed: int):
    """Recover the archived split if present; otherwise make one seeded post-hoc manifest."""
    # 1) archived manifest saved with the production run?
    for cand in (model_dir / "split_manifest.npz", model_dir / "evaluation_manifest.npz"):
        if cand.exists():
            try:
                m = load_evaluation_manifest(cand)
                print(f"Recovered archived manifest: {cand} (provenance={m.provenance})")
                return m
            except Exception:
                # split_manifest.npz (task-2 format) has no 'provenance'; adapt.
                d = np.load(cand, allow_pickle=False)
                manifest_path = out_dir / "evaluation_manifest.npz"
                save_evaluation_manifest(
                    manifest_path, d["train"], d["valid"], d["test"],
                    n_total=int(d["n_total"]), seed=int(d["seed"]),
                    provenance=f"recovered from {cand.name}",
                )
                print(f"Recovered split from {cand.name}; re-saved as {manifest_path}")
                return load_evaluation_manifest(manifest_path)

    # 2) not recoverable -> one seeded post-hoc manifest, stated explicitly.
    manifest_path = out_dir / "evaluation_manifest.npz"
    if manifest_path.exists():
        return load_evaluation_manifest(manifest_path)
    train, valid, test = deterministic_split_indices(
        total_samples, train_fraction=0.8, valid_fraction=0.1, seed=seed)
    save_evaluation_manifest(
        manifest_path, train, valid, test, n_total=total_samples, seed=seed,
        provenance="post-hoc common evaluation; original production indices unavailable",
    )
    print(f"No archived indices; created seeded post-hoc manifest: {manifest_path}")
    return load_evaluation_manifest(manifest_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-boundary audit (task 3).")
    ap.add_argument("--model-dir",
                    default="/sps/grand/macias/Sam_Result/multi_v3_CNN_100epochs_36samples",
                    help="Checkpoint dir of the phase-loss production run (multi_v3 = "
                         "multi-domain L1 with mag+phase, 100 epochs / 36 trials). "
                         "This is the model behind docs/phase_boundary_audit_table.md.")
    ap.add_argument("--data-path", default="/sps/grand/blevy/sims/sims_for_denoising_sept2025")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--trace-length", type=int, default=512,
                    help="Production checkpoint trace length (PROVISIONAL until task-1 audit).")
    ap.add_argument("--sample-spacing-seconds", type=float, default=0.5e-9,
                    help="Sampling dt for the 50-200 MHz band mask (PROVISIONAL; task-1).")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--max-samples", type=int, default=None, help="Debug: cap valid traces.")
    ap.add_argument("--out-dir", default=str(ROOT_DIR / "results" / "phase_boundary_audit"))
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    print(f"Fiducial: {model_dir}")
    model, phase_weight, mag_weight, criterion, _ = _load_fiducial_model(model_dir, device)
    print(f"criterion={criterion}  phase_weight={phase_weight}  mag_weight={mag_weight}")
    if criterion not in ("multi_l1", "multi_mse", "multi_v2", "multi_v3"):
        print(f"WARNING: criterion '{criterion}' has no phase term in the training loss; "
              "the audit still measures the phase-boundary statistics but the "
              "weighted_objective_change is only meaningful for a phase-loss model.")

    print("Loading data...", flush=True)
    noise_signals, clean_signals = produce_noise_and_noiseless_data(args.data_path)
    total_samples = int(clean_signals.shape[1])

    manifest = _get_manifest(out_dir, model_dir, total_samples, args.seed)
    valid_idx = manifest.valid
    if args.max_samples:
        valid_idx = valid_idx[:args.max_samples]
    print(f"Validation traces: {valid_idx.size}")

    # Deterministic evaluation at the production trace length.
    valid_ds = CustomDataset(
        clean_signals, [noise_signals], indices=valid_idx,
        no_random=True, eval_len=args.trace_length, swap_prob=0.0,
    )
    loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    full_audit = PhaseAuditAccumulator(phase_weight=phase_weight)
    band_mask = rfft_frequency_mask(
        args.trace_length, args.sample_spacing_seconds, f_min_hz=50e6, f_max_hz=200e6)
    band_audit = PhaseAuditAccumulator(phase_weight=phase_weight, frequency_mask=band_mask)

    # Accumulate the non-phase objective terms on the SAME audit traces, so the
    # complete direct vs wrapped counterfactual objectives are computed on one
    # dataset (element-wise-mean L1, matching production multi_domain_loss_v3:
    # time_l1 + mag_weight*mag_l1 + phase_weight*phase_l1).
    t_sum = t_cnt = 0.0
    m_sum = m_cnt = 0.0

    print("Running audit inference...", flush=True)
    with torch.no_grad():
        for bi, (noisy, clean) in enumerate(loader):
            noisy = noisy.to(device)
            clean = clean.to(device)
            reconstructed = model(noisy)
            full_audit.update(clean, reconstructed)
            band_audit.update(clean, reconstructed)
            # time-domain L1
            t_sum += float((reconstructed - clean).abs().sum())
            t_cnt += clean.numel()
            # Fourier-magnitude L1
            cfft = torch.fft.rfft(clean, dim=-1)
            pfft = torch.fft.rfft(reconstructed, dim=-1)
            m_sum += float((pfft.abs() - cfft.abs()).abs().sum())
            m_cnt += cfft.numel()
            if (bi + 1) % 20 == 0:
                print(f"  batch {bi + 1}", flush=True)

    full = full_audit.finalize()
    band = band_audit.finalize()

    # Complete direct vs wrapped objectives on the same 41k traces (task: replace the
    # invalid cross-split comparison with an on-sample counterfactual).
    time_l1 = t_sum / t_cnt
    mag_l1 = m_sum / m_cnt
    base = time_l1 + mag_weight * mag_l1
    obj_direct = base + phase_weight * full.direct_phase_l1
    obj_wrapped = base + phase_weight * full.wrapped_phase_l1
    obj_abs_change = obj_direct - obj_wrapped
    obj_rel_change = (obj_abs_change / obj_direct) if obj_direct else float("nan")
    counterfactual = {
        "time_l1": time_l1,
        "mag_l1": mag_l1,
        "mag_weight": mag_weight,
        "phase_weight": phase_weight,
        "phase_l1_direct": full.direct_phase_l1,
        "phase_l1_wrapped": full.wrapped_phase_l1,
        "objective_direct": obj_direct,
        "objective_wrapped": obj_wrapped,
        "objective_absolute_change": obj_abs_change,
        "objective_relative_change": obj_rel_change,
        "n_traces": int(valid_idx.size),
        "definition": "time_l1 + mag_weight*mag_l1 + phase_weight*phase_l1 "
                      "(element-wise-mean L1; matches production multi_domain_loss_v3)",
    }
    print("\n=== COMPLETE OBJECTIVE ON THE SAME AUDIT TRACES ===")
    for k, v in counterfactual.items():
        print(f"  {k}: {v}")

    print("\n=== FULL BAND ===")
    for k, v in full.to_dict().items():
        print(f"  {k}: {v}")
    print("=== 50-200 MHz BAND (secondary; dt provisional) ===")
    for k, v in band.to_dict().items():
        print(f"  {k}: {v}")

    out_json = out_dir / "phase_boundary_audit.json"
    save_phase_audit_json(
        out_json, full_band=full, analysis_band=band,
        metadata={
            "counterfactual_objective_same_traces": counterfactual,
            "checkpoint": str(model_dir / "best_model.pth"),
            "criterion": criterion,
            "phase_weight": phase_weight,
            "mag_weight": mag_weight,
            "trace_length": args.trace_length,
            "sample_spacing_seconds": args.sample_spacing_seconds,
            "evaluation_manifest": str(out_dir / "evaluation_manifest.npz"),
            "manifest_provenance": manifest.provenance,
            "n_valid_traces": int(valid_idx.size),
            "note_trace_length_and_dt": "PROVISIONAL pending the task-1 production-config audit",
        },
    )
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
