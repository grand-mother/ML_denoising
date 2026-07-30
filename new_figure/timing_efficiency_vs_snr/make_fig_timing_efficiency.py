#!/usr/bin/env python3
"""
make_fig_timing_efficiency.py

Regenerates the paper's timing-reconstruction figure (Fig. 7) with the corrected
legend units.

Referee/OM note addressed:
    "Relabel the figure legend from 10 ns to 5 ns. The underlying 10-sample
     selection does not need to be recomputed."

The timing selection is UNCHANGED: it is still |denoised_time - clean_time| <= 10
*samples*, exactly as in the published figure (the notebook passes
thresholds_list=[10]). Only the label changes, via the sampling interval
dt = 0.5 ns/sample, so 10 samples is displayed as 5 ns.

Two deliberate differences from the published run, both documented in
Referee_update/reply4.md:
  1. Evaluation is deterministic: seeded 80/10/10 split (seed 12345), no random
     crop, no swap augmentation. The published run used an unseeded
     `split_indices` and swap_prob=0.5 on the test set, so it could not be
     reproduced bit-for-bit.
  2. The SNR axis uses the revised off-pulse definition
     max|Hilbert(clean)| / std(noisy off-pulse), the single canonical definition
     adopted for every SNR-binned figure in this revision.

Usage:
  python new_figure/timing_efficiency_vs_snr/make_fig_timing_efficiency.py
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

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
# DU_response_computation lives one level above the repo (needed for voltage_to_adc).
if str(ROOT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR.parent))

import matplotlib
matplotlib.use("Agg")

from training.models.cnn import DualBranchAutoencoder
from training.raytune_training_function import CustomDataset
from utils.data_preprocessing import produce_noise_and_noiseless_data
from utils.referee_revision_utils import (
    deterministic_split_indices,
    save_evaluation_manifest,
    load_evaluation_manifest,
)
from visualization.overleaf_plots import peak_time_analysis_for_all_channels

DEFAULT_MODEL_DIR = "/sps/grand/macias/Sam_Result/multi_v3_CNN_100epochs_36samples"


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate Fig. 7 with 5 ns legend labels.")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                    help="Checkpoint dir; default is the multi_v3 model the published "
                         "figure used (loaded with the current training/models/cnn.py, "
                         "which is identical to the published branch).")
    ap.add_argument("--data-path", default="/sps/grand/blevy/sims/sims_for_denoising_sept2025")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--threshold-samples", type=int, default=10,
                    help="Timing tolerance in SAMPLES (published value: 10 -> 5 ns).")
    ap.add_argument("--trace-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--max-samples", type=int, default=None, help="Debug cap on test traces.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    with open(model_dir / "best_trial_config.json") as f:
        config = json.load(f)
    model_config = config["model_config"]
    print(f"Checkpoint: {model_dir}")
    print(f"criterion={config.get('criterion')}  model_config={model_config}", flush=True)

    print("Loading data...", flush=True)
    noise_signals, clean_signals = produce_noise_and_noiseless_data(args.data_path)
    total = int(clean_signals.shape[1])

    manifest_path = out_dir / "evaluation_manifest.npz"
    if manifest_path.exists():
        manifest = load_evaluation_manifest(manifest_path)
        print(f"Reusing manifest: {manifest_path}")
    else:
        tr, va, te = deterministic_split_indices(
            total, train_fraction=0.8, valid_fraction=0.1, seed=args.seed)
        save_evaluation_manifest(
            manifest_path, tr, va, te, n_total=total, seed=args.seed,
            provenance="deterministic test split for the regenerated timing figure",
        )
        manifest = load_evaluation_manifest(manifest_path)
        print(f"Created manifest: {manifest_path}")

    test_idx = manifest.test
    if args.max_samples:
        test_idx = test_idx[:args.max_samples]
    print(f"Test traces: {test_idx.size}", flush=True)

    # Deterministic evaluation: no random crop, no swap. voltage_to_adc as in the
    # published notebook's test_dataset.
    test_ds = CustomDataset(
        clean_signals, [noise_signals], indices=test_idx,
        no_random=True, eval_len=args.trace_length, swap_prob=0.0,
        target_start=120, target_end=480, voltage_to_adc=True,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = DualBranchAutoencoder(model_config)

    print("Running timing analysis (this loads the checkpoint internally)...", flush=True)
    peak_time_analysis_for_all_channels(
        dataloader=test_loader,
        model=model,
        model_path=str(model_dir / "best_model.pth"),
        device=args.device,
        min_snr=1,
        max_snr=1e3,
        thresholds_list=[args.threshold_samples],   # SAMPLES; labelled as ns via dt=0.5
        save_path=str(out_dir),
    )
    print(f"\nDone. Figure written under: {out_dir}")
    for p in sorted(out_dir.glob("*.pdf")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
