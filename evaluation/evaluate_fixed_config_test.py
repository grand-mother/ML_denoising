#!/usr/bin/env python3
"""
evaluate_fixed_config_test.py

Referee follow-up: the time-only vs dual-branch comparison was reported on the
VALIDATION split (the split used for checkpoint selection during training). This
script re-evaluates both existing checkpoints on the held-out TEST split.

Inference only - no retraining. The two checkpoints come from
results/fixed_config_runs/{time_only,dual_branch}, which were trained at the
multi_v3 production operating point and differ only by use_freq_branch.

Both cells are evaluated on the SAME test indices from the shared manifest
(results/fixed_config_runs/split_manifest.npz, seed 12345), with deterministic
preprocessing: no random crop, no swap augmentation, fixed 512-sample window.

Usage:
  python evaluation/evaluate_fixed_config_test.py
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
from training.raytune_train_sept25 import get_criterion
from training.raytune_training_function import CustomDataset
from utils.data_preprocessing import produce_noise_and_noiseless_data
from utils.referee_revision_utils import load_evaluation_manifest

TRACE_LEN = 512


def _psnr_db(clean: torch.Tensor, pred: torch.Tensor) -> float:
    """Per-trace PSNR against that trace's own peak, averaged over the batch.

    Identical to the definition used during training in train_fixed_config.py,
    so the test numbers are directly comparable with the validation numbers.
    """
    mse = ((pred - clean) ** 2).mean(dim=(1, 2))
    peak = clean.abs().amax(dim=(1, 2))
    valid = (mse > 0) & (peak > 0)
    if not valid.any():
        return float("nan")
    return float((20 * torch.log10(peak[valid] / torch.sqrt(mse[valid]))).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Test-split evaluation of the fixed-config pair.")
    ap.add_argument("--runs-root", default=str(ROOT_DIR / "results" / "fixed_config_runs"))
    ap.add_argument("--data-path", default="/sps/grand/blevy/sims/sims_for_denoising_sept2025")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=None, help="Debug cap on test traces.")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    manifest = load_evaluation_manifest(runs_root / "split_manifest.npz")
    test_idx = manifest.test
    if args.max_samples:
        test_idx = test_idx[:args.max_samples]
    print(f"Shared manifest seed={manifest.seed}  "
          f"train/valid/test = {manifest.train.size}/{manifest.valid.size}/{manifest.test.size}")
    print(f"Evaluating on TEST split: {test_idx.size} traces", flush=True)

    print("Loading data...", flush=True)
    noise_signals, clean_signals = produce_noise_and_noiseless_data(args.data_path)

    # Deterministic evaluation, identical for both cells.
    test_ds = CustomDataset(
        clean_signals, [noise_signals], indices=test_idx,
        no_random=True, eval_len=TRACE_LEN, swap_prob=0.0,
    )
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    results = {}
    for tag in ("dual_branch", "time_only"):
        run_dir = runs_root / tag
        with open(run_dir / "best_trial_config.json") as f:
            config = json.load(f)

        model = DualBranchAutoencoder(config["model_config"])
        state = torch.load(run_dir / "best_model.pth", map_location=device, weights_only=True)
        if any(k.startswith("module.") for k in state):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)
        model.to(device).eval()

        criterion = get_criterion(config["criterion"], config)

        tot_loss, tot_psnr, nb = 0.0, 0.0, 0
        with torch.no_grad():
            for bi, (noisy, clean) in enumerate(loader):
                noisy, clean = noisy.to(device), clean.to(device)
                out = model(noisy)
                loss = criterion(out, clean)
                if not torch.isfinite(loss):
                    continue
                tot_loss += float(loss)
                tot_psnr += _psnr_db(clean, out)
                nb += 1
                if (bi + 1) % 10 == 0:
                    print(f"  [{tag}] batch {bi + 1}/{len(loader)}", flush=True)

        with open(run_dir / "best_trial_metrics.json") as f:
            val_metrics = json.load(f)

        results[tag] = {
            "use_freq_branch": config["use_freq_branch"],
            "criterion": config["criterion"],
            "checkpoint": str(run_dir / "best_model.pth"),
            "n_test_traces": int(test_idx.size),
            "test_loss": tot_loss / max(nb, 1),
            "test_psnr_db": tot_psnr / max(nb, 1),
            "validation_loss_for_reference": val_metrics["validation_loss"],
            "validation_psnr_db_for_reference": val_metrics["validation_psnr"],
            "selected_epoch": val_metrics["epoch"],
        }
        print(f"\n[{tag}]  test_loss={results[tag]['test_loss']:.6f}  "
              f"test_psnr={results[tag]['test_psnr_db']:.3f} dB "
              f"(validation was {val_metrics['validation_loss']:.6f} / "
              f"{val_metrics['validation_psnr']:.3f} dB)\n", flush=True)

    db, to = results["dual_branch"], results["time_only"]
    summary = {
        "split": "test",
        "manifest": str(runs_root / "split_manifest.npz"),
        "manifest_seed": int(manifest.seed),
        "trace_length": TRACE_LEN,
        "note": "Inference only; no retraining. Both cells share the same test "
                "indices and identical deterministic preprocessing, and differ "
                "only by use_freq_branch.",
        "dual_branch": db,
        "time_only": to,
        "comparison": {
            "test_loss_relative_increase_time_only_vs_dual":
                (to["test_loss"] - db["test_loss"]) / db["test_loss"],
            "test_psnr_gain_db_dual_minus_time_only":
                db["test_psnr_db"] - to["test_psnr_db"],
        },
    }
    out_path = runs_root / "test_split_comparison.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 66)
    print(f"{'Model':<28}{'test loss':>13}{'test PSNR (dB)':>18}")
    print(f"{'Dual-branch (with Fourier)':<28}{db['test_loss']:>13.4f}{db['test_psnr_db']:>18.3f}")
    print(f"{'Time-only (no Fourier)':<28}{to['test_loss']:>13.4f}{to['test_psnr_db']:>18.3f}")
    print("=" * 66)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
