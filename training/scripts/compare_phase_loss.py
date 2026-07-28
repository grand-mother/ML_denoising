#!/usr/bin/env python3
"""
compare_phase_loss.py

Small, controlled validation study for the referee revision (task 1):
compare the wrapped multi-domain phase loss with
  (1) phase_weighting = "none"              (unweighted wrapped loss)
  (2) phase_weighting = "target_magnitude"  (magnitude-weighted wrapped loss)
across a small grid of phase_weight values, with EVERYTHING ELSE FIXED:

  * one frozen architecture (the production best_trial_config by default),
  * one frozen set of optimizer / schedule / mag_weight hyperparameters,
  * one deterministic, seeded 80/10/10 split (utils.paper_losses_metrics),
  * deterministic data pipeline (no trace swapping, no val shuffle).

Only phase_weighting and phase_weight vary. This is a *small* run (few tens of
epochs), not the production sweep.

Parallel execution
------------------
Each GPU worker runs a subset of cells via --cells and a pinned
CUDA_VISIBLE_DEVICES; cells write into their own cell_<...>/ directory, so
workers never collide. After all workers finish, an --aggregate-only pass
collects every cell_*/result.json into the comparison table + figure.

Fair comparison metric
-----------------------
The training criterion differs from cell to cell (different phase_weight /
weighting), so the criterion value is NOT comparable across cells. We select the
best epoch and rank cells by a PHASE-INDEPENDENT validation metric: the
time-domain validation MSE (time-domain L1 is also logged). The criterion value
is logged only to monitor convergence.

Outputs (in --out-dir)
----------------------
  split_manifest.npz                  the frozen split (task-2 preview)
  cell_<weighting>_pw<pw>/best_model.pth + config.json + result.json
  phase_loss_comparison.csv           one row per cell (after --aggregate-only)
  phase_loss_val_curves.pdf           val-MSE vs epoch, all cells
"""
from __future__ import annotations

import os
import sys
import csv
import json
import glob
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.models.cnn import DualBranchAutoencoder
from training.raytune_training_function import CustomDataset
from training.raytune_train_sept25 import get_criterion
from utils.data_preprocessing import produce_noise_and_noiseless_data
from utils.paper_losses_metrics import deterministic_split_indices, save_split_manifest


# Frozen production best_trial_config
# (/sps/grand/macias/Sam_Result/multi_l1_CNN_100epochs_24samples/best_trial_config.json).
# EVERYTHING is fixed at the production operating point; only phase_weighting
# (none vs target_magnitude) varies. phase_weight is fixed at the production value
# (0.0236) and passed via --phase-weights (single value -> exactly 2 runs).
PROD_CONFIG = {
    "lr": 2.079694835624803e-05,
    "max_lr": 3.401369290706597e-05,
    "step_size": 30000,
    "lr_mode": "triangular2",
    "weight_decay": 1.2920908492083683e-05,
    "batch_size": 1024,
    "mag_weight": 0.019268098921250677,
    "model_config": {
        "time_branch": {"conv_channels": 128, "res_channels": [128, 256]},
        "freq_branch": {"conv_channels": 128, "res_channels": [128, 256]},
        "decoder_channels": [256, 128, 64, 3],
    },
}


def cell_dirname(weighting: str, phase_weight: float) -> str:
    return f"cell_{weighting}_pw{phase_weight}"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(clean_signals, noise_signals, train_idx, valid_idx, batch_size):
    # Deterministic pipeline: no trace swapping; val is not shuffled.
    train_ds = CustomDataset(clean_signals, [noise_signals], indices=train_idx, swap_prob=0.0, no_random=True)
    valid_ds = CustomDataset(clean_signals, [noise_signals], indices=valid_idx, swap_prob=0.0, no_random=True)
    g = torch.Generator()
    g.manual_seed(0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=True, generator=g)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, valid_loader


@torch.no_grad()
def validate(model, valid_loader, criterion, device):
    model.eval()
    tot_crit = tot_mse = tot_l1 = 0.0
    n = 0
    for noisy, clean in valid_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        pred = model(noisy)
        bs = noisy.size(0)
        tot_crit += float(criterion(pred, clean)) * bs
        tot_mse += float(F.mse_loss(pred, clean)) * bs
        tot_l1 += float(F.l1_loss(pred, clean)) * bs
        n += bs
    return tot_crit / n, tot_mse / n, tot_l1 / n


def train_one_cell(weighting, phase_weight, base_cfg, loaders, epochs, device, seed, out_dir):
    train_loader, valid_loader = loaders

    cfg = dict(base_cfg)
    cfg.update({"criterion": "multi_l1", "phase_weight": float(phase_weight),
                "phase_weighting": weighting, "model_type": "CNN"})

    # Identical initialization across all cells: differences come from the loss only.
    set_seed(seed)
    model = DualBranchAutoencoder(cfg["model_config"])
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion = get_criterion("multi_l1", cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=cfg["lr"], max_lr=cfg["max_lr"],
        step_size_up=cfg["step_size"], cycle_momentum=False, mode=cfg["lr_mode"])

    history = []
    best = {"val_mse": float("inf"), "val_l1": float("nan"),
            "val_crit": float("nan"), "epoch": -1}
    cell_dir = Path(out_dir) / cell_dirname(weighting, phase_weight)
    cell_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            loss = criterion(model(noisy), clean)
            loss.backward()
            optimizer.step()
            scheduler.step()

        val_crit, val_mse, val_l1 = validate(model, valid_loader, criterion, device)
        history.append([epoch, val_crit, val_mse, val_l1])
        print(f"[{weighting} pw={phase_weight}] epoch {epoch+1}/{epochs} "
              f"val_crit={val_crit:.6f} val_mse={val_mse:.6f} val_l1={val_l1:.6f}", flush=True)

        if val_mse < best["val_mse"]:
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            best = {"val_mse": val_mse, "val_l1": val_l1, "val_crit": val_crit, "epoch": epoch}
            torch.save(state, cell_dir / "best_model.pth")
            with open(cell_dir / "config.json", "w") as f:
                json.dump({"model_type": "CNN", "model_config": cfg["model_config"],
                           "weighting": weighting, "phase_weight": phase_weight,
                           "mag_weight": cfg["mag_weight"]}, f, indent=2)

    result = {"weighting": weighting, "phase_weight": phase_weight,
              "best_epoch": best["epoch"] + 1, "best_val_mse": best["val_mse"],
              "best_val_l1": best["val_l1"], "best_val_crit": best["val_crit"],
              "history": history}
    with open(cell_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def parse_cells(cells_arg, weightings, phase_weights):
    if cells_arg:
        out = []
        for tok in cells_arg.split(","):
            w, pw = tok.split(":")
            out.append((w.strip(), float(pw)))
        return out
    return [(w, pw) for w in weightings for pw in phase_weights]


def aggregate(out_dir):
    rows = []
    for rj in sorted(glob.glob(str(Path(out_dir) / "cell_*" / "result.json"))):
        with open(rj) as f:
            rows.append(json.load(f))
    if not rows:
        raise RuntimeError(f"No cell_*/result.json found under {out_dir}")

    csv_path = Path(out_dir) / "phase_loss_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["weighting", "phase_weight", "best_epoch",
                    "best_val_mse", "best_val_l1", "best_val_crit"])
        for r in sorted(rows, key=lambda r: r["best_val_mse"]):
            w.writerow([r["weighting"], r["phase_weight"], r["best_epoch"],
                        f"{r['best_val_mse']:.6f}", f"{r['best_val_l1']:.6f}",
                        f"{r['best_val_crit']:.6f}"])
    print(f"Saved comparison table: {csv_path}", flush=True)

    plt.figure(figsize=(10, 6))
    for r in rows:
        h = np.array(r["history"])
        plt.plot(h[:, 0] + 1, h[:, 2], marker="o", markersize=3,
                 label=f"{r['weighting']} pw={r['phase_weight']}")
    plt.xlabel("epoch"); plt.ylabel("validation MSE (phase-independent)")
    plt.yscale("log"); plt.grid(True, alpha=0.3); plt.legend(fontsize=8)
    plt.title("Wrapped phase loss: per-bin weighting mode A/B "
              "(fixed architecture, fixed phase weight, same init and saved split)")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "phase_loss_val_curves.pdf", dpi=200)
    print("Saved curves: phase_loss_val_curves.pdf", flush=True)

    best = min(rows, key=lambda r: r["best_val_mse"])
    print("\n=== summary (best val MSE per cell) ===", flush=True)
    for r in sorted(rows, key=lambda r: r["best_val_mse"]):
        print(f"  {r['weighting']:16s} pw={r['phase_weight']:<4} "
              f"val_mse={r['best_val_mse']:.6f} (epoch {r['best_epoch']})", flush=True)
    print(f"\nBest: weighting={best['weighting']} phase_weight={best['phase_weight']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Phase-loss A/B comparison (task 1).")
    ap.add_argument("--data-path", default="/sps/grand/blevy/sims/sims_for_denoising_sept2025")
    ap.add_argument("--out-dir", default=str(ROOT_DIR / "results" / "phase_loss_compare"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--split-seed", type=int, default=12345)
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--phase-weights", default="0.1,0.3,0.6,1.0")
    ap.add_argument("--weightings", default="none,target_magnitude")
    ap.add_argument("--cells", default=None,
                    help="Subset of cells for this worker, e.g. 'none:0.1,none:0.3'. "
                         "Default = full weightings x phase_weights grid.")
    ap.add_argument("--write-manifest", action="store_true",
                    help="Write split_manifest.npz (set on exactly one worker).")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Skip training; build the comparison table/figure from cell_*/result.json.")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max-samples", type=int, default=None, help="debug: cap total samples")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.aggregate_only:
        aggregate(args.out_dir)
        return

    phase_weights = [float(x) for x in args.phase_weights.split(",")]
    weightings = [w.strip() for w in args.weightings.split(",")]
    cells = parse_cells(args.cells, weightings, phase_weights)
    print(f"This worker will run cells: {cells}", flush=True)

    print("Loading data...", flush=True)
    noise_signals, clean_signals = produce_noise_and_noiseless_data(args.data_path)
    total = clean_signals.shape[1]
    if args.max_samples:
        total = min(total, args.max_samples)

    # One frozen, seeded split (identical across workers because it is deterministic).
    train_idx, valid_idx, test_idx = deterministic_split_indices(
        total, train_fraction=0.8, valid_fraction=0.1, seed=args.split_seed)
    if args.write_manifest:
        save_split_manifest(Path(args.out_dir) / "split_manifest.npz",
                            train_idx, valid_idx, test_idx, n_total=total, seed=args.split_seed)
        print("Wrote split_manifest.npz", flush=True)
    print(f"Split: train={train_idx.size} valid={valid_idx.size} test={test_idx.size}", flush=True)

    loaders = build_loaders(clean_signals, noise_signals, train_idx, valid_idx,
                            PROD_CONFIG["batch_size"])

    for weighting, pw in cells:
        print(f"\n=== cell: weighting={weighting} phase_weight={pw} ===", flush=True)
        train_one_cell(weighting, pw, PROD_CONFIG, loaders, args.epochs,
                       args.device, args.init_seed, args.out_dir)

    print("\nWorker finished its cells.", flush=True)

    # Single-process full-grid run: aggregate here so no separate step is needed.
    # (Multi-worker runs pass --cells and aggregate via a final --aggregate-only call.)
    if args.cells is None:
        print("\nAggregating full grid...", flush=True)
        aggregate(args.out_dir)


if __name__ == "__main__":
    main()
