#!/usr/bin/env python3
"""
train_fixed_config.py

One fixed-configuration training run. No Ray Tune search.

Used for the referee's time-only baseline question: no matched time-only
checkpoint exists, so a single time-only run at the production settings is
trained. The same script trains the dual-branch counterpart by flipping
--use-freq-branch, so the two differ ONLY by the presence of the Fourier
(magnitude + phase) branches, as the referee asked.

Everything else is held fixed at the multi_v3 production operating point
(mag/phase weights, optimizer, cyclic schedule, batch size, 512-sample traces)
and both cells share the same seeded 80/10/10 manifest.

Usage:
  python training/scripts/train_fixed_config.py --use-freq-branch false --tag time_only
  python training/scripts/train_fixed_config.py --use-freq-branch true  --tag dual_branch
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.models.cnn import DualBranchAutoencoder
from training.raytune_train_sept25 import get_criterion
from training.raytune_training_function import CustomDataset
from utils.data_preprocessing import produce_noise_and_noiseless_data
from utils.referee_revision_utils import (
    deterministic_split_indices,
    save_evaluation_manifest,
    load_evaluation_manifest,
)

# Production operating point: multi_v3_CNN_100epochs_36samples/best_trial_config.json
PRODUCTION_CONFIG = {
    "lr": 1.6563229048775545e-05,
    "max_lr": 0.0005066725384466413,
    "step_size": 10000,
    "lr_mode": "triangular2",
    "weight_decay": 0.0002565942350522923,
    "batch_size": 1024,
    "mag_weight": 0.01167558670579447,
    "phase_weight": 0.028892879445242478,
    "model_config": {
        "time_branch": {"conv_channels": 64, "res_channels": [128, 256]},
        "freq_branch": {"conv_channels": 64, "res_channels": [128, 256]},
        "decoder_channels": [128, 64, 32, 3],
    },
}

# Production CustomDataset settings (raytune_main_sept25.py built it with all defaults).
TRACE_LEN = 512
TRAIN_SWAP_PROB = 0.5
TARGET_START = 300
TARGET_END = 500


def _psnr_db(clean: torch.Tensor, pred: torch.Tensor) -> float:
    """Per-trace PSNR against that trace's own peak, averaged over the batch."""
    mse = ((pred - clean) ** 2).mean(dim=(1, 2))
    peak = clean.abs().amax(dim=(1, 2))
    valid = (mse > 0) & (peak > 0)
    if not valid.any():
        return float("nan")
    return float((20 * torch.log10(peak[valid] / torch.sqrt(mse[valid]))).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Single fixed-config training (no Ray Tune).")
    ap.add_argument("--use-freq-branch", choices=["true", "false"], required=True,
                    help="true = dual-branch; false = time-only (Fourier branches removed).")
    ap.add_argument("--tag", required=True, help="Run name; also the output subdirectory.")
    ap.add_argument("--criterion", default="multi_l1",
                    help="Loss key understood by get_criterion. multi_l1 = time + |FFT| + "
                         "wrapped-phase L1, the current code's multi-domain L1.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--data-path", default="/sps/grand/blevy/sims/sims_for_denoising_sept2025")
    ap.add_argument("--out-root", default=str(ROOT_DIR / "results" / "fixed_config_runs"))
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=None, help="Debug: cap traces per split.")
    args = ap.parse_args()

    use_freq = args.use_freq_branch == "true"
    out_dir = Path(args.out_root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = json.loads(json.dumps(PRODUCTION_CONFIG))  # deep copy
    config["model_config"]["use_freq_branch"] = use_freq
    # Record the fusion wiring explicitly so evaluation reconstructs the exact
    # architecture. The existing fixed-config cells were trained with the
    # 'global_pool' wiring; the archived production checkpoints use 'spectral'
    # (the DualBranchAutoencoder default).
    config["model_config"]["freq_fusion"] = "global_pool"
    config["criterion"] = args.criterion
    config["epochs"] = args.epochs
    config["model_type"] = "CNN"
    config["trace_length"] = TRACE_LEN
    config["use_freq_branch"] = use_freq

    print("=" * 70)
    print(f"Fixed-config run: tag={args.tag}  use_freq_branch={use_freq}")
    print(f"criterion={args.criterion}  epochs={args.epochs}  trace_len={TRACE_LEN}")
    print(json.dumps(config, indent=2))
    print("=" * 70, flush=True)

    print("Loading data...", flush=True)
    noise_signals, clean_signals = produce_noise_and_noiseless_data(args.data_path)
    total = int(clean_signals.shape[1])

    # Shared seeded 80/10/10 manifest so both cells see identical traces.
    manifest_path = Path(args.out_root) / "split_manifest.npz"
    if manifest_path.exists():
        manifest = load_evaluation_manifest(manifest_path)
        print(f"Reusing shared manifest: {manifest_path}")
    else:
        tr, va, te = deterministic_split_indices(
            total, train_fraction=0.8, valid_fraction=0.1, seed=args.seed)
        save_evaluation_manifest(
            manifest_path, tr, va, te, n_total=total, seed=args.seed,
            provenance="fixed-config time-only vs dual-branch comparison",
        )
        manifest = load_evaluation_manifest(manifest_path)
        print(f"Created shared manifest: {manifest_path}")

    train_idx, valid_idx = manifest.train, manifest.valid
    if args.max_samples:
        train_idx, valid_idx = train_idx[:args.max_samples], valid_idx[:args.max_samples]
    print(f"train={train_idx.size}  valid={valid_idx.size}", flush=True)

    # Training: production augmentation. Validation: deterministic, fixed 512-sample window.
    train_ds = CustomDataset(
        clean_signals, [noise_signals], indices=train_idx, traces_len=TRACE_LEN,
        swap_prob=TRAIN_SWAP_PROB, target_start=TARGET_START, target_end=TARGET_END,
    )
    valid_ds = CustomDataset(
        clean_signals, [noise_signals], indices=valid_idx,
        no_random=True, eval_len=TRACE_LEN, swap_prob=0.0,
    )
    bs = config["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              pin_memory=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=bs, shuffle=False,
                              pin_memory=True, num_workers=args.num_workers)

    model = DualBranchAutoencoder(config["model_config"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}", flush=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"DataParallel over {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"],
                           weight_decay=config["weight_decay"])
    criterion = get_criterion(config["criterion"], config)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=config["lr"], max_lr=config["max_lr"],
        step_size_up=config["step_size"], cycle_momentum=False, mode=config["lr_mode"],
    )

    history = {"epochs": [], "training_losses": [], "validation_losses": [],
               "validation_psnr": [], "learning_rates": []}
    best_val = float("inf")
    t0 = time.time()

    for epoch in range(config["epochs"]):
        model.train()
        run_loss, nb = 0.0, 0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            if torch.isnan(noisy).any() or torch.isnan(clean).any():
                continue
            optimizer.zero_grad()
            out = model(noisy)
            loss = criterion(out, clean)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            optimizer.step()
            scheduler.step()
            run_loss += float(loss)
            nb += 1
        train_loss = run_loss / max(nb, 1)

        model.eval()
        vl, vp, vnb = 0.0, 0.0, 0
        with torch.no_grad():
            for noisy, clean in valid_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                out = model(noisy)
                loss = criterion(out, clean)
                if not torch.isfinite(loss):
                    continue
                vl += float(loss)
                vp += _psnr_db(clean, out)
                vnb += 1
        val_loss = vl / max(vnb, 1)
        val_psnr = vp / max(vnb, 1)

        history["epochs"].append(epoch)
        history["training_losses"].append(train_loss)
        history["validation_losses"].append(val_loss)
        history["validation_psnr"].append(val_psnr)
        history["learning_rates"].append(float(optimizer.param_groups[0]["lr"]))

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            sd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(sd, out_dir / "best_model.pth")
            with open(out_dir / "best_trial_metrics.json", "w") as f:
                json.dump({"validation_loss": val_loss, "validation_psnr": val_psnr,
                           "training_loss": train_loss, "epoch": epoch,
                           "use_freq_branch": use_freq, "criterion": config["criterion"]},
                          f, indent=2)
            marker = "  <- best"

        print(f"epoch {epoch:3d}/{config['epochs']}  train={train_loss:.6f}  "
              f"val={val_loss:.6f}  psnr={val_psnr:.3f} dB  "
              f"({(time.time() - t0) / 60:.1f} min){marker}", flush=True)

        with open(out_dir / "detailed_metrics.json", "w") as f:
            json.dump(history, f, indent=2)

    with open(out_dir / "best_trial_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nDone in {(time.time() - t0) / 3600:.2f} h. Best val loss {best_val:.6f}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
