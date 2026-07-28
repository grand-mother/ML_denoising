#!/usr/bin/env python3
"""
eval_phase_loss_cells.py

Post-hoc, phase-aware evaluation of the phase-loss A/B study (task 1).

The training harness (compare_phase_loss.py) selects each cell's best epoch by a
phase-INDEPENDENT time-domain MSE, which is the right *selection* metric but does
not reveal what a phase loss is meant to improve. This script loads every cell's
saved best_model.pth and, on the SAME frozen validation split, computes a
consistent, phase-aware metric set so the two loss variants can actually be
compared where they differ: in the phase and frequency domains.

Metrics per cell (all with a single fixed definition, hence comparable):
  time_mse, time_l1               time-domain reconstruction error
  psnr_db                         peak-based PSNR (paper definition), mean over traces
  mag_spec_l1                     L1 on FFT magnitude   || |FFT(pred)| - |FFT(clean)| ||_1
  power_spec_relerr               mean| |P|^2 - |C|^2 | / mean|C|^2   (power spectrum)
  phase_err_unw                   mean wrapped |angle(pred) - angle(clean)| over all bins
  phase_err_magweighted           the same, weighted by |FFT(clean)| (bins that matter)
  amp_bias_median                 median delta_A over all trace-channels
  amp_bias_median_lowsnr          median delta_A for snr in [--lowsnr-lo, --lowsnr-hi)
  timing_pass_frac                fraction with |peak-time err| <= --timing-threshold (all)
  timing_pass_frac_lowsnr         the same, restricted to the low-SNR band

delta_A and SNR reuse the canonical paper definitions
(utils.paper_losses_metrics: paper_input_snr, signed_peak_amplitude_bias).

The SNR off-pulse exclusion half-width is provisional (default 64 samples);
replace with the recovered production value once the audit (task 3) is finalized.

Usage
-----
python eval_phase_loss_cells.py \
    --compare-dir results/phase_loss_compare \
    --device cuda
"""
from __future__ import annotations

import os
import sys
import csv
import glob
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.signal import hilbert

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.models.cnn import DualBranchAutoencoder
from training.raytune_training_function import CustomDataset
from utils.data_preprocessing import produce_noise_and_noiseless_data
from utils.paper_losses_metrics import (
    wrapped_phase_difference,
    paper_input_snr,
    signed_peak_amplitude_bias,
)


def _envelope_peak_index(x: np.ndarray) -> np.ndarray:
    """Index of the Hilbert-envelope peak along the last axis.

    x: (B, C, L) -> (B, C). Matches the peak-time definition used by the paper
    timing-efficiency figure (visualization/overleaf_plots.py::peak_time_analysis,
    where peak time = argmax|Hilbert(trace)|).
    """
    return np.argmax(np.abs(hilbert(x, axis=-1)), axis=-1)


@torch.no_grad()
def eval_one_model(model, loader, device, exclude_half_width):
    """Stream the frozen validation set through one model; return a metric dict."""
    n = 0
    s_mse = s_l1 = 0.0
    s_psnr = 0.0
    s_mag = 0.0
    s_pow_num = s_pow_den = 0.0
    s_ph_unw = 0.0
    s_ph_w_num = s_ph_w_den = 0.0

    dA_list, snr_list, dt_list = [], [], []

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        pred = model(noisy)
        bs = noisy.size(0)
        n += bs

        # --- time domain ---
        s_mse += float(F.mse_loss(pred, clean)) * bs
        s_l1 += float(F.l1_loss(pred, clean)) * bs

        # --- peak-based PSNR (per trace, mean over channels) ---
        peak = clean.abs().amax(dim=(1, 2)).clamp_min(1e-12)          # (B,)
        mse_pt = ((pred - clean) ** 2).mean(dim=(1, 2)).clamp_min(1e-20)
        s_psnr += float((10.0 * torch.log10(peak ** 2 / mse_pt)).sum())

        # --- spectra ---
        cfft = torch.fft.rfft(clean, dim=-1)
        pfft = torch.fft.rfft(pred, dim=-1)
        cmag, pmag = cfft.abs(), pfft.abs()
        s_mag += float((pmag - cmag).abs().mean()) * bs
        s_pow_num += float((pmag ** 2 - cmag ** 2).abs().sum())
        s_pow_den += float((cmag ** 2).sum())

        # --- wrapped phase error (unweighted + magnitude-weighted) ---
        dphi = wrapped_phase_difference(torch.angle(pfft), torch.angle(cfft)).abs()
        s_ph_unw += float(dphi.mean()) * bs
        w = cmag / cmag.amax(dim=-1, keepdim=True).clamp_min(1e-12)
        s_ph_w_num += float((w * dphi).sum())
        s_ph_w_den += float(w.sum())

        # --- amplitude bias + SNR (per trace-channel, numpy) ---
        c = clean.detach().cpu().numpy()
        no = noisy.detach().cpu().numpy()
        p = pred.detach().cpu().numpy()
        dA_list.append(signed_peak_amplitude_bias(c, p).ravel())     # (B*C,)
        snr_list.append(
            paper_input_snr(c, no, exclude_half_width_samples=exclude_half_width).snr.ravel()
        )
        # --- peak-time error (samples) for timing efficiency ---
        dt_list.append(
            np.abs(_envelope_peak_index(p) - _envelope_peak_index(c)).ravel()   # (B*C,)
        )

    dA = np.concatenate(dA_list)
    snr = np.concatenate(snr_list)
    dt = np.concatenate(dt_list)
    return {
        "time_mse": s_mse / n,
        "time_l1": s_l1 / n,
        "psnr_db": s_psnr / n,
        "mag_spec_l1": s_mag / n,
        "power_spec_relerr": s_pow_num / max(s_pow_den, 1e-20),
        "phase_err_unw": s_ph_unw / n,
        "phase_err_magweighted": s_ph_w_num / max(s_ph_w_den, 1e-20),
        "_dA": dA,
        "_snr": snr,
        "_dt": dt,
    }


def main():
    ap = argparse.ArgumentParser(description="Phase-aware eval of phase-loss cells.")
    ap.add_argument("--compare-dir", default=str(ROOT_DIR / "results" / "phase_loss_compare"))
    ap.add_argument("--data-path", default="/sps/grand/blevy/sims/sims_for_denoising_sept2025")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--exclude-half-width", type=int, default=64,
                    help="Off-pulse SNR exclusion half-width (samples). Provisional; "
                         "replace with the recovered production value after the audit.")
    ap.add_argument("--lowsnr-lo", type=float, default=3.0)
    ap.add_argument("--lowsnr-hi", type=float, default=5.0)
    ap.add_argument("--timing-threshold", type=float, default=10.0,
                    help="Peak-time pass threshold in SAMPLES (paper timing-efficiency "
                         "convention). The sample<->ns conversion is provisional pending "
                         "the task-4 sampling-dt audit.")
    args = ap.parse_args()

    compare_dir = Path(args.compare_dir)
    manifest = compare_dir / "split_manifest.npz"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing {manifest}; run the comparison first.")
    m = np.load(manifest)
    valid_idx = m["valid"]
    print(f"Frozen split loaded: valid n={valid_idx.size} (seed={int(m['seed'])})", flush=True)

    print("Loading data...", flush=True)
    noise_signals, clean_signals = produce_noise_and_noiseless_data(args.data_path)
    valid_ds = CustomDataset(clean_signals, [noise_signals], indices=valid_idx, swap_prob=0.0, no_random=True)
    loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    cell_dirs = sorted(glob.glob(str(compare_dir / "cell_*")))
    if not cell_dirs:
        raise RuntimeError(f"No cell_* directories under {compare_dir}")

    rows = []
    for cd in cell_dirs:
        cd = Path(cd)
        cfg_path, ckpt = cd / "config.json", cd / "best_model.pth"
        if not ckpt.exists():
            print(f"  skip {cd.name}: no best_model.pth (cell not finished?)", flush=True)
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        model = DualBranchAutoencoder(cfg["model_config"]).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        print(f"Evaluating {cd.name} ...", flush=True)
        met = eval_one_model(model, loader, device, args.exclude_half_width)

        dA, snr, dt = met.pop("_dA"), met.pop("_snr"), met.pop("_dt")
        low = (snr >= args.lowsnr_lo) & (snr < args.lowsnr_hi)
        met["amp_bias_median"] = float(np.median(dA))
        met["amp_bias_median_lowsnr"] = float(np.median(dA[low])) if low.any() else float("nan")
        # Timing efficiency: fraction of trace-channels with |peak-time error| <= threshold.
        met["timing_pass_frac"] = float(np.mean(dt <= args.timing_threshold))
        met["timing_pass_frac_lowsnr"] = (
            float(np.mean(dt[low] <= args.timing_threshold)) if low.any() else float("nan"))

        met.update({"weighting": cfg["weighting"], "phase_weight": cfg["phase_weight"]})
        rows.append(met)

    # Write CSV
    cols = ["weighting", "phase_weight", "psnr_db", "time_mse", "time_l1",
            "mag_spec_l1", "power_spec_relerr", "phase_err_unw",
            "phase_err_magweighted", "amp_bias_median", "amp_bias_median_lowsnr",
            "timing_pass_frac", "timing_pass_frac_lowsnr"]
    out_csv = compare_dir / "eval_phase_loss_metrics.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(rows, key=lambda r: -r["psnr_db"]):
            w.writerow([r["weighting"], r["phase_weight"]]
                       + [f"{r[c]:.6g}" for c in cols[2:]])
    print(f"\nSaved: {out_csv}", flush=True)

    # Console summary, sorted by the phase-weighted phase error (the headline)
    print("\n=== phase-aware comparison (sorted by magnitude-weighted phase error) ===", flush=True)
    print(f"{'weighting':17s}{'pw':>5}  {'PSNR':>7} {'phase_w':>9} {'phase_unw':>9} "
          f"{'magL1':>9} {'ampbias_lo':>11} {'timing_lo':>10}", flush=True)
    for r in sorted(rows, key=lambda r: r["phase_err_magweighted"]):
        print(f"{r['weighting']:17s}{r['phase_weight']:>5}  {r['psnr_db']:7.3f} "
              f"{r['phase_err_magweighted']:9.5f} {r['phase_err_unw']:9.5f} "
              f"{r['mag_spec_l1']:9.4f} {r['amp_bias_median_lowsnr']:11.4f} "
              f"{r['timing_pass_frac_lowsnr']:10.4f}", flush=True)


if __name__ == "__main__":
    main()
