#!/usr/bin/env python
"""
Regenerate the example waveform / spectrum figure (`traces_plot_time_frequency`)
at an explicit sampling interval.

This reproduces cell 22 of `visualization/notebooks/overleaf.ipynb`:

    traces_plot_time_frequency(testloader=test_loader, model=multi_l1_model,
                               num_images=2, device="cpu", save_path=save_path,
                               x_channel_snr=4.0, y_channel_snr=3.0,
                               z_channel_snr=2.0)

with two differences, both to make the output reproducible and comparable
across dt values:

  * the 80/10/10 split is seeded (`deterministic_split_indices`, seed 12345)
    instead of the notebook's unseeded `split_indices`;
  * the test dataset uses `swap_prob=0.0` instead of 0.5, so no augmentation is
    applied when picking the example trace.

Sample selection depends only on per-channel SNR and clean peak amplitude, both
of which are independent of `dt_ns`, so every `--dt-ns` value plots the *same*
trace and the figures differ only through the two axes that dt controls:

    time axis      t = sample_index * dt_ns
    frequency axis f = rfftfreq(N, d=dt_ns*1e-9)

Model: the production multi_v3 checkpoint.
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.dirname(_ROOT) not in sys.path:
    sys.path.insert(0, os.path.dirname(_ROOT))

from training.models.cnn import DualBranchAutoencoder as CNN
from training.raytune_training_function import CustomDataset
from utils.data_preprocessing import produce_noise_and_noiseless_data
from utils.referee_revision_utils import deterministic_split_indices
from evaluation.physics_impact.physics_impact import load_best_trial_from_files
from visualization.overleaf_plots import traces_plot_time_frequency

PRODUCTION_DIR = "/sps/grand/macias/Sam_Result/multi_v3_CNN_100epochs_36samples"
DATA_PATH = "/sps/grand/blevy/sims/sims_for_denoising_sept2025"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dt-ns", type=float, required=True,
                    help="Sampling interval used for BOTH the time and the frequency axis.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-images", type=int, default=2)
    ap.add_argument("--model-dir", default=PRODUCTION_DIR)
    ap.add_argument("--data-path", default=DATA_PATH)
    ap.add_argument("--split-seed", type=int, default=12345)
    ap.add_argument("--eval-len", type=int, default=512,
                    help="Deterministic evaluation crop length. Must be the training "
                         "length (512); leaving CustomDataset to default to 1024 "
                         "degrades the reconstruction by roughly 8 dB PSNR.")
    ap.add_argument("--crop-mode", choices=("deterministic", "notebook"),
                    default="deterministic",
                    help="'deterministic': no_random=True + eval_len, no augmentation - "
                         "reproducible, and the convention used by the other revision "
                         "figures. 'notebook': no_random=False + swap_prob=0.5, i.e. the "
                         "randomised crop the published notebook cell used. Both feed the "
                         "model 512-sample traces; they differ in where the 512-sample "
                         "window is taken from and whether swap augmentation is applied.")
    ap.add_argument("--x-channel-snr", type=float, default=4.0)
    ap.add_argument("--y-channel-snr", type=float, default=3.0)
    ap.add_argument("--z-channel-snr", type=float, default=2.0)
    args = ap.parse_args()

    np.random.seed(args.split_seed)
    torch.manual_seed(args.split_seed)

    print(f"=== traces_plot_time_frequency at dt = {args.dt_ns} ns ===", flush=True)

    metrics_json = os.path.join(args.model_dir, "best_trial_metrics.json")
    config_json = os.path.join(args.model_dir, "best_trial_config.json")
    model_path = os.path.join(args.model_dir, "best_model.pth")

    _, _, model = load_best_trial_from_files(
        metrics_json_path=metrics_json,
        config_json_path=config_json,
        model_path=model_path,
        model_classes={"CNN": CNN},
        device="cpu",
    )

    print("Loading simulation data...", flush=True)
    noise_signals, clean_signals = produce_noise_and_noiseless_data(args.data_path)
    total = clean_signals.shape[1]
    _, _, test_indices = deterministic_split_indices(
        total, train_fraction=0.8, valid_fraction=0.1, seed=args.split_seed
    )
    print(f"test split: {len(test_indices)} traces (seed {args.split_seed})", flush=True)

    if args.crop_mode == "deterministic":
        crop_kwargs = dict(
            swap_prob=0.0,          # no augmentation for a figure
            no_random=True,         # deterministic crop
            # eval_len is REQUIRED alongside no_random. CustomDataset falls back to
            # `initial_len` (1024) when eval_len is None (raytune_training_function.py:142),
            # so omitting it feeds the model 1024-sample traces although it was trained
            # on 512. 512 is the production training length.
            eval_len=args.eval_len,
        )
        n_workers = 4
    else:
        # Exactly what the published notebook cell did: a randomised 512-sample crop
        # positioned so the pulse stays inside the window, plus 50 % swap augmentation.
        crop_kwargs = dict(swap_prob=0.5, no_random=False, traces_len=args.eval_len)
        # num_workers=0 so the single seeded numpy stream drives every random draw and
        # the selected example trace is reproducible.
        n_workers = 0
    print(f"crop mode: {args.crop_mode}  ->  {crop_kwargs}", flush=True)

    test_dataset = CustomDataset(
        clean_signals,
        [noise_signals],
        indices=test_indices,
        target_start=120,
        target_end=480,
        voltage_to_adc=True,
        **crop_kwargs,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=n_workers,
                             shuffle=False, pin_memory=False)

    os.makedirs(args.out_dir, exist_ok=True)
    traces_plot_time_frequency(
        testloader=test_loader,
        model=model,
        num_images=args.num_images,
        device="cpu",
        save_path=args.out_dir,
        dt_ns=args.dt_ns,
        x_channel_snr=args.x_channel_snr,
        y_channel_snr=args.y_channel_snr,
        z_channel_snr=args.z_channel_snr,
    )
    print(f"Wrote figures to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
