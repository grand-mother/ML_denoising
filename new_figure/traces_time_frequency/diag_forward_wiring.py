"""Is the ~8 dB PSNR deficit caused by the rewired DualBranchAutoencoder.forward?

The multi_v3 checkpoint was trained with the ORIGINAL forward (commit 368f8fd):
frequency features keep their spectral axis, all three branches are interpolated
to a common length (~64) and concatenated. During the revision the forward was
rewired (adaptive_avg_pool1d to a per-channel scalar, broadcast over time,
decoder now runs at T=128 and the output is interpolated 1024->512). Parameter
shapes are identical, so strict load_state_dict passes silently either way.

This script loads the SAME checkpoint into both wirings and evaluates the SAME
traces through both, reporting median PSNR. If old-wiring PSNR ~= the notebook's
~32 dB and new-wiring ~= the ~23 dB seen in the regenerated figures, the rewiring
is the cause.
"""
import os, sys, json
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

R = "/pbs/home/o/omacias/Sam_project/raytune_lib_final"
sys.path.insert(0, R); sys.path.insert(0, os.path.dirname(R))

from training.models import cnn as cnn_mod
from training.models.cnn import DualBranchAutoencoder
from training.raytune_training_function import CustomDataset, psnr
from utils.data_preprocessing import produce_noise_and_noiseless_data
from utils.referee_revision_utils import deterministic_split_indices

M = "/sps/grand/macias/Sam_Result/multi_v3_CNN_100epochs_36samples"
with open(f"{M}/best_trial_config.json") as f:
    config = json.load(f)
model_config = config["model_config"]

state = torch.load(f"{M}/best_model.pth", map_location="cpu")


def forward_old(self, x):
    """The ORIGINAL forward from commit 368f8fd (training-time wiring)."""
    if torch.isnan(x).any():
        x = torch.nan_to_num(x, nan=0.0)
    time_features = self.time_branch(x)
    x_fft = torch.fft.rfft(x, dim=-1)
    magnitude = torch.abs(x_fft)
    phase = torch.angle(x_fft)
    mag_features = self.magnitude_branch(magnitude)
    phase_features = self.phase_branch(phase)
    min_size = min(time_features.size(2), mag_features.size(2), phase_features.size(2))
    time_features = F.interpolate(time_features, size=min_size, mode='linear', align_corners=False)
    mag_features = F.interpolate(mag_features, size=min_size, mode='linear', align_corners=False)
    phase_features = F.interpolate(phase_features, size=min_size, mode='linear', align_corners=False)
    combined = torch.cat((time_features, mag_features, phase_features), dim=1)
    fused = self.fusion_layer(combined)
    out = self.decoder(fused)
    if out.shape[-1] != x.shape[-1]:
        out = F.interpolate(out, size=x.shape[-1], mode='linear', align_corners=False)
    return out


# Two instances of the same checkpoint: current wiring vs original wiring.
model_new = DualBranchAutoencoder(model_config); model_new.load_state_dict(state); model_new.eval()
model_old = DualBranchAutoencoder(model_config); model_old.load_state_dict(state); model_old.eval()
model_old.forward = forward_old.__get__(model_old, DualBranchAutoencoder)

noise_signals, clean_signals = produce_noise_and_noiseless_data(
    "/sps/grand/blevy/sims/sims_for_denoising_sept2025")
total = clean_signals.shape[1]
_, _, test_idx = deterministic_split_indices(
    total, train_fraction=0.8, valid_fraction=0.1, seed=12345)
sub = test_idx[:400]

ds = CustomDataset(clean_signals, [noise_signals], indices=sub,
                   no_random=True, eval_len=512, swap_prob=0.0,
                   target_start=120, target_end=480, voltage_to_adc=True)
dl = DataLoader(ds, batch_size=32, num_workers=4, shuffle=False)

res = {"new": [], "old": []}
snrs = []
with torch.no_grad():
    for noisy, clean in dl:
        d_new = model_new(noisy).numpy()
        d_old = model_old(noisy).numpy()
        c = clean.numpy(); n = noisy.numpy()
        for i in range(c.shape[0]):
            for ch in range(3):
                res["new"].append(psnr(c[i, ch], d_new[i, ch], np.max(c[i, ch])))
                res["old"].append(psnr(c[i, ch], d_old[i, ch], np.max(c[i, ch])))
                sd = np.std(n[i, ch])
                snrs.append(np.max(c[i, ch]) / sd if sd > 0 else np.inf)

snrs = np.array(snrs)
sel = (snrs > 2) & (snrs < 5)
for k in ("old", "new"):
    a = np.array(res[k])
    print(f"forward={k:3s}  median PSNR = {np.median(a):6.2f} dB   "
          f"(2<SNR<5 subset: {np.median(a[sel]):6.2f} dB, n={sel.sum()})", flush=True)
print("\nmedian per-trace PSNR difference (old - new):",
      f"{np.median(np.array(res['old']) - np.array(res['new'])):.2f} dB")
