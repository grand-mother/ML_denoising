#!/usr/bin/env python
# coding: utf-8

# 
# # Ablation Test: Time Domain + Frequency Domain vs Time Domain Only
# 
# This notebook performs an ablation test by comparing `DualBranchAutoencoder` and `TimeOnlyAutoencoder`. 
# The hyperparameters have been carefully selected so that both models have an equivalent number of parameters (~2.02M).
# 

# In[1]:


import sys
import os
from pathlib import Path
import json
import os

OUT_DIR = "figures_csv"
os.makedirs(OUT_DIR, exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Adjust the path to include the ML_denoising root directory
ROOT_DIR = Path("/pbs/home/o/omacias/Sam_project/raytune_lib_final").resolve()
sys.path.insert(0, str(ROOT_DIR))

from training.models.cnn import DualBranchAutoencoder, TimeOnlyAutoencoder
from training.raytune_training_function import CustomDataset, split_indices, multi_domain_l1_loss, psnr, get_peak_amplitude
from utils.data_preprocessing import produce_noise_and_noiseless_data

# Ensure we're running on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# In[2]:


# Config: Ensure parameter sizes are matched

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# 1. Retain Frequency Branch (Control Group / Full Model)
# ==========================================
dual_config = {
    'use_freq_branch': True,    
    'time_branch': {'conv_channels': 32, 'res_channels': (128, 256)},
    'freq_branch': {'conv_channels': 16, 'res_channels': (128, 256)},
    'decoder_channels': [128, 64, 32, 3]
}

# ==========================================
# 2. Disable Frequency Branch (Experimental Group / Ablation)
# ==========================================
time_only_config = {
    'use_freq_branch': False,   
    'time_branch': {'conv_channels': 32, 'res_channels': (128, 256)},
    'freq_branch': {'conv_channels': 16, 'res_channels': (128, 256)}, 
    'decoder_channels': [128, 64, 32, 3]
}

model_dual = DualBranchAutoencoder(dual_config).to(device)
model_time = DualBranchAutoencoder(time_only_config).to(device)

print(f"DualBranchAutoencoder params: {count_parameters(model_dual)}")
print(f"DualBranch_NoFreq params:   {count_parameters(model_time)}")

# Save configs for record keeping
ablation_configs = {
    "DualBranch": dual_config,
    "TimeOnly": time_only_config
}
with open(os.path.join(OUT_DIR, "ablation_configs.json"), "w") as f:
    json.dump(ablation_configs, f, indent=4)


# In[3]:


# Data Loading Setup
# NOTE: Make sure `sim_data_dir` points to your dataset.
sim_data_dir = "/sps/grand/blevy/sims/sims_for_denoising_sept2025"

print("Loading data... (This might take a while)")
try:
    noise_signals, clean_signals = produce_noise_and_noiseless_data(sim_data_dir)
    clean_signals = np.array(clean_signals)  # Load entirely into RAM to avoid I/O bottleneck
    print("Data loaded successfully.")
    
    # Split the dataset
    n_samples = noise_signals.shape[1]
    train_indices, valid_indices, test_indices = split_indices(n_samples)
    
    train_dataset = CustomDataset(clean_signals, [noise_signals], indices=train_indices, 
                                    swap_prob=0.5, target_start=120, target_end=480, voltage_to_adc=True)


    valid_dataset = CustomDataset(clean_signals, [noise_signals], indices=valid_indices,
                                    swap_prob=0.5, target_start=120, target_end=480, voltage_to_adc=True)


    test_dataset = CustomDataset(clean_signals, [noise_signals], indices=test_indices,
                            swap_prob=0.5, target_start=120, target_end=480, voltage_to_adc=True)
    
    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    
    data_available = True
except Exception as e:
    print(f"Failed to load data from {sim_data_dir}.")
    print(e)
    data_available = False


# In[4]:


import os
import csv

def train_model(model, name, epochs=50, lr=1e-4, force_train=False):
    if not data_available:
        print("Data not available. Skipping training.")
        return [], []
        
    ckpt_path = os.path.join(OUT_DIR, f"{name}_best.pt")
    csv_path = os.path.join(OUT_DIR, f"{name}_loss_history.csv")
    
    # Check if checkpoint exists
    if os.path.exists(ckpt_path):
        print(f"[{name}] Checkpoint found at {ckpt_path}. Loading weights...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        # If we don't want to force training after loading, just return
        if not force_train:
            print(f"[{name}] Skipping training. Set force_train=True to resume/force training.")
            
            # 尝试读取历史 CSV loss
            if os.path.exists(csv_path):
                print(f"[{name}] Loading loss history from {csv_path}...")
                t_loss, v_loss = [], []
                with open(csv_path, mode='r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        t_loss.append(float(row['train_loss']))
                        v_loss.append(float(row['val_loss']))
                return t_loss, v_loss
                
            return [], []
        else:
            print(f"[{name}] Resuming training from checkpoint...")
            
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        for batch_idx, (noised, clean) in enumerate(train_loader):
            noised, clean = noised.to(device), clean.to(device)
            optimizer.zero_grad()
            
            output = model(noised)
            loss = multi_domain_l1_loss(clean, output, mag_weight=0.1, phase_weight=0.1)
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for noised, clean in valid_loader:
                noised, clean = noised.to(device), clean.to(device)
                output = model(noised)
                loss = multi_domain_l1_loss(clean, output, mag_weight=0.1, phase_weight=0.1)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(valid_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        
        print(f"[{name}] Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), ckpt_path)
            
        # ----------------------------------------------------
        # NEW: 实时把每个 epoch 的结果写入 CSV
        # ----------------------------------------------------
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss'])
            for e, (t_l, v_l) in enumerate(zip(train_losses, val_losses), 1):
                writer.writerow([e, t_l, v_l])
            
    return train_losses, val_losses


# In[5]:


# ==========================================
# 1. Retain Frequency Branch (Control Group / Full Model)
# ==========================================
config_with_freq = {
    'use_freq_branch': True,    # Keep frequency branch enabled
    'time_branch': {'conv_channels': 32, 'res_channels': (128, 256)},
    'freq_branch': {'conv_channels': 16, 'res_channels': (128, 256)},
    'decoder_channels': [128, 64, 32, 3]
}
model_with_freq = DualBranchAutoencoder(config_with_freq).to(device)
# ==========================================
# 2. Disable Frequency Branch (Experimental Group / Ablation)
# ==========================================
config_no_freq = {
    'use_freq_branch': False,   # Disable frequency branch
    'time_branch': {'conv_channels': 32, 'res_channels': (128, 256)},
    # The following line is optional since it will be ignored when use_freq_branch is False
    'freq_branch': {'conv_channels': 16, 'res_channels': (128, 256)}, 
    'decoder_channels': [128, 64, 32, 3]
}
model_no_freq = DualBranchAutoencoder(config_no_freq).to(device)


# In[ ]:


epochs = 150 #Adjust based on time/resources
print("Starting training for DualBranchAutoencoder...")
train_loss_freq, val_loss_freq = train_model(model_with_freq, "DualBranch_WithFreq", epochs=epochs, force_train=True)


print("\nStarting training for TimeOnlyAutoencoder...")
train_loss_nofreq, val_loss_nofreq = train_model(model_no_freq, "DualBranch_NoFreq", epochs=epochs, force_train=True)


# ##### Loss vs Epochs 

# In[ ]:


if data_available:
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_freq, label='Train')
    plt.plot(val_loss_freq, label='Validation')
    plt.title('DualBranch Autoencoder Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Multi-Domain L1 Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_loss_nofreq, label='Train')
    plt.plot(val_loss_nofreq, label='Validation')
    plt.title('TimeOnly Autoencoder Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Multi-Domain L1 Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'ablation_learning_curves.pdf'))
    # plt.show()


# ##### Evaluation plot

# In[ ]:


def evaluate_model(model, name):
    if not data_available:
        return {"psnr_avg": 0, "peak_ratio_avg": 0}
        
    model.eval()
    psnr_list = []
    peak_ratio_list = []
    
    print(f"Evaluating {name} on test set...")
    with torch.no_grad():
        for noised, clean in test_loader:
            noised, clean = noised.to(device), clean.to(device)
            output = model(noised)
            
            clean_np = clean.cpu().numpy().squeeze()
            out_np = output.cpu().numpy().squeeze()
            
            # Use max peak amplitude as scale for PSNR
            peak_amp = get_peak_amplitude(clean_np)
            current_psnr = psnr(out_np, clean_np, peak_amp)
            psnr_list.append(current_psnr)
            
            # Peak to peak ratio
            out_peak = get_peak_amplitude(out_np)
            ratio = abs(peak_amp - out_peak) / peak_amp
            peak_ratio_list.append(ratio)
            
    return {
        "psnr_avg": float(np.mean(psnr_list)),
        "psnr_std": float(np.std(psnr_list)),
        "peak_ratio_avg": float(np.mean(peak_ratio_list)),
        "peak_ratio_std": float(np.std(peak_ratio_list))
    }


# #### best checking pt

# In[ ]:


if data_available:
    # Load best weights before evaluating
    if os.path.exists("DualBranch_best.pt"):
        model_dual.load_state_dict(torch.load("DualBranch_WithFreq_best.pt", map_location=device, weights_only=True))
    if os.path.exists("TimeOnly_best.pt"):
        model_time.load_state_dict(torch.load("DualBranch_NoFreq_best.pt", map_location=device, weights_only=True))
        
    dual_results = evaluate_model(model_dual, "DualBranch")
    time_results = evaluate_model(model_time, "TimeOnly")

    results = {
        "DualBranch": dual_results,
        "TimeOnly": time_results
    }

    # Save to JSON
    with open(os.path.join(OUT_DIR, "ablation_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    # Save to CSV
    import pandas as pd
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(OUT_DIR, "ablation_metrics.csv"))
    
    print("\nFinal Test Metrics:")
    print(df)


# In[ ]:


import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
# Ensure the paths are available
sys.path.append("/pbs/home/o/omacias/Sam_project/raytune_lib_final")
from visualization.plot_usable_antenna_vs_SNR.make_fig_event_multiplicity_usable_nmse import run_in_notebook
print("Collecting test set waveforms for evaluation...")
clean_list, noisy_list, dual_list, time_list = [], [], [], []
model_dual.eval()
model_time.eval()
with torch.no_grad():
    for noisy, clean in test_loader:
        noisy_device = noisy.to(device)
        
        # Get model predictions
        out_dual = model_dual(noisy_device).cpu().numpy()
        out_time = model_time(noisy_device).cpu().numpy()
        
        noisy_np = noisy.numpy()
        clean_np = clean.numpy()
        
        # Ensure shapes are (N, 3, T)
        if noisy_np.ndim == 4: noisy_np = noisy_np.squeeze(1)
        if clean_np.ndim == 4: clean_np = clean_np.squeeze(1)
        if out_dual.ndim == 4: out_dual = out_dual.squeeze(1)
        if out_time.ndim == 4: out_time = out_time.squeeze(1)
            
        noisy_list.append(noisy_np)
        clean_list.append(clean_np)
        dual_list.append(out_dual)
        time_list.append(out_time)
noisy_all = np.concatenate(noisy_list, axis=0)
clean_all = np.concatenate(clean_list, axis=0)
dual_all = np.concatenate(dual_list, axis=0)
time_all = np.concatenate(time_list, axis=0)
# Generate 1-to-1 event IDs assuming independent traces
event_ids = np.arange(len(clean_all))
print("Plotting Usable Antenna Fraction...")
# labels=("TimeOnly", "DualBranch") maps to ("Standard", "ML") in the plotting script
res = run_in_notebook(
    clean=clean_all,
    noisy=noisy_all,
    denoised=dual_all,
    standard=time_all,
    event_ids=event_ids,
    labels=("TimeOnly", "DualBranch")
)


# In[ ]:


def get_peak_time(trace):
    # Returns peak time in ns assuming dt=2.0 ns
    dt_ns = 2.0
    return np.argmax(np.abs(trace), axis=-1) * dt_ns

# Reuse the SNR values computed by run_in_notebook above
snr_vals = res["snr_trace"]  # Shape: (N, 3)

plt.figure(figsize=(18, 5))
snr_bins = np.linspace(1, 15, 15)

channels = ["X Channel", "Y Channel", "Z Channel"]

for ch in range(3):
    plt.subplot(1, 3, ch+1)
    
    clean_times = get_peak_time(clean_all[:, ch, :])
    dual_times = get_peak_time(dual_all[:, ch, :])
    time_times = get_peak_time(time_all[:, ch, :])
    
    # Efficiency: Fraction of traces where |predicted peak - clean peak| <= 10 ns
    dual_eff = []
    time_eff = []
    
    snr_ch = snr_vals[:, ch]
    for i in range(len(snr_bins)-1):
        mask = (snr_ch >= snr_bins[i]) & (snr_ch < snr_bins[i+1])
        if np.sum(mask) == 0:
            dual_eff.append(np.nan)
            time_eff.append(np.nan)
            continue
        dual_eff.append(np.mean(np.abs(dual_times[mask] - clean_times[mask]) <= 10))
        time_eff.append(np.mean(np.abs(time_times[mask] - clean_times[mask]) <= 10))
        
    plt.step(snr_bins[:-1], dual_eff, where='post', label='DualBranch', color='k', linewidth=2.5)
    plt.step(snr_bins[:-1], time_eff, where='post', label='TimeOnly', color='C1', linewidth=2.5, linestyle='--')
    
    plt.xlabel('Signal-to-Noise Ratio (SNR)', fontsize=14)
    if ch == 0:
        plt.ylabel('Denoising Efficiency ($|\\Delta t| \\leq 10$ns)', fontsize=14)
    plt.title(f'Peak Time Efficiency - {channels[ch]}', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.xlim(1.5, 14)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'peak_time_efficiency.pdf'), bbox_inches='tight')
# plt.show()


# In[ ]:


import torch
import numpy as np
from visualization.nmse_snr_gain_vs_snr.make_fig_nmse_snr_gain_vs_snr__option3_truth_cleanpower_gate import plot_nmse_and_snr_gain_vs_snr, FidelityPlotConfig

print("Running inference on test set...")
model_dual.eval()
model_time.eval()

noisy_list = []
clean_list = []
den_dual_list = []
den_time_list = []

# Collect all waveforms from the test loader
with torch.no_grad():
    for noised, clean in test_loader:
        noised_device = noised.to(device)
        
        # Inference for both models
        out_dual = model_dual(noised_device)
        out_time = model_time(noised_device)
        
        # Move back to CPU and numpy
        noisy_list.append(noised.numpy())
        clean_list.append(clean.numpy())
        den_dual_list.append(out_dual.cpu().numpy())
        den_time_list.append(out_time.cpu().numpy())

# Concatenate along the batch dimension -> Shape: (N, 3, T)
noisy_arr = np.concatenate(noisy_list, axis=0)
clean_arr = np.concatenate(clean_list, axis=0)
den_dual_arr = np.concatenate(den_dual_list, axis=0)
den_time_arr = np.concatenate(den_time_list, axis=0)

print(f"Collected waveforms shape: {clean_arr.shape}")

# Configure the plot settings
# We assign TimeOnly as the "standard" (orange) and DualBranch as the "ML" (blue) model
cfg = FidelityPlotConfig(
    labels=("Noisy", "TimeOnly", "DualBranch"),
    colors=("tab:red", "tab:orange", "tab:blue"),
    # 如果想把图保存下来可以加上这行：
    # savepath="ablation_nmse_snrgain_vs_snr.pdf"
)

print("Generating fidelity plot...")
# This will plot the top row (NMSE vs SNR) and bottom row (ΔSNR_out vs SNR)
plot_nmse_and_snr_gain_vs_snr(
    clean_waveforms=clean_arr,
    noisy_waveforms=noisy_arr,
    denoised_waveforms=den_dual_arr,       # This gets the 'ML' label (DualBranch) den_dual_arr
    standard_waveforms=den_time_arr,       # This gets the 'Standard' label (TimeOnly) den_time_arr
    cfg=cfg
)


# In[ ]:


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def ablation_traces_plot_comparison(testloader, 
                                    models,
                                    num_images=6, 
                                    device="cpu", 
                                    save_path='',
                                    snr_min=1,
                                    snr_max=4,
                                    peak_amp_threshold=15):
    """
    Plot traces comparison specifically for ablation study models.
    Automatically adjusts the number of rows based on len(models).
    """
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    with torch.no_grad():  
        device = torch.device(device)
        for model in models.values():
            model.to(device).eval()
            
        model_names = list(models.keys())
        num_models = len(model_names)
        
        # Assign colors to each model
        color_palette = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple']
        colors = {name: color_palette[i % len(color_palette)] for i, name in enumerate(model_names)}
        
        count = 0
        for noisy_data, clean_data in testloader:
            if count >= num_images: 
                break
            
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            
            # Get outputs from all models
            model_outputs = {}
            for model_name, model in models.items():
                model_outputs[model_name] = model(noisy_data)
            
            batch_size = noisy_data.size(0)
            for sample_idx in range(batch_size):
                if count >= num_images:
                    break
                
                # Check SNR and peak amplitude for all channels first
                valid_sample = True
                snrs = []
                peak_amps = []
                
                for channel_idx in range(3):
                    clean_np = clean_data[sample_idx, channel_idx].cpu().numpy()
                    noisy_np = noisy_data[sample_idx, channel_idx].cpu().numpy()
                    
                    if np.std(noisy_np) != 0:
                        snr = np.max(clean_np) / np.std(noisy_np)
                    else:
                        snr = float('inf')
                    
                    envelope_clean = np.abs(hilbert(clean_np))
                    peak_amp = np.max(envelope_clean)
                    
                    snrs.append(snr)
                    peak_amps.append(peak_amp)
                    
                    if not (snr_min < snr < snr_max) or peak_amp <= peak_amp_threshold:
                        valid_sample = False
                        break
                
                if not valid_sample:
                    continue
            
                channel_names = ['X Channel', 'Y Channel', 'Z Channel']
                
                # Create Grid: num_models x 3
                fig, axes = plt.subplots(num_models, 3, figsize=(30, 8 * num_models))
                if num_models == 1:
                    axes = np.expand_dims(axes, axis=0)
                
                fontsize = 30
                
                for row_idx, model_name in enumerate(model_names):
                    for channel_idx in range(3):
                        ax = axes[row_idx, channel_idx]
                        
                        clean_np = clean_data[sample_idx, channel_idx].cpu().numpy()
                        noisy_np = noisy_data[sample_idx, channel_idx].cpu().numpy()
                        denoised_np = model_outputs[model_name][sample_idx, channel_idx].cpu().numpy()
                        
                        snr = snrs[channel_idx]
                        
                        # Calculate PSNR
                        psnr_value = psnr(clean_np, denoised_np, np.max(clean_np))
                        
                        time_bin = np.arange(0, clean_np.size)
                        envelope_true = np.abs(hilbert(clean_np))
                        peak_time_true = time_bin[np.argmax(envelope_true)]
                        
                        # Plot traces
                        ax.plot(noisy_np, label='Noisy', linestyle=':', color='tab:red', linewidth=2, alpha=0.7)
                        ax.plot(clean_np, label='Clean', color='black', linewidth=3)
                        ax.plot(denoised_np, label='Denoised', color=colors[model_name], alpha=0.9, linewidth=3)
                        
                        ax.set_xlim(peak_time_true - 50, peak_time_true + 100)
                        
                        # Add performance metrics
                        metrics_text = f"SNR = {snr:.2f}\nPSNR = {psnr_value:.2f}"
                        ax.text(0.98, 0.95, metrics_text, 
                               transform=ax.transAxes, 
                               fontsize=fontsize-6, 
                               verticalalignment='top', 
                               horizontalalignment='right',
                               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
                        
                        # Column titles
                        if row_idx == 0:
                            ax.set_title(f'{channel_names[channel_idx]}', fontsize=fontsize, pad=20)
                        
                        # Model name
                        if channel_idx == 0:
                            ax.text(0.02, 0.02, f'{model_name}', 
                                   transform=ax.transAxes, 
                                   fontsize=fontsize-4, 
                                   verticalalignment='bottom', 
                                   horizontalalignment='left',
                                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round'))
                        
                        # Legend
                        if row_idx == 0 and channel_idx == 0:
                            legend = ax.legend(fontsize=fontsize-8, loc='lower right')
                            legend.get_frame().set_edgecolor('none')
                        
                        # X-axis label
                        if row_idx == num_models - 1 and channel_idx == 1:
                            ax.set_xlabel('Time [ns]', fontsize=fontsize)
                        
                        # Y-axis label
                        if channel_idx == 0:
                            ax.set_ylabel('ADC counts', fontsize=fontsize)
                        
                        ax.tick_params(axis='both', labelsize=fontsize-6)
                        ax.grid(True, alpha=0.3)

                plt.tight_layout(rect=[0.02, 0.02, 1, 0.96])
                
                filename = f'ablation_traces_comparison_{count:03d}_snr_{snrs[0]:.2f}.pdf'
                plt.savefig(os.path.join(save_path, filename) if save_path else filename, dpi=200, bbox_inches='tight')
                # plt.show()
                count += 1

            if count >= num_images:
                break
    
    print('Traces comparison completed')

# ==============================================================
# 调用部分 (请在同一个 Cell 下方执行即可)
# ==============================================================

# 把您要比较的模型放在字典里
models_to_compare = {
    'Time Only': model_time,
    'Dual Branch': model_dual
}

# 调用函数生成图像
# snr_min 和 snr_max 决定了我们要挑选在什么信噪比范围内的事件来画图
ablation_traces_plot_comparison(
    testloader=test_loader, 
    models=models_to_compare,
    num_images=4,            # 要画多少张图
    device='cuda',           # 使用 gpu
    snr_min=5,               # 最低 SNR 要求
    snr_max=10,               # 最高 SNR 要求
    peak_amp_threshold=15,    # 幅度要求
    save_path=OUT_DIR
)


# In[ ]:




