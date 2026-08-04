"""
Visualization and plotting functions for publication-ready figures.
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scipy.signal import stft

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from training.raytune_training_function import psnr
from torch.utils.data import Dataset

# Import model classes for comparison
from training.models.cnn import DualBranchAutoencoder

# Paper SNR definition used throughout the analysis:
#
#     SNR = max(clean) / std(noisy)
#
# with the standard deviation evaluated over the FULL trace (no off-pulse
# exclusion, no band-limiting, raw signed maximum of the clean trace, computed
# per channel). This is the definition used for every reported figure and is the
# single definition for the whole repository; do not substitute an off-pulse or
# envelope-based variant.
def _paper_snr_1d(clean_np, noisy_np):
    """Per-trace paper SNR for a 1D clean/noisy trace: max(clean)/std(noisy)."""
    denom = float(np.std(np.asarray(noisy_np)))
    if denom == 0.0:
        return float("inf")
    return float(np.max(np.asarray(clean_np)) / denom)

# Sampling interval of the simulated traces. Peak times are computed as sample
# indices, so the timing selections below are sample counts; this constant is the
# single place that converts them to nanoseconds for the axis/legend labels.
# A 10-sample tolerance is therefore 5 ns.
_DT_NS = 0.5  # nanoseconds per sample

def _samples_to_ns_label(n_samples, dt_ns=_DT_NS):
    """Format a sample-count timing threshold as a nanosecond value for labels."""
    return f"{n_samples * dt_ns:g}"

class CustomDataset(Dataset):
    def __init__(self, noised_signals, clean_signals, indices=None):
        """
        Args:
            noised_signals: Tuple of lists containing noised X, Y, Z signal components.
            clean_signals: Tuple of lists containing clean X, Y, Z signal components.
            indices: Array-like list of indices specifying which samples to include.
        """
        self.indices = indices if indices is not None else list(range(len(noised_signals[0])))

        self.noised_signals = noised_signals
        self.clean_signals = clean_signals


    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]

        # Properly access the sample data
        noised_x = self.noised_signals[0][actual_idx]
        noised_y = self.noised_signals[1][actual_idx]
        noised_z = self.noised_signals[2][actual_idx]
        clean_x = self.clean_signals[0][actual_idx]
        clean_y = self.clean_signals[1][actual_idx]
        clean_z = self.clean_signals[2][actual_idx]

        # Convert to PyTorch tensors
        noised_signals = np.stack([noised_x, noised_y, noised_z], axis=0)
        clean_signals = np.stack([clean_x, clean_y, clean_z], axis=0)

        return torch.tensor(noised_signals, dtype=torch.float32), torch.tensor(clean_signals, dtype=torch.float32)
    
def spectral_convergence_stft(clean, pred, n_fft=256, hop_length=128, eps=1e-8):
    # Compute STFT
    f_c, t_c, Zxx_c = stft(clean, nperseg=n_fft, noverlap=n_fft-hop_length)
    f_p, t_p, Zxx_p = stft(pred,  nperseg=n_fft, noverlap=n_fft-hop_length)
    # Compute Frobenius norms
    num = np.linalg.norm(np.abs(Zxx_p) - np.abs(Zxx_c), 'fro')
    denom = np.linalg.norm(np.abs(Zxx_c), 'fro') + eps
    return num / denom

#### Plot the peak time efficiency
def plot_peak_time_efficiency(snr_vals, clean_time, noisy_time, denoised_time, channel_name, save_path=None): #
    """
    Plots the fraction of traces with |Δt| > threshold for denoised and noisy signals as a function of SNR.
    """
    # Create save directory if it doesn't exist
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    snr_bins = np.linspace(1, 20, 20)  # Adjust as needed
    thresholds = [10, 20]  # SAMPLES (peak times are sample indices; previously mislabeled "ns"; convert with dt once recovered)
    colors = ['orange', 'blue']
    linestyles = ['-', '--']

    plt.figure(figsize=(12, 6))
    fontsize = 16
    # Denoising efficiency: fraction of denoised traces with |Δt| <= 10 samples
    denoising_efficiency = []
    for i in range(len(snr_bins)-1):
        mask = (snr_vals >= snr_bins[i]) & (snr_vals < snr_bins[i+1])
        if np.sum(mask) == 0:
            denoising_efficiency.append(np.nan)
            continue
        # Denoising efficiency: fraction with |Δt| <= 10 samples
        denoising_efficiency.append(np.mean(np.abs(denoised_time[mask] - clean_time[mask]) <= 10))
    plt.step(snr_bins[:-1], denoising_efficiency, where='post', color='k', linewidth=2, label=f'Denoising efficiency - {channel_name}')

    # For each threshold and each type (denoised/noisy)
    for idx, threshold in enumerate(thresholds):
        # Denoised (solid)
        frac_denoised = []
        for i in range(len(snr_bins)-1):
            mask = (snr_vals >= snr_bins[i]) & (snr_vals < snr_bins[i+1])
            if np.sum(mask) == 0:
                frac_denoised.append(np.nan)
                continue
            frac_denoised.append(np.mean(np.abs(denoised_time[mask] - clean_time[mask]) > threshold))
        plt.step(snr_bins[:-1], frac_denoised, where='post', color=colors[idx], linestyle='-', label=fr'$\Delta t_{{peak}} > {threshold}$ samples, denoised')

        # Noisy (dashed)
        frac_noisy = []
        for i in range(len(snr_bins)-1):
            mask = (snr_vals >= snr_bins[i]) & (snr_vals < snr_bins[i+1])
            if np.sum(mask) == 0:
                frac_noisy.append(np.nan)
                continue
            frac_noisy.append(np.mean(np.abs(noisy_time[mask] - clean_time[mask]) > threshold))
        plt.step(snr_bins[:-1], frac_noisy, where='post', color=colors[idx], linestyle='--', label=fr'$\Delta t_{{peak}} > {threshold}$ samples, noisy')

    plt.xlabel('Signal-to-Noise ratio (SNR)', fontsize=fontsize)
    plt.ylabel('Fraction', fontsize=fontsize)
    # plt.title(f'Peak Time Efficiency - {channel_name}', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(1, 10)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=fontsize)
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(os.path.join(save_path, f'Peak_Time_Efficiency_{channel_name}.png'))
    plt.show()

def peak_time_analysis(dataloader, 
                   model, 
                   model_path,
                   device: str = 'cpu', 
                   min_snr: int = 1, 
                   max_snr: int = 1e3,
                   save_path: str ='' ): 
    device = torch.device(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()
    peak_times = {'X Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                  'Y Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                  'Z Channel': {'Clean': [], 'Noisy': [], 'Denoised': []}}
    
    snr_values = {'X Channel': [], 'Y Channel': [], 'Z Channel': []}
    channel_names = ['X Channel', 'Y Channel', 'Z Channel']

    with torch.no_grad():
        for noisy_data, clean_data in dataloader:
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            denoised_output = model(noisy_data)

            batch_size = noisy_data.size(0)
            for i in range(batch_size):
                for idx, channel in enumerate(channel_names):  # idx: 0,1,2
                    clean_np = clean_data[i, idx].cpu().numpy()
                    noisy_np = noisy_data[i, idx].cpu().numpy()
                    denoised_np = denoised_output[i, idx].cpu().numpy()
                    timing = np.arange(clean_np.size)
                    
                    if np.std(noisy_np) != 0:
                        snr = _paper_snr_1d(clean_np, noisy_np)
                    else:
                        snr = float('inf')
                    
                    if  max_snr > snr > min_snr:
                        # Calculate envelopes
                        envelope_clean = np.abs(hilbert(clean_np))
                        envelope_noisy = np.abs(hilbert(noisy_np))
                        envelope_denoised = np.abs(hilbert(denoised_np))

                        # Find peak times
                        peak_time_clean = timing[np.argmax(envelope_clean)]
                        peak_time_noisy = timing[np.argmax(envelope_noisy)]
                        peak_time_denoised = timing[np.argmax(envelope_denoised)]

                        # Store peak times
                        peak_times[channel]['Clean'].append(peak_time_clean)
                        peak_times[channel]['Noisy'].append(peak_time_noisy)
                        peak_times[channel]['Denoised'].append(peak_time_denoised)

                        snr_values[channel].append(snr)

    # Example for X channel
    plot_peak_time_efficiency(
        snr_vals=np.array(snr_values['X Channel']),
        clean_time=np.array(peak_times['X Channel']['Clean']),
        noisy_time=np.array(peak_times['X Channel']['Noisy']),
        denoised_time=np.array(peak_times['X Channel']['Denoised']),
        channel_name='X Channel',
        save_path = save_path
    )

    # Example for Y channel
    plot_peak_time_efficiency(
        snr_vals=np.array(snr_values['Y Channel']),
        clean_time=np.array(peak_times['Y Channel']['Clean']),
        noisy_time=np.array(peak_times['Y Channel']['Noisy']),
        denoised_time=np.array(peak_times['Y Channel']['Denoised']),
        channel_name='Y Channel',
        save_path= save_path
    )        

    plot_peak_time_efficiency(
        snr_vals=np.array(snr_values['Z Channel']),
        clean_time=np.array(peak_times['Z Channel']['Clean']),
        noisy_time=np.array(peak_times['Z Channel']['Noisy']),
        denoised_time=np.array(peak_times['Z Channel']['Denoised']),
        channel_name='Z Channel',
        save_path= save_path
    )   

#### Plot amplitude section

def plot_amplitude_ratio_vs_snr(peak_amplitudes, snr_values, channel, save_path=None):
    # Convert to numpy arrays
    snr = np.array(snr_values[channel])
    clean = np.array(peak_amplitudes[channel]['Clean'])
    noisy = np.array(peak_amplitudes[channel]['Noisy'])
    denoised = np.array(peak_amplitudes[channel]['Denoised'])

    # Calculate ratios
    ratio_noisy = noisy / clean
    ratio_denoised = denoised / clean

    # Bin by SNR
    snr_bins = np.linspace(1, 10, 10)
    bin_centers = (snr_bins[:-1] + snr_bins[1:]) / 2
    means_noisy, stds_noisy = [], []
    means_denoised, stds_denoised = [], []

    for i in range(len(snr_bins)-1):
        mask = (snr >= snr_bins[i]) & (snr < snr_bins[i+1])
        means_noisy.append(np.mean(ratio_noisy[mask]) if np.any(mask) else np.nan)
        stds_noisy.append(np.std(ratio_noisy[mask]) if np.any(mask) else np.nan)
        means_denoised.append(np.mean(ratio_denoised[mask]) if np.any(mask) else np.nan)
        stds_denoised.append(np.std(ratio_denoised[mask]) if np.any(mask) else np.nan)

    # Plot
    plt.figure(figsize=(12, 6))
    fontsize = 16
    plt.errorbar(bin_centers, means_denoised, yerr=stds_denoised, fmt='o', color='red', label='denoised')
    plt.errorbar(bin_centers, means_noisy, yerr=stds_noisy, fmt='^', color='orange', label=f'noisy - {channel}')
    plt.axhline(1, color='gray', linestyle='--')
    plt.xlabel('Signal-to-Noise ratio (SNR)', fontsize=fontsize)
    plt.ylabel('Amplitude ratio', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # plt.title(f'Amplitude Ratio vs SNR - {channel}', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    if save_path:
        plt.savefig(os.path.join(save_path, f'Amplitude_Ratio_{channel}.png'))
    plt.show()

def plot_peak_amplitude_distribution(peak_amplitudes, channel, save_path=None):
    # Get data
    clean = np.array(peak_amplitudes[channel]['Clean'])
    denoised = np.array(peak_amplitudes[channel]['Denoised'])
    plt.figure(figsize=(12, 6))
    fontsize = 16
    # Plot histogram
    plt.hist(clean, bins=30, density=True, histtype='step', color='orange', linestyle='--', label=f'clean - {channel}')
    plt.hist(denoised, bins=30, density=True, histtype='step', color='red', label='denoised')
    plt.yscale('log')
    plt.xlabel('Maximum peak amplitude [ADC]', fontsize=fontsize)
    plt.ylabel('Density', fontsize=fontsize)
    # plt.title(f'Peak Amplitude Distribution - {channel}', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    if save_path:
        plt.savefig(os.path.join(save_path, f'Peak_Amplitude_Distribution_{channel}.pdf'))
    plt.show()

def peak_amplitude_analysis(dataloader, 
                            model, 
                            model_path,
                            device: str = 'cpu', 
                            min_snr: int = 1, 
                            max_snr: int = 1e3,
                            save_path: str =''):
    device = torch.device(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()

    peak_amplitudes = {'X Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                      'Y Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                      'Z Channel': {'Clean': [], 'Noisy': [], 'Denoised': []}}
    
    snr_values = {'X Channel': [], 'Y Channel': [], 'Z Channel': []}
    channel_names = ['X Channel', 'Y Channel', 'Z Channel']

    with torch.no_grad():
        for noisy_data, clean_data in dataloader:
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            denoised_output = model(noisy_data)
            
            batch_size = noisy_data.size(0)
            for i in range(batch_size):
                for idx, channel in enumerate(channel_names):  # idx: 0,1,2
                    clean_np = clean_data[i, idx].cpu().numpy()
                    noisy_np = noisy_data[i, idx].cpu().numpy()
                    denoised_np = denoised_output[i, idx].cpu().numpy()
                    
                    if np.std(noisy_np) != 0:
                        snr = _paper_snr_1d(clean_np, noisy_np)
                    else:
                        snr = float('inf')

                    if  max_snr > snr > min_snr:
                        # Calculate envelopes
                        envelope_clean = np.abs(hilbert(clean_np))
                        envelope_noisy = np.abs(hilbert(noisy_np))
                        envelope_denoised = np.abs(hilbert(denoised_np))

                        # Find peak amplitudes
                        peak_amp_clean = np.max(envelope_clean)
                        peak_amp_noisy = np.max(envelope_noisy)
                        peak_amp_denoised = np.max(envelope_denoised)

                        # Store peak amplitudes
                        peak_amplitudes[channel]['Clean'].append(peak_amp_clean)
                        peak_amplitudes[channel]['Noisy'].append(peak_amp_noisy)
                        peak_amplitudes[channel]['Denoised'].append(peak_amp_denoised)

                        snr_values[channel].append(snr)
                    
    # Plot amplitude ratio vs SNR
    plot_amplitude_ratio_vs_snr(peak_amplitudes, snr_values, 'X Channel', save_path)
    plot_amplitude_ratio_vs_snr(peak_amplitudes, snr_values, 'Y Channel', save_path)
    plot_amplitude_ratio_vs_snr(peak_amplitudes, snr_values, 'Z Channel', save_path)

    # Plot peak amplitude distribution
    plot_peak_amplitude_distribution(peak_amplitudes, 'X Channel', save_path)
    plot_peak_amplitude_distribution(peak_amplitudes, 'Y Channel', save_path)
    plot_peak_amplitude_distribution(peak_amplitudes, 'Z Channel', save_path)

def traces_plot(testloader, 
         model, 
         num_images = 12, 
         device="cpu",
         save_path = ''
        ):
    """
    Plot the traces of the clean, noisy and denoised signals for 1 row with 3 channels of images,
    sharing x and y axes, with shared axis labels.
    """
    with torch.no_grad():  
        device = torch.device(device)
        model = model.to(device)
        model.eval() 
        count = 0  # To count the number of images saved
        for noisy_data, clean_data in testloader:
            if count >= num_images: 
                break
            
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            denoised_output = model(noisy_data)
            
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
                    
                    # Calculate SNR
                    if np.std(noisy_np) != 0:
                        snr = _paper_snr_1d(clean_np, noisy_np)
                    else:
                        snr = float('inf')
                    
                    # Calculate peak amplitude
                    envelope_clean = np.abs(hilbert(clean_np))
                    peak_amp = np.max(envelope_clean)
                    
                    snrs.append(snr)
                    peak_amps.append(peak_amp)
                    
                    # Check if this channel meets criteria
                    if snr <= 1 or peak_amp <= 15:
                        valid_sample = False
                        break
                
                # Only plot if all channels meet criteria
                if not valid_sample:
                    continue
                    
                channel_names = ['X Channel', 'Y Channel', 'Z Channel']
                fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True, sharey=True)
                
                for channel_idx in range(3): 
                    ax = axes[channel_idx]
                    fontsize = 32
                    clean_np = clean_data[sample_idx, channel_idx].cpu().numpy()
                    noisy_np = noisy_data[sample_idx, channel_idx].cpu().numpy()
                    denoised_np = denoised_output[sample_idx, channel_idx].cpu().numpy()
                    snr = snrs[channel_idx]

                    psnr_value = psnr(clean_np, denoised_np, np.max(clean_np))
                    time_bin = np.arange(0, clean_np.size)
                    envelope_true = np.abs(hilbert(clean_np))
                    peak_time_true = time_bin[np.argmax(envelope_true)]
                    
                    ax.plot(noisy_np, linestyle='--', color='orange')
                    # ax.plot(clean_np, label='True', color='red')
                    # ax.plot(denoised_np, label='Denoised', color='blue')
                    ax.axhline(y = np.std(noisy_np), color='black', linestyle='--')
                    ax.axhline(y = -np.std(noisy_np), color='black', linestyle='--')
                    ax.set_xlim(peak_time_true - 125, peak_time_true + 200)
                    # ax.text(
                    #         0.02, 0.95,
                    #         f"{channel_names[channel_idx]}",
                    #         transform=ax.transAxes,
                    #         fontsize=fontsize,
                    #         verticalalignment='top',
                    #         horizontalalignment='left',
                    #         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                    #     )
                    if channel_idx == 0:
                            legend = ax.legend(fontsize=fontsize-4)
                            legend.get_frame().set_edgecolor('none')
                    ax.tick_params(axis='both', labelsize=fontsize)

                # Shared axis labels
                fig.text(0.5, 0.04, 'Time [ns]', ha='center', va='center', fontsize=fontsize+2)
                fig.text(0.04, 0.5, 'ADC counts', ha='center', va='center', rotation='vertical', fontsize=fontsize+2)
                plt.tight_layout(rect=[0.06, 0.06, 1, 1])
                if save_path:
                    plt.savefig(os.path.join(save_path, f'sample_{count:03d}_noisy.pdf'))
                    plt.close()
                else:
                    plt.show()
                
                figs, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True, sharey=True)
                
                for channel_idx in range(3): 
                    ax = axes[channel_idx]
                    fontsize = 32
                    clean_np = clean_data[sample_idx, channel_idx].cpu().numpy()
                    noisy_np = noisy_data[sample_idx, channel_idx].cpu().numpy()
                    denoised_np = denoised_output[sample_idx, channel_idx].cpu().numpy()
                    snr = snrs[channel_idx]

                    psnr_value = psnr(clean_np, denoised_np, np.max(clean_np))
                    time_bin = np.arange(0, clean_np.size)
                    envelope_true = np.abs(hilbert(clean_np))
                    peak_time_true = time_bin[np.argmax(envelope_true)]
                    
                    # ax.plot(noisy_np, label='Noisy', linestyle='--', color='orange')
                    ax.plot(clean_np, color='red', alpha = 0.5)
                    ax.plot(denoised_np, color='blue', alpha = 0.8)
                    # ax.axhline(y = np.std(noisy_np), color='black', linestyle='--')
                    # ax.axhline(y = -np.std(noisy_np), color='black', linestyle='--')
                    ax.set_xlim(peak_time_true - 125, peak_time_true + 200)
                    # ax.text(
                    #         0.02, 0.95,
                    #         f"{channel_names[channel_idx]}",
                    #         transform=ax.transAxes,
                    #         fontsize=fontsize,
                    #         verticalalignment='top',
                    #         horizontalalignment='left',
                    #         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                    #     )
                    if channel_idx == 0:
                            legend = ax.legend(fontsize=fontsize-4)
                            legend.get_frame().set_edgecolor('none')
                    ax.tick_params(axis='both', labelsize=fontsize)

                # Shared axis labels
                figs.text(0.5, 0.04, 'Time [ns]', ha='center', va='center', fontsize=fontsize+2)
                figs.text(0.04, 0.5, 'ADC counts', ha='center', va='center', rotation='vertical', fontsize=fontsize+2)
                plt.tight_layout(rect=[0.06, 0.06, 1, 1])
                count += 1  # Increment the count
                if save_path:
                    plt.savefig(os.path.join(save_path, f'sample_{count:03d}_clean_denoised.pdf'))
                    plt.close()
                else:
                    plt.show()
            if count >= num_images:
                break
    print('test is completed')


### All channels plots
def plot_amplitude_ratio_vs_snr_all_channels(peak_amplitudes, snr_values, save_path=None):
    channel_names = ['X Channel', 'Y Channel', 'Z Channel']
    colors = {'X Channel': 'red', 'Y Channel': 'blue', 'Z Channel': 'green'}
    # Channel-specific markers: X=rectangle, Y=triangle, Z=circle
    channel_markers = {'X Channel': 's', 'Y Channel': '^', 'Z Channel': 'o'}
    # Different marker sizes for each channel for visual distinction
    marker_sizes = {'X Channel': 16, 'Y Channel': 10, 'Z Channel': 22}
    
    # Create save directory if it doesn't exist
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(15, 15))
    fontsize = 25
    snr_bins = np.linspace(1, 10, 10)
    bin_centers = (snr_bins[:-1] + snr_bins[1:]) / 2

    # Store handles and labels for custom legend
    handles_noisy = []
    labels_noisy = []
    handles_denoised = []
    labels_denoised = []

    for channel in channel_names:
        snr = np.array(snr_values[channel])
        clean = np.array(peak_amplitudes[channel]['Clean'])
        noisy = np.array(peak_amplitudes[channel]['Noisy'])
        denoised = np.array(peak_amplitudes[channel]['Denoised'])

        ratio_noisy = noisy / clean
        ratio_denoised = denoised / clean

        means_noisy, stds_noisy = [], []
        means_denoised, stds_denoised = [], []

        for i in range(len(snr_bins)-1):
            mask = (snr >= snr_bins[i]) & (snr < snr_bins[i+1])
            means_noisy.append(np.mean(ratio_noisy[mask]) if np.any(mask) else np.nan)
            stds_noisy.append(np.std(ratio_noisy[mask]) if np.any(mask) else np.nan)
            means_denoised.append(np.mean(ratio_denoised[mask]) if np.any(mask) else np.nan)
            stds_denoised.append(np.std(ratio_denoised[mask]) if np.any(mask) else np.nan)

        # Denoised with channel-specific marker shape
        h_den, _, _ = plt.errorbar(bin_centers, means_denoised, yerr=stds_denoised, 
                     fmt=channel_markers[channel], 
                     color=colors[channel], label=f'Denoised - {channel}', 
                     markersize=marker_sizes[channel], linewidth=2, capsize=5,
                     linestyle='--', markerfacecolor='none', markeredgewidth=2)
        handles_denoised.append(h_den)
        labels_denoised.append(f'{channel}')
        
        # Noisy with channel-specific marker shape (use solid line)
        h_noisy, _, _ = plt.errorbar(bin_centers, means_noisy, yerr=stds_noisy, 
                     fmt=channel_markers[channel], 
                     color=colors[channel], label=f'Noisy - {channel}', 
                     markersize=marker_sizes[channel], linewidth=2, capsize=5)
        handles_noisy.append(h_noisy)
        labels_noisy.append(f'{channel}')

    plt.axhline(1, color='gray', linestyle='--')
    plt.xlabel('Signal-to-Noise ratio (SNR)', fontsize=fontsize+8)
    plt.ylabel('Amplitude ratio', fontsize=fontsize+8)
    plt.xticks(fontsize=fontsize+8)
    plt.yticks(fontsize=fontsize+8)
    # plt.gca().set_aspect('equal', adjustable='box')
    # Create custom legend with two columns: Noisy and Denoised
    all_handles = handles_noisy + handles_denoised
    all_labels = [f'Noisy - {label}' for label in labels_noisy] + [f'Denoised - {label}' for label in labels_denoised]
    
    legend = plt.legend(all_handles, all_labels, fontsize=fontsize-2, loc='upper right', 
                        frameon=True, ncol=2, columnspacing=1.2)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('lightgray')
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_alpha(0.9)
    
    if save_path:
        plt.savefig(os.path.join(save_path, f'Amplitude_Ratio_All_Channels.pdf'))
    plt.show()

# def plot_peak_amplitude_distribution_all_channels(peak_amplitudes, save_path=None):
#     channel_names = ['X Channel', 'Y Channel', 'Z Channel']
#     colors = {'X Channel': 'orange', 'Y Channel': 'blue', 'Z Channel': 'green'}
#     all_values = []
#     for channel in channel_names:
#         all_values.extend(peak_amplitudes[channel]['Clean'])
#         all_values.extend(peak_amplitudes[channel]['Denoised'])
       
#     # Create save directory if it doesn't exist
#     if save_path and not os.path.exists(save_path):
#         os.makedirs(save_path, exist_ok=True)
    
#     plt.figure(figsize=(12, 8))
#     fontsize = 16

#     for channel in channel_names:
#         clean = np.array(peak_amplitudes[channel]['Clean'])
#         denoised = np.array(peak_amplitudes[channel]['Denoised'])
#         plt.hist(clean, bins=30, density=True, histtype='step', color=colors[channel], linestyle='--', label=f'Clean - {channel}')
#         plt.hist(denoised, bins=30, density=True, histtype='step', color=colors[channel], label=f'Denoised - {channel}')
#     plt.xlim(0, 1000)
#     plt.yscale('log')
#     plt.xlabel('Maximum peak amplitude [ADC]', fontsize=fontsize+8)
#     plt.ylabel('Density', fontsize=fontsize+8)
#     plt.xticks(fontsize=fontsize+8)
#     plt.yticks(fontsize=fontsize+8)
#     plt.legend(fontsize=fontsize+8)
#     if save_path:
#         plt.savefig(os.path.join(save_path, f'Peak_Amplitude_Distribution_All_Channels.pdf'))
#     plt.show()


def peak_amplitude_analysis_all_channels(dataloader, 
                            model, 
                            model_path,
                            device: str = 'cpu', 
                            min_snr: int = 1, 
                            max_snr: int = 1e3,
                            save_path: str =''):
    device = torch.device(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()

    peak_amplitudes = {'X Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                      'Y Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                      'Z Channel': {'Clean': [], 'Noisy': [], 'Denoised': []}}
    
    snr_values = {'X Channel': [], 'Y Channel': [], 'Z Channel': []}
    channel_names = ['X Channel', 'Y Channel', 'Z Channel']

    with torch.no_grad():
        for noisy_data, clean_data in dataloader:
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            denoised_output = model(noisy_data)
            
            batch_size = noisy_data.size(0)
            for i in range(batch_size):
                for idx, channel in enumerate(channel_names):  # idx: 0,1,2
                    clean_np = clean_data[i, idx].cpu().numpy()
                    noisy_np = noisy_data[i, idx].cpu().numpy()
                    denoised_np = denoised_output[i, idx].cpu().numpy()
                    
                    if np.std(noisy_np) != 0:
                        snr = _paper_snr_1d(clean_np, noisy_np)
                    else:
                        snr = float('inf')

                    if  max_snr > snr > min_snr:
                        # Calculate envelopes
                        envelope_clean = np.abs(hilbert(clean_np))
                        envelope_noisy = np.abs(hilbert(noisy_np))
                        envelope_denoised = np.abs(hilbert(denoised_np))

                        # Find peak amplitudes
                        peak_amp_clean = np.max(envelope_clean)
                        peak_amp_noisy = np.max(envelope_noisy)
                        peak_amp_denoised = np.max(envelope_denoised)

                        # Store peak amplitudes
                        peak_amplitudes[channel]['Clean'].append(peak_amp_clean)
                        peak_amplitudes[channel]['Noisy'].append(peak_amp_noisy)
                        peak_amplitudes[channel]['Denoised'].append(peak_amp_denoised)

                        snr_values[channel].append(snr)

    # Plot amplitude ratio vs SNR for all channels together
    plot_amplitude_ratio_vs_snr_all_channels(peak_amplitudes, snr_values, save_path)
                    


def traces_plot_time_frequency(testloader, 
                   model, 
                   num_images = 50, 
                   device="cpu", 
                   save_path = '',
                   dt_ns = _DT_NS,   # 0.5 ns/sample, the value stated in the paper
                                     # (was 1.0, which mislabelled both the time and
                                     #  the frequency axis of this figure)
                   x_channel_snr = 4.0,
                   y_channel_snr = 3.0,
                   z_channel_snr = 2.0,
                   ):
    """
    Plot the traces of the clean, noisy and denoised signals.
    Layout: 3 rows (X, Y, Z channels) × 2 columns (Time domain, Frequency domain)
    Left column: Time domain signals (noisy and denoised)
    Right column: Frequency domain signals (noisy and denoised)
    
    Args:
        testloader: DataLoader with test data
        model: Trained model
        num_images: Number of images to generate
        device: Device to run inference on
        save_path: Path to save plots
        dt_ns: Time step in nanoseconds (default: 1.0 ns per sample)
    """
    with torch.no_grad():  
        device = torch.device(device)
        model = model.to(device)
        model.eval() 
        count = 0  # To count the number of images saved
        
        # Search for a suitable sample
        for noisy_data, clean_data in testloader:
            if count >= num_images:
                break
            
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            denoised_output = model(noisy_data)

            batch_size = noisy_data.size(0)
            for sample_idx in range(batch_size):
                if count >= num_images:
                    break
                
                # Check SNR and peak amplitude for all channels
                snrs = []
                peak_amps = []
                valid_sample = True
                
                for channel_idx in range(3):
                    clean_np = clean_data[sample_idx, channel_idx].cpu().numpy()
                    noisy_np = noisy_data[sample_idx, channel_idx].cpu().numpy()
                    
                    # Calculate SNR
                    if np.std(noisy_np) != 0:
                        snr = _paper_snr_1d(clean_np, noisy_np)
                    else:
                        snr = float('inf')
                    
                    # Calculate peak amplitude
                    envelope_clean = np.abs(hilbert(clean_np))
                    peak_amp = np.max(envelope_clean)
                    
                    snrs.append(snr)
                    peak_amps.append(peak_amp)
                    
                    # Check basic criteria - all channels must have SNR > 1 and peak > 15
                    if snr <= 1 or peak_amp <= 15:
                        valid_sample = False
                        break
                
                if not valid_sample:
                    continue
                # Check if SNRs match target values (X≈2, Y≈3, Z≈4) with tolerance
                target_snrs = [x_channel_snr, y_channel_snr, z_channel_snr]
                tolerance = 1  # Allow ±1 tolerance
                snr_match = True
                for channel_idx in range(3):
                    target_snr = target_snrs[channel_idx]
                    if not (target_snr - tolerance <= snrs[channel_idx] <= target_snr + tolerance):
                        snr_match = False
                        break

                if not snr_match:
                    continue
                # Found a valid sample, create the plot
                channel_names = ['X Channel', 'Y Channel', 'Z Channel']
                fig, axes = plt.subplots(3, 2, figsize=(16, 12))
                fontsize = 20
                
                # Calculate sampling parameters
                signal_length = clean_data[sample_idx, 0].cpu().numpy().size
                dt_s = dt_ns * 1e-9  # seconds per sample
                
                for channel_idx in range(3):
                    # Extract data for this channel
                    clean_np = clean_data[sample_idx, channel_idx].cpu().numpy()
                    noisy_np = noisy_data[sample_idx, channel_idx].cpu().numpy()
                    denoised_np = denoised_output[sample_idx, channel_idx].cpu().numpy()
                    snr = snrs[channel_idx]
                    peak_amp = peak_amps[channel_idx]
                    
                    # Calculate PSNR and find peak time for centering
                    psnr_value = psnr(clean_np, denoised_np, np.max(clean_np))
                    time_bin = np.arange(0, clean_np.size) * dt_ns
                    envelope_true = np.abs(hilbert(clean_np))
                    peak_time_true = time_bin[np.argmax(envelope_true)]
                    
                    # ========== LEFT COLUMN: TIME DOMAIN ==========
                    ax_time = axes[channel_idx, 0]
                    
                    # Plot the traces
                    ax_time.plot(time_bin, noisy_np, label='Noisy', linestyle='--', color='red', alpha=0.9, linewidth=1.5)
                    ax_time.plot(time_bin, clean_np, label='True', color='black', linestyle=':', alpha=0.7, linewidth=1.5)
                    ax_time.plot(time_bin, denoised_np, label='Denoised', color='blue', linestyle='-', alpha=0.5, linewidth=1.5)
                    
                    # Set zoom around peak
                    ax_time.set_xlim(peak_time_true - 50, peak_time_true + 100)
                    ax_time.set_xlabel('Times [ns]', fontsize=fontsize)
                    ax_time.set_ylabel('ADC Counts', fontsize=fontsize)
                    ax_time.set_title(f'{channel_names[channel_idx]}', fontsize=fontsize)
                    ax_time.tick_params(axis='both', labelsize=fontsize-2)
                    ax_time.grid(True, alpha=0.3)
                    
                    # Add legend only to top-left subplot
                    if channel_idx == 0:
                        legend = ax_time.legend(fontsize=fontsize-2, loc='lower right')
                        legend.get_frame().set_edgecolor('none')
                    
                    # Add SNR and PSNR text to time domain plot
                    ax_time.text(
                        0.98, 0.95,
                        f"SNR = {snr:.2f}\nPSNR = {psnr_value:.2f}",
                        transform=ax_time.transAxes,
                        fontsize=fontsize-2,
                        verticalalignment='top',
                        horizontalalignment='right',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                    )
                    
                    # ========== RIGHT COLUMN: FREQUENCY DOMAIN ==========
                    ax_freq = axes[channel_idx, 1]
                    
                    # Compute FFT
                    freqs_hz = np.fft.rfftfreq(signal_length, d=dt_s)
                    freqs_mhz = freqs_hz / 1e6  # Convert to MHz
                    
                    # Compute FFT magnitude
                    fft_noisy = np.fft.rfft(noisy_np)
                    fft_clean = np.fft.rfft(clean_np)
                    fft_denoised = np.fft.rfft(denoised_np)
                    
                    # Magnitude spectrum (normalized by signal length)
                    mag_noisy = np.abs(fft_noisy) / signal_length
                    mag_clean = np.abs(fft_clean) / signal_length
                    mag_denoised = np.abs(fft_denoised) / signal_length
                    
                    # Plot frequency domain
                    ax_freq.semilogy(freqs_mhz, mag_noisy, label='Noisy', linestyle='--', color='red', alpha=0.9, linewidth=1.5)
                    ax_freq.semilogy(freqs_mhz, mag_clean, label='True', color='black', linestyle=':', alpha=0.7, linewidth=1.5)
                    ax_freq.semilogy(freqs_mhz, mag_denoised, label='Denoised', color='blue', linestyle='-', alpha=0.5, linewidth=1.5)
                    
                    # Set frequency range (adjust based on your needs, e.g., 50-350 MHz from image)
                    ax_freq.set_xlim(50, 350)
                    ax_freq.set_xlabel('Freqs [MHz]', fontsize=fontsize)
                    ax_freq.set_ylabel('ADC Counts/MHz', fontsize=fontsize)
                    ax_freq.set_title(f'{channel_names[channel_idx]}', fontsize=fontsize)
                    ax_freq.tick_params(axis='both', labelsize=fontsize-2)
                    ax_freq.grid(True, alpha=0.3, which='both')
                    
                    # # Add legend only to top-right subplot
                    # if channel_idx == 0:
                    #     legend = ax_freq.legend(fontsize=fontsize-2, loc='upper right')
                    #     legend.get_frame().set_edgecolor('none')
                
                plt.tight_layout()
                
                if save_path:
                    os.makedirs(save_path, exist_ok=True)
                    plt.savefig(os.path.join(save_path, f'traces_time_freq_{count:03d}.pdf'), dpi=300, bbox_inches='tight')
                    plt.show()
                else:
                    plt.show()
                count += 1
                break  # Process one sample per batch
            
            if count >= num_images:
                break
            
    print(f'traces_plot_time_frequency completed. Generated {count} plots.')

### Peak time efficiency plots
def _robust_sigma_mad(x: np.ndarray, eps: float = 1e-12) -> float:
    """
    Robust scale estimate using MAD, converted to an equivalent Gaussian sigma.

    sigma ≈ 1.4826 * median(|x - median(x)|)
    (1.4826 makes MAD consistent for N(0, sigma^2)).  :contentReference[oaicite:3]{index=3}
    """
    x = np.asarray(x)
    if x.size == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad + eps)


def _estimate_noise_sigma(
    noisy_trace: np.ndarray,
    peak_index: int,
    exclude_radius: int = 32,
    method: str = "mad",
) -> float:
    """
    Estimate per-trace noise scale σ from the noisy TRACE itself.
    We exclude a window around the detected peak to reduce signal leakage.

    method:
      - "mad": robust MAD-based σ
      - "std":  standard deviation (less robust to pulse leakage/outliers)
    """
    n = noisy_trace.size
    if n == 0:
        return np.nan

    # Exclude a small window around the peak to avoid contaminating the noise estimate.
    mask = np.ones(n, dtype=bool)
    lo = max(0, peak_index - exclude_radius)
    hi = min(n, peak_index + exclude_radius + 1)
    mask[lo:hi] = False

    # If exclusion wipes out too much data, fall back to full trace.
    noise_samples = noisy_trace[mask]
    if noise_samples.size < max(16, int(0.1 * n)):
        noise_samples = noisy_trace

    if method.lower() == "std":
        sig = float(np.std(noise_samples))
    elif method.lower() == "mad":
        sig = _robust_sigma_mad(noise_samples)
    else:
        raise ValueError(f"Unknown method='{method}'. Use 'mad' or 'std'.")

    # Guard against pathological zeros
    if not np.isfinite(sig) or sig <= 0:
        sig = float(np.std(noisy_trace))
        if not np.isfinite(sig) or sig <= 0:
            sig = np.nan
    return sig



def plot_peak_time_efficiency_combined(
    snr_values: Dict[str, List[float]],
    peak_times: Dict[str, Dict[str, List[float]]],
    peak_amplitudes: Dict[str, Dict[str, List[float]]],
    noise_sigmas: Dict[str, List[float]],
    timing_thresholds_list: List[int],
    save_path: Optional[str] = None,
    trigger_k: float = 1.0,
    dt_ns: float = _DT_NS,
) -> None:
    """
    Plots the peak time efficiency for all 3 channels in one row.

    Key change vs your previous version (:contentReference[oaicite:4]{index=4}):
      - Trigger-like selection: require noisy_amp >= trigger_k * sigma (per-trace)
      - No fixed adc_threshold.

    Args:
        snr_values: dict keyed by channel name
        peak_times: dict[channel]['Clean'/'Noisy'/'Denoised'] arrays
        peak_amplitudes: dict[channel]['Clean'/'Noisy'/'Denoised'] envelope peak amplitudes
        noise_sigmas: dict[channel] per-trace noise σ in the SAME space as noisy_amp
        timing_thresholds_list: list of thresholds for |Δt_peak|, in SAMPLES
            (peak times are sample indices, so the selection is a sample count)
        save_path: path to save the plot
        trigger_k: multiplier for the dynamic threshold; default k=1.0
        dt_ns: sampling interval, used ONLY to label the sample-count thresholds
            in nanoseconds. The selection itself is unchanged: a 10-sample
            tolerance is displayed as 5 ns at dt = 0.5 ns/sample.
    """
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fontsize = 16
    channel_names = ["X Channel", "Y Channel", "Z Channel"]

    snr_bins = np.linspace(1, 20, 20)  # same as before
    timing_thresholds = timing_thresholds_list

    for ax_idx, channel_name in enumerate(channel_names):
        ax = axes[ax_idx]

        snr_vals = np.array(snr_values[channel_name], dtype=float)
        clean_time = np.array(peak_times[channel_name]["Clean"], dtype=float)
        noisy_time = np.array(peak_times[channel_name]["Noisy"], dtype=float)
        denoised_time = np.array(peak_times[channel_name]["Denoised"], dtype=float)

        # Envelope peak amplitudes
        noisy_amp = np.array(peak_amplitudes[channel_name]["Noisy"], dtype=float)
        denoised_amp = np.array(peak_amplitudes[channel_name]["Denoised"], dtype=float)  # kept for future use
        _ = denoised_amp  # silence linter / placeholder

        sigmas = np.array(noise_sigmas[channel_name], dtype=float)

        # Trigger-like: per-trace pass/fail based on observed (noisy) envelope peak
        pass_trigger = noisy_amp >= (trigger_k * sigmas)

        # --- Efficiency curves (black) and fractions > threshold (orange/red) ---
        denoising_efficiency = []
        noisy_masks = []

        for i in range(len(snr_bins) - 1):
            in_bin = (snr_vals >= snr_bins[i]) & (snr_vals < snr_bins[i + 1])
            mask = in_bin & pass_trigger

            noisy_masks.append(mask)

            if np.sum(mask) == 0:
                denoising_efficiency.append(np.nan)
            else:
                denoising_efficiency.append(
                    np.mean(np.abs(denoised_time[mask] - clean_time[mask]) <= timing_thresholds[0])
                )

        denoising_efficiency = np.array(denoising_efficiency, dtype=float)

        # “Continuous histogram”: step plot (piecewise-constant per SNR bin) :contentReference[oaicite:5]{index=5}
        ax.step(
            snr_bins[:-1],
            denoising_efficiency,
            where="post",
            color="black",
            linewidth=3,
            label=fr"Denoised: $|\Delta t_{{peak}}| \leq {_samples_to_ns_label(timing_thresholds[0], dt_ns)}$ ns",
        )

        # For each threshold: show fraction exceeding threshold for denoised (orange) and noisy (red dashed)
        for idx, threshold in enumerate(timing_thresholds):
            frac_denoised = []
            frac_noisy = []

            for i in range(len(snr_bins) - 1):
                mask = noisy_masks[i]  # exact same denominator for both denoised and noisy fractions
                if np.sum(mask) == 0:
                    frac_denoised.append(np.nan)
                    frac_noisy.append(np.nan)
                    continue

                frac_denoised.append(np.mean(np.abs(denoised_time[mask] - clean_time[mask]) > threshold))
                frac_noisy.append(np.mean(np.abs(noisy_time[mask] - clean_time[mask]) > threshold))

            ax.step(
                snr_bins[:-1],
                frac_denoised,
                where="post",
                color="orange",
                linestyle="-",
                linewidth=2.5,
                label=fr"Denoised: $|\Delta t_{{peak}}| > {_samples_to_ns_label(timing_thresholds[idx], dt_ns)}$ ns",
            )
            ax.step(
                snr_bins[:-1],
                frac_noisy,
                where="post",
                color="red",
                linestyle="--",
                linewidth=2,
                label=fr"Noisy: $|\Delta t_{{peak}}| > {_samples_to_ns_label(timing_thresholds[idx], dt_ns)}$ ns",
            )

        ax.axhline(y=0.95, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, label="95% threshold")
        ax.axhline(y=0.05, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, label="5% threshold")

        ax.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=fontsize)
        if ax_idx == 0:
            ax.set_ylabel("Fraction", fontsize=fontsize)
        ax.set_title(f"{channel_name}", fontsize=fontsize)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        ax.set_xlim(1, 10)
        ax.set_ylim(-0.05, 1.05)
        if ax_idx == 0:
            ax.legend(fontsize=fontsize - 4, loc="center right", frameon=False)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(
            os.path.join(save_path, "Peak_Time_Efficiency_All_Channels.pdf"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()


def peak_time_analysis_for_all_channels(
    dataloader,
    model,
    model_path,
    device: str = "cpu",
    min_snr: int = 1,
    max_snr: float = 1e3,
    thresholds_list: List[int] = (5, 10),
    save_path: str = "",
    trigger_k: float = 2.0,
    sigma_method: str = "mad",
    exclude_radius: int = 32,
    sigma_source: str = "envelope",
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Computes peak times, envelope peak amplitudes, per-trace noise σ, and then plots efficiency curves.

    trigger selection uses:
        noisy_amp >= trigger_k * σ

    sigma_source:
        - "envelope" (default): σ estimated from the noisy envelope (same domain as noisy_amp)
        - "waveform": σ estimated from the raw noisy waveform (classic RMS notion)

    Notes:
      - envelope uses abs(hilbert(x)) (analytic signal magnitude). :contentReference[oaicite:6]{index=6}
    """
    device = torch.device(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()

    peak_times = {
        "X Channel": {"Clean": [], "Noisy": [], "Denoised": []},
        "Y Channel": {"Clean": [], "Noisy": [], "Denoised": []},
        "Z Channel": {"Clean": [], "Noisy": [], "Denoised": []},
    }

    peak_amplitudes = {
        "X Channel": {"Clean": [], "Noisy": [], "Denoised": []},
        "Y Channel": {"Clean": [], "Noisy": [], "Denoised": []},
        "Z Channel": {"Clean": [], "Noisy": [], "Denoised": []},
    }

    # NEW: store per-trace σ used for the trigger-like selection
    noise_sigmas = {"X Channel": [], "Y Channel": [], "Z Channel": []}

    snr_values = {"X Channel": [], "Y Channel": [], "Z Channel": []}
    channel_names = ["X Channel", "Y Channel", "Z Channel"]

    with torch.no_grad():
        for noisy_data, clean_data in dataloader:
            noisy_data = noisy_data.to(device)
            clean_data = clean_data.to(device)
            denoised_output = model(noisy_data)

            batch_size = noisy_data.size(0)
            for i in range(batch_size):
                for idx, channel in enumerate(channel_names):
                    clean_np = clean_data[i, idx].cpu().numpy()
                    noisy_np = noisy_data[i, idx].cpu().numpy()
                    denoised_np = denoised_output[i, idx].cpu().numpy()

                    timing = np.arange(clean_np.size)

                    # Paper SNR: max(clean) / std(noisy) over the full trace.
                    snr = _paper_snr_1d(clean_np, noisy_np)

                    if max_snr > snr > min_snr:
                        envelope_clean = np.abs(hilbert(clean_np))
                        envelope_noisy = np.abs(hilbert(noisy_np))
                        envelope_denoised = np.abs(hilbert(denoised_np))

                        # Peak indices/times from envelopes
                        idx_peak_clean = int(np.argmax(envelope_clean))
                        idx_peak_noisy = int(np.argmax(envelope_noisy))
                        idx_peak_denoised = int(np.argmax(envelope_denoised))

                        peak_time_clean = float(timing[idx_peak_clean])
                        peak_time_noisy = float(timing[idx_peak_noisy])
                        peak_time_denoised = float(timing[idx_peak_denoised])

                        # Envelope peak amplitudes
                        peak_amp_clean = float(np.max(envelope_clean))
                        peak_amp_noisy = float(np.max(envelope_noisy))
                        peak_amp_denoised = float(np.max(envelope_denoised))

                        # Per-trace σ estimate for trigger-like selection
                        if sigma_source.lower() == "envelope":
                            # σ estimated in the same space as peak_amp_noisy
                            sigma = _estimate_noise_sigma(
                                noisy_trace=envelope_noisy,
                                peak_index=idx_peak_noisy,
                                exclude_radius=exclude_radius,
                                method=sigma_method,
                            )
                        elif sigma_source.lower() == "waveform":
                            # Alternative (classic): σ from raw waveform (1-line swap)
                            sigma = _estimate_noise_sigma(
                                noisy_trace=noisy_np,
                                peak_index=idx_peak_noisy,
                                exclude_radius=exclude_radius,
                                method=sigma_method,
                            )
                        else:
                            raise ValueError("sigma_source must be 'envelope' or 'waveform'.")

                        # Store
                        peak_times[channel]["Clean"].append(peak_time_clean)
                        peak_times[channel]["Noisy"].append(peak_time_noisy)
                        peak_times[channel]["Denoised"].append(peak_time_denoised)

                        peak_amplitudes[channel]["Clean"].append(peak_amp_clean)
                        peak_amplitudes[channel]["Noisy"].append(peak_amp_noisy)
                        peak_amplitudes[channel]["Denoised"].append(peak_amp_denoised)

                        noise_sigmas[channel].append(float(sigma))
                        snr_values[channel].append(float(snr))

    plot_peak_time_efficiency_combined(
        snr_values=snr_values,
        peak_times=peak_times,
        peak_amplitudes=peak_amplitudes,
        noise_sigmas=noise_sigmas,
        timing_thresholds_list=list(thresholds_list),
        save_path=save_path,
        trigger_k=trigger_k,
    )

    return snr_values, peak_times, peak_amplitudes, noise_sigmas

def plot_peak_time_efficiency_for_hilbert_filter(dataloader, 
                                                device: str = 'cpu', 
                                                min_snr: int = 1, 
                                                max_snr: int = 1e3,
                                                thresholds_list: List[int] = [5, 10],
                                                save_path: str ='' ):
    """
    Plot peak time efficiency based on the hilber filter.
    """
    peak_times = {'X Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                  'Y Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                  'Z Channel': {'Clean': [], 'Noisy': [], 'Denoised': []}}
    
    peak_amplitudes = {'X Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                       'Y Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                       'Z Channel': {'Clean': [], 'Noisy': [], 'Denoised': []}}
    
    noise_sigmas = {'X Channel': [], 'Y Channel': [], 'Z Channel': []}
    
    snr_values = {'X Channel': [], 'Y Channel': [], 'Z Channel': []}
    channel_names = ['X Channel', 'Y Channel', 'Z Channel']

    for noisy_data, clean_data in dataloader:
        noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
        ### add the hilber filter to the noisy data 
        noisy_data_np = noisy_data.cpu().numpy() 
        noisy_data_hilbert = np.abs(hilbert(noisy_data_np))
        
        batch_size = noisy_data.size(0)
        for i in range(batch_size):
            for idx, channel in enumerate(channel_names):  # idx: 0,1,2
                clean_np = clean_data[i, idx].cpu().numpy()
                noisy_np = noisy_data[i, idx].cpu().numpy()
                noisy_np_hilbert = noisy_data_hilbert[i, idx]
                timing = np.arange(clean_np.size)
                
                if np.std(noisy_np) != 0:
                    snr = _paper_snr_1d(clean_np, noisy_np)
                else:
                    snr = float('inf')
                    
                if  max_snr > snr > min_snr:
                    # Calculate envelopes
                    envelope_clean = np.abs(hilbert(clean_np))
                    envelope_noisy = np.abs(hilbert(noisy_np))
                    envelope_hilbert = noisy_np_hilbert  # This is the "denoised" output via Hilbert filter
                    
                    # Find peak indices
                    idx_peak_clean = int(np.argmax(envelope_clean))
                    idx_peak_noisy = int(np.argmax(envelope_noisy))
                    idx_peak_hilbert = int(np.argmax(envelope_hilbert))
                    
                    # Find peak times
                    peak_time_clean = float(timing[idx_peak_clean])
                    peak_time_noisy = float(timing[idx_peak_noisy])
                    peak_time_hilbert = float(timing[idx_peak_hilbert])

                    # Calculate peak amplitudes
                    peak_amp_clean = float(np.max(envelope_clean))
                    peak_amp_noisy = float(np.max(envelope_noisy))
                    peak_amp_hilbert = float(np.max(envelope_hilbert))
                    
                    # Estimate noise sigma from the noisy envelope
                    sigma = _estimate_noise_sigma(
                        noisy_trace=envelope_noisy,
                        peak_index=idx_peak_noisy,
                        exclude_radius=32,
                        method="mad",
                    )

                    # Store peak times
                    peak_times[channel]['Clean'].append(peak_time_clean)
                    peak_times[channel]['Noisy'].append(peak_time_noisy)
                    peak_times[channel]['Denoised'].append(peak_time_hilbert)
                    
                    # Store peak amplitudes
                    peak_amplitudes[channel]['Clean'].append(peak_amp_clean)
                    peak_amplitudes[channel]['Noisy'].append(peak_amp_noisy)
                    peak_amplitudes[channel]['Denoised'].append(peak_amp_hilbert)
                    
                    # Store noise sigma
                    noise_sigmas[channel].append(float(sigma))

                    snr_values[channel].append(snr)

    # Plot peak time efficiency for all channels
    plot_peak_time_efficiency_combined(
        snr_values=snr_values,
        peak_times=peak_times,
        peak_amplitudes=peak_amplitudes,
        noise_sigmas=noise_sigmas,
        timing_thresholds_list=thresholds_list,
        save_path=save_path,
    )

# ========== MODEL COMPARISON FUNCTIONS ==========

def load_model(model_path, model_config, device='cpu'):
    """
    Load a DualBranchAutoencoder model.
    
    Args:
        model_path: Path to the model weights
        model_config: Dictionary containing model configuration
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    device = torch.device(device)
    
    # Initialize and load model
    model = DualBranchAutoencoder(model_config)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device).eval()
    
    return model


def load_models_for_comparison(model_paths, model_configs, device='cpu'):
    """
    Load multiple models for comparison.
    
    Args:
        model_paths: Dict mapping model names to their weight paths
        model_configs: Dict mapping model names to their configurations
        device: Device to load models on
        
    Returns:
        Dictionary containing loaded models
    """
    device = torch.device(device)
    models = {}
    
    for name, path in model_paths.items():
        config = model_configs.get(name, model_configs.get('CNN', model_configs))
        model = DualBranchAutoencoder(config)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.to(device).eval()
        models[name] = model
    
    return models



def plot_peak_time_efficiency_comparison(snr_values_models, peak_times_models, channel, save_path=None):
    """
    Compare peak time efficiency between multiple models for a specific channel.
    
    Args:
        snr_values_models: Dict with model names as keys, each containing channel SNR values
        peak_times_models: Dict with model names as keys, each containing peak times data
        channel: Channel name ('X Channel', 'Y Channel', or 'Z Channel')
        save_path: Path to save the plot
    """
    # Create save directory if it doesn't exist
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    fontsize = 16
    
    colors = {'DualBranchCNN': 'blue', 'SingleBranchCNN': 'red'}
    linestyles = {'DualBranchCNN': '-', 'SingleBranchCNN': '--'}
    
    snr_bins = np.linspace(1, 20, 20)
    thresholds = [10, 20]  # SAMPLES (peak times are sample indices; previously mislabeled "ns"; convert with dt once recovered)
    threshold_colors = ['orange', 'green']
    
    for model_name in ['DualBranchCNN', 'SingleBranchCNN']:
        snr_vals = np.array(snr_values_models[model_name][channel])
        clean_time = np.array(peak_times_models[model_name][channel]['Clean'])
        noisy_time = np.array(peak_times_models[model_name][channel]['Noisy'])  
        denoised_time = np.array(peak_times_models[model_name][channel]['Denoised'])
        
        # Denoising efficiency: fraction of denoised traces with |Δt| <= 10 samples
        denoising_efficiency = []
        for i in range(len(snr_bins)-1):
            mask = (snr_vals >= snr_bins[i]) & (snr_vals < snr_bins[i+1])
            if np.sum(mask) == 0:
                denoising_efficiency.append(np.nan)
                continue
            denoising_efficiency.append(np.mean(np.abs(denoised_time[mask] - clean_time[mask]) <= 10))
        
        plt.step(snr_bins[:-1], denoising_efficiency, where='post', 
                color=colors[model_name], linewidth=3, 
                linestyle=linestyles[model_name],
                label=f'{model_name} - Denoising efficiency')

        # Plot thresholds
        for idx, threshold in enumerate(thresholds):
            # Denoised
            frac_denoised = []
            for i in range(len(snr_bins)-1):
                mask = (snr_vals >= snr_bins[i]) & (snr_vals < snr_bins[i+1])
                if np.sum(mask) == 0:
                    frac_denoised.append(np.nan)
                    continue
                frac_denoised.append(np.mean(np.abs(denoised_time[mask] - clean_time[mask]) > threshold))
            
            plt.step(snr_bins[:-1], frac_denoised, where='post', 
                    color=threshold_colors[idx], linestyle=linestyles[model_name], 
                    alpha=0.7, linewidth=2,
                    label=fr'{model_name} - $\Delta t_{{peak}} > {threshold}$ samples')

    plt.xlabel('Signal-to-Noise ratio (SNR)', fontsize=fontsize)
    plt.ylabel('Fraction', fontsize=fontsize)
    plt.title(f'Peak Time Efficiency Comparison - {channel}', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(1, 10)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=fontsize-2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, f'Peak_Time_Efficiency_Comparison_{channel}.pdf'), 
                   dpi=300, bbox_inches='tight')
    plt.show()

def plot_amplitude_ratio_comparison(peak_amplitudes_models, snr_values_models, channel, save_path=None):
    """
    Compare amplitude ratios between multiple models for a specific channel.
    """
    # Create save directory if it doesn't exist
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    fontsize = 16
    
    colors = {'DualBranchCNN': 'blue', 'SingleBranchCNN': 'red'}
    markers = {'DualBranchCNN': 'o', 'SingleBranchCNN': '^'}
    
    snr_bins = np.linspace(1, 10, 10)
    bin_centers = (snr_bins[:-1] + snr_bins[1:]) / 2

    for model_name in ['DualBranchCNN', 'SingleBranchCNN']:
        snr = np.array(snr_values_models[model_name][channel])
        clean = np.array(peak_amplitudes_models[model_name][channel]['Clean'])
        denoised = np.array(peak_amplitudes_models[model_name][channel]['Denoised'])

        ratio_denoised = denoised / clean
        means_denoised, stds_denoised = [], []

        for i in range(len(snr_bins)-1):
            mask = (snr >= snr_bins[i]) & (snr < snr_bins[i+1])
            means_denoised.append(np.mean(ratio_denoised[mask]) if np.any(mask) else np.nan)
            stds_denoised.append(np.std(ratio_denoised[mask]) if np.any(mask) else np.nan)

        plt.errorbar(bin_centers, means_denoised, yerr=stds_denoised, 
                    fmt=markers[model_name], color=colors[model_name], 
                    linewidth=2, markersize=8, capsize=5,
                    label=f'{model_name} - Denoised')

    # Add noisy baseline (same for both models)
    snr = np.array(snr_values_models['DualBranchCNN'][channel])  # Use CNN data as reference
    clean = np.array(peak_amplitudes_models['DualBranchCNN'][channel]['Clean'])
    noisy = np.array(peak_amplitudes_models['DualBranchCNN'][channel]['Noisy'])
    ratio_noisy = noisy / clean
    means_noisy, stds_noisy = [], []
    
    for i in range(len(snr_bins)-1):
        mask = (snr >= snr_bins[i]) & (snr < snr_bins[i+1])
        means_noisy.append(np.mean(ratio_noisy[mask]) if np.any(mask) else np.nan)
        stds_noisy.append(np.std(ratio_noisy[mask]) if np.any(mask) else np.nan)
    
    plt.errorbar(bin_centers, means_noisy, yerr=stds_noisy, fmt='s', 
                color='orange', alpha=0.7, linewidth=2, markersize=6, capsize=5,
                label='Noisy (reference)')

    plt.axhline(1, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel('Signal-to-Noise ratio (SNR)', fontsize=fontsize)
    plt.ylabel('Amplitude ratio', fontsize=fontsize)
    plt.title(f'Amplitude Ratio Comparison - {channel}', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, f'Amplitude_Ratio_Comparison_{channel}.pdf'), 
                   dpi=300, bbox_inches='tight')
    plt.show()

def model_comparison_analysis(dataloader, 
                            models,
                            device: str = 'cpu', 
                            min_snr: int = 1, 
                            max_snr: int = 1e3,
                            save_path: str = ''):
    """
    Comprehensive comparison analysis between multiple models.
    
    Args:
        dataloader: DataLoader for test data
        models: Dictionary containing loaded models, e.g. {'model1': model, 'model2': model}
        device: Device to run analysis on
        min_snr, max_snr: SNR range to include in analysis  
        save_path: Path to save plots
    """
    device = torch.device(device)
    
    # Create save directory if it doesn't exist
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        print(f"Created directory: {save_path}")
    
    # Data structures for each model
    peak_times_models = {}
    peak_amplitudes_models = {}
    snr_values_models = {}
    psnr_values_models = {}
    
    channel_names = ['X Channel', 'Y Channel', 'Z Channel']
    
    # Initialize data structures
    for model_name in models.keys():
        peak_times_models[model_name] = {
            channel: {'Clean': [], 'Noisy': [], 'Denoised': []} 
            for channel in channel_names
        }
        peak_amplitudes_models[model_name] = {
            channel: {'Clean': [], 'Noisy': [], 'Denoised': []} 
            for channel in channel_names
        }
        snr_values_models[model_name] = {channel: [] for channel in channel_names}
        psnr_values_models[model_name] = {channel: [] for channel in channel_names}

    with torch.no_grad():
        for noisy_data, clean_data in dataloader:
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            
            # Get outputs from both models
            model_outputs = {}
            for model_name, model in models.items():
                model_outputs[model_name] = model(noisy_data)
            
            batch_size = noisy_data.size(0)
            for i in range(batch_size):
                for idx, channel in enumerate(channel_names):
                    clean_np = clean_data[i, idx].cpu().numpy()
                    noisy_np = noisy_data[i, idx].cpu().numpy()
                    
                    if np.std(noisy_np) != 0:
                        snr = _paper_snr_1d(clean_np, noisy_np)
                    else:
                        snr = float('inf')
                    
                    if max_snr > snr > min_snr:
                        timing = np.arange(clean_np.size)
                        
                        # Process for each model
                        for model_name in models.keys():
                            denoised_np = model_outputs[model_name][i, idx].cpu().numpy()
                            
                            # Calculate envelopes
                            envelope_clean = np.abs(hilbert(clean_np))
                            envelope_noisy = np.abs(hilbert(noisy_np))
                            envelope_denoised = np.abs(hilbert(denoised_np))

                            # Peak times
                            peak_time_clean = timing[np.argmax(envelope_clean)]
                            peak_time_noisy = timing[np.argmax(envelope_noisy)]
                            peak_time_denoised = timing[np.argmax(envelope_denoised)]

                            # Peak amplitudes
                            peak_amp_clean = np.max(envelope_clean)
                            peak_amp_noisy = np.max(envelope_noisy)
                            peak_amp_denoised = np.max(envelope_denoised)

                            # Store data
                            peak_times_models[model_name][channel]['Clean'].append(peak_time_clean)
                            peak_times_models[model_name][channel]['Noisy'].append(peak_time_noisy)
                            peak_times_models[model_name][channel]['Denoised'].append(peak_time_denoised)

                            peak_amplitudes_models[model_name][channel]['Clean'].append(peak_amp_clean)
                            peak_amplitudes_models[model_name][channel]['Noisy'].append(peak_amp_noisy)
                            peak_amplitudes_models[model_name][channel]['Denoised'].append(peak_amp_denoised)

                            snr_values_models[model_name][channel].append(snr)
                            
                            # Calculate PSNR
                            psnr_val = psnr(clean_np, denoised_np, np.max(clean_np))
                            psnr_values_models[model_name][channel].append(psnr_val)

    # Generate comparison plots
    # for channel in channel_names:
    #     plot_peak_time_efficiency_comparison(snr_values_models, peak_times_models, channel, save_path)
    #     plot_amplitude_ratio_comparison(peak_amplitudes_models, snr_values_models, channel, save_path)
    
    # # Combined plots for all channels
    # plot_peak_time_efficiency_combined_comparison(snr_values_models, peak_times_models, save_path)
    # plot_amplitude_ratio_all_channels_comparison(peak_amplitudes_models, snr_values_models, save_path)
    
    return peak_times_models, peak_amplitudes_models, snr_values_models, psnr_values_models

def load_models_v2(model_paths: dict, 
                   model_classes: dict,
                   device: str = 'cpu'):
    """
    Load multiple CNN models for comparison using JSON config files.
    
    Args:
        model_paths: Dictionary mapping model names to tuples of (metrics_json_path, config_json_path, model_path)
                    Example: {
                        'CNN_multiloss': (metrics_path, config_path, model_path),
                        'CNN_mse': (metrics_path, config_path, model_path),
                        'CNN_l1': (metrics_path, config_path, model_path),
                        'CNN_psnr': (metrics_path, config_path, model_path)
                    }
        model_classes: Dict mapping model_type -> class, e.g. {"CNN": CNNModel, "SingleBranchCNN": SingleBranchCNN}
        device: Device to load models on ('cpu' or 'cuda:0')
        
    Returns:
        Dictionary containing loaded models with model names as keys
    """
    device = torch.device(device)
    loaded_models = {}
    
    for model_name, (metrics_json_path, config_json_path, model_path) in model_paths.items():
        # Load config from JSON
        with open(config_json_path, "r") as f:
            config = json.load(f)
        
        model_type = config.get("model_type")
        model_config = config.get("model_config")
        
        if model_type not in model_classes:
            raise ValueError(f"Unknown model_type '{model_type}' for {model_name}. Available: {list(model_classes.keys())}")
        if model_config is None:
            raise ValueError(f"model_config missing in config file for {model_name}: {config_json_path}")
        
        # Build model
        ModelClass = model_classes[model_type]
        model = ModelClass(model_config)
        
        # Load weights (handle PyTorch versions without weights_only)
        try:
            state = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location=device)
        
        model.load_state_dict(state)
        model.to(device).eval()
        
        loaded_models[model_name] = model
    
    return loaded_models

### Usable Antenna VS SNR ###
def traces_plot_comparison(testloader, 
                          models,
                          num_images=6, 
                          device="cpu", 
                          save_path='',
                          snr_min=1,
                          snr_max=4,
                          peak_amp_threshold=15
                          ):
    """
    Plot traces comparison between CNN models with different loss functions.
    Upper row: 3 channels (X, Y, Z) for CNN model
    Lower row: 3 channels (X, Y, Z) for CNN model with different loss functions
    All channels must have SNR > 1 and peak amplitude > 15.
    """
    # Create save directory if it doesn't exist
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    with torch.no_grad():  
        device = torch.device(device)
        for model in models.values():
            model.to(device).eval()
        
        count = 0
        for noisy_data, clean_data in testloader:
            if count >= num_images: 
                break
            
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            
            # Get outputs from both models
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
                    
                    # Calculate SNR
                    if np.std(noisy_np) != 0:
                        snr = _paper_snr_1d(clean_np, noisy_np)
                    else:
                        snr = float('inf')
                    
                    # Calculate peak amplitude
                    envelope_clean = np.abs(hilbert(clean_np))
                    peak_amp = np.max(envelope_clean)
                    
                    snrs.append(snr)
                    peak_amps.append(peak_amp)
                    
                    # Check if this channel meets criteria
                    if not (snr_min < snr < snr_max) or peak_amp <= peak_amp_threshold:
                        valid_sample = False
                        break
                
                # Only plot if all channels meet criteria
                if not valid_sample:
                    continue
            
                channel_names = ['X Channel', 'Y Channel', 'Z Channel']
                model_names = ['Multi MSE Loss', 'Multi L1 Loss']# , 'MSE Loss', 'L1 Loss', 'PSNR Loss'
                colors = {'Multi MSE Loss': 'red', 'Multi L1 Loss': 'red'}#'MSE Loss': 'red', 'L1 Loss': 'red', 'PSNR Loss': 'red'
                
                # Create 4x3 subplot grid: rows=models, columns=channels
                fig, axes = plt.subplots(2, 3, figsize=(24, 12))
                fontsize = 30
                
                for row_idx, model_name in enumerate(model_names):
                    for channel_idx in range(3):
                        ax = axes[row_idx, channel_idx]
                        
                        clean_np = clean_data[sample_idx, channel_idx].cpu().numpy()
                        noisy_np = noisy_data[sample_idx, channel_idx].cpu().numpy()
                        denoised_np = model_outputs[model_name][sample_idx, channel_idx].cpu().numpy()
                        
                        snr = snrs[channel_idx]
                        peak_amp = peak_amps[channel_idx]
                        
                        # Calculate PSNR for this model
                        psnr_value = psnr(clean_np, denoised_np, np.max(clean_np))
                        scstft_value = spectral_convergence_stft(clean_np, denoised_np, n_fft=512, hop_length=256, eps=1e-8)

                        time_bin = np.arange(0, clean_np.size)
                        envelope_true = np.abs(hilbert(clean_np))
                        peak_time_true = time_bin[np.argmax(envelope_true)]
                        
                        # Plot traces
                        ax.plot(noisy_np, label='Noisy', linestyle=':', color='orange', linewidth=2)
                        ax.plot(clean_np, label='Clean', color='black', linewidth=3)
                        ax.plot(denoised_np, label=f'Denoised', color=colors[model_name], alpha=0.8, linewidth=2)
                        
                        ax.set_xlim(peak_time_true - 50, peak_time_true + 100)
                        
                        # Add performance metrics
                        metrics_text = f"SNR = {snr:.2f}\nPSNR = {psnr_value:.2f}"
                        ax.text(0.98, 0.95, metrics_text, 
                               transform=ax.transAxes, 
                               fontsize=fontsize-2, 
                               verticalalignment='top', 
                               horizontalalignment='right',
                               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
                        
                        # Add column titles (channel names) only to top row
                        if row_idx == 0:
                            ax.set_title(f'{channel_names[channel_idx]}', fontsize=fontsize, pad=20)
                        

                        # Add model name as text box in bottom left of first column panels
                        if channel_idx == 0:
                            ax.text(0.02, 0.02, f'{model_name}', 
                                   transform=ax.transAxes, 
                                   fontsize=fontsize-12, 
                                   verticalalignment='bottom', 
                                   horizontalalignment='left',
                                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round'))
                        
                        # Add legend only to top-left subplot
                        if row_idx == 0 and channel_idx == 0:
                            legend = ax.legend(fontsize=fontsize-10, loc='lower right')
                            legend.get_frame().set_edgecolor('none')
                        
                        # Set x-axis labels only for bottom row and 2nd column
                        if row_idx == 4 and channel_idx == 1:
                            ax.set_xlabel('Time [ns]', fontsize=fontsize+4)
                        
                        # Set y-axis labels only for 3rd row
                        if row_idx == 2 and channel_idx == 0:
                            ax.set_ylabel('ADC counts', fontsize=fontsize+4)
                        
                        ax.tick_params(axis='both', labelsize=fontsize-4)
                        ax.grid(True, alpha=0.3)

                plt.tight_layout(rect=[0.08, 0.06, 1, 0.94])
                
                if save_path:
                    plt.savefig(os.path.join(save_path, f'traces_comparison_2x3_{count:03d}_snr_{snrs[0]:.2f}.pdf'), 
                               dpi=300, bbox_inches='tight')
                plt.show()
                count += 1

            if count >= num_images:
                break
    
    print('Traces comparison completed')
def ablation_peak_time_efficiency_comparison(
    dataloader,
    model_dual,
    model_time,
    device: str = "cpu",
    min_snr: float = 1.0,
    max_snr: float = 15.0,
    snr_bins_count: int = 14,
    dt_ns: float = _DT_NS,   # 0.5 ns/sample, the value stated in the paper (was 2.0).
                             # Peak times here are argmax(envelope) * dt_ns, so this
                             # also rescales the |dt| <= time_tolerance_ns selection.
    time_tolerance_ns: float = 10.0,
    save_path: str = "",
):
    """
    Compares the peak time efficiency of two models (e.g., DualBranch vs TimeOnly)
    using the rigorous Hilbert envelope method.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from scipy.signal import hilbert
    import os

    model_dual.eval()
    model_time.eval()
    
    # Store errors and SNRs
    # Structure: { channel_idx: {"snr": [], "dual_err": [], "time_err": []} }
    results = {0: {"snr": [], "dual_err": [], "time_err": []},
               1: {"snr": [], "dual_err": [], "time_err": []},
               2: {"snr": [], "dual_err": [], "time_err": []}}

    with torch.no_grad():
        for noisy_data, clean_data in dataloader:
            noisy_data = noisy_data.to(device)
            clean_data = clean_data.to(device)
            
            dual_output = model_dual(noisy_data)
            time_output = model_time(noisy_data)

            batch_size = noisy_data.size(0)
            for i in range(batch_size):
                for ch in range(3):
                    clean_np = clean_data[i, ch].cpu().numpy()
                    noisy_np = noisy_data[i, ch].cpu().numpy()
                    dual_np = dual_output[i, ch].cpu().numpy()
                    time_np = time_output[i, ch].cpu().numpy()

                    # Paper SNR: max(clean) / std(noisy) over the full trace,
                    # the same definition used by every other figure here.
                    denom = np.std(noisy_np)
                    snr = (np.max(clean_np) / denom) if denom != 0 else float("inf")

                    if min_snr <= snr <= max_snr:
                        env_clean = np.abs(hilbert(clean_np))
                        env_dual = np.abs(hilbert(dual_np))
                        env_time = np.abs(hilbert(time_np))

                        peak_clean = np.argmax(env_clean) * dt_ns
                        peak_dual = np.argmax(env_dual) * dt_ns
                        peak_time = np.argmax(env_time) * dt_ns

                        results[ch]["snr"].append(snr)
                        results[ch]["dual_err"].append(np.abs(peak_dual - peak_clean))
                        results[ch]["time_err"].append(np.abs(peak_time - peak_clean))

    # Plotting
    plt.figure(figsize=(18, 5))
    snr_bins = np.linspace(min_snr, max_snr, snr_bins_count + 1)
    channels = ["X Channel", "Y Channel", "Z Channel"]

    for ch in range(3):
        plt.subplot(1, 3, ch + 1)
        
        snr_arr = np.array(results[ch]["snr"])
        dual_err_arr = np.array(results[ch]["dual_err"])
        time_err_arr = np.array(results[ch]["time_err"])
        
        dual_eff = []
        time_eff = []
        
        for i in range(len(snr_bins) - 1):
            mask = (snr_arr >= snr_bins[i]) & (snr_arr < snr_bins[i+1])
            if np.sum(mask) == 0:
                dual_eff.append(np.nan)
                time_eff.append(np.nan)
            else:
                dual_eff.append(np.mean(dual_err_arr[mask] <= time_tolerance_ns))
                time_eff.append(np.mean(time_err_arr[mask] <= time_tolerance_ns))
                
        plt.step(snr_bins[:-1], dual_eff, where='post', label='DualBranch', color='k', linewidth=2.5)
        plt.step(snr_bins[:-1], time_eff, where='post', label='TimeOnly', color='C1', linewidth=2.5, linestyle='--')
        
        plt.xlabel('Signal-to-Noise Ratio (SNR)', fontsize=14)
        if ch == 0:
            plt.ylabel(f'Denoising Efficiency ($|\\Delta t| \\leq {time_tolerance_ns:g}$ ns)', fontsize=14)
        plt.title(f'Peak Time Efficiency - {channels[ch]}', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.xlim(min_snr, max_snr)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    plt.show()

