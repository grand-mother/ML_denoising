"""
Model comparison and evaluation utilities.
"""
import os
import sys
from pathlib import Path
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import hilbert, stft
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
from torch.utils.data import Dataset

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from training.raytune_training_function import psnr
from training.models.cnn import DualBranchAutoencoder as CNNModel

def band_energy_ratio(clean, pred, fs, band_min, band_max):
    """
    Computes the ratio of predicted to clean energy in a specified frequency band.
    
    Args:
        clean (np.ndarray): Clean reference signal, shape (N,)
        pred (np.ndarray): Model output signal, shape (N,)
        fs (float): Sampling rate in Hz
        band_min (float): Lower band edge in Hz
        band_max (float): Upper band edge in Hz
        
    Returns:
        float: Fractional energy recovery in the band (1.0 = perfect).
    """
    N = clean.shape[-1]
    freqs = np.fft.rfftfreq(N, d=1/fs)  # Frequency axis
    # Find indices corresponding to the band
    idx = np.where((freqs >= band_min) & (freqs <= band_max))[0]

    # Compute FFT and energy in band
    clean_fft = np.fft.rfft(clean)
    pred_fft = np.fft.rfft(pred)
    clean_energy = np.sum(np.abs(clean_fft[idx])**2)
    pred_energy = np.sum(np.abs(pred_fft[idx])**2)
    # To avoid zero-division, add small eps
    return pred_energy / (clean_energy + 1e-12)

def spectral_convergence_stft(clean, pred, n_fft=256, hop_length=128, eps=1e-8):
    # Compute STFT
    f_c, t_c, Zxx_c = stft(clean, nperseg=n_fft, noverlap=n_fft-hop_length)
    f_p, t_p, Zxx_p = stft(pred,  nperseg=n_fft, noverlap=n_fft-hop_length)
    # Compute Frobenius norms
    num = np.linalg.norm(np.abs(Zxx_p) - np.abs(Zxx_c), 'fro')
    denom = np.linalg.norm(np.abs(Zxx_c), 'fro') + eps
    return num / denom

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
        model_classes: Dict mapping model_type -> class, e.g. {"CNN": CNNModel}
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

def plot_peak_time_efficiency_comparison(snr_values_models, peak_times_models, channel, save_path=None):
    """
    Compare peak time efficiency between different models for a specific channel.
    
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
    
    colors = {'CNN_multi_mse_loss': 'blue','CNN_multi_l1_loss': 'blue', 'CNN_mse': 'red', 'CNN_l1': 'green', 'CNN_psnr': 'purple'}
    linestyles = {'CNN_multi_l1_loss': '-', 'CNN_mse': '--', 'CNN_l1': '-.', 'CNN_psnr': ':'}
    
    snr_bins = np.linspace(1, 20, 20)
    thresholds = [10, 20]  # ns
    threshold_colors = ['orange', 'green']
    
    for model_name in ['CNN_multiloss', 'CNN_mse', 'CNN_l1', 'CNN_psnr']:
        snr_vals = np.array(snr_values_models[model_name][channel])
        clean_time = np.array(peak_times_models[model_name][channel]['Clean'])
        noisy_time = np.array(peak_times_models[model_name][channel]['Noisy'])  
        denoised_time = np.array(peak_times_models[model_name][channel]['Denoised'])
        
        # Denoising efficiency: fraction of denoised traces with |Δt| <= 10 ns
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
                    label=fr'{model_name} - $\Delta t_{{peak}} > {threshold}$ns')

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
    Compare amplitude ratios between different models for a specific channel.
    """
    # Create save directory if it doesn't exist
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    fontsize = 16
    
    colors = {'CNN_multi_mse_loss': 'blue', 'CNN_multi_l1_loss': 'blue', 'CNN_mse': 'red', 'CNN_l1': 'green', 'CNN_psnr': 'purple'}
    markers = {'CNN_multi_mse_loss': 'o', 'CNN_multi_l1_loss': 'o', 'CNN_mse': '^', 'CNN_l1': 's', 'CNN_psnr': 'd'}
    
    snr_bins = np.linspace(1, 10, 10)
    bin_centers = (snr_bins[:-1] + snr_bins[1:]) / 2

    for model_name in ['CNN_multi_mse_loss', 'CNN_multi_l1_loss', 'CNN_mse', 'CNN_l1', 'CNN_psnr']:
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
    snr = np.array(snr_values_models['CNN_multi_mse_loss'][channel])  # Use CNN data as reference
    clean = np.array(peak_amplitudes_models['CNN_multi_mse_loss'][channel]['Clean'])
    noisy = np.array(peak_amplitudes_models['CNN_multi_mse_loss'][channel]['Noisy'])
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
    Comprehensive comparison analysis between CNN models with different loss functions.
    
    Args:
        dataloader: DataLoader for test data
        models: Dictionary containing loaded models {'CNN_multi_mse_loss': model, 'CNN_multi_l1_loss': model, 'CNN_mse': model, 'CNN_l1': model, 'CNN_psnr': model}
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
    sc_values_models = {}
    band_energy_ratio_values_models = {}
    
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
        sc_values_models[model_name] = {channel: [] for channel in channel_names}
        band_energy_ratio_values_models[model_name] = {channel: [] for channel in channel_names}

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
                    
                    # Paper SNR: max(clean) / std(noisy) over the full trace.
                    if np.std(noisy_np) != 0:
                        snr = np.max(clean_np) / np.std(noisy_np)
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

                            # Calculate spectral convergence
                            sc_val = spectral_convergence_stft(clean_np, denoised_np, n_fft=512, hop_length=256, eps=1e-8)
                            sc_values_models[model_name][channel].append(sc_val)

                            # Calculate log-magnitude STFT PSNR
                            band_energy_ratio_val = band_energy_ratio(clean_np, denoised_np, fs=2e9, band_min=50e6, band_max=250e6)
                            band_energy_ratio_values_models[model_name][channel].append(band_energy_ratio_val)

    # Generate comparison plots
    # for channel in channel_names:
    #     plot_peak_time_efficiency_comparison(snr_values_models, peak_times_models, channel, save_path)
    #     plot_amplitude_ratio_comparison(peak_amplitudes_models, snr_values_models, channel, save_path)
    
    # # Combined plots for all channels
    # plot_peak_time_efficiency_combined_comparison(snr_values_models, peak_times_models, save_path)
    # plot_amplitude_ratio_all_channels_comparison(peak_amplitudes_models, snr_values_models, save_path)
    
    return peak_times_models, peak_amplitudes_models, snr_values_models, psnr_values_models, sc_values_models, band_energy_ratio_values_models

def plot_peak_time_efficiency_combined_comparison(snr_values_models, peak_times_models, save_path=None):
    """
    Plot peak time efficiency comparison for all 3 channels in one row, comparing different models.
    """
    # Create save directory if it doesn't exist
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fontsize = 16
    channel_names = ['X Channel', 'Y Channel', 'Z Channel']
    
    colors = {'CNN_multi_mse_loss': 'blue', 'CNN_multi_l1_loss': 'blue', 'CNN_mse': 'red', 'CNN_l1': 'green', 'CNN_psnr': 'purple'}
    linestyles = {'CNN_multiloss': '-', 'CNN_mse': '--', 'CNN_l1': '-.', 'CNN_psnr': ':'}
    
    for ax_idx, channel_name in enumerate(channel_names):
        ax = axes[ax_idx]
        
        for model_name in ['CNN_multiloss', 'CNN_mse']:
            snr_vals = np.array(snr_values_models[model_name][channel_name])
            clean_time = np.array(peak_times_models[model_name][channel_name]['Clean'])
            denoised_time = np.array(peak_times_models[model_name][channel_name]['Denoised'])
            
            snr_bins = np.linspace(1, 20, 20)
            
            # Denoising efficiency: fraction with |Δt| <= 10 ns
            denoising_efficiency = []
            for i in range(len(snr_bins)-1):
                mask = (snr_vals >= snr_bins[i]) & (snr_vals < snr_bins[i+1])
                if np.sum(mask) == 0:
                    denoising_efficiency.append(np.nan)
                    continue
                denoising_efficiency.append(np.mean(np.abs(denoised_time[mask] - clean_time[mask]) <= 10))
            
            ax.step(snr_bins[:-1], denoising_efficiency, where='post', 
                   color=colors[model_name], linewidth=3, 
                   linestyle=linestyles[model_name],
                   label=f'{model_name}')

        ax.set_xlabel('Signal-to-Noise ratio (SNR)', fontsize=fontsize)
        if ax_idx == 0:
            ax.set_ylabel('Denoising Efficiency', fontsize=fontsize)
        ax.set_title(f'{channel_name}', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.set_xlim(1, 10)
        ax.set_ylim(0, 1.05)
        if ax_idx == 2:  # Add legend to rightmost plot
            ax.legend(fontsize=fontsize, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'Peak_Time_Efficiency_Combined_Comparison.pdf'), 
                   dpi=300, bbox_inches='tight')
    plt.show()

def plot_amplitude_ratio_all_channels_comparison(peak_amplitudes_models, snr_values_models, save_path=None):
    """
    Compare amplitude ratios for all channels between CNN models with different loss functions.
    """
    # Create save directory if it doesn't exist
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    channel_names = ['X Channel', 'Y Channel', 'Z Channel']
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fontsize = 14
    
    colors = {'CNN_multiloss': 'blue', 'CNN_mse': 'red', 'CNN_l1': 'green', 'CNN_psnr': 'purple'}
    markers = {'CNN_multiloss': 'o', 'CNN_mse': '^', 'CNN_l1': 's', 'CNN_psnr': 'd'}
    
    snr_bins = np.linspace(1, 10, 10)
    bin_centers = (snr_bins[:-1] + snr_bins[1:]) / 2

    for ax_idx, channel in enumerate(channel_names):
        ax = axes[ax_idx]
        
        for model_name in ['CNN_multiloss', 'CNN_mse', 'CNN_l1', 'CNN_psnr']:
            snr = np.array(snr_values_models[model_name][channel])
            clean = np.array(peak_amplitudes_models[model_name][channel]['Clean'])
            denoised = np.array(peak_amplitudes_models[model_name][channel]['Denoised'])

            ratio_denoised = denoised / clean
            means_denoised, stds_denoised = [], []

            for i in range(len(snr_bins)-1):
                mask = (snr >= snr_bins[i]) & (snr < snr_bins[i+1])
                means_denoised.append(np.mean(ratio_denoised[mask]) if np.any(mask) else np.nan)
                stds_denoised.append(np.std(ratio_denoised[mask]) if np.any(mask) else np.nan)

            ax.errorbar(bin_centers, means_denoised, yerr=stds_denoised, 
                       fmt=markers[model_name], color=colors[model_name], 
                       linewidth=2, markersize=6, capsize=4,
                       label=f'{model_name}')

        ax.axhline(1, color='gray', linestyle='--', alpha=0.7)
        ax.set_xlabel('Signal-to-Noise ratio (SNR)', fontsize=fontsize)
        if ax_idx == 0:
            ax.set_ylabel('Amplitude ratio', fontsize=fontsize)
        ax.set_title(f'{channel}', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        if ax_idx == 2:  # Add legend to rightmost plot
            ax.legend(fontsize=fontsize, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'Amplitude_Ratio_All_Channels_Comparison.pdf'), 
                   dpi=300, bbox_inches='tight')
    plt.show()

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
                    # Paper SNR: max(clean) / std(noisy) over the full trace.
                    if np.std(noisy_np) != 0:
                        snr = np.max(clean_np) / np.std(noisy_np)
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
                model_names = ['Multi MSE Loss', 'Multi L1 Loss', 'MSE Loss', 'L1 Loss', 'PSNR Loss']
                colors = {'Multi MSE Loss': 'red', 'Multi L1 Loss': 'red', 'MSE Loss': 'red', 'L1 Loss': 'red', 'PSNR Loss': 'red'}
                
                # Create 4x3 subplot grid: rows=models, columns=channels
                fig, axes = plt.subplots(5, 3, figsize=(45, 50))
                fontsize = 50
                
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
                        enrgy_ratio = band_energy_ratio(clean_np, denoised_np, fs=2e9, band_min=50e6, band_max=250e6)
                        
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

