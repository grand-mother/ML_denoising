"""
Functions for checking the metrics and training utilities.
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import numpy as np
from scipy.signal import hilbert
import torch.nn.functional as F
import matplotlib.pyplot as plt
import auraloss

# Add project root to path for external dependencies
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Optional: Import RF chain if available
try:
    from DU_response_computation import apply_rfchain as rfc
except ImportError:
    rfc = None
    print("Warning: DU_response_computation not found, RF chain functions unavailable")

class CustomDataset(Dataset):
    def __init__(self, clean_signals, noise_signals, traces_len=512, indices=None, 
                 no_random=False, swap_prob=0.5, target_start=300, target_end=500, voltage_to_adc=False):
        """
        Args:
            clean_signals: Array containing clean X, Y, Z microvolt signal components with the shape (3, n_samples, 1024).
            noise_signals: List of noise signal arrays or single concatenated array with the shape (3, n_noise_samples, 1024).
            indices: Array-like list of indices specifying which samples to include.
            swap_prob: Probability of applying trace swapping (default 0.5).
            target_start: Start position for swap target region (default 300).
            target_end: End position for swap target region (default 500).
        """
        self.total_samples = sum(arr.shape[1] for arr in noise_signals)
        self.indices = indices if indices is not None else list(range(self.total_samples))
        self.initial_len = clean_signals.shape[-1]
        self.clean_signals = clean_signals
        self.noise_signals_list = noise_signals
        self.traces_len = traces_len
        self.no_random = no_random
        self.swap_prob = swap_prob
        self.target_start = target_start
        self.target_end = target_end
        self.voltage_to_adc = voltage_to_adc
        # Pre-calculate cumulative indices for efficient file lookup
        self.cumulative_samples = np.cumsum([0] + [arr.shape[1] for arr in noise_signals])
    
    def __len__(self):
        return len(self.indices)
    
    def _get_file_and_local_idx(self, global_idx):
        """Convert global index to file index and local index within that file."""
        file_idx = np.searchsorted(self.cumulative_samples, global_idx + 1) - 1
        local_idx = global_idx - self.cumulative_samples[file_idx]
        return file_idx, local_idx

    def _maybe_slice_swap_traces(self, clean_x, clean_y, clean_z, noise_x, noise_y, noise_z):
        """
        With probability swap_prob, swap a slice [target_start:target_end]
        between clean and noisy triplets along a random source region.
        Returns the (clean_x, noise_x, clean_y, noise_y, clean_z, noise_z).
        """
        if np.random.rand() >= self.swap_prob:
            return clean_x, noise_x, clean_y, noise_y, clean_z, noise_z
        
        target_length = self.target_end - self.target_start
        sig_size = len(clean_x)
        
        # Choose source region: either before target_start or after target_end
        if np.random.rand() < 0.5:
            # Pick from region before target_start
            max_start = self.target_start - target_length
            if max_start < 0:
                # Not enough space before, skip swap
                return clean_x, noise_x, clean_y, noise_y, clean_z, noise_z
            source_start = np.random.randint(0, max_start + 1)
        else:
            # Pick from region after target_end
            min_start = self.target_end
            max_start = sig_size - target_length
            if max_start < min_start:
                # Not enough space after, skip swap
                return clean_x, noise_x, clean_y, noise_y, clean_z, noise_z
            source_start = np.random.randint(min_start, max_start + 1)
        
        source_end = source_start + target_length
        
        # Make copies
        x_c, y_c, z_c = clean_x.copy(), clean_y.copy(), clean_z.copy()
        x_n, y_n, z_n = noise_x.copy(), noise_y.copy(), noise_z.copy()
        
        # Store target regions temporarily
        x_tmp_c = x_c[self.target_start:self.target_end].copy()
        y_tmp_c = y_c[self.target_start:self.target_end].copy()
        z_tmp_c = z_c[self.target_start:self.target_end].copy()
        
        x_tmp_n = x_n[self.target_start:self.target_end].copy()
        y_tmp_n = y_n[self.target_start:self.target_end].copy()
        z_tmp_n = z_n[self.target_start:self.target_end].copy()
        
        # Copy source to target
        x_c[self.target_start:self.target_end] = x_c[source_start:source_end]
        y_c[self.target_start:self.target_end] = y_c[source_start:source_end]
        z_c[self.target_start:self.target_end] = z_c[source_start:source_end]
        
        x_n[self.target_start:self.target_end] = x_n[source_start:source_end]
        y_n[self.target_start:self.target_end] = y_n[source_start:source_end]
        z_n[self.target_start:self.target_end] = z_n[source_start:source_end]
        
        # Copy stored target to source
        x_c[source_start:source_end] = x_tmp_c
        y_c[source_start:source_end] = y_tmp_c
        z_c[source_start:source_end] = z_tmp_c
        
        x_n[source_start:source_end] = x_tmp_n
        y_n[source_start:source_end] = y_tmp_n
        z_n[source_start:source_end] = z_tmp_n
        
        return x_c, x_n, y_c, y_n, z_c, z_n

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]

        if self.no_random:
            idstart = 0
            traces_len = self.initial_len
        else:
            pulse_start = 240
            pulse_end = 280
            traces_len = self.traces_len

            # We want pulse to appear anywhere in the window from position ~50 to ~240
            min_pulse_position = 50   # Earliest position in window for pulse center
            max_pulse_position = 240  # Latest position (limited by pulse_start)
            
            # Calculate idstart range based on desired pulse position
            # Pulse position in window = pulse_start - idstart
            # We want: min_pulse_position <= (pulse_start - idstart) <= max_pulse_position
            # Rearranging: pulse_start - max_pulse_position <= idstart <= pulse_start - min_pulse_position
            min_idstart = pulse_start - max_pulse_position
            max_idstart = pulse_start - min_pulse_position
            
            # Ensure pulse end is also in window: pulse_end - idstart <= traces_len
            # So: idstart >= pulse_end - traces_len
            min_idstart = max(min_idstart, pulse_end - traces_len)
            
            # Ensure we don't go negative or past the end of signal
            min_idstart = max(0, min_idstart)
            max_idstart = min(max_idstart, self.initial_len - traces_len)
            
            if min_idstart <= max_idstart:
                idstart = int(min_idstart + np.random.rand(1) * (max_idstart - min_idstart))
            else:
                # If constraints are impossible, center the pulse
                idstart = max(0, min(pulse_start - traces_len // 2, self.initial_len - traces_len))

        # Get file index and local index for the noise signal
        noise_idx = actual_idx % self.total_samples
        file_idx, local_idx = self._get_file_and_local_idx(noise_idx)
        
        # Extract signals
        clean_x = self.clean_signals[0][actual_idx][idstart:idstart+traces_len]
        clean_y = self.clean_signals[1][actual_idx][idstart:idstart+traces_len]
        clean_z = self.clean_signals[2][actual_idx][idstart:idstart+traces_len]
        
        _noise_component_x = self.noise_signals_list[file_idx][0, local_idx, idstart:idstart+traces_len]
        _noise_component_y = self.noise_signals_list[file_idx][1, local_idx, idstart:idstart+traces_len]
        _noise_component_z = self.noise_signals_list[file_idx][2, local_idx, idstart:idstart+traces_len]
        
        noise_x = clean_x + _noise_component_x
        noise_y = clean_y + _noise_component_y
        noise_z = clean_z + _noise_component_z
        
        # Apply swap if enabled
        if self.swap_prob > 0:
            clean_x, noise_x, clean_y, noise_y, clean_z, noise_z = self._maybe_slice_swap_traces(
                clean_x, clean_y, clean_z, noise_x, noise_y, noise_z
            )
        
        # Convert to PyTorch tensors
        if self.voltage_to_adc:
            noised_signals = rfc.voltage_to_adc(np.stack([noise_x, noise_y, noise_z], axis=0), micro=True)
            clean_signals = rfc.voltage_to_adc(np.stack([clean_x, clean_y, clean_z], axis=0), micro=True)
        else:
            noised_signals = np.stack([noise_x, noise_y, noise_z], axis=0)
            clean_signals = np.stack([clean_x, clean_y, clean_z], axis=0)

        # Handle Inf and NaN values by replacing with clamped values
        MAX_SIGNAL_VALUE = 1e6  # Reasonable max for ADC signals
        if np.isinf(noised_signals).any() or np.isnan(noised_signals).any():
            noised_signals = np.clip(np.nan_to_num(noised_signals, nan=0.0, posinf=MAX_SIGNAL_VALUE, neginf=-MAX_SIGNAL_VALUE), -MAX_SIGNAL_VALUE, MAX_SIGNAL_VALUE)
        if np.isinf(clean_signals).any() or np.isnan(clean_signals).any():
            clean_signals = np.clip(np.nan_to_num(clean_signals, nan=0.0, posinf=MAX_SIGNAL_VALUE, neginf=-MAX_SIGNAL_VALUE), -MAX_SIGNAL_VALUE, MAX_SIGNAL_VALUE)

        return torch.tensor(noised_signals, dtype=torch.float32), torch.tensor(clean_signals, dtype=torch.float32)

def split_indices(n, train_frac=0.8, valid_frac=0.1):
    """
    Split indices into training, validation, and test sets.
    default: 80% are train data, 10% are validation data
    
    """
    indices = np.arange(n)
    np.random.shuffle(indices)

    train_size = int(n * train_frac)
    valid_size = int(n * valid_frac)

    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
    test_indices = indices[train_size + valid_size:]

    return train_indices, valid_indices, test_indices

def get_peak_amplitude(signal, eps=1e-8):
    """
    Function to get peak amplitude of a signal using Hilbert transform
    
    return peak amplitude (with numerical stability protection)
    """
    # Check for NaN or Inf in input signal
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        # Return a safe default value instead of propagating NaN
        return eps
    
    try:
        hilbert_amp = np.abs(hilbert(signal))  # Compute Hilbert transform and get amplitude
        peak_amplitude = np.max(hilbert_amp)  # Find peak amplitude
        
        # Ensure we don't return 0 or negative values
        if peak_amplitude <= 0 or np.isnan(peak_amplitude) or np.isinf(peak_amplitude):
            return eps
        
        return peak_amplitude
    except Exception:
        # If Hilbert transform fails, return a safe default
        return eps

def calculate_psnr_with_peak(original_signal, reconstructed_signal):
    """
    Function to calculate PSNR using peak amplitude of the original signal
    
    return psnr
    """
    peak_amplitude = get_peak_amplitude(original_signal)  # Get peak amplitude of original signal
    mse_loss = np.mean((original_signal - reconstructed_signal) ** 2)  # Calculate MSE
    if mse_loss == 0:
        return float('inf')  # Return infinity if MSE is zero to indicate perfect reconstruction
    max_i = peak_amplitude  # Use peak amplitude as MAX_I for PSNR calculation
    with np.errstate(divide='ignore'):
        psnr_value = 10 * np.log10((max_i ** 2) / mse_loss)  # Calculate PSNR
    return psnr_value

def peak_to_peak_ratio(original, reconstructed):
    """
    Peak to peak ratio metrics
    
    return ratio 
    """
    original_amp = np.abs(hilbert(original))
    reconstructed_amp = np.abs(hilbert(reconstructed))
    max_original_amp = np.max(original_amp)
    if max_original_amp == 0:
        return float('inf')  # Return infinity if max_original_amp is zero to avoid division by zero
    ratio = np.abs((np.max(original_amp) - np.max(reconstructed_amp))) / max_original_amp
    return ratio

#### metrics 

def psnr(target, ref, scale):
    target_data = np.array(target)
    ref_data = np.array(ref)
    diff = ref_data - target_data
    rmse = np.sqrt(np.mean(diff ** 2))
    max_pixel = scale
    psnr = 10 * np.log10(max_pixel**2 / rmse)
    return psnr


### psnr_loss
def psnr_loss(input, target, device='cpu', eps=1e-8):
    """
    PSNR loss for training loop with numerical stability protection.
    
    Args:
        input: Predicted signal tensor
        target: Ground truth signal tensor
        device: Device for computation
        eps: Small value to prevent division by zero and log(0)
    
    Returns:
        -psnr (negative PSNR for minimization)
    """
    # Check for NaN in input tensors
    if torch.isnan(input).any() or torch.isnan(target).any():
        # Return a large but finite loss to allow training to continue
        return torch.tensor(100.0, device=device, requires_grad=True)
    
    # Ensure input is on the correct device and compute MSE loss
    mse_loss = F.mse_loss(input.to(device), target.to(device))
    
    # Add eps to prevent division by zero
    mse_loss = mse_loss + eps
    
    # Detach the tensor, move it to CPU, and convert to NumPy array for get_peak_amplitude
    input_detached = input.detach().cpu().numpy()
    
    # Calculate peak amplitude using the detached array (now with built-in protection)
    peak_amplitude = get_peak_amplitude(input_detached, eps=eps)
    
    # Compute PSNR with numerical stability
    # Clamp the ratio to prevent extreme values
    ratio = (peak_amplitude**2) / mse_loss
    ratio = torch.clamp(ratio, min=eps, max=1e10)
    
    psnr = 10 * torch.log10(ratio)
    
    # Final NaN check - if still NaN, return large finite loss
    if torch.isnan(psnr) or torch.isinf(psnr):
        return torch.tensor(100.0, device=device, requires_grad=True)
    
    return -psnr

def multi_domain_mse_loss(clean, pred, mag_weight, phase_weight):
    # Time domain loss
    time_loss = F.mse_loss(pred, clean)
    
    # Frequency domain components
    clean_fft = torch.fft.rfft(clean, dim=-1)
    pred_fft = torch.fft.rfft(pred, dim=-1)
    
    # Magnitude loss - preserves energy distribution
    mag_loss = F.mse_loss(torch.abs(pred_fft), torch.abs(clean_fft))
    
    # Phase loss - preserves waveform shape and timing
    phase_loss = F.mse_loss(torch.angle(pred_fft), torch.angle(clean_fft))
    
    # Combined frequency loss
    freq_loss = mag_weight * mag_loss + phase_weight * phase_loss
    
    return time_loss + freq_loss

def multi_domain_l1_loss(clean, pred, mag_weight, phase_weight):
    # Time domain loss
    time_loss = F.l1_loss(pred, clean)
    
    # Frequency domain components
    clean_fft = torch.fft.rfft(clean, dim=-1)
    pred_fft = torch.fft.rfft(pred, dim=-1)
    
    # Magnitude loss - preserves energy distribution
    mag_loss = F.l1_loss(torch.abs(pred_fft), torch.abs(clean_fft))
    
    # Phase loss - preserves waveform shape and timing
    phase_loss = F.l1_loss(torch.angle(pred_fft), torch.angle(clean_fft))
    
    # Combined frequency loss
    freq_loss = mag_weight * mag_loss + phase_weight * phase_loss
    
    return time_loss + freq_loss

def plot_metrics(epochs, training_losses, validation_losses, validation_psnr, learning_rates, validation_peak_to_peak, save_folder):
    """
    Plot four metrics versus epochs and save the figures to a specified folder.

    Training Loss and validation loss versus epochs
    Validation PSNR versus epochs
    Peak to Peak ratio versus epochs
    Learning rate versus epochs
    
    save_folder for saving the metrics into the folder, string: name of the file
    """
    # Debug print statements
    print(f"DEBUG: plotting metrics with {len(epochs)} epochs")
    print(f"DEBUG: training_losses length: {len(training_losses)}, range: {np.min(training_losses):.6f} - {np.max(training_losses):.6f}")
    print(f"DEBUG: validation_losses length: {len(validation_losses)}, range: {np.min(validation_losses):.6f} - {np.max(validation_losses):.6f}")
    print(f"DEBUG: training_losses sample: {training_losses[:5]}")
    print(f"DEBUG: validation_losses sample: {validation_losses[:5]}")
    # Create directory if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    
    plt.figure(figsize=(20, 15))

    # Plotting Training and Validation Loss
    plt.subplot(4, 1, 1)
    plt.title('Training and Validation loss vs Epochs', fontsize = 20)
    plt.plot(epochs, training_losses, 
                label='Training loss', linewidth=2, marker='o', markersize=3)
    plt.plot(epochs, validation_losses, 
                label='Validation loss', color='orange', linewidth=2, marker='s', markersize=3)
    plt.xlabel('Epochs', fontsize = 20)
    plt.ylabel('Loss', fontsize = 20)
    # plt.yscale('log')
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(fontsize = 20)
    
    # Plotting Validation PSNR
    plt.subplot(4, 1, 2)
    plt.plot(epochs, validation_psnr, label='Validation PSNR', color='green')
    plt.title('PSNR vs Epochs', fontsize = 20)
    plt.xlabel('Epochs', fontsize = 20)
    plt.ylabel('PSNR', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(fontsize = 20)
    
    # Plotting Learning Rate
    plt.subplot(4, 1, 3)
    plt.plot(epochs, learning_rates, label='Learning Rate', color='cyan')
    plt.xlabel('Epochs', fontsize = 20)
    plt.ylabel('Learning Rate', fontsize = 20)
    plt.title('Learning Rate vs Epochs', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(fontsize = 20)
    plt.legend(fontsize = 20)

    # Plotting Peak-to-Peak Amplitude
    plt.subplot(4, 1, 4)
    plt.plot(epochs, validation_peak_to_peak, label='Validation Peak-to-Peak', color='magenta')
    plt.xlabel('Epochs', fontsize = 20)
    plt.ylabel('Peak-to-Peak Amplitude', fontsize = 20)
    plt.title('Peak-to-Peak Amplitude vs Epochs', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(fontsize = 20)
    plt.legend(fontsize = 20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'metrics.pdf'))
    plt.close()

def calculate_snr(clean_array, noisy_array):
    """Calculate the SNR of a signal."""
    snr = np.max(clean_array) / np.std(noisy_array)
    return snr

def plot_snr_distribution(noised_signals, clean_signals, save_folder=None):
    """Plot the distribution of SNRs, separated by channel, using different linestyles."""
    snr_x_values, snr_y_values, snr_z_values = [], [], []

    for noised_x, noised_y, noised_z, clean_x, clean_y, clean_z in zip(*noised_signals, *clean_signals):
        snr_x_values.append(calculate_snr(clean_x, noised_x))
        snr_y_values.append(calculate_snr(clean_y, noised_y))
        snr_z_values.append(calculate_snr(clean_z, noised_z))

    plt.figure(figsize=(12, 10))
    # Unit-width bins: [0,1), [1,2), ...
    max_snr = max(max(snr_x_values), max(snr_y_values), max(snr_z_values))
    bins = np.arange(0, 16, 1.0)
    # Outline histograms with different linestyles
    plt.hist(snr_x_values, bins=bins, histtype='step', linewidth=2.0, linestyle='-',  color='tab:red',   label='X channel')
    plt.hist(snr_y_values, bins=bins, histtype='step', linewidth=2.0, linestyle='--', color='tab:green', label='Y channel')
    plt.hist(snr_z_values, bins=bins, histtype='step', linewidth=2.0, linestyle=':',  color='tab:blue',  label='Z channel')
    fontsize = 26
    # plt.title('SNR Distribution by Channel', fontsize=18)
    plt.xlabel('Signal-to-Noise Ratio', fontsize=fontsize)
    plt.ylabel('Count', fontsize=fontsize)
    plt.yscale('log')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_folder:
        plt.savefig(os.path.join(save_folder, 'snr_distribution_by_channel.pdf'))
        plt.show()
    else:
        plt.show()