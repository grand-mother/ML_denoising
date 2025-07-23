"""
Functions for checking the metrics

"""
import os
import sys
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader 
import numpy as np
from scipy.signal import hilbert
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('/Users/923714256/ML_denoising')


# from ml_denoising_lib.utils.fft import psd, bandwidth_filter

from DU_response_computation import apply_rfchain as rfc

class CustomDataset(Dataset):
    def __init__(self, noised_signals, clean_signals, traces_len=256, indices=None, no_random=False):
        """
        Args:
            noised_signals: Tuple of lists containing noised X, Y, Z signal components.
            clean_signals: Tuple of lists containing clean X, Y, Z signal components.
            indices: Array-like list of indices specifying which samples to include.
        """
        self.indices = indices if indices is not None else list(range(len(noised_signals[0])))

        self.noised_signals = noised_signals
        self.initial_len = noised_signals.shape[-1]
        self.clean_signals = clean_signals
        self.traces_len = traces_len
        self.no_random = no_random


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        if self.no_random:
            idstart = 0
            self.traces_len = self.initial_len
        else:
            idstart = int(np.random.rand(1)*(self.initial_len - self.traces_len))


        # Properly access the sample data
        noised_x = self.noised_signals[0][actual_idx][idstart:idstart+self.traces_len]
        noised_y = self.noised_signals[1][actual_idx][idstart:idstart+self.traces_len]
        noised_z = self.noised_signals[2][actual_idx][idstart:idstart+self.traces_len]
        clean_x = self.clean_signals[0][actual_idx][idstart:idstart+self.traces_len]
        clean_y = self.clean_signals[1][actual_idx][idstart:idstart+self.traces_len]
        clean_z = self.clean_signals[2][actual_idx][idstart:idstart+self.traces_len]

        # Convert to PyTorch tensors
        noised_signals = np.stack([noised_x, noised_y, noised_z], axis=0)
        clean_signals = np.stack([clean_x, clean_y, clean_z], axis=0)

        return torch.tensor(noised_signals, dtype=torch.float32), torch.tensor(clean_signals, dtype=torch.float32)

   

class CustomDataset_hdf5(Dataset):
    def __init__(self, clean_signals, noiser, traces_len=256, indices=None, no_random=False, lst=18, xy_mode=False, which_noise='gauss_gal', real_an_noisefile=''):
        """
        Args:
           
            clean_signals: Tuple of lists containing clean X, Y, Z signal components.
            noiser: noise_compute object from rfc
            indices: Array-like list of indices specifying which samples to include.
        """
        self.indices = indices if indices is not None else list(range(len(clean_signals[0])))
        self.initial_len = clean_signals.shape[-1]
        self.clean_signals = clean_signals
        self.traces_len = traces_len
        self.no_random = no_random

        self.noiser = noiser
        self.adc = 0.9/2**13

        self.n_traces = len(self.indices)
        self.compt = 0
        self.lst = lst
        self.xy_mode = xy_mode
        if which_noise == 'gauss_an':
            self.noise_directory = '/sps/grand/blevy/sims/noise_spectrum_from_anfiles_full/'
        else:
            self.noise_directory = '/sps/grand/blevy/sims/noise_spectrum_June_MD/'
        self.which_noise = which_noise
        self.real_an_noisefile = real_an_noisefile

        self.regenerate_noise()

    def regenerate_noise(self):
        print('Time to regenerate the noise !!!')
        print('we use the {} noise'.format(self.which_noise))
        lst = self.lst
        if self.which_noise == 'gauss_gal':
            self.gal_noise, _ = self.noiser.noise_samples(self.lst, self.n_traces, micro=False)
        elif self.which_noise == 'gauss_an':
            self.gal_noise, _ = self.noiser.noise_samples_from_existing_spectra(self.noise_directory, n_samples=self.n_traces, micro=False)
            self.gal_noise *= self.adc
        elif self.which_noise == 'gauss_md':
            self.gal_noise, _ = self.noiser.noise_samples_from_existing_spectra(self.noise_directory, n_samples=self.n_traces, micro=False)
            self.gal_noise *= self.adc
        elif self.which_noise == 'real_an':
            self.gal_noise = self.noiser.noise_samples_from_existing_traces(self.real_an_noisefile, n_samples=self.n_traces, seed=None, micro=False)
            self.gal_noise = self.gal_noise.astype(np.float64)
            self.gal_noise *= self.adc
        elif self.which_noise == 'pure_gauss_an':
            self.gal_noise, _ = self.noiser.noise_samples_from_existing_spectra(self.noise_directory, n_samples=self.n_traces, micro=False)
            self.gal_noise *= self.adc
            self.clean_signals *= 0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]

        if self.no_random:
            idstart = 0
            self.traces_len = self.initial_len
        else:
            idstart = int(np.random.rand(1)*(self.initial_len - self.traces_len-300))

        clean_x = self.clean_signals[0][actual_idx][idstart:idstart+self.traces_len] * self.adc
        clean_y = self.clean_signals[1][actual_idx][idstart:idstart+self.traces_len] * self.adc
        if not (self.xy_mode):
            clean_z = self.clean_signals[2][actual_idx][idstart:idstart+self.traces_len] * self.adc

        noised_x = clean_x + self.gal_noise[idx, 0][idstart:idstart+self.traces_len]
        noised_y = clean_y + self.gal_noise[idx, 1][idstart:idstart+self.traces_len]
        if not (self.xy_mode):
            noised_z = clean_z + self.gal_noise[idx, 2][idstart:idstart+self.traces_len]

        # Convert to PyTorch tensors
        if self.xy_mode:
            noised_signals = rfc.voltage_to_adc(np.stack([noised_x, noised_y], axis=0), micro=False)
            clean_signals = rfc.voltage_to_adc(np.stack([clean_x, clean_y], axis=0), micro=False)
        else:
            noised_signals = rfc.voltage_to_adc(np.stack([noised_x, noised_y, noised_z], axis=0), micro=False)
            clean_signals = rfc.voltage_to_adc(np.stack([clean_x, clean_y, clean_z], axis=0), micro=False)

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

def get_peak_amplitude(signal):
    """
    Function to get peak amplitude of a signal using Hilbert transform
    
    return peak amplitude
    """
    hilbert_amp = np.abs(hilbert(signal))  # Compute Hilbert transform and get amplitude
    peak_amplitude = np.max(hilbert_amp)  # Find peak amplitude
    return peak_amplitude

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

def psnr(target, ref, scale):
    target_data = np.array(target)
    ref_data = np.array(ref)
    diff = ref_data - target_data
    rmse = np.sqrt(np.mean(diff ** 2))
    max_pixel = scale
    psnr = 10 * np.log10(max_pixel**2 / rmse)
    return psnr

def psnr_loss(input, target, device='cpu'):
    """
    Psnr loss that use in the training loop plis

    return -psnr
    """
    # Ensure input is on the correct device and compute MSE loss
    mse_loss = F.mse_loss(input.to(device), target.to(device))
    
    # Detach the tensor, move it to CPU, and convert to NumPy array for get_peak_amplitude
    input_detached = input.detach().cpu().numpy()
    
    # Calculate peak amplitude using the detached array
    peak_amplitude = get_peak_amplitude(input_detached)
    
    # No need to move peak_amplitude to a device, as it's now a scalar value and will be used as such
    psnr = 10 * torch.log10((peak_amplitude**2) / mse_loss)
    
    return - psnr

def multi_domain_loss(clean, pred, fft_weight=0.1):
    # time-domain psnr + log-magnitude STFT psnr loss (frequency domain)
    psnr_time = psnr_loss(pred, clean, device= 'cpu')

    clean_fft = torch.fft.rfft(clean, dim=-1)
    pred_fft  = torch.fft.rfft(pred , dim=-1)
    psnr_freq = psnr_loss(torch.log1p(torch.abs(pred_fft)),
                       torch.log1p(torch.abs(clean_fft)), device= 'cpu')
    return psnr_time + fft_weight * psnr_freq

def multi_domain_loss_v2(clean, pred, mag_weight, phase_weight):
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

def plot_snr_distribution(noised_signals, clean_signals, save_folder):
    """Plot the distribution of SNRs."""
    snr_values = []

    # Zip the tuples to iterate over corresponding elements
    for noised_x, noised_y, noised_z, clean_x, clean_y, clean_z in zip(*noised_signals, *clean_signals):
        # Calculate SNR for each channel and append to snr_values
        snr_x = calculate_snr(clean_x, noised_x)
        snr_y = calculate_snr(clean_y, noised_y)
        snr_z = calculate_snr(clean_z, noised_z)
        
        # Append the SNR values for each channel
        snr_values.extend([snr_x, snr_y, snr_z])

    plt.figure(figsize=(12, 8))
    plt.hist(snr_values, bins=15, color='blue', alpha=0.7)
    plt.title('SNR Distribution', fontsize = 15)
    plt.xlabel('Signal-to-Noise Ratio',fontsize =15)
    plt.ylabel('Number Data', fontsize =15)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'snr_distribution.png'))
    plt.close()