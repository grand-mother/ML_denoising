import os

# Set the PYTHONPATH
# os.environ['PYTHONPATH'] = '/Users/923714256/ML_denoising'
# print(os.environ['PYTHONPATH'])

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
from raytune_training_function import psnr
from torch.utils.data import Dataset

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

#### Plot the peak time efficiency
def plot_peak_time_efficiency(snr_vals, clean_time, noisy_time, denoised_time, channel_name, save_path=None): #
    """
    Plots the fraction of traces with |Δt| > threshold for denoised and noisy signals as a function of SNR.
    """
    snr_bins = np.linspace(1, 20, 20)  # Adjust as needed
    thresholds = [10, 20]  # ns
    colors = ['orange', 'blue']
    linestyles = ['-', '--']

    plt.figure(figsize=(12, 6))
    fontsize = 16
    # Denoising efficiency: fraction of denoised traces with |Δt| <= 10 ns
    denoising_efficiency = []
    for i in range(len(snr_bins)-1):
        mask = (snr_vals >= snr_bins[i]) & (snr_vals < snr_bins[i+1])
        if np.sum(mask) == 0:
            denoising_efficiency.append(np.nan)
            continue
        # Denoising efficiency: fraction with |Δt| <= 10 ns
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
        plt.step(snr_bins[:-1], frac_denoised, where='post', color=colors[idx], linestyle='-', label=fr'$\Delta t_{{peak}} > {threshold}$ns, denoised')

        # Noisy (dashed)
        frac_noisy = []
        for i in range(len(snr_bins)-1):
            mask = (snr_vals >= snr_bins[i]) & (snr_vals < snr_bins[i+1])
            if np.sum(mask) == 0:
                frac_noisy.append(np.nan)
                continue
            frac_noisy.append(np.mean(np.abs(noisy_time[mask] - clean_time[mask]) > threshold))
        plt.step(snr_bins[:-1], frac_noisy, where='post', color=colors[idx], linestyle='--', label=fr'$\Delta t_{{peak}} > {threshold}$ns, noisy')

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
                        snr = np.max(clean_np) / np.std(noisy_np)
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
    noisy = np.array(peak_amplitudes[channel]['Noisy'])
    denoised = np.array(peak_amplitudes[channel]['Denoised'])
    plt.figure(figsize=(12, 6))
    fontsize = 16
    # Plot histogram
    plt.hist(noisy, bins=30, density=True, histtype='step', color='orange', linestyle='--', label=f'noisy - {channel}')
    plt.hist(denoised, bins=30, density=True, histtype='step', color='red', label='denoised')
    plt.yscale('log')
    plt.xlabel('Maximum peak amplitude [ADC]', fontsize=fontsize)
    plt.ylabel('Density', fontsize=fontsize)
    # plt.title(f'Peak Amplitude Distribution - {channel}', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    if save_path:
        plt.savefig(os.path.join(save_path, f'Peak_Amplitude_Distribution_{channel}.png'))
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
                        snr = np.max(clean_np) / np.std(noisy_np)
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
            sample_idx = 0  # Index of the sample to plot
            channel_names = ['X Channel', 'Y Channel', 'Z Channel']
            fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharex=True, sharey=True)
            
            for channel_idx in range(3): 
                # Create a single figure with 3 subplots, sharing x and y axes
                ax = axes[channel_idx]
                fontsize = 20
                clean_np = clean_data[sample_idx, channel_idx].cpu().numpy()
                noisy_np = noisy_data[sample_idx, channel_idx].cpu().numpy()
                denoised_np = denoised_output[sample_idx, channel_idx].cpu().numpy()
                snr = np.max(clean_np) / np.std(noisy_np)

                psnr_value = psnr(clean_np, denoised_np, np.max(clean_np))
                time_bin = np.arange(0, clean_np.size)
                envelope_true = np.abs(hilbert(clean_np))
                peak_time_true = time_bin[np.argmax(envelope_true)]
                if snr > 0:
                    ax.plot(noisy_np, label='Noisy', linestyle='--', color='orange')
                    ax.plot(clean_np, label='True', color='red')
                    ax.plot(denoised_np, label='Denoised', color='blue')
                    ax.axhline(y = np.std(noisy_np), color='black', linestyle='--')
                    ax.axhline(y = -np.std(noisy_np), color='black', linestyle='--')
                    ax.set_xlim(peak_time_true - 50, peak_time_true + 100)
    #                    ax.set_title(f"{channel_names[channel_idx]} (SNR = {snr:.2f}, PSNR = {psnr_value:.2f})", fontsize=fontsize)
                    ax.text(
                            0.98, 0.95, 
                            f"SNR = {snr:.2f}", 
                            transform=ax.transAxes, 
                            fontsize=fontsize, 
                            verticalalignment='top', 
                            horizontalalignment='right',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                        )
                    if channel_idx == 0:
                            legend = ax.legend(fontsize=fontsize)
                            legend.get_frame().set_edgecolor('none')
                    ax.tick_params(axis='both', labelsize=fontsize)

            # Shared axis labels
            fig.text(0.5, 0.04, 'Time [ns]', ha='center', va='center', fontsize=fontsize+2)
            fig.text(0.04, 0.5, 'ADC counts', ha='center', va='center', rotation='vertical', fontsize=fontsize+2)
            plt.tight_layout(rect=[0.06, 0.06, 1, 1])
            if save_path:
                plt.savefig(os.path.join(save_path, f'sample_{count:03d}_channel_{channel_idx}_snr_{snr:.2f}.png'))
            plt.show()
            count += 1  # Increment the count

            if count >= num_images:
                break
    print('test is completed') 

# def traces_plot(testloader, model, num_images=12, device="cpu"):
#     """
#     For each batch, pick one high-SNR sample and one low-SNR sample,
#     then plot their 3 channels in a 2×3 grid:
#     - Top row: high-SNR (>5)
#     - Bottom row: low-SNR (<5)
#     """
#     device = torch.device(device)
#     model = model.to(device).eval()
#     count = 0

#     with torch.no_grad():
#         for noisy_data, clean_data in testloader:
#             if count >= num_images:
#                 break

#             noisy_data = noisy_data.to(device)
#             clean_data = clean_data.to(device)
#             denoised_output = model(noisy_data)

#             # Compute an average-SNR per sample over the 3 channels
#             snrs = []
#             for i in range(noisy_data.size(0)):
#                 ch_snrs = []
#                 for ch in range(3):
#                     clean_np = clean_data[i, ch].cpu().numpy()
#                     noisy_np = noisy_data[i, ch].cpu().numpy()
#                     ch_snrs.append(np.max(clean_np) / np.std(noisy_np))
#                 snrs.append(np.mean(ch_snrs))

#             snrs = np.array(snrs)
#             # find one high‐SNR and one low‐SNR index
#             high_candidates = np.where(snrs > 5)[0]
#             low_candidates  = np.where(snrs < 5)[0]
#             if len(high_candidates)==0 or len(low_candidates)==0:
#                 # skip this batch if we can't find both
#                 continue

#             hi_idx = int(high_candidates[0])
#             lo_idx = int(low_candidates[0])

#             # prepare figure
#             fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex=True, sharey=True)
#             fontsize = 20

#             # Plot row 0 = high-SNR, row 1 = low-SNR
#             for row, sample_idx in enumerate([hi_idx, lo_idx]):
#                 for ch in range(3):
#                     ax = axes[row, ch]

#                     clean_np    = clean_data[sample_idx, ch].cpu().numpy()
#                     noisy_np    = noisy_data[sample_idx, ch].cpu().numpy()
#                     denoised_np = denoised_output[sample_idx, ch].cpu().numpy()

#                     # find the true peak to center the window
#                     envelope = np.abs(hilbert(clean_np))
#                     peak     = np.argmax(envelope)

#                     ax.plot(noisy_np,    linestyle='--', label='Noisy',    color='orange')
#                     ax.plot(clean_np,               label='True',     color='red')
#                     ax.plot(denoised_np,            label='Denoised', color='blue')
#                     ax.set_xlim(peak - 50, peak + 100)

#                     # annotate the actual SNR
#                     ax.text(
#                         0.98, 0.95,
#                         f"SNR = {snrs[sample_idx]:.2f}",
#                         transform=ax.transAxes,
#                         fontsize=fontsize,
#                         ha='right', va='top',
#                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
#                     )

#                     if ch == 0:
#                         ax.legend(fontsize=fontsize).get_frame().set_edgecolor('none')
#                     ax.tick_params(labelsize=fontsize)

#             # shared axis labels
#             fig.text(0.5, 0.04, 'Time [ns]',    ha='center', fontsize=fontsize+2)
#             fig.text(0.04, 0.5, 'ADC counts',    va='center', rotation='vertical', fontsize=fontsize+2)
#             plt.tight_layout(rect=[0.06, 0.06, 1, 1])
#             plt.show()

#             count += 1
#             if count >= num_images:
#                 break

#     print('test is completed')


### All channels plots
def plot_amplitude_ratio_vs_snr_all_channels(peak_amplitudes, snr_values, save_path=None):
    channel_names = ['X Channel', 'Y Channel', 'Z Channel']
    colors = {'X Channel': 'red', 'Y Channel': 'blue', 'Z Channel': 'green'}
    markers = {'Noisy': '^', 'Denoised': 'o'}
    plt.figure(figsize=(12, 6))
    fontsize = 14
    snr_bins = np.linspace(1, 10, 10)
    bin_centers = (snr_bins[:-1] + snr_bins[1:]) / 2

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

        plt.errorbar(bin_centers, means_denoised, yerr=stds_denoised, fmt=markers['Denoised'], 
                     color=colors[channel], label=f'Denoised - {channel}')
        plt.errorbar(bin_centers, means_noisy, yerr=stds_noisy, fmt=markers['Noisy'], 
                     color=colors[channel], label=f'Noisy - {channel}')

    plt.axhline(1, color='gray', linestyle='--')
    plt.xlabel('Signal-to-Noise ratio (SNR)', fontsize=fontsize)
    plt.ylabel('Amplitude ratio', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    if save_path:
        plt.savefig(os.path.join(save_path, f'Amplitude_Ratio_All_Channels.png'))
    plt.show()

def plot_peak_amplitude_distribution_all_channels(peak_amplitudes, save_path=None):
    channel_names = ['X Channel', 'Y Channel', 'Z Channel']
    colors = {'X Channel': 'orange', 'Y Channel': 'blue', 'Z Channel': 'green'}
    plt.figure(figsize=(12, 6))
    fontsize = 14

    for channel in channel_names:
        noisy = np.array(peak_amplitudes[channel]['Noisy'])
        denoised = np.array(peak_amplitudes[channel]['Denoised'])
        plt.hist(noisy, bins=30, density=True, histtype='step', color=colors[channel], linestyle='--', label=f'Noisy - {channel}')
        plt.hist(denoised, bins=30, density=True, histtype='step', color=colors[channel], label=f'Denoised - {channel}')

    plt.yscale('log')
    plt.xlabel('Maximum peak amplitude [ADC]', fontsize=fontsize)
    plt.ylabel('Density', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    if save_path:
        plt.savefig(os.path.join(save_path, f'Peak_Amplitude_Distribution_All_Channels.png'))
    plt.show()


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
                        snr = np.max(clean_np) / np.std(noisy_np)
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

    # Plot peak amplitude distribution for all channels together
    plot_peak_amplitude_distribution_all_channels(peak_amplitudes, save_path)                        


def traces_plot_v2(testloader, 
                   model, 
                   num_images = 12, 
                   device="cpu", 
                   save_path = ''
                   ):
    """
    Plot the traces of the clean, noisy and denoised signals for 2 row with 3 channels of images,
    sharing x and y axes, with shared axis labels.
    Upper row: SNR > 5 for all 3 channels
    Lower row: SNR < 5 for all 3 channels
    """
    with torch.no_grad():  
        device = torch.device(device)
        model = model.to(device)
        model.eval() 
        count = 0  # To count the number of images saved
        high_snr_sample = None
        low_snr_sample = None
        
        # Search for samples with high and low SNR
        for noisy_data, clean_data in testloader:
            if count >= num_images or (high_snr_sample is not None and low_snr_sample is not None): 
                break
            
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            denoised_output = model(noisy_data)

            # Check SNR for all channels of the first sample in batch
            sample_idx = 0
            snrs = []
            for channel_idx in range(3):
                clean_np = clean_data[sample_idx, channel_idx].cpu().numpy()
                noisy_np = noisy_data[sample_idx, channel_idx].cpu().numpy()
                snr = np.max(clean_np) / np.std(noisy_np)
                snrs.append(snr)
            
            # Check if all channels have high SNR (> 5) or low SNR (2 < 5)
            if high_snr_sample is None and all(snr > 5 for snr in snrs):
                high_snr_sample = {
                    'noisy': noisy_data.clone(),
                    'clean': clean_data.clone(), 
                    'denoised': denoised_output.clone(),
                    'snrs': snrs.copy()
                }
            elif low_snr_sample is None and all(2 < snr < 5 for snr in snrs):
                low_snr_sample = {
                    'noisy': noisy_data.clone(),
                    'clean': clean_data.clone(),
                    'denoised': denoised_output.clone(), 
                    'snrs': snrs.copy()
                }
                
        # If we found both types of samples, create the plot
        if high_snr_sample is not None and low_snr_sample is not None:
            channel_names = ['X Channel', 'Y Channel', 'Z Channel']
            fig, axes = plt.subplots(2, 3, figsize=(20, 10))#sharex=True, sharey=True)
            fontsize = 20
            
            samples = [high_snr_sample, low_snr_sample]
            row_labels = ['High SNR (>5)', 'Low SNR (2<5)']
            
            for row_idx, sample_data in enumerate(samples):
                for channel_idx in range(3):
                    ax = axes[row_idx, channel_idx]
                    
                    # Extract data for this channel
                    clean_np = sample_data['clean'][0, channel_idx].cpu().numpy()
                    noisy_np = sample_data['noisy'][0, channel_idx].cpu().numpy()
                    denoised_np = sample_data['denoised'][0, channel_idx].cpu().numpy()
                    snr = sample_data['snrs'][channel_idx]

                    # Calculate PSNR and find peak time for centering
                    psnr_value = psnr(clean_np, denoised_np, np.max(clean_np))
                    time_bin = np.arange(0, clean_np.size)
                    envelope_true = np.abs(hilbert(clean_np))
                    peak_time_true = time_bin[np.argmax(envelope_true)]
                    
                    # Plot the traces
                    ax.plot(noisy_np, label='Noisy', linestyle='--', color='orange', alpha=0.6)
                    ax.plot(clean_np, label='True', color='red', alpha=0.4)
                    ax.plot(denoised_np, label='Denoised', color='blue', alpha=0.5)
                    ax.axhline(y=np.std(noisy_np), color='black', linestyle='--', alpha=0.5)
                    ax.axhline(y=-np.std(noisy_np), color='black', linestyle='--', alpha=0.5)
                    
                    # Set zoom around peak
                    ax.set_xlim(peak_time_true - 50, peak_time_true + 100)
                    
                    # Add SNR and PSNR text
                    ax.text(
                        0.98, 0.95,
                        f"SNR = {snr:.2f}\nPSNR = {psnr_value:.2f}",
                        transform=ax.transAxes,
                        fontsize=fontsize-2,
                        verticalalignment='top',
                        horizontalalignment='right',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                    )
                    
                    # Add column titles (channel names) only to top row
                    # if row_idx == 0:
                    #     ax.set_title(f'{channel_names[channel_idx]}', fontsize=fontsize)
                    
                    # Add row labels only to first column
                    # if channel_idx == 0:
                    #     ax.text(
                    #         -0.15, 0.5,
                    #         row_labels[row_idx],
                    #         transform=ax.transAxes,
                    #         fontsize=fontsize,
                    #         verticalalignment='center',
                    #         horizontalalignment='center',
                    #         rotation=90,
                    #         weight='bold'
                    #     )
                    
                    # Add legend only to top-left subplot
                    if row_idx == 0 and channel_idx == 0:
                        legend = ax.legend(fontsize=fontsize-2, loc='lower right')
                        legend.get_frame().set_edgecolor('none')
                    
                    ax.tick_params(axis='both', labelsize=fontsize-2)
            
            # Shared axis labels
            fig.text(0.5, 0.04, 'Time [ns]', ha='center', va='center', fontsize=fontsize+2)
            fig.text(0.04, 0.5, 'ADC counts', ha='center', va='center', rotation='vertical', fontsize=fontsize+2)
            
            plt.tight_layout(rect=[0.08, 0.08, 1, 0.96])
            
            if save_path:
                plt.savefig(os.path.join(save_path, f'comparison_high_low_snr_{count:03d}.pdf'), dpi=300, bbox_inches='tight')
            plt.show()
            count += 1
            
        else:
            print(f"Warning: Could not find both high SNR (>5) and low SNR (<5) samples")
            
    print('traces_plot_v2 completed')



def plot_peak_time_efficiency_combined(snr_values, peak_times, save_path=None):
    """
    Plots the peak time efficiency for all 3 channels in one row.
    
    Args:
        snr_values: Dictionary with keys 'X Channel', 'Y Channel', 'Z Channel'
        peak_times: Dictionary with nested structure for each channel containing 'Clean', 'Noisy', 'Denoised' arrays
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fontsize = 16
    channel_names = ['X Channel', 'Y Channel', 'Z Channel']
    
    for ax_idx, channel_name in enumerate(channel_names):
        ax = axes[ax_idx]
        
        snr_vals = np.array(snr_values[channel_name])
        clean_time = np.array(peak_times[channel_name]['Clean'])
        noisy_time = np.array(peak_times[channel_name]['Noisy'])
        denoised_time = np.array(peak_times[channel_name]['Denoised'])
        
        snr_bins = np.linspace(1, 20, 20)
        thresholds = [10, 20]  # ns
        colors = ['orange', 'blue']
        
        # Denoising efficiency: fraction of denoised traces with |Δt| <= 10 ns
        denoising_efficiency = []
        for i in range(len(snr_bins)-1):
            mask = (snr_vals >= snr_bins[i]) & (snr_vals < snr_bins[i+1])
            if np.sum(mask) == 0:
                denoising_efficiency.append(np.nan)
                continue
            denoising_efficiency.append(np.mean(np.abs(denoised_time[mask] - clean_time[mask]) <= 10))
        ax.step(snr_bins[:-1], denoising_efficiency, where='post', color='k', linewidth=2, 
                label=f'Denoising efficiency')

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
            ax.step(snr_bins[:-1], frac_denoised, where='post', color=colors[idx], linestyle='-', 
                   label=fr'$\Delta t_{{peak}} > {threshold}$ns, denoised')

            # Noisy (dashed)
            frac_noisy = []
            for i in range(len(snr_bins)-1):
                mask = (snr_vals >= snr_bins[i]) & (snr_vals < snr_bins[i+1])
                if np.sum(mask) == 0:
                    frac_noisy.append(np.nan)
                    continue
                frac_noisy.append(np.mean(np.abs(noisy_time[mask] - clean_time[mask]) > threshold))
            ax.step(snr_bins[:-1], frac_noisy, where='post', color=colors[idx], linestyle='--', 
                   label=fr'$\Delta t_{{peak}} > {threshold}$ns, noisy')

        ax.set_xlabel('Signal-to-Noise ratio (SNR)', fontsize=fontsize)
        if ax_idx == 0:  # Only add ylabel to the leftmost plot
            ax.set_ylabel('Fraction', fontsize=fontsize)
        ax.set_title(f'{channel_name}', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.set_xlim(1, 10)
        ax.set_ylim(0, 1.05)
        if ax_idx == 2:  # Only add legend to the rightmost plot
            ax.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'Peak_Time_Efficiency_All_Channels.pdf'), 
                   dpi=300, bbox_inches='tight')
    plt.show()

def peak_time_analysis_for_all_channels(dataloader, 
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
                        snr = np.max(clean_np) / np.std(noisy_np)
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

    # Plot all 3 channels in one row
    plot_peak_time_efficiency_combined(snr_values, peak_times, save_path)