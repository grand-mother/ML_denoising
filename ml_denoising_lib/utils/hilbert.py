import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
import os



def peak_time_and_amplitude(
    dataloader, model, device='cpu', min_snr=1, max_snr=1e3, save_folder=''
):
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    
    # Initialize dictionaries to store data per channel
    peak_amplitudes = {'X Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                       'Y Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                       'Z Channel': {'Clean': [], 'Noisy': [], 'Denoised': []}}
    
    # Initialize dictionaries to store data per channel
    peak_times = {'X Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                  'Y Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                  'Z Channel': {'Clean': [], 'Noisy': [], 'Denoised': []}}
    snr_values = {'X Channel': [], 'Y Channel': [], 'Z Channel': []}
    channel_names = ['X Channel', 'Y Channel', 'Z Channel']

    with torch.no_grad():
        for noisy_data, clean_data in dataloader:
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            denoised_output = model(noisy_data)
            n_channels = denoised_output.shape[1]
            batch_size = noisy_data.size(0)
            for i in range(batch_size):
                for idx, channel in enumerate(channel_names[0:n_channels]):
                    clean_np = clean_data[i, idx].cpu().numpy()
                    noisy_np = noisy_data[i, idx].cpu().numpy()
                    denoised_np = denoised_output[i, idx].cpu().numpy()
                    
                    timing = np.arange(clean_np.size)

                    # Calculate envelopes
                    envelope_clean = np.abs(hilbert(clean_np))
                    envelope_noisy = np.abs(hilbert(noisy_np))
                    envelope_denoised = np.abs(hilbert(denoised_np))

                    if np.std(noisy_np) != 0:
                        snr = np.max(envelope_clean) / np.std(noisy_np)
                    else:
                        snr = float('inf')
                    

                    

                    # Store peak amplitudes
                    peak_amplitudes[channel]['Clean'].append(np.max(envelope_clean).astype(np.float64))
                    peak_amplitudes[channel]['Noisy'].append(np.max(envelope_noisy).astype(np.float64))
                    peak_amplitudes[channel]['Denoised'].append(np.max(envelope_denoised).astype(np.float64))

                    # Find peak times
                    peak_time_clean = timing[np.argmax(envelope_clean)]
                    peak_time_noisy = timing[np.argmax(envelope_noisy)]
                    peak_time_denoised = timing[np.argmax(envelope_denoised)]

                    # Store peak times
                    peak_times[channel]['Clean'].append(peak_time_clean.astype(np.float64))
                    peak_times[channel]['Noisy'].append(peak_time_noisy.astype(np.float64))
                    peak_times[channel]['Denoised'].append(peak_time_denoised.astype(np.float64))

                    snr_values[channel].append(snr.astype(np.float64))


                            # Plotting per channel and saving individual figures
    for idx, channel in enumerate(channel_names):
        # peak amplitudes
        clean_amplitude = np.array(peak_amplitudes[channel]['Clean'])
        noisy_amplitude = np.array(peak_amplitudes[channel]['Noisy'])
        denoised_amplitude = np.array(peak_amplitudes[channel]['Denoised'])

        # peak times
        clean_time = np.array(peak_times[channel]['Clean'])
        noisy_time = np.array(peak_times[channel]['Noisy'])
        denoised_time = np.array(peak_times[channel]['Denoised'])

        snr_vals = np.array(snr_values[channel])

        # Check if arrays are not empty
        if clean_time.size == 0 or noisy_time.size == 0 or denoised_time.size == 0 or clean_amplitude.size == 0 or noisy_amplitude.size == 0 or denoised_amplitude.size == 0:
            print(f"No data to plot for {channel}. Skipping...")
            continue

        # Calculating MSE between scatter points and x=y line for peak times
        errors_noisy_time = noisy_time - clean_time
        mse_noisy_time = np.mean(errors_noisy_time ** 2)

        errors_denoised_time = denoised_time - clean_time
        mse_denoised_time = np.mean(errors_denoised_time ** 2)

        # Calculating MSE between scatter points and x=y line for peak amplitudes
        errors_noisy_amplitude = noisy_amplitude - clean_amplitude
        mse_noisy_amplitude = np.mean(errors_noisy_amplitude ** 2)

        errors_denoised_amplitude = denoised_amplitude - clean_amplitude
        mse_denoised_amplitude = np.mean(errors_denoised_amplitude ** 2)

        fig, axs = plt.subplots(2, 2, figsize=(16, 16), constrained_layout=True)
        # Find global min and max values to ensure equal axis limits
        min_val_time = max(min(clean_time.min(), noisy_time.min(), denoised_time.min()), 1e-2)
        max_val_time = max(clean_time.max(), noisy_time.max(), denoised_time.max())

        min_val_amp = max(min(clean_amplitude.min(), noisy_amplitude.min(), denoised_amplitude.min()), 1e-2)
        max_val_amp = max(clean_amplitude.max(), noisy_amplitude.max(), denoised_amplitude.max())

        # Noisy_time vs Clean_time
        sc1= axs[0,0].scatter(clean_time, noisy_time, c=snr_vals, cmap='viridis', alpha=0.6)
        axs[0,0].plot([min_val_time, max_val_time], [min_val_time, max_val_time], 'r--', label=f'x=y (MSE: {mse_noisy_time:.2f})')
        axs[0,0].set_xlabel('Peak Time of Clean Data (Counts)', fontsize = 18)
        axs[0,0].set_ylabel('Peak Time of Noisy Data (Counts)', fontsize = 18)
        axs[0,0].set_title(f'Noisy vs Clean - {channel}', fontsize = 18)
        axs[0,0].set_xscale('log')
        axs[0,0].set_yscale('log')
        axs[0, 0].set_xlim(min_val_time, max_val_time)
        axs[0, 0].set_ylim(min_val_time, max_val_time)
        axs[0, 0].set_aspect('equal', adjustable='box')
        axs[0, 0].set_title('Noisy vs Clean - Peak Time', fontsize = 18)
        axs[0, 0].tick_params(axis='both', which='major', labelsize=18)


        # Denoised_time vs Clean_time
        sc2 = axs[1,0].scatter(clean_time, denoised_time, c=snr_vals, cmap='viridis', alpha=0.6)
        axs[1,0].plot([min_val_time, max_val_time], [min_val_time, max_val_time], 'r--', label=f'x=y (MSE: {mse_denoised_time:.2f})')
        axs[1,0].set_xlabel('Peak Time of Clean Data (Counts)', fontsize = 18)
        axs[1,0].set_ylabel('Peak Time of Denoised Data (Counts)', fontsize = 18)
        axs[1,0].set_title(f'Denoised vs Clean - {channel}', fontsize = 18)
        axs[1,0].set_xscale('log')
        axs[1,0].set_yscale('log')
        axs[1, 0].set_xlim(min_val_time, max_val_time)
        axs[1, 0].set_ylim(min_val_time, max_val_time)
        axs[1, 0].set_aspect('equal', adjustable='box')
        axs[1, 0].set_title('Denoised vs Clean - Peak Time', fontsize = 18)
        axs[1, 0].tick_params(axis='both', which='major', labelsize=18)

        # Noisy_amplitude vs Clean_amplitude
        sc3 = axs[0,1].scatter(clean_amplitude, noisy_amplitude, c=snr_vals, cmap='viridis', alpha=0.6)
        axs[0,1].plot([min_val_amp, max_val_amp], [min_val_amp, max_val_amp], 'r--', label=f'x=y (MSE: {mse_noisy_amplitude:.2f})')
        axs[0,1].set_xlabel('Peak Amplitude of Clean Data (Counts)', fontsize = 18)
        axs[0,1].set_ylabel('Peak Amplitude of Noisy Data (Counts)', fontsize = 18)
        axs[0,1].set_title(f'Noisy vs Clean - {channel}', fontsize = 18)
        axs[0,1].set_xscale('log')
        axs[0,1].set_yscale('log')     
        axs[0, 1].set_xlim(min_val_amp, max_val_amp)
        axs[0, 1].set_ylim(min_val_amp, max_val_amp)
        axs[0, 1].set_aspect('equal', adjustable='box')
        axs[0, 1].set_title('Noisy vs Clean - Peak Amplitude', fontsize = 18)
        axs[0, 1].tick_params(axis='both', which='major', labelsize=18)

        # Denoised_amplitude vs Clean_amplitude
        sc4 = axs[1,1].scatter(clean_amplitude, denoised_amplitude, c=snr_vals, cmap='viridis', alpha=0.6)
        axs[1,1].plot([min_val_amp, max_val_amp], [min_val_amp, max_val_amp], 'r--', label=f'x=y (MSE: {mse_denoised_amplitude:.2f})')
        axs[1,1].set_xlabel('Peak Amplitude of Clean Data (Counts)', fontsize = 18)
        axs[1,1].set_ylabel('Peak Amplitude of Denoised Data (Counts)', fontsize = 18)
        axs[1,1].set_title(f'Denoised vs Clean - {channel}', fontsize = 18)
        axs[1,1].set_xscale('log')
        axs[1,1].set_yscale('log')
        axs[1, 1].set_xlim(min_val_amp, max_val_amp)
        axs[1, 1].set_ylim(min_val_amp, max_val_amp)
        axs[1, 1].set_aspect('equal', adjustable='box')
        axs[1, 1].set_title('Denoised vs Clean - Peak Amplitude', fontsize = 18)
        axs[1, 1].tick_params(axis='both', which='major', labelsize=18)

        # Add colorbars
        # cbar1 = fig.colorbar(sc1, ax=axs[0, 0], orientation='vertical', label='SNR')
        # cbar2 = fig.cozlorbar(sc2, ax=axs[1, 0], orientation='vertical', label='SNR')
        cbar3 = fig.colorbar(sc3, ax=axs[0, 1], orientation='vertical', label='SNR')
        cbar3.ax.tick_params(labelsize=18)  # Increases tick font size
        cbar3.set_label('SNR', fontsize=16) 

        cbar4 = fig.colorbar(sc4, ax=axs[1, 1], orientation='vertical', label='SNR')
        cbar4.ax.tick_params(labelsize=14)  # Increases tick font size
        cbar4.set_label('SNR', fontsize=16)  # Increases label font size
        # Save the figure for the current channel
        plt.savefig(os.path.join(save_folder, f'Peak_Amplitude_{channel.replace(" ", "_")}_leftMSE={mse_noisy_amplitude:.2f}_rightMSE={mse_denoised_amplitude:.2f}_Peak_Time_{channel.replace(" ", "_")}_leftMSE={mse_noisy_time:.2f}_rightMSE={mse_denoised_time:.2f}.png'))
        plt.close()
    print('Peak amplitude and time graphs have been saved individually for each channel.')
    return peak_amplitudes, snr_values, peak_times




def peak_time_and_amplitude_v2(
    dataloader, model, device='cpu', min_snr=1, max_snr=1e3, save_folder=''
):

    ''' this is meant tyo be used on full test traces size1024, where the pulse, is always in the first half
        since the noise is computed in the second half of the traces so as not bias the noise variance
    '''
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    
    # Initialize dictionaries to store data per channel
    peak_amplitudes = {'X Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                       'Y Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                       'Z Channel': {'Clean': [], 'Noisy': [], 'Denoised': []}}
    abs_amplitudes = {'X Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                       'Y Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                       'Z Channel': {'Clean': [], 'Noisy': [], 'Denoised': []}}
    
    # Initialize dictionaries to store data per channel
    peak_times = {'X Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                  'Y Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                  'Z Channel': {'Clean': [], 'Noisy': [], 'Denoised': []}}
    snr_values = {'X Channel': [], 'Y Channel': [], 'Z Channel': []}
    channel_names = ['X Channel', 'Y Channel', 'Z Channel']

    with torch.no_grad():
        for noisy_data, clean_data in dataloader:
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            denoised_output = model(noisy_data)
            n_channels = denoised_output.shape[1]
            batch_size = noisy_data.size(0)
            for i in range(batch_size):
                for idx, channel in enumerate(channel_names[0:n_channels]):
                    clean_np = clean_data[i, idx].cpu().numpy()
                    noisy_np = noisy_data[i, idx].cpu().numpy()
                    denoised_np = denoised_output[i, idx].cpu().numpy()
                    
                    timing = np.arange(clean_np.size)

                    # Calculate envelopes
                    envelope_clean = np.abs(hilbert(clean_np))
                    envelope_noisy = np.abs(hilbert(noisy_np))
                    envelope_denoised = np.abs(hilbert(denoised_np))
                    noisy_std = noisy_np[int((clean_np.size)/2):].std()
                    
                    if noisy_std != 0:
                        snr = np.max(envelope_clean) / noisy_std
                    else:
                        snr = float('inf')
                    

                    
                    # Store peak amplitudes
                    peak_amplitudes[channel]['Clean'].append(np.max(envelope_clean).astype(np.float64))
                    peak_amplitudes[channel]['Noisy'].append(np.max(envelope_noisy).astype(np.float64))
                    peak_amplitudes[channel]['Denoised'].append(np.max(envelope_denoised).astype(np.float64))

                    abs_amplitudes[channel]['Clean'].append(np.max(np.abs(clean_np)).astype(np.float64))
                    abs_amplitudes[channel]['Noisy'].append(np.max(np.abs(noisy_np)).astype(np.float64))
                    abs_amplitudes[channel]['Denoised'].append(np.max(np.abs(denoised_np)).astype(np.float64))



                    # Find peak times
                    peak_time_clean = timing[np.argmax(envelope_clean)]
                    peak_time_noisy = timing[np.argmax(envelope_noisy)]
                    peak_time_denoised = timing[np.argmax(envelope_denoised)]

                    # Store peak times
                    peak_times[channel]['Clean'].append(peak_time_clean.astype(np.float64))
                    peak_times[channel]['Noisy'].append(peak_time_noisy.astype(np.float64))
                    peak_times[channel]['Denoised'].append(peak_time_denoised.astype(np.float64))

                    snr_values[channel].append(snr.astype(np.float64))


                            # Plotting per channel and saving individual figures
    for idx, channel in enumerate(channel_names):
        # peak amplitudes
        clean_amplitude = np.array(peak_amplitudes[channel]['Clean'])
        noisy_amplitude = np.array(peak_amplitudes[channel]['Noisy'])
        denoised_amplitude = np.array(peak_amplitudes[channel]['Denoised'])

        # peak times
        clean_time = np.array(peak_times[channel]['Clean'])
        noisy_time = np.array(peak_times[channel]['Noisy'])
        denoised_time = np.array(peak_times[channel]['Denoised'])

        snr_vals = np.array(snr_values[channel])

        # Check if arrays are not empty
        if clean_time.size == 0 or noisy_time.size == 0 or denoised_time.size == 0 or clean_amplitude.size == 0 or noisy_amplitude.size == 0 or denoised_amplitude.size == 0:
            print(f"No data to plot for {channel}. Skipping...")
            continue

        # Calculating MSE between scatter points and x=y line for peak times
        errors_noisy_time = noisy_time - clean_time
        mse_noisy_time = np.mean(errors_noisy_time ** 2)

        errors_denoised_time = denoised_time - clean_time
        mse_denoised_time = np.mean(errors_denoised_time ** 2)

        # Calculating MSE between scatter points and x=y line for peak amplitudes
        errors_noisy_amplitude = noisy_amplitude - clean_amplitude
        mse_noisy_amplitude = np.mean(errors_noisy_amplitude ** 2)

        errors_denoised_amplitude = denoised_amplitude - clean_amplitude
        mse_denoised_amplitude = np.mean(errors_denoised_amplitude ** 2)

        fig, axs = plt.subplots(2, 2, figsize=(16, 16), constrained_layout=True)
        # Find global min and max values to ensure equal axis limits
        min_val_time = max(min(clean_time.min(), noisy_time.min(), denoised_time.min()), 1e-2)
        max_val_time = max(clean_time.max(), noisy_time.max(), denoised_time.max())

        min_val_amp = max(min(clean_amplitude.min(), noisy_amplitude.min(), denoised_amplitude.min()), 1e-2)
        max_val_amp = max(clean_amplitude.max(), noisy_amplitude.max(), denoised_amplitude.max())

        # Noisy_time vs Clean_time
        sc1= axs[0,0].scatter(clean_time, noisy_time, c=snr_vals, cmap='viridis', alpha=0.6)
        axs[0,0].plot([min_val_time, max_val_time], [min_val_time, max_val_time], 'r--', label=f'x=y (MSE: {mse_noisy_time:.2f})')
        axs[0,0].set_xlabel('Peak Time of Clean Data (Counts)', fontsize = 18)
        axs[0,0].set_ylabel('Peak Time of Noisy Data (Counts)', fontsize = 18)
        axs[0,0].set_title(f'Noisy vs Clean - {channel}', fontsize = 18)
        axs[0,0].set_xscale('log')
        axs[0,0].set_yscale('log')
        axs[0, 0].set_xlim(min_val_time, max_val_time)
        axs[0, 0].set_ylim(min_val_time, max_val_time)
        axs[0, 0].set_aspect('equal', adjustable='box')
        axs[0, 0].set_title('Noisy vs Clean - Peak Time', fontsize = 18)
        axs[0, 0].tick_params(axis='both', which='major', labelsize=18)


        # Denoised_time vs Clean_time
        sc2 = axs[1,0].scatter(clean_time, denoised_time, c=snr_vals, cmap='viridis', alpha=0.6)
        axs[1,0].plot([min_val_time, max_val_time], [min_val_time, max_val_time], 'r--', label=f'x=y (MSE: {mse_denoised_time:.2f})')
        axs[1,0].set_xlabel('Peak Time of Clean Data (Counts)', fontsize = 18)
        axs[1,0].set_ylabel('Peak Time of Denoised Data (Counts)', fontsize = 18)
        axs[1,0].set_title(f'Denoised vs Clean - {channel}', fontsize = 18)
        axs[1,0].set_xscale('log')
        axs[1,0].set_yscale('log')
        axs[1, 0].set_xlim(min_val_time, max_val_time)
        axs[1, 0].set_ylim(min_val_time, max_val_time)
        axs[1, 0].set_aspect('equal', adjustable='box')
        axs[1, 0].set_title('Denoised vs Clean - Peak Time', fontsize = 18)
        axs[1, 0].tick_params(axis='both', which='major', labelsize=18)

        # Noisy_amplitude vs Clean_amplitude
        sc3 = axs[0,1].scatter(clean_amplitude, noisy_amplitude, c=snr_vals, cmap='viridis', alpha=0.6)
        axs[0,1].plot([min_val_amp, max_val_amp], [min_val_amp, max_val_amp], 'r--', label=f'x=y (MSE: {mse_noisy_amplitude:.2f})')
        axs[0,1].set_xlabel('Peak Amplitude of Clean Data (Counts)', fontsize = 18)
        axs[0,1].set_ylabel('Peak Amplitude of Noisy Data (Counts)', fontsize = 18)
        axs[0,1].set_title(f'Noisy vs Clean - {channel}', fontsize = 18)
        axs[0,1].set_xscale('log')
        axs[0,1].set_yscale('log')     
        axs[0, 1].set_xlim(min_val_amp, max_val_amp)
        axs[0, 1].set_ylim(min_val_amp, max_val_amp)
        axs[0, 1].set_aspect('equal', adjustable='box')
        axs[0, 1].set_title('Noisy vs Clean - Peak Amplitude', fontsize = 18)
        axs[0, 1].tick_params(axis='both', which='major', labelsize=18)

        # Denoised_amplitude vs Clean_amplitude
        sc4 = axs[1,1].scatter(clean_amplitude, denoised_amplitude, c=snr_vals, cmap='viridis', alpha=0.6)
        axs[1,1].plot([min_val_amp, max_val_amp], [min_val_amp, max_val_amp], 'r--', label=f'x=y (MSE: {mse_denoised_amplitude:.2f})')
        axs[1,1].set_xlabel('Peak Amplitude of Clean Data (Counts)', fontsize = 18)
        axs[1,1].set_ylabel('Peak Amplitude of Denoised Data (Counts)', fontsize = 18)
        axs[1,1].set_title(f'Denoised vs Clean - {channel}', fontsize = 18)
        axs[1,1].set_xscale('log')
        axs[1,1].set_yscale('log')
        axs[1, 1].set_xlim(min_val_amp, max_val_amp)
        axs[1, 1].set_ylim(min_val_amp, max_val_amp)
        axs[1, 1].set_aspect('equal', adjustable='box')
        axs[1, 1].set_title('Denoised vs Clean - Peak Amplitude', fontsize = 18)
        axs[1, 1].tick_params(axis='both', which='major', labelsize=18)

        # Add colorbars
        # cbar1 = fig.colorbar(sc1, ax=axs[0, 0], orientation='vertical', label='SNR')
        # cbar2 = fig.cozlorbar(sc2, ax=axs[1, 0], orientation='vertical', label='SNR')
        cbar3 = fig.colorbar(sc3, ax=axs[0, 1], orientation='vertical', label='SNR')
        cbar3.ax.tick_params(labelsize=18)  # Increases tick font size
        cbar3.set_label('SNR', fontsize=16) 

        cbar4 = fig.colorbar(sc4, ax=axs[1, 1], orientation='vertical', label='SNR')
        cbar4.ax.tick_params(labelsize=14)  # Increases tick font size
        cbar4.set_label('SNR', fontsize=16)  # Increases label font size
        # Save the figure for the current channel
        plt.savefig(os.path.join(save_folder, f'Peak_Amplitude_{channel.replace(" ", "_")}_leftMSE={mse_noisy_amplitude:.2f}_rightMSE={mse_denoised_amplitude:.2f}_Peak_Time_{channel.replace(" ", "_")}_leftMSE={mse_noisy_time:.2f}_rightMSE={mse_denoised_time:.2f}.png'))
        plt.close()
    print('Peak amplitude and time graphs have been saved individually for each channel.')
    return peak_amplitudes, snr_values, peak_times, abs_amplitudes








def peak_amplitude(dataloader, model, device='cpu', min_snr=1, max_snr=1e3, save_folder=''):
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    # Initialize dictionaries to store data per channel
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

                    # Calculate envelopes
                    envelope_clean = np.abs(hilbert(clean_np))
                    envelope_noisy = np.abs(hilbert(noisy_np))
                    envelope_denoised = np.abs(hilbert(denoised_np))

                    # Store peak amplitudes
                    peak_amplitudes[channel]['Clean'].append(np.max(envelope_clean).astype(np.float64))
                    peak_amplitudes[channel]['Noisy'].append(np.max(envelope_noisy).astype(np.float64))
                    peak_amplitudes[channel]['Denoised'].append(np.max(envelope_denoised).astype(np.float64))

                    if np.std(noisy_np) != 0:
                        snr = np.max(envelope_clean) / np.std(noisy_np)
                    else:
                        snr = float('inf')

                    snr = snr.astype(np.float64)
                    snr_values[channel].append(snr)

    # Plotting per channel and saving individual figures
    for idx, channel in enumerate(channel_names):
        clean = np.array(peak_amplitudes[channel]['Clean'])
        noisy = np.array(peak_amplitudes[channel]['Noisy'])
        denoised = np.array(peak_amplitudes[channel]['Denoised'])
        snr_vals = np.array(snr_values[channel])
                # Check if arrays are not empty
        if clean.size == 0 or noisy.size == 0 or denoised.size == 0:
            print(f"No data to plot for {channel}. Skipping...")
            continue
        
        # Calculating MSE between scatter points and x=y line
        errors_noisy = noisy - clean
        mse_noisy = np.mean(errors_noisy ** 2)
        
        errors_denoised = denoised - clean
        mse_denoised = np.mean(errors_denoised ** 2)
        
        plt.figure(figsize=(16, 10))
        
        # Noisy vs Clean
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(clean, noisy, c=snr_vals, cmap='viridis', alpha=0.6)
        plt.plot([clean.min(), clean.max()], [clean.min(), clean.max()], 'r--', label=f'x=y (MSE: {mse_noisy:.2f})')
        cbar = plt.colorbar(scatter, label='SNR')
        cbar.ax.tick_params(labelsize = 18)
        plt.xlabel('Peak Amplitude of Clean Data (Counts)', fontsize = 18)
        plt.ylabel('Peak Amplitude of Noisy Data (Counts)', fontsize = 18)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.title(f'Noisy vs Clean - {channel}', fontsize = 18)
        plt.legend(fontsize = 18)
        plt.grid(True)
        plt.xscale('log')
        plt.yscale('log')
        
        # Denoised vs Clean
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(clean, denoised, c=snr_vals, cmap='viridis', alpha=0.6)
        plt.plot([clean.min(), clean.max()], [clean.min(), clean.max()], 'r--', label=f'x=y (MSE: {mse_denoised:.2f})')
        cbar = plt.colorbar(scatter, label='SNR')
        cbar.ax.tick_params(labelsize = 18)
        plt.xlabel('Peak Amplitude of Clean Data (Counts)', fontsize = 18)
        plt.ylabel('Peak Amplitude of Denoised Data (Counts)', fontsize = 18)
        plt.title(f'Denoised vs Clean - {channel}', fontsize = 18)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.legend(fontsize = 18)
        plt.grid(True)
        plt.xscale('log')
        plt.yscale('log')
        
        plt.tight_layout()
        # Save the figure for the current channel
        plt.savefig(os.path.join(save_folder, f'Peak_Amplitude_{channel.replace(" ", "_")}_leftMSE={mse_noisy:.2f}_rightMSE={mse_denoised:.2f}.png'))
        plt.close()
    print('Peak amplitude graphs have been saved individually for each channel.')
    return peak_amplitudes, snr_values


def peak_time(dataloader, model, device='cpu', min_snr=1, max_snr=1e3, save_folder=''):
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    
    # Initialize dictionaries to store data per channel
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
                for idx, channel in enumerate(channel_names):
                    clean_np = clean_data[i, idx].cpu().numpy()
                    noisy_np = noisy_data[i, idx].cpu().numpy()
                    denoised_np = denoised_output[i, idx].cpu().numpy()
                    
                    timing = np.arange(clean_np.size)
                    
                    if np.std(noisy_np) != 0:
                        snr = np.max(clean_np) / np.std(noisy_np)
                    else:
                        snr = float('inf')
                    
                    if max_snr > snr > min_snr:
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
    
    # Plotting per channel and saving individual figures
    for idx, channel in enumerate(channel_names):
        clean = np.array(peak_times[channel]['Clean'])
        noisy = np.array(peak_times[channel]['Noisy'])
        denoised = np.array(peak_times[channel]['Denoised'])
        snr_vals = np.array(snr_values[channel])
        
        # Calculating MSE between scatter points and x=y line
        errors_noisy = noisy - clean
        mse_noisy = np.mean(errors_noisy ** 2)
        
        errors_denoised = denoised - clean
        mse_denoised = np.mean(errors_denoised ** 2)
        
        plt.figure(figsize=(16,10))
        
        # Noisy vs Clean
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(clean, noisy, c=snr_vals, cmap='viridis', alpha=0.6)
        plt.plot([clean.min(), clean.max()], [clean.min(), clean.max()], 'r--', label=f'x=y (MSE: {mse_noisy:.2f})')
        cbar = plt.colorbar(scatter, label='SNR')
        cbar.ax.tick_params(labelsize = 18)
        plt.xlabel('Peak Time Bins of Clean Data', fontsize = 18 )
        plt.ylabel('Peak Time Bins of Noisy Data', fontsize = 18 )
        plt.title(f'Noisy vs Clean - {channel}', fontsize = 18)
        plt.legend(fontsize = 18)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.grid(True)
        y_limits = [min(noisy.min(), clean.min()), max(noisy.max(), clean.max())]
        plt.ylim(y_limits)

        # Denoised vs Clean
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(clean, denoised, c=snr_vals, cmap='viridis', alpha=0.6)
        plt.plot([clean.min(), clean.max()], [clean.min(), clean.max()], 'r--', label=f'x=y (MSE: {mse_denoised:.2f})')
        cbar = plt.colorbar(scatter, label='SNR')
        cbar.ax.tick_params(labelsize = 18)
        plt.xlabel('Peak Time of Clean Data (ns)', fontsize = 18 )
        plt.ylabel('Peak Time of Denoised Data (ns)', fontsize = 18 )
        plt.title(f'Denoised vs Clean - {channel}', fontsize =18)
        plt.legend(fontsize = 18)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.grid(True) 
        plt.ylim(y_limits)
        plt.tight_layout()
        # Save the figure for the current channel
        plt.savefig(os.path.join(save_folder, f'Peak_Time_{channel.replace(" ", "_")}_leftMSE={mse_noisy:.2f}_rightMSE={mse_denoised:.2f}.png'))
        plt.close()
    print('Peak time graphs have been saved individually for each channel.')
