
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np 
import os 
from scipy.signal import hilbert

from .training_function import psnr
from .hilbert import peak_amplitude, peak_time, peak_time_and_amplitude


def test_func(testloader, 
         time, 
         frequency, 
         model, 
         num_images = 200, 
         device="cpu", 
         min_snr =3, 
         max_snr = 1e3, 
         save_folder ='', 
         fft_mode = False,
         voltage = True, 
         efield = False, 
         ADC = False):
    

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
            nb_channel = denoised_output.shape[1]
            sample_idx = 0  # Index of the sample to plot
            channel_names = ['X Channel', 'Y Channel', 'Z Channel']
            
            for channel_idx in range(nb_channel): 
                clean_np = clean_data[sample_idx, channel_idx].cpu().numpy()
                noisy_np = noisy_data[sample_idx, channel_idx].cpu().numpy()
                denoised_np = denoised_output[sample_idx, channel_idx].cpu().numpy()
                snr = np.max(clean_np) / np.std(noisy_np)
                if True:
                    plt.figure(figsize=(25, 16))  # Set figure size for each channel
                    
                    # Metrics calculations
                    mse_value = np.mean((clean_np - denoised_np) ** 2)
                    psnr_value = psnr(clean_np, denoised_np, np.max(clean_np))

                    # Plot the clean signal
                    plt.subplot(2, 1, 1)

                    if fft_mode:
                        plt.title(f'FFT of the Signal- {channel_names[channel_idx]}, SNR:{snr:.2f}',fontsize = 20)
                        plt.yscale('log')
                        plt.plot(frequency[0][:4096], clean_np, label=f'Pure - {channel_names[channel_idx]}', color='blue')
                        plt.plot(frequency[0][:4096], denoised_np, label=f'Denoised - MSE: {mse_value:.2f}, PSNR: {psnr_value:.2f}', linestyle='--', color='orange')
                        plt.xlabel('Frequency (MHz)',fontsize = 20)
                        plt.ylabel('Amplitude (mV/MHz)',fontsize = 20)
                        plt.xlim(0, 300)  # Only plot up to Nyquist frequency, converted to MHz
                        plt.legend(fontsize=20)
                        plt.xticks(fontsize=20)
                        plt.yticks(fontsize=20)
                    elif voltage:
                        plt.plot(time[0],clean_np, label=f'Pure - {channel_names[channel_idx]}', color='blue')
                        plt.plot(time[0],denoised_np, label=f'Denoised - MSE: {mse_value:.2f}, PSNR: {psnr_value:.2f}', linestyle='--', color='orange')
                        plt.xlabel(r'Time [ns]',fontsize = 24)
                        plt.ylabel(r'Voltage [$\mu$V]',fontsize = 24)
                        # plt.title(f'Denoised vs Pure Signal - {channel_names[channel_idx]}', fontsize=20)
                        plt.xlim(0, 300)
                        # plt.legend(fontsize=20)
                        plt.xticks(fontsize=24)
                        plt.yticks(fontsize=24)
                    elif ADC:
                        plt.plot(clean_np, label=f'Pure - {channel_names[channel_idx]}', color='blue')
                        plt.plot(denoised_np, label=f'Denoised - MSE: {mse_value:.2f}, PSNR: {psnr_value:.2f}', linestyle='--', color='orange')
                        plt.xlabel(f'Time Bin[ns]',fontsize = 24)
                        plt.ylabel(f'Counts',fontsize = 24)
                        plt.title(f'Pure Signal - {channel_names[channel_idx]}', fontsize=24)
                        # plt.xlim(200, 712)
                        plt.legend(fontsize=24)
                        plt.xticks(fontsize=24)
                        plt.yticks(fontsize=24)
                    elif efield:
                        plt.plot(time[0], clean_np, label=f'Pure - {channel_names[channel_idx]}', color='blue')
                        plt.plot(time[0], denoised_np, label=f'Denoised - MSE: {mse_value:.2f}, PSNR: {psnr_value:.2f}', linestyle='--', color='orange')
                        plt.xlabel('Time (ns)',fontsize = 24)
                        plt.ylabel('Efield (µV/m)',fontsize = 24)
                        plt.title(f'Denoised vs Pure Signal - {channel_names[channel_idx]}', fontsize=24)
                        plt.legend(fontsize=24)
                        plt.xticks(fontsize=24)
                        plt.yticks(fontsize=24)

                    else:
                        sys.exit('You must select either voltage, adc, or efeild')

                    # Plot the noisy signal
                    plt.subplot(2, 1, 2)


                    if fft_mode:
                        plt.plot(frequency[0][:4096], noisy_np, label=f'Noisy signal - {channel_names[channel_idx]}, SNR = {snr:.2f}', color='red')
                        plt.title(f'FFT of the Signal - {channel_names[channel_idx]}, SNR:{snr:.2f}' ,fontsize =20)
                        plt.yscale('log')
                        plt.xlabel('Frequency (MHz)',fontsize = 20)
                        plt.ylabel('Amplitude (mV/MHz)',fontsize = 20)
                        plt.xlim(0, 300)   # Only plot up to Nyquist frequency, converted to MHz
                        plt.legend(fontsize = 20)
                        plt.xticks(fontsize = 20)
                        plt.yticks(fontsize = 20)  

                    elif voltage:
                        plt.plot(noisy_np, label=f'Noisy signal - {channel_names[channel_idx]}, SNR = {snr:.2f}', color='red')
                        plt.xlabel(r'Time [ns]',fontsize = 24)
                        plt.ylabel(r'Voltage [$\mu$V]',fontsize = 24)
                        plt.xlim(0, 300) 
                        # plt.legend(fontsize = 20)
                        plt.xticks(fontsize = 24)
                        plt.yticks(fontsize = 24)    
                    
                    elif efield: 
                        plt.plot(time[0], noisy_np, label=f'Noisy signal - {channel_names[channel_idx]}, SNR = {snr:.2f}', color='red')
                        plt.xlabel('Time (ns)', fontsize = 24)
                        plt.ylabel('Efield (µV/m)', fontsize = 24)
                        plt.title(f'Noisy Signal - {channel_names[channel_idx]}, SNR:{snr:.2f}', fontsize = 24)
                        plt.legend(fontsize = 24)
                        plt.xticks(fontsize = 24)
                        plt.yticks(fontsize = 24)

                    elif ADC:
                        plt.plot(noisy_np, label=f'Noisy signal - {channel_names[channel_idx]}, SNR = {snr:.2f}', color='red')
                        plt.xlabel('Time Bin(ns)',fontsize = 24)
                        # plt.xlim(200, 712)
                        plt.ylabel('Counts',fontsize = 24)
                        plt.title(f'Noisy Signal - {channel_names[channel_idx]}, SNR:{snr:.2f}', fontsize=24)
                        plt.legend(fontsize=24)
                        plt.xticks(fontsize=24)
                        plt.yticks(fontsize=24)
                    else:
                        sys.exit('You must select either voltage, adc, or efeild')

                    # Save the figure
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_folder, f'sample_{count:03d}_channel_{channel_idx}_snr_{snr:.2f}.png'))
                    plt.close()
                    
            count += 1  # Increment the count

                 
            if count >= num_images:  # Stop after saving 100 images
                break
    print('test is completed') 


def test_func_v2(testloader, 
         time, 
         frequency, 
         model, 
         num_images = 200, 
         device="cpu", 
         min_snr =3, 
         max_snr = 1e3, 
         save_folder ='', 
         fft_mode = False,
         voltage = True, 
         efield = False, 
         ADC = False):
    

    path_plot_snr_low = os.path.join(save_folder, 'plots_snr_low')
    path_plot_snr_med = os.path.join(save_folder, 'plots_snr_med')
    path_plot_snr_med34 = os.path.join(save_folder, 'plots_snr_med34')
    path_plot_snr_high = os.path.join(save_folder, 'plots_snr_high')

    path_plot_list = [path_plot_snr_low, path_plot_snr_med, path_plot_snr_med34, path_plot_snr_high]

    os.makedirs(path_plot_snr_low, exist_ok=True)
    os.makedirs(path_plot_snr_med, exist_ok=True)
    os.makedirs(path_plot_snr_med34, exist_ok=True)
    os.makedirs(path_plot_snr_high, exist_ok=True)

    count = 0
    with torch.no_grad():  
        device = torch.device(device)
        model = model.to(device)
        model.eval() 
        count_low = 0  # To count the number of images saved
        count_med = 0
        count_med34 = 0
        count_high = 0

        for noisy_data, clean_data in testloader:
            
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            denoised_output = model(noisy_data)
            nb_channel = denoised_output.shape[1]
            sample_idx = 0  # Index of the sample to plot
            channel_names = ['X Channel', 'Y Channel', 'Z Channel']
            
            for channel_idx in range(nb_channel): 
                clean_np = clean_data[sample_idx, channel_idx].cpu().numpy()
                noisy_np = noisy_data[sample_idx, channel_idx].cpu().numpy()
                denoised_np = denoised_output[sample_idx, channel_idx].cpu().numpy()

                    # Calculate envelopes
                envelope_clean = np.abs(hilbert(clean_np))
                envelope_noisy = np.abs(hilbert(noisy_np))
                envelope_denoised = np.abs(hilbert(denoised_np))
                noisy_std = noisy_np[int((clean_np.size)/2):].std()
                
                if noisy_std != 0:
                    snr = np.max(envelope_clean) / noisy_std
                else:
                    snr = float('inf')
                    
                #snr = np.max(clean_np) / np.std(noisy_np)

                do_plots = False
                if (snr<=1) and count_low < 30 :
                    flag = 0
                    do_plots = True
                    count_low += 1

                elif (snr > 1)  * (snr <=3) and count_med < 30:
                    flag = 1
                    do_plots = True
                    count_med += 1

                elif (snr > 3)  * (snr <=5) and count_med34 < 30:
                    flag = 2
                    do_plots = True
                    count_med34 += 1

                elif snr > 5 and count_high < 30:
                    flag =3
                    do_plots = True
                    count_high += 1

                if do_plots:
                    plt.figure(figsize=(25, 16))  # Set figure size for each channel
                    
                    # Metrics calculations
                    mse_value = np.mean((clean_np - denoised_np) ** 2)
                    psnr_value = psnr(clean_np, denoised_np, np.max(clean_np))

                    # Plot the clean signal
                    plt.subplot(2, 1, 1)

                    if fft_mode:
                        plt.title(f'FFT of the Signal- {channel_names[channel_idx]}, SNR:{snr:.2f}',fontsize = 20)
                        plt.yscale('log')
                        plt.plot(frequency[0][:4096], clean_np, label=f'Pure - {channel_names[channel_idx]}', color='blue')
                        plt.plot(frequency[0][:4096], denoised_np, label=f'Denoised - MSE: {mse_value:.2f}, PSNR: {psnr_value:.2f}', linestyle='--', color='orange')
                        plt.xlabel('Frequency (MHz)',fontsize = 20)
                        plt.ylabel('Amplitude (mV/MHz)',fontsize = 20)
                        plt.xlim(0, 300)  # Only plot up to Nyquist frequency, converted to MHz
                        plt.legend(fontsize=20)
                        plt.xticks(fontsize=20)
                        plt.yticks(fontsize=20)
                    elif voltage:
                        plt.plot(time[0],clean_np, label=f'Pure - {channel_names[channel_idx]}', color='blue')
                        plt.plot(time[0],denoised_np, label=f'Denoised - MSE: {mse_value:.2f}, PSNR: {psnr_value:.2f}', linestyle='--', color='orange')
                        plt.xlabel(r'Time [ns]',fontsize = 24)
                        plt.ylabel(r'Voltage [$\mu$V]',fontsize = 24)
                        # plt.title(f'Denoised vs Pure Signal - {channel_names[channel_idx]}', fontsize=20)
                        plt.xlim(0, 300)
                        # plt.legend(fontsize=20)
                        plt.xticks(fontsize=24)
                        plt.yticks(fontsize=24)
                    elif ADC:
                        plt.plot(clean_np, label=f'Pure - {channel_names[channel_idx]}', color='blue')
                        plt.plot(denoised_np, label=f'Denoised - MSE: {mse_value:.2f}, PSNR: {psnr_value:.2f}', linestyle='--', color='orange')
                        plt.xlabel(f'Time Bin[ns]',fontsize = 24)
                        plt.ylabel(f'Counts',fontsize = 24)
                        plt.title(f'Pure Signal - {channel_names[channel_idx]}', fontsize=24)
                        # plt.xlim(200, 712)
                        plt.legend(fontsize=24)
                        plt.xticks(fontsize=24)
                        plt.yticks(fontsize=24)
                    elif efield:
                        plt.plot(time[0], clean_np, label=f'Pure - {channel_names[channel_idx]}', color='blue')
                        plt.plot(time[0], denoised_np, label=f'Denoised - MSE: {mse_value:.2f}, PSNR: {psnr_value:.2f}', linestyle='--', color='orange')
                        plt.xlabel('Time (ns)',fontsize = 24)
                        plt.ylabel('Efield (µV/m)',fontsize = 24)
                        plt.title(f'Denoised vs Pure Signal - {channel_names[channel_idx]}', fontsize=24)
                        plt.legend(fontsize=24)
                        plt.xticks(fontsize=24)
                        plt.yticks(fontsize=24)

                    else:
                        sys.exit('You must select either voltage, adc, or efeild')

                    # Plot the noisy signal
                    plt.subplot(2, 1, 2)


                    if fft_mode:
                        plt.plot(frequency[0][:4096], noisy_np, label=f'Noisy signal - {channel_names[channel_idx]}, SNR = {snr:.2f}', color='red')
                        plt.title(f'FFT of the Signal - {channel_names[channel_idx]}, SNR:{snr:.2f}' ,fontsize =20)
                        plt.yscale('log')
                        plt.xlabel('Frequency (MHz)',fontsize = 20)
                        plt.ylabel('Amplitude (mV/MHz)',fontsize = 20)
                        plt.xlim(0, 300)   # Only plot up to Nyquist frequency, converted to MHz
                        plt.legend(fontsize = 20)
                        plt.xticks(fontsize = 20)
                        plt.yticks(fontsize = 20)  

                    elif voltage:
                        plt.plot(noisy_np, label=f'Noisy signal - {channel_names[channel_idx]}, SNR = {snr:.2f}', color='red')
                        plt.xlabel(r'Time [ns]',fontsize = 24)
                        plt.ylabel(r'Voltage [$\mu$V]',fontsize = 24)
                        plt.xlim(0, 300) 
                        # plt.legend(fontsize = 20)
                        plt.xticks(fontsize = 24)
                        plt.yticks(fontsize = 24)    
                    
                    elif efield: 
                        plt.plot(time[0], noisy_np, label=f'Noisy signal - {channel_names[channel_idx]}, SNR = {snr:.2f}', color='red')
                        plt.xlabel('Time (ns)', fontsize = 24)
                        plt.ylabel('Efield (µV/m)', fontsize = 24)
                        plt.title(f'Noisy Signal - {channel_names[channel_idx]}, SNR:{snr:.2f}', fontsize = 24)
                        plt.legend(fontsize = 24)
                        plt.xticks(fontsize = 24)
                        plt.yticks(fontsize = 24)

                    elif ADC:
                        plt.plot(noisy_np, label=f'Noisy signal - {channel_names[channel_idx]}, SNR = {snr:.2f}', color='red')
                        plt.xlabel('Time Bin(ns)',fontsize = 24)
                        # plt.xlim(200, 712)
                        plt.ylabel('Counts',fontsize = 24)
                        plt.title(f'Noisy Signal - {channel_names[channel_idx]}, SNR:{snr:.2f}', fontsize=24)
                        plt.legend(fontsize=24)
                        plt.xticks(fontsize=24)
                        plt.yticks(fontsize=24)
                    else:
                        sys.exit('You must select either voltage, adc, or efeild')

                    # Save the figure
                    plt.tight_layout()

                    
                    plot_path = path_plot_list[flag]
                    
                    plt.savefig(os.path.join(plot_path, f'sample_{count:03d}_channel_{channel_idx}_snr_{snr:.2f}.png'))
                    plt.close()
                    np.save(os.path.join(plot_path, f'clean_{count:03d}_channel_{channel_idx}_snr_{snr:.2f}.npy'), clean_np)
                    np.save(os.path.join(plot_path, f'noisy_{count:03d}_channel_{channel_idx}_snr_{snr:.2f}.npy'), noisy_np)
                    np.save(os.path.join(plot_path, f'denoised_{count:03d}_channel_{channel_idx}_snr_{snr:.2f}.npy'), denoised_np)
                    
                do_plots = False

            count += 1  # Increment the count

            if (count_low + count_med + count_med34 + count_high) >= 120:  # Stop after saving 100 images
                break

    print('test is completed') 



def test_func_v3(testloader, 
         time, 
         frequency, 
         model, 
         num_images = 200, 
         device="cpu", 
         min_snr =3, 
         max_snr = 1e3, 
         save_folder ='', 
         fft_mode = False,
         voltage = True, 
         efield = False, 
         ADC = False):
    

    clean_ = []
    noisy_ = []
    denoised_ = []

    count = 0
    with torch.no_grad():  
        device = torch.device(device)
        model = model.to(device)
        model.eval() 
        
        for noisy_data, clean_data in testloader:
            
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            denoised_output = model(noisy_data)
            nb_channel = denoised_output.shape[1]
            sample_idx = 0  # Index of the sample to plot
            channel_names = ['X Channel', 'Y Channel', 'Z Channel']
            clean_.append(clean_data.cpu().numpy())
            noisy_.append(noisy_data.cpu().numpy())
            denoised_.append(denoised_output.cpu().numpy())

    clean_ = np.array(clean_)
    noisy_ = np.array(noisy_)
    denoised_ = np.array(denoised_)

    np.save(os.path.join(save_folder, 'clean_all.npy'), clean_)
    np.save(os.path.join(save_folder, 'noisy_all.npy'), noisy_)
    np.save(os.path.join(save_folder, 'denoised_all.npy'), denoised_)

    print('test is completed') 
