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
#from sim2root.Common.IllustrateSimPipe import *
#import grand.dataio.root_trees as groot

from ml_denoising_lib.utils.fft import psd, bandwidth_filter

from DU_response_computation import apply_rfchain as rfc






def traces(directory, 
           NJ_directory, 
           nb_events=1000, 
           mpe=1e9, 
           min_zenith=85, 
           max_zenith=88, 
           fft_mode = False, 
           band_filter = False, 
           band_filter_adc= False, 
           band_filter_efield = False,  
           voltage= False, 
           ADC = False, 
           efield= False ):
    """
    directory: path for the data file
    nb_event: number of events want to select
    min_primary_energy: minimum energy to filter out the data
    min_zenith, max_zenith: minimum and maximum zenith of the data in the data filter
    """
    #plot=False, 
    #random_mode = False, 
    freq = []
    time = [] 
    x_train_dataset = []
    y_train_dataset = []
    z_train_dataset = []
    x_test_dataset = []
    y_test_dataset = []
    z_test_dataset = []
    d_input, d_NJ_input = groot.DataDirectory(directory), groot.DataDirectory(NJ_directory)
    
    tvoltage_l0, tvoltage_l0_NJ = d_input.tvoltage_l0 , d_NJ_input.tvoltage_l0
    # tvoltage_l1, tvoltage_l1_NJ = d_input.tvoltage_l1 , d_NJ_input.tvoltage_l1
    tshower_l0, tshower_l0_NJ = d_input.tshower_l0 , d_NJ_input.tshower_l0
    trunefieldsim_l0, trunefieldsim_l0_NJ = d_input.trunefieldsim_l0, d_NJ_input.trunefieldsim_l0
    tefield_l0, tefield_l0_NJ = d_input.tefield_l0 , d_NJ_input.tefield_l0
    tefield_l1, tefield_l1_NJ = d_input.tefield_l1 , d_NJ_input.tefield_l1
    trun_l0, trun_l0_NJ = d_input.trun_l0, d_NJ_input.trun_l0
    tadc_l1, tadc_l1_NJ = d_input.tadc_l1,  d_NJ_input.tadc_l1
    # Get the list of events
    events_list, NJ_events_list = tvoltage_l0.get_list_of_events(), tvoltage_l0_NJ.get_list_of_events()
    
    nb_events, nb_events_NJ = len(events_list),len(NJ_events_list) 
    
    print(f'Number of events: {nb_events}. Number of NJ events: {nb_events_NJ}') 
    
    # If there are no events in the file, exit
    if nb_events == 0:
        sys.exit("There are no events in the file! Exiting.")
        
    event_counter = 0
    max_events_to_store = nb_events
    previous_run = None    
    
    for event_number, run_number in events_list:
        assert isinstance(event_number, int)
        assert isinstance(run_number, int)
        
        if event_counter < max_events_to_store:
            tshower_l0.get_event(event_number, run_number)
            tshower_l0_NJ.get_event(event_number, run_number)

            zenith = tshower_l0.zenith

            energy_primary = tshower_l0.energy_primary
            # Filter events based on zenith angle
            if energy_primary > mpe:
                if min_zenith <= zenith <= max_zenith:
                    # tvoltage_l1.get_event(event_number, run_number)
                    tvoltage_l0.get_event(event_number, run_number)
                    tefield_l0.get_event(event_number, run_number)
                    tefield_l1.get_event(event_number, run_number)
                    tadc_l1.get_event(event_number, run_number)


                    tvoltage_l0_NJ.get_event(event_number, run_number)
                    tefield_l0_NJ.get_event(event_number, run_number)
                    tefield_l1_NJ.get_event(event_number, run_number)
                    tadc_l1_NJ.get_event(event_number, run_number)


                    if previous_run != run_number:                          # Load only for new run.
                        trun_l0.get_run(run_number)                         # Update run info to get site latitude and longitude.       
                        trunefieldsim_l0.get_run(run_number)  
                        trun_l0_NJ.get_run(run_number)
                        trunefieldsim_l0_NJ.get_run(run_number)     
                        previous_run = run_number

                    du_id = np.asarray(tefield_l0.du_id) # Used for printing info and saving in voltage tree.

                    # t0 calculations
                    event_second = tshower_l0.core_time_s
                    event_nano = tshower_l0.core_time_ns

                    event_second_NJ = tshower_l0_NJ.core_time_s
                    event_nano_NJ = tshower_l0_NJ.core_time_ns

                    if voltage:
                        trace_voltage = np.asarray(tvoltage_l0.trace, dtype = np.float32)
                        trace_voltage_NJ = np.asarray(tvoltage_l0_NJ.trace, dtype= np.float32)
                        trace_shape = trace_voltage.shape
                        trace_shape_NJ = trace_voltage_NJ.shape
                        t0_voltage_L1 = (tvoltage_l0.du_seconds-event_second)*1e9 - event_nano + tvoltage_l0.du_nanoseconds 
                        t0_voltage_L1_NJ = (tvoltage_l0_NJ.du_seconds-event_second_NJ)*1e9 - event_nano_NJ + tvoltage_l0_NJ.du_nanoseconds
                        nb_du = trace_shape[0]
                        nb_du_NJ = trace_shape_NJ[0]
                        sig_size = trace_shape[-1]
                        sig_size_Nj = trace_shape[-1]

                    elif ADC:
                        trace_adc = np.asarray(tadc_l1.trace_ch, dtype = np.float32) 
                        trace_adc_NJ = np.asarray(tadc_l1_NJ.trace_ch, dtype= np.float32)
                        trace_shape = trace_adc.shape
                        trace_shape_NJ = trace_adc_NJ.shape
                        t0_adc_L1 = (tadc_l1.du_seconds-event_second)*1e9  - event_nano + tadc_l1.du_nanoseconds
                        t0_adc_L1_NJ = (tadc_l1_NJ.du_seconds-event_second_NJ)*1e9 - event_nano_NJ + tadc_l1.du_nanoseconds
                        nb_du = trace_shape[0]
                        nb_du_NJ = trace_shape_NJ[0]
                        sig_size = trace_shape[-1]
                        sig_size_Nj = trace_shape[-1]
                    
                    elif efield:
                        trace_efield = np.asarray(tefield_l1.trace, dtype = np.float32) #### modify here if you want ADC_l1
                        trace_efield_NJ = np.asarray(tefield_l1_NJ.trace, dtype = np.float32)
                        trace_shape = trace_efield.shape
                        trace_shape_NJ = trace_efield_NJ.shape
                        t0_efield_L1 = (tefield_l1.du_seconds-event_second)*1e9  - event_nano + tefield_l1.du_nanoseconds
                        t0_efield_L1_NJ = (tefield_l1_NJ.du_seconds-event_second_NJ)*1e9 - event_nano_NJ + tefield_l1.du_nanoseconds
                        nb_du = trace_shape[0]
                        nb_du_NJ = trace_shape_NJ[0]
                        sig_size = trace_shape[-1]
                        sig_size_Nj = trace_shape[-1]
                    else:
                        sys.exit("You must select either voltage, adc, or efield")

                    event_counter += 1


                    t_pre_L0 = trunefieldsim_l0.t_pre
                    t_pre_L0_NJ = trunefieldsim_l0_NJ.t_pre

                    event_dus_indices = tefield_l0.get_dus_indices_in_run(trun_l0)

                    event_dus_indices_NJ = tefield_l0_NJ.get_dus_indices_in_run(trun_l0_NJ)


                    dt_ns_l0 = np.asarray(trun_l0.t_bin_size)[event_dus_indices]

                    dt_ns_l0_NJ = np.asarray(trun_l0_NJ.t_bin_size)[event_dus_indices_NJ]
                    
                    # if random_mode:
                    #         xmin = np.random.randint(0,1024)
                    #         xmax = xmin + 1024
                    #         ymin = np.random.randint(0,1024)
                    #         ymax = ymin + 1024
                    #         zmin = np.random.randint(0,1024)
                    #         zmax = zmin + 1024                            

                    for du_idx in range(nb_du):
                        if voltage:
                            trace_voltage_x = trace_voltage[du_idx, 0]
                            trace_voltage_y = trace_voltage[du_idx, 1]
                            trace_voltage_z = trace_voltage[du_idx, 2]

                            trace_voltage_x_NJ = trace_voltage_NJ[du_idx, 0]
                            trace_voltage_y_NJ = trace_voltage_NJ[du_idx, 1]
                            trace_voltage_z_NJ = trace_voltage_NJ[du_idx, 2]

                            trace_voltage_time = np.arange(0, len(trace_voltage_z)) * dt_ns_l0[du_idx] - t_pre_L0
                            if band_filter:
                                ifft_clean_x, ifft_noisy_x = bandwidth_filter(trace_voltage_time, trace_voltage_x_NJ, trace_voltage_x)
                                ifft_clean_y, ifft_noisy_y = bandwidth_filter(trace_voltage_time, trace_voltage_y_NJ, trace_voltage_y)
                                ifft_clean_z, ifft_noisy_z = bandwidth_filter(trace_voltage_time, trace_voltage_z_NJ, trace_voltage_z)
                                x_train_dataset.append(ifft_noisy_x)
                                y_train_dataset.append(ifft_noisy_y)
                                z_train_dataset.append(ifft_noisy_z)
                                x_test_dataset.append(ifft_clean_x)
                                y_test_dataset.append(ifft_clean_y)
                                z_test_dataset.append(ifft_clean_z)
                                time.append(trace_voltage_time) 
                            else:
                                x_train_dataset.append(trace_voltage_x)
                                y_train_dataset.append(trace_voltage_y)
                                z_train_dataset.append(trace_voltage_z)
                                x_test_dataset.append(trace_voltage_x_NJ)
                                y_test_dataset.append(trace_voltage_y_NJ)
                                z_test_dataset.append(trace_voltage_z_NJ)
                                time.append(trace_voltage_time) 

                        elif ADC:
                            trace_adc_x = trace_adc[du_idx, 0]
                            trace_adc_y = trace_adc[du_idx, 1]
                            trace_adc_z = trace_adc[du_idx, 2]
                            trace_adc_x_NJ = trace_adc_NJ[du_idx, 0]
                            trace_adc_y_NJ = trace_adc_NJ[du_idx, 1]
                            trace_adc_z_NJ = trace_adc_NJ[du_idx, 2]

                            # -------------------------------
                            # Slice-swap a fixed clean segment (indices 300-500 ns)
                            # with a random noised segment.
                            # That is, the clean signal region in the NJ trace [300:500]
                            # will be replaced by a randomly selected segment from the noisy ADC trace,
                            # and vice versa.
                            # -------------------------------
                            target_start, target_end = 300, 500  # fixed indices for the clean segment to swap
                            target_length = target_end - target_start  # e.g. 200 ns
                            # Choose a random noised segment (you can adjust the domain of selection as needed)
                            if np.random.rand() < 0.5:
                                # Choose from before the target (if there’s enough room)
                                source_start = np.random.randint(0, target_start - target_length + 1)
                            else:
                                # Choose from after the target region
                                source_start = np.random.randint(target_end, sig_size - target_length + 1)
                            source_end = source_start + target_length

                            # Create copies to manipulate without altering the originals
                            spliced_x = trace_adc_x.copy()
                            spliced_y = trace_adc_y.copy()
                            spliced_z = trace_adc_z.copy()
                            spliced_x_NJ = trace_adc_x_NJ.copy()
                            spliced_y_NJ = trace_adc_y_NJ.copy()
                            spliced_z_NJ = trace_adc_z_NJ.copy()

                            # Perform the swap:
                            # Save the original clean segment from the NJ (clean) trace
                            temp_clean_x_NJ = spliced_x_NJ[target_start:target_end].copy()
                            temp_clean_y_NJ = spliced_y_NJ[target_start:target_end].copy()
                            temp_clean_z_NJ = spliced_z_NJ[target_start:target_end].copy()
                            temp_clean_x = spliced_x[target_start:target_end].copy()
                            temp_clean_y = spliced_y[target_start:target_end].copy()
                            temp_clean_z = spliced_z[target_start:target_end].copy()


                            # Replace the clean region [300:500] in the clean trace with the noisy segment
                            spliced_x_NJ[target_start:target_end] = spliced_x_NJ[source_start:source_end]
                            spliced_y_NJ[target_start:target_end] = spliced_y_NJ[source_start:source_end]
                            spliced_z_NJ[target_start:target_end] = spliced_z_NJ[source_start:source_end]

                            spliced_x[target_start:target_end] = spliced_x[source_start:source_end]
                            spliced_y[target_start:target_end] = spliced_y[source_start:source_end]
                            spliced_z[target_start:target_end] = spliced_z[source_start:source_end]
                            # Put the clean segment into the noisy trace at the random location

                            spliced_x[source_start:source_end] = temp_clean_x
                            spliced_y[source_start:source_end] = temp_clean_y
                            spliced_z[source_start:source_end] = temp_clean_z

                            spliced_x_NJ[source_start:source_end] = temp_clean_x_NJ
                            spliced_y_NJ[source_start:source_end] = temp_clean_y_NJ
                            spliced_z_NJ[source_start:source_end] = temp_clean_z_NJ

                            trace_adc_time = np.arange(0, len(trace_adc_z)) * dt_ns_l0[du_idx] - t_pre_L0
                            if not (np.all(spliced_x == 0) and np.all(spliced_y == 0) and np.all(spliced_z == 0) and
                                    np.all(spliced_x_NJ == 0) and np.all(spliced_y_NJ == 0) and np.all(spliced_z_NJ == 0)):
                                if band_filter_adc:
                                    ifft_clean_x, ifft_noisy_x = bandwidth_filter(trace_adc_time , spliced_x_NJ, spliced_x)
                                    ifft_clean_y, ifft_noisy_y = bandwidth_filter(trace_adc_time , spliced_y_NJ, spliced_y)
                                    ifft_clean_z, ifft_noisy_z = bandwidth_filter(trace_adc_time , spliced_z_NJ, spliced_z)
                                    x_train_dataset.append(ifft_noisy_x)
                                    y_train_dataset.append(ifft_noisy_y)
                                    z_train_dataset.append(ifft_noisy_z)
                                    x_test_dataset.append(ifft_clean_x)
                                    y_test_dataset.append(ifft_clean_y)
                                    z_test_dataset.append(ifft_clean_z)
                                    time.append(trace_adc_time) 
                                else:
                                    x_train_dataset.append(spliced_x)
                                    y_train_dataset.append(spliced_y)
                                    z_train_dataset.append(spliced_z)
                                    x_test_dataset.append(spliced_x_NJ)
                                    y_test_dataset.append(spliced_y_NJ)
                                    z_test_dataset.append(spliced_z_NJ)
                                    time.append(trace_adc_time) 

                        elif efield:
                            trace_efield_x = trace_efield[du_idx, 0]
                            trace_efield_y = trace_efield[du_idx, 1]
                            trace_efield_z = trace_efield[du_idx, 2]
                            trace_efield_x_NJ = trace_efield_NJ[du_idx, 0]
                            trace_efield_y_NJ = trace_efield_NJ[du_idx, 1]
                            trace_efield_z_NJ = trace_efield_NJ[du_idx, 2]
                            trace_efield_time = np.arange(0, len(trace_efield_z)) * dt_ns_l0[du_idx] - t_pre_L0
                            if band_filter_efield:
                                ifft_clean_x, ifft_noisy_x = bandwidth_filter(trace_efield_time , trace_efield_x_NJ, trace_efield_x)
                                ifft_clean_y, ifft_noisy_y = bandwidth_filter(trace_efield_time , trace_efield_y_NJ, trace_efield_y)
                                ifft_clean_z, ifft_noisy_z = bandwidth_filter(trace_efield_time , trace_efield_z_NJ, trace_efield_z)
                                x_train_dataset.append(ifft_noisy_x)
                                y_train_dataset.append(ifft_noisy_y)
                                z_train_dataset.append(ifft_noisy_z)
                                x_test_dataset.append(ifft_clean_x)
                                y_test_dataset.append(ifft_clean_y)
                                z_test_dataset.append(ifft_clean_z)
                                time.append(trace_efield_time) 
                            else:
                                x_train_dataset.append(trace_efield_x)
                                y_train_dataset.append(trace_efield_y)
                                z_train_dataset.append(trace_efield_z)
                                x_test_dataset.append(trace_efield_x_NJ)
                                y_test_dataset.append(trace_efield_y_NJ)
                                z_test_dataset.append(trace_efield_z_NJ)
                                time.append(trace_efield_time) 

                        else:
                            sys.exit("One of the traces must be selected. Exit")
            
                        ### default is False
                        if fft_mode:
                            frequency, fft_amplitude_clean_x, fft_amplitude_noisy_x = psd(trace_voltage_time, trace_voltage_x_NJ, trace_voltage_x)
                            frequency, fft_amplitude_clean_y, fft_amplitude_noisy_y = psd(trace_voltage_time, trace_voltage_y_NJ, trace_voltage_y)
                            frequency, fft_amplitude_clean_z, fft_amplitude_noisy_z = psd(trace_voltage_time, trace_voltage_z_NJ, trace_voltage_z)
                            x_train_dataset.append(fft_amplitude_noisy_x[1:])
                            y_train_dataset.append(fft_amplitude_noisy_y[1:])
                            z_train_dataset.append(fft_amplitude_noisy_z[1:])
                            x_test_dataset.append(fft_amplitude_clean_x[1:])
                            y_test_dataset.append(fft_amplitude_clean_y[1:])
                            z_test_dataset.append(fft_amplitude_clean_z[1:])
                            freq.append(frequency) 
                        
                        
                        # if plot:
                        #     fig, axs = plt.subplots(1, 1, figsize=(8, 6))
                        #     axs.plot(trace_voltage_time, trace_voltage_x, alpha=0.5, label="polarization N")
                        #     axs.plot(trace_voltage_time, trace_voltage_y, alpha=0.5, label="polarization E")
                        #     axs.plot(trace_voltage_time, trace_voltage_z, alpha=0.5, label="polarization V")
                        #     axs.legend()
                        #     axs.set_title(f"Voltage antenna {du_idx}")
                        #     axs.set_xlabel("Time in ns")
                        #     axs.set_ylabel("Voltage in uV")
                        #     plt.show()
                        #     plt.close(fig)        

        else:
            break
    
    print("Processing complete for specified number of events!")
    return time, freq, x_train_dataset, x_test_dataset, y_train_dataset, y_test_dataset, z_train_dataset, z_test_dataset #time,frequency,


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

def multi_domain_loss(input, target, device='cpu', fft_weight=0.2):

    # time-domain psnr + log-magnitude STFT psnr loss (frequency domain)
    psnr_time = psnr_loss(input, target, device= 'cpu')

    input_fft = torch.fft.rfft(input, dim= -1)
    target_fft  = torch.fft.rfft(target , dim= -1)
    psnr_freq = psnr_loss(torch.log1p(torch.abs(input_fft)),
                    torch.log1p(torch.abs(target_fft)), device=device)
    return psnr_time + fft_weight * psnr_freq

def plot_metrics(epochs, training_losses, validation_losses, validation_psnr, learning_rates, validation_peak_to_peak, save_folder):
    """
    Plot four metrics versus epochs and save the figures to a specified folder.

    Training Loss and validation loss versus epochs
    Validation PSNR versus epochs
    Peak to Peak ratio versus epochs
    Learning rate versus epochs
    
    save_folder for saving the metrics into the folder, string: name of the file
    """

    plt.figure(figsize=(16, 9))

    # Plotting Training and Validation Loss
    plt.subplot(4, 1, 1)
    plt.plot(epochs, training_losses, label='Training loss')
    plt.plot(epochs, validation_losses, label='Validation loss', color='orange')
    plt.title('Training and Validation loss vs Epochs', fontsize = 20)
    plt.xlabel('Epochs', fontsize = 20)
    plt.ylabel('Loss', fontsize = 20)
    plt.yscale('log')
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(fontsize = 20)
    
    # Plotting Validation PSNR
    plt.subplot(4, 1, 2)
    plt.plot(epochs, validation_psnr, label='Validation PSNR', color='green')
    plt.title('PSNR vs Epochs', fontsize = 20)
    plt.xlabel('Epochs', fontsize = 20)
    plt.ylabel('PSNR (dB)', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(fontsize = 20)
    
    # Plotting Learning Rate
    plt.subplot(4, 1, 3)
    plt.plot(epochs, learning_rates, label='Learning Rate', color='cyan')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate vs Epochs')
    plt.legend()

    # Plotting Peak-to-Peak Amplitude
    plt.subplot(4, 1, 4)
    plt.plot(epochs, validation_peak_to_peak, label='Validation Peak-to-Peak', color='magenta')
    plt.xlabel('Epochs')
    plt.ylabel('Peak-to-Peak Amplitude')
    plt.title('Peak-to-Peak Amplitude vs Epochs')
    plt.legend()
    

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'metrics.png'))
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