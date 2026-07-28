"""
Hilbert filter analysis for peak time efficiency evaluation.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
from scipy.signal import hilbert

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.data_preprocessing import produce_noise_and_noiseless_data
from training.raytune_training_function import CustomDataset, split_indices
from visualization.overleaf_plots import plot_peak_time_efficiency_for_hilbert_filter

if __name__ == "__main__":
    ## Load Sim data
    Sim_data = "/sps/grand/blevy/sims/sims_for_denoising_sept2025" ### (3, n_noise_samples, 1024)
    noise_signals, clean_signals = produce_noise_and_noiseless_data(Sim_data)

    total_samples_train = clean_signals.shape[1]
    train_indices, valid_indices, test_indices = split_indices(total_samples_train, train_frac=0.8, valid_frac=0.1)
    # With swapping enabled (default 50% probability)

    train_dataset = CustomDataset(clean_signals, [noise_signals], indices=train_indices, 
                        swap_prob=0.0, no_random=True, target_start=120, target_end=480)

    valid_dataset = CustomDataset(clean_signals, [noise_signals], indices=valid_indices, 
                        swap_prob=0.0, no_random=True, target_start=120, target_end=480)

    test_dataset = CustomDataset(clean_signals, [noise_signals], indices=test_indices, 
                        swap_prob=0.0, no_random=True, target_start=120, target_end=480, voltage_to_adc=True)
    print(f'shape of test_indices {np.shape(test_indices)}')


    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device set to : {device}')

    ## Plot peak time efficiency for hilbert filter
    plot_peak_time_efficiency_for_hilbert_filter(test_loader, device, min_snr=1, max_snr=1e3, thresholds_list=[5, 10], save_path=str(ROOT_DIR))