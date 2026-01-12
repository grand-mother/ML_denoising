"""
Data preprocessing utilities for loading and preparing training data.
"""
import os
import numpy as np


def produce_noise_and_noiseless_data(sim_data_dir):
    """
    Load the original noiseless signal and original noise lst signals in microV from the sim_data_dir
    and return them
    """
    noise_signals = []
    for i in range(24):
        noise_data = np.load(f"{sim_data_dir}/noise_microV_lst{i}.npy", mmap_mode='r').transpose(1, 0, 2)
        if i == 0:
            print(f"Loaded noise file {i}: shape {noise_data.shape}")
        noise_signals.append(noise_data)
    total_samples = sum(arr.shape[1] for arr in noise_signals)
    noise_signals_concat = np.concatenate(noise_signals, axis=1)
    print(f"Noise signals shape: {np.shape(noise_signals_concat)}")

    clean_signals = np.load(f"{sim_data_dir}/noiseless_traces.npy", mmap_mode='r').transpose(1, 0, 2)
    print(f"Clean signals shape: {np.shape(clean_signals)}")

    return noise_signals_concat, clean_signals


