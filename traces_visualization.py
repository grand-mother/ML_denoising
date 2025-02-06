#MODULES
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import torch


def create_trace(ts, tssig, omega, sigma):
    """Generate a trace using the sine and exponential functions."""
    return np.sin(2. * np.pi * (ts - tssig) * omega) * np.exp(-(ts - tssig)**2 / sigma)


def save_trace_to_csv(ts, trace, idat, directory):
    """Save the generated trace data to a CSV file."""
    df = pd.DataFrame({'Time': ts, 'Trace': trace})
    filename = os.path.join(directory, f'trace_data_{idat}.csv')
    df.to_csv(filename, index=False)
    print(f'Data saved to {filename}')


def generate_dataset(ts, ndat, directory, save_csv):
    """Generate a dataset of traces along with their parameters."""
    dataset = []
    parameters = []
    for idat in range(ndat):
        tssig = random.uniform(512 - 100, 512 + 100)
        omega = random.uniform(0.01, 0.1)
        sigma = random.uniform(200, 800)
        trace = create_trace(ts, tssig, omega, sigma)
        dataset.append([trace])
        parameters.append((tssig, omega, sigma))
        if save_csv:
            save_trace_to_csv(ts, trace, idat, directory)
    return torch.tensor(np.array(dataset), dtype=torch.float32), parameters


def visualize_dataset(ts, dataset_tensor, parameters, ndat):
    """Visualize the generated dataset using matplotlib."""
    for i in range(ndat):
        fig, axs = plt.subplots(figsize=(16, 4))
        trace = np.array(dataset_tensor)[i, 0, :]
        label = f'tssig={parameters[i][0]:.2f}, omega={parameters[i][1]:.5f}, sigma={parameters[i][2]:.2f}'
        axs.plot(ts, trace, color='r', label=label)
        axs.legend(loc='upper right')
        axs.set_title('Trace')
        axs.set(xlabel='time', ylabel='Amplitude')
        plt.show()


def create_dataloaders(train_tensor, valid_tensor, test_tensor, batch_size):
    """Create data loaders for training, validation, and testing."""
    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_tensor, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_tensor, batch_size=1, shuffle=True)
    return train_loader, valid_loader, test_loader


def prepare_denoise_data(ndat_train=1024, ndat_valid=128, ndat_test=5, save_csv=False,
                         directory=os.getcwd(), batch_size=1, plot=False):
    """Prepare the data for denoising."""
    nts = 1024
    ts = np.arange(0, nts, 1)
    train_ds_tensor, train_params = generate_dataset(ts, ndat_train, directory, save_csv)
    valid_ds_tensor, valid_params = generate_dataset(ts, ndat_valid, directory, save_csv)
    test_ds_tensor, test_params = generate_dataset(ts, ndat_test, directory, save_csv)
    if plot:
        visualize_dataset(ts, test_ds_tensor, test_params, ndat_test)
    return create_dataloaders(train_ds_tensor, valid_ds_tensor, test_ds_tensor, batch_size)


