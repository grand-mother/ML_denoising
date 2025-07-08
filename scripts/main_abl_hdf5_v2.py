import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import shutil
from ml_denoising_lib.modules.CNNModel import DualBranchAutoencoder

from ml_denoising_lib.utils import training_function as tr_fu
#from CNNModel import DualBranchAutoencoder

#import training_function as tr_fu

from ml_denoising_lib.utils import train as train_functions
from ml_denoising_lib.utils import test as test
from ml_denoising_lib.utils.hilbert import peak_time_and_amplitude

import json
from DU_response_computation import apply_rfchain as rfc


#def main(tag, which_noise, loss, n_epochs, batch_size, base_lr, max_lr, step_size_up, lst, model_to_load, noise_param_file):
def main(all_params):

    output_path = all_params['output_path']
    print(os.path.basename(json_params_file))

    tag = all_params['tag']

    which_noise = all_params['which_noise']

    n_epochs = all_params['n_epochs']
    batch_size = 1024

    loss = all_params['loss']

    base_lr = all_params['base_lr']
    max_lr = all_params['max_lr']

    step_size_up = all_params['step_size_up']
    lr_mode = all_params['lr_mode']
    lst = all_params['lst']
    model_to_load = all_params['model_to_load']

    xy_mode  = all_params['xy_mode']


    ## Define the noise and rf chain.
    with open(noise_param_file, 'r') as f:
        params_RF2 = json.load(f)


    ### loading RF and noise params
    a2 = rfc.load_parameters_and_compute_stuff(params_RF2)

    l_eff2 = a2[0]
    tf2 = a2[1]
    latitude2 = a2[2]
    out_freqs2 = a2[3]
    LFmap_path = params_RF2['LFmap_path']

    noise_computer2 = rfc.compute_noise(1, latitude2,
                                [f"{LFmap_path}LFmapshort{i}.npy" for i in range(20, 251)],
                                np.arange(20, 251)*1e6,
                                out_freqs2,
                                tf2,
                                duration=params_RF2['duration'], leff_x=l_eff2[0], leff_y=l_eff2[1], leff_z=l_eff2[2])

    noise_computer2.P_nu
    noise_computer2.noise_rms_traces()

    save_folder = os.path.join(output_path, 'results_dev_{}_{}_{}_{}_hdf5'.format(tag, which_noise, n_epochs, loss))
    os.makedirs(save_folder, exist_ok=True)


    real_an_train_file = all_params["real_an_train_file"]
    real_an_valid_file = all_params["real_an_valid_file"]
    real_an_test_file = all_params["real_an_test_file"]

    save_path_clean_sims_train = all_params["save_path_clean_sims_train"]
    save_path_clean_sims_validation = all_params["save_path_clean_sims_validation"]
    save_path_clean_sims_test = all_params["save_path_clean_sims_test"]

    clean_signals_train = np.load(save_path_clean_sims_train).transpose(1, 0, 2)
    total_samples_train = np.shape(clean_signals_train)[1]
    step_size = int(n_epochs * total_samples_train / batch_size/10)
    print(f'total_sample_train = {total_samples_train}')

    clean_signals_validation = np.load(save_path_clean_sims_validation).transpose(1, 0, 2)
    clean_signals_test = np.load(save_path_clean_sims_test).transpose(1, 0, 2)

    train_dataset = tr_fu.CustomDataset_hdf5(clean_signals_train, noise_computer2, traces_len=512, lst=lst, no_random=False, xy_mode=xy_mode, which_noise=which_noise, real_an_noisefile=real_an_train_file)
    valid_dataset = tr_fu.CustomDataset_hdf5(clean_signals_validation, noise_computer2, traces_len=512, lst=lst, no_random=False, xy_mode=xy_mode, which_noise=which_noise, real_an_noisefile=real_an_valid_file)
    # valid_dataset = tr_fu.CustomDataset_hdf5(clean_signals_validation, noise_computer2, traces_len=1024, lst=lst, no_random=True, xy_mode=xy_mode)
    test_dataset = tr_fu.CustomDataset_hdf5(clean_signals_test, noise_computer2, no_random=True, lst=lst, xy_mode=xy_mode, which_noise=which_noise, real_an_noisefile=real_an_test_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device set to : {device}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=True)
    train_loader_for_test = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True, pin_memory=True)

    model_mse = DualBranchAutoencoder(xy_mode=xy_mode, time_only=False).to(device)

    if os.path.isfile(model_to_load):
        print('existing model found, loading those weights')
        model_mse.load_state_dict(torch.load(model_to_load))
    else:
        print('model not found, starting from random weights')

    optimizer = optim.Adam(model_mse.parameters(), lr=base_lr, weight_decay=0.0005)

    # Set the criterion based on the argument
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'psnr':
        criterion = tr_fu.psnr_loss
    elif loss == 'l1':
        criterion = nn.L1Loss()
    elif loss == 'multi':
        criterion = tr_fu.multi_domain_loss

    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size, cycle_momentum=False, mode=lr_mode)

    training_losses, validation_losses, validation_psnr, learning_rates, validation_peak_to_peak = [], [], [], [], []

    for epoch in range(n_epochs):
        if epoch % 10 == 0:
            train_dataset = tr_fu.CustomDataset_hdf5(clean_signals_train, noise_computer2, traces_len=512, lst=lst, no_random=False, xy_mode=xy_mode, which_noise=which_noise, real_an_noisefile=real_an_train_file)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
        print(f'Epoch {epoch}/{n_epochs}')
        avg_train_loss, avg_valid_loss, avg_psnr, lr, avg_peak_to_peak_ratio = train_functions.train_one_epoch_loop(
            train_loader, valid_loader, model_mse, optimizer, criterion, device, save_folder=save_folder, scheduler=scheduler
        )
        if epoch % 20 == 0:
            torch.save(model_mse.state_dict(), os.path.join(save_folder, 'model_epoch{}.pth'.format(epoch)))

        training_losses.append(avg_train_loss)
        validation_losses.append(avg_valid_loss)
        validation_psnr.append(avg_psnr)
        validation_peak_to_peak.append(avg_peak_to_peak_ratio)
        learning_rates.append(optimizer.param_groups[0]['lr'])

    epochs = range(1, n_epochs + 1)
    aa = training_losses, validation_losses, validation_psnr, learning_rates, validation_peak_to_peak
    tr_fu.plot_metrics(epochs, training_losses, validation_losses, validation_psnr, learning_rates= learning_rates, validation_peak_to_peak = validation_peak_to_peak, save_folder = save_folder)

    np.save(os.path.join(save_folder, 'results.npy'), np.array(aa))
    torch.save(model_mse.state_dict(), os.path.join(save_folder, 'best_model.pth'))

    time = np.arange(1024)

    # make plots for the train
    path_plot_trainfortest = os.path.join(save_folder, 'plots_train_for_test')
    os.makedirs(path_plot_trainfortest, exist_ok=True)
    test.test_func(
        time=time,
        frequency=None,
        testloader=train_loader_for_test,
        num_images=50,
        model=model_mse,
        device=device,
        min_snr=0.1,
        max_snr=300,
        save_folder=path_plot_trainfortest,
        voltage=False,
        ADC=True,
        efield=False
    )

    path_plot_test = os.path.join(save_folder, 'plots_test')
    os.makedirs(path_plot_test, exist_ok=True)
    test.test_func(
        time=time,
        frequency=None,
        num_images=50,
        testloader=test_loader,
        model=model_mse,
        device=device,
        min_snr=0.1,
        max_snr=300,
        save_folder=path_plot_test,
        voltage=False,
        ADC=True,
        efield=False
    )

    peak_amp, snr_values, peak_times = peak_time_and_amplitude(
        dataloader=test_loader,
        model=model_mse,
        device=device,
        min_snr=0.1,
        max_snr=300,
        save_folder=save_folder
    )

    with open(os.path.join(save_folder, 'peak_amplitudes.json'), 'w') as f:
        json.dump(peak_amp, f)

    with open(os.path.join(save_folder, 'snr_values.json'), 'w') as f:
        json.dump(snr_values, f)

    with open(os.path.join(save_folder, 'peak_times.json'), 'w') as f:
        json.dump(peak_times, f)

    return peak_amp, snr_values, peak_times


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some traces.')
    parser.add_argument('--json_params_file', help='master json params file')

    args = parser.parse_args()

    json_params_file = args.json_params_file

    with open(json_params_file, 'r') as f:
        all_params = json.load(f)

    output_path = all_params['output_path']

    print(os.path.basename(json_params_file))

    os.makedirs(output_path, exist_ok=True)
    shutil.copy(json_params_file, os.path.join(output_path, os.path.basename(json_params_file)))

    tag = all_params['tag']

    which_noise = all_params['which_noise']

    n_epochs = all_params['n_epochs']
    batch_size = 1024

    loss = all_params['loss']

    base_lr = all_params['base_lr']
    max_lr = all_params['max_lr']

    step_size_up = all_params['step_size_up']
    lr_mode = all_params['lr_mode']
    lst = all_params['lst']
    model_to_load = all_params['model_to_load']

    noise_param_file = all_params['noise_param_file']
    if os.path.isfile(noise_param_file):
        shutil.copy(noise_param_file, os.path.join(output_path, os.path.basename(noise_param_file)))
    else:
        print('error: the noise params file is not there')

    a, b, c = main(all_params)
