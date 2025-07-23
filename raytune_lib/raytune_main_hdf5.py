import argparse
import os
import numpy as np
import torch
import shutil
from torch.utils.data import Dataset, DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt
from raytune_CNN import DualBranchAutoencoder as CNN
from raytune_SingleCNN import SingleBranchAutoencoder as SingleBranchCNN # for the old model
from pathlib import Path
import ray
import sys
sys.path.append('/Users/923714256/ML_denoising/raytune_lib')
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from raytune_lib.overleaf_plots import traces_plot, peak_amplitude_analysis_all_channels, peak_time_analysis, plot_amplitude_ratio_vs_snr_all_channels

from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

import raytune_training_function as tr_fu
# from ml_denoising_lib.utils import train as train_functions
# from ml_denoising_lib.utils import test as test
# from ml_denoising_lib.utils.hilbert import peak_time_and_amplitude

from raytune_train_hdf5 import train_validate, get_criterion

import json
from DU_response_computation import apply_rfchain as rfc


def main(all_params):
    print("Starting the main function")   
    mpl.rcParams['figure.max_open_warning'] = 50
    output_path = all_params['output_path']
    print(os.path.basename(json_params_file))

    tag = all_params['tag']
    loss = all_params['loss']
    which_noise = all_params['which_noise']
    n_epochs = all_params['n_epochs']
    lst = all_params['lst']
    model_to_load = all_params['model_to_load']
    batch_size = 1024
    xy_mode = all_params['xy_mode']
    noise_param_file = all_params['noise_param_file']

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
    print(f'total_sample_train = {total_samples_train}')

    clean_signals_validation = np.load(save_path_clean_sims_validation).transpose(1, 0, 2)
    clean_signals_test = np.load(save_path_clean_sims_test).transpose(1, 0, 2)

    
    # valid_dataset = tr_fu.CustomDataset_hdf5(clean_signals_validation, noise_computer2, traces_len=1024, lst=lst, no_random=True, xy_mode=xy_mode)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device set to : {device}')

    train_dataset = tr_fu.CustomDataset_hdf5(clean_signals_train, noise_computer2, traces_len=512, lst=lst, no_random=False, xy_mode=xy_mode, which_noise=which_noise, real_an_noisefile=real_an_train_file)
    valid_dataset = tr_fu.CustomDataset_hdf5(clean_signals_validation, noise_computer2, traces_len=512, lst=lst, no_random=False, xy_mode=xy_mode, which_noise=which_noise, real_an_noisefile=real_an_valid_file)
    test_dataset = tr_fu.CustomDataset_hdf5(clean_signals_test, noise_computer2, no_random=True, lst=lst, xy_mode=xy_mode, which_noise=which_noise, real_an_noisefile=real_an_test_file)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=True)
    train_loader_for_test = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True, pin_memory=True)
    

    if all_params["model_type"] == "CNN":
        model_config = {
            "time_branch": {
                "conv_channels": tune.choice([16, 32, 64]),
                "res_channels": tune.choice([(16, 32), (32, 64), (64, 128), (128, 256)])
            },
            "freq_branch": {
                "conv_channels": tune.choice([16, 32, 64]),
                "res_channels": tune.choice([(16, 32), (32, 64), (64, 128), (128, 256)])
            },
            "decoder_channels": tune.choice([[128, 64, 32, 3], [256, 128, 64, 3], [64, 32, 16, 3], [32, 16, 8, 3]])
            # You can add CNN-specific params here if needed
        }
    elif all_params["model_type"] == "SingleBranchCNN":
        model_config = {
            "time_branch": {
                "conv_channels": tune.choice([16, 32, 64]),
                "res_channels": tune.choice([(16, 32), (32, 64), (64, 128), (128, 256)])
            },
            "decoder_channels": tune.choice([[128, 64, 32, 3], [256, 128, 64, 3], [64, 32, 16, 3], [32, 16, 8, 3]])
        }
    else:
        raise ValueError("Unknown model_type: {}".format(all_params["model_type"]))

    # Add debugging to see what n_epochs is set to
    print(f"n_epochs from all_params: {n_epochs}")
    print(f"all_params keys: {all_params.keys()}")

    train_config = {
        "lr": tune.loguniform(1e-6, 1e-4),
        "max_lr": tune.loguniform(1e-5, 1e-3),
        "step_size": tune.choice([10000, 20000, 30000, 40000, 50000]),
        "lr_mode": tune.choice(["triangular2", "triangular"]),
        "weight_decay": tune.loguniform(1e-5, 5e-3),
        "epochs": n_epochs, 
        "criterion": loss,
        "save_folder_name": save_folder,
        "model_type": all_params["model_type"],
        "model_config": model_config,
        "mag_weight": tune.loguniform(1e-2, 1.0), 
        "phase_weight": tune.loguniform(1e-2, 1.0),
        "batch_size": tune.choice([512,1024,2048,4096]),
    }

    print(f"train_config epochs: {train_config['epochs']}")
    
    scheduler = ASHAScheduler(
        metric="validation_loss", #loss
        mode="min",
        max_t=n_epochs,
        grace_period = 35 if n_epochs > 35 else n_epochs,
        reduction_factor=4
    )
    clean_signals_train_ref = ray.put(clean_signals_train)
    clean_signals_validation_ref = ray.put(clean_signals_validation)
    noise_computer2_ref = ray.put(noise_computer2)
    def train_validate_wrapper(config):
        print("=== train_validate_wrapper called ===")
        print(f"Config: {config}")
        
        try:
            # Get the data from Ray's object store
            clean_signals_train = ray.get(clean_signals_train_ref)
            clean_signals_validation = ray.get(clean_signals_validation_ref)
            noise_computer2 = ray.get(noise_computer2_ref)
            
            print("=== Data loaded from Ray object store ===")
            
            # Create datasets and loaders
            train_dataset = tr_fu.CustomDataset_hdf5(clean_signals_train, noise_computer2, traces_len=512, lst=lst, no_random=False, xy_mode=xy_mode, which_noise=which_noise, real_an_noisefile=real_an_train_file)
            valid_dataset = tr_fu.CustomDataset_hdf5(clean_signals_validation, noise_computer2, traces_len=512, lst=lst, no_random=False, xy_mode=xy_mode, which_noise=which_noise, real_an_noisefile=real_an_valid_file)
            
            print("=== Datasets created ===")
            
            train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 1024), num_workers=4, shuffle=True, pin_memory=True)
            valid_loader = DataLoader(valid_dataset, batch_size=config.get("batch_size", 1024), num_workers=4, shuffle=True, pin_memory=True)
            
            print("=== DataLoaders created ===")
            print(f"Train loader size: {len(train_loader)}")
            print(f"Valid loader size: {len(valid_loader)}")
            
            result = train_validate(config, train_loader=train_loader, valid_loader=valid_loader)
            print("=== train_validate completed successfully ===")
            print(f"Result type: {type(result)}")
            if isinstance(result, dict):
                print(f"Result keys: {result.keys()}")
            return result
            
        except Exception as e:
            print(f"ERROR in train_validate_wrapper: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    result = tune.run(
        train_validate_wrapper,
        resources_per_trial={"cpu": 12, "gpu": 1}, 
        config=train_config,
        num_samples = all_params["number_samples"],
        scheduler=scheduler,
    )#      


    best_trial = result.get_best_trial("validation_loss", "min", "last")
    print(f"Best trial config: {best_trial.local_path}")

    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    if 'validation_psnr' in best_trial.last_result:
        print(f"Best trial final validation PSNR: {best_trial.last_result['validation_psnr']}")

    # Create the model FIRST based on the best trial config
    model_type = best_trial.config["model_type"]
    if model_type == "SingleBranchCNN":
        best_train_model = SingleBranchCNN(best_trial.config["model_config"])
    elif model_type == "CNN":
        best_train_model = CNN(best_trial.config["model_config"])
    else:
        raise ValueError("Unknown model_type: {}".format(model_type))

    # THEN load the checkpoint and get metrics
    best_checkpoint = result.get_best_checkpoint(trial=best_trial, mode="min", metric="validation_loss")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_train_model.load_state_dict(best_checkpoint_data["net_state_dict"])
        
        # Get metrics from checkpoint if available
        if "metrics_data" in best_checkpoint_data:
            result_metrics = best_checkpoint_data["metrics_data"]
            print("Loaded metrics from checkpoint")
            print(f"Metrics keys: {result_metrics.keys()}")
            print(f"Number of epochs: {len(result_metrics.get('epochs', []))}")
            print(f"Number of training losses: {len(result_metrics.get('training_losses', []))}")
            print(f"Number of validation losses: {len(result_metrics.get('validation_losses', []))}")
        else:
            print("No metrics found in checkpoint")

    tr_fu.plot_metrics(
            result_metrics["epochs"], 
            result_metrics["training_losses"], 
            result_metrics["validation_losses"], 
            result_metrics["validation_psnr"], 
            result_metrics.get("learning_rates", []), 
            result_metrics.get("peak_to_peak_ratio", []), 
            output_path
        )
    
    # Save the best model
    torch.save(best_train_model.state_dict(), os.path.join(save_folder, 'best_model.pth'))
    #test for overleaf plots
    
    ##### data preparation

    #train_loader_for_test
    path_plot_trainfortest = os.path.join(save_folder, 'plots_train_for_test')
    os.makedirs(path_plot_trainfortest, exist_ok=True)
    
    traces_plot(testloader=train_loader_for_test, 
        model= best_train_model, 
        num_images = 99,
        device="cpu",
        save_path = path_plot_trainfortest)
    
    peak_amplitude_analysis_all_channels(dataloader=train_loader_for_test, 
                                        model_path = os.path.join(save_folder, 'best_model.pth'),
                                        model = best_train_model, 
                                        device = "cpu", 
                                        min_snr = 1, 
                                        max_snr = 1e3, 
                                        save_path = path_plot_trainfortest)
    
    peak_time_analysis(dataloader=train_loader_for_test, 
                                    model_path = os.path.join(save_folder, 'best_model.pth'),
                                    model = best_train_model, 
                                    device = "cpu", 
                                    min_snr = 1, 
                                    max_snr = 1e3, 
                                    save_path = path_plot_trainfortest)
    
    #testloader is the test data loader
    path_plot_test = os.path.join(save_folder, 'plots_test')
    os.makedirs(path_plot_test, exist_ok=True)

    traces_plot(testloader=test_loader, 
        model= best_train_model, 
        num_images = 99,
        device="cpu",
        save_path = path_plot_test)
    
    peak_amplitude_analysis_all_channels(dataloader=test_loader, 
                                        model_path = os.path.join(save_folder, 'best_model.pth'),
                                        model = best_train_model, 
                                        device = "cpu", 
                                        min_snr = 1, 
                                        max_snr = 1e3, 
                                        save_path = path_plot_test)
    
    peak_time_analysis(dataloader=test_loader, 
                        model_path = os.path.join(save_folder, 'best_model.pth'),
                        model = best_train_model, 
                        device = "cpu", 
                        min_snr = 1, 
                        max_snr = 1e3, 
                        save_path = path_plot_test)

    print("Process is all finished")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Hyperparameter tuning with Ray Tune.')

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
    loss = all_params['loss']
    lst = all_params['lst']
    model_to_load = all_params['model_to_load']
    xy_mode = all_params['xy_mode']
    noise_param_file = all_params['noise_param_file']
    if os.path.isfile(noise_param_file):
        shutil.copy(noise_param_file, os.path.join(output_path, os.path.basename(noise_param_file)))
    else:
        print('error: the noise params file is not there')

    # main(all_params)
    main(all_params)
