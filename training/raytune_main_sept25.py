"""
Ray Tune Hyperparameter Optimization for Radio Signal Denoising
Main entry point for training.
"""
import argparse
import os
import sys
import numpy as np
import torch
import shutil
from torch.utils.data import Dataset, DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import ray
import ray.cloudpickle as pickle
import ray.tune as tune
from ray.tune.schedulers import ASHAScheduler
from pathlib import Path

# Add project root to path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Model imports
from training.models.cnn import DualBranchAutoencoder as CNN
from training.raytune_training_function import CustomDataset, split_indices, plot_metrics
from training.raytune_train_sept25 import train_validate

# Utils imports
from utils.data_preprocessing import produce_noise_and_noiseless_data
from utils.config_utils import (load_json_config, convert_to_tune_config,
                                save_best_trial_results, save_detailed_metrics)
from utils.paper_losses_metrics import deterministic_split_indices, save_split_manifest

# Default config paths (relative to project root)
DEFAULT_MODEL_CONFIG = ROOT_DIR / "configs" / "model_config.json"
DEFAULT_TRAINING_CONFIG = ROOT_DIR / "configs" / "training_config.json"

def main(all_params):
    print("Starting the main function")   
    mpl.rcParams['figure.max_open_warning'] = 50
    # Convert to absolute path (required by Ray Tune's pyarrow storage)
    output_path = os.path.abspath(all_params["output_path"])
    os.makedirs(output_path, exist_ok=True)
    print(f"Output path (absolute): {output_path}")

    tag = all_params["tag"]
    loss = all_params["loss"]
    n_epochs = all_params["n_epochs"]
    number_samples = all_params["number_samples"]
    model_type = all_params["model_type"]
    model_to_load = all_params["model_to_load"]
    batch_size = 1024

    sim_data_dir = all_params["sim_data_dir"]
    
    noise_signals, clean_signals = produce_noise_and_noiseless_data(sim_data_dir)
    total_samples_train = np.shape(clean_signals)[1]
    total_noise_samples = np.shape(noise_signals)[1]
    print(f'total_sample_train = {total_samples_train}')
    print(f'total_noise_samples = {total_noise_samples}')
    
    # --- Task 2: one frozen, seeded 80/10/10 split, reused by every Ray trial. ---
    # The split is created ONCE here (not per trial) and persisted; the test
    # indices are loaded from this manifest and never regenerated.
    split_seed = int(all_params.get("split_seed", 12345))
    train_indices, valid_indices, test_indices = deterministic_split_indices(
        total_samples_train, train_fraction=0.8, valid_fraction=0.1, seed=split_seed)
    save_split_manifest(
        Path(output_path) / "split_manifest.npz",
        train_indices, valid_indices, test_indices,
        n_total=total_samples_train, seed=split_seed)
    print(f"Frozen split (seed={split_seed}): "
          f"train={train_indices.size} valid={valid_indices.size} test={test_indices.size}")

    # Deterministic evaluation: no trace swapping AND no random cropping
    # (no_random=True -> fixed full-length 1024 trace).
    test_dataset = CustomDataset(clean_signals, [noise_signals], indices=test_indices,
                                 swap_prob=0.0, no_random=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device set to : {device}')

    # Load model configuration from JSON
    model_config_path = all_params.get("model_config_path", str(DEFAULT_MODEL_CONFIG))
    model_config_json = load_json_config(model_config_path)
    
    if all_params["model_type"] not in model_config_json:
        raise ValueError(f"Model type '{all_params['model_type']}' not found in model config. Available types: {list(model_config_json.keys())}")
    
    model_config = convert_to_tune_config(model_config_json[all_params["model_type"]])


    # Add debugging to see what n_epochs is set to
    print(f"n_epochs from all_params: {n_epochs}")
    print(f"all_params keys: {all_params.keys()}")

    # Load training configuration from JSON
    training_config_path = all_params.get("training_config_path", str(DEFAULT_TRAINING_CONFIG))
    training_config_json = load_json_config(training_config_path)
    train_config = convert_to_tune_config(training_config_json)
    
    # Add fixed parameters
    train_config.update({
        "epochs": n_epochs,
        "criterion": loss,
        "save_folder_name": output_path,
        "model_type": all_params["model_type"],
        "model_config": model_config,
    })
    
    # Add loss-specific parameters
    if loss == "multi_l1" or loss == "multi_mse":
        if "mag_weight" not in train_config:
            train_config["mag_weight"] = tune.loguniform(1e-2, 1.0)
        if "phase_weight" not in train_config:
            train_config["phase_weight"] = tune.loguniform(1e-2, 1.0)
    elif loss not in ["l1", "mse", "psnr", "multi_l1", "multi_mse"]:
        raise ValueError("Unknown loss: {}".format(loss))
    print(f"train_config epochs: {train_config['epochs']}")

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        num_cpus=int(os.environ.get("RAY_NUM_CPUS", 4)),
        num_gpus=int(os.environ.get("RAY_NUM_GPUS", 1))
    )


    print("Ray initialized")
    scheduler = ASHAScheduler(
        metric="validation_loss", #loss
        mode="min",
        max_t=n_epochs,
        grace_period = 50 if n_epochs > 50 else n_epochs,
        reduction_factor=4
    )
    # scheduler = tune.schedulers.FIFOScheduler()
    clean_signals_ref = ray.put(clean_signals)
    noise_signals_ref = ray.put(noise_signals)
    # Share the single frozen split with every trial via the object store.
    train_indices_ref = ray.put(train_indices)
    valid_indices_ref = ray.put(valid_indices)

    def train_validate_wrapper(config):
        print("=== train_validate_wrapper called ===")
        print(f"Config: {config}")
        
        try:
            # Get the data from Ray's object store
            clean_signals = ray.get(clean_signals_ref)
            noise_signals = ray.get(noise_signals_ref)

            print("=== Data loaded from Ray object store ===")

            # Task 2: reuse the ONE frozen split (do not re-split per trial).
            train_indices = ray.get(train_indices_ref)
            valid_indices = ray.get(valid_indices_ref)
    
            # Create datasets and loaders
            train_dataset = CustomDataset(clean_signals, [noise_signals], indices=train_indices, no_random=True)
            # Deterministic validation: no swap, no random crop (fixed full 1024 trace).
            valid_dataset = CustomDataset(clean_signals, [noise_signals], indices=valid_indices,
                                          swap_prob=0.0, no_random=True)
            
            print("=== Datasets created ===")
            
            train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 1024), shuffle=True, pin_memory=True)
            valid_loader = DataLoader(valid_dataset, batch_size=config.get("batch_size", 1024), shuffle=False, pin_memory=True)
            
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
        resources_per_trial={"cpu": 4, "gpu": 4}, 
        config=train_config,
        num_samples = all_params["number_samples"],
        scheduler=scheduler,
        max_concurrent_trials=2,
        storage_path=output_path
    )#      


    best_trial = result.get_best_trial("validation_loss", "min", "last")
    print(f"Best trial config: {best_trial.local_path}")

    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    if 'validation_psnr' in best_trial.last_result:
        print(f"Best trial final validation PSNR: {best_trial.last_result['validation_psnr']}")

    # Save best trial results to JSON files
    model_type = best_trial.config["model_type"]
    best_config_path, best_metrics_path = save_best_trial_results(best_trial, output_path, model_type)
    print(f"Best trial config saved to: {best_config_path}")
    print(f"Best trial metrics saved to: {best_metrics_path}")

    # Create the model FIRST based on the best trial config
    if model_type == "CNN":
        best_train_model = CNN(best_trial.config["model_config"])
    elif model_type == "TimeOnlyCNN":
        from training.models.cnn import TimeOnlyAutoencoder
        best_train_model = TimeOnlyAutoencoder(best_trial.config["model_config"])
    else:
        raise ValueError("Unknown model_type: {}. Only 'CNN' or 'TimeOnlyCNN' are supported.".format(model_type))

    # THEN load the checkpoint and get metrics
    best_checkpoint = result.get_best_checkpoint(trial=best_trial, mode="min", metric="validation_loss")
    
    if best_checkpoint is None:
        print("WARNING: No valid checkpoint found. This may be due to NaN values in training.")
        print("Check if the loss function is producing valid gradients.")
        print("Attempting to load checkpoint from trial's local directory...")
        
        # Try to find checkpoint in trial directory
        trial_checkpoint_dir = Path(best_trial.local_path) / "checkpoint_000000"
        if trial_checkpoint_dir.exists():
            checkpoint_dir = trial_checkpoint_dir
            print(f"Found checkpoint at: {checkpoint_dir}")
        else:
            raise RuntimeError(
                f"No checkpoint found. Training may have failed with NaN losses. "
                f"Check the training logs for numerical issues. "
                f"Trial path: {best_trial.local_path}"
            )
    else:
        checkpoint_dir = best_checkpoint.as_directory().__enter__()
    
    with checkpoint_dir if isinstance(checkpoint_dir, Path) else best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        # Handle DataParallel state_dict prefix mismatch
        state_dict = best_checkpoint_data["net_state_dict"]
        
        # Check if state_dict has "module." prefix (from DataParallel training)
        if any(k.startswith("module.") for k in state_dict.keys()):
            # Remove "module." prefix for loading into non-DataParallel model
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            print("Removed 'module.' prefix from state_dict keys (DataParallel -> single model)")
        
        best_train_model.load_state_dict(state_dict)
        
        # Get metrics from checkpoint if available
        if "metrics_data" in best_checkpoint_data:
            result_metrics = best_checkpoint_data["metrics_data"]
            print("Loaded metrics from checkpoint")
            print(f"Metrics keys: {result_metrics.keys()}")
            print(f"Number of epochs: {len(result_metrics.get('epochs', []))}")
            print(f"Number of training losses: {len(result_metrics.get('training_losses', []))}")
            print(f"Number of validation losses: {len(result_metrics.get('validation_losses', []))}")
            
            # Save detailed metrics to JSON file
            detailed_metrics_path = save_detailed_metrics(result_metrics, output_path)
            print(f"Detailed metrics saved to: {detailed_metrics_path}")
        else:
            print("No metrics found in checkpoint")
            result_metrics = {}

    # Plot metrics if available
    if result_metrics and "epochs" in result_metrics:
        plot_metrics(
                result_metrics["epochs"], 
                result_metrics["training_losses"], 
                result_metrics["validation_losses"], 
                result_metrics["validation_psnr"], 
                result_metrics.get("learning_rates", []), 
                result_metrics.get("peak_to_peak_ratio", []), 
                output_path
            )
    else:
        print("No metrics available for plotting")
    
    # Save the best model
    torch.save(best_train_model.state_dict(), os.path.join(output_path, 'best_model.pth'))


    print("Training is all finished")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Hyperparameter tuning with Ray Tune.')

    parser.add_argument('--json_params_file', help='master json params file')

    args = parser.parse_args()

    json_params_file = args.json_params_file

    with open(json_params_file, 'r') as f:
        all_params = json.load(f)

    output_path = all_params["output_path"]

    print(os.path.basename(json_params_file))

    os.makedirs(output_path, exist_ok=True)
    shutil.copy(json_params_file, os.path.join(output_path, os.path.basename(json_params_file)))

    tag = all_params["tag"]
    n_epochs = all_params["n_epochs"]
    loss = all_params["loss"]
    number_samples = all_params["number_samples"]
    model_type = all_params["model_type"]
    model_to_load = all_params["model_to_load"]
    batch_size = 1024
    sim_data_dir = all_params["sim_data_dir"]
    noise_signals, clean_signals = produce_noise_and_noiseless_data(sim_data_dir)
    total_samples_train = np.shape(clean_signals)[1]
    total_noise_samples = np.shape(noise_signals)[1]
    print(f'total_sample_train = {total_samples_train}')
    print(f'total_noise_samples = {total_noise_samples}')

    # main(all_params)
    main(all_params)
