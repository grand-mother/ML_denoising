"""
Training and validation functions for Ray Tune hyperparameter optimization.
"""
import tempfile
from pathlib import Path
import os
import sys

from ray import train
from ray.train import Checkpoint, get_checkpoint
import ray.cloudpickle as pickle
import torch
import torch.optim as optim
import torch.nn as nn
import auraloss.freq
import time
import shutil
import glob

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from training.models.cnn import DualBranchAutoencoder as CNN

from training.raytune_training_function import (
    psnr_loss,
    multi_domain_mse_loss,
    multi_domain_l1_loss,
    calculate_psnr_with_peak,
    peak_to_peak_ratio
)
# Reproduction of archived checkpoints only; see training/legacy_losses.py.
from training.legacy_losses import multi_domain_loss_v2, multi_domain_loss_v3

def cleanup_old_checkpoints_dynamic(epoch, checkpoint_base_dir=None, keep_last_n=6):
    """
    Dynamic checkpoint cleanup that keeps only the last N checkpoints.
    Deletes data.pkl files from old checkpoints to free up space while preserving directory structure.
    """
    if epoch < keep_last_n:
        return  # Don't clean up until we have more than keep_last_n checkpoints
    
    try:
        print(f"DEBUG: Current working directory: {os.getcwd()}")
        print(f"DEBUG: Provided checkpoint_base_dir: {checkpoint_base_dir}")
        
        # Try multiple methods to find the correct checkpoint directory
        if checkpoint_base_dir is None:
            # Method 1: Try to get from Ray Tune checkpoint
            try:
                from ray.tune import get_checkpoint
                ckpt = get_checkpoint()
                if ckpt:
                    with ckpt.as_directory() as d:
                        # Get the parent directory of the checkpoint
                        checkpoint_base_dir = str(Path(d).parent)
                        print(f"DEBUG: Got checkpoint directory from Ray Tune: {checkpoint_base_dir}")
            except Exception as e:
                print(f"DEBUG: Could not get checkpoint directory from Ray Tune: {e}")
            
            # Method 2: Try to find the ray_results directory
            if checkpoint_base_dir is None:
                # Look for ray_results in common locations
                possible_paths = [
                    "/pbs/home/o/omacias/ray_results",
                    os.path.expanduser("~/ray_results"),
                    os.path.join(os.getcwd(), "ray_results")
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        # Find the most recent experiment directory
                        exp_dirs = glob.glob(os.path.join(path, "train_validate_wrapper_*"))
                        if exp_dirs:
                            # Sort by modification time and get the most recent
                            exp_dirs.sort(key=os.path.getmtime, reverse=True)
                            latest_exp = exp_dirs[0]
                            
                            # Find the trial directory that matches current process
                            trial_dirs = glob.glob(os.path.join(latest_exp, "train_validate_wrapper_*"))
                            if trial_dirs:
                                # For now, use the first trial directory found
                                # In a real scenario, you'd want to match by trial ID
                                checkpoint_base_dir = trial_dirs[0]
                                print(f"DEBUG: Found trial directory: {checkpoint_base_dir}")
                                break
            
            # Method 3: Fallback to current working directory
            if checkpoint_base_dir is None:
                checkpoint_base_dir = os.getcwd()
                print(f"DEBUG: Using current working directory: {checkpoint_base_dir}")
        
        print(f"DEBUG: Final checkpoint_base_dir: {checkpoint_base_dir}")
        
        # Find all checkpoint directories
        checkpoint_dirs = glob.glob(os.path.join(checkpoint_base_dir, "checkpoint_*"))
        
        if len(checkpoint_dirs) <= keep_last_n:
            print(f"DEBUG: Only {len(checkpoint_dirs)} checkpoints found, keeping all (keep_last_n={keep_last_n})")
            return  # Not enough checkpoints to clean up
        
        # Sort by checkpoint number
        def extract_checkpoint_num(path):
            import re
            match = re.search(r'checkpoint_(\d+)', path)
            return int(match.group(1)) if match else 0
        
        checkpoint_dirs.sort(key=extract_checkpoint_num)
        
        # Keep only the last N checkpoints
        checkpoints_to_clean = checkpoint_dirs[:-keep_last_n]
        
        print(f"DEBUG: Found {len(checkpoint_dirs)} total checkpoints")
        print(f"DEBUG: Will clean up {len(checkpoints_to_clean)} old checkpoints")
        print(f"DEBUG: Will keep {keep_last_n} most recent checkpoints")
        
        deleted_count = 0
        total_size_freed = 0
        
        for checkpoint_dir in checkpoints_to_clean:
            data_pkl_path = os.path.join(checkpoint_dir, "data.pkl")
            print(f"DEBUG: Checking if exists: {data_pkl_path}")
            
            if os.path.exists(data_pkl_path):
                file_size = os.path.getsize(data_pkl_path)
                total_size_freed += file_size
                
                print(f"DEBUG: Deleting data.pkl: {data_pkl_path} (size: {file_size / (1024*1024):.2f} MB)")
                os.remove(data_pkl_path)
                
                checkpoint_name = os.path.basename(checkpoint_dir)
                print(f"Deleted data.pkl from checkpoint: {checkpoint_name}")
                deleted_count += 1
            else:
                print(f"DEBUG: data.pkl does not exist: {data_pkl_path}")
        
        if deleted_count > 0:
            print(f"Dynamic cleanup completed. Deleted {deleted_count} data.pkl files.")
            print(f"Total space freed: {total_size_freed / (1024*1024):.2f} MB")
        else:
            print("No old checkpoint data.pkl files found to delete.")
        
    except Exception as e:
        print(f"ERROR: Could not clean up checkpoints: {e}")
        import traceback
        traceback.print_exc()

def get_criterion(name, config=None):
    # Handle case where name is already a loss object
    if hasattr(name, '__call__') and hasattr(name, 'forward'):
        return name
    
    if name == "mse":
        return nn.MSELoss()
    elif name == "psnr":
        return psnr_loss  
    elif name == "l1":
        return nn.L1Loss()
    elif name == "multi_mse": ## multi mse loss with mag and phase
        return lambda pred, clean: multi_domain_mse_loss(clean, pred, mag_weight= config["mag_weight"], phase_weight= config["phase_weight"], phase_weighting=config.get("phase_weighting", "none"))
    elif name == "multi_l1": ## multi l1 loss with mag and phase
        return lambda pred, clean: multi_domain_l1_loss(clean, pred, mag_weight= config["mag_weight"], phase_weight= config["phase_weight"], phase_weighting=config.get("phase_weighting", "none"))

    # --- LEGACY names, for reproducing archived checkpoints only. ---------------
    # Archived best_trial_config.json files carry "multi_v2"/"multi_v3", which the
    # current API renamed to "multi_mse"/"multi_l1". The old losses are NOT the new
    # ones: their phase term applies L1/MSE directly to torch.angle, which is
    # discontinuous at +/-pi and mis-scores errors across the wrap-around (reply2).
    # They are resolvable here so that a production config still loads; do not
    # select them for new training.
    elif name == "multi_v2":  ## LEGACY == archived multi_v2 (direct phase, defective)
        return lambda pred, clean: multi_domain_loss_v2(clean, pred, mag_weight=config["mag_weight"], phase_weight=config["phase_weight"])
    elif name == "multi_v3":  ## LEGACY == archived multi_v3, the production checkpoint (direct phase, defective)
        return lambda pred, clean: multi_domain_loss_v3(clean, pred, mag_weight=config["mag_weight"], phase_weight=config["phase_weight"])
    else:
        raise ValueError(f"Unknown criterion: {name}")

def train_validate(config, checkpoint_dir=None, train_loader=None, valid_loader=None):
    # Add a simple test file write at the very beginning
    try:
        with open("/tmp/train_validate_started.txt", "w") as f:
            f.write(f"train_validate started at {time.time()}\n")
            f.write(f"Config: {config}\n")
        print("=== train_validate function STARTED ===")
    except Exception as e:
        print(f"Error writing test file: {e}")
    
    print(f"Config epochs: {config.get('epochs', 'NOT FOUND')}")
    print(f"Config save_folder_name: {config.get('save_folder_name', 'NOT FOUND')}")
    
    if train_loader is None or valid_loader is None:
        raise ValueError("train_loader and valid_loader must be provided")

    if config["model_type"] == "CNN":
        model = CNN(config["model_config"])
    elif config["model_type"] == "TimeOnlyCNN":
        from training.models.cnn import TimeOnlyAutoencoder
        model = TimeOnlyAutoencoder(config["model_config"])
    else:
        raise ValueError(f"Unknown model_type: {config['model_type']}. Only 'CNN' or 'TimeOnlyCNN' are supported.")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    
    # Define the optimizer with hyperparameters from config
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config["lr"], 
        weight_decay=config["weight_decay"]
    )

    # Choose the criterion
    criterion = get_criterion(config["criterion"], config)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, 
        base_lr=config["lr"], 
        max_lr=config["max_lr"], 
        step_size_up=config["step_size"], 
        cycle_momentum=False, 
        mode=config["lr_mode"]
    )

    # Initialize lists to store metrics
    training_losses = []
    validation_losses = []
    validation_psnr_values = []    
    validation_peak_to_peak_values = []
    learning_rates = []

    # Get the base directory for checkpoints
    checkpoint_base_dir = None

    ckpt = get_checkpoint()
    if ckpt:
        with ckpt.as_directory() as d:
            state = pickle.load(open(Path(d) / "data.pkl", "rb"))
        start_epoch = state["epoch"] + 1
        model.load_state_dict(state["net_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        # Extract checkpoint base directory from current checkpoint path
        checkpoint_base_dir = str(Path(d).parent)
    else:
        start_epoch = 0

    for epoch in range(start_epoch, config["epochs"]):
        print(f"=== Training epoch {epoch}/{config['epochs']} ===")
        
        # Training
        with torch.set_grad_enabled(True):
            model.train()
            total_train_loss = 0
            # This loop is kept identical to the production run (2025-10-28,
            # raytune_lib_sept25): zero_grad -> forward -> loss -> backward -> step,
            # with no gradient clipping and no NaN/Inf batch guards. Both had been
            # added here after the fact (this repository dates from 2026-01-12, two and
            # a half months after that run) and were removed on 2026-08-08 so the
            # archived script does not misdescribe the training that produced the
            # published checkpoints. Do not reinstate either without also restating the
            # training description in the paper.
            for k, (noisy_data, clean_data) in enumerate(train_loader):
                noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
                optimizer.zero_grad()
                outputs = model(noisy_data)
                loss = criterion(outputs, clean_data)
                loss.backward()
                optimizer.step()
                if not (scheduler is None):  
                    scheduler.step()

                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)

            model.eval()
            total_valid_loss, total_psnr, total_peak_to_peak_ratio = 0, 0, 0
            with torch.no_grad():
                for noisy_data, clean_data in valid_loader:
                    clean_data, noisy_data = clean_data.to(device), noisy_data.to(device)
                
                    outputs = model(noisy_data)
                    loss = criterion(outputs, clean_data)
                    total_valid_loss += loss.item()
                    psnr_value = calculate_psnr_with_peak(clean_data.cpu().numpy(), outputs.cpu().numpy())
                    total_psnr += psnr_value
                    ratio = peak_to_peak_ratio(clean_data.cpu().numpy(), outputs.cpu().numpy())
                    total_peak_to_peak_ratio += ratio

            avg_valid_loss = total_valid_loss / len(valid_loader)
            avg_psnr = total_psnr / len(valid_loader)
            avg_peak_to_peak_ratio = total_peak_to_peak_ratio / len(valid_loader)
            lr = optimizer.param_groups[0]['lr']

            # Log & record
            training_losses.append(avg_train_loss)
            validation_losses.append(avg_valid_loss)
            validation_psnr_values.append(avg_psnr)
            validation_peak_to_peak_values.append(avg_peak_to_peak_ratio)
            learning_rates.append(lr)

            # Save checkpoint with metrics and report once
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                data = {
                    "epoch": epoch,
                    "net_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics_data": {
                        "epochs": list(range(start_epoch, epoch + 1)),
                        "training_losses": training_losses,
                        "validation_losses": validation_losses,
                        "validation_psnr": validation_psnr_values,
                        "peak_to_peak_ratio": validation_peak_to_peak_values,
                        "learning_rates": learning_rates
                    }
                }
                with open(Path(checkpoint_dir) / "data.pkl", "wb") as fp:
                    pickle.dump(data, fp)

                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                is_final_epoch = (epoch == config["epochs"] - 1)
                report_data = {"loss": avg_valid_loss, "validation_loss": avg_valid_loss, "validation_psnr": avg_psnr}
                if is_final_epoch:
                    report_data["done"] = True
                train.report(report_data, checkpoint=checkpoint)
                
                # Clean up old checkpoints AFTER saving the new one
                # cleanup_old_checkpoints_dynamic(epoch, checkpoint_base_dir, keep_last_n=6)
    
    print("=== Training loop completed ===")
    print(f"Training losses collected: {len(training_losses)}")
    print(f"Validation losses collected: {len(validation_losses)}")
    
    # After training is complete, save the metrics for the best trial
    metrics_data = {
        "epochs": list(range(start_epoch, config["epochs"])),
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "validation_psnr": validation_psnr_values,
        "peak_to_peak_ratio": validation_peak_to_peak_values,
        "learning_rates": learning_rates
    }

    print("=== train_validate function ending ===")
    return metrics_data