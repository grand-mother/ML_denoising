import tempfile
from pathlib import Path

from ray import train
from ray.train import Checkpoint, get_checkpoint
import ray.cloudpickle as pickle
import os
import torch
import torch.optim as optim
import torch.nn as nn
import time


from raytune_CNN import DualBranchAutoencoder as CNN
from raytune_SingleCNN import SingleBranchAutoencoder as SingleBranchAutoencoder

from raytune_training_function import (
    psnr_loss, 
    multi_domain_loss,
    multi_domain_loss_v2,
    calculate_psnr_with_peak, 
    peak_to_peak_ratio
)
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
    elif name == "multi":
        return lambda pred, clean: multi_domain_loss_v2(clean, pred, mag_weight= config["mag_weight"], phase_weight= config["phase_weight"]) # Make sure this is defined/imported
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

    if config["model_type"] == "SingleBranchCNN":
        model = SingleBranchAutoencoder(config["model_config"])
    elif config["model_type"] == "CNN":
        model = CNN(config["model_config"])
    else:
        raise ValueError(f"Unknown model_type: {config['model_type']}")

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

    ckpt = get_checkpoint()
    if ckpt:
        with ckpt.as_directory() as d:
            state = pickle.load(open(Path(d) / "data.pkl", "rb"))
        start_epoch = state["epoch"] + 1
        model.load_state_dict(state["net_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
    else:
        start_epoch = 0

    for epoch in range(start_epoch, config["epochs"]):
        print(f"=== Training epoch {epoch}/{config['epochs']} ===")
        # Training
        with torch.set_grad_enabled(True):
            model.train()
            total_train_loss = 0
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
            avg_train_loss = total_train_loss / len(train_loader) # average loss for each epoch

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
                # Single report with checkpoint
                # train.report({"loss": avg_valid_loss, "validation_psnr": avg_psnr}, checkpoint=checkpoint)
                is_final_epoch = (epoch == config["epochs"] - 1)
                report_data = {"loss": avg_valid_loss, "validation_loss": avg_valid_loss, "validation_psnr": avg_psnr}
                if is_final_epoch:
                    report_data["done"] = True
                train.report(report_data, checkpoint=checkpoint)
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

    # # Save metrics to checkpoint instead of separate file
    # with tempfile.TemporaryDirectory() as checkpoint_dir:
    #     # Save the final model and metrics together
    #     data = {
    #         "epoch": config["epochs"] - 1,
    #         "net_state_dict": model.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "metrics_data": metrics_data  # Add metrics to checkpoint
    #     }
    #     with open(Path(checkpoint_dir) / "data.pkl", "wb") as fp:
    #         pickle.dump(data, fp)

    #     checkpoint = Checkpoint.from_directory(checkpoint_dir)
    #     train.report({"loss": avg_valid_loss, "validation_psnr": avg_psnr, "done": True}, checkpoint=checkpoint)

    print("=== train_validate function ending ===")
    return metrics_data