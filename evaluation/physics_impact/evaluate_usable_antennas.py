#!/usr/bin/env python3
"""
Script to evaluate the number of usable antennas using denoising models.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import json

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import project modules
from training.models.cnn import DualBranchAutoencoder as CNN
from training.raytune_training_function import CustomDataset, split_indices
from utils.data_preprocessing import produce_noise_and_noiseless_data
from evaluation.physics_impact.number_of_usable_antennas import evaluate_usable_antennas_snr4, plot_usable_fraction_comparison, plot_cumulative_improvement

# Set multiprocessing sharing strategy to avoid "too many open files" error
torch.multiprocessing.set_sharing_strategy('file_system')

def load_best_trial_from_files(metrics_json_path: str,
                               config_json_path: str,
                               model_path: str,
                               model_classes: dict,
                               device: str = "cpu"):
    """
    Load best trial metrics, config (incl. model_config), and the trained model.

    Args:
        metrics_json_path: Path to best_trial_metrics.json
        config_json_path: Path to best_trial_config.json (contains model_config and model_type)
        model_path: Path to best_model.pth
        model_classes: Dict mapping model_type -> class, e.g. {"CNN": CNN}
        device: 'cpu' or CUDA device string, e.g. 'cuda:0'

    Returns:
        (metrics_dict, config_dict, model) tuple
    """
    # Load metrics and config
    with open(metrics_json_path, "r") as f:
        metrics = json.load(f)
    with open(config_json_path, "r") as f:
        config = json.load(f)

    model_type = config.get("model_type")
    model_config = config.get("model_config")
    if model_type not in model_classes:
        raise ValueError(f"Unknown model_type '{model_type}'. Available: {list(model_classes.keys())}")
    if model_config is None:
        raise ValueError("model_config missing in best_trial_config.json")

    # Build model
    ModelClass = model_classes[model_type]
    model = ModelClass(model_config)
    model.to(device)
    model.eval()

    # Load weights (handle PyTorch versions without weights_only)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    return metrics, config, model


def number_of_usable_antennas(test_loader, model, model_path, device="cpu", 
                               snr_threshold=4.0, t_tol_ns=5.0):
    """
    Evaluate the number of usable antennas for a given model.
    
    Args:
        test_loader: DataLoader for test data
        model: Trained denoising model
        model_path: Path to the model file
        device: Device to run inference on ('cpu' or 'cuda')
        snr_threshold: SNR threshold for usable antennas (default: 4.0)
        t_tol_ns: Timing tolerance in nanoseconds (default: 5.0)
    
    Returns:
        Dictionary with evaluation results
    """
    with torch.no_grad():
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        
        clean_traces, noisy_traces, denoised_traces = [], [], []
        
        print("Processing test data...")
        for batch_idx, (noisy_data, clean_data) in enumerate(test_loader):
            if (batch_idx + 1) % 1000 == 0:
                print(f"Processed {batch_idx + 1} batches...")
            
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            denoised_output = model(noisy_data)
            
            batch_size = clean_data.shape[0]
            for sample_idx in range(batch_size):
                for channel_idx in range(3): 
                    # Collect all traces from all channels (treat as independent antennas)
                    clean_traces.append(
                        clean_data[sample_idx, channel_idx].cpu().numpy()
                    )
                    noisy_traces.append(
                        noisy_data[sample_idx, channel_idx].cpu().numpy()
                    )
                    denoised_traces.append(
                        denoised_output[sample_idx, channel_idx].cpu().numpy()
                    )
        
        # Convert to numpy arrays: shape (N_total_traces, T_samples)
        clean_traces = np.array(clean_traces)
        noisy_traces = np.array(noisy_traces)
        denoised_traces = np.array(denoised_traces)
        
        print(f"\nTotal traces shape: {clean_traces.shape} (N_traces, T_samples)")
        
        # Evaluate usable antennas
        dt_ns = 0.5   # ns/sample, the value stated in the paper (was 1.0)
        noise_slice = slice(0, 512)  # First 512 samples are noise-only
        
        print(f"\nEvaluating usable antennas...")
        print(f"SNR threshold: {snr_threshold}")
        print(f"Timing tolerance: {t_tol_ns} ns")
        
        results = evaluate_usable_antennas_snr4(
            traces_clean=clean_traces,
            traces_noisy=noisy_traces,
            traces_denoised=denoised_traces,
            dt_ns=dt_ns,
            noise_slice=noise_slice,
            snr_threshold=snr_threshold,
            t_tol_ns=t_tol_ns,
            snr_bins=np.arange(0., 11., 1.),
        )
        
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        for key, value in results["summary"].items():
            print(f"{key}: {value}")
        print("="*60)
        
        return results


def main():
    """Main function to run the evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate usable antennas')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the model file (.pth)')
    parser.add_argument('--metrics-json', type=str, required=True,
                        help='Path to metrics JSON file')
    parser.add_argument('--config-json', type=str, required=True,
                        help='Path to config JSON file')
    parser.add_argument('--data-path', type=str, 
                        default="/sps/grand/blevy/sims/sims_for_denoising_sept2025",
                        help='Path to simulation data')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use for inference')
    parser.add_argument('--snr-threshold', type=float, default=4.0,
                        help='SNR threshold for usable antennas')
    parser.add_argument('--t-tol-ns', type=float, default=5.0,
                        help='Timing tolerance in nanoseconds')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for DataLoader')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of workers for DataLoader (0 to avoid file descriptor issues)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("USABLE ANTENNAS EVALUATION")
    print("="*60)
    print(f"Model path: {args.model_path}")
    print(f"Data path: {args.data_path}")
    print(f"Device: {args.device}")
    print(f"SNR threshold: {args.snr_threshold}")
    print(f"Timing tolerance: {args.t_tol_ns} ns")
    print("="*60)
    
    # Load data
    print("\nLoading simulation data...")
    noise_signals, clean_signals = produce_noise_and_noiseless_data(args.data_path)
    
    total_samples_train = clean_signals.shape[1]
    train_indices, valid_indices, test_indices = split_indices(
        total_samples_train, train_frac=0.8, valid_frac=0.1
    )
    
    print(f"Total samples: {total_samples_train}")
    print(f"Test samples: {len(test_indices)}")
    
    # Create test dataset and loader
    test_dataset = CustomDataset(
        clean_signals, 
        [noise_signals], 
        indices=test_indices, 
        swap_prob=0.0,
        no_random=True, 
        target_start=120, 
        target_end=480, 
        voltage_to_adc=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        shuffle=False, 
        pin_memory=(args.device == 'cuda')
    )
    
    # Load model
    print("\nLoading model...")
    model_classes = {"CNN": CNN}
    
    metrics, config, model = load_best_trial_from_files(
        metrics_json_path=args.metrics_json,
        config_json_path=args.config_json,
        model_path=args.model_path,
        model_classes=model_classes,
        device=args.device
    )
    
    print("Model loaded successfully!")
    
    # Evaluate
    results = number_of_usable_antennas(
        test_loader=test_loader,
        model=model,
        model_path=args.model_path,
        device=args.device,
        snr_threshold=args.snr_threshold,
        t_tol_ns=args.t_tol_ns
    )
    
        # Plot comparison
    save_path = "/pbs/home/o/omacias/Sam_project/raytune_lib_sept25/evaluate_antenna"
    plot_usable_fraction_comparison(results, save_path=save_path)
    plot_cumulative_improvement(results, save_path=save_path)
    print("\nEvaluation completed!")
    
    return results


if __name__ == "__main__":
    main()