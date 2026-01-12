#!/usr/bin/env python3
"""
common_ml_utils.py

Shared utilities for loading ML models and running inference
for figure generation scripts.
"""

from __future__ import annotations

import os
import sys
import json
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

# Set up paths
os.environ['PYTHONPATH'] = '/pbs/home/o/omacias/Sam_project'
sys.path.append('/pbs/home/o/omacias/Sam_project')

from training.models.cnn import DualBranchAutoencoder as CNN
from training.raytune_training_function import CustomDataset, split_indices
from utils.data_preprocessing import produce_noise_and_noiseless_data


@dataclass
class EvalPack:
    """Standard data pack for evaluation figures."""
    clean: np.ndarray      # (N, C, L)
    noisy: np.ndarray      # (N, C, L)
    denoised: np.ndarray   # (N, C, L)
    snr: np.ndarray        # (N,)


def load_model_from_files(
    model_path: str,
    metrics_json: str,
    config_json: str,
    device: str = "cpu"
) -> torch.nn.Module:
    """
    Load trained model from checkpoint files.
    
    Args:
        model_path: Path to .pth model weights
        metrics_json: Path to metrics JSON
        config_json: Path to config JSON
        device: Device for inference
    
    Returns:
        Loaded model in eval mode
    """
    model_classes = {"CNN": CNN}
    
    with open(config_json, "r") as f:
        config = json.load(f)
    
    model_type = config.get("model_type")
    model_config = config.get("model_config")
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model_type '{model_type}'. Available: {list(model_classes.keys())}")
    if model_config is None:
        raise ValueError("model_config missing in config JSON")
    
    ModelClass = model_classes[model_type]
    model = ModelClass(model_config)
    model.to(device)
    model.eval()
    
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    
    return model


def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> np.ndarray:
    """
    Compute SNR per sample: max(|clean|) / std(noise).
    
    Args:
        clean: (N, C, L) clean waveforms
        noisy: (N, C, L) noisy waveforms
    
    Returns:
        snr: (N,) SNR values
    """
    noise = noisy - clean
    # Use channel-combined metrics
    clean_peak = np.max(np.abs(clean), axis=(1, 2))  # (N,)
    noise_std = np.std(noise, axis=(1, 2))           # (N,)
    snr = clean_peak / (noise_std + 1e-12)
    return snr


def run_inference(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str = "cpu",
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model inference on DataLoader.
    
    Returns:
        clean: (N, C, L)
        noisy: (N, C, L)
        denoised: (N, C, L)
        snr: (N,)
    """
    clean_list, noisy_list, denoised_list = [], [], []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (noisy_batch, clean_batch) in enumerate(data_loader):
            if verbose and (batch_idx + 1) % 500 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
            
            noisy_batch = noisy_batch.to(device)
            denoised_batch = model(noisy_batch)
            
            clean_list.append(clean_batch.numpy())
            noisy_list.append(noisy_batch.cpu().numpy())
            denoised_list.append(denoised_batch.cpu().numpy())
    
    clean = np.concatenate(clean_list, axis=0)
    noisy = np.concatenate(noisy_list, axis=0)
    denoised = np.concatenate(denoised_list, axis=0)
    snr = compute_snr(clean, noisy)
    print(f"Inference complete. Shape: {clean.shape}, SNR range: [{snr.min():.2f}, {snr.max():.2f}]")
    return clean, noisy, denoised, snr


def load_data_and_run_inference(
    model_path: str,
    metrics_json: str,
    config_json: str,
    data_path: str = "/sps/grand/blevy/sims/sims_for_denoising_sept2025",
    device: str = "cpu",
    batch_size: int = 32,
    test_only: bool = True,
    max_samples: Optional[int] = None,
    verbose: bool = True
) -> EvalPack:
    """
    Complete pipeline: load model, load data, run inference.
    
    Args:
        model_path: Path to .pth model
        metrics_json: Path to metrics JSON
        config_json: Path to config JSON
        data_path: Path to simulation data
        device: Device for inference
        batch_size: Batch size for DataLoader
        test_only: If True, use only test split
        max_samples: Limit number of samples (for debugging)
        verbose: Print progress
    
    Returns:
        EvalPack with clean, noisy, denoised, snr arrays
    """
    if verbose:
        print("Loading model...")
    model = load_model_from_files(model_path, metrics_json, config_json, device)
    
    if verbose:
        print("Loading simulation data...")
    noise_signals, clean_signals = produce_noise_and_noiseless_data(data_path)
    
    total_samples = clean_signals.shape[1]
    train_indices, valid_indices, test_indices = split_indices(
        total_samples, train_frac=0.8, valid_frac=0.1
    )
    
    if test_only:
        indices = test_indices
    else:
        indices = np.concatenate([train_indices, valid_indices, test_indices])
    
    if max_samples is not None and max_samples < len(indices):
        indices = indices[:max_samples]
    
    if verbose:
        print(f"Using {len(indices)} samples for evaluation")
    
    dataset = CustomDataset(
        clean_signals,
        [noise_signals],
        indices=indices,
        swap_prob=0.0,  # No augmentation for evaluation
        target_start=120,
        target_end=480,
        voltage_to_adc=True
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=(device == "cuda")
    )
    
    if verbose:
        print("Running inference...")
    clean, noisy, denoised, snr = run_inference(model, data_loader, device, verbose)
    
    if verbose:
        print(f"Inference complete. Shape: {clean.shape}, SNR range: [{snr.min():.2f}, {snr.max():.2f}]")
    
    return EvalPack(clean=clean, noisy=noisy, denoised=denoised, snr=snr)


def add_model_arguments(parser) -> None:
    """Add common model-related arguments to argparse parser."""
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model .pth file (enables ML inference mode)")
    parser.add_argument("--metrics-json", type=str, default=None,
                        help="Path to metrics JSON file")
    parser.add_argument("--config-json", type=str, default=None,
                        help="Path to config JSON file")
    parser.add_argument("--data-path", type=str,
                        default="/sps/grand/blevy/sims/sims_for_denoising_sept2025",
                        help="Path to simulation data")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"], help="Device for inference")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples for debugging")


def check_model_args(args) -> bool:
    """Check if model arguments are provided (ML mode)."""
    return args.model_path is not None


def ensure_ncl(x: np.ndarray) -> np.ndarray:
    """Ensure array has shape (N, C, L)."""
    x = np.asarray(x)
    if x.ndim == 2:
        return x[:, None, :]
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected (N,L) or (N,C,L), got {x.shape}")