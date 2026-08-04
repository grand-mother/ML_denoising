"""
Physics impact analysis and evaluation of usable antennas.
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import json
import pywt

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import project modules
from training.models.cnn import DualBranchAutoencoder as CNN
from training.raytune_training_function import CustomDataset, split_indices
from utils.data_preprocessing import produce_noise_and_noiseless_data

# Set multiprocessing sharing strategy
torch.multiprocessing.set_sharing_strategy('file_system')

###############################################################################
# Helper functions (keep your existing helper functions)
###############################################################################

def apply_wavelet_denoising(traces, wavelet='db4', mode='soft', threshold_mode='sure'):
    """
    Apply wavelet denoising using Donoho-Johnstone thresholding.
    """
    traces = np.asarray(traces)
    if traces.ndim == 1:
        traces = traces[None, :]
        was_1d = True
    else:
        was_1d = False
    
    denoised = np.zeros_like(traces)
    for i, trace in enumerate(traces):
        coeffs = pywt.wavedec(trace, wavelet, mode='symmetric')
        # Estimate noise level from finest detail coefficients
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        # Apply threshold
        threshold = sigma * np.sqrt(2 * np.log(len(trace)))
        coeffs_thresh = [pywt.threshold(c, threshold, mode=mode) for c in coeffs]
        denoised[i] = pywt.waverec(coeffs_thresh, wavelet, mode='symmetric')
    
    return denoised.squeeze()

def _ensure_traces_shape(traces):
    """
    Ensure traces have shape (N, P, T).
    Accepts:
      - (N, T)          -> interpreted as single polarization
      - (N, P, T)       -> left as-is
    """
    traces = np.asarray(traces)
    if traces.ndim == 2:  # (N, T)
        traces = traces[:, None, :]  # (N, 1, T)
    elif traces.ndim != 3:
        raise ValueError(f"traces must have shape (N, T) or (N, P, T), got {traces.shape}")
    return traces


def compute_peak_amplitude_and_index(traces):
    """
    Compute peak amplitude and peak index per trace.

    Parameters
    ----------
    traces : array-like, shape (N, T) or (N, P, T)
        Time-domain traces. If multiple polarizations are present, we treat
        the antenna as usable if ANY polarization has a strong pulse, and we
        define the peak as the global maximum over (P, T).

    Returns
    -------
    peak_amp : ndarray, shape (N,)
        Maximum absolute amplitude across all polarizations and time samples.
    peak_idx : ndarray, shape (N,)
        Index (0..T-1) of the time sample where the global maximum occurs.
        This is effectively the "trigger time" for that antenna.
    """
    traces = _ensure_traces_shape(traces)  # (N, P, T)
    abs_traces = np.abs(traces)           # (N, P, T)

    # Max over polarizations -> (N, T)
    abs_max_over_pol = abs_traces.max(axis=1)

    # Argmax over time -> (N,)
    peak_idx = abs_max_over_pol.argmax(axis=-1)

    # Peak amplitude at those indices
    peak_amp = abs_max_over_pol[np.arange(abs_max_over_pol.shape[0]), peak_idx]
    return peak_amp, peak_idx


def compute_efficiencies_per_snr(
    snr,
    clean_traces,
    rec_traces,
    snr_bins,
    amp_threshold,
    t_max_ns,
    dt_ns,
):
    """
    Compute physics-usable efficiencies in SNR bins for a given reconstruction.

    Physics-usable definition:
        - true signal present: A_peak_clean > amp_threshold
        - recovered amplitude above threshold: A_peak_rec > amp_threshold
        - timing error within |Δt_peak| <= t_max_ns

    Parameters
    ----------
    snr : ndarray, shape (N,)
        SNR per trace (antenna).
    clean_traces : array-like, shape (N, T) or (N, P, T)
        Clean (signal-only) traces.
    rec_traces : array-like, shape (N, T) or (N, P, T)
        Reconstructed traces (ML or Hilbert).
    snr_bins : ndarray, shape (M+1,)
        Bin edges for SNR. Efficiencies are computed per bin [bin_i, bin_{i+1}).
    amp_threshold : float
        Detection threshold (e.g. 15 ADC).
    t_max_ns : float
        Maximum allowed absolute peak-time error for physics-usable antennas.
    dt_ns : float
        Time step between consecutive samples, in nanoseconds.

    Returns
    -------
    bin_centers : ndarray, shape (M,)
        Centers of the SNR bins.
    eps_phys : ndarray, shape (M,)
        Physics-usable efficiency per SNR bin (0–1).
    counts_true : ndarray, shape (M,)
        Number of true-signal antennas (denominator) per bin.
    """
    snr = np.asarray(snr)
    clean_traces = _ensure_traces_shape(clean_traces)
    rec_traces = _ensure_traces_shape(rec_traces)

    if snr.shape[0] != clean_traces.shape[0]:
        raise ValueError("snr and clean_traces must have the same number of traces (N).")
    if clean_traces.shape != rec_traces.shape:
        raise ValueError("clean_traces and rec_traces must have the same shape.")

    N = snr.shape[0]
    n_bins = len(snr_bins) - 1

    # Peak amplitudes and times
    clean_peak_amp, clean_peak_idx = compute_peak_amplitude_and_index(clean_traces)
    rec_peak_amp, rec_peak_idx = compute_peak_amplitude_and_index(rec_traces)

    # Only consider traces that truly have a signal above threshold
    has_true_signal = clean_peak_amp > amp_threshold

    # Time differences in ns
    dt_idx = rec_peak_idx - clean_peak_idx
    dt_ns_arr = dt_idx.astype(np.float64) * float(dt_ns)

    eps_phys = np.full(n_bins, np.nan, dtype=float)
    counts_true = np.zeros(n_bins, dtype=int)
    bin_centers = 0.5 * (snr_bins[:-1] + snr_bins[1:])

    for i in range(n_bins):
        lo, hi = snr_bins[i], snr_bins[i + 1]
        in_bin = (snr >= lo) & (snr < hi) & has_true_signal
        denom = in_bin.sum()
        counts_true[i] = denom

        if denom == 0:
            continue  # leave NaN

        # Physics-usable: A_rec > threshold AND |Δt| <= t_max_ns
        sel_phys = in_bin & (rec_peak_amp > amp_threshold) & (np.abs(dt_ns_arr) <= t_max_ns)
        eps_phys[i] = sel_phys.sum() / denom

    return bin_centers, eps_phys, counts_true

def compute_global_physics_usable_fraction(
    snr,
    clean_traces,
    rec_traces,
    amp_threshold,
    t_max_ns,
    dt_ns,
):
    """
    Compute a single global physics-usable fraction (over all SNR).

    Returns
    -------
    frac_phys : float
        Fraction (0–1) of true-signal antennas that are physics-usable.
    """
    snr = np.asarray(snr)
    clean_traces = _ensure_traces_shape(clean_traces)
    rec_traces = _ensure_traces_shape(rec_traces)

    if snr.shape[0] != clean_traces.shape[0]:
        raise ValueError("snr and clean_traces must have the same number of traces (N).")
    if clean_traces.shape != rec_traces.shape:
        raise ValueError("clean_traces and rec_traces must have the same shape.")

    clean_peak_amp, clean_peak_idx = compute_peak_amplitude_and_index(clean_traces)
    rec_peak_amp, rec_peak_idx = compute_peak_amplitude_and_index(rec_traces)

    has_true_signal = clean_peak_amp > amp_threshold

    dt_idx = rec_peak_idx - clean_peak_idx
    dt_ns_arr = dt_idx.astype(np.float64) * float(dt_ns)

    mask_true = has_true_signal
    denom = mask_true.sum()
    if denom == 0:
        return np.nan

    mask_phys = (
        mask_true
        & (rec_peak_amp > amp_threshold)
        & (np.abs(dt_ns_arr) <= t_max_ns)
    )
    num = mask_phys.sum()

    return num / denom


###############################################################################
# High-level function: two-panel figure, percentages
###############################################################################

def plot_usable_antennas_single_panel(
    snr,
    clean_traces,
    ml_traces,
    hilbert_traces,
    amp_threshold=2.0,
    t_max_ns_list=[10.0, 20.0],
    dt_ns=0.5,   # ns/sample, the value stated in the paper (was 1.0)
    snr_min=1.0,
    snr_max=10.0,
    snr_step=0.5,
    hilbert_label="Hilbert",
    save_path=None,
):
    """
    Create separate plots for each t_max_ns value.
    Each plot shows ML vs Hilbert comparison.
    """
    snr = np.asarray(snr)
    clean_traces = _ensure_traces_shape(clean_traces)
    ml_traces = _ensure_traces_shape(ml_traces)
    hilbert_traces = _ensure_traces_shape(hilbert_traces)

    if not (
        snr.shape[0]
        == clean_traces.shape[0]
        == ml_traces.shape[0]
        == hilbert_traces.shape[0]
    ):
        raise ValueError("snr, clean_traces, ml_traces, and hilbert_traces must have same N.")

    # Define SNR bins
    snr_bins = np.arange(snr_min, snr_max + snr_step, snr_step)

    all_results = {}

    # Create a separate figure for each t_max_ns value
    for t_max_ns in t_max_ns_list:
        fig, ax = plt.subplots(figsize=(12, 10))

        # Per-bin physics-usable efficiencies
        centers, eps_phys_ml, counts_true = compute_efficiencies_per_snr(
            snr,
            clean_traces,
            ml_traces,
            snr_bins,
            amp_threshold,
            t_max_ns,
            dt_ns,
        )

        _, eps_phys_hilbert, _ = compute_efficiencies_per_snr(
            snr,
            clean_traces,
            hilbert_traces,
            snr_bins,
            amp_threshold,
            t_max_ns,
            dt_ns,
        )

        # Convert to percentages
        perc_phys_ml = 100.0 * eps_phys_ml
        perc_phys_hilbert = 100.0 * eps_phys_hilbert

        # Plot ML curve
        ax.plot(
            centers,
            perc_phys_ml,
            marker="o",
            linestyle="-",
            color='tab:blue',
            label=f"ML: amp + |Δt| ≤ {t_max_ns:.0f} ns",
        )
        
        # Plot Hilbert curve
        ax.plot(
            centers,
            perc_phys_hilbert,
            marker="s",
            linestyle="--",
            color='tab:orange',
            label=f"{hilbert_label}: amp + |Δt| ≤ {t_max_ns:.0f} ns",
        )
        fontsize = 27
        ax.axhline(95.0, linestyle=":", linewidth=1.0, color='gray', label="95% threshold")
        ax.set_xlabel("SNR", fontsize=fontsize)
        ax.set_ylabel("Usable antennas [%]", fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        # ax.set_title(f"|Δt| ≤ {int(t_max_ns)} ns")
        ax.set_ylim(0.0, 105.0)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right", fontsize=fontsize)

        fig.tight_layout()

        # Save figure
        if save_path:
            save_file = os.path.join(save_path, f"physics_efficiency_{int(t_max_ns)}_ns.pdf")
            fig.savefig(save_file, bbox_inches='tight')
            print(f"Figure saved to: {save_file}")

        # Store results
        all_results[f"t_max_{int(t_max_ns)}ns"] = {
            "eps_phys_ml": eps_phys_ml,
            "eps_phys_hilbert": eps_phys_hilbert,
        }

    results = {
        "snr_bins": snr_bins,
        "snr_centers": centers,
        "counts_true": counts_true,
        "per_t_max": all_results,
    }
    return results


###############################################################################
# Model loading function
###############################################################################

def load_best_trial_from_files(metrics_json_path: str,
                               config_json_path: str,
                               model_path: str,
                               model_classes: dict,
                               device: str = "cpu"):
    """
    Load best trial metrics, config, and the trained model.
    """
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

    ModelClass = model_classes[model_type]
    model = ModelClass(model_config)
    model.to(device)
    model.eval()

    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    return metrics, config, model


###############################################################################
# Function to collect traces from DataLoader
###############################################################################

def collect_traces_from_dataloader(dataloader, model, device):
    """
    Collect clean, noisy, ML-denoised, and Hilbert-filtered traces from dataloader.
    
    Returns:
        clean_traces: (N, T) array
        noisy_traces: (N, T) array  
        ml_traces: (N, T) array - ML model output
        hilbert_traces: (N, T) array - Hilbert envelope of noisy data
        snr_values: (N,) array - SNR per trace
    """
    clean_list, noisy_list, ml_list, hilbert_list, snr_list = [], [], [], [], []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (noisy_data, clean_data) in enumerate(dataloader):
            if (batch_idx + 1) % 1000 == 0:
                print(f"Processed {batch_idx + 1} batches...")
            
            noisy_data = noisy_data.to(device)
            clean_data = clean_data.to(device)
            
            # ML denoising
            ml_output = model(noisy_data)
            
            # Convert to numpy
            noisy_np = noisy_data.cpu().numpy()
            clean_np = clean_data.cpu().numpy()
            ml_np = ml_output.cpu().numpy()
            
            # Hilbert envelope of noisy data
            hilbert_np = np.abs(hilbert(noisy_np, axis=-1))
            
            batch_size = clean_np.shape[0]
            for sample_idx in range(batch_size):
                for channel_idx in range(3):  # X, Y, Z channels
                    clean_trace = clean_np[sample_idx, channel_idx]
                    noisy_trace = noisy_np[sample_idx, channel_idx]
                    ml_trace = ml_np[sample_idx, channel_idx]
                    hilbert_trace = hilbert_np[sample_idx, channel_idx]
                    
                    # Compute SNR: max(clean) / std(noisy)
                    noise_std = np.std(noisy_trace)
                    if noise_std > 0:
                        snr = np.max(np.abs(clean_trace)) / noise_std
                    else:
                        snr = float('inf')
                    
                    clean_list.append(clean_trace)
                    noisy_list.append(noisy_trace)
                    ml_list.append(ml_trace)
                    hilbert_list.append(hilbert_trace)
                    snr_list.append(snr)
    
    return (np.array(clean_list), np.array(noisy_list), 
            np.array(ml_list), np.array(hilbert_list), np.array(snr_list))


###############################################################################
# Main function
###############################################################################

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Physics impact analysis for denoisers')
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
                        choices=['cpu', 'cuda'], help='Device for inference')
    parser.add_argument('--amp-threshold', type=float, default=2.0,
                        help='Amplitude detection threshold')
    parser.add_argument('--t-max-ns', type=float, default=10.0,
                        help='Maximum allowed |Δt_peak| in ns')
    parser.add_argument('--snr-min', type=float, default=1.0,
                        help='Minimum SNR for efficiency curves')
    parser.add_argument('--snr-max', type=float, default=10.0,
                        help='Maximum SNR for efficiency curves')
    parser.add_argument('--snr-step', type=float, default=0.5,
                        help='SNR bin width')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for DataLoader')
    parser.add_argument('--save-path', type=str, 
                        default='/pbs/home/o/omacias/Sam_project/raytune_lib_sept25/evaluate_antenna',
                        help='Path to save output figures')
    
    args = parser.parse_args()
    
    print("="*60)
    print("PHYSICS IMPACT ANALYSIS")
    print("="*60)
    print(f"Model path: {args.model_path}")
    print(f"Data path: {args.data_path}")
    print(f"Device: {args.device}")
    print(f"Amplitude threshold: {args.amp_threshold}")
    print(f"Timing tolerance: {args.t_max_ns} ns")
    print("="*60)
    
    # Load data
    print("\nLoading simulation data...")
    noise_signals, clean_signals = produce_noise_and_noiseless_data(args.data_path)
    
    total_samples = clean_signals.shape[1]
    train_indices, valid_indices, test_indices = split_indices(
        total_samples, train_frac=0.8, valid_frac=0.1
    )
    
    print(f"Total samples: {total_samples}")
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
        num_workers=0, 
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
    
    # Collect traces
    print("\nCollecting traces...")
    device = torch.device(args.device)
    model = model.to(device)
    
    clean_traces, noisy_traces, ml_traces, hilbert_traces, snr_values = \
        collect_traces_from_dataloader(test_loader, model, device)
    
    print(f"Collected {len(clean_traces)} traces")
    print(f"Clean traces shape: {clean_traces.shape}")
    print(f"SNR range: [{snr_values.min():.2f}, {snr_values.max():.2f}]")
    
        # Run physics impact analysis
    print("\nRunning physics impact analysis...")
    results = plot_usable_antennas_single_panel(
        snr=snr_values,
        clean_traces=clean_traces,
        ml_traces=ml_traces,
        hilbert_traces=hilbert_traces,
        amp_threshold=args.amp_threshold,
        t_max_ns_list=[10.0],
        dt_ns=0.5,   # ns/sample, the value stated in the paper (was 1.0)
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        snr_step=args.snr_step,
        hilbert_label="Hilbert",
        save_path=args.save_path,
    )

# Remove the old save logic (it's now handled inside the function)
    plt.show()
    
    if args.save_path:
        import os
        fig = plt.gcf()
        save_file = os.path.join(args.save_path, "physics_efficiency_comparison.pdf")
        fig.savefig(save_file, bbox_inches='tight')
        print(f"Figure saved to: {save_file}")
    
    # Print summary
    # print("\n" + "="*60)
    # print("RESULTS SUMMARY")
    # print("="*60)
    # valid_bins = ~np.isnan(results["eps_phys_ml"])
    # if valid_bins.any():
    #     avg_ml = np.nanmean(results["eps_phys_ml"])
    #     avg_baseline = np.nanmean(results["eps_phys_baseline"])
    #     avg_ratio = np.nanmean(results["ratio_phys"])
    #     print(f"Average physics-usable efficiency (ML): {avg_ml:.3f}")
    #     print(f"Average physics-usable efficiency (Hilbert): {avg_baseline:.3f}")
    #     print(f"Average efficiency ratio (ML/Hilbert): {avg_ratio:.3f}")
    # print("="*60)
    
    plt.show()
    
    return results


if __name__ == "__main__":
    main()