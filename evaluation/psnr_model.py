"""
PSNR evaluation and model comparison scripts.
"""
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from training.models.cnn import DualBranchAutoencoder as CNN
from training.raytune_training_function import CustomDataset, split_indices
from utils.data_preprocessing import produce_noise_and_noiseless_data
from evaluation.model_comparison import load_models_v2, model_comparison_analysis

def plot_average_psnr_by_objective(
    psnr_values_models,
    channel_names=('X Channel', 'Y Channel', 'Z Channel'),
    objective_order=None,
    objective_label_map=None,
    save_path=None,
    figure_name='Average_PSNR_by_Objective.pdf',
    annotate=True,
):
    """
    Draw a grouped bar chart of average PSNR per objective function for each channel.

    Args:
        psnr_values_models: Dict from model_comparison_analysis
            {objective: {'X Channel': [..], 'Y Channel': [..], 'Z Channel': [..]}}
        channel_names: Iterable of channel labels to plot and order.
        objective_order: Optional list to control bar order (default: dict order).
        objective_label_map: Optional dict mapping internal keys -> pretty labels.
        save_path: Folder to write the figure; skipped if None.
        figure_name: Filename (PDF/PNG/etc).
        annotate: Whether to place mean PSNR text on each bar.
    """
    if objective_order is None:
        objective_order = list(psnr_values_models.keys())
    if objective_label_map is None:
        objective_label_map = {name: name for name in objective_order}
    fontsize = 20
    colors = plt.cm.tab20(np.linspace(0, 1, len(objective_order)))
    fig, axes = plt.subplots(
        1, len(channel_names), figsize=(12, 8), sharey=True
    )
    if len(channel_names) == 1:
        axes = [axes]

    for ax, channel in zip(axes, channel_names):
        means, stds = [], []
        for name in objective_order:
            values = np.array(psnr_values_models.get(name, {}).get(channel, []), dtype=float)
            values = values[~np.isnan(values)]
            if values.size:
                means.append(values.mean())
                stds.append(values.std())
            else:
                means.append(np.nan)
                stds.append(np.nan)

        x = np.arange(len(objective_order))
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=6,
            color=colors,
            edgecolor='black',
            alpha=0.9,
            error_kw={'linestyle': '--', 'linewidth': 0.5, 'alpha': 0.4}  # Add this line
        )
        if annotate:
            for bar, val, std_val in zip(bars, means, stds):
                if np.isfinite(val):
                    # Display mean value
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + 0.2,
                        f'{val:.1f}',
                        ha='center',
                        va='bottom',
                        fontsize=fontsize,
                    )
                    # Display std value below the mean
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val - std_val - 0.5,  # Position below the bar
                        f'±{std_val:.1f}',
                        ha='center',
                        va='top',
                        fontsize=fontsize,
                        color='black',
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [objective_label_map.get(name, name) for name in objective_order],
            rotation=35,
            ha='right',
            fontsize=fontsize
        )
        if ax is axes[0]:
            ax.set_ylabel('Average PSNR', fontsize=fontsize)
            ax.set_yticklabels(fontsize=fontsize)
        ax.set_title(channel, fontsize=fontsize)
        ax.grid(axis='y', alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        save_full_path = os.path.join(save_path, figure_name)
        fig.savefig(save_full_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_full_path}")
    plt.show()

def main():
    # --- 1. Data Loading ---
    print("Loading simulation data...")
    Sim_data = "/sps/grand/blevy/sims/sims_for_denoising_sept2025"
    noise_signals, clean_signals = produce_noise_and_noiseless_data(Sim_data)
    
    total_samples_train = clean_signals.shape[1]
    _, _, test_indices = split_indices(total_samples_train, train_frac=0.8, valid_frac=0.1)
    
    # Create Dataset and DataLoader
    test_dataset = CustomDataset(
        clean_signals, 
        [noise_signals], 
        indices=test_indices, 
        swap_prob=0.5, 
        target_start=120, 
        target_end=480, 
        voltage_to_adc=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        num_workers=4, 
        shuffle=False, 
        pin_memory=True
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device set to: {device}")

    # --- 2. Model Setup ---
    parent_dir = '/sps/grand/macias/Sam_Result'
    save_path = "/pbs/home/o/omacias/Sam_project/raytune_lib_sept25"
    
    model_classes = {"CNN": CNN}
    
    # Define paths to models
    model_paths = {
        'CNN_multi_mse_loss': (
            os.path.join(parent_dir, 'multi_v2_CNN_100epochs_36samples', 'best_trial_metrics.json'),
            os.path.join(parent_dir, 'multi_v2_CNN_100epochs_36samples', 'best_trial_config.json'),
            os.path.join(parent_dir, 'multi_v2_CNN_100epochs_36samples', 'best_model.pth')
        ),
        'CNN_multi_l1_loss': (
            os.path.join(parent_dir, 'multi_v3_CNN_100epochs_36samples', 'best_trial_metrics.json'),
            os.path.join(parent_dir, 'multi_v3_CNN_100epochs_36samples', 'best_trial_config.json'),
            os.path.join(parent_dir, 'multi_v3_CNN_100epochs_36samples', 'best_model.pth')
        ),
        'CNN_mse': (
            os.path.join(parent_dir, 'mse_CNN_100epochs_36samples', 'best_trial_metrics.json'),
            os.path.join(parent_dir, 'mse_CNN_100epochs_36samples', 'best_trial_config.json'),
            os.path.join(parent_dir, 'mse_CNN_100epochs_36samples', 'best_model.pth')
        ),
        'CNN_l1': (
            os.path.join(parent_dir, 'l1_CNN_100epochs_36samples', 'best_trial_metrics.json'),
            os.path.join(parent_dir, 'l1_CNN_100epochs_36samples', 'best_trial_config.json'),
            os.path.join(parent_dir, 'l1_CNN_100epochs_36samples', 'best_model.pth')
        ),
        'CNN_psnr': (
            os.path.join(parent_dir, 'psnr_CNN_100epochs_36samples', 'best_trial_metrics.json'),
            os.path.join(parent_dir, 'psnr_CNN_100epochs_36samples', 'best_trial_config.json'),
            os.path.join(parent_dir, 'psnr_CNN_100epochs_36samples', 'best_model.pth')
        )
    }

    print("Loading models...")
    # Load on CPU initially to avoid OOM if loading many models, then move to device in analysis
    models = load_models_v2(model_paths, model_classes, device=device)
    
    objective_labels = {
        'CNN_multi_mse_loss': 'Multi MSE',
        'CNN_multi_l1_loss': 'Multi L1',
        'CNN_mse': 'MSE',
        'CNN_l1': 'L1',
        'CNN_psnr': 'PSNR'
    }

    # --- 3. Low SNR Analysis (1 < SNR < 4) ---
    print("\nRunning Low SNR Analysis (1 < SNR < 4)...")
    (peak_times_low, peak_amplitudes_low, snr_values_low, psnr_values_low, 
     sc_values_low, band_energy_ratio_values_low) = model_comparison_analysis(
        dataloader=test_loader,
        models=models,
        device=device,
        min_snr=1,
        max_snr=4,
        save_path=save_path
    )
    
    print("Plotting Low SNR results...")
    plot_average_psnr_by_objective(
        psnr_values_low,
        channel_names=('X Channel', 'Y Channel', 'Z Channel'),
        objective_order=list(objective_labels.keys()),
        objective_label_map=objective_labels,
        save_path=save_path,
        figure_name='Average_PSNR_by_Objective_Low_SNR.pdf'
    )

    # --- 4. High SNR Analysis (SNR > 4) ---
    print("\nRunning High SNR Analysis (SNR > 4)...")
    (peak_times_high, peak_amplitudes_high, snr_values_high, psnr_values_high, 
     sc_values_high, band_energy_ratio_values_high) = model_comparison_analysis(
        dataloader=test_loader,
        models=models,
        device=device,
        min_snr=4,
        max_snr=1000,
        save_path=save_path
    )
    
    print("Plotting High SNR results...")
    plot_average_psnr_by_objective(
        psnr_values_high,
        channel_names=('X Channel', 'Y Channel', 'Z Channel'),
        objective_order=list(objective_labels.keys()),
        objective_label_map=objective_labels,
        save_path=save_path,
        figure_name='Average_PSNR_by_Objective_High_SNR.pdf'
    )
    
    print("\nAnalysis completed successfully.")

if __name__ == "__main__":
    main()