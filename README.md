# Radio Signal Denoising with CNN and Ray Tune

A deep learning framework for radio signal denoising using Dual-Branch CNN architectures with Ray Tune hyperparameter optimization.

## 📁 Project Structure

```
raytune_lib_final/
├── training/ # Training modules
│ ├── models/ # CNN architectures
│ │ └── cnn.py # Dual-Branch Autoencoder
│ ├── raytune_main_sept25.py # Main training entry point
│ ├── raytune_train_sept25.py # Training loop
│ ├── raytune_training_function.py # Dataset & metrics
│ └── scripts/
│ ├── run_training.sh # SLURM job script
│ └── .out # Training output logs
│
├── evaluation/ # Evaluation modules
│ ├── model_comparison.py # Model comparison analysis
│ ├── psnr_model.py # PSNR evaluation
│ ├── physics_impact/ # Physics-based evaluation
│ │ ├── physics_impact.py
│ │ ├── evaluate_usable_antennas.py
│ │ └── number_of_usable_antennas.py
│ └── scripts/
│ └── run_psnr_eval.sh # SLURM eval script
│
├── visualization/ # Plotting & visualization
│ ├── overleaf_plots.py # Publication-ready plots
│ ├── common_ml_utils.py # Common ML utilities
│ ├── notebooks/ # Jupyter notebooks
│ │ └── overleaf.ipynb # Main analysis notebook
│ ├── nmse_snr_gain_vs_snr/ # NMSE & SNR gain plots
│ │ ├── make_fig_nmse_snr_gain_vs_snr__option3_truth_cleanpower_gate.py
│ │ └── run_fig_nmse_snr_gain_vs_snr_option3.sh
│ ├── NMSE_STD_VS_SNR/ # NMSE standard deviation plots
│ │ ├── make_fig_appendix_check1_nmse_std_vs_snr.py
│ │ └── run_fig_appendix_check1_nmse_std_vs_snr.sh
│ └── plot_usable_antenna_vs_SNR/ # Usable antenna analysis
│ ├── make_fig_event_multiplicity_usable_nmse.py
│ └── make_fig_event_multiplicity_usable_nmse.sh
│
├── utils/ # Utility functions
│ ├── config_utils.py # Configuration handling
│ ├── data_preprocessing.py # Data loading
│ └── hilbert_filter.py # Signal processing
│
├── configs/ # Configuration files
│ ├── model_config.json # Model architecture config
│ ├── training_config.json # Training hyperparameters
│ ├── training_params.json # Main experiment params
│ └── README_config.md # Configuration documentation
│
├── results/ # Training results & checkpoints
├── output/ # Generated plots and figures
└── requirements.txt # Python dependencies
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

1. Edit configuration files in `configs/`:
   - `training_params.json`: Main experiment parameters
   - `model_config.json`: Model architecture search space
   - `training_config.json`: Training hyperparameter search space

2. Run training:
```bash
cd training/scripts
sbatch run_training.sh
```

Or directly:
```bash
python training/raytune_main_sept25.py --json_params_file configs/training_params.json
```

### Visualization

**Jupyter Notebook Analysis:**
```bash
jupyter notebook visualization/notebooks/overleaf.ipynb
```

**Individual Plotting Scripts:**

1. **NMSE and SNR Gain Plots:**
```bash
cd visualization/nmse_snr_gain_vs_snr
bash run_fig_nmse_snr_gain_vs_snr_option3.sh
```

2. **NMSE Standard Deviation Plots:**
```bash
cd visualization/NMSE_STD_VS_SNR
bash run_fig_appendix_check1_nmse_std_vs_snr.sh
```

3. **Usable Antenna Analysis:**
```bash
cd visualization/plot_usable_antenna_vs_SNR
bash make_fig_event_multiplicity_usable_nmse.sh
```

Generated plots are saved to the `output/` directory.


## 📊 Model Architecture

### Dual-Branch CNN (`DualBranchAutoencoder`)
- **Time Branch**: Processes signals in time domain
- **Frequency Branch**: Processes signals in frequency domain (via FFT)
- **Decoder**: Combines both branches for reconstruction

## 📈 Metrics

- **PSNR**: Peak Signal-to-Noise Ratio
- **NMSE**: Normalized Mean Squared Error
- **SNR Gain**: Signal-to-Noise Ratio improvement after denoising
- **Peak Time Accuracy**: Accuracy of peak timing reconstruction
- **Amplitude Ratio**: Ratio of reconstructed to clean signal amplitude
- **Physics Impact**: Usable antenna evaluation and physics-based metrics (timing thresholds, efficiency)

## ⚙️ Configuration

See `configs/README_config.md` for detailed configuration documentation.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `n_epochs` | Number of training epochs |
| `number_samples` | Number of Ray Tune trials |
| `model_type` | `"CNN"` (DualBranchAutoencoder) |
| `loss` | Loss function: `"l1"`, `"mse"`, `"psnr"`, `"multi_l1"`, `"multi_mse"`. |

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{
  title={Deep-Learning Denoising of Radio Observations for Ultra-High-Energy Cosmic-Ray Detection},
  author={Zhisen Lai, Oscar Macias, Aur´elien Benoit--L´evy, Ars`ene Ferri`ere, MMat´ıas Tueros},
  journal={},
  year={2026}
}
```

## 📄 License

MIT License - see LICENSE file for details.
