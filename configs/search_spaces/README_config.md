# Ray Tune Configuration System

This document describes the new JSON-based configuration system for the Ray Tune hyperparameter optimization.

## Configuration Files

### 1. Main Parameters File (`sample_params.json`)
Contains the main experiment parameters:
- `output_path`: Directory to save results
- `tag`: Experiment identifier
- `loss`: Loss function type
- `n_epochs`: Number of training epochs
- `number_samples`: Number of Ray Tune trials
- `model_type`: Type of model ("CNN" - DualBranchAutoencoder)
- `sim_data_dir`: Path to simulation data
- `training_config_path`: Path to training configuration JSON
- `model_config_path`: Path to model configuration JSON

### 2. Training Configuration (`training_config.json`)
Contains hyperparameter search spaces for training:
- `lr`: Learning rate (loguniform distribution)
- `max_lr`: Maximum learning rate for cyclic LR
- `step_size`: Step size for cyclic LR
- `lr_mode`: Cyclic LR mode
- `weight_decay`: Weight decay regularization
- `batch_size`: Batch size options
- `mag_weight`: Magnitude weight for multi-objective loss
- `phase_weight`: Phase weight for multi-objective loss
- `stft_weight`: STFT weight for multi-objective loss

### 3. Model Configuration (`model_config.json`)
Contains model architecture search spaces:
- `CNN`: Configuration for DualBranchAutoencoder

Configuration includes:
- `time_branch`: Time domain branch configuration
- `freq_branch`: Frequency domain branch configuration
- `decoder_channels`: Decoder architecture options

## Output Files

After training, the following JSON files are generated:

### 1. `best_trial_config.json`
Contains the best trial's hyperparameter configuration.

### 2. `best_trial_metrics.json`
Contains the best trial's final metrics:
- `validation_loss`: Final validation loss
- `validation_psnr`: Final validation PSNR
- `training_loss`: Final training loss
- `epoch`: Final epoch number
- `model_type`: Model type used
- `trial_id`: Ray Tune trial ID
- `local_path`: Local path to trial results

### 3. `detailed_metrics.json`
Contains detailed training metrics:
- `epochs`: List of epoch numbers
- `training_losses`: Training loss per epoch
- `validation_losses`: Validation loss per epoch
- `validation_psnr`: Validation PSNR per epoch
- `learning_rates`: Learning rate per epoch
- `peak_to_peak_ratio`: Peak-to-peak ratio per epoch

## Usage

```bash
python raytune_main_sept25.py --json_params_file sample_params.json
```

## Configuration Types

The system supports the following Ray Tune parameter types:
- `choice`: Select from a list of values
- `loguniform`: Log-uniform distribution between min and max
- `uniform`: Uniform distribution between min and max
- `randint`: Random integer between min and max

## Benefits

1. **Modularity**: Separate configuration files for different aspects
2. **Reusability**: Easy to share and modify configurations
3. **Version Control**: JSON files are easily tracked in git
4. **Documentation**: Self-documenting configuration structure
5. **Flexibility**: Easy to add new parameters without code changes
