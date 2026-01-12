"""
Utility functions for loading and converting JSON configurations to Ray Tune format.
"""

import json
import ray.tune as tune
from pathlib import Path


def load_json_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def convert_to_tune_config(json_config):
    """
    Convert JSON configuration to Ray Tune configuration format.
    
    Args:
        json_config (dict): Configuration dictionary from JSON file
        
    Returns:
        dict: Ray Tune compatible configuration
    """
    tune_config = {}
    
    for key, value in json_config.items():
        if isinstance(value, dict) and "type" in value:
            # Handle tune parameters
            if value["type"] == "choice":
                tune_config[key] = tune.choice(value["values"])
            elif value["type"] == "loguniform":
                tune_config[key] = tune.loguniform(value["min"], value["max"])
            elif value["type"] == "uniform":
                tune_config[key] = tune.uniform(value["min"], value["max"])
            elif value["type"] == "randint":
                tune_config[key] = tune.randint(value["min"], value["max"])
            else:
                # Fallback to direct value
                tune_config[key] = value
        elif isinstance(value, dict):
            # Recursively handle nested dictionaries
            tune_config[key] = convert_to_tune_config(value)
        else:
            # Direct value assignment
            tune_config[key] = value
    
    return tune_config


def save_best_trial_results(best_trial, output_path, model_type):
    """
    Save best trial results to JSON files.
    
    Args:
        best_trial: Best trial object from Ray Tune
        output_path (str): Path to save the results
        model_type (str): Type of model used
    """
    import os
    import json
    import ray.cloudpickle as pickle
    from pathlib import Path
    
    # Save best trial configuration
    best_config_path = os.path.join(output_path, 'best_trial_config.json')
    with open(best_config_path, 'w') as f:
        json.dump(best_trial.config, f, indent=2)
    
    # Save best trial metrics
    best_metrics_path = os.path.join(output_path, 'best_trial_metrics.json')
    metrics_data = {
        'validation_loss': best_trial.last_result.get('loss', None),
        'validation_psnr': best_trial.last_result.get('validation_psnr', None),
        'training_loss': best_trial.last_result.get('training_loss', None),
        'epoch': best_trial.last_result.get('epoch', None),
        'model_type': model_type,
        'trial_id': best_trial.trial_id,
        'local_path': best_trial.local_path
    }
    
    with open(best_metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    return best_config_path, best_metrics_path


def save_detailed_metrics(metrics_data, output_path):
    """
    Save detailed metrics data to JSON file.
    
    Args:
        metrics_data (dict): Detailed metrics from training
        output_path (str): Path to save the metrics
    """
    import os
    import json
    
    detailed_metrics_path = os.path.join(output_path, 'detailed_metrics.json')
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_metrics = {}
    for key, value in metrics_data.items():
        if hasattr(value, 'tolist'):  # numpy array
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, list):
            serializable_metrics[key] = value
        else:
            serializable_metrics[key] = value
    
    with open(detailed_metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    return detailed_metrics_path
