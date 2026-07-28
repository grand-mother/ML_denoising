#!/bin/bash
#SBATCH --job-name=amp_bias_q5            # Job name
#SBATCH --partition=gpu_v100              # Partition name
#SBATCH --time=04:00:00                   # Time limit hrs:min:sec
#SBATCH --output=amp_bias_q5_%j.out       # Standard output and error log
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:v100:1
#
# Amplitude-bias diagnostic for referee question 5.
# Produces: amplitude_bias_table.csv, amplitude_bias_vs_snr.pdf,
#           amplitude_bias_per_trace.npz  (in this directory).

# Activate conda
source /pbs/throng/grand/soft/miniconda3/bin/activate
conda activate /sps/grand/macias/conda_envs/ili-torch

if [ $? -ne 0 ]; then
    echo "Failed to activate the Conda environment."
    exit 1
fi

echo "Active Conda Environment: $(conda info --envs | grep '*' | awk '{print $1}')"
python --version

# Set up paths
PROJECT_ROOT=/pbs/home/o/omacias/Sam_project
export PYTHONPATH=$PROJECT_ROOT/raytune_lib_final

# NB: under SLURM the batch script is copied to a spool dir, so ${BASH_SOURCE[0]}
# does NOT point here. Use an absolute path to the script directory.
SCRIPT_DIR="$PROJECT_ROOT/raytune_lib_final/visualization/amplitude_diagnostic_plot"
cd "$SCRIPT_DIR"

# Real, fully-trained freq+time model (100 epochs, 36 Ray Tune trials).
MODEL_DIR=/sps/grand/macias/Sam_Result/l1_CNN_100epochs_36samples

# Run the diagnostic.
# To densify the low-SNR region, add:  --snr-edges 1,1.5,2,2.5,3,4,5,6,8,10
python -u make_fig_amplitude_bias_vs_snr.py \
    --model-path  "$MODEL_DIR/best_model.pth" \
    --metrics-json "$MODEL_DIR/best_trial_metrics.json" \
    --config-json "$MODEL_DIR/best_trial_config.json" \
    --device cuda \
    --batch-size 256 \
    --out-dir "$SCRIPT_DIR"

conda deactivate
