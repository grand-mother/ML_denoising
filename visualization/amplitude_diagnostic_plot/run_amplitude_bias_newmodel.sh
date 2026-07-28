#!/bin/bash
#SBATCH --job-name=amp_bias_q5_new        # Job name
#SBATCH --partition=gpu_v100              # Partition name
#SBATCH --time=04:00:00                   # Time limit hrs:min:sec
#SBATCH --output=amp_bias_q5_new_%j.out   # Standard output and error log
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:v100:1
#
# Amplitude-bias diagnostic (referee Q5) for the RETRAINED model
# (CNN dual-branch, multi_l1, 100 epochs x 24 samples).
# Intended to run automatically via:
#   sbatch --dependency=afterok:<train_jobid> run_amplitude_bias_newmodel.sh
# Outputs to a SEPARATE subdir so the previous (old-model) results are kept
# for an old-vs-new comparison.

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

SCRIPT_DIR="$PROJECT_ROOT/raytune_lib_final/visualization/amplitude_diagnostic_plot"
cd "$SCRIPT_DIR"

# Retrained model from this session's run.
MODEL_DIR=/sps/grand/macias/Sam_Result/multi_l1_CNN_100epochs_24samples

# Separate output dir -> does not overwrite the old-model deliverables.
OUT_DIR="$SCRIPT_DIR/new_model_multi_l1"
mkdir -p "$OUT_DIR"

# Safety: confirm the trained model was actually saved before running.
if [ ! -f "$MODEL_DIR/best_model.pth" ]; then
    echo "ERROR: $MODEL_DIR/best_model.pth not found. Training did not save a model."
    exit 1
fi

# Same SNR definition / bins / amplitude estimator as the old-model run, so the
# two figures are directly comparable.
python -u make_fig_amplitude_bias_vs_snr.py \
    --model-path  "$MODEL_DIR/best_model.pth" \
    --metrics-json "$MODEL_DIR/best_trial_metrics.json" \
    --config-json "$MODEL_DIR/best_trial_config.json" \
    --device cuda \
    --batch-size 256 \
    --out-dir "$OUT_DIR"

conda deactivate
