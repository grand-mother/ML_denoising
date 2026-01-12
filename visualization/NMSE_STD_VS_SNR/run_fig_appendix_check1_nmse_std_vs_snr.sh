#!/bin/bash
#SBATCH --job-name=fig_check1
#SBATCH --partition=hpc
#SBATCH --time=12:00:00
#SBATCH --output=generate_figures_check1_%j.out
#SBATCH --ntasks=1
#SBATCH --mem=32G

# Activate conda
source /pbs/throng/grand/soft/miniconda3/bin/activate
conda init
conda activate /sps/grand/macias/conda_envs/ili-torch

if [ $? -ne 0 ]; then
    echo "Failed to activate the Conda environment."
    exit 1 
fi

echo "Active Conda Environment: $(conda info --envs | grep '*' | awk '{print $1}')"
python --version

# Set PYTHONPATH
export PYTHONPATH=/pbs/home/o/omacias/Sam_project

# Working directory
cd /pbs/home/o/omacias/Sam_project/raytune_lib_sept25/evaluate_antenna/figures

# Model configuration (for ML overlay)
PARENT_DIR="/sps/grand/macias/Sam_Result"
MODEL_NAME="l1_CNN_100epochs_36samples"

MODEL_PATH="${PARENT_DIR}/${MODEL_NAME}/best_model.pth"
METRICS_JSON="${PARENT_DIR}/${MODEL_NAME}/best_trial_metrics.json"
CONFIG_JSON="${PARENT_DIR}/${MODEL_NAME}/best_trial_config.json"
DATA_PATH="/sps/grand/blevy/sims/sims_for_denoising_sept2025"

# Common parameters
DT_NS=2.0
DEVICE="cpu"
BATCH_SIZE=64

# Output directory
OUT_DIR="/pbs/home/o/omacias/Sam_project/raytune_lib_sept25/evaluate_antenna/figures/output"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Generating Appendix Check 1: Standard-method ROI-NMSE vs SNR figure"
echo "Model: $MODEL_NAME (for ML overlay)"
echo "============================================================"

echo ""
echo ">>> make_fig_appendix_check1_nmse_std_vs_snr.py"
srun python -u make_fig_appendix_check1_nmse_std_vs_snr.py \
    --model-path "$MODEL_PATH" \
    --metrics-json "$METRICS_JSON" \
    --config-json "$CONFIG_JSON" \
    --data-path "$DATA_PATH" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --dt-ns "$DT_NS" \
    --roi-half-width-ns 150.0 \
    --snr-peak-half-width-ns 150.0 \
    --snr-exclude-half-width-ns 150.0 \
    --rolling-window 2000 \
    --rolling-grid 60 \
    --rolling-min-points 150 \
    --band-quantiles "0.16,0.84" \
    --high-snr-min 8.0 \
    --clean-power-gate-quantile 0.10 \
    --clean-power-gate-min-count 200 \
    --xlim "1.5,15.0" \
    --show-ml-overlay \
    --out "${OUT_DIR}/fig_appendix_check1_nmse_std_vs_snr.pdf"

conda deactivate

echo ""
echo "============================================================"
echo "Figure generation completed!"
echo "Output: ${OUT_DIR}/fig_appendix_check1_nmse_std_vs_snr.pdf"
echo "============================================================"

