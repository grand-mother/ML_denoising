#!/bin/bash
#SBATCH --job-name=fig_nmse_snrgain_3
#SBATCH --partition=hpc
#SBATCH --time=12:00:00
#SBATCH --output=generate_figures_nmse_snrgain_option3_%j.out
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

# Model configuration
PARENT_DIR="/sps/grand/macias/Sam_Result"
MODEL_NAME="l1_CNN_100epochs_36samples"

MODEL_PATH="${PARENT_DIR}/${MODEL_NAME}/best_model.pth"
METRICS_JSON="${PARENT_DIR}/${MODEL_NAME}/best_trial_metrics.json"
CONFIG_JSON="${PARENT_DIR}/${MODEL_NAME}/best_trial_config.json"

# Common parameters
DT_NS=2.0
DEVICE="cpu"
BATCH_SIZE=64

# Output directory
OUT_DIR="/pbs/home/o/omacias/Sam_project/raytune_lib_sept25/evaluate_antenna/figures/output"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Generating NMSE and SNR gain vs SNR figure (Option 3: ROI-peak SNR, truth-conditioned clean power gate, q05-q95 bands)"
echo "Model: $MODEL_NAME"
echo "============================================================"

echo ""
echo ">>> make_fig_nmse_snr_gain_vs_snr__option3_truth_cleanpower_gate.py"
srun python -u make_fig_nmse_snr_gain_vs_snr__option3_truth_cleanpower_gate.py \
    --model-path "$MODEL_PATH" \
    --metrics-json "$METRICS_JSON" \
    --config-json "$CONFIG_JSON" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --dt-ns "$DT_NS" \
    --roi-half-width-ns 150.0 \
    --snr-peak-half-width-ns 150.0 \
    --snr-exclude-half-width-ns 150.0 \
    --rolling-window 400 \
    --rolling-min-points 80 \
    --rolling-grid 45 \
    --clean-power-gate-quantile 0.10 \
    --clean-power-gate-min-count 200 \
    --band-quantiles "0.05,0.95" \
    --xlim "1.5,15.0" \
    --out "${OUT_DIR}/fig_nmse_snrgain_vs_snr_option3.pdf"

conda deactivate

echo ""
echo "============================================================"
echo "Figure generation completed!"
echo "Output: ${OUT_DIR}/fig_nmse_snrgain_vs_snr_option3.pdf"
echo "============================================================"

