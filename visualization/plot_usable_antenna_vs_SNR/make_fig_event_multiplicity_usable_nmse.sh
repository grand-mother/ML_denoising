#!/bin/bash
#SBATCH --job-name=fig_event_multiplicity
#SBATCH --partition=hpc
#SBATCH --time=12:00:00
#SBATCH --output=generate_figures_event_multiplicity_%j.out
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
cd /pbs/home/o/omacias/Sam_project/raytune_lib_final/visualization/plot_usable_antenna_vs_SNR

# Model configuration
PARENT_DIR="/sps/grand/macias/Sam_Result"
MODEL_NAME="multi_l1_example"

MODEL_PATH="${PARENT_DIR}/${MODEL_NAME}/best_model.pth"
METRICS_JSON="${PARENT_DIR}/${MODEL_NAME}/best_trial_metrics.json"
CONFIG_JSON="${PARENT_DIR}/${MODEL_NAME}/best_trial_config.json"

# Common parameters
DT_NS=2.0
DEVICE="cpu"
BATCH_SIZE=64

# Data path (for auto-loading event_number_list.npy)
DATA_PATH="/sps/grand/blevy/sims/sims_for_denoising_sept2025"

# Event IDs path (optional - will try to auto-load from data_path/event_number_list.npy if not provided)
# EVENT_IDS_PATH="${DATA_PATH}/event_number_list.npy"  # Uncomment to use explicit path
# STANDARD_NPZ="/path/to/standard_denoised.npz"  # Uncomment and set if needed

# Output directory
OUT_DIR="/pbs/home/o/omacias/Sam_project/raytune_lib_final/output"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Generating event multiplicity gain per event figure"
echo "Model: $MODEL_NAME"
echo "============================================================"

echo ""
echo ">>> make_fig_event_multiplicity_usable_nmse.py"
CMD_ARGS=(
    --model-path "$MODEL_PATH"
    --metrics-json "$METRICS_JSON"
    --config-json "$CONFIG_JSON"
    --data-path "$DATA_PATH"
    --device "$DEVICE"
    --batch-size "$BATCH_SIZE"
    --dt-ns "$DT_NS"
    --roi-half-width-ns 150.0
    --snr-peak-half-width-ns 150.0
    --snr-exclude-half-width-ns 150.0
    --clean-power-gate-quantile 0.10
    --clean-power-gate-min-count 200
    --snr-calib-min 6.0
    --calib-pass-rate 0.95
    --event-snr-stat median
    --snr-bins "1.5,2,3,4,5,6,7,8,9,10,12,14"
    --band-quantiles "0.16,0.84"
    --out "${OUT_DIR}/fig_event_multiplicity_usable_nmse.pdf"
)

# Add event-ids-path if explicitly set
if [ -n "${EVENT_IDS_PATH:-}" ]; then
    CMD_ARGS+=(--event-ids-path "$EVENT_IDS_PATH")
fi

# Add standard-npz if explicitly set
if [ -n "${STANDARD_NPZ:-}" ]; then
    CMD_ARGS+=(--standard-npz "$STANDARD_NPZ")
fi

srun python -u make_fig_event_multiplicity_usable_nmse.py "${CMD_ARGS[@]}"

conda deactivate

echo ""
echo "============================================================"
echo "Figure generation completed!"
echo "Output: ${OUT_DIR}/fig_event_multiplicity_usable_nmse.pdf"
echo "============================================================"
