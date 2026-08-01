#!/bin/bash
#SBATCH --job-name=ampbias
#SBATCH --partition=gpu_v100
#SBATCH --time=04:00:00
#SBATCH --output=ampbias_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1

# Peak-amplitude bias versus input SNR, updated for the referee:
#   * binned MEDIANS of delta_A = A_rec/A_true - 1
#   * shaded band defined exactly (16th-84th percentile = central 68%)
#   * number of trigger-passing traces per SNR bin printed in the figure and
#     tabulated in amplitude_bias_table.csv
#   * one shared selection + one shared SNR array for the noisy and denoised
#     curves, enforced by shared_selection_mask()
#
# Model: the PRODUCTION multi_v3 checkpoint (the manuscript's fiducial), evaluated
# at its training length of 512 samples on a seeded, recorded test split.
# Inference + plotting only; no retraining.
#
# Optional arg: --max-samples cap for a smoke test.
#   sbatch new_figure/peak_amplitude_bias/run_amplitude_bias.sh 3000

MAX_SAMPLES=${1:-}

# Clear positional params before sourcing conda's activate (it misreads them).
set --

nvidia-smi

source /pbs/throng/grand/soft/miniconda3/bin/activate
conda activate /sps/grand/macias/conda_envs/ili-torch

PROJECT_ROOT=/pbs/home/o/omacias/Sam_project/raytune_lib_final
export PYTHONPATH=$PROJECT_ROOT:/pbs/home/o/omacias/Sam_project
cd $PROJECT_ROOT

MODEL_DIR=/sps/grand/macias/Sam_Result/multi_v3_CNN_100epochs_36samples
OUT_DIR=$PROJECT_ROOT/new_figure/peak_amplitude_bias

EXTRA=""
if [ -n "$MAX_SAMPLES" ]; then
    EXTRA="--max-samples $MAX_SAMPLES"
fi

echo "=== amplitude-bias figure (wrapped-phase model)  $EXTRA ==="

python -u visualization/amplitude_diagnostic_plot/make_fig_amplitude_bias_vs_snr.py \
    --model-path   "$MODEL_DIR/best_model.pth" \
    --metrics-json "$MODEL_DIR/best_trial_metrics.json" \
    --config-json  "$MODEL_DIR/best_trial_config.json" \
    --data-path    /sps/grand/blevy/sims/sims_for_denoising_sept2025 \
    --device       cuda \
    --batch-size   256 \
    --eval-len     512 \
    --split-seed   12345 \
    --out-dir      "$OUT_DIR" \
    $EXTRA
