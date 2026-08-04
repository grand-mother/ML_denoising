#!/bin/bash
#SBATCH --job-name=appendixfig
#SBATCH --partition=gpu_v100
#SBATCH --time=06:00:00
#SBATCH --output=appendixfig_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1

# Regenerates the appendix band-limited NMSE / output-SNR-gain figure.
#
# Two corrections force this regeneration (both change content, not just labels):
#   1. dt_ns 2.0 -> 0.5, the sampling stated in the paper (adopted for the
#      revision by decision of the corresponding author, 2026-08-05). dt sets
#      the band-pass sampling rate (fs = 1/dt) and every ns->sample conversion.
#   2. The input-SNR x-axis is the restored paper definition max(clean)/std(noisy)
#      over the full trace, replacing the ROI-peak / envelope-MAD variant.
#
# Model: the production multi_v3 checkpoint, evaluated at its 512-sample training
# length on the seeded, recorded test split. Inference + plotting only.
#
# Optional arg: --max-samples cap for a smoke test.

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
OUT_DIR=$PROJECT_ROOT/new_figure/appendix_nmse_snr_gain
mkdir -p "$OUT_DIR"

EXTRA=""
if [ -n "$MAX_SAMPLES" ]; then
    EXTRA="--max-samples $MAX_SAMPLES"
fi

echo "=== appendix NMSE / SNR-gain figure (dt=0.5 ns, paper SNR)  $EXTRA ==="

python -u visualization/nmse_snr_gain_vs_snr/make_fig_nmse_snr_gain_vs_snr__option3_truth_cleanpower_gate.py \
    --model-path   "$MODEL_DIR/best_model.pth" \
    --metrics-json "$MODEL_DIR/best_trial_metrics.json" \
    --config-json  "$MODEL_DIR/best_trial_config.json" \
    --data-path    /sps/grand/blevy/sims/sims_for_denoising_sept2025 \
    --device       cuda \
    --batch-size   256 \
    --dt-ns        0.5 \
    --eval-len     512 \
    --split-seed   12345 \
    --out          "$OUT_DIR/nmse_snr_gain_vs_snr.pdf" \
    $EXTRA
