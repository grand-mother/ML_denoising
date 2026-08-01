#!/bin/bash
#SBATCH --job-name=fixedcfg
#SBATCH --partition=gpu_v100
#SBATCH --time=48:00:00
#SBATCH --output=fixedcfg_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:v100:4

# Single fixed-configuration training (no Ray Tune) for the referee's time-only
# baseline question. Run twice, once per cell:
#
#   sbatch training/scripts/run_fixed_config.sh false time_only
#   sbatch training/scripts/run_fixed_config.sh true  dual_branch
#
# Optional 3rd/4th args: epochs and --max-samples cap (for smoke tests).

USE_FREQ=${1:?usage: run_fixed_config.sh <true|false> <tag> [epochs] [max_samples]}
TAG=${2:?usage: run_fixed_config.sh <true|false> <tag> [epochs] [max_samples]}
EPOCHS=${3:-100}
MAX_SAMPLES=${4:-}

# Clear this script's own positional params before sourcing conda's activate:
# `source` inherits the caller's $1/$2, and conda's legacy activate script
# misreads them ("false time_only") as its own arguments otherwise.
set --

nvidia-smi

source /pbs/throng/grand/soft/miniconda3/bin/activate
conda activate /sps/grand/macias/conda_envs/ili-torch

PROJECT_ROOT=/pbs/home/o/omacias/Sam_project/raytune_lib_final
export PYTHONPATH=$PROJECT_ROOT
cd $PROJECT_ROOT

EXTRA=""
if [ -n "$MAX_SAMPLES" ]; then
    EXTRA="--max-samples $MAX_SAMPLES"
fi

echo "=== use_freq_branch=$USE_FREQ  tag=$TAG  epochs=$EPOCHS  $EXTRA ==="

python -u training/scripts/train_fixed_config.py \
    --use-freq-branch "$USE_FREQ" \
    --tag "$TAG" \
    --epochs "$EPOCHS" \
    $EXTRA
