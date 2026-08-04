#!/bin/bash
#SBATCH --job-name=testeval
#SBATCH --partition=gpu_v100
#SBATCH --time=02:00:00
#SBATCH --output=testeval_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1

# Re-evaluate the existing time-only and dual-branch checkpoints on the held-out
# TEST split (the earlier comparison used the validation split). Inference only,
# no retraining.
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

EXTRA=""
if [ -n "$MAX_SAMPLES" ]; then
    EXTRA="--max-samples $MAX_SAMPLES"
fi

echo "=== test-split evaluation of the fixed-config pair  $EXTRA ==="

python -u evaluation/evaluate_fixed_config_test.py $EXTRA
