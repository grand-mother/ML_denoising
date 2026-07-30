#!/bin/bash
#SBATCH --job-name=timingfig
#SBATCH --partition=gpu_v100
#SBATCH --time=04:00:00
#SBATCH --output=timingfig_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1

# Regenerates the paper timing figure (Fig. 7) with the corrected 5 ns legend.
# Optional arg: --max-samples cap for a smoke test.
#   sbatch new_figure/timing_efficiency_vs_snr/run_timing_figure.sh 2000

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

echo "=== regenerating timing figure  $EXTRA ==="

python -u new_figure/timing_efficiency_vs_snr/make_fig_timing_efficiency.py $EXTRA
