#!/bin/bash
#SBATCH --job-name=phase_loss_cmp_1024          # Job name
#SBATCH --partition=gpu_v100_interactive   # idle node ccwgislurm0103 has 4 free V100s (gpu_v100 batch is full); NB interactive jobs may be preemptible
#SBATCH --time=14:00:00                    # 2 runs (none vs target_magnitude) x 40 ep @1024, fixed production config
#SBATCH --output=phase_loss_cmp_1024_%j.out     # Standard output and error log
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:v100:4

nvidia-smi

# Activate conda
source /pbs/throng/grand/soft/miniconda3/bin/activate
conda activate /sps/grand/macias/conda_envs/ili-torch
if [ $? -ne 0 ]; then
    echo "Failed to activate the Conda environment."
    exit 1
fi
echo "Active Conda Environment: $(conda info --envs | grep '*' | awk '{print $1}')"
python --version

PROJECT_ROOT=/pbs/home/o/omacias/Sam_project
export PYTHONPATH=$PROJECT_ROOT/raytune_lib_final
export PYTHONUNBUFFERED=1          # stream python prints live to the .out
cd $PROJECT_ROOT/raytune_lib_final

OUT="$PROJECT_ROOT/raytune_lib_final/results/phase_loss_compare_1024"

# Task-1 phase-loss A/B: fixed production architecture, one frozen seeded split,
# only phase_weighting (none vs target_magnitude) x phase_weight vary.
# SEQUENTIAL over all 8 cells in a single process; each cell uses all 4 GPUs via
# DataParallel (production-proven ~100s/epoch, no cross-process contention).
# --write-manifest saves the frozen split; the script auto-aggregates at the end.
# NB: run python DIRECTLY (no srun) so stdout streams straight to the .out; the
# previous srun-wrapped run showed zero output before it was cancelled.
python -u training/scripts/compare_phase_loss.py \
    --epochs 40 \
    --phase-weights 0.023580860891914208 \
    --weightings none,target_magnitude \
    --write-manifest \
    --device cuda \
    --out-dir "$OUT"

# Phase-aware evaluation on the SAME frozen validation split for every cell:
# timing efficiency + low-SNR amplitude bias + phase/spectral errors.
python -u training/scripts/eval_phase_loss_cells.py \
    --compare-dir "$OUT" \
    --device cuda

conda deactivate
