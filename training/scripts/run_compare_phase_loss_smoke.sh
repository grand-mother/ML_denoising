#!/bin/bash
#SBATCH --job-name=phase_loss_smoke        # Job name
#SBATCH --partition=gpu_v100_interactive   # idle interactive V100 node (ccwgislurm0103)
#SBATCH --time=00:40:00                    # Short: smoke test only
#SBATCH --output=phase_loss_smoke_%j.out   # Standard output and error log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:2                  # 2 GPUs -> exercises the DataParallel path

# Smoke test for the phase-loss A/B comparison (task 1). Runs the FULL 8-cell
# grid but only 1 epoch on a small sample cap, so it validates the entire path
# (data load -> seeded split + manifest -> get_criterion with phase_weighting ->
# DataParallel train -> validate -> save -> aggregate CSV/PDF) in a few minutes
# BEFORE committing the ~10 h production comparison.

nvidia-smi

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
export PYTHONUNBUFFERED=1
cd $PROJECT_ROOT/raytune_lib_final

OUT="$PROJECT_ROOT/raytune_lib_final/results/phase_loss_compare_smoke"

python -u training/scripts/compare_phase_loss.py \
    --epochs 1 \
    --max-samples 2000 \
    --phase-weights 0.1,0.3,0.6,1.0 \
    --weightings none,target_magnitude \
    --write-manifest \
    --device cuda \
    --out-dir "$OUT"
echo "=== compare exit: $? ==="

# Validate the phase-aware eval step too (timing efficiency + amp bias + phase).
python -u training/scripts/eval_phase_loss_cells.py \
    --compare-dir "$OUT" \
    --device cuda
echo "=== eval exit: $? ==="

echo "=== outputs ==="
ls -la "$OUT"

conda deactivate
