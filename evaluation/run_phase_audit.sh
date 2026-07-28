#!/bin/bash
#SBATCH --job-name=phase_audit
#SBATCH --partition=gpu_v100_interactive   # short job (~5-10 min); interactive node
#SBATCH --time=00:40:00
#SBATCH --output=phase_audit_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:v100:1

nvidia-smi

source /pbs/throng/grand/soft/miniconda3/bin/activate
conda activate /sps/grand/macias/conda_envs/ili-torch
if [ $? -ne 0 ]; then echo "conda activate failed"; exit 1; fi
echo "Active env: $(conda info --envs | grep '*' | awk '{print $1}')"; python --version

PROJECT_ROOT=/pbs/home/o/omacias/Sam_project
export PYTHONPATH=$PROJECT_ROOT/raytune_lib_final
export PYTHONUNBUFFERED=1
cd $PROJECT_ROOT/raytune_lib_final

# Fiducial = dual-branch multi_l1 checkpoint (the one with a phase term).
python -u evaluation/phase_boundary_audit.py \
    --model-dir /sps/grand/macias/Sam_Result/multi_l1_CNN_100epochs_24samples \
    --device cuda \
    --trace-length 512 \
    --sample-spacing-seconds 0.5e-9 \
    --out-dir "$PROJECT_ROOT/raytune_lib_final/results/phase_boundary_audit"

conda deactivate
