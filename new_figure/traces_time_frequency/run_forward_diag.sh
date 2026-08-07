#!/bin/bash
#SBATCH --job-name=fwddiag
#SBATCH --partition=gpu_v100
#SBATCH --time=01:30:00
#SBATCH --output=fwddiag_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:v100:1

# Same checkpoint, same 400 test traces, two forward wirings (original vs
# rewired). Plus the crop-convention control that crashed in job 56035392.

set --

source /pbs/throng/grand/soft/miniconda3/bin/activate
conda activate /sps/grand/macias/conda_envs/ili-torch

PROJECT_ROOT=/pbs/home/o/omacias/Sam_project/raytune_lib_final
export PYTHONPATH=$PROJECT_ROOT:/pbs/home/o/omacias/Sam_project
export PYTHONUNBUFFERED=1
cd $PROJECT_ROOT/new_figure/traces_time_frequency

echo "########## forward wiring A/B ##########"
python -u diag_forward_wiring.py

echo
echo "########## crop-convention control (fixed) ##########"
python -u diag_crop_psnr.py

echo
echo "=== done ==="
