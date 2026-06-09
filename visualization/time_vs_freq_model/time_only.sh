#!/bin/bash
#SBATCH --job-name=time_freq_ablation
#SBATCH --partition=gpu_v100
#SBATCH --gpus=1   
#SBATCH --time=18:00:00
#SBATCH --output=time_freq_ablation_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

# Print Node information
sinfo -o "%n %c %m"
sinfo -N -l
srun --partition=gpu_v100 nvidia-smi

# Activate conda
source /pbs/throng/grand/soft/miniconda3/bin/activate
conda activate /sps/grand/macias/conda_envs/ili-torch

# Verify the active environment and Python version
echo "Active Conda Environment: $(conda info --envs | grep '*' | awk '{print $1}')"
python --version

export RAY_NUM_CPUS=4
export RAY_NUM_GPUS=1

# Set up paths
PROJECT_ROOT=/pbs/home/o/omacias/Sam_project
export PYTHONPATH=$PROJECT_ROOT/raytune_lib_final

# Change to the working directory
cd $PROJECT_ROOT/raytune_lib_final/visualization/time_vs_freq_model

echo "Starting the Python script..."
python time_freq.py
echo "Python script finished."
