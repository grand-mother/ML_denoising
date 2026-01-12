#!/bin/bash
#SBATCH --job-name=psnr_model        # Job name
#SBATCH --partition=gpucluster            # Partition name
#SBATCH --time=18:00:00                   # Time limit hrs:min:sec
#SBATCH --output=psnr_model_%j.out   # Standard output and error log
#SBATCH --ntasks=1                       # Number of tasks

sinfo -o "%n %c %m"  # Shows nodes, their CPU count, and memory
sinfo -N -l  # Detailed node information

# Check NVIDIA GPU status using srun
srun --partition=gpucluster nvidia-smi


# Activate conda
source /pbs/throng/grand/soft/miniconda3/bin/activate
conda init
conda activate /sps/grand/macias/conda_envs/ili-torch

# Check if the environment was activated successfully
if [ $? -ne 0 ]; then
    echo "Failed to activate the Conda environment."
    exit 1 
fi

# Verify the active environment and Python version
echo "Active Conda Environment: $(conda info --envs | grep '*' | awk '{print $1}')"
python --version

export PYTHONPATH=/pbs/home/o/omacias/Sam_project

srun --partition=gpucluster python psnr_model.py


# Deactivate the conda environment
conda deactivate