#!/bin/bash
#SBATCH --job-name=dualautoencoder        # Job name
#SBATCH --partition=gpucluster            # Partition name
#SBATCH --time=48:00:00                   # Time limit hrs:min:sec
#SBATCH --output=dualautoencoder_%j.out   # Standard output and error log
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --gres=gpu:v100:4

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

# Set up paths - adjust PROJECT_ROOT as needed
PROJECT_ROOT=/pbs/home/o/omacias/Sam_project
export PYTHONPATH=$PROJECT_ROOT

# Change to project directory
cd $PROJECT_ROOT/raytune_lib_final

# Run training with configuration file
srun --partition=gpucluster python training/raytune_main_sept25.py --json_params_file 'configs/experiments/training_params.json'


# Deactivate the conda environment
conda deactivate