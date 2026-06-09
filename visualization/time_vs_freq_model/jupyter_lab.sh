#!/bin/bash
#SBATCH --job-name=jupyter_lab            # Job name
#SBATCH --partition=gpu_l40s_interactive
#SBATCH --gpus=1                          # Number of GPUs
#SBATCH --time=18:00:00                   # Time limit hrs:min:sec
#SBATCH --output=jupyter_lab_%j.out       # Standard output and error log
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

# sinfo -o "%n %c %m"  # Shows nodes, their CPU count, and memory
sinfo -N -l  # Detailed node information

nvidia-smi

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

export RAY_NUM_CPUS=4
export RAY_NUM_GPUS=1

# Set up paths - adjust PROJECT_ROOT as needed
PROJECT_ROOT=/pbs/home/o/omacias/Sam_project
export PYTHONPATH=$PROJECT_ROOT/raytune_lib_final

# Change to the working directory
cd $PROJECT_ROOT/raytune_lib_final/visualization/time_vs_freq_model

# Get node info and a random port for Jupyter Lab
export NODE=$(hostname -s)
export USER=$(whoami)
export PORT=$(shuf -i 8000-9999 -n 1)

echo -e "
-----------------------------------------------------------------
Jupyter Lab is running on node: $NODE
Port: $PORT

To connect, run this EXACT command on your Windows computer:
ssh -J ${USER}@ccahm.in2p3.fr -N -L ${PORT}:localhost:${PORT} ${USER}@${SLURM_SUBMIT_HOST}.in2p3.fr

Then open your browser and go to:
http://localhost:${PORT}
-----------------------------------------------------------------
"

# Set up reverse SSH tunnel to the exact login node you submitted from
ssh -o StrictHostKeyChecking=no -N -f -R ${PORT}:localhost:${PORT} ${SLURM_SUBMIT_HOST}.in2p3.fr

# Run Jupyter Lab
jupyter lab --no-browser --port=${PORT} --ip=0.0.0.0

# Deactivate the conda environment
conda deactivate
