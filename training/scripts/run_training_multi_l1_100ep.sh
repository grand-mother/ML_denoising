#!/bin/bash
#SBATCH --job-name=dualae_multi_l1_100ep   # Job name
#SBATCH --partition=gpu_v100             # Partition name
#SBATCH --time=48:00:00                    # Time limit hrs:min:sec
#SBATCH --output=dualae_multi_l1_100ep_%j.out  # Standard output and error log
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=12                  # CPUs (matches RAY_NUM_CPUS below)
#SBATCH --mem=96G                           # Host RAM (large sim arrays + Ray object store)
#SBATCH --gres=gpu:v100:4

sinfo -o "%n %c %m"
sinfo -N -l
srun --partition=gpu_v100 nvidia-smi

# Activate conda
source /pbs/throng/grand/soft/miniconda3/bin/activate
conda init
conda activate /sps/grand/macias/conda_envs/ili-torch

if [ $? -ne 0 ]; then
    echo "Failed to activate the Conda environment."
    exit 1
fi

echo "Active Conda Environment: $(conda info --envs | grep '*' | awk '{print $1}')"
python --version

# Set up paths
PROJECT_ROOT=/pbs/home/o/omacias/Sam_project
export PYTHONPATH=$PROJECT_ROOT

# IMPORTANT: raytune_main_sept25.py now reads Ray resources from these env vars
# (ray.init num_gpus defaults to 1 otherwise). tune.run uses
# resources_per_trial={"cpu":4,"gpu":4}, so Ray MUST see all 4 allocated GPUs or
# trials never get scheduled and the job hangs. Match the SLURM allocation below.
export RAY_NUM_CPUS=12
export RAY_NUM_GPUS=4

cd $PROJECT_ROOT/raytune_lib_final

# Real run: CNN dual-branch, multi_l1 loss, 100 epochs, 24 Ray Tune samples.
# Output -> /sps/grand/macias/Sam_Result/multi_l1_CNN_100epochs_24samples
srun --partition=gpu_v100 python training/raytune_main_sept25.py \
    --json_params_file 'configs/experiments/training_params_multi_l1_100ep_24s.json'

conda deactivate
