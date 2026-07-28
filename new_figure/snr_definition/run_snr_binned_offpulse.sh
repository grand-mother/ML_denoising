#!/bin/bash
#SBATCH --job-name=snr_offpulse_fig
#SBATCH --partition=gpu_v100_interactive
#SBATCH --time=00:40:00
#SBATCH --output=snr_offpulse_fig_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:v100:1
nvidia-smi
source /pbs/throng/grand/soft/miniconda3/bin/activate
conda activate /sps/grand/macias/conda_envs/ili-torch
export PYTHONPATH=/pbs/home/o/omacias/Sam_project/raytune_lib_final
export PYTHONUNBUFFERED=1
cd /pbs/home/o/omacias/Sam_project/raytune_lib_final
# Fiducial l1 checkpoint; off-pulse SNR (paper_input_snr default in the raw_rms path).
python -u visualization/amplitude_diagnostic_plot/make_fig_amplitude_bias_vs_snr.py \
    --model-path  /sps/grand/macias/Sam_Result/l1_CNN_100epochs_36samples/best_model.pth \
    --metrics-json /sps/grand/macias/Sam_Result/l1_CNN_100epochs_36samples/best_trial_metrics.json \
    --config-json /sps/grand/macias/Sam_Result/l1_CNN_100epochs_36samples/best_trial_config.json \
    --device cuda --batch-size 256 \
    --out-dir /pbs/home/o/omacias/Sam_project/raytune_lib_final/new_figure
conda deactivate
