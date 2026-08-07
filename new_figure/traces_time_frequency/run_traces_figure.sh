#!/bin/bash
#SBATCH --job-name=tracesfig
#SBATCH --partition=gpu_v100
#SBATCH --time=03:00:00
#SBATCH --output=tracesfig_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1

# Regenerates the example waveform / spectrum figure (Fig. 1-style panel) with an
# explicit sampling interval.
#
# dt_ns drives BOTH axes of this figure:
#     time axis      t = sample_index * dt_ns
#     frequency axis f = rfftfreq(N, d=dt_ns*1e-9)
# so it is the one figure in the paper where the assumed sampling rate is
# directly visible to the reader.
#
# Two versions are produced from the SAME trace (sample selection depends only on
# SNR and clean peak amplitude, both dt-independent), so they can be compared
# panel by panel:
#     dt0p5/  dt = 0.5 ns, the value stated in the paper
#     dt2p0/  dt = 2.0 ns, the value the archived scripts used
#
# Model: production multi_v3 checkpoint. Inference + plotting only, no training.

set --

nvidia-smi

source /pbs/throng/grand/soft/miniconda3/bin/activate
conda activate /sps/grand/macias/conda_envs/ili-torch

PROJECT_ROOT=/pbs/home/o/omacias/Sam_project/raytune_lib_final
export PYTHONPATH=$PROJECT_ROOT:/pbs/home/o/omacias/Sam_project
export PYTHONUNBUFFERED=1
cd $PROJECT_ROOT

OUT_BASE=$PROJECT_ROOT/new_figure/traces_time_frequency
SCRIPT=$OUT_BASE/make_fig_traces_time_frequency.py

echo "########## dt = 0.5 ns (paper value, requested) ##########"
python -u "$SCRIPT" --dt-ns 0.5 --out-dir "$OUT_BASE/dt0p5" --num-images 2

echo
echo "########## dt = 2.0 ns (archived script value, for comparison) ##########"
python -u "$SCRIPT" --dt-ns 2.0 --out-dir "$OUT_BASE/dt2p0" --num-images 2

echo
echo "=== done ==="
ls -la "$OUT_BASE/dt0p5" "$OUT_BASE/dt2p0"
