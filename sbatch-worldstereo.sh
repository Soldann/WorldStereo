#!/bin/bash

#SBATCH --job-name=ws-lite          # Job name
#SBATCH --time 00:30:00                 # 8-hour run-time
#SBATCH --output=output_%j.log          # Output file (job number will be appended)
#SBATCH --error=error_%j.log            # Error file (job number will be appended)
#SBATCH --cpus-per-task=8		# 8 cores
#SBATCH --mem-per-cpu=8G
#SBATCH --tmp=100G
#SBATCH --gpus=a100:1

source /cluster/home/languo/.bashrc
source /cluster/home/languo/load-skyfall.sh

conda activate worldstereo

cp -r /cluster/work/rsl/languo/huggingface $TMPDIR
export HF_HOME=$TMPDIR/huggingface
cd /cluster/work/rsl/languo/generative_gaussian_splatting/submodules/WorldStereo

#/cluster/home/languo/3dv-venv-gof/bin/python3 /cluster/home/languo/gaussian-opacity-fields/scripts/run_nerf_synthetic.py
torchrun run_camera_control.py --model_type worldstereo-memory-dmd --input_path /cluster/work/rsl/languo/ --scene_name custom-satellite --output_path outputs --seed 1024 --sequential_cpu_offload --bypass-depth-and-initial-frame
# torchrun run_camera_control.py --model_type worldstereo-camera --input_path /cluster/work/rsl/languo/eth_satellite/images --output_path outputs --seed 1024