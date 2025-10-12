#!/bin/bash
#SBATCH --account=free
#SBATCH --job-name=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=00:05:00
#SBATCH --output=gpu_test.out

module load anaconda3 cuda/12.1
conda activate peft_benchmark
python test_gpu.py
