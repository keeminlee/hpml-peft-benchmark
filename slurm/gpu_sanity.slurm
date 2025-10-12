#!/bin/bash
#SBATCH --job-name=gpu_sanity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --account=edu
#SBATCH --output=/insomnia001/depts/edu/COMS-E6998-012/kjl2175/logs/%x-%j.out

module load anaconda/2023.09
eval "$(/insomnia001/shared/apps/anaconda/2023.09/bin/conda shell.bash hook)"
conda activate $HOME/.conda/envs/peft_benchmark

python - << 'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
