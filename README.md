# HPML PEFT Benchmark (LoRA vs QLoRA)
**Course:** COMS E6998-012 High-Performance Machine Learning (Fall 2025)  
**Owner:** Keemin Lee (kjl2175)  
**Date:** October 12, 2025

---

## Overview
This repository contains code and cluster setup for benchmarking **parameter-efficient fine-tuning (PEFT)** methods on large language models.  
We compare full fine-tuning, **LoRA**, and **Q-LoRA** across GPU memory, latency, and accuracy.

The environment and workflow are configured for the **Columbia Insomnia GPU Cluster**.

---

## Project Structure
```
hpml-peft-benchmark/
 ├─ scripts/        # Training, profiling, and evaluation scripts
 ├─ slurm/          # Job submission scripts
 ├─ docs/           # Cluster setup and environment notes
 ├─ env/            # Environment specs (requirements.txt, YAML)
 ├─ outputs/        # Placeholder (real outputs in scratch/outputs)
 ├─ logs/           # Placeholder (real logs in scratch/logs)
 └─ data/           # Placeholder (real datasets in scratch/data)
```

---

## Environment Setup

### 1. Load Anaconda and Initialize Conda
```bash
module load anaconda/2023.09
eval "$(/insomnia001/shared/apps/anaconda/2023.09/bin/conda shell.bash hook)"
```

### 2. Create Project Environment
```bash
conda create -p $HOME/.conda/envs/peft_benchmark python=3.10 -y
conda activate $HOME/.conda/envs/peft_benchmark
```

### 3. Install Dependencies
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install transformers datasets peft bitsandbytes accelerate deepspeed wandb pynvml
```

### 4. Make Persistent (add to ~/.bashrc)
```bash
# Conda setup
eval "$(/insomnia001/shared/apps/anaconda/2023.09/bin/conda shell.bash hook)"
export CONDA_ENVS_DIRS="$HOME/.conda/envs"
export CONDA_PKGS_DIRS="$HOME/.conda/pkgs"

# Shared Hugging Face cache
export HF_HOME=/insomnia001/depts/edu/COMS-E6998-012/kjl2175/cache/hf
export TRANSFORMERS_CACHE=$HF_HOME
```

---

## Cluster Directory Layout
```
/insomnia001/depts/edu/COMS-E6998-012/kjl2175/
 ├─ code/        → cloned GitHub repo
 ├─ data/        → real datasets (e.g., SST-2)
 ├─ outputs/     → model checkpoints, metrics
 ├─ logs/        → SLURM + training logs
 └─ cache/       → shared Hugging Face cache
```

Create this structure:
```bash
SCR=/insomnia001/depts/edu/COMS-E6998-012/kjl2175
mkdir -p $SCR/{code,data,outputs,logs,cache}
cd $SCR/code
git clone git@github.com:keeminlee/hpml-peft-benchmark.git
```

---

## Running Jobs

### Interactive GPU shell
```bash
srun --pty -t 0-01:00 --gres=gpu:1 -A edu /bin/bash
module load anaconda/2023.09
eval "$(/insomnia001/shared/apps/anaconda/2023.09/bin/conda shell.bash hook)"
conda activate $HOME/.conda/envs/peft_benchmark
python scripts/test_gpu.py
```

### Batch job (preferred)
Example: `slurm/train_baseline.slurm`
```bash
#!/bin/bash
#SBATCH --job-name=baseline_sst2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=edu
#SBATCH --output=/insomnia001/depts/edu/COMS-E6998-012/kjl2175/logs/%x-%j.out

module load anaconda/2023.09
eval "$(/insomnia001/shared/apps/anaconda/2023.09/bin/conda shell.bash hook)"
conda activate $HOME/.conda/envs/peft_benchmark

export HF_HOME=/insomnia001/depts/edu/COMS-E6998-012/kjl2175/cache/hf
cd /insomnia001/depts/edu/COMS-E6998-012/kjl2175/code/hpml-peft-benchmark

python scripts/train_baseline.py   --data_dir /insomnia001/depts/edu/COMS-E6998-012/kjl2175/data   --outdir   /insomnia001/depts/edu/COMS-E6998-012/kjl2175/outputs/kjl2175   --logdir   /insomnia001/depts/edu/COMS-E6998-012/kjl2175/logs/kjl2175
```

Submit the job:
```bash
cd slurm
sbatch train_baseline.slurm
squeue -u $USER
tail -f /insomnia001/depts/edu/COMS-E6998-012/kjl2175/logs/baseline_sst2-<JOBID>.out
```

---

## Validation Scripts
- `scripts/test_gpu.py` → verifies GPU availability  
- `slurm/gpu_sanity.slurm` → short GPU test on compute node

Submit:
```bash
sbatch slurm/gpu_sanity.slurm
```

---

## Path Conventions
| Path | Purpose | Git-tracked? |
|------|----------|--------------|
| `/insomnia.../code/hpml-peft-benchmark/` | Code repo | ✅ |
| `/insomnia.../data/` | Datasets | ❌ |
| `/insomnia.../outputs/` | Model results | ❌ |
| `/insomnia.../logs/` | SLURM + training logs | ❌ |
| `/insomnia.../cache/` | HF cache | ❌ |

---

## Sanity Checklist
- [x] SSH access to Insomnia (`ssh kjl2175@insomnia.rcs.columbia.edu`)
- [x] Git + SSH key configured (`ssh -T git@github.com`)
- [x] Conda env created in `$HOME/.conda/envs/peft_benchmark`
- [x] Repo cloned under scratch `code/`
- [x] HF cache and data paths configured
- [x] Baseline SLURM job runs successfully

---

## Notes for Teammates
Until the shared **team scratch space** is granted (or we just keep using this), this project uses Keemin’s scratch directory as the working area.  
Other members should:
1. Clone the repo under their own home or scratch.  
2. Point their `--outdir`, `--logdir`, and `HF_HOME` to:  
   `/insomnia001/depts/edu/COMS-E6998-012/kjl2175/`

---

## Next Steps
- Run baseline (DistilBERT) training → collect accuracy, GPU memory, latency.  
- Extend to LoRA (rank ∈ {4,8,16,32}).  
- Add QLoRA (4-bit, 8-bit).  
- Log metrics with **Weights & Biases**.  
- Summarize results → `reports/` or `notebooks/`.

---

**Maintainer:** Keemin Lee (kjl2175)  
**GitHub:** [keeminlee/hpml-peft-benchmark](https://github.com/keeminlee/hpml-peft-benchmark)
