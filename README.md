# HPML PEFT Benchmark (LoRA vs QLoRA)
**Course:** COMS E6998-012 High-Performance Machine Learning (Fall 2025)  

---

## Overview
This repository contains code and setup for benchmarking **parameter-efficient fine-tuning (PEFT)** methods such as **LoRA** and **QLoRA** against full fine-tuning baselines.  
We evaluate trade-offs across GPU memory, latency, and accuracy using the **Columbia Insomnia GPU Cluster**.

---

## Project Structure
```
hpml-peft-benchmark/
 ├─ scripts/        # Training, profiling, and evaluation scripts
 ├─ slurm/          # SLURM job submission scripts
 ├─ docs/           # Environment and setup documentation
 ├─ env/            # Environment specs (requirements.txt, YAML)
 ├─ reports/        # Experimental results and analysis
 └─ notebooks/      # Optional interactive analysis notebooks
```
> Note: `data/`, `outputs/`, and `logs/` are stored externally on cluster scratch space.

---

## Environment Setup

### 1. Load and initialize Conda
```bash
module load anaconda/2023.09
eval "$(/insomnia001/shared/apps/anaconda/2023.09/bin/conda shell.bash hook)"
```

### 2. Create project environment
```bash
conda create -p $HOME/.conda/envs/peft_benchmark python=3.10 -y
conda activate $HOME/.conda/envs/peft_benchmark
```

### 3. Install dependencies
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install transformers datasets peft bitsandbytes accelerate deepspeed wandb pynvml
```

### 4. Make persistent (optional)
Add this to your `~/.bashrc`:
```bash
eval "$(/insomnia001/shared/apps/anaconda/2023.09/bin/conda shell.bash hook)"
export CONDA_ENVS_DIRS="$HOME/.conda/envs"
export CONDA_PKGS_DIRS="$HOME/.conda/pkgs"
```

---

## Cluster Directory Layout

Keemin's scratch space (serving as shared data/cache source):
```
/insomnia001/depts/edu/COMS-E6998-012/kjl2175/
 ├─ code/        → cloned GitHub repo
 ├─ data/        → read-only shared datasets
 ├─ outputs/     → personal model checkpoints (per-user)
 ├─ logs/        → SLURM + training logs (per-user)
 └─ cache/       → shared Hugging Face cache
```

Each teammate keeps **their own outputs/logs**, while reading shared data and cache from Keemin’s scratch.

### Create your own structure (for each teammate)
```bash
MY_SCR=/insomnia001/depts/edu/COMS-E6998-012/<your-UNI>
mkdir -p $MY_SCR/{outputs,logs}
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
Example SLURM file `slurm/train_baseline.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=baseline_sst2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=edu
#SBATCH --output=/insomnia001/depts/edu/COMS-E6998-012/<your-UNI>/logs/%x-%j.out

module load anaconda/2023.09
eval "$(/insomnia001/shared/apps/anaconda/2023.09/bin/conda shell.bash hook)"
conda activate $HOME/.conda/envs/peft_benchmark

export HF_HOME=/insomnia001/depts/edu/COMS-E6998-012/kjl2175/cache/hf
export TRANSFORMERS_CACHE=$HF_HOME
DATA_DIR=/insomnia001/depts/edu/COMS-E6998-012/kjl2175/data

python scripts/train_baseline.py   --data_dir $DATA_DIR   --outdir   /insomnia001/depts/edu/COMS-E6998-012/<your-UNI>/outputs/${USER}   --logdir   /insomnia001/depts/edu/COMS-E6998-012/<your-UNI>/logs/${USER}
```

Submit and monitor:
```bash
cd slurm
sbatch train_baseline.slurm
squeue -u $USER
tail -f /insomnia001/depts/edu/COMS-E6998-012/<your-UNI>/logs/baseline_sst2-<JOBID>.out
```

---

## Shared Resources and Permissions

ACLs are **not supported** on this filesystem, so direct shared writing is unavailable.  
Instead, follow this model:

| Path | Access | Description |
|------|---------|-------------|
| `/insomnia.../kjl2175/data/` | Read-only | Shared datasets for all |
| `/insomnia.../kjl2175/cache/hf/` | Read-only | Shared Hugging Face model cache |
| `/insomnia.../<UNI>/outputs/` | Read/Write (owner only) | Each teammate’s training outputs |
| `/insomnia.../<UNI>/logs/` | Read/Write (owner only) | Job logs per teammate |

### To make shared data/cache readable (done by Keemin)
```bash
chmod -R a+rX /insomnia001/depts/edu/COMS-E6998-012/kjl2175/data
chmod -R a+rX /insomnia001/depts/edu/COMS-E6998-012/kjl2175/cache
```

---

## Teammate Quickstart

1. **Clone the repo**
   ```bash
   git clone git@github.com:keeminlee/hpml-peft-benchmark.git
   ```

2. **Create environment**
   ```bash
   conda create -p $HOME/.conda/envs/peft_benchmark python=3.10 -y
   conda activate $HOME/.conda/envs/peft_benchmark
   pip install -r env/requirements.txt
   ```

3. **Set shared paths**
   ```bash
   export HF_HOME=/insomnia001/depts/edu/COMS-E6998-012/kjl2175/cache/hf
   export TRANSFORMERS_CACHE=$HF_HOME
   DATA_DIR=/insomnia001/depts/edu/COMS-E6998-012/kjl2175/data
   ```

4. **Run jobs using your own scratch outputs/logs**
   ```bash
   MY_SCR=/insomnia001/depts/edu/COMS-E6998-012/<your-UNI>
   python scripts/train_baseline.py      --data_dir $DATA_DIR      --outdir   $MY_SCR/outputs/${USER}      --logdir   $MY_SCR/logs/${USER}
   ```

---

## Checklist
- [x] SSH access to cluster confirmed  
- [x] Repo cloned from GitHub  
- [x] Conda environment created  
- [x] Shared data/cache accessible (read-only)  
- [x] Jobs writing to per-user scratch directories  
- [x] Baseline SLURM job runs successfully  
