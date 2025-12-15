# HPML Profiling Script (Baseline / LoRA / QLoRA)

This script profiles **DistilBERT fine-tuning** under three parameterization regimes:

- **Baseline** (full fine-tuning)
- **LoRA** (parameter-efficient fine-tuning)
- **QLoRA** (4-bit quantized LoRA, GPU required)

It is designed to:
- Compare **eager vs. torch.compile (inductor)** execution
- Measure **throughput, step latency, and peak GPU memory**
- Produce **PyTorch profiler traces** for operator-level analysis
- Optionally log results to **Weights & Biases (W&B)**

The script is intended for **diagnostic profiling**, not for producing final benchmark accuracy numbers (those come from `train_baseline.py`, `train_lora.py`, and `train_qlora.py`).

---

## 1. Setup & Installation

### Required dependencies
```bash
pip install torch transformers datasets tqdm
```

### For LoRA / QLoRA
```bash
pip install peft
```

### For QLoRA (Linux / WSL recommended)
```bash
pip install bitsandbytes
```

### Optional (for logging)
```bash
pip install wandb
```

> **Note on CUDA:**  
> Ensure your PyTorch installation matches your CUDA driver. See https://pytorch.org for the correct install command.

---

## 2. What This Script Does

- Runs **short fine-tuning loops** (typically 1–2 epochs)
- Records:
  - Epoch time
  - Tokens/sec throughput
  - Step latency (via profiler)
  - Peak VRAM usage
- Saves:
  - `epoch_results.csv`
  - `summary.json`
  - PyTorch profiler traces (TensorBoard-compatible)

Each run is isolated under:
```
reports/<run_id>/
```

---

## 3. How to Run

### Baseline (Full Fine-Tuning)
```bash
python profile_analysis.py \
  --method baseline \
  --dataset sst2 \
  --epochs 2 \
  --compile_mode eager \
  --run_id prof_baseline_eager
```

### LoRA Profiling
```bash
python profile_analysis.py \
  --method lora \
  --dataset sst2 \
  --rank 8 \
  --epochs 2 \
  --compile_mode eager \
  --run_id prof_lora_eager
```

### QLoRA Profiling (GPU + Linux/WSL)
```bash
python profile_analysis.py \
  --method qlora \
  --dataset sst2 \
  --rank 8 \
  --epochs 2 \
  --compile_mode eager \
  --run_id prof_qlora_eager
```

> ⚠️ `torch.compile(inductor)` is **disabled automatically for QLoRA** by default due to instability.

---

## 4. torch.compile (Inductor) Comparison

To compare eager vs. compiled execution:

```bash
# Eager
python profile_analysis.py --method baseline --compile_mode eager --epochs 2 --run_id eager_run

# Compiled
python profile_analysis.py --method baseline --compile_mode inductor --epochs 2 --run_id compile_run
```

---

## 5. Viewing Profiler Results (TensorBoard)

Each run writes profiler traces to:
```
reports/<run_id>/profiler_traces/
```

Launch TensorBoard:
```bash
tensorboard --logdir reports/<run_id>/profiler_traces
```

This allows you to inspect:
- Operator-level CUDA kernels
- Memory allocations
- Forward/backward time breakdowns

---

## 6. Weights & Biases (Optional)

W&B is **opt-in** and disabled by default.

### Enable W&B logging
```bash
python profile_analysis.py \
  --method lora \
  --dataset sst2 \
  --rank 8 \
  --epochs 2 \
  --run_id prof_lora_wandb \
  --wandb \
  --wandb_project hpml-peft-benchmark \
  --wandb_mode offline
```

### Sync offline runs
```bash
wandb sync wandb/offline-run-*
```

You can then view metrics and comparisons on the W&B dashboard.

---

## 7. Outputs

Each profiling run produces:

```
reports/<run_id>/
├── config.json            # Full run configuration
├── epoch_results.csv      # Per-epoch metrics
├── summary.json           # Aggregated stats + top CUDA ops
└── profiler_traces/       # TensorBoard profiler output
```

---

## 8. How This Fits Into the Project

- Use **`train_baseline.py`, `train_lora.py`, `train_qlora.py`** for:
  - Final accuracy benchmarks
  - Reproducible training runs
- Use **`profile_analysis.py`** for:
  - Diagnosing performance bottlenecks
  - Comparing baseline vs. LoRA vs. QLoRA efficiency
  - Generating profiler visualizations for the report

Profiling runs are intentionally **short and separate** to avoid skewing benchmark results.

---

## 9. Recommended Workflow

1. Run benchmark training scripts → collect `summary.json`
2. Run profiling script (1–2 epochs) → collect traces + timing
3. Aggregate results into tables / plots for the final report
