# HPML PEFT Benchmark — Training Scripts (Keemin)

This directory contains the **final, reproducible training and benchmarking scripts** for the HPML project comparing:

- **Baseline full fine-tuning**
- **LoRA (parameter-efficient fine-tuning)**
- **QLoRA (4-bit quantized LoRA)**

All scripts share a common argument interface and metrics pipeline, and are designed to be runnable on **any machine** without hard-coded paths.

---

## Directory Structure

```
scripts/keemin/
├── train_common.py        # Shared utilities, args, callbacks, metrics
├── train_baseline.py      # Full fine-tuning baseline (DistilBERT)
├── train_lora.py          # LoRA fine-tuning
├── train_qlora.py         # QLoRA (4-bit NF4) fine-tuning
├── tmp/
│   ├── hpml_outputs/      # Run outputs (auto-created)
│   └── hpml_logs/         # Logs (W&B, etc.)
└── README.md
```

The `tmp/` directory is repo-local, reproducible, and should not be committed.

---

## Models and Dataset

- **Model:** `distilbert-base-uncased`
- **Task:** GLUE SST-2 (binary sentiment classification)
- **Trainer:** HuggingFace `Trainer`
- **Metrics:** Accuracy, loss, throughput, latency, GPU memory

Baseline performs **full fine-tuning**; LoRA and QLoRA train only a small subset of parameters.

---

## Default Output Locations

By default, scripts write outputs relative to this directory:

- `scripts/keemin/tmp/hpml_outputs/`
- `scripts/keemin/tmp/hpml_logs/`

You may override these with `--outdir` and `--logdir`.

---

## Quick Smoke Tests (Recommended)

Use `--max_steps` for fast validation before long runs.

### Baseline (full fine-tuning)

```bash
python -m scripts.keemin.train_baseline --max_steps 50 --train_bs 8 --eval_bs 64 --report_to none
```

### LoRA

```bash
python -m scripts.keemin.train_lora --max_steps 50 --rank 8 --train_bs 8 --eval_bs 64 --report_to none
```

### QLoRA (4-bit NF4)

```bash
python -m scripts.keemin.train_qlora --max_steps 50 --rank 8 --train_bs 8 --eval_bs 64 --report_to none
```

---

## Outputs Per Run

Each run creates a uniquely named folder containing:

- `summary.json` — single-row result summary
- `summary.csv` — CSV version of summary
- `metrics.csv` — Trainer logs (step/epoch)
- `env.json` — environment + git metadata
- `checkpoint/` (baseline) or `adapter/` (LoRA/QLoRA)

Example:

```
scripts/keemin/tmp/hpml_outputs/
└── baseline-distilbert-base-uncased-sst2-YYYYMMDD-HHMMSS-xxxxxx/
```

---

## Reproducibility

Reproducibility is supported via:

- explicit `--seed`
- logged git commit hash
- logged environment metadata
- standardized output schema (`summary.json` / `summary.csv`)

---

## Notes

- Low accuracy during smoke tests is expected (few steps).
- Full runs should omit `--max_steps`.
- QLoRA requires `bitsandbytes` and a compatible CUDA environment.

---

## Git Hygiene

Add these to `.gitignore`:

```
scripts/keemin/tmp/
wandb/
```
