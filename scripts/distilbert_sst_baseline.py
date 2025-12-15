#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baseline fine-tuning for DistilBERT on GLUE/SST-2 with:
- W&B/none logging
- Accuracy + loss curves (from HF Trainer)
- Step latency (p50/p95) + peak GPU memory
- Per-epoch train time recorded to epoch_time.jsonl and folded into summary.json/.csv

Usage (single-GPU):
  python scripts/train_baseline_distilbert.py \
    --model distilbert-base-uncased --task sst2 \
    --epochs 3 --train_bs 16 --eval_bs 64 --lr 2e-5 --max_length 128 \
    --outdir /insomnia001/depts/edu/COMS-E6998-012/akv2129/outputs/$USER/baseline \
    --logdir  /insomnia001/depts/edu/COMS-E6998-012/akv2129/logs/$USER \
    --report_to wandb
"""

import argparse, os, time, json, csv, uuid, math, subprocess, platform
from datetime import datetime
from pathlib import Path
from statistics import median
import numpy as np
import torch

# Optional deps
try:
    import pynvml
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

from datasets import load_dataset, DownloadConfig
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
    EarlyStoppingCallback,
    __version__ as transformers_version,
)
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import EvalPrediction

# --------------------------
# Helpers
# --------------------------

def get_git_commit():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

class LatencyMemCallback(TrainerCallback):
    """
    Tracks per-step latency (excluding first 'warmup_steps') and peak GPU memory.
    """
    def __init__(self, warmup_steps: int = 5):
        self.warmup_steps = warmup_steps
        self.step_timings = []
        self._step_start = None
        self.peak_torch_mem = 0
        self.peak_nvml_mem = 0

        if _HAS_NVML:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._nvml_handle = None
        else:
            self._nvml_handle = None

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.peak_torch_mem = 0
        self.peak_nvml_mem = 0

    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._step_start = time.perf_counter()

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self._step_start is not None:
            dt = time.perf_counter() - self._step_start
            if state.global_step > self.warmup_steps:
                self.step_timings.append(dt)

        if torch.cuda.is_available():
            self.peak_torch_mem = max(self.peak_torch_mem, torch.cuda.max_memory_allocated())
        if self._nvml_handle is not None:
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle).used
                self.peak_nvml_mem = max(self.peak_nvml_mem, mem)
            except Exception:
                pass

    def summary(self):
        def pct(values, q):
            if not values:
                return None
            arr = sorted(values)
            idx = (len(arr)-1) * q
            low = math.floor(idx); high = math.ceil(idx)
            if low == high:
                return arr[int(idx)]
            return arr[low] * (high - idx) + arr[high] * (idx - low)

        lat_p50 = median(self.step_timings) if self.step_timings else None
        lat_p95 = pct(self.step_timings, 0.95)

        return {
            "step_latency_p50_s": lat_p50,
            "step_latency_p95_s": lat_p95,
            "peak_torch_mem_bytes": int(self.peak_torch_mem),
            "peak_nvml_mem_bytes": int(self.peak_nvml_mem),
        }

class EpochTimeCallback(TrainerCallback):
    """
    Minimal per-epoch train-time logger (no eval timing).
    Writes JSONL lines: {"epoch": <float>, "epoch_time": <seconds>}
    """
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self._t0 = None
        self.path = os.path.join(run_dir, "epoch_time.jsonl")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._t0 is None:
            return
        et = time.perf_counter() - self._t0
        rec = {"epoch": float(state.epoch), "epoch_time": et}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        # also log to W&B if available (online or offline)
        try:
            import wandb
            wandb.log(rec)
        except Exception:
            pass

# --------------------------
# Main
# --------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Baseline fine-tuning: DistilBERT on GLUE")
    # Core
    p.add_argument("--model", type=str, default="distilbert-base-uncased")
    p.add_argument("--task", type=str, default="sst2", help="GLUE task key (sst2)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train_bs", type=int, default=32)
    p.add_argument("--eval_bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    # Paths
    p.add_argument("--outdir", type=str, required=True, help="Root: .../outputs/<UNI>/baseline")
    p.add_argument("--logdir", type=str, required=True, help="Root: .../logs/<UNI>")
    p.add_argument("--data_dir", type=str, default=None, help="Optional local dataset dir (unused for SST-2)")
    # Logging / eval
    p.add_argument("--eval_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    p.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--report_to", type=str, default="wandb", help="'wandb' or 'none'")
    p.add_argument("--run_name", type=str, default=None)
    # Precision/toggles
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--grad_ckpt", action="store_true", help="Enable gradient checkpointing")
    p.add_argument("--early_stop", action="store_true")
    p.add_argument("--early_stop_patience", type=int, default=3)
    # Internals
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--local_files_only", action="store_true", help="Force loading only from local HF caches")
    return p.parse_args()

def main():
    args = parse_args()

    # Seed & device
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run IDs & folders
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = args.run_name or f"baseline-{args.model.replace('/', '_')}-{args.task}-{ts}-{uuid.uuid4().hex[:6]}"
    run_dir = ensure_dir(os.path.join(args.outdir, run_id))
    ckpt_dir = ensure_dir(os.path.join(run_dir, "checkpoint"))
    metrics_path = os.path.join(run_dir, "metrics.csv")
    summary_json = os.path.join(run_dir, "summary.json")
    summary_csv = os.path.join(run_dir, "summary.csv")

    # Log env snapshot
    env_meta = {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": get_git_commit(),
        "seed": args.seed,
        "argv": vars(args),
        "hf_home": os.environ.get("HF_HOME"),
        "hf_datasets_cache": os.environ.get("HF_DATASETS_CACHE"),
    }
    with open(os.path.join(run_dir, "env.json"), "w") as f:
        json.dump(env_meta, f, indent=2)

    # Offline toggle
    local_only = args.local_files_only or os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("HF_DATASETS_OFFLINE") == "1"
    dl_cfg = DownloadConfig(local_files_only=local_only)

    # Load dataset (GLUE/SST-2)
    dataset = load_dataset("glue", args.task, download_config=dl_cfg)

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=local_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, local_files_only=local_only
    )
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    model.to(device)

    # Preprocess
    sentence_key = "sentence"  # GLUE/SST-2
    def tokenize_fn(batch):
        out = tokenizer(batch[sentence_key], truncation=True, max_length=args.max_length)
        out["labels"] = batch["label"]
        return out

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["sentence", "label", "idx"])

    # Data collator
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Metrics (accuracy)
    acc_metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"]}

    # Training args
    report_to = [] if args.report_to.lower() in ("none", "off") else [args.report_to]
    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy=args.eval_strategy,   # << correct kw
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        eval_steps=None if args.eval_strategy != "steps" else args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True if args.eval_strategy != "no" else False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=report_to,
        run_name=run_id,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=0,   # safer on Insomnia nodes
        log_level="passive",
    )

    # Callbacks
    latmem_cb = LatencyMemCallback(warmup_steps=5)
    epoch_time_cb = EpochTimeCallback(run_dir)
    callbacks = [epoch_time_cb, latmem_cb]  # epoch time first; simple & safe
    if args.early_stop and args.eval_strategy != "no":
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience))

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # Train
    wall_start = time.perf_counter()
    train_out = trainer.train()
    wall_end = time.perf_counter()
    wall_clock_s = wall_end - wall_start

    # Evaluate on validation
    eval_metrics = trainer.evaluate()

    # Collect history logs and write metrics.csv
    log_hist = trainer.state.log_history  # list of dicts
    fieldnames = sorted({k for row in log_hist for k in row.keys()})
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in log_hist:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # Latency/memory summary
    latmem = latmem_cb.summary()

    # Convert bytes to MB for readability
    def b2mb(x): return round(x / (1024**2), 2) if x else None

    # Best metrics
    best_acc = trainer.state.best_metric if trainer.state.best_metric is not None else eval_metrics.get("eval_accuracy")

    # Read epoch times from JSONL
    epoch_times = []
    et_path = os.path.join(run_dir, "epoch_time.jsonl")
    if os.path.exists(et_path):
        with open(et_path) as f:
            for line in f:
                try:
                    epoch_times.append(float(json.loads(line)["epoch_time"]))
                except Exception:
                    pass
    epoch_time_mean = round(float(np.mean(epoch_times)), 3) if epoch_times else None

    # Compose summary
    summary = {
        "run_id": run_id,
        "model": args.model,
        "dataset": f"glue/{args.task}",
        "epochs": args.epochs,
        "train_batch_size": args.train_bs,
        "eval_batch_size": args.eval_bs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_accum": args.grad_accum,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "grad_ckpt": args.grad_ckpt,
        "seed": args.seed,
        "eval_accuracy": float(best_acc) if best_acc is not None else None,
        "final_eval_loss": float(eval_metrics.get("eval_loss")) if eval_metrics.get("eval_loss") is not None else None,
        "train_samples": len(tokenized["train"]),
        "val_samples": len(tokenized["validation"]),
        "wall_clock_s": round(wall_clock_s, 3),
        "step_latency_p50_s": latmem["step_latency_p50_s"],
        "step_latency_p95_s": latmem["step_latency_p95_s"],
        "peak_mem_torch_MB": b2mb(latmem["peak_torch_mem_bytes"]),
        "peak_mem_nvml_MB": b2mb(latmem["peak_nvml_mem_bytes"]),
        # NEW: epoch times folded into summary
        "epoch_times_s": epoch_times,
        "epoch_time_mean_s": epoch_time_mean,
        "env": {
            "hostname": env_meta["hostname"],
            "python": env_meta["python"],
            "torch": env_meta["torch"],
            "transformers": env_meta["transformers"],
            "cuda_available": env_meta["cuda_available"],
            "cuda_version": env_meta["cuda_version"],
            "gpu_name": env_meta["gpu_name"],
            "git_commit": env_meta["git_commit"],
            "hf_home": env_meta["hf_home"],
            "hf_datasets_cache": env_meta["hf_datasets_cache"],
        }
    }

    # Save summary JSON + CSV
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    # CSV (add epoch_time_mean_s column)
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id","model","dataset","epochs","train_bs","eval_bs","lr","weight_decay",
            "grad_accum","fp16","bf16","grad_ckpt","seed","eval_accuracy","final_eval_loss",
            "wall_clock_s","step_p50_s","step_p95_s",
            "peak_mem_torch_MB","peak_mem_nvml_MB","epoch_time_mean_s"
        ])
        writer.writerow([
            summary["run_id"], summary["model"], summary["dataset"], summary["epochs"],
            summary["train_batch_size"], summary["eval_batch_size"], summary["lr"], summary["weight_decay"],
            summary["grad_accum"], summary["fp16"], summary["bf16"], summary["grad_ckpt"], summary["seed"],
            summary["eval_accuracy"], summary["final_eval_loss"], summary["wall_clock_s"],
            summary["step_latency_p50_s"], summary["step_latency_p95_s"],
            summary["peak_mem_torch_MB"], summary["peak_mem_nvml_MB"], summary["epoch_time_mean_s"]
        ])

    # Friendly print
    print("\n=== Baseline Completed ===")
    print(f"Run ID:      {run_id}")
    print(f"Best Acc:    {summary['eval_accuracy']}")
    print(f"Wall-clock:  {summary['wall_clock_s']} s")
    print(f"Epoch mean:  {summary['epoch_time_mean_s']} s")
    print(f"Peak mem MB: torch={summary['peak_mem_torch_MB']} nvml={summary['peak_mem_nvml_MB']}")
    print(f"Outputs in:  {run_dir}")

if __name__ == "__main__":
    main()


