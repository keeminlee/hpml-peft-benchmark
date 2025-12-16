import argparse
import csv
import json
import math
import os
import platform
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from datasets import DownloadConfig, load_dataset
from pathlib import Path
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers import __version__ as transformers_version
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import EvalPrediction

try:
    import pynvml

    _HAS_NVML = True
except Exception:
    _HAS_NVML = False


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def default_run_dirs():
    base = Path(__file__).resolve().parent
    outdir = base / "tmp" / "hpml_outputs"
    logdir = base / "tmp" / "hpml_logs"
    return str(outdir), str(logdir)


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


class LatencyMemCallback(TrainerCallback):
    """Tracks per-step latency and peak GPU memory."""

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

    def summary(self) -> Dict[str, Optional[float]]:
        def pct(values, q):
            if not values:
                return None
            arr = sorted(values)
            idx = (len(arr) - 1) * q
            low = math.floor(idx)
            high = math.ceil(idx)
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
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.path = os.path.join(run_dir, "epoch_time.jsonl")
        self._t0 = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._t0 is None:
            return
        et = time.perf_counter() - self._t0
        rec = {"epoch": float(state.epoch), "epoch_time": et}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        try:
            import wandb

            wandb.log(rec)
        except Exception:
            pass


def parse_base_args(description: str, include_peft: bool = False) -> argparse.ArgumentParser:
    default_outdir, default_logdir = default_run_dirs()

    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", type=str, default="distilbert-base-uncased",
                   help="pretrained model name or path")
    p.add_argument("--task", type=str, default="sst2",
                   help="task or dataset name")
    p.add_argument("--epochs", type=int, default=3,
                   help="number of training epochs (ignored if --max_steps > 0)")
    p.add_argument("--train_bs", type=int, default=32,
                   help="training batch size per device")
    p.add_argument("--eval_bs", type=int, default=64,
                   help="evaluation batch size per device")
    p.add_argument("--lr", type=float, default=3e-5,
                   help="learning rate")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed")
    p.add_argument("--outdir", type=str, default=default_outdir,
                   help="output directory for checkpoints and artifacts")
    p.add_argument("--logdir", type=str, default=default_logdir,
                   help="directory for logs and metrics")
    p.add_argument("--report_to", type=str, default="none",
                   choices=["wandb", "none"],
                   help="logging backend")
    p.add_argument("--weight_decay", type=float, default=0.01,
                   help="weight decay coefficient")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="gradient accumulation steps")
    p.add_argument("--max_length", type=int, default=128,
                   help="maximum input sequence length")
    p.add_argument("--warmup_ratio", type=float, default=0.1,
                   help="fraction of steps for learning-rate warmup")
    p.add_argument("--fp16", action="store_true",
                   help="enable FP16 mixed-precision training")
    p.add_argument("--bf16", action="store_true",
                   help="enable BF16 mixed-precision training")
    p.add_argument("--grad_ckpt", action="store_true",
                   help="enable gradient checkpointing")
    p.add_argument("--local_files_only", action="store_true",
                   help="load models/tokenizers from local cache only")
    p.add_argument("--run_name", type=str, default=None,
                   help="optional experiment name")
    p.add_argument("--max_steps", type=int, default=-1,
                   help="maximum number of training steps (overrides epochs)")

    if include_peft:
        p.add_argument("--rank", type=int, default=8,
                       help="LoRA rank (PEFT only)")
        p.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA scaling factor (PEFT only)")
        p.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout probability (PEFT only)")

    return p


def load_sst2(tokenizer, max_length: int, task: str, local_only: bool = False):
    dl_cfg = DownloadConfig(local_files_only=local_only)
    dataset = load_dataset("glue", task, download_config=dl_cfg)

    def tokenize_fn(batch):
        out = tokenizer(batch["sentence"], truncation=True, max_length=max_length)
        out["labels"] = batch["label"]
        return out

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["sentence", "label", "idx"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return tokenized, collator


def compute_metrics_builder():
    import evaluate

    acc_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"]}

    return compute_metrics


def build_env_meta(args) -> Dict:
    return {
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


def get_run_dir(method: str, args) -> Tuple[str, str, str, str]:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = args.run_name or f"{method}-{args.model.replace('/', '_')}-{args.task}-{ts}-{uuid.uuid4().hex[:6]}"
    run_dir = ensure_dir(os.path.join(args.outdir, run_id))
    ckpt_dir = ensure_dir(os.path.join(run_dir, "checkpoint" if method == "baseline" else "adapter"))
    metrics_path = os.path.join(run_dir, "metrics.csv")
    return run_id, run_dir, ckpt_dir, metrics_path


def write_metrics_csv(log_history, metrics_path):
    fieldnames = sorted({k for row in log_history for k in row.keys()})
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in log_history:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def count_parameters(model) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def b2mb(x: Optional[float]) -> Optional[float]:
    return round(x / (1024 ** 2), 2) if x else None


def compute_throughput(train_examples: int, epochs: int, wall_clock_s: float) -> Optional[float]:
    if wall_clock_s == 0:
        return None
    return round((train_examples * epochs) / wall_clock_s, 3)


__all__ = [
    "EpochTimeCallback",
    "LatencyMemCallback",
    "parse_base_args",
    "load_sst2",
    "compute_metrics_builder",
    "build_env_meta",
    "get_run_dir",
    "write_metrics_csv",
    "count_parameters",
    "b2mb",
    "compute_throughput",
    "ensure_dir",
    "get_git_commit",
]
