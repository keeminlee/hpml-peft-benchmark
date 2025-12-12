#!/usr/bin/env python
import argparse
import csv
import json
import os
import time
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from torch.optim import Adam, AdamW, SGD
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)


def accuracy(preds, labels):
    return (preds == labels).float().mean().item()


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_imdb(max_len: int, model_name: str):
    ds = load_dataset("stanfordnlp/imdb")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_len)

    tokenized = ds.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")
    return tokenizer, tokenized, DataCollatorWithPadding(tokenizer=tokenizer)


def load_sst2(max_len: int, model_name: str):
    ds = load_dataset("glue", "sst2")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    def tokenize_function(examples):
        out = tokenizer(examples["sentence"], truncation=True, max_length=max_len)
        out["labels"] = examples["label"]
        return out

    tokenized = ds.map(tokenize_function, batched=True, remove_columns=["sentence", "label", "idx"])
    tokenized.set_format("torch")
    return tokenizer, tokenized, DataCollatorWithPadding(tokenizer=tokenizer)


@torch.no_grad()
def validate_model(model, dataloader, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    for batch in tqdm(dataloader, desc="Validating", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)
        total_loss += float(loss.item())
        total_acc += float(accuracy(preds, batch["labels"]))
    val_loss = total_loss / max(1, len(dataloader))
    val_acc = total_acc / max(1, len(dataloader))
    return val_loss, val_acc


def top_cuda_ops_from_prof(prof, k: int = 8) -> List[Dict[str, float]]:
    """
    Robust extraction: don't parse printed tables; use key_averages.
    Returns top-k by total CUDA time (ms) if available.
    """
    rows: List[Dict[str, float]] = []
    try:
        avgs = prof.key_averages()
        # Some builds may not have CUDA timings; guard
        # We'll sort by cuda_time_total if present, else fallback to cpu_time_total.
        def get_cuda_ms(evt):
            # torch profiler reports microseconds; docs: cuda_time_total in us
            v = getattr(evt, "cuda_time_total", 0.0)
            return float(v) / 1000.0

        def get_cpu_ms(evt):
            v = getattr(evt, "cpu_time_total", 0.0)
            return float(v) / 1000.0

        # Prefer CUDA time if any event has it
        any_cuda = any(getattr(e, "cuda_time_total", 0.0) > 0 for e in avgs)
        key_fn = (lambda e: get_cuda_ms(e)) if any_cuda else (lambda e: get_cpu_ms(e))

        top = sorted(avgs, key=key_fn, reverse=True)[:k]
        for e in top:
            rows.append(
                {
                    "name": e.key,
                    "cuda_time_total_ms": (float(getattr(e, "cuda_time_total", 0.0)) / 1000.0),
                    "cpu_time_total_ms": (float(getattr(e, "cpu_time_total", 0.0)) / 1000.0),
                    "count": int(getattr(e, "count", 0)),
                }
            )
    except Exception:
        return []
    return rows


def try_init_wandb(args) -> Optional[object]:
    """
    Opt-in W&B. If wandb isn't installed or init fails, return None.
    """
    if args.report_to != "wandb" or args.wandb_mode == "disabled":
        return None
    try:
        import wandb  # type: ignore
        os.environ["WANDB_MODE"] = args.wandb_mode  # offline/online
        run = wandb.init(project=args.wandb_project, name=args.run_id, config=vars(args))
        return run
    except Exception as e:
        print(f"[WARN] W&B requested but unavailable ({e}). Continuing without W&B.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HPML profiling (profile last epoch)")
    parser.add_argument("--run_id", type=str, default=f"run_{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "imdb"])
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW", "Adam", "SGD"])
    parser.add_argument("--compile_mode", type=str, default="eager", choices=["eager", "inductor"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    # W&B opt-in
    parser.add_argument("--report_to", type=str, default="none", choices=["none", "wandb"])
    parser.add_argument("--wandb_project", type=str, default="hpml-project_trial")
    parser.add_argument("--wandb_mode", type=str, default="offline", choices=["offline", "online", "disabled"])

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    report_dir = os.path.join("reports", args.run_id)
    os.makedirs(report_dir, exist_ok=True)
    tb_log_dir = os.path.join(report_dir, "profiler_traces")
    os.makedirs(tb_log_dir, exist_ok=True)

    with open(os.path.join(report_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    wandb_run = try_init_wandb(args)

    if args.dataset == "sst2":
        tokenizer, tokenized, collator = load_sst2(args.max_len, args.model_name)
        train_split, eval_split = tokenized["train"], tokenized["validation"]
    else:
        tokenizer, tokenized, collator = load_imdb(args.max_len, args.model_name)
        train_split, eval_split = tokenized["train"], tokenized["test"]

    train_loader = DataLoader(
        train_split,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    eval_loader = DataLoader(
        eval_split,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    model = DistilBertForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)

    if args.optimizer == "AdamW":
        optimizer = AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == "Adam":
        optimizer = Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = SGD(model.parameters(), lr=args.lr)

    if args.compile_mode == "inductor":
        # Note: torch.compile may not be available on older torch installs.
        model = torch.compile(model, backend="inductor")

    csv_path = os.path.join(report_dir, "epoch_results.csv")
    csv_headers = [
        "run_id",
        "epoch",
        "dataset",
        "model_name",
        "device",
        "compile_mode",
        "batch_size",
        "lr",
        "num_workers",
        "seq_len",
        "seed",
        "epoch_time_s",
        "data_time_s",
        "compute_time_s",
        "tokens_seen",
        "throughput_tok_s",
        "peak_vram_gb",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "profiled",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

    # Profiler will write traces to TensorBoard directory.
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        if torch.cuda.is_available()
        else [ProfilerActivity.CPU],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_log_dir),
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    )

    last_epoch_stats: Dict[str, object] = {}

    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()

        total_loss = 0.0
        total_acc = 0.0
        total_tokens = 0

        data_time_s = 0.0
        compute_time_s = 0.0

        profiled = (epoch == args.epochs - 1)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Start profiler ONLY for the last epoch
        if profiled:
            prof.start()

        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        for batch in iterator:
            # Measure "data time" as time between iterations (including collate + host work)
            t0 = time.time()

            # Move to device (part of data time for our breakdown)
            batch = {k: v.to(device) for k, v in batch.items()}

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            data_time_s += (t1 - t0)

            # Compute (forward+backward+step)
            optimizer.zero_grad(set_to_none=True)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.time()

            outputs = model(**batch)
            loss = outputs.loss
            preds = outputs.logits.argmax(dim=-1)
            loss.backward()
            optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t3 = time.time()
            compute_time_s += (t3 - t2)

            total_loss += float(loss.item())
            total_acc += float(accuracy(preds, batch["labels"]))
            total_tokens += int(batch["input_ids"].numel())

            if profiled:
                prof.step()

            avg_loss = total_loss / max(1, (iterator.n))
            avg_acc = total_acc / max(1, (iterator.n))
            iterator.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

        # Stop profiler at end of last epoch
        if profiled:
            prof.stop()

        epoch_time = time.time() - epoch_start
        num_batches = max(1, len(train_loader))
        train_loss = total_loss / num_batches
        train_acc = total_acc / num_batches
        throughput = (total_tokens / epoch_time) if epoch_time else 0.0
        peak_mem_gb = (
            float(torch.cuda.max_memory_allocated()) / (1024**3) if torch.cuda.is_available() else 0.0
        )

        val_loss, val_acc = validate_model(model, eval_loader, device)

        epoch_row = {
            "run_id": args.run_id,
            "epoch": epoch + 1,
            "dataset": args.dataset,
            "model_name": args.model_name,
            "device": str(device),
            "compile_mode": args.compile_mode,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_workers": args.num_workers,
            "seq_len": args.max_len,
            "seed": args.seed,
            "epoch_time_s": epoch_time,
            "data_time_s": data_time_s,
            "compute_time_s": compute_time_s,
            "tokens_seen": total_tokens,
            "throughput_tok_s": throughput,
            "peak_vram_gb": peak_mem_gb,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "profiled": profiled,
        }

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writerow(epoch_row)

        if wandb_run is not None:
            wandb_run.log(epoch_row, step=epoch + 1)

        if profiled:
            last_epoch_stats = epoch_row

    summary = {
        "run_id": args.run_id,
        "compile_mode": args.compile_mode,
        "dataset": args.dataset,
        "peak_vram_gb": last_epoch_stats.get("peak_vram_gb"),
        "throughput_tok_s": last_epoch_stats.get("throughput_tok_s"),
        "top_ops": top_cuda_ops_from_prof(prof, k=10),
        "tensorboard_trace_dir": tb_log_dir,
        "csv_path": csv_path,
    }
    with open(os.path.join(report_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if wandb_run is not None:
        wandb_run.finish()

    print(f"Profiling complete. Reports saved to {report_dir}")
