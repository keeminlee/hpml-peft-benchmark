#!/usr/bin/env python
import argparse
import csv
import json
import os
import time
from typing import Dict, List

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


def load_imdb(max_len: int):
    ds = load_dataset("stanfordnlp/imdb")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_len)

    tokenized = ds.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")
    return tokenizer, tokenized, DataCollatorWithPadding(tokenizer=tokenizer)


def load_sst2(max_len: int):
    ds = load_dataset("glue", "sst2")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

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
    total_loss, total_acc = 0, 0
    for batch in tqdm(dataloader, desc="Validating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)
        total_loss += loss.item()
        total_acc += accuracy(preds, batch["labels"])

    val_loss = total_loss / len(dataloader)
    val_acc = total_acc / len(dataloader)
    return val_loss, val_acc


def top_cuda_ops(prof) -> List[Dict[str, float]]:
    try:
        events = prof.key_averages().table(sort_by="cuda_time_total", row_limit=5)
        rows = []
        for line in events.splitlines()[6:]:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rows.append({"name": parts[-1], "cuda_time_total_ms": float(parts[-2])})
        return rows
    except Exception:
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HPML profiling (single epoch focus)")
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
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    report_dir = os.path.join("reports", args.run_id)
    os.makedirs(report_dir, exist_ok=True)
    tb_log_dir = os.path.join(report_dir, "profiler_traces")
    os.makedirs(tb_log_dir, exist_ok=True)

    with open(os.path.join(report_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.dataset == "sst2":
        tokenizer, tokenized, collator = load_sst2(args.max_len)
        train_split, eval_split = tokenized["train"], tokenized["validation"]
    else:
        tokenizer, tokenized, collator = load_imdb(args.max_len)
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
        model = torch.compile(model, backend="inductor")

    csv_path = os.path.join(report_dir, "epoch_results.csv")
    csv_headers = [
        "run_id",
        "epoch",
        "dataset",
        "model_name",
        "device",
        "compile",
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
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_log_dir),
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    )

    last_epoch_stats = {}
    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        total_loss = total_acc = total_tokens = 0
        data_time = compute_time = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in iterator:
            start_data = time.time()
            compute_start = time.time()
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            preds = outputs.logits.argmax(dim=-1)
            loss.backward()
            optimizer.step()
            compute_time += time.time() - compute_start
            data_time += compute_start - start_data
            total_loss += loss.item()
            total_acc += accuracy(preds, batch["labels"])
            total_tokens += batch["input_ids"].numel()
            if epoch == args.epochs - 1:
                prof.step()

        epoch_time = time.time() - epoch_start
        num_batches = len(train_loader)
        train_loss = total_loss / num_batches
        train_acc = total_acc / num_batches
        throughput = total_tokens / epoch_time if epoch_time else 0
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

        val_loss, val_acc = validate_model(model, eval_loader, device)

        epoch_row = {
            "run_id": args.run_id,
            "epoch": epoch + 1,
            "dataset": args.dataset,
            "model_name": args.model_name,
            "device": str(device),
            "compile": args.compile_mode,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_workers": args.num_workers,
            "seq_len": args.max_len,
            "seed": args.seed,
            "epoch_time_s": epoch_time,
            "data_time_s": data_time,
            "compute_time_s": compute_time,
            "tokens_seen": total_tokens,
            "throughput_tok_s": throughput,
            "peak_vram_gb": peak_mem_gb,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writerow(epoch_row)

        if epoch == args.epochs - 1:
            last_epoch_stats = epoch_row

    prof.stop()

    summary = {
        "run_id": args.run_id,
        "compile_mode": args.compile_mode,
        "dataset": args.dataset,
        "peak_vram_gb": last_epoch_stats.get("peak_vram_gb"),
        "throughput_tok_s": last_epoch_stats.get("throughput_tok_s"),
        "top_cuda_ops": top_cuda_ops(prof),
    }
    with open(os.path.join(report_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Profiling complete. Reports saved to {report_dir}")
