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
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ------------------------
# Utilities
# ------------------------

def accuracy(preds, labels):
    return (preds == labels).float().mean().item()


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------
# Dataset loaders
# ------------------------

def load_sst2(max_len: int, model_name: str):
    ds = load_dataset("glue", "sst2")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    def tokenize(examples):
        out = tokenizer(examples["sentence"], truncation=True, max_length=max_len)
        out["labels"] = examples["label"]
        return out

    tokenized = ds.map(tokenize, batched=True, remove_columns=["sentence", "label", "idx"])
    tokenized.set_format("torch")
    return tokenizer, tokenized, DataCollatorWithPadding(tokenizer)


# ------------------------
# Validation
# ------------------------

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    loss_sum, acc_sum = 0.0, 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        preds = out.logits.argmax(dim=-1)
        loss_sum += float(out.loss.item())
        acc_sum += accuracy(preds, batch["labels"])
    return loss_sum / len(loader), acc_sum / len(loader)


# ------------------------
# Profiler helpers
# ------------------------

def top_cuda_ops_from_prof(prof, k=8):
    rows = []
    try:
        avgs = prof.key_averages()
        for e in sorted(avgs, key=lambda x: getattr(x, "cuda_time_total", 0.0), reverse=True)[:k]:
            rows.append(
                {
                    "name": e.key,
                    "cuda_time_total_ms": float(getattr(e, "cuda_time_total", 0.0)) / 1000.0,
                    "cpu_time_total_ms": float(getattr(e, "cpu_time_total", 0.0)) / 1000.0,
                    "count": int(getattr(e, "count", 0)),
                }
            )
    except Exception:
        pass
    return rows


# ------------------------
# Model factory
# ------------------------

def build_model(args, device):
    """
    Returns (model, optimizer).
    """
    if args.method == "baseline":
        model = DistilBertForSequenceClassification.from_pretrained(
            args.model_name, num_labels=2
        ).to(device)

    elif args.method == "lora":
        model = DistilBertForSequenceClassification.from_pretrained(
            args.model_name, num_labels=2
        ).to(device)

        lora_cfg = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank * 2,
            target_modules=["q_lin", "v_lin"],
            lora_dropout=0.05,
            bias="none",
            task_type=None,
            modules_to_save=[],
        )
        model = get_peft_model(model, lora_cfg)

    elif args.method == "qlora":
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA requires CUDA")

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = DistilBertForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)

        lora_cfg = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank * 2,
            target_modules=["q_lin", "v_lin"],
            lora_dropout=0.05,
            bias="none",
            task_type=None,
            modules_to_save=[],
        )
        model = get_peft_model(model, lora_cfg)

    else:
        raise ValueError(f"Unknown method {args.method}")

    if args.optimizer == "AdamW":
        opt = AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == "Adam":
        opt = Adam(model.parameters(), lr=args.lr)
    else:
        opt = SGD(model.parameters(), lr=args.lr)

    return model, opt


# ------------------------
# Main
# ------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("HPML Profiling (Baseline / LoRA / QLoRA)")
    parser.add_argument("--run_id", type=str, default=f"run_{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--method", choices=["baseline", "lora", "qlora"], required=True)
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--dataset", default="sst2", choices=["sst2"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--compile_mode", choices=["eager", "inductor"], default="eager")
    parser.add_argument("--optimizer", choices=["AdamW", "Adam", "SGD"], default="AdamW")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # QLoRA safety
    if args.method == "qlora" and args.compile_mode == "inductor":
        print("[INFO] Disabling torch.compile for QLoRA")
        args.compile_mode = "eager"

    report_dir = os.path.join("reports", args.run_id)
    os.makedirs(report_dir, exist_ok=True)
    tb_dir = os.path.join(report_dir, "profiler_traces")
    os.makedirs(tb_dir, exist_ok=True)

    with open(os.path.join(report_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    tokenizer, tokenized, collator = load_sst2(args.max_len, args.model_name)
    train_loader = DataLoader(
        tokenized["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        tokenized["validation"],
        batch_size=args.batch_size,
        collate_fn=collator,
    )

    model, optimizer = build_model(args, device)

    if args.compile_mode == "inductor":
        model = torch.compile(model)

    csv_path = os.path.join(report_dir, "epoch_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "epoch_time_s",
                "throughput_tok_s",
                "peak_vram_gb",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
            ]
        )

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        if torch.cuda.is_available()
        else [ProfilerActivity.CPU],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_dir),
        profile_memory=True,
        record_shapes=True,
    )

    for epoch in range(args.epochs):
        model.train()
        start = time.time()
        total_loss, total_acc, total_tokens = 0.0, 0.0, 0

        if epoch == args.epochs - 1:
            prof.start()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(**batch)
            out.loss.backward()
            optimizer.step()

            preds = out.logits.argmax(dim=-1)
            total_loss += float(out.loss.item())
            total_acc += accuracy(preds, batch["labels"])
            total_tokens += int(batch["input_ids"].numel())

            if epoch == args.epochs - 1:
                prof.step()

        if epoch == args.epochs - 1:
            prof.stop()

        epoch_time = time.time() - start
        peak_mem = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
            if torch.cuda.is_available()
            else 0.0
        )

        val_loss, val_acc = validate(model, eval_loader, device)

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    epoch_time,
                    total_tokens / epoch_time,
                    peak_mem,
                    total_loss / len(train_loader),
                    total_acc / len(train_loader),
                    val_loss,
                    val_acc,
                ]
            )

    summary = {
        "method": args.method,
        "compile_mode": args.compile_mode,
        "peak_vram_gb": peak_mem,
        "top_ops": top_cuda_ops_from_prof(prof, k=10),
        "tensorboard_trace_dir": tb_dir,
    }
    with open(os.path.join(report_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Profiling complete. Results in {report_dir}")
