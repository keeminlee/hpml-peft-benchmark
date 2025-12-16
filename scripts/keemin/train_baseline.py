#!/usr/bin/env python
import csv
import json
import os
import time

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed

from scripts.keemin.train_common import (
    EpochTimeCallback,
    LatencyMemCallback,
    b2mb,
    build_env_meta,
    compute_metrics_builder,
    compute_throughput,
    count_parameters,
    get_run_dir,
    load_sst2,
    parse_base_args,
    write_metrics_csv,
)


def main():
    parser = parse_base_args("Baseline fine-tuning")
    args = parser.parse_args()
    set_seed(args.seed)

    run_id, run_dir, ckpt_dir, metrics_path = get_run_dir("baseline", args)
    env_meta = build_env_meta(args)
    with open(os.path.join(run_dir, "env.json"), "w") as f:
        json.dump(env_meta, f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=args.local_files_only)
    tokenized, collator = load_sst2(tokenizer, args.max_length, args.task, args.local_files_only)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, local_files_only=args.local_files_only
    )
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    callbacks = [EpochTimeCallback(run_dir), LatencyMemCallback(warmup_steps=5)]

    report_to = [] if args.report_to == "none" else [args.report_to]
    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        run_name=run_id,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir=os.path.join(args.logdir, run_id),
        logging_steps=25,
        report_to=report_to,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        bf16=args.bf16,
        max_steps=args.max_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_builder(),
        callbacks=callbacks,
    )

    wall_start = time.perf_counter()
    trainer.train()
    wall_clock_s = time.perf_counter() - wall_start

    write_metrics_csv(trainer.state.log_history, metrics_path)

    epoch_times = []
    et_path = os.path.join(run_dir, "epoch_time.jsonl")
    if os.path.exists(et_path):
        with open(et_path) as f:
            for line in f:
                try:
                    epoch_times.append(float(json.loads(line)["epoch_time"]))
                except Exception:
                    pass

    eval_metrics = trainer.evaluate()
    latmem = callbacks[1].summary()
    total_params, trainable_params = count_parameters(trainer.model)
    throughput = compute_throughput(len(tokenized["train"]), args.epochs, wall_clock_s)
    epoch_time_mean = float(sum(epoch_times) / len(epoch_times)) if epoch_times else None

    summary = {
        "run_id": run_id,
        "method": "baseline",
        "model": args.model,
        "dataset": f"glue/{args.task}",
        "rank": None,
        "epochs": args.epochs,
        "train_batch_size": args.train_bs,
        "eval_batch_size": args.eval_bs,
        "lr": args.lr,
        "seed": args.seed,
        "eval_accuracy": float(trainer.state.best_metric or eval_metrics.get("eval_accuracy")),
        "final_eval_loss": float(eval_metrics.get("eval_loss")) if eval_metrics.get("eval_loss") is not None else None,
        "wall_clock_s": round(wall_clock_s, 3),
        "epoch_time_mean_s": round(epoch_time_mean, 3) if epoch_time_mean else None,
        "throughput": throughput,
        "step_latency_p50_s": latmem.get("step_latency_p50_s"),
        "step_latency_p95_s": latmem.get("step_latency_p95_s"),
        "peak_mem_torch_MB": b2mb(latmem.get("peak_torch_mem_bytes")),
        "peak_mem_nvml_MB": b2mb(latmem.get("peak_nvml_mem_bytes")),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "git_commit": env_meta.get("git_commit"),
        "gpu_name": env_meta.get("gpu_name"),
    }

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    csv_fields = [
        "run_id",
        "method",
        "model",
        "dataset",
        "rank",
        "epochs",
        "train_batch_size",
        "eval_batch_size",
        "lr",
        "seed",
        "eval_accuracy",
        "final_eval_loss",
        "wall_clock_s",
        "epoch_time_mean_s",
        "throughput",
        "step_latency_p50_s",
        "step_latency_p95_s",
        "peak_mem_torch_MB",
        "peak_mem_nvml_MB",
        "trainable_params",
        "total_params",
        "git_commit",
        "gpu_name",
    ]
    with open(os.path.join(run_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerow({k: summary.get(k) for k in csv_fields})

    print(f"Run complete. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
