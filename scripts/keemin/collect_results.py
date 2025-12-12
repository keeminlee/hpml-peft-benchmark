#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path

REQUIRED_FIELDS = [
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
    "peak_mem_torch_MB",
    "peak_mem_nvml_MB",
    "trainable_params",
    "total_params",
    "git_commit",
    "gpu_name",
]


def gather_records(root: Path):
    records = []
    for path in root.rglob("summary.json"):
        try:
            with open(path) as f:
                summary = json.load(f)
            row = {k: summary.get(k) for k in REQUIRED_FIELDS}
            records.append(row)
        except Exception:
            continue
    return records


def main():
    parser = argparse.ArgumentParser(description="Aggregate run summaries into reports/results.csv")
    parser.add_argument("--root", type=str, default="outputs", help="Root directory to scan for summary.json files")
    parser.add_argument(
        "--out", type=str, default="reports/results.csv", help="Path to write aggregated CSV"
    )
    args = parser.parse_args()

    root = Path(args.root)
    records = gather_records(root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_FIELDS)
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    print(f"Wrote {len(records)} rows to {out_path}")


if __name__ == "__main__":
    main()
