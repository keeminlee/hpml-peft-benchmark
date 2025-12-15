"""Generate quick visualizations from aggregated results."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parent / "results.csv"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError("reports/results.csv not found. Run scripts/collect_results.py first.")
    return pd.read_csv(RESULTS_PATH)


def plot_accuracy_vs_memory(df: pd.DataFrame):
    plt.figure(figsize=(6, 4))
    for method, group in df.groupby("method"):
        plt.scatter(group["peak_mem_nvml_MB"], group["eval_accuracy"], label=method)
    plt.xlabel("Peak NVML Memory (MB)")
    plt.ylabel("Eval Accuracy")
    plt.title("Accuracy vs GPU Memory")
    plt.legend()
    out = FIG_DIR / "accuracy_vs_memory.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def plot_accuracy_vs_throughput(df: pd.DataFrame):
    plt.figure(figsize=(6, 4))
    for method, group in df.groupby("method"):
        plt.scatter(group["throughput"], group["eval_accuracy"], label=method)
    plt.xlabel("Throughput (samples/s)")
    plt.ylabel("Eval Accuracy")
    plt.title("Accuracy vs Throughput")
    plt.legend()
    out = FIG_DIR / "accuracy_vs_throughput.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def main():
    df = load_results()
    mem_fig = plot_accuracy_vs_memory(df)
    thr_fig = plot_accuracy_vs_throughput(df)
    print(f"Saved {mem_fig} and {thr_fig}")


if __name__ == "__main__":
    main()
