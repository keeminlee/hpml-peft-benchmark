# python profile_analysis.py --optimizer AdamW --compile_mode eager --num_workers 4 --epochs 5 --run_id baseline_run
# python profile_analysis.py --optimizer AdamW --compile_mode inductor --num_workers 4 --epochs 10 --run_id compile_run


import torch
import transformers
import wandb, os, shutil, time, json, csv, argparse
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DataCollatorWithPadding, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD, Adam
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity

# ------------------------------
# 4. Helper Functions
# (It's safe for functions to be defined outside the block)
# ------------------------------
def accuracy(preds, labels):
    return (preds == labels).float().mean().item()

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

# ------------------------------
# NEW: Main execution block
# ------------------------------
# All your script's logic goes inside here
if __name__ == '__main__':

    # ------------------------------
    # 0. Setup Command-Line Arguments
    # ------------------------------
    parser = argparse.ArgumentParser(description="HPML LLM Profiling Script")
    parser.add_argument('--run_id', type=str, default=f"run_{time.strftime('%Y%m%d-%H%M%S')}",
                        help="Unique ID for this run")
    parser.add_argument('--model_name', type=str, default="distilbert-base-uncased",
                        help="HuggingFace model name")
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam', 'SGD'],
                        help="Optimizer to use")
    parser.add_argument('--compile_mode', type=str, default='eager', choices=['eager', 'inductor'],
                        help="Torch compile mode ('eager' means no compile)")
    parser.add_argument('--num_workers', type=int, default=0,  # Defaulting to 0 for Windows might be safer
                        help="Number of workers for DataLoader")
    parser.add_argument('--epochs', type=int, default=5,
                        help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument('--max_len', type=int, default=256,
                        help="Max sequence length")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)

    # ------------------------------
    # 1. Setup Folders & W&B
    # ------------------------------
    report_dir = os.path.join("reports", args.run_id)
    os.makedirs(report_dir, exist_ok=True)
    tb_log_dir = os.path.join(report_dir, "profiler_traces")
    os.makedirs(tb_log_dir, exist_ok=True)

    with open(os.path.join(report_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    wandb.init(project="hpml-project_trial", name=args.run_id, mode="offline", config=args)
    print(f"Run Configuration: {args}")

    # ------------------------------
    # 2. Load dataset & tokenizer
    # ------------------------------
    print("Loading dataset...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_dataset("stanfordnlp/imdb")
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_len)

    tokenized_datasets = ds.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets.set_format("torch")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.num_workers
    )

    test_dataloader = DataLoader(
        tokenized_datasets["test"],
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=args.num_workers
    )

    # ------------------------------
    # 3. Model, device, optimizer
    # ------------------------------
    print(f"Loading model: {args.model_name}")
    model = DistilBertForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)

    if args.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr)

    if args.compile_mode == 'inductor':
        print("Compiling model with torch.compile(backend='inductor')...")
        model = torch.compile(model, backend="inductor")
    else:
        print("Running in eager mode.")

    # ------------------------------
    # 5. Per-Epoch Training & Profiling Loop
    # ------------------------------
    print(f"Starting training for {args.epochs} epochs...")

    csv_path = os.path.join(report_dir, "epoch_results.csv")
    csv_headers = [
        "run_id", "epoch", "model_name", "device", "compile", "batch_size", "lr",
        "num_workers", "seq_len", "seed", "epoch_time_s", "data_time_s",
        "compute_time_s", "tokens_seen", "throughput_tok_s", "peak_vram_gb",
        "train_loss", "train_acc", "test_loss", "test_acc"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

    profiler = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1, 
            warmup=1, 
            active=5,
            skip_first=10,
            repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_log_dir),
        record_shapes=True,
        with_stack=True,
        profile_memory=True
    )

    profiler.start()

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1} / {args.epochs} ---")
        model.train()
        
        epoch_start_time = time.time()
        total_loss, total_acc, total_tokens = 0, 0, 0
        total_data_time, total_compute_time = 0, 0
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        data_timer = time.time()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            total_data_time += (time.time() - data_timer)
            
            compute_timer = time.time()
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            if epoch == args.epochs - 1:
                with record_function("model_forward"):
                    outputs = model(**batch)
                loss = outputs.loss
                preds = outputs.logits.argmax(dim=-1)
                with record_function("model_backward"):
                    loss.backward()
                with record_function("optimizer_step"):
                    optimizer.step()
                profiler.step()
            else:
                outputs = model(**batch)
                loss = outputs.loss
                preds = outputs.logits.argmax(dim=-1)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            total_acc += accuracy(preds, batch["labels"])
            total_tokens += batch["input_ids"].numel()
            total_compute_time += (time.time() - compute_timer)
            
            data_timer = time.time()
        
        profiler.stop()
        
        epoch_time = time.time() - epoch_start_time
        num_batches = len(train_dataloader)
        train_loss = total_loss / num_batches
        train_acc = total_acc / num_batches
        throughput = total_tokens / epoch_time
        
        if torch.cuda.is_available():
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            peak_mem_gb = 0.0

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s, Data Time: {total_data_time:.2f}s, Compute Time: {total_compute_time:.2f}s")
        print(f"Throughput: {throughput:.2f} tokens/s, Peak VRAM: {peak_mem_gb:.2f} GB")

        test_loss, test_acc = validate_model(model, test_dataloader, device)
        print(f"Epoch {epoch+1}: Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

        epoch_results = {
            "run_id": args.run_id,
            "epoch": epoch + 1,
            "model_name": args.model_name,
            "device": str(device),
            "compile": args.compile_mode != 'eager',
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_workers": args.num_workers,
            "seq_len": args.max_len,
            "seed": args.seed,
            "epoch_time_s": epoch_time,
            "data_time_s": total_data_time,
            "compute_time_s": total_compute_time,
            "tokens_seen": total_tokens,
            "throughput_tok_s": throughput,
            "peak_vram_gb": peak_mem_gb,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        }

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writerow(epoch_results)
        
        wandb.log(epoch_results, step=epoch + 1)

    wandb.finish()
    print(f"Training finished. All reports saved to: {report_dir}")