# PyTorch Profiling Script

This script profiles a DistilBert model on the IMDB dataset. It's designed to compare PyTorch's `eager` mode against the `inductor` backend (`torch.compile`) and log detailed performance metrics (timing, throughput, memory).

---

1. Setup & Installation


Install Dependencies:

pip install torch transformers datasets wandb tqdm

Note on PyTorch: For GPU support, ensure you install the correct PyTorch version for your system's CUDA driver. You can get the specific command from pytorch.org.

2. How to Run
The script is controlled by command-line arguments. Here are the two main experiments:

Baseline Run (Eager Mode, 5 Epochs):

python profile_analysis.py --optimizer AdamW --compile_mode eager --num_workers 4 --epochs 5 --run_id baseline_run

Compiled Run (Inductor, 10 Epochs):

python profile_analysis.py --optimizer AdamW --compile_mode inductor --num_workers 4 --epochs 10 --run_id compile_run


3. Changing the W&B Project
To log runs to your own W&B project, you must edit the script:

Open profile_analysis.py.

Find this line (around line 84):

wandb.init(project="hpml-project_trial", name=args.run_id, mode="offline", config=args)
Change "hpml-project_trial" to your new project name.

4. Viewing Results (Syncing W&B)
The script runs in offline mode, so data is saved locally first.

After a run finishes, the terminal output will show a path:

Example: "wandb: Run data is saved locally in ./wandb/offline-run-20251108_171438-eu0i83qk"
To upload the results, activate your environment and use the wandb sync command with that path:

# (Activate your environment first!)
wandb sync ./wandb/offline-run-20251108_171438-eu0i83qk

You can then view the full, interactive plots on the W&B website.