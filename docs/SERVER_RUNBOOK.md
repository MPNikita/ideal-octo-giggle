# Server runbook: cut=18 baseline

## 1. Discover the new machine

Record its IP, SSH port, username, persistent-storage mount, repository path,
GPU/VRAM and driver/CUDA capability. Do not reuse assumptions from an older
server. Run `nvidia-smi` before selecting PyTorch.

## 2. Clone and environment

```bash
git clone https://github.com/MPNikita/ideal-octo-giggle.git
cd ideal-octo-giggle
git rev-parse HEAD
nvidia-smi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Choose an official CUDA-compatible PyTorch wheel from pytorch.org after reading
the driver/GPU information. Do not blindly reuse an old `cu128` build. Then:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/CUDA_INDEX
python -m pip install -r requirements.txt
python -c 'import torch,transformers; print(torch.__version__, torch.version.cuda, transformers.__version__, torch.cuda.is_available())'
```

Replace `CUDA_INDEX` with the compatible official index. Keep Transformers at
5.16.1. Configure persistent paths without committing machine-specific values:

```bash
export HF_HOME="/persistent/.../huggingface"
export STITCH_DATA="/persistent/.../probguard/data"
export STITCH_ARTIFACTS="/persistent/.../probguard/artifacts"
export STITCH_RESULTS="/persistent/.../probguard/results"
export STITCH_LOGS="/persistent/.../probguard/logs"
```

## 3. Smoke and benchmark gate

```bash
bash scripts/server_smoke.sh
```

This explicitly downloads the two models and a tiny dataset, checks exact
revisions/architecture, Base and Guard self-stitch, cached Guard generation,
tiny Direct Matching and persistent writes. It never starts the full dataset.

Before the paid full run, use the smoke summary/timestamps and `nvidia-smi` to
record Base extraction throughput, Guard extraction throughput, native and
stitched generation throughput, and peak VRAM. Adjust only the machine-level
execution choice if memory is insufficient; do not alter scientific defaults.

## 4. Full run under tmux

```bash
tmux new -s cut18
bash scripts/server_run_cut18.sh
```

The wrapper refuses no-CUDA and dirty repositories, prints the commit and
environment, prepares the pinned data, and logs stdout/stderr. To resume after
an interruption, verify artifact metadata and rerun the Python baseline with
`--resume`; use `--overwrite` only when intentionally replacing the selected
output paths.

Monitoring commands:

```bash
watch -n 5 nvidia-smi
tmux list-sessions
tail -f "$STITCH_LOGS"/cut18_*.log
pgrep -af python
df -h
```

## 5. Backup before shutdown

Check Python processes, confirm the GPU is idle, and verify the checkpoint,
prediction CSV, summary JSON and log exist. Compute SHA-256 for each, copy them
off the server, recompute and compare hashes, and only then shut the server
down. Activation caches can be retained for resume but are not scientific
results and must not be committed.
