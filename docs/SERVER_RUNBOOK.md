# Server runbook: Qwen multi-cut baseline

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
export STITCH_MANIFEST="$PWD/data/baseline_manifest.json"
export STITCH_ARTIFACTS="/persistent/.../probguard/artifacts/multicut"
export STITCH_RESULTS="/persistent/.../probguard/results/multicut"
export STITCH_LOGS="/persistent/.../probguard/logs"
export STITCH_CUTS="0,9,18,27,35"
```

## 3. Smoke and benchmark gate

```bash
bash scripts/server_smoke.sh
```

This explicitly downloads the two models and a tiny dataset, checks exact
revisions/architecture, Base and Guard self-stitch at cuts 0/18/35, cached Guard
generation, three separate tiny adapters, shared extraction, aggregate outputs,
resume metadata, and persistent writes. It never starts the full dataset.

Before the paid full run, use the smoke summary/timestamps and `nvidia-smi` to
record Base extraction throughput, Guard extraction throughput, native and
stitched generation throughput, and peak VRAM. Adjust only the machine-level
execution choice if memory is insufficient; do not alter scientific defaults.

## 4. Full run under tmux

```bash
tmux new -s qwen-multicut
bash scripts/server_run_multicut.sh
```

The wrapper refuses no-CUDA and dirty repositories, prints the commit and
environment, validates the exact manifest, runs smoke, prepares pinned data if
needed, and logs stdout/stderr. Shared train/selection boundary extraction runs
once per model, native Guard JBB evaluation runs once, and each cut has an
independent checkpoint and stitched prediction stage. Rerunning the wrapper
uses `--resume` and reuses a stage only after metadata/hash validation. Use
`--overwrite` only when intentionally replacing explicitly selected paths.

Monitoring commands:

```bash
watch -n 5 nvidia-smi
tmux list-sessions
tail -f "$STITCH_LOGS"/multicut_*.log
pgrep -af python
df -h
```

## 5. Backup before shutdown

Check Python processes, confirm the GPU is idle, and verify all five
checkpoints, per-cut prediction CSVs, aggregate CSV/JSON and log exist. Compute
SHA-256 for each, copy them off the server, recompute and compare hashes, and
only then shut the server down. Activation caches can be retained for resume
but are not scientific results and must not be committed.

## Legacy cut18 regression

The completed cut18 scientific result was produced by
`scripts/run_cut18_baseline.py` at commit
`66f93c1c0ec55cd4e4839b785aaa71532f678de2`. Do not overwrite or reinterpret
that archived result. The legacy wrapper remains available only for exact
cut18 reproduction; the next paid run should use the multi-cut wrapper above.
