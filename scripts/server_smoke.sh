#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${STITCH_DATA:=$PWD/data/smoke_generated}"
: "${STITCH_ARTIFACTS:=$PWD/artifacts/smoke}"
: "${STITCH_RESULTS:=$PWD/results/smoke}"
export STITCH_DATA STITCH_ARTIFACTS STITCH_RESULTS

echo "hostname=$(hostname) commit=$(git rev-parse HEAD) python=$(python --version 2>&1)"
python - <<'PY'
import torch, transformers
print(f"torch={torch.__version__} transformers={transformers.__version__} cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()} bf16_supported={torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}")
if not torch.cuda.is_available(): raise SystemExit("CUDA is required by server_smoke.sh")
if not torch.cuda.is_bf16_supported(): raise SystemExit("BF16 is required by this baseline")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY
python scripts/inspect_qwen.py
python scripts/test_self_stitch.py --model Qwen/Qwen3-4B --cuts 18 --device cuda
python scripts/test_self_stitch.py --model Qwen/Qwen3Guard-Gen-4B --cuts 18 --device cuda
python scripts/test_stitched_generation.py --device cuda
python scripts/direct_matching_smoke.py
echo "Preparing a tiny dataset; this performs explicit Hugging Face dataset downloads."
python scripts/prepare_baseline_data.py --tiny --output-dir "$STITCH_DATA" \
  --manifest "$STITCH_DATA/manifest.json" --overwrite
python scripts/run_cut18_baseline.py --data-dir "$STITCH_DATA" \
  --manifest "$STITCH_DATA/manifest.json" --cache-dir "$STITCH_ARTIFACTS/activation_cache" \
  --checkpoint "$STITCH_ARTIFACTS/direct_matching_cut18.pt" --results-dir "$STITCH_RESULTS" \
  --max-epochs 2 --patience 1 --bootstrap 20 --device cuda --overwrite
test -s "$STITCH_ARTIFACTS/direct_matching_cut18.pt"
test -s "$STITCH_RESULTS/cut18_summary.json"
echo "SERVER SMOKE: PASS"
