#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
test -z "$(git status --porcelain)" || { echo "Refusing dirty worktree" >&2; exit 2; }
: "${STITCH_DATA:=$PWD/data/generated}"
: "${STITCH_ARTIFACTS:=$PWD/artifacts}"
: "${STITCH_RESULTS:=$PWD/results}"
: "${STITCH_LOGS:=$PWD/logs}"
export STITCH_DATA STITCH_ARTIFACTS STITCH_RESULTS STITCH_LOGS
mkdir -p "$STITCH_DATA" "$STITCH_ARTIFACTS" "$STITCH_RESULTS" "$STITCH_LOGS"
python - <<'PY'
import torch
if not torch.cuda.is_available(): raise SystemExit("CUDA unavailable: refusing full run")
print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)} bf16={torch.cuda.is_bf16_supported()}")
PY
echo "commit=$(git rev-parse HEAD)"
if [[ ! -s "$STITCH_DATA/stitch_train.jsonl" || ! -s "$STITCH_DATA/model_selection.jsonl" || ! -s "$STITCH_DATA/evaluation.jsonl" || ! -s "$STITCH_DATA/baseline_manifest.json" ]]; then
  python scripts/prepare_baseline_data.py --output-dir "$STITCH_DATA" \
    --manifest "$STITCH_DATA/baseline_manifest.json"
else
  echo "Reusing prepared data; baseline will validate the manifest and cache metadata."
fi
log="$STITCH_LOGS/cut18_$(date -u +%Y%m%dT%H%M%SZ).log"
python scripts/run_cut18_baseline.py --data-dir "$STITCH_DATA" \
  --manifest "$STITCH_DATA/baseline_manifest.json" --resume 2>&1 | tee "$log"
