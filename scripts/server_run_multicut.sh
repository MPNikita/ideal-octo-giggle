#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
test -z "$(git status --porcelain)" || {
  echo "Refusing to run from a dirty worktree" >&2
  exit 2
}

: "${STITCH_DATA:=$PWD/data/generated}"
: "${STITCH_MANIFEST:=$PWD/data/baseline_manifest.json}"
: "${STITCH_ARTIFACTS:=$PWD/artifacts/multicut}"
: "${STITCH_RESULTS:=$PWD/results/multicut}"
: "${STITCH_LOGS:=$PWD/logs}"
: "${STITCH_CUTS:=0,9,18,27,35}"
: "${STITCH_MANIFEST_SHA256:=614209e8f746787d139a4d1d67d6475c30fae365ac21b78531ff704af6d995e0}"
export STITCH_DATA STITCH_MANIFEST STITCH_ARTIFACTS STITCH_RESULTS STITCH_LOGS

mkdir -p "$STITCH_DATA" "$STITCH_ARTIFACTS" "$STITCH_RESULTS" "$STITCH_LOGS"
if [[ "${STITCH_QUEUE_STATUS:-0}" == "1" ]]; then
  python scripts/queue_status.py initialize --cuts "$STITCH_CUTS"
  queue_failure() {
    code=$?
    python scripts/queue_status.py fail "server_run_multicut.sh exited with code $code" || true
    exit "$code"
  }
  trap queue_failure ERR
fi
echo "commit=$(git rev-parse HEAD) cuts=$STITCH_CUTS"
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable: refusing multi-cut server run")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("BF16 unsupported: refusing multi-cut server run")
print(
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"gpu={torch.cuda.get_device_name(0)} bf16={torch.cuda.is_bf16_supported()}"
)
PY

if [[ ! -s "$STITCH_DATA/stitch_train.jsonl" \
   || ! -s "$STITCH_DATA/model_selection.jsonl" \
   || ! -s "$STITCH_DATA/evaluation.jsonl" ]]; then
  python scripts/prepare_baseline_data.py \
    --output-dir "$STITCH_DATA" --manifest "$STITCH_MANIFEST"
fi

python - "$STITCH_MANIFEST" "$STITCH_MANIFEST_SHA256" <<'PY'
import hashlib
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != sys.argv[2]:
    raise SystemExit(f"Manifest hash mismatch: expected {sys.argv[2]}, got {actual}")
print(f"manifest_sha256={actual}")
PY

if [[ "${STITCH_SKIP_SMOKE:-0}" != "1" ]]; then
  bash scripts/server_smoke.sh
fi
log="$STITCH_LOGS/multicut_$(date -u +%Y%m%dT%H%M%SZ).log"
python scripts/run_multicut_baseline.py \
  --cuts "$STITCH_CUTS" \
  --data-dir "$STITCH_DATA" \
  --manifest "$STITCH_MANIFEST" \
  --artifacts-dir "$STITCH_ARTIFACTS" \
  --results-dir "$STITCH_RESULTS" \
  --resume 2>&1 | tee "$log"

test -s "$STITCH_RESULTS/multicut_summary.csv"
test -s "$STITCH_RESULTS/multicut_summary.json"
for cut in ${STITCH_CUTS//,/ }; do
  test -s "$STITCH_ARTIFACTS/direct_matching_cut${cut}.pt"
  test -s "$STITCH_RESULTS/predictions_cut${cut}.csv"
done
echo "QWEN MULTICUT SERVER RUN: PASS"
