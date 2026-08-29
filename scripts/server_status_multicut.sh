#!/usr/bin/env bash
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/probguard_multicut}"
STITCH_LOGS="${STITCH_LOGS:-$PROJECT_ROOT/logs}"
STITCH_RESULTS="${STITCH_RESULTS:-$PROJECT_ROOT/results/multicut}"
status_file="$STITCH_LOGS/queue_status.json"
log_file="$STITCH_LOGS/multicut_queue.log"

process_output="$(pgrep -af "run_multicut_baseline.py|server_run_multicut.sh" || true)"
if [[ -n "$process_output" ]]; then process_running=YES; else process_running=NO; fi
if tmux has-session -t qwen-multicut 2>/dev/null; then tmux_running=YES; else tmux_running=NO; fi

STATUS_FILE="$status_file" PROCESS_RUNNING="$process_running" \
TMUX_RUNNING="$tmux_running" LOG_FILE="$log_file" RESULTS_DIR="$STITCH_RESULTS" \
python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["STATUS_FILE"])
process_running = os.environ["PROCESS_RUNNING"]
tmux_running = os.environ["TMUX_RUNNING"]
if not path.exists():
    print("QUEUE:\nNOT STARTED")
    print(f"\nPROCESS:\nrunning: {process_running}")
    print(f"\nTMUX:\nqwen-multicut: {tmux_running}")
    raise SystemExit

try:
    status = json.loads(path.read_text(encoding="utf-8"))
except Exception as error:
    print("QUEUE:\nSTALE / POSSIBLY FAILED")
    print(f"\nFAILURE:\ninvalid status file: {error}")
    raise SystemExit

state = status.get("queue_state", "NOT_STARTED")
display_state = (
    "STALE / POSSIBLY FAILED"
    if state == "RUNNING" and process_running == "NO"
    else state
)
print(f"QUEUE:\n{display_state}")
print(f"\nPHASE:\n{status.get('phase') or '-'}")
print(f"\nCUT:\n{status.get('current_cut') if status.get('current_cut') is not None else '-'}")

adapters = status.get("completed_adapter_cuts", [])
evaluations = status.get("completed_eval_cuts", [])
print("\nCOMPLETED ADAPTERS:")
print(", ".join(map(str, adapters)) or "-")
print(f"{len(adapters)} / 5")
print("\nCOMPLETED STITCHED EVALS:")
print(", ".join(map(str, evaluations)) or "-")
print(f"{len(evaluations)} / 5")
print(f"\nNEXT CUT:\n{status.get('next_cut') if status.get('next_cut') is not None else '-'}")
print(f"\nPROCESS:\nrunning: {process_running}\npid: {status.get('pid') or '-'}")
print(f"\nTMUX:\nqwen-multicut: {tmux_running}")

def parse_time(value):
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))

now = datetime.now(timezone.utc)
start = parse_time(status.get("start_time"))
updated = parse_time(status.get("last_update"))
elapsed = (now - start).total_seconds() / 60 if start else None
age = (now - updated).total_seconds() if updated else None
print("\nTIME:")
print(f"started: {status.get('start_time') or '-'}")
print(f"elapsed: {elapsed:.1f} min" if elapsed is not None else "elapsed: -")
print(f"last update: {age:.0f} sec ago" if age is not None else "last update: -")
print(f"\nLOG:\n{os.environ['LOG_FILE']}")
print(f"\nRESULTS:\n{os.environ['RESULTS_DIR']}")
if status.get("failure"):
    print(f"\nFAILURE:\n{status['failure']}")
PY

echo
echo "GPU:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi \
    --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader,nounits
else
  echo "unavailable"
fi
