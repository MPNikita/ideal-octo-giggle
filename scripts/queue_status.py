"""Minimal atomic status file for the multi-cut server queue."""

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def enabled():
    return os.environ.get("STITCH_QUEUE_STATUS", "0") == "1"


def status_path():
    project_root = Path(
        os.environ.get("PROJECT_ROOT", Path.home() / "probguard_multicut")
    )
    logs = Path(os.environ.get("STITCH_LOGS", project_root / "logs"))
    return logs / "queue_status.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(payload):
    if not enabled():
        return
    path = status_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["last_update"] = now()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def read_status():
    path = status_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def initialize(cuts):
    if not enabled():
        return
    timestamp = now()
    write_status(
        {
            "queue_state": "RUNNING",
            "phase": "SETUP",
            "current_cut": None,
            "completed_cuts": [],
            "completed_adapter_cuts": [],
            "completed_eval_cuts": [],
            "next_cut": cuts[0] if cuts else None,
            "cuts": list(cuts),
            "start_time": timestamp,
            "pid": os.getpid(),
            "failure": None,
        }
    )


def update(**changes):
    if not enabled():
        return
    payload = read_status()
    if not payload:
        initialize([])
        payload = read_status()
    payload.update(changes)
    payload["pid"] = os.getpid()
    write_status(payload)


def fail(message):
    if not enabled():
        return
    existing = read_status()
    if existing.get("queue_state") == "FAILED" and existing.get("failure"):
        message = existing["failure"]
    update(
        queue_state="FAILED",
        phase="FAILED",
        failure=str(message).replace("\n", " ")[:500],
    )


def parse_cuts(value):
    return [int(part) for part in value.split(",") if part]


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize_parser = subparsers.add_parser("initialize")
    initialize_parser.add_argument("--cuts", type=parse_cuts, required=True)
    fail_parser = subparsers.add_parser("fail")
    fail_parser.add_argument("message")
    args = parser.parse_args()
    if args.command == "initialize":
        initialize(args.cuts)
    else:
        fail(args.message)


if __name__ == "__main__":
    main()
