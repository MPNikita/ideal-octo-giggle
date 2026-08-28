"""Prepare fixed, disjoint train/selection/JBB JSONL files and an audit manifest."""

import argparse
import hashlib
import json
import random
import re
import statistics
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from stitching_core import GUARD, REVISIONS, render_guard_prompt

SEED = 20260827
DATASETS = {
    "wildchat": {
        "repository": "allenai/WildChat-1M",
        "revision": "7d6490e462285cf85d91eabea0f9a954fbddcd1f",
        "config": "default", "split": "train",
        "fields_used": ["conversation", "conversation.role", "conversation.content",
                        "conversation.language", "conversation.toxic", "conversation.redacted",
                        "toxic", "redacted"],
    },
    "beavertails": {
        "repository": "PKU-Alignment/BeaverTails",
        "revision": "8401fe609d288129cc684a9b3be6a93e41cfe678",
        "config": "default", "split": "330k_train",
        "fields_used": ["prompt"],
        "note": "response/is_safe/category are intentionally not used",
    },
    "jbb": {
        "repository": "JailbreakBench/JBB-Behaviors",
        "revision": "886acc352a31533ffbcf4ef22c744658688086fc",
        "config": "behaviors", "splits": ["harmful", "benign"],
        "fields_used": ["Index", "Goal"],
    },
}


def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_candidates(rows, source, text_getter, wanted, excluded, stats, predicate=lambda row: True):
    selected, exact, normalized_seen = [], set(), set(excluded)
    for row in rows:
        stats["rows_scanned"] += 1
        if not predicate(row):
            stats["filtered"] += 1
            continue
        text = text_getter(row)
        if not isinstance(text, str) or not text.strip():
            stats["filtered"] += 1
            continue
        text = text.strip()
        key = normalize(text)
        if text in exact:
            stats["exact_duplicates"] += 1
            continue
        exact.add(text)
        if key in normalized_seen:
            stats["normalized_duplicates_or_overlap"] += 1
            continue
        normalized_seen.add(key)
        selected.append({"source": source, "text": text})
        if len(selected) == wanted:
            break
    return selected, normalized_seen


def wildchat_text(row):
    users = [m for m in row["conversation"] if m.get("role") == "user"]
    return users[-1].get("content") if users else None


def wildchat_ok(row):
    if row.get("toxic") or row.get("redacted") or row.get("language") != "English":
        return False
    users = [m for m in row["conversation"] if m.get("role") == "user"]
    return bool(users) and not users[-1].get("toxic") and not users[-1].get("redacted") \
        and users[-1].get("language") == "English"


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def length_stats(values):
    values = list(map(int, values))
    return {"min": min(values), "median": statistics.median(values),
            "mean": statistics.fmean(values), "p95": float(np.percentile(values, 95)),
            "max": max(values)}


def token_audit(tokenizer, groups):
    result = {}
    for name, rows in groups.items():
        raw_lengths, rendered_lengths, overheads = [], [], []
        for row in rows:
            raw = len(tokenizer(row["text"], add_special_tokens=False).input_ids)
            rendered = render_guard_prompt(tokenizer, row["text"])
            full = len(tokenizer(rendered, add_special_tokens=False).input_ids)
            raw_lengths.append(raw); rendered_lengths.append(full); overheads.append(full - raw)
        result[name] = {"raw_tokens": length_stats(raw_lengths),
                        "guard_template_tokens": length_stats(rendered_lengths),
                        "template_overhead_tokens": length_stats(overheads),
                        "truncated_examples": 0}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--manifest", type=Path, default=Path("data/baseline_manifest.json"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tiny", action="store_true", help="2+2 train, 1+1 selection, 2+2 JBB")
    parser.add_argument("--max-direct-tokens", type=int, default=2048,
                        help="Exclude (never truncate) longer train/selection candidates")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Refusing non-empty {args.output_dir}; use --overwrite explicitly")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    per_source = 3 if args.tiny else 500
    train_n = 2 if args.tiny else 400
    eval_n = 2 if args.tiny else 100
    stats = {name: {"rows_scanned": 0, "filtered": 0, "exact_duplicates": 0,
                    "normalized_duplicates_or_overlap": 0, "length_excluded": 0}
             for name in ("wildchat", "beavertails")}
    tokenizer = AutoTokenizer.from_pretrained(GUARD, revision=REVISIONS[GUARD])
    def within_limit(text, source):
        if not isinstance(text, str): return False
        rendered = render_guard_prompt(tokenizer, text)
        ok = len(tokenizer(rendered, add_special_tokens=False).input_ids) <= args.max_direct_tokens
        if not ok: stats[source]["length_excluded"] += 1
        return ok

    wc = DATASETS["wildchat"]
    wc_rows = load_dataset(wc["repository"], wc["config"], split=wc["split"],
                           revision=wc["revision"], streaming=True).shuffle(seed=args.seed, buffer_size=10000)
    wild, used = unique_candidates(
        wc_rows, "wildchat", wildchat_text, per_source, set(), stats["wildchat"],
        lambda row: wildchat_ok(row) and within_limit(wildchat_text(row), "wildchat"))
    bt = DATASETS["beavertails"]
    bt_rows = load_dataset(bt["repository"], bt["config"], split=bt["split"],
                           revision=bt["revision"], streaming=True).shuffle(seed=args.seed, buffer_size=10000)
    beaver, used = unique_candidates(
        bt_rows, "beavertails", lambda row: row["prompt"], per_source, used,
        stats["beavertails"], lambda row: within_limit(row.get("prompt"), "beavertails"))
    if len(wild) < per_source or len(beaver) < per_source:
        raise RuntimeError(f"Insufficient unique rows: WildChat={len(wild)}, BeaverTails={len(beaver)}")
    rng.shuffle(wild); rng.shuffle(beaver)
    train = wild[:train_n] + beaver[:train_n]
    selection = wild[train_n:] + beaver[train_n:]
    rng.shuffle(train); rng.shuffle(selection)

    evaluation = []
    jbb = DATASETS["jbb"]
    train_keys = {normalize(row["text"]) for row in train + selection}
    overlap_removed = 0
    for split, truth in (("harmful", "unsafe"), ("benign", "safe")):
        rows = load_dataset(jbb["repository"], jbb["config"], split=split,
                            revision=jbb["revision"])
        added = 0
        for row in rows:
            text = row["Goal"].strip()
            if normalize(text) in train_keys:
                overlap_removed += 1
                continue
            evaluation.append({"example_id": f"jbb_{split}_{int(row['Index']):03d}",
                               "source": f"jbb_{split}", "text": text,
                               "ground_truth": truth})
            added += 1
            if added == eval_n:
                break
        if added != eval_n:
            raise RuntimeError(f"JBB {split}: expected {eval_n}, got {added}")

    for role, rows in (("stitch_train", train), ("model_selection", selection)):
        for index, row in enumerate(rows):
            row["example_id"] = f"{role}_{index:04d}"
    groups = {"stitch_train": train, "model_selection": selection, "evaluation": evaluation}
    paths = {}
    for name, rows in groups.items():
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        paths[name] = {"filename": path.name, "count": len(rows), "sha256": sha256_file(path)}

    audit = token_audit(tokenizer, groups)
    manifest = {
        "schema_version": 1, "seed": args.seed, "tiny": args.tiny,
        "normalization": "Unicode text unchanged; strip; lowercase; collapse whitespace",
        "direct_matching_length_policy": {"max_guard_template_tokens": args.max_direct_tokens,
                                           "action": "exclude candidate; never truncate"},
        "datasets": DATASETS, "outputs": paths, "dedup": stats,
        "evaluation_overlap_with_train_or_selection": overlap_removed,
        "tokenization": {"tokenizer": GUARD, "revision": REVISIONS[GUARD],
                         "template_applied_once": True, "statistics": audit},
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
