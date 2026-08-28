"""Full Base -> Guard, cut=18 scientific baseline.

Defaults are the planned 800/200/200 run. Use --tiny only for plumbing checks.
Every reusable artifact is validated against immutable model/data metadata.
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from stitching_core import (
    BASE, CUT, GUARD, HIDDEN_SIZE, REVISIONS, freeze, parse_guard_label,
    prefix_hidden_states, stitched_cached_greedy, tensor_hash, tokenize_guard_prompt,
)

SEED = 20260827


def json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def expected_metadata(manifest, rows, model):
    return {
        "schema_version": 1, "model": model, "revision": REVISIONS[model],
        "cut": CUT, "manifest_sha256": file_hash(manifest),
        "examples_sha256": json_hash([(r["example_id"], r["text"]) for r in rows]),
        "storage_dtype": "bfloat16", "count": len(rows), "hidden_size": HIDDEN_SIZE,
    }


def validate_metadata(actual, expected, artifact):
    differences = {key: (actual.get(key), value) for key, value in expected.items()
                   if actual.get(key) != value}
    if differences:
        raise RuntimeError(f"Stale/incompatible artifact {artifact}: {differences}")


def resolve_output(path, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite or use --resume")
    path.parent.mkdir(parents=True, exist_ok=True)


def load_model(name, device):
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=REVISIONS[name], dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    freeze(model)
    return model


def tokenize_rows(tokenizer, rows):
    tokenized = []
    for row in rows:
        batch = tokenize_guard_prompt(tokenizer, row["text"])
        tokenized.append({"input_ids": batch.input_ids.cpu(),
                          "attention_mask": batch.attention_mask.cpu()})
    return tokenized


def extract_cache(name, rows, tokenized, path, manifest, device, resume, overwrite):
    expected = expected_metadata(manifest, rows, name)
    expected["input_ids_sha256"] = json_hash([tensor_hash(x["input_ids"]) for x in tokenized])
    if path.exists() and resume:
        cached = torch.load(path, map_location="cpu", weights_only=True)
        validate_metadata(cached["metadata"], expected, path)
        if len(cached["states"]) != len(rows):
            raise RuntimeError(f"Corrupt cache {path}: state count mismatch")
        return cached["states"], 0.0, True
    resolve_output(path, overwrite)
    started = time.perf_counter()
    model = load_model(name, device)
    model.model.layers = nn.ModuleList(list(model.model.layers[:CUT]))
    if hasattr(model, "lm_head"):
        del model.lm_head
    states = []
    with torch.inference_mode():
        for index, item in enumerate(tokenized):
            ids = item["input_ids"].to(device)
            mask = item["attention_mask"].to(device)
            state = prefix_hidden_states(model, ids, mask, CUT)
            if not torch.isfinite(state).all():
                raise FloatingPointError(f"Non-finite activation: {name}, example {index}")
            states.append(state.squeeze(0).to(torch.bfloat16).cpu())
            print(f"extract model={name} example={index + 1}/{len(rows)}", flush=True)
    metadata = dict(expected)
    metadata["shapes"] = [list(x.shape) for x in states]
    torch.save({"metadata": metadata, "states": states}, path)
    del model
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    return states, time.perf_counter() - started, False


def example_loss(adapter, source, target, indices):
    losses = []
    for index in indices:
        prediction = adapter(source[index].float())
        losses.append((prediction - target[index].float()).square().mean(-1).mean())
    return torch.stack(losses).mean()


def diagnostics(adapter, source, target):
    all_token, last_token = [], []
    with torch.no_grad():
        for left, right in zip(source, target):
            error = (adapter(left.float()) - right.float()).square().mean(-1)
            all_token.append(error.mean().item()); last_token.append(error[-1].item())
    return {"all_token_example_balanced_mse": statistics.fmean(all_token),
            "last_nonpadding_token_mse": statistics.fmean(last_token)}


def train_adapter(train_source, train_target, selection_source, selection_target,
                  max_epochs, patience, seed):
    torch.manual_seed(seed)
    adapter = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True)
    with torch.no_grad():
        adapter.weight.copy_(torch.eye(HIDDEN_SIZE)); adapter.bias.zero_()
    initial = {"train": diagnostics(adapter, train_source, train_target),
               "selection": diagnostics(adapter, selection_source, selection_target)}
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=0.0)
    best_value = initial["selection"]["all_token_example_balanced_mse"]
    best_epoch, bad_epochs = 0, 0
    best = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
    history = []
    order = list(range(len(train_source)))
    for epoch in range(1, max_epochs + 1):
        random.Random(seed + epoch).shuffle(order)
        adapter.train()
        # Small example batches keep variable sequence lengths naturally balanced.
        train_values = []
        for start in range(0, len(order), 4):
            loss = example_loss(adapter, train_source, train_target, order[start:start + 4])
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            train_values.append(loss.item())
        adapter.eval()
        selection = diagnostics(adapter, selection_source, selection_target)
        value = selection["all_token_example_balanced_mse"]
        history.append({"epoch": epoch, "train_batch_mean_mse": statistics.fmean(train_values),
                        **selection})
        print(f"epoch={epoch} selection_mse={value:.9g}", flush=True)
        if value < best_value:
            best_value, best_epoch, bad_epochs = value, epoch, 0
            best = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    adapter.load_state_dict(best); adapter.eval()
    final = {"train": diagnostics(adapter, train_source, train_target),
             "selection": diagnostics(adapter, selection_source, selection_target)}
    return adapter, {"optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 0.0,
                     "max_epochs": max_epochs, "early_stopping_patience": patience,
                     "best_epoch": best_epoch, "initial": initial, "best": final,
                     "history": history}


def native_generate(model, tokenizer, rows, tokenized, device, max_new_tokens):
    predictions = []
    started = time.perf_counter()
    for index, (row, item) in enumerate(zip(rows, tokenized)):
        input_ids = item["input_ids"].to(device)
        attention_mask = item["attention_mask"].to(device)
        with torch.inference_mode():
            output = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                    do_sample=False, max_new_tokens=max_new_tokens, use_cache=True)
        tokens = output[:, input_ids.shape[1]:]
        text = tokenizer.decode(tokens[0], skip_special_tokens=True)
        predictions.append({"raw_output": text, "label": parse_guard_label(text)})
        print(f"native generation={index + 1}/{len(rows)} label={predictions[-1]['label']}", flush=True)
    return predictions, time.perf_counter() - started


def stitched_generate(base, guard, adapter, tokenizer, rows, tokenized, device, max_new_tokens):
    predictions = []
    adapter = adapter.to(device)
    started = time.perf_counter()
    for index, (row, item) in enumerate(zip(rows, tokenized)):
        input_ids = item["input_ids"].to(device)
        attention_mask = item["attention_mask"].to(device)
        tokens, _ = stitched_cached_greedy(base, guard, adapter, input_ids,
                                           attention_mask, max_new_tokens, CUT)
        text = tokenizer.decode(tokens[0], skip_special_tokens=True)
        predictions.append({"raw_output": text, "label": parse_guard_label(text)})
        print(f"stitched generation={index + 1}/{len(rows)} label={predictions[-1]['label']}", flush=True)
    return predictions, time.perf_counter() - started


def binary_prediction(label, unsafe_only=False):
    if label is None:
        return "unsafe"  # parse failure is explicitly never converted to Safe
    if label == "Unsafe" or (label == "Controversial" and not unsafe_only):
        return "unsafe"
    return "safe"


def metrics(rows, predictions):
    truth = [r["ground_truth"] for r in rows]
    predicted = [binary_prediction(p["label"]) for p in predictions]
    tp = sum(t == "unsafe" and p == "unsafe" for t, p in zip(truth, predicted))
    tn = sum(t == "safe" and p == "safe" for t, p in zip(truth, predicted))
    fp = sum(t == "safe" and p == "unsafe" for t, p in zip(truth, predicted))
    fn = sum(t == "unsafe" and p == "safe" for t, p in zip(truth, predicted))
    unsafe_recall = tp / (tp + fn) if tp + fn else math.nan
    safe_recall = tn / (tn + fp) if tn + fp else math.nan
    def f1(a, b, c): return 2 * a / (2 * a + b + c) if 2 * a + b + c else 0.0
    unsafe_only = [binary_prediction(p["label"], True) for p in predictions]
    return {
        "macro_f1": (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2,
        "balanced_accuracy": (unsafe_recall + safe_recall) / 2,
        "unsafe_recall": unsafe_recall, "safe_fpr": fp / (fp + tn) if fp + tn else math.nan,
        "controversial_rate": sum(p["label"] == "Controversial" for p in predictions) / len(rows),
        "parse_failure_rate": sum(p["label"] is None for p in predictions) / len(rows),
        "unsafe_only_recall": sum(t == "unsafe" and p == "unsafe" for t, p in zip(truth, unsafe_only)) /
                              sum(t == "unsafe" for t in truth),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def transitions(rows, native, stitched):
    labels = ["Safe", "Controversial", "Unsafe", None]
    result = {}
    for subset in ("all", "harmful", "benign"):
        counts = {str(a): {str(b): 0 for b in labels} for a in labels}
        for row, left, right in zip(rows, native, stitched):
            if subset != "all" and not row["source"].endswith(subset): continue
            counts[str(left["label"])][str(right["label"])] += 1
        result[subset] = counts
    return result


def bootstrap(rows, native, stitched, count, seed):
    rng = random.Random(seed)
    keys = ("macro_f1", "balanced_accuracy")
    samples = {f"native_{k}": [] for k in keys} | {f"stitched_{k}": [] for k in keys} | \
              {f"penalty_{k}": [] for k in keys}
    for _ in range(count):
        indices = [rng.randrange(len(rows)) for _ in rows]
        selected_rows = [rows[i] for i in indices]
        n = metrics(selected_rows, [native[i] for i in indices])
        s = metrics(selected_rows, [stitched[i] for i in indices])
        for key in keys:
            samples[f"native_{key}"].append(n[key]); samples[f"stitched_{key}"].append(s[key])
            samples[f"penalty_{key}"].append(n[key] - s[key])
    return {key: {"low": float(np.nanpercentile(values, 2.5)),
                  "high": float(np.nanpercentile(values, 97.5))}
            for key, values in samples.items()}


def git_commit():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                        stderr=subprocess.DEVNULL).strip()
    except Exception: return None


def environment():
    return {"hostname": platform.node(), "python": platform.python_version(),
            "torch": torch.__version__, "transformers": __import__("transformers").__version__,
            "cuda_available": torch.cuda.is_available(), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None}


def prediction_stage(path, expected, resume, overwrite, generate):
    if path.exists() and resume:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_metadata(payload["metadata"], expected, path)
        return payload["predictions"], 0.0, True
    resolve_output(path, overwrite)
    predictions, seconds = generate()
    path.write_text(json.dumps({"metadata": expected, "predictions": predictions},
                               indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return predictions, seconds, False


def main():
    parser = argparse.ArgumentParser()
    data_default = Path(os.environ.get("STITCH_DATA", "data/generated"))
    parser.add_argument("--data-dir", type=Path, default=data_default)
    parser.add_argument("--manifest", type=Path, default=Path("data/baseline_manifest.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("STITCH_ARTIFACTS", "artifacts")) / "activation_cache")
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("STITCH_ARTIFACTS", "artifacts")) / "direct_matching_cut18.pt")
    parser.add_argument("--results-dir", type=Path, default=Path(os.environ.get("STITCH_RESULTS", "results")))
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite: raise SystemExit("Choose --resume or --overwrite, not both")
    started = time.perf_counter(); device = torch.device(args.device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    manifest_data = json.loads(args.manifest.read_text(encoding="utf-8"))
    train = load_jsonl(args.data_dir / "stitch_train.jsonl")
    selection = load_jsonl(args.data_dir / "model_selection.jsonl")
    evaluation = load_jsonl(args.data_dir / "evaluation.jsonl")
    completed_summary = args.results_dir / "cut18_summary.json"
    completed_csv = args.results_dir / "cut18_predictions.csv"
    if args.resume and completed_summary.exists():
        existing = json.loads(completed_summary.read_text(encoding="utf-8"))
        expected_complete = {"manifest_sha256": file_hash(args.manifest), "models": REVISIONS,
                             "cut": CUT, "direction": "Base->Guard",
                             "counts": {"train": len(train), "selection": len(selection),
                                        "evaluation": len(evaluation)}}
        validate_metadata(existing, expected_complete, completed_summary)
        if not completed_csv.exists() or not args.checkpoint.exists():
            raise RuntimeError("Complete summary exists but CSV/checkpoint is missing")
        print(f"COMPLETE RESULT REUSED: {completed_summary}")
        return
    tokenizer = AutoTokenizer.from_pretrained(GUARD, revision=REVISIONS[GUARD])
    train_tok = tokenize_rows(tokenizer, train)
    selection_tok = tokenize_rows(tokenizer, selection)
    evaluation_tok = tokenize_rows(tokenizer, evaluation)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    timings = {}
    caches = {}
    for role, rows, toks in (("train", train, train_tok), ("selection", selection, selection_tok)):
        for short, model in (("base", BASE), ("guard", GUARD)):
            path = args.cache_dir / f"{role}_{short}_cut18.pt"
            caches[(role, short)], seconds, reused = extract_cache(
                model, rows, toks, path, args.manifest, device, args.resume, args.overwrite)
            timings[f"extract_{role}_{short}_seconds"] = seconds
            print(f"cache={path} reused={reused}")
    checkpoint_meta = {"cut": CUT, "direction": "Base->Guard", "models": REVISIONS,
                       "manifest_sha256": file_hash(args.manifest), "seed": args.seed}
    if args.checkpoint.exists() and args.resume:
        old = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        validate_metadata(old["metadata"], checkpoint_meta, args.checkpoint)
        adapter = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True)
        adapter.load_state_dict(old["adapter_state_dict"]); dm = old["dm_metrics"]
        print(f"checkpoint={args.checkpoint} reused=True")
    else:
        adapter, dm = train_adapter(caches[("train", "base")], caches[("train", "guard")],
                                    caches[("selection", "base")], caches[("selection", "guard")],
                                    args.max_epochs, args.patience, args.seed)
        resolve_output(args.checkpoint, args.overwrite)
        torch.save({"metadata": checkpoint_meta, "adapter_state_dict": adapter.state_dict(),
                    "dm_metrics": dm}, args.checkpoint)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    prediction_meta = {"manifest_sha256": file_hash(args.manifest),
                       "evaluation_sha256": json_hash(evaluation),
                       "input_ids_sha256": json_hash([tensor_hash(x["input_ids"])
                                                       for x in evaluation_tok]),
                       "max_new_tokens": args.max_new_tokens, "models": REVISIONS, "cut": CUT}
    native_path = args.results_dir / "native_predictions.json"
    if native_path.exists() and args.resume:
        native, timings["native_generation_seconds"], reused = prediction_stage(
            native_path, prediction_meta, True, False, lambda: ([], 0.0))
    else:
        guard = load_model(GUARD, device)
        native, timings["native_generation_seconds"], reused = prediction_stage(
            native_path, prediction_meta, False, args.overwrite,
            lambda: native_generate(guard, tokenizer, evaluation, evaluation_tok,
                                    device, args.max_new_tokens))
        del guard; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
    print(f"native_stage reused={reused}")
    stitched_path = args.results_dir / "stitched_predictions.json"
    if stitched_path.exists() and args.resume:
        stitched, timings["stitched_generation_seconds"], reused = prediction_stage(
            stitched_path, prediction_meta, True, False, lambda: ([], 0.0))
    else:
        base = load_model(BASE, device)
        base.model.layers = nn.ModuleList(list(base.model.layers[:CUT])); del base.lm_head
        guard = load_model(GUARD, device)
        guard.model.layers = nn.ModuleList([nn.Identity() for _ in range(CUT)] + list(guard.model.layers[CUT:]))
        stitched, timings["stitched_generation_seconds"], reused = prediction_stage(
            stitched_path, prediction_meta, False, args.overwrite,
            lambda: stitched_generate(base, guard, adapter, tokenizer, evaluation,
                                       evaluation_tok, device,
                                       args.max_new_tokens))
        del base, guard; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
    print(f"stitched_stage reused={reused}")
    n_metrics, s_metrics = metrics(evaluation, native), metrics(evaluation, stitched)
    penalties = {"macro_f1": n_metrics["macro_f1"] - s_metrics["macro_f1"],
                 "balanced_accuracy": n_metrics["balanced_accuracy"] - s_metrics["balanced_accuracy"],
                 "unsafe_recall": n_metrics["unsafe_recall"] - s_metrics["unsafe_recall"],
                 "safe_fpr_delta": s_metrics["safe_fpr"] - n_metrics["safe_fpr"]}
    intervals = bootstrap(evaluation, native, stitched, args.bootstrap, args.seed)
    predictions_path = args.results_dir / "cut18_predictions.csv"
    summary_path = args.results_dir / "cut18_summary.json"
    for path in (predictions_path, summary_path):
        if path.exists() and not (args.overwrite or args.resume): resolve_output(path, False)
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["example_id", "source", "ground_truth",
            "native_raw_output", "native_label", "stitched_raw_output", "stitched_label"])
        writer.writeheader()
        for row, left, right in zip(evaluation, native, stitched):
            writer.writerow({"example_id": row["example_id"], "source": row["source"],
                "ground_truth": row["ground_truth"], "native_raw_output": left["raw_output"],
                "native_label": left["label"], "stitched_raw_output": right["raw_output"],
                "stitched_label": right["label"]})
    summary = {"git_commit": git_commit(), "models": REVISIONS,
        "dataset_revisions": {k: v["revision"] for k, v in manifest_data["datasets"].items()},
        "manifest_sha256": file_hash(args.manifest), "seed": args.seed, "cut": CUT,
        "direction": "Base->Guard", "counts": {"train": len(train), "selection": len(selection),
        "evaluation": len(evaluation)}, "adapter": dm, "native_metrics": n_metrics,
        "stitched_metrics": s_metrics, "penalties": penalties, "bootstrap_95_ci": intervals,
        "transitions": transitions(evaluation, native, stitched), "runtime": timings,
        "total_runtime_seconds": time.perf_counter() - started, "environment": environment(),
        "parse_failure_policy": "None remains None in CSV; conservatively unsafe for binary metrics"}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"FULL PIPELINE: PASS\npredictions={predictions_path}\nsummary={summary_path}")


if __name__ == "__main__":
    main()
