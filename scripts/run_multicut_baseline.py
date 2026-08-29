"""Shared-extraction Base -> Guard baseline for Qwen3 cuts 0, 9, 18, 27, 35.

This is deliberately plain research code. The completed cut18 runner remains
untouched as a reference implementation; this runner reuses its established
loss, generation, parsing, transition, and metric definitions. Its local paired
bootstrap differs only by representing a missing-class recall as NaN for tiny
resamples instead of raising an exception.
"""

import argparse
import csv
import gc
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from transformers.utils.hub import cached_file

from run_cut18_baseline import (
    SEED,
    binary_prediction,
    environment,
    file_hash,
    git_commit,
    json_hash,
    load_jsonl,
    load_model,
    native_generate,
    resolve_output,
    tokenize_rows,
    train_adapter,
    transitions,
    validate_metadata,
)
from stitching_core import (
    BASE,
    GUARD,
    HIDDEN_SIZE,
    NUM_LAYERS,
    REVISIONS,
    SUPPORTED_CUTS,
    boundary_hidden_states,
    cut_semantics,
    parse_guard_label,
    stitched_cached_greedy,
    tensor_hash,
    validate_cuts,
)
from queue_status import fail as fail_queue_status
from queue_status import initialize as initialize_queue_status
from queue_status import update as update_queue_status

FULL_MANIFEST_SHA256 = "614209e8f746787d139a4d1d67d6475c30fae365ac21b78531ff704af6d995e0"
ROLE_FILES = {
    "train": "stitch_train.jsonl",
    "selection": "model_selection.jsonl",
    "evaluation": "evaluation.jsonl",
}


def metrics(rows, predictions):
    """Legacy metrics with a defined result for a resample missing one class.

    This is numerically identical to the cut18 implementation whenever both
    classes are present. Tiny paired-bootstrap samples can contain no unsafe
    rows, so their class-specific recall is represented as NaN instead of
    raising ZeroDivisionError; nanpercentile then uses the valid resamples.
    """
    truth = [row["ground_truth"] for row in rows]
    predicted = [binary_prediction(prediction["label"]) for prediction in predictions]
    tp = sum(t == "unsafe" and p == "unsafe" for t, p in zip(truth, predicted))
    tn = sum(t == "safe" and p == "safe" for t, p in zip(truth, predicted))
    fp = sum(t == "safe" and p == "unsafe" for t, p in zip(truth, predicted))
    fn = sum(t == "unsafe" and p == "safe" for t, p in zip(truth, predicted))
    unsafe_total = tp + fn
    safe_total = tn + fp
    unsafe_recall = tp / unsafe_total if unsafe_total else math.nan
    safe_recall = tn / safe_total if safe_total else math.nan

    def f1(a, b, c):
        return 2 * a / (2 * a + b + c) if 2 * a + b + c else 0.0

    unsafe_only = [
        binary_prediction(prediction["label"], True) for prediction in predictions
    ]
    return {
        "macro_f1": (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2,
        "balanced_accuracy": (unsafe_recall + safe_recall) / 2,
        "unsafe_recall": unsafe_recall,
        "safe_fpr": fp / safe_total if safe_total else math.nan,
        "controversial_rate": sum(
            prediction["label"] == "Controversial" for prediction in predictions
        )
        / len(rows),
        "parse_failure_rate": sum(
            prediction["label"] is None for prediction in predictions
        )
        / len(rows),
        "unsafe_only_recall": (
            sum(
                t == "unsafe" and p == "unsafe"
                for t, p in zip(truth, unsafe_only)
            )
            / unsafe_total
            if unsafe_total
            else math.nan
        ),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def bootstrap(rows, native, stitched, count, seed):
    """Paired bootstrap using identical sampled indices for both systems."""
    rng = random.Random(seed)
    keys = ("macro_f1", "balanced_accuracy")
    samples = (
        {f"native_{key}": [] for key in keys}
        | {f"stitched_{key}": [] for key in keys}
        | {f"penalty_{key}": [] for key in keys}
    )
    for _ in range(count):
        indices = [rng.randrange(len(rows)) for _ in rows]
        selected_rows = [rows[index] for index in indices]
        native_metrics = metrics(
            selected_rows, [native[index] for index in indices]
        )
        stitched_metrics = metrics(
            selected_rows, [stitched[index] for index in indices]
        )
        for key in keys:
            samples[f"native_{key}"].append(native_metrics[key])
            samples[f"stitched_{key}"].append(stitched_metrics[key])
            samples[f"penalty_{key}"].append(
                native_metrics[key] - stitched_metrics[key]
            )
    return {
        key: {
            "low": float(np.nanpercentile(values, 2.5)),
            "high": float(np.nanpercentile(values, 97.5)),
        }
        for key, values in samples.items()
    }


def parse_cuts(value):
    try:
        cuts = validate_cuts(part.strip() for part in value.split(",") if part.strip())
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    unsupported = [cut for cut in cuts if cut not in SUPPORTED_CUTS]
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"This baseline supports cuts {SUPPORTED_CUTS}; unsupported={unsupported}"
        )
    return cuts


def validate_dataset(manifest_path, data_dir, expected_full_hash=None):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = file_hash(manifest_path)
    if expected_full_hash and manifest_hash != expected_full_hash:
        raise RuntimeError(
            f"Manifest hash mismatch: expected {expected_full_hash}, got {manifest_hash}"
        )
    rows = {}
    manifest_roles = {
        "train": "stitch_train",
        "selection": "model_selection",
        "evaluation": "evaluation",
    }
    for role, filename in ROLE_FILES.items():
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        record = manifest["outputs"][manifest_roles[role]]
        actual_hash = file_hash(path)
        if actual_hash != record["sha256"]:
            raise RuntimeError(
                f"Dataset hash mismatch for {path}: expected {record['sha256']}, got {actual_hash}"
            )
        rows[role] = load_jsonl(path)
        if len(rows[role]) != record["count"]:
            raise RuntimeError(
                f"Dataset count mismatch for {path}: expected {record['count']}, "
                f"got {len(rows[role])}"
            )
    return manifest, manifest_hash, rows


def input_ids_hash(tokenized):
    return json_hash([tensor_hash(item["input_ids"]) for item in tokenized])


def cache_metadata(manifest_hash, rows, tokenized, model_name, cuts):
    return {
        "schema_version": 2,
        "model": model_name,
        "revision": REVISIONS[model_name],
        "manifest_sha256": manifest_hash,
        "examples_sha256": json_hash(
            [(row["example_id"], row["text"]) for row in rows]
        ),
        "input_ids_sha256": input_ids_hash(tokenized),
        "cuts": list(cuts),
        "cut_semantics": {str(cut): cut_semantics(cut) for cut in cuts},
        "storage_dtype": "bfloat16",
        "count": len(rows),
        "hidden_size": HIDDEN_SIZE,
        "token_lengths": [int(item["attention_mask"].sum().item()) for item in tokenized],
    }


def validate_cache_payload(payload, expected, path):
    validate_metadata(payload["metadata"], expected, path)
    states_by_cut = payload.get("states_by_cut", {})
    for cut in expected["cuts"]:
        states = states_by_cut.get(str(cut))
        if states is None or len(states) != expected["count"]:
            raise RuntimeError(f"Corrupt cache {path}: missing/count mismatch for cut {cut}")
        for index, (state, length) in enumerate(zip(states, expected["token_lengths"])):
            if tuple(state.shape) != (length, HIDDEN_SIZE):
                raise RuntimeError(
                    f"Corrupt cache {path}: cut={cut} example={index} "
                    f"shape={tuple(state.shape)} expected={(length, HIDDEN_SIZE)}"
                )
            if state.dtype != torch.bfloat16 or not torch.isfinite(state).all():
                raise RuntimeError(
                    f"Corrupt cache {path}: cut={cut} example={index} dtype/finiteness"
                )
    return states_by_cut


def extract_shared_cache(
    model_name, rows, tokenized, cuts, path, manifest_hash, device, resume, overwrite
):
    expected = cache_metadata(manifest_hash, rows, tokenized, model_name, cuts)
    if path.exists() and resume:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        states = validate_cache_payload(payload, expected, path)
        return states, float(payload.get("stage_seconds", 0.0)), True
    resolve_output(path, overwrite)
    started = time.perf_counter()
    model = load_model(model_name, device)
    model.model.layers = nn.ModuleList(list(model.model.layers[: max(cuts)]))
    if hasattr(model, "lm_head"):
        del model.lm_head
    states_by_cut = {str(cut): [] for cut in cuts}
    with torch.inference_mode():
        for index, item in enumerate(tokenized):
            boundaries = boundary_hidden_states(
                model,
                item["input_ids"].to(device),
                item["attention_mask"].to(device),
                cuts,
            )
            for cut in cuts:
                state = boundaries[cut].squeeze(0)
                if not torch.isfinite(state).all():
                    raise FloatingPointError(
                        f"Non-finite activation: {model_name}, cut={cut}, example={index}"
                    )
                states_by_cut[str(cut)].append(state.to(torch.bfloat16).cpu())
            print(
                f"shared extract model={model_name} example={index + 1}/{len(rows)} "
                f"cuts={','.join(map(str, cuts))}",
                flush=True,
            )
    metadata = dict(expected)
    metadata["shapes"] = {
        str(cut): [list(state.shape) for state in states_by_cut[str(cut)]]
        for cut in cuts
    }
    stage_seconds = time.perf_counter() - started
    torch.save(
        {
            "metadata": metadata,
            "states_by_cut": states_by_cut,
            "stage_seconds": stage_seconds,
        },
        path,
    )
    validate_cache_payload(
        torch.load(path, map_location="cpu", weights_only=True), expected, path
    )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return states_by_cut, stage_seconds, False


def checkpoint_metadata(
    cut, manifest_hash, token_hashes, cache_hashes, seed, max_epochs, patience
):
    return {
        "schema_version": 2,
        "cut": cut,
        "cut_semantics": cut_semantics(cut),
        "direction": "Base->Guard",
        "models": REVISIONS,
        "manifest_sha256": manifest_hash,
        "input_ids_sha256": token_hashes,
        "activation_cache_sha256": cache_hashes,
        "seed": seed,
        "training_policy": {
            "adapter": f"Linear({HIDDEN_SIZE},{HIDDEN_SIZE},bias=True)",
            "initialization": "W=I,b=0",
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "max_epochs": max_epochs,
            "patience": patience,
            "selection": "model_selection Direct Matching loss only",
            "loss": "mean hidden MSE -> mean non-padding tokens -> mean examples",
        },
    }


def load_or_train_adapter(
    cut,
    caches,
    checkpoint,
    metadata,
    max_epochs,
    patience,
    seed,
    resume,
    overwrite,
):
    if checkpoint.exists() and resume:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        validate_metadata(payload["metadata"], metadata, checkpoint)
        adapter = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True)
        adapter.load_state_dict(payload["adapter_state_dict"])
        adapter.eval()
        dm_metrics = dict(payload["dm_metrics"])
        with torch.no_grad():
            dm_metrics["weight_identity_frobenius_norm"] = float(
                torch.linalg.vector_norm(adapter.weight - torch.eye(HIDDEN_SIZE)).item()
            )
            dm_metrics["bias_l2_norm"] = float(
                torch.linalg.vector_norm(adapter.bias).item()
            )
        return adapter, dm_metrics, float(payload.get("training_seconds", 0.0)), True
    resolve_output(checkpoint, overwrite)
    started = time.perf_counter()
    adapter, dm_metrics = train_adapter(
        caches[("train", "base")][str(cut)],
        caches[("train", "guard")][str(cut)],
        caches[("selection", "base")][str(cut)],
        caches[("selection", "guard")][str(cut)],
        max_epochs,
        patience,
        seed,
    )
    dm_metrics = dict(dm_metrics)
    with torch.no_grad():
        dm_metrics["weight_identity_frobenius_norm"] = float(
            torch.linalg.vector_norm(adapter.weight - torch.eye(HIDDEN_SIZE)).item()
        )
    dm_metrics["bias_l2_norm"] = float(torch.linalg.vector_norm(adapter.bias).item())
    training_seconds = time.perf_counter() - started
    torch.save(
        {
            "metadata": metadata,
            "adapter_state_dict": adapter.state_dict(),
            "dm_metrics": dm_metrics,
            "training_seconds": training_seconds,
        },
        checkpoint,
    )
    return adapter, dm_metrics, training_seconds, False


def prediction_stage_v2(path, expected, resume, overwrite, generate):
    if path.exists() and resume:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_metadata(payload["metadata"], expected, path)
        return payload["predictions"], float(payload.get("stage_seconds", 0.0)), True
    resolve_output(path, overwrite)
    predictions, seconds = generate()
    path.write_text(
        json.dumps(
            {
                "metadata": expected,
                "predictions": predictions,
                "stage_seconds": seconds,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    return predictions, seconds, False


def set_parameter(module, name, value):
    parts = name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    leaf = parts[-1]
    if leaf not in parent._parameters:
        raise KeyError(f"Checkpoint tensor does not map to a parameter: {name}")
    parent._parameters[leaf] = nn.Parameter(value, requires_grad=False)


class LazySafetensorEmbedding(nn.Module):
    """Exact CPU embedding lookup without retaining the full donor table."""

    def __init__(self, shard, tensor_name, hidden_size):
        super().__init__()
        self.shard = str(shard)
        self.tensor_name = tensor_name
        self.hidden_size = hidden_size

    def forward(self, input_ids):
        flat = input_ids.detach().cpu().reshape(-1)
        unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        with safe_open(self.shard, framework="pt", device="cpu") as handle:
            source = handle.get_slice(self.tensor_name)
            rows = torch.empty((len(unique), self.hidden_size), dtype=torch.bfloat16)
            for offset, index in enumerate(unique):
                rows[offset].copy_(source[int(index)])
        return rows[inverse].reshape(*input_ids.shape, self.hidden_size).to(input_ids.device)


class LazySafetensorLMHead(nn.Module):
    """Exact chunked CPU LM head backed by a tied embedding safetensor."""

    def __init__(self, shard, tensor_name, vocab_size, chunk_size=4096):
        super().__init__()
        self.shard = str(shard)
        self.tensor_name = tensor_name
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size

    def forward(self, hidden_states):
        chunks = []
        with safe_open(self.shard, framework="pt", device="cpu") as handle:
            source = handle.get_slice(self.tensor_name)
            for start in range(0, self.vocab_size, self.chunk_size):
                weight = source[start:min(start + self.chunk_size, self.vocab_size)]
                chunks.append(F.linear(hidden_states, weight))
        return torch.cat(chunks, dim=-1)


def ensure_cpu_embedding_sidecar(model_name):
    """Create a revision-scoped, exact embedding-only mmap for CPU smoke tests."""
    index_path = Path(
        cached_file(
            model_name,
            "model.safetensors.index.json",
            revision=REVISIONS[model_name],
        )
    )
    sidecar = index_path.parent / "cpu_donor_embedding.safetensors"
    if sidecar.exists():
        return sidecar
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    name = "model.embed_tokens.weight"
    with safe_open(
        index_path.parent / weight_map[name], framework="pt", device="cpu"
    ) as handle:
        weight = handle.get_tensor(name).clone()
    save_file({"weight": weight}, sidecar)
    return sidecar


def ensure_cpu_layer_sidecars(model_name, count):
    """Split exact donor layers into small revision-scoped CPU mmap files."""
    if count == 0:
        return []
    index_path = Path(
        cached_file(
            model_name,
            "model.safetensors.index.json",
            revision=REVISIONS[model_name],
        )
    )
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    root = index_path.parent / "cpu_donor_layers"
    root.mkdir(exist_ok=True)
    result = []
    for layer_index in range(count):
        path = root / f"layer_{layer_index:02d}.safetensors"
        expected_metadata = {
            "model": model_name,
            "revision": REVISIONS[model_name],
            "layer": str(layer_index),
        }
        if path.exists():
            with safe_open(path, framework="pt", device="cpu") as handle:
                if handle.metadata() != expected_metadata:
                    raise RuntimeError(f"Invalid CPU layer sidecar metadata: {path}")
            result.append(path)
            continue
        prefix = f"model.layers.{layer_index}."
        names = [name for name in weight_map if name.startswith(prefix)]
        if not names:
            raise RuntimeError(f"No checkpoint tensors found for {prefix}")
        tensors = {}
        for shard in sorted({weight_map[name] for name in names}):
            with safe_open(
                index_path.parent / shard, framework="pt", device="cpu"
            ) as handle:
                for name in names:
                    if weight_map[name] == shard:
                        tensors[name] = handle.get_tensor(name).clone()
        save_file(tensors, path, metadata=expected_metadata)
        result.append(path)
    return result


def load_cpu_fragment(model_name, cut, donor):
    """Materialize only a donor prefix or receiver suffix from safetensors.

    Windows test hosts may not have enough virtual memory to load two complete
    4B models before trimming. The CUDA path keeps the previously verified full
    load-then-trim behavior; this exact-safetensors CPU path changes only memory
    materialization, not forward semantics.
    """
    config = AutoConfig.from_pretrained(model_name, revision=REVISIONS[model_name])
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    index_path = Path(
        cached_file(
            model_name,
            "model.safetensors.index.json",
            revision=REVISIONS[model_name],
        )
    )
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]

    if donor:
        wanted_prefixes = [
            f"model.layers.{index}." for index in range(cut)
        ]
    else:
        wanted_prefixes = ["model.norm."] + [
            f"model.layers.{index}." for index in range(cut, NUM_LAYERS)
        ]
    selected = {
        name: shard
        for name, shard in weight_map.items()
        if any(name.startswith(prefix) for prefix in wanted_prefixes)
    }
    if donor:
        for sidecar in ensure_cpu_layer_sidecars(model_name, cut):
            with safe_open(sidecar, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    set_parameter(model, name, handle.get_tensor(name))
    else:
        by_shard = {}
        for name, shard in selected.items():
            by_shard.setdefault(shard, []).append(name)
        for shard, names in by_shard.items():
            with safe_open(index_path.parent / shard, framework="pt", device="cpu") as handle:
                for name in names:
                    set_parameter(model, name, handle.get_tensor(name))

    model.model.rotary_emb = Qwen3RotaryEmbedding(config, device="cpu")
    if donor:
        model.model.embed_tokens = LazySafetensorEmbedding(
            ensure_cpu_embedding_sidecar(model_name),
            "weight",
            HIDDEN_SIZE,
        )
        model.model.layers = nn.ModuleList(list(model.model.layers[:cut]))
        del model.model.norm
        del model.lm_head
    else:
        head_sidecar = ensure_cpu_embedding_sidecar(model_name)
        model.lm_head = LazySafetensorLMHead(
            head_sidecar, "weight", config.vocab_size
        )
        model.model.layers = nn.ModuleList(
            [nn.Identity() for _ in range(cut)] + list(model.model.layers[cut:])
        )
        del model.model.embed_tokens
    meta_parameters = [name for name, value in model.named_parameters() if value.is_meta]
    meta_buffers = [name for name, value in model.named_buffers() if value.is_meta]
    if meta_parameters or meta_buffers:
        raise RuntimeError(
            f"Incomplete CPU fragment {model_name} cut={cut}: "
            f"meta_parameters={meta_parameters} meta_buffers={meta_buffers}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def stitched_generate_for_cut(
    cut, adapter, tokenizer, rows, tokenized, device, max_new_tokens
):
    def donor_prefix():
        if device.type == "cpu":
            return load_cpu_fragment(BASE, cut, donor=True)
        model = load_model(BASE, device)
        model.model.layers = nn.ModuleList(list(model.model.layers[:cut]))
        del model.lm_head
        return model

    def receiver_suffix():
        if device.type == "cpu":
            return load_cpu_fragment(GUARD, cut, donor=False)
        model = load_model(GUARD, device)
        model.model.layers = nn.ModuleList(
            [nn.Identity() for _ in range(cut)] + list(model.model.layers[cut:])
        )
        # Receiver generation starts from an adapted boundary, never embeddings.
        del model.model.embed_tokens
        return model

    # Load and trim the smaller retained side first to avoid a two-full-model
    # memory peak on 32 GiB local machines and 24 GiB GPUs.
    if device.type == "cpu":
        # Build the small exact donor-embedding sidecar before materializing the
        # receiver, then load suffix before prefix to stay below Windows' low
        # virtual-memory ceiling.
        ensure_cpu_embedding_sidecar(BASE)
        ensure_cpu_embedding_sidecar(GUARD)
        ensure_cpu_layer_sidecars(BASE, cut)
        guard = receiver_suffix()
        gc.collect()
        base = donor_prefix()
    elif cut <= NUM_LAYERS // 2:
        base = donor_prefix()
        gc.collect()
        guard = receiver_suffix()
    else:
        guard = receiver_suffix()
        gc.collect()
        base = donor_prefix()
    adapter = adapter.to(device).eval()
    predictions = []
    started = time.perf_counter()
    try:
        for index, (row, item) in enumerate(zip(rows, tokenized)):
            tokens, _ = stitched_cached_greedy(
                base,
                guard,
                adapter,
                item["input_ids"].to(device),
                item["attention_mask"].to(device),
                max_new_tokens,
                cut,
            )
            text = tokenizer.decode(tokens[0], skip_special_tokens=True)
            predictions.append({"raw_output": text, "label": parse_guard_label(text)})
            print(
                f"stitched cut={cut} generation={index + 1}/{len(rows)} "
                f"label={predictions[-1]['label']}",
                flush=True,
            )
    finally:
        del base, guard
        adapter.to("cpu")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return predictions, time.perf_counter() - started


def penalties(native_metrics, stitched_metrics):
    return {
        "macro_f1": native_metrics["macro_f1"] - stitched_metrics["macro_f1"],
        "balanced_accuracy": (
            native_metrics["balanced_accuracy"] - stitched_metrics["balanced_accuracy"]
        ),
        "unsafe_recall": (
            native_metrics["unsafe_recall"] - stitched_metrics["unsafe_recall"]
        ),
        "safe_fpr_delta": stitched_metrics["safe_fpr"] - native_metrics["safe_fpr"],
    }


def ci_fields(intervals):
    mapping = {
        "native_macro_f1": "native_macro_f1",
        "stitched_macro_f1": "stitched_macro_f1",
        "penalty_macro_f1": "macro_f1_penalty",
        "native_balanced_accuracy": "native_bal_acc",
        "stitched_balanced_accuracy": "stitched_bal_acc",
        "penalty_balanced_accuracy": "bal_acc_penalty",
    }
    flattened = {}
    for source, target in mapping.items():
        flattened[f"{target}_ci_low"] = intervals[source]["low"]
        flattened[f"{target}_ci_high"] = intervals[source]["high"]
    return flattened


def aggregate_record(cut, dm, native_metrics, stitched_metrics, penalty, intervals):
    initial = dm["initial"]["selection"]
    best = dm["best"]["selection"]
    identity_mse = initial["all_token_example_balanced_mse"]
    best_mse = best["all_token_example_balanced_mse"]
    return {
        "cut": cut,
        "relative_depth": cut / NUM_LAYERS,
        "identity_selection_mse": identity_mse,
        "best_selection_mse": best_mse,
        "dm_improvement": (identity_mse - best_mse) / identity_mse,
        "best_epoch": dm["best_epoch"],
        "weight_identity_frobenius_norm": dm["weight_identity_frobenius_norm"],
        "bias_l2_norm": dm["bias_l2_norm"],
        "identity_selection_last_token_mse": initial["last_nonpadding_token_mse"],
        "trained_selection_last_token_mse": best["last_nonpadding_token_mse"],
        "native_macro_f1": native_metrics["macro_f1"],
        "stitched_macro_f1": stitched_metrics["macro_f1"],
        "macro_f1_penalty": penalty["macro_f1"],
        "native_bal_acc": native_metrics["balanced_accuracy"],
        "stitched_bal_acc": stitched_metrics["balanced_accuracy"],
        "bal_acc_penalty": penalty["balanced_accuracy"],
        "native_unsafe_recall": native_metrics["unsafe_recall"],
        "stitched_unsafe_recall": stitched_metrics["unsafe_recall"],
        "unsafe_recall_penalty": penalty["unsafe_recall"],
        "native_safe_fpr": native_metrics["safe_fpr"],
        "stitched_safe_fpr": stitched_metrics["safe_fpr"],
        "safe_fpr_delta": penalty["safe_fpr_delta"],
        "native_unsafe_only_recall": native_metrics["unsafe_only_recall"],
        "stitched_unsafe_only_recall": stitched_metrics["unsafe_only_recall"],
        "native_controversial_rate": native_metrics["controversial_rate"],
        "stitched_controversial_rate": stitched_metrics["controversial_rate"],
        "native_parse_failure_rate": native_metrics["parse_failure_rate"],
        "stitched_parse_failure_rate": stitched_metrics["parse_failure_rate"],
        **ci_fields(intervals),
    }


def write_predictions_csv(path, rows, native, stitched, overwrite):
    resolve_output(path, overwrite)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "example_id",
                "source",
                "ground_truth",
                "native_raw_output",
                "native_label",
                "stitched_raw_output",
                "stitched_label",
            ],
        )
        writer.writeheader()
        for row, left, right in zip(rows, native, stitched):
            writer.writerow(
                {
                    "example_id": row["example_id"],
                    "source": row["source"],
                    "ground_truth": row["ground_truth"],
                    "native_raw_output": left["raw_output"],
                    "native_label": left["label"],
                    "stitched_raw_output": right["raw_output"],
                    "stitched_label": right["label"],
                }
            )


def write_json(path, value, overwrite):
    resolve_output(path, overwrite)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def configure_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuts", type=parse_cuts, default=SUPPORTED_CUTS)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("Choose --resume or --overwrite, not both")
    if args.tiny:
        args.data_dir = args.data_dir or Path("data/generated_tiny")
        args.manifest = args.manifest or Path("data/generated_tiny_manifest.json")
        args.artifacts_dir = args.artifacts_dir or Path("artifacts/multicut_tiny")
        args.results_dir = args.results_dir or Path("results/multicut_tiny")
        args.max_epochs = args.max_epochs if args.max_epochs is not None else 2
        args.patience = args.patience if args.patience is not None else 1
        args.max_new_tokens = (
            args.max_new_tokens if args.max_new_tokens is not None else 4
        )
        args.bootstrap = args.bootstrap if args.bootstrap is not None else 20
    else:
        args.data_dir = args.data_dir or Path(os.environ.get("STITCH_DATA", "data/generated"))
        args.manifest = args.manifest or Path(
            os.environ.get("STITCH_MANIFEST", "data/baseline_manifest.json")
        )
        args.artifacts_dir = args.artifacts_dir or Path(
            os.environ.get("STITCH_ARTIFACTS", "artifacts/multicut")
        )
        args.results_dir = args.results_dir or Path(
            os.environ.get("STITCH_RESULTS", "results/multicut")
        )
        args.max_epochs = args.max_epochs if args.max_epochs is not None else 50
        args.patience = args.patience if args.patience is not None else 5
        args.max_new_tokens = (
            args.max_new_tokens if args.max_new_tokens is not None else 16
        )
        args.bootstrap = args.bootstrap if args.bootstrap is not None else 1000
    if min(args.max_epochs, args.patience, args.max_new_tokens, args.bootstrap) <= 0:
        parser.error("Epochs, patience, max-new-tokens, and bootstrap must be positive")
    return args


def main():
    args = configure_args()
    initialize_queue_status(args.cuts)
    started = time.perf_counter()
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    expected_manifest = None if args.tiny else FULL_MANIFEST_SHA256
    manifest, manifest_hash, rows = validate_dataset(
        args.manifest, args.data_dir, expected_manifest
    )
    tokenizer = AutoTokenizer.from_pretrained(GUARD, revision=REVISIONS[GUARD])
    tokenized = {role: tokenize_rows(tokenizer, value) for role, value in rows.items()}
    token_hashes = {role: input_ids_hash(value) for role, value in tokenized.items()}

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.artifacts_dir / "activation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    timings = {}
    caches = {}
    cache_paths = {}
    for role in ("train", "selection"):
        for short, model_name in (("base", BASE), ("guard", GUARD)):
            update_queue_status(
                queue_state="RUNNING",
                phase="BASE_EXTRACTION" if short == "base" else "GUARD_EXTRACTION",
                current_cut=None,
            )
            path = cache_dir / f"{role}_{short}_boundaries.pt"
            cache_paths[(role, short)] = path
            states, seconds, reused = extract_shared_cache(
                model_name,
                rows[role],
                tokenized[role],
                args.cuts,
                path,
                manifest_hash,
                device,
                args.resume,
                args.overwrite,
            )
            caches[(role, short)] = states
            timings[f"extract_{role}_{short}_seconds"] = seconds
            print(f"cache={path} reused={reused}")

    cache_hashes = {
        f"{role}_{short}": file_hash(path)
        for (role, short), path in cache_paths.items()
    }
    adapters = {}
    dm_by_cut = {}
    checkpoint_hashes = {}
    completed_adapter_cuts = []
    for cut in args.cuts:
        update_queue_status(
            queue_state="RUNNING",
            phase="TRAIN_ADAPTER",
            current_cut=cut,
            next_cut=cut,
        )
        checkpoint = args.artifacts_dir / f"direct_matching_cut{cut}.pt"
        metadata = checkpoint_metadata(
            cut,
            manifest_hash,
            {key: token_hashes[key] for key in ("train", "selection")},
            cache_hashes,
            args.seed,
            args.max_epochs,
            args.patience,
        )
        adapter, dm, training_seconds, reused = load_or_train_adapter(
            cut,
            caches,
            checkpoint,
            metadata,
            args.max_epochs,
            args.patience,
            args.seed,
            args.resume,
            args.overwrite,
        )
        adapters[cut] = adapter
        dm_by_cut[cut] = dm
        timings[f"adapter_training_cut{cut}_seconds"] = training_seconds
        checkpoint_hashes[cut] = file_hash(checkpoint)
        completed_adapter_cuts.append(cut)
        adapter_offset = args.cuts.index(cut) + 1
        update_queue_status(
            completed_adapter_cuts=completed_adapter_cuts,
            next_cut=(
                args.cuts[adapter_offset]
                if adapter_offset < len(args.cuts)
                else None
            ),
        )
        print(f"checkpoint={checkpoint} reused={reused}")

    generation_meta = {
        "schema_version": 2,
        "guard_model": GUARD,
        "guard_revision": REVISIONS[GUARD],
        "tokenizer": GUARD,
        "tokenizer_revision": REVISIONS[GUARD],
        "template": "official Guard chat template applied exactly once",
        "manifest_sha256": manifest_hash,
        "evaluation_sha256": json_hash(rows["evaluation"]),
        "evaluation_input_ids_sha256": token_hashes["evaluation"],
        "max_new_tokens": args.max_new_tokens,
        "generation": "greedy do_sample=False use_cache=True",
        "parser": "Safety: Safe|Controversial|Unsafe regex; parse failure is binary unsafe",
    }
    native_path = args.results_dir / "native_predictions.json"
    update_queue_status(
        queue_state="RUNNING",
        phase="NATIVE_EVAL",
        current_cut=None,
        next_cut=args.cuts[0],
    )
    if native_path.exists() and args.resume:
        native, native_seconds, native_reused = prediction_stage_v2(
            native_path, generation_meta, True, False, lambda: ([], 0.0)
        )
    else:
        guard = load_model(GUARD, device)
        native, native_seconds, native_reused = prediction_stage_v2(
            native_path,
            generation_meta,
            False,
            args.overwrite,
            lambda: native_generate(
                guard,
                tokenizer,
                rows["evaluation"],
                tokenized["evaluation"],
                device,
                args.max_new_tokens,
            ),
        )
        del guard
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    timings["native_generation_seconds"] = native_seconds
    print(f"native_stage reused={native_reused}")

    native_metrics = metrics(rows["evaluation"], native)
    records = []
    per_cut = {}
    completed_eval_cuts = []
    for cut in args.cuts:
        update_queue_status(
            queue_state="RUNNING",
            phase="STITCHED_EVAL",
            current_cut=cut,
            next_cut=cut,
        )
        stitched_meta = {
            **generation_meta,
            "cut": cut,
            "cut_semantics": cut_semantics(cut),
            "base_model": BASE,
            "base_revision": REVISIONS[BASE],
            "adapter_sha256": checkpoint_hashes[cut],
        }
        stitched_path = args.results_dir / f"stitched_predictions_cut{cut}.json"
        if stitched_path.exists() and args.resume:
            stitched, seconds, reused = prediction_stage_v2(
                stitched_path, stitched_meta, True, False, lambda: ([], 0.0)
            )
        else:
            stitched, seconds, reused = prediction_stage_v2(
                stitched_path,
                stitched_meta,
                False,
                args.overwrite,
                lambda cut=cut: stitched_generate_for_cut(
                    cut,
                    adapters[cut],
                    tokenizer,
                    rows["evaluation"],
                    tokenized["evaluation"],
                    device,
                    args.max_new_tokens,
                ),
            )
        timings[f"stitched_generation_cut{cut}_seconds"] = seconds
        print(f"stitched_stage cut={cut} reused={reused}")
        stitched_metrics = metrics(rows["evaluation"], stitched)
        penalty = penalties(native_metrics, stitched_metrics)
        intervals = bootstrap(
            rows["evaluation"], native, stitched, args.bootstrap, args.seed
        )
        transition_matrix = transitions(rows["evaluation"], native, stitched)
        record = aggregate_record(
            cut,
            dm_by_cut[cut],
            native_metrics,
            stitched_metrics,
            penalty,
            intervals,
        )
        records.append(record)
        per_cut[str(cut)] = {
            "cut": cut,
            "relative_depth": cut / NUM_LAYERS,
            "cut_semantics": cut_semantics(cut),
            "adapter": dm_by_cut[cut],
            "native_metrics": native_metrics,
            "stitched_metrics": stitched_metrics,
            "penalties": penalty,
            "bootstrap_95_ci": intervals,
            "transitions": transition_matrix,
            "runtime_seconds": {
                "adapter_training": timings[f"adapter_training_cut{cut}_seconds"],
                "native_generation_shared": timings["native_generation_seconds"],
                "stitched_generation": seconds,
            },
            "artifacts": {
                "checkpoint": str(args.artifacts_dir / f"direct_matching_cut{cut}.pt"),
                "checkpoint_sha256": checkpoint_hashes[cut],
                "stitched_predictions": str(stitched_path),
            },
        }
        write_predictions_csv(
            args.results_dir / f"predictions_cut{cut}.csv",
            rows["evaluation"],
            native,
            stitched,
            args.overwrite or args.resume,
        )
        write_json(
            args.results_dir / f"summary_cut{cut}.json",
            per_cut[str(cut)],
            args.overwrite or args.resume,
        )
        completed_eval_cuts.append(cut)
        eval_offset = args.cuts.index(cut) + 1
        update_queue_status(
            completed_cuts=completed_eval_cuts,
            completed_eval_cuts=completed_eval_cuts,
            next_cut=(
                args.cuts[eval_offset] if eval_offset < len(args.cuts) else None
            ),
        )

    update_queue_status(
        queue_state="RUNNING",
        phase="AGGREGATION",
        current_cut=None,
        next_cut=None,
    )
    csv_path = args.results_dir / "multicut_summary.csv"
    resolve_output(csv_path, args.overwrite or args.resume)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "schema_version": 2,
        "git_commit": git_commit(),
        "direction": "Base->Guard",
        "models": REVISIONS,
        "architecture": {"num_hidden_layers": NUM_LAYERS, "hidden_size": HIDDEN_SIZE},
        "cuts": list(args.cuts),
        "cut_semantics": {str(cut): cut_semantics(cut) for cut in args.cuts},
        "manifest_sha256": manifest_hash,
        "dataset_revisions": {
            key: value["revision"] for key, value in manifest["datasets"].items()
        },
        "input_ids_sha256": token_hashes,
        "counts": {role: len(value) for role, value in rows.items()},
        "seed": args.seed,
        "bootstrap_resamples": args.bootstrap,
        "bootstrap_pairing": "same sampled indices for native and stitched",
        "native_evaluation_runs_total": 1,
        "native_evaluation_runs_this_invocation": 0 if native_reused else 1,
        "native_predictions_reused": native_reused,
        "activation_cache_sha256": cache_hashes,
        "records": records,
        "per_cut": per_cut,
        "timings": timings,
        "total_runtime_seconds": time.perf_counter() - started,
        "environment": environment(),
    }
    json_path = args.results_dir / "multicut_summary.json"
    write_json(json_path, summary, args.overwrite or args.resume)
    update_queue_status(
        queue_state="DONE",
        phase="DONE",
        current_cut=None,
        completed_cuts=list(args.cuts),
        completed_adapter_cuts=list(args.cuts),
        completed_eval_cuts=list(args.cuts),
        next_cut=None,
        failure=None,
    )
    print(
        "MULTICUT PIPELINE: PASS\n"
        f"cuts={','.join(map(str, args.cuts))}\n"
        f"summary_csv={csv_path}\nsummary_json={json_path}"
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        fail_queue_status(f"{type(error).__name__}: {error}")
        raise
