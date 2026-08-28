"""Bitwise identity self-stitch regression on one model and selected cuts."""

import argparse
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stitching_core import (
    BASE, GUARD, REVISIONS, boundary_hidden_states, manual_self_logits,
    prefix_hidden_states, validate_cuts,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=(BASE, GUARD), required=True)
    parser.add_argument("--cuts", nargs="+", type=int, default=[18])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(); device = torch.device(args.device)
    cuts = validate_cuts(args.cuts)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=REVISIONS[args.model])
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=REVISIONS[args.model], dtype=torch.bfloat16,
        low_cpu_mem_usage=True).to(device).eval()
    inputs = tokenizer(["Safety matters."], return_tensors="pt").to(device)
    with torch.inference_mode():
        native = model(**inputs, use_cache=False, output_hidden_states=True)
        boundaries = boundary_hidden_states(
            model, inputs.input_ids, inputs.attention_mask, cuts
        )
        passed = True
        for cut in cuts:
            legacy_boundary = prefix_hidden_states(
                model, inputs.input_ids, inputs.attention_mask, cut
            )
            stitched = manual_self_logits(model, inputs.input_ids, inputs.attention_mask, cut)
            logits_diff = (native.logits.float() - stitched.float()).abs()
            boundary_diff = (
                native.hidden_states[cut].float() - boundaries[cut].float()
            ).abs()
            legacy_diff = (legacy_boundary.float() - boundaries[cut].float()).abs()
            ok = (
                logits_diff.max().item() == 0.0
                and boundary_diff.max().item() == 0.0
                and legacy_diff.max().item() == 0.0
            )
            passed &= ok
            print(
                f"model={args.model} cut={cut} "
                f"boundary_max_abs_diff={boundary_diff.max().item()} "
                f"boundary_mean_abs_diff={boundary_diff.mean().item()} "
                f"legacy_boundary_max_abs_diff={legacy_diff.max().item()} "
                f"logits_max_abs_diff={logits_diff.max().item()} "
                f"logits_mean_abs_diff={logits_diff.mean().item()} "
                f"status={'PASS' if ok else 'FAIL'}"
            )
    del model; gc.collect()
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__": main()
