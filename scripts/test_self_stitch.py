"""Bitwise identity self-stitch regression on one model and selected cuts."""

import argparse
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stitching_core import BASE, GUARD, REVISIONS, manual_self_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=(BASE, GUARD), required=True)
    parser.add_argument("--cuts", nargs="+", type=int, default=[18])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(); device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=REVISIONS[args.model])
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=REVISIONS[args.model], dtype=torch.bfloat16,
        low_cpu_mem_usage=True).to(device).eval()
    inputs = tokenizer(["Safety matters."], return_tensors="pt").to(device)
    with torch.inference_mode():
        native = model(**inputs, use_cache=False).logits
        passed = True
        for cut in args.cuts:
            stitched = manual_self_logits(model, inputs.input_ids, inputs.attention_mask, cut)
            diff = (native.float() - stitched.float()).abs()
            ok = diff.max().item() == 0.0
            passed &= ok
            print(f"model={args.model} cut={cut} max_abs_diff={diff.max().item()} "
                  f"mean_abs_diff={diff.mean().item()} status={'PASS' if ok else 'FAIL'}")
    del model; gc.collect()
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__": main()
