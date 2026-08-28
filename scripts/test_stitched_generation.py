"""One-prompt HF generation == manual cached Guard self-stitch regression."""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stitching_core import GUARD, REVISIONS, stitched_cached_greedy, tokenize_guard_prompt


def identity(value): return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args(); device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(GUARD, revision=REVISIONS[GUARD])
    model = AutoModelForCausalLM.from_pretrained(
        GUARD, revision=REVISIONS[GUARD], dtype=torch.bfloat16,
        low_cpu_mem_usage=True).to(device).eval()
    inputs = tokenize_guard_prompt(tokenizer, "Explain why rain forms.", device)
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False,
                                max_new_tokens=args.max_new_tokens, use_cache=True)
    native = output[:, inputs.input_ids.shape[1]:].cpu()
    manual, _ = stitched_cached_greedy(model, model, identity, inputs.input_ids,
                                       inputs.attention_mask, args.max_new_tokens)
    equal = torch.equal(native, manual)
    print(f"native_tokens={native.tolist()}\nmanual_tokens={manual.tolist()}\nequal={equal}")
    print(f"CACHED SELF-STITCH GENERATION: {'PASS' if equal else 'FAIL'}")
    raise SystemExit(0 if equal else 1)


if __name__ == "__main__": main()
