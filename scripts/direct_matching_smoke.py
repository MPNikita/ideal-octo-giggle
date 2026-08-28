"""Fast adapter-objective regression; the server smoke also runs real models."""

import torch
from torch import nn

from run_cut18_baseline import diagnostics, example_loss
from stitching_core import HIDDEN_SIZE


def main():
    torch.manual_seed(20260827)
    source = [torch.randn(length, HIDDEN_SIZE, dtype=torch.bfloat16) for length in (2, 3, 4)]
    target = [x + 0.05 * torch.randn_like(x) for x in source]
    adapter = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
    with torch.no_grad(): adapter.weight.copy_(torch.eye(HIDDEN_SIZE)); adapter.bias.zero_()
    loss = example_loss(adapter, source, target, [0, 1, 2])
    loss.backward()
    finite = math_isfinite(loss.item()) and all(torch.isfinite(p.grad).all() for p in adapter.parameters())
    print(f"loss={loss.item()} diagnostics={diagnostics(adapter, source, target)} gradients_finite={finite}")
    print(f"DIRECT MATCHING OBJECTIVE SMOKE: {'PASS' if finite else 'FAIL'}")
    raise SystemExit(0 if finite else 1)


def math_isfinite(value):
    return value == value and value not in (float("inf"), float("-inf"))


if __name__ == "__main__": main()
