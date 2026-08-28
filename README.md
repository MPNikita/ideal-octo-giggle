# Qwen3 Base → Guard model stitching

Research baseline for an affine stitch from `Qwen/Qwen3-4B` into
`Qwen/Qwen3Guard-Gen-4B` at `cut=18`. Both model revisions and all dataset
revisions are immutable. The Guard tokenizer and official Guard chat template
are applied exactly once; the resulting IDs are shared by Base and Guard.

The full experiment is intentionally plain Python: paired boundary extraction,
example-balanced Direct Matching, model-selection-only early stopping, native
and cached stitched JBB generation, safety metrics, transitions, bootstrap CIs,
and machine-readable outputs.

## Reproduce

Install a CUDA-compatible PyTorch wheel for the actual server first (PyTorch is
deliberately not pinned before the GPU/driver is known), then:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
bash scripts/server_smoke.sh
```

Only after smoke passes, follow [the server runbook](docs/SERVER_RUNBOOK.md).
The full data/run commands are:

```bash
python scripts/prepare_baseline_data.py
python scripts/run_cut18_baseline.py --resume
```

Defaults produce 800 training, 200 model-selection and 200 independent JBB
examples. Generated JSONL, activations, model weights, checkpoints, logs and
results are ignored by Git. Never infer scientific quality from `--tiny` runs.

## Outputs

- `artifacts/direct_matching_cut18.pt`
- `results/cut18_predictions.csv`
- `results/cut18_summary.json`

Runtime roots can be changed with `HF_HOME`, `STITCH_DATA`,
`STITCH_ARTIFACTS`, `STITCH_RESULTS`, and `STITCH_LOGS`.
