# Qwen3 Base → Guard model stitching

Research baseline for affine stitches from `Qwen/Qwen3-4B` into
`Qwen/Qwen3Guard-Gen-4B` at cuts `0,9,18,27,35`. Both model revisions and all
dataset revisions are immutable. The Guard tokenizer and official Guard chat
template are applied exactly once; the resulting IDs are shared by Base and
Guard. The completed cut18 result and its original runner remain the regression
reference.

The experiment is intentionally plain Python: shared multi-boundary extraction,
one separately selected affine adapter per cut, one native Guard evaluation,
cached stitched JBB generation per cut, safety metrics, transitions, paired
bootstrap CIs, stage-level resume, and machine-readable aggregate outputs.

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
The full multi-cut data/run command is:

```bash
python scripts/prepare_baseline_data.py
bash scripts/server_run_multicut.sh
```

The wrapper runs smoke before the full baseline. Its underlying resumable
command is:

```bash
python scripts/run_multicut_baseline.py --cuts 0,9,18,27,35 --resume
```

For a non-scientific local plumbing check:

```bash
python scripts/run_multicut_baseline.py --cuts 0,18,35 --tiny --resume
```

Defaults produce 800 training, 200 model-selection and 200 independent JBB
examples. Generated JSONL, activations, model weights, checkpoints, logs and
results are ignored by Git. Never infer scientific quality from `--tiny` runs.

## Outputs

- `artifacts/multicut/direct_matching_cut{0,9,18,27,35}.pt`
- `results/multicut/native_predictions.json`
- `results/multicut/predictions_cut{0,9,18,27,35}.csv`
- `results/multicut/multicut_summary.csv`
- `results/multicut/multicut_summary.json`

The legacy `scripts/run_cut18_baseline.py` and its output definitions are kept
unchanged for cut18 regression checks.

## Qwen CKA caveat

The available Qwen CKA tensors have weak embedded provenance. Their exact
representation-index semantics and therefore their CKA-index-to-stitch-cut
mapping are unresolved; see `data/qwen_cka_inventory.json`. Existing CKA was
reported to use native templates separately per model, whereas stitching uses
identical Guard-tokenized IDs. This permits exploratory comparison only and
requires input-aligned re-extraction before a strict correlation claim.

Runtime roots can be changed with `HF_HOME`, `STITCH_DATA`,
`STITCH_ARTIFACTS`, `STITCH_RESULTS`, and `STITCH_LOGS`.
