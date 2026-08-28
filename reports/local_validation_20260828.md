# Local validation before server rental

Date: 2026-08-28 (Europe/Moscow)

This is software/plumbing validation only. The full 800/200/200 experiment was
not run locally, and no scientific conclusion is drawn from the tiny run.

## Data

- `stitch_train`: 800 unique prompts (400 WildChat, 400 BeaverTails)
- `model_selection`: 200 unique prompts (100 + 100)
- `evaluation`: 200 unique JBB goals (100 harmful, 100 benign)
- normalized overlap train/selection/evaluation: 0 / 0 / 0
- Direct Matching candidates above 2048 rendered tokens are excluded, never
  truncated: 17 WildChat candidates excluded
- selected train/selection/evaluation truncated examples: 0 / 0 / 0
- committed manifest SHA-256:
  `614209e8f746787d139a4d1d67d6475c30fae365ac21b78531ff704af6d995e0`

Guard-template token lengths:

| role | min | median | mean | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 298 | 314 | 443.030 | 1044.2 | 2037 |
| selection | 299 | 317 | 442.605 | 1045.4 | 1204 |
| evaluation | 302 | 310 | 311.135 | 320 | 327 |

Median template overhead is 296 tokens for train/selection and 297 for JBB.

## Real-model tiny E2E

- environment: Python 3.13.13, torch 2.13.0+cpu, Transformers 5.16.1
- counts: 4 train, 2 model selection, 4 JBB evaluation
- boundary extraction: Base and Guard completed with finite BF16 caches
- Direct Matching: two epochs, adapter gradients finite, selection checkpoint selected
- native generation: 4/4 parseable
- stitched generation: 4/4 parseable
- CSV, JSON, checkpoint and separate prediction stages created
- native and stitched stage `input_ids_sha256` are identical:
  `2c9cfa329dae48e170c5d77e4eb80e1f8233de5c4499527fc919942cf80d087b`
- full shared-input evaluation replay runtime: 496.90 seconds (cached extraction/training)
- compatible complete `--resume`: reused without modifying summary
- ordinary rerun without `--resume`/`--overwrite`: refused

Result: `FULL PIPELINE: PASS`.

## Regressions

- Base→Base cut18: exact max/mean absolute difference 0/0
- Guard→Guard cut18: exact max/mean absolute difference 0/0
- cached Guard generation: HF and manual generated token IDs exactly equal
- Direct Matching objective: finite loss and gradients

## Deferred to the server

GPU model, VRAM, driver/CUDA, compatible official PyTorch CUDA wheel, batch-size
feasibility, throughput, peak VRAM and full-run ETA depend on the not-yet-rented
machine. `server_smoke.sh` is the gate before the full wrapper.
