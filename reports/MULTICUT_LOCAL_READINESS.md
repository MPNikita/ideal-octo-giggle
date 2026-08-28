# Qwen multi-cut local readiness

Audit completed locally on 2026-08-29 (Europe/Moscow). No full 800/200/200
multi-cut experiment, new Qwen activation extraction, or CKA recomputation was
performed.

## A. Repository

- Commit before: `3a8b026d8d6d27cd9cea95c4b25caec794969724`
- Multi-cut implementation commit: `c5dd8cc27adbde35abc10d84a8cd2317937c80c7`
- Remote: `git@github-mpnikita:MPNikita/ideal-octo-giggle.git`
- Implementation worktree clean immediately after commit: **PASS**
- Normal push to `origin/main`, with no force: **PASS**

The Hugging Face dataset/artifact repository
`probguard/datasets-for-safety-stitching` was not accessible in this local
environment (unauthenticated repository API returned not found), so its remote
inventory could not be independently confirmed. No relevant private team Git
checkout or team CKA implementation was present in the local project tree.

## B. Existing cut18 preservation

- cut18 semantics unchanged: **PASS**
- Legacy `scripts/run_cut18_baseline.py` unchanged: **PASS**
- Reference scientific result untouched: **PASS**
- Scientific execution commit: `66f93c1c0ec55cd4e4839b785aaa71532f678de2`
- Manifest SHA-256:
  `614209e8f746787d139a4d1d67d6475c30fae365ac21b78531ff704af6d995e0`

Archived reference hashes were rechecked after implementation:

| Artifact | SHA-256 |
| --- | --- |
| `direct_matching_cut18.pt` | `b79bb0f42ee5f0ba861529b46e19a469a701a89e34c25a878014835fc228f99c` |
| `cut18_predictions.csv` | `10ea2f30c94482743262b5bed2cf1b54706d8acd14607f803b326ff1f7b14bf0` |
| `cut18_summary.json` | `f942874bf6a91594a6490c2a22956a82973e8774b6f70993cf1507a58c7f0c84` |

The new cut18 boundary was compared against both the legacy prefix function and
the Hugging Face hidden-state boundary on fixed input; both were exactly equal.
Self-stitch native/manual logits and cached generation were also exactly equal.

## C. Multi-cut implementation

| Requirement | Status |
| --- | --- |
| Supported cuts `0,9,18,27,35` | **PASS** |
| Receiver-input boundary semantics | **PASS** |
| One shared boundary extraction pass per model/example | **PASS** |
| Only requested boundaries retained | **PASS** |
| Separate identity-initialized affine adapter per cut | **PASS** |
| Same Direct Matching loss/training/selection policy | **PASS** |
| Single native Guard evaluation shared across cuts | **PASS** |
| Per-cut cached stitched evaluation | **PASS** |
| Paired fixed-seed bootstrap | **PASS** |
| Per-cut and aggregate CSV/JSON | **PASS** |
| Metadata/hash-validated stage resume | **PASS** |
| CUDA/BF16 server wrapper and isolated smoke | **PASS** |

The paired bootstrap uses the same sampled indices for native and stitched
predictions. Missing-class recall in a tiny resample is represented as `NaN`
rather than causing division by zero; this is identical to the legacy metrics
when both classes are present and permits `nanpercentile` to use valid
resamples.

## D. Tiny test

The exact pinned Base and Guard revisions were used locally. The plumbing run
used 4 train, 2 selection, and 4 evaluation examples, two training epochs, 20
paired bootstrap resamples, and cuts 0/18/35. Its metric values are not
scientific results.

| Cut | Base/Guard self-stitch | Adapter training | Stitched generation | Metrics/bootstrap | Outputs |
| ---: | --- | --- | --- | --- | --- |
| 0 | **PASS**, exact zero diff | **PASS** | **PASS** | **PASS** | **PASS** |
| 18 | **PASS**, exact zero diff | **PASS** | **PASS** | **PASS** | **PASS** |
| 35 | **PASS**, exact zero diff | **PASS** | **PASS** | **PASS** | **PASS** |

Additional completed checks:

- Shared extraction produced four compact role/model caches and three cut axes
  in each cache: **PASS**
- Three distinct adapter checkpoints: **PASS**
- Native evaluation once and reused for every cut: **PASS**
- Three stitched prediction stages: **PASS**
- No-op resume reused all caches, checkpoints, native predictions, and stitched
  predictions: **PASS**
- Deliberately stale prediction metadata was refused: **PASS**
- Cached Guard identity generation at cuts 0/18/35 matched native tokens
  exactly: **PASS**

An optional all-five-cut Base self-stitch process was launched separately at the
operator's request and deliberately not monitored. Its outcome is not used as
readiness evidence.

## E. Qwen CKA inventory

The locally available artifacts were inventoried without regenerating hidden
states. Compact audit metadata is in `data/qwen_cka_inventory.json`.

| Artifact | SHA-256 | Shape/rows | Conditions or relation |
| --- | --- | --- | --- |
| `common/native-final-tokens-base.pt` | `f8099480895030917a66c768364e754deb8c7e0af8d8ef0540a2bf9ffdd0ffe7` | clean/suffix: FP16 `[100,36,2560]` | embedded labels: `model=base`, `format=native` |
| `common/native-final-tokens-guard.pt` | `7e3895827775030e11462d4f2b3f4f0830bed439b0d931dbf24c518b32b9b3a4` | clean/suffix: FP16 `[100,36,2560]` | embedded labels: `model=guard`, `format=native` |
| `common/evaluation.csv` | `69d767b3a8eebab4e3cc9521c40ccc8e95c0824b82bbea5a1aadf1b8bf3a5823` | 100 rows, index 0..99 | tensor indices equal CSV index values |

The intended models are `Qwen/Qwen3-4B` and
`Qwen/Qwen3Guard-Gen-4B`, consistent with the surrounding project materials and
hidden size, but exact model identity is not proven by embedded tensor metadata.
Model revisions, tokenizer revisions, serialized input IDs, template snapshots,
extraction source commit, final-token rule, and representation-index semantics
are unknown. Equality of integer indices with `evaluation.csv` is verified, but
their semantic/content binding is not cryptographically proven.

The reported extraction setup was native chat templates separately for Base and
Guard and the final token. This setup is from prior project correspondence, not
from self-contained tensor provenance.

## F. CKA implementation

- Source implementation: **unavailable locally**
- Source commit: **unknown**
- Biased or unbiased estimator: **unknown**
- Bootstrap/shuffle settings: **unknown**
- Recomputed in this audit: **NO**
- Reproducible from existing tensors with the team's exact method: **NO**, until
  the implementation, commit, and tensor semantics are recovered

No third CKA formula was introduced. `scripts/compare_cka_stitching.py` only
joins explicitly proven/confirmed mappings and computes separate clean/suffix
Spearman correlations when at least three valid points exist.

## G. Layer mapping

**QWEN CKA INDEX MAPPING: UNRESOLVED**

| stitch cut | boundary semantics | CKA tensor index | mapping confidence |
| ---------: | ------------------ | ---------------: | ------------------ |
| 0 | embedding output / input to receiver block 0 | unavailable/unknown | **UNKNOWN** |
| 9 | after block 8 / input to receiver block 9 | unknown | **UNKNOWN** |
| 18 | after block 17 / input to receiver block 18 | unknown | **UNKNOWN** |
| 27 | after block 26 / input to receiver block 27 | unknown | **UNKNOWN** |
| 35 | after block 34 / input to receiver block 35 | unknown | **UNKNOWN** |

The 36 CKA positions may be block outputs, but that cannot be used as a mapping
without authoritative extraction code or documentation. In particular, cut 0
must not be substituted with block-0 output if embedding output was not saved.

## H. CKA caveat

Existing Qwen CKA and stitching are not yet perfectly input-aligned: CKA uses
native templates per model; stitching uses identical Guard-tokenized IDs.

The current artifacts are acceptable for an explicitly caveated exploratory
comparison after their index mapping is established. Input-aligned activation
re-extraction is required before a strict CKA-versus-safety-penalty claim.

## I. Server launch command

After clone/pull and environment setup on a future CUDA/BF16 server:

```bash
export HF_HOME="/persistent/.../huggingface"
export STITCH_DATA="/persistent/.../probguard/data"
export STITCH_MANIFEST="$PWD/data/baseline_manifest.json"
export STITCH_ARTIFACTS="/persistent/.../probguard/artifacts/multicut"
export STITCH_RESULTS="/persistent/.../probguard/results/multicut"
export STITCH_LOGS="/persistent/.../probguard/logs"
export STITCH_CUTS="0,9,18,27,35"
bash scripts/server_run_multicut.sh
```

The wrapper verifies a clean commit, CUDA/BF16, the exact manifest, smoke, full
outputs, and resumable stage metadata. It contains no hardcoded server IP or
machine-specific path.

## J. Remaining server-only work

1. Create and verify the CUDA environment on the rented GPU host.
2. Run the GPU smoke gate.
3. Run full shared Base/Guard train and selection activation extraction.
4. Train the five adapters for cuts 0, 9, 18, 27, and 35.
5. Run one native Guard evaluation on the full 200-example JBB split.
6. Run five stitched evaluations on those identical inputs.
7. Copy results off-server with verified hashes.
8. Resolve CKA index semantics and, if scientifically accepted, run the prepared
   final comparison; otherwise perform a later input-aligned CKA re-extraction.

## K. Verdict

QWEN MULTICUT SERVER READINESS: PASS
