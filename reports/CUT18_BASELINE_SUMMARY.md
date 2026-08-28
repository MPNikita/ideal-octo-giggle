# CUT18 Scientific Baseline Summary

## Experiment

```text
Base -> Guard
cut = 18
Direct Matching affine adapter
800 train / 200 selection / 200 JBB eval
```

## Direct Matching

```text
identity selection MSE = 2.1361932568
best selection MSE = 1.6208240596
improvement = 24.1256%

identity selection last-token MSE = 0.4092224848
trained selection last-token MSE = 0.0128020694
```

## Safety

| metric | native | stitched | penalty/delta |
| --- | ---: | ---: | ---: |
| Macro-F1 | 0.657042 | 0.597037 | +0.060004 penalty |
| Balanced accuracy | 0.690000 | 0.645000 | +0.045000 penalty |
| Unsafe recall | 1.000000 | 0.990000 | +0.010000 penalty |
| Safe FPR | 0.620000 | 0.700000 | +0.080000 stitched-native |

Bootstrap:

```text
Macro-F1 penalty 95% CI:
[0.015505, 0.110314]

Balanced accuracy penalty 95% CI:
[0.013559, 0.083339]
```

Transitions:

```text
native Unsafe|Controversial -> stitched Safe: 3
harmful: 1
benign: 2
```

## Interpretation

The affine Direct Matching adapter substantially improves held-out
representation matching, but this does not preserve the Guard function
perfectly.

At cut 18, stitched evaluation shows lower Macro-F1 and balanced accuracy than
the native Guard. Unsafe recall is almost preserved, while the larger observed
degradation is an increase in false positives on benign JBB examples.

This is one cut and one model-selection configuration; layer-wise claims
require the remaining cuts.
