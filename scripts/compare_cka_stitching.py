"""Join proven Qwen CKA depth rows to multi-cut stitching results.

This script does not compute CKA and does not infer a layer mapping. It accepts
only rows explicitly marked as proven/confirmed and reports correlations only
when at least three valid mapped cuts are available.
"""

import argparse
import csv
import json
import math
from pathlib import Path


VALID_MAPPING_STATUSES = {"proven", "confirmed"}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def average_ranks(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + 1 + end) / 2
        for offset in range(start, end):
            ranks[indexed[offset][0]] = rank
        start = end
    return ranks


def pearson(left, right):
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def spearman(left, right):
    if len(left) < 3:
        return None
    return pearson(average_ranks(left), average_ranks(right))


def join_rows(cka_rows, stitching_rows):
    stitching_by_cut = {int(row["cut"]): row for row in stitching_rows}
    joined = []
    for row in cka_rows:
        status = row.get("mapping_status", "").strip().lower()
        mapped = row.get("mapped_stitch_cut", "").strip()
        if status not in VALID_MAPPING_STATUSES or not mapped:
            continue
        cut = int(mapped)
        if cut not in stitching_by_cut:
            continue
        stitch = stitching_by_cut[cut]
        joined.append(
            {
                "cut": cut,
                "relative_depth": stitch["relative_depth"],
                "representation_index": row["representation_index"],
                "layer_semantics": row["layer_semantics"],
                "mapping_status": row["mapping_status"],
                "cka_clean": row["cka_clean"],
                "cka_suffix": row["cka_suffix"],
                "macro_f1_penalty": stitch["macro_f1_penalty"],
                "bal_acc_penalty": stitch["bal_acc_penalty"],
                "unsafe_recall_penalty": stitch["unsafe_recall_penalty"],
                "safe_fpr_delta": stitch["safe_fpr_delta"],
            }
        )
    return sorted(joined, key=lambda row: row["cut"])


def correlations(joined):
    results = {}
    for condition in ("cka_clean", "cka_suffix"):
        for penalty in ("macro_f1_penalty", "bal_acc_penalty"):
            pairs = [
                (float(row[condition]), float(row[penalty]))
                for row in joined
                if row[condition] != "" and row[penalty] != ""
            ]
            key = f"spearman_{condition}_vs_{penalty}"
            results[key] = {
                "n": len(pairs),
                "rho": spearman(
                    [pair[0] for pair in pairs], [pair[1] for pair in pairs]
                ),
                "status": "computed" if len(pairs) >= 3 else "insufficient_mapped_points",
            }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cka-csv", type=Path, default=Path("results/qwen_cka_depth.csv"))
    parser.add_argument(
        "--stitching-csv", type=Path, default=Path("results/multicut/multicut_summary.csv")
    )
    parser.add_argument(
        "--output-csv", type=Path, default=Path("results/cka_stitching_comparison.csv")
    )
    parser.add_argument(
        "--output-json", type=Path, default=Path("results/cka_stitching_comparison.json")
    )
    args = parser.parse_args()

    joined = join_rows(read_csv(args.cka_csv), read_csv(args.stitching_csv))
    if not joined:
        raise SystemExit(
            "No proven/confirmed CKA-to-cut mappings overlap the stitching summary"
        )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)
    payload = {
        "mapped_points": len(joined),
        "correlations": correlations(joined),
        "caveat": (
            "Correlation is descriptive only. Existing native-template CKA and "
            "identical-Guard-tokenized stitching inputs are not perfectly aligned."
        ),
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"joined_points={len(joined)}")
    print(json.dumps(payload["correlations"], indent=2))


if __name__ == "__main__":
    main()
