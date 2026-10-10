"""Evaluate the classical baseline against OSCD labels, per region and in aggregate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .oscd import load_oscd_pairs, read_oscd_pair
from .metrics import binary_metrics
from pipeline.change import run_baseline


def evaluate(root: Path, split: str = "test", limit: int | None = None, histogram_match: bool = True,
             min_object_size: int = 8, opening_radius: int = 1) -> dict:
    pairs = load_oscd_pairs(root, split=split)
    if limit is not None:
        pairs = pairs[:limit]
    if split == "test":
        print("NOTE: test split selected. Do not change thresholds based on these results.")
    results = []
    started_all = time.perf_counter()
    for pair in pairs:
        started = time.perf_counter()
        before, after, label, valid = read_oscd_pair(pair)
        pred, magnitude, threshold = run_baseline(
            before, after, valid_mask=valid, histogram_match=histogram_match,
            min_object_size=min_object_size, opening_radius=opening_radius,
        )
        row = {"pair": pair.name, **binary_metrics(label, pred, valid),
               "otsu_threshold": float(threshold), "mean_change_magnitude": float(magnitude[valid].mean()),
               "runtime_seconds": round(time.perf_counter() - started, 4)}
        results.append(row)
        print(f"{pair.name}: F1={row['f1']:.4f} IoU={row['iou']:.4f} "
              f"precision={row['precision']:.4f} recall={row['recall']:.4f} "
              f"FP={row['fp']} runtime={row['runtime_seconds']:.2f}s")

    metric_keys = ("precision", "recall", "f1", "iou", "false_positive_rate",
                   "false_positive_fraction_of_valid")
    aggregate = {key: float(np.mean([r[key] for r in results])) for key in metric_keys}
    # Also report micro-F1/IoU derived by summing confusion counts rather than averaging pairs.
    totals = {key: int(sum(r[key] for r in results)) for key in ("tp", "fp", "fn", "tn", "valid_pixels")}
    p = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    r = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    aggregate["micro_precision"] = p
    aggregate["micro_recall"] = r
    aggregate["micro_f1"] = 2 * p * r / (p + r) if p + r else 0.0
    denominator = totals["tp"] + totals["fp"] + totals["fn"]
    aggregate["micro_iou"] = totals["tp"] / denominator if denominator else 0.0
    return {
        "dataset": "OSCD / Onera Satellite Change Detection Dataset",
        "split": split,
        "pair_count": len(results),
        "baseline": {"name": "histogram_match + normalized_change_vector + Otsu + morphology",
                     "histogram_match": histogram_match, "min_object_size_pixels": min_object_size,
                     "opening_radius_pixels": opening_radius},
        "aggregate_macro": aggregate,
        "aggregate_confusion_counts": totals,
        "total_runtime_seconds": round(time.perf_counter() - started_all, 4),
        "pairs": results,
        "warnings": [
            "These results evaluate a classical baseline, not a trained neural model.",
            "OSCD is an urban-change benchmark, not a structural-damage severity benchmark.",
            "Do not tune parameters against the test split; use train/development data for ablations."
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Directory containing the extracted OSCD folders")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--limit", type=int, default=None, help="Optional small smoke-test limit")
    parser.add_argument("--output", type=Path, default=Path("outputs/oscd_results.json"))
    parser.add_argument("--histogram-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-object-size", type=int, default=8)
    parser.add_argument("--opening-radius", type=int, default=1)
    args = parser.parse_args()
    report = evaluate(args.root, args.split, args.limit, args.histogram_match,
                      args.min_object_size, args.opening_radius)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote report: {args.output.resolve()}")
    print(json.dumps(report["aggregate_macro"], indent=2))


if __name__ == "__main__":
    main()
