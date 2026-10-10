"""Create a hash-bound deployment manifest from an actual held-out evaluation report.

This utility does not train models or certify their suitability. Without the explicit
--approve-for-inference flag it creates a pending manifest that keeps API inference blocked.
Review is a human responsibility; the script verifies report structure and arithmetic, not
whether the dataset labels, split provenance, or reviewer claims are truthful.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METRIC_KEYS = ("precision", "recall", "f1", "iou", "false_positive_rate")
DEFAULT_THRESHOLDS = {
    "minimum_precision": 0.70,
    "minimum_recall": 0.70,
    "minimum_f1": 0.70,
    "minimum_iou": 0.50,
    "maximum_false_positive_rate": 0.10,
}
TASKS = {"urban_change", "flood_extent", "building_damage"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _binary_metrics_from_confusion(matrix: Any, *, positive_classes: set[int]) -> dict[str, float]:
    """Collapse a multiclass confusion matrix into positive-vs-rest metrics.

    Matrix convention in this project is rows=ground truth and columns=prediction.
    """
    import numpy as np

    cm = np.asarray(matrix)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or cm.shape[0] < 2:
        raise ValueError("Held-out report has no valid square confusion matrix.")
    if not np.issubdtype(cm.dtype, np.number) or not np.isfinite(cm).all() or (cm < 0).any():
        raise ValueError("Confusion matrix contains invalid counts.")
    if not np.allclose(cm, np.rint(cm)):
        raise ValueError("Confusion matrix values must be integer counts.")
    class_count = int(cm.shape[0])
    if not positive_classes or any(c < 0 or c >= class_count for c in positive_classes):
        raise ValueError("Positive class IDs do not match the confusion matrix.")
    pos = np.zeros(class_count, dtype=bool)
    pos[list(positive_classes)] = True
    tp = int(cm[np.ix_(pos, pos)].sum())
    fp = int(cm[np.ix_(~pos, pos)].sum())
    fn = int(cm[np.ix_(pos, ~pos)].sum())
    tn = int(cm[np.ix_(~pos, ~pos)].sum())
    precision = _safe_rate(tp, tp + fp)
    recall = _safe_rate(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _safe_rate(2 * tp, 2 * tp + fp + fn),
        "iou": _safe_rate(tp, tp + fp + fn),
        "false_positive_rate": _safe_rate(fp, fp + tn),
    }


def _scene_count(report: dict[str, Any]) -> int:
    if isinstance(report.get("per_scene"), dict):
        return len(report["per_scene"])
    if isinstance(report.get("per_scene"), list):
        return len(report["per_scene"])
    if isinstance(report.get("pairs"), list):
        return len(report["pairs"])
    value = report.get("num_scenes", report.get("pair_count"))
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def normalize_held_out_report(report: dict[str, Any], task: str) -> tuple[dict[str, float], str, int, str]:
    """Normalize the real trainer's OSCD or manifest-evaluator JSON into deployment metrics."""
    if task not in TASKS:
        raise ValueError(f"Unsupported task: {task}")
    dataset_name = report.get("dataset") or report.get("task")
    if task == "urban_change":
        # The OSCD evaluator writes micro metrics plus one row per official test scene.
        micro = report.get("micro")
        if not isinstance(dataset_name, str) or "test" not in dataset_name.lower():
            raise ValueError("Urban deployment requires the OSCD evaluator's test-split report.")
        if report.get("model_selection_used_test_data") is not False:
            raise ValueError("The OSCD report must state model_selection_used_test_data=false.")
        if not isinstance(micro, dict):
            raise ValueError("OSCD report is missing pooled micro metrics.")
        metrics = {key: micro.get(key) for key in METRIC_KEYS}
        split_type = "held_out_by_scene"
    else:
        if report.get("split") != "test":
            raise ValueError("Deployment requires an evaluated test split; validation metrics cannot unlock inference.")
        if report.get("model_selection_used_evaluated_split") is not False:
            raise ValueError("Manifest report must state model_selection_used_evaluated_split=false.")
        expected_task = report.get("task")
        if expected_task != task:
            raise ValueError(f"Evaluation task mismatch: requested {task}, report says {expected_task!r}.")
        pooled = report.get("pooled")
        if not isinstance(pooled, dict) or not isinstance(pooled.get("confusion_matrix"), list):
            raise ValueError("Manifest evaluation report is missing the pooled confusion matrix.")
        if task == "flood_extent":
            metrics = _binary_metrics_from_confusion(pooled["confusion_matrix"], positive_classes={1})
        else:
            # The release gate for building damage is for identifying any damage (minor/major/
            # destroyed), not the separate severity class. The full class metrics remain in the
            # original test report and must accompany this deployment summary.
            metrics = _binary_metrics_from_confusion(pooled["confusion_matrix"], positive_classes={2, 3, 4})
        split_type = "held_out_by_scene"
    scene_count = _scene_count(report)
    if scene_count < 1:
        raise ValueError("Evaluation report contains no test scenes/pairs.")
    normalized: dict[str, float] = {}
    for key in METRIC_KEYS:
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError(f"Evaluation metric {key!r} is missing or outside [0,1].")
        normalized[key] = float(value)
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("Evaluation report must identify a dataset.")
    return normalized, split_type, scene_count, dataset_name.strip()[:160]


def create_manifest(
    checkpoint: Path,
    report_path: Path,
    task: str,
    *,
    approve_for_inference: bool = False,
    reviewer: str | None = None,
    reviewed_at: str | None = None,
    thresholds: dict[str, float] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    checkpoint = checkpoint.resolve()
    report_path = report_path.resolve()
    if not checkpoint.is_file():
        raise ValueError("Checkpoint file does not exist.")
    if not report_path.is_file():
        raise ValueError("Evaluation report file does not exist.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Evaluation report must be a JSON object.")
    metric_values, split_type, scene_count, dataset_name = normalize_held_out_report(report, task)
    threshold_values = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    if set(threshold_values) != set(DEFAULT_THRESHOLDS):
        raise ValueError("All five deployment thresholds must be explicitly defined.")
    for key, val in threshold_values.items():
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)) or not 0 <= float(val) <= 1:
            raise ValueError(f"Threshold {key} must be a number in [0,1].")
        threshold_values[key] = float(val)
    thresholds_satisfied = (
        metric_values["precision"] >= threshold_values["minimum_precision"]
        and metric_values["recall"] >= threshold_values["minimum_recall"]
        and metric_values["f1"] >= threshold_values["minimum_f1"]
        and metric_values["iou"] >= threshold_values["minimum_iou"]
        and metric_values["false_positive_rate"] <= threshold_values["maximum_false_positive_rate"]
    )
    if approve_for_inference:
        if not reviewer or not reviewer.strip():
            raise ValueError("--reviewer is required when --approve-for-inference is set.")
        if not reviewed_at:
            raise ValueError("--reviewed-at is required when --approve-for-inference is set.")
        try:
            dt = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("--reviewed-at must be ISO-8601, for example 2026-10-10T12:00:00Z.") from exc
        if dt.tzinfo is None:
            raise ValueError("--reviewed-at must include a timezone offset or Z.")
        if not thresholds_satisfied:
            raise ValueError("The held-out metrics do not meet all configured deployment thresholds; inference approval was not created.")
    metrics_path = checkpoint.parent / "deployment_metrics.json"
    source_report_copy = checkpoint.parent / "deployment_source_report.json"
    # Retain the exact evaluated report bytes beside the checkpoint so the API can verify
    # the lineage without depending on a user-supplied absolute path.
    source_report_bytes = report_path.read_bytes()
    source_report_copy.write_bytes(source_report_bytes)
    source_report_hash = sha256_file(source_report_copy)
    metrics_payload = {
        "task": task,
        "dataset_name": dataset_name,
        "split_type": split_type,
        "held_out_scene_count": scene_count,
        "metrics": metric_values,
        "metrics_interpretation": (
            "binary candidate-change metrics" if task == "urban_change" else
            "binary water-vs-not-water metrics" if task == "flood_extent" else
            "binary any-damage-vs-background-or-no-damage metrics; full severity scores are in the source report"
        ),
        "source_report_file": source_report_copy.name,
        "source_report_sha256": source_report_hash,
        "source_report_split": report.get("split") or report.get("dataset"),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, allow_nan=False), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "task": task,
        "checkpoint_sha256": sha256_file(checkpoint),
        "approved_for_inference": bool(approve_for_inference),
        "evaluation": {
            "metrics_file": metrics_path.name,
            "metrics_sha256": sha256_file(metrics_path),
            "source_report_file": source_report_copy.name,
            "source_report_sha256": source_report_hash,
            "split_type": split_type,
            "held_out_scene_count": scene_count,
            "dataset_name": dataset_name,
            "evaluation_date": datetime.now(timezone.utc).date().isoformat(),
            "deployment_thresholds": threshold_values,
        },
        "review": {
            "status": "APPROVED" if approve_for_inference else "PENDING",
            "reviewer": reviewer.strip() if approve_for_inference and reviewer else None,
            "reviewed_at": reviewed_at if approve_for_inference else None,
        },
        "notes": (
            "Approval records a human-reviewed deployment decision; it does not guarantee generalization beyond the evaluated dataset."
        ),
    }
    manifest_path = checkpoint.parent / "deployment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return metrics_path, manifest_path, {"metrics": metric_values, "thresholds_satisfied": thresholds_satisfied, "approved": bool(approve_for_inference)}


def _threshold_arg(name: str):
    def parse(value: str) -> float:
        try:
            x = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be numeric") from exc
        if not math.isfinite(x) or not 0 <= x <= 1:
            raise argparse.ArgumentTypeError("must be in [0,1]")
        return x
    return parse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True, help="Output of the corresponding frozen test evaluator")
    parser.add_argument("--approve-for-inference", action="store_true", help="Explicitly approve after reviewing real held-out evidence")
    parser.add_argument("--reviewer", help="Human reviewer identifier/name (required for approval)")
    parser.add_argument("--reviewed-at", help="ISO-8601 timestamp with timezone (required for approval)")
    parser.add_argument("--minimum-precision", type=_threshold_arg("minimum_precision"), default=DEFAULT_THRESHOLDS["minimum_precision"])
    parser.add_argument("--minimum-recall", type=_threshold_arg("minimum_recall"), default=DEFAULT_THRESHOLDS["minimum_recall"])
    parser.add_argument("--minimum-f1", type=_threshold_arg("minimum_f1"), default=DEFAULT_THRESHOLDS["minimum_f1"])
    parser.add_argument("--minimum-iou", type=_threshold_arg("minimum_iou"), default=DEFAULT_THRESHOLDS["minimum_iou"])
    parser.add_argument("--maximum-fpr", type=_threshold_arg("maximum_false_positive_rate"), default=DEFAULT_THRESHOLDS["maximum_false_positive_rate"])
    args = parser.parse_args()
    thresholds = {
        "minimum_precision": args.minimum_precision, "minimum_recall": args.minimum_recall,
        "minimum_f1": args.minimum_f1, "minimum_iou": args.minimum_iou,
        "maximum_false_positive_rate": args.maximum_fpr,
    }
    try:
        metrics_path, manifest_path, summary = create_manifest(
            args.checkpoint, args.report, args.task,
            approve_for_inference=args.approve_for_inference,
            reviewer=args.reviewer, reviewed_at=args.reviewed_at, thresholds=thresholds,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"metrics_file": str(metrics_path), "manifest_file": str(manifest_path), **summary}, indent=2))
    if args.approve_for_inference and not summary["thresholds_satisfied"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
