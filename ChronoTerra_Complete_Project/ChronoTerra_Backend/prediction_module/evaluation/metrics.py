"""Binary change-detection metrics that correctly handle ignored/nodata pixels."""
from __future__ import annotations
import numpy as np


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, valid_mask: np.ndarray | None = None) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    if truth.shape != pred.shape:
        raise ValueError(f"truth/prediction shapes differ: {truth.shape} != {pred.shape}")
    valid = np.ones(truth.shape, dtype=bool)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != truth.shape:
            raise ValueError("valid_mask shape must match truth and prediction")
    truth, pred = truth[valid], pred[valid]
    tp = int(np.count_nonzero(truth & pred))
    fp = int(np.count_nonzero(~truth & pred))
    fn = int(np.count_nonzero(truth & ~pred))
    tn = int(np.count_nonzero(~truth & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    # Conventional FPR conditions on actual negatives, unlike FP/valid_pixels.
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    return {
        "valid_pixels": int(valid.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision), "recall": float(recall), "f1": float(f1), "iou": float(iou),
        "false_positive_rate": float(false_positive_rate),
        "true_change_pixels": int(truth.sum()), "predicted_change_pixels": int(pred.sum()),
        "false_positive_fraction_of_valid": float(fp / valid.sum()) if valid.sum() else 0.0,
    }
