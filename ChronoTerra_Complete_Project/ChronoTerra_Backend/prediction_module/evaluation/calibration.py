"""Validation-only temperature scaling and reliability diagnostics for segmentation scores.

This module does not guarantee deployment calibration. It fits one scalar temperature to
held-out validation logits, never the official test split, and reports score reliability on
the same validation sample. Spatial pixels are correlated, so ECE/Brier are diagnostics,
not proof of independent uncertainty calibration or out-of-distribution detection.
"""
from __future__ import annotations

from typing import Any
import numpy as np

CALIBRATION_SCHEMA_VERSION = 1


def _as_logits_labels(logits: np.ndarray, labels: np.ndarray, ignore_index: int = 255):
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if z.ndim != 2:
        raise ValueError(f"logits must be [N,K] or [N], got {z.shape}")
    if z.shape[0] != y.size:
        raise ValueError(f"logit rows {z.shape[0]} do not match label count {y.size}")
    finite = np.all(np.isfinite(z), axis=1)
    valid = (y != int(ignore_index)) & finite
    k = z.shape[1]
    if k == 1:
        valid &= np.isin(y, [0, 1])
    else:
        valid &= (y >= 0) & (y < k)
    z, y = z[valid], y[valid]
    if not y.size:
        raise ValueError("No valid finite labeled pixels available for calibration")
    return z, y


def probabilities_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Return [N,K] probabilities. K=1 is interpreted as a single binary logit."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a positive finite number")
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if z.ndim != 2 or not np.all(np.isfinite(z)):
        raise ValueError("logits must be a finite [N,K] array")
    z = z / float(temperature)
    if z.shape[1] == 1:
        # Numerically stable sigmoid.
        p1 = np.empty(z.shape[0], dtype=np.float64)
        positive = z[:, 0] >= 0
        p1[positive] = 1.0 / (1.0 + np.exp(-z[positive, 0]))
        ez = np.exp(z[~positive, 0])
        p1[~positive] = ez / (1.0 + ez)
        return np.stack([1.0 - p1, p1], axis=1)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(np.clip(z, -700, 700))
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-300)


def calibration_metrics(logits: np.ndarray, labels: np.ndarray, *, temperature: float = 1.0,
                        ignore_index: int = 255, bins: int = 15) -> dict[str, Any]:
    z, y = _as_logits_labels(logits, labels, ignore_index=ignore_index)
    probs = probabilities_from_logits(z, temperature)
    n, k = probs.shape
    if k == 2:
        true = y.astype(np.int64)
    else:
        true = y
    chosen = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = (chosen == true).astype(np.float64)
    p_true = np.clip(probs[np.arange(n), true], 1e-12, 1.0)
    nll = float(-np.log(p_true).mean())
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), true] = 1.0
    brier = float(np.square(probs - onehot).sum(axis=1).mean())
    edges = np.linspace(0.0, 1.0, max(2, int(bins)) + 1)
    ece = 0.0
    reliability = []
    for i in range(len(edges) - 1):
        # Include confidence=1 in the final bin.
        mask = (confidence >= edges[i]) & (confidence < edges[i + 1] if i < len(edges) - 2 else confidence <= edges[i + 1])
        count = int(mask.sum())
        if count:
            avg_conf = float(confidence[mask].mean())
            avg_acc = float(correct[mask].mean())
            ece += count / n * abs(avg_acc - avg_conf)
            reliability.append({"bin_low": float(edges[i]), "bin_high": float(edges[i + 1]),
                                "count": count, "mean_confidence": avg_conf,
                                "empirical_accuracy": avg_acc})
        else:
            reliability.append({"bin_low": float(edges[i]), "bin_high": float(edges[i + 1]),
                                "count": 0, "mean_confidence": None, "empirical_accuracy": None})
    return {"n_valid_samples": int(n), "num_probability_classes": int(k),
            "temperature": float(temperature), "negative_log_likelihood": nll,
            "multiclass_brier_score": brier, "top_label_expected_calibration_error": float(ece),
            "reliability_bins": reliability,
            "caveat": "Pixel samples are spatially correlated; these metrics describe the validation sample only and do not guarantee deployment calibration."}


def _nll_for_temperature(z: np.ndarray, y: np.ndarray, temperature: float) -> float:
    """Fast numerically-stable NLL objective used during scalar temperature search."""
    scaled = np.asarray(z, dtype=np.float64) / float(temperature)
    if scaled.shape[1] == 1:
        # Binary logistic loss from a single logit.
        yy = y.astype(np.float64)
        return float(np.mean(np.logaddexp(0.0, scaled[:, 0]) - yy * scaled[:, 0]))
    row_max = np.max(scaled, axis=1, keepdims=True)
    log_partition = row_max[:, 0] + np.log(np.exp(scaled - row_max).sum(axis=1))
    return float(np.mean(log_partition - scaled[np.arange(y.size), y]))


def fit_temperature(logits: np.ndarray, labels: np.ndarray, *, ignore_index: int = 255,
                    max_samples: int = 200_000, seed: int = 42,
                    temperature_min: float = 0.05, temperature_max: float = 20.0,
                    grid_points: int = 241) -> dict[str, Any]:
    """Fit a single scalar temperature by minimizing validation negative log-likelihood.

    Input logits must be [N,K] / [N] for classification outputs. For binary segmentation,
    pass the *one-channel logit* rather than thresholded masks. For large scenes the valid
    sample is uniformly subsampled with a fixed seed. Test labels must never be passed.
    """
    if max_samples < 1 or grid_points < 3:
        raise ValueError("max_samples must be >=1 and grid_points >=3")
    if not (np.isfinite(temperature_min) and np.isfinite(temperature_max) and 0 < temperature_min < temperature_max):
        raise ValueError("temperature bounds must be finite and satisfy 0 < min < max")
    z, y = _as_logits_labels(logits, labels, ignore_index=ignore_index)
    original_n = int(y.size)
    if y.size > max_samples:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(y.size, size=max_samples, replace=False))
        z, y = z[selected], y[selected]
    before = calibration_metrics(z, y, temperature=1.0, ignore_index=ignore_index)
    candidates = np.geomspace(temperature_min, temperature_max, int(grid_points))
    nlls = np.asarray([_nll_for_temperature(z, y, float(t)) for t in candidates])
    best_idx = int(np.argmin(nlls))
    # A deterministic logarithmic grid is dependency-free and avoids unstable optimizer starts.
    temperature = float(candidates[best_idx])
    after = calibration_metrics(z, y, temperature=temperature, ignore_index=ignore_index)
    return {"schema_version": CALIBRATION_SCHEMA_VERSION, "method": "single_scalar_temperature_scaling",
            "fit_split": "validation_only_required_by_caller", "temperature": temperature,
            "fit_objective": "minimize validation negative log-likelihood",
            "temperature_search": {"min": float(temperature_min), "max": float(temperature_max), "grid_points": int(grid_points)},
            "valid_samples_in_source": original_n, "valid_samples_used": int(y.size),
            "subsampled": bool(original_n > y.size), "seed": int(seed),
            "metrics_before": before, "metrics_after": after,
            "status": "FITTED_ON_VALIDATION_SAMPLE", "deployment_guarantee": False,
            "warning": "Temperature scaling calibrates a scalar on the validation sample only. It does not solve input shift; re-evaluate on independent test data and avoid calling this guaranteed confidence."}


def calibrate_binary_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scale an already tiled/overlap-averaged binary probability map.

    This matches the urban inference path, which averages tile probabilities. Return p(class 1).
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be finite before calibration")
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    logits = np.log(p) - np.log1p(-p)
    return (1.0 / (1.0 + np.exp(-np.clip(logits / float(temperature), -80, 80)))).astype(np.float32)
