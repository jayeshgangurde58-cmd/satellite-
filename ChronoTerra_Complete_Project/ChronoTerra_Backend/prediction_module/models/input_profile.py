"""Training-distribution profiles for diagnostic input-shift screening.

This is not a calibrated out-of-distribution detector. Profiles are computed only from
training inputs, saved into the model checkpoint, and compared with inference inputs after
the exact training normalizer. Diagnostics are advisory and never silently modify data.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Any
import numpy as np

PROFILE_VERSION = 1


def _sample_pixels(array_chw: np.ndarray, valid_hw: np.ndarray, *, max_pixels: int, seed: int) -> np.ndarray:
    arr = np.asarray(array_chw, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected CHW array, got shape {arr.shape}")
    # Never mutate the caller-owned mask; validity arrays may be reused by inference/output code.
    valid = np.array(valid_hw, dtype=bool, copy=True)
    if valid.shape != arr.shape[1:]:
        raise ValueError(f"Validity mask {valid.shape} does not match image {arr.shape[1:]}")
    valid &= np.all(np.isfinite(arr), axis=0)
    idx = np.flatnonzero(valid)
    if not idx.size:
        return np.empty((arr.shape[0], 0), dtype=np.float32)
    if idx.size > max_pixels:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_pixels, replace=False))
    flat = arr.reshape(arr.shape[0], -1)
    return flat[:, idx]


def build_input_profile(samples: Iterable[tuple[np.ndarray, np.ndarray]], *,
                        channel_names: Sequence[str] | None = None,
                        max_pixels_per_sample: int = 5000, seed: int = 42) -> dict[str, Any]:
    """Build per-channel quantiles from normalized training inputs only.

    Each tuple is (normalized CHW input, valid HW mask). For a paired task, concatenate the
    before/after channels in the same order used by inference. Sampling is bounded per scene.
    """
    if max_pixels_per_sample < 1:
        raise ValueError("max_pixels_per_sample must be positive")
    channel_values: list[list[np.ndarray]] = []
    n_samples = 0
    n_pixels = 0
    for i, (arr, valid) in enumerate(samples):
        sampled = _sample_pixels(arr, valid, max_pixels=max_pixels_per_sample, seed=seed + i)
        if not channel_values:
            channel_values = [[] for _ in range(sampled.shape[0])]
        if sampled.shape[0] != len(channel_values):
            raise ValueError("All samples must have the same number/order of channels")
        if sampled.shape[1]:
            for ch in range(sampled.shape[0]):
                channel_values[ch].append(sampled[ch])
            n_pixels += sampled.shape[1]
        n_samples += 1
    if not channel_values or n_pixels == 0:
        raise ValueError("Cannot build input profile: no valid training pixels")
    names = list(channel_names or [f"channel_{i+1}" for i in range(len(channel_values))])
    if len(names) != len(channel_values):
        raise ValueError("channel_names count does not match input channel count")
    channels = []
    for idx, pieces in enumerate(channel_values):
        vals = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)
        if not vals.size:
            raise ValueError(f"No valid samples for channel {names[idx]}")
        q = np.percentile(vals, [1, 5, 50, 95, 99])
        channels.append({"index_1based": idx + 1, "name": names[idx],
                         "p01": float(q[0]), "p05": float(q[1]), "p50": float(q[2]),
                         "p95": float(q[3]), "p99": float(q[4]), "sample_count": int(vals.size)})
    return {"schema_version": PROFILE_VERSION, "source": "training_split_only",
            "normalization_applied_before_profile": True,
            "profile_method": "per-channel p01/p05/p50/p95/p99 of bounded valid-pixel samples",
            "training_scene_or_row_samples": int(n_samples), "sampled_valid_pixels_per_channel": int(n_pixels),
            "channels": channels,
            "warning": "Diagnostic reference distribution, not a probability model or guarantee of domain compatibility."}


def compare_input_profile(input_chw: np.ndarray, valid_hw: np.ndarray,
                          train_profile: dict[str, Any] | None, *,
                          max_pixels: int = 50000, seed: int = 17,
                          range_fraction_review: float = 0.20,
                          median_shift_range_review: float = 2.0) -> dict[str, Any]:
    """Compare a normalized input against training quantiles without changing predictions.

    Heuristic thresholds are deliberately labelled unvalidated. This function must not be
    described as calibrated OOD detection; it never rejects an input on profile drift alone.
    """
    if not train_profile or not train_profile.get("channels"):
        return {"status": "NOT_VERIFIED", "method": "training_quantile_profile", "channels": [],
                "reason": "Checkpoint has no saved training-distribution profile; retrain or treat drift as unverified."}
    arr = np.asarray(input_chw, dtype=np.float32)
    # Never mutate the caller-owned mask; validity arrays may be reused by inference/output code.
    valid = np.array(valid_hw, dtype=bool, copy=True)
    if arr.ndim != 3 or valid.shape != arr.shape[1:]:
        return {"status": "REJECT", "method": "training_quantile_profile", "channels": [],
                "reason": "Input shape/mask is inconsistent for profile comparison."}
    reference = train_profile["channels"]
    if len(reference) != arr.shape[0]:
        return {"status": "REJECT", "method": "training_quantile_profile", "channels": [],
                "reason": f"Checkpoint profile has {len(reference)} channels but inference profile has {arr.shape[0]}."}
    sampled = _sample_pixels(arr, valid, max_pixels=max_pixels, seed=seed)
    if sampled.shape[1] == 0:
        return {"status": "REJECT", "method": "training_quantile_profile", "channels": [],
                "reason": "No valid pixels to compare against training profile."}
    rows = []
    flags = []
    for i, ref in enumerate(reference):
        vals = sampled[i]
        q01, q05, q50, q95, q99 = np.percentile(vals, [1, 5, 50, 95, 99])
        lo, hi = float(ref["p01"]), float(ref["p99"])
        outside = float(np.mean((vals < lo) | (vals > hi)))
        robust = max(float(ref["p95"]) - float(ref["p05"]), 0.05)
        shift = float((q50 - float(ref["p50"])) / robust)
        review = outside > range_fraction_review or abs(shift) > median_shift_range_review
        row = {"index_1based": i + 1, "name": ref.get("name", f"channel_{i+1}"),
               "input_p01_p05_p50_p95_p99": [float(q01), float(q05), float(q50), float(q95), float(q99)],
               "training_p01_p05_p50_p95_p99": [float(ref[k]) for k in ("p01", "p05", "p50", "p95", "p99")],
               "fraction_outside_training_p01_p99": outside,
               "median_shift_in_training_robust_ranges": shift,
               "heuristic_review_flag": bool(review)}
        rows.append(row)
        if review:
            flags.append(row["name"])
    return {"status": "DIAGNOSTIC_ONLY", "method": "training_quantile_profile",
            "review_flags": flags, "channels": rows,
            "heuristic_thresholds": {"outside_training_p01_p99_fraction": range_fraction_review,
                                      "absolute_median_shift_in_training_p05_p95_range": median_shift_range_review},
            "threshold_validation": "NOT_VALIDATED; values are screening heuristics, not calibrated OOD probabilities",
            "valid_pixels_sampled": int(sampled.shape[1]),
            "warning": "This profile can flag sensor/season/radiometry/content shift but cannot diagnose its cause. Do not automatically alter or reject the input from this report alone."}
