"""Transparent change-priority tiers (heuristic, not validated damage severity)."""
from __future__ import annotations


def change_priority_tier(area_m2: float, magnitude_relative: float) -> tuple[str, float]:
    """Return priority tier and score based on area + relative scene magnitude.

    The fixed thresholds are a demo heuristic. They must be calibrated and
    validated for a target use case before being interpreted operationally.
    """
    area_score = min(max(float(area_m2), 0.0) / 10_000.0, 1.0)
    mag_score = min(max(float(magnitude_relative), 0.0), 1.0)
    score = 0.5 * area_score + 0.5 * mag_score
    if score < 0.25:
        tier = "Low"
    elif score < 0.50:
        tier = "Medium"
    elif score < 0.75:
        tier = "High"
    else:
        tier = "Critical"
    return tier, score
