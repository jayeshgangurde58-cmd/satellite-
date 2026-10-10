"""Unit tests for train-only input distribution diagnostics (not accuracy tests)."""
import numpy as np
from models.input_profile import build_input_profile, compare_input_profile


def test_profile_is_built_from_valid_training_pixels_only():
    arr = np.stack([np.linspace(0, 1, 100).reshape(10, 10), np.full((10, 10), 0.5)]).astype(np.float32)
    valid = np.ones((10, 10), dtype=bool)
    valid[:5, :] = False
    profile = build_input_profile([(arr, valid)], channel_names=["blue", "green"], max_pixels_per_sample=200)
    assert profile["source"] == "training_split_only"
    assert profile["channels"][0]["p50"] > 0.7
    assert profile["channels"][1]["p50"] == 0.5


def test_profile_comparison_flags_distribution_shift_without_rejecting():
    base = np.random.default_rng(2).normal(0.5, 0.1, size=(2, 64, 64)).astype(np.float32)
    valid = np.ones((64, 64), dtype=bool)
    profile = build_input_profile([(base, valid)], channel_names=["VV", "VH"], max_pixels_per_sample=5000)
    shifted = np.clip(base + 0.35, 0, 1)
    report = compare_input_profile(shifted, valid, profile)
    assert report["status"] == "DIAGNOSTIC_ONLY"
    assert report["review_flags"]
    assert "not calibrated" in report["threshold_validation"].lower()


def test_missing_profile_is_explicit_not_verified():
    x = np.ones((3, 16, 16), dtype=np.float32)
    report = compare_input_profile(x, np.ones((16, 16), dtype=bool), None)
    assert report["status"] == "NOT_VERIFIED"


def test_profile_does_not_mutate_callers_valid_mask():
    arr = np.array([[[1.0, np.nan], [2.0, 3.0]]], dtype=np.float32)
    valid = np.ones((2, 2), dtype=bool)
    original = valid.copy()
    build_input_profile([(arr, valid)], channel_names=["band"], max_pixels_per_sample=10)
    assert np.array_equal(valid, original)
