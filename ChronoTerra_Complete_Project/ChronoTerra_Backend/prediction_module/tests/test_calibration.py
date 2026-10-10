"""Calibration tests use synthetic logits only and are not satellite accuracy evidence."""
import numpy as np
import pytest
from evaluation.calibration import (fit_temperature, calibration_metrics,
                                    probabilities_from_logits, calibrate_binary_probabilities)


def test_temperature_scaling_improves_nll_for_overconfident_synthetic_logits():
    rng = np.random.default_rng(100)
    truth = rng.integers(0, 2, size=5000)
    base = (truth * 2 - 1) * 0.8 + rng.normal(0, 1.0, size=truth.size)
    overconfident = base * 4.0
    fit = fit_temperature(overconfident, truth, seed=3, grid_points=101)
    assert fit["status"] == "FITTED_ON_VALIDATION_SAMPLE"
    assert fit["temperature"] > 1.0
    assert fit["metrics_after"]["negative_log_likelihood"] <= fit["metrics_before"]["negative_log_likelihood"]
    assert fit["deployment_guarantee"] is False


def test_multiclass_calibration_ignores_sentinel_labels():
    logits = np.array([[5., 0., 0.], [0., 5., 0.], [0., 0., 5.], [4., 0., 0.]])
    labels = np.array([0, 1, 2, 255])
    report = calibration_metrics(logits, labels, ignore_index=255)
    assert report["n_valid_samples"] == 3
    assert report["negative_log_likelihood"] < 0.1
    assert len(report["reliability_bins"]) == 15


def test_binary_probability_temperature_scaling_is_monotonic():
    p = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    out = calibrate_binary_probabilities(p, 2.0)
    assert out[0] < out[1] < out[2]
    assert out[1] == pytest.approx(0.5, abs=1e-6)
    assert np.allclose(probabilities_from_logits(np.array([[1., 0.]]), 1.0).sum(axis=1), [1.0])
