import numpy as np
import pytest
from evaluation.metrics import binary_metrics


def test_known_binary_metrics():
    truth = np.array([[1, 1, 0, 0]], dtype=bool)
    pred = np.array([[1, 0, 1, 0]], dtype=bool)
    got = binary_metrics(truth, pred)
    assert (got["tp"], got["fp"], got["fn"], got["tn"]) == (1, 1, 1, 1)
    assert got["precision"] == pytest.approx(0.5)
    assert got["recall"] == pytest.approx(0.5)
    assert got["f1"] == pytest.approx(0.5)
    assert got["iou"] == pytest.approx(1 / 3)


def test_ignored_pixels_do_not_affect_scores():
    truth = np.array([[1, 0], [0, 1]], bool)
    pred = np.array([[1, 1], [1, 0]], bool)
    valid = np.array([[1, 0], [1, 0]], bool)
    got = binary_metrics(truth, pred, valid)
    assert got["valid_pixels"] == 2
    assert got["tp"] == 1 and got["fp"] == 1 and got["fn"] == 0
    assert got["recall"] == pytest.approx(1.0)


def test_empty_positive_denominators_use_zero_not_nan():
    got = binary_metrics(np.zeros((3, 3), bool), np.zeros((3, 3), bool))
    assert got["precision"] == got["recall"] == got["f1"] == got["iou"] == 0.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shapes differ"):
        binary_metrics(np.zeros((2, 2)), np.zeros((3, 3)))


def test_valid_mask_shape_mismatch_raises():
    with pytest.raises(ValueError, match="valid_mask shape"):
        binary_metrics(np.zeros((2, 2)), np.zeros((2, 2)), np.ones((3, 3)))


def test_conventional_false_positive_rate_is_reported():
    truth = np.array([[1, 0, 0, 0]], dtype=bool)
    pred = np.array([[1, 1, 0, 0]], dtype=bool)
    got = binary_metrics(truth, pred)
    assert got["false_positive_rate"] == pytest.approx(1 / 3)
    assert got["false_positive_fraction_of_valid"] == pytest.approx(1 / 4)
