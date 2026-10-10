import numpy as np
import pytest
from pipeline.change import change_vector_magnitude, threshold_change, run_baseline


def test_change_magnitude_shape_and_finiteness():
    before = np.ones((3, 20, 20), dtype=np.float32) * 100
    after = before.copy()
    after[:, 5:15, 5:15] += 50
    magnitude = change_vector_magnitude(before, after, histogram_match=False)
    assert magnitude.shape == (20, 20)
    assert np.isfinite(magnitude).all()
    assert magnitude[8, 8] > magnitude[1, 1]


def test_baseline_detects_large_synthetic_patch_without_histogram_match():
    rng = np.random.default_rng(7)
    before = rng.normal(100, 2, size=(3, 48, 48)).astype(np.float32)
    after = before.copy()
    after[:, 16:32, 16:32] += 80
    pred, mag, threshold = run_baseline(before, after, histogram_match=False,
                                        min_object_size=8, opening_radius=1)
    assert pred.shape == (48, 48)
    assert np.isfinite(threshold)
    assert pred[22:26, 22:26].mean() > 0.5
    assert pred[0:8, 0:8].mean() < 0.2


def test_invalid_pixels_not_predicted():
    before = np.zeros((2, 10, 10), dtype=np.float32)
    after = before.copy()
    after[:, 2:8, 2:8] = 10
    valid = np.ones((10, 10), dtype=bool)
    valid[2:8, 2:8] = False
    pred, _, _ = run_baseline(before, after, valid, histogram_match=False)
    assert not pred[~valid].any()


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shapes differ"):
        change_vector_magnitude(np.zeros((3, 10, 10)), np.zeros((3, 9, 10)))


def test_histogram_matching_ignores_invalid_pixel_values():
    rng = np.random.default_rng(42)
    before = rng.normal(100, 5, size=(3, 20, 20)).astype(np.float32)
    after = before.copy()
    after[:, 5:10, 5:10] += 30
    valid = np.ones((20, 20), dtype=bool)
    valid[:4, :] = False
    before_extreme = before.copy()
    after_extreme = after.copy()
    before_extreme[:, ~valid] = 1e6
    after_extreme[:, ~valid] = -1e6
    m1 = change_vector_magnitude(before, after, valid, histogram_match=True)
    m2 = change_vector_magnitude(before_extreme, after_extreme, valid, histogram_match=True)
    assert np.allclose(m1[valid], m2[valid], atol=1e-5)
    assert not m2[~valid].any()
