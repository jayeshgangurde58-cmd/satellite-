"""Regression tests for geospatial/radiometric safeguards; synthetic fixtures only."""
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from PIL import Image

from quality_gate.pair_preflight import validate_pair, _footprint_overlap_metrics
from quality_gate.prediction_preflight import read_scl_on_image_grid, _apply_scl
from quality_gate.registration_diagnostic import estimate_translation


def _write_tif(path: Path, array: np.ndarray, *, transform=None, crs="EPSG:32632", nodata=None, descriptions=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        array = array[None]
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[1], width=array.shape[2],
                       count=array.shape[0], dtype=array.dtype, crs=crs,
                       transform=transform or from_origin(500000, 1000000, 10, 10), nodata=nodata) as dst:
        dst.write(array)
        if descriptions:
            for i, name in enumerate(descriptions, 1): dst.set_band_description(i, name)
    return path


def test_common_valid_pixels_are_exactly_reported(tmp_path):
    b = np.ones((3, 32, 32), np.uint16)
    a = np.ones((3, 32, 32), np.uint16)
    b[:, :8, :8] = 0; a[:, 16:24, 16:24] = 0
    bp = _write_tif(tmp_path / "b.tif", b, nodata=0)
    ap = _write_tif(tmp_path / "a.tif", a, nodata=0)
    report = validate_pair(bp, ap, expected_channels_each=3, require_georeferencing=True)
    check = next(c for c in report["checks"] if c["check"] == "common_valid_pixel_coverage")
    assert check["details"]["status"] == "PASS"
    assert check["details"]["common_valid_fraction_of_grid"] == pytest.approx(0.875)
    assert check["details"]["intersection_over_union_of_valid_masks"] == pytest.approx(0.875 / 1.0)


def test_scl_is_nearest_reprojected_to_target_image_grid(tmp_path):
    image = _write_tif(tmp_path / "image.tif", np.ones((13, 32, 32), np.uint16),
                       transform=from_origin(500000, 1000000, 10, 10))
    scl = np.full((16, 16), 4, dtype=np.uint8)
    scl[4:8, 4:8] = 9  # high-probability cloud class; categorical IDs must not be bilinear-resampled
    qa = _write_tif(tmp_path / "scl.tif", scl,
                     transform=from_origin(500000, 1000000, 20, 20), nodata=0)
    classes, good, report = read_scl_on_image_grid(image, qa)
    assert classes.shape == good.shape == (32, 32)
    assert report["reprojected"] is True
    assert report["method"] == "nearest_reproject_categorical_SCL_to_model_grid"
    assert 0.92 < good.mean() < 0.96
    # Preflight must not fail just because a large image was downsampled during basic inspection.
    assert _apply_scl(qa, image).get("status") == "PASS"


def test_feature_registration_reports_diagnostics_without_claiming_proof(tmp_path):
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(12)
    before = np.zeros((256, 256, 3), np.uint8)
    for _ in range(100):
        x, y = rng.integers(8, 248, size=2)
        radius = int(rng.integers(2, 8))
        color = int(rng.integers(40, 255))
        cv2.circle(before, (int(x), int(y)), radius, (color, color, color), -1)
    affine = np.float32([[1, 0, 6], [0, 1, -4]])
    after = cv2.warpAffine(before, affine, (256, 256))
    bp, ap = tmp_path / "before.png", tmp_path / "after.png"
    Image.fromarray(before).save(bp); Image.fromarray(after).save(ap)
    report = estimate_translation(bp, ap)
    assert report["status"] == "DIAGNOSTIC_ONLY"
    assert report["feature_registration"]["same_geographic_scene_proven"] is False
    assert "No resampling or warping was applied" in report["warning"]
    assert report["feature_registration"].get("correspondence_count", 0) >= 8
    shift = report.get("shift_sample_pixels")
    assert shift is not None and abs(abs(shift["x"]) - 6) < 1.0 and abs(abs(shift["y"]) - 4) < 1.0


def test_scl_class_2_is_masked_by_default_for_modern_pb04_products_and_can_be_kept_explicitly(tmp_path):
    image = _write_tif(tmp_path / "image.tif", np.ones((13, 16, 16), np.uint16),
                       transform=from_origin(500000, 1000000, 10, 10))
    scl = np.full((16, 16), 4, dtype=np.uint8)
    scl[5:10, 5:10] = 2  # modern Sentinel-2 L2A PB04+ class 2 is CAST_SHADOW
    qa = _write_tif(tmp_path / "scl.tif", scl,
                     transform=from_origin(500000, 1000000, 10, 10), nodata=0)
    classes, good, report = read_scl_on_image_grid(image, qa)
    assert not np.any(good[5:10, 5:10])
    assert 2 in report["excluded_scl_classes"]
    _legacy_classes, legacy_good, legacy_report = read_scl_on_image_grid(image, qa, class2_policy="keep")
    assert np.all(legacy_good[5:10, 5:10])
    assert 2 in legacy_report["advisory_scl_classes_kept_valid"]
