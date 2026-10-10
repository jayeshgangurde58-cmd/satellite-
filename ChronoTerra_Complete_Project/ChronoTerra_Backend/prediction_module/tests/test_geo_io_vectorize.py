import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from pipeline.geo_io import load_geotiff_pair
from pipeline.vectorize import vectorize_change_mask


def _write(path, array, transform=from_origin(500000, 2100000, 10, 10), crs="EPSG:32643"):
    profile = {"driver": "GTiff", "height": array.shape[1], "width": array.shape[2],
               "count": array.shape[0], "dtype": "float32", "transform": transform, "crs": crs}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32))


def test_geo_io_loads_aligned_pair(tmp_path):
    arr = np.ones((2, 12, 12), dtype=np.float32)
    b, a = tmp_path / "before.tif", tmp_path / "after.tif"
    _write(b, arr); _write(a, arr + 1)
    pair = load_geotiff_pair(b, a)
    assert pair.before.shape == pair.after.shape == (2, 12, 12)
    assert pair.valid_mask.all()


def test_geo_io_rejects_grid_mismatch(tmp_path):
    arr = np.ones((2, 12, 12), dtype=np.float32)
    b, a = tmp_path / "before.tif", tmp_path / "after.tif"
    _write(b, arr); _write(a, arr, transform=from_origin(500010, 2100000, 10, 10))
    with pytest.raises(ValueError, match="transforms differ"):
        load_geotiff_pair(b, a)


def test_vectorization_exports_metric_area_and_wgs84_centroid():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:6] = True
    magnitude = np.zeros((10, 10), dtype=np.float32)
    magnitude[mask] = 0.8
    gdf = vectorize_change_mask(mask, magnitude, from_origin(500000, 2100000, 10, 10), "EPSG:32643")
    assert len(gdf) == 1
    assert gdf.crs.to_epsg() == 4326
    assert gdf.iloc[0]["area_m2"] == pytest.approx(900, rel=0.02)
    assert -180 <= gdf.iloc[0]["centroid_lon"] <= 180
    assert -90 <= gdf.iloc[0]["centroid_lat"] <= 90
    assert gdf.iloc[0]["severity_tier"] in {"Low", "Medium", "High", "Critical"}
