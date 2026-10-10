from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin
from api.app import _vectorize_prediction_mask


def test_image_space_prediction_mask_vectorizes_real_components(tmp_path: Path):
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[2:6, 3:8] = 1
    mask[12:17, 18:25] = 1
    path = tmp_path / 'mask.png'
    Image.fromarray(mask, mode='L').save(path)
    out = tmp_path / 'regions.json'
    result = _vectorize_prediction_mask(path, out, 'urban_change')
    payload = json.loads(out.read_text())
    assert result['region_count'] == 2
    assert result['coordinate_space'] == 'image_pixels'
    assert payload['coordinate_space'] == 'image_pixels_not_geographic'
    assert all(f['properties']['class_name'] == 'candidate_change' for f in payload['features'])


def test_georeferenced_prediction_mask_exports_wgs84_polygons(tmp_path: Path):
    path = tmp_path / 'mask.tif'
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[3:8, 4:9] = 1
    with rasterio.open(path, 'w', driver='GTiff', width=12, height=12, count=1, dtype='uint8', crs='EPSG:32633', transform=from_origin(500000, 5100000, 10, 10), nodata=255) as dst:
        dst.write(mask, 1)
    out = tmp_path / 'regions.json'
    result = _vectorize_prediction_mask(path, out, 'urban_change')
    payload = json.loads(out.read_text())
    assert result['region_count'] == 1
    assert result['coordinate_space'] == 'geographic'
    assert payload['coordinate_space'] == 'EPSG:4326'
    assert result['regions'][0]['area_m2'] > 0
    assert -180 <= result['regions'][0]['centroid_lng'] <= 180
    assert -90 <= result['regions'][0]['centroid_lat'] <= 90
