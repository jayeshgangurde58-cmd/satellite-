from pathlib import Path
import shutil

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from api.app import app, OUTPUT_ROOT


def _bytes_geotiff(path: Path, values: np.ndarray) -> bytes:
    with rasterio.open(
        path, "w", driver="GTiff", height=values.shape[1], width=values.shape[2],
        count=values.shape[0], dtype="float32", crs="EPSG:32643",
        transform=from_origin(500000, 2100000, 10, 10),
    ) as dst:
        dst.write(values)
    return path.read_bytes()


def test_api_end_to_end_synthetic_pair(tmp_path):
    rng = np.random.default_rng(21)
    before = rng.normal(100, 2, (3, 64, 64)).astype(np.float32)
    after = before.copy()
    after[:, 20:40, 24:44] += 50
    b = _bytes_geotiff(tmp_path / "before.tif", before)
    a = _bytes_geotiff(tmp_path / "after.tif", after)
    client = TestClient(app)
    response = client.post(
        "/api/analyze",
        files={
            "before_file": ("before.tif", b, "image/tiff"),
            "after_file": ("after.tif", a, "image/tiff"),
        },
        data={"histogram_match": "false", "min_object_size": "4", "opening_radius": "1"},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["metrics"]["valid_pixels"] == 64 * 64
    assert result["metrics"]["polygon_count"] >= 1
    assert result["bounds"] and len(result["bounds"]) == 2
    urls = [result["before_image_url"], result["after_image_url"], result["geojson_url"], result["mask_url"]]
    if "gpkg_url" in result:
        urls.append(result["gpkg_url"])
    for url in urls:
        downloaded = client.get(url)
        assert downloaded.status_code == 200, url
    # Raw uploaded imagery and internal session files must not be served publicly.
    for private_name in ("before.tif", "after.tif"):
        private_response = client.get(f"/outputs/{result['session_id']}/{private_name}")
        assert private_response.status_code == 404
    shutil.rmtree(OUTPUT_ROOT / result["session_id"], ignore_errors=True)
