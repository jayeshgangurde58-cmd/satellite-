from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import app  # noqa: E402

client = TestClient(app)


def png_bytes(size=(64, 48), color=(90, 130, 150)) -> bytes:
    image = Image.new("RGB", size, color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def tiff_bytes(*, crs="EPSG:32633", transform=None, width=32, height=24, bands=3, nodata=None, invalid=False) -> bytes:
    transform = transform or from_origin(500000, 5100000, 10, 10)
    arr = np.full((bands, height, width), 100, dtype=np.uint16)
    if invalid:
        arr[:] = 0
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": bands,
        "dtype": "uint16",
        "transform": transform,
    }
    if crs:
        profile["crs"] = crs
    if nodata is not None:
        profile["nodata"] = nodata
        arr[:] = nodata
    with MemoryFile() as memfile:
        with memfile.open(**profile) as ds:
            ds.write(arr)
        return memfile.read()


def upload_pair(before: bytes, after: bytes, *, before_name="before.png", after_name="after.png", **data):
    return client.post(
        "/api/validate-pair",
        files={
            "before_image": (before_name, before, "application/octet-stream"),
            "after_image": (after_name, after, "application/octet-stream"),
        },
        data=data,
    )


def test_health_is_ok_and_ml_api_is_mounted_without_claiming_weights():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["prediction_service"] in {"mounted", "unavailable"}
    model_status = client.get("/ml/api/models/status")
    assert model_status.status_code == 200
    assert set(model_status.json()["models"]) == {"urban_change", "flood_extent", "building_damage"}


def test_inspect_png_reports_actual_properties_and_missing_georeferencing():
    response = client.post("/api/inspect-image", files={"file": ("image.png", png_bytes(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["dimensions"] == {"width": 64, "height": 48, "pixels": 3072}
    assert body["channels"] == 3
    assert body["metadata"]["crs"] is None
    assert body["status"] == "WARNING"
    assert any(c["name"] == "georeferencing" and c["status"] == "NOT_VERIFIED" for c in body["checks"])


def test_corrupt_png_is_rejected():
    response = client.post("/api/inspect-image", files={"file": ("bad.png", b"\x89PNG\r\n\x1a\nnot-a-real-image", "image/png")})
    assert response.status_code == 422


def test_unsupported_txt_is_rejected():
    response = client.post("/api/inspect-image", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 422


def test_jpeg_png_pair_does_not_claim_geographic_overlap():
    response = upload_pair(png_bytes(), png_bytes(color=(100, 140, 160)), same_area_confirmed="true", analysis_mode="urban")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "WARNING"
    assert body["pair"]["geographic_overlap"]["status"] == "NOT_VERIFIED"
    assert body["pair"]["geographic_overlap"]["footprint_intersection_over_union"] is None
    assert body["user_metadata"]["same_area_confirmed_by_user"] is True
    assert body["next_step"]["prediction_status"] == "NOT_CONFIGURED"


def test_matching_geotiff_grid_reports_geographic_overlap_but_not_registration():
    before = tiff_bytes()
    after = tiff_bytes()
    response = upload_pair(before, after, before_name="before.tif", after_name="after.tif", analysis_mode="urban")
    assert response.status_code == 200
    body = response.json()
    assert body["pair"]["same_crs"] is True
    assert body["pair"]["same_grid"] is True
    assert body["pair"]["geographic_overlap"]["footprint_intersection_over_union"] == pytest.approx(1.0)
    assert body["pair"]["common_valid_pixel_coverage_fraction"] == pytest.approx(1.0)
    assert body["pair"]["registration_status"] == "NOT_VERIFIED"


def test_disjoint_geotiff_footprints_are_rejected():
    before = tiff_bytes(transform=from_origin(100, 100, 1, 1), width=10, height=10)
    after = tiff_bytes(transform=from_origin(1000, 1000, 1, 1), width=10, height=10)
    response = upload_pair(before, after, before_name="before.tif", after_name="after.tif")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "REJECT"
    assert body["pair"]["geographic_overlap"]["status"] == "REJECT"


def test_different_geotiff_grids_warn_and_do_not_auto_warp():
    before = tiff_bytes(transform=from_origin(500000, 5100000, 10, 10))
    after = tiff_bytes(transform=from_origin(500005, 5100000, 10, 10))
    response = upload_pair(before, after, before_name="before.tif", after_name="after.tif")
    assert response.status_code == 200
    body = response.json()
    assert body["pair"]["same_grid"] is False
    assert any(c["name"] == "grid_compatibility" and c["status"] == "WARNING" for c in body["pair"]["checks"])
    assert "does not reproject or warp" in next(c["message"] for c in body["pair"]["checks"] if c["name"] == "grid_compatibility")


def test_pair_rejects_invalid_before_file():
    response = upload_pair(b"bad", png_bytes())
    assert response.status_code == 422


def test_date_order_rejects_when_after_precedes_before():
    response = upload_pair(png_bytes(), png_bytes(), before_date="2025-10-01", after_date="2024-10-01")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "REJECT"
    assert any(c["name"] == "date_order" and c["status"] == "REJECT" for c in body["pair"]["checks"])


def test_all_nodata_geotiff_is_rejected_by_individual_inspection():
    data = tiff_bytes(nodata=9999)
    response = client.post("/api/inspect-image", files={"file": ("empty.tif", data, "image/tiff")})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "REJECT"
    assert any(c["name"] == "valid_pixels" and c["status"] == "REJECT" for c in body["checks"])


def test_mismatched_band_count_is_reported_and_not_claimed_compatible():
    before = tiff_bytes(bands=3)
    after = tiff_bytes(bands=4)
    response = upload_pair(before, after, before_name="before.tif", after_name="after.tif")
    assert response.status_code == 200
    body = response.json()
    assert body["pair"]["same_band_count"] is False
    assert any(c["name"] == "band_count_compatibility" and c["status"] == "WARNING" for c in body["pair"]["checks"])


def test_extremely_narrow_png_does_not_crash_visual_sampling():
    image = Image.new("RGB", (2, 2000), (80, 120, 160))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    response = client.post("/api/inspect-image", files={"file": ("narrow.png", buf.getvalue(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["dimensions"]["width"] == 2
    assert body["dimensions"]["height"] == 2000
    assert body["visual_quality"]["gradient_energy_heuristic"] is None


def test_fully_transparent_png_is_not_reported_as_visual_quality_pass():
    image = Image.new("RGBA", (32, 32), (255, 255, 255, 0))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    response = client.post("/api/inspect-image", files={"file": ("transparent.png", buf.getvalue(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "REJECT"
    assert any(c["name"] == "visual_quality" and c["status"] == "REJECT" for c in body["checks"])


def test_legacy_jobs_endpoint_never_returns_mock_regions():
    response = client.post('/api/jobs')
    assert response.status_code == 410
    assert response.json()['detail']['status'] == 'LEGACY_ENDPOINT_RETIRED'
    assert client.get('/api/jobs/legacy-id').status_code == 410
    assert client.get('/api/jobs/legacy-id/artifacts/anything.png').status_code == 410


def test_prediction_endpoint_refuses_missing_checkpoint_instead_of_fabricating_output():
    response = client.post('/ml/api/predict', data={'task': 'urban_change'})
    assert response.status_code == 503
    body = response.json()['detail']
    assert body['status'] == 'MODEL_NOT_AVAILABLE'
    assert 'will not fabricate' in body['message']


def test_reversed_or_equal_acquisition_dates_are_rejected_by_pair_gate():
    before = png_bytes()
    after = png_bytes(color=(120, 150, 170))
    reversed_response = upload_pair(before, after, before_date="2026-10-09", after_date="2026-10-01", same_area_confirmed="true", analysis_mode="urban")
    assert reversed_response.status_code == 200
    assert reversed_response.json()["status"] == "REJECT"
    assert any(check["name"] == "date_order" and check["status"] == "REJECT" for check in reversed_response.json()["pair"]["checks"])
    same_day_response = upload_pair(before, after, before_date="2026-10-09", after_date="2026-10-09", same_area_confirmed="true", analysis_mode="urban")
    assert same_day_response.status_code == 200
    assert same_day_response.json()["status"] == "REJECT"
