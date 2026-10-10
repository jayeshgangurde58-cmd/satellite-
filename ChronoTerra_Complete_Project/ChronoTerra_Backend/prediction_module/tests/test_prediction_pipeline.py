"""End-to-end software tests using synthetic fixtures only, never accuracy evidence."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform
import torch
from fastapi.testclient import TestClient

from api.app import app, OUTPUT_ROOT
from models.generic_unet import SegmentationUNet
from models.predict_chronoterra import predict_task
from models.siamese_unet import SiameseUNet
from quality_gate.prediction_preflight import run_prediction_preflight

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
TRANSFORM = from_origin(500000, 2100000, 10, 10)


def _write_raster(path: Path, data: np.ndarray, descriptions=None, *, dtype=None, nodata=None, tags=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        data = data[None]
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
                       count=data.shape[0], dtype=dtype or str(data.dtype), crs="EPSG:32643",
                       transform=TRANSFORM, nodata=nodata) as dst:
        dst.write(data)
        if descriptions:
            for i, d in enumerate(descriptions, 1): dst.set_band_description(i, d)
        if tags: dst.update_tags(**tags)
    return path


def _aoi_geojson_for_left_half(raster_path: Path, out_path: Path):
    with rasterio.open(raster_path) as ds:
        pts = [ds.transform * (0, 0), ds.transform * (ds.width // 2, 0),
               ds.transform * (ds.width // 2, ds.height), ds.transform * (0, ds.height)]
        xs, ys = warp_transform(ds.crs, "EPSG:4326", [p[0] for p in pts], [p[1] for p in pts])
    ring = [[float(x), float(y)] for x, y in zip(xs, ys)]
    ring.append(ring[0])
    out_path.write_text(json.dumps({"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[ring]}}),encoding="utf-8")
    return out_path


def test_spectral_spatial_siamese_is_symmetric_and_trainable():
    torch.manual_seed(4)
    model = SiameseUNet(13, base_channels=4).eval()
    before = torch.rand(1, 13, 32, 32)
    after = before.clone(); after[:, :, 8:20, 9:21] += 0.3
    with torch.no_grad():
        a = model(before, after)
        b = model(after, before)
    assert a.shape == (1, 1, 32, 32)
    assert torch.allclose(a, b, atol=1e-6), "symmetric change model should be date-order invariant"
    model.train(); loss = model(before, after).square().mean(); loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters())


def test_urban_change_prediction_writes_geospatial_artifacts_and_factor_report(tmp_path):
    rng = np.random.default_rng(123)
    before = rng.integers(700, 2400, size=(13, 32, 32), dtype=np.uint16)
    after = before.copy(); after[:, 12:22, 10:20] += 900
    b = _write_raster(tmp_path / "before.tif", before, BANDS, tags={"ACQUISITION_DATE": "2023-10-01"})
    a = _write_raster(tmp_path / "after.tif", after, BANDS, tags={"ACQUISITION_DATE": "2024-10-02"})
    model = SiameseUNet(13, base_channels=4)
    ck = tmp_path / "urban.pt"
    aoi_path = _aoi_geojson_for_left_half(b, tmp_path / "aoi.geojson")
    torch.save({"model_state_dict": model.state_dict(),
                "model_config": {"architecture": "SiameseUNet-SpectralSpatial-v2", "in_channels": 13, "base_channels": 4},
                "bands": BANDS, "scale_factor": 10000.0, "threshold": 0.5, "epoch": 1,
                "augmentation": {"radiometric_jitter_enabled": True}}, ck)
    out = predict_task("urban_change", ck, tmp_path / "out" / "prediction", before_path=b, after_path=a,
                       assume_band_order=False, aoi_geojson=aoi_path, tile_size=32, overlap=0, device_name="cpu")
    for key in ("mask_output", "scores_output", "uncertainty_output", "overlay_output", "report_output"):
        assert Path(out[key]).is_file(), key
    assert out["status"].startswith("INFERENCE_COMPLETED")
    assert out["scores_are_calibrated_probabilities"] is False
    assert out["preflight"]["factors"]["radiometric_differences"]["status"] == "DIAGNOSTIC_ONLY"
    assert out["preflight"]["factors"]["seasonality"]["status"] in {"PASS", "WARNING"}
    with rasterio.open(out["mask_output"]) as ds:
        assert ds.count == 1 and ds.crs.to_epsg() == 32643 and ds.nodata == 255
    report = json.loads(Path(out["report_output"]).read_text())
    assert report["checkpoint_sha256"] and report["preflight"]["checks"]["footprint_overlap_fraction_of_smaller"] >= 0.999


def test_flood_and_damage_prediction_routes_load_compatible_checkpoints(tmp_path):
    # These are pipeline tests on untrained/randomly initialized weights, not model-quality claims.
    rng = np.random.default_rng(44)
    sar = rng.normal(-18, 3, (2, 32, 32)).astype(np.float32)
    sar_path = _write_raster(tmp_path / "sar.tif", sar, ["VV", "VH"])
    flood_model = SegmentationUNet(2, 2, 4)
    flood_ck = tmp_path / "flood.pt"
    torch.save({"state_dict": flood_model.state_dict(), "config": {"task": "flood_extent", "in_channels": 2,
                "num_classes": 2, "base_channels": 4, "normalization": "sar_db", "decision_threshold": 0.5}, "epoch": 1}, flood_ck)
    flood = predict_task("flood_extent", flood_ck, tmp_path / "flood_out" / "pred", input_path=sar_path,
                         tile_size=32, overlap=0, device_name="cpu")
    assert Path(flood["mask_output"]).is_file() and Path(flood["report_output"]).is_file()
    assert flood["task"] == "flood_extent"

    before = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    after = before.copy(); after[8:20, 8:20] = 220
    bp, ap = tmp_path / "before.png", tmp_path / "after.png"
    Image.fromarray(before).save(bp); Image.fromarray(after).save(ap)
    damage_model = SegmentationUNet(6, 5, 4)
    damage_ck = tmp_path / "damage.pt"
    torch.save({"state_dict": damage_model.state_dict(), "config": {"task": "building_damage", "in_channels": 6,
                "num_classes": 5, "base_channels": 4, "normalization": "rgb255"}, "epoch": 1}, damage_ck)
    damage = predict_task("building_damage", damage_ck, tmp_path / "damage_out" / "pred", before_path=bp,
                          after_path=ap, tile_size=32, overlap=0, device_name="cpu")
    assert Path(damage["mask_output"]).is_file() and Path(damage["overlay_output"]).is_file()
    assert damage["task_output_semantics"]["4"] == "destroyed"


def test_preflight_rejects_wrong_urban_channel_count(tmp_path):
    x = np.ones((3, 32, 32), dtype=np.uint16)
    b = _write_raster(tmp_path / "b.tif", x, ["B02", "B03", "B04"])
    a = _write_raster(tmp_path / "a.tif", x, ["B02", "B03", "B04"])
    report = run_prediction_preflight("urban_change", before_path=b, after_path=a)
    assert report["status"] == "REJECT"
    assert any("13 channels" in e for e in report["hard_errors"])


def test_api_preflight_runs_without_trained_checkpoints(tmp_path):
    rng = np.random.default_rng(6)
    before = rng.integers(300, 2000, (13, 32, 32), dtype=np.uint16)
    after = before.copy()
    bpath = _write_raster(tmp_path / "api_before.tif", before, BANDS)
    apath = _write_raster(tmp_path / "api_after.tif", after, BANDS)
    client = TestClient(app)
    aoi_path = _aoi_geojson_for_left_half(bpath, tmp_path / "api_aoi.geojson")
    response = client.post("/api/preflight", data={"task": "urban_change", "assume_band_order": "false", "scl_class2_policy":"mask"}, files={
        "before_file": ("before.tif", bpath.read_bytes(), "image/tiff"),
        "after_file": ("after.tif", apath.read_bytes(), "image/tiff"),
        "aoi_geojson_file": ("aoi.geojson", aoi_path.read_bytes(), "application/geo+json"),
    })
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["task"] == "urban_change"
    assert data["status"] in {"PASS", "WARNING", "REJECT"}
    assert data["factors"]["aoi_coverage"]["per_asset"]["before"]["status"] == "PASS"
    assert data["report_url"].startswith("/outputs/preflight-")
    report_url = data["report_url"]
    assert client.get(report_url).status_code == 200
    # keep the test workspace clean
    session_name = report_url.split("/")[2]
    import shutil
    shutil.rmtree(OUTPUT_ROOT / session_name, ignore_errors=True)


def test_api_prediction_fails_closed_when_checkpoint_is_missing(tmp_path, monkeypatch):
    # A missing checkpoint must produce an explicit unavailable state, never random/fake outputs.
    from api.app import MODEL_CHECKPOINTS
    monkeypatch.setitem(MODEL_CHECKPOINTS, "urban_change", tmp_path / "no_real_checkpoint.pt")
    client = TestClient(app)
    status = client.get("/api/models/status")
    assert status.status_code == 200
    assert status.json()["models"]["urban_change"]["status"] == "MODEL_NOT_AVAILABLE"
    response = client.post("/api/predict", data={"task": "urban_change"})
    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "MODEL_NOT_AVAILABLE"
    assert not (OUTPUT_ROOT / "should_not_exist").exists()


def test_homepage_exposes_learned_prediction_controls():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Learned-model prediction" in response.text
    assert "mlPreflight" in response.text and "mlPredict" in response.text
    assert "/api/models/status" in response.text
