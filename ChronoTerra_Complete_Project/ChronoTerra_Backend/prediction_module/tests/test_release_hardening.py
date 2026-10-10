import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import app, OUTPUT_ROOT, _validate_pair_metadata_dates
from api.boundary_analysis import analyze_boundary_candidates, validate_boundary


def _no_absolute_path(value, root: str) -> bool:
    if isinstance(value, dict):
        return all(_no_absolute_path(v, root) for v in value.values())
    if isinstance(value, list):
        return all(_no_absolute_path(v, root) for v in value)
    return not (isinstance(value, str) and root in value)


def test_model_status_does_not_leak_absolute_checkpoint_paths(tmp_path, monkeypatch):
    from api import app as module
    monkeypatch.setitem(module.MODEL_CHECKPOINTS, "urban_change", tmp_path / "missing-private-checkpoint.pt")
    client = TestClient(app)
    response = client.get("/api/models/status")
    assert response.status_code == 200
    payload = response.json()
    assert _no_absolute_path(payload, str(tmp_path))
    assert "checkpoint_reference" in payload["models"]["urban_change"]
    assert "checkpoint" not in payload["models"]["urban_change"]


def test_pair_metadata_rejects_reversed_dates(tmp_path):
    path = tmp_path / "pair_metadata.json"
    path.write_text(json.dumps({
        "before": {"acquisition_date": "2026-10-09"},
        "after": {"acquisition_date": "2026-10-01"},
    }), encoding="utf-8")
    with pytest.raises(Exception, match="after-image acquisition date must be later"):
        _validate_pair_metadata_dates(path)


def test_preflight_response_does_not_leak_server_source_paths(tmp_path):
    # Use the existing test fixture convention directly to create a valid 13-band pair.
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
    before = tmp_path / "before.tif"
    after = tmp_path / "after.tif"
    rng = np.random.default_rng(9)
    for path in (before, after):
        with rasterio.open(path, "w", driver="GTiff", height=32, width=32, count=13,
                           dtype="uint16", crs="EPSG:32643", transform=from_origin(500000, 2100000, 10, 10)) as dst:
            dst.write(rng.integers(100, 2000, size=(13, 32, 32), dtype=np.uint16))
            for idx, band in enumerate(bands, start=1):
                dst.set_band_description(idx, band)
    client = TestClient(app)
    response = client.post("/api/preflight", data={"task": "urban_change", "assume_band_order": "true"}, files={
        "before_file": ("before.tif", before.read_bytes(), "image/tiff"),
        "after_file": ("after.tif", after.read_bytes(), "image/tiff"),
    })
    assert response.status_code == 200, response.text
    payload = response.json()
    assert _no_absolute_path(payload, str(OUTPUT_ROOT.parent.parent))
    report_url = payload.get("report_url")
    if report_url:
        session = report_url.split("/")[2]
        import shutil
        shutil.rmtree(OUTPUT_ROOT / session, ignore_errors=True)


def test_declared_oversized_request_is_rejected_before_parsing():
    from api.app import MAX_REQUEST_UPLOAD_BYTES
    client = TestClient(app)
    response = client.get("/api/health", headers={"Content-Length": str(MAX_REQUEST_UPLOAD_BYTES + 5 * 1024 * 1024)})
    assert response.status_code == 413


def test_reference_zone_computes_overlap_and_near_candidates(tmp_path):
    boundary_path = tmp_path / "reference.geojson"
    boundary = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {"zone": "test"}, "geometry": {
            "type": "Polygon", "coordinates": [[[0, 0], [0.001, 0], [0.001, 0.001], [0, 0.001], [0, 0]]]
        }}],
    }
    boundary_path.write_text(json.dumps(boundary), encoding="utf-8")
    qa = validate_boundary(boundary_path)
    assert qa["status"] == "PASS"
    # Candidate 1 overlaps. Candidate 2 is about 2.2 m from the east edge.
    regions_path = tmp_path / "regions.geojson"
    collection = {
        "type": "FeatureCollection", "coordinate_space": "EPSG:4326", "features": [
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[0.0005, 0.0002], [0.0015, 0.0002], [0.0015, 0.0008], [0.0005, 0.0008], [0.0005, 0.0002]]]}, "properties": {"id": "R0001", "class_name": "candidate_change"}},
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[0.00102, 0.0002], [0.0012, 0.0002], [0.0012, 0.0008], [0.00102, 0.0008], [0.00102, 0.0002]]]}, "properties": {"id": "R0002", "class_name": "candidate_change"}},
        ]
    }
    regions_path.write_text(json.dumps(collection), encoding="utf-8")
    overlay_path = tmp_path / "overlay.geojson"
    summary = analyze_boundary_candidates(regions_path, boundary_path, overlay_path, source_name="test layer", proximity_m=10)
    assert summary["boundary_overlap_candidate_count"] == 1
    assert summary["near_boundary_candidate_count"] == 1
    updated = json.loads(regions_path.read_text(encoding="utf-8"))
    relations = {f["properties"]["id"]: f["properties"]["boundary_relation"] for f in updated["features"]}
    assert relations["R0001"] == "BOUNDARY_OVERLAP_CANDIDATE"
    assert relations["R0002"] == "NEAR_BOUNDARY_CANDIDATE"
    assert updated["features"][0]["properties"]["boundary_authority_status"] == "USER_SUPPLIED_NOT_INDEPENDENTLY_VERIFIED"
    assert overlay_path.exists()


def test_model_inference_gate_requires_hash_matched_held_out_evidence_and_review(tmp_path):
    import hashlib
    from api.app import _model_evidence
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fixture weights are only for gate testing; never used for inference")
    source_report = tmp_path / "deployment_source_report.json"
    per_scene = {f"scene_{i}": {"f1": 0.79} for i in range(12)}
    source_payload = {
        "dataset": "OSCD official test split", "model_selection_used_test_data": False,
        "per_scene": per_scene,
        "micro": {"precision": 0.81, "recall": 0.77, "f1": 0.79, "iou": 0.65, "false_positive_rate": 0.04},
    }
    source_report.write_text(json.dumps(source_payload), encoding="utf-8")
    source_hash = hashlib.sha256(source_report.read_bytes()).hexdigest()
    metrics_path = tmp_path / "held_out_metrics.json"
    metrics_path.write_text(json.dumps({
        "task": "urban_change", "dataset_name": "OSCD official test split",
        "split_type": "held_out_by_scene", "held_out_scene_count": 12,
        "metrics": {"precision": 0.81, "recall": 0.77, "f1": 0.79, "iou": 0.65, "false_positive_rate": 0.04},
        "source_report_file": source_report.name, "source_report_sha256": source_hash,
    }), encoding="utf-8")
    manifest_path = tmp_path / "deployment_manifest.json"
    manifest = {
        "task": "urban_change",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "approved_for_inference": True,
        "evaluation": {"metrics_file": metrics_path.name, "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
                       "source_report_file": source_report.name, "source_report_sha256": source_hash,
                       "split_type": "held_out_by_scene", "held_out_scene_count": 12,
                       "dataset_name": "OSCD official test split", "evaluation_date": "2026-10-01",
                       "deployment_thresholds": {"minimum_precision": 0.7, "minimum_recall": 0.7, "minimum_f1": 0.7,
                                                  "minimum_iou": 0.5, "maximum_false_positive_rate": 0.1}},
        "review": {"status": "APPROVED", "reviewer": "QA reviewer", "reviewed_at": "2026-10-02"},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = _model_evidence(checkpoint, "urban_change")
    assert report["approved"] is True
    assert report["status"] == "APPROVED_FOR_INFERENCE"
    assert report["evaluation_summary"]["held_out_scene_count"] == 12
    manifest["checkpoint_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = _model_evidence(checkpoint, "urban_change")
    assert report["approved"] is False
    assert report["checkpoint_hash_matches"] is False
    manifest["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metrics_path.write_text(json.dumps({
        "task": "urban_change", "dataset_name": "OSCD official test split", "split_type": "held_out_by_scene",
        "held_out_scene_count": 12,
        "metrics": {"precision": 0.81, "recall": 0.77, "f1": 0.01, "iou": 0.65, "false_positive_rate": 0.04},
        "source_report_file": source_report.name, "source_report_sha256": source_hash,
    }), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = _model_evidence(checkpoint, "urban_change")
    assert report["approved"] is False
    assert report["metrics_hash_matches"] is False
    manifest["evaluation"]["metrics_sha256"] = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = _model_evidence(checkpoint, "urban_change")
    assert report["approved"] is False
    assert report["thresholds_satisfied"] is False
    source_report.write_text(json.dumps({**source_payload, "model_selection_used_test_data": True}), encoding="utf-8")
    manifest["evaluation"]["source_report_sha256"] = hashlib.sha256(source_report.read_bytes()).hexdigest()
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_payload["source_report_sha256"] = manifest["evaluation"]["source_report_sha256"]
    metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")
    manifest["evaluation"]["metrics_sha256"] = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = _model_evidence(checkpoint, "urban_change")
    assert report["approved"] is False
    assert report["source_report_hash_matches"] is False or report["evaluation_evidence_valid"] is False


def test_artifact_route_does_not_serve_internal_report_or_uploaded_source(tmp_path):
    import shutil
    from api.app import SESSION_ID_RE
    session_id = "a" * 32
    session_dir = OUTPUT_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "before.tif").write_bytes(b"private-source")
    (session_dir / "preflight_report.json").write_text('{"private":true}', encoding="utf-8")
    client = TestClient(app)
    assert client.get(f"/outputs/{session_id}/before.tif").status_code == 404
    assert client.get(f"/outputs/{session_id}/preflight_report.json").status_code == 404
    shutil.rmtree(session_dir, ignore_errors=True)


def test_expired_artifacts_are_rejected_even_before_periodic_cleanup(tmp_path):
    import os
    import shutil
    import time
    from api.app import SESSION_RETENTION_HOURS, _artifact_url
    session_id = "b" * 32
    session_dir = OUTPUT_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    artifact = session_dir / "before_preview.png"
    artifact.write_bytes(b"expired preview")
    signed_url = _artifact_url(artifact)
    expired = time.time() - (SESSION_RETENTION_HOURS + 2) * 3600
    os.utime(session_dir, (expired, expired))
    client = TestClient(app)
    response = client.get(signed_url)
    assert response.status_code == 404
    shutil.rmtree(session_dir, ignore_errors=True)


def test_artifact_routes_require_valid_signed_expiring_links(tmp_path):
    from urllib.parse import parse_qs, urlsplit
    import shutil
    import time
    from api.app import _artifact_url, _artifact_signature
    session_id = "c" * 32
    session_dir = OUTPUT_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    artifact = session_dir / "before_preview.png"
    artifact.write_bytes(b"preview bytes")
    client = TestClient(app)
    signed_url = _artifact_url(artifact)
    assert client.get(signed_url).status_code == 200
    parsed = urlsplit(signed_url)
    assert client.get(parsed.path).status_code == 404
    query = parse_qs(parsed.query)
    tampered = f"{parsed.path}?expires={query['expires'][0]}&sig={'0' * 64}"
    assert client.get(tampered).status_code == 404
    expired_at = int(time.time()) - 2
    expired_sig = _artifact_signature(session_id, artifact.name, expired_at)
    expired_url = f"{parsed.path}?expires={expired_at}&sig={expired_sig}"
    assert client.get(expired_url).status_code == 404
    shutil.rmtree(session_dir, ignore_errors=True)


def test_request_body_middleware_enforces_limit_without_content_length():
    import asyncio
    from api.request_limits import RequestBodyLimitMiddleware

    async def fake_app(scope, receive, send):
        while True:
            message = await receive()
            if message["type"] == "http.request" and not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    app_with_limit = RequestBodyLimitMiddleware(fake_app, max_body_bytes=8)
    messages = iter([
        {"type": "http.request", "body": b"12345", "more_body": True},
        {"type": "http.request", "body": b"67890", "more_body": False},
    ])
    sent = []

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    asyncio.run(app_with_limit({"type": "http", "method": "POST", "path": "/upload", "headers": []}, receive, send))
    assert sent[0]["status"] == 413
    assert b"configured aggregate upload limit" in sent[1]["body"]


def test_structured_sidecar_limit_is_enforced_during_streaming(tmp_path):
    import asyncio
    import io
    from starlette.datastructures import UploadFile
    from fastapi import HTTPException
    from api.app import _save_upload, UploadBudget

    upload = UploadFile(filename="large.json", file=io.BytesIO(b"12345"))
    target = tmp_path / "large.json"
    with pytest.raises(HTTPException) as exc:
        asyncio.run(_save_upload(upload, target, allowed={".json"}, budget=UploadBudget(),
                                 max_bytes=4, size_label="Test metadata JSON"))
    assert exc.value.status_code == 413
    assert "Test metadata JSON" in str(exc.value.detail)
    assert not target.exists()


def test_single_invalid_acquisition_date_is_rejected(tmp_path):
    path = tmp_path / "single-date.json"
    path.write_text(json.dumps({"before": {"acquisition_date": "not-a-date"}, "after": {}}), encoding="utf-8")
    with pytest.raises(Exception, match="valid ISO dates"):
        _validate_pair_metadata_dates(path)


def test_reference_boundary_future_effective_date_blocks_spatial_classification(tmp_path):
    from api.app import _boundary_temporal_compatibility
    metadata = tmp_path / "pair.json"
    metadata.write_text(json.dumps({
        "before": {"acquisition_date": "2024-01-01"},
        "after": {"acquisition_date": "2024-12-01"},
    }), encoding="utf-8")
    result = _boundary_temporal_compatibility("2025-01-01", metadata)
    assert result["status"] == "TEMPORAL_MISMATCH"
    assert "Boundary comparison was skipped" in result["message"]


def test_reference_boundary_effective_date_does_not_claim_full_temporal_validity(tmp_path):
    from api.app import _boundary_temporal_compatibility
    metadata = tmp_path / "pair.json"
    metadata.write_text(json.dumps({
        "before": {"acquisition_date": "2024-01-01"},
        "after": {"acquisition_date": "2024-12-01"},
    }), encoding="utf-8")
    result = _boundary_temporal_compatibility("2024-05-01", metadata)
    assert result["status"] == "NOT_VERIFIED"
    assert "does not establish" in result["message"]


def test_signed_artifact_capability_works_through_mounted_root_api():
    """Check the real public URL shape, including the parent FastAPI /ml mount."""
    import shutil
    import uuid
    from fastapi.testclient import TestClient
    from api.app import _artifact_url
    from app.main import app as root_app

    session_id = uuid.uuid4().hex
    session_dir = OUTPUT_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=False)
    artifact = session_dir / "before_preview.png"
    artifact.write_bytes(b"mounted-signed-artifact")
    try:
        signed_url = _artifact_url(artifact)
        client = TestClient(root_app)
        response = client.get(f"/ml{signed_url}")
        assert response.status_code == 200
        assert response.content == b"mounted-signed-artifact"
        assert response.headers.get("cache-control") == "private, no-store"
        assert client.get(f"/ml{signed_url.split('?')[0]}").status_code == 404
    finally:
        shutil.rmtree(session_dir, ignore_errors=True)
