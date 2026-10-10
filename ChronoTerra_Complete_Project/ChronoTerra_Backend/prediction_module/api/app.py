"""Local upload-and-analyze demo for GEOAI-04. CPU-only; no database required."""
from __future__ import annotations

import json
import os
import uuid
import time
import threading
import logging
import hashlib
import hmac
import secrets
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
import re

import numpy as np
import rasterio
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from PIL import Image
from rasterio.warp import transform_bounds, transform_geom
from rasterio.transform import from_origin
from rasterio.features import shapes
from pipeline.geo_io import load_geotiff_pair
from pipeline.change import run_baseline
from pipeline.vectorize import vectorize_change_mask, write_vectors
from shapely.geometry import shape, mapping
from pyproj import Geod
CLASS_NAMES = {"urban_change", "flood_extent", "building_damage"}
from api.request_limits import RequestBodyLimitMiddleware

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
OUTPUT_ROOT = ROOT / "outputs" / "sessions"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ChronoTerra Change Intelligence", version="0.5.0")
logger = logging.getLogger("chronoterra.api")


@app.middleware("http")
async def bound_declared_request_size(request: Request, call_next):
    """Reject oversized declared bodies before FastAPI parses multipart uploads.

    Per-file and aggregate streaming limits still apply when Content-Length is absent.
    This header guard is defence-in-depth, not a replacement for reverse-proxy limits.
    """
    raw_length = request.headers.get("content-length")
    if raw_length:
        try:
            declared_length = int(raw_length)
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
        if declared_length < 0:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
        if declared_length > MAX_REQUEST_UPLOAD_BYTES + 2 * 1024 * 1024:
            return JSONResponse(status_code=413, content={"detail": "Request exceeds the configured aggregate upload limit."})
    return await call_next(request)

# Only derived artifacts are retrievable. Raw inputs and internal reports remain in
# session directories but are never exposed through a blanket StaticFiles mount.
PUBLIC_ARTIFACT_NAMES = {
    "preflight_report_public.json", "prediction_report_public.json",
    "prediction_mask_preview.png", "prediction_regions.json",
    "before_preview.png", "after_preview.png", "input_preview.png",
    "prediction_mask.tif", "prediction_mask.png",
    "prediction_scores.tif", "prediction_scores.npz",
    "prediction_uncertainty.tif", "prediction_uncertainty.png",
    "prediction_overlay.png", "before.png", "after.png",
    "change_polygons.geojson", "change_polygons.gpkg", "change_mask.tif",
    "reference_boundary_overlay.geojson",
}
SESSION_ID_RE = re.compile(r"^(?:[a-f0-9]{32}|preflight-[a-f0-9]{32})$")


def _artifact_signature(session_id: str, filename: str, expires: int) -> str:
    payload = f"{session_id}\n{filename}\n{expires}".encode("utf-8")
    return hmac.new(ARTIFACT_SIGNING_KEY, payload, hashlib.sha256).hexdigest()


def _artifact_url(path: Path) -> str:
    """Create an allowlisted, signed artifact URL that cannot outlive its session."""
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("Artifact path is outside the private output root.") from exc
    parts = relative.parts
    if len(parts) != 2 or not SESSION_ID_RE.fullmatch(parts[0]) or parts[1] not in PUBLIC_ARTIFACT_NAMES:
        raise ValueError("Only allowlisted derived artifacts can have public URLs.")
    if not resolved.is_file():
        raise FileNotFoundError("Requested output artifact does not exist.")
    now = int(time.time())
    session_dir = resolved.parent
    session_expiry = int(session_dir.stat().st_mtime + SESSION_RETENTION_HOURS * 3600)
    expires = min(now + SESSION_RETENTION_HOURS * 3600, session_expiry)
    if expires <= now:
        raise ValueError("Artifact session has already expired.")
    session_id, filename = parts
    signature = _artifact_signature(session_id, filename, expires)
    return f"/outputs/{session_id}/{filename}?expires={expires}&sig={signature}"


def _public_report(report: dict) -> dict:
    """Return data safe for a browser download, omitting local paths and internals."""
    forbidden_keys = {
        "path", "filepath", "file_path", "absolute_path", "checkpoint", "checkpoint_path",
        "input_paths", "mask_output", "scores_output", "uncertainty_output", "overlay_output",
        "report_output", "aoi_geojson", "pair_metadata_json", "regions_path", "source_path",
    }
    def clean(value):
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                normalized = str(key).lower()
                if normalized in forbidden_keys or normalized.endswith("_path") or "absolute_path" in normalized:
                    continue
                out[key] = clean(item)
            return out
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, str):
            return _redact_public_text(value)
        return value
    return clean(report)


@app.get("/outputs/{artifact_path:path}")
def serve_public_artifact(artifact_path: str, expires: int | None = None, sig: str | None = None) -> FileResponse:
    """Serve a derived artifact only with a valid, unexpired HMAC capability URL."""
    _cleanup_expired_sessions()
    rel = PurePosixPath(artifact_path)
    if len(rel.parts) != 2 or rel.is_absolute() or any(part in {".", ".."} for part in rel.parts):
        raise HTTPException(404, "Artifact not found")
    session_id, filename = rel.parts
    if not SESSION_ID_RE.fullmatch(session_id) or filename not in PUBLIC_ARTIFACT_NAMES:
        raise HTTPException(404, "Artifact not found")
    now = int(time.time())
    if expires is None or sig is None or expires < now or expires > now + SESSION_RETENTION_HOURS * 3600 + 60:
        raise HTTPException(404, "Artifact not found or expired")
    expected_signature = _artifact_signature(session_id, filename, expires)
    if not hmac.compare_digest(sig, expected_signature):
        raise HTTPException(404, "Artifact not found")
    session_dir = (OUTPUT_ROOT / session_id).resolve()
    try:
        session_dir.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(404, "Artifact not found") from exc
    try:
        cutoff = datetime.now(timezone.utc).timestamp() - SESSION_RETENTION_HOURS * 3600
        if session_dir.stat().st_mtime < cutoff:
            import shutil
            shutil.rmtree(session_dir, ignore_errors=True)
            raise HTTPException(404, "Artifact expired")
    except FileNotFoundError as exc:
        raise HTTPException(404, "Artifact not found") from exc
    target = session_dir / filename
    try:
        target.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(404, "Artifact not found") from exc
    if not target.is_file():
        raise HTTPException(404, "Artifact not found")
    headers = {"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"}
    # Browser previews need inline rendering; binary/data artifacts download as attachments.
    if target.suffix.lower() == ".png":
        return FileResponse(target, headers=headers)
    return FileResponse(target, filename=target.name, headers=headers)

# Checkpoints are resolved from environment variables or conventional training output folders.
# The upload request never chooses an arbitrary filesystem path.
MODEL_CHECKPOINTS = {
    "urban_change": Path(os.environ.get("CHRONOTERRA_URBAN_CHECKPOINT", ROOT / "outputs/models/oscd_siamese_v2/best.pt")),
    "flood_extent": Path(os.environ.get("CHRONOTERRA_FLOOD_CHECKPOINT", ROOT / "outputs/models/sen1floods11_sar_unet_v1/best.pt")),
    "building_damage": Path(os.environ.get("CHRONOTERRA_DAMAGE_CHECKPOINT", ROOT / "outputs/models/xbd_damage_unet_v1/best.pt")),
}
MAX_UPLOAD_MB = int(os.environ.get("CHRONOTERRA_MAX_UPLOAD_MB", "150"))
MAX_REQUEST_UPLOAD_MB = int(os.environ.get("CHRONOTERRA_MAX_REQUEST_UPLOAD_MB", "256"))
if MAX_UPLOAD_MB < 1 or MAX_REQUEST_UPLOAD_MB < MAX_UPLOAD_MB:
    raise RuntimeError("CHRONOTERRA_MAX_UPLOAD_MB must be positive and CHRONOTERRA_MAX_REQUEST_UPLOAD_MB must be at least as large")
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_REQUEST_UPLOAD_BYTES = MAX_REQUEST_UPLOAD_MB * 1024 * 1024
# Multipart boundaries/headers add small overhead beyond file bytes. This guard applies
# before form parsing and also covers chunked requests with no Content-Length header.
app.add_middleware(RequestBodyLimitMiddleware, max_body_bytes=MAX_REQUEST_UPLOAD_BYTES + 2 * 1024 * 1024)
UPLOAD_CHUNK_BYTES = 1024 * 1024
SESSION_RETENTION_HOURS = max(1, int(os.environ.get("CHRONOTERRA_SESSION_RETENTION_HOURS", "24")))
ARTIFACT_SIGNING_SECRET = os.environ.get("CHRONOTERRA_ARTIFACT_SIGNING_SECRET", "").strip()
if ARTIFACT_SIGNING_SECRET and len(ARTIFACT_SIGNING_SECRET.encode("utf-8")) < 32:
    raise RuntimeError("CHRONOTERRA_ARTIFACT_SIGNING_SECRET must contain at least 32 UTF-8 bytes")
ARTIFACT_SIGNING_KEY = ARTIFACT_SIGNING_SECRET.encode("utf-8") if ARTIFACT_SIGNING_SECRET else secrets.token_bytes(32)
CLEANUP_CHECK_INTERVAL_SECONDS = 300
_last_cleanup_monotonic = 0.0
_cleanup_lock = threading.Lock()
# The current models retain whole-scene tensors. Only one inference may run at a time;
# a second request is rejected with 429 rather than multiplying memory usage.
INFERENCE_GATE = threading.BoundedSemaphore(1)
MAX_AXIS_PIXELS = 12000
# This implementation currently holds several whole-raster arrays in memory. Keep these
# budgets conservative until true windowed input/output inference is implemented.
TASK_MAX_PIXELS = {"urban_change": 750_000, "flood_extent": 2_000_000, "building_damage": 1_500_000}
ALLOWED_UPLOAD_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
MODEL_REFERENCES = {
    "urban_change": "outputs/models/oscd_siamese_v2/best.pt",
    "flood_extent": "outputs/models/sen1floods11_sar_unet_v1/best.pt",
    "building_damage": "outputs/models/xbd_damage_unet_v1/best.pt",
}

MODEL_REFERENCE_ENV = {
    "urban_change": "CHRONOTERRA_URBAN_CHECKPOINT",
    "flood_extent": "CHRONOTERRA_FLOOD_CHECKPOINT",
    "building_damage": "CHRONOTERRA_DAMAGE_CHECKPOINT",
}


def _checkpoint_reference(task: str) -> str:
    """Return a safe human-readable reference without disclosing custom filesystem paths."""
    env_name = MODEL_REFERENCE_ENV[task]
    if os.environ.get(env_name):
        return f"external/configured checkpoint (path redacted; configured by {env_name})"
    return MODEL_REFERENCES[task]


class UploadBudget:
    def __init__(self) -> None:
        self.used = 0


def _cleanup_expired_sessions() -> None:
    """Opportunistically delete old session artifacts, at most once per five minutes."""
    global _last_cleanup_monotonic
    now_mono = time.monotonic()
    with _cleanup_lock:
        if now_mono - _last_cleanup_monotonic < CLEANUP_CHECK_INTERVAL_SECONDS:
            return
        _last_cleanup_monotonic = now_mono
    cutoff = datetime.now(timezone.utc).timestamp() - SESSION_RETENTION_HOURS * 3600
    try:
        for child in OUTPUT_ROOT.iterdir():
            if not child.is_dir() or not SESSION_ID_RE.fullmatch(child.name):
                continue
            try:
                if child.stat().st_mtime < cutoff:
                    import shutil
                    shutil.rmtree(child, ignore_errors=True)
            except OSError:
                logger.warning("Could not clean expired session artifact directory")
    except OSError:
        logger.warning("Could not inspect session artifact directory for cleanup")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_evidence(checkpoint: Path, task: str) -> dict:
    """Require a hash-bound, explicitly reviewed deployment manifest before inference."""
    if not checkpoint.is_file():
        return {"status": "MODEL_NOT_AVAILABLE", "approved": False, "checkpoint_exists": False,
                "evidence_manifest_exists": False, "held_out_metrics_exists": False}
    manifest_path = checkpoint.parent / "deployment_manifest.json"
    status = {"status": "NEEDS_EVIDENCE_REVIEW", "approved": False, "checkpoint_exists": True,
              "evidence_manifest_exists": manifest_path.is_file(), "held_out_metrics_exists": False,
              "evaluation_summary": None}
    if not manifest_path.is_file():
        return status
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return status
        actual_hash = _sha256_file(checkpoint)
        evaluation = manifest.get("evaluation") if isinstance(manifest.get("evaluation"), dict) else {}
        review = manifest.get("review") if isinstance(manifest.get("review"), dict) else {}
        metrics_name = evaluation.get("metrics_file")
        if not isinstance(metrics_name, str) or Path(metrics_name).name != metrics_name:
            return status
        metrics_path = checkpoint.parent / metrics_name
        if not metrics_path.is_file():
            return status
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics_hash_matches = evaluation.get("metrics_sha256") == _sha256_file(metrics_path)
        source_report_name = evaluation.get("source_report_file")
        source_report_hash = evaluation.get("source_report_sha256")
        source_report_matches = False
        if (isinstance(source_report_name, str) and Path(source_report_name).name == source_report_name
                and source_report_name == metrics.get("source_report_file")
                and isinstance(source_report_hash, str) and source_report_hash == metrics.get("source_report_sha256")):
            source_report_path = checkpoint.parent / source_report_name
            if source_report_path.is_file() and _sha256_file(source_report_path) == source_report_hash:
                source_payload = json.loads(source_report_path.read_text(encoding="utf-8"))
                if isinstance(source_payload, dict):
                    if task == "urban_change":
                        source_dataset = source_payload.get("dataset")
                        per_scene = source_payload.get("per_scene")
                        source_count = len(per_scene) if isinstance(per_scene, (dict, list)) else 0
                        source_report_matches = (
                            isinstance(source_dataset, str) and "test" in source_dataset.lower()
                            and source_payload.get("model_selection_used_test_data") is False
                            and source_count > 0 and source_count == evaluation.get("held_out_scene_count")
                            and source_dataset == evaluation.get("dataset_name")
                        )
                    else:
                        per_scene = source_payload.get("per_scene")
                        source_count = len(per_scene) if isinstance(per_scene, (dict, list)) else 0
                        source_report_matches = (
                            source_payload.get("task") == task and source_payload.get("split") == "test"
                            and source_payload.get("model_selection_used_evaluated_split") is False
                            and source_count > 0 and source_count == evaluation.get("held_out_scene_count")
                            and source_payload.get("task") == evaluation.get("dataset_name")
                        )
        required_metrics = {"precision", "recall", "f1", "iou", "false_positive_rate"}
        def has_metric(key: str) -> bool:
            value = metrics.get(key) if isinstance(metrics, dict) else None
            if value is None and isinstance(metrics, dict) and isinstance(metrics.get("metrics"), dict):
                value = metrics["metrics"].get(key)
            return isinstance(value, (int, float)) and np.isfinite(value) and 0 <= float(value) <= 1
        metrics_ok = all(has_metric(key) for key in required_metrics)
        scenes = evaluation.get("held_out_scene_count")
        metric_values = {}
        if metrics_ok:
            for key in sorted(required_metrics):
                value = metrics.get(key) if isinstance(metrics, dict) else None
                if value is None and isinstance(metrics, dict) and isinstance(metrics.get("metrics"), dict):
                    value = metrics["metrics"].get(key)
                metric_values[key] = float(value)
        thresholds = evaluation.get("deployment_thresholds") if isinstance(evaluation.get("deployment_thresholds"), dict) else {}
        required_thresholds = {"minimum_precision", "minimum_recall", "minimum_f1", "minimum_iou", "maximum_false_positive_rate"}
        thresholds_valid = all(isinstance(thresholds.get(key), (int, float)) and not isinstance(thresholds.get(key), bool)
                               and np.isfinite(thresholds.get(key)) and 0 <= float(thresholds.get(key)) <= 1
                               for key in required_thresholds)
        thresholds_satisfied = bool(thresholds_valid and metrics_ok and
            metric_values.get("precision", -1) >= float(thresholds["minimum_precision"]) and
            metric_values.get("recall", -1) >= float(thresholds["minimum_recall"]) and
            metric_values.get("f1", -1) >= float(thresholds["minimum_f1"]) and
            metric_values.get("iou", -1) >= float(thresholds["minimum_iou"]) and
            metric_values.get("false_positive_rate", 2) <= float(thresholds["maximum_false_positive_rate"]))
        evaluation_ok = (evaluation.get("split_type") == "held_out_by_scene" and isinstance(scenes, int)
                         and not isinstance(scenes, bool) and scenes > 0 and bool(evaluation.get("dataset_name"))
                         and metrics_ok and metrics_hash_matches and source_report_matches and thresholds_satisfied
                         and metrics.get("task") == task and metrics.get("dataset_name") == evaluation.get("dataset_name")
                         and metrics.get("held_out_scene_count") == scenes and metrics.get("split_type") == "held_out_by_scene")
        metrics_summary = None
        if (metrics_ok and metrics_hash_matches and source_report_matches
                and evaluation.get("split_type") == "held_out_by_scene"
                and isinstance(scenes, int) and not isinstance(scenes, bool) and scenes > 0):
            metrics_summary = {
                "dataset_name": str(evaluation.get("dataset_name"))[:160],
                "held_out_scene_count": int(scenes) if isinstance(scenes, int) and not isinstance(scenes, bool) and scenes > 0 else None,
                "split_type": str(evaluation.get("split_type") or "Not verified")[:80],
                "metrics": metric_values,
                "evaluation_date": str(evaluation.get("evaluation_date") or "Not supplied")[:80],
                "metrics_hash_matches": bool(metrics_hash_matches),
                "deployment_thresholds": {key: float(thresholds[key]) for key in sorted(required_thresholds) if thresholds_valid},
                "thresholds_satisfied": bool(thresholds_satisfied),
            }
        review_ok = (review.get("status") == "APPROVED" and isinstance(review.get("reviewer"), str)
                     and bool(review.get("reviewer").strip()) and isinstance(review.get("reviewed_at"), str)
                     and bool(review.get("reviewed_at").strip()))
        hash_ok = manifest.get("checkpoint_sha256") == actual_hash
        task_ok = manifest.get("task") == task
        approved = bool(manifest.get("approved_for_inference") is True and evaluation_ok and review_ok and hash_ok and task_ok)
        return {"status": "APPROVED_FOR_INFERENCE" if approved else "NEEDS_EVIDENCE_REVIEW",
                "approved": approved, "checkpoint_exists": True, "evidence_manifest_exists": True,
                "held_out_metrics_exists": bool(metrics_ok and metrics_hash_matches and source_report_matches and evaluation.get("split_type") == "held_out_by_scene" and isinstance(scenes, int) and not isinstance(scenes, bool) and scenes > 0),
                "checkpoint_hash_matches": hash_ok, "metrics_hash_matches": bool(metrics_hash_matches),
                "source_report_hash_matches": bool(source_report_matches),
                "thresholds_satisfied": bool(thresholds_satisfied),
                "task_matches": task_ok, "evaluation_evidence_valid": evaluation_ok,
                "evaluation_summary": metrics_summary,
                "human_review_approved": review_ok}
    except Exception:
        # Manifest parsing details stay server-side. Malformed evidence never unlocks inference.
        logger.warning("Model deployment evidence could not be validated for task %s", task)
        return status


def _redact_public_text(value: str) -> str:
    text = value
    for secret_path in (str(ROOT), str(OUTPUT_ROOT), str(Path.home()), os.getcwd()):
        if secret_path and secret_path not in {"/", "\\"}:
            text = text.replace(secret_path, "[redacted-path]")
    # Conservative fallbacks for paths outside the known project roots.
    text = re.sub(r'\b[A-Za-z]:\\\S+', '[redacted-path]', text)
    text = re.sub(r'(?<![:\w])/(?:[^\s,;]+/)*[^\s,;]+', '[redacted-path]', text)
    return text


@app.get("/")
def home() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "local-baseline-and-optional-learned-prediction",
            "claim": "models are available only when real-trained checkpoints have been added and independently evaluated"}


@app.get("/api/models/status")
def model_status() -> dict:
    """Show checkpoint availability without implying that a checkpoint is accurate or validated."""
    out = {}
    for task, checkpoint in MODEL_CHECKPOINTS.items():
        evidence = _model_evidence(checkpoint, task)
        out[task] = {
            "checkpoint_reference": _checkpoint_reference(task),
            "checkpoint_exists": evidence["checkpoint_exists"],
            "evidence_manifest_exists": evidence["evidence_manifest_exists"],
            "held_out_test_metrics_exists": evidence["held_out_metrics_exists"],
            "evaluation_summary": evidence.get("evaluation_summary"),
            "status": evidence["status"],
            "evidence": {key: value for key, value in evidence.items() if key != "status"},
            "warning": "Only a hash-bound deployment manifest with held-out scene metrics and explicit human review enables inference. File presence alone is not validation."
        }
    return {"models": out}


async def _save_upload(upload: UploadFile, path: Path, *, allowed: set[str] = ALLOWED_UPLOAD_SUFFIXES,
                       budget: UploadBudget | None = None, max_bytes: int = MAX_UPLOAD_BYTES,
                       size_label: str | None = None) -> None:
    """Stream upload contents to disk with per-file and aggregate request budgets.

    Smaller structured sidecars (GeoJSON/metadata) use dedicated limits so they are
    rejected during streaming rather than written to disk and parsed first.
    """
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1 or max_bytes > MAX_UPLOAD_BYTES:
        raise ValueError("max_bytes must be a positive integer no larger than MAX_UPLOAD_BYTES")
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported upload extension {suffix!r}; allowed: {sorted(allowed)}")
    budget = budget or UploadBudget()
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with path.open("wb") as target:
            while True:
                chunk = await upload.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                budget.used += len(chunk)
                if written > max_bytes:
                    label = size_label or "Uploaded file"
                    raise HTTPException(413, f"{label} must be no larger than {max_bytes // (1024*1024)} MiB")
                if budget.used > MAX_REQUEST_UPLOAD_BYTES:
                    raise HTTPException(413, f"Combined files exceed the {MAX_REQUEST_UPLOAD_BYTES // (1024*1024)} MB request upload limit")
                target.write(chunk)
        if written == 0:
            raise HTTPException(400, "Uploaded file is empty")
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _validate_uploaded_asset(path: Path, task: str, label: str) -> dict:
    """Reject compressed-raster decompression bombs before whole-array inference."""
    suffix = path.suffix.lower()
    limit = TASK_MAX_PIXELS.get(task, 2_000_000)
    try:
        if suffix in {".tif", ".tiff", ".vrt"}:
            with rasterio.open(path) as ds:
                pixels = int(ds.width) * int(ds.height)
                if ds.width <= 0 or ds.height <= 0 or ds.width > MAX_AXIS_PIXELS or ds.height > MAX_AXIS_PIXELS:
                    raise HTTPException(413, f"{label} dimensions exceed supported axis limits ({MAX_AXIS_PIXELS}px per side). Crop the AOI before upload.")
                if pixels > limit:
                    raise HTTPException(413, f"{label} contains {pixels:,} pixels; this task currently allows at most {limit:,} pixels because full-raster inference is memory bounded by this limit. Crop the area of interest and retry.")
                if ds.count > 32:
                    raise HTTPException(413, f"{label} has too many bands for safe inspection ({ds.count}).")
                if not all(np.isfinite([ds.transform.a, ds.transform.b, ds.transform.c, ds.transform.d, ds.transform.e, ds.transform.f])):
                    raise HTTPException(422, f"{label} contains a non-finite geotransform.")
                return {"width": ds.width, "height": ds.height, "pixels": pixels, "bands": ds.count, "crs": str(ds.crs) if ds.crs else None}
        with Image.open(path) as image:
            width, height = image.size
            pixels = int(width) * int(height)
            if width <= 0 or height <= 0 or width > MAX_AXIS_PIXELS or height > MAX_AXIS_PIXELS or pixels > limit:
                raise HTTPException(413, f"{label} dimensions exceed this task's safe image limit ({limit:,} pixels). Resize/crop the input and retry.")
            return {"width": width, "height": height, "pixels": pixels, "bands": len(image.getbands()), "crs": None}
    except HTTPException:
        raise
    except Exception as exc:
        logger.info("Uploaded asset failed validation (%s): %s", label, type(exc).__name__)
        raise HTTPException(422, f"{label} is not a readable supported raster or image.") from exc


def _parse_acquisition_date(value: str | None, label: str):
    """Parse one optional acquisition date even when its pair partner is missing."""
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise HTTPException(422, f"{label} acquisition date must be an ISO date string (YYYY-MM-DD).")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError as exc:
        raise HTTPException(422, "Acquisition dates must be valid ISO dates (YYYY-MM-DD).") from exc


def _validate_pair_dates(before_date: str | None, after_date: str | None):
    before = _parse_acquisition_date(before_date, "Before")
    after = _parse_acquisition_date(after_date, "After")
    if before is not None and after is not None and after <= before:
        raise HTTPException(422, "Chronology is invalid: the after-image acquisition date must be later than the before-image date.")
    return before, after


def _validate_pair_metadata_dates(path: Path):
    """Validate user/source dates carried in the optional pair metadata JSON."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(422, "Pair metadata must be valid UTF-8 JSON.") from exc
    if not isinstance(payload, dict):
        raise HTTPException(422, "Pair metadata must be a JSON object.")
    before = payload.get("before") if isinstance(payload.get("before"), dict) else {}
    after = payload.get("after") if isinstance(payload.get("after"), dict) else {}
    before_date = before.get("acquisition_date") or payload.get("before_date")
    after_date = after.get("acquisition_date") or payload.get("after_date")
    return _validate_pair_dates(before_date, after_date)


def _boundary_temporal_compatibility(effective_date: str | None, pair_metadata_path: Path | None) -> dict:
    """Check obvious date mismatch without pretending one date proves legal validity."""
    if not effective_date:
        return {"status": "NOT_VERIFIED", "message": "No reference-zone effective date was supplied. Spatial overlap is not temporally verified."}
    try:
        boundary_day = datetime.fromisoformat(effective_date).date()
    except (TypeError, ValueError):
        return {"status": "INVALID_DATE", "message": "Reference-zone effective date is not a valid ISO date."}
    before_day = after_day = None
    if pair_metadata_path is not None:
        try:
            before_day, after_day = _validate_pair_metadata_dates(pair_metadata_path)
        except HTTPException:
            return {"status": "NOT_VERIFIED", "message": "Satellite acquisition dates could not be used to verify the reference-zone date."}
    if after_day is not None and boundary_day > after_day:
        return {"status": "TEMPORAL_MISMATCH", "message": "The reference-zone effective date is later than the after-image acquisition date. Boundary comparison was skipped to avoid comparing against a future boundary state.", "boundary_effective_date": boundary_day.isoformat(), "after_image_date": after_day.isoformat()}
    if before_day is None or after_day is None:
        return {"status": "NOT_VERIFIED", "message": "One or both satellite acquisition dates are missing. Temporal compatibility cannot be assessed.", "boundary_effective_date": boundary_day.isoformat()}
    return {"status": "NOT_VERIFIED", "message": "The effective date is not later than T2, but one date alone does not establish that this boundary was valid throughout the T1–T2 interval. Confirm its validity period and authority independently.", "boundary_effective_date": boundary_day.isoformat(), "before_image_date": before_day.isoformat(), "after_image_date": after_day.isoformat()}


@app.post("/api/preflight")
async def preflight_endpoint(
    task: str = Form(...),
    input_file: UploadFile | None = File(None),
    before_file: UploadFile | None = File(None),
    after_file: UploadFile | None = File(None),
    before_scl_file: UploadFile | None = File(None),
    after_scl_file: UploadFile | None = File(None),
    pair_metadata_file: UploadFile | None = File(None),
    aoi_geojson_file: UploadFile | None = File(None),
    reference_boundary_geojson_file: UploadFile | None = File(None),
    boundary_source_name: str | None = Form(None),
    boundary_effective_date: str | None = Form(None),
    boundary_proximity_m: float = Form(10.0),
    scl_class2_policy: str = Form("mask"),
    assume_band_order: bool = Form(False),
) -> dict:
    """Inspect compatibility/factors without requiring model weights or running inference."""
    _cleanup_expired_sessions()
    budget = UploadBudget()
    if task not in CLASS_NAMES:
        raise HTTPException(400, f"Unknown task {task!r}; supported tasks: {sorted(CLASS_NAMES)}")
    session_id = "preflight-" + uuid.uuid4().hex
    directory = OUTPUT_ROOT / session_id
    directory.mkdir(parents=True, exist_ok=False)
    try:
        input_path = before_path = after_path = before_scl_path = after_scl_path = None
        pair_metadata_path = None
        aoi_geojson_path = None
        reference_boundary_path = None
        if task == "flood_extent":
            if input_file is None:
                raise HTTPException(400, "flood_extent requires input_file: a two-band Sentinel-1 VV/VH GeoTIFF")
            input_path = directory / ("input" + Path(input_file.filename or "input.tif").suffix.lower())
            await _save_upload(input_file, input_path, allowed={".tif", ".tiff"}, budget=budget)
        else:
            if before_file is None or after_file is None:
                raise HTTPException(400, f"{task} requires before_file and after_file")
            before_path = directory / ("before" + Path(before_file.filename or "before.tif").suffix.lower())
            after_path = directory / ("after" + Path(after_file.filename or "after.tif").suffix.lower())
            await _save_upload(before_file, before_path, budget=budget)
            await _save_upload(after_file, after_path, budget=budget)
            if task == "urban_change" and (before_path.suffix not in {".tif", ".tiff"} or after_path.suffix not in {".tif", ".tiff"}):
                raise HTTPException(400, "urban_change requires georeferenced 13-band Sentinel-2 GeoTIFF pairs")
            if task == "building_damage" and (before_path.suffix in {".tif", ".tiff"}) != (after_path.suffix in {".tif", ".tiff"}):
                raise HTTPException(400, "Do not mix GeoTIFF and ordinary image files in one pair")
        if aoi_geojson_file is not None:
            aoi_geojson_path = directory / "aoi.geojson"
            await _save_upload(aoi_geojson_file, aoi_geojson_path, allowed={".geojson", ".json"}, budget=budget,
                               max_bytes=10 * 1024 * 1024, size_label="AOI GeoJSON")
        if pair_metadata_file is not None:
            if task == "flood_extent":
                raise HTTPException(400, "pair_metadata_file is only supported for paired tasks")
            pair_metadata_path = directory / "pair_metadata.json"
            await _save_upload(pair_metadata_file, pair_metadata_path, allowed={".json"}, budget=budget,
                               max_bytes=1 * 1024 * 1024, size_label="Pair metadata JSON")
            _validate_pair_metadata_dates(pair_metadata_path)
        if before_scl_file is not None or after_scl_file is not None:
            if task != "urban_change":
                raise HTTPException(400, "SCL masks are supported only for urban_change")
            if before_scl_file is None or after_scl_file is None:
                raise HTTPException(400, "Supply both before_scl_file and after_scl_file, or neither")
            before_scl_path = directory / "before_scl.tif"
            after_scl_path = directory / "after_scl.tif"
            await _save_upload(before_scl_file, before_scl_path, allowed={".tif", ".tiff"}, budget=budget)
            await _save_upload(after_scl_file, after_scl_path, allowed={".tif", ".tiff"}, budget=budget)
        for label, asset_path in (("input", input_path), ("before image", before_path), ("after image", after_path),
                                  ("before SCL mask", before_scl_path), ("after SCL mask", after_scl_path)):
            if asset_path is not None:
                _validate_uploaded_asset(asset_path, task, label)
        if reference_boundary_geojson_file is not None:
            if task != "urban_change":
                raise HTTPException(400, "Reference-zone boundary comparison is currently supported only for urban_change.")
            if not math.isfinite(boundary_proximity_m) or not 0 <= boundary_proximity_m <= 1000:
                raise HTTPException(422, "boundary_proximity_m must be between 0 and 1000 metres.")
            if boundary_effective_date:
                try: datetime.fromisoformat(boundary_effective_date).date()
                except ValueError as exc: raise HTTPException(422, "Boundary effective date must be a valid ISO date.") from exc
            reference_boundary_path = directory / "reference_boundary.geojson"
            await _save_upload(reference_boundary_geojson_file, reference_boundary_path, allowed={".geojson", ".json"}, budget=budget,
                               max_bytes=10 * 1024 * 1024, size_label="Reference-zone GeoJSON")
            from api.boundary_analysis import validate_boundary
            boundary_qa = validate_boundary(reference_boundary_path)
            boundary_qa["temporal_compatibility"] = _boundary_temporal_compatibility(boundary_effective_date, pair_metadata_path)
            report_boundary_warning = boundary_qa["temporal_compatibility"]["message"] if boundary_qa["temporal_compatibility"]["status"] == "TEMPORAL_MISMATCH" else None
        else:
            boundary_qa = None
            report_boundary_warning = None
        from quality_gate.prediction_preflight import run_prediction_preflight
        report = run_prediction_preflight(task, input_path=input_path, before_path=before_path, after_path=after_path,
                                          before_scl=before_scl_path, after_scl=after_scl_path,
                                          pair_metadata_json=pair_metadata_path, assume_band_order=assume_band_order,
                                          aoi_geojson=aoi_geojson_path, scl_class2_policy=scl_class2_policy)
        if boundary_qa is not None:
            report.setdefault("factors", {})["reference_boundary"] = boundary_qa
            if report_boundary_warning:
                report.setdefault("warnings", []).append(report_boundary_warning)
        # Keep the detailed report private; publish only a path-redacted copy.
        report_path = directory / "preflight_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        public_report_path = directory / "preflight_report_public.json"
        public_report_path.write_text(json.dumps(_public_report(report), indent=2, default=str), encoding="utf-8")
        public_response = _public_report(report)
        public_response["report_url"] = _artifact_url(public_report_path)
        public_response["note"] = "Preflight is independent of model weights. PASS/WARNING does not prove prediction accuracy or geographic identity for metadata-free RGB pairs."
        if reference_boundary_path is not None:
            public_response["reference_boundary"] = boundary_qa
            public_response["boundary_note"] = "Boundary geometry is user-supplied and its authority/effective legal status are not independently verified."
        public_report_path.write_text(json.dumps(public_response, indent=2, default=str), encoding="utf-8")
        return public_response
    except HTTPException:
        import shutil
        shutil.rmtree(directory, ignore_errors=True)
        raise
    except (ValueError, FileNotFoundError) as exc:
        import shutil
        shutil.rmtree(directory, ignore_errors=True)
        logger.info("Preflight rejected input (%s)", type(exc).__name__)
        raise HTTPException(422, detail={"status": "PREFLIGHT_FAILED", "reason": _redact_public_text(str(exc))}) from exc
    except Exception as exc:
        import shutil
        shutil.rmtree(directory, ignore_errors=True)
        request_id = uuid.uuid4().hex
        logger.exception("Preflight failed (request_id=%s)", request_id)
        if isinstance(exc, (ImportError, OSError)) and "Application Control" in str(exc):
            raise HTTPException(503, detail={"status": "DEPENDENCY_UNAVAILABLE", "request_id": request_id,
                                             "message": "A required ML dependency could not be loaded because Windows Application Control blocked its native library."}) from exc
        raise HTTPException(500, detail={"status": "PREFLIGHT_FAILED", "request_id": request_id,
                                         "message": "Preflight failed unexpectedly. Check server logs using the request ID."}) from exc


@app.post("/api/predict")
async def predict(
    task: str = Form(...),
    input_file: UploadFile | None = File(None),
    before_file: UploadFile | None = File(None),
    after_file: UploadFile | None = File(None),
    before_scl_file: UploadFile | None = File(None),
    after_scl_file: UploadFile | None = File(None),
    pair_metadata_file: UploadFile | None = File(None),
    aoi_geojson_file: UploadFile | None = File(None),
    reference_boundary_geojson_file: UploadFile | None = File(None),
    boundary_source_name: str | None = Form(None),
    boundary_effective_date: str | None = Form(None),
    boundary_proximity_m: float = Form(10.0),
    scl_class2_policy: str = Form("mask"),
    assume_band_order: bool = Form(False),
    tile_size: int = Form(256),
    overlap: int = Form(64),
) -> dict:
    """Run only a task-compatible trained checkpoint; never substitutes the classical baseline."""
    if task not in CLASS_NAMES:
        raise HTTPException(400, f"Unknown task {task!r}; supported tasks: {sorted(CLASS_NAMES)}")
    _cleanup_expired_sessions()
    checkpoint = MODEL_CHECKPOINTS[task]
    evidence = _model_evidence(checkpoint, task)
    if not checkpoint.is_file():
        raise HTTPException(503, detail={
            "status": "MODEL_NOT_AVAILABLE", "task": task,
            "checkpoint_reference": _checkpoint_reference(task),
            "message": "The endpoint will not fabricate a prediction. No trained checkpoint is installed for this task; train and evaluate the matching model first."
        })
    if not evidence.get("approved"):
        raise HTTPException(503, detail={
            "status": "MODEL_NEEDS_EVIDENCE_REVIEW", "task": task,
            "checkpoint_reference": _checkpoint_reference(task),
            "message": "A checkpoint exists but is blocked until a hash-matched deployment manifest, held-out scene metrics, and explicit review approval are present."
        })
    if tile_size < 32 or tile_size % 16 or overlap < 0 or overlap >= tile_size:
        raise HTTPException(400, "tile_size must be >=32 and divisible by 16; overlap must be smaller than tile_size")
    session_id = uuid.uuid4().hex
    directory = OUTPUT_ROOT / session_id
    directory.mkdir(parents=True, exist_ok=False)
    budget = UploadBudget()
    try:
        input_path = before_path = after_path = before_scl_path = after_scl_path = None
        aoi_geojson_path = None
        reference_boundary_path = None
        if task == "flood_extent":
            if input_file is None:
                raise HTTPException(400, "flood_extent requires input_file: a two-band Sentinel-1 VV/VH GeoTIFF")
            input_path = directory / ("input" + Path(input_file.filename or "input.tif").suffix.lower())
            await _save_upload(input_file, input_path, allowed={".tif", ".tiff"}, budget=budget)
        else:
            if before_file is None or after_file is None:
                raise HTTPException(400, f"{task} requires before_file and after_file")
            before_path = directory / ("before" + Path(before_file.filename or "before.tif").suffix.lower())
            after_path = directory / ("after" + Path(after_file.filename or "after.tif").suffix.lower())
            await _save_upload(before_file, before_path, budget=budget)
            await _save_upload(after_file, after_path, budget=budget)
            if task == "urban_change" and (before_path.suffix not in {".tif", ".tiff"} or after_path.suffix not in {".tif", ".tiff"}):
                raise HTTPException(400, "urban_change requires two 13-band georeferenced Sentinel-2 GeoTIFFs; RGB images are incompatible")
            if task == "building_damage" and (before_path.suffix in {".tif", ".tiff"}) != (after_path.suffix in {".tif", ".tiff"}):
                raise HTTPException(400, "Do not mix GeoTIFF and ordinary image files in a damage pair")
        pair_metadata_path = None
        if aoi_geojson_file is not None:
            aoi_geojson_path = directory / "aoi.geojson"
            await _save_upload(aoi_geojson_file, aoi_geojson_path, allowed={".geojson", ".json"}, budget=budget,
                               max_bytes=10 * 1024 * 1024, size_label="AOI GeoJSON")
        if pair_metadata_file is not None:
            if task == "flood_extent":
                raise HTTPException(400, "pair_metadata_file is only supported for paired tasks")
            pair_metadata_path = directory / "pair_metadata.json"
            await _save_upload(pair_metadata_file, pair_metadata_path, allowed={".json"}, budget=budget,
                               max_bytes=1 * 1024 * 1024, size_label="Pair metadata JSON")
            _validate_pair_metadata_dates(pair_metadata_path)
        if before_scl_file is not None or after_scl_file is not None:
            if task != "urban_change":
                raise HTTPException(400, "SCL quality masks are supported only by urban_change")
            if before_scl_file is None or after_scl_file is None:
                raise HTTPException(400, "Supply both before_scl_file and after_scl_file, or neither")
            before_scl_path = directory / "before_scl.tif"
            after_scl_path = directory / "after_scl.tif"
            await _save_upload(before_scl_file, before_scl_path, allowed={".tif", ".tiff"}, budget=budget)
            await _save_upload(after_scl_file, after_scl_path, allowed={".tif", ".tiff"}, budget=budget)
        for label, asset_path in (("input", input_path), ("before image", before_path), ("after image", after_path),
                                  ("before SCL mask", before_scl_path), ("after SCL mask", after_scl_path)):
            if asset_path is not None:
                _validate_uploaded_asset(asset_path, task, label)
        if reference_boundary_geojson_file is not None:
            if task != "urban_change":
                raise HTTPException(400, "Reference-zone boundary comparison is currently supported only for urban_change.")
            if not math.isfinite(boundary_proximity_m) or not 0 <= boundary_proximity_m <= 1000:
                raise HTTPException(422, "boundary_proximity_m must be between 0 and 1000 metres.")
            if boundary_effective_date:
                try: datetime.fromisoformat(boundary_effective_date).date()
                except ValueError as exc: raise HTTPException(422, "Boundary effective date must be a valid ISO date.") from exc
            reference_boundary_path = directory / "reference_boundary.geojson"
            await _save_upload(reference_boundary_geojson_file, reference_boundary_path, allowed={".geojson", ".json"}, budget=budget,
                               max_bytes=10 * 1024 * 1024, size_label="Reference-zone GeoJSON")
            validate_boundary(reference_boundary_path)
            # The change model requires correctly georeferenced input; the precise overlay is run after vectorization.
        if not INFERENCE_GATE.acquire(blocking=False):
            raise HTTPException(429, detail={
                "status": "INFERENCE_BUSY",
                "message": "Another prediction is already running. Retry after it finishes; concurrent inference is disabled to protect memory.",
            })
        try:
            report = await run_in_threadpool(
                predict_task, task, checkpoint, directory / "prediction",
                input_path=input_path, before_path=before_path, after_path=after_path,
                before_scl=before_scl_path, after_scl=after_scl_path, pair_metadata_json=pair_metadata_path,
                assume_band_order=assume_band_order, aoi_geojson=aoi_geojson_path,
                scl_class2_policy=scl_class2_policy, tile_size=tile_size, overlap=overlap, device_name="auto",
            )
        finally:
            INFERENCE_GATE.release()
        def url_for(key: str) -> str:
            p = Path(report[key])
            return _artifact_url(p)
        mask_preview_path = directory / "prediction_mask_preview.png"
        _mask_preview(Path(report["mask_output"]), mask_preview_path, task)
        regions_path = directory / "prediction_regions.json"
        regions_info = _vectorize_prediction_mask(Path(report["mask_output"]), regions_path, task)
        boundary_analysis = None
        boundary_overlay_path = None
        before_preview_path = after_preview_path = input_preview_path = None
        before_bounds = after_bounds = bounds = None
        if task == "flood_extent":
            input_preview_path = directory / "input_preview.png"
            _preview_file(input_path, input_preview_path, task)
            bounds = _geo_bounds(input_path)
        else:
            before_preview_path = directory / "before_preview.png"
            after_preview_path = directory / "after_preview.png"
            _preview_pair(before_path, after_path, before_preview_path, after_preview_path, task)
            before_bounds, after_bounds = _geo_bounds(before_path), _geo_bounds(after_path)
            # Geographic overlays are only offered when both rasters carry the same extent.
            if before_bounds and after_bounds and np.allclose(np.asarray(before_bounds), np.asarray(after_bounds), rtol=0, atol=1e-7):
                bounds = before_bounds
        if reference_boundary_path is not None:
            temporal = _boundary_temporal_compatibility(boundary_effective_date, pair_metadata_path)
            if temporal["status"] == "TEMPORAL_MISMATCH":
                boundary_analysis = {
                    "status": "NOT_RUN_TEMPORAL_MISMATCH",
                    "source_name": (boundary_source_name or "User-supplied reference zone")[:160],
                    "effective_date": boundary_effective_date or None,
                    "authority_status": "USER_SUPPLIED_NOT_INDEPENDENTLY_VERIFIED",
                    "temporal_compatibility": temporal,
                    "interpretation": "No boundary-overlap classification was run because the reference-zone date is later than the after-image date.",
                }
                report["warnings"].append(temporal["message"])
            elif regions_info["coordinate_space"] == "geographic" and bounds is not None:
                boundary_overlay_path = directory / "reference_boundary_overlay.geojson"
                boundary_analysis = analyze_boundary_candidates(
                    regions_path, reference_boundary_path, boundary_overlay_path,
                    source_name=boundary_source_name, effective_date=boundary_effective_date,
                    proximity_m=boundary_proximity_m,
                )
                boundary_analysis["temporal_compatibility"] = temporal
                if temporal["status"] == "NOT_VERIFIED":
                    report["warnings"].append(temporal["message"])
                # Keep feature attributes, side-panel summaries and downloaded GeoJSON aligned.
                updated_collection = json.loads(regions_path.read_text(encoding="utf-8"))
                summaries_by_id = {item.get("id"): item for item in regions_info["regions"]}
                for feature in updated_collection.get("features", []):
                    props = feature.get("properties", {})
                    props["boundary_temporal_status"] = temporal["status"]
                    props["boundary_temporal_note"] = temporal["message"]
                    summary = summaries_by_id.get(props.get("id"))
                    if summary is not None:
                        for key in ("boundary_relation", "boundary_overlap_m2", "boundary_overlap_fraction",
                                    "distance_to_reference_zone_m", "boundary_source_name", "boundary_effective_date",
                                    "boundary_authority_status", "review_priority"):
                            if key in props:
                                summary[key] = props[key]
                        summary["boundary_temporal_status"] = temporal["status"]
                        summary["boundary_temporal_note"] = temporal["message"]
                regions_path.write_text(json.dumps(updated_collection, indent=2, allow_nan=False), encoding="utf-8")
            else:
                boundary_analysis = {
                    "status": "NOT_RUN_GEOGRAPHIC_ALIGNMENT_UNVERIFIED",
                    "source_name": (boundary_source_name or "User-supplied reference zone")[:160],
                    "effective_date": boundary_effective_date or None,
                    "authority_status": "USER_SUPPLIED_NOT_INDEPENDENTLY_VERIFIED",
                    "temporal_compatibility": temporal,
                    "interpretation": "Boundary comparison was not run because a verified geographic overlay for the paired result is unavailable. No encroachment inference was made.",
                }
                report["warnings"].append("Reference-zone comparison was not run because a verified geographic overlay for the paired result is unavailable.")
        def output_url(path: Path) -> str:
            return _artifact_url(path)
        report["visualization_artifacts"] = {
            "mask_preview": str(mask_preview_path),
            "regions_output": str(regions_path),
            "regions_coordinate_space": regions_info["coordinate_space"],
            "region_count": regions_info["region_count"],
            "regions_truncated": regions_info["truncated"],
            "geographic_overlay_verified": bounds is not None,
            "bounds_wgs84_leaflet": bounds,
            "before_bounds_wgs84_leaflet": before_bounds,
            "after_bounds_wgs84_leaflet": after_bounds,
            "boundary_analysis": boundary_analysis,
        }
        Path(report["report_output"]).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        public_report_path = directory / "prediction_report_public.json"
        public_report_path.write_text(json.dumps(_public_report(report), indent=2, default=str), encoding="utf-8")
        return {
            "session_id": session_id, "task": task, "status": report["status"],
            "mask_url": url_for("mask_output"), "scores_url": url_for("scores_output"),
            "uncertainty_url": url_for("uncertainty_output"), "overlay_url": url_for("overlay_output"),
            "mask_preview_url": output_url(mask_preview_path), "report_url": output_url(public_report_path),
            "regions_url": output_url(regions_path), "regions": regions_info["regions"],
            "region_count": regions_info["region_count"], "regions_coordinate_space": regions_info["coordinate_space"],
            "regions_truncated": regions_info["truncated"],
            "boundary_analysis": boundary_analysis,
            "boundary_overlay_url": output_url(boundary_overlay_path) if boundary_overlay_path else None,
            "before_preview_url": output_url(before_preview_path) if before_preview_path else None,
            "after_preview_url": output_url(after_preview_path) if after_preview_path else None,
            "input_preview_url": output_url(input_preview_path) if input_preview_path else None,
            "bounds": bounds, "before_bounds": before_bounds, "after_bounds": after_bounds,
            "geographic_overlay_verified": bounds is not None,
            "valid_fraction": report["valid_fraction_exact"],
            "predicted_class_pixel_counts": report["predicted_class_pixel_counts"],
            "warnings": [_redact_public_text(str(w)) for w in report["warnings"]],
            "preflight": _public_report(report["preflight"])
        }
    except HTTPException:
        import shutil
        shutil.rmtree(directory, ignore_errors=True)
        raise
    except (ValueError, FileNotFoundError) as exc:
        import shutil
        shutil.rmtree(directory, ignore_errors=True)
        logger.info("Prediction rejected input (%s)", type(exc).__name__)
        raise HTTPException(422, detail={"status": "PREDICTION_REJECTED", "reason": _redact_public_text(str(exc))}) from exc
    except Exception as exc:
        import shutil
        shutil.rmtree(directory, ignore_errors=True)
        request_id = uuid.uuid4().hex
        logger.exception("Prediction failed (request_id=%s)", request_id)
        if isinstance(exc, (ImportError, OSError)) and "Application Control" in str(exc):
            raise HTTPException(503, detail={"status": "DEPENDENCY_UNAVAILABLE", "request_id": request_id,
                                             "message": "A required ML dependency could not be loaded because Windows Application Control blocked its native library."}) from exc
        raise HTTPException(500, detail={"status": "PREDICTION_FAILED", "request_id": request_id,
                                         "message": "Prediction failed unexpectedly. Check server logs using the request ID."}) from exc


def _preview_rgb(stack: np.ndarray, valid: np.ndarray, rgb_indices: tuple[int, int, int],
                 common_limits: list[tuple[float, float]]) -> Image.Image:
    rgb = np.stack([stack[i] for i in rgb_indices], axis=-1).astype(np.float32)
    out = np.zeros(rgb.shape, dtype=np.uint8)
    for channel, (lo, hi) in enumerate(common_limits):
        if not np.isfinite(lo + hi) or hi <= lo:
            hi = lo + 1.0
        scaled = (rgb[..., channel] - lo) / (hi - lo)
        out[..., channel] = np.clip(scaled * 255, 0, 255).astype(np.uint8)
    # Transparent invalid pixels reveal the base map instead of looking like dark land.
    rgba = np.dstack([out, valid.astype(np.uint8) * 255])
    return Image.fromarray(rgba)


def _rgb_indices(count: int) -> tuple[int, int, int]:
    # OSCD ordering starts B01,B02,B03,B04; a typical 4-band S2 stack is B02,B03,B04,B08.
    if count >= 13:
        return (3, 2, 1)
    if count == 4:
        return (2, 1, 0)
    if count >= 3:
        return (0, 1, 2)
    raise ValueError("Preview requires at least 3 bands; the detector may use more.")


def _render_pair(before: np.ndarray, after: np.ndarray, valid: np.ndarray, directory: Path) -> tuple[str, str]:
    idx = _rgb_indices(before.shape[0])
    limits = []
    for band_idx in idx:
        values = np.concatenate([before[band_idx][valid], after[band_idx][valid]])
        values = values[np.isfinite(values)]
        if not values.size:
            limits.append((0.0, 1.0))
        else:
            lo, hi = np.percentile(values, [2, 98])
            limits.append((float(lo), float(hi)))
    bimg = _preview_rgb(before, valid, idx, limits)
    aimg = _preview_rgb(after, valid, idx, limits)
    bp, ap = directory / "before.png", directory / "after.png"
    bimg.save(bp, optimize=True)
    aimg.save(ap, optimize=True)
    return str(bp.relative_to(OUTPUT_ROOT)), str(ap.relative_to(OUTPUT_ROOT))


def _raster_bounds(path: Path) -> list[list[float]]:
    with rasterio.open(path) as src:
        t = src.transform
        # The simple image overlay represents a north-up rectangular raster.
        if abs(t.b) > 1e-9 or abs(t.d) > 1e-9:
            raise ValueError("Rotated raster grids are not supported by the web preview; warp to north-up first.")
        bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
    left, bottom, right, top = bounds
    return [[float(bottom), float(left)], [float(top), float(right)]]


def _vectorize_prediction_mask(mask_path: Path, destination: Path, task: str, max_regions: int = 500) -> dict:
    """Vectorize real mask components; WGS84 for georeferenced rasters, pixels otherwise.

    Only the largest ``max_regions`` components are retained in memory. Ordinary image
    coordinates are explicitly labelled image-space and are not geographic GeoJSON.
    """
    import heapq
    if mask_path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            crs = src.crs
            nodata = src.nodata
            transform = src.transform if crs else from_origin(0, mask.shape[0], 1, 1)
    else:
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L"))
        transform = from_origin(0, mask.shape[0], 1, 1)
        crs = None
        nodata = 255
    class_map = {"urban_change": {1: "candidate_change"}, "flood_extent": {1: "water"},
                 "building_damage": {1: "no_damage", 2: "minor_damage", 3: "major_damage", 4: "destroyed"}}[task]
    geod = Geod(ellps="WGS84")
    heap: list[tuple[float, int, dict, dict]] = []
    counter = 0
    total_components = 0
    for class_id, class_name in class_map.items():
        binary = (mask == class_id)
        if nodata is not None: binary &= (mask != nodata)
        if not binary.any(): continue
        for geom_json, _value in shapes(binary.astype(np.uint8), mask=binary, transform=transform, connectivity=8):
            geom_obj = shape(geom_json)
            if geom_obj.is_empty or not geom_obj.is_valid: continue
            total_components += 1
            if crs:
                wgs_json = transform_geom(crs, "EPSG:4326", geom_json, precision=7)
                wgs_geom = shape(wgs_json)
                area_value = abs(float(geod.geometry_area_perimeter(wgs_geom)[0]))
                centroid = wgs_geom.centroid
                props = {"class_id": int(class_id), "class_name": class_name, "area_m2": area_value, "coordinate_space": "EPSG:4326"}
                summary = {"class_id": int(class_id), "class_name": class_name, "area_m2": area_value, "centroid_lat": float(centroid.y), "centroid_lng": float(centroid.x), "coordinate_space": "geographic"}
                feature_geom = wgs_json
            else:
                area_value = float(geom_obj.area)
                centroid = geom_obj.centroid
                props = {"class_id": int(class_id), "class_name": class_name, "area_pixels": area_value, "coordinate_space": "image_pixels"}
                summary = {"class_id": int(class_id), "class_name": class_name, "area_pixels": area_value, "centroid_y": float(centroid.y), "centroid_x": float(centroid.x), "coordinate_space": "image_pixels"}
                feature_geom = geom_json
            counter += 1
            feature = {"type": "Feature", "geometry": feature_geom, "properties": props}
            entry = (area_value, counter, feature, summary)
            if len(heap) < max_regions:
                heapq.heappush(heap, entry)
            elif area_value > heap[0][0]:
                heapq.heapreplace(heap, entry)
    kept = sorted(heap, key=lambda item: item[0], reverse=True)
    features, summaries = [], []
    for index, (_area, _counter, feature, summary) in enumerate(kept, start=1):
        region_id = f"R{index:04d}"
        feature["properties"]["id"] = region_id
        summary["id"] = region_id
        features.append(feature); summaries.append(summary)
    truncated = total_components > len(kept)
    coordinate_space = "EPSG:4326" if crs else "image_pixels_not_geographic"
    collection = {"type": "FeatureCollection", "name": task + "_prediction_regions", "coordinate_space": coordinate_space, "features": features}
    destination.write_text(json.dumps(collection, indent=2), encoding="utf-8")
    return {"regions": summaries, "region_count": len(summaries), "truncated": truncated, "total_components": total_components, "coordinate_space": "geographic" if crs else "image_pixels", "regions_path": str(destination)}


def _preview_array(source: Path, task: str, max_side: int = 1600) -> np.ndarray:
    """Read a small RGB-like display sample with aspect ratio preserved."""
    if source.suffix.lower() in {".tif", ".tiff", ".vrt"}:
        with rasterio.open(source) as ds:
            scale = min(1.0, max_side / max(ds.width, ds.height))
            out_w = max(1, int(round(ds.width * scale)))
            out_h = max(1, int(round(ds.height * scale)))
            if ds.count >= 13:
                indexes = [4, 3, 2]
            elif ds.count >= 3:
                indexes = [1, 2, 3]
            else:
                indexes = [1, 1, 1]
            values = ds.read(indexes, out_shape=(3, out_h, out_w), masked=True,
                             resampling=rasterio.enums.Resampling.bilinear).astype(np.float32).filled(np.nan)
            return np.moveaxis(values, 0, -1)
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32)


def _stretch_preview(array: np.ndarray, limits: list[tuple[float, float]] | None = None) -> tuple[Image.Image, list[tuple[float, float]]]:
    data = np.asarray(array, dtype=np.float32)
    if data.ndim != 3 or data.shape[-1] != 3:
        raise ValueError("Preview sample must have three display channels.")
    selected_limits = []
    rgb = np.zeros(data.shape, dtype=np.uint8)
    for channel in range(3):
        vals = data[..., channel][np.isfinite(data[..., channel])]
        if limits is None:
            if vals.size:
                lo, hi = (float(v) for v in np.percentile(vals, [2, 98]))
            else:
                lo, hi = 0.0, 1.0
            if hi <= lo: hi = lo + 1.0
        else:
            lo, hi = limits[channel]
        selected_limits.append((lo, hi))
        scaled = (np.nan_to_num(data[..., channel], nan=lo) - lo) / max(hi - lo, 1e-6)
        rgb[..., channel] = np.clip(scaled * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB"), selected_limits


def _preview_file(source: Path, destination: Path, task: str) -> None:
    """Create a visual-only preview; preserve aspect ratio; never use as model input."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    image, _ = _stretch_preview(_preview_array(source, task))
    image.save(destination, format="PNG", optimize=True)


def _preview_pair(before: Path, after: Path, before_dest: Path, after_dest: Path, task: str) -> None:
    """Use matching display limits for before/after previews to avoid render-induced contrast."""
    before_arr = _preview_array(before, task)
    after_arr = _preview_array(after, task)
    limits = []
    for channel in range(3):
        both = np.concatenate((before_arr[..., channel][np.isfinite(before_arr[..., channel])],
                               after_arr[..., channel][np.isfinite(after_arr[..., channel])]))
        if both.size:
            lo, hi = (float(v) for v in np.percentile(both, [2, 98]))
            if hi <= lo: hi = lo + 1.0
        else:
            lo, hi = 0.0, 1.0
        limits.append((lo, hi))
    bimg, _ = _stretch_preview(before_arr, limits)
    aimg, _ = _stretch_preview(after_arr, limits)
    before_dest.parent.mkdir(parents=True, exist_ok=True)
    bimg.save(before_dest, format="PNG", optimize=True)
    aimg.save(after_dest, format="PNG", optimize=True)


def _geo_bounds(path: Path | None) -> list[list[float]] | None:
    if path is None or path.suffix.lower() not in {".tif", ".tiff", ".vrt"}:
        return None
    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                return None
            if abs(src.transform.b) > 1e-9 or abs(src.transform.d) > 1e-9:
                return None
            left, bottom, right, top = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        if not all(np.isfinite([left, bottom, right, top])) or right <= left or top <= bottom:
            return None
        return [[float(bottom), float(left)], [float(top), float(right)]]
    except Exception:
        return None

def _mask_preview(mask_path: Path, destination: Path, task: str) -> None:
    """Convert a real prediction mask into a transparent PNG layer for the map."""
    if mask_path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            invalid = (mask == 255)
    else:
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L"))
        invalid = mask == 255
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    palette = {
        "urban_change": {1: (239, 68, 68, 180)},
        "flood_extent": {1: (0, 210, 255, 180)},
        "building_damage": {1: (50, 205, 80, 180), 2: (255, 215, 0, 190), 3: (255, 135, 25, 200), 4: (235, 35, 45, 210)},
    }.get(task, {})
    for class_id, color in palette.items():
        rgba[mask == class_id] = color
    rgba[invalid] = (0, 0, 0, 0)
    Image.fromarray(rgba, mode="RGBA").save(destination, optimize=True)


@app.post("/api/analyze")
async def analyze(
    before_file: UploadFile = File(...),
    after_file: UploadFile = File(...),
    histogram_match: bool = Form(True),
    min_object_size: int = Form(8),
    opening_radius: int = Form(1),
) -> dict:
    """Explicitly labelled classical baseline; no learned-model claims are made."""
    if min_object_size < 1 or opening_radius < 0:
        raise HTTPException(400, "min_object_size must be >= 1 and opening_radius >= 0")
    _cleanup_expired_sessions()
    session_id = uuid.uuid4().hex
    directory = OUTPUT_ROOT / session_id
    directory.mkdir(parents=True, exist_ok=False)
    budget = UploadBudget()
    try:
        before_path, after_path = directory / "before.tif", directory / "after.tif"
        await _save_upload(before_file, before_path, allowed={".tif", ".tiff"}, budget=budget)
        await _save_upload(after_file, after_path, allowed={".tif", ".tiff"}, budget=budget)
        _validate_uploaded_asset(before_path, "urban_change", "before image")
        _validate_uploaded_asset(after_path, "urban_change", "after image")
        pair = load_geotiff_pair(before_path, after_path)
        predicted, magnitude, threshold = run_baseline(
            pair.before, pair.after, valid_mask=pair.valid_mask,
            histogram_match=histogram_match, min_object_size=min_object_size,
            opening_radius=opening_radius,
        )
        mask_path = directory / "change_mask.tif"
        profile = pair.profile.copy()
        for key in ("photometric", "interleave", "predictor"):
            profile.pop(key, None)
        profile.update(count=1, dtype="uint8", nodata=255, compress="lzw")
        with rasterio.open(mask_path, "w", **profile) as dst:
            dst.write(np.where(pair.valid_mask, predicted.astype(np.uint8), 255).astype(np.uint8), 1)
        gdf = vectorize_change_mask(predicted, magnitude, pair.profile["transform"], pair.crs)
        geojson_path = directory / "change_polygons.geojson"
        gpkg_path = directory / "change_polygons.gpkg"
        write_vectors(gdf, geojson_path)
        write_vectors(gdf, gpkg_path)
        before_rel, after_rel = _render_pair(pair.before, pair.after, pair.valid_mask, directory)
        bounds = _raster_bounds(before_path)
        valid_count = int(pair.valid_mask.sum())
        
        regions = []
        for idx, row in gdf.iterrows():
            regions.append({
                "id": str(idx),
                "class_name": "candidate_change",
                "class_id": 1,
                "area_m2": float(row.get("area_m2", 0.0)),
                "centroid_lng": float(row.get("centroid_lon", 0.0)),
                "centroid_lat": float(row.get("centroid_lat", 0.0)),
                "coordinate_space": "geographic"
            })
            
        return {
            "session_id": session_id,
            "mode": "classical_baseline",
            "message": "Classical baseline completed. These are candidate changes, not verified damage or illegal encroachment.",
            "bounds": bounds,
            "before_image_url": _artifact_url(OUTPUT_ROOT / Path(before_rel.replace(chr(92), "/"))),
            "after_image_url": _artifact_url(OUTPUT_ROOT / Path(after_rel.replace(chr(92), "/"))),
            "geojson_url": _artifact_url(geojson_path),
            "mask_url": _artifact_url(mask_path),
            "regions": regions,
            "regions_coordinate_space": "geographic",
            "region_count": len(regions),
            "metrics": {
                "threshold_otsu": float(threshold),
                "valid_pixels": valid_count,
                "predicted_change_pixels": int(predicted.sum()),
                "predicted_change_fraction": float(predicted.sum() / valid_count) if valid_count else 0.0,
                "polygon_count": int(len(gdf)),
                "histogram_matching": bool(histogram_match),
                "priority_tiers_are_heuristic": True,
            },
        }
    except HTTPException:
        import shutil
        shutil.rmtree(directory, ignore_errors=True)
        raise
    except (ValueError, rasterio.errors.RasterioError) as exc:
        import shutil
        shutil.rmtree(directory, ignore_errors=True)
        logger.info("Classical baseline input rejected (%s)", type(exc).__name__)
        raise HTTPException(422, "The uploaded rasters are incompatible or unreadable. Verify CRS, dimensions, grid alignment and GeoTIFF validity.") from exc
    except Exception as exc:
        import shutil
        shutil.rmtree(directory, ignore_errors=True)
        request_id = uuid.uuid4().hex
        logger.exception("Classical baseline failed (request_id=%s)", request_id)
        raise HTTPException(500, detail={"status": "BASELINE_FAILED", "request_id": request_id,
                                         "message": "Classical baseline failed unexpectedly. Check server logs using the request ID."}) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=False)
