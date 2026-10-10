from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

# Load the task-specific ML prediction sub-application from the bundled module.
PREDICTION_ROOT = Path(__file__).resolve().parents[1] / "prediction_module"
if str(PREDICTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PREDICTION_ROOT))

from .inspection import compare_pair, inspect_image_file
from api.request_limits import RequestBodyLimitMiddleware

APP_VERSION = "0.1.0"
MAX_UPLOAD_MB = int(os.getenv("CHRONOTERRA_MAX_UPLOAD_MB", "150"))
MAX_REQUEST_UPLOAD_MB = int(os.getenv("CHRONOTERRA_MAX_REQUEST_UPLOAD_MB", "256"))
if MAX_UPLOAD_MB < 1 or MAX_REQUEST_UPLOAD_MB < MAX_UPLOAD_MB:
    raise RuntimeError("CHRONOTERRA_MAX_UPLOAD_MB must be positive and CHRONOTERRA_MAX_REQUEST_UPLOAD_MB must be at least as large")
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_REQUEST_UPLOAD_BYTES = MAX_REQUEST_UPLOAD_MB * 1024 * 1024

app = FastAPI(
    title="ChronoTerra Image Inspection API",
    description=(
        "Real file inspection and before/after compatibility checks for ChronoTerra. "
        "This service does not currently run trained ML predictions."
    ),
    version=APP_VERSION,
)
app.add_middleware(RequestBodyLimitMiddleware, max_body_bytes=MAX_REQUEST_UPLOAD_BYTES + 2 * 1024 * 1024)


raw_origins = os.getenv("CHRONOTERRA_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def bound_declared_request_size(request: Request, call_next):
    """Reject oversized declared multipart bodies before framework parsing/spooling."""
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


import tempfile

async def read_upload_limited(upload: UploadFile, limit_bytes: int = MAX_UPLOAD_BYTES) -> str:
    fd, path = tempfile.mkstemp(suffix=".tmp", prefix="chronoterra_upload_")
    written = 0
    try:
        with os.fdopen(fd, "wb") as target:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit_bytes:
                    if limit_bytes == MAX_UPLOAD_BYTES:
                        detail = f"File exceeds the configured {MAX_UPLOAD_MB} MB upload limit."
                    else:
                        detail = "Combined uploaded file contents exceed the configured aggregate request limit."
                    raise HTTPException(status_code=413, detail=detail)
                target.write(chunk)
        if written == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        return path
    except Exception:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        raise


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "chronoterra-image-inspection",
        "version": APP_VERSION,
        "max_upload_mb": MAX_UPLOAD_MB,
        "prediction_service": "mounted" if prediction_app is not None else "unavailable",
        "prediction_status_endpoint": "/ml/api/models/status" if prediction_app is not None else None,
    }


@app.post("/api/inspect-image")
async def inspect_image(file: UploadFile = File(...)) -> dict[str, object]:
    request_id = str(uuid4())
    temp_path = await read_upload_limited(file)
    try:
        report = inspect_image_file(temp_path, file.filename or "upload")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
    report["request_id"] = request_id
    return report


@app.post("/api/validate-pair")
async def validate_pair(
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...),
    analysis_mode: Literal["urban", "disaster"] = Form("urban"),
    location_name: str | None = Form(None),
    before_date: str | None = Form(None),
    after_date: str | None = Form(None),
    same_area_confirmed: bool = Form(False),
) -> dict[str, object]:
    request_id = str(uuid4())
    before_path = await read_upload_limited(before_image)
    before_size = os.path.getsize(before_path)
    remaining_budget = MAX_REQUEST_UPLOAD_BYTES - before_size
    if remaining_budget <= 0:
        os.remove(before_path)
        raise HTTPException(status_code=413, detail="Combined uploaded file contents exceed the configured aggregate request limit.")
    after_limit = min(MAX_UPLOAD_BYTES, remaining_budget)
    after_path = None
    try:
        after_path = await read_upload_limited(after_image, after_limit)
        after_size = os.path.getsize(after_path)
        result = compare_pair(
            before_path,
            before_image.filename or "before-image",
            after_path,
            after_image.filename or "after-image",
            analysis_mode=analysis_mode,
            before_date=before_date,
            after_date=after_date,
            location_name=location_name,
            same_area_confirmed=same_area_confirmed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if os.path.exists(before_path):
            try:
                os.remove(before_path)
            except OSError:
                pass
        if after_path and os.path.exists(after_path):
            try:
                os.remove(after_path)
            except OSError:
                pass
    result["request_id"] = request_id
    result["uploaded_file_sizes"] = {
        "before_image_bytes": before_size,
        "after_image_bytes": after_size,
    }
    return result


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):  # type: ignore[no-untyped-def]
    # Avoid exposing stack traces or raw decoder details to clients.
    return JSONResponse(
        status_code=500,
        content={"detail": "Unexpected inspection-service error. Check backend logs for details."},
    )

@app.post("/api/jobs")
async def create_analysis_job():
    # The previous implementation fabricated random regions. Keep the legacy route explicit
    # rather than silently returning mock results; the frontend uses /ml/api/predict now.
    raise HTTPException(410, detail={
        "status": "LEGACY_ENDPOINT_RETIRED",
        "message": "The old job endpoint generated synthetic regions and has been retired. Use /ml/api/preflight and /ml/api/predict for task-specific real inference."
    })

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    raise HTTPException(status_code=410, detail={
        "status": "LEGACY_ENDPOINT_RETIRED",
        "message": "Legacy job records are disabled. Use the task-specific /ml/api/predict workflow and its session artifact URLs.",
    })


@app.get("/api/jobs/{job_id}/artifacts/{artifact_name}")
async def get_job_artifact(job_id: str, artifact_name: str):
    raise HTTPException(status_code=410, detail={
        "status": "LEGACY_ENDPOINT_RETIRED",
        "message": "Legacy job artifacts are disabled; they are not an authoritative ChronoTerra result source.",
    })


# Real task-specific prediction API. Checkpoints are optional at startup; requests return
# MODEL_NOT_AVAILABLE until the matching model is trained/evaluated and installed.
try:
    from api.app import app as prediction_app
    app.mount("/ml", prediction_app, name="ml-prediction")
except Exception as exc:  # Keep inspection API diagnosable if optional ML deps are missing.
    prediction_app = None
    print(f"ChronoTerra ML API could not be mounted: {type(exc).__name__}: {exc}")
