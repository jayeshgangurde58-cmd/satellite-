# ChronoTerra FastAPI Backend + ML Prediction API

The FastAPI application provides file inspection and pair validation under `/api`, and mounts the task-specific prediction API under `/ml`. It does not generate synthetic detections. If a trained checkpoint is missing, `/ml/api/predict` returns `MODEL_NOT_AVAILABLE`.

## Windows PowerShell setup

Use Python 3.11 or 3.12 for the broadest geospatial package compatibility.

```powershell
cd ChronoTerra_Backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run.py
```

The server runs at `http://127.0.0.1:8000`.

- `GET /api/health`
- `POST /api/inspect-image`
- `POST /api/validate-pair`
- `GET /ml/api/models/status`
- `POST /ml/api/preflight`
- `POST /ml/api/predict`
- API docs: `http://127.0.0.1:8000/docs`

## Tests

```powershell
python -m pytest -q
```

## Training and checkpoints

See `prediction_module/README.md` and `prediction_module/docs/DATASET_SETUP.md`. The model architectures and training scripts are included, but genuine datasets and trained weights are not bundled. Do not interpret checkpoint existence as proof of accuracy; inspect training summaries and held-out test reports.

## CORS and upload limits

The default CORS allow-list is localhost Vite development. Set `CHRONOTERRA_CORS_ORIGINS` in the process environment for another origin. `CHRONOTERRA_MAX_UPLOAD_MB` defaults to 150 MB, aggregate request limit to 256 MB, and session retention to 24 hours. Artifact URLs are HMAC-signed, expiring bearer capabilities. Configure a persistent 32+ byte `CHRONOTERRA_ARTIFACT_SIGNING_SECRET` before launching to keep URLs valid across restarts; for a one-time local value in PowerShell, run `$env:CHRONOTERRA_ARTIFACT_SIGNING_SECRET = (py -c "import secrets; print(secrets.token_urlsafe(48))")` and launch the server in the same shell. A signed URL is not user-scoped authorization. Review upload limits, retention, CORS, reverse-proxy limits, worker quotas, rate limiting and authentication before public deployment.
