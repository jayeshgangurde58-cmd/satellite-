# ChronoTerra — GeoAI Change & Impact Analysis

This folder combines the GEOSATELITE-234 frontend, FastAPI inspection backend, and the task-specific ML prediction/training module. The visual UI is retained, while fake prediction fallbacks and random region generation have been removed.

## Current capability

- React-free Vite + TypeScript frontend with the original UI layout.
- FastAPI image inspection and a mounted ML API at `/ml`.
- Separate task contracts for urban change, flood-water extent, and building damage.
- Preflight QA, optional GeoJSON AOI, optional paired Sentinel-2 SCL masks, and explicit band-order acknowledgement.
- Actual mask preview, mask-derived polygons, per-class pixel counts, raster/vector downloads, uncertainty proxy, and JSON QA report when a valid trained checkpoint is installed.
- Google Places autocomplete is optional and requires a configured restricted API key.
- No random detections, invented metadata defaults, or mock prediction fallback.

**Important:** Model architectures and training code are included, but no genuine trained model checkpoints or labelled dataset archives are bundled. Until you train and evaluate the models, the prediction API intentionally returns `MODEL_NOT_AVAILABLE` rather than fake output.

## Start the backend (Windows PowerShell)

Use Python 3.11 or 3.12 in a clean virtual environment for geospatial dependency compatibility.

```powershell
cd ChronoTerra_Backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run.py
```

Backend endpoints:

- `http://127.0.0.1:8000/api/health`
- `http://127.0.0.1:8000/api/inspect-image`
- `http://127.0.0.1:8000/api/validate-pair`
- `http://127.0.0.1:8000/ml/api/models/status`
- `http://127.0.0.1:8000/ml/api/preflight`
- `http://127.0.0.1:8000/ml/api/predict`
- Swagger docs: `http://127.0.0.1:8000/docs`

## Start the frontend

Open a second PowerShell window at the project root:

```powershell
npm ci
Copy-Item .env.example .env
npm run dev
```

Open the local Vite URL shown in the terminal (normally `http://localhost:5173`). Keep the backend running. `VITE_API_BASE_URL` defaults to `http://127.0.0.1:8000/api`.

### Optional Google Places location picker

1. Create a Google Cloud project and enable the Maps JavaScript API and Places API.
2. Add your key to `.env` as `VITE_GOOGLE_MAPS_API_KEY=...`.
3. Restrict the key to the local development origin and only the APIs you need; use a separate restricted production key later.
4. Restart Vite.

A selected place point is checked against raster bounds when those bounds exist. It does **not** prove image identity or define a precise AOI. Upload a GeoJSON AOI polygon for AOI-limited analysis.

## Input contracts

| Task | Required input | Important limitation |
|---|---|---|
| Urban change | Two aligned, georeferenced 13-band Sentinel-2 GeoTIFFs in the documented OSCD band order | RGB JPG/PNG is not compatible; registration, band order and scaling must be verified. Optional paired SCL masks. |
| Flood extent | One georeferenced two-band Sentinel-1 VV/VH GeoTIFF in the checkpoint's preprocessing scale | Predicts water extent, not before/after flood change or building damage. |
| Building damage | Aligned pre/post RGB image pair using the trained xBD-style input convention | Output class definitions depend on the trained label mapping; RGB files without georeferencing produce image-space polygons only. |

The same-area checkbox and a Google Places point are user-provided context, not proof of geospatial alignment. The system distinguishes footprint overlap, common valid coverage, content registration, sensor compatibility and AOI coverage. It does not blindly warp imagery to make a pair pass.

## Train the real models

Read `ChronoTerra_Backend/prediction_module/docs/DATASET_SETUP.md` and verify each dataset's source/license terms. Put the genuine datasets into the documented folders, then:

```powershell
cd ChronoTerra_Backend\prediction_module
.\Train-All-Models.ps1 -Model urban
.\Train-All-Models.ps1 -Model flood
.\Train-All-Models.ps1 -Model damage
```

Only run `-Model all` after all three dataset audits pass. Each workflow trains, selects checkpoints using validation data, and evaluates on a held-out test split. Inspect reports and failure cases before deploying checkpoints. Expected default checkpoints are under `prediction_module/outputs/models/`.

A checkpoint does **not** enable inference by itself. Follow [`ChronoTerra_Backend/prediction_module/docs/MODEL_DEPLOYMENT_GATES.md`](ChronoTerra_Backend/prediction_module/docs/MODEL_DEPLOYMENT_GATES.md) to create hash-bound deployment evidence. It creates a pending manifest first; inference stays disabled until a reviewer explicitly approves real held-out results that meet documented thresholds.

## Tests and build checks

Backend plus prediction module tests:

```powershell
cd ChronoTerra_Backend
python -m pytest -q
```

Frontend type check and production build:

```powershell
cd ..
npm run build
```

The browser demo images are RGB preview samples only; they are intentionally incompatible with the 13-band urban model and the two-band SAR flood model. They must never be treated as actual model predictions.

## Security / deployment

The backend defaults CORS to local Vite origins. APIs enforce declared and streamed request-body limits, bounded per-file reads, conservative task-specific pixel caps for full-array inference, allowlisted result artifacts, HMAC-signed expiring artifact URLs and access-time session expiry. Set a stable random `CHRONOTERRA_ARTIFACT_SIGNING_SECRET` in the backend process environment for links to survive a restart. In PowerShell, create one with `$env:CHRONOTERRA_ARTIFACT_SIGNING_SECRET = (py -c "import secrets; print(secrets.token_urlsafe(48))")` before starting the backend (use the same long-lived secret on each restart). If the variable is omitted, a temporary signing key is generated on startup and old links stop working after restart. These links remain bearer capabilities, not user authorization; add authentication/access control before public multi-user hosting. Whole-scene inference remains bounded by conservative pixel limits but is not yet windowed; enforce worker, time, memory and rate limits for deployment. Configure `CHRONOTERRA_CORS_ORIGINS` explicitly and enforce upload limits at the reverse proxy. Runtime job files, real `.env` files, datasets and trained weights are excluded by `.gitignore`.

## Final product design and release status

See [`FINAL_PRODUCT_AND_ENGINEERING_BLUEPRINT.md`](FINAL_PRODUCT_AND_ENGINEERING_BLUEPRINT.md) for the recommended page-by-page UX, urban change/encroachment screening logic, evidence presentation, region schema, quality metrics, and release acceptance criteria. See [`FINAL_RELEASE_NOTES.md`](FINAL_RELEASE_NOTES.md) for the final hardening pass and what is still unverified.
