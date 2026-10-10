# ChronoTerra — Patch & Independent Verification Report

**Audit date:** 10 October 2026  
**Package:** `ChronoTerra_Complete_Project`  
**Purpose:** Record the source-level hardening and the tests actually run. This is not a claim that the project is perfectly bug-free or that its ML accuracy has been proven.

## Executive result

The latest source tree includes fixes for the concrete defects from the previous independent audit and extra regression checks. The current verifiable state is **90 Python tests passed**, **67 Python files compiled**, **strict TypeScript source audit passed with TypeScript 5.8.3**, and the final API smoke checks passed. The production Vite bundle and browser end-to-end journey remain unverified because clean dependency installation cannot be completed in this isolated environment. Real model weights, datasets and held-out model metrics are not bundled; learned predictions intentionally remain unavailable until those are produced and approved.

## Fixes in this patch pass

### API privacy and artifact access
- Removed absolute local checkpoint/input paths from public model status, preflight and prediction responses; public QA JSON is path-redacted.
- Replaced general static exposure with an allowlist of derivative artifact filenames.
- Added HMAC-signed, expiring artifact URLs, signature/tamper/expiry checks, private/no-store and nosniff headers, random per-analysis session IDs, and access-time retention cleanup.
- Raw input uploads and internal reports are not allowed by the artifact-serving route. Tests verify direct input and internal-report requests receive 404.
- Added `CHRONOTERRA_ARTIFACT_SIGNING_SECRET`. If not configured, the process makes a random temporary key, so URLs issued before a restart become invalid. For convenience and consistent behavior, set a strong stable secret in the backend process environment. Signed URLs remain bearer capabilities until expiry; they are not per-user authorization.

### Upload and processing resource controls
- Enforced request-body size limits for declared `Content-Length` and actual streamed request chunks, including requests without a length header.
- Kept per-file and aggregate upload limits; the classical baseline now uses bounded upload reads.
- Added axis, band and task-specific pixel ceilings because the current predictor still materializes whole-raster arrays. This mitigates risk but is not a substitute for windowed inference or worker-level memory quotas.
- Moved inference work through a threadpool and serialized expensive model work in the prediction service.

### Dates, task routing and geospatial boundary screening
- Before/after dates must be chronologically ordered when both are supplied; reversed or equal dates are rejected.
- Switching prediction task clears incompatible files, masks, boundary files and stale metadata. Starting a new analysis clears previous-run fields.
- AOI clipping is separate from the optional reference-zone boundary. Reference polygons are validated as RFC 7946 WGS84 polygon geometry, with size and feature-count limits.
- For geographic, valid candidate polygons, the reference-zone workflow calculates candidate overlap area/fraction and near-boundary distance. It labels the source `USER_SUPPLIED_NOT_INDEPENDENTLY_VERIFIED` and explicitly disclaims legal conclusions.
- Invalid, image-space or unsupported geometry must not be interpreted as a legal boundary match.

### Model readiness and truthful UI
- A checkpoint alone no longer unlocks prediction. The runtime expects a task-matched hash-bound deployment manifest, metrics derived from a held-out-by-scene test report, threshold-satisfying metrics, matching hashes and an explicit named human review.
- `tools/create_deployment_manifest.py` writes a pending manifest by default. It does not synthesize metrics or itself prove that dataset provenance is honest.
- The frontend checks model approval state before the user can submit the actual prediction; missing/blocked-model errors remain visible and no fake result is substituted.
- Model/data pages show unavailable or not-evaluated states rather than measured-looking placeholder metrics.
- The history page states that it is browser-session only; server-side persistent history is not implemented.

### Frontend lifecycle and injection handling
- Removed the old inline raw JavaScript error overlay from `index.html` and retained the ChronoTerra favicon.
- Removed unreferenced Vite starter/demo assets and obsolete legacy modules after searching imports/references.
- Leaflet map instances are removed before whole-root DOM rerenders. The current map view is preserved for the same result, stale asynchronous callbacks are guarded, popup text is escaped, and temporary preview object URLs are revoked when their source is replaced or removed.
- Added a visible fallback when Leaflet fails to load and made the before/after comparison slider support pointer and keyboard interaction.

## Verification performed

### Automated test suite
Command:
```powershell
cd ChronoTerra_Backend
python -m pytest -q
```
Result: **90 passed, 4 warnings in 8.12 seconds**. The four warnings are existing PyProj/NumPy deprecations surfaced during coordinate transformation tests; they are not test failures and should be addressed before a future NumPy/PyProj compatibility upgrade.

New/regression scenarios include model-status path redaction, reversed dates, preflight path redaction, declared and streamed request body limits, boundary overlap/proximity classification, deployment manifest hashes/metrics/review gates, artifact access allowlisting, signed URL required, signature tampering, expiry, and session retention.

### Python syntax
All **67 shipped Python source files** passed `python -m py_compile`.

### TypeScript
All frontend `.ts` files passed a strict no-emit TypeScript audit with the globally available **TypeScript 5.8.3** and a temporary config that intentionally omits Vite-specific type declarations because dependencies could not be installed. This is useful source type evidence, not a substitute for the official project build.

### API smoke checks
Using FastAPI `TestClient` against the mounted application:
- `/api/health`: HTTP 200.
- `/ml/api/models/status`: HTTP 200 with no absolute filesystem paths in the response.
- Urban change, flood extent, and building damage each report `MODEL_NOT_AVAILABLE` because their checkpoints are absent.
- A prediction request without an installed model fails closed with HTTP 503 and a path-safe error.
- A signed artifact URL worked through the real parent mount `/ml/outputs/...`; the unsigned equivalent returned HTTP 404. Response included `Cache-Control: private, no-store`. A dedicated regression test now covers this public URL shape.

### Package checks
- `package.json` and `package-lock.json` names were checked for consistency.
- Source scans found no remaining `window.onerror` / `onunhandledrejection` overlay, no “Page not implemented” developer stub, and no stale test-count text after documentation update.
- No actual `.env`, `.db`, `.sqlite`, `.pt`, `.pth` or `.onnx` files were found in the release tree before cleanup.

## Checks blocked or not performed

### Production Vite build — NOT VERIFIED
A clean install was attempted with:
```bash
npm ci --offline --include=optional --fetch-retries=0 --fetch-timeout=10000
```
It failed with `ENOTCACHED` because the package tarball `vite-8.3.4.tgz` was not available in the local npm cache. An earlier online install attempt timed out in this environment. Running `npm run build` without a complete install cannot establish whether the real production bundle passes. On the target Windows machine, run `npm ci` followed by `npm run build`.

### Browser end-to-end journey — NOT PERFORMED
There was no successful real browser test of upload → preflight → approved prediction → interactive result map → downloads. CDN and basemap availability, browser interactions, and actual export content still need testing.

### Real ML/model metrics — NOT AVAILABLE
No genuine labelled datasets or trained checkpoints are bundled in the ZIP, and this audit did not train a model or produce new held-out metrics. For that reason, the three learned prediction tasks return `MODEL_NOT_AVAILABLE`. Do not present the delivered package as having validated predictions.

### GIS export verification — NOT PERFORMED
Vector/raster export unit tests run, but generated artifacts have not been inspected independently in QGIS against a real georeferenced scene. Boundary screening indicates spatial overlap/proximity only and never legal encroachment proof.

## Important residual risks

1. **No guarantee of zero defects:** automated tests cover known cases, not every input, OS, browser, scene, or deployment environment.
2. **Whole-array model inference:** limits reduce risk, but windowed reads and writes plus process-level memory/time controls remain required for large scenes.
3. **Capability URL model:** anyone who receives a signed artifact URL can access that allowlisted artifact until expiry. Add user-level authentication and authorization before public/multi-user service.
4. **Model evidence integrity:** hashes catch modified checkpoints/reports, but a local user who can replace both the evaluation report and manifest may still forge local evidence. Require a separate evaluator/reviewer and immutable experiment records for a serious release.
5. **Scientific validity:** the gate is a deployment hygiene check, not a certificate of performance. Review split provenance, leakage, annotation quality, calibration and per-scene failures independently.
6. **Third-party browser dependencies:** map rendering relies on a CDN/tile provider and needs a no-network/failure path test on the target deployment.
7. **Legacy app contract:** old `/api/jobs` routes return HTTP 410 on purpose. Integrations that still call them must switch to `/ml/api/preflight` and `/ml/api/predict`.

## Target machine acceptance checklist

- [ ] Clean Python 3.11/3.12 venv; install backend requirements.
- [ ] Set `CHRONOTERRA_ARTIFACT_SIGNING_SECRET` in the backend process environment and configure correct CORS origins.
- [ ] Start backend; confirm `/api/health` and `/ml/api/models/status`.
- [ ] Clean `npm ci`; run `npm run build`.
- [ ] Run the frontend and test task switching, cleared prior inputs, date rejection, preview warnings, model-blocked flow, map controls and browser console.
- [ ] Verify uploaded source and internal report URLs remain inaccessible; test unsigned, tampered and expired links.
- [ ] Train/evaluate one real task using genuine, licensed data and a scene/geography-disjoint held-out test set; inspect false positives and per-scene metrics.
- [ ] Install a hash-bound reviewed model bundle and run true inference on compatible imagery.
- [ ] Verify GeoTIFF transform/CRS/nodata and GeoJSON WGS84 axis order independently in QGIS.
- [ ] Confirm the report language says candidate change/reference-zone screening, not confirmed illegal encroachment.

## Release judgment

**Improved and substantially better tested, but not production-ready or scientifically validated yet.** The code-level privacy, limits, date and geometry workflow fixes are covered by tests. The production frontend build, browser flow, deployment environment, real checkpoint training/evaluation and independent GIS validation are outstanding. A claim of “zero errors” would be unsupported by current evidence.
