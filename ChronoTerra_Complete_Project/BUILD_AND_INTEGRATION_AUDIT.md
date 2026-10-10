# ChronoTerra — Current build and integration audit

## Current source layout and supported workflows

- React-free Vite + TypeScript frontend connected to the root FastAPI image-inspection API and mounted task-specific ML API at `/ml`.
- Three distinct prediction contracts: urban change (paired 13-band Sentinel-2 GeoTIFF), flood-water extent (two-band Sentinel-1 VV/VH GeoTIFF), and paired RGB xBD-style building damage.
- Preflight, optional AOI clipping, optional paired SCL masks, optional user-supplied reference-zone GeoJSON for urban change, true mask-derived vector regions, scores/uncertainty artifacts and path-redacted QA reports.
- The reference-zone screen is intentionally labelled as user-supplied and not independently verified. An AOI polygon is not a legal parcel/ownership boundary.
- No random/mocked results in the current primary analysis path. Legacy `/api/jobs` creation, status and artifact retrieval routes return HTTP 410.

## Current verification evidence

- Backend plus prediction module: `python -m pytest -q` — **90 passed**, four existing PyProj/NumPy deprecation warnings.
- Python syntax compilation: **67 Python files passed** `py_compile`.
- TypeScript-only compile: passed using the available global TypeScript compiler with a temporary audit config (Vite module types omitted because dependencies are unavailable in this environment).
- ZIP integrity: checked at package time (result recorded in the patch report).
- Production Vite bundle: not verified; no installed `node_modules` in this packaging runtime and prior network dependency installation timed out.
- Real-dataset model inference: not validated; no real labelled datasets or trained checkpoints are bundled. All three learned prediction tasks should return `MODEL_NOT_AVAILABLE` until matching checkpoint + hash-bound held-out report + reviewed deployment manifest are installed.
- Full browser E2E and independent QGIS export inspection: not performed.

## Hardening implemented in the current pass

- Model status/preflight/prediction responses redact local filesystem paths.
- Result artifact serving is allowlisted; raw uploads and internal reports are denied. Sessions use 128-bit random identifiers, HMAC-signed expiring URLs, private/no-store headers, and access-time expiry checks. URLs are bearer capabilities, not account authorization; the HMAC secret should be persistent in deployments.
- ASGI body-size limit checks actual streamed bytes, even when `Content-Length` is missing; per-file and aggregate upload checks remain in handlers. The inspection baseline uses bounded file reads.
- Whole-array inference uses conservative task-specific pixel caps: urban 750,000 pixels; flood 2,000,000 pixels; building damage 1,500,000 pixels. This mitigates but does not eliminate memory risk.
- Chronological pair dates must be strictly ordered where both are supplied.
- Reference boundary comparison is separate from AOI clipping, validates WGS84 polygon GeoJSON, only runs where a geographic overlay is verified, and reports overlap/proximity as candidate-review evidence rather than a legal conclusion.
- Legacy job endpoints no longer expose stale result records or artifacts. Results-map instances are torn down on DOM replacement; stale image/fetch callbacks are guarded and popup values escaped. Browser interaction still needs E2E verification.
- Deployment manifest tooling derives task-specific screening metrics from actual held-out test-evaluator reports, binds checkpoint/normalized report/raw test report hashes, and defaults to a pending unapproved manifest.

## Important remaining risks

1. Whole-raster arrays are still loaded within the configured pixel limits. Windowed processing, worker isolation, concurrency caps and load tests are required before large-scene or public multi-user use.
2. Current artifact links are HMAC-signed, expiring bearer capabilities. They reduce casual discovery but are not per-user authorization; do not advertise this as a secure public multi-user deployment.
3. Model-manifest fields can be forged by someone who controls local files. The gate is a release guard, not cryptographic certification or an independent audit.
4. The building-damage deployment gate evaluates any-damage vs other classes; severity-specific precision/recall must be reviewed independently.
5. The same geographic bounds do not prove exact image co-registration. The preflight grid and registration checks remain relevant, and a human should review warnings before boundary screening.
6. Browser UX, Windows dependency resolution, Vite production build, real data, trained weights and GIS output export have not been fully verified.

## Release verdict

**Not production-ready yet.** The local code path is more defensively guarded and has broader regression coverage, but the frontend production build, end-to-end browser path and real-data model evidence remain blockers. Satellite change detections are candidate findings—not proof of illegal encroachment, flood causality or structural safety.
