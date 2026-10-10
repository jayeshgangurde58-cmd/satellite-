# ChronoTerra engineering hardening — patch-pass notes

## Implemented in this package

- Raw inputs and internal reports are not exposed through a blanket static mount. Result routes use an explicit derived-artifact allowlist, 128-bit random session identifiers, HMAC-signed expiring capability URLs, `no-store`/`nosniff` headers, and access-time retention checks. A persistent signing secret can be configured; without it, links stop working after a server restart.
- Absolute local checkpoint/source paths are not returned in model-status or preflight/prediction API responses. Public JSON reports are path-redacted.
- Request limits apply to both declared `Content-Length` and actual streamed request bytes, including chunked requests that lack a content-length header. Per-file/aggregate upload checks still apply after multipart parsing.
- The classical baseline handler uses bounded upload reads. Conservative task-specific raster pixel limits reduce peak memory risk while the predictor still uses whole-array inputs.
- Reverse/equal before/after dates are rejected. Frontend date checks run before the analysis request.
- The AOI mask is distinct from the optional user-supplied reference-zone layer. Boundary overlap/proximity is only calculated on a georeferenced result with a verified common geographic overlay, and is labelled as screening, not legal proof.
- Old `/api/jobs` creation/status/artifact routes are retired with HTTP 410; the legacy result store is not a source of current analysis results.
- `tools/create_deployment_manifest.py` produces pending-by-default, hash-bound model deployment evidence from a valid held-out evaluator report; explicit review and threshold-passing metrics are required to enable inference.
- New regression coverage checks model/source-report hashes, date order, streamed upload size limits, artifact privacy/expiry, signed URL tampering, mounted artifact URLs, and reference-zone geometry logic.
- Results-map cleanup removes Leaflet instances before full DOM replacement, preserves view state for the same result, guards stale asynchronous callbacks, escapes popup attributes, and shows a fallback when the map library is unavailable. The map still depends on third-party CDN/tile network access and needs browser testing.
- The frontend blocks the prediction step when there is no checkpoint with hash-bound, held-out metrics and explicit review approval; no model checkpoints are included here.

## Verification in this environment

- Backend plus prediction-module pytest: **90 passed**, four PyProj/NumPy deprecation warnings.
- Python syntax compilation: **67 Python files passed** `py_compile`.
- TypeScript source compile with the globally available compiler and a temporary Vite-type-free audit config: passed after the current UI patches.
- Production Vite bundle: **not verified** in this runtime. `node_modules` is absent and the previous dependency install did not complete; use `npm ci` and `npm run build` on the target OS.
- Browser end-to-end testing: not performed.
- Real trained checkpoints, labelled datasets and genuine held-out accuracy: not bundled or produced in this environment. The ML endpoints intentionally return `MODEL_NOT_AVAILABLE` until a valid, reviewed model bundle is installed.

## Remaining limits — do not mark production-ready yet

1. Inference still loads bounded rasters into whole-array memory; larger geospatial scenes require true windowed input/output inference or a separate worker with resource enforcement.
2. Artifact URLs are HMAC-signed, expiring bearer capabilities, not user/session authorization. A holder can access allowlisted derived files until expiry. Configure a persistent `CHRONOTERRA_ARTIFACT_SIGNING_SECRET`; add authentication/authorization before public multi-user deployment.
3. The deployment-manifest gate validates report structure, file hashes, task/split assertions, thresholds and reviewer fields. It cannot prove the honesty of split provenance or label correctness; human review and an independent evaluation record remain necessary.
4. The building-damage gate metric is any-damage detection, not certification of individual severity classes.
5. One hundred percent error-free operation cannot be guaranteed. The clean Windows install/build, actual browser path, actual model inference and GIS export inspection still need verification.


The safe product claim remains: ChronoTerra produces candidate-change and reference-boundary screening evidence. It does not determine legal encroachment, flood causality or certified structural safety.
