# ChronoTerra — Final Product & Engineering Blueprint
**Prepared:** 10 October 2026  
**Purpose:** A concrete, implementation-ready specification for the pages, processing pipeline, urban-change/encroachment screening, results presentation, exports, and acceptance tests.

## 1. Non-negotiable product truth

ChronoTerra is a geospatial **screening and evidence-review** tool. Its satellite-only urban workflow identifies candidate surface changes between two dates. It does **not** independently establish that a change is illegal encroachment, identify ownership, or prove a building was damaged.

Use three distinct result terms:
- **Candidate change:** a detected difference between two observations.
- **Boundary-overlap candidate:** a candidate change that spatially intersects an uploaded/authoritative boundary layer.
- **Priority for human review:** a transparent ranking based on recorded evidence and data quality.

Never display “illegal encroachment confirmed” from satellite imagery alone. Legal conclusions require authoritative, date-appropriate parcel/road/river/forest/protected-area records and a qualified review process.

The current delivered package has no trained model weights, no bundled genuine labelled datasets, and no real held-out metrics. Until a trained model is installed and evaluated, show **MODEL NOT AVAILABLE** for that learned workflow. A classical baseline may be offered only as a separately named, lower-evidence option—not as a silent replacement for an ML prediction.

## 2. Recommended web-page architecture

Keep the current visual design and navigation, but make each page answer one clear question and expose backend truth instead of static/demo claims.

### A. Overview / Dashboard — “What can I do, and is the system ready?”
Show:
- Three task cards: Urban change, Flood-water extent, Building damage.
- For each: **model state**, accepted inputs, last evaluation date, benchmark availability, and a “Start analysis” action only when the selected mode can run.
- Service status separated into API health, model checkpoint present, evaluation evidence reviewed, and input-data support.
- Recent session results only if they exist; label session-only history accurately.
- A persistent disclaimer: results are candidate detections and require review.

Do not show made-up counts, percentages, “accuracy”, live alerts, or successful analyses when no evidence exists. Demo images must be marked **preview only — not model input/results**.

### B. New Analysis — wizard with five steps

**Step 1 — Choose task**
- Urban change: two-date, same-area multispectral imagery.
- Flood-water extent: one compatible Sentinel-1 VV/VH scene.
- Building damage: aligned pre/post RGB imagery compatible with the installed checkpoint.
- Explain what the task predicts and explicitly state what it does not predict.

**Step 2 — Add data**
- Show separate upload controls for each required input; show filename, size, band count, dimensions, CRS, acquisition date if available, nodata fraction, and validation status.
- Drag/drop is optional; the same constraints must apply to every upload route.
- Require source/provenance fields: source/provider, scene/product ID, acquisition time, processing level, and licence/usage note where applicable.
- Support GeoTIFF for geospatial workflows. Ordinary JPG/PNG may be accepted only for workflows that explicitly support image-space outputs; never pretend they have geographic coordinates.
- Do not let the user select a place name and imply it georeferences the uploaded raster.

**Step 3 — Inspect & quality gate**
Display a readable checklist with PASS / WARNING / BLOCKED and reasons:
- file type, size, dimensions, pixel count, band count and expected band order;
- CRS, affine transform, pixel size, bounds and nodata;
- pair footprint overlap, common valid coverage, AOI coverage and grid compatibility as separate values;
- date order and time gap;
- content-registration diagnostics;
- cloud/shadow/snow/nodata masks where appropriate;
- sensor and product compatibility;
- model checkpoint/evaluation evidence availability;
- optional authoritative boundary layer validity and date/source.
Do not automatically warp imagery merely to make a pair pass. If alignment is uncertain, stop or allow a clearly labelled manual review path.

**Step 4 — Run**
- Show a real queued/running/completed/failed state.
- Show elapsed time and stage messages without fabricating progress percentages.
- A model missing error must be actionable: `MODEL_NOT_AVAILABLE`, task name, expected checkpoint, and a link to the model status/training instructions.
- Never show a “completed” state before the backend returns a successful result with validated artifacts.

**Step 5 — Review results**
- Open the results page only with a real result payload.
- Keep the inputs, QA report, model/version, thresholds, processing configuration and output artifact IDs linked to the result.

### C. Datasets & Models — “What evidence supports this result?”
For every task show:
- dataset name/source URL, licence/access terms, version, label definition, sample/scene count, split method and date;
- model architecture/version, checkpoint hash, preprocessing profile and band order;
- validation metrics vs held-out test metrics (never conflate them);
- per-scene metrics and failure gallery;
- baseline comparison and ablation results;
- date last evaluated, code version, and evidence-review status.
If missing, display **Not available / not evaluated**. Do not hardcode “ready” labels or infer quality from the existence of a checkpoint file.

### D. Analyses / History — “What has actually run?”
Each row/card should include:
- analysis ID, task, input scene IDs, dates, AOI, mode (baseline or ML), status, model/checkpoint hash, and result timestamp;
- clear distinction between a saved result and a browser-session record;
- reopen only the actual stored result; do not recreate a result from placeholder metadata.
For a no-login local prototype, session-only history is acceptable if disclosed. For shared/public deployment, use server-side records and access-controlled artifacts; do not make history persistent by exposing the upload directory.

### E. Results — primary operational workspace
Recommended layout:
1. **Top status strip:** task, before/after dates, model/baseline name, result status, valid coverage, key warnings.
2. **Map canvas:** synchronized before/after swipe; change-mask overlay; opacity slider; boundary/AOI layers; uncertainty/invalid-data layer; legend; scale bar; north arrow only if projection/map design warrants it.
3. **KPI row:** candidate regions, total candidate area (only when georeferenced and area method is valid), valid AOI coverage, high-priority review count, boundary-overlap area if authoritative boundaries were supplied.
4. **Region table/list:** ID, class, area, calibrated probability or raw score with honest label, QA flags, boundary relation, dates, review priority, and action to zoom/select.
5. **Evidence panel:** before crop, after crop, mask crop, boundary overlay, acquisition metadata, per-region metrics, quality warnings, and reason for review priority.
6. **Downloads:** mask GeoTIFF, valid GeoJSON, GeoPackage where supported, CSV region table, QA JSON, and human-readable PDF report. Each download must come from an actual generated artifact.
7. **Interpretation panel:** “What this result means,” “What it does not establish,” “Known limitations,” and “Suggested human checks.”

### F. Settings / About (or a compact System Status page)
Show environment and version, API status, supported profiles, configured upload limit, artifact-retention policy, model checkpoint/evaluation state, and privacy warning. Never expose filesystem paths, API keys, secrets, or raw internal stack traces to ordinary users.

## 3. Urban change and encroachment: correct analysis logic

### 3.1 Separate the two questions
**Question A — What changed?** Compare valid, compatible observations at time T1 and T2 to create a candidate-change mask and polygons.

**Question B — Does a candidate intersect a regulated/owned boundary?** Only answer this when the user supplies or the application accesses an authoritative, appropriately dated boundary dataset with a known CRS and usage rights.

Do not label Question A as a legal answer to Question B.

### 3.2 Minimum data requirements
For a Sentinel-2 model profile:
- enforce the exact expected product, band order, scaling/reflectance convention and model input profile;
- track the fact that Sentinel-2 MSI bands have different native resolutions (10 m, 20 m, 60 m); resampling to a common grid does not create finer detail in the coarser bands;
- record acquisition dates, processing level, cloud/nodata quality, pixel grid, CRS and transforms;
- require sufficient common valid coverage and a supported alignment state;
- avoid comparing unharmonized radiometry, seasonal vegetation differences, cloud shadows or different pixel grids as if they were real construction.

Sentinel-2 is suitable for broad land-cover/urban-change screening. Its 10 m bands are not sufficient to reliably delineate every individual building edge or determine narrow property-line violations. For parcel/building-level review, prefer appropriately licensed, georeferenced higher-resolution imagery and authoritative cadastral boundaries.

### 3.3 Pipeline
1. **Ingest:** validate file signatures, size, pixel count, bands, CRS, transform, nodata and metadata.
2. **Pair validation:** footprint overlap, common valid coverage, date gap, grid compatibility and content registration are distinct checks.
3. **Mask invalid observations:** cloud, shadow, snow, saturation and nodata using supported quality layers; keep an explicit valid-data mask.
4. **Harmonize conservatively:** apply only the preprocessing required by the trained model. Record every resampling and normalization step.
5. **Generate candidate mask:** trained model when a compatible, evaluated checkpoint exists; otherwise an explicitly labelled classical baseline if supported.
6. **Post-process:** remove tiny components only using a documented scale-aware rule; do not smooth away narrow real features. Keep raw and filtered masks available for audit.
7. **Vectorize:** use the raster transform and CRS; validate geometries; export GeoJSON in WGS84 longitude/latitude axis order; compute metric area/centroid in a suitable projected CRS or geodesically. Never derive geographic area from degree-squared geometry.
8. **Overlay official boundaries:** transform both layers into a common CRS for calculation; preserve source/date/licence; calculate overlap area, overlap fraction, and distance with a documented method and tolerance.
9. **Rank for review:** use transparent, versioned criteria. A ranking is a queueing aid—not a legal or calibrated probability.
10. **Package evidence:** retain source references, masks, QA, model/config hashes, output checksums, and operator notes under an explicit retention policy.

### 3.4 Encroachment result types that are defensible
- `CANDIDATE_CHANGE`: satellite comparison indicates a difference.
- `BOUNDARY_OVERLAP_CANDIDATE`: candidate geometry intersects the supplied authoritative boundary or its defined buffer.
- `NEAR_BOUNDARY_CANDIDATE`: candidate lies within a configured distance but does not intersect.
- `PERSISTENT_CHANGE_CANDIDATE`: change remains visible in a later valid observation; requires a third observation, not just two images.
- `REVIEW_REQUIRED`: QA, alignment, date, cloud or model uncertainty warrants human inspection.
- `INSUFFICIENT_EVIDENCE`: no reliable conclusion due to missing/poor data.
Never emit `ILLEGAL_ENCROACHMENT_CONFIRMED` from the satellite model.

### 3.5 Review priority, not “severity”
A configurable review-priority score can combine:
- detection score (only call it probability if calibration has been measured);
- valid-data fraction and registration quality;
- size/shape stability and persistence across observations;
- boundary overlap fraction and/or distance to a supplied boundary;
- source quality and recency.
Show each factor beside the total and version the formula. Do not hide an arbitrary weighted sum behind a “risk 94%” badge. If weights have not been validated, call it a heuristic priority score, not a probability of encroachment.

## 4. Best ways to represent the result

Use complementary views; no single visual tells the whole story.

### View 1 — Before/after swipe
Best for human interpretation of where the land surface changed. Synchronize map extent and zoom; preserve correct geospatial bounds; show dates and source labels on both sides.

### View 2 — Candidate-change overlay
Use a clear mask/polygon outline over the after image, with opacity controls. Distinguish predicted change from invalid/cloudy pixels. Provide a legend and avoid relying on colour alone.

### View 3 — Official boundary overlay
Only show when a real boundary file/service is supplied and validated. Distinguish parcel/road/river/protected-area boundaries by layer style. Highlight intersection segments and show source, effective date, CRS, and licence. A boundary layer is evidence context, not proof of ownership/legal status unless its authority is established.

### View 4 — Region inspector
Selecting a polygon should:
- zoom to it and highlight it on the map;
- show before/after/mask crops at the same extent;
- show candidate area, pixel count, bounding box, centroid, detection score, QA flags, boundary overlap, distance and persistence status;
- show which values are unavailable and why;
- permit a human disposition: `needs_review`, `likely_real_change`, `likely_artifact`, `insufficient_evidence`, with reviewer/time/comment if review records are enabled.

### View 5 — Summary charts
Useful charts:
- candidate area by class/type (only if class taxonomy exists);
- candidate count/area by review-priority band;
- candidate boundary-overlap area and non-overlap area;
- change area by date interval for multi-temporal analyses;
- QA breakdown (valid coverage, cloud/nodata, rejected scenes);
- precision/recall/F1/IoU/FPR by scene for benchmark evaluation.
Never show model-accuracy charts for a checkpoint without a valid held-out report.

### View 6 — Exports and report
- **GeoTIFF:** mask plus CRS, affine transform and nodata.
- **GeoJSON:** vector features in WGS84 longitude/latitude order.
- **GeoPackage:** richer GIS layer and typed attributes.
- **CSV:** one row per candidate with stable IDs and the same attributes as the map.
- **QA JSON:** machine-readable inputs, decisions, warnings, model/version, thresholds and artifact hashes.
- **PDF report:** purpose, dates, data provenance, coverage/QA, map snapshots, findings, limitations, review list, and export checksums.
Check exported files in QGIS or another independent GIS tool before claiming GIS-ready output.

## 5. Recommended region data schema

Every feature should have stable, documented properties. Fields can be null only with an explicit reason; do not fabricate values.

```json
{
  "candidate_id": "ct_<stable-id>",
  "task": "urban_change",
  "result_type": "CANDIDATE_CHANGE",
  "class_id": 1,
  "class_name": "candidate_change",
  "area_m2": null,
  "area_method": "not_available_without_georeferencing",
  "pixel_count": 0,
  "detection_score": null,
  "score_semantics": "uncalibrated_model_score",
  "valid_data_fraction": null,
  "registration_status": "not_assessed",
  "before_date": null,
  "after_date": null,
  "boundary_source_id": null,
  "boundary_overlap_m2": null,
  "boundary_overlap_fraction": null,
  "distance_to_boundary_m": null,
  "persistence_status": "not_assessed",
  "review_priority": "unranked",
  "review_priority_method_version": null,
  "qa_flags": [],
  "model_id": null,
  "checkpoint_sha256": null,
  "processing_config_hash": null
}
```

The example uses null/zero placeholders to illustrate the schema, **not an actual result**. Production output must distinguish a real measured zero from a missing value. For GeoJSON, coordinates must be WGS84 longitude/latitude. Use projected/geodesic calculations for metric area and distance.

## 6. Quality metrics and evidence

### Model evaluation
On a genuinely held-out set, report:
- precision, recall, F1, IoU and false-positive rate;
- per-scene metrics and aggregate metrics with a stated aggregation method;
- no-change scenes and difficult cases;
- valid coverage and rejected/abstained coverage;
- threshold selection procedure, class imbalance and calibration if scores are called probabilities;
- baseline versus ML results on identical scenes;
- inference time, hardware, model/checkpoint hash and preprocessing version.

Do not use pixel accuracy alone: a model can score highly by predicting the dominant unchanged class. Do not tune thresholds on the final test set. Split by scene/geography/event to reduce spatial/temporal leakage.

### Operational acceptance
A result is “complete” only if:
- input checks pass;
- the correct task/model profile is recorded;
- model status and checkpoint hash are recorded;
- prediction output is generated;
- mask dimensions/transform/CRS/nodata are validated;
- vector output is valid and coordinates are independently checked;
- QA report and checksums are generated;
- UI can open and download each artifact;
- failures never render stale or demo results as new predictions.

## 7. Reliability, security and privacy
- Never mount the directory containing raw uploads as a public static directory.
- Separate private raw inputs/internal logs from approved output artifacts; use opaque expiring artifact access tokens or equivalent access control.
- Set upload byte limits **while streaming**, plus raster pixel/dimension/band limits before allocating arrays.
- Use windowed raster reads and memory-bounded tile processing; enforce processing time and output-size limits.
- Validate file signatures as well as extensions; reject malformed GeoTIFF/GeoJSON and unsupported CRS.
- Clean temporary inputs on failure and apply a defined retention policy.
- Keep API keys server-side when secret; restrict browser map keys by origin and enabled API.
- Restrict CORS, rate-limit CPU-intensive endpoints, and do not return filesystem paths or stack traces to clients.
- Keep the product public/no-login if required, but do not make uploaded artifacts publicly discoverable. No-login does not mean no access controls for data.
- Add regression tests proving that source uploads cannot be fetched from output URLs.

## 8. Suggested implementation order

### P0 — before any public demo
1. Remove public static serving of the session/upload directory; add an allowlist artifact endpoint and privacy regression tests.
2. Enforce streaming upload limits and raster pixel/memory limits; move to windowed reads.
3. Make model availability and baseline mode explicit on Dashboard and New Analysis.
4. Remove stale API calls, placeholder result data and ambiguous “complete” states.
5. Add a true end-to-end API/UI test for each status: missing model, bad input, QA blocked, successful prediction, and failed export.

### P1 — prove one workflow
6. Choose urban change as the primary workflow if it matches the judging rubric.
7. Acquire an authentic, licensed dataset; record scene manifest, acquisition dates, labels and split hashes.
8. Run the classical baseline and one trained model on identical held-out scenes.
9. Inspect false positives: clouds/shadows, seasonality, vegetation, water, construction, registration errors, no-change scenes.
10. Freeze a checkpoint and publish a held-out report before enabling the model card as “evaluated”.

### P2 — make results useful to reviewers
11. Unify region schema and boundary-overlap calculations.
12. Add a boundary upload/validation step and clearly label the source as authoritative/user-provided/unverified.
13. Implement map layers, region inspector, charts and exports against actual backend artifacts.
14. Validate GeoJSON/GPKG independently in QGIS; run browser tests on the target Windows machine.
15. Only then consider multi-temporal persistence, disaster damage, more sensors or public deployment.

## 9. Definition of done
A release is accepted only when all are true:
- clean install and production build pass on the target OS;
- unit and integration tests pass;
- no public route exposes raw uploaded files;
- upload/raster resource limits are tested;
- a real supported input completes the selected workflow with a real checkpoint or explicitly labelled baseline;
- no mock output appears in the real-results view;
- all geospatial exports are valid and independently checked;
- held-out metrics and known failure modes are published;
- every KPI and label has a documented source/calculation;
- legal/encroachment language remains a candidate-screening claim;
- the UI explains when data is insufficient and never forces a confident result.

## 10. Source references
- ESA Sentinel-2 facts and figures: https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2/Facts_and_figures
- Copernicus Sentinel-2 MSI resolution and band documentation: https://s2.pages.eopf.copernicus.eu/pdfs-adfs/MSI/index.html
- IETF RFC 7946, GeoJSON coordinate system and format: https://www.rfc-editor.org/rfc/rfc7946.html
- OGC JSON-FG overview (context on GeoJSON's constraints): https://www.ogc.org/standards/json-fg/
- OSCD dataset: https://rcdaudt.github.io/oscd/
- Sen1Floods11: https://github.com/cloudtostreet/Sen1Floods11
- LEVIR-CD: https://justchenhao.github.io/LEVIR/
- AnyChange, NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/hash/9415416201aa201902d1743c7e65787b-Abstract-Conference.html

## Final engineering judgment
The strongest defensible ChronoTerra product is not “AI proves encroachment.” It is **a quality-gated, traceable satellite-change screening workflow that turns candidate changes into reviewable GIS evidence and overlays them with authoritative boundaries when available**. The advantage must be demonstrated with lower false alarms, reliable geospatial outputs, and transparent evidence compared with a baseline—not claimed from interface features alone.
