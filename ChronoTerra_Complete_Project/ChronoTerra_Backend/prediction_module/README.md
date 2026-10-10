# ChronoTerra Prediction Module — standalone folder

This folder contains the complete current prediction/training workbench for ChronoTerra GEOAI-04: quality gates, task-specific model architectures, dataset adapters, training/evaluation scripts, inference code, API routes, output reports, tests, and the local browser demo.

**Important status boundary:** the source workflows are implemented and software-tested. No genuine OSCD, Sen1Floods11 or xBD training archives were available in the build environment, so no real-data-trained checkpoint or real-data accuracy result is included. Missing checkpoints are reported explicitly; the inference API does not invent predictions or use random weights as a fallback.

## Task models — distinct input contracts

| Task | Model / dataset | Expected input | Output and limitation |
|---|---|---|---|
| Urban change | Siamese spectral-spatial U-Net, OSCD | Two aligned 13-band Sentinel-2 GeoTIFFs in documented OSCD band order | `unchanged` / `candidate_change` mask. It does not decide legal encroachment. |
| Flood extent | U-Net, Sen1Floods11 hand-labelled subset | One georeferenced two-band Sentinel-1 VV/VH GeoTIFF with compatible preprocessing | `not_water` / `water` mask. Single-image water mapping is not temporal flood-change detection. |
| Building damage | U-Net, xBD-style RGB pairs | Aligned pre/post RGB images consistent with xBD scale and label semantics | background/no-damage/minor/major/destroyed mask. Not engineering certification. |

The xBD branch requires separately obtaining the original dataset through its official access route and reviewing its current terms. The xBD-S12 dataset is a separate Sentinel-1/Sentinel-2 derivation and is not interchangeable with this package's six-channel RGB xBD contract.

## Factors and implemented behavior

- **Spectral/model profile:** explicit band/channel contracts, fixed training/inference normalizers, OSCD development-scene raw-range audit, range-scale mismatch checks, and per-channel training quantile profiles saved with new checkpoints.
- **Spatial features:** multiscale U-Net features and a direct raw spectral-difference branch in the urban Siamese model. The accuracy gain of that branch is a hypothesis; compare against the baseline on fixed splits.
- **Pair geometry:** footprint overlap from affine footprints where georeferencing exists, CRS/transform/shape checks, sampled image quality, and exact common-valid coverage where the rasters share a grid.
- **Registration:** phase-correlation and feature/RANSAC diagnostic estimates. These are advisory only; no blind warp occurs and a diagnostic does not prove scene identity or correct alignment.
- **Cloud/shadow:** optional Sentinel-2 SCL quality masks; a differently sampled SCL grid is nearest-neighbour reprojected to the model-image grid. SCL class 2 is masked by default for modern Sentinel-2 L2A processing baseline PB04+ (renamed CAST_SHADOW); it can be retained only with an explicit legacy policy after verifying source processing metadata. SCL is not atmospheric correction and does not catch every cloud/shadow. Optional WGS84 GeoJSON AOI coverage, exact pixel-center masking and valid-pixel reporting are supported for georeferenced rasters. Partial AOI coverage warns; no overlap/no valid observations rejects.
- **Seasonality/radiometry:** date/season hints, sampled radiometric statistics, configurable training-only augmentation and distribution-shift diagnostics. These do not solve domain shift; validate on labeled scenes.
- **NoData:** raster masks and finite-value checks, pair-common validity, ignored labels during training, and output class 255 for invalid/unobserved pixels.
- **Class imbalance:** training-only class frequencies/weights and changed-pixel-oriented patch sampling where configured. Benefits are unproven until real-data evaluation.
- **Uncertainty/OOD:** entropy, class-score margin and training-distribution comparison are explicitly uncalibrated screening diagnostics, not calibrated probabilities or validated OOD rejection.
- **Evaluation:** per-scene metrics, class-wise and pooled confusion measures, precision, recall, F1, IoU and false-positive measurements; the held-out split must remain untouched during model selection.

See `docs/PREDICTION_FACTORS_AND_GATES.md` and `docs/RESEARCH_REFERENCES_AND_DESIGN_DECISIONS.md` for the full status matrix, decisions, sources and known limitations.

## Setup on Windows

Use Python 3.11 for the intended Windows environment. From the module root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-ml.txt
python -m pytest -q
python -m tools.check_datasets --root .
```

`requirements-data.txt` contains optional dependencies for STAC-based real Sentinel-2 pair retrieval. Downloads are never started implicitly by inference or training commands.

## Prepare data

Read `docs/DATASET_SETUP.md` in full. Confirm access/terms before use.

### Urban change — OSCD

Use the maintainer page [Onera Satellite Change Detection](https://rcdaudt.github.io/oscd/) and its current linked archives. Then run:

```powershell
python -m tools.analyze_oscd --root data/oscd --output outputs/analysis/oscd_data_analysis.json
```

The maintainer describes 24 registered 13-band Sentinel-2 pairs, 14 training and 10 test pairs, with 10/20/60 m band resolutions depending on the band. This dataset is small. The trainer separates a validation subset from the official training scenes and leaves the official test split for the separate final evaluator. Do not tune based on test results.

### Flood extent — Sen1Floods11

The optional PowerShell helper requires Google Cloud CLI/`gsutil` and fetches the hand-labelled SAR subset plus official split files:

```powershell
.\tools\download_sen1floods11_handlabeled.ps1
python -m tools.make_sen1floods_manifest --root data/sen1floods11 --output data/manifests/sen1floods11.csv
python -m tools.analyze_manifest --manifest data/manifests/sen1floods11.csv --root data/sen1floods11 --num-classes 2 --ignore-index 255 --output outputs/analysis/sen1floods11_data_analysis.json
```

Official labels encode `-1` invalid, `0` not water, `1` water. The manifest preserves official chip split membership; event-level separation cannot be inferred from a country prefix alone, so the generated audit says when event isolation remains unverified. Check current data terms.

### Building damage — xBD

Follow [xView2/xBD data access](https://xview2.org/dataset) and accept/review the current terms. Place labelled split folders under `data/xbd/{train|tier1}/images` and `labels`, and any eligible labelled `test` / `tier3` data according to the guide. Then:

```powershell
python -m tools.prepare_xbd_targets --root data/xbd --splits train test tier3
python -m tools.make_xbd_manifest --root data/xbd --output data/manifests/xbd_damage.csv --val-fraction 0.15 --test-fraction 0.15 --seed 42 --include-tier3
python -m tools.analyze_manifest --manifest data/manifests/xbd_damage.csv --root data/xbd --num-classes 5 --ignore-index 255 --output outputs/analysis/xbd_data_analysis.json
```

Targets are derived from xBD post-disaster polygons; unknown/unclassified buildings are ignored, not guessed to be undamaged. Split by whole event groups and visually review generated masks. If no publicly labelled test data are provided, reported test scores are an internal group-held-out split, not the hidden official challenge test score.

## Train and evaluate

Install the ML dependencies, prepare the chosen real labelled data and review the analysis report before training. You can run one branch at a time:

```powershell
.\Train-All-Models.ps1 -Model urban
.\Train-All-Models.ps1 -Model flood
.\Train-All-Models.ps1 -Model damage
```

The script runs the selected data audit, training, and separate test evaluation; it exits when required paths are missing. `-Model all` needs all three datasets. Default values are a starting point, not proof of optimal hyperparameters. Use validation experiments only for architecture/augmentation/threshold choices, and record settings.

Expected evaluated checkpoints:

- `outputs/models/oscd_siamese_v2/best.pt` + `oscd_test_metrics.json`
- `outputs/models/sen1floods11_sar_unet_v1/best.pt` + `test_metrics.json`
- `outputs/models/xbd_damage_unet_v1/best.pt` + `test_metrics.json`

Review per-scene and class-wise scores, error overlays and data provenance before allowing any checkpoint to serve users. Good performance on one dataset does not guarantee generalization to different regions, sensors, seasons or image quality.

## Deployment evidence gate (required before API inference)

A `.pt` checkpoint does not automatically enable inference. After running a frozen held-out test evaluator, read [`docs/MODEL_DEPLOYMENT_GATES.md`](docs/MODEL_DEPLOYMENT_GATES.md) and use `python -m tools.create_deployment_manifest --task ... --checkpoint ... --report ...`. The first run creates a pending manifest and inference remains disabled. Only after reviewing the source test report and confirming that task-specific metrics meet the documented thresholds should a reviewer re-run the tool with `--approve-for-inference --reviewer ... --reviewed-at ...`. The API verifies the checkpoint, normalized metrics and raw source report hashes and checks the task/split/review fields. The helper cannot verify dataset authenticity or replace independent review.

For building damage, the deployment gate currently measures **any damage vs background/no-damage**; review all severity-class metrics separately. The default thresholds are project release policy starting points, not universal scientific standards.

## Run predictions

### Direct CLI

Run preflight first; it performs the same input checks without needing model weights. Pass a valid WGS84 GeoJSON polygon if the user has an intended area of interest (AOI):

```powershell
python -m quality_gate.prediction_preflight --task urban_change --before path\before_13band.tif --after path\after_13band.tif --aoi-geojson path\aoi.geojson --assume-band-order --output outputs\preflight\urban.json
```

Then run inference. The AOI argument is optional; when supplied, output pixels outside it are marked invalid. `--before-scl` and `--after-scl` accept optional corresponding Sentinel-2 Scene Classification Layer rasters; class 2 is masked by default for modern PB04+ products.

```powershell
# Urban change: 13-band aligned GeoTIFF pair
python -m models.predict_chronoterra --task urban_change --checkpoint outputs/models/oscd_siamese_v2/best.pt --before path\before_13band.tif --after path\after_13band.tif --output-prefix outputs\predictions\urban --tile-size 256 --overlap 64 --assume-band-order --aoi-geojson path\aoi.geojson

# Flood extent: one 2-band VV/VH GeoTIFF
python -m models.predict_chronoterra --task flood_extent --checkpoint outputs/models/sen1floods11_sar_unet_v1/best.pt --input path\sentinel1_vv_vh.tif --output-prefix outputs\predictions\flood --assume-band-order

# Building damage: an aligned compatible RGB pre/post pair
python -m models.predict_chronoterra --task building_damage --checkpoint outputs/models/xbd_damage_unet_v1/best.pt --before path\pre.png --after path\post.png --output-prefix outputs\predictions\damage
```

`--assume-band-order` is a user assertion that source band order was independently verified. It does not rearrange data or prove compatibility.

### Local API and browser demo

```powershell
python -m uvicorn api.app:app --reload
```

Open `http://127.0.0.1:8000`. Key routes: `GET /api/health`, `GET /api/models/status`, `POST /api/preflight`, `POST /api/predict`, and the pre-existing `POST /api/analyze` classical baseline. `/api/predict` returns `MODEL_NOT_AVAILABLE` (HTTP 503) when a matching checkpoint is missing; it never substitutes random weights or the baseline. The bundled static web UI is a local demo and is not merged into the user's separate React/Vite application.

### Output artifacts

Learned prediction writes a class mask (invalid pixels set to 255), class score raster/array, entropy/margin proxy, visual overlay, and JSON report containing input paths, checkpoint hash, model/task contract, valid fraction, class counts, factor diagnostics and warnings. Scores are temperature-scaled using the checkpoint’s validation-only calibration when available, but they are still **not guaranteed calibrated probabilities**; entropy/margin remain uncertainty proxies.

## Verification facts for this release

- Python syntax compilation passed for API, models, quality gates, tools, pipeline and evaluation files.
- Earlier isolated workbench snapshot: 51 passed; the integrated ChronoTerra project now includes additional hardening tests. See root `PATCH_AND_VERIFICATION_REPORT.md` for the current full-project result.
- One-epoch smoke training succeeded for the OSCD-format trainer and the generic manifest trainer using temporary synthetic fixtures. Those output checkpoints were discarded and are not shipped.
- No real labelled datasets or real-trained checkpoint files are included, and no real-dataset test scores were produced during this build. This package cannot truthfully promise good prediction accuracy until the genuine data are trained and the held-out results are inspected.
- Windows/Python 3.11 and the user's Antigravity/React workspace are not verified by this build environment.

## Deep-research sources

- OSCD maintainer page and labels: https://rcdaudt.github.io/oscd/
- Sen1Floods11 maintainer schema, hand labels, official splits and event metadata: https://github.com/cloudtostreet/Sen1Floods11 and https://github.com/cloudtostreet/Sen1Floods11/blob/master/docs/README.md
- xBD data access and terms: https://xview2.org/dataset
- xBD-S12 (separate Sentinel-1/Sentinel-2 damage dataset; separate data terms): https://github.com/prs-eth/xbd-s12
- Rasterio reprojection: https://rasterio.readthedocs.io/en/stable/topics/reproject.html
- Temperature-scaling background (Guo et al., ICML 2017): https://proceedings.mlr.press/v70/guo17a.html
- ESA Sentinel-2 PSD v15.0 and SCL PB04+ class-2 semantics: https://sentinels.copernicus.eu/-/sentinel-2-product-specification-document-psd-in-version-15.0
- Change detection review covering registration, radiometry, seasonal variability and sensor heterogeneity: https://www.mdpi.com/rs/16/13/2355
- Temporal/spatial domain shift: https://arxiv.org/abs/2407.07616
- Generalization to unseen disaster locations for xBD damage prediction: https://arxiv.org/abs/2401.17271

## Main limitations

- No real trained weights or real held-out metrics are included.
- Some key checks are diagnostics, not solutions: feature registration, seasonality, atmospheric correction, cloud/shadow detection, calibrated uncertainty and out-of-distribution rejection remain incomplete or unvalidated.
- The models are trained from scratch. More epochs or a more complex model may not improve performance; the relevant validation evidence must decide.
- Legal encroachment requires authoritative cadastral/zoning/protected-area boundaries; no learned change mask proves illegality.
- Competition rules should be checked before reusing generated/prewritten code or external model checkpoints.
