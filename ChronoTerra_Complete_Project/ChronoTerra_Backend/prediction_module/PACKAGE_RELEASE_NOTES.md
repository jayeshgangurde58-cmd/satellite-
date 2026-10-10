> **Superseded verification note (10 October 2026):** this document records the earlier standalone workbench snapshot. The integrated source has since been hardened and has its current count/status in the root `PATCH_AND_VERIFICATION_REPORT.md`. Do not use the historical 51-test count as the current release count. The frontend production bundle and real-dataset model accuracy remain unverified.

# ChronoTerra Prediction Workbench — release notes

## Included

- Three separate model workflows: OSCD 13-band optical urban change; Sen1Floods11 2-band Sentinel-1 water extent; xBD-style paired RGB building damage.
- Urban Siamese U-Net with multi-scale spatial features and an explicit spectral-difference branch.
- Training-only, configurable radiometric augmentation; validation sweep support to compare augmentation on/off.
- Model-specific input contracts and rejection checks for channels, band descriptions/order, CRS/grid, overlap, data scaling/range, invalid pixels, and incompatible formats.
- Diagnostics for sampled radiometry, acquisition dates/seasonality, registration shifts, invalid coverage and optional Sentinel-2 SCL cloud/shadow masks. Optional polygon AOI geometry coverage is measured independently of pair overlap and masks predictions outside the AOI. SCL classes are reprojected nearest-neighbour to the model grid when appropriate; class 2 is masked by default for modern PB04+ products, with explicit legacy keep option.
- Prediction CLI outputs: class mask, score image/array, uncertainty proxy, overlay, JSON provenance/factor report, and inference-time training-distribution comparison.
- FastAPI `/api/preflight`, `/api/predict`, `/api/models/status`, existing `/api/analyze` baseline, and a bundled browser demo with input selection and result artifact links.
- Dataset preparation scripts, group-leakage checks, training/evaluation scripts, validation-only temperature scaling / threshold tuning, train-only per-channel input-profile diagnostics, expanded research decision record, software tests and documentation.
- Sen1Floods11 manifest split semantics clearly distinguish official chip split from unverified event-level isolation; xBD source split handling accepts `tier1` or `train`.

## Verification performed in this package environment

- Python syntax compilation passed across models, quality gate, API, data tools, pipeline and evaluation modules.
- `python -m pytest -q`: 51 passed after the latest source fixes and AOI/calibration integration. Four existing PyProj deprecation warnings remain.
- The bundled inline JavaScript passed `node --check` syntax checks.
- Smoke prediction tests use small synthetic rasters and randomly initialized test-only checkpoints; they only verify the code path and output format.
- One-epoch smoke-training runs completed for the generic manifest trainer and OSCD-format trainer using temporary synthetic fixtures only; generated checkpoints were discarded and are not included.

- The final validation run reports **51 passed, 4 existing PyProj deprecation warnings**; `py_compile`, `--help` smoke checks, all four embedded web JavaScript blocks, and one-epoch synthetic smoke training for both urban OSCD format and generic manifest training succeeded. Temporary synthetic checkpoints were deleted.

## Not claimed / not present

- There are no genuine real-data-trained model checkpoints included.
- The real OSCD, Sen1Floods11 and xBD dataset archives are not bundled and were not found in this build environment.
- Real-data training, real held-out accuracy metrics, real-data calibration assessment, and generalization studies have not completed. The calibration code has only been exercised with synthetic training fixtures in this environment. The new input-distribution profile flags are advisory screening heuristics, not validated OOD detection.
- The quality gate provides diagnostics and strict input contracts but does not perfectly solve atmospheric correction, clouds/shadows, seasonality, co-registration, OOD detection, or all cross-sensor effects.
- The bundled `web/` UI is a basic local API demo. The user's separate React/Vite project has not been merged or tested in this package.
- Windows/Python 3.11/Antigravity execution has not been verified by this Linux/Python 3.13 packaging runtime.

## The honest path to learned prediction

1. Check current dataset access/use terms and the competition's code/data rules.
2. Download real data using the bundled scripts or the maintainer sources documented in `docs/DATASET_SETUP.md`.
3. Run `python -m tools.check_datasets --root .` and inspect the data analysis reports.
4. Train each task separately with `Train-All-Models.ps1 -Model urban|flood|damage`.
5. Inspect frozen held-out per-scene metrics and failure cases before accepting or distributing each checkpoint.
6. Only install a checkpoint in the matching `outputs/models/.../best.pt` path once its provenance and test report have been reviewed.

Never interpret a synthetic/random-checkpoint output as real satellite model performance.
