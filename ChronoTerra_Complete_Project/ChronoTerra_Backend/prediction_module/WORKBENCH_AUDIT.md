> **Superseded verification note (10 October 2026):** this document records the earlier standalone workbench snapshot. The integrated source has since been hardened and has its current count/status in the root `PATCH_AND_VERIFICATION_REPORT.md`. Do not use the historical 51-test count as the current release count. The frontend production bundle and real-dataset model accuracy remain unverified.

# ChronoTerra workbench verification report

## Verified in this packaging runtime

- Runtime: Python 3.13.5, PyTorch 2.10.0+cpu; CUDA is unavailable.
- `python -m py_compile models/*.py quality_gate/*.py api/app.py tools/*.py pipeline/*.py evaluation/*.py`: passed.
- `python -m pytest -q`: **51 passed, 4 existing PyProj deprecation warnings**.
- Software tests cover data-loader fixtures, spectral-spatial Siamese forward/backward behavior, all three task-specific prediction routes using tiny synthetic rasters and randomly initialized test checkpoints, invalid-input rejection, preflight/report output, browser control presence, and explicit HTTP 503 when the trained checkpoint is missing.
- FastAPI now provides `/api/preflight`, `/api/predict`, `/api/models/status`, and the original `/api/analyze` classical baseline. The bundled static demo includes task-specific input selection, preflight, model status, prediction artifact links and JSON report display.
- The urban architecture has a multi-scale Siamese feature path plus an explicit spectral-difference branch. Train-time radiometric augmentation can be switched off for a validation-only ablation.
- Training-only per-channel input-distribution profiles are written to new checkpoints and compared at inference as an unvalidated diagnostic heuristic.
- Sentinel-2 SCL class rasters can be nearest-neighbour reprojected to the input grid. SCL class 2 is masked by default for modern PB04+ products because it denotes cast shadow; keeping it requires an explicit legacy policy after verifying processing-baseline metadata.
- Additional regression tests verify input-profile mask immutability, SCL handling, AOI coverage/masking, validation calibration utilities, and the API AOI upload path.
- A one-epoch manifest-training smoke test and a one-epoch OSCD-format smoke-training test completed using synthetic fixtures; those generated temporary outputs were not promoted to model weights.

- Additional final check: generic manifest training, validation temperature fitting and held-out test evaluation completed end-to-end on a temporary three-scene synthetic SAR-format fixture; OSCD-format training, validation temperature fitting and test evaluation completed on a temporary synthetic OSCD layout. Temporary checkpoints/data were deleted. These are pipeline smoke checks only.

## Dataset status and actual model evidence

- No real OSCD, Sen1Floods11 or xBD training files were present in `data/`; only dataset README/instructions were present. A bounded synthetic manifest smoke training/evaluation was also run during this review; its temporary checkpoint was deleted and is not evidence of satellite accuracy.
- Multiple direct external archive/host requests failed with DNS resolution errors in this runtime.
- No real-data training run completed, no real-trained checkpoint was produced, and no real held-out metrics exist. The manifest training smoke run validated temperature-scaling and test-evaluation code paths only on synthetic fixtures. The synthetic random-checkpoint tests are software plumbing tests only—not accuracy evidence.
- CUDA is unavailable in this runtime. CPU training might be possible on a small data set, but the archives could not be obtained here.
- The user's separate React/Vite project has not been merged or tested in this workbench. The included `web/` static UI is a local API demo.

## Remaining validation before presenting model performance

- Download/verify the labelled dataset archives and review the terms for the intended use.
- Audit extracted data, label semantics, band order, data scales, NoData masks and event/scene splits.
- Train and save task-specific checkpoints. Tune architecture, preprocessing and decision thresholds using only training/validation data.
- Evaluate once on the frozen test split; report per-scene and aggregate precision, recall, F1, IoU, false-positive behavior, class support and failure-case examples.
- Evaluate seasonal, radiometric, cloud/shadow, resolution, sensor-shift and registration effects. The current quality gate provides diagnostics and rejection rules but does not solve all those phenomena automatically.
- Verify in the intended Windows/Python 3.11/Antigravity workspace and connect to the user's real React UI.

## Interpretation

This package is a test-passing prediction/training framework, not an accuracy-validated model release. It fails closed when a trained checkpoint is missing and labels preflight/prediction limitations. Never use the random test checkpoints generated inside software tests as model weights for user predictions.
