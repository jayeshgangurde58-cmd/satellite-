# ChronoTerra trained-model and prediction workflows

## Separate model contracts

- `siamese_unet.py` / `train_oscd.py`: 13-band Sentinel-2 OSCD-order before/after GeoTIFF pair -> binary candidate urban-change mask. The network uses multi-scale Siamese spatial features plus an explicit spectral-difference branch.
- `generic_unet.py` / `train_manifest.py`: U-Net segmentation for 2-band Sentinel-1 VV/VH water extent (Sen1Floods11) and 6-channel paired RGB building-damage classes (xBD-style labels).
- `evaluate_oscd_model.py` and `evaluate_manifest.py`: held-out evaluation; model selection and threshold selection are based on validation data, not official test scores.
- `predict_chronoterra.py`: unified task-specific inference CLI; produces class mask, uncalibrated class score raster/array, uncertainty proxy, preview overlay, and a provenance/quality-factor JSON report.
- `input_profile.py`: bounded training-only per-channel quantile profiles saved in checkpoints and compared with inference input after the fixed normalizer; shift output is a screening heuristic, not calibrated OOD detection.
- `api/app.py`: `/api/preflight`, `/api/predict`, and `/api/models/status` expose the prediction and quality-gate workflows. `/api/predict` returns `503 MODEL_NOT_AVAILABLE` when the checkpoint does not exist and never substitutes random weights or the classical baseline.

## Important limits

No genuine real-data-trained checkpoints are bundled. Software tests use synthetic fixtures only and are not accuracy evidence. Do not serve a checkpoint until it has been trained on the corresponding real labelled dataset and the held-out evaluation report has been reviewed.

Some factors are diagnostics/quality gates, not solved scientific problems: metadata does not prove content alignment; missing cloud/shadow QA remains an explicit warning; seasonal and radiometric diagnostics are not a substitute for atmospheric correction; input-profile drift flags are heuristic; uncertainty scores are not calibrated; and the model may fail on out-of-distribution inputs. See `docs/PREDICTION_FACTORS_AND_GATES.md` and `docs/RESEARCH_REFERENCES_AND_DESIGN_DECISIONS.md`.
