# ChronoTerra prediction: research-backed design decisions

**Status:** design/reliability note for the current source snapshot. This document is not a training report and does not claim real-data accuracy.

## 1. What each public labelled dataset can and cannot teach

### OSCD / Onera Satellite Change Detection

Primary source: https://rcdaudt.github.io/oscd/

The maintainer describes 24 registered 13-band Sentinel-2 date-pairs, with pixel-level urban-change labels and an official split of 14 training pairs and 10 test pairs. The data span native spatial resolutions of 10, 20 and 60 metres. The small scene count makes overfitting and scene-level variance a core concern. The current urban model is trained from scratch and predicts *candidate change*, not building identity, physical damage, or legal encroachment.

Design decisions:
- Enforce exactly 13 input bands in the expected order B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12; metadata-free band order must be explicitly verified from authoritative provenance.
- Audit raw value ranges consistently across the official training-derived fit/validation scenes. Do not independently auto-normalize every input pair, which can create inconsistent model meaning.
- Create validation scenes only from the official training split; select epoch and threshold on validation only. Keep the official test split untouched until final evaluation.
- Report per-scene metrics and variability; 10 official test scenes do not establish universal sensor/region generalization.

### Sen1Floods11

Maintainer source and schema: https://github.com/cloudtostreet/Sen1Floods11/blob/master/README.md and https://github.com/cloudtostreet/Sen1Floods11/blob/master/docs/README.md

The maintainer documents a public Google Cloud Storage route, approximately 14 GB for the full v1.1 dataset, 512x512 chip rasters, Sentinel-1 VV/VH values in dB, and hand labels encoded as -1 invalid, 0 not-water, and 1 water. Metadata include event IDs, acquisition dates, orbit direction and relative orbit. A filename's first token/country prefix is not necessarily a unique event ID. The current manifest preserves supplied chip split assignments and emits an audit that true event-level separation is **NOT_VERIFIED** unless an authoritative chip-to-event mapping is used.

Design decisions:
- Train the current branch for single-image water extent, not before/after flood change. Flood-change needs compatible observations at two times plus explicit change labels or a separately designed temporal target.
- Keep SAR preprocessing separate from optical preprocessing. VV/VH dB values are not RGB and must not be substituted with optical reflectance.
- Never count label -1 as background; it is ignored in loss and metrics.
- Preserve the provided split. Report the split as chip-level until true event isolation can be proven. Where an event metadata table is available, augment the audit; do not guess from country prefixes.
- Check use/redistribution conditions and cite the original dataset paper; public access alone is not a complete license determination.

### xBD / xView2 and the xBD-S12 alternative

Original dataset paper: https://openaccess.thecvf.com/content/CVPRW/2019/html/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_Imagery_CVPRW_2019_paper.html
Challenge/data access: https://xview2.org/dataset
xBD-S12 research alternative: https://github.com/prs-eth/xbd-s12

xBD provides paired pre/post disaster imagery, building polygons and ordinal building damage labels. xBD-S12 is a separate Sentinel-1/Sentinel-2 derived dataset and must not be confused with the current paired-RGB xBD branch. The xBD-S12 maintainers explicitly note the original very-high-resolution xBD images/labels cannot be redistributed through their code repo due to licensing restrictions. Dataset terms must be checked independently of the software/code license.

Design decisions:
- Current model consumes concatenated RGB pre/post chips (six channels) and rasterized five-class targets: background, no-damage, minor-damage, major-damage, destroyed. Unknown/unclassified buildings are ignored, not relabelled as undamaged.
- The target converter writes derived masks and retains raw source annotations. Visual QC is required for polygon rasterization and image/mask alignment.
- Split by event where possible. Do not use adjacent tiles from one event across train and test.
- Published work found that both sophisticated and simplified xBD models can fail to generalize to unseen locations; therefore in-domain scores cannot be used as a blanket generalization claim (see https://arxiv.org/abs/2401.17271).

## 2. Why the factor checks are separate

### Spectral compatibility and radiometry

Band identity/order, measurement type, scaling, sensor, acquisition processing, and radiometric differences are separate concerns. Channel count equality is not enough. Fixed task normalizers are used: OSCD's audited raw scale, Sen1Floods11 VV/VH dB mapping, or xBD RGB 0..255 mapping. Range/quantile profiles are diagnostics only; they do not provide atmospheric correction or prove that two sensors are interchangeable.

The urban Siamese network combines shared multi-scale spatial features, an absolute feature-difference pyramid, and a direct per-band spectral-difference branch. This architecture is a hypothesis to be tested—not a claim of accuracy improvement. A controlled ablation must compare: (a) existing classical baseline, (b) Siamese spatial feature differences, (c) spectral-spatial model, and (d) optional radiometric augmentation, on the same development split.

### Spatial overlap, valid coverage and registration

These are different measurements:
1. Footprint overlap describes shared geometry in a common coordinate reference system.
2. AOI coverage describes whether the requested analysis area is covered; this is not inferred merely from pairwise overlap.
3. Common valid coverage requires both dates to contain usable observations at the same grid cells.
4. Content registration checks whether real stable features align despite matching metadata.

The preflight uses affine-derived footprint geometry where possible, exact same-grid common-valid pixel counts, and advisory phase-correlation plus feature/RANSAC diagnostics. None proves location identity. No automatic warping is performed. Registration itself may align away real change, and terrain/relief, local distortion, shadows or clouds can bias feature correspondences. If the pair is not already supported and aligned to the model's declared grid, stop or require explicit, recorded preprocessing.

Raster reprojection is different from content registration. Rasterio documentation describes reprojection/regridding using source/destination coordinate transforms; it relies on the georeferencing supplied, and does not itself prove that features are registered. Categorical masks such as SCL/label IDs use nearest-neighbour resampling; continuous reflectance/SAR intensity requires a selected and recorded continuous-data resampler appropriate to the task. See https://rasterio.readthedocs.io/en/stable/topics/reproject.html.

### Cloud/shadow and SCL

For Sentinel-2, the optional SCL layer is reprojected to the model image grid using nearest-neighbour when needed and georeferencing exists. The current default excludes classes 0, 1, 2, 3, 7, 8, 9, 10 and 11. ESA states that for Processing Baseline 04.00 and later SCL class 2 is CAST_SHADOW; dark features such as dark soils/burned areas are assigned to UNCLASSIFIED class 7. A deliberate `keep` policy exists only for a verified legacy product where class 2 semantics differ; record the actual processing baseline. SCL is imperfect and is not atmospheric correction; thin clouds/shadows and scene-specific failures still require evaluation. Generic RGB cloud/shadow detection is not a validated component in this package.

### Seasonality and domain shift

Acquisition metadata and sampled per-channel statistics can flag an input pair that may differ from training conditions. The new checkpoint input profile stores training-only p01/p05/p50/p95/p99 and compares inference data after the exact model normalizer. This is a **screening heuristic only**, not calibrated OOD detection. It never automatically changes the images or rejects an otherwise compatible scene. Thresholds are not scientifically validated until a labelled domain-shift benchmark is run.

Change detection is especially sensitive to temporal/spatial domain shift. The literature reports that spatial domain shift can be difficult and temporal shift can degrade change detection; see https://arxiv.org/abs/2407.07616 and the review https://arxiv.org/abs/2305.05813. Seasonal/illumination effects should therefore be measured in a controlled holdout experiment, not “solved” by blindly histogram-matching inputs.

## 2A. AOI coverage and score calibration updates

- An optional GeoJSON Polygon/MultiPolygon may define the intended analysis area. GeoJSON without a legacy CRS member is interpreted as WGS84 longitude/latitude under RFC 7946. The geometry is transformed into each input raster CRS; source imagery is not automatically warped. Coverage is measured using an equal-area geometry intersection and separately using valid rasterized pixel centers. Partial AOI coverage is a warning, no rasterized AOI observations is rejected, and predictions outside the requested AOI are marked invalid. The AOI is checked per asset, so a before/after pair can have a different coverage report for each input.
- Checkpoint score temperature is fit on validation-only samples, and binary change/water thresholds are tuned on validation only. NLL, Brier score and ECE are reported as diagnostics. Pixel-wise samples are spatially dependent, and the calibration set may not represent another sensor, region or season; score scaling is not a deployment confidence guarantee. The rationale follows Guo et al. (ICML 2017), but the project's own calibration outcome must be measured on its own held-out scenes.
- Modern Sentinel-2 L2A Processing Baseline 04.00+ SCL class 2 is `CAST_SHADOW` (ESA PSD v15.0). It is masked by default; the `keep` override is available only for a known legacy interpretation after verifying product metadata. SCL remains an imperfect quality layer rather than atmospheric correction.

## 3. Training and evaluation rules

1. Use only authoritative original dataset download/source routes, record archive SHA256 and current use terms.
2. Run data audit before training: sample count, group/event split, band order, normalization/range, class/pixel counts, image/mask dimensions, label values, valid fraction and overlap.
3. Fit class frequencies, distribution profiles, normalization, augmentation parameters, architecture choice, checkpoint selection and decision thresholds using fit/validation data only.
4. Do not use official test labels for preprocessing selection, early stopping, threshold search, or hyperparameter tuning.
5. Report per-scene/event precision, recall, F1, IoU/mIoU, false-positive rate and class counts; for multiclass damage report each class and macro summaries. Include confusion matrices, valid fraction, runtime, checkpoint/data hashes and representative failure overlays.
6. Don't headline raw pixel accuracy alone; large unchanged/background classes can dominate it.
7. Compare against a meaningful baseline and run ablations of spectral branch and radiometric augmentation. Do not claim accuracy gain unless measured on the same validation protocol and confirmed on untouched test.
8. Treat softmax outputs, entropy and top-class margin as uncalibrated unless probability calibration is tested on held-out data. Distribution profile warnings are not proof of a specific failure cause.
9. A successful software test or checkpoint file is not real-data evidence. Real-data checkpoint status is tracked in `outputs/MODEL_WEIGHTS_STATUS.md`.

## 4. What is implemented vs not verified

Implemented in this source snapshot:
- Task-specific model input contracts and fail-closed missing-checkpoint behavior.
- Same-grid/CRS and footprint diagnostics; common-valid pixel coverage when raster grids match.
- Advisory phase-correlation and feature/RANSAC registration estimates, with no blind warp.
- Optional geospatial SCL reprojection using nearest neighbour, with invalid classes recorded.
- Explicit urban spectral-spatial architecture, train-only radiometric augmentation, class-balanced patch sampling/weighted losses where configured, validation-only threshold/epoch selection.
- Training-only input quantile profile stored in new checkpoints and inference shift comparison.
- Output masks with invalid pixels, score layers, uncertainty proxy, overlay and JSON provenance reports.

Not established without real data:
- Any real-data-trained checkpoint or accuracy value.
- An accuracy improvement from the direct spectral branch or radiometric augmentation.
- Complete atmosphere/cloud/shadow/seasonality robustness.
- Calibrated prediction uncertainty or a validated statistical OOD detector.
- True event-independent evaluation of the current Sen1Floods11 official chip splits from filename alone.
- Legal encroachment determination or engineering-grade building-damage certification.
- Windows/Python 3.11/Antigravity end-to-end execution in the user's actual project.

## 5. Primary references

- OSCD maintainer dataset description and downloads: https://rcdaudt.github.io/oscd/
- Original OSCD paper DOI: https://doi.org/10.1109/IGARSS.2018.8518015
- Sen1Floods11 maintainer schema/splits/event metadata: https://github.com/cloudtostreet/Sen1Floods11
- Sen1Floods11 paper: https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html
- xBD dataset paper: https://openaccess.thecvf.com/content/CVPRW/2019/html/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_Imagery_CVPRW_2019_paper.html
- xBD-S12 code/data note and licensing warning: https://github.com/prs-eth/xbd-s12
- xBD unseen-location generalization analysis: https://arxiv.org/abs/2401.17271
- Domain shift in satellite image time series change detection: https://arxiv.org/abs/2407.07616
- Change detection methods review: https://arxiv.org/abs/2305.05813
- Rasterio reprojection documentation: https://rasterio.readthedocs.io/en/stable/topics/reproject.html


## Additional primary-source references added during the prediction review

- ESA Sentinel-2 Product Specification Document (PSD) v15.0, including the PB04+ SCL class-2 semantics: https://sentinels.copernicus.eu/-/sentinel-2-product-specification-document-psd-in-version-15.0
- Sen1Floods11 maintainer README documenting `QC` labels (-1 invalid, 0 not water, 1 water), VV/VH dB channels, 512-pixel chips, event metadata, and split layout: https://github.com/cloudtostreet/Sen1Floods11/blob/master/README.md
- Guo et al., “On Calibration of Modern Neural Networks”, ICML 2017, temperature scaling background: https://proceedings.mlr.press/v70/guo17a.html
