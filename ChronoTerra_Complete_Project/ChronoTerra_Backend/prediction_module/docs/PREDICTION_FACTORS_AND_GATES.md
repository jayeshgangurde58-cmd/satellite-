# ChronoTerra prediction factors and decision gates

This file describes code present in the current prediction module versus behavior still requiring real-data evidence. **A software check passing is not the same as a scientifically validated prediction.**

## Factor/status matrix

| Factor | Current implementation | Decision behavior | Still required / limitations |
|---|---|---|---|
| File integrity and supported formats | `quality_gate.pair_preflight.inspect_image`, `quality_gate.prediction_preflight.inspect_asset` | Reject unreadable, unsupported, empty/all-invalid input | A valid file does not establish authentic source data |
| Dimensions/channels | Preflight and task checkpoint contracts | Reject channel/dimension mismatch | Cannot infer semantic bands from channel count alone |
| Spectral identity/order | OSCD expected 13-band order; explicit assumption option; VV/VH profile checked where metadata exists | Reject known conflicts; require verified assumptions if descriptions absent | Metadata may be incomplete or wrong; source provenance must be reviewed |
| Spectral values/scaling | Cross-scene development split range audit for OSCD; fixed modality-specific normalizers; inference input range checks | Reject dangerous obvious scale mismatch; report distribution diagnostics | DN range heuristic is not atmospheric/radiometric calibration |
| Spatial features/context | Siamese shared multiscale U-Net encoder/decoder, feature-difference pyramid and raw spectral-difference branch; generic U-Net for other tasks | Learned during training | Accuracy contribution is unmeasured until ablation on real labelled scenes |
| Geographic footprint overlap | Densified affine-derived footprint polygon transformed to equal-area CRS when possible | Reject no overlap; warn on partial overlap | Footprint overlap alone does not prove content registration |
| User-defined AOI coverage | `quality_gate.aoi_coverage` transforms polygon AOIs to each raster CRS, computes equal-area geometric coverage, rasterizes a pixel-center mask and counts valid pixels inside the AOI | Reject if AOI has no raster pixels/valid observations; warn if coverage is partial; prediction output is masked outside AOI | Requires trusted raster CRS/transform and valid GeoJSON; AOI coverage is not proof of image-pair alignment or complete coverage of the requested AOI |
| Common valid pixels | Windowed pixel-validity counts when rasters share the exact grid | Report common-valid fraction; reject none | If grids differ, metrics become not verified; no silent reprojection of imagery |
| Content registration | Phase correlation plus SIFT/ORB feature matches and RANSAC transform diagnostics | Advisory only; no blind warp | No real-scene validation; scene changes/clouds/repeated patterns can bias estimates |
| Radiometric differences | Sampled per-channel percentiles and median-shift diagnostics; optional train-only augmentation | Warning/diagnostic; do not auto-correct | Validate augmentation/harmonization with controlled ablations; genuine change also shifts histograms |
| Cloud/shadow | Optional Sentinel-2 SCL layer aligned to target grid using nearest-neighbour; selected invalid classes removed | Reject unusable SCL alignment; warn if absent | SCL can miss thin cloud/shadow; generic RGB cloud detection not validated; not atmospheric correction |
| Seasonal differences | Acquisition date metadata, date-gap/day-of-year heuristic | Warning where evidence indicates seasonal gap; otherwise not verified | Heuristic is not climate/phenology-aware; test on seasonal holdouts |
| Spatial resolution | Record transform and resolution; task-specific input profile | Reject known incompatible data profiles where implemented | Full minimum mapping unit and all supported resolution thresholds require calibration for each model |
| NoData/invalid pixels | Per-band masks, finite checks, nodata tags, common masks and mask value 255 in outputs | Exclude from loss/metrics and outputs | Untagged sentinel values may be ambiguous; audit each archive's convention |
| Class imbalance | Training-only class weights where configured, changed-patch sampling for OSCD, class/pixel audit tools | Record class frequencies; optimize on validation only | Small/rare classes still may have poor recall; inspect class-level scores |
| Input shift / unfamiliar inputs | New training-only per-channel p01/p05/p50/p95/p99 profile saved in checkpoint and compared at inference | `DIAGNOSTIC_ONLY`; never auto-reject on this alone | Thresholds are unvalidated screening heuristics, not calibrated OOD detection |
| Predictive uncertainty | Entropy and top-1/top-2 score margin written as proxies | Exposed as explicitly uncalibrated | Calibration and selective prediction need a real validation study |
| Validation score scaling | Training scripts fit a scalar temperature on held-out validation pixels/scenes and choose binary thresholds only on validation | Applies checkpoint-stored temperature during inference and records calibration diagnostics | This is not a deployment guarantee; pixel samples are correlated and domain shift can break calibration; validate with independent scenes and report Brier/NLL/ECE |
| Temporal/task semantics | Urban and building-damage consume pairs; flood branch consumes one Sentinel-1 scene | Different checkpoints and task labels | Flood output is water extent only, not flood change; change is not damage or legal encroachment |
| Legal / structural interpretation | Reports use candidate-change/water/damage class terminology | Does not make legal or safety decision | Legal encroachment needs authoritative boundaries; damage mask is not engineering certification |

## Task contracts

### Urban change
- Input: 13-band Sentinel-2 GeoTIFF pair, expected OSCD order `B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12`, compatible fixed normalization and same supported pixel grid.
- Architecture: Siamese U-Net with shared multi-scale spatial features, absolute feature deltas and an explicit raw spectral-difference branch.
- Output: `unchanged` vs `candidate_change`, mask/scores/uncertainty-proxy/overlay/report.
- Dataset: OSCD includes 24 registered pairs (14 train, 10 test) and different native band resolutions. See https://rcdaudt.github.io/oscd/.
- Constraints: OSCD is small; use validation split from official train scenes and reserve official test for one frozen final evaluation. This does not support legal encroachment or building-by-building claims at 10 m.

### Flood extent
- Input: one georeferenced two-band Sentinel-1 SAR VV/VH raster using the training checkpoint's expected dB normalization.
- Dataset: Sen1Floods11 labels are `-1` invalid, `0` not water, `1` water; `-1` must be ignored. See https://github.com/cloudtostreet/Sen1Floods11.
- Output: water/non-water segmentation, not whether water increased between two dates. A temporal flood-change model needs a separate paired target and training/evaluation design.
- Constraints: event-level split isolation cannot be inferred from a country prefix; manifest audits explicitly state when it is not verified. SAR orbit, incidence angle, terrain effects and speckle remain data-quality factors.

### Building damage
- Input: paired pre/post RGB image chips consistent with xBD preprocessing, six channels total, aligned to rasterized post-disaster target.
- Classes: background, no-damage, minor-damage, major-damage, destroyed; unclassified annotations are ignore (255), not guessed as no-damage.
- Dataset use terms must be accepted and reviewed independently of the code license. Data are not bundled.
- Split by event if possible; report class-wise metrics and variability. Damage classification is dataset-specific and is not structural engineering certification.

## Decision statuses

- `PASS`: the named measurable compatibility check passed.
- `WARNING`: known limitation exists and is reported.
- `REJECT`: task/model hard constraint failed; do not run that model.
- `NOT_VERIFIED`: evidence is absent or a measurement could not be performed.
- `DIAGNOSTIC_ONLY`: value can guide review but is not a validated gate or calibrated probability.

## Minimum evidence before deploying a checkpoint

1. Dataset source, terms, version, checksum and file inventory recorded.
2. Data audit passes: band identity, scaling, label encoding, masks, image/mask grid, class distribution and group split.
3. Training profile is fit from training rows/scenes only. Hyperparameters, thresholds, early stopping and augmentations use validation data only.
4. A checkpoint is accompanied by training config/log, hash, held-out per-scene/event metrics and failure overlays.
5. Report precision, recall, F1, IoU/mIoU and false-positive rates; do not rely on pixel accuracy alone.
6. Ablate spectral branch and radiometric augmentation on identical splits before claiming improvements.
7. Evaluate at least one independent location/domain and challenging cloud/season/resolution/alignment cases before saying “robust.”
8. Synthetic tests may verify software correctness, but are never accuracy evidence.
9. Temperature scaling is fitted only on validation data; report its NLL/Brier/ECE diagnostics and never describe scores as reliable confidence without independent calibration evidence.
10. If a GeoJSON AOI is supplied, report geometric footprint coverage and rasterized valid coverage separately for every input raster; apply an AOI mask to prediction output without pretending this corrects pair alignment.

See `RESEARCH_REFERENCES_AND_DESIGN_DECISIONS.md` for dataset and literature links plus expanded design rationale.
