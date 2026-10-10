# ChronoTerra — Model, input-quality and analysis plan

## Goal

Produce reproducible, auditable model checkpoints for the project's separate prediction tasks, rather than one universal model. Input quality and matching happen before inference. A prediction must include its input profile, label meaning, checkpoint, preprocessing, threshold/configuration, valid-pixel mask and limitations.

## Model families and label semantics

| Branch | Dataset | Input profile | Label target | Model | Important non-claim |
|---|---|---|---|---|---|
| Urban change | ONERA / OSCD | Paired aligned 13-band Sentinel-2 optical GeoTIFF | Binary candidate change mask | Siamese U-Net | Does not mean illegal encroachment, damage or even specifically building change |
| Flood extent | Sen1Floods11 hand-labelled | Single Sentinel-1 SAR raster with two channels VV/VH in dB | 0 not water; 1 water; -1 ignored | U-Net | Water mask alone does not establish flood-caused change; permanent water and SAR geometry matter |
| Building damage | xBD / xView2 | Paired RGB pre/post images concatenated to 6 channels | 0 background; 1 no damage; 2 minor; 3 major; 4 destroyed | U-Net | Only the dataset's image-based damage label; not engineering/legal certification |

The urban branch's OSCD model accepts the known OSCD band order only. The generic trainer is for the compatible data profiles described above. Do not silently convert three RGB channels into thirteen optical bands or optical imagery into Sentinel-1 SAR.

## Factors to consider before prediction

### Individual image checks

- Can every file be decoded? Is format supported?
- Width, height, channels, dtype/bit depth and pixel range.
- Non-finite pixels, raster validity masks, NoData and valid-pixel fraction.
- CRS, affine transform, bounds, resolution and data extent when GeoTIFF metadata exist.
- Available sensor/platform/date/band descriptions/tags; missing metadata remain missing, not guessed.
- Image sharpness, contrast/exposure, compression artifacts, clouds and shadows. Cloud/shadow conditions must be assessed from a suitable QA/SCL mask or a separately validated detector; generic file inspection cannot reliably identify all clouds/shadows.

### Pair matching and overlap

- Determine whether both footprints intersect in a common coordinate system.
- Keep footprint overlap, intended AOI coverage and common valid-data coverage separate.
- Compare CRS, affine transform, dimensions, resolution, channel/band profile, acquisition date and known sensor metadata.
- Matching transforms are metadata evidence—not proof of pixel-level content registration. Visual RGBs without coordinates cannot prove geographic overlap.
- If registration is not measured, return NOT_VERIFIED or WARNING. Never invent a registration score or blindly warp the images.
- After a justified reprojection/resampling operation, preserve original files, record transform history and recalculate coverage/masks. Use nearest-neighbor for label masks.

### Model checks and analysis factors

- Check model input profile (channel count and band order, sensor modality, range/normalization, resolution and crop convention).
- Mask invalid pixels for loss, metrics and outputs. Do not turn NoData into unchanged/non-water.
- Inspect class imbalance, rare class recall and scene/event variability.
- Tune epochs/checkpoint/threshold only on validation data.
- Report confusion matrix, precision, recall, F1, IoU/mIoU, false-positive rate, valid-pixel count and per-scene metrics. Do not use raw pixel accuracy as the headline metric.
- Save seed, manifest hash, data source/version, split/group IDs, preprocessing, hyperparameters, software versions, run logs and checkpoint hash.
- Results on OSCD cannot by themselves substantiate generalization to a new sensor, region, resolution or disaster event.

## Training protocol

1. Acquire real labeled data under its allowed terms; do not include data inside this source-code ZIP.
2. Build a manifest with explicit `train`, `val` and `test` splits. Keep related geographic scenes/events within a single split where feasible.
3. Run `tools/analyze_manifest.py`; check the masks, class distribution, channels, dimensions, missing files, invalid-pixel fractions and split-group leakage.
4. Run the tiny training smoke test on a temporary fixture only to verify the code path, not to report performance.
5. Train one branch at a time; record seed and configuration. The provided starting values are not claimed optimal.
6. Select checkpoint based on validation foreground mIoU. For binary tasks, select decision threshold from validation only.
7. Freeze code, preprocessing and threshold; evaluate the held-out test split once. Record per-scene variation and failure examples.
8. Compare against a simple baseline and ablate factors such as normalization, class weighting, masking and augmentation on validation/dev data. Never tune from the test report.
9. Test the model on independent scenes and real user-type inputs before enabling that workflow in the frontend.

## Runtime inputs and rejection contract

`quality_gate/pair_preflight.py` is a metadata/footprint preflight, not the whole quality gate. It can reject corrupted files, incompatible channel counts (if expected), and non-overlapping geospatial footprints. It returns NOT_VERIFIED if location metadata is missing or content-level registration is not established. It does not perform model inference or automatic warping.

Use four states:

- PASS: the specific test measured and passed.
- WARNING: a known limitation or correctable issue remains.
- REJECT: a required condition failed for the selected model/task.
- NOT_VERIFIED: enough evidence is not available.

Do not collapse different failures into one vague "low image match percentage". Report the check, evidence and required correction.

## What is still required before the product can say "well-trained model"

- Real dataset archives present and validated.
- Verified manifests and group-aware splits.
- Real training runs producing checkpoints.
- Held-out test reports from the actual dataset, plus failure-case overlays.
- Independent geographic/sensor testing for wider claims.
- Backend integration that chooses the correct model and checks its profile before inference.
- Live Windows/Antigravity and end-to-end frontend testing.
