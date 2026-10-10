# Pair preflight

Run from the repository root:

```powershell
python -m quality_gate.pair_preflight --before path\before.tif --after path\after.tif --expected-channels-each 13 --require-georeferencing --output outputs\analysis\pair_preflight.json
```

This inspects file readability, dimensions, channels, valid pixels and available GeoTIFF metadata, then estimates footprint overlap if both inputs have usable CRS/bounds. It does **not** establish content-level registration, cloud coverage, radiometric calibration, model accuracy or legal boundaries. The returned overall status is deliberately `NOT_VERIFIED` if content registration was not independently established, even when metadata overlap passes.

A secondary translation estimate can be produced for review:

```powershell
python -m quality_gate.registration_diagnostic --before path\before.tif --after path\after.tif --output outputs\analysis\registration_estimate.json
```

The reported phase-correlation shift and response are diagnostics only. They are **not** universal acceptance thresholds and they must not trigger automatic warping. Real changes and clouds can affect the estimate. Rotation, local distortion, AOI mismatch and actual alignment still need to be assessed.

## Descriptive image-quality report

Run `python -m quality_gate.image_quality --image path/to/image.tif --profile oscd13 --output outputs/analysis/image_quality.json` for an OSCD-ordered 13-band raster, or omit `--profile oscd13` for auto-detection. It reports per-channel robust percentiles and validity plus advisory brightness/contrast/sharpness proxies where a visual channel profile can be established. Do not reject on an uncalibrated proxy threshold. The output explicitly leaves cloud/shadow detection as `NOT_VERIFIED` unless a QA mask or validated detector is separately applied.
