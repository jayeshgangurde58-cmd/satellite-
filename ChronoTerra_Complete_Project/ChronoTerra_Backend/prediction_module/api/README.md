# ChronoTerra local API

From the repository root: `python -m uvicorn api.app:app --reload`; open `http://127.0.0.1:8000`.

## Routes

- `GET /api/health`: liveness and execution-mode metadata.
- `GET /api/models/status`: checkpoint existence, training-summary existence, and held-out metrics-file existence. File presence alone does not prove model validity.
- `POST /api/preflight`: multipart quality-factor report without needing trained weights. Paired tasks use `before_file` and `after_file`; flood uses `input_file`. Optional `before_scl_file`, `after_scl_file`, `pair_metadata_file`, `aoi_geojson_file`, and `scl_class2_policy` add QA/provenance evidence and user-defined AOI clipping. AOI upload requires GeoTIFFs with trustworthy georeferencing; it is assessed per raster and masks output outside the AOI.
- `POST /api/predict`: task-specific trained-model inference. It requires `task` in `urban_change`, `flood_extent`, `building_damage`, and matching uploaded files. If a checkpoint is absent, it returns HTTP 503 with `MODEL_NOT_AVAILABLE`; it never fabricates predictions.
- `POST /api/analyze`: the classical candidate-change baseline for compatible GeoTIFF pairs; this is distinct from learned-model inference.
- `/outputs/<session>/<artifact>`: masks, scores, uncertainty proxies, overlays, reports, and classical output artifacts.

Checkpoint paths default to `outputs/models/oscd_siamese_v2/best.pt`, `outputs/models/sen1floods11_sar_unet_v1/best.pt`, and `outputs/models/xbd_damage_unet_v1/best.pt`. Override using `CHRONOTERRA_URBAN_CHECKPOINT`, `CHRONOTERRA_FLOOD_CHECKPOINT`, and `CHRONOTERRA_DAMAGE_CHECKPOINT`.

## Input contracts

- Urban change: two compatible 13-band OSCD-order Sentinel-2 GeoTIFFs, same CRS/affine grid and supported scale. If band descriptions are missing, use `assume_band_order=true` only after independently verifying source provenance.
- Flood extent: two-band Sentinel-1 VV then VH GeoTIFF in dB scale compatible with model preprocessing. Missing descriptions require explicit, verified band-order assertion.
- Building damage: paired RGB pre/post-disaster images in byte 0–255 range, matching dimensions; geographic identity cannot be proved from dimensions alone.

The API is intended for a local research/demo environment, not a hardened public upload service. No real trained checkpoints are bundled; therefore learned prediction routes remain unavailable until each model has been trained and evaluated. The UI in `web/` is a minimal local API demo; the user's separate React workspace has not been merged or verified here.
