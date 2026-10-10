"""Complete task-specific ChronoTerra prediction entry point.

Tasks:
  urban_change   : 13-band OSCD-order Sentinel-2 GeoTIFF pair -> candidate-change mask
  flood_extent   : 2-band Sentinel-1 VV/VH GeoTIFF -> water/non-water mask
  building_damage: aligned pre/post RGB images -> xBD-style 5-class pixel mask

All outputs preserve invalid pixels as nodata/255. The report records quality gates, input
factor diagnostics, model/checkpoint provenance, class distributions and uncertainty proxies.
No model is created or trained here; the requested real-data-trained checkpoint must exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from models.generic_unet import SegmentationUNet
from models.manifest_data import read_image, normalize_input
from models.input_profile import compare_input_profile
from models.siamese_unet import SiameseUNet
from models.train_manifest import tile_logits
from models.train_oscd import tiled_predict
from quality_gate.prediction_preflight import run_prediction_preflight, SCL_INVALID_CLASSES, read_scl_on_image_grid
from quality_gate.aoi_coverage import aoi_mask_and_report
from evaluation.calibration import calibrate_binary_probabilities

try:
    import rasterio
except Exception:  # pragma: no cover
    rasterio = None

CLASS_NAMES = {
    "urban_change": {0: "unchanged", 1: "candidate_change"},
    "flood_extent": {0: "not_water", 1: "water"},
    "building_damage": {0: "background", 1: "no_damage", 2: "minor_damage", 3: "major_damage", 4: "destroyed"},
}
PALETTES = {
    "urban_change": {1: (255, 55, 45)},
    "flood_extent": {1: (0, 210, 255)},
    "building_damage": {1: (50, 205, 80), 2: (255, 215, 0), 3: (255, 135, 25), 4: (235, 35, 45)},
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location=device)
    if not isinstance(obj, dict):
        raise ValueError("Checkpoint is not a ChronoTerra-compatible dictionary")
    return obj


def _read_geotiff(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[str]]:
    if rasterio is None:
        raise RuntimeError("Rasterio is required for GeoTIFF prediction")
    with rasterio.open(path) as ds:
        arr = ds.read().astype(np.float32)
        masks = ds.read_masks() > 0
        valid = np.all(masks, axis=0) & np.all(np.isfinite(arr), axis=0)
        if ds.nodata is not None:
            valid &= ~np.any(arr == ds.nodata, axis=0)
        meta = {"width": ds.width, "height": ds.height, "count": ds.count,
                "crs": ds.crs.to_string() if ds.crs else None,
                "transform": tuple(ds.transform), "bounds": tuple(ds.bounds),
                "nodata": ds.nodata, "driver": ds.driver}
        desc = [(d or "").strip() for d in ds.descriptions]
    return arr, valid, meta, desc


def _apply_scl(valid: np.ndarray, image_path: str | Path, scl_path: str | Path | None, *, class2_policy: str = "mask") -> tuple[np.ndarray, dict[str, Any]]:
    if not scl_path:
        return valid, {"status": "NOT_VERIFIED", "reason": "No Sentinel-2 SCL mask supplied; cloud/shadow removal cannot be verified."}
    try:
        _classes, scl_valid, report = read_scl_on_image_grid(image_path, scl_path, class2_policy=class2_policy)
        if scl_valid.shape != valid.shape:
            raise ValueError("SCL validity grid does not match model image grid")
        combined = valid & scl_valid
        report["combined_valid_fraction"] = float(combined.mean())
        report["excluded_fraction_after_intersection"] = float((valid & ~scl_valid).mean())
        report["status"] = "PASS" if combined.any() else "REJECT"
        if not combined.any():
            report["reason"] = "No valid pixels remain after intersecting raster validity and SCL quality mask"
        return combined, report
    except Exception as exc:
        return np.zeros_like(valid, dtype=bool), {"status": "REJECT", "reason": f"Could not align/apply SCL: {type(exc).__name__}: {exc}"}


def _stretch_rgb(arr: np.ndarray, valid: np.ndarray, channels: tuple[int, int, int]) -> np.ndarray:
    chosen = np.stack([arr[c] for c in channels], axis=-1).astype(np.float32)
    rgb = np.zeros(chosen.shape, dtype=np.uint8)
    sample_mask = valid & np.all(np.isfinite(chosen), axis=-1)
    for ci in range(3):
        values = chosen[:, :, ci][sample_mask]
        if values.size == 0:
            continue
        lo, hi = np.percentile(values, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        rgb[:, :, ci] = np.clip((chosen[:, :, ci] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return rgb


def _save_mask(path: Path, pred: np.ndarray, source_path: Path | None, nodata: int = 255) -> None:
    if source_path and source_path.suffix.lower() in {".tif", ".tiff", ".vrt"}:
        if rasterio is None:
            raise RuntimeError("Rasterio is required to save geospatial prediction")
        with rasterio.open(source_path) as src:
            profile = src.profile.copy()
        for key in ("photometric", "interleave", "predictor", "blockxsize", "blockysize", "tiled"):
            profile.pop(key, None)
        profile.update(count=1, dtype="uint8", nodata=nodata, compress="deflate")
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(pred.astype(np.uint8), 1)
            dst.set_band_description(1, "predicted_class_id; 255 means invalid/unobserved")
    else:
        Image.fromarray(pred.astype(np.uint8), mode="L").save(path)


def _save_scores(path: Path, scores: np.ndarray, source_path: Path | None, class_names: dict[int, str], valid: np.ndarray) -> None:
    safe = scores.astype(np.float32).copy()
    safe[:, ~valid] = np.nan
    if source_path and source_path.suffix.lower() in {".tif", ".tiff", ".vrt"}:
        if rasterio is None:
            raise RuntimeError("Rasterio is required to save geospatial score rasters")
        with rasterio.open(source_path) as src:
            profile = src.profile.copy()
        for key in ("photometric", "interleave", "predictor", "blockxsize", "blockysize", "tiled"):
            profile.pop(key, None)
        profile.update(count=safe.shape[0], dtype="float32", nodata=-9999.0, compress="deflate")
        with rasterio.open(path, "w", **profile) as dst:
            for i in range(safe.shape[0]):
                dst.write(np.where(valid, safe[i], -9999.0).astype(np.float32), i + 1)
                dst.set_band_description(i + 1, f"score_{class_names.get(i, str(i))}_uncalibrated")
    else:
        np.savez_compressed(path.with_suffix(".npz"), class_scores=safe,
                            class_names=np.asarray([class_names[k] for k in sorted(class_names)]))


def _save_uncertainty(path: Path, scores: np.ndarray, valid: np.ndarray, source_path: Path | None) -> dict[str, Any]:
    eps = 1e-8
    entropy = -(np.clip(scores, eps, 1.0) * np.log(np.clip(scores, eps, 1.0))).sum(axis=0)
    if scores.shape[0] > 1:
        entropy = entropy / np.log(scores.shape[0])
        sorted_scores = np.sort(scores, axis=0)
        margin = sorted_scores[-1] - sorted_scores[-2]
    else:
        margin = np.ones_like(entropy)
    entropy = np.where(valid, entropy, np.nan).astype(np.float32)
    margin = np.where(valid, margin, np.nan).astype(np.float32)
    if source_path and source_path.suffix.lower() in {".tif", ".tiff", ".vrt"}:
        with rasterio.open(source_path) as src:
            profile = src.profile.copy()
        for key in ("photometric", "interleave", "predictor", "blockxsize", "blockysize", "tiled"):
            profile.pop(key, None)
        profile.update(count=2, dtype="float32", nodata=-9999.0, compress="deflate")
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(np.where(valid, entropy, -9999.0).astype(np.float32), 1)
            dst.set_band_description(1, "normalized_entropy_uncertainty_proxy_not_calibrated")
            dst.write(np.where(valid, margin, -9999.0).astype(np.float32), 2)
            dst.set_band_description(2, "top1_minus_top2_score_margin_not_calibrated")
    else:
        image = (np.clip(np.nan_to_num(entropy, nan=0.0), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(image, mode="L").save(path)
        np.savez_compressed(path.with_suffix(".npz"), normalized_entropy=entropy, top1_top2_margin=margin)
    vals = entropy[valid]
    return {"definition": "normalized entropy and top-1/top-2 score margin; not calibrated predictive uncertainty",
            "mean_normalized_entropy": float(np.mean(vals)) if vals.size else None,
            "p95_normalized_entropy": float(np.percentile(vals, 95)) if vals.size else None}


def _save_overlay(path: Path, rgb: np.ndarray, pred: np.ndarray, valid: np.ndarray, task: str) -> None:
    base = rgb.astype(np.float32)
    overlay = base.copy()
    pal = PALETTES[task]
    # Avoid coloring invalid/unobserved cells.
    for cls, color in pal.items():
        m = (pred == cls) & valid
        if m.any():
            overlay[m] = 0.55 * overlay[m] + 0.45 * np.asarray(color, dtype=np.float32)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").save(path)


def _validate_checkpoint(task: str, ck: dict[str, Any]) -> tuple[torch.nn.Module, dict[str, Any]]:
    if task == "urban_change":
        cfg = ck.get("model_config", {})
        arch = cfg.get("architecture")
        if arch != "SiameseUNet-SpectralSpatial-v2":
            raise ValueError(f"Urban checkpoint architecture '{arch}' is not the current spectral-spatial v2 architecture; retrain using the supplied updated trainer.")
        if ck.get("bands") != ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]:
            raise ValueError("Urban checkpoint band order does not match ChronoTerra's OSCD profile")
        model = SiameseUNet(in_channels=int(cfg.get("in_channels", 13)), base_channels=int(cfg.get("base_channels", 16)))
        model.load_state_dict(ck["model_state_dict"], strict=True)
        return model, cfg
    cfg = ck.get("config", {})
    if cfg.get("task") != task:
        raise ValueError(f"Checkpoint task '{cfg.get('task')}' does not match requested task '{task}'")
    expected = {"flood_extent": (2, 2), "building_damage": (6, 5)}[task]
    if (int(cfg.get("in_channels", -1)), int(cfg.get("num_classes", -1))) != expected:
        raise ValueError(f"Checkpoint profile mismatch for {task}: expected in_channels/classes={expected}")
    model = SegmentationUNet(int(cfg["in_channels"]), int(cfg["num_classes"]), int(cfg["base_channels"]))
    model.load_state_dict(ck["state_dict"], strict=True)
    return model, cfg


def predict_task(task: str, checkpoint: str | Path, output_prefix: str | Path, *,
                 input_path: str | Path | None = None, before_path: str | Path | None = None,
                 after_path: str | Path | None = None, before_scl: str | Path | None = None,
                 after_scl: str | Path | None = None, pair_metadata_json: str | Path | None = None,
                 assume_band_order: bool = False, aoi_geojson: str | Path | None = None,
                 scl_class2_policy: str = "mask",
                 tile_size: int = 256, overlap: int = 64, device_name: str = "auto") -> dict[str, Any]:
    if task not in CLASS_NAMES:
        raise ValueError(f"Unknown task: {task}")
    if tile_size < 32 or tile_size % 16 or overlap < 0 or overlap >= tile_size:
        raise ValueError("tile-size must be >=32 and divisible by 16; overlap must be in [0,tile-size)")
    checkpoint = Path(checkpoint).resolve(); output_prefix = Path(output_prefix).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Trained checkpoint not found: {checkpoint}. Train and evaluate the model before enabling inference.")
    if device_name == "auto": device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but no CUDA-enabled PyTorch device is available")

    preflight = run_prediction_preflight(task, before_path=before_path, after_path=after_path,
                                         input_path=input_path, before_scl=before_scl, after_scl=after_scl,
                                         pair_metadata_json=pair_metadata_json, assume_band_order=assume_band_order,
                                         aoi_geojson=aoi_geojson, scl_class2_policy=scl_class2_policy)
    if preflight["status"] == "REJECT":
        raise ValueError("Prediction preflight rejected the input pair: " + "; ".join(preflight["hard_errors"]))

    ck = _torch_load(checkpoint, device)
    model, cfg = _validate_checkpoint(task, ck)
    model = model.to(device).eval()
    source_path: Path
    input_paths: dict[str, str]
    warnings = list(preflight.get("warnings", []))

    if task == "urban_change":
        braw, bv, bm, bdesc = _read_geotiff(before_path)
        araw, av, am, adesc = _read_geotiff(after_path)
        if braw.shape != araw.shape:
            raise ValueError("Urban model requires matching before/after shapes")
        valid = bv & av
        scl_reports = {}
        for lbl, sclpath, imagepath in (("before", before_scl, before_path), ("after", after_scl, after_path)):
            valid, qa_report = _apply_scl(valid, imagepath, sclpath, class2_policy=scl_class2_policy)
            scl_reports[lbl] = qa_report
            if qa_report.get("status") == "NOT_VERIFIED":
                warnings.append(f"{lbl} cloud/shadow filtering was not verified; review output carefully")
            elif qa_report.get("status") == "REJECT":
                raise ValueError(f"{lbl} SCL quality mask could not be applied: {qa_report.get('reason')}")
        if not valid.any(): raise ValueError("No common valid pixels after NoData and QA masks")
        scale = float(ck["scale_factor"])
        if not np.isfinite(scale) or scale <= 0: raise ValueError("Invalid scale factor in urban checkpoint")
        # Catch the most dangerous silent scale mismatch before normalizing.
        sample_step = max(1, int(max(braw.shape[-2:]) / 512))
        scale_sample = np.concatenate([braw[:, ::sample_step, ::sample_step].reshape(-1), araw[:, ::sample_step, ::sample_step].reshape(-1)])
        scale_sample = scale_sample[np.isfinite(scale_sample)]
        p99_abs = float(np.percentile(np.abs(scale_sample), 99)) if scale_sample.size else 0.0
        if scale >= 100.0 and p99_abs <= 1.5:
            raise ValueError("Input appears pre-normalized/unit-scale, but this checkpoint expects raw reflectance/DN values divided by its stored scale_factor. Check for double normalization.")
        if scale <= 1.0 and p99_abs > 2.0:
            raise ValueError("Input appears raw DN rather than unit reflectance, but this checkpoint was trained with scale_factor=1. Verify preprocessing; refusing silent scale mismatch.")
        b = np.clip(np.nan_to_num(braw, nan=0.0, posinf=0.0, neginf=0.0) / scale, 0, 1).astype(np.float32)
        a = np.clip(np.nan_to_num(araw, nan=0.0, posinf=0.0, neginf=0.0) / scale, 0, 1).astype(np.float32)
        prob_raw = tiled_predict(model, b, a, device, tile_size=tile_size, overlap=overlap)
        temperature = float(ck.get("temperature", 1.0))
        prob = calibrate_binary_probabilities(prob_raw, temperature)
        threshold = float(ck.get("threshold", 0.5))
        pred = (prob >= threshold).astype(np.uint8)
        scores = np.stack([1.0-prob, prob], axis=0)
        drift_report = compare_input_profile(np.concatenate([b, a], axis=0), valid,
                                             ck.get("input_profile"), seed=17)
        preflight["factors"]["input_distribution_shift"] = drift_report
        if drift_report.get("review_flags"):
            warnings.append("Input channel distributions differ from training reference; review the input-shift diagnostic before interpreting results")
        elif drift_report.get("status") == "NOT_VERIFIED":
            warnings.append("Checkpoint has no training-distribution profile; input shift is not verified")
        rgb = _stretch_rgb(braw, valid, (3, 2, 1))
        source_path = Path(before_path).resolve()
        input_paths = {"before": str(Path(before_path).resolve()), "after": str(Path(after_path).resolve())}
        class_names = CLASS_NAMES[task]
        model_description = {"architecture": cfg.get("architecture"), "bands": ck.get("bands"), "scale_factor": scale,
                             "threshold_from_validation": threshold, "training_epoch": ck.get("epoch"),
                             "radiometric_augmentation": ck.get("augmentation", "checkpoint did not record augmentation details")}
        preflight["factors"]["cloud_shadow_masks_used_at_inference"] = scl_reports
    elif task == "flood_extent":
        xraw, meta = read_image(str(input_path))
        source_path = Path(input_path).resolve()
        valid = meta.get("valid_mask")
        if not isinstance(valid, np.ndarray): valid = np.all(np.isfinite(xraw), axis=0)
        x = normalize_input(xraw, cfg["normalization"]).astype(np.float32)
        logits = tile_logits(model, torch.from_numpy(x), device, tile_size=tile_size, overlap=min(overlap, tile_size-1))
        temperature = float(cfg.get("temperature", 1.0))
        scores = torch.softmax(logits / max(temperature, 1e-6), dim=0).cpu().numpy()
        pred = scores.argmax(axis=0).astype(np.uint8)
        drift_report = compare_input_profile(x, valid, cfg.get("input_profile"), seed=17)
        preflight["factors"]["input_distribution_shift"] = drift_report
        if drift_report.get("review_flags"):
            warnings.append("Input channel distributions differ from training reference; review the input-shift diagnostic before interpreting results")
        elif drift_report.get("status") == "NOT_VERIFIED":
            warnings.append("Checkpoint has no training-distribution profile; input shift is not verified")
        if cfg.get("decision_threshold") is not None:
            pred = (scores[1] >= float(cfg["decision_threshold"])).astype(np.uint8)
        pred[~valid] = 255; scores[:, ~valid] = np.nan
        rgb = np.repeat(_stretch_rgb(xraw, valid, (0, 0, 0))[:, :, :1], 3, axis=2)
        source_path = Path(input_path).resolve(); input_paths = {"input": str(source_path)}
        class_names = CLASS_NAMES[task]
        model_description = {"architecture": "SegmentationUNet", "input_profile": "Sentinel-1 VV/VH", "normalization": cfg["normalization"],
                             "training_epoch": ck.get("epoch"), "threshold_from_validation": cfg.get("decision_threshold")}
    else:  # paired RGB xBD-style building damage
        braw, bm = read_image(str(before_path)); araw, am = read_image(str(after_path))
        if braw.shape[1:] != araw.shape[1:]: raise ValueError("Building-damage model requires matching image dimensions")
        if braw.shape[0] != 3 or araw.shape[0] != 3: raise ValueError("Building-damage model requires RGB before/after images")
        if bm.get("driver") != "PIL" and am.get("driver") != "PIL":
            if (bm.get("crs") or am.get("crs")) and (not bm.get("crs") or not am.get("crs") or bm.get("crs") != am.get("crs")):
                raise ValueError("Georeferenced building-damage rasters must have matching valid CRS metadata; reproject explicitly before inference.")
            if (bm.get("transform") or am.get("transform")) and (not bm.get("transform") or not am.get("transform") or bm.get("transform") != am.get("transform")):
                raise ValueError("Georeferenced building-damage rasters must share the same affine grid; blind pixelwise comparison is rejected.")
        xraw = np.concatenate([braw, araw], axis=0)
        valid_b, valid_a = bm.get("valid_mask"), am.get("valid_mask")
        valid = (valid_b & valid_a) if isinstance(valid_b, np.ndarray) and isinstance(valid_a, np.ndarray) else np.ones(xraw.shape[1:], bool)
        x = normalize_input(xraw, cfg["normalization"]).astype(np.float32)
        logits = tile_logits(model, torch.from_numpy(x), device, tile_size=tile_size, overlap=min(overlap, tile_size-1))
        temperature = float(cfg.get("temperature", 1.0))
        scores = torch.softmax(logits / max(temperature, 1e-6), dim=0).cpu().numpy()
        pred = scores.argmax(axis=0).astype(np.uint8)
        drift_report = compare_input_profile(x, valid, cfg.get("input_profile"), seed=17)
        preflight["factors"]["input_distribution_shift"] = drift_report
        if drift_report.get("review_flags"):
            warnings.append("Input channel distributions differ from training reference; review the input-shift diagnostic before interpreting results")
        elif drift_report.get("status") == "NOT_VERIFIED":
            warnings.append("Checkpoint has no training-distribution profile; input shift is not verified")
        pred[~valid] = 255; scores[:, ~valid] = np.nan
        after_rgb = np.moveaxis(araw, 0, -1)
        rgb = np.clip(after_rgb, 0, 255).astype(np.uint8)
        source_path = Path(before_path).resolve()
        input_paths = {"before": str(Path(before_path).resolve()), "after": str(Path(after_path).resolve())}
        class_names = CLASS_NAMES[task]
        model_description = {"architecture": "SegmentationUNet", "input_profile": "paired RGB pre/post-disaster", "normalization": cfg["normalization"],
                             "training_epoch": ck.get("epoch")}

    # AOI is a user-supplied geographic geometry, distinct from image footprint overlap.
    aoi_runtime_report = None
    if aoi_geojson is not None:
        aoi_mask, aoi_runtime_report = aoi_mask_and_report(aoi_geojson, source_path)
        if aoi_mask.shape != valid.shape:
            raise ValueError(f"AOI mask grid {aoi_mask.shape} does not match prediction grid {valid.shape}")
        before_aoi_valid = int(valid.sum())
        valid_inside_aoi = valid & aoi_mask
        inside_count = int(aoi_mask.sum())
        aoi_runtime_report["valid_common_pixels_inside_aoi_for_prediction"] = int(valid_inside_aoi.sum())
        aoi_runtime_report["valid_fraction_inside_aoi_after_pair_and_quality_masks"] = (float(valid_inside_aoi.sum()/inside_count) if inside_count else 0.0)
        aoi_runtime_report["valid_pixels_before_aoi_restriction"] = before_aoi_valid
        valid = valid_inside_aoi
        if not valid.any():
            raise ValueError("No valid model input pixels remain inside the supplied AOI after intersecting source, pair and quality masks")
        pred[~valid] = 255
        scores[:, ~valid] = np.nan
        preflight.setdefault("factors", {})["aoi_coverage"]["runtime_prediction_mask"] = aoi_runtime_report
        if aoi_runtime_report.get("status") == "WARNING":
            warnings.append("AOI extends beyond image coverage; outputs are limited to covered, valid pixels")

    # All task branches share the same file-writing and uncertainty-report contract.
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    geo_output = source_path.suffix.lower() in {".tif", ".tiff", ".vrt"}
    mask_path = output_prefix.with_name(output_prefix.name + ("_mask.tif" if geo_output else "_mask.png"))
    score_path = output_prefix.with_name(output_prefix.name + ("_scores.tif" if geo_output else "_scores.npz"))
    unc_path = output_prefix.with_name(output_prefix.name + ("_uncertainty.tif" if geo_output else "_uncertainty.png"))
    overlay_path = output_prefix.with_name(output_prefix.name + "_overlay.png")
    report_path = output_prefix.with_name(output_prefix.name + "_report.json")
    pred_out = pred.astype(np.uint8).copy(); pred_out[~valid] = 255
    _save_mask(mask_path, pred_out, source_path if geo_output else None)
    _save_scores(score_path, scores, source_path if geo_output else None, class_names, valid)
    uncertainty = _save_uncertainty(unc_path, scores, valid, source_path if geo_output else None)
    _save_overlay(overlay_path, rgb, pred_out, valid, task)
    values, counts = np.unique(pred_out[valid], return_counts=True)
    dist = {class_names.get(int(v), f"class_{int(v)}"): int(n) for v, n in zip(values, counts)}
    report = {
        "status": "INFERENCE_COMPLETED_WITH_WARNINGS" if warnings else "INFERENCE_COMPLETED",
        "task": task, "task_output_semantics": {str(k): v for k, v in class_names.items()},
        "input_paths": input_paths, "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_epoch": ck.get("epoch"), "model": model_description, "device": str(device),
        "prediction_score_shape_chw": [int(scores.shape[0]), int(pred.shape[0]), int(pred.shape[1])],
        "input_valid_grid_shape_hw": [int(pred.shape[0]), int(pred.shape[1])],
        "valid_fraction_exact": float(valid.mean()), "predicted_class_pixel_counts": dist,
        "mask_output": str(mask_path), "scores_output": str(score_path), "uncertainty_output": str(unc_path),
        "overlay_output": str(overlay_path), "report_output": str(report_path),
        "score_calibration": ({"method": "validation-only scalar temperature scaling", "temperature": float(ck.get("temperature", 1.0)), "report": ck.get("calibration_report")} if task == "urban_change" else {"method": "validation-only scalar temperature scaling", "temperature": float(cfg.get("temperature", 1.0)), "report": cfg.get("calibration_report")}),
        "scores_are_calibrated_probabilities": False,
        "scores_temperature_scaled_on_validation": bool((ck.get("calibration_report") if task == "urban_change" else cfg.get("calibration_report")) and (ck.get("calibration_report") if task == "urban_change" else cfg.get("calibration_report")).get("status") == "FITTED_ON_VALIDATION_SAMPLE"),
        "uncertainty_is_calibrated": False, "aoi_geojson": str(Path(aoi_geojson).resolve()) if aoi_geojson else None,
        "uncertainty_summary": uncertainty, "preflight": preflight,
        "warnings": warnings + ["Prediction output is a model estimate, not ground-truth validation.",
                                 "Scores are not calibrated probabilities. Do not interpret candidate urban change as legal encroachment or flood-water mapping as building damage.",
                                 "A checkpoint file alone does not prove that its training data were real or its test performance is acceptable; inspect its training/held-out evaluation reports."],
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=sorted(CLASS_NAMES), required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--input", help="Single 2-band Sentinel-1 GeoTIFF for flood extent")
    ap.add_argument("--before", help="Before image for urban change/building damage")
    ap.add_argument("--after", help="After image for urban change/building damage")
    ap.add_argument("--before-scl", help="Optional co-gridded Sentinel-2 Scene Classification Layer (SCL) GeoTIFF")
    ap.add_argument("--after-scl", help="Optional co-gridded Sentinel-2 Scene Classification Layer (SCL) GeoTIFF")
    ap.add_argument("--pair-metadata-json", help="Optional source pair_metadata.json with date/sensor provenance")
    ap.add_argument("--aoi-geojson", help="Optional WGS84 GeoJSON Polygon/MultiPolygon to limit valid prediction pixels")
    ap.add_argument("--scl-class2-policy", choices=["mask", "keep"], default="mask", help="Keep class 2 only for a verified legacy SCL product")
    ap.add_argument("--assume-band-order", action="store_true", help="Use only after you verified the OSCD source band order")
    ap.add_argument("--tile-size", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a = ap.parse_args()
    try:
        report = predict_task(a.task, a.checkpoint, a.output_prefix, input_path=a.input,
                              before_path=a.before, after_path=a.after, before_scl=a.before_scl,
                              after_scl=a.after_scl, pair_metadata_json=a.pair_metadata_json,
                              assume_band_order=a.assume_band_order, aoi_geojson=a.aoi_geojson,
                              scl_class2_policy=a.scl_class2_policy,
                              tile_size=a.tile_size, overlap=a.overlap, device_name=a.device)
        print(json.dumps(report, indent=2, default=str))
    except Exception as exc:
        print(json.dumps({"status": "PREDICTION_FAILED", "error_type": type(exc).__name__, "reason": str(exc)}, indent=2))
        raise SystemExit(2)

if __name__ == "__main__":
    main()
