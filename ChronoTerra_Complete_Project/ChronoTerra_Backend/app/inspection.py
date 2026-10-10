from __future__ import annotations

import io
import math
import os
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
import rasterio
from rasterio.io import MemoryFile


MAX_IMAGE_PIXELS = int(os.getenv("CHRONOTERRA_MAX_IMAGE_PIXELS", "40000000"))
SAMPLE_MAX_DIM = int(os.getenv("CHRONOTERRA_SAMPLE_MAX_DIM", "512"))

SUPPORTED_MAGICS = {
    "jpeg": (b"\xff\xd8\xff",),
    "png": (b"\x89PNG\r\n\x1a\n",),
    "tiff": (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"),
}


def detect_format(data: bytes) -> str | None:
    for fmt, signatures in SUPPORTED_MAGICS.items():
        if any(data.startswith(signature) for signature in signatures):
            return fmt
    return None


def _status(checks: list[dict[str, Any]], *, base_warning: bool = False) -> str:
    if any(c.get("status") == "REJECT" for c in checks):
        return "REJECT"
    if base_warning or any(c.get("status") in {"WARNING", "NOT_VERIFIED"} for c in checks):
        return "WARNING"
    return "PASS"


def _check(name: str, status: str, message: str, value: Any = None) -> dict[str, Any]:
    result = {"name": name, "status": status, "message": message}
    if value is not None:
        result["value"] = value
    return result


def _safe_json_number(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


def _visual_statistics(rgb: np.ndarray, alpha: np.ndarray | None = None) -> dict[str, Any]:
    """Compute transparent, explicitly heuristic visual diagnostics on a small sample."""
    if rgb.size == 0:
        return {"status": "NOT_VERIFIED", "message": "No readable sample pixels were available."}

    sample = np.asarray(rgb, dtype=np.float32)
    finite = np.isfinite(sample).all(axis=2)
    if alpha is not None:
        valid_alpha = np.asarray(alpha) > 0
        finite &= valid_alpha
        transparent_fraction = float(1.0 - np.mean(valid_alpha))
    else:
        transparent_fraction = None

    valid = sample[finite]
    if valid.size == 0:
        return {"status": "REJECT", "message": "No finite, visible sample pixels were found."}

    channel_stats = []
    for idx in range(3):
        ch = valid[:, idx]
        channel_stats.append({
            "min": _safe_json_number(np.min(ch)),
            "max": _safe_json_number(np.max(ch)),
            "mean": _safe_json_number(np.mean(ch)),
            "stddev": _safe_json_number(np.std(ch)),
            "clipped_low_fraction": _safe_json_number(np.mean(ch <= 1.0)),
            "clipped_high_fraction": _safe_json_number(np.mean(ch >= 254.0)),
        })

    gray = sample[..., 0] * 0.299 + sample[..., 1] * 0.587 + sample[..., 2] * 0.114
    gray_valid = gray[finite]
    contrast_std = _safe_json_number(np.std(gray_valid))
    p05, p95 = np.percentile(gray_valid, [5, 95])
    dynamic_range_p05_p95 = _safe_json_number(p95 - p05)

    # A simple gradient-energy statistic is an image-quality clue only; it is not
    # a calibrated blur score and must never be presented as a definitive rejection.
    if gray.shape[0] >= 2 and gray.shape[1] >= 2:
        gy, gx = np.gradient(gray.astype(np.float32))
        gradient_energy = _safe_json_number(np.mean((gx[finite] ** 2) + (gy[finite] ** 2)))
    else:
        # Extremely narrow images can become one pixel wide after sampling.
        # In that case a 2-D gradient estimate is mathematically unavailable.
        gradient_energy = None

    notes: list[str] = []
    if contrast_std is not None and contrast_std < 8.0:
        notes.append("Low contrast is suspected from a small grayscale standard deviation; inspect visually.")
    if dynamic_range_p05_p95 is not None and dynamic_range_p05_p95 < 20.0:
        notes.append("The sampled 5th–95th percentile intensity range is narrow; inspect exposure and contrast.")
    if gradient_energy is not None and gradient_energy < 2.0:
        notes.append("Low edge/gradient energy may indicate blur or a naturally smooth scene; this is only a heuristic.")
    if any((s["clipped_low_fraction"] or 0) > 0.2 for s in channel_stats):
        notes.append("A high proportion of low-valued pixels was found in at least one sampled channel.")
    if any((s["clipped_high_fraction"] or 0) > 0.2 for s in channel_stats):
        notes.append("A high proportion of high-valued pixels was found in at least one sampled channel.")

    return {
        "status": "WARNING" if notes else "PASS",
        "sampled_pixel_count": int(sample.shape[0] * sample.shape[1]),
        "channel_statistics": channel_stats,
        "grayscale_stddev": contrast_std,
        "grayscale_p05_p95_range": dynamic_range_p05_p95,
        "gradient_energy_heuristic": gradient_energy,
        "transparent_fraction": transparent_fraction,
        "notes": notes,
        "limitations": [
            "Visual quality statistics are heuristic and do not establish satellite product quality.",
            "Cloud, shadow, haze and semantic scene quality are not reliably classified by these statistics alone.",
        ],
    }


def _sample_size(width: int, height: int) -> tuple[int, int]:
    scale = min(1.0, SAMPLE_MAX_DIM / max(width, height))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def inspect_raster(path: str, filename: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    with rasterio.open(path) as ds:
        width, height, count = ds.width, ds.height, ds.count
        if width <= 0 or height <= 0 or count <= 0:
            checks.append(_check("raster_dimensions", "REJECT", "Raster dimensions or band count are invalid."))
            return {
                "filename": filename, "detected_format": "tiff", "status": "REJECT",
                "dimensions": {"width": width, "height": height, "pixels": max(0, width) * max(0, height)},
                "channels": max(0, count), "mode": "raster", "metadata": {"crs": None, "bounds": None},
                "valid_pixel_sample": None, "visual_quality": {"status": "NOT_VERIFIED"},
                "checks": checks, "warnings": [], "limitations": ["Raster dimensions or bands are invalid."],
            }
        if count > 64:
            checks.append(_check("band_limit", "REJECT", f"Raster has {count} bands, exceeding the configured inspection limit of 64 bands.", {"bands": count, "max_bands": 64}))
            return {
                "filename": filename, "detected_format": "tiff", "status": "REJECT",
                "dimensions": {"width": width, "height": height, "pixels": width * height},
                "channels": count, "mode": "raster", "metadata": {"crs": ds.crs.to_string() if ds.crs else None},
                "valid_pixel_sample": None, "visual_quality": {"status": "NOT_VERIFIED"},
                "checks": checks, "warnings": [checks[-1]["message"]],
                "limitations": ["Inspection band-count limit was enforced before sampling raster values."],
            }
        pixels = width * height
        if pixels > MAX_IMAGE_PIXELS:
            checks.append(_check("pixel_limit", "REJECT", f"Raster contains {pixels:,} pixels, exceeding the configured limit of {MAX_IMAGE_PIXELS:,}.", {"pixels": pixels, "max_pixels": MAX_IMAGE_PIXELS}))
        else:
            checks.append(_check("raster_dimensions", "PASS", "Raster dimensions and band count are readable.", {"width": width, "height": height, "bands": count, "pixels": pixels}))
            checks.append(_check("pixel_limit", "PASS", "Raster is within the configured pixel limit."))

        descriptions = list(ds.descriptions) if ds.descriptions else []
        nodata_values = list(ds.nodatavals) if ds.nodatavals else []
        crs = ds.crs.to_string() if ds.crs else None
        raw_transform = tuple(float(v) for v in ds.transform) if ds.transform else None
        transform = raw_transform if raw_transform and all(math.isfinite(v) for v in raw_transform) else None
        raw_bounds = {
            "left": _safe_json_number(ds.bounds.left),
            "bottom": _safe_json_number(ds.bounds.bottom),
            "right": _safe_json_number(ds.bounds.right),
            "top": _safe_json_number(ds.bounds.top),
        }
        bounds_are_finite = all(v is not None for v in raw_bounds.values())
        bounds = raw_bounds if bounds_are_finite else None
        resolution = [_safe_json_number(v) for v in ds.res]
        sample_h, sample_w = _sample_size(width, height)
        out_shape = (count, sample_h, sample_w)
        sampled = ds.read(out_shape=out_shape, masked=True)
        mask = ds.dataset_mask(out_shape=(sample_h, sample_w)) > 0
        all_bands_finite = np.isfinite(np.asarray(sampled.astype(np.float64).filled(np.nan), dtype=np.float64)).all(axis=0)
        valid_sample = mask & all_bands_finite
        invalid_fraction = float(1.0 - np.mean(valid_sample)) if valid_sample.size else 1.0
        if not np.any(valid_sample):
            checks.append(_check("valid_pixels", "REJECT", "No valid finite pixels were found in the sampled raster."))
        else:
            valid_count = int(np.count_nonzero(valid_sample))
            checks.append(_check("valid_pixels", "PASS" if invalid_fraction == 0 else "WARNING", "Valid-pixel sample inspected; invalid cells remain excluded.", {"sampled_valid_pixels": valid_count, "sampled_pixels": int(valid_sample.size), "invalid_fraction_estimate": invalid_fraction}))

        per_band_stats: list[dict[str, Any]] = []
        sampled_data = np.asarray(sampled.astype(np.float64).filled(np.nan), dtype=np.float64)
        for band_idx in range(count):
            values = sampled_data[band_idx][valid_sample & np.isfinite(sampled_data[band_idx])]
            if values.size == 0:
                per_band_stats.append({"band": band_idx + 1, "status": "NOT_VERIFIED"})
            else:
                per_band_stats.append({
                    "band": band_idx + 1,
                    "description": descriptions[band_idx] if band_idx < len(descriptions) else None,
                    "dtype": ds.dtypes[band_idx],
                    "nodata": _safe_json_number(nodata_values[band_idx]) if band_idx < len(nodata_values) else None,
                    "sample_min": _safe_json_number(np.min(values)),
                    "sample_max": _safe_json_number(np.max(values)),
                    "sample_mean": _safe_json_number(np.mean(values)),
                    "sample_stddev": _safe_json_number(np.std(values)),
                })

        if crs:
            checks.append(_check("georeferencing", "PASS", "A CRS is present in the raster metadata.", {"crs": crs, "bounds": bounds, "resolution": resolution, "transform": transform}))
        else:
            checks.append(_check("georeferencing", "NOT_VERIFIED", "No CRS was found. Geographic bounds/coordinates cannot be trusted from this file alone."))

        is_georeferenced = bool(crs and transform and bounds_are_finite and not ds.transform.is_identity)
        metadata = {
            "crs": crs,
            "bounds": bounds if is_georeferenced else None,
            "transform": transform,
            "resolution": resolution,
            "band_descriptions": descriptions,
            "nodata_values": nodata_values,
            "dtypes": list(ds.dtypes),
            "driver": ds.driver,
            "block_shapes": [list(shape) for shape in ds.block_shapes],
            "is_georeferenced": is_georeferenced,
        }
        # Rendering sample only: choose first three bands where possible, repeat
        # a single band for visual diagnostics; these are not altered source data.
        if count >= 3:
            visual_bands = sampled_data[:3]
        elif count == 2:
            visual_bands = np.stack([sampled_data[0], sampled_data[1], sampled_data[1]])
        else:
            visual_bands = np.repeat(sampled_data[:1], 3, axis=0)
        rgb = np.zeros((sample_h, sample_w, 3), dtype=np.float32)
        for channel in range(3):
            values = visual_bands[channel]
            finite_vals = values[valid_sample & np.isfinite(values)]
            if finite_vals.size:
                lo, hi = np.percentile(finite_vals, [2, 98])
                if hi > lo:
                    rgb[..., channel] = np.clip((values - lo) / (hi - lo) * 255.0, 0, 255)
        visual = _visual_statistics(rgb, alpha=valid_sample.astype(np.uint8))

        if count not in (1, 3, 4):
            checks.append(_check("display_channels", "WARNING", f"Raster has {count} bands. Display rendering is only a preview; model compatibility depends on the selected model's input contract.", {"band_count": count}))
        else:
            checks.append(_check("display_channels", "PASS", "Raster band count was read successfully; this does not prove compatibility with a prediction model.", {"band_count": count}))
        checks.append(_check("model_input_compatibility", "NOT_VERIFIED", "No trained model input contract is configured yet; channel count alone cannot approve this raster for prediction."))
        checks.append(_check("cloud_shadow_quality_mask", "NOT_VERIFIED", "No product-specific cloud/shadow mask was interpreted in this generic raster inspection. Supply and validate the correct mask/product metadata for optical analysis."))
        checks.append(_check("radiometric_compatibility", "NOT_VERIFIED", "Band units, scale/offset and product calibration are not proven compatible merely by reading the raster."))

        status = _status(checks)
        return {
            "filename": filename,
            "detected_format": "tiff",
            "status": status,
            "dimensions": {"width": width, "height": height, "pixels": width * height},
            "channels": count,
            "mode": "raster",
            "metadata": metadata,
            "valid_pixel_sample": {"sampled_pixels": int(valid_sample.size), "valid_pixels": int(np.count_nonzero(valid_sample)), "invalid_fraction_estimate": invalid_fraction},
            "band_statistics": per_band_stats,
            "visual_quality": visual,
            "checks": checks,
            "warnings": [c["message"] for c in checks if c["status"] in {"WARNING", "NOT_VERIFIED"}],
            "limitations": [
                "No source image was written to persistent storage by this inspection request.",
                "The band visualization is a sampled preview transform, not a model preprocessing pipeline.",
                "CRS presence does not by itself prove accurate georeferencing or content registration.",
            ],
        }


def inspect_standard_image(path: str, filename: str, detected_fmt: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
            bands = image.getbands()
            pixels = width * height
            if width < 2 or height < 2:
                checks.append(_check("dimensions", "REJECT", "Image must be at least 2×2 pixels."))
                return {
                    "filename": filename, "detected_format": detected_fmt, "status": "REJECT",
                    "dimensions": {"width": width, "height": height, "pixels": pixels},
                    "channels": len(bands), "mode": mode, "metadata": {"crs": None, "bounds": None},
                    "valid_pixel_sample": None, "visual_quality": {"status": "NOT_VERIFIED"},
                    "checks": checks, "warnings": [], "limitations": ["Image was too small for useful inspection."],
                }
            if pixels > MAX_IMAGE_PIXELS:
                checks.append(_check("pixel_limit", "REJECT", f"Image contains {pixels:,} pixels, exceeding the configured limit of {MAX_IMAGE_PIXELS:,}.", {"pixels": pixels, "max_pixels": MAX_IMAGE_PIXELS}))
                return {
                    "filename": filename, "detected_format": detected_fmt, "status": "REJECT",
                    "dimensions": {"width": width, "height": height, "pixels": pixels},
                    "channels": len(bands), "mode": mode, "metadata": {"crs": None, "bounds": None},
                    "valid_pixel_sample": None, "visual_quality": {"status": "NOT_VERIFIED"},
                    "checks": checks, "warnings": [checks[-1]["message"]],
                    "limitations": ["Pixel limit was enforced before full raster decoding/visual analysis."],
                }
            checks.append(_check("dimensions", "PASS", "Image dimensions are valid.", {"width": width, "height": height, "pixels": pixels}))
            checks.append(_check("pixel_limit", "PASS", "Image is within the configured pixel limit."))

            if "A" in bands or "transparency" in image.info:
                rgba = image.convert("RGBA")
                sample_h, sample_w = _sample_size(width, height)
                rgba.thumbnail((sample_w, sample_h))
                array = np.asarray(rgba, dtype=np.uint8)
                visual = _visual_statistics(array[..., :3], alpha=array[..., 3])
                transparent_fraction = float(1.0 - np.mean(array[..., 3] > 0))
            else:
                rgb_image = image.convert("RGB")
                rgb_image.thumbnail((SAMPLE_MAX_DIM, SAMPLE_MAX_DIM))
                array = np.asarray(rgb_image, dtype=np.uint8)
                visual = _visual_statistics(array)
                transparent_fraction = None

            checks.append(_check("file_decode", "PASS", "The image was decoded and verified successfully.", {"format": image.format, "mode": mode}))
            checks.append(_check("georeferencing", "NOT_VERIFIED", "Standard JPG/PNG files do not provide verified geographic coordinates through this inspection path."))
            if detected_fmt == "jpeg" and image.format != "JPEG":
                checks.append(_check("format_consistency", "WARNING", "The file signature and decoded image format disagree."))
            if detected_fmt == "png" and image.format != "PNG":
                checks.append(_check("format_consistency", "WARNING", "The file signature and decoded image format disagree."))
            if visual.get("status") == "REJECT":
                checks.append(_check("visual_quality", "REJECT", "No visible finite pixels were available for visual inspection; this file cannot be compared reliably."))
            elif visual.get("status") == "WARNING":
                checks.append(_check("visual_quality", "WARNING", "One or more heuristic visual-quality indicators warrant review; these do not prove the image is unusable."))
            else:
                checks.append(_check("visual_quality", "PASS", "Basic visual statistics were computed; these do not establish satellite quality."))

            return {
                "filename": filename,
                "detected_format": "jpeg" if image.format == "JPEG" else "png" if image.format == "PNG" else detected_fmt,
                "status": _status(checks, base_warning=True),
                "dimensions": {"width": width, "height": height, "pixels": pixels},
                "channels": len(bands),
                "mode": mode,
                "metadata": {
                    "crs": None,
                    "bounds": None,
                    "transform": None,
                    "resolution": None,
                    "exif_available": bool(image.getexif()),
                    "has_icc_profile": bool(image.info.get("icc_profile")),
                    "transparency_present": transparent_fraction is not None,
                    "transparent_fraction_sample": transparent_fraction,
                },
                "valid_pixel_sample": None,
                "visual_quality": visual,
                "checks": checks,
                "warnings": [c["message"] for c in checks if c["status"] in {"WARNING", "NOT_VERIFIED"}],
                "limitations": [
                    "No geographic CRS, affine transform, footprint or ground sampling distance was verified.",
                    "A user's confirmation that two images show the same place is not geospatial verification.",
                    "Cloud/shadow and satellite-specific quality masks were not available in this standard-image inspection.",
                ],
            }
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(f"The uploaded image could not be decoded safely: {exc.__class__.__name__}.") from exc


def inspect_image_file(path: str, filename: str) -> dict[str, Any]:
    if not os.path.getsize(path):
        raise ValueError("The uploaded file is empty.")
    with open(path, "rb") as f:
        header = f.read(1024)
    detected = detect_format(header)
    if detected is None:
        raise ValueError("Unsupported or unrecognized file content. Supported formats are JPEG, PNG and GeoTIFF.")
    try:
        if detected == "tiff":
            result = inspect_raster(path, filename)
        else:
            result = inspect_standard_image(path, filename, detected)
    except (rasterio.errors.RasterioError, OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Unsupported"):
            raise
        raise ValueError(f"The uploaded {detected.upper()} file could not be inspected safely: {exc.__class__.__name__}.") from exc

    ext = os.path.splitext(filename or "")[1].lower()
    expected = {"jpeg": {".jpg", ".jpeg"}, "png": {".png"}, "tiff": {".tif", ".tiff"}}[detected]
    if ext and ext not in expected:
        warning = f"Filename extension '{ext}' does not match detected content type '{detected}'. The file was classified by content."
        result["warnings"].append(warning)
        result["checks"].append(_check("filename_extension", "WARNING", warning))
        result["status"] = _status(result["checks"], base_warning=result["detected_format"] in {"jpeg", "png"})
    result["file_size_bytes"] = os.path.getsize(path)
    return result


def _bounds_area(bounds: dict[str, Any]) -> float:
    return max(0.0, (bounds["right"] - bounds["left"])) * max(0.0, (bounds["top"] - bounds["bottom"]))


def _intersection_bounds(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float] | None:
    left = max(a["left"], b["left"])
    bottom = max(a["bottom"], b["bottom"])
    right = min(a["right"], b["right"])
    top = min(a["top"], b["top"])
    if right <= left or top <= bottom:
        return None
    return {"left": left, "bottom": bottom, "right": right, "top": top}


def compare_pair(
    before_path: str,
    before_name: str,
    after_path: str,
    after_name: str,
    *,
    analysis_mode: str = "urban",
    before_date: str | None = None,
    after_date: str | None = None,
    location_name: str | None = None,
    same_area_confirmed: bool = False,
) -> dict[str, Any]:
    before = inspect_image_file(before_path, before_name)
    after = inspect_image_file(after_path, after_name)
    pair_checks: list[dict[str, Any]] = []
    warnings_out: list[str] = []

    before_meta = before.get("metadata") or {}
    after_meta = after.get("metadata") or {}
    before_crs, after_crs = before_meta.get("crs"), after_meta.get("crs")
    before_bounds, after_bounds = before_meta.get("bounds"), after_meta.get("bounds")
    before_dims, after_dims = before.get("dimensions") or {}, after.get("dimensions") or {}

    geographic = {
        "status": "NOT_VERIFIED",
        "footprint_intersection_over_union": None,
        "before_footprint_coverage_by_after": None,
        "after_footprint_coverage_by_before": None,
        "intersection_bounds": None,
        "aoi_coverage_before": None,
        "aoi_coverage_after": None,
        "method": "Axis-aligned bounds only when both rasters have a CRS expressed in the same CRS; no AOI polygon was supplied. For rotated rasters this uses the axis-aligned envelope, not the exact footprint polygon.",
    }

    both_crs_bounds = bool(before_crs and after_crs and before_bounds and after_bounds)
    if both_crs_bounds and before_crs == after_crs:
        a_area, b_area = _bounds_area(before_bounds), _bounds_area(after_bounds)
        inter = _intersection_bounds(before_bounds, after_bounds)
        if inter and a_area > 0 and b_area > 0:
            inter_area = _bounds_area(inter)
            union = a_area + b_area - inter_area
            geographic.update({
                "status": "PASS" if union > 0 else "NOT_VERIFIED",
                "footprint_intersection_over_union": inter_area / union if union > 0 else None,
                "before_footprint_coverage_by_after": inter_area / a_area if a_area > 0 else None,
                "after_footprint_coverage_by_before": inter_area / b_area if b_area > 0 else None,
                "intersection_bounds": inter,
            })
            pair_checks.append(_check("geographic_footprint_overlap", "PASS", "The two CRS-tagged raster bounds intersect. This is bounds-based overlap, not proof of content registration.", geographic))
        else:
            geographic["status"] = "REJECT"
            pair_checks.append(_check("geographic_footprint_overlap", "REJECT", "The two CRS-tagged raster footprints have no positive-area intersection."))
    elif both_crs_bounds:
        geographic["status"] = "NOT_VERIFIED"
        pair_checks.append(_check("geographic_footprint_overlap", "NOT_VERIFIED", "The rasters use different CRS values; overlap was not computed without an explicit reprojection policy.", {"before_crs": before_crs, "after_crs": after_crs}))
    else:
        message = "Geographic footprint overlap could not be verified because one or both images lack usable CRS/bounds metadata."
        if same_area_confirmed:
            message += " The user's same-area confirmation is retained as user-provided information, not technical verification."
        geographic["status"] = "NOT_VERIFIED"
        pair_checks.append(_check("geographic_footprint_overlap", "NOT_VERIFIED", message))

    same_crs = bool(before_crs and after_crs and before_crs == after_crs)
    pair_checks.append(_check(
        "crs_compatibility",
        "PASS" if same_crs else "WARNING" if before_crs and after_crs else "NOT_VERIFIED",
        "CRS values match." if same_crs else "CRS values differ; an explicit reprojection policy is required." if before_crs and after_crs else "CRS compatibility cannot be verified for one or both images.",
        {"before_crs": before_crs, "after_crs": after_crs},
    ))

    same_dims = before_dims.get("width") == after_dims.get("width") and before_dims.get("height") == after_dims.get("height")
    pair_checks.append(_check("dimensions_compatibility", "PASS" if same_dims else "WARNING", "Image dimensions match." if same_dims else "Image dimensions differ; an explicit resampling/registration policy would be needed for pixelwise comparison.", {"before": before_dims, "after": after_dims}))

    before_transform, after_transform = before_meta.get("transform"), after_meta.get("transform")
    before_geo = before.get("detected_format") == "tiff" and bool(before_meta.get("is_georeferenced"))
    after_geo = after.get("detected_format") == "tiff" and bool(after_meta.get("is_georeferenced"))
    same_transform = bool(before_geo and after_geo and before_transform and after_transform and len(before_transform) == len(after_transform) and all(abs(float(x) - float(y)) < 1e-9 for x, y in zip(before_transform, after_transform)))
    same_grid = bool(same_crs and same_dims and same_transform)
    pair_checks.append(_check("grid_compatibility", "PASS" if same_grid else "WARNING" if before_geo and after_geo else "NOT_VERIFIED", "CRS, dimensions and affine transform match." if same_grid else "Raster grids are not verified as identical; this service does not reproject or warp inputs automatically." if before_geo and after_geo else "A common geographic grid cannot be verified because georeferenced raster metadata is incomplete.", {"same_crs": same_crs, "same_dimensions": same_dims, "same_affine_transform": same_transform}))

    same_band_count = before.get("channels") == after.get("channels")
    pair_checks.append(_check("band_count_compatibility", "PASS" if same_band_count else "WARNING", "Band/channel counts match, but semantic equivalence of bands is not established." if same_band_count else "Band/channel counts differ; the selected model's input contract must be checked before prediction.", {"before_channels": before.get("channels"), "after_channels": after.get("channels")}))

    if same_grid and before.get("detected_format") == "tiff" and after.get("detected_format") == "tiff":
        # Re-open and calculate the sampled common valid mask without materializing full rasters.
        with rasterio.open(before_path) as bds, rasterio.open(after_path) as ads:
            sh, sw = _sample_size(bds.width, bds.height)
            bmask = bds.dataset_mask(out_shape=(sh, sw)) > 0
            amask = ads.dataset_mask(out_shape=(sh, sw)) > 0
            bvals = bds.read(out_shape=(bds.count, sh, sw), masked=False)
            avals = ads.read(out_shape=(ads.count, sh, sw), masked=False)
            bfinite = np.isfinite(np.asarray(bvals, dtype=np.float64)).all(axis=0)
            afinite = np.isfinite(np.asarray(avals, dtype=np.float64)).all(axis=0)
            common = bmask & amask & bfinite & afinite
            common_fraction = float(np.mean(common)) if common.size else None
        pair_checks.append(_check("common_valid_pixel_coverage", "PASS" if common_fraction is not None and common_fraction > 0 else "REJECT", "Estimated common valid-pixel fraction on a downsampled mask for identical grids.", {"fraction": common_fraction, "sampled_pixels": int(common.size)}))
    else:
        common_fraction = None
        pair_checks.append(_check("common_valid_pixel_coverage", "NOT_VERIFIED", "Common pixel coverage was not computed because identical georeferenced grids were not verified."))

    pair_checks.append(_check("content_registration", "NOT_VERIFIED", "No content-registration estimator or automatic image warp was run. Equal metadata/grid does not prove that ground features align."))
    pair_checks.append(_check("radiometric_compatibility", "NOT_VERIFIED", "Radiometric/sensor compatibility requires product-specific band, calibration and acquisition metadata; it is not inferred from file format alone."))

    if before_date and after_date:
        try:
            from datetime import date
            bdate = date.fromisoformat(before_date)
            adate = date.fromisoformat(after_date)
            if adate <= bdate:
                pair_checks.append(_check("date_order", "REJECT", "Chronology is invalid: the after-image date must be later than the before-image date.", {"before_date": before_date, "after_date": after_date}))
            else:
                pair_checks.append(_check("date_order", "PASS", "Provided date ordering is valid.", {"before_date": before_date, "after_date": after_date}))
        except ValueError:
            pair_checks.append(_check("date_order", "REJECT", "Supplied acquisition dates must be valid ISO dates (YYYY-MM-DD).", {"before_date": before_date, "after_date": after_date}))
    else:
        pair_checks.append(_check("date_order", "NOT_VERIFIED", "One or both acquisition dates were not supplied. Dates are not inferred from image appearance."))

    if before["status"] == "REJECT" or after["status"] == "REJECT":
        pair_checks.append(_check("individual_image_gate", "REJECT", "At least one image failed individual inspection."))
    elif before["status"] == "WARNING" or after["status"] == "WARNING":
        pair_checks.append(_check("individual_image_gate", "WARNING", "At least one individual image has warnings or unverifiable metadata."))
    else:
        pair_checks.append(_check("individual_image_gate", "PASS", "Both individual image inspections passed."))

    if geographic["status"] == "REJECT" or before["status"] == "REJECT" or after["status"] == "REJECT" or any(c["status"] == "REJECT" for c in pair_checks):
        status = "REJECT"
    elif any(c["status"] in {"WARNING", "NOT_VERIFIED"} for c in pair_checks):
        status = "WARNING"
    else:
        status = "PASS"

    warnings_out = [c["message"] for c in pair_checks if c["status"] in {"WARNING", "NOT_VERIFIED"}]
    blockers = [c["message"] for c in pair_checks if c["status"] == "REJECT"]
    # The backend in this phase validates inputs only. It is not a prediction endpoint.
    return {
        "analysis_mode": analysis_mode if analysis_mode in {"urban", "disaster"} else "urban",
        "status": status,
        "before_image": before,
        "after_image": after,
        "pair": {
            "geographic_overlap": geographic,
            "same_crs": same_crs,
            "same_dimensions": same_dims,
            "same_grid": same_grid,
            "same_band_count": same_band_count,
            "common_valid_pixel_coverage_fraction": common_fraction,
            "registration_status": "NOT_VERIFIED",
            "radiometric_compatibility_status": "NOT_VERIFIED",
            "checks": pair_checks,
            "warnings": warnings_out,
            "blockers": blockers,
        },
        "user_metadata": {
            "location_name": location_name or None,
            "before_date": before_date or None,
            "after_date": after_date or None,
            "same_area_confirmed_by_user": bool(same_area_confirmed),
        },
        "next_step": {
            "can_continue_to_review": status != "REJECT",
            "prediction_status": "NOT_CONFIGURED",
            "message": "This endpoint performs image inspection and pair validation only. No trained prediction model is connected.",
        },
        "limitations": [
            "A bounding-box intersection is not proof that image contents are correctly registered.",
            "AOI coverage is not computed because this endpoint does not accept a geospatial AOI polygon.",
            "Image-space compatibility is not the same as compatibility with a trained model input contract.",
            "No images are persisted by this request; this phase performs inspection only.",
        ],
    }
