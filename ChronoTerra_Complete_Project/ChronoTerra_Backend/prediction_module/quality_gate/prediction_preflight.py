"""Sensor-aware preflight for ChronoTerra learned inference.

This module reports evidence and hard compatibility failures before inference. It does not
claim that any single statistic proves correct scene identity, co-registration, or cloud-free
conditions. Inference can continue with explicit warnings when the model's hard input contract
is satisfied; reports preserve every unverified factor.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.warp import transform_bounds, reproject
except Exception:  # pragma: no cover - optional dependency on non-geospatial installs
    rasterio = None
    transform_bounds = None
    reproject = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from quality_gate.pair_preflight import inspect_image, _footprint_overlap_fraction
from quality_gate.aoi_coverage import aoi_mask_and_report
from quality_gate.registration_diagnostic import estimate_translation

OSCD_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
# Exclude NoData, defective, cloud-shadow, unclassified, cloud and snow/ice classes.
# SCL class 2 is "dark area pixels"; it can include legitimate dark surfaces/water, so
# it is reported as advisory rather than unconditionally discarded.
SCL_INVALID_CLASSES = {0, 1, 2, 3, 7, 8, 9, 10, 11}
SCL_ADVISORY_CLASSES = set()
TASK_CONTRACTS = {
    "urban_change": {"format": "geotiff_pair", "channels_each": 13, "model_input": "Sentinel-2 OSCD order B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12"},
    "flood_extent": {"format": "geotiff_single", "channels_each": 2, "model_input": "Sentinel-1 SAR VV/VH in the checkpoint's documented preprocessing scale"},
    "building_damage": {"format": "rgb_pair", "channels_each": 3, "model_input": "paired pre/post RGB chips aligned to the xBD training convention"},
}


def _normal_band_name(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def _find_date(tags: dict[str, Any]) -> str | None:
    candidates = []
    for k, v in tags.items():
        kl = str(k).lower()
        if any(t in kl for t in ("datetime", "acquisition_date", "date_acquired", "acq_date", "timestamp", "acquired")):
            candidates.append(str(v).strip())
    for raw in candidates:
        raw = raw.replace("Z", "+00:00")
        try:
            d = datetime.fromisoformat(raw)
            return d.date().isoformat()
        except Exception:
            m = re.search(r"(20\d{2})[-:]?(\d{2})[-:]?(\d{2})", raw)
            if m:
                try:
                    return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date().isoformat()
                except ValueError:
                    pass
    return None



def _find_date_in_metadata_json(path: str | Path | None, which: str) -> tuple[str | None, dict[str, Any]]:
    """Read acquisition date/provenance hints from pair_metadata.json when available."""
    if not path:
        return None, {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pair metadata JSON not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    obj = data.get(which, {}) if isinstance(data, dict) else {}
    if not isinstance(obj, dict):
        return None, {}
    dt = _find_date(obj)
    metadata = obj.get("metadata", {})
    if not dt and isinstance(metadata, dict):
        dt = _find_date(metadata)
    props = metadata.get("properties", {}) if isinstance(metadata, dict) else {}
    if not dt and isinstance(props, dict):
        dt = _find_date(props)
    origin = data.get("source", "unspecified") if isinstance(data, dict) else "unspecified"
    return dt, {"source_object_keys": sorted(map(str, obj.keys())), "metadata_origin": origin, "date_is_user_supplied": str(origin).startswith("user_supplied")}

def _stats_2d(a: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float | None]:
    x = np.asarray(a, dtype=np.float32)
    m = np.isfinite(x)
    if valid is not None and valid.shape == x.shape:
        m &= valid
    vals = x[m]
    if vals.size == 0:
        return {"p02": None, "p50": None, "p98": None, "robust_range": None, "std": None}
    p02, p50, p98 = np.percentile(vals, [2, 50, 98])
    return {"p02": float(p02), "p50": float(p50), "p98": float(p98),
            "robust_range": float(p98-p02), "std": float(np.std(vals))}


def inspect_asset(path: str | Path, sample_dim: int = 512) -> dict[str, Any]:
    """Bounded-memory metadata and sampled radiometry inspection for TIFF or image inputs."""
    p = Path(path)
    base = inspect_image(p)
    if base.get("status") == "REJECT":
        return {"asset": base, "band_statistics": [], "acquisition_date": None}
    detail: dict[str, Any] = {"asset": base, "band_statistics": [], "acquisition_date": None,
                             "sensor_hint": None, "band_order": {"status": "NOT_VERIFIED", "names": []},
                             "quality_indicators": {"cloud_shadow": "NOT_VERIFIED", "sharpness": "NOT_VERIFIED"}}
    if p.suffix.lower() in {".tif", ".tiff", ".vrt"}:
        if rasterio is None:
            detail["asset"]["status"] = "NOT_VERIFIED"
            detail["asset"]["reason"] = "Rasterio unavailable"
            return detail
        with rasterio.open(p) as ds:
            h, w = ds.height, ds.width
            oh, ow = min(sample_dim, h), min(sample_dim, w)
            arr = ds.read(out_shape=(ds.count, oh, ow), out_dtype="float32", resampling=rasterio.enums.Resampling.bilinear)
            masks = ds.dataset_mask(out_shape=(oh, ow), resampling=rasterio.enums.Resampling.nearest) > 0
            valid = masks & np.all(np.isfinite(arr), axis=0)
            if ds.nodata is not None:
                valid &= ~np.any(arr == ds.nodata, axis=0)
            descriptions = [(x or "").strip() for x in ds.descriptions]
            tags = dict(ds.tags())
            for _, values in ds.tags(ns="IMAGE_STRUCTURE").items():
                tags.setdefault(_, values)
            band_names = []
            for i, desc in enumerate(descriptions):
                # Band descriptions are only accepted as metadata, not guessed from values.
                band_names.append(desc)
            detail.update({
                "driver": ds.driver,
                "width": w, "height": h, "channels": ds.count,
                "crs": ds.crs.to_string() if ds.crs else None,
                "bounds": [float(ds.bounds.left), float(ds.bounds.bottom), float(ds.bounds.right), float(ds.bounds.top)],
                "transform": [float(x) for x in ds.transform],
                "resolution": [float(abs(ds.res[0])), float(abs(ds.res[1]))],
                "dtypes": list(ds.dtypes), "nodata": ds.nodata,
                "valid_fraction_sampled": float(valid.mean()),
                "band_descriptions": descriptions,
                "acquisition_date": _find_date(tags),
                "metadata_tags": {str(k): str(v)[:200] for k, v in tags.items()
                                  if any(t in str(k).lower() for t in ("date", "time", "sensor", "platform", "satellite", "mission", "product", "orbit"))},
            })
            normalized = [_normal_band_name(n) for n in band_names]
            if all(normalized) and all(n in "".join(normalized) for n in OSCD_BANDS) and ds.count == 13:
                # Verify ordering, not just presence.
                positions = [next((i for i, n in enumerate(normalized) if b in n), -1) for b in OSCD_BANDS]
                status = "PASS" if positions == list(range(13)) else "REJECT"
                detail["band_order"] = {"status": status, "names": band_names, "expected": OSCD_BANDS}
            elif any(normalized):
                detail["band_order"] = {"status": "NOT_VERIFIED", "names": band_names, "reason": "Band descriptions are present but don't prove the requested profile."}
            else:
                detail["band_order"] = {"status": "NOT_VERIFIED", "names": band_names, "reason": "No band descriptions; verify order from authoritative source metadata."}
            stats = []
            for idx, band in enumerate(arr):
                stats.append({"band_index_1based": idx+1, "description": descriptions[idx], **_stats_2d(band, valid)})
            detail["band_statistics"] = stats
            if tags.get("SPACECRAFT_NAME") or tags.get("PLATFORM"):
                detail["sensor_hint"] = tags.get("SPACECRAFT_NAME") or tags.get("PLATFORM")
            detail["quality_indicators"]["cloud_shadow"] = "NOT_VERIFIED: use aligned Sentinel-2 SCL/QA mask or validated detector"
            if ds.count >= 3:
                detail["quality_indicators"]["sharpness"] = "NOT_VERIFIED: a display-band sharpness proxy is not a universal accept/reject metric"
    else:
        with Image.open(p) as im:
            im.load()
            arr = np.asarray(im.convert("RGB") if im.mode not in ("L", "RGB") else im)
            detail.update({"width": im.width, "height": im.height, "channels": 1 if arr.ndim == 2 else arr.shape[-1],
                           "crs": None, "bounds": None, "transform": None, "resolution": None,
                           "acquisition_date": None, "band_descriptions": None,
                           "valid_fraction_sampled": float(np.isfinite(arr).all(axis=-1).mean()) if arr.ndim == 3 else float(np.isfinite(arr).mean())})
            detail["quality_indicators"]["cloud_shadow"] = "NOT_VERIFIED: generic RGB images have no authoritative cloud/shadow mask"
            if cv2 is not None and arr.shape[0] >= 16 and arr.shape[1] >= 16:
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
                detail["quality_indicators"]["sharpness"] = {"laplacian_variance": float(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F).var()),
                                                               "interpretation": "advisory only; depends on resolution, content and sensor"}
    return detail


def _raster_details(path: str | Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if rasterio is None:
        raise RuntimeError("Rasterio is required for raster factor analysis")
    with rasterio.open(path) as ds:
        h, w = ds.height, ds.width
        oh, ow = min(512, h), min(512, w)
        arr = ds.read(out_shape=(ds.count, oh, ow), out_dtype="float32", resampling=rasterio.enums.Resampling.bilinear)
        valid = (ds.dataset_mask(out_shape=(oh, ow), resampling=rasterio.enums.Resampling.nearest) > 0) & np.all(np.isfinite(arr), axis=0)
        if ds.nodata is not None:
            valid &= ~np.any(arr == ds.nodata, axis=0)
        stats = [_stats_2d(arr[i], valid) for i in range(ds.count)]
        return ({"width": ds.width, "height": ds.height, "count": ds.count,
                 "crs": ds.crs.to_string() if ds.crs else None,
                 "transform": tuple(ds.transform), "bounds": [float(ds.bounds.left), float(ds.bounds.bottom), float(ds.bounds.right), float(ds.bounds.top)],
                 "resolution": (float(abs(ds.res[0])), float(abs(ds.res[1]))),
                 "descriptions": [(x or "").strip() for x in ds.descriptions],
                 "date": _find_date(ds.tags()), "nodata": ds.nodata,
                 "valid_fraction_sampled": float(valid.mean()), "sample_scale": [oh, ow]}, arr, valid)


def _compare_radiometry(before_path: str | Path, after_path: str | Path) -> dict[str, Any]:
    """Sample-based same-channel distribution comparison, explicitly diagnostic only."""
    b, ba, bv = _raster_details(before_path)
    a, aa, av = _raster_details(after_path)
    if b["count"] != a["count"]:
        return {"status": "REJECT", "reason": "Channel counts differ; spectral statistics are not comparable."}
    if ba.shape != aa.shape or b["transform"] != a["transform"] or b["crs"] != a["crs"]:
        return {"status": "NOT_VERIFIED", "reason": "Common pixelwise radiometric diagnostics require the same raster grid."}
    valid = bv & av
    if not valid.any():
        return {"status": "REJECT", "reason": "No common valid pixels for radiometric diagnostics."}
    rows = []
    for i in range(b["count"]):
        bs, ass = _stats_2d(ba[i], valid), _stats_2d(aa[i], valid)
        if bs["p50"] is None or ass["p50"] is None:
            rows.append({"band_index_1based": i+1, "status": "NOT_VERIFIED"}); continue
        scale = max(float(bs["robust_range"] or 0), float(ass["robust_range"] or 0), 1e-6)
        rows.append({"band_index_1based": i+1,
                     "before_p02_p50_p98": [bs["p02"], bs["p50"], bs["p98"]],
                     "after_p02_p50_p98": [ass["p02"], ass["p50"], ass["p98"]],
                     "median_shift_normalized_by_robust_range": float((ass["p50"]-bs["p50"])/scale),
                     "interpretation": "descriptive only; genuine land change also shifts distributions"})
    return {"status": "DIAGNOSTIC_ONLY", "valid_fraction_in_sample": float(valid.mean()), "per_band": rows,
            "warning": "Do not automatically histogram-match or reject based on this report; real change and seasonal effects can alter histograms."}


def read_scl_on_image_grid(image_path: str | Path, scl_path: str | Path, *, class2_policy: str = "mask") -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read a categorical Sentinel-2 SCL layer on the model image grid.

    If grids differ, require trustworthy CRS/affine metadata and use nearest-neighbour
    reprojection because SCL class IDs are categorical. This operation is only for the QA
    layer; it does not warp either bi-temporal image. No overlap yields all-invalid and fails.
    """
    if rasterio is None or reproject is None:
        raise RuntimeError("Rasterio reproject is required for SCL alignment")
    if class2_policy not in {"mask", "keep"}:
        raise ValueError("class2_policy must be mask or keep")
    from rasterio.enums import Resampling
    with rasterio.open(image_path) as img, rasterio.open(scl_path) as qa:
        if qa.count != 1:
            raise ValueError("SCL/QA layer must be single-band")
        if not img.crs or not qa.crs:
            if (img.width, img.height, img.transform, img.crs) != (qa.width, qa.height, qa.transform, qa.crs):
                raise ValueError("Cannot align SCL layer with mismatched grid because trustworthy CRS metadata is missing")
        same = (img.width == qa.width and img.height == qa.height and img.crs == qa.crs and img.transform == qa.transform)
        raw = qa.read(1)
        raw_valid = (qa.read_masks(1) > 0) & np.isfinite(raw)
        if same:
            cls = raw
            qa_valid = raw_valid
            method = "direct_same_grid"
        else:
            cls = np.zeros((img.height, img.width), dtype=raw.dtype)
            qa_valid_u8 = np.zeros((img.height, img.width), dtype=np.uint8)
            reproject(source=raw, destination=cls, src_transform=qa.transform, src_crs=qa.crs,
                      dst_transform=img.transform, dst_crs=img.crs, src_nodata=qa.nodata,
                      dst_nodata=0, resampling=Resampling.nearest)
            reproject(source=raw_valid.astype(np.uint8), destination=qa_valid_u8,
                      src_transform=qa.transform, src_crs=qa.crs, dst_transform=img.transform,
                      dst_crs=img.crs, src_nodata=0, dst_nodata=0, resampling=Resampling.nearest)
            qa_valid = qa_valid_u8 > 0
            method = "nearest_reproject_categorical_SCL_to_model_grid"
        invalid_classes = set(SCL_INVALID_CLASSES) - ({2} if class2_policy == "keep" else set())
        good = qa_valid & ~np.isin(cls, list(invalid_classes))
        if not good.any():
            raise ValueError("SCL layer has no valid accepted pixels on the image grid; check overlap/NoData")
        report = {"status": "PASS", "method": method, "reprojected": not same,
                  "source_crs": str(qa.crs) if qa.crs else None,
                  "target_crs": str(img.crs) if img.crs else None,
                  "excluded_scl_classes": sorted(invalid_classes),
                  "advisory_scl_classes_kept_valid": [2] if class2_policy == "keep" else [],
                  "class2_policy": class2_policy,
                  "valid_fraction_scl_on_target_grid": float(good.mean()),
                  "excluded_fraction_on_target_grid": float((~good).mean()),
                  "warning": ("SCL class 2 is retained only by explicit user policy; verify legacy processing-baseline semantics before using this option. SCL misses some thin clouds/shadows and is not atmospheric correction." if class2_policy == "keep" else "Modern Sentinel-2 L2A PB04+ SCL class 2 is CAST_SHADOW and is masked. For legacy products, verify processing-baseline semantics; SCL remains imperfect and is not atmospheric correction.")}
        return cls, good, report


def _apply_scl(path: str | Path | None, image_path: str | Path, valid: np.ndarray | None = None, *, class2_policy: str = "mask") -> dict[str, Any]:
    if not path:
        return {"status": "NOT_VERIFIED", "reason": "No SCL quality layer was supplied; cloud/shadow pixels have not been explicitly masked."}
    try:
        _cls, good, report = read_scl_on_image_grid(image_path, path, class2_policy=class2_policy)
        if valid is not None and good.shape != valid.shape:
            return {"status": "REJECT", "reason": "Aligned SCL mask shape does not match model validity mask"}
        return report
    except Exception as exc:
        return {"status": "REJECT", "reason": f"SCL alignment/masking failed: {type(exc).__name__}: {exc}"}


def run_prediction_preflight(task: str, *, before_path: str | Path | None = None,
                             after_path: str | Path | None = None, input_path: str | Path | None = None,
                             before_scl: str | Path | None = None, after_scl: str | Path | None = None,
                             pair_metadata_json: str | Path | None = None,
                             assume_band_order: bool = False,
                             aoi_geojson: str | Path | None = None,
                             scl_class2_policy: str = "mask") -> dict[str, Any]:
    """Return model-specific hard gates plus factor diagnostics; never performs inference."""
    if task not in TASK_CONTRACTS:
        raise ValueError(f"Unsupported task {task!r}. Supported tasks: {sorted(TASK_CONTRACTS)}")
    if scl_class2_policy not in {"mask", "keep"}:
        raise ValueError("scl_class2_policy must be mask or keep")
    contract = TASK_CONTRACTS[task]
    report: dict[str, Any] = {"task": task, "contract": contract, "status": "PASS", "hard_errors": [], "warnings": [], "checks": {},
                              "factors": {"spectral_compatibility": "NOT_VERIFIED", "spatial_context": "model architecture feature, requires real-data validation",
                                          "radiometric_differences": "NOT_VERIFIED", "cloud_shadow": "NOT_VERIFIED",
                                          "seasonality": "NOT_VERIFIED", "registration": "NOT_VERIFIED",
                                          "invalid_pixels": "NOT_VERIFIED"}}
    assets: dict[str, Any] = {}
    if task == "flood_extent":
        if not input_path:
            report["hard_errors"].append("flood_extent requires --input with a 2-band Sentinel-1 SAR GeoTIFF")
        else:
            input_info = inspect_asset(input_path); assets["input"] = input_info
            asset = input_info.get("asset", {})
            if Path(input_path).suffix.lower() not in {".tif", ".tiff", ".vrt"}:
                report["hard_errors"].append("Flood extent requires georeferenced Sentinel-1 VV/VH GeoTIFF; RGB JPG/PNG is incompatible.")
            if asset.get("channels") != 2:
                report["hard_errors"].append(f"Flood model expects exactly two SAR bands, received {asset.get('channels')}.")
            if not asset.get("crs"):
                report["hard_errors"].append("Flood model requires a valid CRS.")
            names = [(_normal_band_name(x)) for x in input_info.get("band_order", {}).get("names", [])]
            nonempty = [x for x in names if x]
            if len(nonempty) == 2:
                first = "VV" if nonempty[0].startswith("VV") else "VH" if nonempty[0].startswith("VH") else "UNKNOWN"
                second = "VV" if nonempty[1].startswith("VV") else "VH" if nonempty[1].startswith("VH") else "UNKNOWN"
                if [first, second] != ["VV", "VH"]:
                    report["hard_errors"].append("Declared band descriptions conflict with required VV then VH order; --assume-band-order cannot override contradictory metadata.")
            elif not assume_band_order:
                report["hard_errors"].append("VV/VH order is not encoded in TIFF descriptions. Verify it from authoritative source metadata, then use --assume-band-order explicitly.")
            else:
                report["warnings"].append("VV then VH order was explicitly assumed; retain the source metadata/provenance used to verify it.")
            # Sen1Floods11 and the current model contract use SAR backscatter in dB.
            # Quantiles are diagnostic; a median outside this range is a strong scale mismatch.
            bstats = input_info.get("band_statistics", [])
            medians = [st.get("p50") for st in bstats if st.get("p50") is not None]
            if len(medians) == 2 and any(float(v) < -35.0 or float(v) > 5.0 for v in medians):
                report["hard_errors"].append("SAR band median is outside expected dB input range [-35, 5]; inspect source units/scaling instead of clipping incompatible data.")
            elif len(medians) == 2:
                p02s = [st.get("p02") for st in bstats if st.get("p02") is not None]
                p98s = [st.get("p98") for st in bstats if st.get("p98") is not None]
                if any(float(v) < -35.0 for v in p02s) or any(float(v) > 5.0 for v in p98s):
                    report["warnings"].append("Some SAR values fall outside the model's [-35, 5] dB normalization range and will be clipped; verify the input units.")
            report["checks"]["input"] = {"status": "REJECT" if report["hard_errors"] else asset.get("status", "NOT_VERIFIED"),
                                                  "crs": asset.get("crs"), "channels": asset.get("channels"),
                                                  "valid_fraction_sampled": input_info.get("valid_fraction_sampled", asset.get("valid_fraction"))}
            report["factors"]["invalid_pixels"] = {"valid_fraction_sampled": input_info.get("valid_fraction_sampled"), "method": "Rasterio masks + finite values"}
            report["factors"]["spectral_compatibility"] = {"input_channels": asset.get("channels"), "expected": "Sentinel-1 VV,VH", "band_order": input_info.get("band_order"), "per_band_quantiles": bstats, "expected_value_scale": "dB [-35,5] normalization range"}
            report["factors"]["cloud_shadow"] = "Not relevant as an optical SCL factor; SAR still has layover/shadow and speckle effects not fully diagnosed here."
    else:
        if not before_path or not after_path:
            report["hard_errors"].append(f"{task} requires both before and after files")
        else:
            bi, ai = inspect_asset(before_path), inspect_asset(after_path)
            assets["before"], assets["after"] = bi, ai
            b, a = bi.get("asset", {}), ai.get("asset", {})
            geo_b = Path(before_path).suffix.lower() in {".tif", ".tiff", ".vrt"}
            geo_a = Path(after_path).suffix.lower() in {".tif", ".tiff", ".vrt"}
            expected = contract["channels_each"]
            if int(b.get("channels", -1)) != expected or int(a.get("channels", -1)) != expected:
                report["hard_errors"].append(f"{task} requires {expected} channels per image; before={b.get('channels')}, after={a.get('channels')}.")
            if b.get("width") != a.get("width") or b.get("height") != a.get("height"):
                report["hard_errors"].append("Before/after dimensions differ. Reconcile the AOI and pixel grid before this model can infer.")
            if task == "urban_change":
                if not (geo_b and geo_a): report["hard_errors"].append("OSCD-style urban-change model requires two georeferenced 13-band GeoTIFFs.")
                if not b.get("crs") or not a.get("crs"): report["hard_errors"].append("Both urban images must have valid CRS metadata.")
                if b.get("crs") != a.get("crs") or b.get("transform") != a.get("transform"):
                    report["hard_errors"].append("Urban images must already be on the same CRS and affine grid; no blind automatic warping is performed.")
                for label, info in (("before", bi), ("after", ai)):
                    bo = info.get("band_order", {})
                    if bo.get("status") == "REJECT": report["hard_errors"].append(f"{label} band descriptions conflict with the 13-band OSCD order.")
                    elif bo.get("status") != "PASS" and not assume_band_order:
                        report["hard_errors"].append(f"{label} band order is not verified. Verify source metadata and pass --assume-band-order explicitly, or write verified band descriptions.")
                    elif bo.get("status") != "PASS" and assume_band_order:
                        report["warnings"].append(f"{label} band order was explicitly assumed; keep authoritative source metadata for provenance.")
                try:
                    overlap = _footprint_overlap_fraction(b, a) if b.get("crs") and a.get("crs") else None
                    report["checks"]["footprint_overlap_fraction_of_smaller"] = overlap
                    if overlap is not None and overlap < 0.999:
                        report["hard_errors"].append(f"Urban images do not cover the same footprint completely (overlap={overlap:.3f}); align/crop explicitly.")
                except Exception as exc:
                    report["warnings"].append(f"Footprint overlap could not be computed: {exc}")
                for label, scl in (("before", before_scl), ("after", after_scl)):
                    if scl:
                        q = _apply_scl(scl, before_path if label == "before" else after_path, class2_policy=scl_class2_policy)
                        if q.get("status") == "REJECT": report["hard_errors"].append(f"{label} SCL mask invalid: {q['reason']}")
                        else: report["factors"][f"{label}_cloud_shadow_mask"] = q
                    else:
                        report["warnings"].append(f"{label} optical cloud/shadow mask not supplied; invalid cloud/shadow pixels may cause false detections.")
                if not before_scl or not after_scl:
                    report["factors"]["cloud_shadow"] = "NOT_VERIFIED: supply matching Sentinel-2 SCL masks when available"
                else:
                    report["factors"]["cloud_shadow"] = "SCL masks supplied and grid-checked; SCL is not a perfect cloud/shadow detector"
                if geo_b and geo_a and b.get("crs") == a.get("crs") and b.get("transform") == a.get("transform"):
                    try:
                        rad = _compare_radiometry(before_path, after_path)
                        report["factors"]["radiometric_differences"] = rad
                    except Exception as exc:
                        report["factors"]["radiometric_differences"] = {"status": "NOT_VERIFIED", "reason": str(exc)}
                else:
                    report["factors"]["radiometric_differences"] = "NOT_VERIFIED: grids differ"
                dates = [bi.get("acquisition_date"), ai.get("acquisition_date")]
                metadata_provenance = {}
                if pair_metadata_json:
                    try:
                        db, meta_b = _find_date_in_metadata_json(pair_metadata_json, "before")
                        da, meta_a = _find_date_in_metadata_json(pair_metadata_json, "after")
                        dates = [dates[0] or db, dates[1] or da]
                        metadata_provenance = {"pair_metadata_json": str(pair_metadata_json), "before_metadata": meta_b, "after_metadata": meta_a}
                    except Exception as exc:
                        report["warnings"].append(f"Pair metadata JSON could not be used for acquisition dates: {exc}")
                if metadata_provenance:
                    report["factors"]["source_metadata_provenance"] = metadata_provenance
                    if "user_supplied_form_metadata_not_sensor_verified" in str(metadata_provenance):
                        report["warnings"].append("Dates/location were entered by the user and are not verified sensor metadata.")
                if all(dates):
                    try:
                        d0, d1 = [datetime.fromisoformat(str(d).replace("Z", "+00:00")).date() for d in dates]
                        if d1 <= d0:
                            report["hard_errors"].append("Acquisition chronology is invalid: the after-image date must be later than the before-image date.")
                            report["factors"]["chronology"] = {"status": "REJECT", "before_date": dates[0], "after_date": dates[1],
                                                                 "rule": "after_date must be strictly later than before_date"}
                            report["factors"]["seasonality"] = {"status": "NOT_CALCULATED_INVALID_CHRONOLOGY", "before_date": dates[0], "after_date": dates[1]}
                        else:
                            gap = (d1-d0).days
                            doy = abs(d1.timetuple().tm_yday-d0.timetuple().tm_yday)
                            seasonal_doy_gap = min(doy, 366-doy)
                            season = "WARNING" if seasonal_doy_gap > 45 else "PASS"
                            report["factors"]["chronology"] = {"status": "PASS", "before_date": dates[0], "after_date": dates[1], "days_between": gap}
                            report["factors"]["seasonality"] = {"status": season, "before_date": dates[0], "after_date": dates[1],
                                                                  "days_between": gap, "circular_day_of_year_gap": seasonal_doy_gap,
                                                                  "rule": "45-day day-of-year gap is a review heuristic, not a universal validity threshold"}
                            if season == "WARNING": report["warnings"].append("Acquisition day-of-year differs substantially; seasonal variation can increase false detections.")
                    except Exception:
                        report["factors"]["seasonality"] = "NOT_VERIFIED: acquisition metadata not parseable"
                else:
                    report["factors"]["seasonality"] = "NOT_VERIFIED: acquisition date missing from embedded metadata; supply source metadata for stronger QA"
                    report["warnings"].append("Acquisition dates/seasons could not be verified from embedded image metadata.")
                try:
                    reg = estimate_translation(before_path, after_path, max_dim=512)
                    report["factors"]["registration"] = reg
                    shift = reg.get("shift_native_pixels") or {}
                    response = reg.get("response")
                    if shift and response is not None and (abs(shift.get("x", 0)) > 0.5 or abs(shift.get("y", 0)) > 0.5):
                        report["warnings"].append("Registration diagnostic estimates >0.5 native-pixel translation. Review stable features; no automatic warp was applied.")
                    if response is not None and response < 0.05:
                        report["warnings"].append("Registration response is weak; content alignment remains unverified.")
                except Exception as exc:
                    report["factors"]["registration"] = {"status": "NOT_VERIFIED", "reason": str(exc)}
                    report["warnings"].append("Content registration could not be estimated reliably.")
                before_valid_sampled = float(bi.get("valid_fraction_sampled", b.get("valid_fraction", 0.0)))
                after_valid_sampled = float(ai.get("valid_fraction_sampled", a.get("valid_fraction", 0.0)))
                common_upper_bound = min(before_valid_sampled, after_valid_sampled)
                common_valid_fraction_sampled = None
                if geo_b and geo_a and b.get("crs") == a.get("crs") and b.get("transform") == a.get("transform") and (b.get("width"), b.get("height")) == (a.get("width"), a.get("height")):
                    try:
                        _bd, _barr, bvalid = _raster_details(before_path)
                        _ad, _aarr, avalid = _raster_details(after_path)
                        if bvalid.shape == avalid.shape:
                            common_valid_fraction_sampled = float((bvalid & avalid).mean())
                    except Exception as exc:
                        report["warnings"].append(f"Common valid coverage diagnostic failed: {exc}")
                report["factors"]["invalid_pixels"] = {"before_valid_fraction_sampled": before_valid_sampled,
                                                          "after_valid_fraction_sampled": after_valid_sampled,
                                                          "common_valid_fraction_upper_bound": common_upper_bound,
                                                          "common_valid_fraction_sampled": common_valid_fraction_sampled,
                                                          "note": "Coverage is sampled for bounded preflight; inference applies exact per-pixel masks on the input grid."}
            elif task == "building_damage":
                if geo_b != geo_a:
                    report["hard_errors"].append("Do not mix georeferenced and image-space inputs in one pair.")
                if geo_b and geo_a and (b.get("crs") != a.get("crs") or b.get("transform") != a.get("transform")):
                    report["hard_errors"].append("Georeferenced damage pair must share CRS/transform; align explicitly first.")
                # xBD-style model was trained for byte-range RGB, not 16-bit optical reflectance.
                if bi.get("band_statistics") or ai.get("band_statistics"):
                    for label, info in (("before", bi), ("after", ai)):
                        stats = info.get("band_statistics", [])
                        p98 = [st.get("p98") for st in stats if st.get("p98") is not None]
                        if p98 and max(float(v) for v in p98) > 255.0:
                            report["hard_errors"].append(f"{label} image values exceed byte RGB range 0..255; the xBD RGB model does not accept unscaled multispectral/reflectance values.")
                        elif p98 and max(float(v) for v in p98) <= 1.5:
                            report["hard_errors"].append(f"{label} appears already normalized to approximately 0..1; the xBD model expects byte RGB values in 0..255.")
                report["factors"]["spatial_overlap"] = "NOT_VERIFIED" if not (geo_b and geo_a) else "Footprint metadata checked"
                try:
                    reg = estimate_translation(before_path, after_path, max_dim=512)
                    report["factors"]["registration"] = reg
                    shift = reg.get("shift_native_pixels") or {}
                    response = reg.get("response")
                    if response is None or response < 0.05 or abs(shift.get("x", 0)) > 0.5 or abs(shift.get("y", 0)) > 0.5:
                        report["warnings"].append("RGB alignment diagnostic is weak or estimates a notable shift; review stable features and source pairing.")
                except Exception as exc:
                    report["factors"]["registration"] = {"status": "NOT_VERIFIED", "reason": str(exc)}
                report["warnings"].append("RGB pair location and acquisition season are not guaranteed by pixels/dimensions alone; use a semantically valid pre/post-disaster pair.")
                dates = [bi.get("acquisition_date"), ai.get("acquisition_date")]
                if pair_metadata_json:
                    try:
                        db, meta_b = _find_date_in_metadata_json(pair_metadata_json, "before")
                        da, meta_a = _find_date_in_metadata_json(pair_metadata_json, "after")
                        dates = [dates[0] or db, dates[1] or da]
                        report["factors"]["source_metadata_provenance"] = {"before_metadata": meta_b, "after_metadata": meta_a, "metadata_origin": meta_b.get("metadata_origin")}
                        if meta_b.get("date_is_user_supplied") or meta_a.get("date_is_user_supplied"):
                            report["warnings"].append("Dates/location were entered by the user and are not verified sensor metadata.")
                    except Exception as exc:
                        report["warnings"].append(f"Pair metadata JSON could not be used for dates: {exc}")
                report["factors"]["seasonality"] = {"status": "PASS" if all(dates) else "NOT_VERIFIED", "before_date": dates[0], "after_date": dates[1],
                                                       "note": "Dates alone do not establish seasonal comparability or disaster interval."}
            report["checks"]["before"] = {"width": b.get("width"), "height": b.get("height"), "channels": b.get("channels"), "crs": b.get("crs"), "resolution": b.get("resolution"), "valid_fraction_sampled": bi.get("valid_fraction_sampled", b.get("valid_fraction"))}
            report["checks"]["after"] = {"width": a.get("width"), "height": a.get("height"), "channels": a.get("channels"), "crs": a.get("crs"), "resolution": a.get("resolution"), "valid_fraction_sampled": ai.get("valid_fraction_sampled", a.get("valid_fraction"))}
            if b.get("channels") == a.get("channels") and b.get("channels"):
                report["factors"]["spectral_compatibility"] = {"channel_count_match": True, "before_band_descriptions": bi.get("band_order", {}).get("names"),
                                                                 "after_band_descriptions": ai.get("band_order", {}).get("names"),
                                                                 "expected_profile": contract["model_input"]}
    # Optional user AOI: assess geometry coverage independently from pair overlap and registration.
    if aoi_geojson is not None:
        aoi_targets: list[tuple[str, str | Path | None]] = []
        if task == "flood_extent":
            aoi_targets = [("input", input_path)]
        else:
            aoi_targets = [("before", before_path), ("after", after_path)]
        aoi_reports: dict[str, Any] = {}
        aoi_hard_error = False
        for label, target_path in aoi_targets:
            if not target_path:
                report["hard_errors"].append(f"AOI supplied but {label} raster path is missing")
                aoi_hard_error = True
                continue
            suffix = Path(target_path).suffix.lower()
            if suffix not in {".tif", ".tiff", ".vrt"}:
                report["hard_errors"].append("AOI filtering requires georeferenced raster inputs; metadata-free RGB images cannot be clipped by geographic AOI")
                aoi_hard_error = True
                continue
            try:
                _aoi_mask, aoi_rep = aoi_mask_and_report(aoi_geojson, target_path)
                aoi_reports[label] = aoi_rep
                if aoi_rep.get("status") == "REJECT":
                    report["hard_errors"].append(f"AOI check for {label} rejected: {aoi_rep.get('reason')}")
                    aoi_hard_error = True
                elif aoi_rep.get("status") == "WARNING":
                    report["warnings"].append(f"AOI coverage for {label} is partial; only covered pixels can be assessed")
            except Exception as exc:
                report["hard_errors"].append(f"AOI GeoJSON/raster validation failed for {label}: {type(exc).__name__}: {exc}")
                aoi_hard_error = True
        report["factors"]["aoi_coverage"] = {"status": "REJECT" if aoi_hard_error or any(x.get("status") == "REJECT" for x in aoi_reports.values()) else "WARNING" if any(x.get("status") == "WARNING" for x in aoi_reports.values()) else "PASS" if aoi_reports else "REJECT",
                                               "requested_aoi_geojson": str(Path(aoi_geojson).resolve()),
                                               "per_asset": aoi_reports,
                                               "note": "AOI polygon coverage, rasterized pixel mask and valid data fraction are separate from before/after registration."}

    report["assets"] = assets
    if report["hard_errors"]:
        report["status"] = "REJECT"
    elif report["warnings"]:
        report["status"] = "WARNING"
    report["inference_gate"] = "BLOCK" if report["hard_errors"] else "ALLOW_WITH_WARNINGS" if report["warnings"] else "ALLOW"
    report["must_not_be_interpreted_as"] = ["proof of legal encroachment", "proof of structural safety/damage certification", "a calibrated confidence report", "guaranteed generalization to arbitrary imagery"]
    return report


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=sorted(TASK_CONTRACTS), required=True)
    ap.add_argument("--input"); ap.add_argument("--before"); ap.add_argument("--after")
    ap.add_argument("--before-scl"); ap.add_argument("--after-scl"); ap.add_argument("--pair-metadata-json")
    ap.add_argument("--aoi-geojson", help="Optional WGS84 GeoJSON Polygon/MultiPolygon to restrict analysis")
    ap.add_argument("--scl-class2-policy", choices=["mask", "keep"], default="mask", help="Modern SCL defaults to masking class 2 cast shadows; keep only if legacy SCL semantics have been verified")
    ap.add_argument("--assume-band-order", action="store_true", help="Use only after verifying source metadata proves the intended order")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = run_prediction_preflight(args.task, before_path=args.before, after_path=args.after, input_path=args.input,
                                      before_scl=args.before_scl, after_scl=args.after_scl, pair_metadata_json=args.pair_metadata_json,
                                      assume_band_order=args.assume_band_order, aoi_geojson=args.aoi_geojson,
                                      scl_class2_policy=args.scl_class2_policy)
    dumped = json.dumps(result, indent=2, default=str)
    print(dumped)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(dumped, encoding="utf-8")
    raise SystemExit(2 if result["status"] == "REJECT" else 0)

if __name__ == "__main__":
    main()
