"""AOI geometry validation, footprint coverage and pixel-center masking.

GeoJSON without a legacy ``crs`` member is interpreted per RFC 7946 as WGS 84 longitude/latitude.
The module never warps source images; it transforms the AOI geometry into each existing raster grid.
Coverage is geometric and pixel-grid based, not proof that paired pixels are co-registered.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import rasterio
    from rasterio.features import geometry_mask
    from affine import Affine
    from pyproj import CRS, Transformer
    from shapely.geometry import Polygon, shape, mapping
    from shapely.ops import transform as shapely_transform, unary_union
except Exception:  # pragma: no cover - dependency failures are surfaced to caller
    rasterio = None
    geometry_mask = None
    Affine = None
    CRS = None
    Transformer = None
    Polygon = None
    shape = None
    mapping = None
    shapely_transform = None
    unary_union = None


def _require_geo_stack() -> None:
    if any(x is None for x in (rasterio, geometry_mask, Affine, CRS, Transformer, Polygon, shape, mapping, shapely_transform, unary_union)):
        raise RuntimeError("AOI coverage needs Rasterio, PyProj, Affine and Shapely; install requirements.txt")


def _crs_from_geojson(payload: dict[str, Any]) -> tuple[Any, str]:
    """Return CRS and source note. Modern RFC 7946 GeoJSON is always EPSG:4326."""
    legacy = payload.get("crs")
    if not legacy:
        return CRS.from_epsg(4326), "EPSG:4326_assumed_per_RFC7946"
    props = legacy.get("properties", {}) if isinstance(legacy, dict) else {}
    name = props.get("name") if isinstance(props, dict) else None
    if not name:
        raise ValueError("Legacy GeoJSON crs member is present but cannot be interpreted; provide standard WGS84 GeoJSON")
    try:
        return CRS.from_user_input(name), f"legacy_crs:{name}"
    except Exception as exc:
        raise ValueError(f"Invalid GeoJSON CRS {name!r}: {exc}") from exc


def load_aoi(path: str | Path):
    _require_geo_stack()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"AOI GeoJSON not found: {p}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ValueError(f"AOI is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("AOI GeoJSON top-level value must be an object")
    source_crs, crs_note = _crs_from_geojson(payload)
    items = []
    kind = payload.get("type")
    if kind == "FeatureCollection":
        features = payload.get("features")
        if not isinstance(features, list) or not features:
            raise ValueError("AOI FeatureCollection must contain at least one feature")
        for feature in features:
            if not isinstance(feature, dict) or not isinstance(feature.get("geometry"), dict):
                raise ValueError("Every AOI feature must contain a geometry")
            items.append(shape(feature["geometry"]))
    elif kind == "Feature":
        if not isinstance(payload.get("geometry"), dict):
            raise ValueError("AOI Feature must contain a geometry")
        items.append(shape(payload["geometry"]))
    elif kind in {"Polygon", "MultiPolygon"}:
        items.append(shape(payload))
    else:
        raise ValueError("AOI must be a Polygon, MultiPolygon, Feature or FeatureCollection of polygon geometries")
    if any(g.is_empty or not g.is_valid for g in items):
        raise ValueError("AOI contains an empty or invalid geometry; repair the GeoJSON geometry before analysis")
    if any(g.geom_type not in {"Polygon", "MultiPolygon"} for g in items):
        raise ValueError("AOI geometry must be polygonal")
    geom = unary_union(items)
    if geom.is_empty or not geom.is_valid or geom.area <= 0:
        raise ValueError("AOI union is empty, invalid or has zero area")
    return geom, source_crs, crs_note


def _to_equal_area(geom, source_crs):
    transformer = Transformer.from_crs(source_crs, CRS.from_epsg(6933), always_xy=True)
    return shapely_transform(transformer.transform, geom)


def aoi_mask_and_report(aoi_geojson: str | Path, raster_path: str | Path,
                        *, all_touched: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """Rasterize AOI on source grid and report geometric / valid-pixel coverage.

    ``all_touched=False`` uses pixel-center-style rasterization, avoiding systematic expansion
    of narrow AOIs. Set True deliberately if the application policy prefers any touched pixel.
    Returns a boolean ``[height,width]`` mask and JSON-safe report.
    """
    _require_geo_stack()
    geom_src, source_crs, crs_note = load_aoi(aoi_geojson)
    raster = Path(raster_path)
    if not raster.is_file():
        raise FileNotFoundError(f"Raster for AOI coverage not found: {raster}")
    with rasterio.open(raster) as ds:
        if ds.crs is None:
            raise ValueError(f"AOI analysis requires a georeferenced raster with CRS: {raster}")
        if ds.width <= 0 or ds.height <= 0:
            raise ValueError("Raster has empty dimensions")
        geom_raster = shapely_transform(Transformer.from_crs(source_crs, CRS.from_user_input(ds.crs), always_xy=True).transform, geom_src)
        if geom_raster.is_empty or not geom_raster.is_valid:
            raise ValueError("AOI transform into raster CRS produced invalid geometry")
        mask = geometry_mask([mapping(geom_raster)], out_shape=(ds.height, ds.width),
                             transform=ds.transform, invert=True, all_touched=bool(all_touched))
        n_aoi_pixels = int(mask.sum())
        if n_aoi_pixels == 0:
            return mask, {
                "status": "REJECT", "reason": "AOI does not include any rasterized pixels on this grid",
                "aoi_geojson": str(Path(aoi_geojson).resolve()), "raster_path": str(raster.resolve()),
                "source_crs_interpretation": crs_note, "raster_crs": str(ds.crs),
                "aoi_rasterized_pixels": 0, "aoi_rasterized_pixel_fraction_of_grid": 0.0,
                "aoi_footprint_coverage_fraction": 0.0, "all_touched": bool(all_touched),
                "coverage_semantics": "geometric area in equal-area EPSG:6933; pixel mask on existing raster grid"
            }
        # The dataset is processed in native blocks; only count pixels inside AOI, never guess nodata.
        valid_in_aoi = 0
        for _band_index, window in ds.block_windows(1):
            row0, col0 = int(window.row_off), int(window.col_off)
            row1, col1 = row0 + int(window.height), col0 + int(window.width)
            aoi_window = mask[row0:row1, col0:col1]
            if not aoi_window.any():
                continue
            values = ds.read(window=window, out_dtype="float32")
            band_masks = ds.read_masks(window=window) > 0
            valid = np.all(band_masks, axis=0) & np.all(np.isfinite(values), axis=0)
            if ds.nodata is not None:
                valid &= ~np.any(values == ds.nodata, axis=0)
            valid_in_aoi += int(np.count_nonzero(aoi_window & valid))
        # Compute how much of the AOI polygon is covered by the true affine raster footprint.
        # Densified pixel-edge rings reduce errors for rotated grids / nonlinear CRS transforms.
        steps = 64
        pixels = []
        for i in range(steps): pixels.append((ds.width * i / steps, 0))
        for i in range(steps): pixels.append((ds.width, ds.height * i / steps))
        for i in range(steps): pixels.append((ds.width * (steps-i) / steps, ds.height))
        for i in range(steps): pixels.append((0, ds.height * (steps-i) / steps))
        footprint_xy = [ds.transform * pt for pt in pixels]
        footprint = Polygon(footprint_xy)
        equal_aoi = _to_equal_area(geom_src, source_crs)
        equal_footprint = _to_equal_area(footprint, CRS.from_user_input(ds.crs))
        aoi_area = float(equal_aoi.area)
        intersection_area = float(equal_aoi.intersection(equal_footprint).area)
        footprint_fraction = float(np.clip(intersection_area / aoi_area, 0.0, 1.0)) if aoi_area > 0 else 0.0
        valid_fraction = float(valid_in_aoi / n_aoi_pixels)
        if valid_in_aoi == 0:
            status, reason = "REJECT", "AOI pixels exist, but no valid raster observations fall inside it"
        elif footprint_fraction < 0.999999:
            status, reason = "WARNING", "The raster covers only part of the requested AOI; predictions are restricted to covered pixels"
        else:
            status, reason = "PASS", "Raster footprint covers the AOI geometry; exact pixel mask is available"
        report = {
            "status": status, "reason": reason, "aoi_geojson": str(Path(aoi_geojson).resolve()),
            "raster_path": str(raster.resolve()), "source_crs_interpretation": crs_note,
            "raster_crs": str(ds.crs), "aoi_area_m2_equal_area_approx": aoi_area,
            "intersection_area_m2_equal_area_approx": intersection_area,
            "aoi_footprint_coverage_fraction": footprint_fraction,
            "aoi_rasterized_pixels": n_aoi_pixels,
            "aoi_rasterized_pixel_fraction_of_grid": float(n_aoi_pixels / (ds.width * ds.height)),
            "valid_pixels_inside_aoi": int(valid_in_aoi), "valid_fraction_within_rasterized_aoi": valid_fraction,
            "all_touched": bool(all_touched),
            "coverage_semantics": "geometry fraction measured in equal-area EPSG:6933; pixel validity from per-band masks + finite values + declared nodata",
            "note": "AOI footprint coverage does not prove image-pair registration. Valid fraction is relative to AOI pixel centers rasterized onto this raster grid."
        }
        return mask, report
