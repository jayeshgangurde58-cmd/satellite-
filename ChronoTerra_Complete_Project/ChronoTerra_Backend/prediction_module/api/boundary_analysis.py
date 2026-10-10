"""Reference-zone overlay validation and conservative candidate intersection analysis.

Input GeoJSON must be RFC 7946 WGS84 coordinates and polygonal features. Uploaded
geometry is user-provided context, not independently verified legal/ownership evidence.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pyproj import CRS, Geod, Transformer
from shapely.geometry import shape, mapping
from shapely.ops import transform as shapely_transform, unary_union

MAX_BOUNDARY_FEATURES = 10_000
MAX_BOUNDARY_BYTES = 10 * 1024 * 1024
GEOD = Geod(ellps="WGS84")


def _load_geojson_boundary(path: str | Path) -> tuple[list[Any], Any]:
    p = Path(path)
    if p.stat().st_size > MAX_BOUNDARY_BYTES:
        raise ValueError("Reference-zone GeoJSON exceeds the 10 MiB limit.")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Reference-zone file is not valid UTF-8 GeoJSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Reference-zone GeoJSON must be a FeatureCollection or Feature.")
    if "crs" in payload:
        raise ValueError("GeoJSON with a legacy/custom CRS declaration is not accepted; provide RFC 7946 WGS84 longitude/latitude coordinates.")
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features")
        if not isinstance(features, list) or not features:
            raise ValueError("Reference-zone FeatureCollection must contain at least one polygon feature.")
    elif payload.get("type") == "Feature":
        features = [payload]
    else:
        raise ValueError("Reference-zone GeoJSON must be a FeatureCollection or Feature containing polygons.")
    if len(features) > MAX_BOUNDARY_FEATURES:
        raise ValueError(f"Reference-zone GeoJSON may contain at most {MAX_BOUNDARY_FEATURES} features.")

    geoms: list[Any] = []
    for idx, feature in enumerate(features):
        if not isinstance(feature, dict) or feature.get("type") != "Feature" or not isinstance(feature.get("geometry"), dict):
            raise ValueError(f"Reference-zone feature {idx + 1} is malformed.")
        try:
            geom = shape(feature["geometry"])
        except Exception as exc:
            raise ValueError(f"Reference-zone feature {idx + 1} has an unreadable geometry.") from exc
        if geom.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError(f"Reference-zone feature {idx + 1} must be Polygon or MultiPolygon.")
        if geom.is_empty or not geom.is_valid:
            raise ValueError(f"Reference-zone feature {idx + 1} is empty or invalid; repair it in a GIS and retry.")
        minx, miny, maxx, maxy = geom.bounds
        if not all(math.isfinite(v) for v in (minx, miny, maxx, maxy)):
            raise ValueError(f"Reference-zone feature {idx + 1} contains non-finite coordinates.")
        if minx < -180 or maxx > 180 or miny < -90 or maxy > 90:
            raise ValueError("Reference-zone coordinates fall outside WGS84 longitude/latitude bounds. Re-export RFC 7946 GeoJSON from a GIS.")
        geoms.append(geom)
    try:
        merged = unary_union(geoms)
    except Exception as exc:
        raise ValueError("Reference-zone polygons could not be combined; check geometry validity and topology.") from exc
    if merged.is_empty or not merged.is_valid:
        raise ValueError("Combined reference-zone geometry is invalid; repair overlaps or self-intersections in a GIS.")
    return geoms, merged


def _geodesic_area(geom: Any) -> float:
    if geom.is_empty:
        return 0.0
    area, _ = GEOD.geometry_area_perimeter(geom)
    return abs(float(area))


def _metric_distance(candidate: Any, reference: Any) -> float:
    """Distance in metres using a local azimuthal-equidistant CRS around the candidate."""
    centroid = candidate.representative_point()
    lon, lat = float(centroid.x), float(centroid.y)
    local_crs = CRS.from_proj4(f"+proj=aeqd +lat_0={lat:.8f} +lon_0={lon:.8f} +datum=WGS84 +units=m +no_defs")
    transformer = Transformer.from_crs(CRS.from_epsg(4326), local_crs, always_xy=True)
    project = lambda x, y, z=None: transformer.transform(x, y)
    c_local = shapely_transform(project, candidate)
    r_local = shapely_transform(project, reference)
    return float(c_local.distance(r_local))


def analyze_boundary_candidates(
    regions_path: str | Path,
    boundary_path: str | Path,
    boundary_overlay_path: str | Path,
    *,
    source_name: str | None = None,
    effective_date: str | None = None,
    proximity_m: float = 10.0,
) -> dict[str, Any]:
    """Annotate retained geographic candidate regions against a reference-zone polygon layer.

    The analysis is not a legal encroachment decision. Region results are limited to the
    polygons emitted in regions_path (which can be truncated to the largest components).
    """
    if not math.isfinite(proximity_m) or proximity_m < 0 or proximity_m > 1000:
        raise ValueError("Reference-zone proximity must be between 0 and 1000 metres.")
    _, reference = _load_geojson_boundary(boundary_path)
    regions_file = Path(regions_path)
    try:
        collection = json.loads(regions_file.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Candidate region output is not valid GeoJSON.") from exc
    if collection.get("coordinate_space") != "EPSG:4326":
        raise ValueError("Reference-zone comparison requires georeferenced prediction polygons. Image-space predictions cannot be compared to geographic boundaries.")

    overlap_count = 0
    near_count = 0
    tested = 0
    for feature in collection.get("features", []):
        geom = shape(feature["geometry"])
        if geom.is_empty or not geom.is_valid:
            feature.setdefault("properties", {})["boundary_analysis_status"] = "UNVERIFIED_INVALID_CANDIDATE_GEOMETRY"
            continue
        tested += 1
        intersection = geom.intersection(reference)
        overlap_m2 = _geodesic_area(intersection)
        candidate_m2 = _geodesic_area(geom)
        overlap_fraction = min(1.0, max(0.0, overlap_m2 / candidate_m2)) if candidate_m2 > 0 else 0.0
        intersects_area = overlap_m2 > max(0.01, candidate_m2 * 1e-9)
        distance_m = 0.0 if intersects_area else _metric_distance(geom, reference)
        if intersects_area:
            relation = "BOUNDARY_OVERLAP_CANDIDATE"
            overlap_count += 1
        elif distance_m <= proximity_m:
            relation = "NEAR_BOUNDARY_CANDIDATE"
            near_count += 1
        else:
            relation = "CANDIDATE_CHANGE"
        props = feature.setdefault("properties", {})
        props.update({
            "boundary_relation": relation,
            "boundary_overlap_m2": overlap_m2,
            "boundary_overlap_fraction": overlap_fraction,
            "distance_to_reference_zone_m": distance_m,
            "boundary_source_name": (source_name or "User-supplied reference zone")[:160],
            "boundary_effective_date": effective_date or None,
            "boundary_authority_status": "USER_SUPPLIED_NOT_INDEPENDENTLY_VERIFIED",
            "review_priority": "HIGHER_REVIEW_PRIORITY" if relation == "BOUNDARY_OVERLAP_CANDIDATE" else "REVIEW_IF_CONTEXT_REQUIRES" if relation == "NEAR_BOUNDARY_CANDIDATE" else "UNRANKED",
        })
    collection["boundary_analysis"] = {
        "status": "COMPLETED",
        "source_name": (source_name or "User-supplied reference zone")[:160],
        "effective_date": effective_date or None,
        "authority_status": "USER_SUPPLIED_NOT_INDEPENDENTLY_VERIFIED",
        "proximity_threshold_m": float(proximity_m),
        "candidate_regions_tested": tested,
        "boundary_overlap_candidate_count": overlap_count,
        "near_boundary_candidate_count": near_count,
        "interpretation": "Spatial screening only; not proof of ownership, violation, or illegal encroachment.",
    }
    regions_file.write_text(json.dumps(collection, indent=2, allow_nan=False), encoding="utf-8")

    # Publish a geometry-only copy rather than arbitrary user properties.
    clean_features = []
    for idx, geom in enumerate(_load_geojson_boundary(boundary_path)[0], start=1):
        clean_features.append({"type": "Feature", "geometry": mapping(geom), "properties": {"reference_zone_id": f"Z{idx:05d}"}})
    clean = {"type": "FeatureCollection", "name": "user_supplied_reference_zone", "coordinate_space": "EPSG:4326", "features": clean_features}
    Path(boundary_overlay_path).write_text(json.dumps(clean, indent=2, allow_nan=False), encoding="utf-8")
    return collection["boundary_analysis"]


def validate_boundary(path: str | Path) -> dict[str, Any]:
    geoms, merged = _load_geojson_boundary(path)
    minx, miny, maxx, maxy = merged.bounds
    return {
        "status": "PASS",
        "feature_count": len(geoms),
        "geometry_type": merged.geom_type,
        "bounds_wgs84": [float(minx), float(miny), float(maxx), float(maxy)],
        "coordinate_reference": "RFC7946 WGS84 longitude/latitude (assumed as required by GeoJSON)",
        "authority_status": "USER_SUPPLIED_NOT_INDEPENDENTLY_VERIFIED",
        "note": "This validates geometry and coordinate conventions; it does not validate legal authority, ownership, effective date, or source identity.",
    }
