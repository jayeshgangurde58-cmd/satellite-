"""Fetch a small same-MGRS-tile Sentinel-2 L2A pair from Planetary Computer.

Optional dependencies: install requirements-data.txt. This reads COG windows for
an AOI, writes 4-band B02/B03/B04/B08 GeoTIFFs with an SCL-derived dataset mask,
and records item IDs/provenance. It does NOT guarantee content-level registration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
BANDS = ("B02", "B03", "B04", "B08")
# SCL values excluded for conservative prototype masking: no data, saturated,
# cloud shadow, unclassified, medium/high probability cloud, cirrus, snow/ice.
INVALID_SCL = (0, 1, 3, 7, 8, 9, 10, 11)


def list_items(client: Client, bbox: list[float], period: str, max_cloud: float,
               max_items: int) -> list[Any]:
    search = client.search(
        collections=[COLLECTION], bbox=bbox, datetime=period, max_items=max_items,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.items())
    return [i for i in items if i.properties.get("s2:mgrs_tile")]


def choose_pair(before_items: list[Any], after_items: list[Any]) -> tuple[Any, Any]:
    before_by_tile: dict[str, list[Any]] = {}
    after_by_tile: dict[str, list[Any]] = {}
    for item in before_items:
        before_by_tile.setdefault(item.properties["s2:mgrs_tile"], []).append(item)
    for item in after_items:
        after_by_tile.setdefault(item.properties["s2:mgrs_tile"], []).append(item)
    common = set(before_by_tile) & set(after_by_tile)
    if not common:
        raise RuntimeError("No MGRS tile appears in both periods. Widen dates/bbox or relax cloud limit.")
    candidates = []
    for tile in common:
        for b in before_by_tile[tile]:
            for a in after_by_tile[tile]:
                # Prefer low cloud cover, then shorter item processing strings for deterministic ties.
                cloud_score = float(b.properties.get("eo:cloud_cover", 100)) + float(a.properties.get("eo:cloud_cover", 100))
                sensor_penalty = 10.0 if b.properties.get("platform") != a.properties.get("platform") else 0.0
                baseline_penalty = 5.0 if b.properties.get("s2:processing_baseline") != a.properties.get("s2:processing_baseline") else 0.0
                score = cloud_score + sensor_penalty + baseline_penalty
                candidates.append((score, tile, str(b.id), str(a.id), b, a))
    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    best = candidates[0]
    return best[4], best[5]


def _read_window(href: str, bbox: list[float]) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(href) as src:
        projected = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
        window = from_bounds(*projected, transform=src.transform).round_offsets().round_lengths()
        window = window.intersection(Window(0, 0, src.width, src.height))
        if window.width < 2 or window.height < 2:
            raise RuntimeError("Requested AOI is outside or too small within the selected Sentinel-2 tile")
        arr = src.read(1, window=window, boundless=False)
        mask = src.read_masks(1, window=window) > 0
        meta = {
            "crs": src.crs,
            "transform": src.window_transform(window),
            "width": int(window.width),
            "height": int(window.height),
            "profile": src.profile.copy(),
            "valid_mask": mask,
        }
        return arr, meta


def read_scene(item: Any, bbox: list[float]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import planetary_computer as pc
    signed = pc.sign(item)
    arrays: list[np.ndarray] = []
    band_metas: list[dict[str, Any]] = []
    for band in BANDS:
        if band not in signed.assets:
            raise RuntimeError(f"Selected STAC item {item.id} has no {band} asset")
        arr, meta = _read_window(signed.assets[band].href, bbox)
        arrays.append(arr)
        band_metas.append(meta)
    ref = band_metas[0]
    for band, meta, arr in zip(BANDS, band_metas, arrays):
        if meta["crs"] != ref["crs"] or meta["transform"] != ref["transform"] or arr.shape != arrays[0].shape:
            raise RuntimeError(f"Band grid mismatch at {band}; refusing to silently resample 10m bands")

    if "SCL" not in signed.assets:
        raise RuntimeError(f"Selected STAC item {item.id} has no SCL asset; cannot apply cloud/shadow mask")
    scl, scl_meta = _read_window(signed.assets["SCL"].href, bbox)
    scl_10m = np.zeros(arrays[0].shape, dtype=np.uint8)
    reproject(
        source=scl, destination=scl_10m,
        src_transform=scl_meta["transform"], src_crs=scl_meta["crs"],
        dst_transform=ref["transform"], dst_crs=ref["crs"],
        resampling=Resampling.nearest, src_nodata=0, dst_nodata=0,
    )
    valid = np.logical_and.reduce([m["valid_mask"] for m in band_metas])
    valid &= ~np.isin(scl_10m, INVALID_SCL)
    valid &= np.logical_and.reduce([np.isfinite(x) for x in arrays])
    metadata = {
        "id": item.id,
        "datetime": item.properties.get("datetime"),
        "mgrs_tile": item.properties.get("s2:mgrs_tile"),
        "platform": item.properties.get("platform"),
        "processing_baseline": item.properties.get("s2:processing_baseline"),
        "eo_cloud_cover_percent": item.properties.get("eo:cloud_cover"),
        "collection": COLLECTION,
        "crs": ref["crs"].to_string(),
        "transform": list(ref["transform"]),
        "width": ref["width"], "height": ref["height"],
        "bands": list(BANDS), "native_resolution_m": 10,
        "mask": f"SCL exclusions {list(INVALID_SCL)}; per-band dataset masks; common valid pixels={int(valid.sum())}",
        "license_link": next((link.href for link in item.links if link.rel == "license"),
                              "https://cds.climate.copernicus.eu/licences/ec-sentinel"),
    }
    return np.stack(arrays, axis=0), valid, {"grid": ref, "metadata": metadata}


def save_scene(path: Path, stack: np.ndarray, valid: np.ndarray, grid: dict[str, Any]) -> None:
    profile = grid["profile"].copy()
    for key in ("photometric", "interleave", "predictor", "blockxsize", "blockysize", "tiled"):
        profile.pop(key, None)
    profile.update(
        driver="GTiff", height=stack.shape[1], width=stack.shape[2], count=stack.shape[0],
        dtype=str(stack.dtype), crs=grid["crs"], transform=grid["transform"],
        nodata=None, compress="deflate",
    )
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(stack)
            dst.write_mask(valid.astype(np.uint8) * 255)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", nargs=4, type=float, required=True,
                        metavar=("WEST", "SOUTH", "EAST", "NORTH"), help="WGS84 degrees")
    parser.add_argument("--before-period", required=True, help="STAC interval, e.g. 2023-10-01/2023-11-30")
    parser.add_argument("--after-period", required=True, help="STAC interval, e.g. 2025-10-01/2025-11-30")
    parser.add_argument("--max-cloud", type=float, default=20.0, help="Item-level cloud-cover percentage")
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("data/real_pair"))
    args = parser.parse_args()
    west, south, east, north = args.bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        parser.error("bbox must be ordered W,S,E,N in valid WGS84 ranges")
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        from pystac_client import Client
    except ImportError as exc:
        raise SystemExit("Optional STAC dependencies missing. Run: pip install -r requirements-data.txt") from exc
    client = Client.open(STAC_URL)
    before_items = list_items(client, args.bbox, args.before_period, args.max_cloud, args.max_items)
    after_items = list_items(client, args.bbox, args.after_period, args.max_cloud, args.max_items)
    if not before_items or not after_items:
        raise SystemExit("No candidate scenes. Widen date periods or increase --max-cloud / --max-items.")
    before_item, after_item = choose_pair(before_items, after_items)
    print("Selected STAC items:")
    print(" BEFORE", before_item.id, before_item.properties.get("datetime"), before_item.properties.get("eo:cloud_cover"))
    print(" AFTER ", after_item.id, after_item.properties.get("datetime"), after_item.properties.get("eo:cloud_cover"))
    before, before_valid, before_info = read_scene(before_item, args.bbox)
    after, after_valid, after_info = read_scene(after_item, args.bbox)
    if before.shape != after.shape:
        raise SystemExit(f"Shape mismatch: before {before.shape}; after {after.shape}. Select another pair.")
    bg = before_info["grid"]; ag = after_info["grid"]
    if bg["crs"] != ag["crs"] or bg["transform"] != ag["transform"]:
        raise SystemExit("Grid mismatch between dates. Select a same-MGRS pair; no implicit warp is performed.")
    save_scene(args.out / "before.tif", before, before_valid, bg)
    save_scene(args.out / "after.tif", after, after_valid, ag)
    metadata = {
        "purpose": "real Sentinel-2 L2A pair; candidate inputs only, not validated registration",
        "stac_api": STAC_URL, "collection": COLLECTION, "bbox_wgs84": args.bbox,
        "bands": list(BANDS), "grid_crs": bg["crs"].to_string(),
        "grid_transform": list(bg["transform"]), "shape_bands_rows_cols": list(before.shape),
        "before_valid_pixels": int(before_valid.sum()), "after_valid_pixels": int(after_valid.sum()),
        "common_valid_pixels": int((before_valid & after_valid).sum()),
        "before": before_info["metadata"], "after": after_info["metadata"],
        "caveat": "Matching grid metadata is not proof of content-level co-registration. Visually inspect static features and check residual shifts before interpreting change polygons. Sentinel-2 L2A radiometric offsets/processing baselines must be documented; these default periods both start after the Jan 2022 baseline change, but this does not make all acquisition differences disappear.",
        "license_note": "Sentinel data are under Copernicus Sentinel data terms. Preserve provenance and review terms for all data and derived outputs before public/commercial reuse.",
    }
    (args.out / "pair_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved:", args.out.resolve())
    print("Shape:", before.shape, "CRS:", bg["crs"], "Valid overlap:", int((before_valid & after_valid).sum()))
    print("IMPORTANT: inspect the pair visually and verify content alignment before presenting results.")


if __name__ == "__main__":
    main()
