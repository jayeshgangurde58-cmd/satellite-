"""Polygonize change masks and write vectors with metric area + EPSG:4326 output."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
from skimage.measure import label
from .severity import change_priority_tier


def vectorize_change_mask(
    change_mask: np.ndarray,
    magnitude: np.ndarray,
    transform,
    source_crs,
    min_area_m2: float = 0.0,
) -> gpd.GeoDataFrame:
    mask = np.asarray(change_mask, dtype=bool)
    mag = np.asarray(magnitude, dtype=np.float32)
    if mask.ndim != 2 or mag.shape != mask.shape:
        raise ValueError("change_mask and magnitude must be same-sized 2-D arrays")
    if source_crs is None:
        raise ValueError("source_crs is required for geospatial polygon output")
    labels = label(mask, connectivity=2).astype(np.int32)
    records = []
    if labels.max() == 0:
        return gpd.GeoDataFrame(
            columns=["area_m2", "change_magnitude", "centroid_lon", "centroid_lat", "severity_tier", "priority_score", "geometry"],
            geometry="geometry", crs=source_crs,
        ).to_crs("EPSG:4326")

    p95 = float(np.percentile(mag[mask], 95)) if np.any(mask) else 0.0
    for geom_json, value in shapes(labels, mask=labels > 0, transform=transform, connectivity=8):
        component_id = int(value)
        component_geom = shape(geom_json)
        component_pixels = labels == component_id
        mean_magnitude = float(np.mean(mag[component_pixels]))
        relative_magnitude = mean_magnitude / p95 if p95 > 1e-12 else 0.0
        records.append({"component_id": component_id, "change_magnitude": mean_magnitude,
                        "magnitude_relative": min(relative_magnitude, 1.0), "geometry": component_geom})

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=source_crs)
    # UTM estimated from the data extent provides metric units for local area.
    metric_crs = gdf.estimate_utm_crs()
    if metric_crs is None:
        raise ValueError("Could not estimate a metric CRS; supply suitable georeferencing")
    metric = gdf.to_crs(metric_crs)
    metric["area_m2"] = metric.geometry.area
    keep = metric["area_m2"] >= float(min_area_m2)
    metric = metric.loc[keep].copy()
    metric["centroid_metric"] = metric.geometry.centroid
    centroids = gpd.GeoSeries(metric["centroid_metric"], crs=metric.crs).to_crs("EPSG:4326")
    metric["centroid_lon"] = centroids.x
    metric["centroid_lat"] = centroids.y
    tiers = [change_priority_tier(a, m) for a, m in zip(metric["area_m2"], metric["magnitude_relative"])]
    metric["severity_tier"] = [x[0] for x in tiers]
    metric["priority_score"] = [x[1] for x in tiers]
    metric = metric.drop(columns=["centroid_metric", "magnitude_relative"])
    return metric.to_crs("EPSG:4326")


def write_vectors(gdf: gpd.GeoDataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix not in {".geojson", ".json", ".gpkg"}:
        raise ValueError("Output extension must be .geojson, .json, or .gpkg")
    
    if suffix in {".geojson", ".json"}:
        path.write_text(gdf.to_json(), encoding="utf-8")
    else:
        try:
            gdf.to_file(path, driver="GPKG", index=False)
        except ImportError as e:
            import logging
            logging.getLogger(__name__).warning("Skipping GPKG export due to missing pyogrio/GDAL DLLs: %s", e)
