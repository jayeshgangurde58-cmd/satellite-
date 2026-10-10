"""Strict GeoTIFF input validation for real geospatial runs."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import rasterio


@dataclass
class RasterPair:
    before: np.ndarray  # (bands, rows, cols)
    after: np.ndarray
    valid_mask: np.ndarray  # (rows, cols)
    profile: dict
    crs: object


def load_geotiff_pair(before_path: str | Path, after_path: str | Path) -> RasterPair:
    """Load same-grid, co-registered multiband GeoTIFFs; fail rather than silently warp."""
    with rasterio.open(before_path) as bsrc, rasterio.open(after_path) as asrc:
        if bsrc.crs is None or asrc.crs is None:
            raise ValueError("Both GeoTIFFs must define a CRS")
        if bsrc.crs != asrc.crs:
            raise ValueError(f"CRS mismatch: {bsrc.crs} != {asrc.crs}; reproject explicitly first")
        if bsrc.width != asrc.width or bsrc.height != asrc.height:
            raise ValueError("Raster dimensions differ; align/resample explicitly before analysis")
        if bsrc.transform != asrc.transform:
            raise ValueError("Affine transforms differ; co-register/alignment is required")
        if bsrc.count != asrc.count:
            raise ValueError(f"Band count differs: {bsrc.count} != {asrc.count}")

        before = bsrc.read().astype(np.float32)
        after = asrc.read().astype(np.float32)
        valid = (bsrc.read_masks() > 0).all(axis=0) & (asrc.read_masks() > 0).all(axis=0)
        valid &= np.isfinite(before).all(axis=0) & np.isfinite(after).all(axis=0)
        if not valid.any():
            raise ValueError("No valid overlapping pixels in the raster pair")
        return RasterPair(before, after, valid, bsrc.profile.copy(), bsrc.crs)
