"""Inspect GeoTIFF metadata and optionally estimate a rough residual translation.

The shift estimate is a diagnostic only: real change, seasonality, masks and
radiometric differences can bias whole-image phase correlation. Never auto-warp
based solely on this estimate.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from skimage.registration import phase_cross_correlation
from skimage.transform import resize


def describe(path: Path) -> dict:
    with rasterio.open(path) as src:
        masks = src.read_masks() > 0
        data = src.read()
        per_band_valid = masks.reshape(src.count, -1).mean(axis=1).tolist()
        tags = src.tags()
        return {
            "path": str(path.resolve()), "driver": src.driver,
            "crs": src.crs.to_string() if src.crs else None,
            "transform": list(src.transform), "bounds": list(src.bounds),
            "bounds_epsg4326": list(transform_bounds(src.crs, "EPSG:4326", *src.bounds)) if src.crs else None,
            "width": src.width, "height": src.height, "count": src.count,
            "dtypes": list(src.dtypes), "res": list(src.res), "nodata": src.nodata,
            "valid_fraction_per_band": per_band_valid,
            "finite_fraction": float(np.isfinite(data).mean()), "tags": tags,
        }


def rough_shift(before_path: Path, after_path: Path, max_side: int = 512) -> dict:
    with rasterio.open(before_path) as b, rasterio.open(after_path) as a:
        if b.crs != a.crs or b.transform != a.transform or b.width != a.width or b.height != a.height:
            raise ValueError("Cannot estimate a meaningful shift before grids are equal")
        out_h = min(max_side, b.height); out_w = min(max_side, b.width)
        bdata = b.read(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
        adata = a.read(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
        bm = b.read_masks(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.nearest) > 0
        am = a.read_masks(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.nearest) > 0
        valid = bm & am & np.isfinite(bdata) & np.isfinite(adata)
        if valid.mean() < 0.25:
            raise ValueError("Too little valid overlap for rough phase-correlation diagnostic")
        def normalise(x):
            vals = x[valid]
            lo, hi = np.percentile(vals, [2, 98])
            return np.clip((x - lo) / max(hi-lo, 1e-6), 0, 1)
        ref, mov = normalise(bdata), normalise(adata)
        ref[~valid] = np.median(ref[valid]); mov[~valid] = np.median(mov[valid])
        shift, error, phase = phase_cross_correlation(ref, mov, upsample_factor=10)
        return {"estimated_shift_pixels_downsampled": [float(shift[0]), float(shift[1])],
                "downsample_shape": [out_h, out_w], "valid_fraction": float(valid.mean()),
                "registration_error_indicator": float(error) if np.isfinite(error) else None,
                "phase_difference": float(phase) if np.isfinite(phase) else None,
                "warning": "Whole-image shift estimate can be biased by actual change or radiometric differences; it is not proof of registration and must not trigger automatic warping without review."}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--after", type=Path, required=True)
    ap.add_argument("--estimate-shift", action="store_true")
    ap.add_argument("--output", type=Path, default=None, help="Optional JSON report output")
    args = ap.parse_args()
    b = describe(args.before); a = describe(args.after)
    same_grid = (b["crs"] == a["crs"] and b["transform"] == a["transform"]
                 and b["width"] == a["width"] and b["height"] == a["height"])
    report = {"before": b, "after": a, "same_grid_metadata": same_grid,
              "warning": "Same grid metadata is not proof of content-level co-registration."}
    if args.estimate_shift and same_grid:
        report["rough_shift_diagnostic"] = rough_shift(args.before, args.after)
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not same_grid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
