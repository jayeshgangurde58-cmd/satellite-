"""Run the baseline on two same-grid, georeferenced multiband GeoTIFFs."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import rasterio
from .geo_io import load_geotiff_pair
from .change import run_baseline
from .vectorize import vectorize_change_mask, write_vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--mask-out", type=Path, default=Path("outputs/change_mask.tif"))
    parser.add_argument("--vectors-out", type=Path, default=Path("outputs/change_polygons.geojson"))
    parser.add_argument("--min-object-size", type=int, default=8)
    parser.add_argument("--opening-radius", type=int, default=1)
    parser.add_argument("--min-area-m2", type=float, default=0.0)
    parser.add_argument("--histogram-match", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    pair = load_geotiff_pair(args.before, args.after)
    predicted, magnitude, threshold = run_baseline(
        pair.before, pair.after, pair.valid_mask, histogram_match=args.histogram_match,
        min_object_size=args.min_object_size, opening_radius=args.opening_radius,
    )
    args.mask_out.parent.mkdir(parents=True, exist_ok=True)
    profile = pair.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=255, compress="lzw")
    output_mask = np.where(pair.valid_mask, predicted.astype(np.uint8), 255).astype(np.uint8)
    with rasterio.open(args.mask_out, "w", **profile) as dst:
        dst.write(output_mask, 1)

    gdf = vectorize_change_mask(predicted, magnitude, pair.profile["transform"], pair.crs,
                                min_area_m2=args.min_area_m2)
    write_vectors(gdf, args.vectors_out)
    print(f"Otsu threshold: {threshold:.6f}")
    print(f"Valid pixels: {int(pair.valid_mask.sum())}; predicted change pixels: {int(predicted.sum())}")
    print(f"Polygons: {len(gdf)}")
    print(f"Mask: {args.mask_out.resolve()}")
    print(f"Vectors: {args.vectors_out.resolve()}")
    print("Warning: severity_tier is a configurable change-priority heuristic, not validated damage severity.")


if __name__ == "__main__":
    main()
