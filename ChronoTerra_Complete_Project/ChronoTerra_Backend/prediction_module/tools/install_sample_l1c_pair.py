"""Download a genuine Sentinel-2 L1C pair and stage it for ChronoTerra inference tests.

Fetches 13-band Sentinel-2 L1C data over a small AOI via the public AWS Open Data registry.
Resamples 20m and 60m bands to a 10m target grid (bilinear) and outputs a pair of 13-band
GeoTIFFs. No API credentials are required.

Writes outputs to data/sample_l1c_pair/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform

logger = logging.getLogger(__name__)

BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")

# We target a specific known pair over Rome (33TTG) on public AWS S3.
# These tiles are publicly readable over HTTPS without Requester Pays.
BEFORE_BASE = "https://sentinel-s2-l1c.s3.amazonaws.com/tiles/33/T/TG/2023/5/25/0"
AFTER_BASE = "https://sentinel-s2-l1c.s3.amazonaws.com/tiles/33/T/TG/2023/8/23/0"
TARGET_SIZE = 512
# An arbitrary offset from the top-left of the 10m grid to crop an interesting area
TARGET_X_OFF = 4000
TARGET_Y_OFF = 4000

def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def fetch_tile_info(base_url: str) -> dict[str, Any]:
    url = f"{base_url}/tileInfo.json"
    logger.info("Fetching tile info from %s", url)
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))

def fetch_and_resample_bands(base_url: str) -> tuple[np.ndarray, dict[str, Any]]:
    # We use B02 (10m) to establish the master 10m grid
    b02_url = f"{base_url}/B02.jp2"
    
    with rasterio.Env(GDAL_HTTP_RETRY=3, GDAL_HTTP_TIMEOUT=15):
        with rasterio.open(b02_url) as src:
            target_win = Window(TARGET_X_OFF, TARGET_Y_OFF, TARGET_SIZE, TARGET_SIZE)
            bounds = rasterio.windows.bounds(target_win, src.transform)
            target_transform = window_transform(target_win, src.transform)
            target_crs = src.crs
            
            # The profile for the output multi-band GeoTIFF
            profile = src.profile.copy()
            for key in ("photometric", "interleave", "predictor", "blockxsize", "blockysize", "tiled"):
                profile.pop(key, None)
            
            profile.update(
                driver="GTiff",
                height=TARGET_SIZE,
                width=TARGET_SIZE,
                count=13,
                dtype="uint16",
                crs=target_crs,
                transform=target_transform,
                nodata=0,  # S2 L1C nodata is typically 0
                compress="deflate"
            )

    out_stack = np.zeros((13, TARGET_SIZE, TARGET_SIZE), dtype=np.uint16)
    
    with rasterio.Env(GDAL_HTTP_RETRY=3, GDAL_HTTP_TIMEOUT=15):
        for idx, band in enumerate(BANDS):
            band_url = f"{base_url}/{band}.jp2"
            logger.info("Reading %s...", band_url)
            with rasterio.open(band_url) as src:
                # Read exactly the bounds in the source grid
                src_win = from_bounds(*bounds, transform=src.transform).round_lengths().round_offsets()
                src_arr = src.read(1, window=src_win)
                src_transform = window_transform(src_win, src.transform)
                
                if src_arr.shape == (TARGET_SIZE, TARGET_SIZE) and src.transform == target_transform:
                    out_stack[idx] = src_arr
                else:
                    # Resample to the 10m grid
                    reproject(
                        source=src_arr,
                        destination=out_stack[idx],
                        src_transform=src_transform,
                        src_crs=src.crs,
                        dst_transform=target_transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear
                    )
                    
    return out_stack, profile

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/sample_l1c_pair"))
    parser.add_argument("--before-base", type=str, default=BEFORE_BASE, help="Override before tile URL base")
    parser.add_argument("--after-base", type=str, default=AFTER_BASE, help="Override after tile URL base")
    args = parser.parse_args()
    
    out_dir = args.out_dir.resolve()
    tmp_dir = out_dir / ".tmp_download"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        before_info = fetch_tile_info(args.before_base)
        after_info = fetch_tile_info(args.after_base)
        
        before_date = before_info.get("timestamp", "2023-05-22T00:00:00.000Z")
        after_date = after_info.get("timestamp", "2023-08-20T00:00:00.000Z")
        
        before_stack, before_prof = fetch_and_resample_bands(args.before_base)
        after_stack, after_prof = fetch_and_resample_bands(args.after_base)
        
        before_tmp = tmp_dir / "before.tif"
        after_tmp = tmp_dir / "after.tif"
        
        for stack, prof, path, date_str in [
            (before_stack, before_prof, before_tmp, before_date),
            (after_stack, after_prof, after_tmp, after_date)
        ]:
            with rasterio.open(path, "w", **prof) as dst:
                dst.write(stack)
                for i, b in enumerate(BANDS, 1):
                    dst.set_band_description(i, b)
                dst.update_tags(ACQUISITION_DATE=date_str)
                
        before_hash = _hash_file(before_tmp)
        after_hash = _hash_file(after_tmp)
        
        manifest = {
            "schema": "chronoterra_l1c_sample_v1",
            "provider": "AWS Open Data Sentinel-2 L1C",
            "bbox_grid_offset": [TARGET_X_OFF, TARGET_Y_OFF, TARGET_SIZE, TARGET_SIZE],
            "before": {
                "base_url": args.before_base,
                "timestamp": before_date,
                "sha256": before_hash
            },
            "after": {
                "base_url": args.after_base,
                "timestamp": after_date,
                "sha256": after_hash
            },
            "band_order": list(BANDS),
            "target_resolution_m": 10,
            "resampling_method": "bilinear for 20m/60m bands to match 10m grid",
            "output_crs": before_prof["crs"].to_string(),
            "output_transform": list(before_prof["transform"]),
            "attribution": "Contains modified Copernicus Sentinel data [2023]",
            "warning": "This is a structurally compatible 13-band sample for software validation. It does not prove model suitability."
        }
        
        (tmp_dir / "pair_metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        
        # Move successfully generated files to final directory atomically
        if (out_dir / "before.tif").exists(): os.remove(out_dir / "before.tif")
        if (out_dir / "after.tif").exists(): os.remove(out_dir / "after.tif")
        if (out_dir / "pair_metadata.json").exists(): os.remove(out_dir / "pair_metadata.json")
        
        shutil.move(str(before_tmp), str(out_dir / "before.tif"))
        shutil.move(str(after_tmp), str(out_dir / "after.tif"))
        shutil.move(str(tmp_dir / "pair_metadata.json"), str(out_dir / "pair_metadata.json"))
        
        logger.info("Successfully installed 13-band Sentinel-2 L1C pair to %s", out_dir)
        
    except Exception as exc:
        logger.error("Installation failed: %s", exc)
        raise
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
