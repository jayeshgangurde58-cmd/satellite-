"""Experimental inference on a georeferenced 13-band GeoTIFF pair.

This checkpoint is for OSCD-compatible 13-band optical input only. It is not
for arbitrary JPG/PNG uploads, Sentinel-1 SAR, or building-damage classification.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import numpy as np
import rasterio
import torch

from models.oscd_data import BANDS
from models.siamese_unet import SiameseUNet
from models.train_oscd import tiled_predict


def _torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def read_stack(path: Path) -> tuple[np.ndarray, np.ndarray, dict, list[str]]:
    with rasterio.open(path) as src:
        if src.count != 13:
            raise ValueError(f"{path} has {src.count} bands; this model requires exactly 13 OSCD-order bands")
        arr = src.read().astype(np.float32)
        valid = np.logical_and.reduce([(src.read_masks(i + 1) > 0) & np.isfinite(arr[i]) for i in range(src.count)])
        profile = src.profile.copy()
        descriptions = [d or "" for d in src.descriptions]
        return arr, valid, profile, descriptions


def check_pair(before_path: Path, after_path: Path, assume_band_order: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, list[str]]:
    b_raw, b_valid, b_profile, b_desc = read_stack(before_path)
    a_raw, a_valid, a_profile, a_desc = read_stack(after_path)
    for key, bval, aval in [
        ("shape", b_raw.shape, a_raw.shape), ("CRS", b_profile.get("crs"), a_profile.get("crs")),
        ("transform", b_profile.get("transform"), a_profile.get("transform")),
    ]:
        if bval != aval:
            raise ValueError(f"Before/after {key} mismatch; perform a reviewed geospatial alignment first")
    if b_profile.get("crs") is None:
        raise ValueError("Input GeoTIFFs must have CRS metadata")
    warnings = []
    for label, descs in [("before", b_desc), ("after", a_desc)]:
        normalized = [re.sub(r"[^A-Z0-9]", "", x.upper()) for x in descs]
        recognized = all(band in desc for band, desc in zip(BANDS, normalized))
        if not recognized:
            if not assume_band_order:
                raise ValueError(f"{label} band descriptions do not prove the expected B01..B12/B8A order. Verify order and rerun with --assume-oscd-band-order only if you have confirmed it.")
            warnings.append(f"{label} band order was manually assumed to match {list(BANDS)}")
    valid = b_valid & a_valid
    if not valid.any():
        raise ValueError("No common valid pixels between inputs")
    if not b_valid.all() or not a_valid.all():
        warnings.append("Invalid/NoData pixels exist and will be marked as NoData in the result")
    return b_raw, a_raw, valid, b_profile, warnings


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-prefix", type=Path, required=True, help="Output prefix, e.g. outputs/urban_prediction")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--assume-oscd-band-order", action="store_true", help="Use only after independently verifying band order")
    args = p.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but not available")
    ckpt = _torch_load(args.checkpoint, device)
    cfg = ckpt["model_config"]
    if ckpt.get("bands") != list(BANDS):
        raise SystemExit("Checkpoint band profile does not match supported OSCD order")
    b_raw, a_raw, valid, profile, warnings = check_pair(args.before, args.after, args.assume_oscd_band_order)
    scale = float(ckpt["scale_factor"])
    # Invalid cells are replaced only for model execution; their outputs are masked to NoData below.
    b = np.clip(np.nan_to_num(b_raw, nan=0.0, posinf=0.0, neginf=0.0) / scale, 0.0, 1.0).astype(np.float32)
    a = np.clip(np.nan_to_num(a_raw, nan=0.0, posinf=0.0, neginf=0.0) / scale, 0.0, 1.0).astype(np.float32)

    model = SiameseUNet(in_channels=int(cfg["in_channels"]), base_channels=int(cfg["base_channels"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    prob = tiled_predict(model, b, a, device, tile_size=args.tile_size, overlap=args.overlap)
    threshold = float(ckpt.get("threshold", 0.5))
    mask = (prob >= threshold).astype(np.uint8)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    prob_path = args.out_prefix.with_name(args.out_prefix.name + "_score.tif")
    mask_path = args.out_prefix.with_name(args.out_prefix.name + "_mask.tif")
    prob_profile = {**profile, "count": 1, "dtype": "float32", "nodata": -9999.0, "compress": "deflate"}
    mask_profile = {**profile, "count": 1, "dtype": "uint8", "nodata": 255, "compress": "deflate"}
    prob_out = np.where(valid, prob, -9999.0).astype(np.float32)
    mask_out = np.where(valid, mask, 255).astype(np.uint8)
    with rasterio.open(prob_path, "w", **prob_profile) as dst:
        dst.write(prob_out, 1)
        dst.set_band_description(1, "model_change_score_not_calibrated_probability")
    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(mask_out, 1)
        dst.set_band_description(1, f"candidate_change_mask_threshold_{threshold:.3f}")
    report = {
        "status": "INFERENCE_COMPLETED",
        "task": "OSCD-compatible 13-band optical urban-change segmentation",
        "before": str(args.before), "after": str(args.after), "checkpoint": str(args.checkpoint),
        "score_raster": str(prob_path), "mask_raster": str(mask_path),
        "threshold": threshold, "scale_factor": scale, "valid_fraction": float(valid.mean()),
        "candidate_change_fraction_of_valid": float((mask[valid] > 0).mean()),
        "crs": str(profile.get("crs")), "height": int(mask.shape[0]), "width": int(mask.shape[1]),
        "warnings": warnings + ["Output score is not a calibrated probability or legal encroachment finding.",
                                 "Validate sensor, bands, reflectance scaling, registration and AOI compatibility before interpreting results."],
    }
    report_path = args.out_prefix.with_name(args.out_prefix.name + "_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
