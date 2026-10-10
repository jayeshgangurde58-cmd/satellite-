"""Evaluate a frozen trained checkpoint on the official OSCD test split.

Run only after architecture, preprocessing, threshold and training configuration
have been frozen. Do not change settings based on this test report.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch

from models.oscd_data import discover_pairs, load_scene
from models.siamese_unet import SiameseUNet
from models.train_oscd import evaluate_scenes


def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # compatibility with older PyTorch
        return torch.load(path, map_location=device)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/models/oscd_siamese/test_metrics.json"))
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--tile-overlap", type=int, default=64)
    args = p.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but not available")
    ckpt = safe_torch_load(args.checkpoint, device)
    cfg = ckpt["model_config"]
    model = SiameseUNet(in_channels=int(cfg["in_channels"]), base_channels=int(cfg["base_channels"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    scale_factor = float(ckpt["scale_factor"])
    pairs = discover_pairs(args.root, "test")
    scenes = [load_scene(pair, scale_factor) for pair in pairs]
    threshold = float(ckpt.get("threshold", 0.5))
    temperature = float(ckpt.get("temperature", 1.0))
    report = evaluate_scenes(model, scenes, device, args.tile_size, args.tile_overlap, threshold=threshold, temperature=temperature)
    report.update({"dataset": "OSCD official test split", "checkpoint": str(args.checkpoint),
                   "checkpoint_epoch": ckpt.get("epoch"), "scale_factor": scale_factor, "validation_temperature": temperature,
                   "validation_calibration_report": ckpt.get("calibration_report"),
                   "bands": ckpt.get("bands"), "model_selection_used_test_data": False,
                   "warning": "The OSCD test result is not a guarantee of performance on other sensors, regions, RGB uploads or disaster-damage tasks."})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
