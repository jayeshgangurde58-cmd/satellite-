"""Train a from-scratch Siamese U-Net using OSCD train labels only.

Run from the project root:
  python -m models.train_oscd --root data/oscd --output outputs/models/oscd_siamese

The official OSCD test split is never used for model selection by this script.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.oscd_data import BANDS, LoadedScene, RandomOSCDPatches, audit_scale_factor_across_scenes, discover_pairs, load_scene, sampled_raw_percentiles
from models.siamese_unet import SiameseUNet
from models.input_profile import build_input_profile
from evaluation.calibration import fit_temperature, calibrate_binary_probabilities


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Keep runs more reproducible; exact determinism may still vary by OS/library.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def masked_bce_dice(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                    pos_weight: torch.Tensor) -> torch.Tensor:
    bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
    valid_count = valid.sum().clamp_min(1.0)
    bce = (bce_map * valid).sum() / valid_count
    probs = torch.sigmoid(logits)
    intersection = (probs * target * valid).sum()
    denom = ((probs + target) * valid).sum()
    dice_loss = 1.0 - (2.0 * intersection + 1.0) / (denom + 1.0)
    return 0.5 * bce + 0.5 * dice_loss


def _tile_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


@torch.inference_mode()
def tiled_predict(model: SiameseUNet, before: np.ndarray, after: np.ndarray, device: torch.device,
                 tile_size: int = 256, overlap: int = 64, batch_size: int = 4) -> np.ndarray:
    """Return probability HW; pads only tile edges and averages overlapping logits/probabilities."""
    if tile_size % 16 != 0:
        raise ValueError("tile_size must be divisible by 16")
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")
    _, h, w = before.shape
    starts = [(y, x) for y in _tile_starts(h, tile_size, stride) for x in _tile_starts(w, tile_size, stride)]
    out_sum = np.zeros((h, w), dtype=np.float32)
    out_count = np.zeros((h, w), dtype=np.float32)
    for offset in range(0, len(starts), batch_size):
        batch_coords = starts[offset:offset + batch_size]
        b_list, a_list, shapes = [], [], []
        for top, left in batch_coords:
            b = before[:, top:min(top + tile_size, h), left:min(left + tile_size, w)]
            a = after[:, top:min(top + tile_size, h), left:min(left + tile_size, w)]
            bh, bw = b.shape[-2:]
            if bh < tile_size or bw < tile_size:
                pad_h, pad_w = tile_size - bh, tile_size - bw
                b = np.pad(b, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
                a = np.pad(a, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
            b_list.append(b)
            a_list.append(a)
            shapes.append((bh, bw))
        bt = torch.from_numpy(np.stack(b_list)).to(device=device, dtype=torch.float32)
        at = torch.from_numpy(np.stack(a_list)).to(device=device, dtype=torch.float32)
        probs = torch.sigmoid(model(bt, at)).squeeze(1).detach().cpu().numpy()
        for (top, left), pred, (bh, bw) in zip(batch_coords, probs, shapes):
            out_sum[top:top + bh, left:left + bw] += pred[:bh, :bw]
            out_count[top:top + bh, left:left + bw] += 1.0
    return out_sum / np.maximum(out_count, 1.0)


def scene_metrics(prob: np.ndarray, target: np.ndarray, valid: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    pred = prob >= threshold
    m = valid.astype(bool)
    tp = int(np.logical_and(np.logical_and(pred, target), m).sum())
    fp = int(np.logical_and(np.logical_and(pred, ~target), m).sum())
    fn = int(np.logical_and(np.logical_and(~pred, target), m).sum())
    tn = int(np.logical_and(np.logical_and(~pred, ~target), m).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "iou": iou,
            "false_positive_rate": fpr, "valid_pixels": int(m.sum()),
            "valid_fraction": float(m.mean()), "threshold": threshold}


@torch.inference_mode()
def evaluate_scenes(model: SiameseUNet, scenes: list[LoadedScene], device: torch.device,
                    tile_size: int, overlap: int, threshold: float = 0.5, temperature: float = 1.0) -> dict[str, Any]:
    per_scene: dict[str, dict[str, Any]] = {}
    for s in scenes:
        prob_raw = tiled_predict(model, s.before, s.after, device, tile_size=tile_size, overlap=overlap)
        prob = calibrate_binary_probabilities(prob_raw, temperature) if temperature != 1.0 else prob_raw
        per_scene[s.name] = scene_metrics(prob, s.label, s.valid, threshold=threshold)
    keys = ("precision", "recall", "f1", "iou", "false_positive_rate", "valid_fraction")
    macro = {key: float(np.mean([v[key] for v in per_scene.values()])) for key in keys}
    macro_std = {key: float(np.std([v[key] for v in per_scene.values()])) for key in keys}
    totals = {key: sum(v[key] for v in per_scene.values()) for key in ("tp", "fp", "fn", "tn", "valid_pixels")}
    tp, fp, fn, tn = totals["tp"], totals["fp"], totals["fn"], totals["tn"]
    micro = {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        **totals,
    }
    return {"threshold": threshold, "temperature": float(temperature), "per_scene": per_scene, "macro_mean": macro,
            "macro_std": macro_std, "micro": micro}


@torch.inference_mode()
def tune_validation_threshold(model: SiameseUNet, scenes: list[LoadedScene], device: torch.device,
                              tile_size: int, overlap: int, temperature: float = 1.0) -> tuple[float, dict[str, Any]]:
    """Tune one global decision threshold on validation scenes only, never on test labels."""
    probabilities = {}
    for scene in scenes:
        p_raw = tiled_predict(model, scene.before, scene.after, device, tile_size=tile_size, overlap=overlap)
        probabilities[scene.name] = calibrate_binary_probabilities(p_raw, temperature) if temperature != 1.0 else p_raw
    thresholds = [round(float(x), 3) for x in np.arange(0.10, 0.951, 0.025)]
    candidates = []
    for threshold in thresholds:
        per_scene = [scene_metrics(probabilities[s.name], s.label, s.valid, threshold=threshold) for s in scenes]
        macro_f1 = float(np.mean([m["f1"] for m in per_scene]))
        macro_iou = float(np.mean([m["iou"] for m in per_scene]))
        macro_fpr = float(np.mean([m["false_positive_rate"] for m in per_scene]))
        candidates.append({"threshold": threshold, "macro_f1": macro_f1, "macro_iou": macro_iou,
                           "macro_false_positive_rate": macro_fpr})
    # Primary objective is macro-F1; a larger threshold breaks an exact tie to
    # favor fewer false positives. This choice is based only on validation data.
    best = max(candidates, key=lambda row: (row["macro_f1"], row["threshold"]))
    return float(best["threshold"]), {"selected_threshold": best["threshold"], "temperature_from_validation": float(temperature),
        "selection_rule": "maximize validation macro-F1; prefer larger threshold on exact ties",
        "candidates": candidates}


def save_checkpoint(path: Path, model: SiameseUNet, args: argparse.Namespace, scale_factor: float,
                    epoch: int, val_report: dict[str, Any], val_names: list[str]) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {"architecture": "SiameseUNet-SpectralSpatial-v2", "in_channels": 13, "base_channels": args.base_channels},
        "bands": list(BANDS),
        "scale_factor": float(scale_factor),
        "normalization": "clip(raw / scale_factor, 0, 1); verify OSCD values before use",
        "model_features": ["shared multi-scale spatial encoder", "absolute feature-difference pyramid", "explicit raw per-band spectral-difference branch"],
        "input_profile": getattr(args, "input_profile", None),
        "augmentation": {"paired_geometry": ["horizontal_flip", "vertical_flip", "90_degree_rotations", "date_swap"],
                         "radiometric_jitter_enabled": not getattr(args, "no_radiometric_augmentation", False),
                         "radiometric_jitter": ("independent per-date/per-band gain [0.92,1.08] and offset [-0.015,0.015] with clipping; robustness hypothesis requiring ablation"
                                                if not getattr(args, "no_radiometric_augmentation", False) else "disabled for controlled ablation")},
        "threshold": 0.5, "temperature": 1.0, "calibration_report": None,
        "epoch": epoch,
        "seed": args.seed,
        "val_scene_names": val_names,
        "val_macro_mean": val_report["macro_mean"],
        "val_macro_std": val_report["macro_std"],
        "val_micro": val_report["micro"],
    }, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True, help="Extracted OSCD root containing Images/Train Labels/Test Labels")
    p.add_argument("--output", type=Path, default=Path("outputs/models/oscd_siamese"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=100, help="Number of batches sampled each epoch")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--base-channels", type=int, default=16, help="Use 8 for faster CPU experiments")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--val-count", type=int, default=3, help="Scenes held out from the 14 official training scenes")
    p.add_argument("--positive-patch-probability", type=float, default=0.5)
    p.add_argument("--scale-factor", default="auto", help="auto, 1, or 10000; verify data ranges in run_config.json")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda")
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--tile-overlap", type=int, default=64)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--no-radiometric-augmentation", action="store_true", help="Disable per-date/per-band jitter for a controlled validation ablation")
    args = p.parse_args()

    if args.epochs < 1 or args.steps_per_epoch < 1 or args.batch_size < 1:
        p.error("epochs, steps-per-epoch and batch-size must be positive")
    if args.val_count < 1:
        p.error("val-count must be at least 1")
    if not (0.0 <= args.positive_patch_probability <= 1.0):
        p.error("positive-patch-probability must be between 0 and 1")
    args.output.mkdir(parents=True, exist_ok=True)
    set_seeds(args.seed)
    torch.set_num_threads(max(1, args.threads))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA was requested but is not available")

    train_files = discover_pairs(args.root, "train")
    if len(train_files) < args.val_count + 1:
        p.error(f"Need at least val-count+1 training scenes; found {len(train_files)}")
    rng = random.Random(args.seed)
    shuffled = sorted(train_files, key=lambda x: x.name)
    rng.shuffle(shuffled)
    val_files = shuffled[:args.val_count]
    fit_files = shuffled[args.val_count:]

    development_files = fit_files + val_files
    try:
        inferred_scale, per_scene_scale_audit = audit_scale_factor_across_scenes(development_files)
    except ValueError as exc:
        p.error(str(exc))
    if str(args.scale_factor).lower() == "auto":
        scale_factor = inferred_scale
        scale_source = "cross-scene development audit (train+validation only; no official test files read)"
    else:
        try:
            scale_factor = float(args.scale_factor)
        except ValueError as exc:
            raise SystemExit("--scale-factor must be 'auto', 1, or 10000") from exc
        if scale_factor not in (1.0, 10000.0):
            p.error("--scale-factor must be auto, 1 or 10000")
        if scale_factor != inferred_scale:
            p.error(f"Explicit scale factor {scale_factor:g} conflicts with the cross-scene range audit ({inferred_scale:g}). Resolve scaling consistently before training.")
        scale_source = "explicit CLI value confirmed by cross-scene development audit"
    raw_scale_audit = {"inferred_scale_factor": inferred_scale, "scenes": per_scene_scale_audit,
                       "scope": "official training split only; train/validation development scenes only; official test was not accessed"}

    print(f"Device: {device}; training scenes: {len(fit_files)}; validation scenes: {len(val_files)}; scale factor: {scale_factor:g}")
    print("Loading OSCD train split only. Official test labels are not used for training or model selection.")
    fit_scenes = [load_scene(x, scale_factor) for x in fit_files]
    val_scenes = [load_scene(x, scale_factor) for x in val_files]

    audit = []
    for s in fit_scenes + val_scenes:
        audit.append({"scene": s.name, "shape": list(s.before.shape), "valid_fraction": float(s.valid.mean()),
                      "change_fraction_of_valid": float((s.label & s.valid).sum() / max(1, s.valid.sum())),
                      "before_min": float(s.before.min()), "before_p99": float(np.percentile(s.before, 99)),
                      "after_min": float(s.after.min()), "after_p99": float(np.percentile(s.after, 99)),
                      "scale_factor": s.scale_factor})
    # Fit distribution diagnostics on model-fit scenes only; validation and official test scenes
    # are excluded from the profile to avoid information leakage into the training reference.
    args.input_profile = build_input_profile(
        [(np.concatenate([scene.before, scene.after], axis=0), scene.valid) for scene in fit_scenes],
        channel_names=[f"before_{b}" for b in BANDS] + [f"after_{b}" for b in BANDS],
        max_pixels_per_sample=5000, seed=args.seed)
    run_config = {
        "task": "OSCD binary urban-change segmentation",
        "architecture": "SiameseUNet-SpectralSpatial-v2 from scratch; no external pretrained weights",
        "seed": args.seed, "device": str(device), "scale_factor": scale_factor,
        "scale_factor_source": scale_source, "scale_detection_sample_raw_percentiles": raw_scale_audit, "bands": list(BANDS),
        "epochs_requested": args.epochs, "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size, "patch_size": args.patch_size,
        "base_channels": args.base_channels, "learning_rate": args.learning_rate,
        "radiometric_augmentation_enabled": not args.no_radiometric_augmentation,
        "val_count": args.val_count, "positive_patch_probability": args.positive_patch_probability,
        "tile_size": args.tile_size, "tile_overlap": args.tile_overlap,
        "training_scenes": [s.name for s in fit_scenes], "validation_scenes": [s.name for s in val_scenes],
        "data_audit": audit,
        "input_profile": args.input_profile,
        "warning": "OSCD contains few scenes; validation results do not guarantee generalization to arbitrary imagery.",
    }
    (args.output / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (args.output / "splits.json").write_text(json.dumps({"seed": args.seed,
        "fit_train_scenes": [s.name for s in fit_scenes], "validation_scenes": [s.name for s in val_scenes],
        "official_test_split_reserved": True}, indent=2), encoding="utf-8")

    train_dataset = RandomOSCDPatches(fit_scenes, args.patch_size,
        length=args.steps_per_epoch * args.batch_size,
        positive_patch_probability=args.positive_patch_probability, augment=True, seed=args.seed,
        radiometric_augmentation=not args.no_radiometric_augmentation)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = SiameseUNet(in_channels=13, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    positives = sum(int((s.label & s.valid).sum()) for s in fit_scenes)
    valid_pixels = sum(int(s.valid.sum()) for s in fit_scenes)
    negatives = max(1, valid_pixels - positives)
    pos_weight_value = min(5.0, max(1.0, float(np.sqrt(negatives / max(positives, 1)))))
    pos_weight = torch.tensor([pos_weight_value], device=device)
    print(f"Training change fraction (valid pixels): {positives / max(valid_pixels, 1):.6f}; BCE positive weight: {pos_weight_value:.3f}")

    history_path = args.output / "history.csv"
    best_f1 = -1.0
    best_epoch = 0
    no_improvement = 0
    rows: list[dict[str, Any]] = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_values = []
        for before, after, labels, valid in loader:
            before, after = before.to(device), after.to(device)
            labels, valid = labels.to(device), valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(before, after)
            loss = masked_bce_dice(logits, labels, valid, pos_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}; inspect input scale and masks")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_values.append(float(loss.detach().cpu()))

        model.eval()
        val_report = evaluate_scenes(model, val_scenes, device, args.tile_size, args.tile_overlap)
        val_f1 = val_report["macro_mean"]["f1"]
        row = {"epoch": epoch, "train_loss": float(np.mean(loss_values)),
               "val_macro_f1": val_f1, "val_macro_iou": val_report["macro_mean"]["iou"],
               "val_macro_precision": val_report["macro_mean"]["precision"],
               "val_macro_recall": val_report["macro_mean"]["recall"],
               "val_micro_f1": val_report["micro"]["f1"], "elapsed_seconds": time.time() - start}
        rows.append(row)
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
        (args.output / "latest_validation.json").write_text(json.dumps(val_report, indent=2), encoding="utf-8")
        save_checkpoint(args.output / "last.pt", model, args, scale_factor, epoch, val_report, [s.name for s in val_scenes])
        print(f"epoch {epoch:03d}/{args.epochs} loss={row['train_loss']:.4f} val_macro_F1={val_f1:.4f} "
              f"val_macro_IoU={row['val_macro_iou']:.4f} ({row['elapsed_seconds']:.0f}s)")

        if val_f1 > best_f1 + 1e-5:
            best_f1, best_epoch, no_improvement = val_f1, epoch, 0
            save_checkpoint(args.output / "best.pt", model, args, scale_factor, epoch, val_report, [s.name for s in val_scenes])
            (args.output / "best_validation.json").write_text(json.dumps(val_report, indent=2), encoding="utf-8")
        else:
            no_improvement += 1
        if no_improvement >= args.patience:
            print(f"Early stopping after {no_improvement} epochs without improved validation macro-F1.")
            break

    # Tune only the decision threshold using the held-out validation scenes. The official
    # test split remains untouched until the separate evaluation command is run.
    best_ckpt_path = args.output / "best.pt"
    try:
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
    model.eval()
    # Fit score temperature on a bounded, scene-balanced sample of validation pixels only.
    # These are logits reconstructed from overlap-averaged validation probabilities; calibration
    # is therefore explicitly tied to this inference aggregation procedure.
    z_samples, y_samples = [], []
    rng_cal = np.random.default_rng(args.seed)
    max_calibration_pixels_per_scene = 20_000
    for scene in val_scenes:
        raw_prob = tiled_predict(model, scene.before, scene.after, device,
                                 tile_size=args.tile_size, overlap=args.tile_overlap)
        valid_idx = np.flatnonzero(scene.valid.reshape(-1))
        if valid_idx.size == 0:
            continue
        if valid_idx.size > max_calibration_pixels_per_scene:
            valid_idx = np.sort(rng_cal.choice(valid_idx, max_calibration_pixels_per_scene, replace=False))
        pp = np.clip(raw_prob.reshape(-1)[valid_idx], 1e-6, 1.0-1e-6)
        zz = np.log(pp) - np.log1p(-pp)
        z_samples.append(zz.reshape(-1, 1))
        y_samples.append(scene.label.reshape(-1)[valid_idx].astype(np.int64))
    if not z_samples:
        raise RuntimeError("No valid validation pixels available for temperature calibration")
    calibration_report = fit_temperature(np.concatenate(z_samples, axis=0), np.concatenate(y_samples),
                                         ignore_index=255, max_samples=200_000, seed=args.seed)
    temperature = float(calibration_report["temperature"])
    selected_threshold, threshold_search = tune_validation_threshold(
        model, val_scenes, device, args.tile_size, args.tile_overlap, temperature=temperature)
    tuned_val_report = evaluate_scenes(model, val_scenes, device, args.tile_size, args.tile_overlap,
                                       threshold=selected_threshold, temperature=temperature)
    best_ckpt["temperature"] = temperature
    best_ckpt["calibration_report"] = calibration_report
    best_ckpt["threshold"] = selected_threshold
    best_ckpt["val_macro_mean"] = tuned_val_report["macro_mean"]
    best_ckpt["val_macro_std"] = tuned_val_report["macro_std"]
    best_ckpt["val_micro"] = tuned_val_report["micro"]
    best_ckpt["validation_threshold_search"] = threshold_search
    torch.save(best_ckpt, best_ckpt_path)
    (args.output / "calibration_validation.json").write_text(json.dumps(calibration_report, indent=2), encoding="utf-8")
    (args.output / "threshold_tuning_validation.json").write_text(json.dumps(threshold_search, indent=2), encoding="utf-8")
    (args.output / "best_validation.json").write_text(json.dumps(tuned_val_report, indent=2), encoding="utf-8")
    summary = {"status": "TRAINING_COMPLETED", "best_epoch_selected_at_threshold_0_5": best_epoch,
               "selection_macro_f1_at_threshold_0_5": best_f1,
               "validation_selected_threshold": selected_threshold, "validation_temperature": temperature,
               "validation_calibration_nll_before": calibration_report["metrics_before"]["negative_log_likelihood"],
               "validation_calibration_nll_after": calibration_report["metrics_after"]["negative_log_likelihood"],
               "validation_macro_f1_at_selected_threshold": tuned_val_report["macro_mean"]["f1"],
               "validation_macro_iou_at_selected_threshold": tuned_val_report["macro_mean"]["iou"],
               "checkpoint": str(best_ckpt_path), "history_csv": str(history_path),
               "official_test_evaluated": False, "elapsed_seconds": time.time() - start,
               "caveat": "Small-dataset research checkpoint. Temperature scaling and threshold are fit on validation scenes only. Measure once on the reserved OSCD test split and then on independent imagery; validation calibration does not guarantee deployment calibration."}
    (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
