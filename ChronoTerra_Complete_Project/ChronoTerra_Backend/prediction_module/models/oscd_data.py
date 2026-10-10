"""OSCD file discovery, guarded loading and patch sampling (no data download)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import random
from typing import Literal

import numpy as np
import rasterio
from PIL import Image
from torch.utils.data import Dataset

BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")


@dataclass(frozen=True)
class OSCDPairFiles:
    name: str
    before_paths: tuple[Path, ...]
    after_paths: tuple[Path, ...]
    label_path: Path


@dataclass
class LoadedScene:
    name: str
    before: np.ndarray  # normalized float32 CHW
    after: np.ndarray   # normalized float32 CHW
    label: np.ndarray   # bool HW
    valid: np.ndarray   # bool HW
    scale_factor: float


def _find_named_dir(root: Path, name: str) -> Path:
    candidates = [p for p in (root, *root.rglob(name)) if p.is_dir() and p.name == name]
    if not candidates:
        raise FileNotFoundError(f"Could not find '{name}' below {root}; check your extracted OSCD folder layout")
    return min(candidates, key=lambda p: len(p.parts))


def _band_key(path: Path) -> tuple[int, str]:
    stem = path.stem.upper()
    for idx, band in enumerate(BANDS):
        if re.search(rf"(?<![A-Z0-9]){re.escape(band)}(?![A-Z0-9])", stem):
            return idx, path.name.lower()
    raise ValueError(f"Cannot identify OSCD spectral band from filename: {path.name}")


def _list_bands(folder: Path) -> tuple[Path, ...]:
    paths = tuple(sorted(folder.glob("*.tif"), key=_band_key))
    if len(paths) != 13 or tuple(_band_key(p)[0] for p in paths) != tuple(range(13)):
        raise ValueError(f"Expected exactly one TIFF for each of 13 OSCD bands in {folder}; found {[p.name for p in paths]}")
    return paths


def discover_pairs(root: str | Path, split: Literal["train", "test"]) -> list[OSCDPairFiles]:
    root = Path(root).expanduser().resolve()
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    images_root = _find_named_dir(root, "Onera Satellite Change Detection dataset - Images")
    labels_root = _find_named_dir(root, f"Onera Satellite Change Detection dataset - {split.capitalize()} Labels")
    result: list[OSCDPairFiles] = []
    for city_dir in sorted(p for p in labels_root.iterdir() if p.is_dir()):
        image_city = images_root / city_dir.name
        bdir, adir, label = image_city / "imgs_1_rect", image_city / "imgs_2_rect", city_dir / "cm" / "cm.png"
        if not bdir.is_dir() or not adir.is_dir() or not label.is_file():
            raise FileNotFoundError(f"Incomplete OSCD scene {city_dir.name}: expected before/after bands and {label}")
        result.append(OSCDPairFiles(city_dir.name, _list_bands(bdir), _list_bands(adir), label))
    if not result:
        raise FileNotFoundError(f"No {split} labelled OSCD pairs found under {labels_root}")
    return result


def sampled_raw_percentiles(pair: OSCDPairFiles) -> dict[str, float]:
    """Summarize raw values from a sample of representative bands before scaling."""
    samples: list[np.ndarray] = []
    for path in (pair.before_paths[1], pair.after_paths[1], pair.before_paths[7], pair.after_paths[7]):
        with rasterio.open(path) as src:
            oh, ow = min(512, src.height), min(512, src.width)
            arr = src.read(1, out_shape=(oh, ow), out_dtype="float32")
            valid = src.read_masks(1, out_shape=(oh, ow), resampling=rasterio.enums.Resampling.nearest) > 0
            valid &= np.isfinite(arr)
            if src.nodata is not None:
                valid &= arr != src.nodata
            vals = arr[valid]
            if vals.size:
                stride = max(1, vals.size // 200_000)
                samples.append(vals[::stride])
    if not samples:
        raise ValueError(f"Cannot infer data scale; selected bands contain no valid pixels in {pair.name}")
    values = np.concatenate(samples)
    return {"p01": float(np.nanpercentile(values, 1)),
            "p50": float(np.nanpercentile(values, 50)),
            "p99": float(np.nanpercentile(values, 99)),
            "min": float(np.nanmin(values)), "max": float(np.nanmax(values))}


def detect_scale_factor(pair: OSCDPairFiles) -> float:
    """Heuristic for the two common ranges 0..10000 DN and 0..1 reflectance.

    This is only a range heuristic, not radiometric calibration. Review the
    raw percentile audit and explicitly override if its inference is wrong.
    """
    stats = sampled_raw_percentiles(pair)
    return 10000.0 if stats["p99"] > 2.0 else 1.0



def audit_scale_factor_across_scenes(pairs: list[OSCDPairFiles]) -> tuple[float, dict[str, dict[str, float]]]:
    """Audit DN-vs-unit scale consistently on train/validation scenes only.

    Per-scene auto-normalization can create silent domain inconsistency. This function instead
    requires all development scenes to match one documented regime. It must never receive the
    official test split for scale inference/model selection.
    """
    if not pairs:
        raise ValueError("At least one development scene is required for scale audit")
    audits = {pair.name: sampled_raw_percentiles(pair) for pair in pairs}
    factors = {name: (10000.0 if stats["p99"] > 2.0 else 1.0) for name, stats in audits.items()}
    distinct = sorted(set(factors.values()))
    if len(distinct) != 1:
        detail = {name: {"inferred_scale_factor": factor, **audits[name]} for name, factor in factors.items()}
        raise ValueError("OSCD development scenes appear to mix raw-DN and unit-reflectance ranges. "
                         "Do not normalize each scene independently; inspect data and preprocess the "
                         f"dataset consistently. Per-scene audit: {detail}")
    return distinct[0], audits

def load_scene(pair: OSCDPairFiles, scale_factor: float) -> LoadedScene:
    def read_stack(paths: tuple[Path, ...]) -> tuple[np.ndarray, np.ndarray]:
        data, masks = [], []
        shapes: set[tuple[int, int]] = set()
        for path in paths:
            with rasterio.open(path) as src:
                band = src.read(1).astype(np.float32)
                band_mask = (src.read_masks(1) > 0) & np.isfinite(band)
            shapes.add(band.shape)
            data.append(band)
            masks.append(band_mask)
        if len(shapes) != 1:
            raise ValueError(f"Bands have different dimensions within {paths[0].parent}: {sorted(shapes)}")
        raw = np.stack(data, axis=0)
        valid = np.logical_and.reduce(masks) & np.all(np.isfinite(raw), axis=0)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(raw / float(scale_factor), 0.0, 1.0).astype(np.float32), valid

    before, valid_before = read_stack(pair.before_paths)
    after, valid_after = read_stack(pair.after_paths)
    if before.shape != after.shape:
        raise ValueError(f"Before/after shapes differ for {pair.name}: {before.shape} vs {after.shape}")
    with Image.open(pair.label_path) as im:
        label = np.asarray(im.convert("L")) > 0
    if label.shape != before.shape[-2:]:
        raise ValueError(f"Label shape {label.shape} != image shape {before.shape[-2:]} for {pair.name}; refusing silent resize")
    valid = valid_before & valid_after
    if not valid.any():
        raise ValueError(f"No common valid pixels for scene {pair.name}")
    return LoadedScene(pair.name, before, after, label.astype(bool), valid, float(scale_factor))


class RandomOSCDPatches(Dataset):
    """Random aligned patches; sampling around labelled changed pixels reduces class starvation."""
    def __init__(self, scenes: list[LoadedScene], patch_size: int = 128, length: int = 400,
                 positive_patch_probability: float = 0.5, augment: bool = True, seed: int = 42,
                 radiometric_augmentation: bool = True) -> None:
        if not scenes:
            raise ValueError("Training requires at least one scene")
        if patch_size < 16 or patch_size % 16 != 0:
            raise ValueError("patch_size must be >=16 and divisible by 16")
        self.scenes, self.patch_size, self.length = scenes, patch_size, length
        self.positive_patch_probability, self.augment = positive_patch_probability, augment
        self.radiometric_augmentation = bool(radiometric_augmentation)
        self.rng = random.Random(seed)
        self.positive_coords = [np.argwhere(s.label & s.valid) for s in scenes]
        self.cumulative = np.cumsum([max(1, int((s.label & s.valid).sum())) for s in scenes])

    def __len__(self) -> int:
        return self.length

    def _choose_scene(self) -> int:
        # Sample across scenes rather than favoring only the largest image.
        return self.rng.randrange(len(self.scenes))

    def __getitem__(self, index: int):
        import torch
        scene_idx = self._choose_scene()
        s = self.scenes[scene_idx]
        b, a, y, v = s.before, s.after, s.label, s.valid
        ph = pw = self.patch_size
        h, w = y.shape
        if h < ph or w < pw:
            pad_h, pad_w = max(0, ph - h), max(0, pw - w)
            b = np.pad(b, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
            a = np.pad(a, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
            y = np.pad(y, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
            v = np.pad(v, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
            h, w = y.shape

        choose_positive = self.rng.random() < self.positive_patch_probability and len(self.positive_coords[scene_idx]) > 0
        if choose_positive:
            py, px = self.positive_coords[scene_idx][self.rng.randrange(len(self.positive_coords[scene_idx]))]
            top = int(np.clip(int(py) - self.rng.randrange(ph), 0, h - ph))
            left = int(np.clip(int(px) - self.rng.randrange(pw), 0, w - pw))
        else:
            top = self.rng.randrange(h - ph + 1)
            left = self.rng.randrange(w - pw + 1)
        b = b[:, top:top + ph, left:left + pw]
        a = a[:, top:top + ph, left:left + pw]
        y = y[top:top + ph, left:left + pw]
        v = v[top:top + ph, left:left + pw]

        if self.augment:
            if self.rng.random() < 0.5:
                b, a, y, v = (np.flip(x, axis=-1) for x in (b, a, y, v))
            if self.rng.random() < 0.5:
                b, a, y, v = (np.flip(x, axis=-2) for x in (b, a, y, v))
            k = self.rng.randrange(4)
            if k:
                b, a, y, v = (np.rot90(x, k=k, axes=(-2, -1)) for x in (b, a, y, v))
            if self.rng.random() < 0.5:
                b, a = a, b
            # Mild, independent per-date band gains/offsets improve robustness to
            # acquisition/radiometric variation. This is augmentation, not a
            # replacement for atmospheric correction, QA masks or seasonal validation.
            if self.radiometric_augmentation and self.rng.random() < 0.70:
                def radiometric_jitter(arr):
                    gains = np.asarray([self.rng.uniform(0.92, 1.08) for _ in range(arr.shape[0])], dtype=np.float32)[:, None, None]
                    offsets = np.asarray([self.rng.uniform(-0.015, 0.015) for _ in range(arr.shape[0])], dtype=np.float32)[:, None, None]
                    return np.clip(arr * gains + offsets, 0.0, 1.0)
                b = radiometric_jitter(b)
                a = radiometric_jitter(a)
        return (
            torch.from_numpy(np.ascontiguousarray(b)).float(),
            torch.from_numpy(np.ascontiguousarray(a)).float(),
            torch.from_numpy(np.ascontiguousarray(y[None])).float(),
            torch.from_numpy(np.ascontiguousarray(v[None])).float(),
        )
