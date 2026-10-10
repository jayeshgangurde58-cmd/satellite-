"""Reader for the official OSCD directory layout described by TorchGeo.

Expected root contains:
  Onera Satellite Change Detection dataset - Images/<city>/imgs_1_rect/*.tif
  Onera Satellite Change Detection dataset - Images/<city>/imgs_2_rect/*.tif
  Onera Satellite Change Detection dataset - Train Labels/<city>/cm/cm.png
  Onera Satellite Change Detection dataset - Test Labels/<city>/cm/cm.png
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import rasterio
from PIL import Image

BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")


@dataclass(frozen=True)
class OSCDPair:
    name: str
    before_paths: tuple[Path, ...]
    after_paths: tuple[Path, ...]
    label_path: Path


def _find_named_dir(root: Path, name: str) -> Path:
    candidates = [p for p in (root, *root.rglob(name)) if p.is_dir() and p.name == name]
    if not candidates:
        raise FileNotFoundError(f"Could not find directory '{name}' under {root}")
    # Prefer the closest occurrence to root to avoid accidental nested duplicates.
    return min(candidates, key=lambda p: len(p.parts))


def _band_key(path: Path) -> tuple[int, str]:
    text = path.stem.upper()
    for idx, band in enumerate(BANDS):
        if re.search(rf"(?<![A-Z0-9]){re.escape(band)}(?![A-Z0-9])", text):
            return idx, path.name.lower()
    raise ValueError(f"Could not identify Sentinel-2 band name in {path.name}")


def _band_paths(folder: Path) -> tuple[Path, ...]:
    paths = tuple(sorted(folder.glob("*.tif"), key=_band_key))
    if len(paths) != len(BANDS):
        raise ValueError(f"Expected {len(BANDS)} OSCD band TIFFs in {folder}, found {len(paths)}")
    detected = tuple(_band_key(p)[0] for p in paths)
    if detected != tuple(range(len(BANDS))):
        raise ValueError(f"Expected each OSCD band exactly once in {folder}; found {[p.name for p in paths]}")
    return paths


def load_oscd_pairs(root: str | Path, split: str = "test") -> list[OSCDPair]:
    """Discover labelled OSCD image pairs for `train` or `test` split."""
    root = Path(root).expanduser().resolve()
    split = split.lower()
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    images_root = _find_named_dir(root, "Onera Satellite Change Detection dataset - Images")
    labels_name = f"Onera Satellite Change Detection dataset - {split.capitalize()} Labels"
    labels_root = _find_named_dir(root, labels_name)
    pairs: list[OSCDPair] = []
    for label_city in sorted(p for p in labels_root.iterdir() if p.is_dir()):
        city = label_city.name
        image_city = images_root / city
        before_dir = image_city / "imgs_1_rect"
        after_dir = image_city / "imgs_2_rect"
        label_path = label_city / "cm" / "cm.png"
        if not before_dir.is_dir() or not after_dir.is_dir() or not label_path.is_file():
            raise FileNotFoundError(
                f"Incomplete OSCD entry '{city}'. Expected {before_dir}, {after_dir}, and {label_path}"
            )
        pairs.append(OSCDPair(city, _band_paths(before_dir), _band_paths(after_dir), label_path))
    if not pairs:
        raise FileNotFoundError(f"No labelled {split} regions found under {labels_root}")
    return pairs


def read_oscd_pair(pair: OSCDPair) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return before(CHW), after(CHW), binary label(HW), valid(HW).

    Each band's raster must already have the same pixel shape within an image
    date, and both dates must agree. No silent resize/resampling is performed.
    OSCD labels encode no-change=0 and change=255 (any nonzero value is change).
    """
    def read_stack(paths: tuple[Path, ...]) -> tuple[np.ndarray, np.ndarray]:
        data, masks = [], []
        shapes = set()
        for path in paths:
            with rasterio.open(path) as src:
                arr = src.read(1)
                valid_band = src.read_masks(1) > 0
            shapes.add(arr.shape)
            data.append(arr.astype(np.float32))
            masks.append(valid_band & np.isfinite(arr))
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent band dimensions in {paths[0].parent}: {sorted(shapes)}")
        return np.stack(data, axis=0), np.logical_and.reduce(masks)

    before, valid_before = read_stack(pair.before_paths)
    after, valid_after = read_stack(pair.after_paths)
    if before.shape != after.shape:
        raise ValueError(f"OSCD before/after shape mismatch for {pair.name}: {before.shape} vs {after.shape}")
    with Image.open(pair.label_path) as image:
        label = np.asarray(image.convert("L")) > 0
    if label.shape != before.shape[1:]:
        raise ValueError(
            f"Label shape {label.shape} does not match imagery {before.shape[1:]} for {pair.name}; "
            "resample/align explicitly instead of silently resizing labels"
        )
    valid = valid_before & valid_after
    if not valid.any():
        raise ValueError(f"No valid pixels for OSCD pair {pair.name}")
    return before, after, label.astype(bool), valid
