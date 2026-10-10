"""Manifest-driven raster/pair dataset for flood and xBD damage training.

CSV required columns: split, mask_path, group_id and either input_path or before_path.
Optional after_path makes a concatenated before+after tensor. Paths are relative to --root
unless absolute. Supported data: GeoTIFF via rasterio and PNG/JPEG via Pillow.
"""
from __future__ import annotations
import csv, random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

try:
    import rasterio
except Exception:
    rasterio = None

@dataclass
class ManifestRow:
    split: str
    mask_path: str
    group_id: str
    input_path: str = ""
    before_path: str = ""
    after_path: str = ""


def read_manifest(path: str | Path, root: str | Path) -> list[dict[str, Any]]:
    path = Path(path); root = Path(root)
    if not path.exists(): raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames: raise ValueError("Manifest must have a CSV header")
        rows=[]
        for idx, raw in enumerate(reader, start=2):
            row={str(k).strip(): (v or "").strip() for k,v in raw.items() if k is not None}
            if not row.get("split") or not row.get("mask_path"):
                raise ValueError(f"Manifest row {idx}: split and mask_path are required")
            if not (row.get("input_path") or row.get("before_path")):
                raise ValueError(f"Manifest row {idx}: input_path or before_path is required")
            if row.get("before_path") and not row.get("after_path"):
                raise ValueError(f"Manifest row {idx}: before_path requires after_path")
            row["group_id"] = row.get("group_id") or Path(row.get("input_path") or row.get("before_path")).stem
            for field in ("input_path", "before_path", "after_path", "mask_path"):
                if row.get(field):
                    q=Path(row[field]); row[field]=str(q if q.is_absolute() else root/q)
            rows.append(row)
    if not rows: raise ValueError("Manifest contains no data rows")
    groups={}
    for r in rows:
        groups.setdefault(r["group_id"], set()).add(r["split"].lower())
    leaks={g: sorted(s) for g,s in groups.items() if len(s)>1}
    if leaks: raise ValueError(f"Group leakage across splits is not allowed: {dict(list(leaks.items())[:5])}")
    present={r["split"].lower() for r in rows}
    if "train" not in present or "val" not in present:
        raise ValueError(f"Manifest must include train and val rows; found {sorted(present)}")
    return rows


def read_image(path: str) -> tuple[np.ndarray, dict[str, Any]]:
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f"Image not found: {p}")
    if p.suffix.lower() in {".tif", ".tiff", ".vrt"}:
        if rasterio is None: raise RuntimeError("Rasterio is required for GeoTIFF inputs")
        with rasterio.open(p) as ds:
            a=ds.read().astype(np.float32)
            meta={"driver": ds.driver, "width": ds.width, "height": ds.height, "count": ds.count,
                  "crs": str(ds.crs) if ds.crs else None, "transform": tuple(ds.transform),
                  "bounds": tuple(ds.bounds), "nodata": ds.nodata, "dtype": ds.dtypes[0]}
            masks=ds.read_masks()>0
            valid=np.all(masks, axis=0) & np.all(np.isfinite(a), axis=0)
            if ds.nodata is not None:
                valid &= ~np.any(a == ds.nodata, axis=0)
            meta["valid_mask"] = valid
    else:
        with Image.open(p) as im:
            # Preserve alpha as validity rather than silently treating transparent pixels as data.
            has_alpha = "A" in im.getbands() or "transparency" in im.info
            if has_alpha:
                rgba = np.asarray(im.convert("RGBA"))
                alpha_valid = rgba[:, :, 3] > 0
                a = rgba[:, :, :3]
            else:
                im = im.convert("RGB") if im.mode not in {"L", "RGB"} else im
                a = np.asarray(im)
                alpha_valid = np.ones(a.shape[:2], dtype=bool) if a.ndim == 3 else np.ones(a.shape, dtype=bool)
        if a.ndim==2: a=a[None,:,:]
        else: a=np.moveaxis(a, -1, 0)
        a=a.astype(np.float32)
        meta={"driver":"PIL", "width": a.shape[2], "height": a.shape[1], "count": a.shape[0],
              "crs":None, "transform":None, "bounds":None, "nodata":None, "dtype":str(a.dtype),
              "valid_mask":np.all(np.isfinite(a), axis=0) & alpha_valid}
    if a.ndim!=3 or a.shape[0]<1: raise ValueError(f"Invalid image dimensions: {p}")
    return a, meta


def read_mask(path: str) -> np.ndarray:
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f"Mask not found: {p}")
    if p.suffix.lower() in {".tif", ".tiff", ".vrt"}:
        if rasterio is None: raise RuntimeError("Rasterio is required for GeoTIFF masks")
        with rasterio.open(p) as ds:
            a=ds.read(1)
            nodata=ds.nodata
            if nodata is not None: a=np.where(a==nodata, -1, a)
    else:
        with Image.open(p) as im: a=np.asarray(im)
        if a.ndim==3: a=a[:,:,0]
    return a.astype(np.int64)


def read_mask_metadata(path: str) -> dict[str, Any] | None:
    """Return georeferencing metadata when the annotation itself is a raster."""
    p=Path(path)
    if p.suffix.lower() not in {".tif", ".tiff", ".vrt"}:
        return None
    if rasterio is None:
        raise RuntimeError("Rasterio is required to inspect raster mask metadata")
    with rasterio.open(p) as ds:
        return {"driver": ds.driver, "width": ds.width, "height": ds.height,
                "crs": str(ds.crs) if ds.crs else None, "transform": tuple(ds.transform),
                "nodata": ds.nodata}


def _check_pair_grid(before_meta: dict[str, Any], after_meta: dict[str, Any], before_path: str, after_path: str) -> None:
    """Prevent paired raster samples with silently different geospatial grids."""
    b_is_raster = before_meta.get("driver") != "PIL"
    a_is_raster = after_meta.get("driver") != "PIL"
    if b_is_raster != a_is_raster:
        raise ValueError(f"Paired input mixes georeferenced/raster and ordinary image sources: {before_path} vs {after_path}")
    if b_is_raster and a_is_raster:
        b_crs, a_crs = before_meta.get("crs"), after_meta.get("crs")
        b_tr, a_tr = before_meta.get("transform"), after_meta.get("transform")
        if (b_crs or a_crs) and (not b_crs or not a_crs or b_crs != a_crs):
            raise ValueError(f"Paired raster CRS mismatch/missing CRS: {before_path} vs {after_path}")
        if (b_tr or a_tr) and (not b_tr or not a_tr or b_tr != a_tr):
            raise ValueError(f"Paired raster affine transforms differ: {before_path} vs {after_path}; align explicitly with provenance")


def normalize_input(x: np.ndarray, method: str) -> np.ndarray:
    if not np.all(np.isfinite(x)):
        x=np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if method=="rgb255": return np.clip(x, 0, 255)/255.0
    if method=="optical10000": return np.clip(x, 0, 10000)/10000.0
    if method=="sar_db": return (np.clip(x, -35.0, 5.0)+35.0)/40.0
    if method=="unit": return x
    if method=="auto":
        # Deliberately deterministic range-based choice; inspect run_config before use.
        p=float(np.nanpercentile(np.abs(x), 99.5)) if x.size else 0
        if p > 255: return np.clip(x, 0, 10000)/10000.0
        if p > 1.5: return np.clip(x, 0, 255)/255.0
        return x
    raise ValueError(f"Unsupported normalization method: {method}")

class ManifestSegmentationDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], split: str, input_channels: int, normalization: str,
                 num_classes: int, ignore_index: int=255, patch_size: int=256, patches_per_scene: int=8,
                 training: bool=False, seed: int=42, radiometric_augmentation: bool=True):
        self.rows=[r for r in rows if r["split"].lower()==split]
        if not self.rows: raise ValueError(f"No rows for split '{split}'")
        self.input_channels=input_channels; self.normalization=normalization; self.num_classes=num_classes
        self.ignore_index=ignore_index; self.patch_size=patch_size; self.patches_per_scene=patches_per_scene
        self.training=training; self.seed=seed; self.epoch=0; self.radiometric_augmentation=bool(radiometric_augmentation)
        self.repeat=patches_per_scene if training else 1

    def set_epoch(self, epoch: int) -> None:
        """Make random patch selection/augmentation change by epoch but stay reproducible."""
        self.epoch=int(epoch)

    def __len__(self): return len(self.rows)*self.repeat

    def _get_sample(self, idx):
        row=self.rows[idx%len(self.rows)]
        if row.get("input_path"):
            x, meta=read_image(row["input_path"])
        else:
            b, mb=read_image(row["before_path"]); a, ma=read_image(row["after_path"])
            if b.shape[1:] != a.shape[1:]: raise ValueError(f"Input spatial dimensions mismatch: {row['before_path']} vs {row['after_path']}")
            _check_pair_grid(mb, ma, row["before_path"], row["after_path"])
            x=np.concatenate([b,a], axis=0)
            valid_b=mb.get('valid_mask'); valid_a=ma.get('valid_mask')
            pair_valid=None
            if isinstance(valid_b,np.ndarray) and isinstance(valid_a,np.ndarray): pair_valid=valid_b & valid_a
            elif isinstance(valid_b,np.ndarray): pair_valid=valid_b
            elif isinstance(valid_a,np.ndarray): pair_valid=valid_a
            meta={"before":mb,"after":ma,"valid_mask":pair_valid}
        y=read_mask(row["mask_path"])
        if x.shape[1:] != y.shape: raise ValueError(f"Image/mask shape mismatch for {row['mask_path']}: input={x.shape[1:]}, mask={y.shape}")
        mask_meta=read_mask_metadata(row["mask_path"])
        if mask_meta is not None:
            image_meta = meta if "driver" in meta else None
            if image_meta is None and "before" in meta:
                # A paired raster target aligns to both dates; pair-grid check below has already run.
                image_meta = meta.get("before")
            if image_meta and image_meta.get("driver") != "PIL":
                if not image_meta.get("crs") or not mask_meta.get("crs") or image_meta.get("crs") != mask_meta.get("crs"):
                    raise ValueError(f"Raster image/target CRS cannot be verified or differs: {row['mask_path']}")
                if image_meta.get("transform") != mask_meta.get("transform"):
                    raise ValueError(f"Raster image/target transform differs: {row['mask_path']}; refusing shape-only alignment")
            elif mask_meta.get("crs") or mask_meta.get("transform"):
                raise ValueError(f"Georeferenced raster mask paired with non-georeferenced image: {row['mask_path']}")
        if x.shape[0] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {x.shape[0]} for row {row}")
        valid=np.all(np.isfinite(x), axis=0)
        vm=meta.get("valid_mask")
        if isinstance(vm, np.ndarray): valid &= vm
        y=np.where(valid, y, self.ignore_index)
        y=np.where(y == -1, self.ignore_index, y)
        y=np.where((y>=0)&(y<self.num_classes), y, self.ignore_index)
        x=normalize_input(x, self.normalization)
        return x.astype(np.float32), y.astype(np.int64), row, meta

    def __getitem__(self, index):
        rng=random.Random(self.seed+self.epoch*1_000_003+index*104729)
        x,y,row,meta=self._get_sample(index)
        h,w=y.shape; p=self.patch_size
        if self.training:
            if h<p or w<p:
                pad_h=max(0,p-h); pad_w=max(0,p-w)
                x=np.pad(x, ((0,0),(0,pad_h),(0,pad_w)), mode="reflect")
                y=np.pad(y, ((0,pad_h),(0,pad_w)), mode="constant", constant_values=self.ignore_index)
                h,w=y.shape
            top=rng.randint(0,h-p); left=rng.randint(0,w-p)
            x=x[:,top:top+p,left:left+p]; y=y[top:top+p,left:left+p]
            # Apply only geometry-preserving transforms to both input and mask.
            if rng.random()<0.5: x=x[:,:,::-1].copy(); y=y[:,::-1].copy()
            if rng.random()<0.5: x=x[:,::-1,:].copy(); y=y[::-1,:].copy()
            k=rng.randrange(4)
            if k: x=np.rot90(x,k,axes=(1,2)).copy(); y=np.rot90(y,k,axes=(0,1)).copy()
            # Small modality-aware radiometric augmentation. It is applied only to the
            # training split and must be ablated using validation scenes before claiming a gain.
            if self.radiometric_augmentation:
                if self.normalization == "sar_db" and x.shape[0] == 2:
                    for ch in range(2):
                        gain = rng.uniform(0.97, 1.03)
                        offset = rng.uniform(-0.02, 0.02)
                        x[ch] = np.clip(x[ch] * gain + offset, 0.0, 1.0)
                    # Mild sensor-noise perturbation; not a substitute for SAR calibration or speckle QA.
                    noise_rng = np.random.default_rng(rng.getrandbits(32))
                    noise = noise_rng.normal(0.0, 0.008, size=x.shape).astype(np.float32)
                    x += noise
                    x = np.clip(x, 0.0, 1.0)
                elif self.normalization in {"rgb255", "optical10000", "unit"} and x.shape[0] >= 6 and x.shape[0] % 2 == 0:
                    date_channels = x.shape[0] // 2
                    # For expected six-channel RGB or paired multispectral inputs, jitter
                    # each date independently to discourage reliance on identical radiometry.
                    for start in (0, date_channels):
                        gains = np.asarray([rng.uniform(0.92, 1.08) for _ in range(date_channels)], dtype=np.float32)[:, None, None]
                        offsets = np.asarray([rng.uniform(-0.025, 0.025) for _ in range(date_channels)], dtype=np.float32)[:, None, None]
                        x[start:start + date_channels] = np.clip(x[start:start + date_channels] * gains + offsets, 0.0, 1.0)
        else:
            # Full-scene mode; eval/predict code will tile if needed.
            if h%16 or w%16:
                ph=(16-h%16)%16; pw=(16-w%16)%16
                x=np.pad(x, ((0,0),(0,ph),(0,pw)), mode="reflect")
                y=np.pad(y, ((0,ph),(0,pw)), mode="constant", constant_values=self.ignore_index)
        return {"image":torch.from_numpy(x.copy()), "mask":torch.from_numpy(y.copy()),
                "row_id":row.get("group_id", row.get("mask_path", str(index))), "row":row}
