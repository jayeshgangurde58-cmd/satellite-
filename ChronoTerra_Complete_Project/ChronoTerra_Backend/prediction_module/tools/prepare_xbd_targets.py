"""Rasterize xBD post-disaster JSON building polygons into 5-class target masks.

Input per split: <root>/<split>/images/*_post_disaster.png and
<root>/<split>/labels/*_post_disaster.json.
Output per split: <root>/<split>/targets/*_post_disaster_target.png.
Class IDs: 0 background, 1 no-damage, 2 minor-damage, 3 major-damage,
4 destroyed; un-classified becomes ignore value 255, not an invented damage class.
Original files are never changed.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None
try:
    from shapely import wkt
except Exception:
    wkt = None

DAMAGE_CLASS = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 255,
}


def _polygons(geom):
    """Yield exterior coordinate arrays from Polygon or MultiPolygon-like geometry."""
    if geom.geom_type == "Polygon":
        yield np.asarray(geom.exterior.coords, dtype=np.float64)
    elif geom.geom_type == "MultiPolygon":
        for part in geom.geoms:
            yield np.asarray(part.exterior.coords, dtype=np.float64)
    elif geom.geom_type == "GeometryCollection":
        for part in geom.geoms:
            yield from _polygons(part)


def build_mask(annotation_path: str | Path) -> tuple[np.ndarray, dict]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required: pip install opencv-python-headless")
    if wkt is None:
        raise RuntimeError("Shapely is required: pip install shapely")
    p = Path(annotation_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    meta = data.get("metadata", {})
    width = int(meta.get("width", 1024)); height = int(meta.get("height", 1024))
    if not (1 <= width <= 8192 and 1 <= height <= 8192):
        raise ValueError(f"Unreasonable annotation dimensions {width}x{height} in {p}")
    feats = data.get("features", {}).get("xy", [])
    mask = np.zeros((height, width), dtype=np.uint8)
    counts = {k: 0 for k in DAMAGE_CLASS}
    unknown = []
    for feature in feats:
        props = feature.get("properties", {})
        subtype = str(props.get("subtype", "un-classified")).strip().lower()
        if subtype not in DAMAGE_CLASS:
            subtype = "un-classified"
        wkt_text = feature.get("wkt")
        if not wkt_text:
            # Some converted archives may provide a pixel-space XY coordinate list.
            coords = feature.get("geometry", {}).get("coordinates")
            if coords and feature.get("geometry", {}).get("type") == "Polygon":
                poly_parts = [np.asarray(coords[0], dtype=np.float64)]
            else:
                raise ValueError(f"Feature lacks pixel-space WKT/Polygon geometry in {p}")
        else:
            geom = wkt.loads(wkt_text)
            poly_parts = list(_polygons(geom))
        for coords in poly_parts:
            if len(coords) < 3 or not np.isfinite(coords).all():
                continue
            pts = np.rint(coords[:, :2]).astype(np.int32).reshape((-1, 1, 2))
            if subtype == "un-classified":
                unknown.append(pts)
            else:
                cv2.fillPoly(mask, [pts], int(DAMAGE_CLASS[subtype]))
        counts[subtype] += 1
    # Ignore unknown-grade building areas rather than counting them as background or no-damage.
    for pts in unknown:
        cv2.fillPoly(mask, [pts], 255)
    return mask, {"annotation_file": str(p), "width": width, "height": height,
                  "building_annotations": len(feats), "annotation_counts": counts,
                  "class_values": {"0": "background", "1": "no_damage", "2": "minor_damage",
                                   "3": "major_damage", "4": "destroyed", "255": "ignore/un_classified"}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='data/xbd')
    ap.add_argument('--splits', nargs='+', default=['train', 'test', 'tier3'], choices=['train', 'tier1', 'test', 'tier3', 'hold', 'holdout'])
    ap.add_argument('--overwrite', action='store_true', help='Regenerate already-existing masks')
    a = ap.parse_args(); root = Path(a.root).resolve()
    reports = []; missing_labels = []
    for requested_split in a.splits:
        aliases = {"train": ("train", "tier1"), "tier1": ("tier1",), "test": ("test",),
                   "tier3": ("tier3",), "holdout": ("holdout", "hold"), "hold": ("hold",)}
        split = next((name for name in aliases.get(requested_split, (requested_split,))
                      if (root / name / 'images').exists() and (root / name / 'labels').exists()), None)
        if split is None:
            continue
        images = root / split / 'images'; labels = root / split / 'labels'; outdir = root / split / 'targets'
        outdir.mkdir(parents=True, exist_ok=True)
        for post in sorted(labels.glob('*_post_disaster.json')):
            image = images / (post.stem + '.png')
            if not image.exists():
                # Annotation-only files are not training examples without matching post image.
                missing_labels.append(str(post)); continue
            out = outdir / (post.stem + '_target.png')
            if out.exists() and not a.overwrite:
                with Image.open(out) as existing:
                    mask=np.asarray(existing.convert('L'))
                with Image.open(image) as im:
                    if im.size != (mask.shape[1],mask.shape[0]):
                        raise ValueError(f"Existing mask/image dimensions differ: {out} {mask.shape[::-1]} vs {im.size}")
                data=json.loads(post.read_text(encoding='utf-8'))
                counts={k:0 for k in DAMAGE_CLASS}
                for feature in data.get('features',{}).get('xy',[]):
                    subtype=str(feature.get('properties',{}).get('subtype','un-classified')).strip().lower()
                    counts[subtype if subtype in counts else 'un-classified'] += 1
                report={"annotation_file":str(post),"width":mask.shape[1],"height":mask.shape[0],
                        "building_annotations":sum(counts.values()),"annotation_counts":counts,"split":split,
                        "image":str(image),"output_mask":str(out),"reused_existing_mask":True,
                        "ignore_pixel_fraction":float(np.mean(mask==255)),
                        "class_pixel_counts":{str(k):int(np.sum(mask==k)) for k in [0,1,2,3,4,255]}}
                reports.append(report)
                continue
            mask, report = build_mask(post)
            with Image.open(image) as im:
                if im.size != (report['width'], report['height']):
                    raise ValueError(f"Image/annotation dimension mismatch: {image} {im.size} vs {(report['width'], report['height'])}")
            Image.fromarray(mask).save(out)
            report.update({"split": split, "image": str(image), "output_mask": str(out),
                           "reused_existing_mask":False,"ignore_pixel_fraction": float(np.mean(mask == 255)),
                           "class_pixel_counts": {str(k): int(np.sum(mask == k)) for k in [0,1,2,3,4,255]}})
            reports.append(report)
    if not reports:
        raise SystemExit(f"No xBD post-disaster JSON/image pairs were converted under {root}. Expected <split>/images and <split>/labels.")
    summary = {"root": str(root), "masks_created": len(reports), "missing_matching_images": len(missing_labels),
               "by_split": {}, "warning": "Generated masks are rasterized from the source polygon labels; visually spot-check examples before training."}
    for split in a.splits:
        group = [r for r in reports if r['split'] == split]
        if group:
            summary['by_split'][split] = {"masks": len(group),
                "buildings_annotated": int(sum(r['building_annotations'] for r in group)),
                "class_pixel_counts": {str(k): int(sum(r['class_pixel_counts'][str(k)] for r in group)) for k in [0,1,2,3,4,255]}}
    out_report = root / 'targets_preparation_report.json'
    out_report.write_text(json.dumps({"summary": summary, "items": reports}, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f"Detailed report: {out_report}")

if __name__ == '__main__': main()
