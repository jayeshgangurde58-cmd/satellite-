"""Quality gate for image-pair analysis. Does not perform blind warping or ML inference.

Statuses are PASS, WARNING, REJECT or NOT_VERIFIED. Metadata overlap is a footprint check,
not proof of content registration. Visual-only files without georeferencing cannot prove AOI.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
try:
    import rasterio
    from rasterio.warp import transform_bounds, transform as warp_transform
    from affine import Affine
    from shapely.geometry import Polygon, box
except Exception:
    rasterio = None
    transform_bounds = None
    warp_transform = None
    Affine = None
    Polygon = None
    box = None

SUPPORTED={'.jpg','.jpeg','.png','.tif','.tiff'}


def _sample_statistics(sample: np.ndarray) -> dict[str, Any]:
    """Descriptive percentiles only; no universal blur/exposure threshold is applied."""
    values=sample[np.isfinite(sample)]
    if values.size==0: return {"sample_p02":None,"sample_p50":None,"sample_p98":None,"sample_dynamic_range_02_98":None}
    p02,p50,p98=np.percentile(values,[2,50,98])
    return {"sample_p02":float(p02),"sample_p50":float(p50),"sample_p98":float(p98),
            "sample_dynamic_range_02_98":float(p98-p02)}


def inspect_image(path: str | Path) -> dict[str, Any]:
    p=Path(path)
    if not p.exists(): return {"path":str(p),"status":"REJECT","reason":"File does not exist"}
    if p.suffix.lower() not in SUPPORTED: return {"path":str(p),"status":"REJECT","reason":f"Unsupported file extension: {p.suffix}"}
    try:
        if p.suffix.lower() in {'.tif','.tiff'}:
            if rasterio is None: return {"path":str(p),"status":"NOT_VERIFIED","reason":"Rasterio is unavailable"}
            with rasterio.open(p) as ds:
                valid_count=0; total=ds.width*ds.height
                # Windowed pass avoids reading a potentially huge raster into RAM.
                for _idx, window in ds.block_windows(1):
                    block=ds.read(window=window,out_dtype='float32')
                    masks=ds.read_masks(window=window)>0
                    valid=np.all(masks,axis=0)&np.all(np.isfinite(block),axis=0)
                    if ds.nodata is not None: valid &= ~np.any(block==ds.nodata,axis=0)
                    valid_count += int(valid.sum())
                oh=min(256,ds.height); ow=min(256,ds.width)
                sample=ds.read(out_shape=(ds.count,oh,ow),out_dtype='float32')
                stats=_sample_statistics(sample)
                tags=ds.tags()
                tag_values={str(k):str(v)[:256] for k,v in tags.items() if any(tok in k.lower() for tok in ('date','time','sensor','platform','satellite','product','orbit','mission'))}
                descriptions=[str(v) if v else None for v in ds.descriptions]
                vf=float(valid_count/total) if total else 0.0
                meta={"path":str(p),"format":"GeoTIFF","width":ds.width,"height":ds.height,"channels":ds.count,
                      "dtype":list(ds.dtypes),"crs":ds.crs.to_string() if ds.crs else None,
                      "bounds":[float(ds.bounds.left),float(ds.bounds.bottom),float(ds.bounds.right),float(ds.bounds.top)],
                      "transform":[float(v) for v in ds.transform],"resolution":[float(abs(ds.res[0])),float(abs(ds.res[1]))],
                      "nodata":ds.nodata,"band_descriptions":descriptions,"selected_metadata_tags":tag_values,
                      "valid_fraction":vf,"valid_pixels":valid_count,"total_pixels":int(total),
                      "valid_fraction_exact_window_scan":True,**stats}
                if total==0 or valid_count==0: meta.update(status='REJECT',reason='No valid pixels')
                elif not ds.crs: meta.update(status='WARNING',reason='Missing CRS: exact geographic placement cannot be verified')
                else: meta.update(status='PASS',reason='Readable georeferenced raster')
                return meta
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            w,h=im.size; channels=len(im.getbands()); mode=im.mode; fmt=im.format
            im.thumbnail((256,256))
            a=np.asarray(im)
            if a.ndim==2: finite=np.isfinite(a)
            elif a.ndim==3: finite=np.all(np.isfinite(a),axis=2)
            else: finite=np.zeros((im.height,im.width),bool)
            if a.ndim==3 and a.shape[2]==4:
                finite &= a[:,:,3]>0
            valid_fraction=float(finite.mean()) if finite.size else 0.0
            sample=np.moveaxis(a, -1, 0) if a.ndim==3 else a[None,:,:]
            stats=_sample_statistics(sample.astype(np.float32))
        return {"path":str(p),"format":fmt or p.suffix.lower().lstrip('.').upper(),"width":w,"height":h,
                "channels":channels,"mode":mode,"dtype":str(a.dtype),"crs":None,"bounds":None,"transform":None,
                "resolution":None,"nodata":None,"band_descriptions":None,"selected_metadata_tags":{},
                "valid_fraction":valid_fraction,"sample_valid_pixels":int(finite.sum()),"sampled_pixels":int(finite.size),"total_pixels":int(w*h),
                "valid_fraction_exact_window_scan":False,**stats,"status":"PASS" if finite.any() else "REJECT",
                "reason":"Readable image; geographic location is not verified from file metadata"}
    except Exception as e:
        return {"path":str(p),"status":"REJECT","reason":f"Could not decode/inspect file: {type(e).__name__}: {e}"}

def _affine_footprint(meta: dict[str, Any], target_crs: str = "EPSG:6933"):
    """Build a densified polygon from the raster's affine grid, not its axis-aligned bounds.

    Densification matters when a rotated/sheared raster is transformed into an equal-area CRS.
    The output remains a footprint approximation, not proof of content registration.
    """
    if rasterio is None or warp_transform is None or Polygon is None or Affine is None:
        return None
    crs = meta.get("crs")
    transform_values = meta.get("transform")
    width, height = meta.get("width"), meta.get("height")
    if not crs or not transform_values or width is None or height is None:
        return None
    try:
        affine = Affine(*list(transform_values)[:6])
        # Pixel-edge perimeter, sampled to approximate curved projected edges.
        steps = 32
        pixels = []
        for i in range(steps): pixels.append((width*i/steps, 0))
        for i in range(steps): pixels.append((width, height*i/steps))
        for i in range(steps): pixels.append((width*(steps-i)/steps, height))
        for i in range(steps): pixels.append((0, height*(steps-i)/steps))
        xs, ys = [], []
        for col, row in pixels:
            x, y = affine * (col, row)
            xs.append(float(x)); ys.append(float(y))
        if str(crs) == target_crs:
            tx, ty = xs, ys
        else:
            tx, ty = warp_transform(crs, target_crs, xs, ys)
        poly = Polygon(list(zip(tx, ty)))
        if not poly.is_valid: poly = poly.buffer(0)
        return poly if not poly.is_empty and poly.area > 0 else None
    except Exception:
        return None


def _footprint_overlap_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float] | None:
    """Return robust geometric overlap metrics in a common equal-area CRS.

    `fraction_of_smaller` is intersection / smaller-footprint area. IoU is also reported
    because overlap relative to the smaller footprint alone can hide a much larger AOI.
    Rotated affine grids are represented by their actual pixel-edge polygon, not a bounds box.
    """
    if rasterio is None or transform_bounds is None or Polygon is None or box is None:
        return None
    if not left.get('crs') or not right.get('crs'):
        return None
    try:
        pa = _affine_footprint(left)
        pb = _affine_footprint(right)
        if pa is None or pb is None:
            # Bounds-only fallback is retained but clearly marked by caller/report.
            crs_a, crs_b = left['crs'], right['crs']
            ba, bb = left.get('bounds'), right.get('bounds')
            if not ba or not bb: return None
            aa = transform_bounds(crs_a, 'EPSG:6933', *ba, densify_pts=41) if crs_a != 'EPSG:6933' else ba
            bb2 = transform_bounds(crs_b, 'EPSG:6933', *bb, densify_pts=41) if crs_b != 'EPSG:6933' else bb
            pa, pb = box(*aa), box(*bb2)
            method = 'densified_bounds_fallback'
        else:
            method = 'densified_affine_footprints'
        inter = pa.intersection(pb).area
        min_area = min(pa.area, pb.area)
        union = pa.union(pb).area
        return {"fraction_of_smaller": float(inter/min_area) if min_area > 0 else 0.0,
                "intersection_over_union": float(inter/union) if union > 0 else 0.0,
                "intersection_area_equal_area_m2": float(inter),
                "before_footprint_area_equal_area_m2": float(pa.area),
                "after_footprint_area_equal_area_m2": float(pb.area),
                "geometry_method": method}
    except Exception:
        return None


def _footprint_overlap_fraction(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    """Backward-compatible smaller-footprint overlap metric."""
    result = _footprint_overlap_metrics(left, right)
    return None if result is None else result["fraction_of_smaller"]


def _common_valid_grid_metrics(before_path: str | Path, after_path: str | Path) -> dict[str, Any]:
    """Exact, windowed common-valid metrics when two rasters occupy the identical pixel grid."""
    if rasterio is None:
        return {"status": "NOT_VERIFIED", "reason": "Rasterio unavailable"}
    try:
        with rasterio.open(before_path) as b, rasterio.open(after_path) as a:
            same = (b.width == a.width and b.height == a.height and b.crs == a.crs and b.transform == a.transform)
            if not same:
                return {"status": "NOT_VERIFIED", "reason": "A shared pixel grid is required for pixelwise common-valid coverage"}
            n_total = n_b = n_a = n_common = 0
            for _, win in b.block_windows(1):
                ba = b.read(window=win, out_dtype='float32')
                aa = a.read(window=win, out_dtype='float32')
                bm = (b.read_masks(window=win) > 0).all(axis=0) & np.isfinite(ba).all(axis=0)
                am = (a.read_masks(window=win) > 0).all(axis=0) & np.isfinite(aa).all(axis=0)
                if b.nodata is not None: bm &= ~np.any(ba == b.nodata, axis=0)
                if a.nodata is not None: am &= ~np.any(aa == a.nodata, axis=0)
                common = bm & am
                n_total += int(bm.size); n_b += int(bm.sum()); n_a += int(am.sum()); n_common += int(common.sum())
            union_valid = n_b + n_a - n_common
            return {"status": "PASS" if n_common else "REJECT", "grid_exact_match": True,
                    "total_pixels": n_total, "before_valid_pixels": n_b, "after_valid_pixels": n_a,
                    "common_valid_pixels": n_common,
                    "common_valid_fraction_of_grid": float(n_common/max(n_total, 1)),
                    "before_valid_also_valid_after": float(n_common/max(n_b, 1)),
                    "after_valid_also_valid_before": float(n_common/max(n_a, 1)),
                    "intersection_over_union_of_valid_masks": float(n_common/max(union_valid, 1)),
                    "method": "exact windowed scan using raster masks, finite values and declared NoData"}
    except Exception as exc:
        return {"status": "NOT_VERIFIED", "reason": f"Common-valid scan failed: {type(exc).__name__}: {exc}"}


def validate_pair(before_path: str | Path, after_path: str | Path, *, expected_channels_each: int | None=None,
                  require_georeferencing: bool=False) -> dict[str, Any]:
    before=inspect_image(before_path); after=inspect_image(after_path)
    checks=[]; hard_reject=False; warning=False; not_verified=False
    for label, img in [('before',before),('after',after)]:
        checks.append({"check":f"{label}_file_integrity","status":img['status'],"details":img.get('reason')})
        if img['status']=='REJECT': hard_reject=True
        elif img['status'] in ('WARNING','NOT_VERIFIED'): warning=True
    if hard_reject:
        return {"status":"REJECT","before":before,"after":after,"checks":checks,
                "reason":"At least one input is unreadable or unusable; no model should run."}

    same_shape=(before['width']==after['width'] and before['height']==after['height'])
    checks.append({"check":"same_pixel_dimensions","status":"PASS" if same_shape else "WARNING",
                   "details":f"before={before['width']}x{before['height']}; after={after['width']}x{after['height']}"})
    if not same_shape: warning=True
    if expected_channels_each is not None:
        for label,img in [('before',before),('after',after)]:
            good=img.get('channels')==expected_channels_each
            checks.append({"check":f"{label}_model_channel_compatibility","status":"PASS" if good else "REJECT",
                           "details":f"expected {expected_channels_each}, received {img.get('channels')}"})
            if not good: hard_reject=True

    geo_a=bool(before.get('crs') and before.get('bounds')); geo_b=bool(after.get('crs') and after.get('bounds'))
    if geo_a and geo_b:
        overlap_metrics = _footprint_overlap_metrics(before, after)
        overlap = None if overlap_metrics is None else overlap_metrics["fraction_of_smaller"]
        if overlap is None:
            checks.append({"check":"geographic_footprint_overlap","status":"NOT_VERIFIED","details":"Could not transform raster footprints into a common equal-area coordinate system"}); not_verified=True
        elif overlap<=1e-9:
            checks.append({"check":"geographic_footprint_overlap","status":"REJECT","overlap_metrics":overlap_metrics,"details":"Geospatial footprints do not overlap"}); hard_reject=True
        elif overlap<0.5:
            checks.append({"check":"geographic_footprint_overlap","status":"WARNING","overlap_metrics":overlap_metrics,"details":"Footprints overlap only partially; set task-specific AOI/common-coverage requirements"}); warning=True
        else:
            checks.append({"check":"geographic_footprint_overlap","status":"PASS","overlap_metrics":overlap_metrics,
                           "details":"Footprints overlap; this does not prove image content is registered"})
        if before.get('crs')!=after.get('crs'):
            checks.append({"check":"crs_match","status":"WARNING","details":"CRS differs; supported reprojection may be needed before comparison"}); warning=True
        else: checks.append({"check":"crs_match","status":"PASS","details":"CRS metadata match"})
        same_grid=(same_shape and before.get('transform')==after.get('transform') and before.get('crs')==after.get('crs'))
        checks.append({"check":"grid_transform_match","status":"PASS" if same_grid else "WARNING",
                       "details":"Matching grid metadata is necessary but not sufficient for content registration"})
        if not same_grid: warning=True
        if Path(before_path).suffix.lower() in {'.tif','.tiff'} and Path(after_path).suffix.lower() in {'.tif','.tiff'}:
            common_valid = _common_valid_grid_metrics(before_path, after_path)
            checks.append({"check":"common_valid_pixel_coverage", "status":common_valid.get("status", "NOT_VERIFIED"), "details":common_valid})
            if common_valid.get("status") == "REJECT": hard_reject=True
            elif common_valid.get("status") != "PASS": not_verified=True
            if common_valid.get("status") == "PASS" and common_valid.get("common_valid_fraction_of_grid", 1.0) < 0.5:
                warning=True
    else:
        checks.append({"check":"geographic_footprint_overlap","status":"NOT_VERIFIED",
                       "details":"At least one image lacks usable geospatial metadata; file dimensions cannot prove same location"})
        not_verified=True
        if require_georeferencing:
            checks.append({"check":"required_georeferencing","status":"REJECT","details":"Selected model requires georeferenced imagery"}); hard_reject=True
        else:
            checks.append({"check":"image_space_comparison","status":"WARNING","details":"Only visual/image-space matching may be possible; geographic overlap remains unverified"}); warning=True
    for label,img in [('before',before),('after',after)]:
        if img.get('valid_fraction',1.0)<0.8:
            checks.append({"check":f"{label}_valid_coverage","status":"WARNING","valid_fraction":img['valid_fraction'],
                           "details":"Large invalid/no-data fraction; a task-specific valid coverage gate is required"}); warning=True
    checks.append({"check":"content_registration","status":"NOT_VERIFIED",
                   "details":"CRS/bounds/transform metadata do not prove pixel-level alignment. Run the registration diagnostic and review its limitations."})
    not_verified=True
    checks.append({"check":"cloud_shadow_and_seasonality","status":"NOT_VERIFIED",
                   "details":"Requires reliable sensor QA masks/metadata or a separately validated detector; file integrity checks cannot establish this."})
    if hard_reject: status='REJECT'
    elif not_verified: status='NOT_VERIFIED'
    elif warning: status='WARNING'
    else: status='PASS'
    return {"status":status,"before":before,"after":after,"checks":checks,
            "reason":("Required compatibility check failed" if hard_reject else
                      "Some prerequisites remain unverified" if not_verified else
                      "Correctable or task-specific warnings remain" if warning else
                      "Basic metadata checks passed; content-level registration and model-specific conditions still need validation"),
            "not_performed":["automatic warping","content-level registration proof","ML inference","legal encroachment determination"]}

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--before',required=True); ap.add_argument('--after',required=True)
    ap.add_argument('--expected-channels-each',type=int); ap.add_argument('--require-georeferencing',action='store_true'); ap.add_argument('--output')
    a=ap.parse_args(); report=validate_pair(a.before,a.after,expected_channels_each=a.expected_channels_each,require_georeferencing=a.require_georeferencing)
    s=json.dumps(report,indent=2); print(s)
    if a.output: Path(a.output).write_text(s,encoding='utf-8')
    raise SystemExit(1 if report['status']=='REJECT' else 0)
