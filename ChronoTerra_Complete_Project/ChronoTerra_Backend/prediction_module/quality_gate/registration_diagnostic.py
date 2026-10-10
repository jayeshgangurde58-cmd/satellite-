"""Advisory global-translation diagnostic; it never warps or marks a pair as verified.

Phase correlation assumes mostly consistent scene appearance and translation-only motion. Real
construction, clouds, crop differences and radiometric changes can bias it. Interpret alongside
geospatial metadata and stable-feature evidence, not as automatic proof of co-registration.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
try:
 import cv2
except Exception: cv2=None
try:
 import rasterio
except Exception: rasterio=None


def _load_gray(path: str | Path, max_dim=512):
 p=Path(path)
 if p.suffix.lower() in {'.tif','.tiff'}:
  if rasterio is None: raise RuntimeError('Rasterio is required to read GeoTIFF')
  with rasterio.open(p) as ds:
   oh=min(max_dim,ds.height); ow=min(max_dim,ds.width)
   a=ds.read(out_shape=(ds.count,oh,ow),out_dtype='float32')
   desc=[(d or '').upper() for d in ds.descriptions]
   valid=ds.dataset_mask(out_shape=(oh,ow))>0
   if ds.count>=4 and ('B04' in desc or ds.count==13):
    if 'B04' in desc: ri=desc.index('B04'); gi=desc.index('B03') if 'B03' in desc else min(2,ds.count-1); bi=desc.index('B02') if 'B02' in desc else min(1,ds.count-1)
    else: ri,gi,bi=3,2,1  # Only a display composite assumption; confirm OSCD ordering separately.
    chans=np.stack([a[ri],a[gi],a[bi]],axis=-1)
    # Per-band robust stretching is diagnostic only and does not alter the model inputs.
    scaled=[]
    for i in range(3):
     lo,hi=np.percentile(chans[:,:,i][valid] if valid.any() else chans[:,:,i],[2,98]); scaled.append(np.clip((chans[:,:,i]-lo)/max(hi-lo,1e-6),0,1))
    gray=.2126*scaled[0]+.7152*scaled[1]+.0722*scaled[2]
   else:
    band=a[0]; vals=band[valid] if valid.any() else band.ravel(); lo,hi=np.percentile(vals,[2,98]); gray=np.clip((band-lo)/max(hi-lo,1e-6),0,1)
   return np.nan_to_num(gray.astype(np.float32)), {"width":ds.width,"height":ds.height,"sample_width":ow,"sample_height":oh,"crs":ds.crs.to_string() if ds.crs else None}
 with Image.open(p) as im:
  w,h=im.size; im.thumbnail((max_dim,max_dim)); arr=np.asarray(im.convert('RGB')).astype(np.float32)/255.0
 gray=(.2126*arr[:,:,0]+.7152*arr[:,:,1]+.0722*arr[:,:,2]).astype(np.float32)
 return gray,{"width":w,"height":h,"sample_width":gray.shape[1],"sample_height":gray.shape[0],"crs":None}


def _robust_uint8(gray: np.ndarray) -> np.ndarray:
    """Robust display stretch for feature diagnostics only; never used as model input."""
    vals = gray[np.isfinite(gray)]
    if vals.size == 0:
        return np.zeros(gray.shape, np.uint8)
    lo, hi = np.percentile(vals, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(gray.shape, np.uint8)
    out = np.clip((gray - lo) / (hi - lo), 0, 1)
    return np.round(out * 255).astype(np.uint8)


def estimate_feature_transform(before_gray: np.ndarray, after_gray: np.ndarray) -> dict:
    """Estimate a robust similarity transform from candidate stable visual features.

    This is an advisory diagnostic only. True changed pixels, repeated structures, clouds,
    shadows, textureless scenes and perspective/relief differences can bias keypoint matches.
    The returned transform must never be applied automatically without explicit QA.
    """
    if cv2 is None:
        return {"status": "NOT_VERIFIED", "reason": "OpenCV unavailable"}
    b = _robust_uint8(np.asarray(before_gray, dtype=np.float32))
    a = _robust_uint8(np.asarray(after_gray, dtype=np.float32))
    if min(*b.shape, *a.shape) < 32:
        return {"status": "NOT_VERIFIED", "reason": "Images are too small for a meaningful feature diagnostic"}
    detector_name = "SIFT" if hasattr(cv2, "SIFT_create") else "ORB"
    detector = cv2.SIFT_create(nfeatures=3000) if detector_name == "SIFT" else cv2.ORB_create(nfeatures=3000, fastThreshold=8)
    kp_b, desc_b = detector.detectAndCompute(b, None)
    kp_a, desc_a = detector.detectAndCompute(a, None)
    base = {"status": "DIAGNOSTIC_ONLY", "detector": detector_name,
            "keypoints_before": len(kp_b or []), "keypoints_after": len(kp_a or []),
            "same_geographic_scene_proven": False,
            "warning": "Feature correspondences are evidence for review, not proof; changed regions can bias registration. No warp was applied."}
    if desc_b is None or desc_a is None or len(kp_b) < 8 or len(kp_a) < 8:
        return {**base, "correspondence_count": 0, "inlier_count": 0,
                "reason": "Insufficient visual keypoints"}
    try:
        if detector_name == "SIFT":
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        pairs = matcher.knnMatch(desc_b, desc_a, k=2)
        good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            return {**base, "correspondence_count": len(good), "inlier_count": 0,
                    "inlier_ratio": None, "reason": "Fewer than 8 ratio-filtered feature correspondences"}
        src = np.float32([kp_b[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_a[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
            maxIters=3000, confidence=0.995, refineIters=20)
        if matrix is None or inliers is None:
            return {**base, "correspondence_count": len(good), "inlier_count": 0,
                    "inlier_ratio": 0.0, "reason": "Robust transform estimation failed"}
        keep = inliers.reshape(-1).astype(bool)
        predicted = cv2.transform(src[keep], matrix).reshape(-1, 2)
        observed = dst[keep].reshape(-1, 2)
        residual = np.linalg.norm(predicted - observed, axis=1)
        inlier_count = int(keep.sum())
        linear = matrix[:, :2]
        scale = float(np.sqrt(np.sum(linear**2) / 2.0))
        rotation = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
        return {**base, "correspondence_count": len(good), "inlier_count": inlier_count,
                "inlier_ratio": float(keep.mean()), "median_reprojection_error_sample_pixels": float(np.median(residual)) if residual.size else None,
                "p95_reprojection_error_sample_pixels": float(np.percentile(residual, 95)) if residual.size else None,
                "estimated_rotation_degrees": rotation, "estimated_scale": scale,
                "estimated_translation_sample_pixels": {"x": float(matrix[0, 2]), "y": float(matrix[1, 2])},
                "affine_partial_2d_matrix": [[float(v) for v in row] for row in matrix],
                "reason": "RANSAC transform estimated; review inlier geometry and residuals before any correction"}
    except Exception as exc:
        return {**base, "status": "NOT_VERIFIED", "reason": f"Feature diagnostic failed: {type(exc).__name__}: {exc}"}


def estimate_translation(before_path, after_path, max_dim=512):
    """Run phase correlation plus robust feature registration diagnostics; never warps.

    Phase correlation is translation-only. Feature-based similarity estimates can indicate
    rotation/scale as well, but neither diagnostic is a geospatial truth source or a universal
    pass/fail threshold.
    """
    b, mb = _load_gray(before_path, max_dim)
    a, ma = _load_gray(after_path, max_dim)
    feature = estimate_feature_transform(b, a)
    report = {"status": "DIAGNOSTIC_ONLY", "before": mb, "after": ma,
              "feature_registration": feature,
              "assumptions": ["phase correlation assumes mostly consistent appearance and translation-only motion",
                              "feature registration assumes sufficient stable and distinctive scene features"],
              "warning": "Neither diagnostic proves true geographic overlap or content registration. Real change/clouds/shadows can bias both. No resampling or warping was applied."}
    if b.shape != a.shape:
        report.update({"shift_sample_pixels": None, "shift_native_pixels": None, "response": None,
                       "reason": "Sample image dimensions differ; phase correlation was not run. Review the feature diagnostic and reconcile AOI/resolution with justified preprocessing."})
        return report
    if cv2 is None:
        report.update({"shift_sample_pixels": None, "shift_native_pixels": None, "response": None,
                       "reason": "OpenCV is required for phase correlation; feature diagnostic unavailable"})
        return report
    # Remove global brightness offset and taper borders to reduce edge artifacts.
    b0 = np.nan_to_num(b - b.mean()).astype(np.float32)
    a0 = np.nan_to_num(a - a.mean()).astype(np.float32)
    win = cv2.createHanningWindow((b0.shape[1], b0.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(b0, a0, win)
    scale_x = mb['width'] / max(mb['sample_width'], 1)
    scale_y = mb['height'] / max(mb['sample_height'], 1)
    report.update({"shift_sample_pixels": {"x": float(shift[0]), "y": float(shift[1])},
                   "shift_native_pixels": {"x": float(shift[0] * scale_x), "y": float(shift[1] * scale_y)},
                   "response": float(response),
                   "reason": "Translation estimate emitted for review; no universal pass threshold is applied."})
    return report

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--before',required=True); ap.add_argument('--after',required=True); ap.add_argument('--max-dim',type=int,default=512); ap.add_argument('--output')
 a=ap.parse_args(); r=estimate_translation(a.before,a.after,a.max_dim); out=json.dumps(r,indent=2); print(out)
 if a.output: Path(a.output).write_text(out,encoding='utf-8')
