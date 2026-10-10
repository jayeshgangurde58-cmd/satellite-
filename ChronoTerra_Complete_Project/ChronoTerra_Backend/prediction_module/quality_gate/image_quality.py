"""Descriptive image-quality checks for ChronoTerra inputs.

This records objective file/pixel statistics and optional blur/exposure proxies; it does not
claim to detect all cloud, shadow, atmospheric or geolocation problems. Thresholds are not
universal and results are advisory until calibrated against task-specific validation data.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
try:
    import rasterio
except Exception:
    rasterio = None
try:
    import cv2
except Exception:
    cv2 = None


def _stats(a: np.ndarray, valid: np.ndarray | None = None):
    v=np.isfinite(a)
    if valid is not None: v &= valid
    values=a[v]
    if not values.size:
        return {"p01":None,"p50":None,"p99":None,"range_p01_p99":None,"constant_or_invalid":True}
    p01,p50,p99=np.percentile(values,[1,50,99])
    return {"p01":float(p01),"p50":float(p50),"p99":float(p99),
            "range_p01_p99":float(p99-p01),"constant_or_invalid":bool(p99-p01 <= 1e-8)}


def inspect_quality(path: str | Path, profile: str = 'auto', max_dim: int = 512) -> dict:
    p=Path(path)
    out={"path":str(p),"status":"PASS","visual_quality_profile":profile,
         "blur_proxy":None,"brightness_proxy":None,"contrast_proxy":None,
         "cloud_shadow_status":"NOT_VERIFIED",
         "limitations":["No universal quality thresholds are applied.",
                        "Cloud/shadow detection requires QA masks or a separately validated detector.",
                        "Visual proxies do not prove geometric alignment or model suitability."]}
    if not p.exists(): return {**out,"status":"REJECT","reason":"File does not exist"}
    try:
        channels=None; desc=[]; valid=None; dtype=None; img_format=None
        if p.suffix.lower() in {'.tif','.tiff'}:
            if rasterio is None: return {**out,"status":"NOT_VERIFIED","reason":"Rasterio unavailable"}
            with rasterio.open(p) as ds:
                h,w=ds.height,ds.width; c=ds.count; dtype=list(ds.dtypes); img_format='GeoTIFF'
                ow=min(max_dim,w); oh=min(max_dim,h)
                arr=ds.read(out_shape=(c,oh,ow),out_dtype='float32')
                masks=ds.dataset_mask(out_shape=(oh,ow))>0
                valid=masks & np.all(np.isfinite(arr),axis=0)
                if ds.nodata is not None: valid &= ~np.any(arr==ds.nodata,axis=0)
                desc=[(d or '').upper() for d in ds.descriptions]
            a=np.moveaxis(arr,0,-1)
            if c==1: vis=arr[0]
            elif c==3: vis=0.2126*arr[0]+0.7152*arr[1]+0.0722*arr[2]
            elif profile=='oscd13' and c>=13:
                # OSCD/B01..B12 ordering: B04 red=index 3, B03 green=index 2, B02 blue=index 1.
                chans=np.stack([arr[3],arr[2],arr[1]],axis=-1)
                scaled=[]
                for band in range(3):
                    values=chans[:,:,band][valid] if valid.any() else chans[:,:,band].ravel()
                    lo,hi=np.percentile(values,[2,98]) if values.size else (0,1)
                    scaled.append(np.clip((chans[:,:,band]-lo)/max(float(hi-lo),1e-6),0,1))
                vis=0.2126*scaled[0]+0.7152*scaled[1]+0.0722*scaled[2]
            elif {'B04','B03','B02'}.issubset(set(desc)):
                chans=np.stack([arr[desc.index('B04')],arr[desc.index('B03')],arr[desc.index('B02')]],axis=-1)
                scaled=[]
                for band in range(3):
                    values=chans[:,:,band][valid] if valid.any() else chans[:,:,band].ravel()
                    lo,hi=np.percentile(values,[2,98]) if values.size else (0,1)
                    scaled.append(np.clip((chans[:,:,band]-lo)/max(float(hi-lo),1e-6),0,1))
                vis=0.2126*scaled[0]+0.7152*scaled[1]+0.0722*scaled[2]
            else:
                vis=None
        else:
            with Image.open(p) as im:
                img_format=im.format; w,h=im.size; mode=im.mode; im.thumbnail((max_dim,max_dim))
                ar=np.asarray(im)
            if ar.ndim==2:
                a=ar[:,:,None]; vis=ar.astype(np.float32); valid=np.isfinite(vis)
            else:
                if ar.shape[2]>=4:
                    valid=ar[:,:,3]>0
                    ar=ar[:,:,:3]
                else: valid=np.ones(ar.shape[:2],bool)
                a=ar; vis=(0.2126*ar[:,:,0]+0.7152*ar[:,:,1]+0.0722*ar[:,:,2]).astype(np.float32) if ar.shape[2]>=3 else ar[:,:,0].astype(np.float32)
            c=a.shape[2]; dtype=str(a.dtype)
        per_channel=[]
        if 'a' in locals():
            if a.ndim==2: per_channel=[_stats(a,valid)]
            else:
                for i in range(a.shape[-1]): per_channel.append(_stats(a[:,:,i],valid))
        if vis is not None:
            vv=np.asarray(vis)[valid] if isinstance(valid,np.ndarray) and valid.shape==np.asarray(vis).shape else np.asarray(vis).ravel()
            vv=vv[np.isfinite(vv)]
            if vv.size:
                out['brightness_proxy']={"p05":float(np.percentile(vv,5)),"median":float(np.percentile(vv,50)),"p95":float(np.percentile(vv,95))}
                out['contrast_proxy']={"p01_p99_range":float(np.percentile(vv,99)-np.percentile(vv,1)),
                                       "std":float(np.std(vv))}
                if cv2 is not None and min(vis.shape)>=16:
                    v=np.asarray(vis,dtype=np.float32)
                    lo,hi=np.percentile(vv,[1,99]); v=np.clip((v-lo)/max(float(hi-lo),1e-8),0,1).astype(np.float32)
                    out['blur_proxy']={"method":"variance_of_laplacian_after_robust_visual_stretch",
                                       "value":float(cv2.Laplacian(v,cv2.CV_32F).var()),
                                       "interpretation":"descriptive only; compare only within a consistent imagery/profile regime"}
            else: out['status']='REJECT'; out['reason']='No finite valid pixels to assess'
        elif c>3:
            out['blur_proxy']={"status":"NOT_VERIFIED","reason":"No declared RGB band/profile; provide profile=oscd13 only for verified OSCD band order"}
        out.update({"format":img_format,"width":int(w),"height":int(h),"channels":int(c),"dtype":dtype,
                    "sample_valid_fraction":float(valid.mean()) if isinstance(valid,np.ndarray) else None,
                    "per_channel_statistics":per_channel,"reason":"Descriptive image quality values computed; suitability thresholds remain task-specific"})
        return out
    except Exception as e:
        return {**out,"status":"REJECT","reason":f"Could not inspect image: {type(e).__name__}: {e}"}


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--image',required=True); ap.add_argument('--profile',choices=['auto','oscd13'],default='auto'); ap.add_argument('--output')
    a=ap.parse_args(); report=inspect_quality(a.image,a.profile); text=json.dumps(report,indent=2); print(text)
    if a.output:
        p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(text,encoding='utf-8')
    raise SystemExit(1 if report['status']=='REJECT' else 0)
if __name__=='__main__': main()
