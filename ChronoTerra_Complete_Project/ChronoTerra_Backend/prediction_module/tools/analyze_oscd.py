"""Audit OSCD train/test scenes, masks, band shapes, ranges and valid coverage; no training."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import rasterio
from PIL import Image
from models.oscd_data import BANDS, discover_pairs, detect_scale_factor, sampled_raw_percentiles, load_scene


def band_info(paths):
    items=[]
    for expected,path in zip(BANDS, paths):
        with rasterio.open(path) as ds:
            a=ds.read(1).astype(np.float32)
            m=(ds.read_masks(1)>0)&np.isfinite(a)
            vals=a[m]
            q=np.percentile(vals,[1,50,99]) if vals.size else [None,None,None]
            items.append({"expected_band":expected,"file":str(path),"shape_hw":[ds.height,ds.width],
                          "dtype":ds.dtypes[0],"crs":ds.crs.to_string() if ds.crs else None,
                          "transform":list(ds.transform),"valid_fraction":float(m.mean()),
                          "p01":None if q[0] is None else float(q[0]),"p50":None if q[1] is None else float(q[1]),
                          "p99":None if q[2] is None else float(q[2])})
    return items


def inspect_split(root: Path, split: str, scale_factor: float):
    pairs=discover_pairs(root,split); scenes=[]; errors=[]
    for pair in pairs:
        try:
            scene=load_scene(pair,scale_factor)
            raw=sampled_raw_percentiles(pair)
            before_info=band_info(pair.before_paths); after_info=band_info(pair.after_paths)
            # Unique pixel values and proportions from the stored mask.
            with Image.open(pair.label_path) as im: y=np.asarray(im.convert('L'))
            u,c=np.unique(y,return_counts=True); label_counts={str(int(k)):int(v) for k,v in zip(u,c)}
            n_valid=int(scene.valid.sum()); n_changed=int((scene.label & scene.valid).sum())
            scenes.append({"scene":pair.name,"split":split,"before_shape_chw":list(scene.before.shape),
                           "after_shape_chw":list(scene.after.shape),"label_shape_hw":list(y.shape),
                           "common_valid_pixels":n_valid,"total_pixels":int(scene.valid.size),
                           "common_valid_fraction":float(scene.valid.mean()),"label_value_counts":label_counts,
                           "changed_fraction_of_valid":float(n_changed/max(n_valid,1)),"scale_factor_used":scale_factor,
                           "sampled_raw_percentiles":raw,"before_bands":before_info,"after_bands":after_info})
        except Exception as e:
            errors.append({"scene":pair.name,"error":f"{type(e).__name__}: {e}"})
    return {"split":split,"scene_count":len(pairs),"scenes":scenes,"errors":errors}


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--root',default='data/oscd'); ap.add_argument('--output',default='outputs/analysis/oscd_data_analysis.json')
    a=ap.parse_args(); root=Path(a.root).resolve()
    train_pairs=discover_pairs(root,'train')
    scale=detect_scale_factor(train_pairs[0])
    train=inspect_split(root,'train',scale); test=inspect_split(root,'test',scale)
    out={"dataset":"ONERA/OSCD","root":str(root),"band_order":list(BANDS),"scale_factor_inferred_from_first_train_scene":scale,
         "scale_factor_caveat":"Range heuristic only. Review band-level percentiles and override training scale if needed.",
         "train":train,"test":test,"summary":{"train_scenes":train['scene_count']-len(train['errors']),
         "test_scenes":test['scene_count']-len(test['errors']),"train_errors":len(train['errors']),"test_errors":len(test['errors'])},
         "reproducibility":{"report_sha256_excludes_self":True},
         "warning":"A data audit is not a model evaluation. Do not tune hyperparameters on the official test report."}
    outp=Path(a.output); outp.parent.mkdir(parents=True,exist_ok=True); payload=json.dumps(out,indent=2)
    outp.write_text(payload,encoding='utf-8'); print(json.dumps(out['summary'],indent=2)); print(f'Full report: {outp}')
    if train['errors'] or test['errors']: raise SystemExit(1)
if __name__=='__main__': main()
