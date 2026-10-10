"""Legacy compatibility predictor retained for older tests/scripts.

For the current supported task-specific CLI and FastAPI routes, use
``models.predict_chronoterra``. This legacy entry point has fewer preflight checks and
should not be the public ChronoTerra prediction path. It never fabricates predictions.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from models.generic_unet import SegmentationUNet
from models.manifest_data import read_image, normalize_input
from models.train_manifest import tile_logits
try:
    import rasterio
except Exception:
    rasterio = None

TASK_CLASSES = {
    "flood_extent": {0:"not_water",1:"water"},
    "building_damage": {0:"background",1:"no_damage",2:"minor_damage",3:"major_damage",4:"destroyed"},
    "building_change": {0:"unchanged",1:"candidate_change"},
}


def _torch_load(path, device):
    try: return torch.load(path, map_location=device, weights_only=True)
    except TypeError: return torch.load(path, map_location=device)


def prepare_input(task, input_path=None, before_path=None, after_path=None, expected_channels=None, normalization=None):
    if task == "flood_extent":
        if not input_path: raise ValueError("flood_extent requires --input with a two-band Sentinel-1 VV/VH GeoTIFF")
        if Path(input_path).suffix.lower() not in {'.tif','.tiff'}: raise ValueError("Flood model accepts a georeferenced two-band Sentinel-1 GeoTIFF, not RGB JPG/PNG")
        x, meta = read_image(str(input_path))
        if x.shape[0] != expected_channels: raise ValueError(f"Model expects {expected_channels} bands, received {x.shape[0]}")
        if not meta.get('crs'): raise ValueError("Flood input must include valid CRS metadata")
        valid = meta.get('valid_mask')
        paths={"input":str(input_path)}
        output_source=str(input_path)
    else:
        if not before_path or not after_path: raise ValueError(f"{task} requires both --before and --after")
        before, mb = read_image(str(before_path)); after, ma = read_image(str(after_path))
        if before.shape[1:] != after.shape[1:]: raise ValueError(f"Image pair dimensions differ: {before.shape[1:]} vs {after.shape[1:]}. Align explicitly first.")
        if before.shape[0] != after.shape[0]: raise ValueError(f"Image pair channel counts differ: {before.shape[0]} vs {after.shape[0]}")
        if before.shape[0]*2 != expected_channels: raise ValueError(f"Model requires {expected_channels} total channels ({expected_channels//2} per image), got {before.shape[0]} per image")
        b_is_geo=Path(before_path).suffix.lower() in {'.tif','.tiff'}
        a_is_geo=Path(after_path).suffix.lower() in {'.tif','.tiff'}
        if b_is_geo != a_is_geo: raise ValueError("Do not mix georeferenced rasters with ordinary JPG/PNG in one pair")
        if b_is_geo:
            for name, lhs, rhs in [('CRS',mb.get('crs'),ma.get('crs')),('transform',mb.get('transform'),ma.get('transform')),('bounds',mb.get('bounds'),ma.get('bounds'))]:
                if lhs != rhs: raise ValueError(f"Before/after {name} mismatch. This model inference path requires a pre-aligned identical grid; it will not warp images automatically.")
            if not mb.get('crs'): raise ValueError("GeoTIFF pair must include CRS metadata")
        vm_b=mb.get('valid_mask'); vm_a=ma.get('valid_mask')
        valid=(vm_b & vm_a) if isinstance(vm_b,np.ndarray) and isinstance(vm_a,np.ndarray) else vm_b if isinstance(vm_b,np.ndarray) else vm_a if isinstance(vm_a,np.ndarray) else None
        x=np.concatenate([before,after],axis=0)
        meta={"driver":"GeoTIFF" if b_is_geo else "PIL","before":mb,"after":ma,
              "width":mb['width'],"height":mb['height']}
        if isinstance(valid,np.ndarray): meta['valid_mask']=valid
        paths={"before":str(before_path),"after":str(after_path)}
        output_source=str(before_path)
    if x.shape[0] != expected_channels:
        raise ValueError(f"Expected {expected_channels} model input channels but assembled {x.shape[0]}")
    x=normalize_input(x,normalization).astype(np.float32)
    valid=meta.get('valid_mask')
    if not isinstance(valid,np.ndarray): valid=np.all(np.isfinite(x),axis=0)
    else: valid=valid & np.all(np.isfinite(x),axis=0)
    if not valid.any(): raise ValueError("Inputs contain no common valid pixels")
    return x, valid, meta, paths, output_source


def infer(task, checkpoint, output_prefix, input_path=None, before_path=None, after_path=None, patch_size=256, overlap=32, device_name='auto'):
    checkpoint=Path(checkpoint).resolve(); output_prefix=Path(output_prefix)
    device=torch.device('cuda' if device_name=='auto' and torch.cuda.is_available() else 'cpu' if device_name=='auto' else device_name)
    if device.type=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA requested but unavailable')
    ck=_torch_load(checkpoint,device); cfg=ck.get('config',{})
    if cfg.get('task') != task: raise ValueError(f"Checkpoint task is '{cfg.get('task')}', but requested task is '{task}'")
    x,valid,meta,paths,output_source=prepare_input(task,input_path,before_path,after_path,cfg['in_channels'],cfg['normalization'])
    model=SegmentationUNet(cfg['in_channels'],cfg['num_classes'],cfg['base_channels']).to(device)
    model.load_state_dict(ck['state_dict'],strict=True); model.eval()
    logits=tile_logits(model,torch.from_numpy(x),device,tile_size=patch_size,overlap=overlap)
    scores=torch.softmax(logits,dim=0).cpu().numpy()
    pred=scores.argmax(axis=0).astype(np.uint8)
    if cfg['num_classes']==2 and ck.get('config',{}).get('decision_threshold') is not None:
        pred=(scores[1]>=float(cfg['decision_threshold'])).astype(np.uint8)
    pred[~valid]=255; scores[:,~valid]=np.nan
    output_prefix.parent.mkdir(parents=True,exist_ok=True)
    mask_path=output_prefix.with_name(output_prefix.name+'_mask.tif' if Path(output_source).suffix.lower() in {'.tif','.tiff'} else output_prefix.name+'_mask.png')
    if mask_path.suffix.lower()=='.tif':
        if rasterio is None: raise RuntimeError('Rasterio is required to save georeferenced output')
        with rasterio.open(output_source) as src:
            profile=src.profile.copy(); profile.update(count=1,dtype='uint8',nodata=255,compress='deflate')
        with rasterio.open(mask_path,'w',**profile) as dst:
            dst.write(pred,1); dst.set_band_description(1,f"{task}_predicted_class_id")
    else:
        Image.fromarray(pred,mode='L').save(mask_path)
    score_path=output_prefix.with_name(output_prefix.name+'_scores.npz'); np.savez_compressed(score_path,class_scores=scores)
    report_path=output_prefix.with_name(output_prefix.name+'_report.json')
    report={"status":"INFERENCE_COMPLETED","task":task,"input_paths":paths,"checkpoint":str(checkpoint),
            "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),"checkpoint_epoch":ck.get('epoch'),
            "class_ids":{str(k):v for k,v in TASK_CLASSES[task].items()},"input_profile":{"channels":int(x.shape[0]),"height":int(x.shape[1]),"width":int(x.shape[2]),"normalization":cfg['normalization']},
            "valid_fraction":float(valid.mean()),"mask_output":str(mask_path.resolve()),"class_score_output":str(score_path.resolve()),
            "scores_are_calibrated_probabilities":False,"decision_threshold":cfg.get('decision_threshold'),
            "geospatial_output":bool(mask_path.suffix.lower()=='.tif' and meta.get('driver')=='GeoTIFF'),
            "warnings":["Softmax scores are not calibrated confidence.","Predictions inherit the limits of the dataset and task; no legal encroachment or structural certification is produced.","Only the uploaded pair's metadata/grid is checked here; run quality_gate preflight and review the report before inference."]}
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2)); return report


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--task',choices=sorted(TASK_CLASSES),required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output-prefix',required=True)
    ap.add_argument('--input',help='Single input raster for flood_extent'); ap.add_argument('--before',help='Before image for paired tasks'); ap.add_argument('--after',help='After image for paired tasks')
    ap.add_argument('--patch-size',type=int,default=256); ap.add_argument('--overlap',type=int,default=32); ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=ap.parse_args(); infer(a.task,a.checkpoint,a.output_prefix,a.input,a.before,a.after,a.patch_size,a.overlap,a.device)
if __name__=='__main__': main()
