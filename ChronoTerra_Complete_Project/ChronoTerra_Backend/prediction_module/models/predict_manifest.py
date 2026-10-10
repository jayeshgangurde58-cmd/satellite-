"""Predict one row from a manifest and save class mask + score maps.

For image pairs, the model input is the concatenation [before channels, after channels].
GeoTIFF outputs preserve the before-image profile only when its grid matches the after image.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from models.generic_unet import SegmentationUNet
from models.manifest_data import read_manifest,ManifestSegmentationDataset,read_image
from models.train_manifest import tile_logits
try:
 import rasterio
except Exception: rasterio=None

def main():
 ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--manifest',required=True); ap.add_argument('--root',default='.'); ap.add_argument('--checkpoint',required=True); ap.add_argument('--row',type=int,default=0)
 ap.add_argument('--split',default='test',choices=['train','val','test']); ap.add_argument('--output-dir',required=True); ap.add_argument('--patch-size',type=int,default=256); ap.add_argument('--overlap',type=int,default=32)
 a=ap.parse_args(); root=Path(a.root).resolve(); rows=read_manifest(a.manifest,root); splitrows=[r for r in rows if r['split'].lower()==a.split]
 if not 0<=a.row<len(splitrows): raise SystemExit(f'--row {a.row} out of range; {len(splitrows)} rows in split {a.split}')
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False); c=ck['config']; model=SegmentationUNet(c['in_channels'],c['num_classes'],c['base_channels']); model.load_state_dict(ck['state_dict'])
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)
 ds=ManifestSegmentationDataset(rows,a.split,c['in_channels'],c['normalization'],c['num_classes'],c['ignore_index'],a.patch_size,1,False,42)
 sample=ds[a.row]; logits=tile_logits(model,sample['image'],device,a.patch_size,a.overlap); temperature=float(c.get('temperature',1.0)); scores=torch.softmax(logits/max(temperature,1e-6),dim=0).numpy(); pred=scores.argmax(0).astype(np.uint8)
 if c['num_classes']==2 and c.get('decision_threshold') is not None: pred=(scores[1]>=float(c['decision_threshold'])).astype(np.uint8)
 mask_path=sample['row']['mask_path']; from models.manifest_data import read_mask
 rawmask=read_mask(mask_path); h,w=rawmask.shape; pred=pred[:h,:w]; scores=scores[:,:h,:w]
 # Do not report predictions over invalid swath/NoData pixels as valid classes.
 valid_masks=[]
 for key in ('input_path','before_path','after_path'):
  path=sample['row'].get(key)
  if path:
   _arr,_meta=read_image(path)
   vm=_meta.get('valid_mask')
   if isinstance(vm,np.ndarray): valid_masks.append(vm[:h,:w])
 if valid_masks:
  valid=np.logical_and.reduce(valid_masks); pred[~valid]=255; scores[:,~valid]=np.nan
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); stem=Path(mask_path).stem.replace('_target','')
 before_path=sample['row'].get('input_path') or sample['row'].get('before_path')
 saved_mask=out/f'{stem}_prediction.tif' if Path(before_path).suffix.lower() in {'.tif','.tiff'} else out/f'{stem}_prediction.png'
 if saved_mask.suffix=='.tif' and rasterio:
  with rasterio.open(before_path) as src:
   profile=src.profile.copy(); profile.update(count=1,dtype='uint8',compress='deflate',nodata=255)
   # Class masks for nodata are available in class scores only if source mask is not part of input;
   # output uses no nodata only when the source raster has none.
   with rasterio.open(saved_mask,'w',**profile) as dst: dst.write(pred,1)
 else: Image.fromarray(pred).save(saved_mask)
 score_path=out/f'{stem}_scores.npz'; np.savez_compressed(score_path,class_scores=scores)
 report={'task':c['task'],'model_checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_epoch':ck.get('epoch'),
         'manifest_row':a.row,'split':a.split,'group_id':sample['row'].get('group_id'),'input_paths':{k:sample['row'].get(k) for k in ('input_path','before_path','after_path') if sample['row'].get(k)},
         'mask_output':str(saved_mask.resolve()),'class_score_output':str(score_path.resolve()),'class_ids':list(range(c['num_classes'])),
         'decision_threshold':c.get('decision_threshold'),'temperature_from_validation':temperature,'calibration_report':c.get('calibration_report'),'scores_are_calibrated_probabilities':False,'scores_temperature_scaled_on_validation':bool(c.get('calibration_report') and c.get('calibration_report',{}).get('status')=='FITTED_ON_VALIDATION_SAMPLE'),'warning':'Prediction score channels are softmax outputs and are not calibrated confidence. Geographic export is preserved only for GeoTIFF inputs with compatible grids.'}
 (out/f'{stem}_prediction.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
