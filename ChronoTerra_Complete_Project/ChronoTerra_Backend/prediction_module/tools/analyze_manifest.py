"""Create a real dataset readiness/label analysis report from a manifest; no training is done."""
from __future__ import annotations
import argparse,csv,json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from models.manifest_data import read_manifest,read_image,read_mask

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--manifest',required=True); ap.add_argument('--root',default='.'); ap.add_argument('--num-classes',type=int,required=True); ap.add_argument('--ignore-index',type=int,default=255); ap.add_argument('--output',required=True)
    a=ap.parse_args(); root=Path(a.root).resolve(); rows=read_manifest(a.manifest,root)
    split=Counter(); channel_pairs=Counter(); class_counts=Counter(); valid=[]; scenes=[]; problems=[]; groups=defaultdict(set)
    for i,r in enumerate(rows):
        split[r['split'].lower()]+=1; groups[r['group_id']].add(r['split'].lower())
        try:
            if r.get('input_path'):
                x,meta=read_image(r['input_path'])
                valid_mask=meta.get('valid_mask')
            else:
                b,mb=read_image(r['before_path']); z,mz=read_image(r['after_path'])
                if b.shape!=z.shape: raise ValueError(f'paired image shape mismatch {b.shape} vs {z.shape}')
                x=np.concatenate([b,z],0); meta={'before':mb,'after':mz}
                vb=mb.get('valid_mask'); vz=mz.get('valid_mask')
                valid_mask=(vb & vz) if isinstance(vb,np.ndarray) and isinstance(vz,np.ndarray) else vb if isinstance(vb,np.ndarray) else vz if isinstance(vz,np.ndarray) else None
            y=read_mask(r['mask_path'])
            if x.shape[1:]!=y.shape: raise ValueError(f'image/mask dimensions differ {x.shape[1:]} vs {y.shape}')
            channel_pairs[x.shape[0]]+=1
            v=np.isfinite(x).all(axis=0)
            if isinstance(valid_mask,np.ndarray): v &= valid_mask
            valid.append(float(v.mean()))
            # Class balance used for training/evaluation excludes pixels invalid in either input.
            y_valid=np.where(v,y,a.ignore_index)
            u,c=np.unique(y,return_counts=True); vals={str(int(k)):int(n) for k,n in zip(u,c)}
            uv,cv=np.unique(y_valid,return_counts=True); valid_vals={str(int(k)):int(n) for k,n in zip(uv,cv)}
            for k,n in zip(uv,cv):
                kk=int(k)
                if kk==-1 or kk==a.ignore_index: continue
                if 0<=kk<a.num_classes: class_counts[kk]+=int(n)
                else: problems.append({'row':i+2,'issue':f'out-of-range label {kk}'})
            scenes.append({'group_id':r['group_id'],'split':r['split'],'input_channels':int(x.shape[0]),'shape':[int(x.shape[2]),int(x.shape[1])],
                           'mask_values_raw':vals,'mask_values_with_invalid_input_excluded':valid_vals,
                           'valid_input_pixels':int(v.sum()),'valid_fraction':float(v.mean())})
        except Exception as e: problems.append({'row':i+2,'group_id':r.get('group_id'),'issue':str(e)})
    group_leaks={g:sorted(s) for g,s in groups.items() if len(s)>1}
    total=sum(class_counts.values())
    out={'manifest':str(Path(a.manifest).resolve()),'manifest_sha256':hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
         'row_count':len(rows),'split_counts':dict(split),'channel_count_counts':dict(channel_pairs),
         'class_pixel_counts':{str(k):v for k,v in sorted(class_counts.items())},
         'class_pixel_fraction':{str(k):v/total for k,v in sorted(class_counts.items())} if total else {},
         'mean_valid_fraction':float(np.mean(valid)) if valid else None,'group_split_leaks':group_leaks,'problems':problems,'scenes':scenes,
         'warning':'Dataset analysis is descriptive. It does not prove spatial registration, label correctness or model accuracy.'}
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps({k:v for k,v in out.items() if k!='scenes'},indent=2))
    if problems or group_leaks: raise SystemExit(1)
if __name__=='__main__': main()
