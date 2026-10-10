"""Evaluate one frozen generic U-Net checkpoint on manifest's test split only."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from models.generic_unet import SegmentationUNet
from models.manifest_data import read_manifest
from models.train_manifest import evaluate_split

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',required=True); ap.add_argument('--root',default='.'); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--split',choices=['val','test'],default='test'); ap.add_argument('--patch-size',type=int,default=256); ap.add_argument('--overlap',type=int,default=32); ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args(); ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False); c=ck['config']
    rows=read_manifest(a.manifest,Path(a.root).resolve())
    if not any(r['split'].lower()==a.split for r in rows): raise SystemExit(f"Manifest has no '{a.split}' rows")
    model=SegmentationUNet(c['in_channels'],c['num_classes'],c['base_channels']); model.load_state_dict(ck['state_dict'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)
    args=argparse.Namespace(in_channels=c['in_channels'],normalization=c['normalization'],num_classes=c['num_classes'],ignore_index=c['ignore_index'],patch_size=a.patch_size,overlap=a.overlap,seed=a.seed)
    report=evaluate_split(model,rows,Path(a.root).resolve(),a.split,args,device,binary_threshold=c.get('decision_threshold') if a.split=='test' and c['num_classes']==2 else None,temperature=float(c.get('temperature',1.0)))
    report.update({'task':c['task'],'checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_epoch':ck.get('epoch'),'validation_temperature':float(c.get('temperature',1.0)),
                   'validation_calibration_report':c.get('calibration_report'),
                   'manifest_sha256':__import__('hashlib').sha256(Path(a.manifest).read_bytes()).hexdigest(),
                   'model_selection_used_evaluated_split':False if a.split=='test' else None,
                   'warning':'Test metrics are meaningful only if the manifest uses an untouched, group-separated real dataset split.'})
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
