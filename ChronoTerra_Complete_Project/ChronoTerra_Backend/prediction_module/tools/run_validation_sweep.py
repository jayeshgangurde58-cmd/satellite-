"""Small reproducible hyperparameter sweep selected by validation metrics only.

Does not inspect the official/test split to select a configuration. Each trial gets a unique
output directory and a captured log. Review per-scene variation before trusting the winner.
"""
from __future__ import annotations
import argparse, itertools, json, subprocess, sys, time
from pathlib import Path


def floats(s):
    vals=[float(x.strip()) for x in s.split(',') if x.strip()]
    if not vals or any(v<=0 for v in vals): raise ValueError('Expected comma-separated positive numbers')
    return vals

def ints(s):
    vals=[int(x.strip()) for x in s.split(',') if x.strip()]
    if not vals or any(v<4 for v in vals): raise ValueError('Expected comma-separated integers >= 4')
    return vals

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kind',choices=['oscd','manifest'],required=True); ap.add_argument('--root',default='data/oscd'); ap.add_argument('--output',required=True)
    ap.add_argument('--base-channels',default='8,16'); ap.add_argument('--learning-rates',default='0.001,0.0003')
    ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--batch-size',type=int,default=2); ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--patience',type=int,default=6); ap.add_argument('--patch-size',type=int,default=128); ap.add_argument('--steps-per-epoch',type=int,default=60)
    ap.add_argument('--val-count',type=int,default=3); ap.add_argument('--manifest'); ap.add_argument('--task',choices=['flood_extent','building_damage','building_change'])
    ap.add_argument('--in-channels',type=int); ap.add_argument('--num-classes',type=int); ap.add_argument('--normalization',choices=['rgb255','sar_db','optical10000','unit','auto'])
    ap.add_argument('--patches-per-scene',type=int,default=4); ap.add_argument('--workers',type=int,default=0); ap.add_argument('--no-radiometric-augmentation',action='store_true',help='Disable radiometric jitter for a controlled validation ablation')
    a=ap.parse_args(); root=Path(a.root).resolve(); out=Path(a.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    try: bases=ints(a.base_channels); lrs=floats(a.learning_rates)
    except ValueError as e: ap.error(str(e))
    if a.epochs<1 or a.batch_size<1: ap.error('epochs and batch-size must be positive')
    if a.kind=='manifest' and not all([a.manifest,a.task,a.in_channels,a.num_classes,a.normalization]):
        ap.error('manifest kind requires --manifest, --task, --in-channels, --num-classes and --normalization')
    if a.kind=='manifest' and a.patch_size%16: ap.error('manifest --patch-size must be divisible by 16')
    trials=[]; start=time.time(); best=None
    for idx,(bc,lr) in enumerate(itertools.product(bases,lrs),start=1):
        name=f'trial_{idx:02d}_bc{bc}_lr{lr:g}'; trial_dir=out/name; trial_dir.mkdir(parents=True,exist_ok=True)
        if a.kind=='oscd':
            cmd=[sys.executable,'-m','models.train_oscd','--root',str(root),'--output',str(trial_dir),'--epochs',str(a.epochs),
                 '--steps-per-epoch',str(a.steps_per_epoch),'--batch-size',str(a.batch_size),'--patch-size',str(a.patch_size),
                 '--base-channels',str(bc),'--learning-rate',str(lr),'--val-count',str(a.val_count),'--patience',str(a.patience),'--seed',str(a.seed)] + ([ '--no-radiometric-augmentation'] if a.no_radiometric_augmentation else [])
        else:
            cmd=[sys.executable,'-m','models.train_manifest','--manifest',str(Path(a.manifest).resolve()),'--root',str(root),'--output',str(trial_dir),
                 '--task',a.task,'--in-channels',str(a.in_channels),'--num-classes',str(a.num_classes),'--normalization',a.normalization,
                 '--epochs',str(a.epochs),'--batch-size',str(a.batch_size),'--patch-size',str(a.patch_size),'--patches-per-scene',str(a.patches_per_scene),
                 '--base-channels',str(bc),'--learning-rate',str(lr),'--patience',str(a.patience),'--seed',str(a.seed),'--workers',str(a.workers)] + ([ '--no-radiometric-augmentation'] if a.no_radiometric_augmentation else [])
        started=time.time()
        proc=subprocess.run(cmd,cwd=Path.cwd(),capture_output=True,text=True)
        log=trial_dir/'console.log'; log.write_text(proc.stdout+'\n--- STDERR ---\n'+proc.stderr,encoding='utf-8')
        entry={"trial":name,"base_channels":bc,"learning_rate":lr,"return_code":proc.returncode,
               "seconds":time.time()-started,"output_dir":str(trial_dir),"log":str(log)}
        if proc.returncode==0 and (trial_dir/'training_summary.json').exists():
            summary=json.loads((trial_dir/'training_summary.json').read_text(encoding='utf-8'))
            if a.kind=='oscd':
                value=summary.get('validation_macro_f1_at_selected_threshold'); objective='validation_macro_f1_at_selected_threshold'
            else:
                value=summary.get('best_validation_mean_foreground_iou'); objective='validation_mean_scene_foreground_iou'
            entry.update({"objective":objective,"validation_score":value,"training_summary":summary})
            if value is not None and (best is None or float(value)>float(best['validation_score'])): best=entry
        else:
            entry['error']='Trial failed or training_summary.json was not written; inspect console.log.'
        trials.append(entry)
        print(f"[{idx}/{len(bases)*len(lrs)}] {name}: return={proc.returncode}, validation={entry.get('validation_score')}")
    report={"kind":a.kind,"output":str(out),"trials":trials,"selected_trial":best,
            "selection_policy":"validation metric only; this script does not open or score the test split for selection", "radiometric_augmentation_disabled":bool(a.no_radiometric_augmentation),
            "elapsed_seconds":time.time()-start,
            "warning":"A winner among a small search grid is not proof of generalization. Keep the split fixed, report variance/failures, then evaluate the untouched test split once."}
    (out/'sweep_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({"selected_trial":best['trial'] if best else None,"validation_score":best['validation_score'] if best else None,
                      "sweep_summary":str(out/'sweep_summary.json'),"trials":len(trials)},indent=2))
    if best is None: raise SystemExit('Every sweep trial failed; inspect individual console.log files.')
if __name__=='__main__': main()
