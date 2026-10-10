"""Create a leak-aware manifest for xBD paired-RGB damage segmentation.

Expected labelled split folders: <root>/<split>/{images,targets}, where targets were
created from official post-disaster JSON labels by tools.prepare_xbd_targets. If a
labelled test split exists it is retained and training events that overlap it are excluded.
If no labelled test split exists, whole event groups are partitioned from labelled train
pairs into train/validation/test; unlabeled challenge images are never treated as labels.
"""
from __future__ import annotations
import argparse,csv,re,random
from pathlib import Path


def event_key(name: str) -> str:
    m=re.match(r'^(.*)_\d+_post_disaster$',name)
    return m.group(1) if m else name.split('_')[0]


def get_pairs(root: Path, split: str):
    # xBD/xView2 releases use both train/ and tier1/ for labelled development imagery.
    aliases = {"train": ("train", "tier1"), "test": ("test",), "tier3": ("tier3",), "holdout": ("holdout", "hold")}
    candidates = aliases.get(split, (split,))
    physical = next((root / name for name in candidates
                     if (root / name / 'images').exists() and (root / name / 'targets').exists()), None)
    if physical is None:
        return []
    imgdir = physical / 'images'; targetdir = physical / 'targets'
    result=[]
    for post in sorted(imgdir.glob('*_post_disaster.png')):
        stem=post.name[:-len('_post_disaster.png')]
        before=imgdir/(stem+'_pre_disaster.png')
        mask=targetdir/(stem+'_post_disaster_target.png')
        if not before.exists() or not mask.exists(): continue
        result.append({'event':event_key(post.stem),'before_path':before.relative_to(root).as_posix(),
                       'after_path':post.relative_to(root).as_posix(),'mask_path':mask.relative_to(root).as_posix(),
                       'source_split':physical.name,'pair_id':stem})
    return result


def group_split(rows, val_fraction, test_fraction, seed):
    events=sorted({r['event'] for r in rows})
    if len(events)<3:
        raise SystemExit(f'At least 3 independent event groups are required for train/validation/test; found {len(events)}: {events}')
    rng=random.Random(seed); rng.shuffle(events)
    ntest=max(1, round(len(events)*test_fraction))
    nval=max(1, round(len(events)*val_fraction))
    if ntest+nval>=len(events): ntest=1; nval=1
    test_events=set(events[:ntest]); val_events=set(events[ntest:ntest+nval])
    return [('test' if r['event'] in test_events else 'val' if r['event'] in val_events else 'train',r) for r in rows]


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--root',default='data/xbd'); ap.add_argument('--output',default='data/manifests/xbd_damage.csv')
    ap.add_argument('--val-fraction',type=float,default=.15); ap.add_argument('--test-fraction',type=float,default=.15); ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--include-tier3',action='store_true',help='Include explicitly labelled tier3 folder in development data.')
    a=ap.parse_args(); root=Path(a.root).resolve()
    if not 0.05<=a.val_fraction<=0.4 or not 0.05<=a.test_fraction<=0.4 or a.val_fraction+a.test_fraction>=0.8:
        raise SystemExit('Use val/test fractions in [0.05,0.4] whose sum is below 0.8')
    train=get_pairs(root,'train'); test=get_pairs(root,'test')
    if a.include_tier3: train += get_pairs(root,'tier3')
    if not train: raise SystemExit(f'No labelled paired samples under {root}/train/images and targets. Run tools.prepare_xbd_targets first.')
    output_rows=[]
    if test:
        test_events={r['event'] for r in test}
        train=[r for r in train if r['event'] not in test_events]
        if len({r['event'] for r in train}) < 2:
            raise SystemExit('Not enough training events remain after preventing overlap with labelled test events.')
        events=sorted({r['event'] for r in train}); rng=random.Random(a.seed); rng.shuffle(events)
        nval=max(1,min(len(events)-1,round(len(events)*a.val_fraction))); val_events=set(events[:nval])
        output_rows += [('val' if r['event'] in val_events else 'train',r) for r in train]
        output_rows += [('test',r) for r in test]
        mode='labelled_test_split_preserved'
    else:
        output_rows=group_split(train,a.val_fraction,a.test_fraction,a.seed)
        mode='event_group_holdout_from_labelled_train_data'
    rows=[]
    for split,r in output_rows:
        rows.append({'split':split,'before_path':r['before_path'],'after_path':r['after_path'],'mask_path':r['mask_path'],
                     'group_id':r['event'],'pair_id':r['pair_id']})
    groups={}
    for r in rows: groups.setdefault(r['group_id'],set()).add(r['split'])
    leaked={g:sorted(s) for g,s in groups.items() if len(s)>1}
    if leaked: raise SystemExit(f'Event/group leakage across splits found: {dict(list(leaked.items())[:8])}')
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['split','before_path','after_path','mask_path','group_id','pair_id']); w.writeheader(); w.writerows(rows)
    counts={s:sum(r['split']==s for r in rows) for s in ('train','val','test')}
    events={s:sorted({r['group_id'] for r in rows if r['split']==s}) for s in ('train','val','test')}
    print(f'Wrote {len(rows)} rows: {counts}; split_mode={mode}')
    print('Event groups by split:',events)
    print('Important: if this uses a split derived from labelled training data, describe it as an internal held-out test split, not the official challenge test set.')
if __name__=='__main__': main()
