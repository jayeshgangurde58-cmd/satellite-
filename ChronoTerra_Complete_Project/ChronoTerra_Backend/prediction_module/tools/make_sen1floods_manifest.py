"""Create a manifest from the hand-labeled Sen1Floods11 v1.1 SAR subset.

Example root expected from gsutil sync:
  data/sen1floods11/data/flood_events/HandLabeled/{S1Hand,LabelHand}
  data/sen1floods11/splits/flood_handlabeled/{train,val,test}.csv
The provided official split CSVs are used; no random pixel/chip split is invented.
"""
from __future__ import annotations
import argparse,csv,re,json
from pathlib import Path


def norm_stem(s: str) -> str:
    p=Path(str(s).strip())
    stem=p.stem
    for suf in ('_S1Hand','_LabelHand','_S1','_Label','_QC','_S2Hand'):
        if stem.endswith(suf): stem=stem[:-len(suf)]
    return stem

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='data/sen1floods11'); ap.add_argument('--output',default='data/manifests/sen1floods11.csv')
    ap.add_argument('--s1-dir'); ap.add_argument('--labels-dir'); ap.add_argument('--splits-dir')
    a=ap.parse_args(); root=Path(a.root).resolve()
    s1dir=Path(a.s1_dir).resolve() if a.s1_dir else root/'data/flood_events/HandLabeled/S1Hand'
    labdir=Path(a.labels_dir).resolve() if a.labels_dir else root/'data/flood_events/HandLabeled/LabelHand'
    splitdir=Path(a.splits_dir).resolve() if a.splits_dir else root/'splits/flood_handlabeled'
    if not s1dir.exists() or not labdir.exists(): raise SystemExit(f'Missing S1Hand or LabelHand directories: {s1dir} ; {labdir}')
    s1s={norm_stem(p.name):p for p in s1dir.glob('*.tif')}
    labels={norm_stem(p.name):p for p in labdir.glob('*.tif')}
    keys=sorted(set(s1s)&set(labels))
    if not keys: raise SystemExit('No matched *_S1Hand.tif and *_LabelHand.tif files were found')
    split_files={}
    if not splitdir.exists(): raise SystemExit(f'Official split directory not found: {splitdir}')
    # Actual v1.1 filenames commonly use flood_train_data.csv, flood_valid_data.csv,
    # and flood_test_data.csv; also accept concise aliases used in converted datasets.
    aliases={'train':{'train'},'val':{'val','valid','validation'},'test':{'test'}}
    for p in splitdir.rglob('*'):
        if p.suffix.lower() not in {'.csv','.txt'}: continue
        lname=p.stem.lower()
        split=next((sp for sp,names in aliases.items() if lname in names),None)
        if split is None:
            # Support maintainer-style names such as flood_valid_data and flood_train_data.
            matches=[sp for sp,names in aliases.items() if any(re.search(rf'(^|[^a-z]){re.escape(n)}([^a-z]|$)',lname) for n in names)]
            if len(matches)==1: split=matches[0]
        if split is None: continue
        values=set()
        try:
            if p.suffix.lower()=='.csv':
                with p.open('r',newline='',encoding='utf-8-sig') as f:
                    for row in csv.reader(f):
                        for cell in row:
                            cell=str(cell).strip()
                            if cell:
                                values.add(norm_stem(cell))
                                # Some split tables only store the numeric chip component.
                                m=re.search(r'([A-Za-z0-9-]+_\d+)',cell)
                                if m: values.add(m.group(1))
            else:
                values.update(norm_stem(line) for line in p.read_text(encoding='utf-8-sig').splitlines() if line.strip())
        except Exception as e: raise SystemExit(f'Cannot read split file {p}: {e}')
        split_files[split]=values
    if not {'train','val','test'}.issubset(split_files):
        missing={'train','val','test'}-set(split_files)
        raise SystemExit(f'Expected official train/val/test split files, missing {sorted(missing)} under {splitdir}')
    # Match keys against complete entries; if the split files store only IDs, use numeric suffix matching.
    assignment={}
    for k in keys:
        hits=[]
        for split,values in split_files.items():
            if k in values or any(v.endswith(k) for v in values if v): hits.append(split)
            else:
                numeric=k.rsplit('_',1)[-1]
                if numeric in values: hits.append(split)
        hits=set(hits)
        if len(hits)>1: raise SystemExit(f'Chip {k} appears in multiple official splits: {sorted(hits)}')
        if len(hits)==1: assignment[k]=hits.pop()
    rows=[]
    for k in keys:
        if k not in assignment: continue
        # Official splits assign individual chips. A filename prefix such as 'Bolivia'
        # is not a unique flood-event ID, so do not use it as the leakage group.
        rows.append({'split':assignment[k], 'input_path':s1s[k].relative_to(root).as_posix(),
                     'mask_path':labels[k].relative_to(root).as_posix(), 'group_id':k,
                     'filename_prefix_audit_only':k.split('_')[0]})
    if not rows: raise SystemExit('Could not map any chip to split files. Inspect split CSV columns and pass manual CSV with split,input_path,mask_path,group_id.')
    groups={}
    for r in rows: groups.setdefault(r['group_id'],set()).add(r['split'])
    leaked={g:s for g,s in groups.items() if len(s)>1}
    if leaked: raise SystemExit(f'Chip-level leakage across official splits found (sample): {dict(list(leaked.items())[:8])}')
    prefix_splits={}
    for r in rows: prefix_splits.setdefault(r['filename_prefix_audit_only'],set()).add(r['split'])
    prefix_overlap={g:sorted(s) for g,s in prefix_splits.items() if len(s)>1}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        fields=['split','input_path','mask_path','group_id','filename_prefix_audit_only']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    counts={sp:sum(r['split']==sp for r in rows) for sp in ('train','val','test')}
    audit={'rows':len(rows),'split_counts':counts,'duplicate_chip_cross_split_groups':leaked,
           'filename_prefixes_appearing_in_multiple_splits':prefix_overlap,
           'filename_prefix_semantics':'filename token only; often country/region label, not a guaranteed event ID',
           'event_isolation':'NOT_VERIFIED from filename alone; use authoritative chip-to-event mapping before claiming event-independent generalization',
           'official_chip_split_preserved':True}
    out.with_name(out.stem+'_split_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(f'Wrote {len(rows)} rows to {out}; split counts={counts}')
    if prefix_overlap: print(f'Warning: filename prefixes appear in multiple splits (not proof of same event): {dict(list(prefix_overlap.items())[:8])}')
    print('Official chip-level splits are preserved. True event isolation is NOT_VERIFIED by filename prefix alone.')
if __name__=='__main__': main()
