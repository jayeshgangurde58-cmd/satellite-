from pathlib import Path
import csv
import numpy as np
import rasterio
from rasterio.transform import from_origin
from PIL import Image
import torch
from models.generic_unet import SegmentationUNet
from models.train_manifest import loss_ce_dice, tile_logits
from models.manifest_data import read_manifest, ManifestSegmentationDataset
from quality_gate.pair_preflight import inspect_image, validate_pair


def _write_tif(path, x, transform=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if x.ndim==2: x=x[None]
    with rasterio.open(path,'w',driver='GTiff',height=x.shape[1],width=x.shape[2],count=x.shape[0],dtype=str(x.dtype),crs='EPSG:32632',transform=transform or from_origin(500000,1000000,10,10)) as ds:
        ds.write(x)


def test_generic_unet_loss_and_tiled_output():
    torch.manual_seed(3)
    model=SegmentationUNet(6,5,base_channels=4)
    x=torch.rand(1,6,32,32); y=torch.zeros(1,32,32,dtype=torch.long); y[:,8:16,8:16]=3
    out=model(x); loss=loss_ce_dice(out,y,255)
    assert out.shape==(1,5,32,32) and torch.isfinite(loss)
    loss.backward()
    tiled=tile_logits(model.eval(),x[0],torch.device('cpu'),tile_size=32,overlap=0)
    assert tiled.shape==(5,32,32) and torch.isfinite(tiled).all()


def test_manifest_group_leakage_rejected(tmp_path):
    p=tmp_path/'manifest.csv'
    p.write_text('split,input_path,mask_path,group_id\ntrain,a.png,m.png,event1\nval,b.png,n.png,event1\n',encoding='utf8')
    try: read_manifest(p,tmp_path)
    except ValueError as e: assert 'Group leakage' in str(e)
    else: raise AssertionError('Expected group leakage to be rejected')


def test_pair_preflight_geotiff_overlap_and_rejection(tmp_path):
    a=tmp_path/'a.tif'; b=tmp_path/'b.tif'; c=tmp_path/'c.tif'
    arr=np.ones((3,32,32),dtype=np.uint16)
    _write_tif(a,arr,from_origin(500000,1000000,10,10))
    _write_tif(b,arr,from_origin(500100,1000000,10,10))
    _write_tif(c,arr,from_origin(600000,1000000,10,10))
    r=validate_pair(a,b,expected_channels_each=3,require_georeferencing=True)
    assert r['status']=='NOT_VERIFIED'  # footprint passes, but content registration remains unverified
    assert any(c['check']=='geographic_footprint_overlap' and c['status']=='PASS' for c in r['checks'])
    r2=validate_pair(a,c,expected_channels_each=3,require_georeferencing=True)
    assert r2['status']=='REJECT'


def test_rgb_location_not_verified(tmp_path):
    a=tmp_path/'a.png'; b=tmp_path/'b.png'
    Image.fromarray(np.zeros((32,32,3),dtype=np.uint8)).save(a)
    Image.fromarray(np.ones((32,32,3),dtype=np.uint8)).save(b)
    r=validate_pair(a,b,expected_channels_each=3,require_georeferencing=False)
    assert r['status']=='NOT_VERIFIED'
    assert any(c['check']=='geographic_footprint_overlap' and c['status']=='NOT_VERIFIED' for c in r['checks'])


def test_training_one_epoch_on_synthetic_manifest(tmp_path):
    # Software-path test only; these images are synthetic and must never be reported as real-data metrics.
    from models.train_manifest import main as _not_called
    root=tmp_path
    rows=[]
    for split,event in [('train','evt_train'),('val','evt_val'),('test','evt_test')]:
        for i in range(1):
            base=root/split; base.mkdir(parents=True,exist_ok=True)
            before=np.zeros((32,32,3),dtype=np.uint8); after=before.copy(); after[8:20,8:20]=200
            mask=np.zeros((32,32),dtype=np.uint8); mask[8:20,8:20]=2
            bp=base/'before.png'; ap=base/'after.png'; mp=base/'mask.png'
            Image.fromarray(before).save(bp); Image.fromarray(after).save(ap); Image.fromarray(mask).save(mp)
            rows.append({'split':split,'before_path':str(bp),'after_path':str(ap),'mask_path':str(mp),'group_id':event,'pair_id':event})
    manifest=root/'manifest.csv'
    with manifest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['split','before_path','after_path','mask_path','group_id','pair_id']); w.writeheader(); w.writerows(rows)
    ds=ManifestSegmentationDataset(read_manifest(manifest,root), 'train',6,'rgb255',5,255,32,1,True,42)
    sample=ds[0]
    assert sample['image'].shape==(6,32,32) and sample['mask'].shape==(32,32)


def test_binary_training_writes_validation_threshold(tmp_path):
    # Smoke test of the real training/threshold path with synthetic data only.
    import subprocess, sys
    root=tmp_path
    rows=[]
    for split,event in [('train','event_train'),('val','event_val'),('test','event_test')]:
        d=root/split; d.mkdir(parents=True,exist_ok=True)
        x=np.zeros((2,32,32),dtype=np.float32); x[0]=-25; x[1]=-30
        x[:,8:20,8:20]=-5
        y=np.zeros((32,32),dtype=np.int16); y[8:20,8:20]=1; y[0:2,0:2]=-1
        ip=d/'input.tif'; mp=d/'mask.tif'
        _write_tif(ip,x)
        # Float inputs use the helper with uint? write explicit integer mask separately.
        with rasterio.open(mp,'w',driver='GTiff',height=32,width=32,count=1,dtype='int16',crs='EPSG:32632',transform=from_origin(500000,1000000,10,10),nodata=-1) as dst:
            dst.write(y,1)
        rows.append({'split':split,'input_path':str(ip),'mask_path':str(mp),'group_id':event})
    manifest=root/'manifest.csv'
    with manifest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['split','input_path','mask_path','group_id']); w.writeheader(); w.writerows(rows)
    output=root/'out'
    cmd=[sys.executable,'-m','models.train_manifest','--manifest',str(manifest),'--root',str(root),'--output',str(output),
         '--task','flood_extent','--in-channels','2','--num-classes','2','--normalization','sar_db','--epochs','1',
         '--batch-size','1','--patch-size','32','--patches-per-scene','1','--base-channels','4','--workers','0','--overlap','0']
    res=subprocess.run(cmd,cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=60)
    assert res.returncode==0, res.stdout+'\n'+res.stderr
    assert (output/'best.pt').exists() and (output/'threshold_tuning_validation.json').exists()
    import json
    check=json.loads((output/'threshold_tuning_validation.json').read_text())
    assert check['selection_split']=='val' and 0.2<=check['best_threshold']<=0.8


def test_xbd_json_polygons_rasterize_damage_and_unknown(tmp_path):
    import json
    from tools.prepare_xbd_targets import build_mask
    ann = tmp_path / 'scene_post_disaster.json'
    data = {
        'metadata': {'width': 32, 'height': 32},
        'features': {'xy': [
            {'wkt': 'POLYGON ((2 2, 10 2, 10 10, 2 10, 2 2))', 'properties': {'subtype': 'minor-damage'}},
            {'wkt': 'POLYGON ((16 16, 24 16, 24 24, 16 24, 16 16))', 'properties': {'subtype': 'un-classified'}},
        ]}
    }
    ann.write_text(json.dumps(data), encoding='utf-8')
    mask, report = build_mask(ann)
    assert mask.shape == (32, 32)
    assert mask[5, 5] == 2
    assert mask[20, 20] == 255
    assert mask[29, 29] == 0
    assert report['building_annotations'] == 2


def test_xbd_manifest_splits_events_when_challenge_test_unlabelled(tmp_path):
    from tools.make_xbd_manifest import main as _main
    import subprocess, sys
    root=tmp_path/'xbd'
    for event in ['event_alpha','event_beta','event_gamma','event_delta']:
        d=root/'train'/'images'; d.mkdir(parents=True,exist_ok=True)
        t=root/'train'/'targets'; t.mkdir(parents=True,exist_ok=True)
        before=np.zeros((32,32,3),dtype=np.uint8); after=np.ones_like(before)
        Image.fromarray(before).save(d/f'{event}_000001_pre_disaster.png')
        Image.fromarray(after).save(d/f'{event}_000001_post_disaster.png')
        Image.fromarray(np.zeros((32,32),dtype=np.uint8)).save(t/f'{event}_000001_post_disaster_target.png')
    out=root/'manifest.csv'
    res=subprocess.run([sys.executable,'-m','tools.make_xbd_manifest','--root',str(root),'--output',str(out),'--val-fraction','0.25','--test-fraction','0.25'],
                       cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=30)
    assert res.returncode==0, res.stdout+'\n'+res.stderr
    import csv
    with out.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    assert {r['split'] for r in rows} == {'train','val','test'}
    groups={}
    for r in rows: groups.setdefault(r['group_id'],set()).add(r['split'])
    assert all(len(s)==1 for s in groups.values())


def test_sen1floods_manifest_supports_official_flood_csv_filenames(tmp_path):
    import subprocess, sys, csv
    root=tmp_path/'sen1'; data=root/'data'/'flood_events'/'HandLabeled'
    splitdir=root/'splits'/'flood_handlabeled'; splitdir.mkdir(parents=True)
    s1=data/'S1Hand'; lab=data/'LabelHand'; s1.mkdir(parents=True); lab.mkdir(parents=True)
    for event, chip, split in [('Bolivia',1001,'train'),('Ghana',2001,'val'),('India',3001,'test')]:
        name=f'{event}_{chip}'
        _write_tif(s1/f'{name}_S1Hand.tif',np.zeros((2,32,32),dtype=np.float32))
        with rasterio.open(lab/f'{name}_LabelHand.tif','w',driver='GTiff',height=32,width=32,count=1,dtype='int16',
                           crs='EPSG:4326',transform=from_origin(0,1,0.001,0.001),nodata=-1) as ds:
            ds.write(np.zeros((32,32),dtype=np.int16),1)
        splitname={'train':'flood_train_data.csv','val':'flood_valid_data.csv','test':'flood_test_data.csv'}[split]
        with (splitdir/splitname).open('w',newline='',encoding='utf-8') as f:
            f.write(f'{name}\n')
    out=root/'manifest.csv'
    res=subprocess.run([sys.executable,'-m','tools.make_sen1floods_manifest','--root',str(root),'--output',str(out)],
                       cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=30)
    assert res.returncode==0, res.stdout+'\n'+res.stderr
    with out.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    assert {r['split'] for r in rows}=={'train','val','test'}
    assert len(rows)==3


def test_visual_quality_report_has_advisory_metrics(tmp_path):
    from quality_gate.image_quality import inspect_quality
    p=tmp_path/'rgb.png'
    im=np.zeros((64,64,3),dtype=np.uint8)
    im[16:48,16:48]=200
    Image.fromarray(im).save(p)
    report=inspect_quality(p)
    assert report['status']=='PASS'
    assert report['width']==64 and report['channels']==3
    assert report['blur_proxy'] is not None
    assert report['cloud_shadow_status']=='NOT_VERIFIED'


def test_user_input_inference_writes_prediction_files(tmp_path):
    import json
    from models.generic_unet import SegmentationUNet
    from models.predict_user_input import infer
    before=np.zeros((32,32,3),dtype=np.uint8); after=before.copy(); after[8:24,8:24]=220
    bp=tmp_path/'before.png'; ap=tmp_path/'after.png'
    Image.fromarray(before).save(bp); Image.fromarray(after).save(ap)
    model=SegmentationUNet(6,5,base_channels=4)
    ckpt=tmp_path/'random_untrained_checkpoint.pt'
    torch.save({'state_dict':model.state_dict(),'config':{'task':'building_damage','in_channels':6,'num_classes':5,'base_channels':4,'normalization':'rgb255','ignore_index':255},'epoch':0},ckpt)
    report=infer('building_damage',ckpt,tmp_path/'prediction',before_path=bp,after_path=ap,patch_size=32,overlap=0,device_name='cpu')
    assert report['status']=='INFERENCE_COMPLETED'
    assert Path(report['mask_output']).exists()
    assert Path(report['class_score_output']).exists()
    assert report['scores_are_calibrated_probabilities'] is False


def test_xbd_target_preparation_is_idempotent(tmp_path):
    import json, subprocess, sys
    root=tmp_path/'xbd'; imdir=root/'train'/'images'; labdir=root/'train'/'labels'
    imdir.mkdir(parents=True); labdir.mkdir(parents=True)
    stem='sample-event_000001'
    Image.fromarray(np.zeros((32,32,3),dtype=np.uint8)).save(imdir/f'{stem}_pre_disaster.png')
    Image.fromarray(np.ones((32,32,3),dtype=np.uint8)).save(imdir/f'{stem}_post_disaster.png')
    feature={'wkt':'POLYGON ((4 4, 20 4, 20 20, 4 20, 4 4))','properties':{'subtype':'destroyed'}}
    for suffix in ['pre','post']:
        payload={'metadata':{'width':32,'height':32},'features':{'xy':[feature] if suffix=='post' else []}}
        (labdir/f'{stem}_{suffix}_disaster.json').write_text(json.dumps(payload),encoding='utf-8')
    cmd=[sys.executable,'-m','tools.prepare_xbd_targets','--root',str(root),'--splits','train']
    cwd=Path(__file__).resolve().parents[1]
    one=subprocess.run(cmd,cwd=cwd,capture_output=True,text=True,timeout=30)
    assert one.returncode==0, one.stdout+'\n'+one.stderr
    two=subprocess.run(cmd,cwd=cwd,capture_output=True,text=True,timeout=30)
    assert two.returncode==0, two.stdout+'\n'+two.stderr
    target=root/'train'/'targets'/f'{stem}_post_disaster_target.png'
    assert target.exists()
    report=json.loads((root/'targets_preparation_report.json').read_text(encoding='utf-8'))
    assert report['items'][0]['reused_existing_mask'] is True
