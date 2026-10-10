"""Train a U-Net from an explicit CSV manifest (flood or building damage).

This never downloads data and never fabricates samples. Dataset splits must be real and
provided explicitly. Test rows are evaluated only by the separate evaluation command.
"""
from __future__ import annotations
import argparse, csv, json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from models.generic_unet import SegmentationUNet
from models.manifest_data import ManifestSegmentationDataset, read_manifest, read_mask
from models.input_profile import build_input_profile
from evaluation.calibration import fit_temperature


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def loss_ce_dice(logits, target, ignore_index: int, class_weights=None):
    valid=target != ignore_index
    if not bool(valid.any()): return logits.sum()*0.0
    ce=F.cross_entropy(logits, target, weight=class_weights, ignore_index=ignore_index)
    probs=F.softmax(logits, dim=1)
    classes=logits.shape[1]
    target_safe=target.masked_fill(~valid, 0)
    onehot=F.one_hot(target_safe, num_classes=classes).permute(0,3,1,2).float()
    mask=valid.unsqueeze(1).float()
    probs=probs*mask; onehot=onehot*mask
    dims=(0,2,3)
    inter=(probs*onehot).sum(dims)
    denom=probs.sum(dims)+onehot.sum(dims)
    dice=(2*inter+1e-6)/(denom+1e-6)
    # Ignore classes with no positive target in this batch except background; otherwise
    # the dice term would penalize predicting zero for absent classes.
    active=onehot.sum(dims)>0
    if classes>1: active[0]=False
    dice_loss=(1-dice[active]).mean() if bool(active.any()) else torch.zeros((), device=logits.device)
    return ce + 0.5*dice_loss


def tile_logits(model, image: torch.Tensor, device, tile_size=256, overlap=32):
    """image is [C,H,W] float tensor; returns [K,H,W] logits by weighted averaging."""
    c,h,w=image.shape
    stride=max(16, tile_size-overlap)
    ys=list(range(0, max(h-tile_size,0)+1, stride)); xs=list(range(0, max(w-tile_size,0)+1, stride))
    if not ys or ys[-1]+tile_size<h: ys.append(max(0,h-tile_size))
    if not xs or xs[-1]+tile_size<w: xs.append(max(0,w-tile_size))
    sums=None; counts=torch.zeros((1,h,w), dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        for y0 in sorted(set(ys)):
            for x0 in sorted(set(xs)):
                patch=image[:,y0:min(y0+tile_size,h),x0:min(x0+tile_size,w)]
                ph,pw=patch.shape[-2:]
                if ph<16 or pw<16: continue
                pad_h=(16-ph%16)%16; pad_w=(16-pw%16)%16
                patch=F.pad(patch, (0,pad_w,0,pad_h), mode="replicate")
                out=model(patch.unsqueeze(0).to(device))[0,:,:ph,:pw].cpu()
                if sums is None: sums=torch.zeros((out.shape[0],h,w), dtype=torch.float32)
                sums[:,y0:y0+ph,x0:x0+pw] += out
                counts[:,y0:y0+ph,x0:x0+pw] += 1
    if sums is None: raise ValueError(f"Scene too small to infer: {(h,w)}")
    return sums / counts.clamp_min(1)


def conf_counts(pred, target, num_classes, ignore_index):
    valid=target!=ignore_index
    p=pred[valid].reshape(-1).long(); y=target[valid].reshape(-1).long()
    good=(y>=0)&(y<num_classes)&(p>=0)&(p<num_classes)
    code=y[good]*num_classes+p[good]
    return torch.bincount(code, minlength=num_classes*num_classes).reshape(num_classes,num_classes).cpu().numpy().astype(np.int64)


def metrics_from_conf(conf):
    n=conf.shape[0]; result={}; ious=[]; f1s=[]
    for c in range(n):
        tp=int(conf[c,c]); fp=int(conf[:,c].sum()-tp); fn=int(conf[c,:].sum()-tp); tn=int(conf.sum()-tp-fp-fn)
        prec=tp/(tp+fp) if tp+fp else 0.0
        rec=tp/(tp+fn) if tp+fn else 0.0
        iou=tp/(tp+fp+fn) if tp+fp+fn else None
        f1=2*prec*rec/(prec+rec) if prec+rec else 0.0
        fpr=fp/(fp+tn) if fp+tn else 0.0
        result[str(c)]={"support":int(conf[c,:].sum()),"tp":tp,"fp":fp,"fn":fn,"tn":tn,
                        "precision":prec,"recall":rec,"f1":f1,"iou":iou,"false_positive_rate":fpr}
        if iou is not None and c>0: ious.append(iou)
        if (conf[c,:].sum()>0 or tp+fp>0) and c>0: f1s.append(f1)
    result_summary={"mean_foreground_iou":float(np.mean(ious)) if ious else None,
                    "macro_foreground_f1":float(np.mean(f1s)) if f1s else None,
                    "valid_pixels":int(conf.sum()),"confusion_matrix":conf.tolist()}
    return result, result_summary


def evaluate_split(model, rows, root, split, args, device, binary_threshold=None, temperature=1.0):
    ds=ManifestSegmentationDataset(rows, split, args.in_channels, args.normalization, args.num_classes,
                                   args.ignore_index, args.patch_size, 1, False, args.seed)
    per_scene=[]; pooled=np.zeros((args.num_classes,args.num_classes), dtype=np.int64)
    for i in range(len(ds)):
        sample=ds[i]; x=sample["image"]; y=sample["mask"].numpy()
        logits=tile_logits(model,x,device,args.patch_size,args.overlap)
        scaled_logits = logits / max(float(temperature), 1e-6)
        pred=scaled_logits.argmax(0).numpy() if binary_threshold is None or args.num_classes!=2 else (torch.softmax(scaled_logits,dim=0)[1].numpy()>=binary_threshold).astype(np.uint8)
        # Use the dataset's mask, which already excludes invalid pixels from either paired input.
        raw_y=read_mask(sample["row"]["mask_path"])
        h,w=raw_y.shape; pred=pred[:h,:w]; yy=y[:h,:w].copy()
        yy[yy==-1]=args.ignore_index
        yy[(yy<0)|(yy>=args.num_classes)]=args.ignore_index
        cm=conf_counts(torch.from_numpy(pred),torch.from_numpy(yy),args.num_classes,args.ignore_index)
        cls, summary=metrics_from_conf(cm); pooled += cm
        per_scene.append({"group_id":sample["row_id"],"mask_path":sample["row"]["mask_path"],"per_class":cls,**summary})
    pooled_cls, pooled_summary=metrics_from_conf(pooled)
    defined=[s["mean_foreground_iou"] for s in per_scene if s.get("mean_foreground_iou") is not None]
    report={"split":split,"temperature":float(temperature),"num_scenes":len(per_scene),"per_scene":per_scene,"pooled_per_class":pooled_cls,
            "pooled":pooled_summary,"scene_mean_foreground_iou":float(np.mean(defined)) if defined else None,
            "scene_std_foreground_iou":float(np.std(defined)) if defined else None}
    return report




def collect_validation_logits(model, rows, root, args, device, *, max_pixels_per_scene=12000):
    """Collect bounded, per-scene-balanced logits/labels from validation only."""
    ds=ManifestSegmentationDataset(rows,"val",args.in_channels,args.normalization,args.num_classes,
                                   args.ignore_index,args.patch_size,1,False,args.seed)
    rng=np.random.default_rng(args.seed); logits_parts=[]; labels_parts=[]; scene_counts=[]
    model.eval()
    for i in range(len(ds)):
        sample=ds[i]
        logits=tile_logits(model,sample["image"],device,args.patch_size,args.overlap).numpy()
        y=sample["mask"].numpy()
        raw_y=read_mask(sample["row"]["mask_path"]); h,w=raw_y.shape
        logits=logits[:,:h,:w]; y=y[:h,:w]
        valid=(y>=0)&(y<args.num_classes)&(y!=args.ignore_index)
        idx=np.flatnonzero(valid.reshape(-1))
        if idx.size==0:
            scene_counts.append({"group_id":sample["row_id"],"valid_pixels":0,"sampled_pixels":0})
            continue
        if idx.size>max_pixels_per_scene:
            idx=np.sort(rng.choice(idx,max_pixels_per_scene,replace=False))
        z=np.moveaxis(logits,0,-1).reshape(-1,args.num_classes)[idx]
        yy=y.reshape(-1)[idx].astype(np.int64)
        finite=np.all(np.isfinite(z),axis=1)
        z,yy=z[finite],yy[finite]
        if z.size:
            logits_parts.append(z.astype(np.float32)); labels_parts.append(yy)
        scene_counts.append({"group_id":sample["row_id"],"valid_pixels":int(valid.sum()),"sampled_pixels":int(len(yy))})
    if not logits_parts:
        raise RuntimeError("No finite labelled validation pixels available for calibration")
    return np.concatenate(logits_parts,axis=0),np.concatenate(labels_parts),scene_counts

def tune_binary_threshold(model, rows, root, args, device, out_dir, temperature=1.0):
    """Select a binary threshold using validation only; never reads test labels."""
    ds=ManifestSegmentationDataset(rows,"val",args.in_channels,args.normalization,args.num_classes,args.ignore_index,args.patch_size,1,False,args.seed)
    thresholds=np.round(np.arange(0.20,0.801,0.025),3)
    per_t={float(t):[] for t in thresholds}
    model.eval()
    for i in range(len(ds)):
        sample=ds[i]; logits=tile_logits(model,sample["image"],device,args.patch_size,args.overlap)
        probability=torch.softmax(logits/max(float(temperature),1e-6),dim=0)[1].numpy()
        raw_y=read_mask(sample["row"]["mask_path"]); h,w=raw_y.shape
        probability=probability[:h,:w]; y=sample["mask"].numpy()[:h,:w].copy(); y[y==-1]=args.ignore_index
        y[(y<0)|(y>=2)]=args.ignore_index
        yt=torch.from_numpy(y)
        for t in thresholds:
            p=torch.from_numpy((probability>=float(t)).astype(np.uint8))
            cm=conf_counts(p,yt,2,args.ignore_index); _,summ=metrics_from_conf(cm)
            per_t[float(t)].append(summ.get("mean_foreground_iou"))
    rows_out=[]
    for t,vals in per_t.items():
        vv=[v for v in vals if v is not None]
        rows_out.append({"threshold":t,"mean_scene_foreground_iou":float(np.mean(vv)) if vv else 0.0,
                         "scene_std_foreground_iou":float(np.std(vv)) if vv else 0.0,"scenes_scored":len(vv)})
    # Prefer threshold nearest 0.5 when validation IoU ties.
    best=sorted(rows_out,key=lambda x:(-x['mean_scene_foreground_iou'],abs(x['threshold']-0.5)))[0]
    report={"selection_split":"val","objective":"mean per-scene foreground IoU","temperature_from_validation":float(temperature),"best_threshold":best['threshold'],
            "best_validation_mean_scene_foreground_iou":best['mean_scene_foreground_iou'],"sweep":rows_out,
            "warning":"Threshold selected on validation scenes only; do not retune it on test data."}
    (Path(out_dir)/"threshold_tuning_validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return float(best['threshold']),report


def auto_class_weights(rows, root, num_classes, ignore_index, device,
                       in_channels, normalization, seed=42):
    """Estimate class weights only from train pixels that are valid in the input.

    This uses the same dataset validity rules as training, so nodata/invalid pixels
    cannot distort class frequencies. Validation and test labels are never counted.
    """
    counts=np.zeros(num_classes,dtype=np.float64)
    train_ds=ManifestSegmentationDataset(rows,"train",in_channels,normalization,
                                         num_classes,ignore_index,32,1,False,seed)
    for i in range(len(train_ds)):
        y=train_ds[i]["mask"].numpy()
        for c in range(num_classes): counts[c] += np.count_nonzero(y==c)
    valid=counts.sum()
    if valid==0: return None, counts
    freq=counts/max(valid,1)
    weights=np.ones_like(freq)
    present=counts>0
    weights[present]=1.0/np.sqrt(np.maximum(freq[present],1e-8))
    if present.any(): weights[present]/=weights[present].mean()
    weights=np.clip(weights,0.25,4.0)
    return torch.tensor(weights,dtype=torch.float32,device=device), counts


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",required=True); ap.add_argument("--root",default="."); ap.add_argument("--output",required=True)
    ap.add_argument("--task",choices=["flood_extent","building_damage","building_change"],required=True)
    ap.add_argument("--in-channels",type=int,required=True); ap.add_argument("--num-classes",type=int,required=True)
    ap.add_argument("--normalization",choices=["rgb255","optical10000","sar_db","unit"],required=True, help="Explicit training normalization; auto is disabled to keep train/inference consistent")
    ap.add_argument("--ignore-index",type=int,default=255); ap.add_argument("--epochs",type=int,default=40)
    ap.add_argument("--batch-size",type=int,default=2); ap.add_argument("--patch-size",type=int,default=256)
    ap.add_argument("--patches-per-scene",type=int,default=8); ap.add_argument("--base-channels",type=int,default=16)
    ap.add_argument("--learning-rate",type=float,default=3e-4); ap.add_argument("--weight-decay",type=float,default=1e-4)
    ap.add_argument("--workers",type=int,default=0); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--patience",type=int,default=8); ap.add_argument("--overlap",type=int,default=32)
    ap.add_argument("--no-auto-class-weights",action="store_true")
    ap.add_argument("--no-radiometric-augmentation", action="store_true", help="Disable modality-aware radiometric jitter for a controlled validation ablation")
    args=ap.parse_args()
    if args.patch_size<32 or args.patch_size%16: raise SystemExit("--patch-size must be >=32 and divisible by 16")
    if args.epochs<1 or args.batch_size<1: raise SystemExit("epochs and batch-size must be positive")
    root=Path(args.root).resolve(); manifest=Path(args.manifest)
    if not manifest.is_absolute(): manifest=(Path.cwd()/manifest).resolve()
    rows=read_manifest(manifest, root)
    task_contracts={"flood_extent": (2, 2, "sar_db"), "building_damage": (6, 5, "rgb255"), "building_change": (6, 2, "rgb255")}
    exp_channels, exp_classes, exp_norm = task_contracts[args.task]
    if (args.in_channels, args.num_classes) != (exp_channels, exp_classes):
        raise SystemExit(f"Task {args.task} contract requires in_channels={exp_channels}, num_classes={exp_classes}; received {args.in_channels}, {args.num_classes}")
    if args.normalization != exp_norm:
        raise SystemExit(f"Task {args.task} requires normalization={exp_norm} to match its documented dataset scale; received {args.normalization}")
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    seed_all(args.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds=ManifestSegmentationDataset(rows,"train",args.in_channels,args.normalization,args.num_classes,args.ignore_index,args.patch_size,args.patches_per_scene,True,args.seed,not args.no_radiometric_augmentation)
    val_ds=ManifestSegmentationDataset(rows,"val",args.in_channels,args.normalization,args.num_classes,args.ignore_index,args.patch_size,1,False,args.seed)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=(device.type=="cuda"))
    # Training-only distribution reference for inference drift screening; never samples val/test.
    profile_ds=ManifestSegmentationDataset(rows,"train",args.in_channels,args.normalization,args.num_classes,
                                           args.ignore_index,args.patch_size,1,False,args.seed,False)
    def _profile_samples():
        for i in range(len(profile_ds.rows)):
            xx, _yy, _row, meta = profile_ds._get_sample(i)
            vm = meta.get("valid_mask")
            if not isinstance(vm, np.ndarray): vm = np.all(np.isfinite(xx), axis=0)
            yield xx.astype(np.float32), np.asarray(vm, dtype=bool)
    channel_names = ( ["VV_dB", "VH_dB"] if args.task == "flood_extent" else
                      ["before_R", "before_G", "before_B", "after_R", "after_G", "after_B"] )
    input_profile = build_input_profile(_profile_samples(), channel_names=channel_names,
                                        max_pixels_per_sample=5000, seed=args.seed)
    model=SegmentationUNet(args.in_channels,args.num_classes,args.base_channels).to(device)
    weights,counts=(None,None) if args.no_auto_class_weights else auto_class_weights(rows,root,args.num_classes,args.ignore_index,device,args.in_channels,args.normalization,args.seed)
    optim=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim,mode="max",factor=0.5,patience=3,min_lr=1e-6)
    config={"task":args.task,"manifest":str(manifest),"manifest_sha256":__import__("hashlib").sha256(manifest.read_bytes()).hexdigest(),
            "root":str(root),"in_channels":args.in_channels,"num_classes":args.num_classes,"normalization":args.normalization,
            "ignore_index":args.ignore_index,"epochs_requested":args.epochs,"batch_size":args.batch_size,"patch_size":args.patch_size,
            "patches_per_scene":args.patches_per_scene,"base_channels":args.base_channels,"learning_rate":args.learning_rate,
            "weight_decay":args.weight_decay,"workers":args.workers,"seed":args.seed,"patience":args.patience,
            "device":str(device),"radiometric_augmentation_enabled":not args.no_radiometric_augmentation,"train_rows":len(train_ds.rows),"validation_rows":len(val_ds.rows),
            "input_profile": input_profile,
            "test_rows":sum(r["split"].lower()=="test" for r in rows),"class_pixel_counts_train":counts.tolist() if counts is not None else None,
            "class_weights":weights.detach().cpu().tolist() if weights is not None else None,
            "note":"All metric thresholds/selections use validation split only. Test is not accessed by this script."}
    (out/"run_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    history=[]; best=-1.0; best_epoch=0; stale=0; t0=time.time()
    for epoch in range(1,args.epochs+1):
        train_ds.set_epoch(epoch)
        model.train(); losses=[]
        for batch in train_loader:
            xx=batch["image"].to(device); yy=batch["mask"].to(device)
            optim.zero_grad(set_to_none=True); logits=model(xx)
            loss=loss_ce_dice(logits,yy,args.ignore_index,weights)
            if not torch.isfinite(loss): raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optim.step(); losses.append(float(loss.detach().cpu()))
        val=evaluate_split(model,rows,root,"val",args,device)
        score=val["scene_mean_foreground_iou"]
        if score is None: score=val["pooled"].get("mean_foreground_iou")
        score=float(score or 0.0); scheduler.step(score)
        record={"epoch":epoch,"train_loss":float(np.mean(losses)) if losses else None,
                "val_scene_mean_foreground_iou":val["scene_mean_foreground_iou"],
                "val_pooled_mean_foreground_iou":val["pooled"].get("mean_foreground_iou"),
                "val_macro_foreground_f1":val["pooled"].get("macro_foreground_f1"),"learning_rate":optim.param_groups[0]["lr"]}
        history.append(record)
        (out/"latest_validation.json").write_text(json.dumps(val,indent=2),encoding="utf-8")
        torch.save({"state_dict":model.state_dict(),"config":config,"epoch":epoch,"val_report":val},out/"last.pt")
        print(f"epoch {epoch:03d}/{args.epochs} loss={record['train_loss']:.4f} val_mean_foreground_IoU={score:.4f}")
        if best_epoch == 0 or score>best+1e-6:
            best=score; best_epoch=epoch; stale=0
            torch.save({"state_dict":model.state_dict(),"config":config,"epoch":epoch,"val_report":val},out/"best.pt")
            (out/"best_validation.json").write_text(json.dumps(val,indent=2),encoding="utf-8")
        else: stale+=1
        with (out/"history.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(history[0].keys())); writer.writeheader(); writer.writerows(history)
        if stale>=args.patience:
            print(f"early stopping: no validation improvement for {args.patience} epochs")
            break
    # Calibrate multi-class scores on balanced samples from the validation split only.
    best_ck=torch.load(out/"best.pt",map_location=device,weights_only=False)
    model.load_state_dict(best_ck["state_dict"]); model.eval()
    calibration_logits, calibration_labels, calibration_scene_counts = collect_validation_logits(
        model, rows, root, args, device, max_pixels_per_scene=12000)
    calibration_report=fit_temperature(calibration_logits, calibration_labels,
                                       ignore_index=args.ignore_index, seed=args.seed)
    temperature=float(calibration_report["temperature"])
    calibration_report["per_scene_sample_counts"] = calibration_scene_counts
    config["temperature"] = temperature
    config["calibration_report"] = calibration_report
    decision_threshold=None
    if args.num_classes==2:
        decision_threshold,_=tune_binary_threshold(model,rows,root,args,device,out,temperature=temperature)
        config["decision_threshold"]=decision_threshold
    best_ck["config"]=config
    torch.save(best_ck,out/"best.pt")
    (out/"calibration_validation.json").write_text(json.dumps(calibration_report,indent=2),encoding="utf-8")
    run_json=json.loads((out/"run_config.json").read_text(encoding="utf-8"))
    run_json["validation_temperature"]=temperature
    run_json["decision_threshold_from_validation"]=decision_threshold
    (out/"run_config.json").write_text(json.dumps(run_json,indent=2),encoding="utf-8")
    val_report=evaluate_split(model,rows,root,"val",args,device,binary_threshold=decision_threshold,temperature=temperature)
    (out/"best_validation.json").write_text(json.dumps(val_report,indent=2),encoding="utf-8")
    summary={"status":"TRAINING_COMPLETED","task":args.task,"best_epoch":best_epoch,"best_validation_mean_foreground_iou":best,
             "decision_threshold_from_validation":decision_threshold,"temperature_from_validation":temperature,
             "calibration_nll_before":calibration_report["metrics_before"]["negative_log_likelihood"],
             "calibration_nll_after":calibration_report["metrics_after"]["negative_log_likelihood"],
             "epochs_completed":len(history),"elapsed_seconds":time.time()-t0,
             "checkpoint":str((out/"best.pt").resolve()),"official_test_evaluated":False,
             "caveat":"Temperature and binary threshold were fit on validation only. Evaluate the untouched test split separately; validation temperature scaling does not guarantee deployment calibration."}
    (out/"training_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
