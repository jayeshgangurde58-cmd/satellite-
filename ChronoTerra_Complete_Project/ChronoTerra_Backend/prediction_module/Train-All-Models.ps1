param([ValidateSet('all','urban','flood','damage')][string]$Model = 'all')
$ErrorActionPreference = 'Stop'
# Run only after checking FUSION's current rules, preparing licensed data and installing dependencies.
# Override with environment vars to fit hardware: CHRONOTERRA_EPOCHS, CHRONOTERRA_BATCH, CHRONOTERRA_PATCHES.
$Epochs = if ($env:CHRONOTERRA_EPOCHS) { [int]$env:CHRONOTERRA_EPOCHS } else { 30 }
$Batch = if ($env:CHRONOTERRA_BATCH) { [int]$env:CHRONOTERRA_BATCH } else { 2 }
$Patches = if ($env:CHRONOTERRA_PATCHES) { [int]$env:CHRONOTERRA_PATCHES } else { 4 }
python -m tools.check_datasets --root .

$oscdImages = 'data/oscd/Onera Satellite Change Detection dataset - Images'
$oscdTrain = 'data/oscd/Onera Satellite Change Detection dataset - Train Labels'
$oscdTest = 'data/oscd/Onera Satellite Change Detection dataset - Test Labels'
$floodS1 = 'data/sen1floods11/data/flood_events/HandLabeled/S1Hand'
$floodLabels = 'data/sen1floods11/data/flood_events/HandLabeled/LabelHand'
$floodSplits = 'data/sen1floods11/splits/flood_handlabeled'
$xbdSplit = if ((Test-Path 'data/xbd/train/images') -and (Test-Path 'data/xbd/train/labels')) { 'data/xbd/train' } elseif ((Test-Path 'data/xbd/tier1/images') -and (Test-Path 'data/xbd/tier1/labels')) { 'data/xbd/tier1' } else { 'data/xbd/train' }
$xbdTrainImages = Join-Path $xbdSplit 'images'; $xbdTrainLabels = Join-Path $xbdSplit 'labels' 
$requiredPaths = @()
if ($Model -in @('all','urban')) { $requiredPaths += @($oscdImages,$oscdTrain,$oscdTest) }
if ($Model -in @('all','flood')) { $requiredPaths += @($floodS1,$floodLabels,$floodSplits) }
if ($Model -in @('all','damage')) { $requiredPaths += @($xbdTrainImages,$xbdTrainLabels) }
foreach ($p in $requiredPaths) {
  if (-not (Test-Path $p)) { throw "Required dataset path missing: $p. See docs/DATASET_SETUP.md. No training was started. Use -Model urban/flood/damage to train one prepared branch." }
}

if ($Model -in @('all','urban')) {
Write-Host '[Urban] OSCD data audit and Siamese U-Net'
python -m tools.analyze_oscd --root data/oscd --output outputs/analysis/oscd_data_analysis.json
if ($LASTEXITCODE -ne 0) { throw 'OSCD data audit found problems; inspect report before training.' }
python -m models.train_oscd --root data/oscd --output outputs/models/oscd_siamese_v2 --epochs $Epochs --steps-per-epoch 100 --batch-size $Batch --patch-size 128 --base-channels 16 --learning-rate 0.0003 --val-count 3 --seed 42
if ($LASTEXITCODE -ne 0) { throw 'OSCD training failed; inspect its error and do not continue blindly.' }
python -m models.evaluate_oscd_model --root data/oscd --checkpoint outputs/models/oscd_siamese_v2/best.pt --output outputs/models/oscd_siamese_v2/oscd_test_metrics.json
if ($LASTEXITCODE -ne 0) { throw 'OSCD held-out evaluation failed.' }
}

if ($Model -in @('all','flood')) {
Write-Host '[Flood] Sen1Floods11 Sentinel-1 U-Net'
python -m tools.make_sen1floods_manifest --root data/sen1floods11 --output data/manifests/sen1floods11.csv
if ($LASTEXITCODE -ne 0) { throw 'Sen1Floods11 manifest creation failed; inspect the official split files and event separation.' }
python -m tools.analyze_manifest --manifest data/manifests/sen1floods11.csv --root data/sen1floods11 --num-classes 2 --ignore-index 255 --output outputs/analysis/sen1floods11_data_analysis.json
if ($LASTEXITCODE -ne 0) { throw 'Sen1Floods11 dataset analysis failed quality checks.' }
python -m models.train_manifest --manifest data/manifests/sen1floods11.csv --root data/sen1floods11 --output outputs/models/sen1floods11_sar_unet_v1 --task flood_extent --in-channels 2 --num-classes 2 --normalization sar_db --ignore-index 255 --epochs $Epochs --batch-size $Batch --patch-size 256 --patches-per-scene $Patches --base-channels 16 --learning-rate 0.0003 --patience 8 --seed 42
if ($LASTEXITCODE -ne 0) { throw 'Flood model training failed.' }
python -m models.evaluate_manifest --manifest data/manifests/sen1floods11.csv --root data/sen1floods11 --checkpoint outputs/models/sen1floods11_sar_unet_v1/best.pt --output outputs/models/sen1floods11_sar_unet_v1/test_metrics.json --split test
if ($LASTEXITCODE -ne 0) { throw 'Flood held-out evaluation failed.' }
}

if ($Model -in @('all','damage')) {
Write-Host "[Damage] xBD paired RGB U-Net (development split: $xbdSplit)"
python -m tools.prepare_xbd_targets --root data/xbd --splits train test tier3
if ($LASTEXITCODE -ne 0) { throw 'xBD annotation-to-mask preparation failed.' }
python -m tools.make_xbd_manifest --root data/xbd --output data/manifests/xbd_damage.csv --val-fraction 0.15 --test-fraction 0.15 --seed 42 --include-tier3
if ($LASTEXITCODE -ne 0) { throw 'xBD manifest creation failed; inspect target encoding and event split.' }
python -m tools.analyze_manifest --manifest data/manifests/xbd_damage.csv --root data/xbd --num-classes 5 --ignore-index 255 --output outputs/analysis/xbd_data_analysis.json
if ($LASTEXITCODE -ne 0) { throw 'xBD dataset analysis failed quality checks.' }
python -m models.train_manifest --manifest data/manifests/xbd_damage.csv --root data/xbd --output outputs/models/xbd_damage_unet_v1 --task building_damage --in-channels 6 --num-classes 5 --normalization rgb255 --ignore-index 255 --epochs $Epochs --batch-size $Batch --patch-size 256 --patches-per-scene $Patches --base-channels 16 --learning-rate 0.0003 --patience 8 --seed 42
if ($LASTEXITCODE -ne 0) { throw 'Building-damage training failed.' }
python -m models.evaluate_manifest --manifest data/manifests/xbd_damage.csv --root data/xbd --checkpoint outputs/models/xbd_damage_unet_v1/best.pt --output outputs/models/xbd_damage_unet_v1/test_metrics.json --split test
if ($LASTEXITCODE -ne 0) { throw 'Building-damage held-out evaluation failed.' }
}

Write-Host "Training workflow '$Model' completed. Review per-scene metrics, validation reports and warnings before connecting any model to the UI."
