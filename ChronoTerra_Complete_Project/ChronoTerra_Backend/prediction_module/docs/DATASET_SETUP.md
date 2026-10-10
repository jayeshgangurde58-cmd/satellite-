# Dataset preparation (real labelled data only)

This project does not bundle satellite datasets and no training command downloads data automatically.
Create one dataset at a time, check use terms, then inspect manifests before training.

## A. Urban change — ONERA / OSCD

Maintainer page: https://rcdaudt.github.io/oscd/

The dataset reports 24 registered Sentinel-2 pairs, 13 spectral bands and pixel-level change labels, with a small official split (14 train / 10 test pairs in the current project notes). The supplied `models/train_oscd.py` uses official training scenes to create a separate validation subset. It must not train on the official test labels.

Use the official page and follow the linked archive instructions. Expected layout under `data/oscd/`:

```text
Onera Satellite Change Detection dataset - Images/<scene>/imgs_1_rect/*.tif
Onera Satellite Change Detection dataset - Images/<scene>/imgs_2_rect/*.tif
Onera Satellite Change Detection dataset - Train Labels/<scene>/cm/cm.png
Onera Satellite Change Detection dataset - Test Labels/<scene>/cm/cm.png
```

Run `python -m tools.check_datasets --root .` after extraction. Confirm band order, scale factor, mask shape, NoData handling and the current dataset terms before public redistribution. OSCD results are not a proxy for RGB, SAR, building damage or legal encroachment.

## B. Flood-water segmentation — Sen1Floods11 v1.1 hand-labelled subset

Maintainer repository and data schema: https://github.com/cloudtostreet/Sen1Floods11

Dataset docs: https://github.com/cloudtostreet/Sen1Floods11/blob/master/docs/README.md

Use the provided PowerShell helper (requires Google Cloud CLI / `gsutil`):

```powershell
.\tools\download_sen1floods11_handlabeled.ps1
python -m tools.make_sen1floods_manifest --root data/sen1floods11 --output data/manifests/sen1floods11.csv
python -m tools.analyze_manifest --manifest data/manifests/sen1floods11.csv --root data/sen1floods11 --num-classes 2 --ignore-index 255 --output outputs/analysis/sen1floods11_data_analysis.json
```

The helper downloads only the hand-labelled S1 chips, LabelHand masks and official split files, not the full weakly-labelled dataset. S1 inputs have two bands (VV/VH in dB); label values are `-1` invalid, `0` not water, `1` water. The loader maps `-1` to ignore index 255. The model output is **water extent**, not automatically “flood change” or building damage.

The manifest builder fails if it detects a split/event conflict or cannot map official split rows. If it fails, inspect the downloaded split CSV schema and event metadata; do not weaken the leakage guard simply to make it pass. Sen1Floods11 is publicly accessible, but check its current terms/citation instructions because the repository README does not provide an explicit standalone license declaration.

## C. Building damage — xBD / xView2

Official challenge/download page: https://xview2.org/dataset

Dataset paper: https://openaccess.thecvf.com/content/CVPRW/2019/papers/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_Imagery_CVPRW_2019_paper.pdf

Access requires following the dataset site's access process and accepting the current terms. Do not download unofficial mirrors or redistribute data without verifying rights. A typical raw split contains paired RGB images in `images/` and annotation JSON files in `labels/`:

```text
data/xbd/tier1/images/<event>_<id>_pre_disaster.png
 data/xbd/tier1/images/<event>_<id>_post_disaster.png
 data/xbd/tier1/labels/<event>_<id>_pre_disaster.json
 data/xbd/tier1/labels/<event>_<id>_post_disaster.json
```

First rasterize the post-disaster polygon labels into masks. The tool writes generated files under `targets/` without modifying the raw source files. Unclassified building polygons become ignore value `255`, not background/no-damage:

```powershell
python -m tools.prepare_xbd_targets --root data/xbd --splits train test tier3
```

Some extracts call the labelled development split `tier1`; others use `train`. The scripts accept either and the readiness checker reports which path it selects. If a separately labelled Tier 3 archive is available and its terms permit use, place it at `data/xbd/tier3/{images,labels}`; the recommended command below includes it. The builder keeps entire event groups within one split and excludes training events that overlap a labelled test event. Do not place the challenge holdout or unlabeled challenge images into training.

Then build and inspect the manifest:

```powershell
python -m tools.make_xbd_manifest --root data/xbd --output data/manifests/xbd_damage.csv --val-fraction 0.15 --test-fraction 0.15 --seed 42 --include-tier3
python -m tools.analyze_manifest --manifest data/manifests/xbd_damage.csv --root data/xbd --num-classes 5 --ignore-index 255 --output outputs/analysis/xbd_data_analysis.json
```

The generated target-mask encoding is `0=background, 1=no damage, 2=minor, 3=major, 4=destroyed, 255=ignore/unclassified`. Inspect mask PNGs visually against their source images before training. If a labelled test folder exists, it is retained after checking event leakage; if challenge test images do not have public labels, the script creates an internal event-group-held-out test split from labelled samples. Such metrics must be described as an internal held-out test, not the hidden official challenge test. The mask and manifest preparation stages must stop on mismatched dimensions or event leakage rather than guess. If the challenge `test` folder has no public labels, the generated test split is an internal event-group holdout and must not be described as the official hidden test score. The `xBD-S12` dataset is a different Sentinel-1/Sentinel-2 derivation; this current damage checkpoint expects RGB xBD chips and cannot accept xBD-S12 tensors without a separately implemented/validated preprocessing contract. See https://github.com/prs-eth/xbd-s12 for its own preparation and licensing notes.

## Data terms and reporting

- The data files and annotations have terms separate from this source code. Check the current maintainer terms for each dataset before publishing weights, screenshots, samples or derived data.
- Keep dataset ID/version, archive checksums, manifest hash, code version, seed, hyperparameters, normalization, test split and model checkpoint with every result.
- The package's tests use tiny synthetic fixtures for software behavior. They are not real satellite accuracy tests.
