# Model weights and deployment status

**Status as of package build:** no model weights trained on real satellite datasets are included. The `synthetic_demo` rasters and test fixtures are generated software fixtures, not satellite imagery. The three model architectures and their dataset-specific training/inference pipelines are source code only; no accuracy results are claimed. The trainers now fit scalar temperature and binary operating threshold on validation-only samples where applicable, but this does not guarantee deployment calibration.

Expected checkpoints after real training and evaluation:

| Task | Expected checkpoint | Required held-out report | Input contract |
|---|---|---|---|
| Urban change | `outputs/models/oscd_siamese_v2/best.pt` | `oscd_test_metrics.json` | 13-band aligned Sentinel-2 GeoTIFF pair in documented OSCD band order |
| Flood extent | `outputs/models/sen1floods11_sar_unet_v1/best.pt` | `test_metrics.json` | Two-band georeferenced Sentinel-1 SAR VV/VH GeoTIFF |
| Building damage | `outputs/models/xbd_damage_unet_v1/best.pt` | `test_metrics.json` | Compatible pre/post-disaster RGB pair, trained with xBD-compatible five-class targets |

Training and evaluation files should record data source/provenance, data manifest hash, split, seed, normalization, hyperparameters, software/device, checkpoint hash and output metrics. Do not treat a file's existence as evidence that the data were genuine or that performance is good.

Inference is now disabled unless each checkpoint directory also contains these three files created by `tools/create_deployment_manifest.py`:

- `deployment_source_report.json` — exact copy of the raw held-out evaluator report.
- `deployment_metrics.json` — normalized, task-specific screening metrics derived from that report.
- `deployment_manifest.json` — matching hashes, test-scene count, minimum metrics thresholds and explicit reviewer approval.

The tool creates a pending/blocked manifest by default. It refuses validation/unknown-split reports, absent metric fields, mismatched task metadata or a requested approval when configured thresholds fail. Follow `docs/MODEL_DEPLOYMENT_GATES.md`. This is a software release gate, not proof that the source dataset/split/labels are honest; humans must inspect per-scene/class results and preserve an independent review record.
