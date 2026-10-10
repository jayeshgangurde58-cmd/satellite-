# ChronoTerra model deployment evidence gate

The API intentionally refuses to infer from a checkpoint file alone. Each task needs a task-compatible checkpoint, an untouched held-out evaluation report, a normalized metric summary, hashes that bind the files together, and an explicitly reviewed deployment manifest.

## 1. Train and evaluate one model first

Use the matching commands in `DATASET_SETUP.md` and `../README.md`. Do not put real datasets, checkpoints, or generated evaluation reports in the source-control archive unless redistribution is permitted. The report must come from the frozen `test` split; do not select thresholds or tune the model using test labels.

## 2. Create a pending manifest (inference remains disabled)

Run from `ChronoTerra_Backend\prediction_module` after the relevant test evaluator has created its report.

```powershell
# Urban change (OSCD)
python -m tools.create_deployment_manifest --task urban_change --checkpoint outputs/models/oscd_siamese_v2/best.pt --report outputs/models/oscd_siamese_v2/oscd_test_metrics.json

# Flood extent (Sen1Floods11)
python -m tools.create_deployment_manifest --task flood_extent --checkpoint outputs/models/sen1floods11_sar_unet_v1/best.pt --report outputs/models/sen1floods11_sar_unet_v1/test_metrics.json

# Building damage (xBD-style)
python -m tools.create_deployment_manifest --task building_damage --checkpoint outputs/models/xbd_damage_unet_v1/best.pt --report outputs/models/xbd_damage_unet_v1/test_metrics.json
```

For each model directory, the tool writes:

- `deployment_source_report.json`: byte-for-byte copy of the source test report;
- `deployment_metrics.json`: normalized metrics used by the deployment gate;
- `deployment_manifest.json`: hashes, dataset/split metadata, thresholds and review state.

The default manifest is `PENDING`, with `approved_for_inference=false`. The API should continue returning `MODEL_NEEDS_EVIDENCE_REVIEW` for a present checkpoint with a pending/invalid manifest.

## 3. Inspect the report and approve explicitly

Do not run the approval command until a reviewer has inspected the full per-scene report, confusion data/class support, failure examples, input compatibility and dataset/split provenance. The default thresholds (precision 0.70, recall 0.70, F1 0.70, IoU 0.50, false-positive rate at most 0.10) are **starting deployment policy values**, not universally correct scientific requirements. Set different thresholds only with a documented task/use-case justification.

```powershell
python -m tools.create_deployment_manifest --task urban_change --checkpoint outputs/models/oscd_siamese_v2/best.pt --report outputs/models/oscd_siamese_v2/oscd_test_metrics.json --approve-for-inference --reviewer "Reviewer name or team" --reviewed-at "2026-10-10T12:00:00Z"
```

Use the matching task/checkpoint/report for flood and damage. The example timestamp is illustrative; replace it with the real review time and timezone. The command fails closed if the report is not the expected held-out test format, required metrics are missing, hashes cannot be created, or the requested metrics do not meet all configured thresholds. Omitting `--approve-for-inference` never approves the model.

### Metric interpretation by task

- **Urban change:** binary candidate-change metrics from the OSCD evaluator's pooled micro metrics.
- **Flood extent:** binary water-vs-not-water metrics collapsed from the task's held-out confusion matrix.
- **Building damage:** the deployment gate uses a binary **any-damage (minor/major/destroyed) vs background/no-damage** decision derived from the multiclass confusion matrix. This does not certify the individual severity classes; inspect class-wise performance separately before use.

The API re-computes the checkpoint, normalized metrics, and source-report hashes, validates the task and test-split assertions, and checks the reviewer/threshold fields at inference time. If a bound file changes, the gate closes. Hashes prove file consistency, not that labels are correct or that a claimed test split is honest; the human review and external audit trail remain essential.

## 4. What the gate does not prove

An approved deployment manifest is a local release-control record, not independent certification. It does not establish performance beyond the tested dataset, legal status of urban changes, flood causality, engineering structural safety, or robust calibration across regions/sensors/seasons. Keep metrics, data provenance, split hashes and known limitations with every release.
