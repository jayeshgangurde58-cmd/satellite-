import json
from pathlib import Path

import pytest

from tools.create_deployment_manifest import create_manifest, normalize_held_out_report


def test_normalize_oscd_held_out_report_requires_test_provenance():
    report = {
        "dataset": "OSCD official test split",
        "model_selection_used_test_data": False,
        "per_scene": {f"scene-{idx}": {} for idx in range(10)},
        "micro": {"precision": 0.81, "recall": 0.72, "f1": 0.76, "iou": 0.61, "false_positive_rate": 0.04},
    }
    metrics, split_type, scene_count, dataset = normalize_held_out_report(report, "urban_change")
    assert metrics["f1"] == pytest.approx(0.76)
    assert split_type == "held_out_by_scene"
    assert scene_count == 10
    assert dataset == "OSCD official test split"
    report["model_selection_used_test_data"] = True
    with pytest.raises(ValueError, match="model_selection_used_test_data=false"):
        normalize_held_out_report(report, "urban_change")


def test_manifest_tool_derives_flood_metrics_and_starts_pending(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fixture checkpoint - this test does not run model inference")
    report_path = tmp_path / "heldout.json"
    report_path.write_text(json.dumps({
        "task": "flood_extent", "split": "test", "model_selection_used_evaluated_split": False,
        "num_scenes": 3, "per_scene": [{}, {}, {}],
        "pooled": {"confusion_matrix": [[90, 10], [20, 80]]},
    }), encoding="utf-8")
    metrics_path, manifest_path, summary = create_manifest(checkpoint, report_path, "flood_extent")
    assert summary["approved"] is False
    assert summary["thresholds_satisfied"] is True
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["metrics"]["precision"] == pytest.approx(80 / 90)
    assert metrics["metrics"]["recall"] == pytest.approx(80 / 100)
    assert metrics["metrics"]["false_positive_rate"] == pytest.approx(10 / 100)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["review"]["status"] == "PENDING"
    assert manifest["approved_for_inference"] is False
    assert (tmp_path / "deployment_source_report.json").exists()


def test_manifest_tool_refuses_approval_when_thresholds_fail(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    report_path = tmp_path / "heldout.json"
    report_path.write_text(json.dumps({
        "task": "flood_extent", "split": "test", "model_selection_used_evaluated_split": False,
        "num_scenes": 2, "per_scene": [{}, {}],
        "pooled": {"confusion_matrix": [[98, 2], [80, 20]]},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="do not meet all configured deployment thresholds"):
        create_manifest(checkpoint, report_path, "flood_extent", approve_for_inference=True,
                        reviewer="test reviewer", reviewed_at="2026-10-10T12:00:00Z")


def test_damage_deployment_metric_is_any_damage_vs_other(tmp_path):
    # Row = actual class, column = predicted class. Classes 2, 3, 4 mean any damage.
    report = {
        "task": "building_damage", "split": "test", "model_selection_used_evaluated_split": False,
        "num_scenes": 2, "per_scene": [{}, {}],
        "pooled": {"confusion_matrix": [
            [50, 0, 1, 0, 0],
            [2, 30, 3, 0, 0],
            [1, 2, 20, 1, 0],
            [0, 0, 2, 15, 1],
            [0, 0, 0, 1, 10],
        ]},
    }
    metrics, _, count, _ = normalize_held_out_report(report, "building_damage")
    assert count == 2
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["iou"] <= 1


def test_approved_manifest_from_normalized_real_report_passes_runtime_gate(tmp_path):
    import hashlib
    from api.app import _model_evidence

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fixture-only checkpoint bytes; no inference occurs in this test")
    report_path = tmp_path / "oscd_test_report.json"
    report_path.write_text(json.dumps({
        "dataset": "OSCD official test split", "model_selection_used_test_data": False,
        "per_scene": {f"scene-{i}": {} for i in range(10)},
        "micro": {"precision": 0.91, "recall": 0.88, "f1": 0.895, "iou": 0.81, "false_positive_rate": 0.025},
    }), encoding="utf-8")
    _, manifest_path, summary = create_manifest(
        checkpoint, report_path, "urban_change", approve_for_inference=True,
        reviewer="independent-reviewer", reviewed_at="2026-10-10T12:00:00Z",
        thresholds={"minimum_precision": 0.7, "minimum_recall": 0.7, "minimum_f1": 0.7,
                    "minimum_iou": 0.5, "maximum_false_positive_rate": 0.1},
    )
    assert summary["approved"] is True
    runtime = _model_evidence(checkpoint, "urban_change")
    assert runtime["approved"] is True
    assert runtime["checkpoint_hash_matches"] is True
    assert runtime["metrics_hash_matches"] is True
    assert runtime["source_report_hash_matches"] is True
    # Altering the retained raw report must immediately close the deployment gate.
    source_copy = tmp_path / "deployment_source_report.json"
    source_copy.write_text(source_copy.read_text(encoding="utf-8") + " ", encoding="utf-8")
    runtime = _model_evidence(checkpoint, "urban_change")
    assert runtime["approved"] is False
    assert runtime["source_report_hash_matches"] is False
