"""One-time interrupted-scope registry/action-log repair regressions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from pydrake.all import RigidTransform

from scenesmith.agent_utils import asset_registry as registry_module
from scenesmith.agent_utils.asset_registry import AssetRegistry
from scenesmith.agent_utils.room import ObjectType, SceneObject, UniqueID
from scripts import repair_manipuland_registry as repair


def _asset(root: Path, object_id: str) -> SceneObject:
    directory = root / "sdf" / object_id
    directory.mkdir(parents=True)
    geometry = directory / "mesh.gltf"
    geometry.write_text('{"asset":{"version":"2.0"}}\n', encoding="utf-8")
    sdf = directory / "model.sdf"
    sdf.write_text(
        "<sdf version='1.7'><model name='x'><link name='base'><visual "
        "name='v'><geometry><mesh><uri>mesh.gltf</uri></mesh></geometry>"
        "</visual></link></model></sdf>",
        encoding="utf-8",
    )
    return SceneObject(
        object_id=UniqueID(object_id),
        object_type=ObjectType.MANIPULAND,
        name=object_id,
        description=object_id,
        transform=RigidTransform(),
        geometry_path=geometry,
        sdf_path=sdf,
        metadata={"source": "test"},
        bbox_min=np.array([-0.1, -0.1, 0.0]),
        bbox_max=np.array([0.1, 0.1, 0.1]),
    )


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    namespace = tmp_path / "generated_assets" / "manipuland"
    namespace.mkdir(parents=True)
    scene_states = tmp_path / "scene_states"
    scene_states.mkdir()
    path = namespace / "asset_registry.json"
    registry = AssetRegistry(
        auto_save_path=path,
        required_root=namespace,
        allowed_object_types=frozenset({ObjectType.MANIPULAND}),
    )
    baseline_id = "teacher_base_0"
    checkpoint_id = "teacher_leaf_0"
    aborted_id = "aborted_folder_0"
    registry.register(_asset(namespace, baseline_id))
    registry.register(_asset(namespace, checkpoint_id))
    target = path.read_bytes()
    registry.register(_asset(namespace, aborted_id))
    current = path.read_bytes()

    receipt = {
        "schema_version": 2,
        "status": "pass",
        "furniture_id": "teacher_desk_0",
        "furniture_index": 0,
        "added_manipuland_ids": [checkpoint_id, "composite_stack_0"],
    }
    receipt["attestation"] = repair._checkpoint_attestation(receipt)
    receipt_path = scene_states / "teacher_completion_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    entries = [
        {"step_number": index, "tool_name": f"tool_{index}"}
        for index in range(1, 6)
    ]
    action_path = tmp_path / "action_log.json"
    action_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    prefix = json.dumps(entries[:2], indent=2).encode("utf-8")
    monkeypatch.setattr(
        repair,
        "_checkpoint_retained_ids",
        lambda _path, *, expected_receipt_sha256, current_assets: {
            baseline_id,
            checkpoint_id,
        },
    )
    return {
        "namespace": namespace,
        "registry": path,
        "receipt": receipt_path,
        "receipt_sha": repair._sha256(receipt_path),
        "current": current,
        "target": target,
        "action": action_path,
        "action_current": action_path.read_bytes(),
        "action_prefix": prefix,
        "quarantine": scene_states / "manipuland_registry_repair_quarantine_test",
        "aborted_id": aborted_id,
    }


def _run(case: dict[str, object], *, apply: bool) -> dict[str, object]:
    return repair.repair(
        registry_path=case["registry"],
        checkpoint_receipt=case["receipt"],
        expected_checkpoint_sha256=case["receipt_sha"],
        expected_input_sha256=hashlib.sha256(case["current"]).hexdigest(),
        expected_output_sha256=hashlib.sha256(case["target"]).hexdigest(),
        quarantine_dir=case["quarantine"],
        action_log_path=case["action"],
        expected_action_log_input_sha256=hashlib.sha256(
            case["action_current"]
        ).hexdigest(),
        expected_action_log_prefix_sha256=hashlib.sha256(
            case["action_prefix"]
        ).hexdigest(),
        action_log_prefix_count=2,
        apply=apply,
    )


def test_hash_bound_repair_quarantines_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    before = case["registry"].read_bytes()
    verified = _run(case, apply=False)
    assert verified["status"] == "verified"
    assert case["registry"].read_bytes() == before
    assert not case["quarantine"].exists()

    repaired = _run(case, apply=True)
    assert repaired["status"] == "repaired"
    assert case["registry"].read_bytes() == case["target"]
    assert case["action"].read_bytes() == case["action_prefix"]
    assert case["quarantine"].is_dir()
    manifest = json.loads((case["quarantine"] / "manifest.json").read_text())
    assert manifest["removed_asset_ids"] == [case["aborted_id"]]
    assert len(manifest["removed_asset_files"]) == 2
    assert _run(case, apply=True)["status"] == "repaired"


def test_wrong_target_hash_has_no_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    registry_before = case["registry"].read_bytes()
    action_before = case["action"].read_bytes()
    with pytest.raises(repair.RepairError, match="Reconstructed.*mismatch"):
        repair.repair(
            registry_path=case["registry"],
            checkpoint_receipt=case["receipt"],
            expected_checkpoint_sha256=case["receipt_sha"],
            expected_input_sha256=hashlib.sha256(case["current"]).hexdigest(),
            expected_output_sha256="0" * 64,
            quarantine_dir=case["quarantine"],
            action_log_path=case["action"],
            expected_action_log_input_sha256=hashlib.sha256(
                case["action_current"]
            ).hexdigest(),
            expected_action_log_prefix_sha256=hashlib.sha256(
                case["action_prefix"]
            ).hexdigest(),
            action_log_prefix_count=2,
            apply=True,
        )
    assert case["registry"].read_bytes() == registry_before
    assert case["action"].read_bytes() == action_before
    assert not case["quarantine"].exists()
