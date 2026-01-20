"""Durable and legacy per-furniture manipuland checkpoint regressions."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import xml.etree.ElementTree as ET

from contextlib import nullcontext
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
import numpy as np

from omegaconf import OmegaConf
from pydrake.all import RigidTransform

from scenesmith.agent_utils.asset_registry import AssetRegistry
from scenesmith.agent_utils.house import RoomGeometry
from scenesmith.agent_utils.room import (
    ObjectType,
    PlacementInfo,
    RoomScene,
    SceneObject,
    SupportSurface,
    UniqueID,
)
from scenesmith.agent_utils.scene_analyzer import FurnitureSelection
from scenesmith.manipuland_agents import stateful_manipuland_agent as checkpoint
from scenesmith.manipuland_agents.stateful_manipuland_agent import (
    StatefulManipulandAgent,
)
from scenesmith.agent_utils.scene_analyzer import FurnitureSelection


class _FakeScene:
    def __init__(self, state: dict, scene_dir: Path):
        self._state = copy.deepcopy(state)
        self.scene_dir = scene_dir
        self.room_id = "classroom_01"

    def to_state_dict(self) -> dict:
        return copy.deepcopy(self._state)

    def restore_from_state_dict(self, state: dict) -> None:
        self._state = copy.deepcopy(state)

    def content_hash(self) -> str:
        value = copy.deepcopy(self._state)
        value.pop("timestamp", None)
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class _FakeLogger:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def log_scene(self, scene, name=None, output_dir=None):
        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "scene_state.json").write_text(
            json.dumps(scene.to_state_dict()), encoding="utf-8"
        )
        (output_dir / "scene.dmd.yaml").write_text("directives:\n", encoding="utf-8")
        return output_dir


def _base_state() -> dict:
    return {
        "room_geometry": {"room_id": "classroom_01"},
        "text_description": "room",
        "objects": {
            "teacher_desk_0": {
                "object_id": "teacher_desk_0",
                "object_type": "furniture",
                "transform": {
                    "translation": [0, 0, 0],
                    "rotation_wxyz": [1, 0, 0, 0],
                },
                "support_surfaces": [],
                "metadata": {"authority": "bound"},
                "geometry_path": "generated_assets/furniture/teacher_desk.gltf",
                "sdf_path": "generated_assets/furniture/teacher_desk.sdf",
            },
            "chair_0": {
                "object_id": "chair_0",
                "object_type": "furniture",
                "transform": {
                    "translation": [1, 0, 0],
                    "rotation_wxyz": [1, 0, 0, 0],
                },
                "support_surfaces": [],
                "metadata": {"authority": "bound"},
            },
        },
        "timestamp": 1.0,
    }


def _accepted_state() -> dict:
    state = _base_state()
    state["timestamp"] = 2.0
    state["objects"]["teacher_desk_0"]["transform"] = {
        "translation": [1e-16, 0, 0],
        "rotation_wxyz": [-1, 0, 0, 0],
    }
    state["objects"]["teacher_desk_0"]["support_surfaces"] = [
        {"surface_id": "S_0"}
    ]
    state["objects"]["chair_0"]["transform"] = {
        "translation": [1, 1e-16, 0],
        "rotation_wxyz": [1, 0, 0, 0],
    }
    state["objects"]["gradebook_0"] = {
        "object_id": "gradebook_0",
        "object_type": "manipuland",
        "transform": {
            "translation": [0, 0, 1],
            "rotation_wxyz": [1, 0, 0, 0],
        },
        "geometry_path": "generated_assets/manipuland/gradebook.gltf",
        "sdf_path": "generated_assets/manipuland/gradebook.sdf",
        "support_surfaces": [],
        "metadata": {"asset_source": "objaverse"},
        "placement_info": {
            "placement_method": "surface_placement",
            "parent_surface_id": "S_0",
        },
    }
    return state


def _selection() -> FurnitureSelection:
    return FurnitureSelection(
        furniture_id=UniqueID("teacher_desk_0"),
        suggested_items="gradebook",
        prompt_constraints="teacher materials",
        style_notes="organized",
        context_furniture_ids=[UniqueID("chair_0")],
    )


def _write_evidence(directory: Path, *, state: dict | None = None) -> None:
    directory.mkdir(parents=True)
    score = {
        category: {"grade": 8, "comment": "passes"}
        for category in checkpoint._SCORE_CATEGORIES
    }
    score["summary"] = "accepted"
    (directory / "scores.yaml").write_text(
        __import__("yaml").safe_dump(score, sort_keys=False), encoding="utf-8"
    )
    for name in ("0_side.png", "0_top.png", "1_side.png", "2_side.png", "3_side.png"):
        (directory / name).write_bytes(("image:" + name).encode())
    if state is not None:
        (directory / "scene_state.json").write_text(
            json.dumps(state), encoding="utf-8"
        )
        (directory / "scene.dmd.yaml").write_text("directives:\n", encoding="utf-8")


def _write_added_assets(root: Path) -> None:
    directory = root / "generated_assets" / "manipuland"
    directory.mkdir(parents=True, exist_ok=True)
    binary = b"gradebook-bin"
    (directory / "gradebook.bin").write_bytes(binary)
    (directory / "gradebook.gltf").write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "buffers": [
                    {"uri": "gradebook.bin", "byteLength": len(binary)}
                ],
            }
        ),
        encoding="utf-8",
    )
    (directory / "gradebook_collision.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    (directory / "gradebook.sdf").write_text(
        "<sdf version='1.7'><model name='gradebook'><link name='base'>"
        "<visual><geometry><mesh><uri>gradebook.gltf</uri></mesh></geometry></visual>"
        "<collision><geometry><mesh><uri>gradebook_collision.obj</uri></mesh>"
        "</geometry></collision></link></model></sdf>",
        encoding="utf-8",
    )
    furniture = root / "generated_assets" / "furniture"
    furniture.mkdir(parents=True, exist_ok=True)
    teacher_binary = b"teacher-desk-bin"
    (furniture / "teacher_desk.bin").write_bytes(teacher_binary)
    (furniture / "teacher_desk.gltf").write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "buffers": [
                    {"uri": "teacher_desk.bin", "byteLength": len(teacher_binary)}
                ],
            }
        ),
        encoding="utf-8",
    )
    (furniture / "teacher_desk_collision.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
    )
    (furniture / "teacher_desk.sdf").write_text(
        "<sdf version='1.7'><model name='teacher_desk'><link name='base'>"
        "<visual><geometry><mesh><uri>teacher_desk.gltf</uri></mesh></geometry>"
        "</visual><collision><geometry><mesh><uri>teacher_desk_collision.obj</uri>"
        "</mesh></geometry></collision></link></model></sdf>",
        encoding="utf-8",
    )


def _authorize_legacy_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    agent: StatefulManipulandAgent,
    render_dir: Path,
) -> Path:
    input_dir = tmp_path / "scene_states" / "scene_after_ceiling_objects"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / "scene_state.json"
    input_path.write_text(json.dumps(agent.scene.to_state_dict()), encoding="utf-8")
    accepted_dir = tmp_path / "scene_states" / "manipuland_furniture_teacher_desk_0"
    candidate = json.loads((render_dir / "scene_state.json").read_text(encoding="utf-8"))
    added_ids = sorted(
        set(candidate["objects"]) - set(agent.scene.to_state_dict()["objects"])
    )
    referenced_ids = sorted(set(added_ids) | {"teacher_desk_0"})
    manifest = {
        "schema_version": checkpoint._LEGACY_RECOVERY_SCHEMA,
        "status": "authorized",
        "purpose": "one_time_legacy_manipuland_recovery",
        "room_id": "classroom_01",
        "furniture_index": 0,
        "furniture_id": "teacher_desk_0",
        "input_scene_content_hash": agent.scene.content_hash(),
        "output_scene_content_hash": _FakeScene(
            candidate, tmp_path
        ).content_hash(),
        "input_scene_state_path": input_path.relative_to(tmp_path).as_posix(),
        "input_scene_state": checkpoint._file_record(input_path),
        "render_directory": render_dir.relative_to(tmp_path).as_posix(),
        "scene_state": checkpoint._file_record(render_dir / "scene_state.json"),
        "drake_directive": checkpoint._file_record(render_dir / "scene.dmd.yaml"),
        "accepted_evidence": checkpoint._accepted_render_evidence(
            accepted_dir, minimum_score=8
        ),
        "added_manipuland_ids": added_ids,
        "added_asset_files": checkpoint._referenced_room_asset_records(
            candidate,
            room_root=tmp_path,
            object_ids=set(added_ids),
        ),
        "referenced_asset_object_ids": referenced_ids,
        "referenced_asset_files": checkpoint._referenced_room_asset_records(
            candidate,
            room_root=tmp_path,
            object_ids=set(referenced_ids),
        ),
    }
    manifest_path = tmp_path.parent / f"{tmp_path.name}_legacy_recovery.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv(
        "SCENESMITH_LEGACY_MANIPULAND_RECOVERY_MANIFEST", str(manifest_path)
    )
    monkeypatch.setenv(
        "SCENESMITH_LEGACY_MANIPULAND_RECOVERY_SHA256",
        checkpoint._sha256_file(manifest_path),
    )
    return manifest_path


def _rewrite_authorization(
    manifest_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
) -> None:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(document)
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setenv(
        "SCENESMITH_LEGACY_MANIPULAND_RECOVERY_SHA256",
        checkpoint._sha256_file(manifest_path),
    )


def _agent(tmp_path: Path, state: dict) -> StatefulManipulandAgent:
    _write_added_assets(tmp_path)
    agent = StatefulManipulandAgent.__new__(StatefulManipulandAgent)
    agent.logger = _FakeLogger(tmp_path)
    agent.scene = _FakeScene(state, tmp_path)
    agent.cfg = OmegaConf.create(
        {"checkpoint_test": True, "early_finish_min_score": 8}
    )
    agent._legacy_recovery_consumed = False
    agent._legacy_recovery_satisfied = False
    agent._required_legacy_recovery = None
    current_render = tmp_path / "current_render"
    _write_evidence(current_render, state=state)
    agent.final_render_dir = current_render
    agent.checkpoint_render_dir = None
    return agent


def _real_roomscene_checkpoint_agent(
    tmp_path: Path,
) -> tuple[StatefulManipulandAgent, dict, str]:
    _write_added_assets(tmp_path)
    room_sdf = tmp_path / "room.sdf"
    room_sdf.write_text(
        "<sdf version='1.7'><model name='room'><link name='base'/></model></sdf>",
        encoding="utf-8",
    )
    floor = SceneObject(
        object_id=UniqueID("floor_classroom_01"),
        object_type=ObjectType.FLOOR,
        name="floor",
        description="floor",
        transform=RigidTransform(),
        sdf_path=room_sdf,
    )
    room_geometry = RoomGeometry(
        sdf_tree=ET.parse(room_sdf),
        sdf_path=room_sdf,
        floor=floor,
    )
    furniture_dir = tmp_path / "generated_assets" / "furniture"
    manip_dir = tmp_path / "generated_assets" / "manipuland"
    teacher = SceneObject(
        object_id=UniqueID("teacher_desk_0"),
        object_type=ObjectType.FURNITURE,
        name="teacher desk",
        description="teacher desk",
        transform=RigidTransform(),
        geometry_path=furniture_dir / "teacher_desk.gltf",
        sdf_path=furniture_dir / "teacher_desk.sdf",
        bbox_min=np.array([-0.5, -0.3, 0.0]),
        bbox_max=np.array([0.5, 0.3, 0.8]),
    )
    chair = SceneObject(
        object_id=UniqueID("chair_0"),
        object_type=ObjectType.FURNITURE,
        name="chair",
        description="chair",
        transform=RigidTransform(p=[1.0, 0.0, 0.0]),
        bbox_min=np.array([-0.2, -0.2, 0.0]),
        bbox_max=np.array([0.2, 0.2, 0.8]),
    )
    scene = RoomScene(
        room_geometry=room_geometry,
        scene_dir=tmp_path,
        room_id="classroom_01",
        objects={obj.object_id: obj for obj in (teacher, chair)},
        text_description="room",
    )
    base_state = scene.to_state_dict()
    input_hash = scene.content_hash()
    surface = SupportSurface(
        surface_id=UniqueID("S_0"),
        bounding_box_min=np.array([-0.5, -0.3, 0.0]),
        bounding_box_max=np.array([0.5, 0.3, 1.0]),
        transform=RigidTransform(p=[0.0, 0.0, 0.8]),
    )
    teacher.support_surfaces = [surface]
    gradebook = SceneObject(
        object_id=UniqueID("gradebook_0"),
        object_type=ObjectType.MANIPULAND,
        name="gradebook",
        description="gradebook",
        transform=RigidTransform(p=[0.0, 0.0, 0.8]),
        geometry_path=manip_dir / "gradebook.gltf",
        sdf_path=manip_dir / "gradebook.sdf",
        placement_info=PlacementInfo(
            parent_surface_id=surface.surface_id,
            position_2d=np.array([0.0, 0.0]),
            rotation_2d=0.0,
        ),
        metadata={"asset_source": "objaverse"},
        bbox_min=np.array([-0.1, -0.1, 0.0]),
        bbox_max=np.array([0.1, 0.1, 0.05]),
    )
    scene.add_object(gradebook)
    agent = StatefulManipulandAgent.__new__(StatefulManipulandAgent)
    agent.logger = _FakeLogger(tmp_path)
    agent.scene = scene
    agent.cfg = OmegaConf.create(
        {"checkpoint_test": True, "early_finish_min_score": 8}
    )
    agent._legacy_recovery_consumed = False
    agent._legacy_recovery_satisfied = False
    agent._required_legacy_recovery = None
    accepted_dir = (
        tmp_path / "scene_states" / "manipuland_furniture_teacher_desk_0"
    )
    _write_evidence(accepted_dir)
    current_render = tmp_path / "current_render"
    _write_evidence(current_render, state=scene.to_state_dict())
    agent.final_render_dir = current_render
    agent.checkpoint_render_dir = None
    return agent, base_state, input_hash


def _authorized_legacy_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[StatefulManipulandAgent, Path, str]:
    accepted_dir = tmp_path / "scene_states" / "manipuland_furniture_teacher_desk_0"
    render_dir = (
        tmp_path
        / "scene_renders"
        / "manipulands_teacher_desk_0"
        / "renders_003"
    )
    _write_evidence(accepted_dir)
    _write_evidence(render_dir, state=_accepted_state())
    agent = _agent(tmp_path, _base_state())
    manifest = _authorize_legacy_recovery(
        tmp_path, monkeypatch, agent=agent, render_dir=render_dir
    )
    return agent, manifest, agent.scene.content_hash()


def test_exact_accepted_legacy_render_is_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted_dir = tmp_path / "scene_states" / "manipuland_furniture_teacher_desk_0"
    render_dir = (
        tmp_path
        / "scene_renders"
        / "manipulands_teacher_desk_0"
        / "renders_003"
    )
    _write_evidence(accepted_dir)
    _write_evidence(render_dir, state=_accepted_state())
    _write_added_assets(tmp_path)
    agent = _agent(tmp_path, _base_state())
    _authorize_legacy_recovery(
        tmp_path, monkeypatch, agent=agent, render_dir=render_dir
    )

    source = agent._restore_legacy_accepted_render(
        _selection(), 0, agent.scene.content_hash()
    )

    assert source is not None
    assert source["render_directory"].endswith("renders_003")
    assert source["added_manipuland_ids"] == ["gradebook_0"]
    assert "gradebook_0" in agent.scene.to_state_dict()["objects"]


def test_legacy_input_accepts_hash_preserving_quaternion_roundtrip_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted_dir = tmp_path / "scene_states" / "manipuland_furniture_teacher_desk_0"
    render_dir = (
        tmp_path
        / "scene_renders"
        / "manipulands_teacher_desk_0"
        / "renders_003"
    )
    _write_evidence(accepted_dir)
    _write_evidence(render_dir, state=_accepted_state())
    agent = _agent(tmp_path, _base_state())
    input_hash = agent.scene.content_hash()
    _authorize_legacy_recovery(
        tmp_path, monkeypatch, agent=agent, render_dir=render_dir
    )
    exact_to_state_dict = agent.scene.to_state_dict

    def roundtripped_state() -> dict:
        state = exact_to_state_dict()
        for item in state["objects"].values():
            item["transform"]["rotation_wxyz"][1] += 1e-18
        return state

    monkeypatch.setattr(agent.scene, "to_state_dict", roundtripped_state)
    assert agent.scene.content_hash() == input_hash
    source = agent._restore_legacy_accepted_render(_selection(), 0, input_hash)
    assert source is not None
    assert source["added_manipuland_ids"] == ["gradebook_0"]


def test_legacy_render_cannot_change_protected_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted = _accepted_state()
    accepted["objects"]["chair_0"]["metadata"] = {"authority": "substituted"}
    _write_evidence(
        tmp_path / "scene_states" / "manipuland_furniture_teacher_desk_0"
    )
    render_dir = (
        tmp_path
        / "scene_renders"
        / "manipulands_teacher_desk_0"
        / "renders_003"
    )
    _write_evidence(render_dir, state=accepted)
    _write_added_assets(tmp_path)
    agent = _agent(tmp_path, _base_state())
    _authorize_legacy_recovery(
        tmp_path, monkeypatch, agent=agent, render_dir=render_dir
    )

    with pytest.raises(RuntimeError, match="protected fields"):
        agent._restore_legacy_accepted_render(
            _selection(), 0, agent.scene.content_hash()
        )


def test_published_checkpoint_restores_only_exact_hash_chain(tmp_path: Path) -> None:
    _write_evidence(
        tmp_path / "scene_states" / "manipuland_furniture_teacher_desk_0"
    )
    base = _base_state()
    output = _accepted_state()
    agent = _agent(tmp_path, output)
    input_hash = _FakeScene(base, tmp_path).content_hash()
    agent._publish_furniture_checkpoint(
        furniture_selection=_selection(),
        furniture_index=0,
        input_scene_hash=input_hash,
        input_object_ids=set(base["objects"]),
        input_scene_state=base,
        legacy_source=None,
    )
    agent.scene.restore_from_state_dict(base)

    assert agent._restore_completed_furniture_checkpoint(
        furniture_selection=_selection(),
        furniture_index=0,
        input_scene_hash=input_hash,
    )
    assert agent.scene.content_hash() == _FakeScene(output, tmp_path).content_hash()

    agent.scene.restore_from_state_dict(base)
    with pytest.raises(RuntimeError, match="contract mismatch"):
        agent._restore_completed_furniture_checkpoint(
            furniture_selection=_selection(),
            furniture_index=0,
            input_scene_hash="0" * 64,
        )


def test_legacy_recovery_is_never_automatic(tmp_path: Path) -> None:
    _write_evidence(
        tmp_path / "scene_states" / "manipuland_furniture_teacher_desk_0"
    )
    render_dir = (
        tmp_path
        / "scene_renders"
        / "manipulands_teacher_desk_0"
        / "renders_003"
    )
    _write_evidence(render_dir, state=_accepted_state())
    agent = _agent(tmp_path, _base_state())
    assert (
        agent._restore_legacy_accepted_render(
            _selection(), 0, agent.scene.content_hash()
        )
        is None
    )


def test_legacy_transform_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted = _accepted_state()
    accepted["objects"]["chair_0"]["transform"]["translation"][0] += 0.001
    accepted_dir = tmp_path / "scene_states" / "manipuland_furniture_teacher_desk_0"
    render_dir = (
        tmp_path
        / "scene_renders"
        / "manipulands_teacher_desk_0"
        / "renders_003"
    )
    _write_evidence(accepted_dir)
    _write_evidence(render_dir, state=accepted)
    _write_added_assets(tmp_path)
    agent = _agent(tmp_path, _base_state())
    _authorize_legacy_recovery(
        tmp_path, monkeypatch, agent=agent, render_dir=render_dir
    )
    with pytest.raises(RuntimeError, match="transform/schema"):
        agent._restore_legacy_accepted_render(
            _selection(), 0, agent.scene.content_hash()
        )


def test_legacy_target_may_add_support_surface_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _base_state()
    del base["objects"]["teacher_desk_0"]["support_surfaces"]
    accepted = _accepted_state()
    accepted_dir = tmp_path / "scene_states" / "manipuland_furniture_teacher_desk_0"
    render_dir = (
        tmp_path
        / "scene_renders"
        / "manipulands_teacher_desk_0"
        / "renders_003"
    )
    _write_evidence(accepted_dir)
    _write_evidence(render_dir, state=accepted)
    _write_added_assets(tmp_path)
    agent = _agent(tmp_path, base)
    _authorize_legacy_recovery(
        tmp_path, monkeypatch, agent=agent, render_dir=render_dir
    )

    assert (
        agent._restore_legacy_accepted_render(
            _selection(), 0, agent.scene.content_hash()
        )
        is not None
    )


def test_low_score_cannot_be_accepted(tmp_path: Path) -> None:
    directory = tmp_path / "evidence"
    _write_evidence(directory)
    scores = __import__("yaml").safe_load((directory / "scores.yaml").read_text())
    scores["Layout"]["grade"] = 7
    (directory / "scores.yaml").write_text(
        __import__("yaml").safe_dump(scores, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="does not pass 8"):
        checkpoint._accepted_render_evidence(directory, minimum_score=8)


def test_exact_furniture_plan_is_reused(tmp_path: Path) -> None:
    agent = _agent(tmp_path, _base_state())
    input_hash = agent.scene.content_hash()
    agent._publish_furniture_plan(input_hash, [_selection()])
    restored = agent._load_furniture_plan(input_hash)
    assert restored is not None
    assert [checkpoint._selection_payload(item) for item in restored] == [
        checkpoint._selection_payload(_selection())
    ]
    with pytest.raises(RuntimeError, match="input_scene_content_hash mismatch"):
        agent._load_furniture_plan("f" * 64)


def test_fabricated_predecessor_furniture_plan_runtime_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, _base_state())
    input_hash = agent.scene.content_hash()
    agent._publish_furniture_plan(input_hash, [_selection()])
    path = tmp_path / "scene_states" / "manipuland_furniture_plan.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["checkpoint_runtime_sha256"] = (
        checkpoint._SCOPE_PREDECESSOR_RUNTIME_SHA256
    )
    document["attestation"] = checkpoint._receipt_attestation(document)
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RuntimeError, match="predecessor furniture plan artifact"):
        agent._load_furniture_plan(input_hash)

    monkeypatch.setattr(
        checkpoint, "_SCOPE_PREDECESSOR_INPUT_SCENE_CONTENT_SHA256", input_hash
    )
    monkeypatch.setattr(
        checkpoint, "_SCOPE_PREDECESSOR_PLAN_SHA256", checkpoint._sha256_file(path)
    )
    assert agent._load_furniture_plan(input_hash) == [_selection()]

    document["checkpoint_runtime_sha256"] = "0" * 64
    document["attestation"] = checkpoint._receipt_attestation(document)
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checkpoint_runtime_sha256 mismatch"):
        agent._load_furniture_plan(input_hash)


def test_checkpoint_extra_file_is_rejected(tmp_path: Path) -> None:
    _write_evidence(
        tmp_path / "scene_states" / "manipuland_furniture_teacher_desk_0"
    )
    base = _base_state()
    agent = _agent(tmp_path, _accepted_state())
    input_hash = _FakeScene(base, tmp_path).content_hash()
    agent._publish_furniture_checkpoint(
        furniture_selection=_selection(),
        furniture_index=0,
        input_scene_hash=input_hash,
        input_object_ids=set(base["objects"]),
        input_scene_state=base,
        legacy_source=None,
    )
    directory = agent._checkpoint_directory(_selection(), 0)
    (directory / "substituted.txt").write_text("bad", encoding="utf-8")
    agent.scene.restore_from_state_dict(base)
    with pytest.raises(RuntimeError, match="inventory mismatch"):
        agent._restore_completed_furniture_checkpoint(
            furniture_selection=_selection(),
            furniture_index=0,
            input_scene_hash=input_hash,
        )


def test_atomic_checkpoint_failure_leaves_no_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_evidence(
        tmp_path / "scene_states" / "manipuland_furniture_teacher_desk_0"
    )
    agent = _agent(tmp_path, _accepted_state())
    original_rename = Path.rename

    def fail_publication(path: Path, target: Path):
        if path.name.startswith(".manipuland_checkpoint_"):
            raise OSError("injected rename failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_publication)
    with pytest.raises(OSError, match="injected rename failure"):
        agent._publish_furniture_checkpoint(
            furniture_selection=_selection(),
            furniture_index=0,
            input_scene_hash="a" * 64,
            input_object_ids=set(_base_state()["objects"]),
            input_scene_state=_base_state(),
            legacy_source=None,
        )
    final = agent._checkpoint_directory(_selection(), 0)
    assert not final.exists()
    assert not list(final.parent.glob(f".{final.name}.tmp.*"))


def test_stale_checkpoint_transaction_fails_closed(tmp_path: Path) -> None:
    _write_evidence(
        tmp_path / "scene_states" / "manipuland_furniture_teacher_desk_0"
    )
    agent = _agent(tmp_path, _accepted_state())
    final = agent._checkpoint_directory(_selection(), 0)
    stale = final.with_name(f".{final.name}.tmp.crashed")
    stale.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Stale manipuland checkpoint transaction"):
        agent._publish_furniture_checkpoint(
            furniture_selection=_selection(),
            furniture_index=0,
            input_scene_hash="a" * 64,
            input_object_ids=set(_base_state()["objects"]),
            input_scene_state=_base_state(),
            legacy_source=None,
        )


def test_checkpoint_state_tamper_is_detected(tmp_path: Path) -> None:
    _write_evidence(
        tmp_path / "scene_states" / "manipuland_furniture_teacher_desk_0"
    )
    base = _base_state()
    agent = _agent(tmp_path, _accepted_state())
    input_hash = _FakeScene(base, tmp_path).content_hash()
    agent._publish_furniture_checkpoint(
        furniture_selection=_selection(),
        furniture_index=0,
        input_scene_hash=input_hash,
        input_object_ids=set(base["objects"]),
        input_scene_state=base,
        legacy_source=None,
    )
    directory = agent._checkpoint_directory(_selection(), 0)
    (directory / "scene_state.json").write_text("{}", encoding="utf-8")
    agent.scene.restore_from_state_dict(base)
    with pytest.raises(RuntimeError, match="hash/size mismatch"):
        agent._restore_completed_furniture_checkpoint(
            furniture_selection=_selection(),
            furniture_index=0,
            input_scene_hash=input_hash,
        )


def test_checkpoint_receipt_extra_field_is_rejected(tmp_path: Path) -> None:
    _write_evidence(
        tmp_path / "scene_states" / "manipuland_furniture_teacher_desk_0"
    )
    base = _base_state()
    agent = _agent(tmp_path, _accepted_state())
    input_hash = _FakeScene(base, tmp_path).content_hash()
    agent._publish_furniture_checkpoint(
        furniture_selection=_selection(),
        furniture_index=0,
        input_scene_hash=input_hash,
        input_object_ids=set(base["objects"]),
        input_scene_state=base,
        legacy_source=None,
    )
    directory = agent._checkpoint_directory(_selection(), 0)
    receipt_path = directory / "completion_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["substitution"] = "not allowed"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    agent.scene.restore_from_state_dict(base)
    with pytest.raises(RuntimeError, match="receipt schema mismatch"):
        agent._restore_completed_furniture_checkpoint(
            furniture_selection=_selection(),
            furniture_index=0,
            input_scene_hash=input_hash,
        )


def test_legacy_checkpoint_revalidates_added_asset_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted_dir = tmp_path / "scene_states" / "manipuland_furniture_teacher_desk_0"
    render_dir = (
        tmp_path
        / "scene_renders"
        / "manipulands_teacher_desk_0"
        / "renders_003"
    )
    _write_evidence(accepted_dir)
    _write_evidence(render_dir, state=_accepted_state())
    _write_added_assets(tmp_path)
    base = _base_state()
    agent = _agent(tmp_path, base)
    input_hash = agent.scene.content_hash()
    _authorize_legacy_recovery(
        tmp_path, monkeypatch, agent=agent, render_dir=render_dir
    )
    source = agent._restore_legacy_accepted_render(
        _selection(), 0, input_hash
    )
    assert source is not None
    agent._publish_furniture_checkpoint(
        furniture_selection=_selection(),
        furniture_index=0,
        input_scene_hash=input_hash,
        input_object_ids=set(base["objects"]),
        input_scene_state=base,
        legacy_source=source,
    )
    (tmp_path / "generated_assets" / "manipuland" / "gradebook.bin").write_bytes(
        b"substituted"
    )
    agent.scene.restore_from_state_dict(base)
    with pytest.raises(RuntimeError, match="referenced asset inventory mismatch"):
        agent._restore_completed_furniture_checkpoint(
            furniture_selection=_selection(),
            furniture_index=0,
            input_scene_hash=input_hash,
        )


def test_asset_changed_after_authorization_cannot_be_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _, input_hash = _authorized_legacy_case(tmp_path, monkeypatch)
    (
        tmp_path / "generated_assets" / "manipuland" / "gradebook.bin"
    ).write_bytes(b"substituted-after-authorization")
    with pytest.raises(RuntimeError, match="added asset inventory mismatch"):
        agent._restore_legacy_accepted_render(_selection(), 0, input_hash)


def test_target_transitive_asset_changed_after_authorization_cannot_be_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _, input_hash = _authorized_legacy_case(tmp_path, monkeypatch)
    (
        tmp_path / "generated_assets" / "furniture" / "teacher_desk.bin"
    ).write_bytes(b"substituted-target-after-authorization")
    with pytest.raises(RuntimeError, match="referenced asset inventory mismatch"):
        agent._restore_legacy_accepted_render(_selection(), 0, input_hash)


def test_manifest_output_hash_must_match_candidate_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, manifest, input_hash = _authorized_legacy_case(tmp_path, monkeypatch)
    _rewrite_authorization(
        manifest,
        monkeypatch,
        lambda document: document.__setitem__(
            "output_scene_content_hash", "0" * 64
        ),
    )
    with pytest.raises(RuntimeError, match="output scene content hash mismatch"):
        agent._restore_legacy_accepted_render(_selection(), 0, input_hash)


@pytest.mark.parametrize("mutation", ["authorization", "render", "asset"])
def test_legacy_source_is_revalidated_between_restore_and_checkpoint_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    agent, manifest, input_hash = _authorized_legacy_case(tmp_path, monkeypatch)
    base = _base_state()
    source = agent._restore_legacy_accepted_render(_selection(), 0, input_hash)
    assert source is not None

    if mutation == "authorization":
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        expected = "authorization manifest changed"
    elif mutation == "render":
        render_dir = tmp_path / source["render_directory"]
        (render_dir / "scene.dmd.yaml").write_text(
            "directives:\n- substituted\n", encoding="utf-8"
        )
        expected = "Drake directive hash/size mismatch"
    else:
        (
            tmp_path / "generated_assets" / "manipuland" / "gradebook.bin"
        ).write_bytes(b"substituted-after-restore")
        expected = "referenced asset inventory mismatch"

    with pytest.raises(RuntimeError, match=expected):
        agent._publish_furniture_checkpoint(
            furniture_selection=_selection(),
            furniture_index=0,
            input_scene_hash=input_hash,
            input_object_ids=set(base["objects"]),
            input_scene_state=base,
            legacy_source=source,
        )


def test_manifest_added_ids_or_assets_must_match_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, manifest, input_hash = _authorized_legacy_case(tmp_path, monkeypatch)

    def substitute_ids(document: dict) -> None:
        document["added_manipuland_ids"] = ["substituted_0"]
        document["referenced_asset_object_ids"] = [
            "substituted_0",
            "teacher_desk_0",
        ]

    _rewrite_authorization(
        manifest,
        monkeypatch,
        substitute_ids,
    )
    with pytest.raises(RuntimeError, match="added manipuland inventory mismatch"):
        agent._restore_legacy_accepted_render(_selection(), 0, input_hash)

    agent, manifest, input_hash = _authorized_legacy_case(
        tmp_path / "second_room", monkeypatch
    )
    _rewrite_authorization(
        manifest,
        monkeypatch,
        lambda document: document["added_asset_files"][0].__setitem__(
            "sha256", "0" * 64
        ),
    )
    with pytest.raises(RuntimeError, match="added asset inventory mismatch"):
        agent._restore_legacy_accepted_render(_selection(), 0, input_hash)


def test_resigned_checkpoint_cannot_substitute_manifest_bound_legacy_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _, input_hash = _authorized_legacy_case(tmp_path, monkeypatch)
    base = _base_state()
    source = agent._restore_legacy_accepted_render(_selection(), 0, input_hash)
    assert source is not None
    agent._publish_furniture_checkpoint(
        furniture_selection=_selection(),
        furniture_index=0,
        input_scene_hash=input_hash,
        input_object_ids=set(base["objects"]),
        input_scene_state=base,
        legacy_source=source,
    )
    receipt_path = (
        agent._checkpoint_directory(_selection(), 0) / "completion_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["legacy_source"]["output_scene_content_hash"] = "f" * 64
    receipt["attestation"] = checkpoint._receipt_attestation(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    agent.scene.restore_from_state_dict(base)
    with pytest.raises(RuntimeError, match="does not match authorization manifest"):
        agent._restore_completed_furniture_checkpoint(
            furniture_selection=_selection(),
            furniture_index=0,
            input_scene_hash=input_hash,
        )


def test_extra_object_after_legacy_restore_cannot_be_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _, input_hash = _authorized_legacy_case(tmp_path, monkeypatch)
    base = _base_state()
    source = agent._restore_legacy_accepted_render(_selection(), 0, input_hash)
    assert source is not None
    changed = agent.scene.to_state_dict()
    changed["objects"]["unexpected_0"] = copy.deepcopy(
        changed["objects"]["gradebook_0"]
    )
    changed["objects"]["unexpected_0"]["object_id"] = "unexpected_0"
    agent.scene.restore_from_state_dict(changed)
    with pytest.raises(RuntimeError, match="postprocessed object delta mismatch"):
        agent._publish_furniture_checkpoint(
            furniture_selection=_selection(),
            furniture_index=0,
            input_scene_hash=input_hash,
            input_object_ids=set(base["objects"]),
            input_scene_state=base,
            legacy_source=source,
        )

def test_manifest_inside_room_output_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, manifest, _ = _authorized_legacy_case(tmp_path, monkeypatch)
    inside = tmp_path / "inside_authorization.json"
    inside.write_bytes(manifest.read_bytes())
    monkeypatch.setenv(
        "SCENESMITH_LEGACY_MANIPULAND_RECOVERY_MANIFEST", str(inside)
    )
    monkeypatch.setenv(
        "SCENESMITH_LEGACY_MANIPULAND_RECOVERY_SHA256",
        checkpoint._sha256_file(inside),
    )
    with pytest.raises(RuntimeError, match="must be outside room output"):
        agent._load_explicit_legacy_manifest()


@pytest.mark.parametrize("mode", ["absent", "moved"])
def test_explicit_legacy_target_must_match_exact_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    agent, _, _ = _authorized_legacy_case(tmp_path, monkeypatch)
    chair = FurnitureSelection(
        furniture_id=UniqueID("chair_0"),
        suggested_items="",
        prompt_constraints="",
        style_notes="",
        context_furniture_ids=[],
    )
    selections = [chair] if mode == "absent" else [chair, _selection()]
    with pytest.raises(RuntimeError, match="absent or moved"):
        agent._validate_explicit_legacy_plan(selections)


def test_canonical_absolute_inside_room_is_normalized_and_unsafe_paths_fail(
    tmp_path: Path,
) -> None:
    inside = tmp_path / "assets" / "inside.bin"
    inside.parent.mkdir()
    inside.write_bytes(b"inside")
    if os.name != "nt":
        resolved, record = checkpoint._room_asset_file_record(
            inside.resolve().as_posix(),
            room_root=tmp_path,
            label="inside",
        )
        assert resolved == inside.resolve()
        assert record["path"] == "assets/inside.bin"

    for raw in (
        "./assets/inside.bin",
        "assets/../assets/inside.bin",
        "assets\\inside.bin",
    ):
        with pytest.raises(RuntimeError, match="not canonical"):
            checkpoint._room_asset_file_record(
                raw,
                room_root=tmp_path,
                label="unsafe",
            )
    outside = tmp_path.parent / f"{tmp_path.name}_outside.bin"
    outside.write_bytes(b"outside")
    with pytest.raises(RuntimeError, match="escapes the room"):
        checkpoint._room_asset_file_record(
            outside.resolve().as_posix(),
            room_root=tmp_path,
            label="outside",
        )


def test_transitive_asset_inventory_is_exact_and_ignores_unreferenced_files(
    tmp_path: Path,
) -> None:
    _write_added_assets(tmp_path)
    (tmp_path / "generated_assets" / "manipuland" / "unused.bin").write_bytes(
        b"unused"
    )
    state = _accepted_state()
    records = checkpoint._referenced_room_asset_records(
        state,
        room_root=tmp_path,
        object_ids={"gradebook_0"},
    )
    assert [record["path"] for record in records] == [
        "generated_assets/manipuland/gradebook.bin",
        "generated_assets/manipuland/gradebook.gltf",
        "generated_assets/manipuland/gradebook.sdf",
        "generated_assets/manipuland/gradebook_collision.obj",
    ]


@pytest.mark.skipif(os.name == "nt", reason="production authority uses POSIX paths")
def test_exact_scene_floor_and_project_material_dependencies_are_bound(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    scene_root = project / "outputs" / "2026-07-10" / "run" / "scene_000"
    room_root = scene_root / "room_classroom_01"
    floor_root = scene_root / "floor_plans" / "classroom_01" / "floors"
    room_geometry_root = scene_root / "room_geometry"
    materials_root = project / "materials" / "Wood"
    room_root.mkdir(parents=True)
    floor_root.mkdir(parents=True)
    room_geometry_root.mkdir(parents=True)
    materials_root.mkdir(parents=True)
    (floor_root / "floor.bin").write_bytes(b"floor-bin")
    textures = []
    for name in ("Color.jpg", "NormalGL.jpg", "Roughness.jpg"):
        texture = materials_root / name
        texture.write_bytes(name.encode())
        textures.append(texture)
    texture_uris = [
        os.path.relpath(path, floor_root).replace(os.sep, "/")
        for path in textures
    ]
    floor_gltf = floor_root / "floor.gltf"
    floor_gltf.write_text(
        json.dumps(
            {
                "buffers": [{"uri": "floor.bin"}],
                "images": [{"uri": uri} for uri in texture_uris],
            }
        ),
        encoding="utf-8",
    )
    room_sdf = room_geometry_root / "room_geometry_classroom_01.sdf"
    room_sdf.write_text(
        "<sdf><model><link><visual><geometry><mesh>"
        "<uri>../floor_plans/classroom_01/floors/floor.gltf</uri>"
        "</mesh></geometry></visual></link></model></sdf>",
        encoding="utf-8",
    )
    floor = {
        "object_id": "floor_classroom_01",
        "object_type": "floor",
        "immutable": True,
        "geometry_path": floor_gltf.as_posix(),
        "sdf_path": None,
        "metadata": {},
    }
    state = {
        "objects": {},
        "room_geometry": {"floor": floor, "sdf_path": room_sdf.as_posix()},
    }

    records = checkpoint._referenced_room_asset_records(
        state,
        room_root=room_root,
        object_ids={"floor_classroom_01"},
    )
    assert [record["path"] for record in records] == [
        "_project_materials/Wood/Color.jpg",
        "_project_materials/Wood/NormalGL.jpg",
        "_project_materials/Wood/Roughness.jpg",
        "_scene_floor_plan/floors/floor.bin",
        "_scene_floor_plan/floors/floor.gltf",
        "_scene_room_geometry/room_geometry_classroom_01.sdf",
    ]
    checkpoint._verify_referenced_asset_records(
        records,
        scene_state=state,
        room_root=room_root,
        object_ids={"floor_classroom_01"},
    )
    textures[0].write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="asset inventory mismatch"):
        checkpoint._verify_referenced_asset_records(
            records,
            scene_state=state,
            room_root=room_root,
            object_ids={"floor_classroom_01"},
        )


@pytest.mark.skipif(os.name == "nt", reason="production authority uses POSIX paths")
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("ordinary_object", "escapes the room"),
        ("wrong_floor_id", "does not match the exact room floor"),
        ("wrong_floor_type", "does not match the exact room floor"),
        ("mutable_floor", "does not match the exact room floor"),
        ("cross_room", "not the exact current-room floor.gltf"),
        ("material_escape", "escapes exact project materials"),
    ],
)
def test_scene_floor_authority_cannot_be_claimed_or_escape(
    tmp_path: Path, mutation: str, message: str
) -> None:
    project = tmp_path / "project"
    scene_root = project / "outputs" / "2026-07-10" / "run" / "scene_000"
    room_root = scene_root / "room_classroom_01"
    floor_root = scene_root / "floor_plans" / "classroom_01" / "floors"
    other_floor_root = scene_root / "floor_plans" / "classroom_02" / "floors"
    room_geometry_root = scene_root / "room_geometry"
    materials_root = project / "materials"
    for directory in (
        room_root,
        floor_root,
        other_floor_root,
        room_geometry_root,
        materials_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    (floor_root / "floor.bin").write_bytes(b"floor-bin")
    (other_floor_root / "floor.gltf").write_text("{}", encoding="utf-8")
    outside = project / "outside.jpg"
    outside.write_bytes(b"outside")
    floor_gltf = floor_root / "floor.gltf"
    document: dict = {"buffers": [{"uri": "floor.bin"}]}
    if mutation == "material_escape":
        document["images"] = [
            {"uri": os.path.relpath(outside, floor_root).replace(os.sep, "/")}
        ]
    floor_gltf.write_text(json.dumps(document), encoding="utf-8")
    room_sdf = room_geometry_root / "room_geometry_classroom_01.sdf"
    room_sdf.write_text(
        "<sdf><model><link><visual><geometry><mesh>"
        "<uri>../floor_plans/classroom_01/floors/floor.gltf</uri>"
        "</mesh></geometry></visual></link></model></sdf>",
        encoding="utf-8",
    )
    floor = {
        "object_id": "floor_classroom_01",
        "object_type": "floor",
        "immutable": True,
        "geometry_path": floor_gltf.as_posix(),
        "sdf_path": None,
        "metadata": {},
    }
    state = {
        "objects": {},
        "room_geometry": {"floor": floor, "sdf_path": room_sdf.as_posix()},
    }
    selected = {"floor_classroom_01"}
    if mutation == "ordinary_object":
        state["objects"]["cabinet_0"] = {
            "object_id": "cabinet_0",
            "object_type": "furniture",
            "geometry_path": floor_gltf.as_posix(),
            "sdf_path": None,
            "metadata": {},
        }
        selected = {"cabinet_0"}
    elif mutation == "wrong_floor_id":
        floor["object_id"] = "floor_classroom_02"
        selected = {"floor_classroom_02"}
    elif mutation == "wrong_floor_type":
        floor["object_type"] = "furniture"
    elif mutation == "mutable_floor":
        floor["immutable"] = False
    elif mutation == "cross_room":
        floor["geometry_path"] = (other_floor_root / "floor.gltf").as_posix()

    with pytest.raises(RuntimeError, match=message):
        checkpoint._referenced_room_asset_records(
            state, room_root=room_root, object_ids=selected
        )


@pytest.mark.parametrize("uri", ["a//b.bin", "a/./b.bin", "a/b.bin/"])
def test_noncanonical_dependency_uri_is_rejected(uri: str) -> None:
    with pytest.raises(RuntimeError, match="unsafe or unsupported"):
        checkpoint._dependency_relative_path(
            uri,
            owner_relative=PurePosixPath("asset/model.gltf"),
            label="test",
            allowed_suffixes=frozenset({".bin"}),
        )


def test_sdf_dependency_cycle_and_unsupported_gltf_uri_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "assets"
    root.mkdir()
    (root / "a.sdf").write_text(
        "<sdf><include><uri>b.sdf</uri></include></sdf>", encoding="utf-8"
    )
    (root / "b.sdf").write_text(
        "<sdf><include><uri>a.sdf</uri></include></sdf>", encoding="utf-8"
    )
    state = {
        "objects": {
            "item": {
                "geometry_path": None,
                "sdf_path": "assets/a.sdf",
                "metadata": {},
            }
        }
    }
    with pytest.raises(RuntimeError, match="dependency cycle"):
        checkpoint._referenced_room_asset_records(
            state, room_root=tmp_path, object_ids={"item"}
        )

    (root / "bad.gltf").write_text(
        json.dumps({"buffers": [{"uri": "nested.gltf"}]}), encoding="utf-8"
    )
    state["objects"]["item"] = {
        "geometry_path": "assets/bad.gltf",
        "sdf_path": None,
        "metadata": {},
    }
    with pytest.raises(RuntimeError, match="unsafe or unsupported"):
        checkpoint._referenced_room_asset_records(
            state, room_root=tmp_path, object_ids={"item"}
        )


def test_referenced_glb_is_rejected_without_an_exact_dependency_audit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "assets"
    root.mkdir()
    (root / "model.glb").write_bytes(b"glTF" + b"\x00" * 16)
    state = {
        "objects": {
            "item": {
                "geometry_path": "assets/model.glb",
                "sdf_path": None,
                "metadata": {},
            }
        }
    }
    with pytest.raises(RuntimeError, match="unsupported direct asset suffix"):
        checkpoint._referenced_room_asset_records(
            state, room_root=tmp_path, object_ids={"item"}
        )

    (root / "model.sdf").write_text(
        "<sdf><model><link><visual><geometry><mesh>"
        "<uri>model.glb</uri></mesh></geometry></visual></link></model></sdf>",
        encoding="utf-8",
    )
    state["objects"]["item"] = {
        "geometry_path": None,
        "sdf_path": "assets/model.sdf",
        "metadata": {},
    }
    with pytest.raises(RuntimeError, match="unsafe or unsupported"):
        checkpoint._referenced_room_asset_records(
            state, room_root=tmp_path, object_ids={"item"}
        )


@pytest.mark.parametrize("suffix", [".dae", ".usd", ".foo"])
def test_unknown_direct_geometry_suffix_is_rejected(
    tmp_path: Path, suffix: str
) -> None:
    asset = tmp_path / f"model{suffix}"
    asset.write_bytes(b"external-dependency-capable")
    state = {
        "objects": {
            "item": {
                "geometry_path": asset.name,
                "sdf_path": None,
                "metadata": {},
            }
        }
    }
    with pytest.raises(RuntimeError, match="unsupported direct asset suffix"):
        checkpoint._referenced_room_asset_records(
            state, room_root=tmp_path, object_ids={"item"}
        )


def test_sdf_dae_and_unconsumed_uri_nodes_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "model.dae").write_text("<COLLADA/>", encoding="utf-8")
    sdf = tmp_path / "model.sdf"
    state = {
        "objects": {
            "item": {
                "geometry_path": None,
                "sdf_path": "model.sdf",
                "metadata": {},
            }
        }
    }
    sdf.write_text(
        "<sdf><model><link><visual><geometry><mesh><uri>model.dae</uri>"
        "</mesh></geometry></visual></link></model></sdf>",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="unsafe or unsupported"):
        checkpoint._referenced_room_asset_records(
            state, room_root=tmp_path, object_ids={"item"}
        )

    (tmp_path / "terrain.png").write_bytes(b"png")
    sdf.write_text(
        "<sdf><model><link><visual><geometry><heightmap>"
        "<uri>terrain.png</uri></heightmap></geometry></visual></link></model></sdf>",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="unsupported URI elements"):
        checkpoint._referenced_room_asset_records(
            state, room_root=tmp_path, object_ids={"item"}
        )


@pytest.mark.asyncio
async def test_required_legacy_recovery_must_be_consumed_before_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, _base_state())
    agent.rendering_manager = SimpleNamespace(clear_cache=lambda: None)
    agent._load_furniture_plan = lambda _: [_selection()]

    def require_exact_recovery(_selections) -> None:
        agent._required_legacy_recovery = (0, "teacher_desk_0", "a" * 64)
        agent._legacy_recovery_satisfied = False

    agent._validate_explicit_legacy_plan = require_exact_recovery
    agent._restore_completed_furniture_checkpoint = lambda **_: True
    monkeypatch.setattr(checkpoint, "custom_span", lambda **_: nullcontext())
    monkeypatch.setattr(
        agent.scene,
        "get_object",
        lambda object_id: agent.scene.to_state_dict()["objects"].get(str(object_id)),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="not consumed or restored"):
        await agent.add_manipulands(agent.scene)


def test_room_asset_hardlink_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    alias = tmp_path / "alias.bin"
    source.write_bytes(b"hardlink")
    try:
        os.link(source, alias)
    except OSError as exc:
        pytest.skip(f"hardlink unavailable: {exc}")
    with pytest.raises(RuntimeError, match="hard-linked"):
        checkpoint._room_asset_file_record(
            "source.bin", room_root=tmp_path, label="hardlink"
        )


def test_normal_checkpoint_revalidates_transitive_asset_bytes(tmp_path: Path) -> None:
    _write_evidence(
        tmp_path / "scene_states" / "manipuland_furniture_teacher_desk_0"
    )
    base = _base_state()
    agent = _agent(tmp_path, _accepted_state())
    input_hash = _FakeScene(base, tmp_path).content_hash()
    agent._publish_furniture_checkpoint(
        furniture_selection=_selection(),
        furniture_index=0,
        input_scene_hash=input_hash,
        input_object_ids=set(base["objects"]),
        input_scene_state=base,
        legacy_source=None,
    )
    (
        tmp_path / "generated_assets" / "manipuland" / "gradebook.bin"
    ).write_bytes(b"changed")
    agent.scene.restore_from_state_dict(base)
    with pytest.raises(RuntimeError, match="referenced asset inventory mismatch"):
        agent._restore_completed_furniture_checkpoint(
            furniture_selection=_selection(),
            furniture_index=0,
            input_scene_hash=input_hash,
        )


@pytest.mark.parametrize("mutation", ["extra", "removed_input"])
def test_resigned_checkpoint_state_requires_exact_live_input_delta(
    tmp_path: Path, mutation: str
) -> None:
    _write_evidence(
        tmp_path / "scene_states" / "manipuland_furniture_teacher_desk_0"
    )
    base = _base_state()
    agent = _agent(tmp_path, _accepted_state())
    input_hash = _FakeScene(base, tmp_path).content_hash()
    agent._publish_furniture_checkpoint(
        furniture_selection=_selection(),
        furniture_index=0,
        input_scene_hash=input_hash,
        input_object_ids=set(base["objects"]),
        input_scene_state=base,
        legacy_source=None,
    )
    directory = agent._checkpoint_directory(_selection(), 0)
    state_path = directory / "scene_state.json"
    receipt_path = directory / "completion_receipt.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if mutation == "extra":
        state["objects"]["unlisted_0"] = copy.deepcopy(
            state["objects"]["gradebook_0"]
        )
        state["objects"]["unlisted_0"]["object_id"] = "unlisted_0"
    else:
        del state["objects"]["chair_0"]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifacts"]["scene_state"] = checkpoint._file_record(state_path)
    receipt["output_scene_content_hash"] = _FakeScene(state, tmp_path).content_hash()
    receipt["attestation"] = checkpoint._receipt_attestation(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    agent.scene.restore_from_state_dict(base)
    with pytest.raises(RuntimeError, match="restored object delta mismatch"):
        agent._restore_completed_furniture_checkpoint(
            furniture_selection=_selection(),
            furniture_index=0,
            input_scene_hash=input_hash,
        )


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_checkpoint_referenced_asset_inventory_is_exact(
    tmp_path: Path, mutation: str
) -> None:
    _write_evidence(
        tmp_path / "scene_states" / "manipuland_furniture_teacher_desk_0"
    )
    base = _base_state()
    agent = _agent(tmp_path, _accepted_state())
    input_hash = _FakeScene(base, tmp_path).content_hash()
    agent._publish_furniture_checkpoint(
        furniture_selection=_selection(),
        furniture_index=0,
        input_scene_hash=input_hash,
        input_object_ids=set(base["objects"]),
        input_scene_state=base,
        legacy_source=None,
    )
    receipt_path = (
        agent._checkpoint_directory(_selection(), 0) / "completion_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        receipt["referenced_asset_files"].pop()
    else:
        receipt["referenced_asset_files"].append(
            {
                "path": "substituted.bin",
                "name": "substituted.bin",
                "size_bytes": 1,
                "sha256": "0" * 64,
            }
        )
    receipt["attestation"] = checkpoint._receipt_attestation(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    agent.scene.restore_from_state_dict(base)
    with pytest.raises(RuntimeError, match="referenced asset inventory mismatch"):
        agent._restore_completed_furniture_checkpoint(
            furniture_selection=_selection(),
            furniture_index=0,
            input_scene_hash=input_hash,
        )


def test_stale_copied_evidence_and_dropped_render_object_fail_publication(
    tmp_path: Path,
) -> None:
    accepted_dir = (
        tmp_path / "scene_states" / "manipuland_furniture_teacher_desk_0"
    )
    _write_evidence(accepted_dir)
    base = _base_state()
    agent = _agent(tmp_path, _accepted_state())
    (accepted_dir / "0_side.png").write_bytes(b"stale-substitution")
    with pytest.raises(RuntimeError, match="does not match current accepted render"):
        agent._publish_furniture_checkpoint(
            furniture_selection=_selection(),
            furniture_index=0,
            input_scene_hash=_FakeScene(base, tmp_path).content_hash(),
            input_object_ids=set(base["objects"]),
            input_scene_state=base,
            legacy_source=None,
        )

    second = tmp_path / "second"
    second_accepted = (
        second / "scene_states" / "manipuland_furniture_teacher_desk_0"
    )
    _write_evidence(second_accepted)
    agent = _agent(second, _accepted_state())
    (agent.final_render_dir / "scene_state.json").write_text(
        json.dumps(base), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="did not survive post-processing"):
        agent._publish_furniture_checkpoint(
            furniture_selection=_selection(),
            furniture_index=0,
            input_scene_hash=_FakeScene(base, second).content_hash(),
            input_object_ids=set(base["objects"]),
            input_scene_state=base,
            legacy_source=None,
        )


def test_checkpoint_publication_rejects_semantic_mutation_of_prior_object(
    tmp_path: Path,
) -> None:
    accepted_dir = (
        tmp_path / "scene_states" / "manipuland_furniture_teacher_desk_0"
    )
    _write_evidence(accepted_dir)
    base = _base_state()
    output = _accepted_state()
    output["objects"]["chair_0"]["metadata"]["scope_poison"] = True
    agent = _agent(tmp_path, output)
    with pytest.raises(RuntimeError, match="mutated protected object chair_0"):
        agent._publish_furniture_checkpoint(
            furniture_selection=_selection(),
            furniture_index=0,
            input_scene_hash=_FakeScene(base, tmp_path).content_hash(),
            input_object_ids=set(base["objects"]),
            input_scene_state=base,
            legacy_source=None,
        )


@pytest.mark.parametrize("runtime", ["predecessor", "arbitrary"])
def test_fabricated_or_arbitrary_checkpoint_runtime_is_rejected(
    tmp_path: Path, runtime: str
) -> None:
    _write_evidence(
        tmp_path / "scene_states" / "manipuland_furniture_teacher_desk_0"
    )
    base = _base_state()
    agent = _agent(tmp_path, _accepted_state())
    input_hash = _FakeScene(base, tmp_path).content_hash()
    agent._publish_furniture_checkpoint(
        furniture_selection=_selection(),
        furniture_index=0,
        input_scene_hash=input_hash,
        input_object_ids=set(base["objects"]),
        input_scene_state=base,
        legacy_source=None,
    )
    receipt_path = (
        agent._checkpoint_directory(_selection(), 0) / "completion_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["schema_version"] = 2
    receipt.pop("asset_registry_snapshot")
    receipt["checkpoint_runtime_sha256"] = (
        checkpoint._SCOPE_PREDECESSOR_RUNTIME_SHA256
        if runtime == "predecessor"
        else "0" * 64
    )
    receipt["attestation"] = checkpoint._receipt_attestation(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    agent.scene.restore_from_state_dict(base)
    with pytest.raises(RuntimeError, match="checkpoint_runtime_sha256 mismatch"):
        agent._restore_completed_furniture_checkpoint(
            furniture_selection=_selection(),
            furniture_index=0,
            input_scene_hash=input_hash,
        )


def test_schema3_publication_binds_and_continues_from_real_serialized_roundtrip(
    tmp_path: Path,
) -> None:
    agent, base_state, input_hash = _real_roomscene_checkpoint_agent(tmp_path)
    agent._publish_furniture_checkpoint(
        furniture_selection=_selection(),
        furniture_index=0,
        input_scene_hash=input_hash,
        input_object_ids=set(base_state["objects"]),
        input_scene_state=base_state,
        legacy_source=None,
    )
    directory = agent._checkpoint_directory(_selection(), 0)
    state = json.loads((directory / "scene_state.json").read_text(encoding="utf-8"))
    receipt = json.loads(
        (directory / "completion_receipt.json").read_text(encoding="utf-8")
    )
    restored = RoomScene(
        room_geometry=None,
        scene_dir=tmp_path,
        room_id="classroom_01",
    )
    restored.restore_from_state_dict(state)
    assert restored.content_hash() == receipt["output_scene_content_hash"]
    assert agent.scene.content_hash() == receipt["output_scene_content_hash"]
    agent.scene.restore_from_state_dict(base_state)
    assert agent._restore_completed_furniture_checkpoint(
        furniture_selection=_selection(),
        furniture_index=0,
        input_scene_hash=input_hash,
    )
    assert agent.scene.content_hash() == receipt["output_scene_content_hash"]


def test_authorized_input_uses_exact_roomscene_loader_roundtrip(
    tmp_path: Path,
) -> None:
    scene = RoomScene(
        room_geometry=None,
        scene_dir=tmp_path,
        room_id="classroom_01",
    )
    desk = SceneObject(
        object_id=UniqueID("teacher_desk_0"),
        object_type=ObjectType.FURNITURE,
        name="teacher desk",
        description="authorized teacher desk",
        transform=RigidTransform(),
        bbox_min=np.array([-0.5, -0.3, 0.0]),
        bbox_max=np.array([0.5, 0.3, 0.8]),
    )
    scene.add_object(desk)
    recorded = scene.to_state_dict()
    recorded["objects"]["teacher_desk_0"]["transform"]["rotation_wxyz"] = [
        0.9999557143890737,
        0.0,
        0.0,
        0.009411124302495237,
    ]
    scene.restore_from_state_dict(recorded)
    live = scene.to_state_dict()
    assert recorded != live
    assert (
        checkpoint._serialized_scene_roundtrip_state(scene, recorded)
        == live
    )

    substituted = copy.deepcopy(recorded)
    substituted["objects"]["teacher_desk_0"]["description"] = "substituted"
    assert (
        checkpoint._serialized_scene_roundtrip_state(scene, substituted)
        != live
    )

    removed = copy.deepcopy(recorded)
    del removed["objects"]["teacher_desk_0"]
    assert checkpoint._serialized_scene_roundtrip_state(scene, removed) != live


def test_completed_legacy_checkpoint_normalizes_only_pinned_quaternion_spelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, base_state, _ = _real_roomscene_checkpoint_agent(tmp_path)
    accepted_state = agent.scene.to_state_dict()
    raw_quaternion = [
        0.9999557143890737,
        0.0,
        0.0,
        0.009411124302495237,
    ]
    for state in (base_state, accepted_state):
        state["objects"]["teacher_desk_0"]["transform"][
            "rotation_wxyz"
        ] = list(raw_quaternion)

    render_dir = (
        tmp_path
        / "scene_renders"
        / "manipulands_teacher_desk_0"
        / "renders_003"
    )
    _write_evidence(render_dir, state=accepted_state)
    agent.scene.restore_from_state_dict(base_state)
    live_input_state = agent.scene.to_state_dict()
    assert base_state != live_input_state
    input_hash = agent.scene.content_hash()
    manifest_path = _authorize_legacy_recovery(
        tmp_path,
        monkeypatch,
        agent=agent,
        render_dir=render_dir,
    )
    input_path = (
        tmp_path
        / "scene_states"
        / "scene_after_ceiling_objects"
        / "scene_state.json"
    )
    input_path.write_text(json.dumps(base_state), encoding="utf-8")
    normalized_output = RoomScene(
        room_geometry=None,
        scene_dir=tmp_path,
        room_id="classroom_01",
    )
    normalized_output.restore_from_state_dict(accepted_state)

    def bind_raw_input(document: dict) -> None:
        document["input_scene_state"] = checkpoint._file_record(input_path)
        document["output_scene_content_hash"] = normalized_output.content_hash()

    _rewrite_authorization(manifest_path, monkeypatch, bind_raw_input)
    source = agent._restore_legacy_accepted_render(
        _selection(), 0, input_hash
    )
    assert source is not None
    agent._publish_furniture_checkpoint(
        furniture_selection=_selection(),
        furniture_index=0,
        input_scene_hash=input_hash,
        input_object_ids=set(live_input_state["objects"]),
        input_scene_state=live_input_state,
        legacy_source=source,
    )
    agent.scene.restore_from_state_dict(base_state)
    assert agent._restore_completed_furniture_checkpoint(
        furniture_selection=_selection(),
        furniture_index=0,
        input_scene_hash=input_hash,
    )

    # Even a fully re-recorded/re-attested authorization cannot use loader
    # normalization to hide a non-transform semantic substitution.
    substituted_input = copy.deepcopy(base_state)
    substituted_input["objects"]["teacher_desk_0"]["description"] = (
        "substituted teacher desk"
    )
    input_path.write_text(json.dumps(substituted_input), encoding="utf-8")
    _rewrite_authorization(
        manifest_path,
        monkeypatch,
        lambda document: document.__setitem__(
            "input_scene_state", checkpoint._file_record(input_path)
        ),
    )
    receipt_path = (
        agent._checkpoint_directory(_selection(), 0)
        / "completion_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["legacy_source"]["authorization"] = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "manifest_sha256": checkpoint._sha256_file(manifest_path),
    }
    receipt["attestation"] = checkpoint._receipt_attestation(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    agent.scene.restore_from_state_dict(base_state)
    with pytest.raises(RuntimeError, match="input state does not match live scene"):
        agent._restore_completed_furniture_checkpoint(
            furniture_selection=_selection(),
            furniture_index=0,
            input_scene_hash=input_hash,
        )


def test_schema3_checkpoint_pins_registry_head_and_allows_exact_descendant(
    tmp_path: Path,
) -> None:
    agent, base_state, input_hash = _real_roomscene_checkpoint_agent(tmp_path)
    root = tmp_path / "generated_assets" / "manipuland"
    registry_path = root / "asset_registry.json"
    registry = AssetRegistry(
        auto_save_path=registry_path,
        required_root=root,
        allowed_object_types=frozenset({ObjectType.MANIPULAND}),
    )

    def register_template(object_id: str) -> None:
        directory = root / "sdf" / f"{object_id}_namespace"
        directory.mkdir(parents=True)
        geometry = directory / f"{object_id}.gltf"
        sdf = directory / f"{object_id}.sdf"
        geometry.write_text("{}\n", encoding="utf-8")
        sdf.write_text(
            "<sdf version='1.7'><model name='asset'><link name='base'/></model></sdf>",
            encoding="utf-8",
        )
        registry.register(
            SceneObject(
                object_id=UniqueID(object_id),
                object_type=ObjectType.MANIPULAND,
                name=object_id,
                description=f"reusable {object_id}",
                transform=RigidTransform(),
                geometry_path=geometry,
                sdf_path=sdf,
                metadata={"asset_source": "generated"},
                bbox_min=np.array([-0.1, -0.1, 0.0]),
                bbox_max=np.array([0.1, 0.1, 0.1]),
            )
        )

    register_template("checkpoint_book_0")
    agent.asset_manager = SimpleNamespace(
        registry=registry,
        reconcile_registry_with_scene=lambda **_kwargs: None,
    )
    agent._publish_furniture_checkpoint(
        furniture_selection=_selection(),
        furniture_index=0,
        input_scene_hash=input_hash,
        input_object_ids=set(base_state["objects"]),
        input_scene_state=base_state,
        legacy_source=None,
    )
    directory = agent._checkpoint_directory(_selection(), 0)
    receipt_path = directory / "completion_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    pinned = receipt["asset_registry_snapshot"]
    assert pinned == registry.snapshot()
    assert pinned["head"]["revision"] == 1

    register_template("later_folder_0")
    agent.scene.restore_from_state_dict(base_state)
    assert agent._restore_completed_furniture_checkpoint(
        furniture_selection=_selection(),
        furniture_index=0,
        input_scene_hash=input_hash,
    )

    receipt["asset_registry_snapshot"]["head"]["attestation"] = "f" * 64
    receipt["attestation"] = checkpoint._receipt_attestation(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    agent.scene.restore_from_state_dict(base_state)
    with pytest.raises(RuntimeError, match="head|ancestor"):
        agent._restore_completed_furniture_checkpoint(
            furniture_selection=_selection(),
            furniture_index=0,
            input_scene_hash=input_hash,
        )

def test_exact_schema2_migration_branch_uses_real_roundtrip_hash_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, base_state, input_hash = _real_roomscene_checkpoint_agent(tmp_path)
    agent._publish_furniture_checkpoint(
        furniture_selection=_selection(),
        furniture_index=0,
        input_scene_hash=input_hash,
        input_object_ids=set(base_state["objects"]),
        input_scene_state=base_state,
        legacy_source=None,
    )
    directory = agent._checkpoint_directory(_selection(), 0)
    state_path = directory / "scene_state.json"
    directive_path = directory / "scene.dmd.yaml"
    receipt_path = directory / "completion_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    stable_hash = receipt["output_scene_content_hash"]
    legacy_pre_serialization_hash = "1" * 64
    receipt["schema_version"] = 2
    receipt.pop("asset_registry_snapshot")
    receipt["checkpoint_runtime_sha256"] = (
        checkpoint._SCOPE_PREDECESSOR_RUNTIME_SHA256
    )
    receipt["output_scene_content_hash"] = legacy_pre_serialization_hash
    receipt["attestation"] = checkpoint._receipt_attestation(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(
        checkpoint, "_SCOPE_PREDECESSOR_INPUT_SCENE_CONTENT_SHA256", input_hash
    )
    monkeypatch.setattr(
        checkpoint,
        "_SCOPE_PREDECESSOR_OUTPUT_SCENE_CONTENT_SHA256",
        legacy_pre_serialization_hash,
    )
    monkeypatch.setattr(
        checkpoint,
        "_SCOPE_PREDECESSOR_RECEIPT_SHA256",
        checkpoint._sha256_file(receipt_path),
    )
    monkeypatch.setattr(
        checkpoint,
        "_SCOPE_PREDECESSOR_SCENE_STATE_SHA256",
        checkpoint._sha256_file(state_path),
    )
    monkeypatch.setattr(
        checkpoint,
        "_SCOPE_PREDECESSOR_DRAKE_DIRECTIVE_SHA256",
        checkpoint._sha256_file(directive_path),
    )
    monkeypatch.setattr(
        checkpoint, "_SCOPE_PREDECESSOR_ROUNDTRIP_CONTENT_SHA256", stable_hash
    )
    agent.scene.restore_from_state_dict(base_state)
    assert agent._restore_completed_furniture_checkpoint(
        furniture_selection=_selection(),
        furniture_index=0,
        input_scene_hash=input_hash,
    )
    assert agent.scene.content_hash() == stable_hash


def test_room_geometry_floor_checkpoint_publishes_and_restores(
    tmp_path: Path,
) -> None:
    floor_assets = tmp_path / "room_geometry"
    floor_assets.mkdir()
    (floor_assets / "floor.sdf").write_text(
        "<sdf version='1.7'><model name='floor'><link name='base'>"
        "<collision><geometry><box><size>10 10 0.1</size></box></geometry>"
        "</collision></link></model></sdf>",
        encoding="utf-8",
    )
    base = _base_state()
    del base["objects"]["teacher_desk_0"]
    floor = {
        "object_id": "floor_classroom_01",
        "object_type": "floor",
        "name": "floor",
        "description": "classroom floor",
        "transform": {
            "translation": [0, 0, 0],
            "rotation_wxyz": [1, 0, 0, 0],
        },
        "geometry_path": None,
        "sdf_path": "room_geometry/floor.sdf",
        "image_path": None,
        "support_surfaces": [],
        "placement_info": None,
        "metadata": {"authority": "room_geometry"},
        "bbox_min": [-5, -5, -0.1],
        "bbox_max": [5, 5, 0],
        "immutable": True,
        "scale_factor": 1.0,
    }
    base["room_geometry"] = {
        "floor": floor,
        "walls": [],
        "sdf_path": "room_geometry/floor.sdf",
        "width": 10.0,
        "length": 10.0,
    }
    output = copy.deepcopy(base)
    output_floor = output["room_geometry"]["floor"]
    output_floor["support_surfaces"] = [{"surface_id": "S_floor"}]
    added = copy.deepcopy(_accepted_state()["objects"]["gradebook_0"])
    added["placement_info"]["parent_surface_id"] = "S_floor"
    output["objects"]["gradebook_0"] = added
    selection = FurnitureSelection(
        furniture_id=UniqueID("floor_classroom_01"),
        suggested_items="one floor item",
        prompt_constraints="keep paths clear",
        style_notes="school",
        context_furniture_ids=[UniqueID("chair_0")],
    )
    _write_evidence(
        tmp_path / "scene_states" / "manipuland_furniture_floor_classroom_01"
    )
    agent = _agent(tmp_path, output)
    input_hash = _FakeScene(base, tmp_path).content_hash()
    agent._publish_furniture_checkpoint(
        furniture_selection=selection,
        furniture_index=17,
        input_scene_hash=input_hash,
        input_object_ids=set(base["objects"]),
        input_scene_state=base,
        legacy_source=None,
    )
    receipt = json.loads(
        (agent._checkpoint_directory(selection, 17) / "completion_receipt.json")
        .read_text(encoding="utf-8")
    )
    assert receipt["schema_version"] == 3
    assert receipt["referenced_asset_object_ids"] == [
        "floor_classroom_01",
        "gradebook_0",
    ]
    agent.scene.restore_from_state_dict(base)
    assert agent._restore_completed_furniture_checkpoint(
        furniture_selection=selection,
        furniture_index=17,
        input_scene_hash=input_hash,
    )
    restored = agent.scene.to_state_dict()
    assert "gradebook_0" in restored["objects"]
    assert restored["room_geometry"]["floor"]["support_surfaces"] == [
        {"surface_id": "S_floor"}
    ]
def test_explicit_optional_zero_surface_target_is_allowed_to_noop() -> None:
    optional = FurnitureSelection(
        furniture_id="whiteboard_0",
        suggested_items="Optional: dry-erase markers",
        prompt_constraints=(
            "No specific manipulands required for the rolling whiteboard beyond "
            "general classroom realism."
        ),
        style_notes="Keep minimal.",
    )
    required = FurnitureSelection(
        furniture_id="teacher_desk_0",
        suggested_items="Required: attendance sheet",
        prompt_constraints="Place required classroom items on the desk.",
        style_notes="",
    )

    assert StatefulManipulandAgent._allows_noop_without_support_surfaces(optional)
    assert not StatefulManipulandAgent._allows_noop_without_support_surfaces(required)


def test_required_none_inventory_allows_zero_surface_target_to_noop() -> None:
    optional = FurnitureSelection(
        furniture_id="bulletin_board_0",
        suggested_items=(
            "REQUIRED: none. Optional: school hygiene poster, pinned notices"
        ),
        prompt_constraints="No specific requirements for this surface.",
        style_notes="Keep tidy.",
    )
    required = FurnitureSelection(
        furniture_id="bulletin_board_0",
        suggested_items="REQUIRED: hygiene poster. Optional: pinned notices",
        prompt_constraints="Keep tidy.",
        style_notes="",
    )

    assert StatefulManipulandAgent._allows_noop_without_support_surfaces(optional)
    assert not StatefulManipulandAgent._allows_noop_without_support_surfaces(required)
