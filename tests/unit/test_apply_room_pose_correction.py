import copy
import hashlib
import json
from pathlib import Path

import pytest

import scripts.apply_room_pose_correction as correction


ROOM_ID = "classroom_01"


def _transform(x, y, z, rotation=None):
    return {
        "translation": [float(x), float(y), float(z)],
        "rotation_wxyz": list(rotation or [1.0, 0.0, 0.0, 0.0]),
    }


def _object(
    object_id,
    object_type,
    transform,
    *,
    parent_surface_id=None,
    support_surfaces=None,
    metadata=None,
):
    placement = None
    if parent_surface_id is not None:
        placement = {
            "parent_surface_id": parent_surface_id,
            "position_2d": [0.0, 0.0],
            "rotation_2d": 0.0,
        }
    return {
        "object_id": object_id,
        "object_type": object_type,
        "name": object_id,
        "description": object_id,
        "transform": transform,
        "placement_info": placement,
        "support_surfaces": support_surfaces or [],
        "metadata": metadata or {},
    }


def _source_state():
    root_surface = {
        "surface_id": "root_top",
        "transform": _transform(1.0, 2.0, 1.0),
    }
    child_surface = {
        "surface_id": "child_top",
        "transform": _transform(1.2, 2.1, 1.2),
    }
    objects = {
        "cabinet": _object(
            "cabinet",
            "furniture",
            _transform(1.0, 2.0, 0.0),
            support_surfaces=[root_surface],
        ),
        "box": _object(
            "box",
            "manipuland",
            _transform(1.2, 2.1, 1.0),
            parent_surface_id="root_top",
            support_surfaces=[child_surface],
        ),
        "pile": _object(
            "pile",
            "manipuland",
            _transform(1.3, 2.2, 1.2),
            parent_surface_id="child_top",
            metadata={
                "composite_type": "pile",
                "member_assets": [
                    {"asset_id": "a", "transform": _transform(1.3, 2.2, 1.2)},
                    {"asset_id": "b", "transform": _transform(1.4, 2.3, 1.3)},
                ],
            },
        ),
        "filled": _object(
            "filled",
            "manipuland",
            _transform(1.5, 2.4, 1.0),
            parent_surface_id="root_top",
            metadata={
                "composite_type": "filled_container",
                "container_asset": {
                    "asset_id": "container",
                    "transform": _transform(1.5, 2.4, 1.0),
                },
                "fill_assets": [
                    {"asset_id": "fill", "transform": _transform(1.5, 2.4, 1.1)}
                ],
            },
        ),
        "calendar": {
            **_object(
                "calendar",
                "wall_mounted",
                _transform(5.0, 6.0, 1.65, [0.5, 0.5, 0.5, 0.5]),
                parent_surface_id="south_wall",
            ),
            "placement_info": {
                "parent_surface_id": "south_wall",
                "position_2d": [1.57, 1.65],
                "rotation_2d": 0.002,
            },
            "bbox_min": [-0.5, -0.05, -0.4],
            "bbox_max": [0.5, 0.05, 0.4],
        },
        "unrelated": _object(
            "unrelated", "furniture", _transform(-3.0, 4.0, 0.0)
        ),
    }
    return {
        "timestamp": "2026-07-12T00:00:00Z",
        "text_description": "test classroom",
        "room_geometry": {},
        "objects": objects,
    }


def _spec(source_sha256, house_layout_sha256="1" * 64):
    return {
        "schema_id": correction.SCHEMA_ID,
        "schema_version": correction.SCHEMA_VERSION,
        "room_id": ROOM_ID,
        "source_state_sha256": source_sha256,
        "house_layout_sha256": house_layout_sha256,
        "corrections": [
            {
                "object_id": "cabinet",
                "object_type": "furniture",
                "mode": "world_translation",
                "expected_translation": [1.0, 2.0, 0.0],
                "target_translation": [2.0, 3.0, 0.5],
            },
            {
                "object_id": "calendar",
                "object_type": "wall_mounted",
                "mode": "wall_local",
                "expected_parent_surface_id": "south_wall",
                "expected_position_2d": [1.57, 1.65],
                "target_position_2d": [1.57, 2.5],
                "expected_rotation_2d": 0.002,
            },
        ],
    }


def _wall_resolver(state, item):
    record = state["objects"][item["object_id"]]
    value = copy.deepcopy(record["transform"])
    value["translation"][2] = item["target_position_2d"][1]
    # Exercise the production adapter's harmless quaternion roundoff path.
    value["rotation_wxyz"][0] += 1e-11
    return value


class FakeRuntime:
    def __init__(self, collision_predicate=None):
        self.collision_predicate = collision_predicate

    def resolve_wall_pose(self, state, item):
        return _wall_resolver(state, item)

    def collisions(self, state):
        if self.collision_predicate and self.collision_predicate(state):
            return [{"description": "synthetic collision"}]
        return []

    def roundtrip(self, state):
        return copy.deepcopy(state)

    def dmd(self, state):
        return "directives:\n- add_model: {}\n"

    def sceneeval(self, state):
        return {"format": "SceneEval", "object_ids": sorted(state["objects"])}


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _make_run(tmp_path):
    room = tmp_path / "run" / "scenes" / f"room_{ROOM_ID}"
    final = room / "scene_states" / "final_scene"
    final.mkdir(parents=True)
    source = _source_state()
    state_path = final / "scene_state.json"
    _write_json(state_path, source)
    (final / "scene.blend").write_bytes(b"current blend")
    digest = hashlib.sha256(state_path.read_bytes()).hexdigest()
    house_layout = tmp_path / "house_layout.json"
    _write_json(house_layout, {"rooms": [], "placed_rooms": []})
    house_digest = hashlib.sha256(house_layout.read_bytes()).hexdigest()
    spec_path = tmp_path / "pose_correction.json"
    _write_json(spec_path, _spec(digest, house_digest))
    return room, final, state_path, spec_path, source, house_layout


def test_translation_propagates_support_descendants_and_composites():
    source = _source_state()
    before = copy.deepcopy(source)
    result = correction.apply_corrections_to_state(
        source,
        correction._validate_spec(_spec("0" * 64)),
        wall_pose_resolver=_wall_resolver,
    )
    corrected = result.corrected_state["objects"]

    assert source == before
    assert corrected["cabinet"]["transform"]["translation"] == [2.0, 3.0, 0.5]
    assert corrected["cabinet"]["support_surfaces"][0]["transform"]["translation"] == [
        2.0,
        3.0,
        1.5,
    ]
    assert corrected["box"]["transform"]["translation"] == [2.2, 3.1, 1.5]
    assert corrected["box"]["support_surfaces"][0]["transform"]["translation"] == [
        2.2,
        3.1,
        1.7,
    ]
    assert corrected["pile"]["transform"]["translation"] == [2.3, 3.2, 1.7]
    assert corrected["pile"]["metadata"]["member_assets"][0]["transform"]["translation"] == [
        2.3,
        3.2,
        1.7,
    ]
    assert corrected["filled"]["metadata"]["container_asset"]["transform"]["translation"] == [
        2.5,
        3.4,
        1.5,
    ]
    assert corrected["filled"]["metadata"]["fill_assets"][0]["transform"]["translation"] == [
        2.5,
        3.4,
        1.6,
    ]
    assert corrected["unrelated"] == before["objects"]["unrelated"]
    assert result.translated_ids == {"cabinet", "box", "pile", "filled"}


def test_wall_local_is_height_only_and_preserves_exact_rotation():
    source = _source_state()
    result = correction.apply_corrections_to_state(
        source,
        correction._validate_spec(_spec("0" * 64)),
        wall_pose_resolver=_wall_resolver,
    )
    actual = result.corrected_state["objects"]["calendar"]
    assert actual["placement_info"]["position_2d"] == [1.57, 2.5]
    assert actual["transform"]["translation"] == [5.0, 6.0, 2.5]
    assert actual["transform"]["rotation_wxyz"] == [0.5, 0.5, 0.5, 0.5]


def test_wall_local_rejects_horizontal_world_drift():
    def drifting_resolver(state, item):
        value = _wall_resolver(state, item)
        value["translation"][0] += 0.01
        return value

    with pytest.raises(correction.PoseCorrectionError, match="horizontal position"):
        correction.apply_corrections_to_state(
            _source_state(),
            correction._validate_spec(_spec("0" * 64)),
            wall_pose_resolver=drifting_resolver,
        )


def test_unknown_composite_fails_closed():
    source = _source_state()
    source["objects"]["box"]["metadata"] = {"composite_type": "unknown"}
    with pytest.raises(correction.PoseCorrectionError, match="unsupported composite_type"):
        correction.apply_corrections_to_state(
            source,
            correction._validate_spec(_spec("0" * 64)),
            wall_pose_resolver=_wall_resolver,
        )


def test_duplicate_support_surface_and_cycle_fail_closed():
    duplicate = _source_state()
    duplicate["objects"]["unrelated"]["support_surfaces"] = [
        {"surface_id": "root_top", "transform": _transform(-3.0, 4.0, 1.0)}
    ]
    with pytest.raises(correction.PoseCorrectionError, match="owned by both"):
        correction.apply_corrections_to_state(
            duplicate,
            correction._validate_spec(_spec("0" * 64)),
            wall_pose_resolver=_wall_resolver,
        )

    cyclic = _source_state()
    cyclic["objects"]["cabinet"]["placement_info"] = {
        "parent_surface_id": "child_top",
        "position_2d": [0.0, 0.0],
        "rotation_2d": 0.0,
    }
    with pytest.raises(correction.PoseCorrectionError, match="contains a cycle"):
        correction.apply_corrections_to_state(
            cyclic,
            correction._validate_spec(_spec("0" * 64)),
            wall_pose_resolver=_wall_resolver,
        )


def test_default_run_is_dry_and_does_not_mutate(tmp_path):
    room, final, state_path, spec_path, _source, house_layout = _make_run(tmp_path)
    state_bytes = state_path.read_bytes()
    blend_bytes = (final / "scene.blend").read_bytes()

    result = correction.run(
        room_dir=room,
        house_layout=house_layout,
        spec_path=spec_path,
        runtime=FakeRuntime(),
    )

    assert result["mode"] == "dry_run"
    assert result["status"] == "pass"
    assert state_path.read_bytes() == state_bytes
    assert (final / "scene.blend").read_bytes() == blend_bytes
    assert not (final / correction.RECEIPT_NAME).exists()


def test_apply_quarantines_stale_evidence_and_verify_recomputes(tmp_path):
    room, final, state_path, spec_path, source, house_layout = _make_run(tmp_path)
    scene_dir = room.parent
    stale_render = scene_dir / "review" / "room_review_renders" / f"{ROOM_ID}_top.png"
    stale_gate = scene_dir / "quality_gates" / "room_self_exam_deterministic" / f"{ROOM_ID}.json"
    stale_render.parent.mkdir(parents=True)
    stale_gate.parent.mkdir(parents=True)
    stale_render.write_bytes(b"old render")
    stale_gate.write_text("{}\n", encoding="utf-8")

    receipt = correction.run(
        room_dir=room,
        house_layout=house_layout,
        spec_path=spec_path,
        apply=True,
        runtime=FakeRuntime(),
    )

    assert receipt["status"] == "pass"
    assert not (final / "scene.blend").exists()
    assert not stale_render.exists()
    assert not stale_gate.exists()
    corrected = json.loads(state_path.read_text(encoding="utf-8"))
    assert corrected["objects"]["cabinet"]["transform"]["translation"] == [2.0, 3.0, 0.5]
    assert sorted(receipt["manipuland_ids"]) == ["box", "filled", "pile"]

    run_dir = scene_dir.parent
    transaction_root = run_dir / receipt["backup"]["transaction_root_relative_to_run_dir"]
    backup_final = transaction_root / "original_final_scene"
    assert json.loads((backup_final / "scene_state.json").read_text(encoding="utf-8")) == source
    assert (backup_final / "scene.blend").read_bytes() == b"current blend"
    assert (transaction_root / "stale_evidence" / stale_render.relative_to(scene_dir)).exists()
    assert (transaction_root / "stale_evidence" / stale_gate.relative_to(scene_dir)).exists()

    verified = correction.run(
        room_dir=room,
        house_layout=house_layout,
        spec_path=spec_path,
        verify_only=True,
        runtime=FakeRuntime(),
    )
    assert verified["attestation"] == receipt["attestation"]


def test_publication_failure_restores_original_final_scene(tmp_path):
    room, final, state_path, spec_path, _source, house_layout = _make_run(tmp_path)
    original_state = state_path.read_bytes()

    def fail_after_swap():
        raise RuntimeError("injected publication failure")

    with pytest.raises(RuntimeError, match="injected publication failure"):
        correction.run(
            room_dir=room,
            house_layout=house_layout,
            spec_path=spec_path,
            apply=True,
            runtime=FakeRuntime(),
            failure_injector=fail_after_swap,
        )

    assert state_path.read_bytes() == original_state
    assert (final / "scene.blend").read_bytes() == b"current blend"
    assert not (room / ".pose_correction.lock").exists()
    assert not (final / correction.RECEIPT_NAME).exists()


def test_corrected_collision_rejects_without_mutation(tmp_path):
    room, final, state_path, spec_path, _source, house_layout = _make_run(tmp_path)
    original_state = state_path.read_bytes()
    runtime = FakeRuntime(
        collision_predicate=lambda state: state["objects"]["cabinet"]["transform"][
            "translation"
        ][0]
        > 1.5
    )

    with pytest.raises(correction.PoseCorrectionError, match="real collision"):
        correction.run(
            room_dir=room,
            house_layout=house_layout,
            spec_path=spec_path,
            apply=True,
            runtime=runtime,
        )

    assert state_path.read_bytes() == original_state
    assert (final / "scene.blend").exists()


def test_source_hash_mismatch_rejects_without_mutation(tmp_path):
    room, final, state_path, spec_path, _source, house_layout = _make_run(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["source_state_sha256"] = "f" * 64
    _write_json(spec_path, spec)
    original_state = state_path.read_bytes()

    with pytest.raises(correction.PoseCorrectionError, match="Source final-state SHA-256 changed"):
        correction.run(
            room_dir=room,
            house_layout=house_layout,
            spec_path=spec_path,
            apply=True,
            runtime=FakeRuntime(),
        )

    assert state_path.read_bytes() == original_state
    assert (final / "scene.blend").exists()


def test_verify_rejects_published_state_tampering(tmp_path):
    room, _final, state_path, spec_path, _source, house_layout = _make_run(tmp_path)
    correction.run(
        room_dir=room,
        house_layout=house_layout,
        spec_path=spec_path,
        apply=True,
        runtime=FakeRuntime(),
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["objects"]["cabinet"]["transform"]["translation"][0] += 0.1
    _write_json(state_path, state)

    with pytest.raises(correction.PoseCorrectionError, match="no longer matches its receipt"):
        correction.run(
            room_dir=room,
            house_layout=house_layout,
            spec_path=spec_path,
            verify_only=True,
            runtime=FakeRuntime(),
        )


def test_verify_rejects_tampered_source_backup(tmp_path):
    room, _final, _state_path, spec_path, _source, house_layout = _make_run(tmp_path)
    receipt = correction.run(
        room_dir=room,
        house_layout=house_layout,
        spec_path=spec_path,
        apply=True,
        runtime=FakeRuntime(),
    )
    run_dir = room.parent.parent
    backup_state = run_dir / receipt["source_state"]["path"]
    source = json.loads(backup_state.read_text(encoding="utf-8"))
    source["objects"]["cabinet"]["transform"]["translation"][0] += 0.1
    _write_json(backup_state, source)

    with pytest.raises(correction.PoseCorrectionError, match="source state no longer matches"):
        correction.run(
            room_dir=room,
            house_layout=house_layout,
            spec_path=spec_path,
            verify_only=True,
            runtime=FakeRuntime(),
        )


def test_verify_rejects_missing_commit_marker_and_stale_reappearance(tmp_path):
    room, _final, _state_path, spec_path, _source, house_layout = _make_run(tmp_path)
    stale = room.parent / "review" / "room_review_renders" / f"{ROOM_ID}_top.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old render")
    receipt = correction.run(
        room_dir=room,
        house_layout=house_layout,
        spec_path=spec_path,
        apply=True,
        runtime=FakeRuntime(),
    )
    run_dir = room.parent.parent
    marker = run_dir / receipt["backup"]["commit_marker"]
    marker.unlink()
    with pytest.raises(correction.PoseCorrectionError, match="commit marker"):
        correction.run(
            room_dir=room,
            house_layout=house_layout,
            spec_path=spec_path,
            verify_only=True,
            runtime=FakeRuntime(),
        )

    marker.write_bytes(b"committed\n")
    stale.write_bytes(b"old render")
    with pytest.raises(correction.PoseCorrectionError, match="Stale evidence reappeared"):
        correction.run(
            room_dir=room,
            house_layout=house_layout,
            spec_path=spec_path,
            verify_only=True,
            runtime=FakeRuntime(),
        )


def test_verify_recomputes_generated_artifacts(tmp_path):
    room, _final, _state_path, spec_path, _source, house_layout = _make_run(tmp_path)
    correction.run(
        room_dir=room,
        house_layout=house_layout,
        spec_path=spec_path,
        apply=True,
        runtime=FakeRuntime(),
    )

    runtime = FakeRuntime()
    runtime.dmd = lambda state: "directives:\n- changed: true\n"
    with pytest.raises(correction.PoseCorrectionError, match="differs from recomputation"):
        correction.run(
            room_dir=room,
            house_layout=house_layout,
            spec_path=spec_path,
            verify_only=True,
            runtime=runtime,
        )
