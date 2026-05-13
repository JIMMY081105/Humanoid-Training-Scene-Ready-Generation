from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import struct

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from scripts import recover_reference_classroom_furniture as recovery


@dataclass
class RecoveryFixture:
    room_dir: Path
    database: Path
    initial_state: Path
    registry: Path
    prompt: Path
    state_message_id: int
    cutoff_message_id: int
    failure_message_id: int
    storage_source_sdf: Path
    storage_copied_sdf: Path


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_sdf(path: Path, *, link_name: str = "base_link", static: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<?xml version='1.0'?>\n"
        "<sdf version='1.11'>\n"
        "  <model name='fixture'>\n"
        f"    <static>{str(static).lower()}</static>\n"
        f"    <link name='{link_name}'/>\n"
        "  </model>\n"
        "</sdf>\n",
        encoding="utf-8",
    )


def _write_glb(path: Path, *, include_normals: bool = True) -> None:
    positions = struct.pack("<9f", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    normals = struct.pack("<9f", 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    indices_payload = struct.pack("<3H", 0, 1, 2)
    binary_payload = positions + normals + indices_payload
    binary_payload += b"\x00" * (-len(binary_payload) % 4)
    attributes = {"POSITION": 0}
    accessors = [
        {
            "bufferView": 0,
            "componentType": 5126,
            "type": "VEC3",
            "count": 3,
        }
    ]
    if include_normals:
        attributes["NORMAL"] = len(accessors)
        accessors.append(
            {
                "bufferView": 1,
                "componentType": 5126,
                "type": "VEC3",
                "count": 3,
            }
        )
    index = len(accessors)
    accessors.append(
        {
            "bufferView": 2,
            "componentType": 5123,
            "type": "SCALAR",
            "count": 3,
        }
    )
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary_payload)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": 36, "byteLength": len(normals)},
            {"buffer": 0, "byteOffset": 72, "byteLength": 6},
        ],
        "accessors": accessors,
        "meshes": [
            {
                "primitives": [
                    {"attributes": attributes, "indices": index, "mode": 4}
                ]
            }
        ],
    }
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    payload += b" " * (-len(payload) % 4)
    length = 12 + 8 + len(payload) + 8 + len(binary_payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, length)
        + struct.pack("<II", len(payload), 0x4E4F534A)
        + payload
        + struct.pack("<II", len(binary_payload), 0x004E4942)
        + binary_payload
    )


def _write_artiverse_tree(root: Path, *, include_normals: bool = True) -> Path:
    (root / "objs").mkdir(parents=True)
    obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    (root / "objs" / "cabinet.obj").write_text(obj, encoding="utf-8")
    (root / "objs" / "cabinet__material_wood.obj").write_text(
        obj, encoding="utf-8"
    )
    _write_glb(root / "glbs" / "cabinet.glb", include_normals=include_normals)
    sdf = root / "scenesmith_artiverse.sdf"
    sdf.write_text(
        """<sdf version="1.11">
  <model name="fixture">
    <link name="cabinet_base">
      <visual name="cabinet_visual">
        <geometry><mesh><uri>./objs/cabinet__material_wood.obj</uri><scale>0.5 0.5 0.5</scale></mesh></geometry>
      </visual>
      <collision name="cabinet_collision">
        <geometry><mesh><uri>./objs/cabinet.obj</uri><scale>0.5 0.5 0.5</scale></mesh></geometry>
      </collision>
    </link>
  </model>
</sdf>
""",
        encoding="utf-8",
    )
    return sdf


def _wall(
    object_id: str,
    x: float,
    y: float,
    width: float,
    depth: float,
    height: float,
) -> dict[str, object]:
    return {
        "object_id": object_id,
        "object_type": "wall",
        "name": object_id,
        "description": f"Room {object_id}",
        "transform": {
            "translation": [x, y, height / 2.0],
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "geometry_path": None,
        "sdf_path": None,
        "image_path": None,
        "support_surfaces": [],
        "placement_info": None,
        "metadata": {},
        "bbox_min": [-width / 2.0, -depth / 2.0, -height / 2.0],
        "bbox_max": [width / 2.0, depth / 2.0, height / 2.0],
        "immutable": True,
        "scale_factor": 1.0,
    }


DESCRIPTIONS = {
    "whiteboard_0": "Rolling classroom whiteboard",
    "teacher_desk_0": "Teacher desk",
    "office_chair_0": "Teacher office chair",
    "filing_cabinet_0": "Teacher filing cabinet",
    "cubby_shelf_unit_0": "Classroom cubby shelf",
    "storage_cabinet_0": "Tall classroom storage cabinet",
    "trash_bin_0": "Classroom trash bin",
    "potted_plant_0": "Classroom potted plant",
    "student_desk_0": "Student desk",
    "classroom_student_chair_0": "Student chair",
}


def _base_asset_id(object_id: str) -> str:
    if object_id in recovery.EXPECTED_DESK_IDS:
        return "student_desk_0"
    if object_id in recovery.EXPECTED_CHAIR_IDS:
        return "classroom_student_chair_0"
    return object_id


def _asset_name(asset_id: str) -> str:
    return asset_id.rsplit("_", 1)[0]


def _append_message(
    connection: sqlite3.Connection,
    document: dict[str, object],
    *,
    second: int,
) -> int:
    cursor = connection.execute(
        "INSERT INTO agent_messages(session_id, message_data, created_at) VALUES(?,?,?)",
        (
            "designer",
            json.dumps(document, separators=(",", ":")),
            f"2026-07-12 05:48:{second % 60:02d}",
        ),
    )
    return int(cursor.lastrowid)


def _call(call_id: str, name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments, separators=(",", ":")),
        "status": "completed",
    }


def _output(call_id: str, value: object) -> dict[str, object]:
    output = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _make_fixture(
    tmp_path: Path,
    *,
    failed_move: bool = False,
    disallowed_remove_before_cutoff: bool = False,
) -> RecoveryFixture:
    room = tmp_path / "scene_000" / "room_classroom_01"
    room.mkdir(parents=True)
    prompt = tmp_path / "immutable_prompt.txt"
    prompt.write_text("EXACT PROMPT: keep every detail.\n", encoding="utf-8")

    room_geometry_sdf = tmp_path / "scene_000" / "room_geometry" / "room.sdf"
    _write_sdf(room_geometry_sdf, link_name="room_geometry_body_link", static=True)
    walls = {
        "north_wall": _wall("north_wall", 0.0, 3.725, 9.0, 0.05, 3.2),
        "south_wall": _wall("south_wall", 0.0, -3.725, 9.0, 0.05, 3.2),
        "east_wall": _wall("east_wall", 4.475, 0.0, 0.05, 7.5, 3.2),
        "west_wall": _wall("west_wall", -4.475, 0.0, 0.05, 7.5, 3.2),
    }
    initial = {
        "room_geometry": {
            "sdf_path": str(room_geometry_sdf.resolve()),
            "walls": list(walls.values()),
            "width": 7.5,
            "length": 9.0,
            "wall_height": 3.2,
            "wall_thickness": 0.05,
            "openings": [],
            "floor": None,
        },
        "objects": walls,
        "text_description": prompt.read_text(encoding="utf-8"),
        "timestamp": 1.0,
    }
    initial_path = room / "scene_renders" / "furniture" / "renders_001" / "scene_state.json"
    _write_json(initial_path, initial)

    source_dir = tmp_path / "artiverse_source" / "urdf_w_collider"
    source_sdf = _write_artiverse_tree(source_dir)

    registry: dict[str, dict[str, object]] = {}
    default_dimensions = {
        "whiteboard_0": (2.0, 0.4, 1.8),
        "teacher_desk_0": (1.4, 0.7, 0.7),
        "office_chair_0": (0.6, 0.6, 0.9),
        "filing_cabinet_0": (0.5, 0.5, 1.1),
        "cubby_shelf_unit_0": (2.0, 0.5, 1.2),
        "storage_cabinet_0": (1.0, 0.5, 2.0),
        "trash_bin_0": (0.3, 0.3, 0.4),
        "potted_plant_0": (0.4, 0.4, 0.6),
        "student_desk_0": (0.6, 0.45, 0.525),
        "classroom_student_chair_0": (0.45, 0.44, 0.71),
    }
    storage_copied_sdf: Path | None = None
    for asset_id, dimensions in default_dimensions.items():
        geometry = room / "generated_assets" / "furniture" / "geometry" / asset_id / "mesh.gltf"
        geometry.parent.mkdir(parents=True, exist_ok=True)
        geometry.write_text('{"asset":"fixture"}\n', encoding="utf-8")
        if asset_id == "storage_cabinet_0":
            copied_root = (
                room
                / "generated_assets"
                / "furniture"
                / "sdf"
                / "storage_cabinet_fixture"
            )
            shutil.copytree(source_dir, copied_root)
            sdf = copied_root / source_sdf.name
            storage_copied_sdf = sdf
            metadata: dict[str, object] = {
                "asset_source": "articulated",
                "articulated_source": "artiverse",
                "articulated_id": "artiverse/armoire/fixture",
                "articulated_source_sdf_path": str(source_sdf.resolve()),
                "articulated_source_sdf_sha256": recovery.sha256_file(source_sdf),
                "articulated_source_tree_sha256": recovery.sha256_directory_tree(
                    source_sdf.parent
                ),
                "articulated_copied_sdf_sha256": "0" * 64,
                "articulated_copied_tree_sha256": "1" * 64,
                "is_articulated": True,
            }
            scale = 0.5
        else:
            sdf = room / "generated_assets" / "furniture" / "sdf" / asset_id / "model.sdf"
            _write_sdf(sdf)
            metadata = {"asset_source": "generated"}
            scale = 1.0
        width, depth, height = dimensions
        registry[asset_id] = {
            "object_id": asset_id,
            "object_type": "furniture",
            "name": _asset_name(asset_id),
            "description": DESCRIPTIONS[asset_id],
            "transform": {
                "translation": [0.0, 0.0, 0.0],
                "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "geometry_path": str(geometry.resolve()),
            "sdf_path": str(sdf.resolve()),
            "image_path": None,
            "metadata": metadata,
            "bbox_min": [-width / 2.0, -depth / 2.0, -height / 2.0],
            "bbox_max": [width / 2.0, depth / 2.0, height / 2.0],
            "scale_factor": scale,
        }
    assert storage_copied_sdf is not None
    registry_path = room / "generated_assets" / "furniture" / "asset_registry.json"
    _write_json(registry_path, registry)

    anchor_objects: list[dict[str, object]] = []
    for wall_id in recovery.EXPECTED_WALL_IDS:
        wall = walls[wall_id]
        bbox_min = wall["bbox_min"]
        bbox_max = wall["bbox_max"]
        assert isinstance(bbox_min, list) and isinstance(bbox_max, list)
        anchor_objects.append(
            {
                "object_id": wall_id,
                "description": wall["description"],
                "position_x": wall["transform"]["translation"][0],  # type: ignore[index]
                "position_y": wall["transform"]["translation"][1],  # type: ignore[index]
                "rotation_degrees": 0.0,
                "dimensions": {
                    "width": bbox_max[0] - bbox_min[0],
                    "depth": bbox_max[1] - bbox_min[1],
                    "height": bbox_max[2] - bbox_min[2],
                },
            }
        )

    furniture_order = [
        *recovery.EXPECTED_SINGLE_FURNITURE_IDS,
        *recovery.EXPECTED_DESK_IDS,
        *recovery.EXPECTED_CHAIR_IDS,
    ]
    for index, object_id in enumerate(furniture_order):
        asset_id = _base_asset_id(object_id)
        width, depth, height = default_dimensions[asset_id]
        if object_id == "storage_cabinet_0":
            # The exact state anchor predates the successful 0.5 rescale.
            width, depth, height = (2.0, 1.0, 4.0)
        anchor_objects.append(
            {
                "object_id": object_id,
                "description": DESCRIPTIONS[asset_id],
                "position_x": float(index % 6) - 2.5,
                "position_y": float(index // 6) - 2.0,
                "rotation_degrees": 0.0,
                "dimensions": {"width": width, "depth": depth, "height": height},
            }
        )
    storage_anchor = next(
        value for value in anchor_objects if value["object_id"] == "storage_cabinet_0"
    )
    storage_anchor["position_x"] = 3.0
    storage_anchor["position_y"] = -3.0

    database = room / "designer.db"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE agent_sessions(session_id TEXT PRIMARY KEY, created_at TEXT, updated_at TEXT)"
    )
    connection.execute(
        "CREATE TABLE agent_messages("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
        "message_data TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO agent_sessions VALUES('designer','2026-07-12 05:00:00','2026-07-12 06:00:00')"
    )

    second = 0
    for object_id in furniture_order:
        asset_id = _base_asset_id(object_id)
        call_id = f"add-{object_id}"
        call_message = _append_message(
            connection,
            _call(
                call_id,
                "add_furniture_to_scene_tool",
                {"asset_id": asset_id, "x": 0.0, "y": 0.0, "yaw": 0.0},
            ),
            second=second,
        )
        assert call_message > 0
        _append_message(
            connection,
            _output(
                call_id,
                {
                    "success": True,
                    "asset_id": asset_id,
                    "object_id": object_id,
                    "message": "placed",
                },
            ),
            second=second + 1,
        )
        second += 2

    state_call_id = "state-anchor"
    _append_message(
        connection,
        _call(state_call_id, "get_current_scene_state", {}),
        second=second,
    )
    state_message_id = _append_message(
        connection,
        _output(
            state_call_id,
            {
                "success": True,
                "furniture_count": len(anchor_objects),
                "objects": anchor_objects,
            },
        ),
        second=second + 1,
    )
    second += 2

    move_calls: list[tuple[str, str, float, float]] = []
    for index, chair_id in enumerate(recovery.EXPECTED_CHAIR_IDS):
        call_id = f"move-{chair_id}"
        x = -2.0 + float(index % 3) * 2.0
        y = 1.0 - float(index // 3) * 1.2
        _append_message(
            connection,
            _call(
                call_id,
                "move_furniture_tool",
                {"object_id": chair_id, "x": x, "y": y, "yaw": 0.0},
            ),
            second=second,
        )
        second += 1
        move_calls.append((call_id, chair_id, x, y))
    for index, (call_id, chair_id, _x, _y) in enumerate(move_calls):
        _append_message(
            connection,
            _output(
                call_id,
                {
                    "success": not (failed_move and index == 0),
                    "object_id": chair_id,
                    "message": "moved",
                },
            ),
            second=second,
        )
        second += 1

    if disallowed_remove_before_cutoff:
        call_id = "forbidden-remove"
        _append_message(
            connection,
            _call(call_id, "remove_furniture_tool", {"object_id": "storage_cabinet_0"}),
            second=second,
        )
        second += 1
        _append_message(
            connection,
            _output(call_id, {"success": True, "object_id": "storage_cabinet_0"}),
            second=second,
        )
        second += 1

    rescale_call = "rescale-storage"
    _append_message(
        connection,
        _call(
            rescale_call,
            "rescale_furniture_tool",
            {"object_id": "storage_cabinet_0", "scale_factor": 0.5},
        ),
        second=second,
    )
    second += 1
    _append_message(
        connection,
        _output(
            rescale_call,
            {
                "success": True,
                "object_id": "storage_cabinet_0",
                "asset_id": str(storage_copied_sdf.resolve()),
                "affected_object_ids": ["storage_cabinet_0"],
                "scale_factor": 0.5,
                "new_asset_scale": 0.5,
                "previous_dimensions": {"width": 2.0, "depth": 1.0, "height": 4.0},
                "new_dimensions": {"width": 1.0, "depth": 0.5, "height": 2.0},
            },
        ),
        second=second,
    )
    second += 1
    snap_call = "snap-storage"
    _append_message(
        connection,
        _call(
            snap_call,
            "snap_to_object_tool",
            {
                "object_id": "storage_cabinet_0",
                "target_id": "south_wall",
                "orientation": "away",
            },
        ),
        second=second,
    )
    second += 1
    cutoff_message_id = _append_message(
        connection,
        _output(
            snap_call,
            {
                "success": True,
                "object_id": "storage_cabinet_0",
                "target_id": "south_wall",
                "original_position": {"x": 3.0, "y": -3.0, "z": 0.0},
                "new_position": {"x": 3.0, "y": -3.5, "z": 0.0},
                "distance_moved": 0.5,
                "rotation_applied": True,
                "rotation_angle_degrees": 0.0,
            },
        ),
        second=second,
    )
    second += 1
    _append_message(connection, {"type": "reasoning", "summary": []}, second=second)
    second += 1
    observe_call = "observe-failure"
    _append_message(
        connection,
        _call(observe_call, "observe_scene", {}),
        second=second,
    )
    second += 1
    failure_message_id = _append_message(
        connection,
        _output(
            observe_call,
            "An error occurred: OBJ has no normals: copied/objs/cabinet.obj",
        ),
        second=second,
    )
    second += 1
    # The recovery must ignore this later destructive workaround branch.
    late_remove = "late-remove"
    _append_message(
        connection,
        _call(late_remove, "remove_furniture_tool", {"object_id": "storage_cabinet_0"}),
        second=second,
    )
    _append_message(
        connection,
        _output(late_remove, {"success": True, "object_id": "storage_cabinet_0"}),
        second=second + 1,
    )
    connection.commit()
    connection.close()

    return RecoveryFixture(
        room_dir=room,
        database=database,
        initial_state=initial_path,
        registry=registry_path,
        prompt=prompt,
        state_message_id=state_message_id,
        cutoff_message_id=cutoff_message_id,
        failure_message_id=failure_message_id,
        storage_source_sdf=source_sdf,
        storage_copied_sdf=storage_copied_sdf,
    )


def _recover(fixture: RecoveryFixture) -> Path:
    return recovery.recover_checkpoint(
        room_dir=fixture.room_dir,
        designer_db=fixture.database,
        initial_scene_state=fixture.initial_state,
        asset_registry=fixture.registry,
        state_message_id=fixture.state_message_id,
        cutoff_message_id=fixture.cutoff_message_id,
        first_observe_failure_message_id=fixture.failure_message_id,
        bound_inputs=[fixture.prompt],
    )


def test_recovers_exact_checkpoint_and_ignores_post_failure_branch(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    database_before = hashlib.sha256(fixture.database.read_bytes()).hexdigest()
    source_before = fixture.storage_source_sdf.read_bytes()

    checkpoint = _recover(fixture)

    assert hashlib.sha256(fixture.database.read_bytes()).hexdigest() == database_before
    assert fixture.storage_source_sdf.read_bytes() == source_before
    assert checkpoint.name == recovery.CHECKPOINT_NAME
    assert not (checkpoint.parent / recovery.STAGING_NAME).exists()
    state = json.loads((checkpoint / "scene_state.json").read_text(encoding="utf-8"))
    assert len(state["objects"]) == 36
    assert set(state["objects"]) == recovery.EXPECTED_OBJECT_IDS
    assert state["text_description"] == fixture.prompt.read_text(encoding="utf-8")
    assert state["objects"]["north_wall"]["immutable"] is True

    for index, chair_id in enumerate(recovery.EXPECTED_CHAIR_IDS):
        translation = state["objects"][chair_id]["transform"]["translation"]
        assert translation == pytest.approx(
            [-2.0 + float(index % 3) * 2.0, 1.0 - float(index // 3) * 1.2, 0.0]
        )
    storage = state["objects"]["storage_cabinet_0"]
    assert storage["transform"]["translation"] == [3.0, -3.5, 0.0]
    assert storage["scale_factor"] == 0.5
    assert storage["bbox_min"] == [-0.5, -0.25, -1.0]
    assert storage["bbox_max"] == [0.5, 0.25, 1.0]
    assert storage["metadata"]["articulated_copied_sdf_sha256"] == recovery.sha256_file(
        fixture.storage_copied_sdf
    )
    assert storage["metadata"]["articulated_copied_tree_sha256"] == recovery.sha256_directory_tree(
        fixture.storage_copied_sdf.parent
    )
    normalized_sdf = fixture.storage_copied_sdf.read_text(encoding="utf-8")
    assert (
        "./scenesmith_artiverse_visuals_v2/cabinet.gltf" in normalized_sdf
    )
    assert "./objs/cabinet.obj" in normalized_sdf
    derived_dir = (
        fixture.storage_copied_sdf.parent / "scenesmith_artiverse_visuals_v2"
    )
    derived_gltf = derived_dir / "cabinet.gltf"
    derived_bin = derived_dir / "cabinet.bin"
    derived_manifest = derived_dir / "_derivation_manifest.json"
    assert derived_gltf.is_file()
    assert derived_bin.is_file()
    assert derived_manifest.is_file()
    gltf_document = json.loads(derived_gltf.read_text(encoding="utf-8"))
    assert gltf_document["buffers"] == [{"byteLength": 80, "uri": "cabinet.bin"}]
    manifest_document = json.loads(derived_manifest.read_text(encoding="utf-8"))
    assert manifest_document["schema_version"] == 2
    assert manifest_document["policy"] == "publisher_glb_derived_external_gltf"

    directive = (checkpoint / "scene.dmd.yaml").read_text(encoding="utf-8")
    assert "child: storage_cabinet_0::cabinet_base" in directive
    receipt = json.loads((checkpoint / "recovery_receipt.json").read_text())
    assert receipt["status"] == "pass"
    assert receipt["database_evidence"]["cutoff_message_id"] == fixture.cutoff_message_id
    assert receipt["database_evidence"]["first_observe_failure_message_id"] == fixture.failure_message_id
    assert receipt["database_evidence"]["ignored_messages_after_failure"] == 2
    assert receipt["reconstruction"]["student_desk_count"] == 12
    assert receipt["reconstruction"]["student_chair_count"] == 12
    assert receipt["reconstruction"]["instance_asset_mapping"]["student_desk_b"] == "student_desk_0"
    normalization = receipt["reconstruction"]["artiverse_visual_normalization"]
    assert len(normalization) == 1
    assert normalization[0]["status"] == "pass"
    assert normalization[0]["schema_version"] == 2
    assert normalization[0]["policy"] == "publisher_glb_derived_external_gltf"
    assert normalization[0]["rewritten_link_count"] == 1
    assert normalization[0]["collision_element_count"] == 1
    assert normalization[0]["derived_resource_directory_sha256"] == (
        recovery.sha256_directory_tree(derived_dir)
    )
    normalized_link = normalization[0]["links"][0]
    assert normalized_link["derived_gltf_sha256"] == hashlib.sha256(
        derived_gltf.read_bytes()
    ).hexdigest()
    assert normalized_link["derived_bin_sha256"] == hashlib.sha256(
        derived_bin.read_bytes()
    ).hexdigest()
    bound = receipt["source_inputs"]["bound_immutable_inputs"]
    assert bound[0]["sha256"] == hashlib.sha256(fixture.prompt.read_bytes()).hexdigest()
    state_record = receipt["published_artifacts"]["scene_state"]
    assert state_record["sha256"] == hashlib.sha256(
        (checkpoint / "scene_state.json").read_bytes()
    ).hexdigest()


def test_rejects_failed_operation_inside_replay_window(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path, failed_move=True)
    with pytest.raises(recovery.RecoveryError, match="was not successful"):
        _recover(fixture)
    assert not (fixture.room_dir / "scene_states" / recovery.CHECKPOINT_NAME).exists()


def test_recovery_accepts_pre_normalized_copy_without_rewriting_it(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    first = recovery.normalize_copied_artiverse_visuals(
        fixture.storage_copied_sdf, fixture.storage_copied_sdf.parent
    )
    normalized_bytes = fixture.storage_copied_sdf.read_bytes()
    normalized_tree = recovery.sha256_directory_tree(fixture.storage_copied_sdf.parent)

    checkpoint = _recover(fixture)

    assert fixture.storage_copied_sdf.read_bytes() == normalized_bytes
    assert recovery.sha256_directory_tree(fixture.storage_copied_sdf.parent) == normalized_tree
    receipt = json.loads((checkpoint / "recovery_receipt.json").read_text())
    evidence = receipt["reconstruction"]["artiverse_visual_normalization"][0]
    assert first["rewritten_link_count"] == 1
    assert evidence["rewritten_link_count"] == 0
    assert evidence["already_normalized_link_count"] == 1


def test_recovery_migrates_legacy_direct_glb_copy(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    document = ET.parse(fixture.storage_copied_sdf)
    visual_uri = next(
        element
        for element in document.getroot().iter()
        if str(element.tag).rsplit("}", 1)[-1] == "uri"
        and "material_wood" in (element.text or "")
    )
    visual_uri.text = "./glbs/cabinet.glb"
    document.write(
        fixture.storage_copied_sdf,
        encoding="utf-8",
        xml_declaration=True,
    )

    checkpoint = _recover(fixture)

    receipt = json.loads((checkpoint / "recovery_receipt.json").read_text())
    evidence = receipt["reconstruction"]["artiverse_visual_normalization"][0]
    assert evidence["rewritten_link_count"] == 0
    assert evidence["migrated_legacy_glb_link_count"] == 1
    assert evidence["already_normalized_link_count"] == 0
    normalized_sdf = fixture.storage_copied_sdf.read_text(encoding="utf-8")
    assert "./scenesmith_artiverse_visuals_v2/cabinet.gltf" in normalized_sdf
    assert (
        fixture.storage_copied_sdf.parent
        / "scenesmith_artiverse_visuals_v2"
        / "cabinet.bin"
    ).is_file()


def test_rejects_artiverse_glb_without_authored_normals(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _write_glb(
        fixture.storage_copied_sdf.parent / "glbs" / "cabinet.glb",
        include_normals=False,
    )
    with pytest.raises(recovery.RecoveryError, match="Cannot normalize.*POSITION/NORMAL"):
        _recover(fixture)
    assert not (fixture.room_dir / "scene_states" / recovery.CHECKPOINT_NAME).exists()


def test_rejects_disallowed_mutation_before_cutoff(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path, disallowed_remove_before_cutoff=True)
    with pytest.raises(recovery.RecoveryError, match="Disallowed function call"):
        _recover(fixture)


def test_rejects_stale_immutable_artiverse_source_hash(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    fixture.storage_source_sdf.write_text("mutated source\n", encoding="utf-8")
    with pytest.raises(recovery.RecoveryError, match="immutable source SDF hash is stale"):
        _recover(fixture)


def test_rejects_registry_scale_that_disagrees_with_archived_rescale(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    registry = json.loads(fixture.registry.read_text(encoding="utf-8"))
    registry["storage_cabinet_0"]["scale_factor"] = 0.25
    _write_json(fixture.registry, registry)
    with pytest.raises(recovery.RecoveryError, match="does not match archived rescale"):
        _recover(fixture)


def test_refuses_to_replace_existing_checkpoint(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    existing = fixture.room_dir / "scene_states" / recovery.CHECKPOINT_NAME
    existing.mkdir(parents=True)
    marker = existing / "keep.txt"
    marker.write_text("user data", encoding="utf-8")
    with pytest.raises(recovery.RecoveryError, match="Refusing to replace"):
        _recover(fixture)
    assert marker.read_text(encoding="utf-8") == "user data"
