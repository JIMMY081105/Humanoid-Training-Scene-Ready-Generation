import hashlib
import json
import sys
import types

from pathlib import Path

import pytest


class FakeVector:
    def __init__(self, values):
        self.x, self.y, self.z = (float(value) for value in values)

    def __getitem__(self, index):
        return (self.x, self.y, self.z)[index]

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __add__(self, other):
        return FakeVector((self.x + other.x, self.y + other.y, self.z + other.z))

    def __sub__(self, other):
        return FakeVector((self.x - other.x, self.y - other.y, self.z - other.z))

    def to_track_quat(self, _track, _up):
        return types.SimpleNamespace(to_euler=lambda: (0.0, 0.0, 0.0))


class IdentityMatrix:
    def __matmul__(self, vector):
        return FakeVector(vector)


def _corners(minimum, maximum):
    return [
        (x, y, z)
        for x in (minimum[0], maximum[0])
        for y in (minimum[1], maximum[1])
        for z in (minimum[2], maximum[2])
    ]


class FakeObject:
    def __init__(
        self,
        name,
        minimum=(0, 0, 0),
        maximum=(1, 1, 1),
        *,
        object_type="MESH",
        properties=None,
    ):
        self.name = name
        self.type = object_type
        self.bound_box = _corners(minimum, maximum)
        self.matrix_world = IdentityMatrix()
        self.parent = None
        self.users_collection = []
        self.hide_render = False
        self.hide_viewport = False
        self.data = types.SimpleNamespace()
        self.rotation_euler = None
        self._properties = properties or {}

    def get(self, key, default=None):
        return self._properties.get(key, default)


class FakeObjectList(list):
    def remove(self, obj, do_unlink=False):
        del do_unlink
        super().remove(obj)


class FakeBpy(types.ModuleType):
    def __init__(self):
        super().__init__("bpy")
        objects = FakeObjectList()
        render_settings = types.SimpleNamespace(
            resolution_x=0,
            resolution_y=0,
            film_transparent=True,
            engine="",
            filepath="",
        )
        scene = types.SimpleNamespace(
            objects=objects, render=render_settings, camera=None
        )
        self.context = types.SimpleNamespace(
            scene=scene,
            object=None,
            view_layer=types.SimpleNamespace(update=lambda: None),
        )
        self.data = types.SimpleNamespace(objects=objects)

        def camera_add(*, location):
            camera = FakeObject(
                "Camera", object_type="CAMERA", minimum=(0, 0, 0), maximum=(0, 0, 0)
            )
            camera.location = location
            objects.append(camera)
            self.context.object = camera

        def light_add(*, type, location):
            del type, location
            light = FakeObject(
                "Sun", object_type="LIGHT", minimum=(0, 0, 0), maximum=(0, 0, 0)
            )
            light.data.energy = 0.0
            objects.append(light)
            self.context.object = light

        def render(*, write_still):
            assert write_still is True
            target = Path(scene.render.filepath)
            target.write_bytes(f"png:{target.name}".encode())

        self.ops = types.SimpleNamespace(
            object=types.SimpleNamespace(camera_add=camera_add, light_add=light_add),
            render=types.SimpleNamespace(render=render),
            wm=types.SimpleNamespace(open_mainfile=lambda **_kwargs: None),
        )


FAKE_BPY = FakeBpy()
MATHUTILS = types.ModuleType("mathutils")
MATHUTILS.Vector = FakeVector
# SQZ has real bpy installed; force this unit-test module onto deterministic
# fakes before importing the renderer, then the whole test exercises one API.
sys.modules["bpy"] = FAKE_BPY
sys.modules["mathutils"] = MATHUTILS

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from render_room_review_views import (  # noqa: E402
    CutawayError,
    _restore_visibility,
    _room_mesh_records,
    _visibility_snapshot,
    classify_room_envelope,
    establish_cutaway,
    render_room,
)


@pytest.fixture(autouse=True)
def _reset_scene():
    FAKE_BPY.context.scene.objects.clear()
    FAKE_BPY.context.scene.camera = None
    FAKE_BPY.context.object = None
    yield
    FAKE_BPY.context.scene.objects.clear()


def _complete_room_objects():
    return [
        FakeObject("floor", (0, 0, -0.1), (10, 8, 0.0)),
        # Anonymous name proves the overhead geometry heuristic, not just naming.
        FakeObject("Mesh_Overhead_0", (0, 0, 3.0), (10, 8, 3.2)),
        FakeObject("north_wall", (0, 7.9, 0), (10, 8.1, 3.0)),
        FakeObject("south_wall", (0, -0.1, 0), (10, 0.1, 3.0)),
        FakeObject("east_wall", (9.9, 0, 0), (10.1, 8, 3.0)),
        FakeObject("west_wall", (-0.1, 0, 0), (0.1, 8, 3.0)),
        FakeObject("teacher_table", (4, 3, 0), (6, 5, 1.0)),
    ]


def _classification():
    FAKE_BPY.context.scene.objects.extend(_complete_room_objects())
    records, bounds = _room_mesh_records(x=0, y=0, width=10, depth=8)
    return records, bounds, classify_room_envelope(records, bounds)


def test_oblique_cutaway_hides_overhead_and_only_camera_side_walls() -> None:
    records, bounds, classification = _classification()
    snapshot = _visibility_snapshot(records)

    evidence = establish_cutaway(
        classification=classification,
        room_bounds=bounds,
        camera_location=FakeVector((20, -20, 15)),
        view_name="oblique_a",
    )

    assert evidence["established"] is True
    assert evidence["overhead_state"] == "hidden"
    assert set(evidence["hidden_envelope_object_names"]) == {
        "Mesh_Overhead_0",
        "east_wall",
        "south_wall",
    }
    assert set(evidence["visible_far_wall_object_names"]) == {
        "north_wall",
        "west_wall",
    }
    assert evidence["visible_floor_object_names"] == ["floor"]
    assert evidence["visible_content_object_names"] == ["teacher_table"]
    assert (
        next(
            obj for obj in FAKE_BPY.context.scene.objects if obj.name == "floor"
        ).hide_render
        is False
    )
    assert (
        next(
            obj for obj in FAKE_BPY.context.scene.objects if obj.name == "teacher_table"
        ).hide_render
        is False
    )

    _restore_visibility(snapshot)
    assert all(
        not obj.hide_render and not obj.hide_viewport
        for obj in FAKE_BPY.context.scene.objects
    )


def test_top_cutaway_hides_overhead_but_preserves_all_walls() -> None:
    _records, bounds, classification = _classification()

    evidence = establish_cutaway(
        classification=classification,
        room_bounds=bounds,
        camera_location=FakeVector((5, 4, 40)),
        view_name="top",
    )

    assert evidence["hidden_envelope_object_names"] == ["Mesh_Overhead_0"]
    assert evidence["hidden_camera_side_walls"] == []
    assert set(evidence["visible_far_wall_object_names"]) == {
        "north_wall",
        "south_wall",
        "east_wall",
        "west_wall",
    }


def test_absent_overhead_is_explicitly_verified_while_oblique_walls_are_cut() -> None:
    FAKE_BPY.context.scene.objects.extend(
        obj for obj in _complete_room_objects() if obj.name != "Mesh_Overhead_0"
    )
    records, bounds = _room_mesh_records(x=0, y=0, width=10, depth=8)
    classification = classify_room_envelope(records, bounds)

    top = establish_cutaway(
        classification=classification,
        room_bounds=bounds,
        camera_location=FakeVector((5, 4, 40)),
        view_name="top",
    )
    oblique = establish_cutaway(
        classification=classification,
        room_bounds=bounds,
        camera_location=FakeVector((20, -20, 15)),
        view_name="oblique_a",
    )

    assert top["overhead_state"] == "verified_absent"
    assert top["hidden_envelope_object_names"] == []
    assert oblique["overhead_state"] == "verified_absent"
    assert set(oblique["hidden_envelope_object_names"]) == {
        "east_wall",
        "south_wall",
    }


def test_unclassifiable_envelope_fails_closed() -> None:
    FAKE_BPY.context.scene.objects.append(FakeObject("table", (3, 3, 0), (5, 5, 1)))
    records, bounds = _room_mesh_records(x=0, y=0, width=10, depth=8)

    with pytest.raises(CutawayError, match="No room envelope"):
        classify_room_envelope(records, bounds)


def test_indivisible_combined_room_shell_fails_closed() -> None:
    FAKE_BPY.context.scene.objects.extend(
        [
            FakeObject(
                "Mesh_AnonymousShell",
                (0, 0, -0.1),
                (10, 8, 3.0),
            ),
            FakeObject("desk", (4, 3, 0), (6, 5, 1)),
        ]
    )
    records, bounds = _room_mesh_records(x=0, y=0, width=10, depth=8)

    with pytest.raises(CutawayError, match="indivisible envelope"):
        classify_room_envelope(records, bounds)


def test_render_room_writes_hash_bound_cutaway_evidence_and_restores_scene(
    tmp_path: Path,
) -> None:
    room_objects = _complete_room_objects()
    FAKE_BPY.context.scene.objects.extend(room_objects)
    source_blend = tmp_path / "scene.blend"
    source_blend.write_bytes(b"blend-source")
    source_state = tmp_path / "scene_state.json"
    source_state.write_text('{"objects": {}}', encoding="utf-8")

    result = render_room(
        {"room_id": "classroom_01", "position": [0, 0], "width": 10, "depth": 8},
        tmp_path,
        source_blend=source_blend,
        source_state=source_state,
    )

    assert result["status"] == "pass"
    assert result["rendered_view_count"] == 3
    assert [view["view_name"] for view in result["views"]] == [
        "top",
        "oblique_a",
        "oblique_b",
    ]
    assert all(view["cutaway"]["established"] for view in result["views"])
    assert result["source_blend"]["size_bytes"] == len(b"blend-source")
    assert len(result["source_blend"]["sha256"]) == 64
    assert result["source_state"]["sha256"] == hashlib.sha256(
        source_state.read_bytes()
    ).hexdigest()
    assert result["derivation_receipt"]["source_state"] == result["source_state"]
    assert len(result["derivation_receipt"]["attestation"]["sha256"]) == 64
    for view in result["views"]:
        image = Path(view["image"])
        assert image.is_file()
        assert len(view["image_sha256"]) == 64
        assert view["image_size_bytes"] == image.stat().st_size
    evidence_path = tmp_path / "classroom_01_cutaway_evidence.json"
    persisted = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert persisted == result
    assert all(not obj.hide_render and not obj.hide_viewport for obj in room_objects)


def test_render_room_rejects_distinct_paths_with_duplicate_image_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FAKE_BPY.context.scene.objects.extend(_complete_room_objects())

    def duplicate_render(*, write_still):
        assert write_still is True
        Path(FAKE_BPY.context.scene.render.filepath).write_bytes(b"same-image")

    monkeypatch.setattr(FAKE_BPY.ops.render, "render", duplicate_render)
    with pytest.raises(CutawayError, match="duplicate image bytes"):
        render_room(
            {"room_id": "classroom_01", "position": [0, 0], "width": 10, "depth": 8},
            tmp_path,
        )


def test_render_failure_overwrites_evidence_with_fail_status(tmp_path: Path) -> None:
    # Floor and ceiling establish a top cutaway, but no wall can establish either
    # oblique dollhouse cutaway. The renderer must stop and publish failure evidence.
    FAKE_BPY.context.scene.objects.extend(
        [
            FakeObject("floor", (0, 0, -0.1), (10, 8, 0)),
            FakeObject("ceiling", (0, 0, 3), (10, 8, 3.2)),
            FakeObject("desk", (4, 3, 0), (6, 5, 1)),
        ]
    )

    with pytest.raises(CutawayError, match="No genuinely camera-side"):
        render_room(
            {"room_id": "library", "position": [0, 0], "width": 10, "depth": 8},
            tmp_path,
        )

    evidence = json.loads(
        (tmp_path / "library_cutaway_evidence.json").read_text(encoding="utf-8")
    )
    assert evidence["status"] == "fail"
    assert evidence["error_type"] == "CutawayError"
    assert len(evidence["views"]) == 1
