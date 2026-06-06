import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools" / "sage_scene_checker"
sys.path.insert(0, str(TOOLS_DIR))

from check_scenesmith_output import REQUIRED_SCENE_FILES, _required_files, run_checker  # noqa: E402
from money_guard import PAID_API_ENV_VARS  # noqa: E402


def _clear_paid_env(monkeypatch) -> None:
    for name in PAID_API_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _fake_object(
    object_id: str,
    *,
    translation: list[float] | None = None,
    bbox_min: list[float] | None = None,
    bbox_max: list[float] | None = None,
    geometry_path: str = "assets/chair.gltf",
) -> dict:
    return {
        "object_id": object_id,
        "object_type": "furniture",
        "name": "chair",
        "description": "student chair",
        "transform": {
            "translation": translation or [2.0, 2.0, 0.5],
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "geometry_path": geometry_path,
        "bbox_min": bbox_min or [-0.25, -0.25, -0.5],
        "bbox_max": bbox_max or [0.25, 0.25, 0.5],
        "placement_info": None,
        "support_surfaces": [],
        "metadata": {},
    }


def _write_fake_scene(
    tmp_path: Path,
    *,
    objects: dict[str, dict] | None = None,
) -> Path:
    scene_dir = tmp_path / "scene_000"
    combined = scene_dir / "combined_house"
    objects = objects or {"chair_1": _fake_object("chair_1")}
    house_state = {
        "layout": {
            "wall_height": 3.0,
            "house_prompt": "small classroom",
            "rooms": [{"id": "room_a", "type": "classroom", "width": 4.0, "length": 5.0}],
            "placed_rooms": [
                {
                    "room_id": "room_a",
                    "position": [0.0, 0.0],
                    "width": 5.0,
                    "depth": 4.0,
                    "walls": [
                        {
                            "wall_id": "room_a_south",
                            "room_id": "room_a",
                            "direction": "south",
                            "start_point": [0.0, 0.0],
                            "end_point": [5.0, 0.0],
                            "length": 5.0,
                            "openings": [
                                {
                                    "opening_id": "door_a",
                                    "opening_type": "DOOR",
                                    "position_along_wall": 2.5,
                                    "width": 0.9,
                                    "height": 2.1,
                                    "sill_height": 0.0,
                                }
                            ],
                        }
                    ],
                }
            ],
        },
        "rooms": {"room_a": {"objects": objects}},
    }
    sceneeval_objects = []
    for object_id, obj in objects.items():
        translation = obj["transform"]["translation"]
        sceneeval_objects.append(
            {
                "id": object_id,
                "transform": {
                    "data": [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        translation[0],
                        translation[1],
                        translation[2],
                        1.0,
                    ]
                },
            }
        )
    _write_json(combined / "house_state.json", house_state)
    _write_json(combined / "sceneeval_state.json", {"scene": {"object": sceneeval_objects}})
    asset = scene_dir / "room_room_a" / "assets" / "chair.gltf"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_text("fake asset", encoding="utf-8")
    return scene_dir


def _assert_report_shape(report: dict) -> None:
    assert set(["scene_id", "pass", "summary", "money_guard", "checks", "repair_suggestions"]).issubset(report)
    assert set(["num_rooms", "num_objects", "num_errors", "num_warnings"]).issubset(report["summary"])
    assert report["money_guard"] == {
        "openai_api_calls": 0,
        "gemini_api_calls": 0,
        "anthropic_api_calls": 0,
        "external_paid_api_calls": 0,
        "codex_cli_calls": 0,
    }
    assert all({"id", "severity", "status", "message"}.issubset(check) for check in report["checks"])


def test_report_schema_is_valid_for_minimal_scene(tmp_path, monkeypatch):
    _clear_paid_env(monkeypatch)
    scene_dir = _write_fake_scene(tmp_path)

    report = run_checker(scene_dir=scene_dir, no_paid_api=True)

    _assert_report_shape(report)
    assert report["pass"] is True
    assert report["summary"]["num_rooms"] == 1
    assert report["summary"]["num_objects"] == 1


def test_missing_files_produce_fail_report(tmp_path, monkeypatch):
    _clear_paid_env(monkeypatch)
    scene_dir = tmp_path / "empty_scene"
    scene_dir.mkdir()

    report = run_checker(scene_dir=scene_dir, no_paid_api=True)

    _assert_report_shape(report)
    assert report["pass"] is False
    assert any(check["id"] == "required_file.house_state" and check["status"] == "fail" for check in report["checks"])
    assert any(check["id"] == "required_file.sceneeval_state" and check["status"] == "fail" for check in report["checks"])


def test_invalid_bbox_produces_fail_report(tmp_path, monkeypatch):
    _clear_paid_env(monkeypatch)
    scene_dir = _write_fake_scene(
        tmp_path,
        objects={
            "chair_1": _fake_object(
                "chair_1",
                bbox_min=[0.0, -0.25, -0.5],
                bbox_max=[0.0, 0.25, 0.5],
            )
        },
    )

    report = run_checker(scene_dir=scene_dir, no_paid_api=True)

    assert report["pass"] is False
    assert any(check["id"] == "object.bbox_valid" and check["status"] == "fail" for check in report["checks"])
    assert any(suggestion["check_id"] == "object.bbox_valid" for suggestion in report["repair_suggestions"])


def test_overlapping_boxes_are_detected(tmp_path, monkeypatch):
    _clear_paid_env(monkeypatch)
    scene_dir = _write_fake_scene(
        tmp_path,
        objects={
            "chair_1": _fake_object("chair_1", translation=[2.0, 2.0, 0.5]),
            "chair_2": _fake_object("chair_2", translation=[2.1, 2.0, 0.5]),
        },
    )

    report = run_checker(scene_dir=scene_dir, no_paid_api=True)

    assert any(check["id"] == "object.coarse_collision" and check["status"] == "fail" for check in report["checks"])
    assert report["summary"]["num_warnings"] >= 1


def test_no_sage_installation_is_required_for_basic_checks(tmp_path, monkeypatch):
    _clear_paid_env(monkeypatch)
    scene_dir = _write_fake_scene(tmp_path)

    report = run_checker(scene_dir=scene_dir, sage_root=None, no_paid_api=True)

    assert report["pass"] is True
    assert any(check["id"] == "sage.optional" and check["status"] == "pass" for check in report["checks"])


def test_required_file_map_resolves_under_scene_directory(tmp_path):
    scene_dir = tmp_path / "scene_000"

    required = _required_files(scene_dir)

    assert set(required) == set(REQUIRED_SCENE_FILES)
    assert required["house_state"] == scene_dir / "combined_house" / "house_state.json"
    assert required["sceneeval_state"] == scene_dir / "combined_house" / "sceneeval_state.json"
