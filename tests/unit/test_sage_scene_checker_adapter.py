import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools" / "sage_scene_checker"
sys.path.insert(0, str(TOOLS_DIR))

from adapter import _clamp01, load_scenesmith_output, normalize_category  # noqa: E402


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _fake_object(object_id: str, translation: list[float] | None = None) -> dict:
    return {
        "object_id": object_id,
        "object_type": "furniture",
        "name": "chair",
        "description": "student chair",
        "transform": {
            "translation": translation or [2.0, 2.0, 0.5],
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "geometry_path": "assets/chair.gltf",
        "bbox_min": [-0.25, -0.25, -0.5],
        "bbox_max": [0.25, 0.25, 0.5],
        "placement_info": None,
        "support_surfaces": [],
        "metadata": {},
    }


def _write_fake_scene(tmp_path: Path) -> Path:
    scene_dir = tmp_path / "scene_000"
    combined = scene_dir / "combined_house"
    house_state = {
        "layout": {
            "wall_height": 3.0,
            "house_prompt": "small classroom",
            "rooms": [
                {
                    "id": "room_a",
                    "type": "classroom",
                    "position": [0.0, 0.0],
                    "width": 4.0,
                    "length": 5.0,
                }
            ],
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
        "rooms": {
            "room_a": {
                "text_description": "classroom",
                "objects": {"chair_1": _fake_object("chair_1")},
            }
        },
    }
    sceneeval_state = {
        "scene": {
            "object": [
                {
                    "id": "chair_1",
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
                            2.0,
                            2.0,
                            0.5,
                            1.0,
                        ]
                    },
                }
            ]
        }
    }
    _write_json(combined / "house_state.json", house_state)
    _write_json(combined / "sceneeval_state.json", sceneeval_state)
    (combined / "house.dmd.yaml").write_text("name: fake\n", encoding="utf-8")
    asset = scene_dir / "room_room_a" / "assets" / "chair.gltf"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_text("fake asset", encoding="utf-8")
    return scene_dir


def test_adapter_loads_minimal_house_state_and_sceneeval_state(tmp_path):
    scene_dir = _write_fake_scene(tmp_path)

    floor_plan = load_scenesmith_output(scene_dir)

    assert floor_plan["id"] == "scene_000"
    assert floor_plan["has_house_dmd"] is True
    assert len(floor_plan["rooms"]) == 1
    assert len(floor_plan["objects"]) == 1
    assert floor_plan["rooms"][0]["doors"][0]["id"] == "door_a"
    assert floor_plan["objects"][0]["id"] == "chair_1"
    assert floor_plan["objects"][0]["position"] == {"x": 2.0, "y": 2.0, "z": 0.5}
    assert floor_plan["objects"][0]["asset_path_exists"] is True


def test_adapter_normalizes_common_semantic_categories():
    assert normalize_category("Student Desk") == "furniture"
    assert normalize_category("wall mounted screen") == "wall_mounted"
    assert normalize_category("") == "unknown"


def test_adapter_clamps_opening_position_ratio():
    assert _clamp01(-0.25) == 0.0
    assert _clamp01(0.5) == 0.5
    assert _clamp01(1.25) == 1.0
