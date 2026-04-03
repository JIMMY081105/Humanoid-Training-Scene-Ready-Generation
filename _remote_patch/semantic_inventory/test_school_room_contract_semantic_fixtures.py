from __future__ import annotations

import copy

from scripts.school_room_contract import canonical_room_prompt, evaluate_room_inventory


def obj(name: str, *, dims=(0.5, 0.5, 0.8), xyz=(0.0, 0.0, 0.0)) -> dict:
    return {
        "object_id": name.replace(" ", "_"),
        "name": name,
        "description": name,
        "object_type": "FURNITURE",
        "metadata": {
            "dimension_contract": {
                "status": "pass",
                "measured_final_scene_dimensions_m": list(dims),
            }
        },
        "transform": {"translation": list(xyz), "rotation_wxyz": [1.0, 0.0, 0.0, 0.0]},
        "geometry_path": f"generated_assets/{name.replace(' ', '_')}.glb",
        "placement_info": None,
        "bbox_min": [-dims[0] / 2, -dims[1] / 2, 0.0],
        "bbox_max": [dims[0] / 2, dims[1] / 2, dims[2]],
        "immutable": False,
        "scale_factor": 1.0,
    }


def classroom() -> dict:
    objects = {}
    for index in range(12):
        x, y = -2.7 + 1.8 * (index % 4), -2.2 + 2.0 * (index // 4)
        desk = obj("student classroom desk", xyz=(x, y, 0.0))
        chair = obj("classroom chair", xyz=(x, y + 0.65, 0.0))
        objects[f"classroom_desk_{index}"] = desk
        objects[f"classroom_chair_{index}"] = chair
    for name in (
        "teacher desk",
        "ergonomic teacher swivel chair",
        "storage cabinet",
        "book shelf cubby",
        "educational poster bulletin board",
        "classroom clock",
        "trash bin",
        "potted plant",
        "textbook",
        "notebook",
        "folder",
        "worksheet",
        "pencil holder",
        "classroom marker",
        "rubber eraser",
        "ruler",
        "glue stick",
        "scissors",
        "paper tray",
        "student backpack",
        "water bottle",
        "classroom storage bin",
        "dry-erase marker",
        "whiteboard eraser",
    ):
        objects[name.replace(" ", "_")] = obj(name)
    objects["whiteboard_0"] = obj("large classroom whiteboard with marker tray")
    return {"objects": objects}


def test_generated_classroom_semantic_variants_pass_without_counting_marker_tray() -> None:
    prompt = canonical_room_prompt("classroom_02", "a" * 64, "context")
    state = classroom()
    result = evaluate_room_inventory("classroom_02", state, prompt)
    assert result["status"] == "pass"
    assert result["counts"]["student_desks"] == 12
    assert result["counts"]["student_chairs"] == 12
    assert result["counts"]["teacher_chair"] == 1
    assert result["matched_object_ids"]["whiteboard"] == ["whiteboard_0"]
    assert result["matched_object_ids"]["markers"] == ["classroom_marker"]

    without_marker = copy.deepcopy(state)
    without_marker["objects"].pop("classroom_marker")
    failed = evaluate_room_inventory("classroom_02", without_marker, prompt)
    assert failed["status"] == "fail"
    assert failed["counts"]["markers"] == 0
    assert failed["counts"]["whiteboard"] == 1


def boys_room(*, urinal_dims=(1.3, 0.39, 1.12)) -> dict:
    objects = {
        "double_sink_vanity": obj("double sink vanity", dims=(1.6, 0.53, 0.94)),
        "urinal_station": obj("urinal station with two white urinals", dims=urinal_dims),
        "mirror_0": obj("mirror"),
        "mirror_1": obj("mirror"),
        "soap": obj("soap dispenser"),
        "dryer": obj("hand dryer"),
        "trash": obj("trash bin"),
    }
    for index in range(2):
        objects[f"toilet_{index}"] = obj("toilet fixture")
    for index in range(4):
        objects[f"divider_{index}"] = obj("privacy screen divider")
    return {"objects": objects}


def test_dimension_attested_double_sink_and_urinal_station_expand_fail_closed() -> None:
    prompt = canonical_room_prompt("boys_toilet", "b" * 64, "context")
    passing = evaluate_room_inventory("boys_toilet", boys_room(), prompt)
    assert passing["status"] == "pass"
    assert passing["counts"]["sinks"] == 2
    assert passing["counts"]["urinals"] == 2
    assert passing["counts"]["partitions"] == 2
    assert passing["fixture_multiplicity_evidence"]["double_sink_vanity"]["component_count"] == 2
    assert passing["fixture_multiplicity_evidence"]["urinal_station"]["component_count"] == 2

    too_small = evaluate_room_inventory(
        "boys_toilet", boys_room(urinal_dims=(0.5, 0.2, 0.4)), prompt
    )
    assert too_small["status"] == "fail"
    assert too_small["counts"]["urinals"] == 1
    assert any("plausible measured-dimension" in issue for issue in too_small["critical_issues"])
