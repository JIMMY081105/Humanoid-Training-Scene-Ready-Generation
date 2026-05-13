from __future__ import annotations

import hashlib
import json

import pytest

from scripts.school_room_contract import (
    PROFILE,
    ROOM_IDS,
    bind_layout_prompts,
    canonical_room_prompt,
    collect_required_articulated_roles,
    evaluate_room_inventory,
)
from scripts.room_self_exam import RoomSpec, _score_room


def _object(
    name: str,
    *,
    articulated_source: str | None = None,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict:
    metadata = {}
    if articulated_source:
        metadata = {
            "is_articulated": True,
            "articulated_source": articulated_source,
            "articulated_id": f"{articulated_source}/{name.replace(' ', '_')}",
        }
    return {
        "object_id": name.replace(" ", "_"),
        "name": name,
        "description": name,
        "object_type": "FURNITURE",
        "metadata": metadata,
        "transform": {
            "translation": list(translation),
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "geometry_path": f"generated_assets/{name.replace(' ', '_')}.glb",
        "placement_info": None,
        "bbox_min": [-0.05, -0.05, 0.0],
        "bbox_max": [0.05, 0.05, 0.1],
        "immutable": False,
        "scale_factor": 1.0,
    }


def _classroom_state() -> dict:
    objects = {}
    for index in range(12):
        column = index % 4
        row = index // 4
        desk_position = (-2.7 + 1.8 * column, -2.2 + 2.0 * row, 0.0)
        chair_position = (desk_position[0], desk_position[1] + 0.65, 0.0)
        objects[f"student_desk_{index}"] = _object(
            "student desk", translation=desk_position
        )
        objects[f"student_chair_{index}"] = _object(
            "student chair", translation=chair_position
        )
    for name in (
        "teacher desk",
        "teacher chair",
        "front whiteboard",
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
        "whiteboard board marker",
        "whiteboard eraser",
    ):
        objects[name.replace(" ", "_")] = _object(name)
    return {"objects": objects}


def test_bind_layout_prompts_is_complete_hash_bound_and_idempotent(tmp_path) -> None:
    layout_path = tmp_path / "house_layout.json"
    layout = {
        "rooms": [
            {"id": room_id, "prompt": f"generated context for {room_id}"}
            for room_id in ROOM_IDS
        ],
        "placed_rooms": [
            {"room_id": room_id, "prompt": f"generated context for {room_id}"}
            for room_id in ROOM_IDS
        ],
    }
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    effective_hash = "a" * 64
    manifest_path = tmp_path / "input_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "effective_prompt_sha256": effective_hash,
                "pipeline_contract": {"id": "scenesmith_full_quality_v1"},
            }
        ),
        encoding="utf-8",
    )
    evidence_path = tmp_path / "binding.json"

    first = bind_layout_prompts(layout_path, manifest_path, evidence_path)
    first_bytes = layout_path.read_bytes()
    second = bind_layout_prompts(layout_path, manifest_path, evidence_path)

    assert first["status"] == "pass"
    assert second["status"] == "pass"
    assert layout_path.read_bytes() == first_bytes
    bound = json.loads(layout_path.read_text(encoding="utf-8"))
    for room in [*bound["rooms"], *bound["placed_rooms"]]:
        room_id = room.get("id") or room.get("room_id")
        assert f"room_id={room_id} profile={PROFILE}" in room["prompt"]
        assert effective_hash in room["prompt"]
        assert f"generated context for {room_id}" in room["prompt"]


def test_classroom_inventory_requires_exactly_twelve_desks_and_chairs() -> None:
    prompt = canonical_room_prompt("classroom_02", "b" * 64, "context")
    passing = evaluate_room_inventory("classroom_02", _classroom_state(), prompt)

    state = _classroom_state()
    state["objects"].pop("student_desk_11")
    failing = evaluate_room_inventory("classroom_02", state, prompt)

    assert passing["status"] == "pass"
    assert passing["counts"]["student_desks"] == 12
    assert passing["counts"]["student_chairs"] == 12
    assert failing["status"] == "fail"
    assert any("student desks" in issue for issue in failing["critical_issues"])


def test_prompt_binding_rejects_duplicate_or_ambiguous_room_specs(tmp_path) -> None:
    layout_path = tmp_path / "house_layout.json"
    layout = {
        "rooms": [{"id": room_id, "prompt": room_id} for room_id in ROOM_IDS],
        "placed_rooms": [
            {"room_id": room_id, "prompt": room_id} for room_id in ROOM_IDS
        ],
    }
    layout["rooms"].append(
        {"id": "classroom_01", "prompt": "duplicate worker-selected prompt"}
    )
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    manifest_path = tmp_path / "input_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "effective_prompt_sha256": "f" * 64,
                "pipeline_contract": {"id": "scenesmith_full_quality_v1"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="duplicates=.*classroom_01"):
        bind_layout_prompts(layout_path, manifest_path, tmp_path / "binding.json")


def test_inventory_rejects_unbound_generated_prompt() -> None:
    result = evaluate_room_inventory(
        "classroom_01", _classroom_state(), "shortened floor-planner summary"
    )

    assert result["status"] == "fail"
    assert any("prompt binding failed" in issue.lower() for issue in result["critical_issues"])


@pytest.mark.parametrize(
    ("mutation", "expected_fragment"),
    (
        (lambda obj: obj.update(geometry_path="generated_assets/cube.glb"), "placeholder"),
        (lambda obj: obj.update(immutable=True), "immutable"),
        (lambda obj: obj.pop("bbox_max"), "bounds"),
    ),
)
def test_labeled_primitive_immutable_or_unbounded_object_cannot_count(
    mutation, expected_fragment: str
) -> None:
    state = _classroom_state()
    mutation(state["objects"]["student_desk_0"])
    prompt = canonical_room_prompt("classroom_02", "b" * 64, "context")

    result = evaluate_room_inventory("classroom_02", state, prompt)

    assert result["status"] == "fail"
    rejected = result["invalid_physical_candidates"]["student_desk_0"]
    assert any(expected_fragment in issue for issue in rejected["issues"])


def test_production_inventory_requires_exact_prompt_binding() -> None:
    prompt = canonical_room_prompt("classroom_02", "b" * 64, "context")
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    missing = evaluate_room_inventory(
        "classroom_02",
        _classroom_state(),
        prompt,
        require_prompt_binding=True,
    )
    wrong = evaluate_room_inventory(
        "classroom_02",
        _classroom_state(),
        prompt,
        expected_prompt_sha256="0" * 64,
        require_prompt_binding=True,
    )
    passing = evaluate_room_inventory(
        "classroom_02",
        _classroom_state(),
        prompt,
        expected_prompt_sha256=digest,
        require_prompt_binding=True,
    )

    assert missing["status"] == "fail"
    assert wrong["status"] == "fail"
    assert passing["status"] == "pass"


def test_one_spoofed_object_cannot_satisfy_multiple_classroom_items() -> None:
    state = _classroom_state()
    for key in (
        "textbook",
        "notebook",
        "folder",
        "worksheet",
        "pencil_holder",
        "classroom_marker",
        "rubber_eraser",
        "ruler",
        "glue_stick",
        "scissors",
    ):
        state["objects"].pop(key)
    state["objects"]["all_supplies"] = _object(
        "textbook notebook folder worksheet pencil holder marker eraser ruler glue stick scissors"
    )

    result = evaluate_room_inventory(
        "classroom_03",
        state,
        canonical_room_prompt("classroom_03", "d" * 64, "context"),
    )

    assert result["status"] == "fail"
    assert sum(
        result["counts"][key]
        for key in (
            "textbooks",
            "notebooks",
            "folders",
            "worksheets",
            "pencil_holders",
            "markers",
            "erasers",
            "rulers",
            "glue_sticks",
            "scissors",
        )
    ) == 1


def test_classroom_requires_twelve_spatially_paired_desk_chair_sets() -> None:
    state = _classroom_state()
    state["objects"]["student_chair_11"]["transform"]["translation"] = [20, 20, 0]

    result = evaluate_room_inventory(
        "classroom_04",
        state,
        canonical_room_prompt("classroom_04", "e" * 64, "context"),
    )

    pairing = result["spatial_checks"]["desk_chair_pairing"]
    assert result["status"] == "fail"
    assert pairing["pair_count"] == 11
    assert pairing["status"] == "fail"


def test_desk_chair_pairs_require_orientation_floor_support_and_clearance() -> None:
    prompt = canonical_room_prompt("classroom_04", "e" * 64, "context")

    sideways = _classroom_state()
    sideways["objects"]["student_chair_11"]["transform"]["rotation_wxyz"] = [
        2**-0.5,
        0.0,
        0.0,
        2**-0.5,
    ]
    elevated = _classroom_state()
    elevated["objects"]["student_chair_11"]["transform"]["translation"][2] = 1.0
    overlapping = _classroom_state()
    overlapping["objects"]["student_desk_11"]["transform"]["translation"] = list(
        overlapping["objects"]["student_desk_10"]["transform"]["translation"]
    )

    for state in (sideways, elevated, overlapping):
        result = evaluate_room_inventory("classroom_04", state, prompt)
        assert result["status"] == "fail"
        assert result["spatial_checks"]["desk_chair_pairing"]["status"] == "fail"


def test_required_articulated_roles_and_artiverse_source_are_enforced() -> None:
    states = {
        "library": {
            "objects": {
                "bookcase": _object(
                    "openable library bookcase cabinet with hinged glass doors",
                    articulated_source="artiverse",
                )
            }
        },
        "storage_room": {
            "objects": {
                "utility": _object(
                    "openable school supply utility cabinet with two hinged doors",
                    articulated_source="artvip",
                )
            }
        },
        "classroom_01": {
            "objects": {
                "filing": _object(
                    "teacher filing cabinet with operable drawers",
                    articulated_source="artvip",
                )
            }
        },
    }

    passing = collect_required_articulated_roles(states)
    states["library"]["objects"]["bookcase"]["metadata"][
        "articulated_source"
    ] = "artvip"
    no_artiverse = collect_required_articulated_roles(states)
    states["storage_room"]["objects"] = {}
    missing_role = collect_required_articulated_roles(states)

    assert passing["status"] == "pass"
    assert passing["artiverse_roles"] == ["library_glass_door_bookcase"]
    assert no_artiverse["status"] == "fail"
    assert any("Artiverse" in issue for issue in no_artiverse["critical_issues"])
    assert missing_role["status"] == "fail"
    assert "school_supply_two_door_utility_cabinet" in missing_role["missing_roles"]


def test_canonical_prompt_records_effective_prompt_hash() -> None:
    digest = hashlib.sha256(b"effective prompt").hexdigest()
    prompt = canonical_room_prompt("library", digest, "generated library context")

    assert digest in prompt
    assert "hinged glass doors" in prompt
    assert "generated library context" in prompt


def test_deterministic_room_gate_blocks_semantic_inventory_drift() -> None:
    prompt = canonical_room_prompt("classroom_01", "c" * 64, "context")
    spec = RoomSpec("classroom_01", 9.0, 7.5, prompt)
    passing = _score_room(
        "classroom_01",
        spec,
        _classroom_state(),
        ["classroom_01_top.png"],
        32,
        contract_profile=PROFILE,
        expected_prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    )
    state = _classroom_state()
    state["objects"].pop("student_chair_11")
    failing = _score_room(
        "classroom_01",
        spec,
        state,
        ["classroom_01_top.png"],
        32,
        contract_profile=PROFILE,
        expected_prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    )

    assert passing["status"] == "pass"
    assert passing["metrics"]["semantic_inventory"]["status"] == "pass"
    assert failing["status"] == "fail"
    assert failing["scores"]["prompt_alignment"] == 0


def test_production_room_gate_cannot_omit_profile_or_binding() -> None:
    prompt = canonical_room_prompt("classroom_01", "c" * 64, "context")
    spec = RoomSpec("classroom_01", 9.0, 7.5, prompt)

    no_profile = _score_room(
        "classroom_01",
        spec,
        _classroom_state(),
        ["classroom_01_top.png"],
        32,
        production_mode=True,
    )
    no_binding = _score_room(
        "classroom_01",
        spec,
        _classroom_state(),
        ["classroom_01_top.png"],
        32,
        contract_profile=PROFILE,
    )

    assert no_profile["status"] == "fail"
    assert no_binding["status"] == "fail"
