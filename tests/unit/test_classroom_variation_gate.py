from __future__ import annotations

import json

from pathlib import Path

import pytest

from scripts.classroom_variation_gate import (
    CLASSROOM_IDS,
    CLASSROOM_PAIRS,
    evaluate,
    verify_output,
)


class FakeVLM:
    def __init__(self, response: dict):
        self.response = response
        self.calls = []

    def create_completion(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _object(name: str, x: float, y: float) -> dict:
    return {
        "name": name,
        "description": name,
        "object_type": "FURNITURE",
        "transform": {"translation": [x, y, 0.0]},
    }


def _setup(
    tmp_path: Path,
    *,
    duplicate_pair: bool = False,
    cosmetic_only_pair: bool = False,
) -> tuple[Path, Path]:
    scene = tmp_path / "scene"
    reviews = scene / "review" / "room_review_renders"
    reviews.mkdir(parents=True)
    for room_index, room_id in enumerate(CLASSROOM_IDS):
        seating_variation = (
            0
            if (duplicate_pair or cosmetic_only_pair) and room_index in {0, 1}
            else room_index
        )
        decor_variation = 0 if duplicate_pair and room_index in {0, 1} else room_index
        objects = {}
        for index in range(12):
            column, row = index % 4, index // 4
            x = column * 1.5
            y = row * 1.4
            if index == 11:
                x += seating_variation * 0.12
            objects[f"desk_{index}"] = _object("student desk", x, y)
            objects[f"chair_{index}"] = _object("student chair", x, y + 0.55)
        objects[f"identity_{decor_variation}"] = _object(
            f"classroom identity feature {decor_variation}",
            -2.0,
            1.0 + decor_variation * 0.1,
        )
        state = (
            scene
            / f"room_{room_id}"
            / "scene_states"
            / "final_scene"
            / "scene_state.json"
        )
        state.parent.mkdir(parents=True)
        state.write_text(json.dumps({"objects": objects}), encoding="utf-8")
        (reviews / f"{room_id}_top.png").write_bytes(
            b"\x89PNG\r\n\x1a\n" + room_id.encode("ascii")
        )
    return scene, reviews


def _passing_response() -> dict:
    return {
        "variation_quality_score": 9,
        "classrooms": {
            room_id: {
                "status": "pass",
                "distinctive_features": [
                    f"Distinct seating geometry for {room_id}",
                    f"Distinct decorated zone for {room_id}",
                ],
                "seating_layout": f"Specific non-duplicated layout for {room_id}",
            }
            for room_id in CLASSROOM_IDS
        },
        "too_similar_pairs": [],
        "pairwise_comparisons": [
            {
                "rooms": [first, second],
                "status": "distinct",
                "specific_differences": [
                    f"Different seating geometry between {first} and {second}",
                    f"Different teaching and storage zones between {first} and {second}",
                ],
                "seating_layout_difference": (
                    f"{first} and {second} use meaningfully different desk arrangements"
                ),
            }
            for first, second in CLASSROOM_PAIRS
        ],
    }


def test_six_distinct_classrooms_pass_and_are_hash_bound(tmp_path: Path) -> None:
    scene, reviews = _setup(tmp_path)
    service = FakeVLM(_passing_response())

    result = evaluate(scene, reviews, vlm_service=service)
    output = tmp_path / "variation.json"
    output.write_text(json.dumps(result), encoding="utf-8")

    assert result["status"] == "pass"
    assert len(result["fingerprints"]) == 6
    assert len(result["evidence"]) == 12
    assert len(service.calls) == 1
    assert verify_output(output)["status"] == "pass"


def test_deterministically_identical_pair_fails_even_if_vlm_claims_pass(
    tmp_path: Path,
) -> None:
    scene, reviews = _setup(tmp_path, duplicate_pair=True)

    result = evaluate(scene, reviews, vlm_service=FakeVLM(_passing_response()))

    assert result["status"] == "fail"
    assert any(
        "classroom_01 and classroom_02" in issue
        for issue in result["critical_issues"]
    )


def test_cosmetic_change_does_not_hide_duplicate_seating_layout(
    tmp_path: Path,
) -> None:
    scene, reviews = _setup(tmp_path, cosmetic_only_pair=True)

    result = evaluate(scene, reviews, vlm_service=FakeVLM(_passing_response()))

    assert result["status"] == "fail"
    assert any(
        "identical seating layout" in issue
        and "classroom_01 and classroom_02" in issue
        for issue in result["critical_issues"]
    )


def test_visual_too_similar_pair_or_low_score_fails(tmp_path: Path) -> None:
    scene, reviews = _setup(tmp_path)
    response = _passing_response()
    response["variation_quality_score"] = 6
    response["too_similar_pairs"] = [["classroom_04", "classroom_05"]]
    for item in response["pairwise_comparisons"]:
        if item["rooms"] == ["classroom_04", "classroom_05"]:
            item["status"] = "too_similar"

    result = evaluate(scene, reviews, vlm_service=FakeVLM(response))

    assert result["status"] == "fail"
    assert any("below" in issue for issue in result["critical_issues"])
    assert any("duplicated" in issue for issue in result["critical_issues"])


def test_verify_rejects_mutated_state_and_malformed_vlm_schema(tmp_path: Path) -> None:
    scene, reviews = _setup(tmp_path)
    result = evaluate(scene, reviews, vlm_service=FakeVLM(_passing_response()))
    output = tmp_path / "variation.json"
    output.write_text(json.dumps(result), encoding="utf-8")
    state = (
        scene
        / "room_classroom_03"
        / "scene_states"
        / "final_scene"
        / "scene_state.json"
    )
    state.write_text(state.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact changed"):
        verify_output(output)

    malformed = _passing_response()
    malformed["classrooms"].pop("classroom_06")
    with pytest.raises(ValueError, match="exactly all six"):
        evaluate(scene, reviews, vlm_service=FakeVLM(malformed))


def test_variation_requires_all_fifteen_pairwise_verdicts(tmp_path: Path) -> None:
    scene, reviews = _setup(tmp_path)
    missing = _passing_response()
    missing["pairwise_comparisons"].pop()

    with pytest.raises(ValueError, match="all 15"):
        evaluate(scene, reviews, vlm_service=FakeVLM(missing))

    inconsistent = _passing_response()
    inconsistent["too_similar_pairs"] = [["classroom_01", "classroom_02"]]
    with pytest.raises(ValueError, match="disagrees"):
        evaluate(scene, reviews, vlm_service=FakeVLM(inconsistent))
