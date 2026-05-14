import argparse
import hashlib
import json

from pathlib import Path

import pytest

from scripts.assemble_final_house_and_render import _verify_room_gates
from scripts.room_visual_self_exam import (
    ROOM_SCORE_KEYS,
    evaluate_room,
    room_visual_requirements,
    run_gate as run_room_gate,
    validate_requirement_evidence,
)
from scripts.school_room_contract import PROFILE, ROOM_IDS, bind_layout_prompts
from scripts.whole_floor_reference_gate import (
    FLOOR_SCORE_KEYS,
    evaluate_whole_floor,
    run_gate as run_whole_floor_gate,
    validate_exact_layout,
)


class FakeVLMService:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create_completion(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _image(path: Path) -> Path:
    path.write_bytes(b"\x89PNG\r\n\x1a\nmock-image:" + path.name.encode())
    return path


def _room_gate(*, status="pass", issues=None, score=9):
    return {
        "room_id": "classroom_01",
        "status": status,
        "scores": {key: score for key in ROOM_SCORE_KEYS},
        "critical_issues": issues or [],
        "repair_instructions": [],
    }


def _assessment(keys, *, score=8, issues=None, requirement_keys=()):
    return json.dumps(
        {
            "scores": {key: score for key in keys},
            "critical_issues": issues or [],
            "repair_instructions": [],
            "observations": ["Evidence is visible in the supplied views."],
            "requirement_evidence": {
                key: {
                    "status": "pass",
                    "view_indices": [1, 2],
                    "observation": f"Visible evidence for {key} is clear in the room views.",
                }
                for key in requirement_keys
            },
        }
    )


def _room_images(tmp_path: Path):
    reference = _image(tmp_path / "reference.png")
    reviews = [
        _image(tmp_path / "classroom_01_top.png"),
        _image(tmp_path / "classroom_01_oblique_a.png"),
        _image(tmp_path / "classroom_01_oblique_b.png"),
    ]
    return reference, reviews


def test_contract_visual_checklist_requires_every_item_and_view_citation() -> None:
    required = room_visual_requirements("classroom_02")
    evidence = {
        key: {
            "status": "pass",
            "view_indices": [1],
            "observation": f"The supplied view visibly proves {key}.",
        }
        for key in required
    }

    normalized, failures = validate_requirement_evidence(
        evidence, required, review_count=3
    )
    assert set(normalized) == set(required)
    assert failures == []

    evidence.pop(next(iter(required)))
    with pytest.raises(ValueError, match="keys differ"):
        validate_requirement_evidence(evidence, required, review_count=3)


def test_contract_visual_checklist_explicit_failed_item_blocks_pass() -> None:
    required = room_visual_requirements("library")
    evidence = {
        key: {
            "status": "pass",
            "view_indices": [1, 3],
            "observation": f"Visible library evidence for {key}.",
        }
        for key in required
    }
    evidence["inventory:bookcases"]["status"] = "fail"
    evidence["inventory:bookcases"]["observation"] = (
        "Only three distinct shelf units are visible."
    )

    _normalized, failures = validate_requirement_evidence(
        evidence, required, review_count=3
    )
    assert failures == [
        "Visual checklist requirement failed (inventory:bookcases): "
        "Only three distinct shelf units are visible."
    ]


def _room_evidence_files(tmp_path: Path):
    scene_state = tmp_path / "scene_state.json"
    scene_state.write_text('{"objects": {}}', encoding="utf-8")
    house_layout = tmp_path / "house_layout.json"
    house_layout.write_text('{"rooms": []}', encoding="utf-8")
    return scene_state, house_layout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_cutaway_evidence(
    review_dir: Path, room_id: str, source_blend: Path, source_state: Path | None = None
) -> Path:
    view_names = ("top", "oblique_a", "oblique_b")
    views = []
    for view_name in view_names:
        image = (review_dir / f"{room_id}_{view_name}.png").resolve()
        views.append(
            {
                "view_name": view_name,
                "image": str(image),
                "image_size_bytes": image.stat().st_size,
                "image_sha256": _sha256(image),
                "cutaway": {
                    "established": True,
                    "view_name": view_name,
                    "overhead_state": "hidden",
                    "hidden_overhead": [{"name": "ceiling"}],
                    "hidden_camera_side_walls": (
                        [] if view_name == "top" else [{"name": "near_wall"}]
                    ),
                    "visible_far_wall_object_names": ["far_wall"],
                    "visible_floor_object_names": ["floor"],
                    "visible_content_object_names": ["student_desk"],
                    "classified_content_count": 1,
                },
            }
        )
    source_state = source_state or source_blend.with_name("scene_state.json")
    state_record = {
        "path": str(source_state.resolve()),
        "size_bytes": source_state.stat().st_size,
        "sha256": _sha256(source_state),
    }
    blend_record = {
        "path": str(source_blend.resolve()),
        "size_bytes": source_blend.stat().st_size,
        "sha256": _sha256(source_blend),
    }
    derivation_payload = {
        "schema_id": "scenesmith_state_blend_render_derivation_v1",
        "schema_version": 1,
        "algorithm": "sha256",
        "source_state": state_record,
        "source_blend": blend_record,
        "renders": [
            {
                "view_name": view["view_name"],
                "path": view["image"],
                "size_bytes": view["image_size_bytes"],
                "sha256": view["image_sha256"],
            }
            for view in views
        ],
    }
    evidence = {
        "schema_id": "scenesmith_room_cutaway_review_v1",
        "schema_version": 1,
        "status": "pass",
        "room_id": room_id,
        "expected_views": list(view_names),
        "rendered_view_count": 3,
        "source_state": state_record,
        "source_blend": blend_record,
        "classification": {
            "overhead": [{"name": "ceiling"}],
            "wall": [{"name": "near_wall"}, {"name": "far_wall"}],
            "floor": [{"name": "floor"}],
            "combined_envelope": [],
            "content": [{"name": "student_desk"}],
        },
        "views": views,
        "derivation_receipt": {
            **derivation_payload,
            "attestation": {
                "algorithm": "sha256",
                "sha256": hashlib.sha256(
                    json.dumps(
                        derivation_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            },
        },
    }
    path = review_dir / f"{room_id}_cutaway_evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    return path


def _passing_visual_gate_fixture(tmp_path: Path):
    scene_dir = tmp_path / "scene_000"
    scene_dir.mkdir()
    layout = scene_dir / "house_layout.json"
    layout.write_text(
        json.dumps(
            {
                "rooms": [
                    {
                        "id": "classroom_01",
                        "prompt": "Warm classroom with twelve student desks.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    scene_state = (
        scene_dir
        / "room_classroom_01"
        / "scene_states"
        / "final_scene"
        / "scene_state.json"
    )
    scene_state.parent.mkdir(parents=True)
    scene_state.write_text('{"objects": {}}', encoding="utf-8")
    deterministic_dir = tmp_path / "deterministic"
    deterministic_dir.mkdir()
    (deterministic_dir / "classroom_01.json").write_text(
        json.dumps(_room_gate()), encoding="utf-8"
    )
    review_dir = tmp_path / "reviews"
    review_dir.mkdir()
    reviews = [
        _image(review_dir / "classroom_01_top.png"),
        _image(review_dir / "classroom_01_oblique_a.png"),
        _image(review_dir / "classroom_01_oblique_b.png"),
    ]
    reference = _image(tmp_path / "reference.png")
    gate_dir = tmp_path / "visual_gate"
    args = argparse.Namespace(
        scene_dir=scene_dir,
        deterministic_gate_dir=deterministic_dir,
        review_dir=review_dir,
        reference_image=reference,
        output_dir=gate_dir,
        rooms=None,
        threshold=7.0,
        minimum_review_images=3,
        maximum_review_images=6,
        model="gpt-5.2",
        vlm_backend="openai",
    )
    summary = run_room_gate(
        args,
        vlm_service=FakeVLMService(_assessment(ROOM_SCORE_KEYS, score=8)),
    )
    assert summary["status"] == "pass"
    return {
        "scene_dir": scene_dir,
        "layout": layout,
        "scene_state": scene_state,
        "deterministic_dir": deterministic_dir,
        "review_dir": review_dir,
        "reviews": reviews,
        "reference": reference,
        "gate_dir": gate_dir,
    }


def _existing_summary_args(fixture, *, rooms=None):
    return argparse.Namespace(
        scene_dir=fixture["scene_dir"],
        deterministic_gate_dir=fixture["deterministic_dir"],
        review_dir=fixture["review_dir"],
        reference_image=fixture["reference"],
        output_dir=fixture["gate_dir"],
        rooms=rooms,
        summarize_existing=True,
        threshold=7.0,
        minimum_review_images=3,
        maximum_review_images=6,
        model="gpt-5.2",
        vlm_backend="openai",
    )


def _valid_layout():
    rooms = {
        "main_corridor": ([-2.0, 0.0], 4.0, 10.0),
        "classroom_01": ([-10.0, 0.0], 4.0, 4.0),
        "classroom_03": ([-10.0, 4.0], 4.0, 4.0),
        "classroom_04": ([-10.0, 8.0], 4.0, 4.0),
        "classroom_02": ([6.0, 0.0], 4.0, 4.0),
        "classroom_06": ([6.0, 4.0], 4.0, 4.0),
        "classroom_05": ([6.0, 8.0], 4.0, 4.0),
        "library": ([-2.0, -5.0], 4.0, 4.0),
        "boys_toilet": ([-8.0, -4.0], 2.0, 2.0),
        "girls_toilet": ([-5.0, -4.0], 2.0, 2.0),
        "storage_room": ([6.0, 3.5], 2.0, 2.0),
    }
    return {
        "placed_rooms": [
            {
                "room_id": room_id,
                "position": position,
                "width": width,
                "depth": depth,
            }
            for room_id, (position, width, depth) in rooms.items()
        ]
    }


def _whole_floor_images(tmp_path: Path):
    return {
        "overview_top": _image(tmp_path / "overview_top.png"),
        "overview_isometric": _image(tmp_path / "overview_isometric.png"),
        "overview_front": _image(tmp_path / "overview_front.png"),
    }


def _write_house_cutaway_evidence(
    combined_dir: Path, images: dict[str, Path]
) -> Path:
    blend = combined_dir / "house.blend"
    blend.write_bytes(b"house-blend")
    state = combined_dir / "house_state.json"
    if not state.is_file():
        raise AssertionError("house_state.json must exist before cutaway evidence")
    views = []
    for name in ("overview_top", "overview_isometric", "overview_front"):
        image = images[name]
        views.append(
            {
                "view_name": name,
                "image": str(image.resolve()),
                "image_size_bytes": image.stat().st_size,
                "image_sha256": _sha256(image),
                "cutaway": {
                    "established": True,
                    "view_name": name,
                    "overhead_state": "verified_absent",
                    "hidden_overhead": [],
                    "hidden_camera_side_walls": (
                        [] if name == "overview_top" else ["near_wall"]
                    ),
                    "visible_far_wall_object_names": ["far_wall"],
                    "visible_floor_object_names": ["floor"],
                    "visible_content_object_names": ["desk"],
                },
            }
        )
    state_record = {
        "path": str(state.resolve()),
        "size_bytes": state.stat().st_size,
        "sha256": _sha256(state),
    }
    blend_record = {
        "path": str(blend.resolve()),
        "size_bytes": blend.stat().st_size,
        "sha256": _sha256(blend),
    }
    derivation_payload = {
        "schema_id": "scenesmith_state_blend_render_derivation_v1",
        "schema_version": 1,
        "algorithm": "sha256",
        "source_state": state_record,
        "source_blend": blend_record,
        "renders": [
            {
                "view_name": view["view_name"],
                "path": view["image"],
                "size_bytes": view["image_size_bytes"],
                "sha256": view["image_sha256"],
            }
            for view in views
        ],
    }
    evidence = {
        "schema_id": "scenesmith_house_cutaway_review_v1",
        "schema_version": 1,
        "status": "pass",
        "expected_views": ["overview_top", "overview_isometric", "overview_front"],
        "rendered_view_count": 3,
        "source_state": state_record,
        "source_blend": blend_record,
        "classification": {
            "overhead": [],
            "wall": [{"name": "near_wall"}, {"name": "far_wall"}],
            "floor": [{"name": "floor"}],
            "combined_envelope": [],
            "content": [{"name": "desk"}],
        },
        "views": views,
        "derivation_receipt": {
            **derivation_payload,
            "attestation": {
                "algorithm": "sha256",
                "sha256": hashlib.sha256(
                    json.dumps(
                        derivation_payload, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
            },
        },
    }
    path = combined_dir / "outlook_renders" / "overview_cutaway_evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    return path


def _attached_image_count(call):
    user_message = call["messages"][1]
    return sum(
        item.get("type") == "image_url"
        for item in user_message["content"]
        if isinstance(item, dict)
    )


def test_room_visual_gate_passes_with_three_views_and_combines_scores(tmp_path):
    reference, reviews = _room_images(tmp_path)
    scene_state, house_layout = _room_evidence_files(tmp_path)
    service = FakeVLMService(_assessment(ROOM_SCORE_KEYS, score=8))

    result = evaluate_room(
        room_id="classroom_01",
        room_prompt="Warm classroom with twelve desks facing a whiteboard.",
        deterministic_result=_room_gate(score=9),
        review_images=reviews,
        reference_image=reference,
        scene_state_path=scene_state,
        house_layout_path=house_layout,
        vlm_service=service,
    )

    assert result["status"] == "pass"
    assert result["scores"] == {key: 8.0 for key in ROOM_SCORE_KEYS}
    assert len(service.calls) == 1
    assert _attached_image_count(service.calls[0]) == 4
    assert "twelve desks" in str(service.calls[0]["messages"][1]["content"])
    assert result["evidence"]["scene_state"] == {
        "path": str(scene_state.resolve()),
        "sha256": _sha256(scene_state),
    }
    assert [entry["sha256"] for entry in result["evidence"]["review_images"]] == [
        _sha256(path) for path in reviews
    ]


def test_deterministic_room_failure_is_never_overridden_or_sent_to_vlm(tmp_path):
    reference, reviews = _room_images(tmp_path)
    scene_state, house_layout = _room_evidence_files(tmp_path)
    service = FakeVLMService(_assessment(ROOM_SCORE_KEYS, score=10))

    result = evaluate_room(
        room_id="classroom_01",
        room_prompt="classroom",
        deterministic_result=_room_gate(
            status="fail", issues=["Doorway is blocked."], score=10
        ),
        review_images=reviews,
        reference_image=reference,
        scene_state_path=scene_state,
        house_layout_path=house_layout,
        vlm_service=service,
    )

    assert result["status"] == "fail"
    assert "Doorway is blocked." in result["critical_issues"]
    assert service.calls == []


def test_room_gate_requires_three_review_images_without_calling_vlm(tmp_path):
    reference, reviews = _room_images(tmp_path)
    scene_state, house_layout = _room_evidence_files(tmp_path)
    service = FakeVLMService(_assessment(ROOM_SCORE_KEYS, score=10))

    result = evaluate_room(
        room_id="classroom_01",
        room_prompt="classroom",
        deterministic_result=_room_gate(),
        review_images=reviews[:2],
        reference_image=reference,
        scene_state_path=scene_state,
        house_layout_path=house_layout,
        vlm_service=service,
    )

    assert result["status"] == "fail"
    assert any("at least 3" in issue for issue in result["critical_issues"])
    assert service.calls == []


def test_room_gate_fails_closed_on_malformed_vlm_json(tmp_path):
    reference, reviews = _room_images(tmp_path)
    scene_state, house_layout = _room_evidence_files(tmp_path)
    service = FakeVLMService("not-json")

    result = evaluate_room(
        room_id="classroom_01",
        room_prompt="classroom",
        deterministic_result=_room_gate(),
        review_images=reviews,
        reference_image=reference,
        scene_state_path=scene_state,
        house_layout_path=house_layout,
        vlm_service=service,
    )

    assert result["status"] == "fail"
    assert result["scores"] == {key: 0.0 for key in ROOM_SCORE_KEYS}
    assert any("failed closed" in issue for issue in result["critical_issues"])


def test_room_gate_fails_when_any_combined_score_is_below_seven(tmp_path):
    reference, reviews = _room_images(tmp_path)
    scene_state, house_layout = _room_evidence_files(tmp_path)
    payload = json.loads(_assessment(ROOM_SCORE_KEYS, score=8))
    payload["scores"]["prompt_alignment"] = 6
    service = FakeVLMService(json.dumps(payload))

    result = evaluate_room(
        room_id="classroom_01",
        room_prompt="classroom",
        deterministic_result=_room_gate(),
        review_images=reviews,
        reference_image=reference,
        scene_state_path=scene_state,
        house_layout_path=house_layout,
        vlm_service=service,
    )

    assert result["status"] == "fail"
    assert result["scores"]["prompt_alignment"] == 6.0


def test_room_gate_fails_if_evidence_changes_during_vlm_review(tmp_path):
    reference, reviews = _room_images(tmp_path)
    scene_state, house_layout = _room_evidence_files(tmp_path)

    class MutatingVLMService(FakeVLMService):
        def create_completion(self, **kwargs):
            reviews[0].write_bytes(reviews[0].read_bytes() + b"mutated")
            return super().create_completion(**kwargs)

    service = MutatingVLMService(_assessment(ROOM_SCORE_KEYS, score=10))
    result = evaluate_room(
        room_id="classroom_01",
        room_prompt="classroom",
        deterministic_result=_room_gate(),
        review_images=reviews,
        reference_image=reference,
        scene_state_path=scene_state,
        house_layout_path=house_layout,
        vlm_service=service,
    )

    assert result["status"] == "fail"
    assert any(
        "review_images[0] changed during visual review" in issue
        for issue in result["critical_issues"]
    )


def test_room_cli_runner_writes_final_contract_json(tmp_path):
    scene_dir = tmp_path / "scene_000"
    scene_dir.mkdir()
    (scene_dir / "house_layout.json").write_text(
        json.dumps(
            {
                "rooms": [
                    {
                        "id": "classroom_01",
                        "prompt": "Warm classroom with twelve student desks.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    scene_state = (
        scene_dir
        / "room_classroom_01"
        / "scene_states"
        / "final_scene"
        / "scene_state.json"
    )
    scene_state.parent.mkdir(parents=True)
    scene_state.write_text('{"objects": {}}', encoding="utf-8")
    deterministic_dir = tmp_path / "deterministic"
    deterministic_dir.mkdir()
    (deterministic_dir / "classroom_01.json").write_text(
        json.dumps(_room_gate()), encoding="utf-8"
    )
    review_dir = tmp_path / "reviews"
    review_dir.mkdir()
    _image(review_dir / "classroom_01_top.png")
    _image(review_dir / "classroom_01_oblique_a.png")
    _image(review_dir / "classroom_01_oblique_b.png")
    reference = _image(tmp_path / "reference.png")
    output_dir = tmp_path / "visual_gate"
    args = argparse.Namespace(
        scene_dir=scene_dir,
        deterministic_gate_dir=deterministic_dir,
        review_dir=review_dir,
        reference_image=reference,
        output_dir=output_dir,
        rooms=None,
        threshold=7.0,
        minimum_review_images=3,
        maximum_review_images=6,
        model="gpt-5.2",
        vlm_backend="openai",
    )

    summary = run_room_gate(
        args,
        vlm_service=FakeVLMService(_assessment(ROOM_SCORE_KEYS, score=8)),
    )

    assert summary["status"] == "pass"
    result = json.loads((output_dir / "classroom_01.json").read_text())
    assert result["room_id"] == "classroom_01"
    assert result["status"] == "pass"
    assert result["evidence"]["algorithm"] == "sha256"
    assert result["evidence"]["scene_state"]["path"] == str(scene_state.resolve())
    assert result["evidence"]["scene_state"]["sha256"] == _sha256(scene_state)
    assert len(result["evidence"]["review_images"]) == 3
    assert result["evidence"]["house_layout"]["sha256"] == _sha256(
        scene_dir / "house_layout.json"
    )
    assert result["evidence"]["reference_image"]["sha256"] == _sha256(reference)
    assert (output_dir / "summary.json").is_file()


def test_existing_room_summary_never_calls_vlm_or_rewrites_room_json(tmp_path):
    fixture = _passing_visual_gate_fixture(tmp_path)
    gate_path = fixture["gate_dir"] / "classroom_01.json"
    original_gate = gate_path.read_bytes()
    service = FakeVLMService(RuntimeError("VLM must not be called"))

    summary = run_room_gate(
        _existing_summary_args(fixture, rooms=["classroom_01"]),
        vlm_service=service,
    )

    assert summary["status"] == "pass"
    assert summary["room_count"] == 1
    assert summary["passed_rooms"] == ["classroom_01"]
    assert service.calls == []
    assert gate_path.read_bytes() == original_gate


def test_existing_room_summary_fails_when_bound_evidence_was_tampered(tmp_path):
    fixture = _passing_visual_gate_fixture(tmp_path)
    fixture["reviews"][0].write_bytes(
        fixture["reviews"][0].read_bytes() + b"tampered"
    )
    service = FakeVLMService(RuntimeError("VLM must not be called"))

    summary = run_room_gate(
        _existing_summary_args(fixture, rooms=["classroom_01"]),
        vlm_service=service,
    )

    assert summary["status"] == "fail"
    assert summary["failed_rooms"] == ["classroom_01"]
    assert service.calls == []


def test_existing_room_summary_fails_on_stale_gate_contract(tmp_path):
    fixture = _passing_visual_gate_fixture(tmp_path)
    gate_path = fixture["gate_dir"] / "classroom_01.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["threshold"] = 8.0
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    summary = run_room_gate(
        _existing_summary_args(fixture, rooms=["classroom_01"]),
        vlm_service=FakeVLMService(RuntimeError("VLM must not be called")),
    )

    assert summary["status"] == "fail"
    assert summary["failed_rooms"] == ["classroom_01"]


def test_existing_room_summary_fails_when_selected_room_json_is_missing(tmp_path):
    fixture = _passing_visual_gate_fixture(tmp_path)
    (fixture["gate_dir"] / "classroom_01.json").unlink()
    service = FakeVLMService(RuntimeError("VLM must not be called"))

    summary = run_room_gate(
        _existing_summary_args(fixture, rooms=["classroom_01"]),
        vlm_service=service,
    )

    assert summary["status"] == "fail"
    assert summary["passed_rooms"] == []
    assert summary["failed_rooms"] == ["classroom_01"]
    assert service.calls == []


def test_school_runner_all_room_visual_confirmation_is_vlm_free() -> None:
    runner = (
        Path(__file__).resolve().parents[2]
        / "remote_jobs"
        / "run_full_quality_school_sqz.sh"
    ).read_text(encoding="utf-8")
    block = runner.split(
        'echo "[gate] confirming every room gate together"', 1
    )[1].split(
        'echo "[gate] proving all six classrooms are spatially and visually distinct"',
        1,
    )[0]

    assert "--summarize-existing" in block
    assert "probe_openai" not in block


@pytest.mark.parametrize(
    ("target_name", "expected_label"),
    [
        ("scene_state", "scene_state"),
        ("layout", "house_layout"),
        ("reference", "reference_image"),
        ("review", "review_images[0]"),
    ],
)
def test_assembly_rejects_changed_visual_gate_evidence(
    tmp_path, target_name, expected_label
):
    fixture = _passing_visual_gate_fixture(tmp_path)
    assert (
        _verify_room_gates(
            fixture["gate_dir"], ["classroom_01"], fixture["scene_dir"]
        )
        == []
    )

    target = (
        fixture["reviews"][0]
        if target_name == "review"
        else fixture[target_name]
    )
    target.write_bytes(target.read_bytes() + b"changed-after-pass")

    failures = _verify_room_gates(
        fixture["gate_dir"], ["classroom_01"], fixture["scene_dir"]
    )
    assert any(expected_label in failure for failure in failures)
    assert any("SHA-256 mismatch" in failure for failure in failures)


def test_assembly_rejects_missing_review_evidence(tmp_path):
    fixture = _passing_visual_gate_fixture(tmp_path)
    fixture["reviews"][1].unlink()

    failures = _verify_room_gates(
        fixture["gate_dir"], ["classroom_01"], fixture["scene_dir"]
    )

    assert any("review_images[1]" in failure for failure in failures)
    assert any("evidence file is missing" in failure for failure in failures)


def test_assembly_rejects_legacy_passing_gate_without_hash_bindings(tmp_path):
    fixture = _passing_visual_gate_fixture(tmp_path)
    gate_path = fixture["gate_dir"] / "classroom_01.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate.pop("evidence")
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    failures = _verify_room_gates(
        fixture["gate_dir"], ["classroom_01"], fixture["scene_dir"]
    )

    assert failures == ["classroom_01: passing gate has no evidence manifest"]


def test_assembly_requires_canonical_final_state_path(tmp_path):
    fixture = _passing_visual_gate_fixture(tmp_path)
    alternate = tmp_path / "copied_scene_state.json"
    alternate.write_bytes(fixture["scene_state"].read_bytes())
    gate_path = fixture["gate_dir"] / "classroom_01.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["evidence"]["scene_state"] = {
        "path": str(alternate.resolve()),
        "sha256": _sha256(alternate),
    }
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    failures = _verify_room_gates(
        fixture["gate_dir"], ["classroom_01"], fixture["scene_dir"]
    )

    assert any("does not match canonical file" in failure for failure in failures)


def test_exact_layout_validation_accepts_reference_relative_room_order():
    result = validate_exact_layout(_valid_layout())

    assert result["status"] == "pass"
    assert result["actual_room_count"] == 11


def test_whole_floor_gate_passes_and_sends_reference_plus_three_overviews(tmp_path):
    reference = _image(tmp_path / "reference.png")
    overviews = _whole_floor_images(tmp_path)
    service = FakeVLMService(_assessment(FLOOR_SCORE_KEYS, score=8))

    result = evaluate_whole_floor(
        layout=_valid_layout(),
        overview_images=overviews,
        reference_image=reference,
        vlm_service=service,
    )

    assert result["status"] == "pass"
    assert len(service.calls) == 1
    assert _attached_image_count(service.calls[0]) == 4


def test_wrong_room_count_fails_whole_floor_gate_without_vlm(tmp_path):
    reference = _image(tmp_path / "reference.png")
    overviews = _whole_floor_images(tmp_path)
    layout = _valid_layout()
    layout["placed_rooms"] = [
        room
        for room in layout["placed_rooms"]
        if room["room_id"] != "storage_room"
    ]
    service = FakeVLMService(_assessment(FLOOR_SCORE_KEYS, score=10))

    result = evaluate_whole_floor(
        layout=layout,
        overview_images=overviews,
        reference_image=reference,
        vlm_service=service,
    )

    assert result["status"] == "fail"
    assert any("exactly 11" in issue for issue in result["critical_issues"])
    assert service.calls == []


def test_whole_floor_visual_critical_issue_forces_failure(tmp_path):
    reference = _image(tmp_path / "reference.png")
    overviews = _whole_floor_images(tmp_path)
    service = FakeVLMService(
        _assessment(
            FLOOR_SCORE_KEYS,
            score=9,
            issues=["Several desks visibly block the central corridor."],
        )
    )

    result = evaluate_whole_floor(
        layout=_valid_layout(),
        overview_images=overviews,
        reference_image=reference,
        vlm_service=service,
    )

    assert result["status"] == "fail"
    assert result["critical_issues"] == [
        "Several desks visibly block the central corridor."
    ]


def test_whole_floor_cli_runner_writes_result_json(tmp_path):
    scene_dir = tmp_path / "scene_000"
    scene_dir.mkdir()
    (scene_dir / "house_layout.json").write_text(
        json.dumps(_valid_layout()), encoding="utf-8"
    )
    overview_dir = scene_dir / "combined_house" / "outlook_renders"
    overview_dir.mkdir(parents=True)
    overview_images = _whole_floor_images(overview_dir)
    (scene_dir / "combined_house" / "house_state.json").write_text(
        json.dumps(
            {
                "rooms": [
                    room["room_id"] for room in _valid_layout()["placed_rooms"]
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_house_cutaway_evidence(scene_dir / "combined_house", overview_images)
    (scene_dir / "combined_house" / "artiverse_usage.json").write_text(
        json.dumps({"status": "pass", "surviving_instances": ["fixture_01"]}),
        encoding="utf-8",
    )
    reference = _image(tmp_path / "reference.png")
    (tmp_path / "input_manifest.json").write_text(
        json.dumps({"reference_image_sha256": _sha256(reference)}),
        encoding="utf-8",
    )
    output = scene_dir / "quality_gates" / "whole_floor_reference.json"
    args = argparse.Namespace(
        scene_dir=scene_dir,
        layout=None,
        overview_dir=None,
        reference_image=reference,
        output=output,
        threshold=7.0,
        model="gpt-5.2",
        vlm_backend="openai",
    )

    result = run_whole_floor_gate(
        args,
        vlm_service=FakeVLMService(_assessment(FLOOR_SCORE_KEYS, score=8)),
    )

    assert result["status"] == "pass"
    saved = json.loads(output.read_text())
    assert saved["status"] == "pass"
    assert saved["deterministic_layout_gate"]["actual_room_count"] == 11


def test_contract_room_gate_binds_full_prompt_manifest_and_deterministic_json(
    tmp_path,
):
    scene_dir = tmp_path / "scene_000"
    scene_dir.mkdir()
    layout_path = scene_dir / "house_layout.json"
    layout_path.write_text(
        json.dumps(
                {
                    "rooms": [
                        {"id": room_id, "prompt": f"generated {room_id}"}
                        for room_id in ROOM_IDS
                    ],
                    "placed_rooms": [
                        {
                            "room_id": room_id,
                            "prompt": f"generated {room_id}",
                        }
                        for room_id in ROOM_IDS
                    ],
                }
        ),
        encoding="utf-8",
    )
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    effective_prompt = "Complete immutable warm school prompt.\n"
    effective_prompt_path = input_dir / "prompt.txt"
    effective_prompt_path.write_text(effective_prompt, encoding="utf-8")
    reference = _image(tmp_path / "reference.png")
    manifest_path = input_dir / "input_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "effective_prompt_sha256": hashlib.sha256(
                    effective_prompt.encode("utf-8")
                ).hexdigest(),
                "reference_image_sha256": _sha256(reference),
                "pipeline_contract": {"id": "scenesmith_full_quality_v1"},
            }
        ),
        encoding="utf-8",
    )
    binding_path = scene_dir / "quality_gates" / "room_prompt_binding.json"
    bind_layout_prompts(layout_path, manifest_path, binding_path)

    scene_state = (
        scene_dir
        / "room_classroom_01"
        / "scene_states"
        / "final_scene"
        / "scene_state.json"
    )
    scene_state.parent.mkdir(parents=True)
    scene_state.write_text('{"objects": {}}', encoding="utf-8")
    deterministic_dir = (
        scene_dir / "quality_gates" / "room_self_exam_deterministic"
    )
    deterministic_dir.mkdir(parents=True)
    deterministic = _room_gate()
    deterministic["contract_profile"] = PROFILE
    (deterministic_dir / "classroom_01.json").write_text(
        json.dumps(deterministic), encoding="utf-8"
    )
    source_blend = scene_state.parent / "scene.blend"
    source_blend.write_bytes(b"blend")
    review_dir = scene_dir / "review" / "room_review_renders"
    review_dir.mkdir(parents=True)
    _image(review_dir / "classroom_01_top.png")
    _image(review_dir / "classroom_01_oblique_a.png")
    _image(review_dir / "classroom_01_oblique_b.png")
    cutaway_path = _write_cutaway_evidence(
        review_dir, "classroom_01", source_blend
    )
    output_dir = scene_dir / "quality_gates" / "room_self_exam"
    service = FakeVLMService(
        _assessment(
            ROOM_SCORE_KEYS,
            score=8,
            requirement_keys=room_visual_requirements("classroom_01"),
        )
    )
    args = argparse.Namespace(
        scene_dir=scene_dir,
        deterministic_gate_dir=deterministic_dir,
        review_dir=review_dir,
        reference_image=reference,
        output_dir=output_dir,
        rooms=["classroom_01"],
        threshold=7.0,
        minimum_review_images=3,
        maximum_review_images=6,
        model="gpt-5.2",
        vlm_backend="openai",
        contract_profile=PROFILE,
        effective_prompt=effective_prompt_path,
        input_manifest=manifest_path,
        prompt_binding=binding_path,
    )

    summary = run_room_gate(args, vlm_service=service)
    result = json.loads((output_dir / "classroom_01.json").read_text())

    assert summary["status"] == "pass"
    assert result["contract_profile"] == PROFILE
    assert result["evidence"]["effective_prompt"]["sha256"] == _sha256(
        effective_prompt_path
    )
    assert result["evidence"]["deterministic_gate"]["sha256"] == _sha256(
        deterministic_dir / "classroom_01.json"
    )
    assert result["evidence"]["cutaway_evidence"]["sha256"] == _sha256(
        cutaway_path
    )
    assert result["evidence"]["source_blend"]["sha256"] == _sha256(
        source_blend
    )
    assert "Complete immutable user prompt" in str(
        service.calls[0]["messages"][1]["content"]
    )
    assert (
        _verify_room_gates(
            output_dir,
            ["classroom_01"],
            scene_dir,
            contract_profile=PROFILE,
            input_dir=input_dir,
        )
        == []
    )

    gate_path = output_dir / "classroom_01.json"
    original_gate = gate_path.read_bytes()
    summary_service = FakeVLMService(RuntimeError("VLM must not be called"))
    summary_args = argparse.Namespace(**vars(args), summarize_existing=True)
    existing_summary = run_room_gate(summary_args, vlm_service=summary_service)
    assert existing_summary["status"] == "pass"
    assert summary_service.calls == []
    assert gate_path.read_bytes() == original_gate

    stale_gate = json.loads(original_gate)
    request = stale_gate["evidence"]["vlm_request"]
    request["messages_sha256"] = "0" * 64
    request_payload = {
        key: value for key, value in request.items() if key != "request_sha256"
    }
    request["request_sha256"] = hashlib.sha256(
        json.dumps(
            request_payload, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    gate_path.write_text(json.dumps(stale_gate), encoding="utf-8")
    digest_rejection = run_room_gate(
        summary_args,
        vlm_service=FakeVLMService(RuntimeError("VLM must not be called")),
    )
    assert digest_rejection["status"] == "fail"
    gate_path.write_bytes(original_gate)

    stale_gate = json.loads(original_gate)
    requirement_key = next(
        iter(stale_gate["visual_assessment"]["requirement_evidence"])
    )
    stale_gate["visual_assessment"]["requirement_evidence"][requirement_key][
        "status"
    ] = "fail"
    gate_path.write_text(json.dumps(stale_gate), encoding="utf-8")
    rejected_summary = run_room_gate(
        summary_args,
        vlm_service=FakeVLMService(RuntimeError("VLM must not be called")),
    )
    assert rejected_summary["status"] == "fail"
    assert rejected_summary["failed_rooms"] == ["classroom_01"]
    gate_path.write_bytes(original_gate)

    cutaway = json.loads(cutaway_path.read_text(encoding="utf-8"))
    cutaway["status"] = "fail"
    cutaway_path.write_text(json.dumps(cutaway), encoding="utf-8")
    assembly_failures = _verify_room_gates(
        output_dir,
        ["classroom_01"],
        scene_dir,
        contract_profile=PROFILE,
        input_dir=input_dir,
    )
    assert any("cutaway_evidence" in failure for failure in assembly_failures)
    assert any("status is 'fail'" in failure for failure in assembly_failures)
    blocked_service = FakeVLMService(_assessment(ROOM_SCORE_KEYS, score=10))
    blocked = run_room_gate(args, vlm_service=blocked_service)
    blocked_result = json.loads(
        (output_dir / "classroom_01.json").read_text(encoding="utf-8")
    )

    assert blocked["status"] == "fail"
    assert blocked_service.calls == []
    assert any(
        "Cutaway review evidence failed" in issue
        for issue in blocked_result["critical_issues"]
    )
