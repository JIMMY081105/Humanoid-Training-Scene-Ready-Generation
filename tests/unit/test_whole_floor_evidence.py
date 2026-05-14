from __future__ import annotations

import argparse
import hashlib
import json

from pathlib import Path

import pytest

from scripts.whole_floor_reference_gate import (
    FLOOR_SCORE_KEYS,
    REQUIRED_OVERVIEW_NAMES,
    canonical_evidence_paths,
    run_gate,
    verify_gate_evidence,
)


class FakeVLMService:
    def __init__(self, *, mutate: Path | None = None, fail_if_called: bool = False):
        self.mutate = mutate
        self.fail_if_called = fail_if_called
        self.calls = []

    def create_completion(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_if_called:
            raise AssertionError("VLM must not be called in this test")
        if self.mutate is not None:
            self.mutate.write_bytes(self.mutate.read_bytes() + b"-changed-during-vlm")
        return json.dumps(
            {
                "scores": {key: 9 for key in FLOOR_SCORE_KEYS},
                "critical_issues": [],
                "repair_instructions": [],
                "observations": ["All supplied views meet the contract."],
            }
        )


def _room(room_id: str, x: float, y: float, width: float, depth: float) -> dict:
    return {
        "room_id": room_id,
        "position": [x, y],
        "width": width,
        "depth": depth,
    }


def _valid_layout() -> dict:
    return {
        "placed_rooms": [
            _room("classroom_01", 0, 0, 9, 7.5),
            _room("classroom_03", 0, 10, 9, 7.5),
            _room("classroom_04", 0, 20, 9, 7.5),
            _room("classroom_02", 21, 0, 9, 7.5),
            _room("classroom_06", 21, 12, 9, 7.5),
            _room("classroom_05", 21, 20, 9, 7.5),
            _room("library", 10, -9, 10, 9),
            _room("boys_toilet", 1, -5, 4, 4),
            _room("girls_toilet", 5, -5, 4, 4),
            _room("storage_room", 21, 8, 5, 3.7),
            _room("main_corridor", 9, 0, 12, 22.5),
        ]
    }


def _image(path: Path, marker: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + marker)
    return path


def _write_house_cutaway(
    combined_dir: Path, overviews: dict[str, Path]
) -> tuple[Path, Path]:
    blend = combined_dir / "house.blend"
    blend.write_bytes(b"house-blend")
    state = combined_dir / "house_state.json"
    if not state.is_file():
        raise AssertionError("house_state.json must exist before cutaway evidence")
    views = []
    for name in REQUIRED_OVERVIEW_NAMES:
        image = overviews[name]
        views.append(
            {
                "view_name": name,
                "image": str(image.resolve()),
                "image_size_bytes": image.stat().st_size,
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
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
        "sha256": hashlib.sha256(state.read_bytes()).hexdigest(),
    }
    blend_record = {
        "path": str(blend.resolve()),
        "size_bytes": blend.stat().st_size,
        "sha256": hashlib.sha256(blend.read_bytes()).hexdigest(),
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
        "expected_views": list(REQUIRED_OVERVIEW_NAMES),
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
    return blend, path


def _fixture(tmp_path: Path) -> dict[str, object]:
    scene_dir = tmp_path / "scene_000"
    combined_dir = scene_dir / "combined_house"
    overview_dir = combined_dir / "outlook_renders"
    overview_dir.mkdir(parents=True)

    layout_path = scene_dir / "house_layout.json"
    layout_path.write_text(json.dumps(_valid_layout()), encoding="utf-8")
    reference = _image(tmp_path / "reference.png", b"reference")
    overviews = {
        name: _image(overview_dir / f"{name}.png", name.encode("utf-8"))
        for name in REQUIRED_OVERVIEW_NAMES
    }
    house_state = combined_dir / "house_state.json"
    house_state.write_text('{"rooms": {}}', encoding="utf-8")
    house_blend, house_cutaway = _write_house_cutaway(combined_dir, overviews)
    artiverse_usage = combined_dir / "artiverse_usage.json"
    artiverse_usage.write_text(
        '{"status": "pass", "final_surviving_asset_count": 1}',
        encoding="utf-8",
    )
    input_manifest = tmp_path / "input_manifest.json"
    input_manifest.write_text(
        json.dumps(
            {"reference_image_sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}
        ),
        encoding="utf-8",
    )
    output = scene_dir / "quality_gates" / "whole_floor_reference.json"
    args = argparse.Namespace(
        scene_dir=scene_dir,
        layout=None,
        overview_dir=None,
        house_state=None,
        artiverse_usage=None,
        reference_image=reference,
        input_manifest=input_manifest,
        output=None,
        verify_only=False,
        threshold=7.0,
        model="gpt-5.2",
        vlm_backend="openai",
    )
    expected_paths = canonical_evidence_paths(
        scene_dir=scene_dir,
        layout_path=layout_path,
        reference_image=reference,
        overview_images=overviews,
        house_state_path=house_state,
        artiverse_usage_path=artiverse_usage,
        house_cutaway_path=house_cutaway,
        input_manifest_path=input_manifest,
    )
    return {
        "scene_dir": scene_dir,
        "layout": layout_path,
        "reference": reference,
        "overviews": overviews,
        "house_state": house_state,
        "artiverse_usage": artiverse_usage,
        "house_blend": house_blend,
        "house_cutaway": house_cutaway,
        "input_manifest": input_manifest,
        "output": output,
        "args": args,
        "expected_paths": expected_paths,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_passing_gate_binds_every_canonical_artifact(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = run_gate(fixture["args"], vlm_service=FakeVLMService())

    assert result["status"] == "pass", result["critical_issues"]
    assert result["evidence_verification"] == {
        "status": "pass",
        "critical_issues": [],
    }
    assert fixture["output"].is_file()
    evidence = result["evidence"]
    assert evidence["schema_version"] == 1
    assert evidence["algorithm"] == "sha256"
    assert set(evidence["overview_images"]) == set(REQUIRED_OVERVIEW_NAMES)
    for label in (
        "house_layout",
        "reference_image",
        "house_state",
        "artiverse_usage",
        "house_cutaway",
    ):
        path = Path(evidence[label]["path"])
        assert evidence[label]["sha256"] == _sha256(path)
    for entry in evidence["overview_images"].values():
        path = Path(entry["path"])
        assert entry["sha256"] == _sha256(path)
    assert verify_gate_evidence(
        result, expected_paths=fixture["expected_paths"]
    ) == []


def test_mutation_during_vlm_review_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    target = fixture["overviews"]["overview_isometric"]
    service = FakeVLMService(mutate=target)

    result = run_gate(fixture["args"], vlm_service=service)

    assert result["status"] == "fail"
    assert len(service.calls) == 1
    assert any(
        "changed during visual review" in issue
        and "overview_isometric" in issue
        for issue in result["critical_issues"]
    )
    assert result["evidence_verification"]["status"] == "fail"


@pytest.mark.parametrize(
    ("artifact", "expected_label"),
    [
        ("layout", "house_layout"),
        ("reference", "reference_image"),
        ("overview_top", "overview_top"),
        ("overview_isometric", "overview_isometric"),
        ("overview_front", "overview_front"),
        ("house_state", "house_state"),
        ("artiverse_usage", "artiverse_usage"),
        ("house_cutaway", "house_cutaway"),
    ],
)
def test_mutation_after_pass_is_rejected(
    tmp_path: Path, artifact: str, expected_label: str
) -> None:
    fixture = _fixture(tmp_path)
    result = run_gate(fixture["args"], vlm_service=FakeVLMService())
    assert result["status"] == "pass"

    if artifact.startswith("overview_"):
        target = fixture["overviews"][artifact]
    else:
        target = fixture[artifact]
    target.write_bytes(target.read_bytes() + b"-changed-after-pass")

    failures = verify_gate_evidence(
        result, expected_paths=fixture["expected_paths"]
    )
    assert any(expected_label in failure and "SHA-256 mismatch" in failure for failure in failures)


def test_verify_only_rejects_stale_saved_gate_without_vlm(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = run_gate(fixture["args"], vlm_service=FakeVLMService())
    assert result["status"] == "pass"
    fixture["artiverse_usage"].write_text('{"status": "mutated"}', encoding="utf-8")

    fixture["args"].verify_only = True
    service = FakeVLMService(fail_if_called=True)
    verified = run_gate(fixture["args"], vlm_service=service)

    assert verified["status"] == "fail"
    assert service.calls == []
    assert any(
        "artiverse_usage" in issue and "SHA-256 mismatch" in issue
        for issue in verified["critical_issues"]
    )
    saved = json.loads(fixture["output"].read_text(encoding="utf-8"))
    assert saved["status"] == "fail"


def test_self_consistent_vlm_digest_substitution_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = run_gate(fixture["args"], vlm_service=FakeVLMService())
    assert result["status"] == "pass"
    request = result["evidence"]["vlm_request"]
    request["messages_sha256"] = "0" * 64
    payload = {key: value for key, value in request.items() if key != "request_sha256"}
    request["request_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    failures = verify_gate_evidence(
        result, expected_paths=fixture["expected_paths"]
    )

    assert any("reconstructed messages" in failure for failure in failures)


def test_missing_combined_artifact_blocks_vlm_and_gate(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["house_state"].unlink()
    service = FakeVLMService(fail_if_called=True)

    result = run_gate(fixture["args"], vlm_service=service)

    assert result["status"] == "fail"
    assert service.calls == []
    assert any(
        "Whole-floor evidence preflight failed closed" in issue
        and "house_state.json" in issue
        for issue in result["critical_issues"]
    )
