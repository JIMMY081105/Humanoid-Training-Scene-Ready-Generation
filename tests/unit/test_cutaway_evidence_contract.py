from __future__ import annotations

import hashlib
import json

from pathlib import Path

from scripts.cutaway_evidence_contract import VIEW_NAMES, validate_cutaway_evidence


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, room_id: str = "classroom_01") -> dict[str, object]:
    source_blend = tmp_path / "scene.blend"
    source_blend.write_bytes(b"blend")
    images: list[Path] = []
    views = []
    for view_name in VIEW_NAMES:
        image = tmp_path / f"{room_id}_{view_name}.png"
        image.write_bytes(f"image-{view_name}".encode())
        images.append(image)
        views.append(
            {
                "view_name": view_name,
                "image": str(image.resolve()),
                "image_size_bytes": image.stat().st_size,
                "image_sha256": _sha256(image),
                "cutaway": {
                    "established": True,
                    "view_name": view_name,
                    "overhead_state": "hidden",
                    "hidden_overhead": [{"name": "ceiling"}],
                    "hidden_camera_side_walls": (
                        [] if view_name == "top" else [{"name": f"near_{view_name}"}]
                    ),
                    "visible_far_wall_object_names": ["far_wall"],
                    "visible_floor_object_names": ["floor"],
                    "visible_content_object_names": ["student_desk"],
                    "classified_content_count": 1,
                },
            }
        )
    evidence = {
        "schema_id": "scenesmith_room_cutaway_review_v1",
        "schema_version": 1,
        "status": "pass",
        "room_id": room_id,
        "expected_views": list(VIEW_NAMES),
        "rendered_view_count": 3,
        "source_blend": {
            "path": str(source_blend.resolve()),
            "size_bytes": source_blend.stat().st_size,
            "sha256": _sha256(source_blend),
        },
        "classification": {
            "overhead": [{"name": "ceiling"}],
            "wall": [{"name": "near_wall"}, {"name": "far_wall"}],
            "floor": [{"name": "floor"}],
            "combined_envelope": [],
            "content": [{"name": "student_desk"}],
        },
        "views": views,
    }
    evidence_path = tmp_path / f"{room_id}_cutaway_evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    return {
        "room_id": room_id,
        "source_blend": source_blend,
        "images": images,
        "evidence": evidence,
        "evidence_path": evidence_path,
    }


def _validate(fixture: dict[str, object]) -> list[str]:
    return validate_cutaway_evidence(
        fixture["evidence_path"],
        room_id=fixture["room_id"],
        review_images=fixture["images"],
        source_blend=fixture["source_blend"],
    )


def test_valid_cutaway_evidence_passes(tmp_path: Path) -> None:
    assert _validate(_fixture(tmp_path)) == []


def test_changed_review_image_invalidates_cutaway(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["images"][1].write_bytes(b"changed")

    assert any("SHA-256" in failure for failure in _validate(fixture))


def test_oblique_without_hidden_camera_side_wall_fails(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["evidence"]["views"][1]["cutaway"]["hidden_camera_side_walls"] = []
    fixture["evidence_path"].write_text(
        json.dumps(fixture["evidence"]), encoding="utf-8"
    )

    assert any("no camera-side wall" in failure for failure in _validate(fixture))


def test_substituted_source_blend_fails(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    substitute = tmp_path / "substitute.blend"
    substitute.write_bytes(b"blend")
    fixture["evidence"]["source_blend"]["path"] = str(substitute.resolve())
    fixture["evidence_path"].write_text(
        json.dumps(fixture["evidence"]), encoding="utf-8"
    )

    assert any("canonical file" in failure for failure in _validate(fixture))


def test_indivisible_combined_envelope_fails(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["evidence"]["classification"]["combined_envelope"] = [
        {"name": "room_shell"}
    ]
    fixture["evidence_path"].write_text(
        json.dumps(fixture["evidence"]), encoding="utf-8"
    )

    assert any("indivisible combined shell" in failure for failure in _validate(fixture))
