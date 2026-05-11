from __future__ import annotations

import hashlib
import json

from pathlib import Path

from scripts.house_cutaway_evidence_contract import (
    VIEW_NAMES,
    validate_house_cutaway_evidence,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, object]:
    combined = tmp_path / "combined_house"
    outlook = combined / "outlook_renders"
    outlook.mkdir(parents=True)
    blend = combined / "house.blend"
    blend.write_bytes(b"house-blend")
    images: dict[str, Path] = {}
    views = []
    for view_name in VIEW_NAMES:
        image = outlook / f"{view_name}.png"
        image.write_bytes(f"image-{view_name}".encode())
        images[view_name] = image
        views.append(
            {
                "view_name": view_name,
                "image": str(image.resolve()),
                "image_size_bytes": image.stat().st_size,
                "image_sha256": _sha256(image),
                "cutaway": {
                    "established": True,
                    "view_name": view_name,
                    "overhead_state": "verified_absent",
                    "hidden_overhead": [],
                    "hidden_camera_side_walls": (
                        [] if view_name == "overview_top" else ["near_wall"]
                    ),
                    "visible_far_wall_object_names": ["far_wall"],
                    "visible_floor_object_names": ["floor"],
                    "visible_content_object_names": ["desk"],
                },
            }
        )
    evidence = {
        "schema_id": "scenesmith_house_cutaway_review_v1",
        "schema_version": 1,
        "status": "pass",
        "expected_views": list(VIEW_NAMES),
        "rendered_view_count": 3,
        "source_blend": {
            "path": str(blend.resolve()),
            "size_bytes": blend.stat().st_size,
            "sha256": _sha256(blend),
        },
        "classification": {
            "overhead": [],
            "wall": [{"name": "near_wall"}, {"name": "far_wall"}],
            "floor": [{"name": "floor"}],
            "combined_envelope": [],
            "content": [{"name": "desk"}],
        },
        "views": views,
    }
    evidence_path = outlook / "overview_cutaway_evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    return {
        "blend": blend,
        "images": images,
        "evidence": evidence,
        "evidence_path": evidence_path,
    }


def _validate(fixture: dict[str, object]) -> list[str]:
    return validate_house_cutaway_evidence(
        fixture["evidence_path"],
        source_blend=fixture["blend"],
        overview_images=fixture["images"],
    )


def test_valid_house_cutaway_evidence_passes(tmp_path: Path) -> None:
    assert _validate(_fixture(tmp_path)) == []


def test_house_cutaway_rejects_mutated_overview(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["images"]["overview_front"].write_bytes(b"changed")

    assert any("SHA-256" in failure for failure in _validate(fixture))


def test_house_cutaway_rejects_opaque_oblique(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["evidence"]["views"][1]["cutaway"][
        "hidden_camera_side_walls"
    ] = []
    fixture["evidence_path"].write_text(
        json.dumps(fixture["evidence"]), encoding="utf-8"
    )

    assert any("no camera-side wall" in failure for failure in _validate(fixture))


def test_house_cutaway_rejects_substituted_blend(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    substitute = tmp_path / "substitute.blend"
    substitute.write_bytes(b"house-blend")
    fixture["evidence"]["source_blend"]["path"] = str(substitute.resolve())
    fixture["evidence_path"].write_text(
        json.dumps(fixture["evidence"]), encoding="utf-8"
    )

    assert any("canonical file" in failure for failure in _validate(fixture))
