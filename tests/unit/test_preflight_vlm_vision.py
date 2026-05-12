from __future__ import annotations

import json

from scripts.preflight_vlm_vision import run


class _FakeVisionService:
    def create_completion(self, **_kwargs):
        return {
            "scores": {"image_readability": 9},
            "critical_issues": [],
            "repair_instructions": [],
        }


def test_vision_smoke_binds_image_and_passes_schema(tmp_path) -> None:
    image = tmp_path / "reference.png"
    image.write_bytes(b"fake-image-for-hash")
    output = tmp_path / "vision.json"

    result = run(
        image=image,
        output=output,
        model="fake-model",
        backend="openai",
        vlm_service=_FakeVisionService(),
    )

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "pass"
    assert persisted["status"] == "pass"
    assert len(persisted["image_sha256"]) == 64


def test_vision_smoke_fails_closed_for_missing_image(tmp_path) -> None:
    output = tmp_path / "vision.json"
    result = run(
        image=tmp_path / "missing.png",
        output=output,
        model="fake-model",
        backend="openai",
        vlm_service=_FakeVisionService(),
    )
    assert result["status"] == "fail"
    assert result["error_type"] == "FileNotFoundError"
