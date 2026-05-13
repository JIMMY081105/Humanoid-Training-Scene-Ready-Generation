from __future__ import annotations

import hashlib
import json
import os

from pathlib import Path

import pytest

from scripts.school_room_contract import PROFILE, ROOM_IDS
from scripts.select_room_resume_stage import (
    CHECKPOINTS,
    ResumeContractError,
    select_resume_stage,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _setup_scene(tmp_path: Path, room_id: str = "classroom_01") -> tuple[Path, Path, str]:
    scene_dir = tmp_path / "scene_000"
    prompts = {candidate: f"immutable prompt for {candidate}" for candidate in ROOM_IDS}
    rooms = [{"id": candidate, "prompt": prompts[candidate]} for candidate in ROOM_IDS]
    placed = [
        {"room_id": candidate, "prompt": prompts[candidate]} for candidate in ROOM_IDS
    ]
    layout_path = scene_dir / "house_layout.json"
    _write_json(layout_path, {"rooms": rooms, "placed_rooms": placed})
    layout_hash = hashlib.sha256(layout_path.read_bytes()).hexdigest()
    binding_path = scene_dir / "quality_gates" / "room_prompt_binding.json"
    _write_json(
        binding_path,
        {
            "status": "pass",
            "profile": PROFILE,
            "layout": {"path": str(layout_path.resolve()), "sha256": layout_hash},
            "room_prompt_sha256": {
                candidate: hashlib.sha256(prompts[candidate].encode()).hexdigest()
                for candidate in ROOM_IDS
            },
            "occurrence_counts": {candidate: 2 for candidate in ROOM_IDS},
        },
    )
    return scene_dir, binding_path, prompts[room_id]


def _checkpoint(scene_dir: Path, room_id: str, name: str, prompt: str) -> Path:
    path = scene_dir / f"room_{room_id}" / "scene_states" / name / "scene_state.json"
    _write_json(path, {"text_description": prompt, "objects": {}})
    return path


@pytest.mark.parametrize(
    ("checkpoint_count", "expected"),
    [
        (0, "furniture"),
        (1, "wall_mounted"),
        (2, "ceiling_mounted"),
        (3, "manipuland"),
        (4, "complete"),
    ],
)
def test_selects_highest_contiguous_checkpoint(
    tmp_path: Path, checkpoint_count: int, expected: str
) -> None:
    room_id = "classroom_01"
    scene_dir, binding_path, prompt = _setup_scene(tmp_path, room_id)
    for name, _stage in CHECKPOINTS[:checkpoint_count]:
        _checkpoint(scene_dir, room_id, name, prompt)

    assert select_resume_stage(
        scene_dir=scene_dir,
        room_id=room_id,
        prompt_binding_path=binding_path,
    ) == expected


def test_rejects_gapped_checkpoint_chain(tmp_path: Path) -> None:
    room_id = "library"
    scene_dir, binding_path, prompt = _setup_scene(tmp_path, room_id)
    _checkpoint(scene_dir, room_id, "scene_after_wall_objects", prompt)

    with pytest.raises(ResumeContractError, match="gapped"):
        select_resume_stage(
            scene_dir=scene_dir,
            room_id=room_id,
            prompt_binding_path=binding_path,
        )


def test_rejects_checkpoint_prompt_mismatch(tmp_path: Path) -> None:
    room_id = "storage_room"
    scene_dir, binding_path, _prompt = _setup_scene(tmp_path, room_id)
    _checkpoint(scene_dir, room_id, "scene_after_furniture", "stale prompt")

    with pytest.raises(ResumeContractError, match="immutable room prompt"):
        select_resume_stage(
            scene_dir=scene_dir,
            room_id=room_id,
            prompt_binding_path=binding_path,
        )


def test_rejects_changed_layout_after_binding(tmp_path: Path) -> None:
    scene_dir, binding_path, _prompt = _setup_scene(tmp_path)
    layout_path = scene_dir / "house_layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    layout["extra"] = "mutation"
    _write_json(layout_path, layout)

    with pytest.raises(ResumeContractError, match="changed after prompt binding"):
        select_resume_stage(
            scene_dir=scene_dir,
            room_id="classroom_01",
            prompt_binding_path=binding_path,
        )


def test_rejects_malformed_checkpoint(tmp_path: Path) -> None:
    room_id = "classroom_02"
    scene_dir, binding_path, _prompt = _setup_scene(tmp_path, room_id)
    path = (
        scene_dir
        / f"room_{room_id}"
        / "scene_states"
        / "scene_after_furniture"
        / "scene_state.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ResumeContractError, match="Cannot parse"):
        select_resume_stage(
            scene_dir=scene_dir,
            room_id=room_id,
            prompt_binding_path=binding_path,
        )


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink creation needs privileges")
def test_rejects_symlink_checkpoint(tmp_path: Path) -> None:
    room_id = "classroom_03"
    scene_dir, binding_path, prompt = _setup_scene(tmp_path, room_id)
    real = tmp_path / "outside.json"
    _write_json(real, {"text_description": prompt, "objects": {}})
    link = (
        scene_dir
        / f"room_{room_id}"
        / "scene_states"
        / "scene_after_furniture"
        / "scene_state.json"
    )
    link.parent.mkdir(parents=True)
    link.symlink_to(real)

    with pytest.raises(ResumeContractError, match="link/junction"):
        select_resume_stage(
            scene_dir=scene_dir,
            room_id=room_id,
            prompt_binding_path=binding_path,
        )
