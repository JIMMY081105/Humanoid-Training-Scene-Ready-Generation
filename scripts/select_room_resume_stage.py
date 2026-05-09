#!/usr/bin/env python3
"""Select the highest trustworthy SceneSmith checkpoint for one school room."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat

from pathlib import Path
from typing import Any, Sequence

try:
    from .school_room_contract import (
        PROFILE,
        ROOM_IDS,
        _layout_collection,
    )
except ImportError:  # Direct execution: python scripts/select_room_resume_stage.py
    from school_room_contract import (  # type: ignore[no-redef]
        PROFILE,
        ROOM_IDS,
        _layout_collection,
    )


CHECKPOINTS: tuple[tuple[str, str], ...] = (
    ("scene_after_furniture", "wall_mounted"),
    ("scene_after_wall_objects", "ceiling_mounted"),
    ("scene_after_ceiling_objects", "manipuland"),
    ("final_scene", "complete"),
)


class ResumeContractError(RuntimeError):
    """Raised when an existing checkpoint cannot be trusted for resume."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _require_regular_unlinked_file(path: Path, *, stop_at: Path) -> None:
    """Reject links/special files in the checkpoint path without following them."""

    current = path
    while True:
        if _is_link_or_junction(current):
            raise ResumeContractError(f"Resume path is a link/junction: {current}")
        try:
            mode = current.stat(follow_symlinks=False).st_mode
        except OSError as exc:
            raise ResumeContractError(f"Cannot stat resume path {current}: {exc}") from exc
        if current == path:
            if not stat.S_ISREG(mode):
                raise ResumeContractError(f"Resume evidence is not a regular file: {path}")
        elif not stat.S_ISDIR(mode):
            raise ResumeContractError(f"Resume parent is not a directory: {current}")
        if current == stop_at:
            break
        if current.parent == current or stop_at not in current.parents:
            raise ResumeContractError(
                f"Resume evidence escapes the expected room directory: {path}"
            )
        current = current.parent


def _read_object(path: Path, *, stop_at: Path) -> dict[str, Any]:
    _require_regular_unlinked_file(path, stop_at=stop_at)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResumeContractError(f"Cannot parse JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResumeContractError(f"Expected a JSON object in {path}")
    return value


def _load_bound_prompt(
    *, scene_dir: Path, room_id: str, prompt_binding_path: Path
) -> str:
    layout_path = scene_dir / "house_layout.json"
    layout = _read_object(layout_path, stop_at=scene_dir)
    room_specs = _layout_collection(layout, "rooms")
    placed_rooms = _layout_collection(layout, "placed_rooms")
    room_prompt = room_specs[room_id].get("prompt")
    placed_prompt = placed_rooms[room_id].get("prompt")
    if not isinstance(room_prompt, str) or not room_prompt:
        raise ResumeContractError(f"Layout has no prompt for {room_id}")
    if placed_prompt != room_prompt:
        raise ResumeContractError(
            f"RoomSpec/PlacedRoom prompt mismatch for {room_id}"
        )

    binding = _read_object(prompt_binding_path, stop_at=scene_dir)
    if binding.get("status") != "pass" or binding.get("profile") != PROFILE:
        raise ResumeContractError("Prompt binding is not a passing school contract")
    binding_layout = binding.get("layout")
    if not isinstance(binding_layout, dict):
        raise ResumeContractError("Prompt binding has no layout evidence")
    if binding_layout.get("path") != str(layout_path.resolve()):
        raise ResumeContractError("Prompt binding names a different house layout")
    if binding_layout.get("sha256") != _sha256_file(layout_path):
        raise ResumeContractError("House layout changed after prompt binding")

    prompt_hashes = binding.get("room_prompt_sha256")
    if not isinstance(prompt_hashes, dict) or set(prompt_hashes) != set(ROOM_IDS):
        raise ResumeContractError("Prompt binding room hash inventory is incomplete")
    expected_hash = hashlib.sha256(room_prompt.encode("utf-8")).hexdigest()
    if prompt_hashes.get(room_id) != expected_hash:
        raise ResumeContractError(f"Bound prompt hash mismatch for {room_id}")
    occurrence_counts = binding.get("occurrence_counts")
    if occurrence_counts != {candidate: 2 for candidate in ROOM_IDS}:
        raise ResumeContractError("Prompt binding occurrence inventory is invalid")
    return room_prompt


def _checkpoint_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _validate_checkpoint(path: Path, *, room_dir: Path, prompt: str) -> None:
    state = _read_object(path, stop_at=room_dir)
    if state.get("text_description") != prompt:
        raise ResumeContractError(
            f"Checkpoint prompt does not match immutable room prompt: {path}"
        )
    objects = state.get("objects")
    if not isinstance(objects, (dict, list)):
        raise ResumeContractError(f"Checkpoint has no valid objects collection: {path}")


def select_resume_stage(
    *, scene_dir: Path, room_id: str, prompt_binding_path: Path
) -> str:
    if room_id not in ROOM_IDS:
        raise ResumeContractError(f"Unsupported school room ID: {room_id}")
    scene_dir = scene_dir.absolute()
    if _is_link_or_junction(scene_dir) or not scene_dir.is_dir():
        raise ResumeContractError(f"Scene directory is missing or linked: {scene_dir}")
    prompt_binding_path = prompt_binding_path.absolute()
    room_dir = scene_dir / f"room_{room_id}"
    if room_dir.exists() and (_is_link_or_junction(room_dir) or not room_dir.is_dir()):
        raise ResumeContractError(f"Room directory is not a real directory: {room_dir}")

    prompt = _load_bound_prompt(
        scene_dir=scene_dir,
        room_id=room_id,
        prompt_binding_path=prompt_binding_path,
    )
    state_paths = [
        room_dir / "scene_states" / checkpoint / "scene_state.json"
        for checkpoint, _next_stage in CHECKPOINTS
    ]
    present = [_checkpoint_exists(path) for path in state_paths]
    for index, exists in enumerate(present):
        if exists and not all(present[:index]):
            raise ResumeContractError(
                f"Checkpoint chain is gapped before {CHECKPOINTS[index][0]}"
            )
        if exists:
            _validate_checkpoint(state_paths[index], room_dir=room_dir, prompt=prompt)

    highest_index = max((index for index, exists in enumerate(present) if exists), default=-1)
    if highest_index < 0:
        return "furniture"
    return CHECKPOINTS[highest_index][1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--room-id", required=True)
    parser.add_argument("--prompt-binding", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stage = select_resume_stage(
        scene_dir=args.scene_dir,
        room_id=args.room_id,
        prompt_binding_path=args.prompt_binding,
    )
    print(stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
