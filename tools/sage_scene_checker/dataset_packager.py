"""Prepare a local Hugging Face-ready dataset folder for checked scenes."""

from __future__ import annotations

import json
import shutil

from pathlib import Path
from typing import Any


def package_scene(
    *,
    scene_dir: str | Path,
    validation_report: str | Path,
    output_dir: str | Path,
    prompt_text: str | None = None,
) -> Path:
    """Copy key scene artifacts and metadata into a dataset-style folder."""

    scene_path = Path(scene_dir)
    report_path = Path(validation_report)
    output_path = Path(output_dir)
    target = output_path / scene_path.name
    target.mkdir(parents=True, exist_ok=True)

    combined = scene_path / "combined_house"
    files = [
        combined / "house_state.json",
        combined / "sceneeval_state.json",
        combined / "house.dmd.yaml",
        combined / "house.blend",
        scene_path / "asset_registry.json",
    ]
    copied: list[str] = []
    for source in files:
        if source.exists():
            destination = target / source.name
            shutil.copy2(source, destination)
            copied.append(destination.name)

    if report_path.exists():
        shutil.copy2(report_path, target / "validation_report.json")
        copied.append("validation_report.json")

    prompt = prompt_text or _prompt_from_house_state(combined / "house_state.json")
    if prompt:
        (target / "prompt.txt").write_text(prompt, encoding="utf-8")
        copied.append("prompt.txt")

    metadata: dict[str, Any] = {
        "scene_id": scene_path.name,
        "source_scene_dir": str(scene_path),
        "files": copied,
        "uploaded": False,
    }
    (target / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return target


def _prompt_from_house_state(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    prompt = (data.get("layout") or {}).get("house_prompt")
    return str(prompt) if prompt else None

