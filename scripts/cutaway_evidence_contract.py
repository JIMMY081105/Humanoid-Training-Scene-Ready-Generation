#!/usr/bin/env python3
"""Validate hash-bound, three-view cutaway room-render evidence without Blender."""

from __future__ import annotations

import hashlib
import json

from pathlib import Path
from typing import Any, Sequence


SCHEMA_ID = "scenesmith_room_cutaway_review_v1"
SCHEMA_VERSION = 1
VIEW_NAMES = ("top", "oblique_a", "oblique_b")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_bound_file(
    record: Any,
    *,
    expected_path: Path,
    label: str,
) -> list[str]:
    if not isinstance(record, dict):
        return [f"{label}: file record is missing or malformed"]
    path_value = record.get("path") if "path" in record else record.get("image")
    if not isinstance(path_value, str) or not path_value:
        return [f"{label}: path is missing"]
    path = Path(path_value)
    if not path.is_absolute():
        return [f"{label}: path is not absolute: {path}"]
    resolved = path.resolve()
    expected = expected_path.resolve()
    failures: list[str] = []
    if resolved != expected:
        failures.append(
            f"{label}: path differs from canonical file; "
            f"recorded={resolved}, expected={expected}"
        )
        return failures
    if not resolved.is_file() or resolved.stat().st_size < 1:
        failures.append(f"{label}: file is missing or empty: {resolved}")
        return failures
    size_value = (
        record.get("size_bytes")
        if "size_bytes" in record
        else record.get("image_size_bytes")
    )
    if size_value != resolved.stat().st_size:
        failures.append(f"{label}: recorded size differs from current file")
    digest_value = (
        record.get("sha256")
        if "sha256" in record
        else record.get("image_sha256")
    )
    if digest_value != _sha256_file(resolved):
        failures.append(f"{label}: SHA-256 differs from current file")
    return failures


def validate_cutaway_evidence(
    evidence_path: Path,
    *,
    room_id: str,
    review_images: Sequence[Path],
    source_blend: Path,
) -> list[str]:
    """Return every reason an alleged passing cutaway proof is invalid."""

    evidence_path = evidence_path.resolve()
    canonical_images = [path.resolve() for path in review_images]
    expected_names = [f"{room_id}_{name}.png" for name in VIEW_NAMES]
    failures: list[str] = []
    if len(canonical_images) != len(VIEW_NAMES):
        failures.append("cutaway evidence requires exactly three canonical images")
    elif [path.name for path in canonical_images] != expected_names:
        failures.append(
            "cutaway image names/order differ from the canonical top and oblique views"
        )
    if not evidence_path.is_file():
        return [*failures, f"cutaway evidence file is missing: {evidence_path}"]
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [*failures, f"cannot read cutaway evidence: {exc}"]
    if not isinstance(evidence, dict):
        return [*failures, "cutaway evidence is not a JSON object"]
    if evidence.get("schema_id") != SCHEMA_ID:
        failures.append("cutaway evidence schema_id is unsupported")
    if evidence.get("schema_version") != SCHEMA_VERSION:
        failures.append("cutaway evidence schema_version is unsupported")
    if evidence.get("status") != "pass":
        failures.append(f"cutaway evidence status is {evidence.get('status')!r}, not pass")
    if evidence.get("room_id") != room_id:
        failures.append("cutaway evidence room_id differs from the reviewed room")
    if evidence.get("expected_views") != list(VIEW_NAMES):
        failures.append("cutaway expected_views are not the exact three-view contract")
    if evidence.get("rendered_view_count") != len(VIEW_NAMES):
        failures.append("cutaway rendered_view_count is not exactly three")

    failures.extend(
        _verify_bound_file(
            evidence.get("source_blend"),
            expected_path=source_blend,
            label="cutaway/source_blend",
        )
    )
    classification = evidence.get("classification")
    if not isinstance(classification, dict):
        failures.append("cutaway envelope classification is missing or malformed")
    else:
        for role in ("wall", "floor", "content"):
            values = classification.get(role)
            if not isinstance(values, list) or not values:
                failures.append(f"cutaway classification has no {role} records")
        if classification.get("combined_envelope") not in ([], None):
            failures.append("cutaway evidence contains an indivisible combined shell")

    views = evidence.get("views")
    if not isinstance(views, list) or len(views) != len(VIEW_NAMES):
        failures.append("cutaway evidence does not contain exactly three views")
        views = []
    if [view.get("view_name") for view in views if isinstance(view, dict)] != list(
        VIEW_NAMES
    ):
        failures.append("cutaway view names/order are malformed")

    for index, view_name in enumerate(VIEW_NAMES):
        if index >= len(views) or not isinstance(views[index], dict):
            continue
        view = views[index]
        if index < len(canonical_images):
            failures.extend(
                _verify_bound_file(
                    view,
                    expected_path=canonical_images[index],
                    label=f"cutaway/views/{view_name}/image",
                )
            )
        cutaway = view.get("cutaway")
        if not isinstance(cutaway, dict):
            failures.append(f"cutaway/views/{view_name}: proof is missing")
            continue
        if cutaway.get("established") is not True:
            failures.append(f"cutaway/views/{view_name}: established is not true")
        if cutaway.get("view_name") != view_name:
            failures.append(f"cutaway/views/{view_name}: view identity is inconsistent")
        overhead_state = cutaway.get("overhead_state")
        hidden_overhead = cutaway.get("hidden_overhead")
        if overhead_state not in {"hidden", "verified_absent"}:
            failures.append(f"cutaway/views/{view_name}: overhead state is unproven")
        if overhead_state == "hidden" and not hidden_overhead:
            failures.append(f"cutaway/views/{view_name}: hidden overhead has no records")
        if overhead_state == "verified_absent" and hidden_overhead not in ([], None):
            failures.append(f"cutaway/views/{view_name}: absent overhead has hidden records")
        camera_side = cutaway.get("hidden_camera_side_walls")
        if view_name == "top":
            if camera_side not in ([], None):
                failures.append("cutaway/views/top: top view must preserve boundary walls")
        elif not isinstance(camera_side, list) or not camera_side:
            failures.append(
                f"cutaway/views/{view_name}: no camera-side wall was hidden"
            )
        for label in (
            "visible_far_wall_object_names",
            "visible_floor_object_names",
            "visible_content_object_names",
        ):
            values = cutaway.get(label)
            if not isinstance(values, list) or not values:
                failures.append(f"cutaway/views/{view_name}: {label} is empty")
        if not isinstance(cutaway.get("classified_content_count"), int) or cutaway.get(
            "classified_content_count", 0
        ) < 1:
            failures.append(f"cutaway/views/{view_name}: no classified content")
    return list(dict.fromkeys(failures))
