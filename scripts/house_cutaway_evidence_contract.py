#!/usr/bin/env python3
"""Validate hash-bound cutaway evidence for final whole-school overview renders."""

from __future__ import annotations

import hashlib
import json

from pathlib import Path
from typing import Any, Mapping


SCHEMA_ID = "scenesmith_house_cutaway_review_v1"
SCHEMA_VERSION = 1
VIEW_NAMES = ("overview_top", "overview_isometric", "overview_front")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file_record(
    record: Any,
    *,
    expected_path: Path,
    label: str,
    path_key: str = "path",
    size_key: str = "size_bytes",
    hash_key: str = "sha256",
) -> list[str]:
    if not isinstance(record, dict):
        return [f"{label}: record is missing or malformed"]
    path_value = record.get(path_key)
    if not isinstance(path_value, str) or not path_value:
        return [f"{label}: path is missing"]
    path = Path(path_value)
    if not path.is_absolute():
        return [f"{label}: path is not absolute: {path}"]
    resolved = path.resolve()
    expected = expected_path.resolve()
    if resolved != expected:
        return [
            f"{label}: path differs from canonical file; "
            f"recorded={resolved}, expected={expected}"
        ]
    if not resolved.is_file() or resolved.stat().st_size < 1:
        return [f"{label}: file is missing or empty: {resolved}"]
    failures: list[str] = []
    if record.get(size_key) != resolved.stat().st_size:
        failures.append(f"{label}: recorded size differs from current file")
    if record.get(hash_key) != _sha256_file(resolved):
        failures.append(f"{label}: SHA-256 differs from current file")
    return failures


def validate_house_cutaway_evidence(
    evidence_path: Path,
    *,
    source_blend: Path,
    overview_images: Mapping[str, Path],
) -> list[str]:
    evidence_path = evidence_path.resolve()
    failures: list[str] = []
    if set(overview_images) != set(VIEW_NAMES):
        failures.append("house cutaway validator requires the exact three overview names")
    if not evidence_path.is_file():
        return [*failures, f"house cutaway evidence is missing: {evidence_path}"]
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [*failures, f"cannot read house cutaway evidence: {exc}"]
    if not isinstance(evidence, dict):
        return [*failures, "house cutaway evidence is not a JSON object"]
    if evidence.get("schema_id") != SCHEMA_ID:
        failures.append("house cutaway evidence schema_id is unsupported")
    if evidence.get("schema_version") != SCHEMA_VERSION:
        failures.append("house cutaway evidence schema_version is unsupported")
    if evidence.get("status") != "pass":
        failures.append(
            f"house cutaway evidence status is {evidence.get('status')!r}, not pass"
        )
    if evidence.get("expected_views") != list(VIEW_NAMES):
        failures.append("house cutaway expected_views are malformed")
    if evidence.get("rendered_view_count") != len(VIEW_NAMES):
        failures.append("house cutaway rendered_view_count is not exactly three")
    failures.extend(
        _verify_file_record(
            evidence.get("source_blend"),
            expected_path=source_blend,
            label="house_cutaway/source_blend",
        )
    )

    classification = evidence.get("classification")
    if not isinstance(classification, dict):
        failures.append("house cutaway classification is missing or malformed")
    else:
        for role in ("wall", "floor", "content"):
            records = classification.get(role)
            if not isinstance(records, list) or not records:
                failures.append(f"house cutaway classification has no {role} records")
        if classification.get("combined_envelope") not in ([], None):
            failures.append("house cutaway contains an indivisible combined envelope")

    views = evidence.get("views")
    if not isinstance(views, list) or len(views) != len(VIEW_NAMES):
        failures.append("house cutaway evidence does not contain exactly three views")
        views = []
    if [view.get("view_name") for view in views if isinstance(view, dict)] != list(
        VIEW_NAMES
    ):
        failures.append("house cutaway view names/order are malformed")

    for index, view_name in enumerate(VIEW_NAMES):
        if index >= len(views) or not isinstance(views[index], dict):
            continue
        view = views[index]
        expected_image = overview_images.get(view_name)
        if expected_image is not None:
            failures.extend(
                _verify_file_record(
                    view,
                    expected_path=expected_image,
                    label=f"house_cutaway/views/{view_name}/image",
                    path_key="image",
                    size_key="image_size_bytes",
                    hash_key="image_sha256",
                )
            )
        cutaway = view.get("cutaway")
        if not isinstance(cutaway, dict):
            failures.append(f"house_cutaway/views/{view_name}: proof is missing")
            continue
        if cutaway.get("established") is not True:
            failures.append(f"house_cutaway/views/{view_name}: established is not true")
        if cutaway.get("view_name") != view_name:
            failures.append(
                f"house_cutaway/views/{view_name}: view identity is inconsistent"
            )
        overhead_state = cutaway.get("overhead_state")
        hidden_overhead = cutaway.get("hidden_overhead")
        if overhead_state not in {"hidden", "verified_absent"}:
            failures.append(
                f"house_cutaway/views/{view_name}: overhead state is unproven"
            )
        if overhead_state == "hidden" and not hidden_overhead:
            failures.append(
                f"house_cutaway/views/{view_name}: hidden overhead has no records"
            )
        if overhead_state == "verified_absent" and hidden_overhead not in ([], None):
            failures.append(
                f"house_cutaway/views/{view_name}: absent overhead has hidden records"
            )
        camera_side = cutaway.get("hidden_camera_side_walls")
        if view_name == "overview_top":
            if camera_side not in ([], None):
                failures.append(
                    "house_cutaway/views/overview_top: top view must preserve walls"
                )
        elif not isinstance(camera_side, list) or not camera_side:
            failures.append(
                f"house_cutaway/views/{view_name}: no camera-side wall was hidden"
            )
        for label in (
            "visible_far_wall_object_names",
            "visible_floor_object_names",
            "visible_content_object_names",
        ):
            values = cutaway.get(label)
            if not isinstance(values, list) or not values:
                failures.append(f"house_cutaway/views/{view_name}: {label} is empty")
    return list(dict.fromkeys(failures))
