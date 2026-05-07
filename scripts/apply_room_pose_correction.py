#!/usr/bin/env python3
"""Apply a hash-bound, manipuland-preserving correction to a final room state.

The default mode is a read-only dry run.  ``--apply`` stages a corrected
``final_scene`` without carrying forward the stale blend, swaps it into place,
and quarantines stale review/gate evidence.  ``--verify-only`` rehashes the
published artifacts and independently repeats the real collision check.

Corrections are intentionally limited to world translations with unchanged
rotation and wall-local placements.  Moving furniture after manipuland
generation also moves every support surface, supported descendant, and embedded
composite-member transform by the identical world-space translation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import stat
import sys
import time
import uuid

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA_ID = "scenesmith_room_pose_correction_spec_v1"
SCHEMA_VERSION = 1
RECEIPT_SCHEMA_ID = "scenesmith_room_pose_correction_receipt_v1"
RECEIPT_SCHEMA_VERSION = 1
RECEIPT_NAME = "pose_correction_receipt.json"
HASH_CHUNK_SIZE = 1024 * 1024
DEFAULT_COLLISION_POLICY = {
    "penetration_threshold": 0.001,
    "floor_penetration_tolerance": 0.05,
    "manipuland_furniture_tolerance_m": 0.02,
}


class PoseCorrectionError(RuntimeError):
    """Raised when a correction cannot be proven or published safely."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if is_junction is not None and is_junction():
        return True
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, FileNotFoundError, OSError):
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_flag and attributes & reparse_flag)


def _reject_link_components(path: Path, *, label: str) -> None:
    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if (current.exists() or current.is_symlink()) and _is_link_like(current):
            raise PoseCorrectionError(
                f"{label} path contains a symlink/junction: {current}"
            )


def _inside(root: Path, candidate: Path) -> bool:
    try:
        return os.path.commonpath((os.fspath(root), os.fspath(candidate))) == os.fspath(root)
    except ValueError:
        return False


def _require_real_directory(path: Path, *, label: str) -> Path:
    absolute = path.expanduser().absolute()
    _reject_link_components(absolute, label=label)
    if not absolute.exists() or not absolute.is_dir() or _is_link_like(absolute):
        raise PoseCorrectionError(f"{label} is not a real directory: {absolute}")
    return absolute.resolve(strict=True)


def _require_regular_file(path: Path, *, label: str, nonempty: bool = True) -> Path:
    absolute = path.expanduser().absolute()
    _reject_link_components(absolute, label=label)
    if not absolute.exists() or _is_link_like(absolute):
        raise PoseCorrectionError(f"{label} is missing or link-like: {absolute}")
    try:
        metadata = absolute.lstat()
    except OSError as exc:
        raise PoseCorrectionError(f"Cannot inspect {label}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise PoseCorrectionError(f"{label} is not a regular file: {absolute}")
    if nonempty and metadata.st_size <= 0:
        raise PoseCorrectionError(f"{label} is empty: {absolute}")
    return absolute


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_replace(source: Path, destination: Path) -> None:
    source_parent = source.parent
    destination_parent = destination.parent
    os.replace(source, destination)
    _fsync_directory(destination_parent)
    if source_parent != destination_parent:
        _fsync_directory(source_parent)


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    safe = _require_regular_file(path, label=label)
    before = safe.stat()
    raw = safe.read_bytes()
    after = safe.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise PoseCorrectionError(f"{label} changed while it was being read")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PoseCorrectionError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise PoseCorrectionError(f"{label} root must be a JSON object")
    return value, raw


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PoseCorrectionError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PoseCorrectionError(f"{label} must be finite")
    return result


def _vector(value: Any, *, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise PoseCorrectionError(f"{label} must contain exactly {length} numbers")
    return [_finite_number(item, label=f"{label}[{index}]") for index, item in enumerate(value)]


def _close_vector(first: Sequence[float], second: Sequence[float], *, atol: float = 1e-9) -> bool:
    return len(first) == len(second) and all(
        math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)
        for left, right in zip(first, second)
    )


def _validate_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise PoseCorrectionError(f"{label} must be a lowercase SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise PoseCorrectionError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    spec = copy.deepcopy(dict(value))
    if spec.get("schema_id") != SCHEMA_ID or spec.get("schema_version") != SCHEMA_VERSION:
        raise PoseCorrectionError("Unsupported pose-correction specification schema")
    room_id = spec.get("room_id")
    if not isinstance(room_id, str) or not room_id.strip():
        raise PoseCorrectionError("Correction specification has no room_id")
    _validate_sha256(spec.get("source_state_sha256"), label="source_state_sha256")
    _validate_sha256(spec.get("house_layout_sha256"), label="house_layout_sha256")
    corrections = spec.get("corrections")
    if not isinstance(corrections, list) or not corrections:
        raise PoseCorrectionError("Correction specification has no corrections")
    seen: set[str] = set()
    for index, correction in enumerate(corrections):
        label = f"corrections[{index}]"
        if not isinstance(correction, dict):
            raise PoseCorrectionError(f"{label} must be an object")
        object_id = correction.get("object_id")
        if not isinstance(object_id, str) or not object_id:
            raise PoseCorrectionError(f"{label}.object_id is missing")
        if object_id in seen:
            raise PoseCorrectionError(f"Correction object_id is duplicated: {object_id}")
        seen.add(object_id)
        object_type = correction.get("object_type")
        if not isinstance(object_type, str) or not object_type:
            raise PoseCorrectionError(f"{label}.object_type is missing")
        mode = correction.get("mode")
        if mode == "world_translation":
            _vector(correction.get("expected_translation"), length=3, label=f"{label}.expected_translation")
            _vector(correction.get("target_translation"), length=3, label=f"{label}.target_translation")
        elif mode == "wall_local":
            parent = correction.get("expected_parent_surface_id")
            if not isinstance(parent, str) or not parent:
                raise PoseCorrectionError(f"{label}.expected_parent_surface_id is missing")
            _vector(correction.get("expected_position_2d"), length=2, label=f"{label}.expected_position_2d")
            _vector(correction.get("target_position_2d"), length=2, label=f"{label}.target_position_2d")
            _finite_number(correction.get("expected_rotation_2d"), label=f"{label}.expected_rotation_2d")
        else:
            raise PoseCorrectionError(f"{label}.mode is unsupported: {mode!r}")
    policy = spec.get("collision_policy", DEFAULT_COLLISION_POLICY)
    if not isinstance(policy, dict):
        raise PoseCorrectionError("collision_policy must be an object")
    normalized_policy = {}
    for key, default in DEFAULT_COLLISION_POLICY.items():
        normalized_policy[key] = _finite_number(policy.get(key, default), label=f"collision_policy.{key}")
        if not math.isclose(normalized_policy[key], default, rel_tol=0.0, abs_tol=0.0):
            raise PoseCorrectionError(
                f"collision_policy.{key} must equal the production value {default}"
            )
    spec["collision_policy"] = normalized_policy
    prefix = spec.get("sceneeval_asset_id_prefix", "scenesmith")
    if not isinstance(prefix, str) or not prefix:
        raise PoseCorrectionError("sceneeval_asset_id_prefix must be a nonempty string")
    spec["sceneeval_asset_id_prefix"] = prefix
    return spec


def _objects(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    objects = state.get("objects")
    if not isinstance(objects, dict):
        raise PoseCorrectionError("Room state has no objects dictionary")
    for object_id, record in objects.items():
        if not isinstance(object_id, str) or not isinstance(record, dict):
            raise PoseCorrectionError("Room state contains a malformed object record")
        if record.get("object_id") != object_id:
            raise PoseCorrectionError(f"Object dictionary key/id mismatch: {object_id}")
    return objects


def _transform(record: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    transform = record.get("transform")
    if not isinstance(transform, dict):
        raise PoseCorrectionError(f"{label} has no transform")
    translation = _vector(transform.get("translation"), length=3, label=f"{label}.transform.translation")
    rotation = _vector(transform.get("rotation_wxyz"), length=4, label=f"{label}.transform.rotation_wxyz")
    transform["translation"] = translation
    transform["rotation_wxyz"] = rotation
    return transform


def _translate_serialized_transform(value: Any, delta: Sequence[float], *, label: str) -> None:
    if not isinstance(value, dict):
        raise PoseCorrectionError(f"{label} is not a serialized transform")
    transform = _transform(value, label=label) if "transform" in value else value
    if "translation" not in transform or "rotation_wxyz" not in transform:
        raise PoseCorrectionError(f"{label} is not a serialized transform")
    translation = _vector(transform["translation"], length=3, label=f"{label}.translation")
    transform["translation"] = [translation[index] + float(delta[index]) for index in range(3)]


def _translate_composite_metadata(record: dict[str, Any], delta: Sequence[float], *, object_id: str) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise PoseCorrectionError(f"Object {object_id} metadata is malformed")
    composite_type = metadata.get("composite_type")
    if composite_type is None:
        return
    if composite_type in ("stack", "pile"):
        members = metadata.get("member_assets")
        if not isinstance(members, list):
            raise PoseCorrectionError(f"Composite {object_id} has no member_assets list")
        for index, member in enumerate(members):
            if not isinstance(member, dict) or "transform" not in member:
                raise PoseCorrectionError(f"Composite {object_id} member {index} has no transform")
            _translate_serialized_transform(member["transform"], delta, label=f"{object_id}.member_assets[{index}]")
        return
    if composite_type == "filled_container":
        container = metadata.get("container_asset")
        fills = metadata.get("fill_assets")
        if not isinstance(container, dict) or "transform" not in container:
            raise PoseCorrectionError(f"Composite {object_id} has no container transform")
        if not isinstance(fills, list):
            raise PoseCorrectionError(f"Composite {object_id} has no fill_assets list")
        _translate_serialized_transform(container["transform"], delta, label=f"{object_id}.container_asset")
        for index, fill in enumerate(fills):
            if not isinstance(fill, dict) or "transform" not in fill:
                raise PoseCorrectionError(f"Composite {object_id} fill {index} has no transform")
            _translate_serialized_transform(fill["transform"], delta, label=f"{object_id}.fill_assets[{index}]")
        return
    raise PoseCorrectionError(f"Object {object_id} has unsupported composite_type {composite_type!r}")


def _surface_ids(record: Mapping[str, Any], *, object_id: str) -> list[str]:
    surfaces = record.get("support_surfaces", [])
    if not isinstance(surfaces, list):
        raise PoseCorrectionError(f"Object {object_id} support_surfaces is malformed")
    result: list[str] = []
    for index, surface in enumerate(surfaces):
        if not isinstance(surface, dict):
            raise PoseCorrectionError(f"Object {object_id} support surface {index} is malformed")
        surface_id = surface.get("surface_id")
        if not isinstance(surface_id, str) or not surface_id:
            raise PoseCorrectionError(f"Object {object_id} support surface {index} has no id")
        if surface_id in result:
            raise PoseCorrectionError(f"Object {object_id} repeats support surface {surface_id}")
        result.append(surface_id)
    return result


def _validate_support_graph(
    objects: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    surface_owner: dict[str, str] = {}
    for object_id, record in objects.items():
        for surface_id in _surface_ids(record, object_id=object_id):
            previous = surface_owner.get(surface_id)
            if previous is not None:
                raise PoseCorrectionError(
                    f"Support surface {surface_id} is owned by both {previous} and {object_id}"
                )
            surface_owner[surface_id] = object_id

    parent_owner: dict[str, str] = {}
    for object_id, record in objects.items():
        parent_surface = _placement_parent(record)
        if parent_surface in surface_owner:
            parent_owner[object_id] = surface_owner[parent_surface]

    for start in objects:
        path: list[str] = []
        seen: set[str] = set()
        current = start
        while current in parent_owner:
            if current in seen:
                cycle_start = path.index(current)
                cycle = path[cycle_start:] + [current]
                raise PoseCorrectionError(
                    "Support graph contains a cycle: " + " -> ".join(cycle)
                )
            seen.add(current)
            path.append(current)
            current = parent_owner[current]
    return surface_owner


def _translate_object(record: dict[str, Any], delta: Sequence[float], *, object_id: str) -> list[str]:
    transform = _transform(record, label=f"object {object_id}")
    translation = transform["translation"]
    transform["translation"] = [translation[index] + float(delta[index]) for index in range(3)]
    surfaces = record.get("support_surfaces", [])
    surface_ids = _surface_ids(record, object_id=object_id)
    for index, surface in enumerate(surfaces):
        if "transform" not in surface:
            raise PoseCorrectionError(
                f"Object {object_id} support surface {index} has no transform"
            )
        _translate_serialized_transform(surface["transform"], delta, label=f"{object_id}.support_surfaces[{index}]")
    _translate_composite_metadata(record, delta, object_id=object_id)
    return surface_ids


def _placement_parent(record: Mapping[str, Any]) -> str | None:
    placement = record.get("placement_info")
    if placement is None:
        return None
    if not isinstance(placement, dict):
        raise PoseCorrectionError("Object placement_info is malformed")
    parent = placement.get("parent_surface_id")
    if not isinstance(parent, str) or not parent:
        raise PoseCorrectionError("Object placement_info has no parent_surface_id")
    return parent


def _translate_tree(
    objects: dict[str, dict[str, Any]],
    *,
    root_id: str,
    delta: Sequence[float],
    claimed_by: dict[str, str],
) -> list[str]:
    pending = [root_id]
    affected: list[str] = []
    while pending:
        object_id = pending.pop(0)
        previous = claimed_by.get(object_id)
        if previous is not None and previous != root_id:
            raise PoseCorrectionError(
                f"Object {object_id} is a descendant of both {previous} and {root_id}"
            )
        if previous is not None:
            continue
        claimed_by[object_id] = root_id
        record = objects[object_id]
        surface_ids = set(_translate_object(record, delta, object_id=object_id))
        affected.append(object_id)
        if not surface_ids:
            continue
        for child_id, child in objects.items():
            if _placement_parent(child) in surface_ids:
                previous = claimed_by.get(child_id)
                if previous is not None and previous != root_id:
                    raise PoseCorrectionError(
                        f"Object {child_id} is a descendant of both {previous} and {root_id}"
                    )
                if previous is None:
                    pending.append(child_id)
    return affected


def _mask_composite_transforms(record: dict[str, Any]) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return
    composite_type = metadata.get("composite_type")
    if composite_type in ("stack", "pile"):
        for member in metadata.get("member_assets", []):
            if isinstance(member, dict) and "transform" in member:
                member["transform"] = "<pose-correction>"
    elif composite_type == "filled_container":
        container = metadata.get("container_asset")
        if isinstance(container, dict) and "transform" in container:
            container["transform"] = "<pose-correction>"
        for fill in metadata.get("fill_assets", []):
            if isinstance(fill, dict) and "transform" in fill:
                fill["transform"] = "<pose-correction>"


def _masked_state(
    state: Mapping[str, Any], *, translated_ids: set[str], wall_local_ids: set[str]
) -> dict[str, Any]:
    masked = copy.deepcopy(dict(state))
    objects = _objects(masked)
    for object_id in translated_ids:
        record = objects[object_id]
        record["transform"] = "<pose-correction>"
        for surface in record.get("support_surfaces", []):
            if isinstance(surface, dict):
                surface["transform"] = "<pose-correction>"
        _mask_composite_transforms(record)
    for object_id in wall_local_ids:
        record = objects[object_id]
        record["transform"] = "<pose-correction>"
        placement = record.get("placement_info")
        if isinstance(placement, dict):
            position = placement.get("position_2d")
            if isinstance(position, list) and len(position) == 2:
                position[1] = "<pose-correction>"
    return masked


@dataclass
class CorrectionResult:
    corrected_state: dict[str, Any]
    records: list[dict[str, Any]]
    translated_ids: set[str]
    wall_local_ids: set[str]


WallPoseResolver = Callable[[dict[str, Any], Mapping[str, Any]], dict[str, Any]]


def apply_corrections_to_state(
    source_state: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    wall_pose_resolver: WallPoseResolver,
) -> CorrectionResult:
    """Return a corrected deep copy while enforcing an exact structural diff."""
    baseline = copy.deepcopy(dict(source_state))
    corrected = copy.deepcopy(dict(source_state))
    objects = _objects(corrected)
    _validate_support_graph(objects)
    claimed_by: dict[str, str] = {}
    translated: set[str] = set()
    wall_local: set[str] = set()
    records: list[dict[str, Any]] = []

    for correction in spec["corrections"]:
        object_id = correction["object_id"]
        record = objects.get(object_id)
        if record is None:
            raise PoseCorrectionError(f"Correction target is missing: {object_id}")
        if record.get("object_type") != correction["object_type"]:
            raise PoseCorrectionError(
                f"Correction target {object_id} has type {record.get('object_type')!r}, "
                f"expected {correction['object_type']!r}"
            )
        before_transform = copy.deepcopy(_transform(record, label=f"object {object_id}"))
        before_rotation = copy.deepcopy(before_transform["rotation_wxyz"])

        if correction["mode"] == "world_translation":
            if record.get("object_type") != "furniture":
                raise PoseCorrectionError(
                    f"World-translation target {object_id} must be furniture"
                )
            if record.get("placement_info") is not None:
                raise PoseCorrectionError(
                    f"World-translation target {object_id} must be an unparented root"
                )
            expected = _vector(correction["expected_translation"], length=3, label=f"{object_id}.expected_translation")
            target = _vector(correction["target_translation"], length=3, label=f"{object_id}.target_translation")
            if not _close_vector(before_transform["translation"], expected):
                raise PoseCorrectionError(
                    f"Correction target {object_id} source translation changed: "
                    f"{before_transform['translation']} != {expected}"
                )
            delta = [target[index] - expected[index] for index in range(3)]
            affected = _translate_tree(
                objects, root_id=object_id, delta=delta, claimed_by=claimed_by
            )
            translated.update(affected)
            after_transform = _transform(objects[object_id], label=f"object {object_id}")
            if not _close_vector(after_transform["translation"], target):
                raise PoseCorrectionError(f"Correction target {object_id} did not reach target")
            if after_transform["rotation_wxyz"] != before_rotation:
                raise PoseCorrectionError(f"Correction target {object_id} rotation changed")
            records.append(
                {
                    "object_id": object_id,
                    "mode": "world_translation",
                    "source_transform": before_transform,
                    "target_transform": copy.deepcopy(after_transform),
                    "world_delta": delta,
                    "affected_object_ids": affected,
                }
            )
            continue

        if record.get("object_type") != "wall_mounted":
            raise PoseCorrectionError(
                f"Wall-local target {object_id} must be wall_mounted"
            )
        if _surface_ids(record, object_id=object_id):
            raise PoseCorrectionError(
                f"Wall-local target {object_id} must not own support surfaces"
            )
        metadata = record.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("composite_type") is not None:
            raise PoseCorrectionError(
                f"Wall-local target {object_id} must be a non-composite leaf"
            )
        placement = record.get("placement_info")
        if not isinstance(placement, dict):
            raise PoseCorrectionError(f"Wall-local target {object_id} has no placement_info")
        expected_parent = correction["expected_parent_surface_id"]
        if placement.get("parent_surface_id") != expected_parent:
            raise PoseCorrectionError(
                f"Wall-local target {object_id} parent changed: "
                f"{placement.get('parent_surface_id')!r} != {expected_parent!r}"
            )
        expected_position = _vector(correction["expected_position_2d"], length=2, label=f"{object_id}.expected_position_2d")
        target_position = _vector(correction["target_position_2d"], length=2, label=f"{object_id}.target_position_2d")
        current_position = _vector(placement.get("position_2d"), length=2, label=f"{object_id}.placement_info.position_2d")
        current_rotation = _finite_number(placement.get("rotation_2d"), label=f"{object_id}.placement_info.rotation_2d")
        expected_rotation = _finite_number(correction["expected_rotation_2d"], label=f"{object_id}.expected_rotation_2d")
        if not _close_vector(current_position, expected_position):
            raise PoseCorrectionError(f"Wall-local target {object_id} source position changed")
        if not math.isclose(current_rotation, expected_rotation, rel_tol=0.0, abs_tol=1e-12):
            raise PoseCorrectionError(f"Wall-local target {object_id} source rotation changed")
        if not math.isclose(target_position[0], expected_position[0], rel_tol=0.0, abs_tol=1e-12):
            raise PoseCorrectionError("Wall-local correction may change height only")
        placement["position_2d"] = target_position
        resolved_transform = wall_pose_resolver(corrected, correction)
        if not isinstance(resolved_transform, dict) or set(resolved_transform) != set(before_transform):
            raise PoseCorrectionError(
                f"Wall-local target {object_id} resolver changed the transform schema"
            )
        resolved_translation = _vector(
            resolved_transform.get("translation") if isinstance(resolved_transform, dict) else None,
            length=3,
            label=f"{object_id}.resolved_transform.translation",
        )
        resolved_rotation = _vector(
            resolved_transform.get("rotation_wxyz") if isinstance(resolved_transform, dict) else None,
            length=4,
            label=f"{object_id}.resolved_transform.rotation_wxyz",
        )
        if not _close_vector(resolved_translation[:2], before_transform["translation"][:2], atol=1e-9):
            raise PoseCorrectionError(
                f"Wall-local target {object_id} changed its world horizontal position"
            )
        if not _close_vector(resolved_rotation, before_rotation, atol=1e-9):
            raise PoseCorrectionError(f"Wall-local target {object_id} rotation changed")
        # Wall-surface reconstruction can round an otherwise identical pose.
        # Keep exact source horizontal/rotation bytes: this correction is height-only.
        resolved_transform["translation"][:2] = before_transform["translation"][:2]
        resolved_transform["rotation_wxyz"] = before_rotation
        record["transform"] = resolved_transform
        after_transform = copy.deepcopy(_transform(record, label=f"object {object_id}"))
        wall_local.add(object_id)
        records.append(
            {
                "object_id": object_id,
                "mode": "wall_local",
                "parent_surface_id": expected_parent,
                "source_position_2d": expected_position,
                "target_position_2d": target_position,
                "rotation_2d": expected_rotation,
                "source_transform": before_transform,
                "target_transform": after_transform,
            }
        )

    if _canonical_json(
        _masked_state(baseline, translated_ids=translated, wall_local_ids=wall_local)
    ) != _canonical_json(
        _masked_state(corrected, translated_ids=translated, wall_local_ids=wall_local)
    ):
        raise PoseCorrectionError("Correction changed state outside its exact pose allowance")

    return CorrectionResult(
        corrected_state=corrected,
        records=records,
        translated_ids=translated,
        wall_local_ids=wall_local,
    )


def _collision_record(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "object_a_id",
        "object_b_id",
        "object_a_name",
        "object_b_name",
        "penetration_depth",
        "penetration_depth_m",
    ):
        if hasattr(value, key):
            field = getattr(value, key)
            if isinstance(field, (str, int, float, bool)) or field is None:
                result[key] = field
            else:
                result[key] = str(field)
    description = getattr(value, "to_description", None)
    result["description"] = str(description() if callable(description) else value)
    return result


class ProductionRuntime:
    """Lazy SceneSmith/Drake adapter; importing this module itself stays lightweight."""

    def __init__(
        self,
        *,
        room_dir: Path,
        room_id: str,
        house_layout_path: Path,
        collision_policy: Mapping[str, float],
        sceneeval_asset_id_prefix: str,
    ) -> None:
        from scenesmith.agent_utils.house import HouseLayout
        from scenesmith.agent_utils.physics_validation import compute_scene_collisions
        from scenesmith.agent_utils.room import RoomScene
        from scenesmith.agent_utils.sceneeval_exporter import (
            SceneEvalExportConfig,
            SceneEvalExporter,
        )
        from scenesmith.utils.sdf_utils import serialize_rigid_transform
        from scenesmith.wall_agents.tools.wall_surface import (
            extract_wall_surfaces_from_room_geometry,
        )

        layout_value, _raw = _read_json(house_layout_path, label="house layout")
        self.room_dir = room_dir
        self.room_id = room_id
        self.collision_policy = dict(collision_policy)
        self.room_scene_cls = RoomScene
        self.compute_scene_collisions = compute_scene_collisions
        self.serialize_rigid_transform = serialize_rigid_transform
        self.extract_wall_surfaces = extract_wall_surfaces_from_room_geometry
        self.house_layout = HouseLayout.from_dict(layout_value, house_dir=room_dir.parent)
        self.sceneeval_config = SceneEvalExportConfig(
            asset_id_prefix=sceneeval_asset_id_prefix
        )
        self.sceneeval_exporter_cls = SceneEvalExporter

    def scene(self, state: Mapping[str, Any]) -> Any:
        scene = self.room_scene_cls(
            room_geometry=None, scene_dir=self.room_dir, room_id=self.room_id
        )
        scene.restore_from_state_dict(copy.deepcopy(dict(state)))
        return scene

    def resolve_wall_pose(
        self, state: dict[str, Any], correction: Mapping[str, Any]
    ) -> dict[str, Any]:
        scene = self.scene(state)
        surfaces = self.extract_wall_surfaces(
            scene.room_geometry, room_id=self.room_id
        )
        expected_id = correction["expected_parent_surface_id"]
        matches = [surface for surface in surfaces if str(surface.surface_id) == expected_id]
        if len(matches) != 1:
            raise PoseCorrectionError(
                f"Expected exactly one wall surface {expected_id}, found {len(matches)}"
            )
        surface = matches[0]
        target = correction["target_position_2d"]
        if not surface.contains_point_2d(float(target[0]), float(target[1])):
            raise PoseCorrectionError(
                f"Wall-local target is outside or inside an opening on {expected_id}"
            )
        record = _objects(state)[correction["object_id"]]
        bbox_min = _vector(record.get("bbox_min"), length=3, label="wall object bbox_min")
        bbox_max = _vector(record.get("bbox_max"), length=3, label="wall object bbox_max")
        valid, error = surface.check_object_bounds(
            position_x=float(target[0]),
            position_z=float(target[1]),
            object_width=float(bbox_max[0] - bbox_min[0]),
            object_height=float(bbox_max[2] - bbox_min[2]),
        )
        if not valid:
            raise PoseCorrectionError(f"Wall-local object bounds are invalid: {error}")
        pose = surface.to_world_pose(
            position_x=float(target[0]),
            position_z=float(target[1]),
            rotation_deg=math.degrees(float(correction["expected_rotation_2d"])),
        )
        return self.serialize_rigid_transform(pose)

    def collisions(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        scene = self.scene(state)
        collisions = self.compute_scene_collisions(
            scene=scene,
            penetration_threshold=self.collision_policy["penetration_threshold"],
            floor_penetration_tolerance=self.collision_policy[
                "floor_penetration_tolerance"
            ],
            current_furniture_id=None,
            manipuland_furniture_tolerance_m=self.collision_policy[
                "manipuland_furniture_tolerance_m"
            ],
        )
        return sorted(
            (_collision_record(collision) for collision in collisions),
            key=lambda item: _canonical_json(item),
        )

    def roundtrip(self, state: Mapping[str, Any]) -> dict[str, Any]:
        value = self.scene(state).to_state_dict()
        if "timestamp" in state:
            value["timestamp"] = state["timestamp"]
        return value

    def dmd(self, state: Mapping[str, Any]) -> str:
        return self.scene(state).to_drake_directive()

    def sceneeval(self, state: Mapping[str, Any]) -> dict[str, Any]:
        exporter = self.sceneeval_exporter_cls(
            scene=self.scene(state),
            scene_dir=self.room_dir,
            config=self.sceneeval_config,
            house_layout=self.house_layout,
        )
        # The public export method fixes its output path to live final_scene.  Build
        # the value in memory so publication can remain staged and fail-closed.
        value = exporter._build_scene_state()
        if not isinstance(value, dict):
            raise PoseCorrectionError("SceneEval exporter returned a malformed value")
        return value


@dataclass
class PreparedCorrection:
    corrected_state: dict[str, Any]
    correction_records: list[dict[str, Any]]
    source_collisions: list[dict[str, Any]]
    corrected_collisions: list[dict[str, Any]]
    dmd: str
    sceneeval: dict[str, Any]
    object_ids: list[str]
    manipuland_ids: list[str]


def _object_and_manipuland_ids(state: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    objects = _objects(state)
    object_ids = sorted(objects)
    manipulands = sorted(
        object_id
        for object_id, record in objects.items()
        if record.get("object_type") == "manipuland"
    )
    return object_ids, manipulands


def prepare_correction(
    source_state: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    runtime: Any,
) -> PreparedCorrection:
    source_ids, source_manipulands = _object_and_manipuland_ids(source_state)
    source_collisions = runtime.collisions(source_state)
    result = apply_corrections_to_state(
        source_state, spec, wall_pose_resolver=runtime.resolve_wall_pose
    )
    corrected_ids, corrected_manipulands = _object_and_manipuland_ids(
        result.corrected_state
    )
    if corrected_ids != source_ids:
        raise PoseCorrectionError("Correction changed the exact object-ID inventory")
    if corrected_manipulands != source_manipulands:
        raise PoseCorrectionError("Correction changed the exact manipuland inventory")

    roundtrip = runtime.roundtrip(result.corrected_state)
    if _canonical_json(roundtrip) != _canonical_json(result.corrected_state):
        raise PoseCorrectionError(
            "Corrected state is not byte-semantically stable after restore/serialize"
        )
    corrected_collisions = runtime.collisions(roundtrip)
    if corrected_collisions:
        descriptions = "; ".join(
            item.get("description", "unknown collision")
            for item in corrected_collisions
        )
        raise PoseCorrectionError(
            "Corrected state still has real collision violations: " + descriptions
        )
    dmd = runtime.dmd(roundtrip)
    if not isinstance(dmd, str) or not dmd.strip():
        raise PoseCorrectionError("Corrected Drake directive is empty")
    sceneeval = runtime.sceneeval(roundtrip)
    return PreparedCorrection(
        corrected_state=roundtrip,
        correction_records=result.records,
        source_collisions=source_collisions,
        corrected_collisions=corrected_collisions,
        dmd=dmd,
        sceneeval=sceneeval,
        object_ids=corrected_ids,
        manipuland_ids=corrected_manipulands,
    )


def _write_bytes_durable(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    _fsync_directory(path.parent)


def _write_json_durable(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes_durable(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )


def _file_record(path: Path, *, logical_path: str | None = None) -> dict[str, Any]:
    safe = _require_regular_file(path, label=logical_path or path.name)
    return {
        "path": logical_path or safe.name,
        "size_bytes": safe.stat().st_size,
        "sha256": _sha256_file(safe),
    }


def _receipt_with_attestation(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    result["attestation"] = {
        "algorithm": "sha256",
        "sha256": _sha256_bytes(_canonical_json(result)),
    }
    return result


def _verify_attestation(receipt: Mapping[str, Any]) -> None:
    attestation = receipt.get("attestation")
    if not isinstance(attestation, dict) or attestation.get("algorithm") != "sha256":
        raise PoseCorrectionError("Correction receipt has no SHA-256 attestation")
    payload = {key: value for key, value in receipt.items() if key != "attestation"}
    expected = _sha256_bytes(_canonical_json(payload))
    if attestation.get("sha256") != expected:
        raise PoseCorrectionError("Correction receipt attestation is invalid")


def _stale_evidence_paths(scene_dir: Path, room_id: str) -> list[Path]:
    scene_dir = _require_real_directory(scene_dir, label="scene directory")
    review = scene_dir / "review" / "room_review_renders"
    deterministic = scene_dir / "quality_gates" / "room_self_exam_deterministic"
    visual = scene_dir / "quality_gates" / "room_self_exam"
    candidates = [
        review / f"{room_id}_top.png",
        review / f"{room_id}_oblique_a.png",
        review / f"{room_id}_oblique_b.png",
        review / f"{room_id}_cutaway_evidence.json",
        deterministic / f"{room_id}.json",
        deterministic / "summary.json",
        visual / f"{room_id}.json",
        visual / "summary.json",
    ]
    result = []
    for path in candidates:
        if path.exists() or path.is_symlink():
            safe = _require_regular_file(path, label="stale evidence")
            if not _inside(scene_dir, safe):
                raise PoseCorrectionError(f"Stale evidence escapes scene directory: {safe}")
            result.append(safe)
    return result


def _stale_evidence_records(scene_dir: Path, paths: Sequence[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        relative = path.relative_to(scene_dir).as_posix()
        record = _file_record(path, logical_path=relative)
        record["backup_path"] = f"stale_evidence/{relative}"
        records.append(record)
    return records


def _safe_descendant(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PoseCorrectionError(f"{label} path is missing")
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise PoseCorrectionError(f"{label} path is unsafe: {relative!r}")
    candidate = (root / value).absolute()
    if not _inside(root, candidate):
        raise PoseCorrectionError(f"{label} path escapes its root: {relative!r}")
    _reject_link_components(candidate, label=label)
    return candidate


def _acquire_lock(path: Path) -> int:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise PoseCorrectionError(f"Pose-correction lock already exists: {path}") from exc
    os.write(descriptor, f"pid={os.getpid()} time={time.time()}\n".encode("ascii"))
    os.fsync(descriptor)
    return descriptor


def _release_lock(path: Path, descriptor: int) -> None:
    try:
        os.close(descriptor)
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _rollback_publication(
    *,
    final_dir: Path,
    backup_final: Path,
    failed_final: Path,
    moved_evidence: list[tuple[Path, Path]],
) -> None:
    errors: list[str] = []
    for original, backup in reversed(moved_evidence):
        try:
            if backup.exists() and not original.exists():
                original.parent.mkdir(parents=True, exist_ok=True)
                _durable_replace(backup, original)
        except Exception as exc:  # pragma: no cover - catastrophic filesystem failure.
            errors.append(f"restore {original}: {exc}")
    try:
        if final_dir.exists() and not failed_final.exists():
            _durable_replace(final_dir, failed_final)
    except Exception as exc:  # pragma: no cover - catastrophic filesystem failure.
        errors.append(f"quarantine failed final: {exc}")
    try:
        if backup_final.exists() and not final_dir.exists():
            _durable_replace(backup_final, final_dir)
    except Exception as exc:  # pragma: no cover - catastrophic filesystem failure.
        errors.append(f"restore original final: {exc}")
    if errors:
        raise PoseCorrectionError("Publication rollback was incomplete: " + "; ".join(errors))


def _publish(
    *,
    room_dir: Path,
    final_dir: Path,
    source_state_path: Path,
    source_state_sha256: str,
    spec_path: Path,
    spec_sha256: str,
    spec: Mapping[str, Any],
    house_layout_record: Mapping[str, Any],
    prepared: PreparedCorrection,
    failure_injector: Callable[[], None] | None = None,
) -> dict[str, Any]:
    scene_dir = room_dir.parent
    run_dir = scene_dir.parent
    transaction_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    )
    transactions_dir = run_dir / "pose_correction_transactions"
    if transactions_dir.exists() or transactions_dir.is_symlink():
        transactions_dir = _require_real_directory(
            transactions_dir, label="pose-correction transaction directory"
        )
    else:
        transactions_dir.mkdir(parents=False)
        _fsync_directory(run_dir)
        transactions_dir = _require_real_directory(
            transactions_dir, label="pose-correction transaction directory"
        )
    transaction_root = transactions_dir / transaction_id
    transaction_root.mkdir(parents=True, exist_ok=False)
    _fsync_directory(transactions_dir)
    staging = transaction_root / "staged_final_scene"
    staging.mkdir()
    backup_final = transaction_root / "original_final_scene"
    failed_final = transaction_root / "failed_corrected_final_scene"

    corrected_state_path = staging / "scene_state.json"
    dmd_path = staging / "scene.dmd.yaml"
    sceneeval_path = staging / "sceneeval_state.json"
    _write_json_durable(corrected_state_path, prepared.corrected_state)
    _write_bytes_durable(dmd_path, prepared.dmd.encode("utf-8"))
    _write_json_durable(sceneeval_path, prepared.sceneeval)

    stale_paths = _stale_evidence_paths(scene_dir, spec["room_id"])
    stale_records = _stale_evidence_records(scene_dir, stale_paths)
    transaction_relative = transaction_root.relative_to(run_dir).as_posix()
    backup_relative = backup_final.relative_to(run_dir).as_posix()
    source_record = _file_record(
        source_state_path,
        logical_path=f"{backup_relative}/scene_state.json",
    )
    receipt_payload = {
        "schema_id": RECEIPT_SCHEMA_ID,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "pass",
        "transaction_id": transaction_id,
        "room_id": spec["room_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "specification": {
            "path": str(spec_path),
            "sha256": spec_sha256,
            "source_state_sha256": spec["source_state_sha256"],
            "house_layout_sha256": spec["house_layout_sha256"],
        },
        "source_state": source_record,
        "house_layout": dict(house_layout_record),
        "published_artifacts": {
            "scene_state": _file_record(corrected_state_path, logical_path="scene_state.json"),
            "drake_directive": _file_record(dmd_path, logical_path="scene.dmd.yaml"),
            "sceneeval_state": _file_record(sceneeval_path, logical_path="sceneeval_state.json"),
        },
        "blend_policy": {
            "status": "intentionally_absent_pending_hash_bound_refresh",
            "required_command": "scripts/run_single_room_worker.py --refresh-final-blend",
        },
        "object_ids": prepared.object_ids,
        "manipuland_ids": prepared.manipuland_ids,
        "manipuland_count": len(prepared.manipuland_ids),
        "corrections": prepared.correction_records,
        "collision_proof": {
            "policy": dict(spec["collision_policy"]),
            "source_collision_count": len(prepared.source_collisions),
            "source_collisions": prepared.source_collisions,
            "corrected_collision_count": len(prepared.corrected_collisions),
            "corrected_collisions": prepared.corrected_collisions,
            "roundtrip_recomputed": True,
        },
        "quarantined_stale_evidence": stale_records,
        "backup": {
            "transaction_root_relative_to_run_dir": transaction_relative,
            "original_final_scene": backup_relative,
            "commit_marker": f"{transaction_relative}/COMMITTED",
        },
    }
    receipt = _receipt_with_attestation(receipt_payload)
    _write_bytes_durable(transaction_root / "PREPARED", b"prepared\n")

    # Re-read the source immediately before publication.  A worker that changed
    # final_scene while staging therefore cannot be overwritten.
    if _sha256_file(source_state_path) != source_state_sha256:
        raise PoseCorrectionError("Source final state changed before publication")

    moved_evidence: list[tuple[Path, Path]] = []
    published = False
    try:
        _durable_replace(final_dir, backup_final)
        backup_source_path = backup_final / "scene_state.json"
        if _file_record(backup_source_path, logical_path=source_record["path"]) != source_record:
            raise PoseCorrectionError("Backed-up source final state changed during publication")
        _write_bytes_durable(transaction_root / "FINAL_BACKED_UP", b"backed_up\n")
        _durable_replace(staging, final_dir)
        published = True
        _write_bytes_durable(transaction_root / "FINAL_PUBLISHED", b"published\n")
        if failure_injector is not None:
            failure_injector()
        for original in stale_paths:
            relative = original.relative_to(scene_dir)
            backup = transaction_root / "stale_evidence" / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            _durable_replace(original, backup)
            moved_evidence.append((original, backup))
        _write_bytes_durable(
            transaction_root / "EVIDENCE_QUARANTINED", b"quarantined\n"
        )
        if _file_record(backup_source_path, logical_path=source_record["path"]) != source_record:
            raise PoseCorrectionError("Backed-up source final state changed after publication")
        artifacts = receipt["published_artifacts"]
        for key, label in (
            ("scene_state", "published state"),
            ("drake_directive", "published Drake directive"),
            ("sceneeval_state", "published SceneEval state"),
        ):
            _verify_file_record(final_dir, artifacts[key], label=label)
        # The pass receipt is the final live write.  A crash at any earlier point
        # leaves no live pass receipt; a crash before COMMITTED is fail-closed.
        _write_json_durable(final_dir / RECEIPT_NAME, receipt)
        _write_bytes_durable(transaction_root / "COMMITTED", b"committed\n")
    except Exception:
        if published or backup_final.exists():
            _rollback_publication(
                final_dir=final_dir,
                backup_final=backup_final,
                failed_final=failed_final,
                moved_evidence=moved_evidence,
            )
        raise
    return receipt


def _verify_file_record(final_dir: Path, record: Mapping[str, Any], *, label: str) -> None:
    path_value = record.get("path")
    if not isinstance(path_value, str) or Path(path_value).name != path_value:
        raise PoseCorrectionError(f"{label} has an unsafe artifact path")
    actual = _file_record(final_dir / path_value, logical_path=path_value)
    if actual != dict(record):
        raise PoseCorrectionError(f"{label} no longer matches its receipt")


def verify_published(
    *,
    room_dir: Path,
    spec_path: Path,
    spec: Mapping[str, Any],
    spec_sha256: str,
    house_layout_path: Path,
    house_layout_record: Mapping[str, Any],
    runtime: Any,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    final_dir = _require_real_directory(
        room_dir / "scene_states" / "final_scene", label="published final_scene"
    )
    scene_dir = _require_real_directory(room_dir.parent, label="scene directory")
    run_dir = _require_real_directory(scene_dir.parent, label="run directory")
    receipt_path = receipt_path or final_dir / RECEIPT_NAME
    receipt, _raw = _read_json(receipt_path, label="pose-correction receipt")
    _verify_attestation(receipt)
    if receipt.get("schema_id") != RECEIPT_SCHEMA_ID or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise PoseCorrectionError("Unsupported pose-correction receipt schema")
    if receipt.get("status") != "pass":
        raise PoseCorrectionError("Pose-correction receipt does not have pass status")
    if receipt.get("room_id") != spec["room_id"]:
        raise PoseCorrectionError("Receipt room_id does not match the specification")
    specification = receipt.get("specification")
    if not isinstance(specification, dict) or specification.get("sha256") != spec_sha256:
        raise PoseCorrectionError("Receipt specification hash does not match")
    if specification.get("source_state_sha256") != spec["source_state_sha256"]:
        raise PoseCorrectionError("Receipt source-state hash does not match the specification")
    if specification.get("house_layout_sha256") != spec["house_layout_sha256"]:
        raise PoseCorrectionError("Receipt house-layout hash does not match the specification")
    if dict(house_layout_record) != receipt.get("house_layout"):
        raise PoseCorrectionError("Current house layout no longer matches the receipt")
    if _sha256_file(house_layout_path) != spec["house_layout_sha256"]:
        raise PoseCorrectionError("Current house layout no longer matches the specification")

    backup = receipt.get("backup")
    if not isinstance(backup, dict):
        raise PoseCorrectionError("Receipt has no transaction backup record")
    transaction_root = _safe_descendant(
        run_dir,
        backup.get("transaction_root_relative_to_run_dir"),
        label="transaction root",
    )
    transaction_root = _require_real_directory(
        transaction_root, label="transaction root"
    )
    commit_marker = _safe_descendant(
        run_dir, backup.get("commit_marker"), label="transaction commit marker"
    )
    _require_regular_file(commit_marker, label="transaction commit marker")
    original_final = _safe_descendant(
        run_dir, backup.get("original_final_scene"), label="original final_scene"
    )
    original_final = _require_real_directory(
        original_final, label="original final_scene"
    )
    if original_final.parent != transaction_root:
        raise PoseCorrectionError("Original final_scene is outside its transaction")

    source_record = receipt.get("source_state")
    if not isinstance(source_record, dict):
        raise PoseCorrectionError("Receipt has no source-state record")
    source_path = _safe_descendant(
        run_dir, source_record.get("path"), label="backed-up source state"
    )
    if source_path != original_final / "scene_state.json":
        raise PoseCorrectionError("Receipt source-state path is not the transaction backup")
    actual_source_record = _file_record(
        source_path, logical_path=str(source_record.get("path"))
    )
    if actual_source_record != source_record:
        raise PoseCorrectionError("Backed-up source state no longer matches the receipt")
    if source_record.get("sha256") != spec["source_state_sha256"]:
        raise PoseCorrectionError("Backed-up source state does not match the specification")
    source_state, _source_raw = _read_json(
        source_path, label="backed-up source state"
    )

    artifacts = receipt.get("published_artifacts")
    if not isinstance(artifacts, dict):
        raise PoseCorrectionError("Receipt has no published artifact inventory")
    for key, label in (
        ("scene_state", "published state"),
        ("drake_directive", "published Drake directive"),
        ("sceneeval_state", "published SceneEval state"),
    ):
        record = artifacts.get(key)
        if not isinstance(record, dict):
            raise PoseCorrectionError(f"Receipt has no {label} record")
        _verify_file_record(final_dir, record, label=label)

    blend_policy = receipt.get("blend_policy")
    if not isinstance(blend_policy, dict) or blend_policy.get("status") != (
        "intentionally_absent_pending_hash_bound_refresh"
    ):
        raise PoseCorrectionError("Receipt has an unsupported blend lifecycle")
    blend_path = final_dir / "scene.blend"
    if blend_path.exists() or blend_path.is_symlink():
        raise PoseCorrectionError(
            "A blend exists before the required hash-bound refresh superseded this receipt"
        )

    evidence = receipt.get("quarantined_stale_evidence")
    if not isinstance(evidence, list):
        raise PoseCorrectionError("Receipt has no stale-evidence inventory")
    for index, record in enumerate(evidence):
        if not isinstance(record, dict):
            raise PoseCorrectionError(f"Stale-evidence record {index} is malformed")
        original = _safe_descendant(
            scene_dir, record.get("path"), label=f"stale evidence {index}"
        )
        if original.exists() or original.is_symlink():
            raise PoseCorrectionError(f"Stale evidence reappeared: {record.get('path')}")
        quarantined = _safe_descendant(
            transaction_root,
            record.get("backup_path"),
            label=f"quarantined evidence {index}",
        )
        actual = _file_record(
            quarantined, logical_path=str(record.get("path"))
        )
        expected = {
            key: record[key] for key in ("path", "size_bytes", "sha256") if key in record
        }
        if actual != expected:
            raise PoseCorrectionError(
                f"Quarantined evidence no longer matches: {record.get('path')}"
            )

    state, _state_raw = _read_json(final_dir / "scene_state.json", label="published room state")
    object_ids, manipuland_ids = _object_and_manipuland_ids(state)
    if object_ids != receipt.get("object_ids"):
        raise PoseCorrectionError("Published object inventory differs from the receipt")
    if manipuland_ids != receipt.get("manipuland_ids"):
        raise PoseCorrectionError("Published manipuland inventory differs from the receipt")
    recomputed = prepare_correction(source_state, spec, runtime=runtime)
    if _canonical_json(recomputed.corrected_state) != _canonical_json(state):
        raise PoseCorrectionError(
            "Published state differs from the correction recomputed from its source"
        )
    dmd_path = _require_regular_file(
        final_dir / "scene.dmd.yaml", label="published Drake directive"
    )
    if dmd_path.read_bytes() != recomputed.dmd.encode("utf-8"):
        raise PoseCorrectionError("Published Drake directive differs from recomputation")
    sceneeval, _sceneeval_raw = _read_json(
        final_dir / "sceneeval_state.json", label="published SceneEval state"
    )
    if _canonical_json(sceneeval) != _canonical_json(recomputed.sceneeval):
        raise PoseCorrectionError("Published SceneEval state differs from recomputation")
    if receipt.get("corrections") != recomputed.correction_records:
        raise PoseCorrectionError("Receipt corrections differ from recomputation")
    collision_proof = receipt.get("collision_proof")
    expected_collision_proof = {
        "policy": dict(spec["collision_policy"]),
        "source_collision_count": len(recomputed.source_collisions),
        "source_collisions": recomputed.source_collisions,
        "corrected_collision_count": len(recomputed.corrected_collisions),
        "corrected_collisions": recomputed.corrected_collisions,
        "roundtrip_recomputed": True,
    }
    if collision_proof != expected_collision_proof:
        raise PoseCorrectionError("Receipt collision proof differs from recomputation")
    if receipt.get("manipuland_count") != len(recomputed.manipuland_ids):
        raise PoseCorrectionError("Receipt manipuland count differs from recomputation")
    return receipt


def run(
    *,
    room_dir: Path | str,
    house_layout: Path | str,
    spec_path: Path | str,
    apply: bool = False,
    verify_only: bool = False,
    receipt_path: Path | str | None = None,
    runtime: Any | None = None,
    failure_injector: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if apply and verify_only:
        raise PoseCorrectionError("--apply and --verify-only are mutually exclusive")
    room = _require_real_directory(Path(room_dir), label="room directory")
    spec_file = _require_regular_file(Path(spec_path), label="correction specification")
    spec_value, spec_raw = _read_json(spec_file, label="correction specification")
    spec = _validate_spec(spec_value)
    spec_sha = _sha256_bytes(spec_raw)
    if room.name != f"room_{spec['room_id']}":
        raise PoseCorrectionError(
            f"Room directory {room.name!r} does not match room_id {spec['room_id']!r}"
        )
    final_dir = room / "scene_states" / "final_scene"
    final_dir = _require_real_directory(final_dir, label="final_scene directory")
    source_state_path = final_dir / "scene_state.json"
    house_layout_file = _require_regular_file(Path(house_layout), label="house layout")
    house_layout_record = _file_record(
        house_layout_file, logical_path="house_layout.json"
    )
    if house_layout_record["sha256"] != spec["house_layout_sha256"]:
        raise PoseCorrectionError(
            f"House-layout SHA-256 changed: {house_layout_record['sha256']} != "
            f"{spec['house_layout_sha256']}"
        )

    if runtime is None:
        runtime = ProductionRuntime(
            room_dir=room,
            room_id=spec["room_id"],
            house_layout_path=house_layout_file,
            collision_policy=spec["collision_policy"],
            sceneeval_asset_id_prefix=spec["sceneeval_asset_id_prefix"],
        )

    if verify_only:
        return verify_published(
            room_dir=room,
            spec_path=spec_file,
            spec=spec,
            spec_sha256=spec_sha,
            house_layout_path=house_layout_file,
            house_layout_record=house_layout_record,
            runtime=runtime,
            receipt_path=Path(receipt_path) if receipt_path else None,
        )

    source_state, source_raw = _read_json(source_state_path, label="source final state")
    source_sha = _sha256_bytes(source_raw)
    if source_sha != spec["source_state_sha256"]:
        raise PoseCorrectionError(
            f"Source final-state SHA-256 changed: {source_sha} != "
            f"{spec['source_state_sha256']}"
        )
    if source_state.get("text_description") is None:
        raise PoseCorrectionError("Source room state has no text_description")
    prepared = prepare_correction(source_state, spec, runtime=runtime)
    dry_result = {
        "schema_id": RECEIPT_SCHEMA_ID,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "pass",
        "mode": "dry_run",
        "room_id": spec["room_id"],
        "source_state_sha256": source_sha,
        "specification_sha256": spec_sha,
        "object_count": len(prepared.object_ids),
        "manipuland_count": len(prepared.manipuland_ids),
        "corrections": prepared.correction_records,
        "source_collision_count": len(prepared.source_collisions),
        "corrected_collision_count": len(prepared.corrected_collisions),
        "would_remove_stale_blend": (final_dir / "scene.blend").exists(),
        "would_quarantine_stale_evidence": [
            path.relative_to(room.parent).as_posix()
            for path in _stale_evidence_paths(room.parent, spec["room_id"])
        ],
    }
    if not apply:
        return dry_result

    lock_path = room / ".pose_correction.lock"
    lock_descriptor = _acquire_lock(lock_path)
    try:
        receipt = _publish(
            room_dir=room,
            final_dir=final_dir,
            source_state_path=source_state_path,
            source_state_sha256=source_sha,
            spec_path=spec_file,
            spec_sha256=spec_sha,
            spec=spec,
            house_layout_record=house_layout_record,
            prepared=prepared,
            failure_injector=failure_injector,
        )
    finally:
        _release_lock(lock_path, lock_descriptor)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room-dir", type=Path, required=True)
    parser.add_argument("--house-layout", type=Path, required=True)
    parser.add_argument("--spec", dest="spec_path", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Publish the proven correction transactionally.")
    mode.add_argument("--verify-only", action="store_true", help="Rehash and recompute an existing published receipt.")
    parser.add_argument("--receipt", type=Path, help="Receipt path for --verify-only; defaults inside final_scene.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(
            room_dir=args.room_dir,
            house_layout=args.house_layout,
            spec_path=args.spec_path,
            apply=args.apply,
            verify_only=args.verify_only,
            receipt_path=args.receipt,
        )
    except PoseCorrectionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
