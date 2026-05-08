#!/usr/bin/env python3
"""Recover the reference classroom furniture checkpoint from an archived agent log.

The furniture agent can finish a substantial amount of valid scene work before a
renderer-specific failure prevents the normal checkpoint writer from running.  This
tool reconstructs that state without replaying model calls or asset generation.

Recovery is deliberately narrow and fail closed:

* the SQLite database is opened read-only and its main/WAL bytes must remain stable;
* one exact ``get_current_scene_state`` output anchors the reconstructed scene;
* only successful move, rescale, and snap calls through an explicit cutoff are replayed;
* the cutoff must precede a specified failed ``observe_scene`` output;
* every non-wall object must map to a successful earlier placement and an exact asset
  registry template;
* the reference classroom's fixed 36-object inventory is enforced;
* each used copied Artiverse SDF is normalized to its in-tree publisher GLB visuals,
  copied hashes are refreshed from the resulting bytes, and immutable source hashes
  are revalidated; and
* publication uses a new transaction directory and refuses to replace an existing
  ``scene_after_furniture`` checkpoint.

No prompts, source assets, registry files, or database rows are modified.  The only
generated-asset mutation is the normalizer's atomic, visual-only copied-SDF rewrite;
collision XML and publisher files remain byte-identical.  Checkpoint publication is
also atomic.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sqlite3
import stat
import sys
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scenesmith.agent_utils.artiverse_visual_normalization import (
    ArtiverseVisualNormalizationError,
    normalize_copied_artiverse_visuals,
)


SCHEMA_VERSION = 1
CHECKPOINT_NAME = "scene_after_furniture"
STAGING_NAME = f".{CHECKPOINT_NAME}.recovery-staging"
ROOM_ID = "classroom_01"
EXPECTED_WALL_IDS = ("north_wall", "south_wall", "east_wall", "west_wall")
EXPECTED_SINGLE_FURNITURE_IDS = (
    "whiteboard_0",
    "teacher_desk_0",
    "office_chair_0",
    "filing_cabinet_0",
    "cubby_shelf_unit_0",
    "storage_cabinet_0",
    "trash_bin_0",
    "potted_plant_0",
)
EXPECTED_DESK_IDS = tuple(
    f"student_desk_{suffix}" for suffix in (*map(str, range(10)), "a", "b")
)
EXPECTED_CHAIR_IDS = tuple(
    f"classroom_student_chair_{suffix}"
    for suffix in (*map(str, range(10)), "a", "b")
)
EXPECTED_OBJECT_IDS = frozenset(
    (*EXPECTED_WALL_IDS, *EXPECTED_SINGLE_FURNITURE_IDS, *EXPECTED_DESK_IDS, *EXPECTED_CHAIR_IDS)
)
ALLOWED_REPLAY_CALLS = frozenset(
    {"move_furniture_tool", "rescale_furniture_tool", "snap_to_object_tool"}
)
FLOAT_ABS_TOLERANCE = 1.0e-9
FLOAT_REL_TOLERANCE = 1.0e-9


class RecoveryError(RuntimeError):
    """Raised when archived evidence cannot support an exact recovery."""


@dataclass(frozen=True)
class MessageRow:
    id: int
    session_id: str
    raw: str
    created_at: str
    document: Mapping[str, Any]


@dataclass
class LogicalObject:
    object_id: str
    description: str
    x: float
    y: float
    z: float
    yaw_degrees: float
    dimensions: tuple[float, float, float]
    asset_id: str | None = None
    final_asset_scale: float | None = None


def _require_regular_file(path: Path, label: str) -> Path:
    candidate = path.expanduser().resolve(strict=True)
    info = candidate.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise RecoveryError(f"{label} must be an unlinked regular file: {path}")
    return candidate


def _require_directory(path: Path, label: str) -> Path:
    candidate = path.expanduser().resolve(strict=True)
    info = candidate.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISDIR(info.st_mode):
        raise RecoveryError(f"{label} must be an unlinked directory: {path}")
    return candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_directory_tree(root: Path) -> str:
    """Hash a regular-file tree exactly like the Artiverse provenance contract."""

    canonical = _require_directory(root, "Articulated asset tree")
    files: list[Path] = []
    for candidate in canonical.rglob("*"):
        info = candidate.stat(follow_symlinks=False)
        if candidate.is_symlink():
            raise RecoveryError(f"Asset tree contains a symbolic link: {candidate}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise RecoveryError(f"Asset tree contains a special file: {candidate}")
        files.append(candidate)
    files.sort(key=lambda candidate: candidate.relative_to(canonical).as_posix())
    if not files:
        raise RecoveryError(f"Articulated asset tree is empty: {canonical}")

    digest = hashlib.sha256()
    for candidate in files:
        relative = candidate.relative_to(canonical).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(candidate)))
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"Cannot read {label} as JSON: {path}: {exc}") from exc


def _write_json_exclusive(path: Path, value: Any) -> None:
    serialized = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


def _write_text_exclusive(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecoveryError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RecoveryError(f"{label} must be finite")
    return result


def _positive_dimensions(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, Mapping):
        raise RecoveryError(f"{label} must be an object")
    result = tuple(
        _finite_number(value.get(key), f"{label}.{key}")
        for key in ("width", "depth", "height")
    )
    if any(component <= 0.0 for component in result):
        raise RecoveryError(f"{label} must contain positive dimensions")
    return result  # type: ignore[return-value]


def _close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=FLOAT_REL_TOLERANCE,
        abs_tol=FLOAT_ABS_TOLERANCE,
    )


def _require_vector_close(
    actual: Sequence[float], expected: Sequence[float], label: str
) -> None:
    if len(actual) != len(expected) or any(
        not _close(float(left), float(right))
        for left, right in zip(actual, expected, strict=True)
    ):
        raise RecoveryError(f"{label} mismatch: actual={list(actual)}, expected={list(expected)}")


def _parse_message_document(raw: str, message_id: int) -> Mapping[str, Any]:
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RecoveryError(f"designer.db message {message_id} is not JSON: {exc}") from exc
    if not isinstance(document, Mapping):
        raise RecoveryError(f"designer.db message {message_id} is not a JSON object")
    return document


def _db_family_snapshot(database: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for suffix, label in (("", "main"), ("-wal", "wal")):
        candidate = Path(f"{database}{suffix}")
        if not candidate.exists():
            if suffix == "":
                raise RecoveryError(f"designer.db is missing: {database}")
            continue
        canonical = _require_regular_file(candidate, f"designer.db {label}")
        result[label] = {
            "path": str(canonical),
            "size_bytes": canonical.stat().st_size,
            "sha256": sha256_file(canonical),
        }
    return result


def _load_messages_read_only(database: Path) -> list[MessageRow]:
    uri = database.as_uri() + "?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise RecoveryError(f"Cannot open designer.db read-only: {exc}") from exc
    try:
        connection.execute("PRAGMA query_only = ON")
        table = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='agent_messages'"
        ).fetchone()
        if table is None:
            raise RecoveryError("designer.db has no agent_messages table")
        rows = connection.execute(
            "SELECT id, session_id, message_data, created_at "
            "FROM agent_messages ORDER BY id"
        ).fetchall()
    except sqlite3.Error as exc:
        raise RecoveryError(f"Cannot read designer.db agent messages: {exc}") from exc
    finally:
        connection.close()

    messages: list[MessageRow] = []
    previous_id = 0
    for raw_id, session_id, raw, created_at in rows:
        message_id = int(raw_id)
        if message_id <= previous_id:
            raise RecoveryError("designer.db message IDs are not strictly increasing")
        previous_id = message_id
        if not isinstance(session_id, str) or not session_id:
            raise RecoveryError(f"designer.db message {message_id} has no session ID")
        if not isinstance(raw, str) or not isinstance(created_at, str):
            raise RecoveryError(f"designer.db message {message_id} has malformed columns")
        messages.append(
            MessageRow(
                id=message_id,
                session_id=session_id,
                raw=raw,
                created_at=created_at,
                document=_parse_message_document(raw, message_id),
            )
        )
    if not messages:
        raise RecoveryError("designer.db contains no agent messages")
    return messages


def _message_index(messages: Sequence[MessageRow]) -> dict[int, MessageRow]:
    return {row.id: row for row in messages}


def _pair_calls(
    messages: Sequence[MessageRow],
) -> tuple[dict[str, MessageRow], dict[str, MessageRow]]:
    calls: dict[str, MessageRow] = {}
    outputs: dict[str, MessageRow] = {}
    for row in messages:
        kind = row.document.get("type")
        if kind not in {"function_call", "function_call_output"}:
            continue
        call_id = row.document.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise RecoveryError(f"Function message {row.id} has no call_id")
        destination = calls if kind == "function_call" else outputs
        if call_id in destination:
            raise RecoveryError(f"Duplicate {kind} for call_id {call_id}")
        destination[call_id] = row
    return calls, outputs


def _parse_output_json(row: MessageRow, label: str) -> Mapping[str, Any]:
    value = row.document.get("output")
    if not isinstance(value, str):
        raise RecoveryError(f"{label} output in message {row.id} is not a string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RecoveryError(f"{label} output in message {row.id} is not JSON: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise RecoveryError(f"{label} output in message {row.id} is not an object")
    return parsed


def _parse_arguments(row: MessageRow, label: str) -> Mapping[str, Any]:
    value = row.document.get("arguments")
    if not isinstance(value, str):
        raise RecoveryError(f"{label} arguments in message {row.id} are not a string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RecoveryError(f"{label} arguments in message {row.id} are not JSON: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise RecoveryError(f"{label} arguments in message {row.id} are not an object")
    return parsed


def _validate_reference_inventory(object_ids: Iterable[str], label: str) -> None:
    actual = frozenset(object_ids)
    if actual != EXPECTED_OBJECT_IDS:
        missing = sorted(EXPECTED_OBJECT_IDS - actual)
        extra = sorted(actual - EXPECTED_OBJECT_IDS)
        raise RecoveryError(
            f"{label} does not have the exact 36-object reference inventory: "
            f"missing={missing}, extra={extra}"
        )


def _logical_anchor(
    state_output: Mapping[str, Any],
) -> tuple[list[str], dict[str, LogicalObject]]:
    if state_output.get("success") is not True:
        raise RecoveryError("get_current_scene_state did not report success")
    objects = state_output.get("objects")
    if not isinstance(objects, list) or not objects:
        raise RecoveryError("get_current_scene_state has no object list")
    if state_output.get("furniture_count") != len(objects):
        raise RecoveryError("get_current_scene_state furniture_count is stale")

    order: list[str] = []
    logical: dict[str, LogicalObject] = {}
    for index, value in enumerate(objects):
        if not isinstance(value, Mapping):
            raise RecoveryError(f"get_current_scene_state object {index} is malformed")
        object_id = value.get("object_id")
        description = value.get("description")
        if not isinstance(object_id, str) or not object_id:
            raise RecoveryError(f"get_current_scene_state object {index} has no ID")
        if object_id in logical:
            raise RecoveryError(f"get_current_scene_state repeats object {object_id}")
        if not isinstance(description, str) or not description:
            raise RecoveryError(f"get_current_scene_state object {object_id} has no description")
        dimensions = _positive_dimensions(value.get("dimensions"), f"{object_id}.dimensions")
        logical[object_id] = LogicalObject(
            object_id=object_id,
            description=description,
            x=_finite_number(value.get("position_x"), f"{object_id}.position_x"),
            y=_finite_number(value.get("position_y"), f"{object_id}.position_y"),
            z=0.0,
            yaw_degrees=_finite_number(
                value.get("rotation_degrees"), f"{object_id}.rotation_degrees"
            ),
            dimensions=dimensions,
        )
        order.append(object_id)
    _validate_reference_inventory(order, "Archived anchor state")
    return order, logical


def _successful_prior_add_mapping(
    messages: Sequence[MessageRow],
    calls: Mapping[str, MessageRow],
    outputs: Mapping[str, MessageRow],
    *,
    before_message_id: int,
    needed_ids: set[str],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    mapping: dict[str, str] = {}
    evidence: list[dict[str, Any]] = []
    for call in messages:
        if call.id >= before_message_id:
            break
        if call.document.get("type") != "function_call" or call.document.get(
            "name"
        ) != "add_furniture_to_scene_tool":
            continue
        call_id = str(call.document.get("call_id"))
        output = outputs.get(call_id)
        if output is None or output.id >= before_message_id:
            continue
        result = _parse_output_json(output, "add_furniture_to_scene_tool")
        if result.get("success") is not True:
            continue
        arguments = _parse_arguments(call, "add_furniture_to_scene_tool")
        object_id = result.get("object_id")
        asset_id = result.get("asset_id")
        if object_id != result.get("object_id") or asset_id != arguments.get("asset_id"):
            raise RecoveryError(f"Placement call/output mismatch for call_id {call_id}")
        if not isinstance(object_id, str) or not isinstance(asset_id, str):
            raise RecoveryError(f"Placement call/output {call_id} has malformed IDs")
        if object_id not in needed_ids:
            continue
        previous = mapping.get(object_id)
        if previous is not None and previous != asset_id:
            raise RecoveryError(f"Object {object_id} maps to multiple registry assets")
        mapping[object_id] = asset_id
        evidence.append(
            {
                "call_message_id": call.id,
                "output_message_id": output.id,
                "call_id": call_id,
                "object_id": object_id,
                "asset_id": asset_id,
                "call_sha256": hashlib.sha256(call.raw.encode("utf-8")).hexdigest(),
                "output_sha256": hashlib.sha256(output.raw.encode("utf-8")).hexdigest(),
            }
        )
    missing = sorted(needed_ids - set(mapping))
    if missing:
        raise RecoveryError(f"No successful archived placement maps these objects: {missing}")
    return mapping, evidence


def _created_at_epoch(created_at: str) -> float:
    try:
        parsed = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise RecoveryError(f"Unsupported SQLite created_at timestamp: {created_at}") from exc
    return parsed.timestamp()


def _replay_operations(
    *,
    messages: Sequence[MessageRow],
    calls: Mapping[str, MessageRow],
    outputs: Mapping[str, MessageRow],
    logical: dict[str, LogicalObject],
    state_message_id: int,
    cutoff_message_id: int,
) -> tuple[list[dict[str, Any]], set[int]]:
    rows_in_window = [
        row for row in messages if state_message_id < row.id <= cutoff_message_id
    ]
    consumed: set[int] = set()
    operations: list[dict[str, Any]] = []

    operation_calls = [
        row for row in rows_in_window if row.document.get("type") == "function_call"
    ]
    for call in operation_calls:
        name = call.document.get("name")
        if name not in ALLOWED_REPLAY_CALLS:
            raise RecoveryError(
                f"Disallowed function call before recovery cutoff: message {call.id} {name}"
            )
        if call.document.get("status") != "completed":
            raise RecoveryError(f"Replay call {call.id} is not completed")
        call_id = str(call.document.get("call_id"))
        output = outputs.get(call_id)
        if output is None or not (state_message_id < output.id <= cutoff_message_id):
            raise RecoveryError(f"Replay call {call.id} has no output inside the cutoff")
        result = _parse_output_json(output, str(name))
        if result.get("success") is not True:
            raise RecoveryError(f"Replay operation {call.id}/{output.id} was not successful")
        arguments = _parse_arguments(call, str(name))
        object_id = arguments.get("object_id")
        if not isinstance(object_id, str) or object_id not in logical:
            raise RecoveryError(f"Replay operation {call.id} targets an unknown object")
        if object_id in EXPECTED_WALL_IDS:
            raise RecoveryError(f"Replay operation {call.id} attempts to mutate a wall")
        target = logical[object_id]

        if name == "move_furniture_tool":
            if result.get("object_id") != object_id:
                raise RecoveryError(f"Move output {output.id} has the wrong object ID")
            target.x = _finite_number(arguments.get("x"), f"move {call.id} x")
            target.y = _finite_number(arguments.get("y"), f"move {call.id} y")
            target.z = 0.0
            target.yaw_degrees = _finite_number(
                arguments.get("yaw", 0.0), f"move {call.id} yaw"
            )

        elif name == "rescale_furniture_tool":
            if result.get("object_id") != object_id:
                raise RecoveryError(f"Rescale output {output.id} has the wrong object ID")
            factor = _finite_number(arguments.get("scale_factor"), f"rescale {call.id}")
            if factor <= 0.0 or not _close(
                factor, _finite_number(result.get("scale_factor"), f"rescale {output.id}")
            ):
                raise RecoveryError(f"Rescale call/output factor mismatch at {call.id}")
            previous = _positive_dimensions(
                result.get("previous_dimensions"), f"rescale {output.id}.previous_dimensions"
            )
            _require_vector_close(target.dimensions, previous, f"Rescale {call.id} previous dimensions")
            new_dimensions = _positive_dimensions(
                result.get("new_dimensions"), f"rescale {output.id}.new_dimensions"
            )
            _require_vector_close(
                tuple(component * factor for component in previous),
                new_dimensions,
                f"Rescale {call.id} new dimensions",
            )
            affected = result.get("affected_object_ids")
            if not isinstance(affected, list) or object_id not in affected:
                raise RecoveryError(f"Rescale output {output.id} has invalid affected objects")
            new_scale = _finite_number(
                result.get("new_asset_scale"), f"rescale {output.id}.new_asset_scale"
            )
            if new_scale <= 0.0:
                raise RecoveryError(f"Rescale output {output.id} has invalid cumulative scale")
            for affected_id in affected:
                if not isinstance(affected_id, str) or affected_id not in logical:
                    raise RecoveryError(f"Rescale output {output.id} names an unknown object")
                affected_object = logical[affected_id]
                _require_vector_close(
                    affected_object.dimensions,
                    previous,
                    f"Rescale {call.id} affected dimensions for {affected_id}",
                )
                affected_object.dimensions = new_dimensions
                affected_object.final_asset_scale = new_scale

        elif name == "snap_to_object_tool":
            target_id = arguments.get("target_id")
            if result.get("object_id") != object_id or result.get("target_id") != target_id:
                raise RecoveryError(f"Snap call/output identity mismatch at {call.id}")
            if not isinstance(target_id, str) or target_id not in logical:
                raise RecoveryError(f"Snap operation {call.id} has an unknown target")
            original = result.get("original_position")
            new = result.get("new_position")
            if not isinstance(original, Mapping) or not isinstance(new, Mapping):
                raise RecoveryError(f"Snap output {output.id} has no positions")
            original_xyz = tuple(
                _finite_number(original.get(axis), f"snap {output.id}.original_{axis}")
                for axis in ("x", "y", "z")
            )
            _require_vector_close(
                (target.x, target.y, target.z), original_xyz, f"Snap {call.id} original position"
            )
            new_xyz = tuple(
                _finite_number(new.get(axis), f"snap {output.id}.new_{axis}")
                for axis in ("x", "y", "z")
            )
            reported_distance = _finite_number(
                result.get("distance_moved"), f"snap {output.id}.distance_moved"
            )
            actual_distance = math.dist(original_xyz, new_xyz)
            if not _close(reported_distance, actual_distance):
                raise RecoveryError(f"Snap output {output.id} has a stale distance")
            target.x, target.y, target.z = new_xyz
            if result.get("rotation_applied") is True:
                target.yaw_degrees = _finite_number(
                    result.get("rotation_angle_degrees"),
                    f"snap {output.id}.rotation_angle_degrees",
                )
            elif result.get("rotation_applied") is not False:
                raise RecoveryError(f"Snap output {output.id} has malformed rotation evidence")

        consumed.update({call.id, output.id})
        operations.append(
            {
                "name": name,
                "call_id": call_id,
                "call_message_id": call.id,
                "output_message_id": output.id,
                "object_id": object_id,
                "arguments": arguments,
                "result": result,
                "call_sha256": hashlib.sha256(call.raw.encode("utf-8")).hexdigest(),
                "output_sha256": hashlib.sha256(output.raw.encode("utf-8")).hexdigest(),
            }
        )

    for row in rows_in_window:
        kind = row.document.get("type")
        if row.id in consumed or kind == "reasoning":
            continue
        raise RecoveryError(
            f"Unconsumed message inside replay window: {row.id} type={kind}"
        )
    return operations, consumed


def _validate_first_observe_failure(
    *,
    messages: Sequence[MessageRow],
    calls: Mapping[str, MessageRow],
    failure_message_id: int,
    cutoff_message_id: int,
    expected_substring: str,
) -> tuple[MessageRow, MessageRow]:
    by_id = _message_index(messages)
    failure = by_id.get(failure_message_id)
    if failure is None or failure.document.get("type") != "function_call_output":
        raise RecoveryError("Specified observe failure is not a function_call_output")
    call_id = failure.document.get("call_id")
    call = calls.get(str(call_id))
    if call is None or call.document.get("name") != "observe_scene":
        raise RecoveryError("Specified failure does not belong to observe_scene")
    if not (cutoff_message_id < call.id < failure.id):
        raise RecoveryError("Recovery cutoff does not precede the observe failure")
    output = failure.document.get("output")
    if not isinstance(output, str) or expected_substring not in output:
        raise RecoveryError(
            f"Observe failure does not contain expected marker: {expected_substring!r}"
        )
    for row in messages:
        if (
            cutoff_message_id < row.id < call.id
            and row.document.get("type") == "function_call"
            and row.document.get("name") == "observe_scene"
        ):
            raise RecoveryError(
                f"Specified failure is not the first observe_scene after cutoff: {row.id}"
            )
        if cutoff_message_id < row.id < failure.id and row.id != call.id:
            if row.document.get("type") != "reasoning":
                raise RecoveryError(
                    f"Unexpected message between cutoff and observe failure: {row.id}"
                )
    return call, failure


def _yaw_from_quaternion(value: Any, label: str) -> float:
    if not isinstance(value, list) or len(value) != 4:
        raise RecoveryError(f"{label} is not a wxyz quaternion")
    w, x, y, z = (_finite_number(component, label) for component in value)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        raise RecoveryError(f"{label} is a zero quaternion")
    w, x, y, z = (component / norm for component in (w, x, y, z))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return math.degrees(yaw)


def _validate_initial_walls(
    initial: Mapping[str, Any], logical: Mapping[str, LogicalObject]
) -> Mapping[str, Any]:
    objects = initial.get("objects")
    if not isinstance(objects, Mapping) or set(objects) != set(EXPECTED_WALL_IDS):
        raise RecoveryError("Initial scene state must contain exactly the four room walls")
    if not isinstance(initial.get("room_geometry"), Mapping):
        raise RecoveryError("Initial scene state has no room_geometry")
    if not isinstance(initial.get("text_description"), str) or not initial.get(
        "text_description"
    ):
        raise RecoveryError("Initial scene state has no immutable room prompt")
    for wall_id in EXPECTED_WALL_IDS:
        value = objects[wall_id]
        if not isinstance(value, Mapping):
            raise RecoveryError(f"Initial wall {wall_id} is malformed")
        if value.get("object_type") != "wall" or value.get("immutable") is not True:
            raise RecoveryError(f"Initial wall {wall_id} is not an immutable wall")
        transform = value.get("transform")
        if not isinstance(transform, Mapping):
            raise RecoveryError(f"Initial wall {wall_id} has no transform")
        translation = transform.get("translation")
        if not isinstance(translation, list) or len(translation) != 3:
            raise RecoveryError(f"Initial wall {wall_id} has malformed translation")
        anchor = logical[wall_id]
        _require_vector_close(
            (anchor.x, anchor.y),
            (
                _finite_number(translation[0], f"{wall_id}.x"),
                _finite_number(translation[1], f"{wall_id}.y"),
            ),
            f"Initial wall {wall_id} position",
        )
        yaw = _yaw_from_quaternion(
            transform.get("rotation_wxyz"), f"{wall_id}.rotation_wxyz"
        )
        if not _close(anchor.yaw_degrees, yaw):
            raise RecoveryError(f"Initial wall {wall_id} yaw differs from archived state")
        bbox_min = value.get("bbox_min")
        bbox_max = value.get("bbox_max")
        if not isinstance(bbox_min, list) or not isinstance(bbox_max, list):
            raise RecoveryError(f"Initial wall {wall_id} has no bounds")
        dimensions = tuple(float(b) - float(a) for a, b in zip(bbox_min, bbox_max, strict=True))
        _require_vector_close(anchor.dimensions, dimensions, f"Initial wall {wall_id} dimensions")
        if value.get("description") != anchor.description:
            raise RecoveryError(f"Initial wall {wall_id} description differs from archive")
    return objects


def _resolve_registry_path(value: Any, label: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise RecoveryError(f"{label} path is malformed")
    path = Path(value)
    if not path.is_absolute():
        raise RecoveryError(f"{label} must use an absolute registry path: {path}")
    return _require_regular_file(path, label)


def _relative_state_path(path: Path | None, room_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(room_dir).as_posix()
    except ValueError:
        return str(path)


def _refresh_artiverse_metadata(metadata: dict[str, Any], copied_sdf: Path) -> dict[str, str]:
    source_value = metadata.get("articulated_source_sdf_path")
    if not isinstance(source_value, str) or not source_value:
        raise RecoveryError("Artiverse registry metadata has no source SDF path")
    source_sdf = _require_regular_file(Path(source_value), "Artiverse source SDF")
    source_sdf_hash = sha256_file(source_sdf)
    source_tree_hash = sha256_directory_tree(source_sdf.parent)
    if metadata.get("articulated_source_sdf_sha256") != source_sdf_hash:
        raise RecoveryError(f"Artiverse immutable source SDF hash is stale: {source_sdf}")
    if metadata.get("articulated_source_tree_sha256") != source_tree_hash:
        raise RecoveryError(f"Artiverse immutable source tree hash is stale: {source_sdf.parent}")

    copied_sdf_hash = sha256_file(copied_sdf)
    copied_tree_hash = sha256_directory_tree(copied_sdf.parent)
    metadata["articulated_copied_sdf_sha256"] = copied_sdf_hash
    metadata["articulated_copied_tree_sha256"] = copied_tree_hash
    return {
        "articulated_id": str(metadata.get("articulated_id", "")),
        "source_sdf_path": str(source_sdf),
        "source_sdf_sha256": source_sdf_hash,
        "source_tree_sha256": source_tree_hash,
        "copied_sdf_path": str(copied_sdf),
        "copied_sdf_sha256": copied_sdf_hash,
        "copied_tree_sha256": copied_tree_hash,
    }


def _quaternion_for_yaw(yaw_degrees: float) -> list[float]:
    radians = math.radians(yaw_degrees)
    return [math.cos(radians / 2.0), 0.0, 0.0, math.sin(radians / 2.0)]


def _materialize_objects(
    *,
    order: Sequence[str],
    logical: Mapping[str, LogicalObject],
    initial_walls: Mapping[str, Any],
    registry: Mapping[str, Any],
    placement_mapping: Mapping[str, str],
    room_dir: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, str]],
    list[dict[str, Any]],
    dict[str, str],
]:
    objects: dict[str, Any] = {}
    artiverse_bindings: list[dict[str, str]] = []
    normalization_evidence: list[dict[str, Any]] = []
    normalized_by_sdf: dict[Path, dict[str, Any]] = {}
    instance_assets: dict[str, str] = {}

    for object_id in order:
        if object_id in initial_walls:
            objects[object_id] = copy.deepcopy(initial_walls[object_id])
            continue
        asset_id = placement_mapping[object_id]
        raw_template = registry.get(asset_id)
        if not isinstance(raw_template, Mapping):
            raise RecoveryError(f"Registry has no valid template for {asset_id}")
        template = copy.deepcopy(dict(raw_template))
        if template.get("object_id") != asset_id:
            raise RecoveryError(f"Registry key/object_id mismatch for {asset_id}")
        if template.get("object_type") != "furniture":
            raise RecoveryError(f"Registry asset {asset_id} is not furniture")
        target = logical[object_id]
        if template.get("description") != target.description:
            raise RecoveryError(f"Registry description differs for {object_id}/{asset_id}")

        bbox_min = template.get("bbox_min")
        bbox_max = template.get("bbox_max")
        if not isinstance(bbox_min, list) or not isinstance(bbox_max, list) or len(
            bbox_min
        ) != 3 or len(bbox_max) != 3:
            raise RecoveryError(f"Registry asset {asset_id} has malformed bounds")
        dimensions = tuple(
            _finite_number(high, f"{asset_id}.bbox_max")
            - _finite_number(low, f"{asset_id}.bbox_min")
            for low, high in zip(bbox_min, bbox_max, strict=True)
        )
        _require_vector_close(dimensions, target.dimensions, f"Final dimensions for {object_id}")
        registry_scale = _finite_number(template.get("scale_factor", 1.0), f"{asset_id}.scale")
        if target.final_asset_scale is not None and not _close(
            registry_scale, target.final_asset_scale
        ):
            raise RecoveryError(
                f"Registry scale for {asset_id} does not match archived rescale output"
            )

        geometry_path = _resolve_registry_path(
            template.get("geometry_path"), f"{asset_id} geometry"
        )
        sdf_path = _resolve_registry_path(template.get("sdf_path"), f"{asset_id} SDF")
        image_path = _resolve_registry_path(template.get("image_path"), f"{asset_id} image")
        if geometry_path is None or sdf_path is None:
            raise RecoveryError(f"Registry asset {asset_id} lacks geometry or SDF")

        metadata = template.get("metadata")
        if not isinstance(metadata, Mapping):
            raise RecoveryError(f"Registry asset {asset_id} has malformed metadata")
        refreshed_metadata = copy.deepcopy(dict(metadata))
        if str(refreshed_metadata.get("articulated_source", "")).lower() == "artiverse":
            evidence = normalized_by_sdf.get(sdf_path)
            if evidence is None:
                try:
                    evidence = normalize_copied_artiverse_visuals(
                        sdf_path, sdf_path.parent
                    )
                except ArtiverseVisualNormalizationError as exc:
                    raise RecoveryError(
                        f"Cannot normalize copied Artiverse visuals for {object_id}: {exc}"
                    ) from exc
                normalized_by_sdf[sdf_path] = evidence
                normalization_evidence.append(evidence)
            binding = _refresh_artiverse_metadata(refreshed_metadata, sdf_path)
            if (
                evidence.get("status") != "pass"
                or evidence.get("copied_sdf_path") != str(sdf_path)
                or evidence.get("sdf_sha256_after")
                != binding["copied_sdf_sha256"]
                or evidence.get("copied_tree_sha256_after")
                != binding["copied_tree_sha256"]
            ):
                raise RecoveryError(
                    f"Artiverse visual-normalization evidence is stale for {object_id}"
                )
            binding["object_id"] = object_id
            binding["asset_id"] = asset_id
            artiverse_bindings.append(binding)

        template.update(
            {
                "object_id": object_id,
                "transform": {
                    "translation": [target.x, target.y, target.z],
                    "rotation_wxyz": _quaternion_for_yaw(target.yaw_degrees),
                },
                "geometry_path": _relative_state_path(geometry_path, room_dir),
                "sdf_path": _relative_state_path(sdf_path, room_dir),
                "image_path": _relative_state_path(image_path, room_dir),
                "support_surfaces": [],
                "placement_info": None,
                "metadata": refreshed_metadata,
                "immutable": False,
            }
        )
        objects[object_id] = template
        instance_assets[object_id] = asset_id

    _validate_reference_inventory(objects, "Materialized checkpoint")
    if not any(binding["object_id"] == "storage_cabinet_0" for binding in artiverse_bindings):
        raise RecoveryError("Recovered storage_cabinet_0 is not bound to Artiverse")
    if not normalization_evidence:
        raise RecoveryError("Recovered checkpoint has no Artiverse visual normalization")
    return objects, artiverse_bindings, normalization_evidence, instance_assets


def _resolve_state_path(value: Any, room_dir: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RecoveryError(f"{label} path is missing")
    path = Path(value)
    if not path.is_absolute():
        path = room_dir / path
    return _require_regular_file(path, label)


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _sdf_model_info(path: Path) -> tuple[str, bool]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise RecoveryError(f"Cannot parse SDF for directive: {path}: {exc}") from exc
    base_link = None
    is_static = False
    for element in root.iter():
        local = _xml_local_name(str(element.tag))
        if local == "link" and base_link is None:
            name = element.get("name")
            if isinstance(name, str) and name:
                base_link = name
        elif local == "static" and (element.text or "").strip().lower() == "true":
            is_static = True
    if base_link is None:
        raise RecoveryError(f"SDF has no named link: {path}")
    return base_link, is_static


def _angle_axis_for_yaw(yaw_degrees: float) -> tuple[float, tuple[float, float, float]]:
    normalized = math.degrees(
        math.atan2(math.sin(math.radians(yaw_degrees)), math.cos(math.radians(yaw_degrees)))
    )
    if abs(normalized) <= FLOAT_ABS_TOLERANCE:
        return 0.0, (1.0, 0.0, 0.0)
    return abs(normalized), (0.0, 0.0, 1.0 if normalized > 0.0 else -1.0)


def _render_directive(state: Mapping[str, Any], room_dir: Path) -> str:
    room_geometry = state.get("room_geometry")
    if not isinstance(room_geometry, Mapping):
        raise RecoveryError("Recovered state has no room geometry")
    room_sdf = _resolve_state_path(
        room_geometry.get("sdf_path"), room_dir, "Room geometry SDF"
    )
    lines = [
        "directives:",
        "- add_model:",
        "    name: room_geometry",
        f"    file: file://{room_sdf}",
        "- add_weld:",
        "    parent: world",
        "    child: room_geometry::room_geometry_body_link",
    ]
    objects = state.get("objects")
    if not isinstance(objects, Mapping):
        raise RecoveryError("Recovered state has no objects")
    for object_id, value in objects.items():
        if not isinstance(value, Mapping) or value.get("object_type") == "wall":
            continue
        sdf_value = value.get("sdf_path")
        if not sdf_value:
            continue
        sdf_path = _resolve_state_path(sdf_value, room_dir, f"{object_id} SDF")
        base_link, is_static = _sdf_model_info(sdf_path)
        transform = value.get("transform")
        if not isinstance(transform, Mapping):
            raise RecoveryError(f"Recovered object {object_id} has no transform")
        translation = transform.get("translation")
        if not isinstance(translation, list) or len(translation) != 3:
            raise RecoveryError(f"Recovered object {object_id} has malformed translation")
        tx, ty, tz = (
            _finite_number(component, f"{object_id}.translation") for component in translation
        )
        yaw = _yaw_from_quaternion(
            transform.get("rotation_wxyz"), f"{object_id}.rotation_wxyz"
        )
        angle, axis = _angle_axis_for_yaw(yaw)
        name_value = value.get("name")
        if not isinstance(name_value, str) or not name_value:
            raise RecoveryError(f"Recovered object {object_id} has no name")
        suffix = str(object_id).split("_")[-1][:8]
        model_name = f"{name_value.lower().replace(' ', '_')}_{suffix}"
        lines.extend(
            [
                "- add_model:",
                f"    name: {model_name}",
                f"    file: file://{sdf_path}",
            ]
        )
        if is_static:
            lines.extend(
                [
                    "    default_free_body_pose:",
                    f"      {base_link}:",
                    "        base_frame: world",
                    f"        translation: [{tx}, {ty}, {tz}]",
                    "        rotation: !AngleAxis",
                    f"          angle_deg: {angle}",
                    f"          axis: [{axis[0]}, {axis[1]}, {axis[2]}]",
                ]
            )
        else:
            lines.extend(
                [
                    "- add_weld:",
                    "    parent: world",
                    f"    child: {model_name}::{base_link}",
                    "    X_PC:",
                    f"      translation: [{tx}, {ty}, {tz}]",
                    "      rotation: !AngleAxis",
                    f"        angle_deg: {angle}",
                    f"        axis: [{axis[0]}, {axis[1]}, {axis[2]}]",
                ]
            )
    return "\n".join(lines) + "\n"


def _input_record(path: Path, label: str) -> dict[str, Any]:
    canonical = _require_regular_file(path, label)
    return {
        "path": str(canonical),
        "size_bytes": canonical.stat().st_size,
        "sha256": sha256_file(canonical),
    }


def recover_checkpoint(
    *,
    room_dir: Path,
    designer_db: Path,
    initial_scene_state: Path,
    asset_registry: Path,
    state_message_id: int,
    cutoff_message_id: int,
    first_observe_failure_message_id: int,
    expected_failure_substring: str = "OBJ has no normals",
    bound_inputs: Sequence[Path] = (),
) -> Path:
    room = _require_directory(room_dir, "Room directory")
    if room.name != f"room_{ROOM_ID}":
        raise RecoveryError(
            f"Recovery is restricted to room_{ROOM_ID}, got {room.name}"
        )
    database = _require_regular_file(designer_db, "designer.db")
    initial_path = _require_regular_file(initial_scene_state, "Initial scene state")
    registry_path = _require_regular_file(asset_registry, "Asset registry")
    if not (0 < state_message_id < cutoff_message_id < first_observe_failure_message_id):
        raise RecoveryError("Message IDs must satisfy state < cutoff < observe failure")
    if not expected_failure_substring:
        raise RecoveryError("Expected observe failure substring cannot be empty")

    final_dir = room / "scene_states" / CHECKPOINT_NAME
    staging_dir = room / "scene_states" / STAGING_NAME
    if final_dir.exists() or final_dir.is_symlink():
        raise RecoveryError(f"Refusing to replace existing checkpoint: {final_dir}")
    if staging_dir.exists() or staging_dir.is_symlink():
        raise RecoveryError(f"Stale recovery staging path requires inspection: {staging_dir}")

    database_before = _db_family_snapshot(database)
    initial_record = _input_record(initial_path, "Initial scene state")
    registry_record = _input_record(registry_path, "Asset registry")
    bound_records = [_input_record(path, "Bound immutable input") for path in bound_inputs]
    script_record = _input_record(Path(__file__), "Recovery script")

    initial = _read_json(initial_path, "initial scene state")
    registry = _read_json(registry_path, "asset registry")
    if not isinstance(initial, Mapping) or not isinstance(registry, Mapping):
        raise RecoveryError("Initial scene state and registry must be JSON objects")

    messages = _load_messages_read_only(database)
    by_id = _message_index(messages)
    calls, outputs = _pair_calls(messages)
    state_row = by_id.get(state_message_id)
    if state_row is None or state_row.document.get("type") != "function_call_output":
        raise RecoveryError("State message is not a function_call_output")
    state_call_id = state_row.document.get("call_id")
    state_call = calls.get(str(state_call_id))
    if (
        state_call is None
        or state_call.id + 1 != state_row.id
        or state_call.document.get("name") != "get_current_scene_state"
        or _parse_arguments(state_call, "get_current_scene_state") != {}
    ):
        raise RecoveryError("State output is not paired with an exact empty-argument state call")
    state_output = _parse_output_json(state_row, "get_current_scene_state")
    order, logical = _logical_anchor(state_output)
    initial_walls = _validate_initial_walls(initial, logical)

    nonwall_ids = set(order) - set(EXPECTED_WALL_IDS)
    placement_mapping, placement_evidence = _successful_prior_add_mapping(
        messages,
        calls,
        outputs,
        before_message_id=state_message_id,
        needed_ids=nonwall_ids,
    )
    for object_id, asset_id in placement_mapping.items():
        logical[object_id].asset_id = asset_id

    operations, _consumed = _replay_operations(
        messages=messages,
        calls=calls,
        outputs=outputs,
        logical=logical,
        state_message_id=state_message_id,
        cutoff_message_id=cutoff_message_id,
    )

    for operation in operations:
        if operation["name"] != "rescale_furniture_tool":
            continue
        object_id = str(operation["object_id"])
        asset_id = placement_mapping[object_id]
        template = registry.get(asset_id)
        if not isinstance(template, Mapping):
            raise RecoveryError(f"Rescaled object {object_id} has no registry template")
        registry_sdf = _resolve_registry_path(
            template.get("sdf_path"), f"{asset_id} rescale SDF"
        )
        result_asset = operation["result"].get("asset_id")
        if (
            registry_sdf is None
            or not isinstance(result_asset, str)
            or not result_asset
            or Path(result_asset).resolve() != registry_sdf
        ):
            raise RecoveryError(
                f"Archived rescale output is bound to a different SDF for {object_id}"
            )
        affected = operation["result"].get("affected_object_ids")
        assert isinstance(affected, list)  # Already validated by replay.
        if any(placement_mapping[str(candidate)] != asset_id for candidate in affected):
            raise RecoveryError(
                f"Archived rescale affected objects do not share asset {asset_id}"
            )

    failure_call, failure_output = _validate_first_observe_failure(
        messages=messages,
        calls=calls,
        failure_message_id=first_observe_failure_message_id,
        cutoff_message_id=cutoff_message_id,
        expected_substring=expected_failure_substring,
    )

    selected_session_ids = {
        state_call.session_id,
        state_row.session_id,
        failure_call.session_id,
        failure_output.session_id,
        *(row.session_id for row in messages if state_message_id < row.id <= cutoff_message_id),
    }
    if len(selected_session_ids) != 1:
        raise RecoveryError(
            f"Recovery evidence crosses designer sessions: {sorted(selected_session_ids)}"
        )

    (
        objects,
        artiverse_bindings,
        normalization_evidence,
        instance_assets,
    ) = _materialize_objects(
        order=order,
        logical=logical,
        initial_walls=initial_walls,
        registry=registry,
        placement_mapping=placement_mapping,
        room_dir=room,
    )
    state = {
        "room_geometry": copy.deepcopy(initial["room_geometry"]),
        "objects": objects,
        "text_description": initial["text_description"],
        "timestamp": _created_at_epoch(by_id[cutoff_message_id].created_at),
    }
    directive = _render_directive(state, room)

    database_after_parse = _db_family_snapshot(database)
    if database_after_parse != database_before:
        raise RecoveryError("designer.db main/WAL bytes changed during read-only recovery")
    if _input_record(initial_path, "Initial scene state") != initial_record:
        raise RecoveryError("Initial scene state changed during recovery")
    if _input_record(registry_path, "Asset registry") != registry_record:
        raise RecoveryError("Asset registry changed during recovery")
    current_bound = [_input_record(Path(record["path"]), "Bound immutable input") for record in bound_records]
    if current_bound != bound_records:
        raise RecoveryError("A bound immutable input changed during recovery")

    scene_states_dir = room / "scene_states"
    scene_states_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(exist_ok=False)
    state_path = staging_dir / "scene_state.json"
    directive_path = staging_dir / "scene.dmd.yaml"
    receipt_path = staging_dir / "recovery_receipt.json"
    _write_json_exclusive(state_path, state)
    _write_text_exclusive(directive_path, directive)

    message_records = [
        {
            "id": row.id,
            "session_id": row.session_id,
            "created_at": row.created_at,
            "message_data_sha256": hashlib.sha256(row.raw.encode("utf-8")).hexdigest(),
        }
        for row in messages
        if row.id
        in {
            state_call.id,
            state_row.id,
            failure_call.id,
            failure_output.id,
            *(
                message_id
                for operation in operations
                for message_id in (
                    operation["call_message_id"],
                    operation["output_message_id"],
                )
            ),
            *(
                message_id
                for placement in placement_evidence
                for message_id in (
                    placement["call_message_id"],
                    placement["output_message_id"],
                )
            ),
        }
    ]
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "recovery_kind": "reference_classroom_furniture_from_designer_db",
        "room_id": ROOM_ID,
        "checkpoint_name": CHECKPOINT_NAME,
        "source_inputs": {
            "designer_db_family": database_before,
            "initial_scene_state": initial_record,
            "asset_registry": registry_record,
            "bound_immutable_inputs": bound_records,
            "recovery_code": script_record,
        },
        "database_evidence": {
            "session_id": state_row.session_id,
            "anchor_call_message_id": state_call.id,
            "anchor_state_message_id": state_row.id,
            "cutoff_message_id": cutoff_message_id,
            "first_observe_failure_call_message_id": failure_call.id,
            "first_observe_failure_message_id": failure_output.id,
            "expected_failure_substring": expected_failure_substring,
            "selected_messages": message_records,
            "selected_messages_sha256": _json_sha256(message_records),
            "maximum_archived_message_id": messages[-1].id,
            "ignored_messages_after_failure": sum(
                row.id > failure_output.id for row in messages
            ),
        },
        "reconstruction": {
            "object_count": len(objects),
            "wall_count": len(EXPECTED_WALL_IDS),
            "student_desk_count": len(EXPECTED_DESK_IDS),
            "student_chair_count": len(EXPECTED_CHAIR_IDS),
            "object_ids": list(order),
            "instance_asset_mapping": instance_assets,
            "placement_evidence": placement_evidence,
            "successful_replayed_operations": operations,
            "artiverse_bindings": artiverse_bindings,
            "artiverse_visual_normalization": normalization_evidence,
        },
        "published_artifacts": {
            "scene_state": {
                "path": str(final_dir / "scene_state.json"),
                "size_bytes": state_path.stat().st_size,
                "sha256": sha256_file(state_path),
            },
            "scene_dmd": {
                "path": str(final_dir / "scene.dmd.yaml"),
                "size_bytes": directive_path.stat().st_size,
                "sha256": sha256_file(directive_path),
            },
        },
    }
    receipt["attestation"] = {
        "algorithm": "sha256",
        "sha256": _json_sha256(receipt),
    }
    _write_json_exclusive(receipt_path, receipt)

    if _db_family_snapshot(database) != database_before:
        raise RecoveryError("designer.db main/WAL bytes changed before publication")
    os.replace(staging_dir, final_dir)
    return final_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room-dir", type=Path, required=True)
    parser.add_argument("--designer-db", type=Path, required=True)
    parser.add_argument("--initial-scene-state", type=Path, required=True)
    parser.add_argument("--asset-registry", type=Path, required=True)
    parser.add_argument("--state-message-id", type=int, required=True)
    parser.add_argument("--cutoff-message-id", type=int, required=True)
    parser.add_argument(
        "--first-observe-failure-message-id", type=int, required=True
    )
    parser.add_argument(
        "--expected-failure-substring", default="OBJ has no normals"
    )
    parser.add_argument(
        "--bind-input",
        type=Path,
        action="append",
        default=[],
        help="Additional immutable prompt/specification/input file to bind in the receipt",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        checkpoint = recover_checkpoint(
            room_dir=args.room_dir,
            designer_db=args.designer_db,
            initial_scene_state=args.initial_scene_state,
            asset_registry=args.asset_registry,
            state_message_id=args.state_message_id,
            cutoff_message_id=args.cutoff_message_id,
            first_observe_failure_message_id=args.first_observe_failure_message_id,
            expected_failure_substring=args.expected_failure_substring,
            bound_inputs=args.bind_input,
        )
    except RecoveryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "pass",
                "checkpoint": str(checkpoint),
                "receipt": str(checkpoint / "recovery_receipt.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
