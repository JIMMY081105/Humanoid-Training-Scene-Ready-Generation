#!/usr/bin/env python3
"""Prove that the three required food-factory roles really articulate.

Semantic labels and ``is_articulated`` metadata are not motion evidence.  This
gate binds the final room states to regular, contained SDF asset trees, checks
role-specific joint types and finite limits, and then exercises every movable
joint in an isolated Drake ``MultibodyPlant``.  Verify-only mode repeats the
entire parse, hash, load, and kinematics exercise before accepting saved proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import time
import xml.etree.ElementTree as ET

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname

try:
    from .factory_room_contract import (
        ROLE_RULES as ARTICULATED_ROLE_RULES,
        PROFILE,
        collect_required_articulated_roles,
    )
except ImportError:  # Direct execution.
    from factory_room_contract import (  # type: ignore[no-redef]
        ROLE_RULES as ARTICULATED_ROLE_RULES,
        PROFILE,
        collect_required_articulated_roles,
    )


SCHEMA_VERSION = 1
VERIFICATION_SCHEMA_ID = "scenesmith_articulated_motion_verification_v1"
HASH_CHUNK_SIZE = 1024 * 1024
MINIMUM_LIMIT_SPAN = 1.0e-6
MINIMUM_TRANSFORM_DELTA = 1.0e-9
REQUIRED_ROOMS = (
    "cold_storage",
    "maintenance",
    "changing_room",
    "office_administration",
    "break_room",
)
ROLE_MOTION_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "cold_room_insulated_door": {
        "room_ids": ("cold_storage",),
        "joint_types": ("revolute", "prismatic"),
        "minimum_joint_count": 1,
    },
    "maintenance_tool_cabinet_or_changing_locker": {
        "room_ids": ("maintenance", "changing_room"),
        "joint_types": ("revolute", "prismatic"),
        "minimum_joint_count": 1,
    },
    "office_filing_cabinet_or_break_refrigerator": {
        "room_ids": ("office_administration", "break_room"),
        "joint_types": ("revolute", "prismatic"),
        "minimum_joint_count": 1,
    },
}

DrakeLoader = Callable[[Path, Sequence[Mapping[str, Any]]], Mapping[str, Any]]


class ArticulatedMotionError(RuntimeError):
    """Raised when an articulated-motion acceptance invariant is violated."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_regular_file_within(path: Path, root: Path, label: str) -> Path:
    """Resolve one file while rejecting escapes, symlinks, and junctions."""

    lexical_path = _lexical_absolute(path)
    lexical_root = _lexical_absolute(root)
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError as exc:
        raise ArticulatedMotionError(
            f"{label} escapes its required root {lexical_root}: {lexical_path}"
        ) from exc

    current = lexical_root
    if _is_link_or_junction(current):
        raise ArticulatedMotionError(f"{label} root is a symlink or junction: {current}")
    for component in relative.parts:
        current = current / component
        if _is_link_or_junction(current):
            raise ArticulatedMotionError(f"{label} uses a symlink or junction: {current}")

    try:
        resolved_root = lexical_root.resolve(strict=True)
        resolved_path = lexical_path.resolve(strict=True)
        resolved_path.relative_to(resolved_root)
        file_status = lexical_path.stat(follow_symlinks=False)
    except (OSError, ValueError) as exc:
        raise ArticulatedMotionError(f"{label} is missing or escapes its root: {path}") from exc
    if not stat.S_ISREG(file_status.st_mode):
        raise ArticulatedMotionError(f"{label} is not a regular file: {path}")
    return resolved_path


def _asset_tree_evidence(asset_root: Path, room_root: Path, label: str) -> dict[str, Any]:
    """Inventory and hash a complete regular-file asset tree without following links."""

    root = _lexical_absolute(asset_root)
    room = _lexical_absolute(room_root)
    try:
        root.relative_to(room)
        resolved_room = room.resolve(strict=True)
        resolved_root = root.resolve(strict=True)
        resolved_root.relative_to(resolved_room)
    except (OSError, ValueError) as exc:
        raise ArticulatedMotionError(f"{label} escapes the room tree: {root}") from exc
    if _is_link_or_junction(root):
        raise ArticulatedMotionError(f"{label} is a symlink or junction: {root}")

    pending = [resolved_root]
    paths: list[Path] = []
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise ArticulatedMotionError(f"Cannot scan {label} at {directory}: {exc}") from exc
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink() or _is_link_or_junction(path):
                raise ArticulatedMotionError(
                    f"{label} contains a symlink or junction: {path}"
                )
            try:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    paths.append(path)
                else:
                    raise ArticulatedMotionError(
                        f"{label} contains a special filesystem entry: {path}"
                    )
            except OSError as exc:
                raise ArticulatedMotionError(
                    f"Cannot inspect {label} entry {path}: {exc}"
                ) from exc

    paths.sort(key=lambda item: item.relative_to(resolved_root).as_posix())
    if not paths:
        raise ArticulatedMotionError(f"{label} is empty: {resolved_root}")
    digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for path in paths:
        relative = path.relative_to(resolved_root).as_posix()
        file_hash = _sha256_file(path)
        size = path.stat().st_size
        digest.update(len(relative.encode("utf-8")).to_bytes(8, "big"))
        digest.update(relative.encode("utf-8"))
        digest.update(bytes.fromhex(file_hash))
        files.append({"relative_path": relative, "sha256": file_hash, "size_bytes": size})
        total_bytes += size
    return {
        "path": str(resolved_root),
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(element: ET.Element, name: str) -> ET.Element | None:
    return next((item for item in element if _local_name(item.tag) == name), None)


def _text(element: ET.Element | None) -> str:
    return (element.text or "").strip() if element is not None else ""


def _finite_float(raw: str, label: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ArticulatedMotionError(f"{label} is not numeric: {raw!r}") from exc
    if not math.isfinite(value):
        raise ArticulatedMotionError(f"{label} is not finite: {raw!r}")
    return value


def _validate_resource_uris(document: ET.Element, sdf_path: Path, asset_root: Path) -> list[str]:
    resources: list[str] = []
    for ordinal, element in enumerate(document.iter(), start=1):
        if _local_name(element.tag) != "uri":
            continue
        raw = _text(element)
        if not raw or "\x00" in raw:
            raise ArticulatedMotionError(f"SDF resource URI #{ordinal} is empty or invalid")
        parsed = urlsplit(raw)
        windows_absolute = Path(raw).is_absolute() and len(parsed.scheme) == 1
        if parsed.query or parsed.fragment:
            raise ArticulatedMotionError(
                f"SDF resource URI #{ordinal} has a query or fragment: {raw}"
            )
        if parsed.scheme and parsed.scheme.lower() != "file" and not windows_absolute:
            raise ArticulatedMotionError(
                f"SDF resource URI #{ordinal} is not a local file: {raw}"
            )
        if parsed.scheme.lower() == "file":
            if parsed.netloc not in {"", "localhost"}:
                raise ArticulatedMotionError(
                    f"SDF resource URI #{ordinal} uses a remote authority: {raw}"
                )
            decoded = url2pathname(unquote(parsed.path))
        else:
            decoded = unquote(raw)
        candidate = Path(decoded)
        if not candidate.is_absolute():
            candidate = sdf_path.parent / candidate
        resolved = _require_regular_file_within(
            candidate, asset_root, f"SDF resource URI #{ordinal}"
        )
        resources.append(resolved.relative_to(asset_root.resolve()).as_posix())
    return sorted(resources)


def _parse_motion_joints(
    sdf_path: Path, role: str, requirement: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        document = ET.parse(sdf_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ArticulatedMotionError(f"Cannot parse SDF for {role}: {sdf_path}: {exc}") from exc

    expected_types = tuple(str(value) for value in requirement["joint_types"])
    movable: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for joint in (item for item in document.iter() if _local_name(item.tag) == "joint"):
        joint_type = str(joint.attrib.get("type", "")).strip().lower()
        if joint_type == "fixed":
            continue
        name = str(joint.attrib.get("name", "")).strip()
        if not name:
            raise ArticulatedMotionError(f"{role} contains a movable joint without a name")
        if name in seen_names:
            raise ArticulatedMotionError(f"{role} contains duplicate movable joint name {name!r}")
        seen_names.add(name)
        if joint_type not in expected_types:
            raise ArticulatedMotionError(
                f"{role} has wrong joint type {joint_type!r} for {name!r}; "
                f"expected only {expected_types!r} movable joints"
            )

        parent = _text(_child(joint, "parent"))
        child = _text(_child(joint, "child"))
        if not parent or not child or child == "world" or parent == child:
            raise ArticulatedMotionError(
                f"{role}/{name} has invalid parent/child links: {parent!r} -> {child!r}"
            )
        axis = _child(joint, "axis")
        axis_xyz = _text(_child(axis, "xyz")) if axis is not None else ""
        components = axis_xyz.split()
        if len(components) != 3:
            raise ArticulatedMotionError(f"{role}/{name} has no valid three-component axis")
        vector = [_finite_float(value, f"{role}/{name} axis") for value in components]
        if math.sqrt(sum(value * value for value in vector)) <= MINIMUM_LIMIT_SPAN:
            raise ArticulatedMotionError(f"{role}/{name} has a zero-length motion axis")

        limit = _child(axis, "limit") if axis is not None else None
        lower = _finite_float(_text(_child(limit, "lower")), f"{role}/{name} lower limit")
        upper = _finite_float(_text(_child(limit, "upper")), f"{role}/{name} upper limit")
        span = upper - lower
        if span <= MINIMUM_LIMIT_SPAN:
            raise ArticulatedMotionError(
                f"{role}/{name} has no finite positive joint range: [{lower}, {upper}]"
            )
        positions = [lower + 0.25 * span, lower + 0.75 * span]
        movable.append(
            {
                "joint_name": name,
                "joint_type": joint_type,
                "parent_link": parent,
                "child_link": child,
                "axis": vector,
                "limits": {"lower": lower, "upper": upper, "span": span},
                "tested_positions": positions,
            }
        )

    minimum = int(requirement["minimum_joint_count"])
    if len(movable) < minimum:
        raise ArticulatedMotionError(
            f"{role} has {len(movable)} valid {expected_types} joints; at least {minimum} required"
        )
    movable.sort(key=lambda item: item["joint_name"])
    return movable, _validate_resource_uris(document, sdf_path, sdf_path.parent)


def _matrix_delta(first: Any, second: Any) -> dict[str, float]:
    matrix_a = [[float(value) for value in row] for row in first]
    matrix_b = [[float(value) for value in row] for row in second]
    if len(matrix_a) != 4 or len(matrix_b) != 4 or any(
        len(row) != 4 for row in [*matrix_a, *matrix_b]
    ):
        raise ArticulatedMotionError("Drake returned a non-4x4 body transform")
    differences = [
        matrix_b[row][column] - matrix_a[row][column]
        for row in range(4)
        for column in range(4)
    ]
    if not all(math.isfinite(value) for value in differences):
        raise ArticulatedMotionError("Drake returned a non-finite body transform")
    translation = [matrix_b[row][3] - matrix_a[row][3] for row in range(3)]
    rotation = [
        matrix_b[row][column] - matrix_a[row][column]
        for row in range(3)
        for column in range(3)
    ]
    return {
        "max_abs_matrix_delta": max(abs(value) for value in differences),
        "translation_norm": math.sqrt(sum(value * value for value in translation)),
        "rotation_frobenius_norm": math.sqrt(sum(value * value for value in rotation)),
    }


def _exercise_with_pydrake(
    sdf_path: Path, joint_requests: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    """Load one asset in its own plant and exercise all requested joints."""

    try:
        from pydrake.all import (
            AddMultibodyPlantSceneGraph,
            DiagramBuilder,
            JointIndex,
            Parser,
            PrismaticJoint,
            RevoluteJoint,
        )
    except ImportError as exc:
        raise ArticulatedMotionError(
            "pydrake is required for articulated-motion acceptance"
        ) from exc

    builder = DiagramBuilder()
    plant, _scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    model_instances = list(parser.AddModels(str(sdf_path)))
    if not model_instances:
        raise ArticulatedMotionError(f"Drake loaded no models from {sdf_path}")
    plant.Finalize()
    _diagram = builder.Build()

    model_names = [plant.GetModelInstanceName(index) for index in model_instances]
    joints_by_name: dict[str, list[Any]] = {}
    for ordinal in range(plant.num_joints()):
        joint = plant.get_joint(JointIndex(ordinal))
        joints_by_name.setdefault(joint.name(), []).append(joint)

    exercises: list[dict[str, Any]] = []
    for request in joint_requests:
        name = str(request["joint_name"])
        candidates = joints_by_name.get(name, [])
        if len(candidates) != 1:
            raise ArticulatedMotionError(
                f"Drake resolved {len(candidates)} joints named {name!r} in {sdf_path}"
            )
        joint = candidates[0]
        expected_type = str(request["joint_type"])
        if isinstance(joint, RevoluteJoint):
            drake_type = "revolute"
            setter = joint.set_angle
        elif isinstance(joint, PrismaticJoint):
            drake_type = "prismatic"
            setter = joint.set_translation
        else:
            drake_type = type(joint).__name__
            setter = None
        if drake_type != expected_type or setter is None:
            raise ArticulatedMotionError(
                f"Drake joint {name!r} has type {drake_type!r}, expected {expected_type!r}"
            )

        drake_lower = float(joint.position_lower_limits()[0])
        drake_upper = float(joint.position_upper_limits()[0])
        expected_limits = request["limits"]
        if not all(math.isfinite(value) for value in (drake_lower, drake_upper)):
            raise ArticulatedMotionError(f"Drake joint {name!r} has non-finite limits")
        lower_matches = math.isclose(
            drake_lower, float(expected_limits["lower"]), abs_tol=1e-9
        )
        upper_matches = math.isclose(
            drake_upper, float(expected_limits["upper"]), abs_tol=1e-9
        )
        if not lower_matches or not upper_matches:
            raise ArticulatedMotionError(
                f"Drake joint limits for {name!r} differ from the parsed SDF"
            )

        context = plant.CreateDefaultContext()
        first_position, second_position = [float(value) for value in request["tested_positions"]]
        setter(context, first_position)
        first_pose = plant.EvalBodyPoseInWorld(context, joint.child_body()).GetAsMatrix4()
        setter(context, second_position)
        second_pose = plant.EvalBodyPoseInWorld(context, joint.child_body()).GetAsMatrix4()
        delta = _matrix_delta(first_pose, second_pose)
        exercises.append(
            {
                "joint_name": name,
                "joint_type": drake_type,
                "model_name": plant.GetModelInstanceName(joint.model_instance()),
                "parent_body_name": joint.parent_body().name(),
                "child_body_name": joint.child_body().name(),
                "limits": {"lower": drake_lower, "upper": drake_upper},
                "tested_positions": [first_position, second_position],
                "transform_delta": delta,
                "child_body_pose_changed": (
                    delta["max_abs_matrix_delta"] > MINIMUM_TRANSFORM_DELTA
                ),
            }
        )

    return {
        "status": "pass",
        "model_names": model_names,
        "num_model_instances": plant.num_model_instances(),
        "num_bodies": plant.num_bodies(),
        "num_joints": plant.num_joints(),
        "num_positions": plant.num_positions(),
        "joint_exercises": exercises,
    }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArticulatedMotionError(f"{label} is not numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ArticulatedMotionError(f"{label} is not finite")
    return normalized


def _bounded_count(value: Any, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArticulatedMotionError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return value


def _validated_motion_result(
    raw: Mapping[str, Any], requests: Sequence[Mapping[str, Any]], role: str
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or raw.get("status") != "pass":
        raise ArticulatedMotionError(f"Drake motion load failed for {role}: {raw!r}")
    model_names = raw.get("model_names")
    if (
        not isinstance(model_names, list)
        or not model_names
        or any(not isinstance(name, str) or not name.strip() for name in model_names)
    ):
        raise ArticulatedMotionError(f"Drake reported no valid model names for {role}")
    raw_exercises = raw.get("joint_exercises")
    if not isinstance(raw_exercises, list):
        raise ArticulatedMotionError(f"Drake reported no joint exercises for {role}")
    by_name: dict[str, Mapping[str, Any]] = {}
    for item in raw_exercises:
        if not isinstance(item, Mapping):
            raise ArticulatedMotionError(f"Drake returned a malformed joint exercise for {role}")
        name = str(item.get("joint_name", ""))
        if not name or name in by_name:
            raise ArticulatedMotionError(f"Drake returned duplicate/missing joint names for {role}")
        by_name[name] = item
    expected_names = {str(request["joint_name"]) for request in requests}
    if set(by_name) != expected_names:
        raise ArticulatedMotionError(
            f"Drake joint exercise set differs for {role}: "
            f"expected={sorted(expected_names)}, actual={sorted(by_name)}"
        )

    normalized: list[dict[str, Any]] = []
    for request in requests:
        name = str(request["joint_name"])
        item = by_name[name]
        if item.get("joint_type") != request["joint_type"]:
            raise ArticulatedMotionError(f"Drake returned the wrong joint type for {role}/{name}")
        for key in ("model_name", "parent_body_name", "child_body_name"):
            if not isinstance(item.get(key), str) or not str(item[key]).strip():
                raise ArticulatedMotionError(f"Drake omitted {key} for {role}/{name}")
        positions = item.get("tested_positions")
        if not isinstance(positions, list) or len(positions) != 2:
            raise ArticulatedMotionError(f"Drake omitted tested positions for {role}/{name}")
        normalized_positions = [
            _finite_number(value, f"{role}/{name} tested position") for value in positions
        ]
        expected_positions = [float(value) for value in request["tested_positions"]]
        if any(
            not math.isclose(actual, expected, abs_tol=1e-10)
            for actual, expected in zip(normalized_positions, expected_positions)
        ):
            raise ArticulatedMotionError(f"Drake tested stale positions for {role}/{name}")
        drake_limits = item.get("limits")
        if not isinstance(drake_limits, Mapping):
            raise ArticulatedMotionError(f"Drake omitted joint limits for {role}/{name}")
        normalized_limits = {
            bound: _finite_number(
                drake_limits.get(bound), f"{role}/{name} Drake {bound} limit"
            )
            for bound in ("lower", "upper")
        }
        for bound in ("lower", "upper"):
            if not math.isclose(
                normalized_limits[bound],
                float(request["limits"][bound]),
                abs_tol=1e-9,
            ):
                raise ArticulatedMotionError(
                    f"Drake {bound} limit differs from the parsed SDF for {role}/{name}"
                )
        delta = item.get("transform_delta")
        if not isinstance(delta, Mapping):
            raise ArticulatedMotionError(f"Drake omitted transform delta for {role}/{name}")
        normalized_delta = {
            key: _finite_number(delta.get(key), f"{role}/{name} {key}")
            for key in (
                "max_abs_matrix_delta",
                "translation_norm",
                "rotation_frobenius_norm",
            )
        }
        if (
            item.get("child_body_pose_changed") is not True
            or normalized_delta["max_abs_matrix_delta"] <= MINIMUM_TRANSFORM_DELTA
            or max(
                normalized_delta["translation_norm"],
                normalized_delta["rotation_frobenius_norm"],
            )
            <= MINIMUM_TRANSFORM_DELTA
        ):
            raise ArticulatedMotionError(
                f"Drake found no child-body motion for {role}/{name}"
            )
        normalized.append(
            {
                **request,
                "drake_model_name": str(item["model_name"]),
                "drake_parent_body_name": str(item["parent_body_name"]),
                "drake_child_body_name": str(item["child_body_name"]),
                "transform_delta": normalized_delta,
                "child_body_pose_changed": True,
            }
        )
    return {
        "status": "pass",
        "model_names": list(model_names),
        "num_model_instances": _bounded_count(
            raw.get("num_model_instances"), f"{role} Drake model-instance count", 1
        ),
        "num_bodies": _bounded_count(
            raw.get("num_bodies"), f"{role} Drake body count", 2
        ),
        "num_joints": _bounded_count(
            raw.get("num_joints"), f"{role} Drake joint count", len(normalized)
        ),
        "num_positions": _bounded_count(
            raw.get("num_positions"), f"{role} Drake position count", len(normalized)
        ),
        "joint_exercises": normalized,
    }


def _load_required_states(scene_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    evidence: dict[str, Any] = {}
    for room_id in REQUIRED_ROOMS:
        room_dir = scene_dir / f"room_{room_id}"
        state_candidate = room_dir / "scene_states" / "final_scene" / "scene_state.json"
        state_path = _require_regular_file_within(
            state_candidate, room_dir, f"Final room state for {room_id}"
        )
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArticulatedMotionError(
                f"Cannot read final room state for {room_id}: {exc}"
            ) from exc
        if not isinstance(state, dict):
            raise ArticulatedMotionError(f"Final room state for {room_id} is not an object")
        states[room_id] = state
        evidence[room_id] = {
            "path": str(state_path),
            "sha256": _sha256_file(state_path),
            "size_bytes": state_path.stat().st_size,
        }
    return states, evidence


def _evidence_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": result.get("schema_version"),
        "profile": result.get("profile"),
        "scene_dir": result.get("scene_dir"),
        "required_roles": result.get("required_roles"),
        "artiverse_roles": result.get("artiverse_roles"),
        "state_evidence": result.get("state_evidence"),
        "roles": result.get("roles"),
        "drake_motion_exercised": result.get("drake_motion_exercised"),
    }


def _attest(result: dict[str, Any]) -> dict[str, Any]:
    result.pop("attestation", None)
    result["attestation"] = {
        "algorithm": "sha256",
        "sha256": _canonical_sha256(result),
    }
    return result


def _attestation_failures(result: Any) -> list[str]:
    if not isinstance(result, dict):
        return ["saved articulated-motion proof is not a JSON object"]
    failures: list[str] = []
    if result.get("schema_version") != SCHEMA_VERSION:
        failures.append("saved articulated-motion schema_version is unsupported")
    if result.get("status") != "pass":
        failures.append(f"saved articulated-motion status is {result.get('status')!r}, not pass")
    attestation = result.get("attestation")
    if not isinstance(attestation, dict) or attestation.get("algorithm") != "sha256":
        failures.append("saved articulated-motion attestation is missing or malformed")
    elif not _is_sha256(attestation.get("sha256")):
        failures.append("saved articulated-motion attestation SHA-256 is malformed")
    else:
        unattested = dict(result)
        unattested.pop("attestation", None)
        if _canonical_sha256(unattested) != attestation["sha256"]:
            failures.append("saved articulated-motion attestation does not match its contents")
    evidence_hash = result.get("evidence_sha256")
    if not _is_sha256(evidence_hash) or evidence_hash != _canonical_sha256(
        _evidence_payload(result)
    ):
        failures.append("saved articulated-motion evidence digest is missing or stale")
    return failures


def evaluate(
    scene_dir: Path, *, drake_loader: DrakeLoader | None = None
) -> dict[str, Any]:
    scene_dir = scene_dir.resolve(strict=True)
    if set(ROLE_MOTION_REQUIREMENTS) != set(ARTICULATED_ROLE_RULES):
        raise ArticulatedMotionError(
            "Articulated-motion role table differs from the factory room contract"
        )
    states, state_evidence = _load_required_states(scene_dir)
    role_contract = collect_required_articulated_roles(
        states, require_runtime_provenance=True
    )
    if role_contract.get("status") != "pass":
        raise ArticulatedMotionError(
            "Required articulated-role contract failed: "
            + "; ".join(str(issue) for issue in role_contract.get("critical_issues", []))
        )
    artiverse_roles = sorted(set(role_contract.get("artiverse_roles", [])))
    if not artiverse_roles:
        raise ArticulatedMotionError("At least one required furniture role must use Artiverse")

    loader = drake_loader or _exercise_with_pydrake
    role_evidence: dict[str, Any] = {}
    for role in sorted(ROLE_MOTION_REQUIREMENTS):
        requirement = ROLE_MOTION_REQUIREMENTS[role]
        records = role_contract.get("roles", {}).get(role)
        if not isinstance(records, list) or len(records) != 1:
            raise ArticulatedMotionError(
            f"{role} must resolve to exactly one final role asset, found "
                f"{len(records) if isinstance(records, list) else 0}"
            )
        record = records[0]
        room_id = str(record.get("room_id", ""))
        if room_id not in requirement["room_ids"]:
            raise ArticulatedMotionError(
                f"{role} resolved to room {room_id!r}, expected one of {requirement['room_ids']!r}"
            )
        room_dir = scene_dir / f"room_{room_id}"
        raw_sdf = str(record.get("sdf_path", "")).strip()
        if not raw_sdf:
            raise ArticulatedMotionError(f"{role} has no runtime SDF path")
        sdf_candidate = Path(raw_sdf)
        if not sdf_candidate.is_absolute():
            sdf_candidate = room_dir / sdf_candidate
        sdf_path = _require_regular_file_within(
            sdf_candidate, room_dir, f"Runtime SDF for {role}"
        )
        tree = _asset_tree_evidence(
            sdf_path.parent, room_dir, f"Runtime asset tree for {role}"
        )
        requests, referenced_resources = _parse_motion_joints(
            sdf_path, role, requirement
        )
        raw_motion = loader(sdf_path, requests)
        motion = _validated_motion_result(raw_motion, requests, role)
        final_tree = _asset_tree_evidence(
            sdf_path.parent, room_dir, f"Runtime asset tree for {role} after Drake load"
        )
        if final_tree != tree:
            raise ArticulatedMotionError(
                f"Runtime asset tree for {role} changed during the Drake motion exercise"
            )
        role_evidence[role] = {
            "room_id": room_id,
            "object_id": str(record.get("object_id", "")),
            "articulated_id": str(record.get("articulated_id", "")),
            "articulated_source": str(record.get("articulated_source", "")),
            "asset_source": str(record.get("asset_source", "")),
            "state_sha256": state_evidence[room_id]["sha256"],
            "sdf": {
                "path": str(sdf_path),
                "sha256": _sha256_file(sdf_path),
                "size_bytes": sdf_path.stat().st_size,
                "referenced_resources": referenced_resources,
            },
            "resource_tree": tree,
            "motion_requirement": dict(requirement),
            "drake_motion": motion,
        }

    for room_id, state_record in state_evidence.items():
        current_state = _require_regular_file_within(
            Path(state_record["path"]),
            scene_dir / f"room_{room_id}",
            f"Final room state for {room_id} after Drake load",
        )
        if (
            _sha256_file(current_state) != state_record["sha256"]
            or current_state.stat().st_size != state_record["size_bytes"]
        ):
            raise ArticulatedMotionError(
                f"Final room state for {room_id} changed during the motion gate"
            )

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "profile": PROFILE,
        "scene_dir": str(scene_dir),
        "required_roles": sorted(ROLE_MOTION_REQUIREMENTS),
        "artiverse_roles": artiverse_roles,
        "state_evidence": state_evidence,
        "roles": role_evidence,
        "drake_motion_exercised": True,
        "critical_issues": [],
    }
    result["evidence_sha256"] = _canonical_sha256(_evidence_payload(result))
    return _attest(result)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run(
    scene_dir: Path,
    output: Path,
    *,
    drake_loader: DrakeLoader | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = evaluate(scene_dir, drake_loader=drake_loader)
    except Exception as exc:
        result = _attest(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "fail",
                "scene_dir": str(scene_dir.resolve()),
                "drake_motion_exercised": False,
                "critical_issues": [f"{type(exc).__name__}: {exc}"],
            }
        )
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    # elapsed time is evidence too; refresh the attestation after recording it.
    _attest(result)
    _write_json(output, result)
    return result


def verify_output(
    scene_dir: Path,
    output: Path,
    *,
    drake_loader: DrakeLoader | None = None,
    verification_output: Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    issues: list[str] = []
    source_path = output.resolve()
    if (
        verification_output is not None
        and verification_output.resolve() == source_path
    ):
        raise ArticulatedMotionError(
            "verification output must differ from the source motion-gate output"
        )
    source_sha256: str | None = None
    source_size_bytes: int | None = None
    try:
        source_sha256 = _sha256_file(source_path)
        source_size_bytes = source_path.stat().st_size
        saved = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        saved = None
        issues.append(f"Cannot read saved articulated-motion proof: {exc}")
    if saved is not None:
        issues.extend(_attestation_failures(saved))

    fresh: dict[str, Any] | None = None
    if not issues:
        try:
            fresh = evaluate(scene_dir, drake_loader=drake_loader)
        except Exception as exc:
            issues.append(
                f"Fresh articulated-motion exercise failed: {type(exc).__name__}: {exc}"
            )
    if fresh is not None and isinstance(saved, dict):
        if saved.get("scene_dir") != fresh.get("scene_dir"):
            issues.append("Saved articulated-motion proof belongs to a different scene directory")
        if saved.get("evidence_sha256") != fresh.get("evidence_sha256"):
            issues.append(
                "Current state, SDF/resource hashes, or repeated Drake motion differ "
                "from the saved articulated-motion evidence"
            )

    saved_attestation = saved.get("attestation") if isinstance(saved, dict) else None
    fresh_attestation = fresh.get("attestation") if fresh else None
    result: dict[str, Any] = {
        "verification_schema_id": VERIFICATION_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if not issues else "fail",
        "mode": "verify-only",
        "scene_dir": str(scene_dir.resolve()),
        "saved_output": str(source_path),
        "source_gate": {
            "path": str(source_path),
            "sha256": source_sha256,
            "size_bytes": source_size_bytes,
            "evidence_sha256": (
                saved.get("evidence_sha256") if isinstance(saved, dict) else None
            ),
            "attestation_sha256": (
                saved_attestation.get("sha256")
                if isinstance(saved_attestation, dict)
                else None
            ),
        },
        "saved_evidence_sha256": (
            saved.get("evidence_sha256") if isinstance(saved, dict) else None
        ),
        "fresh_evidence_sha256": fresh.get("evidence_sha256") if fresh else None,
        "recomputed_gate": {
            "result": fresh,
            "status": fresh.get("status") if fresh else None,
            "evidence_sha256": fresh.get("evidence_sha256") if fresh else None,
            "attestation": fresh_attestation,
            "result_sha256": _canonical_sha256(fresh) if fresh else None,
            "role_count": (
                len(fresh.get("roles", {}))
                if fresh and isinstance(fresh.get("roles"), dict)
                else 0
            ),
        },
        "drake_motion_repeated": fresh is not None,
        "critical_issues": issues,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    _attest(result)
    if verification_output is not None:
        _write_json(verification_output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Rehash all bound artifacts and repeat every Drake motion exercise.",
    )
    parser.add_argument(
        "--verification-output",
        type=Path,
        help=(
            "Atomically save the hash-bound repeat-execution receipt; valid only "
            "with --verify-only."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verification_output is not None and not args.verify_only:
        build_parser().error("--verification-output requires --verify-only")
    result = (
        verify_output(
            args.scene_dir,
            args.output,
            verification_output=args.verification_output,
        )
        if args.verify_only
        else run(args.scene_dir, args.output)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
