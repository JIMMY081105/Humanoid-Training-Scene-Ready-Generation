"""
Stateful manipuland agent with planner/designer/critic workflow.

This module implements manipuland placement using persistent agents that work
per-furniture, with fresh contexts for each furniture surface to bound token usage.
"""

import copy
import hashlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import time
import uuid
import xml.etree.ElementTree as ET

from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from agents import Agent, FunctionTool, Runner, RunResult, custom_span
from omegaconf import DictConfig, OmegaConf
import yaml

from scenesmith.agent_utils.base_stateful_agent import (
    BaseStatefulAgent,
    log_agent_usage,
)
from scenesmith.agent_utils.asset_registry import AssetRegistry
from scenesmith.agent_utils.physical_feasibility import (
    apply_per_furniture_postprocessing,
)
from scenesmith.agent_utils.placement_noise import PlacementNoiseMode
from scenesmith.agent_utils.rendering_manager import RenderingManager
from scenesmith.agent_utils.room import (
    AgentType,
    ObjectType,
    RoomScene,
    SupportSurface,
    UniqueID,
    extract_and_propagate_support_surfaces,
)
from scenesmith.agent_utils.scene_analyzer import FurnitureSelection, SceneAnalyzer
from scenesmith.agent_utils.scoring import (
    ManipulandCritiqueWithScores,
    log_agent_response,
)
from scenesmith.agent_utils.support_surface_extraction import (
    SupportSurfaceExtractionConfig,
)
from scenesmith.agent_utils.workflow_tools import WorkflowTools
from scenesmith.manipuland_agents.base_manipuland_agent import BaseManipulandAgent
from scenesmith.manipuland_agents.tools.manipuland_tools import ManipulandTools
from scenesmith.manipuland_agents.tools.vision_tools import ManipulandVisionTools
from scenesmith.prompts.registry import ManipulandAgentPrompts
from scenesmith.utils.logging import BaseLogger

console_logger = logging.getLogger(__name__)

_FURNITURE_CHECKPOINT_SCHEMA = 3
_FURNITURE_PLAN_SCHEMA = 1
_LEGACY_RECOVERY_SCHEMA = 3
_SCOPE_PREDECESSOR_RUNTIME_SHA256 = (
    "3b50f11db94ba3824d18bca034f79430e81a0a69419d9261b2575f6b6da2778c"
)
_SCOPE_PREDECESSOR_OUTPUT_SCENE_CONTENT_SHA256 = (
    "196981bb84396f6c1e084a0c32824612d6253e12d7f7a0c151eba1d9f0054837"
)
_SCOPE_PREDECESSOR_ROUNDTRIP_CONTENT_SHA256 = (
    "ec5d204593c23d4fc4aba56dd89e9e14367c4fe80f975c3b84cc1f4089ed1372"
)
_SCOPE_PREDECESSOR_RECEIPT_SHA256 = (
    "da8140238d59b2fc0abc95c11512839ff217475620fc321e938c5257651e91ec"
)
_SCOPE_PREDECESSOR_SCENE_STATE_SHA256 = (
    "9ab91d19053626dc8d2bad5485d32babd1a7992d4d6ab66e63ec99fff3b9db77"
)
_SCOPE_PREDECESSOR_DRAKE_DIRECTIVE_SHA256 = (
    "94784fa816a06507d7b2de2d4da06bf7f8ab2a1df95ff15cc7d187179d3e8726"
)
_SCOPE_PREDECESSOR_INPUT_SCENE_CONTENT_SHA256 = (
    "4b918d3e8781d8ea409a3900623ad25ab59b48a48ce64c9208b624d4bc417559"
)
_SCOPE_PREDECESSOR_PLAN_SHA256 = (
    "33906752fe705460f933a80d181b53ee0fe7eba117ff5c7ea929a99cbbfe4805"
)
# This is the one durable Room 1 filing-cabinet receipt created immediately before
# the optional-no-support-surface policy was corrected.  It may be restored only
# when the receipt bytes, identity, and both scene-content hashes are exact.
_FILING_CABINET_RUNTIME_PREDECESSOR_SHA256 = (
    "ede55b31fd4d7f94d16b3c8a6891d6fa1296ae4dc34556983ee02b7a2f655f16"
)
_FILING_CABINET_PREDECESSOR_INPUT_SCENE_CONTENT_SHA256 = (
    "ec5d204593c23d4fc4aba56dd89e9e14367c4fe80f975c3b84cc1f4089ed1372"
)
_FILING_CABINET_PREDECESSOR_OUTPUT_SCENE_CONTENT_SHA256 = (
    "83277726010ad845b8c8400cf5f4ba6aeac3ddbbcf511f982ad34aedd3bc145a"
)
_FILING_CABINET_PREDECESSOR_RECEIPT_SHA256 = (
    "ce68283accd5a7c924a0bc99565747c36cf3f0fd7261e61f7f6fea0dfe0da8a4"
)
_SAFE_CHECKPOINT_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_ACCEPTED_IMAGE_NAMES = frozenset(
    {"0_side.png", "0_top.png", "1_side.png", "2_side.png", "3_side.png"}
)
_SCORE_CATEGORIES = (
    "Realism",
    "Functionality",
    "Layout",
    "Holistic Completeness",
    "Prompt Following",
)
_TRANSFORM_TOLERANCE = 1e-9


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


@contextmanager
def _exclusive_checkpoint_lock(scene_states_dir: Path):
    scene_states_dir.mkdir(parents=True, exist_ok=True)
    lock_path = scene_states_dir / ".manipuland_checkpoint.lock"
    if lock_path.is_symlink():
        raise RuntimeError(f"Manipuland checkpoint lock is a symlink: {lock_path}")
    with lock_path.open("a+b") as lock_file:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
                os.fsync(lock_file.fileno())
            lock_file.seek(0)
            while True:
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.01)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_json_fsynced(path: Path, document: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json_fsynced(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        _write_json_fsynced(temporary, document)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size <= 0
        or path.stat().st_nlink != 1
    ):
        raise RuntimeError(f"Checkpoint evidence is missing or not regular: {path}")
    return {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _valid_file_record(record: Any, *, include_path: bool = False) -> bool:
    expected = {"name", "size_bytes", "sha256"}
    if include_path:
        expected.add("path")
    if not isinstance(record, dict) or set(record) != expected:
        return False
    if (
        not isinstance(record["name"], str)
        or not record["name"]
        or PurePosixPath(record["name"]).name != record["name"]
        or isinstance(record["size_bytes"], bool)
        or not isinstance(record["size_bytes"], int)
        or record["size_bytes"] <= 0
        or not isinstance(record["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
    ):
        return False
    if include_path:
        raw_path = record["path"]
        relative = PurePosixPath(raw_path) if isinstance(raw_path, str) else None
        if (
            relative is None
            or relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative.as_posix() != raw_path
            or relative.name != record["name"]
        ):
            return False
    return True


def _canonical_relative_path(raw: Any, *, label: str) -> PurePosixPath:
    if (
        not isinstance(raw, str)
        or not raw
        or raw != raw.strip()
        or "\\" in raw
        or "%" in raw
    ):
        raise RuntimeError(f"{label} is not canonical room-relative")
    relative = PurePosixPath(raw)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != raw
    ):
        raise RuntimeError(f"{label} is not canonical room-relative")
    return relative


def _validate_legacy_recovery_manifest_document(
    manifest: Any, *, expected_room_id: str
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "status",
        "purpose",
        "room_id",
        "furniture_index",
        "furniture_id",
        "input_scene_content_hash",
        "output_scene_content_hash",
        "input_scene_state_path",
        "input_scene_state",
        "render_directory",
        "scene_state",
        "drake_directive",
        "accepted_evidence",
        "added_manipuland_ids",
        "added_asset_files",
        "referenced_asset_object_ids",
        "referenced_asset_files",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_keys
        or manifest["schema_version"] != _LEGACY_RECOVERY_SCHEMA
        or manifest["status"] != "authorized"
        or manifest["purpose"] != "one_time_legacy_manipuland_recovery"
        or manifest["room_id"] != expected_room_id
        or isinstance(manifest["furniture_index"], bool)
        or not isinstance(manifest["furniture_index"], int)
        or manifest["furniture_index"] < 0
        or not isinstance(manifest["furniture_id"], str)
        or not _SAFE_CHECKPOINT_ID.fullmatch(manifest["furniture_id"])
        or not isinstance(manifest["input_scene_content_hash"], str)
        or re.fullmatch(r"[0-9a-f]{64}", manifest["input_scene_content_hash"])
        is None
        or not isinstance(manifest["output_scene_content_hash"], str)
        or re.fullmatch(r"[0-9a-f]{64}", manifest["output_scene_content_hash"])
        is None
        or not _valid_file_record(manifest["input_scene_state"])
        or not _valid_file_record(manifest["scene_state"])
        or not _valid_file_record(manifest["drake_directive"])
        or not isinstance(manifest["accepted_evidence"], dict)
    ):
        raise RuntimeError("Legacy recovery manifest has a non-exact schema")
    input_relative = _canonical_relative_path(
        manifest["input_scene_state_path"], label="legacy input scene-state path"
    )
    _canonical_relative_path(
        manifest["render_directory"], label="legacy render directory"
    )
    if input_relative.name != manifest["input_scene_state"]["name"]:
        raise RuntimeError("Legacy recovery input scene-state record/path mismatch")
    added_ids = manifest["added_manipuland_ids"]
    referenced_ids = manifest["referenced_asset_object_ids"]
    if (
        not isinstance(added_ids, list)
        or not added_ids
        or added_ids != sorted(set(added_ids))
        or not all(
            isinstance(value, str) and _SAFE_CHECKPOINT_ID.fullmatch(value)
            for value in added_ids
        )
        or not isinstance(referenced_ids, list)
        or referenced_ids != sorted(set(referenced_ids))
        or set(referenced_ids)
        != set(added_ids) | {manifest["furniture_id"]}
    ):
        raise RuntimeError("Legacy recovery manifest object inventory is malformed")
    for key in ("added_asset_files", "referenced_asset_files"):
        records = manifest[key]
        if (
            not isinstance(records, list)
            or not records
            or not all(_valid_file_record(record, include_path=True) for record in records)
            or [record["path"] for record in records]
            != sorted({record["path"] for record in records})
        ):
            raise RuntimeError("Legacy recovery manifest asset inventory is malformed")
    return manifest


def _regular_file_without_symlink_components(path: Path, *, label: str) -> Path:
    lexical = path.expanduser()
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"{label} contains a symlink component: {current}")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"{label} is missing: {lexical}") from exc
    if (
        not resolved.is_file()
        or resolved.stat().st_size <= 0
        or resolved.stat().st_nlink != 1
    ):
        raise RuntimeError(f"{label} is not a nonempty regular file: {resolved}")
    return resolved


def _regular_directory_without_symlink_components(
    path: Path,
    *,
    label: str,
    required_root: Path | None = None,
) -> Path:
    lexical = path.expanduser()
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"{label} contains a symlink component: {current}")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"{label} is missing: {lexical}") from exc
    if not resolved.is_dir():
        raise RuntimeError(f"{label} is not a regular directory: {resolved}")
    if required_root is not None and not resolved.is_relative_to(
        required_root.resolve(strict=True)
    ):
        raise RuntimeError(f"{label} escapes its required root: {resolved}")
    return resolved


def _room_asset_file_record(
    raw_path: Any,
    *,
    room_root: Path,
    label: str,
    allow_absolute_within_room: bool = True,
    scene_floor_root: Path | None = None,
    allow_scene_floor_absolute: bool = False,
    allow_scene_floor_logical: bool = False,
    project_materials_root: Path | None = None,
    allow_project_materials_logical: bool = False,
    scene_room_geometry_root: Path | None = None,
    allow_scene_room_geometry_absolute: bool = False,
    allow_scene_room_geometry_logical: bool = False,
) -> tuple[Path, dict[str, Any]]:
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or raw_path != raw_path.strip()
        or "\\" in raw_path
        or "%" in raw_path
    ):
        raise RuntimeError(f"{label} path is not canonical: {raw_path!r}")
    pure = PurePosixPath(raw_path)
    if (
        not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != raw_path
    ):
        raise RuntimeError(f"{label} path is not canonical: {raw_path!r}")

    room_authority_root = room_root.resolve(strict=True)
    authority_root = room_authority_root
    record_prefix: PurePosixPath | None = None
    if pure.is_absolute():
        if not allow_absolute_within_room:
            raise RuntimeError(f"{label} path must be room-relative: {raw_path!r}")
        lexical = Path(raw_path)
        try:
            relative_native = lexical.relative_to(room_authority_root)
        except ValueError as exc:
            if allow_scene_floor_absolute and scene_floor_root is not None:
                authority_root = scene_floor_root.resolve(strict=True)
                try:
                    relative_native = lexical.relative_to(authority_root)
                except ValueError as floor_exc:
                    raise RuntimeError(
                        f"{label} absolute path escapes the exact scene-floor root: "
                        f"{raw_path}"
                    ) from floor_exc
                record_prefix = PurePosixPath("_scene_floor_plan")
            elif (
                allow_scene_room_geometry_absolute
                and scene_room_geometry_root is not None
            ):
                authority_root = scene_room_geometry_root.resolve(strict=True)
                try:
                    relative_native = lexical.relative_to(authority_root)
                except ValueError as geometry_exc:
                    raise RuntimeError(
                        f"{label} absolute path escapes the exact scene-room-geometry "
                        f"root: {raw_path}"
                    ) from geometry_exc
                record_prefix = PurePosixPath("_scene_room_geometry")
            else:
                raise RuntimeError(
                    f"{label} absolute path escapes the room: {raw_path}"
                ) from exc
        relative = PurePosixPath(relative_native.as_posix())
    else:
        if (
            allow_scene_floor_logical
            and scene_floor_root is not None
            and pure.parts[0] == "_scene_floor_plan"
        ):
            if len(pure.parts) == 1:
                raise RuntimeError(f"{label} scene-floor path is empty")
            authority_root = scene_floor_root.resolve(strict=True)
            relative = PurePosixPath(*pure.parts[1:])
            record_prefix = PurePosixPath("_scene_floor_plan")
        elif (
            allow_project_materials_logical
            and project_materials_root is not None
            and pure.parts[0] == "_project_materials"
        ):
            if len(pure.parts) == 1:
                raise RuntimeError(f"{label} project-materials path is empty")
            authority_root = project_materials_root.resolve(strict=True)
            relative = PurePosixPath(*pure.parts[1:])
            record_prefix = PurePosixPath("_project_materials")
        elif (
            allow_scene_room_geometry_logical
            and scene_room_geometry_root is not None
            and pure.parts[0] == "_scene_room_geometry"
        ):
            if len(pure.parts) == 1:
                raise RuntimeError(f"{label} scene-room-geometry path is empty")
            authority_root = scene_room_geometry_root.resolve(strict=True)
            relative = PurePosixPath(*pure.parts[1:])
            record_prefix = PurePosixPath("_scene_room_geometry")
        else:
            relative = pure

    current = authority_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"{label} path contains a symlink: {current}")
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"{label} file is missing: {current}") from exc
    if not resolved.is_relative_to(authority_root):
        raise RuntimeError(f"{label} file escapes its exact asset root: {resolved}")
    if resolved.stat().st_nlink != 1:
        raise RuntimeError(f"{label} file is hard-linked: {resolved}")
    record = _file_record(resolved)
    record["path"] = (
        (record_prefix / relative).as_posix()
        if record_prefix is not None
        else relative.as_posix()
    )
    return resolved, record


def _exact_scene_floor_asset_root(
    floor: Any,
    *,
    room_root: Path,
) -> Path:
    """Derive the sole scene-level authority granted to the room's real floor."""

    root = room_root.resolve(strict=True)
    if not root.name.startswith("room_"):
        raise RuntimeError("Scene-floor authority requires a canonical room directory")
    room_id = root.name.removeprefix("room_")
    if (
        not room_id
        or _SAFE_CHECKPOINT_ID.fullmatch(room_id) is None
        or not isinstance(floor, dict)
        or floor.get("object_id") != f"floor_{room_id}"
        or floor.get("object_type") != "floor"
        or floor.get("immutable") is not True
    ):
        raise RuntimeError("Scene-floor authority does not match the exact room floor")
    scene_root = root.parent
    expected = scene_root / "floor_plans" / room_id
    return _regular_directory_without_symlink_components(
        expected,
        label="exact scene-floor asset root",
        required_root=scene_root,
    )


def _exact_scene_room_geometry_asset_root(
    floor: Any,
    *,
    room_root: Path,
) -> Path:
    """Derive the exact scene-level room-geometry directory for this floor."""

    # Reuse the exact room/floor identity proof before granting a second role.
    _exact_scene_floor_asset_root(floor, room_root=room_root)
    root = room_root.resolve(strict=True)
    scene_root = root.parent
    expected = scene_root / "room_geometry"
    return _regular_directory_without_symlink_components(
        expected,
        label="exact scene-room-geometry asset root",
        required_root=scene_root,
    )


def _exact_project_materials_root(*, room_root: Path) -> Path:
    """Derive the immutable shared-material root from the canonical output layout."""

    root = room_root.resolve(strict=True)
    scene_root = root.parent
    if len(scene_root.parents) < 4 or scene_root.parents[2].name != "outputs":
        raise RuntimeError("Scene-floor materials lack a canonical outputs layout")
    project_root = scene_root.parents[3]
    expected = project_root / "materials"
    return _regular_directory_without_symlink_components(
        expected,
        label="exact project materials root",
        required_root=project_root,
    )


def _scene_floor_material_dependency_path(
    raw_uri: Any,
    *,
    owner_path: Path,
    materials_root: Path,
    label: str,
) -> PurePosixPath:
    """Resolve an upward floor texture URI into the typed materials namespace."""

    if (
        not isinstance(raw_uri, str)
        or not raw_uri
        or raw_uri != raw_uri.strip()
        or "\\" in raw_uri
        or "%" in raw_uri
    ):
        raise RuntimeError(f"{label} URI is not canonical: {raw_uri!r}")
    parsed = urlsplit(raw_uri)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise RuntimeError(f"{label} URI is not a local relative file: {raw_uri!r}")
    relative = PurePosixPath(parsed.path)
    leading_parents = 0
    for part in relative.parts:
        if part == "..":
            leading_parents += 1
        else:
            break
    remaining = relative.parts[leading_parents:]
    if (
        relative.is_absolute()
        or leading_parents == 0
        or not remaining
        or any(part in {"", ".", ".."} for part in remaining)
        or relative.as_posix() != parsed.path
        or relative.suffix.lower()
        not in {".png", ".jpg", ".jpeg", ".webp", ".ktx2"}
    ):
        raise RuntimeError(f"{label} URI is unsafe or unsupported: {raw_uri!r}")
    current = owner_path.parent
    for part in relative.parts:
        current = current.parent if part == ".." else current / part
        if current.is_symlink():
            raise RuntimeError(f"{label} URI contains a symlink: {current}")
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"{label} dependency is missing: {current}") from exc
    material_root = materials_root.resolve(strict=True)
    if not resolved.is_relative_to(material_root):
        raise RuntimeError(f"{label} URI escapes exact project materials: {raw_uri!r}")
    material_relative = resolved.relative_to(material_root)
    return PurePosixPath("_project_materials") / PurePosixPath(
        material_relative.as_posix()
    )


def _scene_room_geometry_dependency_path(
    raw_uri: Any,
    *,
    owner_path: Path,
    floor_plan_root: Path,
    label: str,
) -> PurePosixPath:
    """Resolve room-geometry mesh URIs only into this room's floor-plan tree."""

    if (
        not isinstance(raw_uri, str)
        or not raw_uri
        or raw_uri != raw_uri.strip()
        or "\\" in raw_uri
        or "%" in raw_uri
    ):
        raise RuntimeError(f"{label} URI is not canonical: {raw_uri!r}")
    parsed = urlsplit(raw_uri)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise RuntimeError(f"{label} URI is not a local relative file: {raw_uri!r}")
    relative = PurePosixPath(parsed.path)
    leading_parents = 0
    for part in relative.parts:
        if part == "..":
            leading_parents += 1
        else:
            break
    remaining = relative.parts[leading_parents:]
    if (
        relative.is_absolute()
        or leading_parents == 0
        or not remaining
        or any(part in {"", ".", ".."} for part in remaining)
        or relative.as_posix() != parsed.path
        or relative.suffix.lower() not in {".gltf", ".obj", ".stl", ".ply"}
    ):
        raise RuntimeError(f"{label} URI is unsafe or unsupported: {raw_uri!r}")
    current = owner_path.parent
    for part in relative.parts:
        current = current.parent if part == ".." else current / part
        if current.is_symlink():
            raise RuntimeError(f"{label} URI contains a symlink: {current}")
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"{label} dependency is missing: {current}") from exc
    authority_root = floor_plan_root.resolve(strict=True)
    if not resolved.is_relative_to(authority_root):
        raise RuntimeError(
            f"{label} URI escapes exact current-room floor plan: {raw_uri!r}"
        )
    plan_relative = resolved.relative_to(authority_root)
    return PurePosixPath("_scene_floor_plan") / PurePosixPath(
        plan_relative.as_posix()
    )


def _dependency_relative_path(
    raw_uri: Any,
    *,
    owner_relative: PurePosixPath,
    label: str,
    allowed_suffixes: frozenset[str],
) -> PurePosixPath:
    if (
        not isinstance(raw_uri, str)
        or not raw_uri
        or raw_uri != raw_uri.strip()
        or "\\" in raw_uri
        or "%" in raw_uri
    ):
        raise RuntimeError(f"{label} URI is not canonical: {raw_uri!r}")
    parsed = urlsplit(raw_uri)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise RuntimeError(f"{label} URI is not a local relative file: {raw_uri!r}")
    path = parsed.path[2:] if parsed.path.startswith("./") else parsed.path
    relative = PurePosixPath(path)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != path
        or relative.suffix.lower() not in allowed_suffixes
    ):
        raise RuntimeError(f"{label} URI is unsafe or unsupported: {raw_uri!r}")
    return owner_relative.parent / relative


def _referenced_room_asset_records(
    scene_state: dict[str, Any],
    *,
    room_root: Path,
    object_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Bind direct, composite-member, SDF, and glTF files for selected objects."""

    objects = scene_state.get("objects")
    if not isinstance(objects, dict):
        raise RuntimeError("Scene asset inventory has no exact object mapping")
    selectable: dict[str, Any] = dict(objects)
    room_geometry = scene_state.get("room_geometry")
    floor = room_geometry.get("floor") if isinstance(room_geometry, dict) else None
    floor_id: str | None = None
    if isinstance(floor, dict) and isinstance(floor.get("object_id"), str):
        floor_id = floor["object_id"]
        if floor_id in selectable and selectable[floor_id] != floor:
            raise RuntimeError("Floor target aliases a different scene object")
        selectable[floor_id] = floor
    selected_ids = set(selectable) if object_ids is None else set(object_ids)
    if not selected_ids or not selected_ids.issubset(selectable):
        raise RuntimeError("Scene asset inventory object selection is invalid")

    pending: list[tuple[str, str, str]] = []

    def add_direct_path(
        raw: Any, *, key: str, label: str, authority: str = "room"
    ) -> None:
        if not isinstance(raw, str):
            raise RuntimeError(f"{label} is not a path string")
        suffix = PurePosixPath(raw).suffix.lower()
        allowed = (
            frozenset({".gltf", ".obj", ".stl", ".ply"})
            if key == "geometry_path"
            else frozenset({".sdf"})
        )
        if suffix not in allowed:
            raise RuntimeError(
                f"{label} has an unsupported direct asset suffix: {suffix!r}"
            )
        pending.append((raw, label, authority))

    for object_id in sorted(selected_ids):
        item = selectable[object_id]
        if not isinstance(item, dict):
            raise RuntimeError(f"Scene object {object_id} is malformed")
        for key in ("geometry_path", "sdf_path"):
            if item.get(key) is not None:
                add_direct_path(
                    item[key],
                    key=key,
                    label=f"{object_id} {key}",
                    authority="floor_plan" if object_id == floor_id else "room",
                )
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            continue

        def add_composite_asset(record: Any, *, label: str) -> None:
            if not isinstance(record, dict):
                raise RuntimeError(f"Scene object {object_id} {label} is malformed")
            for key in ("geometry_path", "sdf_path"):
                if record.get(key) is None:
                    raise RuntimeError(
                        f"Scene object {object_id} {label} lacks {key}"
                    )
                add_direct_path(
                    record[key],
                    key=key,
                    label=f"{object_id} {label} {key}",
                )

        members = metadata.get("member_assets")
        if members is not None:
            if not isinstance(members, list) or not members:
                raise RuntimeError(f"Scene object {object_id} member assets are malformed")
            for index, member in enumerate(members):
                add_composite_asset(member, label=f"member {index}")

        container = metadata.get("container_asset")
        if container is not None:
            add_composite_asset(container, label="container asset")
        fill_assets = metadata.get("fill_assets")
        if fill_assets is not None:
            if not isinstance(fill_assets, list) or not fill_assets:
                raise RuntimeError(f"Scene object {object_id} fill assets are malformed")
            for index, fill_asset in enumerate(fill_assets):
                add_composite_asset(fill_asset, label=f"fill asset {index}")

    if floor_id in selected_ids and isinstance(room_geometry, dict):
        room_sdf = room_geometry.get("sdf_path")
        if room_sdf is not None:
            add_direct_path(
                room_sdf,
                key="sdf_path",
                label=f"{floor_id} room_geometry sdf_path",
                authority="room_geometry",
            )

    floor_authority_root: Path | None = None
    room_geometry_authority_root: Path | None = None
    materials_authority_root: Path | None = None
    room_authority_root = room_root.resolve(strict=True)
    if floor_id in selected_ids:
        for raw, _label, authority in pending:
            if (
                authority in {"floor_plan", "room_geometry"}
                and PurePosixPath(raw).is_absolute()
                and not Path(raw).is_relative_to(room_authority_root)
            ):
                floor_authority_root = _exact_scene_floor_asset_root(
                    floor, room_root=room_root
                )
                room_geometry_authority_root = (
                    _exact_scene_room_geometry_asset_root(
                        floor, room_root=room_root
                    )
                )
                materials_authority_root = _exact_project_materials_root(
                    room_root=room_root
                )
                break
        if floor_authority_root is not None:
            exact_floor_geometry = floor_authority_root / "floors" / "floor.gltf"
            geometry_path = floor.get("geometry_path") if isinstance(floor, dict) else None
            if (
                not isinstance(geometry_path, str)
                or Path(geometry_path) != exact_floor_geometry
            ):
                raise RuntimeError(
                    "Scene-floor geometry is not the exact current-room floor.gltf"
                )
            room_sdf = room_geometry.get("sdf_path")
            exact_room_sdf = (
                room_geometry_authority_root / f"room_geometry_{room_root.name.removeprefix('room_')}.sdf"
                if room_geometry_authority_root is not None
                else None
            )
            if not isinstance(room_sdf, str) or Path(room_sdf) != exact_room_sdf:
                raise RuntimeError(
                    "Scene room geometry is not the exact current-room SDF"
                )

    records: dict[str, dict[str, Any]] = {}
    resolved_paths: dict[str, Path] = {}

    def add_path(
        raw: Any,
        label: str,
        *,
        allow_absolute: bool = True,
        allow_floor_absolute: bool = False,
        allow_floor_logical: bool = False,
        allow_materials_logical: bool = False,
        allow_room_geometry_absolute: bool = False,
        allow_room_geometry_logical: bool = False,
    ) -> None:
        resolved, record = _room_asset_file_record(
            raw,
            room_root=room_root,
            label=label,
            allow_absolute_within_room=allow_absolute,
            scene_floor_root=floor_authority_root,
            allow_scene_floor_absolute=allow_floor_absolute,
            allow_scene_floor_logical=allow_floor_logical,
            project_materials_root=materials_authority_root,
            allow_project_materials_logical=allow_materials_logical,
            scene_room_geometry_root=room_geometry_authority_root,
            allow_scene_room_geometry_absolute=allow_room_geometry_absolute,
            allow_scene_room_geometry_logical=allow_room_geometry_logical,
        )
        relative = record["path"]
        previous = records.setdefault(relative, record)
        if previous != record:
            raise RuntimeError(f"Scene asset record changed during inventory: {relative}")
        resolved_paths[relative] = resolved

    for raw, label, authority in pending:
        add_path(
            raw,
            label,
            allow_floor_absolute=authority == "floor_plan",
            allow_room_geometry_absolute=authority == "room_geometry",
        )

    visited_dependencies: set[str] = set()
    dependency_graph: dict[str, set[str]] = {}

    def add_dependency_edge(owner: str, dependency: PurePosixPath) -> None:
        dependency_string = dependency.as_posix()
        stack = [dependency_string]
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current == owner:
                raise RuntimeError(
                    f"Referenced asset dependency cycle: {owner} -> "
                    f"{dependency_string}"
                )
            if current in seen:
                continue
            seen.add(current)
            stack.extend(dependency_graph.get(current, set()))
        dependency_graph.setdefault(owner, set()).add(dependency_string)

    while True:
        unvisited = sorted(set(records) - visited_dependencies)
        if not unvisited:
            break
        relative_string = unvisited[0]
        visited_dependencies.add(relative_string)
        path = resolved_paths[relative_string]
        relative = PurePosixPath(relative_string)
        is_scene_floor_dependency = relative.parts[0] == "_scene_floor_plan"
        is_scene_room_geometry_dependency = (
            relative.parts[0] == "_scene_room_geometry"
        )
        is_project_material_dependency = relative.parts[0] == "_project_materials"
        suffix = path.suffix.lower()
        dependencies: list[PurePosixPath] = []
        if suffix == ".sdf":
            try:
                root = ET.parse(path).getroot()
            except (OSError, ET.ParseError) as exc:
                raise RuntimeError(f"Cannot parse referenced SDF: {path}") from exc
            consumed_uri_ids: set[int] = set()
            for geometry in root.iter():
                if geometry.tag.rsplit("}", 1)[-1] != "geometry":
                    continue
                for mesh in geometry:
                    if mesh.tag.rsplit("}", 1)[-1] != "mesh":
                        continue
                    mesh_uris = [
                        child
                        for child in mesh
                        if child.tag.rsplit("}", 1)[-1] == "uri"
                    ]
                    if len(mesh_uris) != 1:
                        raise RuntimeError(
                            f"Referenced SDF mesh has a non-exact URI: {relative}"
                        )
                    consumed_uri_ids.add(id(mesh_uris[0]))
                    raw_uri = mesh_uris[0].text or ""
                    if (
                        is_scene_room_geometry_dependency
                        and PurePosixPath(raw_uri).parts
                        and PurePosixPath(raw_uri).parts[0] == ".."
                    ):
                        if floor_authority_root is None:
                            raise RuntimeError(
                                "Referenced room-geometry floor-plan authority is unavailable"
                            )
                        dependencies.append(
                            _scene_room_geometry_dependency_path(
                                raw_uri,
                                owner_path=path,
                                floor_plan_root=floor_authority_root,
                                label=f"referenced room-geometry SDF {relative}",
                            )
                        )
                    else:
                        dependencies.append(
                            _dependency_relative_path(
                                raw_uri,
                                owner_relative=relative,
                                label=f"referenced SDF {relative}",
                                allowed_suffixes=frozenset(
                                    {".gltf", ".obj", ".stl", ".ply"}
                                ),
                            )
                        )
            for include in root.iter():
                if include.tag.rsplit("}", 1)[-1] != "include":
                    continue
                include_uris = [
                    child
                    for child in include
                    if child.tag.rsplit("}", 1)[-1] == "uri"
                ]
                if len(include_uris) != 1:
                    raise RuntimeError(
                        f"Referenced SDF include has a non-exact URI: {relative}"
                    )
                consumed_uri_ids.add(id(include_uris[0]))
                dependencies.append(
                    _dependency_relative_path(
                        include_uris[0].text or "",
                        owner_relative=relative,
                        label=f"referenced SDF include {relative}",
                        allowed_suffixes=frozenset({".sdf"}),
                    )
                )
            all_uri_elements = [
                element
                for element in root.iter()
                if element.tag.rsplit("}", 1)[-1] == "uri"
            ]
            if any(id(element) not in consumed_uri_ids for element in all_uri_elements):
                raise RuntimeError(
                    f"Referenced SDF has unsupported URI elements: {relative}"
                )
        elif suffix == ".glb":
            raise RuntimeError(
                f"Referenced GLB is unsupported by the exact dependency audit: {path}"
            )
        elif suffix == ".gltf":
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Cannot parse referenced glTF: {path}") from exc
            uris: list[Any] = []

            def collect_uris(value: Any) -> None:
                if isinstance(value, dict):
                    for key, child in value.items():
                        if key == "uri":
                            uris.append(child)
                        else:
                            collect_uris(child)
                elif isinstance(value, list):
                    for child in value:
                        collect_uris(child)

            collect_uris(document)
            for raw_uri in uris:
                if isinstance(raw_uri, str) and raw_uri.startswith("data:"):
                    continue
                if (
                    is_scene_floor_dependency
                    and isinstance(raw_uri, str)
                    and PurePosixPath(raw_uri).parts
                    and PurePosixPath(raw_uri).parts[0] == ".."
                ):
                    if materials_authority_root is None:
                        raise RuntimeError(
                            "Referenced scene-floor material authority is unavailable"
                        )
                    dependencies.append(
                        _scene_floor_material_dependency_path(
                            raw_uri,
                            owner_path=path,
                            materials_root=materials_authority_root,
                            label=f"referenced scene-floor glTF {relative}",
                        )
                    )
                else:
                    dependencies.append(
                        _dependency_relative_path(
                            raw_uri,
                            owner_relative=relative,
                            label=f"referenced glTF {relative}",
                            allowed_suffixes=frozenset(
                                {
                                    ".bin",
                                    ".png",
                                    ".jpg",
                                    ".jpeg",
                                    ".webp",
                                    ".ktx2",
                                }
                            ),
                        )
                    )
        elif suffix == ".obj":
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError) as exc:
                raise RuntimeError(f"Cannot parse referenced OBJ: {path}") from exc
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    tokens = shlex.split(stripped, comments=True, posix=True)
                except ValueError as exc:
                    raise RuntimeError(f"Cannot parse referenced OBJ: {path}") from exc
                if tokens and tokens[0].lower() == "mtllib":
                    if len(tokens) != 2:
                        raise RuntimeError(
                            f"Referenced OBJ has ambiguous mtllib: {relative}"
                        )
                    dependencies.append(
                        _dependency_relative_path(
                            tokens[1],
                            owner_relative=relative,
                            label=f"referenced OBJ {relative}",
                            allowed_suffixes=frozenset({".mtl"}),
                        )
                    )
        elif suffix == ".mtl":
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError) as exc:
                raise RuntimeError(f"Cannot parse referenced MTL: {path}") from exc
            texture_commands = {
                "map_ka",
                "map_kd",
                "map_ks",
                "map_ke",
                "map_ns",
                "map_d",
                "bump",
                "map_bump",
                "disp",
                "decal",
                "norm",
                "refl",
            }
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    tokens = shlex.split(stripped, comments=True, posix=True)
                except ValueError as exc:
                    raise RuntimeError(f"Cannot parse referenced MTL: {path}") from exc
                if tokens and tokens[0].lower() in texture_commands:
                    if len(tokens) < 2:
                        raise RuntimeError(
                            f"Referenced MTL texture record is empty: {relative}"
                        )
                    dependencies.append(
                        _dependency_relative_path(
                            tokens[-1],
                            owner_relative=relative,
                            label=f"referenced MTL {relative}",
                            allowed_suffixes=frozenset(
                                {
                                    ".png",
                                    ".jpg",
                                    ".jpeg",
                                    ".webp",
                                    ".tga",
                                    ".bmp",
                                    ".exr",
                                }
                            ),
                        )
                    )
        elif suffix not in {
            ".bin",
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".ktx2",
            ".tga",
            ".bmp",
            ".exr",
            ".stl",
            ".ply",
        }:
            raise RuntimeError(
                f"Referenced asset has an unsupported dependency audit: {relative}"
            )
        for dependency in dependencies:
            add_dependency_edge(relative_string, dependency)
            add_path(
                dependency.as_posix(),
                f"transitive dependency of {relative}",
                allow_absolute=False,
                allow_floor_logical=(
                    is_scene_floor_dependency or is_scene_room_geometry_dependency
                ),
                allow_materials_logical=(
                    is_scene_floor_dependency or is_project_material_dependency
                ),
                allow_room_geometry_logical=is_scene_room_geometry_dependency,
            )

    return [records[path] for path in sorted(records)]


def _verify_referenced_asset_records(
    expected: Any,
    *,
    scene_state: dict[str, Any],
    room_root: Path,
    object_ids: set[str],
) -> None:
    current = _referenced_room_asset_records(
        scene_state,
        room_root=room_root,
        object_ids=object_ids,
    )
    if current != expected:
        raise RuntimeError("Checkpoint referenced asset inventory mismatch")


def _selection_payload(selection: FurnitureSelection) -> dict[str, Any]:
    return {
        "furniture_id": str(selection.furniture_id),
        "suggested_items": selection.suggested_items,
        "prompt_constraints": selection.prompt_constraints,
        "style_notes": selection.style_notes,
        "context_furniture_ids": [
            str(item) for item in selection.context_furniture_ids
        ],
    }


def _accepted_render_evidence(
    directory: Path,
    *,
    minimum_score: int,
    allowed_extra: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Bind the final score and exact copied critic-view inventory."""

    directory = _regular_directory_without_symlink_components(
        directory,
        label="accepted manipuland evidence directory",
    )
    if isinstance(minimum_score, bool) or not isinstance(minimum_score, int):
        raise RuntimeError(f"Manipuland minimum score is not an integer: {minimum_score!r}")
    if minimum_score < 0 or minimum_score > 10:
        raise RuntimeError(f"Manipuland minimum score is outside 0..10: {minimum_score}")
    score = directory / "scores.yaml"
    actual = {path.name for path in directory.iterdir()}
    non_image_names = {score.name, *allowed_extra}
    image_names = sorted(actual - non_image_names)
    unexpected_entries = [
        name for name in image_names if Path(name).suffix.lower() != ".png"
    ]
    if unexpected_entries:
        raise RuntimeError(
            f"Accepted manipuland evidence inventory mismatch: "
            f"unexpected={unexpected_entries}"
        )
    if not image_names:
        raise RuntimeError("Accepted manipuland evidence has no copied critic views")
    images = [directory / name for name in image_names]
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Accepted manipuland evidence entry is not regular: {path}")
    try:
        scores = yaml.safe_load(score.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Cannot parse accepted manipuland scores: {score}") from exc
    if not isinstance(scores, dict) or set(scores) != {*_SCORE_CATEGORIES, "summary"}:
        raise RuntimeError(f"Accepted manipuland scores have a non-exact schema: {score}")
    grades: dict[str, int] = {}
    for category in _SCORE_CATEGORIES:
        item = scores[category]
        if not isinstance(item, dict) or set(item) != {"grade", "comment"}:
            raise RuntimeError(f"Accepted score category {category!r} is malformed")
        grade = item["grade"]
        if (
            isinstance(grade, bool)
            or not isinstance(grade, int)
            or grade < 0
            or grade > 10
            or grade < minimum_score
            or not isinstance(item["comment"], str)
            or not item["comment"].strip()
        ):
            raise RuntimeError(
                f"Accepted score category {category!r} does not pass {minimum_score}"
            )
        grades[category] = grade
    if not isinstance(scores["summary"], str) or not scores["summary"].strip():
        raise RuntimeError("Accepted manipuland score summary is empty")
    return {
        "scores": _file_record(score),
        "images": [_file_record(path) for path in images],
        "minimum_score": minimum_score,
        "grades": grades,
    }


def _evidence_matches(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left == right


def _receipt_attestation(document: dict[str, Any]) -> str:
    payload = {key: value for key, value in document.items() if key != "attestation"}
    return _sha256_bytes(_canonical_json_bytes(payload))


def _transforms_equivalent(before: Any, after: Any) -> bool:
    if (
        not isinstance(before, dict)
        or not isinstance(after, dict)
        or set(before) != {"translation", "rotation_wxyz"}
        or set(after) != {"translation", "rotation_wxyz"}
    ):
        return False
    try:
        before_translation = [float(item) for item in before["translation"]]
        after_translation = [float(item) for item in after["translation"]]
        before_quaternion = [float(item) for item in before["rotation_wxyz"]]
        after_quaternion = [float(item) for item in after["rotation_wxyz"]]
    except (TypeError, ValueError):
        return False
    if (
        len(before_translation) != 3
        or len(after_translation) != 3
        or len(before_quaternion) != 4
        or len(after_quaternion) != 4
        or not all(
            math.isfinite(item)
            for item in (
                *before_translation,
                *after_translation,
                *before_quaternion,
                *after_quaternion,
            )
        )
    ):
        return False
    translation_error = max(
        abs(left - right)
        for left, right in zip(before_translation, after_translation)
    )
    same_sign_error = max(
        abs(left - right) for left, right in zip(before_quaternion, after_quaternion)
    )
    opposite_sign_error = max(
        abs(left + right) for left, right in zip(before_quaternion, after_quaternion)
    )
    return translation_error <= _TRANSFORM_TOLERANCE and min(
        same_sign_error, opposite_sign_error
    ) <= _TRANSFORM_TOLERANCE


def _serialized_scene_roundtrip_state(
    scene: Any, state: dict[str, Any]
) -> dict[str, Any]:
    """Normalize pinned serialized state through the exact resume loader once.

    Drake may canonicalize a quaternion by a final floating-point bit while
    reconstructing ``RigidTransform``.  The live scene has already undergone that
    load, so authorization compares against the exact state produced by applying
    the same loader to the pinned input artifact—not against its pre-load numeric
    spelling.  Every non-loader semantic remains subject to exact dictionary
    equality at the caller.
    """
    if not isinstance(state, dict):
        raise RuntimeError("Serialized scene state is not a mapping")
    if isinstance(scene, RoomScene):
        restored = RoomScene(
            room_geometry=None,
            scene_dir=Path(scene.scene_dir),
            room_id=scene.room_id,
            room_type=scene.room_type,
        )
    else:
        restored = copy.deepcopy(scene)
    try:
        restored.restore_from_state_dict(copy.deepcopy(state))
        normalized = restored.to_state_dict()
    except Exception as exc:
        raise RuntimeError(
            "Serialized scene state cannot complete exact round trip"
        ) from exc
    if not isinstance(normalized, dict):
        raise RuntimeError("Serialized scene round trip did not produce a mapping")
    return normalized


def _serialized_scene_content_hash(scene: Any, state: dict[str, Any]) -> str:
    """Hash the exact semantics obtained by restoring serialized scene state.

    ``RoomScene.to_state_dict`` can normalize numeric transforms when it is loaded
    again.  Durable checkpoints therefore bind the post-serialization state that
    future processes actually resume, rather than an in-memory pre-write hash.
    """
    if isinstance(scene, RoomScene):
        restored = RoomScene(
            room_geometry=None,
            scene_dir=Path(scene.scene_dir),
            room_id=scene.room_id,
            room_type=scene.room_type,
        )
    else:
        # Unit doubles must exercise the same restore/hash protocol without
        # imposing the full RoomScene schema on focused checkpoint tests.
        restored = copy.deepcopy(scene)
    restored.restore_from_state_dict(copy.deepcopy(state))
    value = restored.content_hash()
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError("Serialized scene produced an invalid content hash")
    return value


MANIPULAND_CRITIC_EVIDENCE_CONSTRAINTS = """
CRITIC EVIDENCE CONSTRAINTS (authoritative): Articulated furniture is
intentionally rendered fully open so every drawer/door and support surface is
visible. Never penalize or request changes to drawer/door openness in these
evidence renders. Rigid manipulands must remain collision-free: close adjacency
at an object's edge/corner is the functional equivalent of attachment for this
pipeline. Never request mesh contact, overlap, clipping, or holding between
rigid collision meshes. If physics reports no manipuland collision, required
items are present, bounds/clearance and furniture access pass, and density
matches the assignment, optional micro-styling preferences alone must not hold
Realism, Functionality, Layout, or Holistic Completeness below 8. Do not repeat
a refinement that physics already rejected or that caused automatic removal;
accept the nearest collision-free arrangement.
""".strip()


class StatefulManipulandAgent(BaseStatefulAgent, BaseManipulandAgent):
    """Manipuland placement with planner/designer/critic agents per furniture.

    Workflow:
    1. Initial analysis: Identify which furniture to populate
    2. Per-furniture loop: Create fresh agents for each furniture surface
    3. Per-furniture workflow: Planner coordinates designer/critic
    4. Agent-driven termination: Planner decides when surface is complete
    """

    @property
    def agent_type(self) -> AgentType:
        """Return agent type for collision filtering."""
        return AgentType.MANIPULAND

    def __init__(
        self,
        cfg: DictConfig,
        logger: BaseLogger,
        geometry_server_host: str = "127.0.0.1",
        geometry_server_port: int = 7000,
        hssd_server_host: str = "127.0.0.1",
        hssd_server_port: int = 7001,
        articulated_server_host: str = "127.0.0.1",
        articulated_server_port: int = 7002,
        materials_server_host: str = "127.0.0.1",
        materials_server_port: int = 7008,
        num_workers: int = 1,
        render_gpu_id: int | None = None,
        objaverse_server_host: str = "127.0.0.1",
        objaverse_server_port: int = 7009,
    ):
        # Initialize base agent (sessions, checkpoint state, prompt registry).
        BaseStatefulAgent.__init__(
            self,
            cfg=cfg,
            logger=logger,
            geometry_server_host=geometry_server_host,
            geometry_server_port=geometry_server_port,
            hssd_server_host=hssd_server_host,
            hssd_server_port=hssd_server_port,
        )
        # Initialize manipuland-specific base class.
        BaseManipulandAgent.__init__(
            self,
            cfg=cfg,
            logger=logger,
            geometry_server_host=geometry_server_host,
            geometry_server_port=geometry_server_port,
            hssd_server_host=hssd_server_host,
            hssd_server_port=hssd_server_port,
            objaverse_server_host=objaverse_server_host,
            objaverse_server_port=objaverse_server_port,
            articulated_server_host=articulated_server_host,
            articulated_server_port=articulated_server_port,
            materials_server_host=materials_server_host,
            materials_server_port=materials_server_port,
            num_workers=num_workers,
            render_gpu_id=render_gpu_id,
        )

        # Initialize pending images for image injection during critique.
        self.pending_images: list[dict[str, Any]] = []
        self._legacy_recovery_consumed = False
        self._legacy_recovery_satisfied = False
        self._required_legacy_recovery: tuple[int, str, str] | None = None
        self._protected_object_ids: frozenset[str] = frozenset()
        self._furniture_scope_input_scene_state: dict[str, Any] | None = None
        self._furniture_scope_target_id: str | None = None
        self._protected_asset_object_ids: frozenset[str] = frozenset()
        self._protected_asset_files: list[dict[str, Any]] | None = None
        self._protected_registry_snapshot: dict[str, Any] | None = None
        self._protected_registry_sdf_paths: frozenset[Path] = frozenset()

        # Current furniture selection context (set per-furniture in workflow).
        self.current_furniture_selection: FurnitureSelection | None = None

        # Context image for manipuland designer initialization (per-furniture).
        self.manipuland_context_image_path: Path | None = None

    def _begin_furniture_scope(
        self,
        *,
        furniture_id: UniqueID,
        input_scene_state: dict[str, Any],
    ) -> None:
        """Freeze the exact scene inventory before one furniture workflow."""
        if self._furniture_scope_input_scene_state is not None:
            raise RuntimeError("A manipuland furniture scope is already active")
        self._protected_registry_snapshot = None
        self._protected_registry_sdf_paths = frozenset()
        input_objects = input_scene_state.get("objects")
        live_objects = self.scene.to_state_dict().get("objects")
        if not isinstance(input_objects, dict) or not isinstance(live_objects, dict):
            raise RuntimeError("Manipuland furniture scope has no object mapping")
        if set(input_objects) != set(live_objects):
            raise RuntimeError("Manipuland furniture scope input inventory changed")
        # Validate the global support-surface namespace before any observation,
        # physics ownership lookup, or mutation tool can consume it.
        self.scene._synchronize_surface_id_counter(reset=False)
        protected_ids = {str(key) for key in input_objects}
        room_geometry = input_scene_state.get("room_geometry")
        floor = room_geometry.get("floor") if isinstance(room_geometry, dict) else None
        if isinstance(floor, dict) and isinstance(floor.get("object_id"), str):
            protected_ids.add(floor["object_id"])
        self._protected_object_ids = frozenset(protected_ids)
        self._protected_asset_object_ids = frozenset(protected_ids)
        self._protected_asset_files = _referenced_room_asset_records(
            input_scene_state,
            room_root=Path(self.scene.scene_dir),
            object_ids=set(protected_ids),
        )
        asset_manager = getattr(self, "asset_manager", None)
        if asset_manager is None:
            if not bool(getattr(self, "cfg", {}).get("checkpoint_test", False)):
                raise RuntimeError("Manipuland scope lacks an asset registry")
        else:
            asset_manager.reconcile_registry_with_scene(scene=self.scene)
            self._protected_registry_snapshot = asset_manager.registry.snapshot()
            self._protected_registry_sdf_paths = frozenset(
                asset_manager.registry.snapshot_sdf_paths(
                    self._protected_registry_snapshot
                )
            )
        self._furniture_scope_input_scene_state = copy.deepcopy(input_scene_state)
        self._furniture_scope_target_id = str(furniture_id)

    def _clear_furniture_scope(self) -> None:
        """Clear per-furniture ownership after success or failure."""
        self._protected_object_ids = frozenset()
        self._furniture_scope_input_scene_state = None
        self._furniture_scope_target_id = None
        self._protected_asset_object_ids = frozenset()
        self._protected_asset_files = None
        self._protected_registry_snapshot = None
        self._protected_registry_sdf_paths = frozenset()

    @staticmethod
    def _verify_input_object_semantics(
        *,
        input_scene_state: dict[str, Any],
        output_scene_state: dict[str, Any],
        target_furniture_id: str,
    ) -> None:
        """Prove every preexisting object stayed semanticly immutable.

        Support extraction may change only the current furniture's
        ``support_surfaces`` field.  Numerically equivalent transform round trips
        are accepted using the same tolerance as legacy recovery.
        """
        def verify_object(
            object_id: str,
            before: Any,
            after: Any,
            *,
            allow_surface_delta: bool,
        ) -> None:
            if not isinstance(before, dict) or not isinstance(after, dict):
                raise RuntimeError(
                    f"Manipuland checkpoint object {object_id} is malformed"
                )
            if not _transforms_equivalent(
                before.get("transform"), after.get("transform")
            ):
                raise RuntimeError(
                    f"Manipuland checkpoint mutated protected object {object_id}"
                )
            before_semantics = copy.deepcopy(before)
            after_semantics = copy.deepcopy(after)
            before_semantics.pop("transform", None)
            after_semantics.pop("transform", None)
            if allow_surface_delta:
                before_semantics.pop("support_surfaces", None)
                after_semantics.pop("support_surfaces", None)
            if before_semantics != after_semantics:
                raise RuntimeError(
                    f"Manipuland checkpoint mutated protected object {object_id}"
                )

        input_objects = input_scene_state.get("objects")
        output_objects = output_scene_state.get("objects")
        if not isinstance(input_objects, dict) or not isinstance(output_objects, dict):
            raise RuntimeError("Manipuland checkpoint semantic input is malformed")
        missing = sorted(set(input_objects) - set(output_objects))
        if missing:
            raise RuntimeError(
                f"Manipuland checkpoint removed protected input objects: {missing}"
            )
        for object_id, before in input_objects.items():
            verify_object(
                object_id,
                before,
                output_objects[object_id],
                allow_surface_delta=object_id == target_furniture_id,
            )

        if input_scene_state.get("text_description") != output_scene_state.get(
            "text_description"
        ):
            raise RuntimeError("Manipuland checkpoint changed the room description")
        input_geometry = input_scene_state.get("room_geometry")
        output_geometry = output_scene_state.get("room_geometry")
        if not isinstance(input_geometry, dict) or not isinstance(output_geometry, dict):
            if input_geometry != output_geometry:
                raise RuntimeError("Manipuland checkpoint changed room geometry")
            return
        input_geometry = copy.deepcopy(input_geometry)
        output_geometry = copy.deepcopy(output_geometry)
        input_floor = input_geometry.pop("floor", None)
        output_floor = output_geometry.pop("floor", None)
        if input_floor is None or output_floor is None:
            if input_floor != output_floor:
                raise RuntimeError("Manipuland checkpoint changed protected floor")
        else:
            floor_id = input_floor.get("object_id") if isinstance(input_floor, dict) else None
            if not isinstance(floor_id, str) or (
                not isinstance(output_floor, dict)
                or output_floor.get("object_id") != floor_id
            ):
                raise RuntimeError("Manipuland checkpoint changed protected floor")
            verify_object(
                floor_id,
                input_floor,
                output_floor,
                allow_surface_delta=floor_id == target_furniture_id,
            )

        input_walls = input_geometry.pop("walls", None)
        output_walls = output_geometry.pop("walls", None)
        if not isinstance(input_walls, list) or not isinstance(output_walls, list):
            if input_walls != output_walls:
                raise RuntimeError("Manipuland checkpoint changed protected walls")
        else:
            input_wall_ids = [
                wall.get("object_id") if isinstance(wall, dict) else None
                for wall in input_walls
            ]
            output_wall_ids = [
                wall.get("object_id") if isinstance(wall, dict) else None
                for wall in output_walls
            ]
            if (
                None in input_wall_ids
                or input_wall_ids != output_wall_ids
                or len(set(input_wall_ids)) != len(input_wall_ids)
            ):
                raise RuntimeError("Manipuland checkpoint changed protected wall inventory")
            for wall_id, before_wall, after_wall in zip(
                input_wall_ids, input_walls, output_walls
            ):
                verify_object(
                    wall_id,
                    before_wall,
                    after_wall,
                    allow_surface_delta=wall_id == target_furniture_id,
                )
        if input_geometry != output_geometry:
            raise RuntimeError("Manipuland checkpoint changed protected room geometry")

    def _verify_active_furniture_scope(self) -> None:
        if (
            self._furniture_scope_input_scene_state is None
            or self._furniture_scope_target_id is None
        ):
            raise RuntimeError("No active manipuland furniture scope")
        self._verify_input_object_semantics(
            input_scene_state=self._furniture_scope_input_scene_state,
            output_scene_state=self.scene.to_state_dict(),
            target_furniture_id=self._furniture_scope_target_id,
        )
        if self._protected_asset_files is None or not self._protected_asset_object_ids:
            raise RuntimeError("No protected manipuland asset snapshot is active")
        _verify_referenced_asset_records(
            self._protected_asset_files,
            scene_state=self.scene.to_state_dict(),
            room_root=Path(self.scene.scene_dir),
            object_ids=set(self._protected_asset_object_ids),
        )
        asset_manager = getattr(self, "asset_manager", None)
        if getattr(self, "_protected_registry_snapshot", None) is not None:
            if asset_manager is None:
                raise RuntimeError("Protected asset registry disappeared")
            asset_manager.registry.verify_snapshot(
                self._protected_registry_snapshot
            )

    def _current_scope_manipuland_ids(
        self, furniture_id: UniqueID
    ) -> list[UniqueID]:
        """Return and validate exactly the objects created in this scope."""
        if str(furniture_id) != self._furniture_scope_target_id:
            raise RuntimeError("Manipuland furniture scope target changed")
        furniture = self.scene.get_object(furniture_id)
        if furniture is None:
            raise RuntimeError(f"Furniture {furniture_id} disappeared")
        surface_ids = {surface.surface_id for surface in furniture.support_surfaces}
        current_ids = {str(object_id) for object_id in self.scene.objects}
        created_ids = sorted(current_ids - self._protected_object_ids)
        result: list[UniqueID] = []
        for object_id in created_ids:
            obj = self.scene.get_object(UniqueID(object_id))
            if (
                obj is None
                or obj.object_type != ObjectType.MANIPULAND
                or obj.placement_info is None
                or obj.placement_info.parent_surface_id not in surface_ids
            ):
                raise RuntimeError(
                    f"Current furniture created an out-of-scope object: {object_id}"
                )
            result.append(obj.object_id)
        return result

    def _render_furniture_for_context(self) -> Path:
        """Render furniture with clean angled front view for context image input.

        Uses furniture_selection mode with empty annotate_object_types to get
        a clean render without any labels, bounding boxes, or coordinate overlays.
        For articulated furniture, opens joints to show interior surfaces.
        Includes context furniture (e.g., chairs around a table) for spatial reference.

        Uses adaptive camera elevation based on furniture type:
        - Tables (1 surface): High elevation (60°) - looking down at surface
        - Shelves (multiple surfaces): Low elevation (30°) - see all levels from front

        Camera is positioned to view the furniture's front face (+Y in local frame),
        accounting for the furniture's world rotation.

        Special case for floor: Renders top-down view of entire room with all
        furniture visible, similar to observe_scene. This provides spatial context
        for floor item placement (rugs, floor lamps, etc.).

        Returns:
            Path to directory containing rendered images.
        """
        furniture = self.scene.get_object(self.current_furniture_id)

        # Special case: Floor needs top-down view of entire room with all furniture.
        # This provides spatial context for floor item placement.
        if furniture.object_type == ObjectType.FLOOR:
            # Include all furniture objects for room context.
            all_furniture_ids = [
                obj.object_id
                for obj in self.scene.objects.values()
                if obj.object_type == ObjectType.FURNITURE
            ]
            return self.rendering_manager.render_scene(
                scene=self.scene,
                blender_server=self.blender_server,
                include_objects=[self.current_furniture_id] + all_furniture_ids,
                exclude_room_geometry=False,  # Include floor/walls for context
                rendering_mode="furniture_selection",  # Disables grid/frame
                annotate_object_types=[],  # Disables all labels/bboxes
                render_name=f"context_input_{self.current_furniture_id}",
                # Top-down view for floor context.
                include_vertical_views=True,  # Include top view
                override_side_view_count=0,  # No side views, just top
            )

        # Get context furniture IDs from current selection.
        context_ids = (
            self.current_furniture_selection.context_furniture_ids
            if self.current_furniture_selection
            else []
        )

        # Include current furniture + validated context furniture (same pattern as
        # observe_scene).
        valid_context_ids = [
            ctx_id for ctx_id in context_ids if ctx_id in self.scene.objects
        ]
        include_objects = [self.current_furniture_id] + valid_context_ids

        # Check if furniture is articulated (has doors/drawers).
        is_articulated = furniture.metadata.get("is_articulated", False)

        # Determine elevation based on furniture type (number of support surfaces).
        # Tables with 1 surface benefit from high angle looking down at surface.
        # Shelves with multiple surfaces need low angle to see all levels.
        num_surfaces = (
            len(furniture.support_surfaces) if furniture.support_surfaces else 1
        )
        if num_surfaces == 1:
            elevation = 60.0  # High angle - looking down at table surface
        else:
            elevation = 30.0  # Low angle - see all shelf levels from front

        # Calculate camera azimuth to view the furniture's front face.
        # Furniture "front" is +Y in local frame. We need to find where that
        # points in world frame and position the camera there.
        # For a Z-rotation (yaw) of θ, the camera should be at azimuth = 90° + θ.
        rotation_matrix = furniture.transform.rotation().matrix()
        # Extract yaw (Z rotation) from rotation matrix: atan2(R[1,0], R[0,0]).
        yaw_rad = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        # Camera azimuth: 90° (front at +Y) + furniture yaw rotation.
        front_azimuth = 90.0 + math.degrees(yaw_rad)

        return self.rendering_manager.render_scene(
            scene=self.scene,
            blender_server=self.blender_server,
            include_objects=include_objects,
            exclude_room_geometry=True,  # Furniture only, no floor/walls
            rendering_mode="furniture_selection",  # Disables grid/frame
            annotate_object_types=[],  # Disables all labels/bboxes
            articulated_open=is_articulated,  # Open joints to show interior surfaces
            context_furniture_ids=valid_context_ids,  # For proper visibility in render
            render_name=f"context_input_{self.current_furniture_id}",
            # Render single angled view from furniture's front face.
            include_vertical_views=False,  # No pure top/bottom views
            override_side_view_count=1,  # Single angled view
            side_view_start_azimuth_degrees=front_azimuth,  # Front of furniture
            side_view_elevation_degrees=elevation,  # Adaptive elevation
        )

    def _get_furniture_dimensions(self, furniture) -> str:
        """Compute human-readable furniture dimensions from bbox.

        Args:
            furniture: SceneObject with bbox_min and bbox_max.

        Returns:
            Human-readable dimensions string.
        """
        if furniture.bbox_min is None or furniture.bbox_max is None:
            return "dimensions unknown"

        dims = furniture.bbox_max - furniture.bbox_min
        width, depth, height = dims[0], dims[1], dims[2]
        return f"{width:.2f}m wide × {depth:.2f}m deep × {height:.2f}m tall"

    def _generate_manipuland_context_image(self) -> Path | None:
        """Generate context image for manipuland placement.

        Renders the furniture and uses image editing API to add suggested objects.
        This provides visual guidance for the manipuland designer agent.

        Returns:
            Path to generated context image, or None if generation fails or disabled.
        """
        if not self.cfg.context_image_generation.enabled:
            return None

        render_dir = self._render_furniture_for_context()

        selection = self.current_furniture_selection
        furniture = self.scene.get_object(selection.furniture_id)

        # Select correct image based on furniture type.
        # Floor uses top-down view; other furniture uses angled front view.
        if furniture.object_type == ObjectType.FLOOR:
            render = render_dir / "0_top.png"
        else:
            render = render_dir / "0_side.png"

        try:
            return self.asset_manager.image_generator.generate_manipuland_context_image(
                reference_image_path=render,
                furniture_description=furniture.description,
                furniture_dimensions=self._get_furniture_dimensions(furniture),
                suggested_items=selection.suggested_items,
                prompt_constraints=selection.prompt_constraints,
                style_notes=selection.style_notes,
                output_path=render_dir / "context_edited.png",
            )
        except Exception as e:
            console_logger.warning(f"Context image generation failed: {e}")
            return None

    def _get_context_image_path(self) -> Path | None:
        """Get the AI-generated context image for initial design.

        Returns:
            Path to manipuland context image if available, None otherwise.
        """
        return self.manipuland_context_image_path

    def _create_designer_tools(
        self,
        current_furniture_id: UniqueID,
        support_surfaces: dict[str, SupportSurface],
    ) -> list[FunctionTool]:
        """Create designer tools with captured dependencies.

        Args:
            current_furniture_id: ID of furniture being populated.
            support_surfaces: Dictionary mapping surface_id to SupportSurface.

        Returns:
            List of tools for the designer agent.
        """
        # Get context furniture from current selection.
        context_ids = []
        if self.current_furniture_selection:
            context_ids = self.current_furniture_selection.context_furniture_ids

        vision_tools = ManipulandVisionTools(
            scene=self.scene,
            rendering_manager=self.rendering_manager,
            cfg=self.cfg,
            current_furniture_id=current_furniture_id,
            blender_server=self.blender_server,
            context_furniture_ids=context_ids,
            protected_object_ids=set(self._protected_object_ids),
        )
        self.manipuland_tools = ManipulandTools(
            scene=self.scene,
            asset_manager=self.asset_manager,
            cfg=self.cfg,
            current_furniture_id=current_furniture_id,
            support_surfaces=support_surfaces,
            protected_object_ids=set(self._protected_object_ids),
            protected_registry_sdf_paths=set(
                self._protected_registry_sdf_paths
            ),
        )
        workflow_tools = WorkflowTools()

        return [
            *vision_tools.tools.values(),
            *self.manipuland_tools.tools.values(),
            *workflow_tools.tools.values(),
        ]

    def _create_designer_agent(
        self, tools: list[FunctionTool], furniture_description: str
    ) -> Agent:
        """Create designer agent with furniture-specific context.

        Args:
            tools: Tools to provide to the designer.
            furniture_description: Description of furniture being populated.

        Returns:
            Configured designer agent.
        """
        designer_config = self.cfg.agents.designer_agent
        designer_prompt_enum = ManipulandAgentPrompts[designer_config.prompt]

        # Get structured assignment context from current furniture selection.
        selection = self.current_furniture_selection
        if not selection:
            raise ValueError("No current furniture selection set")

        return super()._create_designer_agent(
            tools=tools,
            prompt_enum=designer_prompt_enum,
            furniture_description=furniture_description,
            suggested_items=selection.suggested_items,
            prompt_constraints=selection.prompt_constraints,
            style_notes=selection.style_notes,
            has_reference_image=self.manipuland_context_image_path is not None,
        )

    def _create_critic_agent(
        self, tools: list[FunctionTool], furniture_description: str
    ) -> Agent:
        """Create critic agent with furniture-specific context.

        Args:
            tools: Tools to provide to the critic.
            furniture_description: Description of furniture being populated.

        Returns:
            Configured critic agent with structured output.
        """
        critic_config = self.cfg.agents.critic_agent
        critic_prompt_enum = ManipulandAgentPrompts[critic_config.prompt]

        # Get structured assignment context from current furniture selection.
        selection = self.current_furniture_selection
        if not selection:
            raise ValueError("No current furniture selection set")

        critic_furniture_description = (
            f"{furniture_description}\n\n{MANIPULAND_CRITIC_EVIDENCE_CONSTRAINTS}"
        )
        return super()._create_critic_agent(
            tools=tools,
            prompt_enum=critic_prompt_enum,
            output_type=ManipulandCritiqueWithScores,
            furniture_description=critic_furniture_description,
            suggested_items=selection.suggested_items,
            prompt_constraints=selection.prompt_constraints,
            style_notes=selection.style_notes,
        )

    def _create_planner_agent(
        self, tools: list[FunctionTool], furniture_description: str
    ) -> Agent:
        """Create planner agent with furniture-specific context.

        Args:
            tools: Tools to provide to the planner.
            furniture_description: Description of furniture being populated.

        Returns:
            Configured planner agent.
        """
        planner_config = self.cfg.agents.planner_agent
        planner_prompt_enum = ManipulandAgentPrompts[planner_config.prompt]
        single_threshold = self.cfg.reset_single_category_threshold
        total_threshold = self.cfg.reset_total_sum_threshold

        # Get structured assignment context from current furniture selection.
        selection = self.current_furniture_selection
        if not selection:
            raise ValueError("No current furniture selection set")

        return super()._create_planner_agent(
            tools=tools,
            prompt_enum=planner_prompt_enum,
            furniture_description=furniture_description,
            suggested_items=selection.suggested_items,
            prompt_constraints=selection.prompt_constraints,
            style_notes=selection.style_notes,
            max_critique_rounds=self.cfg.max_critique_rounds,
            reset_single_category_threshold=single_threshold,
            reset_total_sum_threshold=total_threshold,
            early_finish_min_score=self.cfg.early_finish_min_score,
        )

    def _create_tools_for_furniture(
        self, furniture_id: UniqueID
    ) -> tuple[list[FunctionTool], list[FunctionTool], list[FunctionTool]]:
        """Create tools for planner, designer, and critic.

        Args:
            furniture_id: ID of current furniture.

        Returns:
            Tuple of (planner_tools, designer_tools, critic_tools).
        """
        # Get all support surfaces for this furniture.
        furniture = self.scene.get_object(furniture_id)
        if not furniture or not furniture.support_surfaces:
            raise ValueError(f"Furniture {furniture_id} has no support surfaces")

        # Build dict mapping surface_id strings to SupportSurface objects.
        support_surfaces = {
            str(surface.surface_id): surface for surface in furniture.support_surfaces
        }

        # Create designer tools using base class helper method.
        # This ensures consistency with furniture agent architecture and includes
        # WorkflowTools for task management.
        designer_tools = self._create_designer_tools(
            current_furniture_id=furniture_id, support_surfaces=support_surfaces
        )

        # Planner gets all designer tools (same access).
        planner_tools = designer_tools

        # Create critic tools using helper method.
        critic_tools = self._create_critic_tools(furniture_id=furniture_id)

        return planner_tools, designer_tools, critic_tools

    def _get_initial_design_prompt_enum(self) -> Any:
        """Get the prompt enum for initial design instruction.

        Returns:
            Manipuland-specific initial design instruction prompt.
        """
        return ManipulandAgentPrompts.DESIGNER_INITIAL_INSTRUCTION

    def _get_initial_design_prompt_kwargs(self) -> dict:
        """Get prompt kwargs for initial design instruction.

        Returns:
            Dict with has_reference_image flag.
        """
        return {
            "has_reference_image": self.manipuland_context_image_path is not None,
        }

    def _get_design_change_prompt_enum(self) -> Any:
        """Get the prompt enum for design change instruction.

        Returns:
            Manipuland-specific design change instruction prompt.
        """
        return ManipulandAgentPrompts.DESIGNER_CRITIQUE_INSTRUCTION

    def _get_critique_prompt_enum(self) -> Any:
        """Get the prompt enum for critic runner instruction.

        Returns:
            Manipuland-specific critic instruction prompt.
        """
        return ManipulandAgentPrompts.MANIPULAND_CRITIC_RUNNER_INSTRUCTION

    def _set_placement_noise_profile(self, mode: PlacementNoiseMode) -> None:
        """Set placement noise profile for manipuland tools.

        Args:
            mode: Placement noise mode (NATURAL or PERFECT).
        """
        self.manipuland_tools.set_noise_profile(mode)

    def _get_manipuland_physics_scope_ids(self) -> set[UniqueID] | None:
        """Expose only objects created after the active furniture snapshot."""
        if self._furniture_scope_input_scene_state is None:
            return set()
        return {
            obj.object_id
            for obj in self.scene.objects.values()
            if obj.object_type == ObjectType.MANIPULAND
            and str(obj.object_id) not in self._protected_object_ids
        }

    def _create_critic_tools(self, furniture_id: UniqueID) -> list[FunctionTool]:
        """Create critic tools with read-only scene access.

        Args:
            furniture_id: ID of furniture being critiqued (for context rendering).

        Returns:
            List of tools for the critic (read-only scene validation tools).
        """
        # Get context furniture from current selection.
        context_ids = []
        if self.current_furniture_selection:
            context_ids = self.current_furniture_selection.context_furniture_ids

        # Create vision tools for critic (read-only operations).
        vision_tools = ManipulandVisionTools(
            scene=self.scene,
            rendering_manager=self.rendering_manager,
            cfg=self.cfg,
            current_furniture_id=furniture_id,
            blender_server=self.blender_server,
            context_furniture_ids=context_ids,
            protected_object_ids=set(self._protected_object_ids),
        )

        # Critic gets read-only tools (observe only).
        # Note: check_physics is NOT included since physics_context is already
        # injected via the critique runner instruction template.
        return [
            vision_tools.tools["observe_scene"],
            self.manipuland_tools.tools["get_current_scene_state"],
        ]

    def _setup_furniture_context(self, furniture_selection: FurnitureSelection) -> None:
        """Set up per-furniture rendering and analysis context.

        Args:
            furniture_selection: Selection data for this furniture including
                suggested items, prompt constraints, and style notes.
        """
        # Clear pending images from previous furniture iteration.
        # This prevents image leakage if session callback somehow doesn't trigger.
        self.pending_images = []

        furniture_id = furniture_selection.furniture_id

        # Create per-furniture rendering manager with subdirectory.
        self.rendering_manager = RenderingManager(
            cfg=self.cfg.rendering,
            logger=self.logger,
            subdirectory=f"manipulands_{furniture_id}",
        )

        # Update scene_analyzer to use per-furniture rendering manager.
        self.scene_analyzer = SceneAnalyzer(
            vlm_service=self.vlm_service,
            rendering_manager=self.rendering_manager,
            cfg=self.cfg,
            blender_server=self.blender_server,
        )

        # Store current furniture selection for agent creation.
        self.current_furniture_id = furniture_id
        self.current_furniture_selection = furniture_selection

    def _initialize_checkpoint_state(self) -> None:
        """Reset checkpoint state for new furniture iteration.

        Called at the start of each furniture iteration to clear checkpoint
        state from the previous furniture piece. The attributes themselves
        were initialized in __init__().
        """
        # Reset checkpoint state to None for new furniture iteration.
        self.previous_scene_checkpoint = None
        self.scene_checkpoint = None
        self.previous_checkpoint_scores = None
        self.checkpoint_scores = None
        self.previous_scores = None
        self.previous_checkpoint_render_dir = None
        self.checkpoint_render_dir = None
        # Keep placement_style as-is (it persists across furniture iterations).

    def _setup_furniture_agents(
        self, furniture_id: UniqueID, furniture_description: str
    ) -> None:
        """Create agents and sessions for this furniture piece.

        Args:
            furniture_id: ID of furniture being populated.
            furniture_description: Human-readable furniture description.
        """
        # Create fresh tools and agents for this furniture.
        # First create designer/critic tools.
        (
            _,  # planner_tools created later after agents exist
            designer_tools,
            critic_tools,
        ) = self._create_tools_for_furniture(furniture_id)

        # Create sessions using base class helper.
        # Sessions are stored as instance variables for planner tool closures.
        self.designer_session, self.critic_session = self._create_sessions(
            session_prefix=f"{furniture_id}_"
        )

        # Create agents using base class helpers with override methods.
        self.designer = self._create_designer_agent(
            tools=designer_tools, furniture_description=furniture_description
        )

        self.critic = self._create_critic_agent(
            tools=critic_tools, furniture_description=furniture_description
        )

        # Now create planner tools (can reference self.designer/critic/sessions).
        planner_tools = self._create_planner_tools()

        # Create planner agent using base class helper with override method.
        self.planner = self._create_planner_agent(
            tools=planner_tools, furniture_description=furniture_description
        )

    async def _run_furniture_workflow(self, furniture_id: UniqueID) -> None:
        """Execute the multi-agent workflow for a furniture piece.

        Args:
            furniture_id: ID of furniture being populated.
        """
        # Reaching this workflow means no completed checkpoint was restored for
        # this furniture.  Discard any fixed-name SQLite history left by an
        # earlier aborted attempt so it cannot describe a scene that was rolled
        # back, while retaining session history created during this workflow.
        await self.designer_session.clear_session()
        await self.critic_session.clear_session()

        # Get runner instruction for planner to start workflow.
        planner_runner_prompt = (
            ManipulandAgentPrompts.MANIPULAND_PLANNER_RUNNER_INSTRUCTION
        )
        runner_instruction = self.prompt_registry.get_prompt(
            prompt_enum=planner_runner_prompt,
        )

        result: RunResult = await Runner.run(
            starting_agent=self.planner,
            input=runner_instruction,
            max_turns=self.cfg.agents.planner_agent.max_turns,
            run_config=self._create_run_config(),
        )
        log_agent_usage(result=result, agent_name="PLANNER (MANIPULAND)")

        if result.final_output:
            log_agent_response(
                response=result.final_output, agent_name="PLANNER (MANIPULAND)"
            )

        # Compute final critique and scores for completed furniture.
        # Check if scene changed since last checkpoint to avoid redundant critique.
        current_scene_hash = self.scene.content_hash()

        if (
            self.checkpoint_scene_hash is not None
            and current_scene_hash == self.checkpoint_scene_hash
        ):
            console_logger.info(
                "Scene unchanged since last critique, skipping final critique"
            )
        else:
            console_logger.info(
                "Scene changed since last critique, computing final critique"
            )
            # Pass update_checkpoint=False to preserve N-1 checkpoint for reset check.
            await self._request_critique_impl(update_checkpoint=False)

        # Validate final scene and save scores.
        await self._finalize_scene_and_scores()

        console_logger.info(
            f"Completed manipuland placement for furniture {furniture_id}"
        )

    def _get_final_scores_directory(self) -> Path:
        """Get the directory path for saving per-furniture manipuland placement state.

        Returns:
            Path to scene_states/manipuland_furniture_{id} directory.
        """
        return (
            self.logger.output_dir
            / "scene_states"
            / f"manipuland_furniture_{self.current_furniture_id}"
        )

    def _checkpoint_context(
        self, furniture_selection: FurnitureSelection, furniture_index: int
    ) -> dict[str, Any]:
        selection = _selection_payload(furniture_selection)
        resolved_config = OmegaConf.to_container(self.cfg, resolve=True)
        return {
            "furniture_index": furniture_index,
            "furniture_id": str(furniture_selection.furniture_id),
            "selection": selection,
            "selection_sha256": _sha256_bytes(_canonical_json_bytes(selection)),
            "config_sha256": _sha256_bytes(_canonical_json_bytes(resolved_config)),
            "checkpoint_runtime_sha256": _sha256_file(Path(__file__).resolve()),
        }

    def _plan_context(self, input_scene_hash: str) -> dict[str, Any]:
        resolved_config = OmegaConf.to_container(self.cfg, resolve=True)
        return {
            "room_id": str(getattr(self.scene, "room_id", "")),
            "input_scene_content_hash": input_scene_hash,
            "config_sha256": _sha256_bytes(_canonical_json_bytes(resolved_config)),
            "checkpoint_runtime_sha256": _sha256_file(Path(__file__).resolve()),
        }

    def _load_furniture_plan(
        self, input_scene_hash: str
    ) -> list[FurnitureSelection] | None:
        scene_states = self.logger.output_dir / "scene_states"
        path = scene_states / "manipuland_furniture_plan.json"
        with _exclusive_checkpoint_lock(scene_states):
            if not path.exists():
                return None
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Manipuland furniture plan is not regular: {path}")
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Cannot parse manipuland furniture plan: {path}") from exc
            expected_keys = {
                "schema_version",
                "status",
                "room_id",
                "input_scene_content_hash",
                "config_sha256",
                "checkpoint_runtime_sha256",
                "selections",
                "selections_sha256",
                "attestation",
            }
            if (
                not isinstance(document, dict)
                or set(document) != expected_keys
                or document["schema_version"] != _FURNITURE_PLAN_SCHEMA
                or document["status"] != "pass"
                or document["attestation"] != _receipt_attestation(document)
            ):
                raise RuntimeError("Manipuland furniture plan contract mismatch")
            plan_context = self._plan_context(input_scene_hash)
            for key, expected in plan_context.items():
                if key == "checkpoint_runtime_sha256":
                    runtime_sha = document.get(key)
                    if runtime_sha not in {
                        expected,
                        _SCOPE_PREDECESSOR_RUNTIME_SHA256,
                    }:
                        raise RuntimeError(
                            "Manipuland furniture plan checkpoint_runtime_sha256 mismatch"
                        )
                    if runtime_sha == _SCOPE_PREDECESSOR_RUNTIME_SHA256 and (
                        input_scene_hash
                        != _SCOPE_PREDECESSOR_INPUT_SCENE_CONTENT_SHA256
                        or _sha256_file(path) != _SCOPE_PREDECESSOR_PLAN_SHA256
                    ):
                        raise RuntimeError(
                            "Manipuland predecessor furniture plan artifact mismatch"
                        )
                    continue
                if document.get(key) != expected:
                    raise RuntimeError(f"Manipuland furniture plan {key} mismatch")
            selections = document.get("selections")
            if (
                not isinstance(selections, list)
                or not selections
                or document.get("selections_sha256")
                != _sha256_bytes(_canonical_json_bytes(selections))
            ):
                raise RuntimeError("Manipuland furniture plan selection hash mismatch")
            result: list[FurnitureSelection] = []
            seen: set[str] = set()
            expected_selection_keys = {
                "furniture_id",
                "suggested_items",
                "prompt_constraints",
                "style_notes",
                "context_furniture_ids",
            }
            for item in selections:
                if not isinstance(item, dict) or set(item) != expected_selection_keys:
                    raise RuntimeError("Manipuland furniture plan selection is malformed")
                furniture_id = item["furniture_id"]
                contexts = item["context_furniture_ids"]
                if (
                    not isinstance(furniture_id, str)
                    or not _SAFE_CHECKPOINT_ID.fullmatch(furniture_id)
                    or furniture_id in seen
                    or not all(
                        isinstance(item[key], str)
                        for key in ("suggested_items", "prompt_constraints", "style_notes")
                    )
                    or not isinstance(contexts, list)
                    or not all(isinstance(value, str) for value in contexts)
                ):
                    raise RuntimeError("Manipuland furniture plan selection is invalid")
                seen.add(furniture_id)
                result.append(
                    FurnitureSelection(
                        furniture_id=UniqueID(furniture_id),
                        suggested_items=item["suggested_items"],
                        prompt_constraints=item["prompt_constraints"],
                        style_notes=item["style_notes"],
                        context_furniture_ids=[UniqueID(value) for value in contexts],
                    )
                )
            console_logger.info("Restored exact manipuland furniture plan from %s", path)
            return result

    def _publish_furniture_plan(
        self, input_scene_hash: str, selections: list[FurnitureSelection]
    ) -> None:
        if not selections:
            raise RuntimeError("Refusing to publish an empty manipuland furniture plan")
        payload = [_selection_payload(selection) for selection in selections]
        document = {
            "schema_version": _FURNITURE_PLAN_SCHEMA,
            "status": "pass",
            **self._plan_context(input_scene_hash),
            "selections": payload,
            "selections_sha256": _sha256_bytes(_canonical_json_bytes(payload)),
        }
        document["attestation"] = _receipt_attestation(document)
        scene_states = self.logger.output_dir / "scene_states"
        path = scene_states / "manipuland_furniture_plan.json"
        with _exclusive_checkpoint_lock(scene_states):
            if path.exists():
                raise RuntimeError(f"Refusing to replace manipuland furniture plan: {path}")
            _atomic_json_fsynced(path, document)
        console_logger.info("Published exact manipuland furniture plan: %s", path)

    def _minimum_accepted_score(self) -> int:
        value = self.cfg.early_finish_min_score
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(f"Invalid manipuland early_finish_min_score: {value!r}")
        return value

    def _checkpoint_directory(
        self, furniture_selection: FurnitureSelection, furniture_index: int
    ) -> Path:
        furniture_id = str(furniture_selection.furniture_id)
        if not _SAFE_CHECKPOINT_ID.fullmatch(furniture_id):
            raise RuntimeError(f"Unsafe furniture ID for checkpoint: {furniture_id!r}")
        return (
            self.logger.output_dir
            / "scene_states"
            / f"manipuland_checkpoint_{furniture_index:03d}_{furniture_id}"
        )

    @staticmethod
    def _verify_file_record(path: Path, record: Any, *, label: str) -> None:
        if not isinstance(record, dict) or set(record) != {
            "name",
            "size_bytes",
            "sha256",
        }:
            raise RuntimeError(f"Malformed {label} record")
        current = _file_record(path)
        if current != record:
            raise RuntimeError(f"{label} hash/size mismatch: {path}")

    def _verify_legacy_checkpoint_source(
        self,
        source: Any,
        *,
        accepted_evidence: dict[str, Any],
        scene_state: dict[str, Any],
        furniture_selection: FurnitureSelection,
        furniture_index: int,
        input_scene_hash: str,
        input_object_ids: set[str],
        live_input_scene_state: dict[str, Any] | None = None,
    ) -> None:
        if source is None:
            return
        expected_keys = {
            "authorization",
            "render_directory",
            "scene_state",
            "drake_directive",
            "output_scene_content_hash",
            "accepted_evidence",
            "added_manipuland_ids",
            "added_asset_files",
            "referenced_asset_object_ids",
            "referenced_asset_files",
        }
        if not isinstance(source, dict) or set(source) != expected_keys:
            raise RuntimeError("Manipuland checkpoint legacy source schema mismatch")
        authorization = source["authorization"]
        authorization_value = (
            authorization.get("manifest_path") if isinstance(authorization, dict) else None
        )
        authorization_lexical = (
            Path(authorization_value) if isinstance(authorization_value, str) else None
        )
        if (
            not isinstance(authorization, dict)
            or set(authorization)
            != {"manifest_path", "manifest_size_bytes", "manifest_sha256"}
            or not isinstance(authorization["manifest_path"], str)
            or not authorization["manifest_path"]
            or "%" in authorization["manifest_path"]
            or authorization_lexical is None
            or not authorization_lexical.is_absolute()
            or str(authorization_lexical) != authorization["manifest_path"]
            or any(
                part in {"", ".", ".."}
                for part in authorization_lexical.parts[1:]
            )
            or isinstance(authorization["manifest_size_bytes"], bool)
            or not isinstance(authorization["manifest_size_bytes"], int)
            or authorization["manifest_size_bytes"] <= 0
            or not isinstance(authorization["manifest_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", authorization["manifest_sha256"])
        ):
            raise RuntimeError("Manipuland checkpoint legacy authorization is malformed")
        authorization_path = _regular_file_without_symlink_components(
            Path(authorization["manifest_path"]),
            label="legacy authorization manifest",
        )
        output_root = self.logger.output_dir.resolve(strict=True)
        if authorization_path.is_relative_to(output_root):
            raise RuntimeError("Legacy authorization manifest must be outside room output")
        if (
            authorization_path.stat().st_size
            != authorization["manifest_size_bytes"]
            or _sha256_file(authorization_path) != authorization["manifest_sha256"]
        ):
            raise RuntimeError("Legacy authorization manifest changed after promotion")
        try:
            manifest = json.loads(authorization_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Cannot parse checkpoint legacy authorization") from exc
        manifest = _validate_legacy_recovery_manifest_document(
            manifest,
            expected_room_id=str(getattr(self.scene, "room_id", "")),
        )
        configured_sha = os.environ.get(
            "SCENESMITH_LEGACY_MANIPULAND_RECOVERY_SHA256", ""
        ).strip()
        if configured_sha and configured_sha != authorization["manifest_sha256"]:
            raise RuntimeError("Durable legacy checkpoint authorization SHA mismatch")
        if not isinstance(source["output_scene_content_hash"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", source["output_scene_content_hash"]
        ):
            raise RuntimeError("Manipuland checkpoint legacy output hash is malformed")
        expected_source_fields = {
            "render_directory": manifest["render_directory"],
            "scene_state": manifest["scene_state"],
            "drake_directive": manifest["drake_directive"],
            "output_scene_content_hash": manifest["output_scene_content_hash"],
            "accepted_evidence": manifest["accepted_evidence"],
            "added_manipuland_ids": manifest["added_manipuland_ids"],
            "added_asset_files": manifest["added_asset_files"],
            "referenced_asset_object_ids": manifest[
                "referenced_asset_object_ids"
            ],
            "referenced_asset_files": manifest["referenced_asset_files"],
        }
        if any(source.get(key) != value for key, value in expected_source_fields.items()):
            raise RuntimeError(
                "Manipuland checkpoint legacy source does not match authorization manifest"
            )
        if (
            manifest["furniture_index"] != furniture_index
            or manifest["furniture_id"] != str(furniture_selection.furniture_id)
            or manifest["input_scene_content_hash"] != input_scene_hash
        ):
            raise RuntimeError(
                "Manipuland checkpoint legacy authorization context mismatch"
            )
        input_relative = _canonical_relative_path(
            manifest["input_scene_state_path"],
            label="checkpoint legacy input scene-state path",
        )
        input_state_path = _regular_file_without_symlink_components(
            output_root.joinpath(*input_relative.parts),
            label="checkpoint legacy input scene state",
        )
        if not input_state_path.is_relative_to(output_root):
            raise RuntimeError("Checkpoint legacy input scene state escapes room output")
        self._verify_file_record(
            input_state_path,
            manifest["input_scene_state"],
            label="checkpoint legacy input scene state",
        )
        try:
            input_state = json.loads(input_state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Cannot parse checkpoint legacy input scene state") from exc
        manifest_input_objects = input_state.get("objects")
        if (
            not isinstance(manifest_input_objects, dict)
            or set(manifest_input_objects) != input_object_ids
        ):
            raise RuntimeError(
                "Manipuland checkpoint legacy input object inventory mismatch"
            )
        if live_input_scene_state is not None:
            expected_input = copy.deepcopy(live_input_scene_state)
            recorded_input = _serialized_scene_roundtrip_state(
                self.scene, input_state
            )
            expected_input.pop("timestamp", None)
            recorded_input.pop("timestamp", None)
            if expected_input != recorded_input:
                raise RuntimeError(
                    "Manipuland checkpoint legacy input state does not match live scene"
                )
        render_directory = source["render_directory"]
        if not isinstance(render_directory, str) or "\\" in render_directory:
            raise RuntimeError("Manipuland checkpoint legacy render directory is unsafe")
        render_relative = PurePosixPath(render_directory)
        if (
            render_relative.is_absolute()
            or not render_relative.parts
            or any(part in {"", ".", ".."} for part in render_relative.parts)
        ):
            raise RuntimeError("Manipuland checkpoint legacy render directory is unsafe")

        for key in ("scene_state", "drake_directive"):
            record = source[key]
            if not _valid_file_record(record):
                raise RuntimeError(f"Manipuland checkpoint legacy {key} is malformed")
        if source["accepted_evidence"] != accepted_evidence:
            raise RuntimeError("Manipuland checkpoint legacy evidence mismatch")
        added_ids = source["added_manipuland_ids"]
        if (
            not isinstance(added_ids, list)
            or not added_ids
            or len(added_ids) != len(set(added_ids))
            or not all(
                isinstance(value, str) and _SAFE_CHECKPOINT_ID.fullmatch(value)
                for value in added_ids
            )
        ):
            raise RuntimeError("Manipuland checkpoint legacy object inventory is malformed")
        current_objects = scene_state.get("objects")
        if (
            not isinstance(current_objects, dict)
            or not input_object_ids.issubset(current_objects)
            or set(current_objects) - input_object_ids != set(added_ids)
        ):
            raise RuntimeError(
                "Manipuland checkpoint legacy postprocessed object delta mismatch"
            )

        render_dir = _regular_directory_without_symlink_components(
            output_root.joinpath(*render_relative.parts),
            label="checkpoint legacy render directory",
            required_root=output_root,
        )
        rendered_evidence = _accepted_render_evidence(
            render_dir,
            minimum_score=self._minimum_accepted_score(),
            allowed_extra=frozenset({"scene_state.json", "scene.dmd.yaml"}),
        )
        if rendered_evidence != accepted_evidence:
            raise RuntimeError("Manipuland checkpoint legacy render evidence changed")
        self._verify_file_record(
            render_dir / "scene_state.json",
            source["scene_state"],
            label="checkpoint legacy scene state",
        )
        self._verify_file_record(
            render_dir / "scene.dmd.yaml",
            source["drake_directive"],
            label="checkpoint legacy Drake directive",
        )
        try:
            candidate_state = json.loads(
                (render_dir / "scene_state.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Cannot parse checkpoint legacy render state") from exc
        candidate_objects = candidate_state.get("objects")
        if (
            not isinstance(candidate_objects, dict)
            or not input_object_ids.issubset(candidate_objects)
            or set(candidate_objects) - input_object_ids != set(added_ids)
        ):
            raise RuntimeError("Manipuland checkpoint legacy render delta mismatch")
        asset_records = source["added_asset_files"]
        if not isinstance(asset_records, list) or not asset_records:
            raise RuntimeError("Manipuland checkpoint legacy asset inventory is empty")
        _verify_referenced_asset_records(
            asset_records,
            scene_state=candidate_state,
            room_root=Path(self.scene.scene_dir),
            object_ids=set(added_ids),
        )
        referenced_ids = source["referenced_asset_object_ids"]
        referenced_assets = source["referenced_asset_files"]
        for state_document in (candidate_state, scene_state):
            _verify_referenced_asset_records(
                referenced_assets,
                scene_state=state_document,
                room_root=Path(self.scene.scene_dir),
                object_ids=set(referenced_ids),
            )
        _verify_referenced_asset_records(
            asset_records,
            scene_state=scene_state,
            room_root=Path(self.scene.scene_dir),
            object_ids=set(added_ids),
        )

    def _restore_completed_furniture_checkpoint(
        self,
        *,
        furniture_selection: FurnitureSelection,
        furniture_index: int,
        input_scene_hash: str,
    ) -> bool:
        scene_states = self.logger.output_dir / "scene_states"
        with _exclusive_checkpoint_lock(scene_states):
            return self._restore_completed_furniture_checkpoint_unlocked(
                furniture_selection=furniture_selection,
                furniture_index=furniture_index,
                input_scene_hash=input_scene_hash,
            )

    def _restore_completed_furniture_checkpoint_unlocked(
        self,
        *,
        furniture_selection: FurnitureSelection,
        furniture_index: int,
        input_scene_hash: str,
    ) -> bool:
        directory = self._checkpoint_directory(furniture_selection, furniture_index)
        if not directory.exists():
            return False
        if directory.is_symlink() or not directory.is_dir():
            raise RuntimeError(f"Invalid manipuland checkpoint directory: {directory}")
        expected_entries = {
            "scene_state.json",
            "scene.dmd.yaml",
            "completion_receipt.json",
        }
        actual_entries = {path.name for path in directory.iterdir()}
        if actual_entries != expected_entries or any(
            path.is_symlink() or not path.is_file() for path in directory.iterdir()
        ):
            raise RuntimeError(
                f"Manipuland checkpoint inventory mismatch: {directory}"
            )
        receipt_path = directory / "completion_receipt.json"
        receipt_file_sha256 = _sha256_file(receipt_path)
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read manipuland checkpoint: {directory}") from exc
        expected_receipt_keys = {
            "schema_version",
            "status",
            "furniture_index",
            "furniture_id",
            "selection",
            "selection_sha256",
            "config_sha256",
            "checkpoint_runtime_sha256",
            "input_scene_content_hash",
            "output_scene_content_hash",
            "accepted_evidence",
            "added_manipuland_ids",
            "referenced_asset_object_ids",
            "referenced_asset_files",
            "legacy_source",
            "artifacts",
            "attestation",
        }
        if isinstance(receipt, dict) and receipt.get("schema_version") == 3:
            expected_receipt_keys.add("asset_registry_snapshot")
        if not isinstance(receipt, dict) or set(receipt) != expected_receipt_keys:
            raise RuntimeError(f"Manipuland checkpoint receipt schema mismatch: {directory}")
        context = self._checkpoint_context(furniture_selection, furniture_index)
        is_exact_predecessor = (
            receipt.get("schema_version") == 2
            and furniture_index == 0
            and str(furniture_selection.furniture_id) == "teacher_desk_0"
            and receipt.get("checkpoint_runtime_sha256")
            == _SCOPE_PREDECESSOR_RUNTIME_SHA256
            and receipt.get("input_scene_content_hash")
            == _SCOPE_PREDECESSOR_INPUT_SCENE_CONTENT_SHA256
            and receipt.get("output_scene_content_hash")
            == _SCOPE_PREDECESSOR_OUTPUT_SCENE_CONTENT_SHA256
            and receipt_file_sha256 == _SCOPE_PREDECESSOR_RECEIPT_SHA256
        )
        is_exact_filing_cabinet_predecessor = (
            receipt.get("schema_version") == _FURNITURE_CHECKPOINT_SCHEMA
            and furniture_index == 1
            and str(furniture_selection.furniture_id) == "filing_cabinet_0"
            and receipt.get("checkpoint_runtime_sha256")
            == _FILING_CABINET_RUNTIME_PREDECESSOR_SHA256
            and receipt.get("input_scene_content_hash")
            == _FILING_CABINET_PREDECESSOR_INPUT_SCENE_CONTENT_SHA256
            and receipt.get("output_scene_content_hash")
            == _FILING_CABINET_PREDECESSOR_OUTPUT_SCENE_CONTENT_SHA256
            and receipt_file_sha256 == _FILING_CABINET_PREDECESSOR_RECEIPT_SHA256
        )
        for key, expected in context.items():
            if key == "checkpoint_runtime_sha256" and (
                is_exact_predecessor or is_exact_filing_cabinet_predecessor
            ):
                continue
            if receipt.get(key) != expected:
                raise RuntimeError(
                    f"Manipuland checkpoint {key} mismatch for {directory}"
                )
        if (
            (
                receipt.get("schema_version") != _FURNITURE_CHECKPOINT_SCHEMA
                and not is_exact_predecessor
            )
            or receipt.get("status") != "pass"
            or receipt.get("input_scene_content_hash") != input_scene_hash
            or receipt.get("attestation") != _receipt_attestation(receipt)
        ):
            raise RuntimeError(f"Manipuland checkpoint contract mismatch: {directory}")
        registry_snapshot = receipt.get("asset_registry_snapshot")
        asset_manager = getattr(self, "asset_manager", None)
        if receipt.get("schema_version") == _FURNITURE_CHECKPOINT_SCHEMA:
            if asset_manager is not None:
                asset_manager.registry.verify_snapshot(registry_snapshot)
            elif not (
                bool(self.cfg.get("checkpoint_test", False))
                and registry_snapshot == AssetRegistry.empty_snapshot()
            ):
                raise RuntimeError("Checkpoint asset registry snapshot is unavailable")

        state_path = directory / "scene_state.json"
        directive_path = directory / "scene.dmd.yaml"
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != {
            "scene_state",
            "drake_directive",
        }:
            raise RuntimeError(f"Manipuland checkpoint artifacts are missing: {directory}")
        self._verify_file_record(
            state_path, artifacts.get("scene_state"), label="checkpoint scene state"
        )
        self._verify_file_record(
            directive_path,
            artifacts.get("drake_directive"),
            label="checkpoint Drake directive",
        )
        if is_exact_predecessor and (
            _sha256_file(state_path) != _SCOPE_PREDECESSOR_SCENE_STATE_SHA256
            or _sha256_file(directive_path)
            != _SCOPE_PREDECESSOR_DRAKE_DIRECTIVE_SHA256
        ):
            raise RuntimeError("Manipuland predecessor checkpoint artifact mismatch")
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Cannot parse checkpoint scene state") from exc
        live_input_state = self.scene.to_state_dict()
        live_input_objects = live_input_state.get("objects")
        if not isinstance(live_input_objects, dict):
            raise RuntimeError("Live checkpoint input scene has no object mapping")
        live_input_object_ids = set(live_input_objects)
        referenced_ids = receipt["referenced_asset_object_ids"]
        added_ids = receipt["added_manipuland_ids"]
        if (
            not isinstance(referenced_ids, list)
            or not referenced_ids
            or referenced_ids != sorted(set(referenced_ids))
            or not all(
                isinstance(value, str) and _SAFE_CHECKPOINT_ID.fullmatch(value)
                for value in referenced_ids
            )
            or str(furniture_selection.furniture_id) not in referenced_ids
        ):
            raise RuntimeError("Checkpoint referenced object inventory is malformed")
        if (
            not isinstance(added_ids, list)
            or not added_ids
            or added_ids != sorted(set(added_ids))
            or not all(
                isinstance(value, str) and _SAFE_CHECKPOINT_ID.fullmatch(value)
                for value in added_ids
            )
            or set(referenced_ids)
            != set(added_ids) | {str(furniture_selection.furniture_id)}
        ):
            raise RuntimeError("Checkpoint added manipuland inventory is malformed")
        state_objects = state.get("objects")
        if (
            not isinstance(state_objects, dict)
            or not live_input_object_ids.issubset(state_objects)
            or set(state_objects) - live_input_object_ids != set(added_ids)
        ):
            raise RuntimeError("Checkpoint restored object delta mismatch")
        self._verify_input_object_semantics(
            input_scene_state=live_input_state,
            output_scene_state=state,
            target_furniture_id=str(furniture_selection.furniture_id),
        )
        _verify_referenced_asset_records(
            receipt["referenced_asset_files"],
            scene_state=state,
            room_root=Path(self.scene.scene_dir),
            object_ids=set(referenced_ids),
        )
        accepted_dir = (
            self.logger.output_dir
            / "scene_states"
            / f"manipuland_furniture_{furniture_selection.furniture_id}"
        )
        if not _evidence_matches(
            _accepted_render_evidence(
                accepted_dir, minimum_score=self._minimum_accepted_score()
            ),
            receipt.get("accepted_evidence"),
        ):
            raise RuntimeError(f"Accepted manipuland evidence changed: {accepted_dir}")
        self._verify_legacy_checkpoint_source(
            receipt["legacy_source"],
            accepted_evidence=receipt["accepted_evidence"],
            scene_state=state,
            furniture_selection=furniture_selection,
            furniture_index=furniture_index,
            input_scene_hash=input_scene_hash,
            input_object_ids=live_input_object_ids,
            live_input_scene_state=live_input_state,
        )
        serialized_content_hash = _serialized_scene_content_hash(self.scene, state)
        if is_exact_predecessor:
            if (
                serialized_content_hash
                != _SCOPE_PREDECESSOR_ROUNDTRIP_CONTENT_SHA256
            ):
                raise RuntimeError(
                    "Manipuland predecessor checkpoint round-trip hash mismatch"
                )
        elif receipt.get("output_scene_content_hash") != serialized_content_hash:
            raise RuntimeError("Serialized manipuland checkpoint content hash mismatch")
        required = getattr(self, "_required_legacy_recovery", None)
        satisfies_required = False
        if required is not None and required[:2] == (
            furniture_index,
            str(furniture_selection.furniture_id),
        ):
            source = receipt["legacy_source"]
            if (
                not isinstance(source, dict)
                or source.get("authorization", {}).get("manifest_sha256")
                != required[2]
            ):
                raise RuntimeError(
                    "Required legacy target checkpoint lacks exact authorization"
                )
            satisfies_required = True

        previous = self.scene.to_state_dict()
        asset_manager = getattr(self, "asset_manager", None)
        try:
            self.scene.restore_from_state_dict(state)
            if self.scene.content_hash() != serialized_content_hash:
                raise RuntimeError("Restored manipuland checkpoint content hash mismatch")
        except Exception:
            self.scene.restore_from_state_dict(previous)
            raise
        if asset_manager is not None:
            asset_manager.registry._last_save_committed = False
        try:
            if asset_manager is not None:
                asset_manager.reconcile_registry_with_scene(
                    scene=self.scene,
                    leaf_object_ids={UniqueID(value) for value in added_ids},
                )
            elif not bool(self.cfg.get("checkpoint_test", False)):
                raise RuntimeError("Restored checkpoint lacks registry reconciliation")
        except Exception:
            if not (
                asset_manager is not None
                and asset_manager.registry._last_save_committed
            ):
                self.scene.restore_from_state_dict(previous)
            raise
        if satisfies_required:
            self._legacy_recovery_satisfied = True
        console_logger.info(
            "Restored durable manipuland checkpoint for %s",
            furniture_selection.furniture_id,
        )
        return True

    def _load_explicit_legacy_manifest(
        self,
    ) -> tuple[dict[str, Any], Path, str] | None:
        manifest_value = os.environ.get(
            "SCENESMITH_LEGACY_MANIPULAND_RECOVERY_MANIFEST", ""
        ).strip()
        expected_sha = os.environ.get(
            "SCENESMITH_LEGACY_MANIPULAND_RECOVERY_SHA256", ""
        ).strip()
        if not manifest_value and not expected_sha:
            return None
        if not manifest_value or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise RuntimeError(
                "Legacy manipuland recovery requires both an explicit manifest "
                "path and its lowercase SHA-256"
            )
        lexical_manifest = Path(manifest_value)
        if (
            "%" in manifest_value
            or not lexical_manifest.is_absolute()
            or str(lexical_manifest) != manifest_value
            or any(part in {"", ".", ".."} for part in lexical_manifest.parts[1:])
        ):
            raise RuntimeError("Legacy recovery manifest path must be canonical absolute")
        manifest_path = _regular_file_without_symlink_components(
            lexical_manifest,
            label="legacy recovery manifest",
        )
        output_root = self.logger.output_dir.resolve(strict=True)
        if manifest_path.is_relative_to(output_root):
            raise RuntimeError("Legacy recovery manifest must be outside room output")
        if _sha256_file(manifest_path) != expected_sha:
            raise RuntimeError("Legacy recovery manifest SHA-256 mismatch")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Cannot parse legacy recovery manifest") from exc
        manifest = _validate_legacy_recovery_manifest_document(
            manifest,
            expected_room_id=str(getattr(self.scene, "room_id", "")),
        )
        return manifest, manifest_path, expected_sha

    def _validate_explicit_legacy_plan(
        self, selections: list[FurnitureSelection]
    ) -> None:
        self._required_legacy_recovery = None
        self._legacy_recovery_satisfied = False
        loaded = self._load_explicit_legacy_manifest()
        if loaded is None:
            return
        manifest, _, expected_sha = loaded
        index = manifest["furniture_index"]
        target = manifest["furniture_id"]
        if index >= len(selections) or str(selections[index].furniture_id) != target:
            raise RuntimeError(
                "Legacy recovery target/index is absent or moved in furniture plan"
            )
        self._required_legacy_recovery = (index, target, expected_sha)

    def _legacy_render_candidate(
        self,
        furniture_selection: FurnitureSelection,
        furniture_index: int,
        input_scene_hash: str,
    ) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]] | None:
        if self._legacy_recovery_consumed:
            return None
        loaded = self._load_explicit_legacy_manifest()
        if loaded is None:
            return None
        manifest, manifest_path, expected_sha = loaded
        furniture_id = str(furniture_selection.furniture_id)
        if (
            manifest["furniture_index"] != furniture_index
            or manifest["furniture_id"] != furniture_id
            or manifest["input_scene_content_hash"] != input_scene_hash
        ):
            # A manifest for an earlier furniture is harmless after that exact
            # target has already been restored from its durable checkpoint.
            if manifest.get("furniture_id") != furniture_id:
                return None
            raise RuntimeError("Legacy recovery manifest target/input mismatch")

        root = self.logger.output_dir.resolve(strict=True)

        def resolve_relative(raw: Any, *, label: str) -> Path:
            if (
                not isinstance(raw, str)
                or not raw
                or raw != raw.strip()
                or "\\" in raw
                or "%" in raw
            ):
                raise RuntimeError(f"Unsafe {label} in legacy manifest")
            relative = PurePosixPath(raw)
            if (
                relative.is_absolute()
                or relative.as_posix() != raw
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                raise RuntimeError(f"Unsafe {label} in legacy manifest")
            path = root
            for part in relative.parts:
                path = path / part
                if path.is_symlink():
                    raise RuntimeError(f"Legacy {label} contains a symlink")
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise RuntimeError(f"Legacy {label} escapes the room output")
            return resolved

        input_state_path = resolve_relative(
            manifest["input_scene_state_path"], label="input scene-state path"
        )
        self._verify_file_record(
            input_state_path,
            manifest["input_scene_state"],
            label="legacy input scene state",
        )
        input_document = json.loads(input_state_path.read_text(encoding="utf-8"))
        current_document = self.scene.to_state_dict()
        input_objects = input_document.get("objects")
        current_objects = current_document.get("objects")
        if (
            self.scene.content_hash() != manifest["input_scene_content_hash"]
            or not isinstance(input_objects, dict)
            or not isinstance(current_objects, dict)
            or set(input_objects) != set(current_objects)
        ):
            raise RuntimeError(
                "Legacy manifest input semantics are not the live base scene"
            )
        candidate_dir = resolve_relative(
            manifest["render_directory"], label="render directory"
        )
        if candidate_dir.is_symlink() or not candidate_dir.is_dir():
            raise RuntimeError(f"Legacy render directory is invalid: {candidate_dir}")
        accepted_dir = (
            self.logger.output_dir
            / "scene_states"
            / f"manipuland_furniture_{furniture_id}"
        )
        accepted = _accepted_render_evidence(
            accepted_dir, minimum_score=self._minimum_accepted_score()
        )
        rendered = _accepted_render_evidence(
            candidate_dir,
            minimum_score=self._minimum_accepted_score(),
            allowed_extra=frozenset({"scene_state.json", "scene.dmd.yaml"}),
        )
        if accepted != rendered or accepted != manifest["accepted_evidence"]:
            raise RuntimeError("Legacy accepted score/view evidence mismatch")
        self._verify_file_record(
            candidate_dir / "scene_state.json",
            manifest["scene_state"],
            label="legacy render scene state",
        )
        self._verify_file_record(
            candidate_dir / "scene.dmd.yaml",
            manifest["drake_directive"],
            label="legacy render Drake directive",
        )
        authorization = {
            "manifest_path": str(manifest_path.resolve(strict=True)),
            "manifest_size_bytes": manifest_path.stat().st_size,
            "manifest_sha256": expected_sha,
        }
        return candidate_dir, accepted, authorization, manifest

    def _restore_legacy_accepted_render(
        self,
        furniture_selection: FurnitureSelection,
        furniture_index: int,
        input_scene_hash: str,
    ) -> dict[str, Any] | None:
        """Recover the exact accepted render state, never an arbitrary iteration."""

        match = self._legacy_render_candidate(
            furniture_selection, furniture_index, input_scene_hash
        )
        if match is None:
            return None
        candidate_dir, accepted, authorization, manifest = match
        state_path = candidate_dir / "scene_state.json"
        directive_path = candidate_dir / "scene.dmd.yaml"
        state_record = _file_record(state_path)
        directive_record = _file_record(directive_path)
        candidate = json.loads(state_path.read_text(encoding="utf-8"))
        current = self.scene.to_state_dict()
        for document in (candidate, current):
            document.pop("timestamp", None)
        if (
            candidate.get("room_geometry") != current.get("room_geometry")
            or candidate.get("text_description") != current.get("text_description")
            or not isinstance(candidate.get("objects"), dict)
            or not isinstance(current.get("objects"), dict)
        ):
            raise RuntimeError("Legacy manipuland state is not based on this room")

        current_objects = current["objects"]
        candidate_objects = candidate["objects"]
        missing = sorted(set(current_objects) - set(candidate_objects))
        added = sorted(set(candidate_objects) - set(current_objects))
        if missing or not added:
            raise RuntimeError(
                f"Legacy manipuland object delta is invalid: missing={missing}, added={added}"
            )
        target_id = str(furniture_selection.furniture_id)
        target = candidate_objects.get(target_id)
        if not isinstance(target, dict) or not target.get("support_surfaces"):
            raise RuntimeError("Legacy target furniture has no authored support surfaces")
        surface_ids = {
            item.get("surface_id")
            for item in target["support_surfaces"]
            if isinstance(item, dict)
        }
        if None in surface_ids or not surface_ids:
            raise RuntimeError("Legacy target support-surface inventory is malformed")

        for object_id, before in current_objects.items():
            after = candidate_objects[object_id]
            if not isinstance(before, dict) or not isinstance(after, dict):
                raise RuntimeError(f"Legacy object {object_id} is malformed")
            allowed_schema_delta = {"support_surfaces"} if object_id == target_id else set()
            schema_delta = set(before) ^ set(after)
            if schema_delta - allowed_schema_delta or not _transforms_equivalent(
                before.get("transform"), after.get("transform")
            ):
                raise RuntimeError(
                    f"Legacy render changed the transform/schema of {object_id}"
                )
            ignored = {"transform"}
            if object_id == target_id:
                ignored.add("support_surfaces")
            changed = {
                key
                for key in before
                if key not in ignored and before.get(key) != after.get(key)
            }
            if changed:
                raise RuntimeError(
                    f"Legacy render changed protected fields on {object_id}: {changed}"
                )
        for object_id in added:
            item = candidate_objects[object_id]
            placement = item.get("placement_info")
            metadata = item.get("metadata")
            if (
                item.get("object_type") != "manipuland"
                or not isinstance(placement, dict)
                or placement.get("parent_surface_id") not in surface_ids
                or not isinstance(metadata, dict)
            ):
                raise RuntimeError(
                    f"Legacy added object is not an exact supported manipuland: {object_id}"
                )
            composite_type = metadata.get("composite_type")
            if composite_type is None:
                if metadata.get("asset_source") != "objaverse":
                    raise RuntimeError(
                        f"Legacy manipuland lacks ObjectThor provenance: {object_id}"
                    )
                if not item.get("geometry_path") or not item.get("sdf_path"):
                    raise RuntimeError(f"Legacy manipuland lacks assets: {object_id}")
            else:
                if composite_type not in {"stack", "pile", "fill"}:
                    raise RuntimeError(
                        f"Legacy manipuland has unknown composite type: {object_id}"
                    )
                members = metadata.get("member_assets")
                if not isinstance(members, list) or not members:
                    raise RuntimeError(f"Legacy composite has no members: {object_id}")
                for member_index, member in enumerate(members):
                    if not isinstance(member, dict):
                        raise RuntimeError(f"Legacy composite member is malformed: {object_id}")
                    if not member.get("geometry_path") or not member.get("sdf_path"):
                        raise RuntimeError(
                            f"Legacy composite member lacks assets: "
                            f"{object_id} member {member_index}"
                        )

        added_asset_files = _referenced_room_asset_records(
            candidate,
            room_root=Path(self.scene.scene_dir),
            object_ids=set(added),
        )
        referenced_asset_object_ids = sorted(set(added) | {target_id})
        referenced_asset_files = _referenced_room_asset_records(
            candidate,
            room_root=Path(self.scene.scene_dir),
            object_ids=set(referenced_asset_object_ids),
        )
        if added != manifest["added_manipuland_ids"]:
            raise RuntimeError("Legacy manifest added manipuland inventory mismatch")
        if added_asset_files != manifest["added_asset_files"]:
            raise RuntimeError("Legacy manifest added asset inventory mismatch")
        if (
            referenced_asset_object_ids
            != manifest["referenced_asset_object_ids"]
            or referenced_asset_files != manifest["referenced_asset_files"]
        ):
            raise RuntimeError("Legacy manifest referenced asset inventory mismatch")

        previous = self.scene.to_state_dict()
        try:
            self.scene.restore_from_state_dict(candidate)
            if self.scene.content_hash() != manifest["output_scene_content_hash"]:
                raise RuntimeError("Legacy manifest output scene content hash mismatch")
        except Exception:
            self.scene.restore_from_state_dict(previous)
            raise
        console_logger.info(
            "Recovered exact accepted legacy render for %s from %s",
            target_id,
            candidate_dir,
        )
        self._legacy_recovery_consumed = True
        return {
            "authorization": authorization,
            "render_directory": str(candidate_dir.relative_to(self.logger.output_dir)),
            "scene_state": state_record,
            "drake_directive": directive_record,
            "output_scene_content_hash": manifest["output_scene_content_hash"],
            "accepted_evidence": accepted,
            "added_manipuland_ids": added,
            "added_asset_files": added_asset_files,
            "referenced_asset_object_ids": referenced_asset_object_ids,
            "referenced_asset_files": referenced_asset_files,
        }

    def _run_per_furniture_postprocessing(
        self,
        furniture_id: UniqueID,
        manipuland_ids: list[UniqueID],
    ) -> None:
        if not self.cfg.per_furniture_postprocessing.enabled:
            return
        sim_cfg = self.cfg.per_furniture_postprocessing.simulation
        sim_html_path = None
        if sim_cfg.save_html:
            sim_html_path = (
                self.scene.scene_dir
                / "simulation"
                / "per_furniture"
                / f"{furniture_id}_simulation.html"
            )
        self.scene = apply_per_furniture_postprocessing(
            full_scene=self.scene,
            furniture_id=furniture_id,
            manipuland_ids=manipuland_ids,
            config=self.cfg.per_furniture_postprocessing,
            simulation_html_path=sim_html_path,
        )

    def _publish_furniture_checkpoint(
        self,
        *,
        furniture_selection: FurnitureSelection,
        furniture_index: int,
        input_scene_hash: str,
        input_object_ids: set[str],
        input_scene_state: dict[str, Any],
        legacy_source: dict[str, Any] | None,
    ) -> None:
        scene_states = self.logger.output_dir / "scene_states"
        with _exclusive_checkpoint_lock(scene_states):
            self._publish_furniture_checkpoint_unlocked(
                furniture_selection=furniture_selection,
                furniture_index=furniture_index,
                input_scene_hash=input_scene_hash,
                input_object_ids=input_object_ids,
                input_scene_state=input_scene_state,
                legacy_source=legacy_source,
            )

    def _publish_furniture_checkpoint_unlocked(
        self,
        *,
        furniture_selection: FurnitureSelection,
        furniture_index: int,
        input_scene_hash: str,
        input_object_ids: set[str],
        input_scene_state: dict[str, Any],
        legacy_source: dict[str, Any] | None,
    ) -> None:
        final = self._checkpoint_directory(furniture_selection, furniture_index)
        if final.exists():
            raise RuntimeError(f"Refusing to replace existing checkpoint: {final}")
        final.parent.mkdir(parents=True, exist_ok=True)
        stale = sorted(final.parent.glob(f".{final.name}.tmp.*"))
        if stale:
            raise RuntimeError(f"Stale manipuland checkpoint transaction exists: {stale}")
        accepted_dir = (
            self.logger.output_dir
            / "scene_states"
            / f"manipuland_furniture_{furniture_selection.furniture_id}"
        )
        accepted_evidence = _accepted_render_evidence(
            accepted_dir, minimum_score=self._minimum_accepted_score()
        )
        output_state = self.scene.to_state_dict()
        protected_asset_files = getattr(self, "_protected_asset_files", None)
        protected_asset_ids = getattr(
            self, "_protected_asset_object_ids", frozenset()
        )
        if protected_asset_files is not None and protected_asset_ids:
            _verify_referenced_asset_records(
                protected_asset_files,
                scene_state=output_state,
                room_root=Path(self.scene.scene_dir),
                object_ids=set(protected_asset_ids),
            )
        elif not bool(self.cfg.get("checkpoint_test", False)):
            raise RuntimeError("Manipuland checkpoint lacks protected asset snapshot")
        output_objects = output_state.get("objects")
        if not isinstance(output_objects, dict) or not input_object_ids.issubset(
            output_objects
        ):
            raise RuntimeError("Manipuland checkpoint input object inventory changed")
        input_semantic_objects = input_scene_state.get("objects")
        if (
            not isinstance(input_semantic_objects, dict)
            or set(input_semantic_objects) != input_object_ids
        ):
            raise RuntimeError("Manipuland checkpoint semantic input inventory mismatch")
        target_id = str(furniture_selection.furniture_id)
        self._verify_input_object_semantics(
            input_scene_state=input_scene_state,
            output_scene_state=output_state,
            target_furniture_id=target_id,
        )
        added_ids = set(output_objects) - input_object_ids
        if not added_ids:
            raise RuntimeError("Manipuland checkpoint has no added manipulands")
        target = output_objects.get(target_id)
        if target is None:
            room_geometry = output_state.get("room_geometry")
            floor = (
                room_geometry.get("floor")
                if isinstance(room_geometry, dict)
                else None
            )
            if isinstance(floor, dict) and floor.get("object_id") == target_id:
                target = floor
        if target is None:
            raise RuntimeError("Manipuland checkpoint target disappeared")
        if not isinstance(target, dict):
            raise RuntimeError("Manipuland checkpoint target is malformed")
        target_surfaces = {
            surface.get("surface_id")
            for surface in target.get("support_surfaces", [])
            if isinstance(surface, dict)
        }
        if None in target_surfaces or not target_surfaces:
            raise RuntimeError("Manipuland checkpoint target has no exact surfaces")
        for added_id in added_ids:
            added = output_objects[added_id]
            placement = added.get("placement_info") if isinstance(added, dict) else None
            if (
                not isinstance(added, dict)
                or added.get("object_type") != "manipuland"
                or not isinstance(placement, dict)
                or placement.get("parent_surface_id") not in target_surfaces
            ):
                raise RuntimeError(
                    f"Manipuland checkpoint added out-of-scope object {added_id}"
                )
        referenced_object_ids = sorted(added_ids | {target_id})
        referenced_asset_files = _referenced_room_asset_records(
            output_state,
            room_root=Path(self.scene.scene_dir),
            object_ids=set(referenced_object_ids),
        )
        normal_render_records: dict[str, dict[str, Any]] | None = None
        if legacy_source is None:
            render_dir = getattr(self, "final_render_dir", None) or getattr(
                self, "checkpoint_render_dir", None
            )
            if render_dir is None:
                raise RuntimeError("Current manipuland render evidence is unavailable")
            current_evidence = _accepted_render_evidence(
                Path(render_dir),
                minimum_score=self._minimum_accepted_score(),
                allowed_extra=frozenset({"scene_state.json", "scene.dmd.yaml"}),
            )
            if current_evidence != accepted_evidence:
                raise RuntimeError(
                    "Copied manipuland evidence does not match current accepted render"
                )
            render_state_path = Path(render_dir) / "scene_state.json"
            render_directive_path = Path(render_dir) / "scene.dmd.yaml"
            normal_render_records = {
                "scene_state": _file_record(render_state_path),
                "drake_directive": _file_record(render_directive_path),
            }
            try:
                render_state = json.loads(render_state_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("Cannot parse current accepted render state") from exc
            render_objects = render_state.get("objects")
            if not isinstance(render_objects, dict) or not input_object_ids.issubset(
                render_objects
            ):
                raise RuntimeError("Current accepted render object inventory is invalid")
            self._verify_input_object_semantics(
                input_scene_state=input_scene_state,
                output_scene_state=render_state,
                target_furniture_id=target_id,
            )
            accepted_added_ids = set(render_objects) - input_object_ids
            if accepted_added_ids != added_ids:
                raise RuntimeError(
                    "Accepted manipulands did not survive post-processing exactly"
                )
            render_asset_files = _referenced_room_asset_records(
                render_state,
                room_root=Path(self.scene.scene_dir),
                object_ids=set(referenced_object_ids),
            )
            if render_asset_files != referenced_asset_files:
                raise RuntimeError(
                    "Accepted manipuland assets changed during post-processing"
                )
        else:
            self._verify_legacy_checkpoint_source(
                legacy_source,
                accepted_evidence=accepted_evidence,
                scene_state=output_state,
                furniture_selection=furniture_selection,
                furniture_index=furniture_index,
                input_scene_hash=input_scene_hash,
                input_object_ids=input_object_ids,
            )
        stage = final.with_name(f".{final.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
        if stage.exists():
            raise RuntimeError(f"Checkpoint staging path already exists: {stage}")
        try:
            stage.mkdir()
            self.logger.log_scene(scene=self.scene, output_dir=stage)
            if {path.name for path in stage.iterdir()} != {
                "scene_state.json",
                "scene.dmd.yaml",
            }:
                raise RuntimeError("Scene logger emitted unexpected checkpoint artifacts")
            _fsync_file(stage / "scene_state.json")
            _fsync_file(stage / "scene.dmd.yaml")
            staged_state = json.loads(
                (stage / "scene_state.json").read_text(encoding="utf-8")
            )
            self._verify_input_object_semantics(
                input_scene_state=input_scene_state,
                output_scene_state=staged_state,
                target_furniture_id=target_id,
            )
            if protected_asset_files is not None and protected_asset_ids:
                _verify_referenced_asset_records(
                    protected_asset_files,
                    scene_state=staged_state,
                    room_root=Path(self.scene.scene_dir),
                    object_ids=set(protected_asset_ids),
                )
            _verify_referenced_asset_records(
                referenced_asset_files,
                scene_state=staged_state,
                room_root=Path(self.scene.scene_dir),
                object_ids=set(referenced_object_ids),
            )
            serialized_content_hash = _serialized_scene_content_hash(
                self.scene, staged_state
            )
            state_record = _file_record(stage / "scene_state.json")
            directive_record = _file_record(stage / "scene.dmd.yaml")
            receipt = {
                "schema_version": _FURNITURE_CHECKPOINT_SCHEMA,
                "status": "pass",
                **self._checkpoint_context(furniture_selection, furniture_index),
                "input_scene_content_hash": input_scene_hash,
                "output_scene_content_hash": serialized_content_hash,
                "accepted_evidence": accepted_evidence,
                "added_manipuland_ids": sorted(added_ids),
                "referenced_asset_object_ids": referenced_object_ids,
                "referenced_asset_files": referenced_asset_files,
                "asset_registry_snapshot": (
                    self.asset_manager.registry.snapshot()
                    if getattr(self, "asset_manager", None) is not None
                    else AssetRegistry.empty_snapshot()
                ),
                "legacy_source": legacy_source,
                "artifacts": {
                    "scene_state": state_record,
                    "drake_directive": directive_record,
                },
            }
            receipt["attestation"] = _receipt_attestation(receipt)
            _write_json_fsynced(stage / "completion_receipt.json", receipt)
            _verify_referenced_asset_records(
                referenced_asset_files,
                scene_state=output_state,
                room_root=Path(self.scene.scene_dir),
                object_ids=set(referenced_object_ids),
            )
            if (
                _accepted_render_evidence(
                    accepted_dir, minimum_score=self._minimum_accepted_score()
                )
                != accepted_evidence
            ):
                raise RuntimeError(
                    "Copied manipuland evidence changed during checkpoint publication"
                )
            if legacy_source is not None:
                self._verify_legacy_checkpoint_source(
                    legacy_source,
                    accepted_evidence=accepted_evidence,
                    scene_state=output_state,
                    furniture_selection=furniture_selection,
                    furniture_index=furniture_index,
                    input_scene_hash=input_scene_hash,
                    input_object_ids=input_object_ids,
                )
            elif (
                _accepted_render_evidence(
                    Path(render_dir),
                    minimum_score=self._minimum_accepted_score(),
                    allowed_extra=frozenset({"scene_state.json", "scene.dmd.yaml"}),
                )
                != accepted_evidence
            ):
                raise RuntimeError(
                    "Current render evidence changed during checkpoint publication"
                )
            if normal_render_records is not None:
                self._verify_file_record(
                    Path(render_dir) / "scene_state.json",
                    normal_render_records["scene_state"],
                    label="current render scene state",
                )
                self._verify_file_record(
                    Path(render_dir) / "scene.dmd.yaml",
                    normal_render_records["drake_directive"],
                    label="current render Drake directive",
                )
            _fsync_directory(stage)
            stage.rename(final)
            _fsync_directory(final.parent)
        except Exception:
            if stage.exists():
                shutil.rmtree(stage)
                _fsync_directory(final.parent)
            raise
        # Continue from the exact serialized state that a resumed process will
        # load.  This prevents the next furniture's input hash from depending on
        # pre-serialization quaternion/numeric representation.
        self.scene.restore_from_state_dict(staged_state)
        if self.scene.content_hash() != serialized_content_hash:
            raise RuntimeError(
                "Published manipuland checkpoint round-trip hash mismatch"
            )
        console_logger.info(
            "Published durable manipuland checkpoint for %s: %s",
            furniture_selection.furniture_id,
            final,
        )

    async def add_manipulands(self, scene: RoomScene) -> None:
        """Add manipulands to furniture surfaces in the scene.

        This method implements a two-phase workflow:
        1. VLM-based furniture analysis to identify which pieces need manipulands
        2. Per-furniture multi-agent workflow (planner/designer/critic) to
           populate selected furniture with appropriate small objects

        The scene is mutated in place to add manipuland objects. Fresh agent
        contexts are created for each furniture piece to bound token usage.

        Side effects:
        - Scene objects are added (manipulands placed on furniture)
        - Support surfaces are extracted and assigned to furniture
        - Render cache is cleared before processing
        - Per-furniture subdirectories created under logger output directory
        - Checkpoint state saved after each critique iteration
        - Final scores copied to furniture_<id>/final_scene/ directories

        Requirements:
        - Furniture must have geometry_path (non-None)
        - Furniture must have valid bounding boxes (bbox_min, bbox_max)
        - Scene must have text_description for agent context

        Args:
            scene: RoomScene with furniture already placed. Furniture objects must
                have geometry and bounding boxes to be considered for manipuland
                placement.

        Raises:
            Exception: If a selected furniture target, its support extraction,
                its workflow, post-processing, or checkpoint publication fails.
        """
        console_logger.info("Starting manipuland placement")
        self.scene = scene

        # Clear render cache to ensure fresh renders for manipulands.
        # This prevents cache key collisions when object IDs are reused.
        self.rendering_manager.clear_cache()

        # Phase 1: restore an exact prior plan, or analyze once and durably bind
        # the complete ordered selection/context before mutating the scene.
        initial_scene_hash = self.scene.content_hash()
        furniture_data = self._load_furniture_plan(initial_scene_hash)
        new_plan = furniture_data is None
        if furniture_data is None:
            furniture_data = await self._analyze_furniture_for_placement(scene)
            furniture_data = self._apply_furniture_target_limit(scene, furniture_data)

        self._validate_explicit_legacy_plan(furniture_data)
        if not furniture_data:
            console_logger.info("No furniture identified for manipuland placement")
            return

        console_logger.info(
            f"Identified {len(furniture_data)} furniture pieces to populate"
        )

        # Phase 1b: Select context furniture for each selection.
        if new_plan and self.cfg.context_furniture.enabled:
            # Get path to furniture_selection images (already rendered).
            furniture_selection_dir = (
                self.rendering_manager._base_output_dir
                / "scene_renders"
                / "furniture_selection"
            )
            images_dir = (
                furniture_selection_dir if furniture_selection_dir.exists() else None
            )

            context_map = self.scene_analyzer.select_context_furniture(
                scene=scene,
                furniture_selections=furniture_data,
                furniture_selection_images_dir=images_dir,
            )

            # Attach context to each selection.
            for selection in furniture_data:
                selection.context_furniture_ids = context_map.get(
                    selection.furniture_id, []
                )

        if new_plan:
            self._publish_furniture_plan(initial_scene_hash, furniture_data)

        # Phase 2: Per-furniture loop.
        for furniture_index, furniture_selection in enumerate(furniture_data):
            furniture_id = furniture_selection.furniture_id
            # Create custom span for this furniture's manipuland placement.
            with custom_span(
                name=f"manipulands_{furniture_id}",
                data={"furniture_id": str(furniture_id)},
            ):
                console_logger.info(f"Populating furniture: {furniture_id}")
                if furniture_selection.suggested_items:
                    console_logger.info(
                        f"Suggested items: {furniture_selection.suggested_items}"
                    )
                    console_logger.info(
                        f"Prompt constraints: {furniture_selection.prompt_constraints}"
                    )
                    console_logger.info(
                        f"Style notes: {furniture_selection.style_notes}"
                    )

                # Extract support surface for this furniture.
                furniture = self.scene.get_object(furniture_id)
                if not furniture:
                    raise RuntimeError(
                        f"Selected furniture {furniture_id} is missing from the scene"
                    )

                input_scene_state = self.scene.to_state_dict()
                input_objects = input_scene_state.get("objects")
                if not isinstance(input_objects, dict):
                    raise RuntimeError("Manipuland input scene has no object mapping")
                input_object_ids = set(input_objects)
                input_scene_hash = self.scene.content_hash()
                if self._restore_completed_furniture_checkpoint(
                    furniture_selection=furniture_selection,
                    furniture_index=furniture_index,
                    input_scene_hash=input_scene_hash,
                ):
                    continue

                self._begin_furniture_scope(
                    furniture_id=furniture_id,
                    input_scene_state=input_scene_state,
                )
                try:
                    legacy_source = self._restore_legacy_accepted_render(
                        furniture_selection,
                        furniture_index,
                        input_scene_hash,
                    )
                    if legacy_source is not None:
                        if not self.cfg.per_furniture_postprocessing.enabled:
                            raise RuntimeError(
                                "Legacy accepted render cannot be promoted without "
                                "per-furniture post-processing"
                            )
                        self._verify_active_furniture_scope()
                        current_ids = self._current_scope_manipuland_ids(furniture_id)
                        self._run_per_furniture_postprocessing(
                            furniture_id, current_ids
                        )
                        self._verify_active_furniture_scope()
                        self._publish_furniture_checkpoint(
                            furniture_selection=furniture_selection,
                            furniture_index=furniture_index,
                            input_scene_hash=input_scene_hash,
                            legacy_source=legacy_source,
                            input_object_ids=input_object_ids,
                            input_scene_state=input_scene_state,
                        )
                        required = self._required_legacy_recovery
                        if required is not None and required[:2] == (
                            furniture_index,
                            str(furniture_id),
                        ):
                            self._legacy_recovery_satisfied = True
                        continue

                    # Extract surfaces for this target only.  Propagating to identical
                    # furniture would mutate protected input objects.
                    hsm_config = SupportSurfaceExtractionConfig.from_config(
                        cfg=self.cfg.support_surface_extraction
                    )
                    surfaces = extract_and_propagate_support_surfaces(
                        scene=self.scene,
                        furniture_object=furniture,
                        config=hsm_config,
                        propagate_to_identical=False,
                    )

                    console_logger.info(
                        f"Extracted {len(surfaces)} support surface(s) for {furniture_id}"
                    )
                    if not surfaces:
                        if self._allows_noop_without_support_surfaces(
                            furniture_selection
                        ):
                            console_logger.info(
                                "Skipping optional zero-surface manipuland target %s; "
                                "its bound prompt requires no manipulands",
                                furniture_id,
                            )
                            continue
                        raise RuntimeError(
                            f"Selected furniture {furniture_id} has no usable "
                            "support surfaces"
                        )
                    self._verify_active_furniture_scope()

                    # Set up per-furniture context.
                    self._setup_furniture_context(furniture_selection)

                    # Generate context image for manipuland placement (if enabled).
                    self.manipuland_context_image_path = (
                        self._generate_manipuland_context_image()
                    )

                    # Initialize checkpoint state.
                    self._initialize_checkpoint_state()

                    # Get furniture description for agent prompts.
                    furniture_obj = self.scene.get_object(furniture_id)
                    furniture_description = (
                        furniture_obj.description if furniture_obj else "furniture"
                    )

                    # Create agents and sessions.
                    self._setup_furniture_agents(
                        furniture_id=furniture_id,
                        furniture_description=furniture_description,
                    )

                    # Run multi-agent workflow.
                    await self._run_furniture_workflow(furniture_id)
                    self._verify_active_furniture_scope()

                    # Per-furniture post-processing is part of the durable output,
                    # so the checkpoint is published only after it succeeds.
                    current_ids = self._current_scope_manipuland_ids(furniture_id)
                    self._run_per_furniture_postprocessing(furniture_id, current_ids)
                    self._verify_active_furniture_scope()
                    self._publish_furniture_checkpoint(
                        furniture_selection=furniture_selection,
                        furniture_index=furniture_index,
                        input_scene_hash=input_scene_hash,
                        legacy_source=None,
                        input_object_ids=input_object_ids,
                        input_scene_state=input_scene_state,
                    )

                except Exception as e:
                    raise RuntimeError(
                        f"Required manipuland target {furniture_id} failed"
                    ) from e
                finally:
                    self._clear_furniture_scope()

        if (
            self._required_legacy_recovery is not None
            and not self._legacy_recovery_satisfied
        ):
            raise RuntimeError("Required legacy recovery was not consumed or restored")
        console_logger.info("Manipuland placement complete")

    def _apply_furniture_target_limit(
        self, scene: RoomScene, furniture_data: list[FurnitureSelection]
    ) -> list[FurnitureSelection]:
        """Limit duplicated room targets for production room-worker throughput."""
        raw_limit = os.environ.get("SCENESMITH_MANIPULAND_MAX_FURNITURE", "").strip()
        if not raw_limit:
            return furniture_data

        try:
            limit = int(raw_limit)
        except ValueError:
            console_logger.warning(
                "Ignoring invalid SCENESMITH_MANIPULAND_MAX_FURNITURE=%r", raw_limit
            )
            return furniture_data

        if limit <= 0 or len(furniture_data) <= limit:
            return furniture_data

        def selection_text(selection: FurnitureSelection) -> str:
            obj = scene.get_object(selection.furniture_id)
            return (
                f"{selection.furniture_id} "
                f"{getattr(obj, 'name', '') if obj else ''} "
                f"{getattr(obj, 'description', '') if obj else ''}"
            ).lower()

        def take_matching(
            candidates: list[FurnitureSelection],
            count: int,
            predicate: Any,
        ) -> list[FurnitureSelection]:
            picked: list[FurnitureSelection] = []
            for candidate in candidates:
                if len(picked) >= count:
                    break
                if predicate(selection_text(candidate)):
                    picked.append(candidate)
            return picked

        limited: list[FurnitureSelection] = []
        limited.extend(
            take_matching(furniture_data, 1, lambda text: "teacher" in text)
        )
        limited.extend(
            item
            for item in take_matching(
                furniture_data,
                2,
                lambda text: "student" in text and "desk" in text,
            )
            if item not in limited
        )
        remaining = limit - len(limited)
        if remaining > 0:
            limited.extend(
                item
                for item in take_matching(
                    furniture_data,
                    remaining,
                    lambda text: any(
                        term in text
                        for term in (
                            "shelf",
                            "bookshelf",
                            "cabinet",
                            "storage",
                            "bookcase",
                            "counter",
                        )
                    ),
                )
                if item not in limited
            )
        if len(limited) < limit:
            limited.extend(
                item
                for item in furniture_data
                if item not in limited
            )
        limited = limited[:limit]
        limited = sorted(limited, key=furniture_data.index)
        console_logger.info(
            "Limited manipuland furniture targets from %d to %d via "
            "SCENESMITH_MANIPULAND_MAX_FURNITURE=%d: %s",
            len(furniture_data),
            len(limited),
            limit,
            [str(item.furniture_id) for item in limited],
        )
        return limited

    @staticmethod
    def _allows_noop_without_support_surfaces(
        furniture_selection: FurnitureSelection,
    ) -> bool:
        """Whether a zero-surface target is explicitly optional in the prompt.

        A target with required manipulands must still fail closed if its model has
        no usable support surface. Some fixtures, such as a rolling whiteboard,
        explicitly say that no manipulands are required; the furniture itself is
        then the completed visual result and must not stop every later target.
        """
        constraints = furniture_selection.prompt_constraints.casefold()
        return "no specific manipulands required" in constraints

    async def _analyze_furniture_for_placement(
        self, scene: RoomScene
    ) -> list[FurnitureSelection]:
        """Analyze which furniture should have manipulands.

        Delegates to SceneAnalyzer for VLM-based furniture selection.

        Args:
            scene: RoomScene with furniture.

        Returns:
            List of FurnitureSelection objects with assignment context.
        """
        return self.scene_analyzer.analyze_furniture_for_manipulands(
            scene=scene,
            prompt_enum=ManipulandAgentPrompts.ANALYZE_FURNITURE_FOR_PLACEMENT,
        )
