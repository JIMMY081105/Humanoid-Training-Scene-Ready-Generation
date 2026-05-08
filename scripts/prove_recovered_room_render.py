#!/usr/bin/env python3
"""Prove a recovered room renders through Drake and Blender without model calls.

The production room worker has a deliberately narrow ``--refresh-final-blend``
branch: it restores one existing state, starts only the local Blender server, and
exports Drake's RenderEngineGltfClient payload to a blend file.  This harness puts
an earlier recovered checkpoint into a private shadow run, invokes that branch,
then invokes the existing three-view Blender review renderer.

No source checkpoint or asset is modified.  The child environment has paid-model
credentials removed and all non-loopback proxy traffic pointed at a closed local
port.  Source trees are hashed before and after the proof, and the resulting blend,
three distinct PNGs, cutaway evidence, logs, code, and exact prompt bytes are bound
in a receipt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import traceback
import uuid
import xml.etree.ElementTree as ET

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlsplit


SCHEMA_ID = "scenesmith_recovered_room_real_render_proof_v2"
SCHEMA_VERSION = 2
VIEW_NAMES = ("top", "oblique_a", "oblique_b")
FORBIDDEN_CREDENTIALS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DASHSCOPE_API_KEY",
    "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)
PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
RUN_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ProofError(RuntimeError):
    """Raised when the isolated real-render proof cannot be attested."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_regular_file(path: Path, label: str) -> Path:
    candidate = path.expanduser().resolve(strict=True)
    info = candidate.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ProofError(f"{label} must be an unlinked regular file: {path}")
    return candidate


def _require_directory(path: Path, label: str) -> Path:
    candidate = path.expanduser().resolve(strict=True)
    info = candidate.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISDIR(info.st_mode):
        raise ProofError(f"{label} must be an unlinked directory: {path}")
    return candidate


def _file_record(path: Path, label: str) -> dict[str, Any]:
    candidate = _require_regular_file(path, label)
    size = candidate.stat().st_size
    if size <= 0:
        raise ProofError(f"{label} is empty: {candidate}")
    return {
        "path": str(candidate),
        "size_bytes": size,
        "sha256": _sha256_file(candidate),
    }


def _regular_tree_files(root: Path) -> list[Path]:
    canonical = _require_directory(root, "Source tree")
    files: list[Path] = []
    casefold_paths: set[str] = set()
    for current, directories, names in os.walk(canonical, followlinks=False):
        current_path = Path(current)
        for name in (*directories, *names):
            candidate = current_path / name
            info = candidate.stat(follow_symlinks=False)
            if candidate.is_symlink():
                raise ProofError(f"Tree contains a symbolic link: {candidate}")
            if name in directories:
                if not stat.S_ISDIR(info.st_mode):
                    raise ProofError(f"Tree contains a special directory: {candidate}")
                continue
            if not stat.S_ISREG(info.st_mode):
                raise ProofError(f"Tree contains a special file: {candidate}")
            relative = candidate.relative_to(canonical).as_posix()
            folded = relative.casefold()
            if folded in casefold_paths:
                raise ProofError(f"Tree has a case-fold path collision: {relative}")
            casefold_paths.add(folded)
            files.append(candidate)
    files.sort(key=lambda item: item.relative_to(canonical).as_posix())
    if not files:
        raise ProofError(f"Tree is empty: {canonical}")
    return files


def _tree_record(root: Path, label: str) -> dict[str, Any]:
    canonical = _require_directory(root, label)
    files = _regular_tree_files(canonical)
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(canonical).as_posix().encode("utf-8")
        size = path.stat().st_size
        total_bytes += size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(_sha256_file(path)))
    return {
        "path": str(canonical),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def _copy_regular_tree(source: Path, destination: Path) -> None:
    source_record = _tree_record(source, "Tree to copy")
    if destination.exists() or destination.is_symlink():
        raise ProofError(f"Copy destination already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    source_root = Path(source_record["path"])
    for source_file in _regular_tree_files(source_root):
        relative = source_file.relative_to(source_root)
        destination_file = destination / relative
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination_file, follow_symlinks=False)
    copied = _tree_record(destination, "Copied tree")
    comparable = ("file_count", "total_bytes", "sha256")
    if any(copied[key] != source_record[key] for key in comparable):
        raise ProofError(
            f"Copied tree differs from source: {source_root} -> {destination}"
        )


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)


def _marker_root(value: str, marker: str) -> Path | None:
    normalized = value.replace("\\", "/")
    token = f"/{marker}/"
    if token in normalized:
        prefix = normalized.split(token, 1)[0]
        return Path(f"{prefix}/{marker}")
    if normalized.startswith(f"{marker}/"):
        return None
    return None


def _discover_source_tree(
    state: Mapping[str, Any], *, marker: str, source_room: Path
) -> Path:
    roots = {
        root.resolve()
        for value in _iter_strings(state)
        if (root := _marker_root(value, marker)) is not None
    }
    relative_root = source_room / marker
    if any(value.replace("\\", "/").startswith(f"{marker}/") for value in _iter_strings(state)):
        roots.add(relative_root.resolve())
    if not roots and relative_root.is_dir():
        roots.add(relative_root.resolve())
    if len(roots) != 1:
        raise ProofError(
            f"Checkpoint must resolve exactly one {marker!r} source tree, got "
            f"{sorted(map(str, roots))}"
        )
    return _require_directory(next(iter(roots)), f"Checkpoint {marker} tree")


def _rebase_absolute_marker(
    value: str,
    *,
    marker: str,
    destination_root: Path,
) -> str:
    normalized = value.replace("\\", "/")
    token = f"/{marker}/"
    if not Path(value).is_absolute() or token not in normalized:
        return value
    suffix = normalized.split(token, 1)[1]
    return str(destination_root / suffix)


def _rebase_state_paths(
    state: Mapping[str, Any],
    *,
    shadow_room: Path,
    shadow_scene: Path,
) -> dict[str, Any]:
    """Rebase only copied room/scene resource markers in a deep copy."""

    original_prompt = state.get("text_description")

    def visit(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, str):
            return copy.deepcopy(value)
        result = _rebase_absolute_marker(
            value,
            marker="generated_assets",
            destination_root=shadow_room / "generated_assets",
        )
        result = _rebase_absolute_marker(
            result,
            marker="room_geometry",
            destination_root=shadow_scene / "room_geometry",
        )
        result = _rebase_absolute_marker(
            result,
            marker="floor_plans",
            destination_root=shadow_scene / "floor_plans",
        )
        return result

    rebased = visit(state)
    if rebased.get("text_description") != original_prompt:
        raise ProofError("Path rebasing changed the immutable room prompt")
    return rebased


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_shadow_run_destination(repo: Path, output_dir: Path) -> Path:
    """Require the proof run at the exact depth used by production runs.

    Generated floor GLTFs intentionally contain repository-relative material
    URIs.  Adding even one directory between ``outputs/<date>/<run>`` and
    ``scene_000`` changes what those URIs resolve to.  The proof therefore uses
    ``output_dir`` itself as the run root and refuses every other depth.
    """

    outputs = _require_directory(repo / "outputs", "SceneSmith outputs root")
    destination = output_dir.expanduser().resolve(strict=False)
    if destination.parent.parent != outputs:
        raise ProofError(
            "Proof output must itself be a run root at "
            f"{outputs}/<YYYY-MM-DD>/<unique_run>, got {destination}"
        )
    if not RUN_DATE_PATTERN.fullmatch(destination.parent.name):
        raise ProofError(
            "Proof output date directory must use YYYY-MM-DD: "
            f"{destination.parent}"
        )
    if not destination.name or destination.name in {".", ".."}:
        raise ProofError("Proof output has no unique run name")
    return destination


def _iter_json_uris(value: Any, pointer: str = "$") -> Iterable[tuple[str, str]]:
    """Yield every URI-bearing JSON field, including extension-owned fields."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{pointer}/{key}"
            if key == "uri":
                if not isinstance(item, str) or not item:
                    raise ProofError(f"GLTF has an invalid URI at {child}")
                yield child, item
            else:
                yield from _iter_json_uris(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_json_uris(item, f"{pointer}/{index}")


def _resolve_external_uri(
    raw_uri: str,
    *,
    relative_to: Path,
    label: str,
    repo: Path,
    forbidden_roots: Sequence[Path],
    allow_data: bool,
) -> Path | None:
    parsed = urlsplit(raw_uri)
    if parsed.scheme == "data":
        if not allow_data:
            raise ProofError(f"{label} must be an external local URI: {raw_uri}")
        return None
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ProofError(f"{label} is not a plain local URI: {raw_uri}")
    decoded = unquote(parsed.path)
    if not decoded or "\\" in decoded or "\x00" in decoded:
        raise ProofError(f"{label} is unsafe: {raw_uri}")
    relative = Path(decoded)
    if relative.is_absolute():
        raise ProofError(f"{label} must be relative: {raw_uri}")
    try:
        target = (relative_to / relative).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProofError(f"{label} is unresolved: {raw_uri}: {exc}") from exc
    if not _is_within(target, repo):
        raise ProofError(f"{label} escapes the SceneSmith repository: {raw_uri}")
    if any(_is_within(target, root) for root in forbidden_roots):
        raise ProofError(f"{label} resolves into an unbound live source tree: {raw_uri}")
    return _require_regular_file(target, label)


def _scoped_repo_path(path: Path, *, repo: Path, shadow_run: Path) -> dict[str, str]:
    if _is_within(path, shadow_run):
        return {
            "scope": "shadow_run",
            "path": path.relative_to(shadow_run).as_posix(),
        }
    return {"scope": "repo", "path": path.relative_to(repo).as_posix()}


def _gltf_external_uri_manifest(
    *,
    repo: Path,
    shadow_run: Path,
    bound_roots: Sequence[Path],
    forbidden_source_roots: Sequence[Path],
) -> dict[str, Any]:
    """Recursively resolve and hash every external URI used by copied GLTFs."""

    canonical_repo = _require_directory(repo, "SceneSmith repository")
    canonical_run = _require_directory(shadow_run, "Shadow run")
    canonical_forbidden = tuple(
        root.expanduser().resolve(strict=True) for root in forbidden_source_roots
    )
    queue: list[Path] = []
    for root in bound_roots:
        canonical_root = _require_directory(root, "Bound shadow resource tree")
        if not _is_within(canonical_root, canonical_run):
            raise ProofError(f"Bound resource tree is outside the shadow run: {root}")
        queue.extend(
            path
            for path in _regular_tree_files(canonical_root)
            if path.suffix.lower() == ".gltf"
        )
    queue = sorted(set(queue), key=str)
    if not queue:
        raise ProofError("Bound shadow resources contain no GLTF documents")

    seen: set[Path] = set()
    gltf_records: list[dict[str, Any]] = []
    uri_records: list[dict[str, Any]] = []
    while queue:
        gltf = _require_regular_file(queue.pop(0), "Bound GLTF document")
        if gltf in seen:
            continue
        seen.add(gltf)
        try:
            document = json.loads(gltf.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProofError(f"Cannot parse bound GLTF {gltf}: {exc}") from exc
        if not isinstance(document, Mapping):
            raise ProofError(f"Bound GLTF is not a JSON object: {gltf}")
        gltf_scope = _scoped_repo_path(gltf, repo=canonical_repo, shadow_run=canonical_run)
        gltf_records.append(
            {
                **gltf_scope,
                "size_bytes": gltf.stat().st_size,
                "sha256": _sha256_file(gltf),
            }
        )
        for pointer, raw_uri in _iter_json_uris(document):
            target = _resolve_external_uri(
                raw_uri,
                relative_to=gltf.parent,
                label=f"GLTF {gltf} URI {pointer}",
                repo=canonical_repo,
                forbidden_roots=canonical_forbidden,
                allow_data=True,
            )
            if target is None:
                continue
            target_scope = _scoped_repo_path(
                target, repo=canonical_repo, shadow_run=canonical_run
            )
            uri_records.append(
                {
                    "gltf_scope": gltf_scope["scope"],
                    "gltf": gltf_scope["path"],
                    "json_pointer": pointer,
                    "uri": raw_uri,
                    "target_scope": target_scope["scope"],
                    "target": target_scope["path"],
                    "size_bytes": target.stat().st_size,
                    "sha256": _sha256_file(target),
                }
            )
            if target.suffix.lower() == ".gltf" and target not in seen:
                queue.append(target)
                queue.sort(key=str)
    return {
        "gltf_count": len(gltf_records),
        "external_uri_count": len(uri_records),
        "gltf_files": sorted(
            gltf_records, key=lambda item: (item["scope"], item["path"])
        ),
        "external_uris": sorted(
            uri_records,
            key=lambda item: (
                item["gltf_scope"],
                item["gltf"],
                item["json_pointer"],
                item["uri"],
            ),
        ),
    }


def _local_name(tag: Any) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _validate_artiverse_visual_gltf_bindings(
    state: Mapping[str, Any], *, shadow_room: Path, repo: Path, source_roots: Sequence[Path]
) -> list[dict[str, Any]]:
    """Require every used Artiverse visual to use the derived external GLTF."""

    generated_root = _require_directory(
        shadow_room / "generated_assets", "Copied generated assets"
    )
    objects = state.get("objects")
    if not isinstance(objects, Mapping):
        raise ProofError("Shadow state has no objects for Artiverse visual validation")
    records: list[dict[str, Any]] = []
    used_artiverse = 0
    for object_id, raw_object in sorted(objects.items(), key=lambda item: str(item[0])):
        if not isinstance(raw_object, Mapping):
            raise ProofError(f"Shadow object is malformed: {object_id}")
        metadata = raw_object.get("metadata")
        if not isinstance(metadata, Mapping) or str(
            metadata.get("articulated_source", "")
        ).lower() != "artiverse":
            continue
        used_artiverse += 1
        sdf = _resolve_state_path(raw_object.get("sdf_path"), shadow_room, f"{object_id}.sdf_path")
        if not _is_within(sdf, generated_root):
            raise ProofError(f"Used Artiverse SDF is outside copied generated_assets: {sdf}")
        try:
            root = ET.fromstring(sdf.read_bytes())
        except (OSError, ET.ParseError) as exc:
            raise ProofError(f"Cannot parse used Artiverse SDF {sdf}: {exc}") from exc
        visual_count = 0
        for visual_index, visual in enumerate(
            element for element in root.iter() if _local_name(element.tag) == "visual"
        ):
            uris = [
                (element.text or "").strip()
                for element in visual.iter()
                if _local_name(element.tag) == "uri"
            ]
            if len(uris) != 1 or not uris[0]:
                raise ProofError(
                    f"Used Artiverse visual {object_id}[{visual_index}] must have one mesh URI"
                )
            raw_uri = uris[0]
            if not urlsplit(raw_uri).path.lower().endswith(".gltf"):
                raise ProofError(
                    f"Used Artiverse visual is not the derived external GLTF: "
                    f"{object_id}[{visual_index}]={raw_uri}"
                )
            target = _resolve_external_uri(
                raw_uri,
                relative_to=sdf.parent,
                label=f"Used Artiverse visual {object_id}[{visual_index}]",
                repo=repo,
                forbidden_roots=source_roots,
                allow_data=False,
            )
            assert target is not None
            if not _is_within(target, generated_root):
                raise ProofError(
                    f"Used Artiverse visual escapes copied generated_assets: {target}"
                )
            records.append(
                {
                    "object_id": str(object_id),
                    "sdf": sdf.relative_to(shadow_room).as_posix(),
                    "visual_index": visual_index,
                    "uri": raw_uri,
                    "target": target.relative_to(shadow_room).as_posix(),
                    "size_bytes": target.stat().st_size,
                    "sha256": _sha256_file(target),
                }
            )
            visual_count += 1
        if visual_count == 0:
            raise ProofError(f"Used Artiverse object has no visual meshes: {object_id}")
    if used_artiverse == 0:
        raise ProofError("Recovered shadow state uses no Artiverse object")
    return records


def _assert_no_ignored_glb_meshes(log: Path) -> None:
    candidate = _require_regular_file(log, "Render log")
    try:
        lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise ProofError(f"Cannot inspect render log {candidate}: {exc}") from exc
    unsupported_signature = (
        "renderenginegltfclient only supports mesh specifications which use "
        ".obj or .gltf files"
    )
    failures = [
        line
        for line in lines
        if (
            ".glb" in line.lower()
            and "ignor" in line.lower()
        )
        or unsupported_signature in line.lower()
    ]
    if failures:
        raise ProofError(
            "Render logs report ignored GLB meshes: " + " | ".join(failures[:10])
        )


def _audit_no_source_scene_paths(
    state: Mapping[str, Any], *, source_roots: Iterable[Path]
) -> None:
    """Reject any absolute shadow-state string still rooted in live scene data."""

    canonical_roots = sorted(
        {root.expanduser().resolve() for root in source_roots}, key=str
    )
    stale: list[str] = []
    for value in _iter_strings(state):
        candidate = Path(value)
        if not candidate.is_absolute():
            continue
        resolved = candidate.expanduser().resolve(strict=False)
        if any(_is_within(resolved, root) for root in canonical_roots):
            stale.append(value)
    if stale:
        raise ProofError(
            "Shadow state still references unbound source scene/run paths: "
            + "; ".join(sorted(set(stale)))
        )


def _resolve_state_path(value: Any, shadow_room: Path, label: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ProofError(f"{label} is not a valid path")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = shadow_room / candidate
    return _require_regular_file(candidate, label)


def _validate_render_dependencies(state: Mapping[str, Any], shadow_room: Path) -> None:
    room_geometry = state.get("room_geometry")
    if not isinstance(room_geometry, Mapping):
        raise ProofError("Shadow state has no room_geometry object")
    _resolve_state_path(
        room_geometry.get("sdf_path"), shadow_room, "Room geometry SDF"
    )

    def validate_nested_paths(value: Any, label: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                child_label = f"{label}.{key}"
                if key in {"geometry_path", "sdf_path", "image_path"}:
                    if item is not None:
                        _resolve_state_path(item, shadow_room, child_label)
                else:
                    validate_nested_paths(item, child_label)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                validate_nested_paths(item, f"{label}[{index}]")

    # In particular this binds ``room_geometry.floor.geometry_path`` under the
    # copied floor_plans tree; walls and future nested geometry are covered too.
    validate_nested_paths(room_geometry, "room_geometry")
    objects = state.get("objects")
    if not isinstance(objects, Mapping) or not objects:
        raise ProofError("Shadow state has no objects")
    for object_id, raw_object in objects.items():
        if not isinstance(raw_object, Mapping):
            raise ProofError(f"Shadow object is malformed: {object_id}")
        for key in ("geometry_path", "sdf_path", "image_path"):
            if raw_object.get(key) is not None:
                _resolve_state_path(
                    raw_object.get(key), shadow_room, f"{object_id}.{key}"
                )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _sanitized_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if source is None else source)
    for name in FORBIDDEN_CREDENTIALS:
        environment.pop(name, None)
    dead_proxy = "http://127.0.0.1:9"
    for name in PROXY_VARIABLES:
        environment[name] = dead_proxy
    environment["NO_PROXY"] = "127.0.0.1,localhost,::1"
    environment["no_proxy"] = "127.0.0.1,localhost,::1"
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["SCENESMITH_REAL_RENDER_PROOF"] = "1"
    return environment


def _run_logged(
    command: Sequence[str], *, cwd: Path, environment: Mapping[str, str], log: Path
) -> None:
    with log.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("command=" + json.dumps(list(command)) + "\n")
        stream.flush()
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        stream.write(f"\nexit_code={completed.returncode}\n")
    if completed.returncode != 0:
        raise ProofError(
            f"Proof command exited {completed.returncode}; inspect {log}"
        )


def _validate_review_evidence(
    evidence_path: Path,
    *,
    expected_state: Path,
    expected_blend: Path,
) -> dict[str, Any]:
    evidence_record = _file_record(evidence_path, "Cutaway evidence")
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProofError(f"Cannot parse cutaway evidence: {exc}") from exc
    if evidence.get("status") != "pass" or evidence.get("rendered_view_count") != 3:
        raise ProofError("Cutaway renderer did not report an exact three-view pass")
    views = evidence.get("views")
    if not isinstance(views, list) or [view.get("view_name") for view in views] != list(
        VIEW_NAMES
    ):
        raise ProofError("Cutaway evidence has the wrong view sequence")
    hashes = [view.get("image_sha256") for view in views]
    if any(not isinstance(value, str) or len(value) != 64 for value in hashes):
        raise ProofError("Cutaway evidence has malformed image hashes")
    if len(set(hashes)) != 3:
        raise ProofError("Cutaway views are not byte-distinct")
    derivation = evidence.get("derivation_receipt")
    if not isinstance(derivation, Mapping):
        raise ProofError("Cutaway evidence has no state/blend derivation receipt")
    source_state = derivation.get("source_state")
    source_blend = derivation.get("source_blend")
    if not isinstance(source_state, Mapping) or not isinstance(source_blend, Mapping):
        raise ProofError("Cutaway derivation receipt is malformed")
    state_record = _file_record(expected_state, "Shadow state")
    blend_record = _file_record(expected_blend, "Shadow blend")
    for actual, expected, label in (
        (source_state, state_record, "state"),
        (source_blend, blend_record, "blend"),
    ):
        if actual.get("sha256") != expected["sha256"] or actual.get(
            "size_bytes"
        ) != expected["size_bytes"]:
            raise ProofError(f"Cutaway derivation is not bound to the shadow {label}")
    for view in views:
        image = _require_regular_file(Path(str(view["image"])), "Review image")
        if _sha256_file(image) != view["image_sha256"]:
            raise ProofError(f"Review image hash is stale: {image}")
    return {"record": evidence_record, "document": evidence}


def _command_records(commands: Sequence[Sequence[str]]) -> list[list[str]]:
    return [[str(component) for component in command] for command in commands]


def prove_render(
    *,
    repo_dir: Path,
    source_room: Path,
    source_checkpoint: str,
    output_dir: Path,
    csv: Path,
    room_id: str,
    port_offset: int,
    materials_data: Path,
    materials_source_embeddings: Path,
    materials_embeddings: Path,
    python_executable: Path | None = None,
) -> Path:
    repo = _require_directory(repo_dir, "SceneSmith repository")
    room = _require_directory(source_room, "Recovered source room")
    if room.name != f"room_{room_id}":
        raise ProofError(f"Source room name does not match {room_id}: {room}")
    if not source_checkpoint or "/" in source_checkpoint or "\\" in source_checkpoint:
        raise ProofError("Source checkpoint must be one simple directory name")
    destination = _validate_shadow_run_destination(repo, output_dir)
    if destination.exists() or destination.is_symlink():
        raise ProofError(f"Proof output already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    started_at = _utc_now()

    source_state = _require_regular_file(
        room / "scene_states" / source_checkpoint / "scene_state.json",
        "Recovered source state",
    )
    recovery_receipt = _require_regular_file(
        source_state.parent / "recovery_receipt.json", "Recovery receipt"
    )
    prompt_csv = _require_regular_file(
        csv if csv.is_absolute() else repo / csv, "Prompt CSV"
    )
    worker = _require_regular_file(
        repo / "scripts" / "run_single_room_worker.py", "Room worker"
    )
    renderer = _require_regular_file(
        repo / "scripts" / "render_room_review_views.py", "Review renderer"
    )
    # Conda commonly exposes ``bin/python`` as a symlink.  Resolve it first and
    # attest/execute the regular interpreter target rather than rejecting the
    # standard environment entry point.
    python_path = _require_regular_file(
        (python_executable or Path(sys.executable)).expanduser().resolve(strict=True),
        "Python executable",
    )

    try:
        source_document = json.loads(source_state.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProofError(f"Cannot parse recovered source state: {exc}") from exc
    if not isinstance(source_document, Mapping):
        raise ProofError("Recovered source state is not a JSON object")
    prompt = source_document.get("text_description")
    if not isinstance(prompt, str) or not prompt:
        raise ProofError("Recovered source state has no immutable room prompt")

    generated_source = _discover_source_tree(
        source_document, marker="generated_assets", source_room=room
    )
    room_geometry_source = _discover_source_tree(
        source_document,
        marker="room_geometry",
        source_room=room.parent,
    )
    floor_plans_source = _discover_source_tree(
        source_document,
        marker="floor_plans",
        source_room=room.parent,
    )
    source_before = {
        "state": _file_record(source_state, "Recovered source state"),
        "recovery_receipt": _file_record(recovery_receipt, "Recovery receipt"),
        "generated_assets": _tree_record(generated_source, "Generated assets"),
        "room_geometry": _tree_record(room_geometry_source, "Room geometry"),
        "floor_plans": _tree_record(floor_plans_source, "Floor plans"),
    }

    # The requested output directory is the shadow run root.  Do not introduce
    # another component here: generated GLTF material URIs are depth-sensitive.
    shadow_run = destination
    shadow_scene = shadow_run / "scene_000"
    shadow_room = shadow_scene / f"room_{room_id}"
    final_state_dir = shadow_room / "scene_states" / "final_scene"
    final_state_dir.mkdir(parents=True, exist_ok=False)
    _copy_regular_tree(generated_source, shadow_room / "generated_assets")
    _copy_regular_tree(room_geometry_source, shadow_scene / "room_geometry")
    _copy_regular_tree(floor_plans_source, shadow_scene / "floor_plans")

    shadow_document = _rebase_state_paths(
        source_document, shadow_room=shadow_room, shadow_scene=shadow_scene
    )
    if shadow_document.get("text_description") != prompt:
        raise ProofError("Shadow state does not preserve the exact room prompt")
    source_scene_roots = {
        room.resolve(),
        room.parent.resolve(),
        room.parent.parent.resolve(),
        generated_source.parent.resolve(),
        room_geometry_source.parent.resolve(),
        floor_plans_source.parent.resolve(),
    }
    _audit_no_source_scene_paths(
        shadow_document,
        source_roots=source_scene_roots,
    )
    shadow_state = final_state_dir / "scene_state.json"
    _atomic_write_json(shadow_state, shadow_document)
    _validate_render_dependencies(shadow_document, shadow_room)
    bound_roots = (
        shadow_room / "generated_assets",
        shadow_scene / "room_geometry",
        shadow_scene / "floor_plans",
    )
    shadow_bound_before = {
        "generated_assets": _tree_record(bound_roots[0], "Copied generated assets"),
        "room_geometry": _tree_record(bound_roots[1], "Copied room geometry"),
        "floor_plans": _tree_record(bound_roots[2], "Copied floor plans"),
    }
    artiverse_visuals = _validate_artiverse_visual_gltf_bindings(
        shadow_document,
        shadow_room=shadow_room,
        repo=repo,
        source_roots=tuple(source_scene_roots),
    )
    gltf_dependencies_before = _gltf_external_uri_manifest(
        repo=repo,
        shadow_run=shadow_run,
        bound_roots=bound_roots,
        forbidden_source_roots=tuple(source_scene_roots),
    )

    logs = destination / "logs"
    reviews = destination / "review"
    logs.mkdir()
    reviews.mkdir()
    environment = _sanitized_environment()
    inherited_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(repo) + (
        os.pathsep + inherited_pythonpath if inherited_pythonpath else ""
    )
    common = [
        str(python_path),
        str(worker),
        "--repo-dir",
        str(repo),
        "--run-dir",
        str(shadow_run),
        "--csv",
        str(prompt_csv),
        "--run-name",
        f"recovered_{room_id}_real_render_proof",
        "--room-id",
        room_id,
        "--start-stage",
        "furniture",
        "--stop-stage",
        "manipuland",
        "--asset-pipeline",
        "generated_sam3d",
        "--materials-data",
        str(materials_data),
        "--materials-source-embeddings",
        str(materials_source_embeddings),
        "--materials-embeddings",
        str(materials_embeddings),
        "--port-offset",
        str(port_offset),
        "--render-gpu-id",
        "0",
        "--refresh-final-blend",
    ]
    render_command = [
        str(python_path),
        str(renderer),
        "--blend",
        str(final_state_dir / "scene.blend"),
        "--scene-state",
        str(shadow_state),
        "--room-id",
        room_id,
        "--output-dir",
        str(reviews),
    ]
    commands = (common, render_command)
    try:
        _run_logged(common, cwd=repo, environment=environment, log=logs / "blend.log")
        _assert_no_ignored_glb_meshes(logs / "blend.log")
        blend = _require_regular_file(final_state_dir / "scene.blend", "Shadow blend")
        if blend.stat().st_size <= 0:
            raise ProofError(f"Shadow blend is empty: {blend}")
        _run_logged(
            render_command,
            cwd=repo,
            environment=environment,
            log=logs / "review.log",
        )
        _assert_no_ignored_glb_meshes(logs / "review.log")
        review = _validate_review_evidence(
            reviews / f"{room_id}_cutaway_evidence.json",
            expected_state=shadow_state,
            expected_blend=blend,
        )
        source_after = {
            "state": _file_record(source_state, "Recovered source state"),
            "recovery_receipt": _file_record(recovery_receipt, "Recovery receipt"),
            "generated_assets": _tree_record(generated_source, "Generated assets"),
            "room_geometry": _tree_record(room_geometry_source, "Room geometry"),
            "floor_plans": _tree_record(floor_plans_source, "Floor plans"),
        }
        if source_after != source_before:
            raise ProofError("A source checkpoint or source asset changed during proof")
        shadow_bound_after = {
            "generated_assets": _tree_record(bound_roots[0], "Copied generated assets"),
            "room_geometry": _tree_record(bound_roots[1], "Copied room geometry"),
            "floor_plans": _tree_record(bound_roots[2], "Copied floor plans"),
        }
        if shadow_bound_after != shadow_bound_before:
            raise ProofError("A bound copied render resource changed during proof")
        gltf_dependencies_after = _gltf_external_uri_manifest(
            repo=repo,
            shadow_run=shadow_run,
            bound_roots=bound_roots,
            forbidden_source_roots=tuple(source_scene_roots),
        )
        if gltf_dependencies_after != gltf_dependencies_before:
            raise ProofError("A recursively bound GLTF dependency changed during proof")

        output_records = {
            "shadow_state": _file_record(shadow_state, "Shadow state"),
            "shadow_blend": _file_record(blend, "Shadow blend"),
            "cutaway_evidence": review["record"],
            "blend_log": _file_record(logs / "blend.log", "Blend log"),
            "review_log": _file_record(logs / "review.log", "Review log"),
            "images": [
                _file_record(reviews / f"{room_id}_{name}.png", f"{name} image")
                for name in VIEW_NAMES
            ],
        }
        payload = {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "status": "pass",
            "started_at": started_at,
            "completed_at": _utc_now(),
            "room_id": room_id,
            "source_checkpoint": source_checkpoint,
            "exact_prompt": {
                "utf8_length": len(prompt.encode("utf-8")),
                "sha256": _sha256_bytes(prompt.encode("utf-8")),
                "preserved_in_shadow": True,
            },
            "source_before": source_before,
            "source_after": source_after,
            "source_unchanged": True,
            "source_scene_roots_audited": sorted(map(str, source_scene_roots)),
            "shadow_resource_binding": {
                "run_root_depth_policy": "repo/outputs/<YYYY-MM-DD>/<unique_run>",
                "output_dir_is_shadow_run": True,
                "bound_roots_before": shadow_bound_before,
                "bound_roots_after": shadow_bound_after,
                "bound_roots_unchanged": True,
                "gltf_dependencies_before": gltf_dependencies_before,
                "gltf_dependencies_after": gltf_dependencies_after,
                "gltf_dependencies_unchanged": True,
                "used_artiverse_visuals": artiverse_visuals,
                "used_artiverse_visual_format": "derived_external_gltf",
                "ignored_glb_mesh_log_matches": 0,
            },
            "code": {
                "proof": _file_record(Path(__file__), "Proof code"),
                "worker": _file_record(worker, "Room worker"),
                "renderer": _file_record(renderer, "Review renderer"),
            },
            "execution": {
                "commands": _command_records(commands),
                "credentials_removed": list(FORBIDDEN_CREDENTIALS),
                "external_proxy_policy": "dead_loopback_proxy_except_localhost",
                "patched_repo_precedes_pythonpath": True,
                "generation_services_started": False,
                "retrieval_services_started": False,
                "model_or_vlm_calls_allowed": False,
                "pipeline": [
                    "RoomScene.restore_from_state_dict",
                    "Drake SceneGraph",
                    "RenderEngineGltfClient",
                    "headless Blender save_blend",
                    "Blender three-view cutaway render",
                ],
            },
            "outputs": output_records,
            "review_attestation_sha256": review["document"]["derivation_receipt"]
            ["attestation"]["sha256"],
        }
        receipt = {
            **payload,
            "attestation": {
                "algorithm": "sha256",
                "sha256": _sha256_bytes(_canonical_json(payload)),
            },
        }
        receipt_path = destination / "proof_receipt.json"
        _atomic_write_json(receipt_path, receipt)
        return receipt_path
    except Exception as exc:
        failure = {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "status": "fail",
            "started_at": started_at,
            "failed_at": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "commands": _command_records(commands),
        }
        _atomic_write_json(destination / "proof_failure.json", failure)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--source-room", type=Path, required=True)
    parser.add_argument("--source-checkpoint", default="scene_after_furniture")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--room-id", default="classroom_01")
    parser.add_argument("--port-offset", type=int, default=700)
    parser.add_argument("--materials-data", type=Path, default=Path("data/materials"))
    parser.add_argument(
        "--materials-source-embeddings",
        type=Path,
        default=Path("data/materials/embeddings"),
    )
    parser.add_argument(
        "--materials-embeddings",
        type=Path,
        default=Path("data/materials_full_quality_contract/embeddings"),
    )
    parser.add_argument("--python-executable", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = prove_render(
            repo_dir=args.repo_dir,
            source_room=args.source_room,
            source_checkpoint=args.source_checkpoint,
            output_dir=args.output_dir,
            csv=args.csv,
            room_id=args.room_id,
            port_offset=args.port_offset,
            materials_data=args.materials_data,
            materials_source_embeddings=args.materials_source_embeddings,
            materials_embeddings=args.materials_embeddings,
            python_executable=args.python_executable,
        )
    except (ProofError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "pass", "receipt": str(receipt)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
