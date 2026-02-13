#!/usr/bin/env python3
"""Verify an immutable scene transfer and finalize two-GPU Drake acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import stat
import sys
import time

from pathlib import Path, PurePosixPath
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_LINE_RE = re.compile(r"^([0-9A-Fa-f]{64}) ([ *])(.+)$")
RUN_ATTEMPT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
SQZ_RECORD_RELATIVE_PATH = "combined_house/sqz_acceptance_record.json"
SQZ_RECORD_SCHEMA_ID = "scenesmith_sqz_final_acceptance_bundle_v1"
SQZ_COMPLETION_SCHEMA_ID = "scenesmith_sqz_pipeline_completion_v1"
TWO_GPU_RECEIPT_SCHEMA_ID = "scenesmith_two_gpu_drake_acceptance_v2"
RUNTIME_VALIDATION_SCHEMA_ID = "scenesmith_two_gpu_runtime_validation_v1"
PIPELINE_CODE_CONTRACT_LABEL = "pipeline_code_contract"
REQUIRED_RUNTIME_ARTIFACTS = (
    "remote_jobs/TEMPLATE_2gpu_drake_acceptance.sbatch",
    "local_setup/compute_node_env.sh",
    "scripts/two_gpu_drake_acceptance_contract.py",
    "scripts/validate_drake_scene.py",
    "scripts/validate_isaac_sim_scene.py",
    "scripts/pipeline_code_contract.py",
)
SQZ_AWAITING_STATUS = "awaiting_2gpu_acceptance"
REQUIRED_ROOM_IDS = (
    "classroom_01",
    "classroom_02",
    "classroom_03",
    "classroom_04",
    "classroom_05",
    "classroom_06",
    "library",
    "boys_toilet",
    "girls_toilet",
    "storage_room",
    "main_corridor",
)
CRITICAL_SQZ_EVIDENCE_LABELS = frozenset(
    {
        "house_layout",
        "floor_plan_layout_gate",
        "room_prompt_binding",
        "classroom_variation",
        "articulated_motion",
        "school_navigation",
        "room_deterministic_summary",
        "room_visual_summary",
        "house_state",
        "house_dmd",
        "house_blend",
        "artiverse_usage",
        "artiverse_final_validation",
        "whole_floor_gate",
        "drake_sqz",
        "simulator_exports",
        "sage_scene_checker",
        "pipeline_code_contract",
        "materials_contract",
        "artiverse_preparation",
        "sam3d_offline_preflight",
        "sam3d_generation_preflight",
        "sam3d_generation_input",
        "sam3d_generation_glb",
        "sam3d_generation_mask",
        "sam3d_generation_masked_image",
        "objathor_retrieval_preflight",
        "vlm_vision_smoke",
    }
)


class AcceptanceContractError(RuntimeError):
    """Raised when transfer or acceptance evidence violates the contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise AcceptanceContractError(f"output path is a symlink: {path}")
    path = path.parent.resolve(strict=True) / path.name
    if path.exists() and not path.is_file():
        raise AcceptanceContractError(f"output path is not a regular file: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.is_symlink():
        raise AcceptanceContractError(f"temporary output path is a symlink: {temporary}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_sha256(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if not SHA256_RE.fullmatch(normalized):
        raise AcceptanceContractError(f"{label} is not a 64-character SHA-256")
    return normalized


def _require_run_attempt_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or RUN_ATTEMPT_RE.fullmatch(value) is None:
        raise AcceptanceContractError(
            f"{label} must be a 1-128 character safe run-attempt identifier"
        )
    return value


def _require_canonical_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise AcceptanceContractError(f"{label} is not a canonical POSIX path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise AcceptanceContractError(f"{label} is not a safe relative path: {value!r}")
    return value


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _manifest_entries(manifest_path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise AcceptanceContractError("package SHA-256 manifest is empty")
    for line_number, line in enumerate(lines, start=1):
        match = MANIFEST_LINE_RE.fullmatch(line)
        if match is None:
            raise AcceptanceContractError(
                f"invalid SHA-256 manifest line {line_number}: expected "
                "'<sha256>  <relative-path>'"
            )
        digest, _marker, relative_value = match.groups()
        if "\\" in relative_value or "\x00" in relative_value:
            raise AcceptanceContractError(
                f"manifest line {line_number} contains a non-POSIX path"
            )
        relative = PurePosixPath(relative_value)
        if (
            relative.is_absolute()
            or relative.as_posix() != relative_value
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise AcceptanceContractError(
                f"manifest line {line_number} has an unsafe path: {relative_value}"
            )
        if relative_value in entries:
            raise AcceptanceContractError(
                f"manifest contains duplicate path: {relative_value}"
            )
        entries[relative_value] = digest.lower()
    return entries


def create_package_manifest(
    package_root: Path,
    manifest_path: Path,
    expected_run_attempt_id: str,
) -> dict[str, Any]:
    """Create and immediately verify an external exact-file-set manifest."""

    expected_run_attempt_id = _require_run_attempt_id(
        expected_run_attempt_id, "expected run_attempt_id"
    )
    package_root = package_root.resolve(strict=True)
    manifest_path = manifest_path.expanduser().absolute()
    if not package_root.is_dir():
        raise AcceptanceContractError(f"package root is not a directory: {package_root}")
    if _path_is_within(manifest_path, package_root):
        raise AcceptanceContractError(
            "package manifest must be outside the package root to avoid self-reference"
        )
    records: list[tuple[str, str]] = []
    for path in sorted(package_root.rglob("*")):
        if path.is_symlink():
            raise AcceptanceContractError(f"package contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise AcceptanceContractError(f"package contains a special file: {path}")
        resolved = path.resolve(strict=True)
        if not _path_is_within(resolved, package_root):
            raise AcceptanceContractError(f"package file escapes root: {path}")
        records.append(
            (path.relative_to(package_root).as_posix(), _sha256_file(resolved))
        )
    if not records:
        raise AcceptanceContractError("scene package contains no regular files")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            "".join(f"{digest}  {relative}\n" for relative, digest in records),
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    manifest_sha256 = _sha256_file(manifest_path)
    return verify_package(
        package_root,
        manifest_path,
        manifest_sha256,
        expected_run_attempt_id,
    )


def verify_package(
    package_root: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
    expected_run_attempt_id: str,
) -> dict[str, Any]:
    """Verify the exact package plus its semantic SQZ acceptance record."""

    package_root = package_root.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256, "expected manifest digest"
    )
    expected_run_attempt_id = _require_run_attempt_id(
        expected_run_attempt_id, "expected run_attempt_id"
    )
    if not package_root.is_dir():
        raise AcceptanceContractError(f"package root is not a directory: {package_root}")
    if not manifest_path.is_file():
        raise AcceptanceContractError(f"manifest is not a file: {manifest_path}")
    if _path_is_within(manifest_path, package_root):
        raise AcceptanceContractError(
            "package manifest must be outside the package root to avoid self-reference"
        )

    manifest_sha256 = _sha256_file(manifest_path)
    if manifest_sha256 != expected_manifest_sha256:
        raise AcceptanceContractError(
            "package manifest digest mismatch: "
            f"expected {expected_manifest_sha256}, got {manifest_sha256}"
        )
    entries = _manifest_entries(manifest_path)

    actual_files: dict[str, Path] = {}
    for path in sorted(package_root.rglob("*")):
        if path.is_symlink():
            raise AcceptanceContractError(f"package contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise AcceptanceContractError(f"package contains a special file: {path}")
        relative = path.relative_to(package_root).as_posix()
        actual_files[relative] = path
    if not actual_files:
        raise AcceptanceContractError("scene package contains no regular files")

    listed = set(entries)
    actual = set(actual_files)
    if listed != actual:
        missing = sorted(listed - actual)
        unexpected = sorted(actual - listed)
        raise AcceptanceContractError(
            f"package file set differs from manifest; missing={missing}, "
            f"unexpected={unexpected}"
        )

    verified_files: list[dict[str, Any]] = []
    for relative in sorted(entries):
        path = actual_files[relative]
        resolved = path.resolve(strict=True)
        if not _path_is_within(resolved, package_root):
            raise AcceptanceContractError(
                f"manifest path escapes the package root: {relative}"
            )
        actual_sha256 = _sha256_file(resolved)
        if actual_sha256 != entries[relative]:
            raise AcceptanceContractError(
                f"package file digest mismatch for {relative}: "
                f"expected {entries[relative]}, got {actual_sha256}"
            )
        verified_files.append(
            {
                "path": relative,
                "sha256": actual_sha256,
                "size_bytes": resolved.stat().st_size,
            }
        )

    sqz_record = _validate_sqz_acceptance_record(
        package_root=package_root,
        verified_files=verified_files,
        expected_run_attempt_id=expected_run_attempt_id,
    )

    return {
        "schema_version": 1,
        "status": "pass",
        "algorithm": "sha256",
        "package_root": str(package_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "expected_manifest_sha256": expected_manifest_sha256,
        "file_count": len(verified_files),
        "package_content_sha256": _canonical_sha256(verified_files),
        "run_attempt_id": expected_run_attempt_id,
        "sqz_acceptance_record": sqz_record,
        "files": verified_files,
        "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise AcceptanceContractError(f"JSON contains duplicate key {key!r}")
        value[key] = item
    return value


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except AcceptanceContractError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise AcceptanceContractError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AcceptanceContractError(f"{label} is not a JSON object: {path}")
    return value


def _validate_sqz_acceptance_record(
    *,
    package_root: Path,
    verified_files: list[dict[str, Any]],
    expected_run_attempt_id: str,
) -> dict[str, Any]:
    """Interpret the in-package SQZ receipt and bind it to the manifest."""

    expected_run_attempt_id = _require_run_attempt_id(
        expected_run_attempt_id, "expected run_attempt_id"
    )
    files_by_path = {str(item.get("path")): item for item in verified_files}
    receipt_file = files_by_path.get(SQZ_RECORD_RELATIVE_PATH)
    if receipt_file is None:
        raise AcceptanceContractError(
            f"package lacks required SQZ receipt {SQZ_RECORD_RELATIVE_PATH}"
        )
    receipt_path = package_root.joinpath(*PurePosixPath(SQZ_RECORD_RELATIVE_PATH).parts)
    record = _load_json_object(receipt_path, "SQZ acceptance record")
    expected_keys = {
        "schema_id",
        "schema_version",
        "status",
        "run_attempt_id",
        "created_at_utc",
        "hash_algorithm",
        "path_encoding",
        "required_room_ids",
        "required_labels",
        "scene_identity",
        "package_inventory",
        "external_package_manifest",
        "evidence_count",
        "evidence",
        "evidence_attestation",
        "label_specific_validation",
        "classroom_variation_gate",
        "two_gpu_acceptance",
        "self_attestation",
    }
    if set(record) != expected_keys:
        raise AcceptanceContractError(
            "SQZ acceptance record fields differ from schema v1: "
            f"missing={sorted(expected_keys - set(record))}, "
            f"unexpected={sorted(set(record) - expected_keys)}"
        )
    if (
        record.get("schema_id") != SQZ_RECORD_SCHEMA_ID
        or record.get("schema_version") != 1
        or record.get("status") != SQZ_AWAITING_STATUS
    ):
        raise AcceptanceContractError(
            "SQZ acceptance record schema/status is not awaiting 2-GPU acceptance"
        )
    run_attempt_id = _require_run_attempt_id(
        record.get("run_attempt_id"), "SQZ acceptance record run_attempt_id"
    )
    if run_attempt_id != expected_run_attempt_id:
        raise AcceptanceContractError(
            "SQZ acceptance record run_attempt_id differs from the expected attempt"
        )
    if (
        record.get("hash_algorithm") != "sha256"
        or record.get("path_encoding") != "scene-relative-posix"
        or not isinstance(record.get("created_at_utc"), str)
        or not record["created_at_utc"]
        or record.get("required_room_ids") != list(REQUIRED_ROOM_IDS)
    ):
        raise AcceptanceContractError("SQZ acceptance record identity contract is invalid")
    if record.get("classroom_variation_gate") != {
        "required": True,
        "status": "pass",
        "evidence_label": "classroom_variation",
    }:
        raise AcceptanceContractError(
            "SQZ acceptance record does not pass the classroom-variation gate"
        )
    if record.get("two_gpu_acceptance") != {
        "required": True,
        "status": "pending",
        "minimum_visible_gpu_count": 2,
    }:
        raise AcceptanceContractError(
            "SQZ acceptance record improperly claims or weakens 2-GPU acceptance"
        )

    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise AcceptanceContractError("SQZ acceptance evidence inventory is empty")
    if record.get("evidence_count") != len(evidence):
        raise AcceptanceContractError("SQZ acceptance evidence_count is inconsistent")
    labels: set[str] = set()
    evidence_paths: set[str] = set()
    normalized_evidence: list[dict[str, Any]] = []
    for index, item in enumerate(evidence):
        if not isinstance(item, dict) or set(item) != {
            "label",
            "scene_relative_path",
            "size_bytes",
            "sha256",
            "category",
        }:
            raise AcceptanceContractError(
                f"SQZ acceptance evidence[{index}] is malformed"
            )
        label = item.get("label")
        category = item.get("category")
        if (
            not isinstance(label, str)
            or not label
            or label in labels
            or not isinstance(category, str)
            or not category
        ):
            raise AcceptanceContractError(
                f"SQZ acceptance evidence[{index}] has invalid label/category"
            )
        relative = _require_canonical_relative_path(
            item.get("scene_relative_path"),
            f"SQZ acceptance evidence[{index}] path",
        )
        if relative == SQZ_RECORD_RELATIVE_PATH or relative in evidence_paths:
            raise AcceptanceContractError(
                f"SQZ acceptance evidence has duplicate/reserved path: {relative}"
            )
        digest = _require_sha256(
            str(item.get("sha256", "")),
            f"SQZ acceptance evidence[{index}] digest",
        )
        size = item.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise AcceptanceContractError(
                f"SQZ acceptance evidence[{index}] has invalid size"
            )
        manifest_item = files_by_path.get(relative)
        if (
            manifest_item is None
            or manifest_item.get("sha256") != digest
            or manifest_item.get("size_bytes") != size
        ):
            raise AcceptanceContractError(
                f"SQZ evidence is not exactly bound by the package manifest: {relative}"
            )
        labels.add(label)
        evidence_paths.add(relative)
        normalized_evidence.append(item)

    expected_evidence_paths = set(files_by_path) - {SQZ_RECORD_RELATIVE_PATH}
    if evidence_paths != expected_evidence_paths:
        raise AcceptanceContractError(
            "SQZ evidence inventory is not the exact package file set excluding its "
            f"own receipt; missing={sorted(expected_evidence_paths - evidence_paths)}, "
            f"unexpected={sorted(evidence_paths - expected_evidence_paths)}"
        )
    expected_evidence_attestation = {
        "algorithm": "sha256",
        "sha256": _canonical_sha256(normalized_evidence),
    }
    if record.get("evidence_attestation") != expected_evidence_attestation:
        raise AcceptanceContractError("SQZ evidence attestation is inconsistent")

    evidence_by_label = {
        str(item["label"]): item for item in normalized_evidence
    }
    scene_identity = record.get("scene_identity")
    if not isinstance(scene_identity, dict):
        raise AcceptanceContractError("SQZ scene identity is missing")
    _require_canonical_relative_path(
        scene_identity.get("scene_relative_to_run"),
        "SQZ scene_relative_to_run",
    )
    if scene_identity != {
        "scene_relative_to_run": scene_identity["scene_relative_to_run"],
        "house_layout_sha256": evidence_by_label.get("house_layout", {}).get(
            "sha256"
        ),
        "house_state_sha256": evidence_by_label.get("house_state", {}).get(
            "sha256"
        ),
        "house_dmd_sha256": evidence_by_label.get("house_dmd", {}).get("sha256"),
    }:
        raise AcceptanceContractError("SQZ scene identity is stale or malformed")
    package_items = sorted(
        (
            {
                "scene_relative_path": item["scene_relative_path"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in normalized_evidence
        ),
        key=lambda item: item["scene_relative_path"],
    )
    if record.get("package_inventory") != {
        "scope": "all_regular_scene_files_except_this_record",
        "record_scene_relative_path": SQZ_RECORD_RELATIVE_PATH,
        "file_count": len(package_items),
        "content_sha256": _canonical_sha256(package_items),
    }:
        raise AcceptanceContractError("SQZ package inventory is stale or malformed")
    if record.get("external_package_manifest") != {
        "required": True,
        "status": "pending_creation_after_this_record",
        "must_be_outside_scene": True,
        "must_cover_exact_scene_file_set_including_this_record": True,
        "record_scene_relative_path": SQZ_RECORD_RELATIVE_PATH,
    }:
        raise AcceptanceContractError(
            "SQZ external-package-manifest contract is malformed"
        )

    required_labels = record.get("required_labels")
    if (
        not isinstance(required_labels, list)
        or any(not isinstance(value, str) or not value for value in required_labels)
        or required_labels != sorted(set(required_labels))
        or not set(required_labels).issubset(labels)
        or not CRITICAL_SQZ_EVIDENCE_LABELS.issubset(set(required_labels))
    ):
        raise AcceptanceContractError("SQZ required-label contract is incomplete")
    if record.get("label_specific_validation") != {
        label: "pass" for label in required_labels
    }:
        raise AcceptanceContractError(
            "SQZ label-specific validation does not pass every required label"
        )

    attestation = record.get("self_attestation")
    expected_self_sha256 = _canonical_sha256(
        {key: value for key, value in record.items() if key != "self_attestation"}
    )
    if attestation != {
        "algorithm": "sha256",
        "scope": "canonical_json_of_all_fields_except_self_attestation",
        "sha256": expected_self_sha256,
    }:
        raise AcceptanceContractError("SQZ acceptance self-attestation is inconsistent")

    return {
        "schema_id": SQZ_RECORD_SCHEMA_ID,
        "schema_version": 1,
        "status": SQZ_AWAITING_STATUS,
        "relative_path": SQZ_RECORD_RELATIVE_PATH,
        "sha256": receipt_file["sha256"],
        "size_bytes": receipt_file["size_bytes"],
        "run_attempt_id": run_attempt_id,
        "self_attestation_sha256": expected_self_sha256,
        "evidence_attestation_sha256": expected_evidence_attestation["sha256"],
        "evidence_count": len(evidence),
    }


def _passing_package_evidence(
    path: Path,
    label: str,
    *,
    expected_run_attempt_id: str,
) -> dict[str, Any]:
    evidence = _load_json_object(path, label)
    if evidence.get("schema_version") != 1 or evidence.get("status") != "pass":
        raise AcceptanceContractError(f"{label} is not passing schema version 1")
    for key in ("manifest_sha256", "package_content_sha256"):
        _require_sha256(str(evidence.get(key, "")), f"{label} {key}")
    if not isinstance(evidence.get("files"), list) or not evidence["files"]:
        raise AcceptanceContractError(f"{label} has no verified file records")
    if evidence.get("file_count") != len(evidence["files"]):
        raise AcceptanceContractError(f"{label} file count is inconsistent")
    seen_paths: set[str] = set()
    for item in evidence["files"]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise AcceptanceContractError(f"{label} has a malformed file record")
        if item["path"] in seen_paths:
            raise AcceptanceContractError(f"{label} has duplicate file records")
        seen_paths.add(item["path"])
        _require_sha256(str(item.get("sha256", "")), f"{label} file digest")
        if not isinstance(item.get("size_bytes"), int) or item["size_bytes"] < 0:
            raise AcceptanceContractError(f"{label} has an invalid file size")
    if _canonical_sha256(evidence["files"]) != evidence["package_content_sha256"]:
        raise AcceptanceContractError(f"{label} content fingerprint is inconsistent")
    if evidence.get("run_attempt_id") != expected_run_attempt_id:
        raise AcceptanceContractError(f"{label} run_attempt_id is inconsistent")
    package_root = Path(str(evidence.get("package_root", "")))
    manifest_path = Path(str(evidence.get("manifest_path", "")))
    current = verify_package(
        package_root,
        manifest_path,
        str(evidence.get("manifest_sha256", "")),
        expected_run_attempt_id,
    )
    stable_keys = (
        "package_root",
        "manifest_path",
        "manifest_sha256",
        "expected_manifest_sha256",
        "file_count",
        "package_content_sha256",
        "run_attempt_id",
        "sqz_acceptance_record",
        "files",
    )
    changed = [key for key in stable_keys if evidence.get(key) != current.get(key)]
    if changed:
        raise AcceptanceContractError(
            f"{label} differs from a fresh package verification: {', '.join(changed)}"
        )
    return evidence


def _runtime_regular_file(repo_root: Path, relative_value: str) -> Path:
    """Resolve an attested runtime file without following path indirection."""

    relative_value = _require_canonical_relative_path(
        relative_value, "runtime artifact path"
    )
    candidate = repo_root.joinpath(*PurePosixPath(relative_value).parts)
    current = repo_root
    for component in PurePosixPath(relative_value).parts:
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise AcceptanceContractError(
                f"required runtime artifact is missing: {relative_value}: {exc}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise AcceptanceContractError(
                f"runtime artifact path contains a symlink: {relative_value}"
            )
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(repo_root)
    except (OSError, ValueError) as exc:
        raise AcceptanceContractError(
            f"runtime artifact escapes repository: {relative_value}"
        ) from exc
    metadata = resolved.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise AcceptanceContractError(
            f"runtime artifact is not a regular file: {relative_value}"
        )
    return resolved


def _pipeline_contract_from_package(
    package: dict[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    """Load the exact pipeline/code proof named by the in-package SQZ record."""

    package_root = Path(str(package["package_root"])).resolve(strict=True)
    record_path = package_root / SQZ_RECORD_RELATIVE_PATH
    record = _load_json_object(record_path, "SQZ acceptance record")
    evidence = record.get("evidence")
    matches = [
        item
        for item in evidence if isinstance(item, dict)
        and item.get("label") == PIPELINE_CODE_CONTRACT_LABEL
    ] if isinstance(evidence, list) else []
    if len(matches) != 1:
        raise AcceptanceContractError(
            "SQZ record must name exactly one pipeline_code_contract artifact"
        )
    item = matches[0]
    relative_value = _require_canonical_relative_path(
        item.get("scene_relative_path"), "packaged pipeline contract path"
    )
    files_by_path = {str(entry.get("path")): entry for entry in package["files"]}
    package_entry = files_by_path.get(relative_value)
    if (
        package_entry is None
        or package_entry.get("sha256") != item.get("sha256")
        or package_entry.get("size_bytes") != item.get("size_bytes")
    ):
        raise AcceptanceContractError(
            "packaged pipeline contract is not bound by package verification"
        )
    contract_path = package_root.joinpath(*PurePosixPath(relative_value).parts)
    contract = _load_json_object(contract_path, "packaged pipeline/code contract")
    attestation = contract.get("attestation")
    expected_attestation = {
        "schema_version": 1,
        "algorithm": "sha256",
        "sha256": _canonical_sha256(
            {key: value for key, value in contract.items() if key != "attestation"}
        ),
    }
    if (
        contract.get("schema_version") != 1
        or contract.get("status") != "pass"
        or contract.get("hash_algorithm") != "sha256"
        or attestation != expected_attestation
    ):
        raise AcceptanceContractError(
            "packaged pipeline/code contract identity or self-attestation is invalid"
        )
    artifacts = contract.get("artifacts")
    if (
        not isinstance(artifacts, list)
        or contract.get("artifact_count") != len(artifacts)
    ):
        raise AcceptanceContractError(
            "packaged pipeline/code contract artifact inventory is malformed"
        )
    by_path: dict[str, dict[str, Any]] = {}
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise AcceptanceContractError(
                f"packaged pipeline/code artifact[{index}] is malformed"
            )
        path_value = _require_canonical_relative_path(
            artifact["path"], f"packaged pipeline/code artifact[{index}] path"
        )
        if path_value in by_path:
            raise AcceptanceContractError(
                f"packaged pipeline/code contract repeats artifact: {path_value}"
            )
        _require_sha256(
            str(artifact.get("sha256", "")),
            f"packaged pipeline/code artifact[{index}] digest",
        )
        if (
            not isinstance(artifact.get("size_bytes"), int)
            or isinstance(artifact.get("size_bytes"), bool)
            or artifact["size_bytes"] < 1
        ):
            raise AcceptanceContractError(
                f"packaged pipeline/code artifact[{index}] size is invalid"
            )
        by_path[path_value] = artifact
    missing = sorted(set(REQUIRED_RUNTIME_ARTIFACTS) - set(by_path))
    if missing:
        raise AcceptanceContractError(
            "packaged pipeline/code contract lacks runtime artifacts: "
            + ", ".join(missing)
        )
    return contract_path, contract, by_path


def verify_runtime(
    package_preflight_path: Path,
    repo_root: Path,
    *,
    expected_run_attempt_id: str,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Bind the exact executing ParaCloud files to the packaged code proof."""

    package_preflight_path = package_preflight_path.resolve(strict=True)
    package = _passing_package_evidence(
        package_preflight_path,
        "package preflight for runtime verification",
        expected_run_attempt_id=expected_run_attempt_id,
    )
    supplied_root = repo_root.expanduser().absolute()
    if supplied_root.is_symlink():
        raise AcceptanceContractError("runtime repository root must not be a symlink")
    try:
        repo_root = supplied_root.resolve(strict=True)
    except OSError as exc:
        raise AcceptanceContractError(f"runtime repository root is missing: {exc}") from exc
    if not repo_root.is_dir():
        raise AcceptanceContractError("runtime repository root is not a directory")

    contract_path, pipeline_contract, artifacts_by_path = (
        _pipeline_contract_from_package(package)
    )
    runtime_artifacts: list[dict[str, Any]] = []
    for relative_value in REQUIRED_RUNTIME_ARTIFACTS:
        expected = artifacts_by_path[relative_value]
        live_path = _runtime_regular_file(repo_root, relative_value)
        actual_size = live_path.stat().st_size
        actual_sha256 = _sha256_file(live_path)
        if (
            actual_size != expected["size_bytes"]
            or actual_sha256 != expected["sha256"]
        ):
            raise AcceptanceContractError(
                "ParaCloud runtime differs from packaged pipeline/code contract: "
                f"{relative_value}"
            )
        runtime_artifacts.append(
            {
                "path": relative_value,
                "size_bytes": actual_size,
                "sha256": actual_sha256,
            }
        )

    payload: dict[str, Any] = {
        "schema_id": RUNTIME_VALIDATION_SCHEMA_ID,
        "schema_version": 1,
        "status": "pass",
        "run_attempt_id": expected_run_attempt_id,
        "package_root": package["package_root"],
        "package_preflight_path": str(package_preflight_path),
        "package_preflight_sha256": _sha256_file(package_preflight_path),
        "repo_root": str(repo_root),
        "pipeline_code_contract": {
            "scene_relative_path": contract_path.relative_to(
                Path(str(package["package_root"])).resolve(strict=True)
            ).as_posix(),
            "sha256": _sha256_file(contract_path),
            "attestation_sha256": pipeline_contract["attestation"]["sha256"],
        },
        "runtime_artifacts": runtime_artifacts,
        "runtime_content_sha256": _canonical_sha256(runtime_artifacts),
    }
    payload["verification_payload_sha256"] = _canonical_sha256(payload)
    if output_path is not None:
        _atomic_write_json(output_path, payload)
    return payload


def _passing_runtime_validation(
    runtime_validation_path: Path,
    package_preflight_path: Path,
    *,
    expected_run_attempt_id: str,
) -> dict[str, Any]:
    saved = _load_json_object(runtime_validation_path, "runtime validation")
    expected_keys = {
        "schema_id",
        "schema_version",
        "status",
        "run_attempt_id",
        "package_root",
        "package_preflight_path",
        "package_preflight_sha256",
        "repo_root",
        "pipeline_code_contract",
        "runtime_artifacts",
        "runtime_content_sha256",
        "verification_payload_sha256",
    }
    if set(saved) != expected_keys:
        raise AcceptanceContractError("runtime validation fields differ from schema v1")
    if (
        saved.get("schema_id") != RUNTIME_VALIDATION_SCHEMA_ID
        or saved.get("schema_version") != 1
        or saved.get("status") != "pass"
        or saved.get("run_attempt_id") != expected_run_attempt_id
    ):
        raise AcceptanceContractError("runtime validation identity is invalid")
    claimed = _require_sha256(
        str(saved.get("verification_payload_sha256", "")),
        "runtime validation payload digest",
    )
    if claimed != _canonical_sha256(
        {key: value for key, value in saved.items() if key != "verification_payload_sha256"}
    ):
        raise AcceptanceContractError("runtime validation self-attestation is invalid")
    current = verify_runtime(
        package_preflight_path,
        Path(str(saved.get("repo_root", ""))),
        expected_run_attempt_id=expected_run_attempt_id,
    )
    if saved != current:
        raise AcceptanceContractError(
            "runtime validation differs from a fresh packaged-code verification"
        )
    return saved


def create_sqz_completion(
    package_validation_path: Path,
    *,
    expected_run_attempt_id: str,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Create the outside-scene pending receipt for the verified SQZ package."""

    expected_run_attempt_id = _require_run_attempt_id(
        expected_run_attempt_id, "expected run_attempt_id"
    )
    package_validation_path = package_validation_path.resolve(strict=True)
    package = _passing_package_evidence(
        package_validation_path,
        "SQZ package validation",
        expected_run_attempt_id=expected_run_attempt_id,
    )
    if output_path is not None:
        package_root = Path(str(package["package_root"])).resolve()
        output_absolute = output_path.expanduser().absolute()
        output_resolved = output_absolute.parent.resolve() / output_absolute.name
        if _path_is_within(output_resolved, package_root):
            raise AcceptanceContractError(
                "SQZ pipeline completion output must be outside the scene package"
            )
    record = dict(package["sqz_acceptance_record"])
    completion: dict[str, Any] = {
        "schema_id": SQZ_COMPLETION_SCHEMA_ID,
        "schema_version": 1,
        "status": SQZ_AWAITING_STATUS,
        "completion_type": "sqz_pipeline",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_attempt_id": expected_run_attempt_id,
        "package": {
            "manifest_sha256": package["manifest_sha256"],
            "file_count": package["file_count"],
            "content_sha256": package["package_content_sha256"],
        },
        "sqz_acceptance_record": record,
        "source_evidence": {
            "package_validation_sha256": _sha256_file(package_validation_path),
        },
        "two_gpu_acceptance": {
            "required": True,
            "status": "pending",
            "minimum_visible_gpu_count": 2,
        },
    }
    completion["completion_payload_sha256"] = _canonical_sha256(completion)
    return completion


def _passing_sqz_completion(
    path: Path,
    *,
    package: dict[str, Any],
    expected_run_attempt_id: str,
    expected_completion_sha256: str,
) -> dict[str, Any]:
    """Validate the transferred pending completion against fresh package proof."""

    path = path.resolve(strict=True)
    expected_completion_sha256 = _require_sha256(
        expected_completion_sha256, "expected SQZ completion digest"
    )
    actual_completion_sha256 = _sha256_file(path)
    if actual_completion_sha256 != expected_completion_sha256:
        raise AcceptanceContractError(
            "SQZ completion digest mismatch: "
            f"expected {expected_completion_sha256}, got {actual_completion_sha256}"
        )
    completion = _load_json_object(path, "SQZ pipeline completion")
    expected_keys = {
        "schema_id",
        "schema_version",
        "status",
        "completion_type",
        "created_at_utc",
        "run_attempt_id",
        "package",
        "sqz_acceptance_record",
        "source_evidence",
        "two_gpu_acceptance",
        "completion_payload_sha256",
    }
    if set(completion) != expected_keys:
        raise AcceptanceContractError("SQZ pipeline completion fields differ from schema v1")
    if (
        completion.get("schema_id") != SQZ_COMPLETION_SCHEMA_ID
        or completion.get("schema_version") != 1
        or completion.get("status") != SQZ_AWAITING_STATUS
        or completion.get("completion_type") != "sqz_pipeline"
        or completion.get("run_attempt_id") != expected_run_attempt_id
        or not isinstance(completion.get("created_at_utc"), str)
        or not completion["created_at_utc"]
    ):
        raise AcceptanceContractError("SQZ pipeline completion identity is invalid")
    if completion.get("package") != {
        "manifest_sha256": package["manifest_sha256"],
        "file_count": package["file_count"],
        "content_sha256": package["package_content_sha256"],
    }:
        raise AcceptanceContractError(
            "SQZ pipeline completion does not bind the transferred package"
        )
    if completion.get("sqz_acceptance_record") != package["sqz_acceptance_record"]:
        raise AcceptanceContractError(
            "SQZ pipeline completion does not bind the internal SQZ record"
        )
    source = completion.get("source_evidence")
    if not isinstance(source, dict) or set(source) != {"package_validation_sha256"}:
        raise AcceptanceContractError("SQZ pipeline completion source proof is malformed")
    _require_sha256(
        str(source.get("package_validation_sha256", "")),
        "SQZ completion package-validation digest",
    )
    if completion.get("two_gpu_acceptance") != {
        "required": True,
        "status": "pending",
        "minimum_visible_gpu_count": 2,
    }:
        raise AcceptanceContractError(
            "SQZ pipeline completion improperly claims or weakens 2-GPU acceptance"
        )
    claimed_payload_sha256 = _require_sha256(
        str(completion.get("completion_payload_sha256", "")),
        "SQZ completion payload digest",
    )
    expected_payload_sha256 = _canonical_sha256(
        {
            key: value
            for key, value in completion.items()
            if key != "completion_payload_sha256"
        }
    )
    if claimed_payload_sha256 != expected_payload_sha256:
        raise AcceptanceContractError("SQZ completion self-attestation is inconsistent")
    return completion


def finalize_receipt(
    preflight_path: Path,
    postflight_path: Path,
    drake_report_path: Path,
    runtime_validation_path: Path,
    sqz_completion_path: Path,
    *,
    expected_run_attempt_id: str,
    expected_completion_sha256: str,
    slurm_job_id: str,
    node_name: str,
) -> dict[str, Any]:
    """Validate all evidence and build a cryptographically bound pass receipt."""

    preflight_path = preflight_path.resolve(strict=True)
    postflight_path = postflight_path.resolve(strict=True)
    drake_report_path = drake_report_path.resolve(strict=True)
    runtime_validation_path = runtime_validation_path.resolve(strict=True)
    sqz_completion_path = sqz_completion_path.resolve(strict=True)
    expected_run_attempt_id = _require_run_attempt_id(
        expected_run_attempt_id, "expected run_attempt_id"
    )
    if (
        not isinstance(slurm_job_id, str)
        or not slurm_job_id.strip()
        or slurm_job_id.strip().lower() == "unknown"
        or len(slurm_job_id) > 128
        or any(ord(character) < 32 for character in slurm_job_id)
    ):
        raise AcceptanceContractError("a concrete SLURM job id is required")
    if (
        not isinstance(node_name, str)
        or not node_name.strip()
        or len(node_name) > 255
        or any(ord(character) < 32 for character in node_name)
    ):
        raise AcceptanceContractError("a concrete compute node name is required")
    preflight = _passing_package_evidence(
        preflight_path,
        "package preflight",
        expected_run_attempt_id=expected_run_attempt_id,
    )
    postflight = _passing_package_evidence(
        postflight_path,
        "package postflight",
        expected_run_attempt_id=expected_run_attempt_id,
    )
    runtime_validation = _passing_runtime_validation(
        runtime_validation_path,
        preflight_path,
        expected_run_attempt_id=expected_run_attempt_id,
    )

    stable_keys = (
        "package_root",
        "manifest_sha256",
        "expected_manifest_sha256",
        "file_count",
        "package_content_sha256",
        "run_attempt_id",
        "sqz_acceptance_record",
        "files",
    )
    changed = [key for key in stable_keys if preflight.get(key) != postflight.get(key)]
    if changed:
        raise AcceptanceContractError(
            "scene package changed during Drake acceptance: " + ", ".join(changed)
        )

    report = _load_json_object(drake_report_path, "Drake report")
    requirements = report.get("acceptance_requirements", {})
    checks = report.get("checks", {})
    if report.get("status") != "pass":
        raise AcceptanceContractError("Drake validation report is not passing")
    if requirements.get("required_visible_gpu_count") != 2:
        raise AcceptanceContractError("Drake report was not run with --require-gpus 2")
    if requirements.get("required_gpu_execution_count") != 2:
        raise AcceptanceContractError(
            "Drake report does not require real execution on two GPUs"
        )
    if requirements.get("max_collision_elements_per_sdf") != 32:
        raise AcceptanceContractError(
            "Drake report was not run with --max-collision-elements 32"
        )
    if requirements.get("minimum_model_directive_count") != 12:
        raise AcceptanceContractError(
            "Drake report was not run with --minimum-models 12"
        )
    if requirements.get("expected_room_count") != 11:
        raise AcceptanceContractError(
            "Drake report was not run with --expected-rooms 11"
        )
    if int(report.get("visible_gpu_count", 0)) < 2:
        raise AcceptanceContractError("Drake report contains fewer than two GPUs")
    if report.get("two_gpu_acceptance_environment") is not True:
        raise AcceptanceContractError(
            "Drake report does not identify a two-GPU acceptance environment"
        )
    required_checks = {
        "dmd_file_exists",
        "package_root_exists",
        "required_gpus_available",
        "required_gpus_exercised",
        "all_sdf_files_parsed",
        "collision_cap_satisfied",
        "sdf_assets_present",
        "dmd_inventory_nonempty",
        "house_state_inventory_nonempty",
        "expected_room_count_satisfied",
        "drake_model_count_matches_directives",
        "drake_load_succeeded",
    }
    if not required_checks.issubset(checks) or not all(
        checks.get(name) is True for name in required_checks
    ):
        raise AcceptanceContractError("Drake report does not pass every required check")
    if report.get("drake_load", {}).get("status") != "pass":
        raise AcceptanceContractError("Drake load result is not passing")
    if int(report.get("drake_load", {}).get("added_model_count", 0)) < 12:
        raise AcceptanceContractError("Drake load contains fewer than 12 models")
    gpu_inventory = report.get("gpu_inventory")
    exercises = (
        gpu_inventory.get("execution_exercises")
        if isinstance(gpu_inventory, dict)
        else None
    )
    if (
        not isinstance(gpu_inventory, dict)
        or gpu_inventory.get("required_execution_count") != 2
        or gpu_inventory.get("all_required_devices_exercised") is not True
        or not isinstance(exercises, list)
        or len(exercises) != 2
        or [item.get("index") for item in exercises if isinstance(item, dict)]
        != [0, 1]
        or any(
            not isinstance(item, dict)
            or item.get("status") != "pass"
            or item.get("synchronized") is not True
            or int(item.get("allocated_bytes", 0)) < 1
            for item in exercises
        )
    ):
        raise AcceptanceContractError(
            "Drake report lacks two independent passing CUDA execution exercises"
        )
    house_inventory = report.get("house_state_inventory", {})
    if (
        house_inventory.get("status") != "pass"
        or house_inventory.get("room_count") != 11
        or int(house_inventory.get("object_count", 0)) < 1
    ):
        raise AcceptanceContractError("Drake report has an invalid house-state inventory")

    package_root = Path(str(preflight["package_root"])).resolve()
    if _path_is_within(sqz_completion_path, package_root):
        raise AcceptanceContractError(
            "SQZ pipeline completion must remain outside the immutable scene package"
        )
    sqz_completion = _passing_sqz_completion(
        sqz_completion_path,
        package=preflight,
        expected_run_attempt_id=expected_run_attempt_id,
        expected_completion_sha256=expected_completion_sha256,
    )
    report_package_root = Path(str(report.get("package_root", ""))).resolve()
    if package_root != report_package_root:
        raise AcceptanceContractError(
            "Drake report package root differs from the hash-verified package"
        )
    dmd_path = Path(str(report.get("dmd_path", ""))).resolve()
    if not _path_is_within(dmd_path, package_root):
        raise AcceptanceContractError("Drake directives path is outside the package")
    dmd_relative = dmd_path.relative_to(package_root).as_posix()
    files_by_path = {item.get("path"): item for item in preflight["files"]}
    if dmd_relative not in files_by_path:
        raise AcceptanceContractError(
            "Drake directives file is not covered by the package manifest"
        )

    receipt: dict[str, Any] = {
        "schema_id": TWO_GPU_RECEIPT_SCHEMA_ID,
        "schema_version": 2,
        "status": "pass",
        "acceptance_type": "two_gpu_drake",
        "run_attempt_id": expected_run_attempt_id,
        "accepted_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "slurm_job_id": slurm_job_id.strip(),
        "node_name": node_name.strip(),
        "package": {
            "root": str(package_root),
            "manifest_path": preflight["manifest_path"],
            "manifest_sha256": preflight["manifest_sha256"],
            "file_count": preflight["file_count"],
            "content_sha256": preflight["package_content_sha256"],
            "dmd_relative_path": dmd_relative,
            "dmd_sha256": files_by_path[dmd_relative]["sha256"],
        },
        "sqz_acceptance_record": dict(preflight["sqz_acceptance_record"]),
        "sqz_pipeline_completion": {
            "path": str(sqz_completion_path),
            "sha256": _sha256_file(sqz_completion_path),
            "completion_payload_sha256": sqz_completion[
                "completion_payload_sha256"
            ],
            "status_before_two_gpu_acceptance": sqz_completion["status"],
            "run_attempt_id": sqz_completion["run_attempt_id"],
        },
        "evidence": {
            "package_preflight_path": str(preflight_path),
            "package_preflight_sha256": _sha256_file(preflight_path),
            "package_postflight_path": str(postflight_path),
            "package_postflight_sha256": _sha256_file(postflight_path),
            "drake_report_path": str(drake_report_path),
            "drake_report_sha256": _sha256_file(drake_report_path),
            "runtime_validation_path": str(runtime_validation_path),
            "runtime_validation_sha256": _sha256_file(runtime_validation_path),
            "sqz_pipeline_completion_path": str(sqz_completion_path),
            "sqz_pipeline_completion_sha256": _sha256_file(sqz_completion_path),
        },
        "drake": {
            "visible_gpu_count": report["visible_gpu_count"],
            "visible_gpus": report.get("visible_gpus", []),
            "two_gpu_acceptance_environment": True,
            "gpu_execution_exercises": exercises,
            "collision_element_cap": 32,
            "checks": {name: checks[name] for name in sorted(required_checks)},
            "load": report["drake_load"],
        },
        "runtime": {
            "repo_root": runtime_validation["repo_root"],
            "pipeline_code_contract": runtime_validation[
                "pipeline_code_contract"
            ],
            "runtime_artifacts": runtime_validation["runtime_artifacts"],
            "runtime_content_sha256": runtime_validation[
                "runtime_content_sha256"
            ],
            "verification_payload_sha256": runtime_validation[
                "verification_payload_sha256"
            ],
        },
    }
    receipt["receipt_payload_sha256"] = _canonical_sha256(receipt)
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-package")
    verify.add_argument("--package-root", required=True, type=Path)
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--expected-manifest-sha256", required=True)
    verify.add_argument("--expected-run-attempt-id", required=True)
    verify.add_argument("--output", required=True, type=Path)

    create = subparsers.add_parser("create-manifest")
    create.add_argument("--package-root", required=True, type=Path)
    create.add_argument("--manifest", required=True, type=Path)
    create.add_argument("--expected-run-attempt-id", required=True)
    create.add_argument("--output", required=True, type=Path)

    completion = subparsers.add_parser("create-sqz-completion")
    completion.add_argument("--package-validation", required=True, type=Path)
    completion.add_argument("--expected-run-attempt-id", required=True)
    completion.add_argument("--output", required=True, type=Path)

    runtime = subparsers.add_parser("verify-runtime")
    runtime.add_argument("--package-preflight", required=True, type=Path)
    runtime.add_argument("--repo-root", required=True, type=Path)
    runtime.add_argument("--expected-run-attempt-id", required=True)
    runtime.add_argument("--output", required=True, type=Path)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--package-preflight", required=True, type=Path)
    finalize.add_argument("--package-postflight", required=True, type=Path)
    finalize.add_argument("--drake-report", required=True, type=Path)
    finalize.add_argument("--runtime-validation", required=True, type=Path)
    finalize.add_argument("--sqz-pipeline-completion", required=True, type=Path)
    finalize.add_argument("--expected-sqz-completion-sha256", required=True)
    finalize.add_argument("--expected-run-attempt-id", required=True)
    finalize.add_argument("--output", required=True, type=Path)
    finalize.add_argument("--slurm-job-id", default=os.getenv("SLURM_JOB_ID", "unknown"))
    finalize.add_argument("--node-name", default=socket.gethostname())
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        if args.command == "verify-package":
            payload = verify_package(
                args.package_root,
                args.manifest,
                args.expected_manifest_sha256,
                args.expected_run_attempt_id,
            )
        elif args.command == "create-manifest":
            payload = create_package_manifest(
                args.package_root,
                args.manifest,
                args.expected_run_attempt_id,
            )
        elif args.command == "create-sqz-completion":
            payload = create_sqz_completion(
                args.package_validation,
                expected_run_attempt_id=args.expected_run_attempt_id,
                output_path=args.output,
            )
        elif args.command == "verify-runtime":
            payload = verify_runtime(
                args.package_preflight,
                args.repo_root,
                expected_run_attempt_id=args.expected_run_attempt_id,
                output_path=args.output,
            )
        else:
            payload = finalize_receipt(
                args.package_preflight,
                args.package_postflight,
                args.drake_report,
                args.runtime_validation,
                args.sqz_pipeline_completion,
                expected_run_attempt_id=args.expected_run_attempt_id,
                expected_completion_sha256=args.expected_sqz_completion_sha256,
                slurm_job_id=args.slurm_job_id,
                node_name=args.node_name,
            )
        _atomic_write_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (AcceptanceContractError, OSError, ValueError) as exc:
        print(f"two-GPU acceptance contract failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
