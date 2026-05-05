#!/usr/bin/env python3
"""One-time, hash-bound repair of the interrupted classroom manipuland registry.

The tool reconstructs the exact revision-2 authority from the attested current
lineage and the exact teacher-desk checkpoint.  Removed, aborted-scope files are
copied into an immutable quarantine before the registry is atomically replaced;
cleanup is resumable and deletes only files named and hashed by the old attested
registry.  It never guesses paths and never touches outside the room's strict
``generated_assets/manipuland`` authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from scenesmith.agent_utils import asset_registry as registry_module
from scenesmith.agent_utils.asset_registry import AssetRegistry
from scenesmith.agent_utils.room import ObjectType, UniqueID


class RepairError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=registry_module._reject_duplicate_json_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                RepairError(f"JSON contains {item}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepairError(f"Cannot parse strict JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RepairError(f"JSON document is not a mapping: {path}")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _checkpoint_attestation(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _valid_sha(value: str, label: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RepairError(f"{label} is not a lowercase SHA-256")
    return value


def _regular_file(path: Path, label: str) -> Path:
    lexical = path if path.is_absolute() else Path.cwd() / path
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise RepairError(f"{label} contains a symlink: {current}")
    resolved = lexical.resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size <= 0:
        raise RepairError(f"{label} is not a private nonempty regular file")
    return resolved


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_file_fsynced(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _checkpoint_retained_ids(
    receipt_path: Path,
    *,
    expected_receipt_sha256: str,
    current_assets: set[str],
) -> set[str]:
    receipt_path = _regular_file(receipt_path, "teacher checkpoint receipt")
    if _sha256(receipt_path) != expected_receipt_sha256:
        raise RepairError("Teacher checkpoint receipt SHA-256 mismatch")
    receipt = _strict_json(receipt_path)
    payload = {key: value for key, value in receipt.items() if key != "attestation"}
    if (
        receipt.get("schema_version") != 2
        or receipt.get("status") != "pass"
        or receipt.get("furniture_index") != 0
        or receipt.get("furniture_id") != "teacher_desk_0"
        or receipt.get("attestation")
        != _checkpoint_attestation(payload)
        or not isinstance(receipt.get("added_manipuland_ids"), list)
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None
            for value in receipt["added_manipuland_ids"]
        )
    ):
        raise RepairError("Teacher checkpoint receipt is not the exact predecessor")
    retained = set(registry_module._LEGACY_INTERRUPTED_REGISTRY_IDS)
    retained.update(set(receipt["added_manipuland_ids"]) & current_assets)
    return retained


def _build_target(
    registry: AssetRegistry,
    current: dict[str, Any],
    retained_ids: set[str],
) -> tuple[bytes, list[dict[str, Any]]]:
    assets = current.get("assets")
    history = current.get("attestation_history")
    if (
        current.get("schema_version") != 2
        or not isinstance(current.get("revision"), int)
        or current["revision"] < 3
        or not isinstance(assets, dict)
        or not retained_ids.issubset(assets)
        or not isinstance(history, list)
        or len(history) < 2
    ):
        raise RepairError("Current registry cannot prove its revision-2 ancestor")
    selected = {
        UniqueID(asset_id): registry._assets[UniqueID(asset_id)]
        for asset_id in sorted(retained_ids)
    }
    root = registry._registry_root(registry.auto_save_path)
    asset_files = registry._asset_file_inventory(selected, root=root)
    payload = {
        "schema_version": 2,
        "revision": 2,
        "previous_attestation": history[0],
        "attestation_history": [history[0]],
        "lineage_root": current.get("lineage_root"),
        "legacy_source_b64": current.get("legacy_source_b64"),
        "assets": {asset_id: assets[asset_id] for asset_id in sorted(retained_ids)},
        "asset_files": asset_files,
    }
    document = {**payload, "attestation": registry_module._canonical_payload_sha256(payload)}
    if document["attestation"] != history[1]:
        raise RepairError("Reconstructed revision-2 attestation is not in current lineage")
    return _canonical_bytes(document), asset_files


def _quarantine_manifest(
    *,
    input_sha256: str,
    output_sha256: str,
    checkpoint_sha256: str,
    removed_assets: list[str],
    removed_files: list[dict[str, Any]],
    action_log_input_sha256: str,
    action_log_prefix_sha256: str,
    action_log_prefix_count: int,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "status": "prepared",
        "input_registry_sha256": input_sha256,
        "output_registry_sha256": output_sha256,
        "checkpoint_receipt_sha256": checkpoint_sha256,
        "removed_asset_ids": removed_assets,
        "removed_asset_files": removed_files,
        "action_log_input_sha256": action_log_input_sha256,
        "action_log_prefix_sha256": action_log_prefix_sha256,
        "action_log_prefix_count": action_log_prefix_count,
    }
    return {**payload, "attestation": registry_module._canonical_payload_sha256(payload)}


def _validate_quarantine(
    quarantine: Path,
    expected: dict[str, Any],
) -> None:
    manifest = _strict_json(_regular_file(quarantine / "manifest.json", "quarantine manifest"))
    if manifest != expected:
        raise RepairError("Existing quarantine manifest differs from this repair")
    for record in expected["removed_asset_files"]:
        source = quarantine / "files" / PurePosixPath(record["path"])
        source = _regular_file(source, "quarantined asset")
        if source.stat().st_size != record["size_bytes"] or _sha256(source) != record["sha256"]:
            raise RepairError(f"Quarantined asset differs: {record['path']}")
    action_copy = _regular_file(
        quarantine / "action_log_before_repair.json",
        "quarantined action log",
    )
    if _sha256(action_copy) != expected["action_log_input_sha256"]:
        raise RepairError("Quarantined action log differs")


def _prepare_quarantine(
    *,
    root: Path,
    quarantine: Path,
    manifest: dict[str, Any],
    action_log_path: Path,
) -> None:
    if quarantine.exists() or quarantine.is_symlink():
        _validate_quarantine(quarantine, manifest)
        return
    stage = quarantine.with_name(f".{quarantine.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    if stage.exists() or stage.is_symlink():
        raise RepairError("Quarantine staging path already exists")
    stage.mkdir(parents=False)
    try:
        for record in manifest["removed_asset_files"]:
            relative = PurePosixPath(record["path"])
            source = _regular_file(root.joinpath(*relative.parts), "removed registry asset")
            if source.stat().st_size != record["size_bytes"] or _sha256(source) != record["sha256"]:
                raise RepairError(f"Removed registry asset changed: {record['path']}")
            target = stage / "files" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as source_handle, target.open("xb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
                target_handle.flush()
                os.fsync(target_handle.fileno())
            if _sha256(target) != record["sha256"]:
                raise RepairError(f"Quarantine copy mismatch: {record['path']}")
        action_source = _regular_file(action_log_path, "source action log")
        action_target = stage / "action_log_before_repair.json"
        with action_source.open("rb") as source_handle, action_target.open(
            "xb"
        ) as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        if _sha256(action_target) != manifest["action_log_input_sha256"]:
            raise RepairError("Quarantine action-log copy mismatch")
        for directory in sorted(
            {path.parent for path in (stage / "files").rglob("*")},
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            if directory.is_dir():
                _fsync_directory(directory)
        _write_file_fsynced(stage / "manifest.json", _canonical_bytes(manifest))
        _fsync_directory(stage)
        stage.rename(quarantine)
        _fsync_directory(quarantine.parent)
    except Exception:
        if stage.exists() and not stage.is_symlink():
            shutil.rmtree(stage)
        raise


def _publish_registry(path: Path, encoded: bytes) -> None:
    stage = path.with_name(f".{path.name}.repair.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        _write_file_fsynced(stage, encoded)
        os.replace(stage, path)
        _fsync_directory(path.parent)
    except Exception:
        if stage.exists() and not stage.is_symlink():
            stage.unlink()
        raise


def _cleanup_live_removed_files(
    root: Path,
    records: list[dict[str, Any]],
) -> None:
    parents: set[Path] = set()
    for record in records:
        path = root.joinpath(*PurePosixPath(record["path"]).parts)
        if not path.exists() and not path.is_symlink():
            continue
        resolved = _regular_file(path, "live removed registry asset")
        if resolved.stat().st_size != record["size_bytes"] or _sha256(resolved) != record["sha256"]:
            raise RepairError(f"Refusing to remove changed asset: {record['path']}")
        parents.add(resolved.parent)
        resolved.unlink()
        _fsync_directory(resolved.parent)
    for directory in sorted(parents, key=lambda item: len(item.parts), reverse=True):
        current = directory
        while current != root and current.is_relative_to(root):
            try:
                current.rmdir()
            except OSError:
                break
            _fsync_directory(current.parent)
            current = current.parent


def _action_log_prefix(
    path: Path,
    *,
    expected_input_sha256: str,
    expected_prefix_sha256: str,
    prefix_count: int,
) -> tuple[bytes, str]:
    current_sha = _sha256(_regular_file(path, "action log"))
    try:
        entries = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=registry_module._reject_duplicate_json_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepairError("Action log is invalid JSON") from exc
    if (
        not isinstance(entries, list)
        or len(entries) < prefix_count
        or any(
            not isinstance(entry, dict)
            or entry.get("step_number") != index
            for index, entry in enumerate(entries, start=1)
        )
    ):
        raise RepairError("Action log step sequence is malformed")
    prefix = json.dumps(entries[:prefix_count], indent=2).encode("utf-8")
    if hashlib.sha256(prefix).hexdigest() != expected_prefix_sha256:
        raise RepairError("Action-log predecessor prefix SHA-256 mismatch")
    if current_sha not in {expected_input_sha256, expected_prefix_sha256}:
        raise RepairError("Action log is neither expected input nor predecessor")
    if current_sha == expected_prefix_sha256 and path.read_bytes() != prefix:
        raise RepairError("Action-log predecessor bytes mismatch")
    return prefix, current_sha


def _publish_action_log(path: Path, prefix: bytes) -> None:
    from scenesmith.agent_utils.action_logger import _exclusive_action_log_lock

    with _exclusive_action_log_lock(path):
        if path.read_bytes() == prefix:
            return
        stage = path.with_name(
            f".{path.name}.repair.{os.getpid()}.{uuid.uuid4().hex}"
        )
        try:
            _write_file_fsynced(stage, prefix)
            os.replace(stage, path)
            _fsync_directory(path.parent)
        except Exception:
            if stage.exists() and not stage.is_symlink():
                stage.unlink()
            raise


def repair(
    *,
    registry_path: Path,
    checkpoint_receipt: Path,
    expected_checkpoint_sha256: str,
    expected_input_sha256: str,
    expected_output_sha256: str,
    quarantine_dir: Path,
    action_log_path: Path,
    expected_action_log_input_sha256: str,
    expected_action_log_prefix_sha256: str,
    action_log_prefix_count: int,
    apply: bool,
) -> dict[str, Any]:
    expected_checkpoint_sha256 = _valid_sha(expected_checkpoint_sha256, "checkpoint SHA")
    expected_input_sha256 = _valid_sha(expected_input_sha256, "input registry SHA")
    expected_output_sha256 = _valid_sha(expected_output_sha256, "output registry SHA")
    expected_action_log_input_sha256 = _valid_sha(
        expected_action_log_input_sha256, "input action-log SHA"
    )
    expected_action_log_prefix_sha256 = _valid_sha(
        expected_action_log_prefix_sha256, "action-log prefix SHA"
    )
    if action_log_prefix_count <= 0:
        raise RepairError("Action-log prefix count must be positive")
    registry_path = _regular_file(registry_path, "asset registry")
    root = registry_path.parent.resolve(strict=True)
    room_root = root.parent.parent.resolve(strict=True)
    quarantine_parent = quarantine_dir.parent.resolve(strict=True)
    if (
        root.parent.name != "generated_assets"
        or quarantine_parent != (room_root / "scene_states").resolve(strict=True)
        or not re.fullmatch(r"manipuland_registry_repair_quarantine_[A-Za-z0-9_.-]+", quarantine_dir.name)
    ):
        raise RepairError("Repair paths are outside their exact room authorities")
    action_log_path = _regular_file(action_log_path, "action log")
    if action_log_path.parent != room_root or action_log_path.name != "action_log.json":
        raise RepairError("Action log is outside the exact room authority")
    action_prefix, action_current_sha = _action_log_prefix(
        action_log_path,
        expected_input_sha256=expected_action_log_input_sha256,
        expected_prefix_sha256=expected_action_log_prefix_sha256,
        prefix_count=action_log_prefix_count,
    )
    registry = AssetRegistry(
        auto_save_path=registry_path,
        required_root=root,
        allowed_object_types=frozenset({ObjectType.MANIPULAND}),
    )
    lock_file = registry._acquire_scope_lock()
    try:
        journal = room_root / "scene_states" / ".manipuland_registry_scope_transaction.json"
        if journal.exists() or journal.is_symlink():
            raise RepairError("Refusing repair while a furniture scope journal exists")
        current_sha = _sha256(registry_path)
        current = _strict_json(registry_path)
        registry.load_from_file(registry_path)
        retained = _checkpoint_retained_ids(
            checkpoint_receipt,
            expected_receipt_sha256=expected_checkpoint_sha256,
            current_assets=set(current.get("assets", {})),
        )
        target_bytes: bytes | None = None
        if current_sha == expected_input_sha256:
            target_bytes, target_files = _build_target(registry, current, retained)
            if hashlib.sha256(target_bytes).hexdigest() != expected_output_sha256:
                raise RepairError("Reconstructed revision-2 registry SHA-256 mismatch")
            target_paths = {record["path"] for record in target_files}
            current_files = current.get("asset_files")
            if not isinstance(current_files, list):
                raise RepairError("Current registry lacks an asset-file inventory")
            removed_files = [
                record
                for record in current_files
                if record.get("path") not in target_paths
            ]
            removed_assets = sorted(set(current["assets"]) - retained)
            manifest = _quarantine_manifest(
                input_sha256=expected_input_sha256,
                output_sha256=expected_output_sha256,
                checkpoint_sha256=expected_checkpoint_sha256,
                removed_assets=removed_assets,
                removed_files=removed_files,
                action_log_input_sha256=expected_action_log_input_sha256,
                action_log_prefix_sha256=expected_action_log_prefix_sha256,
                action_log_prefix_count=action_log_prefix_count,
            )
        elif current_sha == expected_output_sha256:
            if set(current.get("assets", {})) != retained:
                raise RepairError("Output registry retained-ID set is wrong")
            manifest = _strict_json(
                _regular_file(
                    quarantine_dir / "manifest.json", "quarantine manifest"
                )
            )
            payload = {
                key: value for key, value in manifest.items() if key != "attestation"
            }
            if (
                manifest.get("input_registry_sha256") != expected_input_sha256
                or manifest.get("output_registry_sha256") != expected_output_sha256
                or manifest.get("checkpoint_receipt_sha256")
                != expected_checkpoint_sha256
                or manifest.get("action_log_input_sha256")
                != expected_action_log_input_sha256
                or manifest.get("action_log_prefix_sha256")
                != expected_action_log_prefix_sha256
                or manifest.get("action_log_prefix_count")
                != action_log_prefix_count
                or manifest.get("attestation")
                != registry_module._canonical_payload_sha256(payload)
                or not isinstance(manifest.get("removed_asset_ids"), list)
                or not isinstance(manifest.get("removed_asset_files"), list)
            ):
                raise RepairError("Quarantine manifest is not the expected repair")
            removed_assets = manifest["removed_asset_ids"]
            removed_files = manifest["removed_asset_files"]
        else:
            raise RepairError("Current registry is neither expected input nor output")
        result = {
            "status": "verified",
            "current_registry_sha256": current_sha,
            "target_registry_sha256": expected_output_sha256,
            "retained_asset_ids": sorted(retained),
            "removed_asset_ids": removed_assets,
            "removed_file_count": len(removed_files),
            "quarantine_dir": str(quarantine_dir),
            "current_action_log_sha256": action_current_sha,
            "target_action_log_sha256": expected_action_log_prefix_sha256,
        }
        if not apply:
            return result
        if current_sha == expected_input_sha256:
            _prepare_quarantine(
                root=root,
                quarantine=quarantine_dir,
                manifest=manifest,
                action_log_path=action_log_path,
            )
            assert target_bytes is not None
            _publish_registry(registry_path, target_bytes)
        elif current_sha == expected_output_sha256:
            _validate_quarantine(quarantine_dir, manifest)
        else:
            raise RepairError("Current registry is neither expected input nor output")
        _publish_action_log(action_log_path, action_prefix)
        _cleanup_live_removed_files(root, removed_files)
        repaired = AssetRegistry(
            auto_save_path=registry_path,
            required_root=root,
            allowed_object_types=frozenset({ObjectType.MANIPULAND}),
        )
        repaired.load_from_file(registry_path)
        if _sha256(registry_path) != expected_output_sha256 or set(
            str(asset.object_id) for asset in repaired.list_all()
        ) != retained:
            raise RepairError("Repaired registry failed independent reload")
        _validate_quarantine(quarantine_dir, manifest)
        if _sha256(action_log_path) != expected_action_log_prefix_sha256:
            raise RepairError("Repaired action log failed exact hash verification")
        result["status"] = "repaired"
        return result
    finally:
        registry._release_scope_lock(lock_file)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-output-sha256", required=True)
    parser.add_argument("--quarantine-dir", type=Path, required=True)
    parser.add_argument("--action-log", type=Path, required=True)
    parser.add_argument("--expected-action-log-input-sha256", required=True)
    parser.add_argument("--expected-action-log-prefix-sha256", required=True)
    parser.add_argument("--action-log-prefix-count", type=int, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        result = repair(
            registry_path=args.registry,
            checkpoint_receipt=args.checkpoint_receipt,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            expected_input_sha256=args.expected_input_sha256,
            expected_output_sha256=args.expected_output_sha256,
            quarantine_dir=args.quarantine_dir,
            action_log_path=args.action_log,
            expected_action_log_input_sha256=args.expected_action_log_input_sha256,
            expected_action_log_prefix_sha256=args.expected_action_log_prefix_sha256,
            action_log_prefix_count=args.action_log_prefix_count,
            apply=args.apply,
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
