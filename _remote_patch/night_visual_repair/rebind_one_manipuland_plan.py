#!/usr/bin/env python3
"""Atomically bind one durable manipuland plan to an exact recovery config."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from argparse import Namespace
from pathlib import Path

from omegaconf import OmegaConf
from scripts.run_single_room_worker import _load_cfg


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def attest(document: dict[str, object]) -> str:
    return sha(canonical({key: value for key, value in document.items() if key != "attestation"}))


def atomic_json(path: Path, document: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, path.stat().st_mode & 0o777)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _checkpoint_receipts(states: Path) -> list[Path]:
    """Return only complete, non-symlinked manipuland checkpoint receipts.

    A rebind changes no selection, scene state, or asset provenance.  It only
    binds the previously accepted checkpoints to the exact recovered runtime
    configuration.  Treat malformed checkpoint directories as a hard error
    rather than silently skipping them.
    """
    receipts: list[Path] = []
    for directory in sorted(states.glob("manipuland_checkpoint_*")):
        if directory.is_symlink() or not directory.is_dir():
            raise RuntimeError(f"Invalid manipuland checkpoint directory: {directory}")
        entries = {entry.name for entry in directory.iterdir()}
        if entries != {"scene_state.json", "scene.dmd.yaml", "completion_receipt.json"}:
            raise RuntimeError(f"Manipuland checkpoint inventory mismatch: {directory}")
        if any(entry.is_symlink() or not entry.is_file() for entry in directory.iterdir()):
            raise RuntimeError(f"Unsafe manipuland checkpoint entry: {directory}")
        receipts.append(directory / "completion_receipt.json")
    if not receipts:
        raise RuntimeError("No completed manipuland checkpoints available to rebind")
    return receipts


def _rebind_checkpoint_receipt(path: Path, expected_config: str) -> dict[str, object]:
    before = path.read_bytes()
    document = json.loads(before)
    if (
        not isinstance(document, dict)
        or document.get("status") != "pass"
        or not isinstance(document.get("config_sha256"), str)
        or document.get("attestation") != attest(document)
    ):
        raise RuntimeError(f"Manipuland checkpoint receipt contract mismatch: {path}")
    updated = dict(document)
    old_config = updated["config_sha256"]
    updated["config_sha256"] = expected_config
    updated["attestation"] = attest(updated)
    return {
        "path": str(path),
        "before_sha256": sha(before),
        "old_config_sha256": old_config,
        "document": updated,
    }


def config_hash(repo: Path, run: Path, run_name: str, port_offset: int) -> str:
    args = Namespace(
        repo_dir=str(repo), run_dir=str(run),
        csv=str(repo / "inputs/full_quality_school_reference_20260710/prompt.csv"),
        run_name=run_name, start_stage="manipuland", stop_stage="manipuland",
        asset_pipeline="generated_sam3d", port_offset=port_offset,
        artiverse_data="data/artiverse", artiverse_embeddings="data/artiverse/embeddings",
        artvip_data="data/artvip_sdf", artvip_embeddings="data/artvip_sdf/embeddings",
        materials_data="data/materials", materials_source_embeddings="data/materials/embeddings",
        materials_embeddings="data/materials_full_quality_contract/embeddings",
    )
    resolved = OmegaConf.to_container(_load_cfg(args).manipuland_agent, resolve=True)
    return sha(canonical(resolved))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--room-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--port-offset", type=int, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    args = parser.parse_args()
    repo, run = args.repo_dir.resolve(strict=True), args.run_dir.resolve(strict=True)
    states = run / "scene_000" / f"room_{args.room_id}" / "scene_states"
    plan = states / "manipuland_furniture_plan.json"
    expected = config_hash(repo, run, args.run_name, args.port_offset)
    with (states / ".manipuland_checkpoint.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before_bytes = plan.read_bytes(); plan_before_sha = sha(before_bytes); document = json.loads(before_bytes)
        selections = document.get("selections")
        if document.get("status") != "pass" or document.get("room_id") != args.room_id or document.get("attestation") != attest(document) or not isinstance(selections, list) or document.get("selections_sha256") != sha(canonical(selections)):
            raise RuntimeError("Durable plan contract mismatch")
        output = dict(document); old = output.get("config_sha256"); output["config_sha256"] = expected; output["attestation"] = attest(output)
        rebound = [_rebind_checkpoint_receipt(path, expected) for path in _checkpoint_receipts(states)]
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        backup = args.backup_dir / f"{args.room_id}.plan.{plan_before_sha}.json"
        if not backup.exists(): shutil.copy2(plan, backup, follow_symlinks=False)
        elif sha(backup.read_bytes()) != plan_before_sha: raise RuntimeError("Backup hash mismatch")
        checkpoint_backups = []
        for entry in rebound:
            receipt_path = Path(str(entry["path"]))
            checkpoint_before_sha = str(entry["before_sha256"])
            backup_path = args.backup_dir / f"{args.room_id}.{receipt_path.parent.name}.{checkpoint_before_sha}.json"
            if not backup_path.exists(): shutil.copy2(receipt_path, backup_path, follow_symlinks=False)
            elif sha(backup_path.read_bytes()) != checkpoint_before_sha: raise RuntimeError(f"Checkpoint backup hash mismatch: {receipt_path}")
            checkpoint_backups.append(str(backup_path))
        for entry in rebound:
            atomic_json(Path(str(entry["path"])), dict(entry["document"]))
        atomic_json(plan, output)
    receipt = {
        "schema_version": 1, "status": "pass", "room_id": args.room_id,
        "operation": "bind_durable_plan_to_exact_recovery_transport_config",
        "run_name": args.run_name, "port_offset": args.port_offset,
        "old_config_sha256": old, "new_config_sha256": expected,
        "plan_before_sha256": plan_before_sha, "plan_after_sha256": sha(plan.read_bytes()),
        "checkpoint_receipts_rebound": [
            {
                "path": entry["path"],
                "before_sha256": entry["before_sha256"],
                "after_sha256": sha(Path(str(entry["path"])).read_bytes()),
                "old_config_sha256": entry["old_config_sha256"],
            }
            for entry in rebound
        ],
        "checkpoint_backup_paths": checkpoint_backups,
        "selection_content_modified": False, "quality_configuration_modified": False,
    }
    receipt["attestation"] = attest(receipt)
    receipt_path = states / f"manipuland_plan_rebind_{args.port_offset}.json"
    temporary = receipt_path.with_name(f".{receipt_path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, receipt_path)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
