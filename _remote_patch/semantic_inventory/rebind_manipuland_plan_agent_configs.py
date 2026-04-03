#!/usr/bin/env python3
"""Bind migrated furniture plans to the exact agent config each worker validates."""

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


WORKERS = {
    "classroom_04": ("paracloud_resume_classroom_04", 404),
    "storage_room": ("paracloud_resume_storage_room", 424),
    "classroom_05": ("paracloud_resume_classroom_05", 444),
    "library": ("paracloud_resume_library", 464),
    "boys_toilet": ("resume_boys_toilet_rebound", 617),
    "classroom_06": ("paracloud_resume_classroom_06", 484),
    "classroom_02": ("paracloud_resume_classroom_02", 504),
    "classroom_03": ("paracloud_resume_classroom_03", 524),
    "main_corridor": ("resume_main_corridor_rebound", 609),
    "girls_toilet": ("paracloud_resume_girls_toilet", 675),
}


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def attest(document: dict[str, object]) -> str:
    payload = {key: value for key, value in document.items() if key != "attestation"}
    return sha(canonical(payload))


def atomic_json(path: Path, document: dict[str, object], mode: int) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def agent_config_hash(repo: Path, run: Path, run_name: str, port_offset: int) -> str:
    args = Namespace(
        repo_dir=str(repo),
        run_dir=str(run),
        csv=str(repo / "inputs/full_quality_school_reference_20260710/prompt.csv"),
        run_name=run_name,
        start_stage="manipuland",
        stop_stage="manipuland",
        asset_pipeline="generated_sam3d",
        port_offset=port_offset,
        artiverse_data="data/artiverse",
        artiverse_embeddings="data/artiverse/embeddings",
        artvip_data="data/artvip_sdf",
        artvip_embeddings="data/artvip_sdf/embeddings",
        materials_data="data/materials",
        materials_source_embeddings="data/materials/embeddings",
        materials_embeddings="data/materials_full_quality_contract/embeddings",
    )
    try:
        cfg = _load_cfg(args)
        resolved_agent = OmegaConf.to_container(cfg.manipuland_agent, resolve=True)
        return sha(canonical(resolved_agent))
    finally:
        # A production worker composes one config per process. Clear named
        # resolvers so this multi-room migration audit reproduces that behavior.
        OmegaConf.clear_resolvers()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--runtime-source", type=Path)
    args = parser.parse_args()

    repo = args.repo_dir.resolve(strict=True)
    run = args.run_dir.resolve(strict=True)
    scene = run / "scene_000"
    runtime_source = (
        args.runtime_source
        if args.runtime_source is not None
        else repo / "scenesmith/manipuland_agents/stateful_manipuland_agent.py"
    ).resolve(strict=True)
    runtime_sha256 = sha(runtime_source.read_bytes())
    args.backup_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for room_id, (run_name, port_offset) in WORKERS.items():
        expected = agent_config_hash(repo, run, run_name, port_offset)
        states = scene / f"room_{room_id}" / "scene_states"
        plan = states / "manipuland_furniture_plan.json"
        with (states / ".manipuland_checkpoint.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            before_bytes = plan.read_bytes()
            before_sha = sha(before_bytes)
            document = json.loads(before_bytes)
            selections = document.get("selections")
            if (
                document.get("status") != "pass"
                or document.get("room_id") != room_id
                or document.get("attestation") != attest(document)
                or not isinstance(selections, list)
                or document.get("selections_sha256") != sha(canonical(selections))
            ):
                raise SystemExit(f"plan contract mismatch for {room_id}")

            old_config = document.get("config_sha256")
            backup = args.backup_dir / f"{room_id}.plan.{before_sha}.json"
            if not backup.exists():
                shutil.copy2(plan, backup, follow_symlinks=False)
            elif sha(backup.read_bytes()) != before_sha:
                raise SystemExit(f"backup mismatch for {room_id}")

            output = dict(document)
            output["config_sha256"] = expected
            output["checkpoint_runtime_sha256"] = runtime_sha256
            output["attestation"] = attest(output)
            atomic_json(plan, output, plan.stat().st_mode & 0o777)
            records.append(
                {
                    "room_id": room_id,
                    "run_name": run_name,
                    "port_offset": port_offset,
                    "old_config_sha256": old_config,
                    "new_agent_config_sha256": expected,
                    "old_checkpoint_runtime_sha256": document.get(
                        "checkpoint_runtime_sha256"
                    ),
                    "new_checkpoint_runtime_sha256": runtime_sha256,
                    "plan_before_sha256": before_sha,
                    "plan_after_sha256": sha(plan.read_bytes()),
                }
            )

    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "operation": "rebind_plans_to_exact_manipuland_agent_configs_and_runtime",
        "records": records,
        "checkpoint_runtime_source": str(runtime_source),
        "checkpoint_runtime_sha256": runtime_sha256,
        "quality_configuration_modified": False,
        "selection_content_modified": False,
    }
    receipt["attestation"] = attest(receipt)
    atomic_json(args.receipt, receipt, 0o600)
    print("MANIPULAND_AGENT_CONFIG_REBIND_PASS", len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
