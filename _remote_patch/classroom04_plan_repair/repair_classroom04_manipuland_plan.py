#!/usr/bin/env python3
"""Consolidate Classroom 4's details onto six supported surfaces."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path


EXPECTED_PLAN_SHA256 = "3196d02b60efd7104c7e0720e4de89a2f1b9094cc7c7b95d11f5197ed0358b84"
EXPECTED_RUNTIME_SHA256 = "8ad862129a21f236986573cb1d851223f8fb28e8246ddaff9a3ae27761bec4b7"
EXPECTED_SELECTION_SHA256 = [
    "0a990f4fe687f24b77a2f13ac68169cce81f0e6450ee673a55277b25627d486c",
    "e007e2d42524bb00f70af3adb218a4cafa243564b4e52a39a5c01b5ae8cb383c",
    "adad187131ba9ee4751de67b377faea719fe8e22bfa5ea4851060796507da00c",
    "cb46dae62bd6775ed0891eca9e2ee4cfab6f503bd101d93ef5cb0062b53d2da2",
    "ad5df0f17460dcaddcc2842967a66fcf821e363dbe777d50d0cf2df936de7bf3",
    "684161f5564f40c9caf8972dda1a885b903e39138c2671e4235a31bccbe555be",
    "c6634f523ebe194c947a8f20cf2b824a80db893ef760d4d7e8015f1081cfe722",
    "d17fd34066bd035cd971fcb0fd51f8ab6d815d6c1ece4f8d600e24863af42371",
    "059dd8b80eec6576b1f46387f0ec3a74048ef0969fd85947e6e5ab671c363e91",
    "c6b33648924db642649bf44ddf490e1c1a8f30a381cd8f3cb834472862be9f4b",
    "d9312b179d969ac3dd586bbef5e8000e1cbebeba0455e1fef2a25d3539826790",
    "70fd6c26a6abdb4cf0e22d0e3b297c3211426489e5f995d81b0bcb26101d3d10",
    "f2285234f495f15cad5312335b7a980f77a5a72b085d21b6fbccf1518e5c5a7f",
    "8c2cd05627f40e2bfc0baac6fda1ade6c4b9cf61266ffa12a3755c4541d9fcf1",
    "ea7d9f5258c471be69be2862203ebd67633871f6ddd31523b24ee45c16d5c7d1",
    "a0549a0a46677077075b5ef82df46fa19af08f4b308ccb5d1f1dbe06322fe05a",
    "d8d1dee69ec60d5936794e3c46547fe7ef21bcef75518d1c88813bd76154e50e",
    "ff3d2ba118e269cca77f73104243a198fa5044954e827b7f62c8cf848279f3d0",
    "ce60b7ba4eb21e338dbb57e27ea66374bfffeb8c0cc9f99c22d294fd35285230",
]
KEEP_INDICES = (0, 1, 2, 3, 5, 14)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def attest(document: dict[str, object]) -> str:
    return sha(canonical({k: v for k, v in document.items() if k != "attestation"}))


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
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room-dir", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    room = args.room_dir.resolve(strict=True)
    states = room / "scene_states"
    plan_path = states / "manipuland_furniture_plan.json"
    if list(states.glob("manipuland_checkpoint_*")):
        raise SystemExit("refusing to alter a plan after a checkpoint exists")

    with (states / ".manipuland_checkpoint.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = sha(plan_path.read_bytes())
        if before != EXPECTED_PLAN_SHA256:
            raise SystemExit(f"unexpected plan digest: {before}")
        document = json.loads(plan_path.read_text(encoding="utf-8"))
        selections = document.get("selections")
        if (
            document.get("room_id") != "classroom_04"
            or document.get("status") != "pass"
            or document.get("checkpoint_runtime_sha256") != EXPECTED_RUNTIME_SHA256
            or document.get("attestation") != attest(document)
            or not isinstance(selections, list)
            or len(selections) != len(EXPECTED_SELECTION_SHA256)
            or document.get("selections_sha256") != sha(canonical(selections))
        ):
            raise SystemExit("input plan contract mismatch")
        for index, expected in enumerate(EXPECTED_SELECTION_SHA256):
            if sha(canonical(selections[index])) != expected:
                raise SystemExit(f"unexpected selection at index {index}")

        repaired = [dict(selections[index]) for index in KEEP_INDICES]
        repaired[0]["suggested_items"] = (
            "REQUIRED: paper trays, folders/worksheets, dry-erase markers, "
            "whiteboard eraser. Optional: gradebook/notebook, pen cup, stapler"
        )
        repaired[0]["prompt_constraints"] = (
            "Prompt requires paper trays, folders/worksheets, board markers and a "
            "whiteboard eraser. Stage the board tools together on the teacher desk "
            "rather than relying on the generated trayless wall boards."
        )
        repaired[5]["suggested_items"] = (
            "REQUIRED: textbooks, labeled storage bins, backpacks. Optional: labeled "
            "folders/binders, spare notebooks"
        )

        output = dict(document)
        output["selections"] = repaired
        output["selections_sha256"] = sha(canonical(repaired))
        output["attestation"] = attest(output)
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        backup = args.backup_dir / f"{plan_path.name}.{before}"
        if not backup.exists():
            shutil.copy2(plan_path, backup, follow_symlinks=False)
        elif sha(backup.read_bytes()) != before:
            raise SystemExit("existing backup digest mismatch")
        atomic_json(plan_path, output, plan_path.stat().st_mode & 0o777)
        after = sha(plan_path.read_bytes())

        receipt = {
            "schema_version": 1,
            "status": "pass",
            "operation": "consolidate_required_inventory_on_six_surfaces",
            "room_id": "classroom_04",
            "plan_before_sha256": before,
            "plan_after_sha256": after,
            "kept_targets": [item["furniture_id"] for item in repaired],
            "removed_duplicate_or_optional_targets": [
                selections[index]["furniture_id"]
                for index in range(len(selections))
                if index not in KEEP_INDICES
            ],
            "board_supply_reroute_target": "desk_0",
            "required_categories_preserved": [
                "textbooks", "notebooks", "folders/worksheets",
                "pencil/marker holder", "erasers/rulers", "glue", "scissors",
                "water bottles", "backpacks", "storage bins", "paper trays",
                "board markers", "whiteboard eraser",
            ],
            "reason": (
                "The prompt requests believable details without clutter; six supported "
                "surfaces cover every category while leaving most of the twelve desks "
                "clear and preserving robot aisles."
            ),
        }
        receipt["attestation"] = attest(receipt)
        atomic_json(args.receipt, receipt, 0o600)
        print(f"CLASSROOM04_PLAN_REPAIR_PASS {before} {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
