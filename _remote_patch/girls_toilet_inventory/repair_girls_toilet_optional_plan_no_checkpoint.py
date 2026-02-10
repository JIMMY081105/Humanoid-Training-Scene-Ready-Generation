#!/usr/bin/env python3
"""Drop girls-restroom manipuland scopes when every proposed item is optional."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path


EXPECTED_PLAN_SHA256 = "c913217e920fee18427592fa6aa344c619af58758730e1345ba48d2b8e37779e"
EXPECTED_RUNTIME_SHA256 = "8ad862129a21f236986573cb1d851223f8fb28e8246ddaff9a3ae27761bec4b7"
EXPECTED_SELECTION_SHA256 = (
    "a978b9e83decf03b51668f4c39b2a6422b2de4ea342d795bca69a29ce3e0eaca",
    "825762519a880093ad1fac644f49c88581432a27ccbb8ffead1470d0acc4bbd9",
)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def attest(document: dict[str, object]) -> str:
    return sha(canonical({key: value for key, value in document.items() if key != "attestation"}))


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
    with (states / ".manipuland_checkpoint.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if list(states.glob("manipuland_checkpoint_*")):
            raise SystemExit("a checkpoint exists; no-checkpoint repair is not authorized")
        before = sha(plan_path.read_bytes())
        if before != EXPECTED_PLAN_SHA256:
            raise SystemExit(f"unexpected plan digest: {before}")
        document = json.loads(plan_path.read_text(encoding="utf-8"))
        selections = document.get("selections")
        if (
            document.get("room_id") != "girls_toilet"
            or document.get("status") != "pass"
            or document.get("checkpoint_runtime_sha256") != EXPECTED_RUNTIME_SHA256
            or document.get("attestation") != attest(document)
            or not isinstance(selections, list)
            or len(selections) != 2
            or document.get("selections_sha256") != sha(canonical(selections))
        ):
            raise SystemExit("input plan contract mismatch")
        for index, expected in enumerate(EXPECTED_SELECTION_SHA256):
            selection = selections[index]
            if sha(canonical(selection)) != expected:
                raise SystemExit(f"selection {index} changed")
            if "REQUIRED: none" not in selection.get("suggested_items", ""):
                raise SystemExit(f"selection {index} contains required manipulands")
        if "No specific objects required" not in selections[0].get("prompt_constraints", ""):
            raise SystemExit("vanity prompt constraint is not optional")
        if "already fulfilled" not in selections[1].get("prompt_constraints", ""):
            raise SystemExit("hygiene-station prompt constraint is not already fulfilled")

        output = dict(document)
        output["selections"] = []
        output["selections_sha256"] = sha(canonical(output["selections"]))
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
            "operation": "drop_all_optional_unstable_restroom_manipuland_scopes",
            "room_id": "girls_toilet",
            "plan_before_sha256": before,
            "plan_after_sha256": after,
            "removed_optional_targets": [selection["furniture_id"] for selection in selections],
            "required_prompt_content_preserved": [
                "double-basin vanity and two verified mirror panels",
                "soap dispenser and hygiene dryer/paper-towel station",
                "all restroom furniture, walls, ceiling fixtures and circulation",
            ],
            "reason": (
                "Both scopes explicitly require no manipulands; the vanity bottles were physically "
                "unstable and the hygiene station already fulfills the required drying function."
            ),
        }
        receipt["attestation"] = attest(receipt)
        atomic_json(args.receipt, receipt, 0o600)
        print("GIRLS_TOILET_OPTIONAL_PLAN_REPAIR_PASS", before, after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
