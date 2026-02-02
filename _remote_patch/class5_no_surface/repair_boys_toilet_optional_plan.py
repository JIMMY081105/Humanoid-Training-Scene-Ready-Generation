#!/usr/bin/env python3
"""Bind explicit REQUIRED:none Boys Toilet targets as safe zero-surface no-ops."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path


EXPECTED_PLAN_SHA256 = "93f94b209bfc9b292572ab586cfe795da9a1157fe32a83bf29e70ed2d6081eed"
EXPECTED_TARGETS = ("bulletin_board_0", "floating_shelf_0", "vanity_0")
NOOP_SENTENCE = "No specific manipulands required."


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


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
    plan = states / "manipuland_furniture_plan.json"
    if plan.is_symlink() or not plan.is_file() or plan.stat().st_nlink != 1:
        raise SystemExit(f"plan is not an unlinked regular file: {plan}")
    if list(states.glob("manipuland_checkpoint_*")):
        raise SystemExit("refusing to alter a plan after a furniture checkpoint exists")

    with (states / ".manipuland_checkpoint.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = sha(plan.read_bytes())
        if before != EXPECTED_PLAN_SHA256:
            raise SystemExit(f"unexpected plan digest: {before}")
        document = json.loads(plan.read_text(encoding="utf-8"))
        selections = document.get("selections")
        if (
            document.get("room_id") != "boys_toilet"
            or document.get("status") != "pass"
            or document.get("attestation") != attest(document)
            or not isinstance(selections, list)
            or document.get("selections_sha256") != sha(canonical(selections))
            or tuple(item.get("furniture_id") for item in selections) != EXPECTED_TARGETS
        ):
            raise SystemExit("input plan contract mismatch")

        repaired: list[dict[str, object]] = []
        for item in selections:
            if not str(item.get("suggested_items", "")).casefold().startswith(
                "required: none."
            ):
                raise SystemExit(f"target is not explicitly optional-only: {item}")
            updated = dict(item)
            constraints = str(updated["prompt_constraints"]).rstrip()
            if NOOP_SENTENCE.casefold() not in constraints.casefold():
                updated["prompt_constraints"] = f"{constraints} {NOOP_SENTENCE}"
            repaired.append(updated)

        output = dict(document)
        output["selections"] = repaired
        output["selections_sha256"] = sha(canonical(repaired))
        output["attestation"] = attest(output)

        args.backup_dir.mkdir(parents=True, exist_ok=True)
        backup = args.backup_dir / f"{plan.name}.{before}"
        if not backup.exists():
            shutil.copy2(plan, backup, follow_symlinks=False)
        elif sha(backup.read_bytes()) != before:
            raise SystemExit(f"existing backup digest mismatch: {backup}")
        atomic_json(plan, output, plan.stat().st_mode & 0o777)
        after = sha(plan.read_bytes())

        receipt = {
            "schema_version": 1,
            "status": "pass",
            "operation": "bind_explicit_required_none_zero_surface_policy",
            "room_id": "boys_toilet",
            "plan": str(plan),
            "before_sha256": before,
            "after_sha256": after,
            "targets": list(EXPECTED_TARGETS),
            "required_inventory_changed": False,
        }
        receipt["attestation"] = attest(receipt)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.receipt, receipt, 0o600)
        print(f"PLAN_REPAIR_PASS {before} {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
