#!/usr/bin/env python3
"""Remove only optional rolling pencils from the attested library plan."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path


EXPECTED_PLAN_SHA256 = "6ea0135cd354e0e5cb5648273df51ee2b3fa996f1b50281bc23e10a4ad8095d8"
TARGETS = {
    "circulation_counter_0": {
        "before": "REQUIRED: one flat checkout logbook and one or two low-profile book stacks. Optional: one pencil",
        "after": "REQUIRED: one flat checkout logbook and one or two low-profile book stacks. Optional: none",
        "required": ("checkout logbook", "book stacks"),
    },
    "reading_table_2": {
        "before": "REQUIRED: one flat stack of library flyers/handouts and one event-calendar notice beside two magazines. Optional: one pencil",
        "after": "REQUIRED: one flat stack of library flyers/handouts and one event-calendar notice beside two magazines. Optional: none",
        "required": ("flyers/handouts", "event-calendar notice", "magazines"),
    },
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
            raise SystemExit("library plan repair requires zero committed checkpoints")
        before_bytes = plan_path.read_bytes()
        before_sha = sha(before_bytes)
        if before_sha != EXPECTED_PLAN_SHA256:
            raise SystemExit(f"unexpected plan digest: {before_sha}")
        document = json.loads(before_bytes)
        selections = document.get("selections")
        if (
            document.get("status") != "pass"
            or document.get("room_id") != "library"
            or document.get("attestation") != attest(document)
            or not isinstance(selections, list)
            or document.get("selections_sha256") != sha(canonical(selections))
        ):
            raise SystemExit("library plan contract mismatch")

        output_selections = [dict(selection) for selection in selections]
        changed: list[str] = []
        for selection in output_selections:
            furniture_id = selection.get("furniture_id")
            target = TARGETS.get(furniture_id)
            if target is None:
                continue
            if selection.get("suggested_items") != target["before"]:
                raise SystemExit(f"unexpected optional-pencil scope: {furniture_id}")
            required_text = selection["suggested_items"]
            if any(token not in required_text for token in target["required"]):
                raise SystemExit(f"required content is missing: {furniture_id}")
            selection["suggested_items"] = target["after"]
            selection["style_notes"] = (
                selection["style_notes"]
                + " Do not add a loose pencil or another slender rolling object."
            )
            changed.append(str(furniture_id))
        if changed != ["circulation_counter_0", "reading_table_2"]:
            raise SystemExit(f"unexpected changed targets: {changed}")

        output = dict(document)
        output["selections"] = output_selections
        output["selections_sha256"] = sha(canonical(output_selections))
        output["attestation"] = attest(output)

        args.backup_dir.mkdir(parents=True, exist_ok=True)
        backup = args.backup_dir / f"{plan_path.name}.{before_sha}"
        if not backup.exists():
            shutil.copy2(plan_path, backup, follow_symlinks=False)
        elif sha(backup.read_bytes()) != before_sha:
            raise SystemExit("existing backup digest mismatch")
        atomic_json(plan_path, output, plan_path.stat().st_mode & 0o777)
        after_sha = sha(plan_path.read_bytes())

        receipt: dict[str, object] = {
            "schema_version": 1,
            "status": "pass",
            "operation": "drop_optional_unstable_library_pencils",
            "room_id": "library",
            "plan_before_sha256": before_sha,
            "plan_after_sha256": after_sha,
            "changed_targets": changed,
            "required_prompt_content_modified": False,
            "required_items_preserved": {
                "circulation_counter_0": ["checkout logbook", "book stacks"],
                "reading_table_2": [
                    "library flyers/handouts",
                    "event-calendar notice",
                    "magazines",
                ],
            },
            "reason": (
                "A loose optional pencil rolled 0.101 m and rotated 105.2 degrees "
                "during the mandatory five-second stability simulation."
            ),
        }
        receipt["attestation"] = attest(receipt)
        atomic_json(args.receipt, receipt, 0o600)
        print("LIBRARY_OPTIONAL_PENCIL_REPAIR_PASS", before_sha, after_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
