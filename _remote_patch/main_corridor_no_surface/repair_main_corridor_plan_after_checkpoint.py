#!/usr/bin/env python3
"""Drop optional zero-surface bulletin-board scopes after corridor checkpoint 000."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path


EXPECTED_PLAN_SHA256 = "263a4208a1bc32366d5efa797e0d9cc33dc258bf1a5ddc908de6ec21e3e6a8fc"
EXPECTED_RUNTIME_SHA256 = "8ad862129a21f236986573cb1d851223f8fb28e8246ddaff9a3ae27761bec4b7"
EXPECTED_SELECTION_SHA256 = (
    "1d88d11c4d5df5aaf6c44f4a2c72d8beca97f58773cf6104c2a2e6cf31ab075f",
    "9f4e576ce094c9629414bbabb19cedb95ae2631b0a2ce49071dc679587b60d86",
    "ba1da28baee050eb7019543c02dced33dba284776a459d8511fa9b7680cbe91a",
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
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if len(args.expected_checkpoint_sha256) != 64:
        raise SystemExit("invalid expected checkpoint digest")

    room = args.room_dir.resolve(strict=True)
    states = room / "scene_states"
    plan_path = states / "manipuland_furniture_plan.json"
    checkpoint_path = (
        states / "manipuland_checkpoint_000_console_table_0" / "completion_receipt.json"
    )

    with (states / ".manipuland_checkpoint.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        checkpoint_dirs = sorted(states.glob("manipuland_checkpoint_*"))
        if [path.name for path in checkpoint_dirs] != [
            "manipuland_checkpoint_000_console_table_0"
        ]:
            raise SystemExit("unexpected checkpoint inventory")

        before = sha(plan_path.read_bytes())
        if before != EXPECTED_PLAN_SHA256:
            raise SystemExit(f"unexpected plan digest: {before}")
        document = json.loads(plan_path.read_text(encoding="utf-8"))
        selections = document.get("selections")
        if (
            document.get("room_id") != "main_corridor"
            or document.get("status") != "pass"
            or document.get("checkpoint_runtime_sha256") != EXPECTED_RUNTIME_SHA256
            or document.get("attestation") != attest(document)
            or not isinstance(selections, list)
            or len(selections) != 3
            or document.get("selections_sha256") != sha(canonical(selections))
        ):
            raise SystemExit("input plan contract mismatch")
        for index, expected in enumerate(EXPECTED_SELECTION_SHA256):
            if sha(canonical(selections[index])) != expected:
                raise SystemExit(f"selection {index} changed")

        checkpoint_digest = sha(checkpoint_path.read_bytes())
        if checkpoint_digest != args.expected_checkpoint_sha256:
            raise SystemExit("checkpoint digest differs from the supplied exact hash")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("status") != "pass"
            or checkpoint.get("furniture_index") != 0
            or checkpoint.get("selection") != selections[0]
            or checkpoint.get("selection_sha256") != EXPECTED_SELECTION_SHA256[0]
            or checkpoint.get("checkpoint_runtime_sha256") != EXPECTED_RUNTIME_SHA256
            or checkpoint.get("attestation") != attest(checkpoint)
        ):
            raise SystemExit("checkpoint contract mismatch")

        output = dict(document)
        output["selections"] = [selections[0]]
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
            "operation": "preserve_console_checkpoint_and_drop_optional_zero_surface_scopes",
            "room_id": "main_corridor",
            "plan_before_sha256": before,
            "plan_after_sha256": after,
            "preserved_checkpoint": checkpoint_path.parent.name,
            "preserved_checkpoint_sha256": checkpoint_digest,
            "kept_target": selections[0]["furniture_id"],
            "removed_optional_targets": [
                selections[1]["furniture_id"], selections[2]["furniture_id"]
            ],
            "required_prompt_content_preserved": [
                "visitor console details from checkpoint 000",
                "both bulletin/display furniture objects remain in the room",
                "all corridor furniture, wall, ceiling and circulation content",
            ],
            "reason": (
                "The two display boards already satisfy the required display-area "
                "furniture prompt; their posting contents are explicitly optional, "
                "and vertical boards provide no horizontal manipuland support surface."
            ),
        }
        receipt["attestation"] = attest(receipt)
        atomic_json(args.receipt, receipt, 0o600)
        print(f"MAIN_CORRIDOR_PLAN_REPAIR_PASS {before} {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
