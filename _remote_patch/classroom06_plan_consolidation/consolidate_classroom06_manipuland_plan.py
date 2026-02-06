#!/usr/bin/env python3
"""Preserve Classroom 06 checkpoint 000 and consolidate duplicate item scopes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path


EXPECTED_PLAN_SHA256 = "32acfa5ced6dd40bff6b75835e21cd95c8d5c4fd8f26fbc41a374bc57db85f0f"
EXPECTED_RUNTIME_SHA256 = "8ad862129a21f236986573cb1d851223f8fb28e8246ddaff9a3ae27761bec4b7"
EXPECTED_CHECKPOINT_SHA256 = "f913910620ba3b5f03289d6284d8b06aab21276597f709705558335d1927f1b0"
EXPECTED_SELECTION_SHA256 = [
    "515bfd06b07e2d473551a0c361b4486278e0f526dfb552940321b04e2e187585",
    "7f322cd2c7cf68d6a231cfc50d4441cb182a4f2df411937048aafd5d7dc15a92",
    "c13a75a934302a834d36f87ba1bb1d48c829b4efd8d169b23d3976cf1222220b",
    "fbdfcc7532b6325ccec0bf21b1e1f9297bf9f894ec5653cb3daab14f55f80c54",
    "7ec1f5370d2bfa5494236371ab145604c0cc778c1987b836ae679583f3591e3c",
    "2ae0d7ede26dbccb37f76196de6839227ff074372c8351e6402b1729ab21bb94",
    "797865560c60dd7fcc06dad376fdf65d59ba0b3c08faa07e8f13a43662d8e0f2",
    "906b522f5a910f2768247f36d98a50b80c73be44b25fbdbfe17fba35d5a40c63",
    "9b62b3cda1afd04c50864635ec2727f5d9edc1211825ad313a91637d4b858002",
    "56d16dd3a1c370580f263765fe45c005ecc4e004710444a82612d20dcf8e5338",
    "dc1c1d329f9d4c5a0f043ef0c35719fd0a2e50561b6e025925788e9be38bf580",
    "904f5eb77594dceb4bf38cec774dd49c2067d3bf86493d19d6d247db65249d0d",
    "99e3386ed740d1dea98e2cf97a8f5b545842e89b44949284db19dfe5b9235981",
    "4743c91c5e72e35b7c4e35de0380ae1c146db1a376610173726a202dda3a32bc",
    "1df43fa770a787eff016511cc8cf1407881162b1f0e6d9e44b8426664dc4a06f",
    "3bec76078c6123777ae6c54f015ad2fc87b8539b4191c242d6a90c35e0d27457",
]
KEEP_INDICES = (0, 1, 2, 3, 4, 5, 10)


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
    checkpoint_path = (
        states / "manipuland_checkpoint_000_desk_0" / "completion_receipt.json"
    )
    checkpoint_dirs = sorted(states.glob("manipuland_checkpoint_*"))
    if [path.name for path in checkpoint_dirs] != ["manipuland_checkpoint_000_desk_0"]:
        raise SystemExit("unexpected checkpoint inventory")

    with (states / ".manipuland_checkpoint.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = sha(plan_path.read_bytes())
        if before != EXPECTED_PLAN_SHA256:
            raise SystemExit(f"unexpected plan digest: {before}")

        document = json.loads(plan_path.read_text(encoding="utf-8"))
        selections = document.get("selections")
        if (
            document.get("room_id") != "classroom_06"
            or document.get("status") != "pass"
            or document.get("checkpoint_runtime_sha256") != EXPECTED_RUNTIME_SHA256
            or document.get("attestation") != attest(document)
            or not isinstance(selections, list)
            or len(selections) != len(EXPECTED_SELECTION_SHA256)
            or document.get("selections_sha256") != sha(canonical(selections))
        ):
            raise SystemExit("input plan contract mismatch")
        for index, expected in enumerate(EXPECTED_SELECTION_SHA256):
            actual = sha(canonical(selections[index]))
            if actual != expected:
                raise SystemExit(
                    f"unexpected selection at index {index}: {actual} != {expected}"
                )

        if sha(checkpoint_path.read_bytes()) != EXPECTED_CHECKPOINT_SHA256:
            raise SystemExit("unexpected checkpoint digest")
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

        repaired = [dict(selections[index]) for index in KEEP_INDICES]
        repaired[2]["suggested_items"] = (
            "REQUIRED: labeled storage bins, textbooks, folders/worksheets, "
            "two backpacks fully stored in lower cubbies, glue/scissors caddy. "
            "Optional: pencil box set"
        )
        repaired[2]["prompt_constraints"] = (
            "Prompt requires textbooks, folders/worksheets, backpacks, storage "
            "bins, glue and scissors. Keep backpacks and the low-profile supply "
            "caddy fully inside supported cubbies so aisles remain clear."
        )
        repaired[3]["suggested_items"] = (
            "REQUIRED: notebook, pencil/eraser, water bottle. Optional: worksheet"
        )
        repaired[4]["suggested_items"] = (
            "REQUIRED: textbook, notebook, ruler. Optional: pencil case"
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
            "operation": "preserve_checkpoint_000_and_consolidate_duplicate_inventory",
            "room_id": "classroom_06",
            "plan_before_sha256": before,
            "plan_after_sha256": after,
            "preserved_checkpoint": checkpoint_path.parent.name,
            "preserved_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "kept_targets": [item["furniture_id"] for item in repaired],
            "removed_duplicate_or_floor_targets": [
                selections[index]["furniture_id"]
                for index in range(len(selections))
                if index not in KEEP_INDICES
            ],
            "required_categories_preserved": [
                "textbooks",
                "notebooks",
                "folders/worksheets",
                "pencil/marker holders",
                "erasers",
                "rulers",
                "glue",
                "scissors",
                "water bottles",
                "backpacks",
                "storage bins",
                "paper trays",
            ],
            "reason": (
                "Seven supported surfaces retain every prompt category while "
                "removing nine duplicate or floor-placement scopes."
            ),
        }
        receipt["attestation"] = attest(receipt)
        atomic_json(args.receipt, receipt, 0o600)
        print(f"CLASSROOM06_PLAN_CONSOLIDATION_PASS {before} {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
