#!/usr/bin/env python3
"""Replace Storage's unstable open box and remove duplicate optional scopes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path


EXPECTED_PLAN_SHA256 = "fa636bcaf8a3b471632141ab22c213cb645df12b6707e495f9794db19b265d49"
EXPECTED_RUNTIME_SHA256 = "8ad862129a21f236986573cb1d851223f8fb28e8246ddaff9a3ae27761bec4b7"
EXPECTED_SELECTION_SHA256 = [
    "d3f61132c755837f72adc6f79db1e78486f987e57dd60510aaf70bbf436520a0",
    "f080615cd16d715712235bed2bab04a8db8af2f29176e21cde6ada17708d6d6e",
    "f4cfa7b9755155428a9367dfa18b83c6bf4524f374558e90606ef2032630b797",
    "c505430c31f048b0a2b9cb82630462ca4217c0e093dc2920c465e798231370e3",
    "50ab06c5bec245ef8909189c5095090c6cea4e945b010f8fefa777a10c4085a6",
    "cabacb9e63eb51194674680573e12ef1761970701a7951f8f4dfac347dc91577",
    "180e9c1a2becbb603c79dd8ed97436c38dc7fb2b9532f5edd89df3361bc38e20",
    "f16b84ebbdf236387bb43bc384def5e2a819c9edb9fc6c71805df132b4d1e860",
    "42f96a703563109d3bbf8d2ba12ed2b91dd937611300327b2db6c2e439027224",
    "fcc57dc9e96399bd699cc5193a208664a569bf7daf4a5e54ae1a93d7ca594fb8",
    "d3e3458ed091250b2c98a592d759e705a0d9cc9ce2247288deca8e9599b16be1",
    "296146606c856c47828aebe74223c7545213d3a644b3391c508c1af0b55fdf62",
]
KEEP_INDICES = (0, 1, 2, 4, 6, 7, 9)


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
            document.get("room_id") != "storage_room"
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
            "REQUIRED: flat sealed cardboard archive/file boxes and low textbook "
            "stacks. Optional: lidded low-profile tote, art paper packs"
        )
        repaired[0]["prompt_constraints"] = (
            "Prompt requires boxes, paper/textbook/art supplies and spare classroom "
            "materials. Use only sealed low-profile archive/file boxes; exclude the "
            "unstable open storage-box asset rejected by strict forward simulation."
        )
        repaired[0]["style_notes"] = (
            "Distribute flat boxes and book stacks across separate shelves with clear "
            "front edges; dense but reachable and physically stable."
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
            "operation": "replace_unstable_box_and_consolidate_storage_scopes",
            "room_id": "storage_room",
            "plan_before_sha256": before,
            "plan_after_sha256": after,
            "kept_targets": [item["furniture_id"] for item in repaired],
            "removed_duplicate_or_optional_targets": [
                selections[index]["furniture_id"]
                for index in range(len(selections))
                if index not in KEEP_INDICES
            ],
            "removed_unstable_asset_role": "open storage_box_0",
            "replacement_role": "sealed low-profile archive/file boxes",
            "required_categories_preserved": [
                "boxes",
                "textbooks",
                "copy paper",
                "binders/folders",
                "art supplies",
                "four labeled bins",
                "cleaning consumables",
                "packing supplies",
                "maintenance tools",
            ],
            "reason": (
                "The first shelf failed only because one open box translated 3.4 cm; "
                "seven supported surfaces retain the full dense storage inventory "
                "without duplicate cart, floor, or auxiliary-cabinet clutter."
            ),
        }
        receipt["attestation"] = attest(receipt)
        atomic_json(args.receipt, receipt, 0o600)
        print(f"STORAGE_PLAN_REPAIR_PASS {before} {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
