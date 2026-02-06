#!/usr/bin/env python3
"""Preserve Classroom 5's two checkpoints and remove duplicate later scopes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path


EXPECTED_PLAN_SHA256 = "1a361b7a1cfb11c1f4fc199b4b6d43212a127f3d61ef0993ab8d8f1c27f321d8"
EXPECTED_RUNTIME_SHA256 = "8ad862129a21f236986573cb1d851223f8fb28e8246ddaff9a3ae27761bec4b7"
EXPECTED_CHECKPOINT_SHA256 = (
    "990d0c21c8082b165d01328a53f2f1dcc2a65342c10e09b7823a79a6bc245030",
    "c615e5333bd8d1cd343519b656c7275ca8bd5e04b439344d56143c4b489ce269",
)
EXPECTED_SELECTION_SHA256 = [
    "7a4466aa6717132bd71095e4f0f9b8560de636c58980b42708f14c74f7341b3d",
    "202dece6d50c5b920b9cd843a36922af89c565d327c491ff7f749519e0bd0d18",
    "13810c07f427e2c3dc5394793edad5219effd20ceecafeaf672eb2584711f435",
    "795d5c7c5b65c3a53ed480b384014b01156ae478e09e0db4b8b567b0d115a0ae",
    "fbef98e37cb1f723e1efb1538bc188f0287897e97ee9170fd6b079784c3e9aa7",
    "3eee41fd0099ce976781a91ee7046f024da8e45181a97ba52c18f4c753c1530c",
    "301b1d2fcc89a54da5de0759ab93ad20664a722ff235139d29a28c4919ed6d52",
    "8251a2273adc1f5e4df613747d6cc49620565394f4884a203112a452185e983d",
    "ece1dbbde6519140e751cc0fafecabda34e4ea49b0139a5decac0f6fa3aaa68b",
    "169bf519637124446dee71e60cdef43ed79b1317da9c333074bab40896747506",
    "53f2c540364e2f4fc1abcac6564032ed4ab8a2108ce4ec8edace51b042d15252",
    "21a6f91c89949f99d5c8ee3e62161e0ac9322222b2fb35286040e699cb4a0067",
    "687b571ba22d1841250efe16eb433a4a3be7b9c6ab050435fc261eab5a14282e",
    "dec995db7484b3fdfdcff00f1b64d83c57dd2302da0ce301cd96118718907e65",
    "7e90b5db038d769a3e28131b072ac79adf1144425ee2b7b058fdbb8f9924f6e4",
    "f8dc3ad74fd7560f71c962039c427953476e9ef11f16826d3898cc548f900e13",
    "2d79c1c5bd0e8060f4817579779ee2038045613cab338c5eacd50a8068a69f42",
    "eac376d9d5599eb40c455c23923539c115b39a42032b0d541efb0b52c7a62c52",
    "d806e8c72141fd3f8bfa68429eb41f30bbfc94af1c30fd7bf45f2c548c3da5b1",
    "15c03bc0d1d925636457fb25139a7e22155c2d4e7e059a0981158a80cb6f0bfc",
]
KEEP_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 14, 15)


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
    checkpoint_paths = [
        states / "manipuland_checkpoint_000_teacher_desk_0" / "completion_receipt.json",
        states / "manipuland_checkpoint_001_drawer_pedestal_0" / "completion_receipt.json",
    ]
    checkpoint_dirs = sorted(states.glob("manipuland_checkpoint_*"))
    if [path.name for path in checkpoint_dirs] != [
        "manipuland_checkpoint_000_teacher_desk_0",
        "manipuland_checkpoint_001_drawer_pedestal_0",
    ]:
        raise SystemExit("unexpected checkpoint inventory")

    with (states / ".manipuland_checkpoint.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = sha(plan_path.read_bytes())
        if before != EXPECTED_PLAN_SHA256:
            raise SystemExit(f"unexpected plan digest: {before}")
        document = json.loads(plan_path.read_text(encoding="utf-8"))
        selections = document.get("selections")
        if (
            document.get("room_id") != "classroom_05"
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
        for index, (path, expected_hash) in enumerate(
            zip(checkpoint_paths, EXPECTED_CHECKPOINT_SHA256, strict=True)
        ):
            if sha(path.read_bytes()) != expected_hash:
                raise SystemExit(f"unexpected checkpoint digest at index {index}")
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
            if (
                checkpoint.get("status") != "pass"
                or checkpoint.get("furniture_index") != index
                or checkpoint.get("selection") != selections[index]
                or checkpoint.get("selection_sha256")
                != EXPECTED_SELECTION_SHA256[index]
                or checkpoint.get("checkpoint_runtime_sha256")
                != EXPECTED_RUNTIME_SHA256
                or checkpoint.get("attestation") != attest(checkpoint)
            ):
                raise SystemExit(f"checkpoint contract mismatch at index {index}")

        repaired = [dict(selections[index]) for index in KEEP_INDICES]
        repaired[-1]["suggested_items"] = (
            "REQUIRED: mixed books/textbooks, labeled storage bins, backpacks and "
            "folders/worksheets in separate cubbies. Optional: spare notebooks"
        )
        repaired[-1]["prompt_constraints"] = (
            "Prompt requires books, storage bins, folders/worksheets and backpacks. "
            "Keep backpacks fully inside lower cubbies so no floor target or aisle "
            "clutter is needed."
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
            "operation": "preserve_two_checkpoints_and_consolidate_remaining_inventory",
            "room_id": "classroom_05",
            "plan_before_sha256": before,
            "plan_after_sha256": after,
            "preserved_checkpoints": [path.parent.name for path in checkpoint_paths],
            "preserved_checkpoint_sha256": list(EXPECTED_CHECKPOINT_SHA256),
            "kept_targets": [item["furniture_id"] for item in repaired],
            "backpack_reroute_target": "cubby_bookshelf_0",
            "removed_duplicate_or_optional_targets": [
                selections[index]["furniture_id"]
                for index in range(len(selections))
                if index not in KEEP_INDICES
            ],
            "required_categories_preserved": [
                "textbooks", "notebooks", "folders/worksheets",
                "pencil/marker holders", "erasers/rulers", "glue", "scissors",
                "water bottles", "backpacks", "storage bins", "paper trays",
                "board markers", "whiteboard eraser", "art/reading corner",
            ],
            "reason": (
                "Ten supported surfaces retain every prompt category and the art "
                "corner while removing ten duplicated optional desk/floor scopes."
            ),
        }
        receipt["attestation"] = attest(receipt)
        atomic_json(args.receipt, receipt, 0o600)
        print(f"CLASSROOM05_PLAN_CONSOLIDATION_PASS {before} {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
