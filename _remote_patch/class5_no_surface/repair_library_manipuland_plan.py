#!/usr/bin/env python3
"""Repair Library's unstable counter scope and trayless notice-board targets."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path


EXPECTED_PLAN_SHA256 = "3fadf9569552869bb1dfd5c58df1226e4c76283147e1b9c3b16ed4e4506a2965"
EXPECTED_SELECTION_SHA256 = {
    0: "2105be5d852306081b7cd862fe14fbc683fec611e9c345d8560d079797a78557",
    15: "627e161e443a5f644a33fdfdce1a6953eef23c75401df9bbd26c219cf210e9a3",
    18: "17926d3d3372861f039fa6985aa483b9ea158ce7602474097e0a3c3e8277f1ad",
    19: "7160b0539f9c7cc8a11b2d3f6efa672bf75b382e21a2c09b91661ac08ea3cc4f",
}


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
            document.get("room_id") != "library"
            or document.get("status") != "pass"
            or document.get("attestation") != attest(document)
            or not isinstance(selections, list)
            or len(selections) != 20
            or document.get("selections_sha256") != sha(canonical(selections))
        ):
            raise SystemExit("input plan contract mismatch")
        for index, expected in EXPECTED_SELECTION_SHA256.items():
            if sha(canonical(selections[index])) != expected:
                raise SystemExit(f"unexpected audited selection at index {index}")

        repaired = [dict(item) for item in selections]
        repaired[0]["suggested_items"] = (
            "REQUIRED: one flat checkout logbook and one or two low-profile book "
            "stacks. Optional: one pencil"
        )
        repaired[0]["prompt_constraints"] = (
            "Prompt: include a recognizable librarian/circulation counter and visible "
            "books/stationery. Use only low-profile, physically stable checkout details."
        )
        repaired[0]["style_notes"] = (
            "Group the stable book/logbook set at one staff-side corner and preserve a "
            "large clear transaction surface; no monitor, scanner, loose slips, or tall "
            "container."
        )

        repaired[15]["suggested_items"] = (
            "REQUIRED: one flat stack of library flyers/handouts and one event-calendar "
            "notice beside two magazines. Optional: one pencil"
        )
        repaired[15]["prompt_constraints"] = (
            "Prompt requires reading posters/notices and visible magazines/stationery. "
            "The generated wall boards are visual fixtures without horizontal support, "
            "so keep the take-home notices visible and accessible on this reading table."
        )
        repaired[15]["style_notes"] = (
            "Keep the notice stack flat, tidy, and centered with magazines; preserve clear "
            "edges for all reading chairs."
        )

        for index in (18, 19):
            repaired[index]["suggested_items"] = "REQUIRED: none. Optional: none"
            repaired[index]["prompt_constraints"] = (
                "The wall-mounted bulletin board itself remains the required visible "
                "school display; its generated model has no physical horizontal support. "
                "Notice handouts are assigned to reading_table_2. No specific manipulands "
                "required."
            )
            repaired[index]["style_notes"] = (
                "Preserve the installed board as a clean organized school display."
            )

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
            "operation": "stability_and_zero_surface_manipuland_plan_repair",
            "room_id": "library",
            "plan": str(plan),
            "before_sha256": before,
            "after_sha256": after,
            "stability_target": "circulation_counter_0",
            "notice_reroute_target": "reading_table_2",
            "zero_surface_targets": ["bulletin_board_0", "bulletin_board_1"],
            "required_notice_inventory_preserved": True,
        }
        receipt["attestation"] = attest(receipt)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.receipt, receipt, 0o600)
        print(f"PLAN_REPAIR_PASS {before} {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
