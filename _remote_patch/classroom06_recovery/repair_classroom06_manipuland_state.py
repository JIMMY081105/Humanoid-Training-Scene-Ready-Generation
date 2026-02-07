#!/usr/bin/env python3
"""Reroute Classroom 6's unstable cabinet-top bin to supported cubbies."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path


EXPECTED_PLAN_SHA256 = "32acfa5ced6dd40bff6b75835e21cd95c8d5c4fd8f26fbc41a374bc57db85f0f"
EXPECTED_CHECKPOINT_SHA256 = "f913910620ba3b5f03289d6284d8b06aab21276597f709705558335d1927f1b0"
EXPECTED_SELECTION_SHA256 = {
    0: "515bfd06b07e2d473551a0c361b4486278e0f526dfb552940321b04e2e187585",
    1: "7f322cd2c7cf68d6a231cfc50d4441cb182a4f2df411937048aafd5d7dc15a92",
    2: "c13a75a934302a834d36f87ba1bb1d48c829b4efd8d169b23d3976cf1222220b",
}
EXECUTING_RUNTIME_SHA256 = "8ad862129a21f236986573cb1d851223f8fb28e8246ddaff9a3ae27761bec4b7"


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


def backup(path: Path, backup_dir: Path, expected_sha256: str) -> None:
    target = backup_dir / f"{path.name}.{expected_sha256}"
    if not target.exists():
        shutil.copy2(path, target, follow_symlinks=False)
    elif sha(target.read_bytes()) != expected_sha256:
        raise SystemExit(f"existing backup digest mismatch: {target}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room-dir", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    room = args.room_dir.resolve(strict=True)
    runtime = args.runtime.resolve(strict=True)
    if sha(runtime.read_bytes()) != EXECUTING_RUNTIME_SHA256:
        raise SystemExit("production runtime is not the restored executing version")

    states = room / "scene_states"
    plan_path = states / "manipuland_furniture_plan.json"
    checkpoint_path = (
        states
        / "manipuland_checkpoint_000_desk_0"
        / "completion_receipt.json"
    )
    for path in (plan_path, checkpoint_path):
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise SystemExit(f"artifact is not an unlinked regular file: {path}")
    checkpoint_dirs = sorted(states.glob("manipuland_checkpoint_*"))
    if [path.name for path in checkpoint_dirs] != ["manipuland_checkpoint_000_desk_0"]:
        raise SystemExit("unexpected checkpoint inventory")

    with (states / ".manipuland_checkpoint.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan_before = sha(plan_path.read_bytes())
        checkpoint_before = sha(checkpoint_path.read_bytes())
        if plan_before != EXPECTED_PLAN_SHA256:
            raise SystemExit(f"unexpected plan digest: {plan_before}")
        if checkpoint_before != EXPECTED_CHECKPOINT_SHA256:
            raise SystemExit(f"unexpected checkpoint digest: {checkpoint_before}")

        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        selections = plan.get("selections")
        if (
            plan.get("room_id") != "classroom_06"
            or plan.get("status") != "pass"
            or plan.get("checkpoint_runtime_sha256") != EXECUTING_RUNTIME_SHA256
            or plan.get("attestation") != attest(plan)
            or not isinstance(selections, list)
            or len(selections) != 16
            or plan.get("selections_sha256") != sha(canonical(selections))
        ):
            raise SystemExit("input plan contract mismatch")
        for index, expected in EXPECTED_SELECTION_SHA256.items():
            if sha(canonical(selections[index])) != expected:
                raise SystemExit(f"unexpected selection at index {index}")

        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("status") != "pass"
            or checkpoint.get("furniture_index") != 0
            or checkpoint.get("furniture_id") != "desk_0"
            or checkpoint.get("selection") != selections[0]
            or checkpoint.get("selection_sha256") != EXPECTED_SELECTION_SHA256[0]
            or checkpoint.get("checkpoint_runtime_sha256")
            != EXECUTING_RUNTIME_SHA256
            or checkpoint.get("attestation") != attest(checkpoint)
        ):
            raise SystemExit("input checkpoint contract mismatch")

        repaired_selections = [dict(item) for item in selections]
        repaired_selections[1]["suggested_items"] = (
            "REQUIRED: extra paper tray(s). Optional: stack of worksheets, tissue "
            "box, spare markers/colored pencils container"
        )
        repaired_selections[1]["prompt_constraints"] = (
            "Prompt requires storage bins and paper trays. Keep the extra paper trays "
            "on this cabinet top; the labeled storage bins are placed in "
            "cubby_bookshelf_0 because the retrieved open bin had invalid rigid-body "
            "inertia and failed strict stability validation."
        )
        repaired_selections[1]["style_notes"] = (
            "Top surface only; low-profile and stable; 1–3 items with clear edges."
        )
        repaired_selections[2]["suggested_items"] = (
            "REQUIRED: labeled storage bin(s), textbooks, folders/worksheets. "
            "Optional: pencil box set, glue/scissors caddy, a couple of backpacks "
            "stored in cubbies"
        )
        repaired_selections[2]["prompt_constraints"] = (
            "Prompt: add textbooks, folders/worksheets, backpacks, storage bins and "
            "classroom supplies. The labeled storage-bin requirement is fulfilled "
            "inside these physically supported cubbies rather than by the unstable "
            "retrieved open bin on the cabinet top."
        )

        repaired_plan = dict(plan)
        repaired_plan["selections"] = repaired_selections
        repaired_plan["selections_sha256"] = sha(canonical(repaired_selections))
        repaired_plan["attestation"] = attest(repaired_plan)

        args.backup_dir.mkdir(parents=True, exist_ok=True)
        backup(plan_path, args.backup_dir, plan_before)
        atomic_json(plan_path, repaired_plan, plan_path.stat().st_mode & 0o777)
        plan_after = sha(plan_path.read_bytes())
        checkpoint_after = sha(checkpoint_path.read_bytes())
        if checkpoint_after != checkpoint_before:
            raise SystemExit("completed desk checkpoint changed during plan repair")

        receipt = {
            "schema_version": 1,
            "status": "pass",
            "operation": "reroute_unstable_cabinet_bin_to_supported_cubbies",
            "room_id": "classroom_06",
            "plan_before_sha256": plan_before,
            "plan_after_sha256": plan_after,
            "checkpoint_before_sha256": checkpoint_before,
            "checkpoint_after_sha256": checkpoint_after,
            "preserved_checkpoint": "manipuland_checkpoint_000_desk_0",
            "preserved_selection_sha256": EXPECTED_SELECTION_SHA256[0],
            "executing_runtime_sha256": EXECUTING_RUNTIME_SHA256,
            "removed_unstable_asset_role": "open storage bin on cabinet top",
            "reroute_target": "cubby_bookshelf_0",
            "required_categories_preserved": [
                "labeled storage bins",
                "paper trays",
                "textbooks",
                "folders/worksheets",
            ],
        }
        receipt["attestation"] = attest(receipt)
        atomic_json(args.receipt, receipt, 0o600)
        print(
            "CLASSROOM06_STATE_REPAIR_PASS",
            plan_before,
            plan_after,
            checkpoint_before,
            checkpoint_after,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
