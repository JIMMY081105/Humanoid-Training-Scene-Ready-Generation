#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_MIRROR_DESCRIPTION = (
    "framed photograph print of a neutral reflective glass surface (gradient), "
    "portrait orientation"
)
MIRROR_IDS = ("photograph_print_1", "photograph_print_2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def require_regular(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise SystemExit(f"not a regular unlinked file: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--expected-state-sha256", required=True)
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    require_regular(args.state)
    before = sha256(args.state)
    if before != args.expected_state_sha256:
        raise SystemExit(f"state digest mismatch: {before}")
    document = json.loads(args.state.read_text(encoding="utf-8"))
    objects = document.get("objects")
    if not isinstance(objects, dict):
        raise SystemExit("state objects mapping is missing")

    repaired = []
    mirror_positions = []
    for object_id in MIRROR_IDS:
        obj = objects.get(object_id)
        if not isinstance(obj, dict):
            raise SystemExit(f"missing mirror-panel object: {object_id}")
        if obj.get("description") != EXPECTED_MIRROR_DESCRIPTION:
            raise SystemExit(f"unexpected mirror-panel description: {object_id}")
        placement = obj.get("placement_info")
        dimensions = (obj.get("metadata") or {}).get("dimension_contract", {}).get(
            "measured_final_scene_dimensions_m"
        )
        if (
            not isinstance(placement, dict)
            or placement.get("parent_surface_id") != "girls_toilet_east"
            or not isinstance(dimensions, list)
            or len(dimensions) != 3
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in dimensions)
            or not (0.60 <= dimensions[0] <= 0.70)
            or not (0.005 <= dimensions[1] <= 0.04)
            or not (0.90 <= dimensions[2] <= 1.00)
        ):
            raise SystemExit(f"invalid physical mirror-panel evidence: {object_id}")
        position = placement.get("position_2d")
        if not isinstance(position, list) or len(position) != 2:
            raise SystemExit(f"invalid mirror placement: {object_id}")
        mirror_positions.append(float(position[0]))
        metadata = obj.setdefault("metadata", {})
        metadata["semantic_repair"] = {
            "original_name": obj.get("name"),
            "original_description": obj.get("description"),
            "reason": "wall critic verified mirror-like reflective panels with correct portrait dimensions",
            "schema_version": 1,
        }
        obj["name"] = "restroom_mirror"
        obj["description"] = (
            "Portrait school restroom mirror with a neutral reflective glass face and thin frame"
        )
        repaired.append(object_id)
    if abs(mirror_positions[0] - mirror_positions[1]) < 0.70:
        raise SystemExit("mirror panels do not have the required clear horizontal separation")

    sink = objects.get("sink_vanity_0")
    if not isinstance(sink, dict) or "double-basin sink vanity" not in str(
        sink.get("description", "")
    ).lower():
        raise SystemExit("verified double-basin sink vanity is missing")
    sink_contract = (sink.get("metadata") or {}).get("dimension_contract")
    if not isinstance(sink_contract, dict) or sink_contract.get("status") != "pass":
        raise SystemExit("double-basin vanity dimension contract is not passing")

    args.backup_dir.mkdir(parents=True, exist_ok=True)
    backup = args.backup_dir / f"{args.state.name}.{before}.backup"
    if not backup.exists():
        shutil.copy2(args.state, backup)
    require_regular(backup)
    if sha256(backup) != before:
        raise SystemExit("backup digest mismatch")
    atomic_json(args.state, document)
    after = sha256(args.state)
    receipt = {
        "schema_version": 1,
        "status": "pass",
        "operation": "bind_verified_reflective_panels_as_restroom_mirrors",
        "state_path": str(args.state.resolve()),
        "state_sha256_before": before,
        "state_sha256_after": after,
        "backup_path": str(backup.resolve()),
        "backup_sha256": sha256(backup),
        "repaired_object_ids": repaired,
        "verified_double_basin_parent": "sink_vanity_0",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    receipt["attestation"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_json(args.receipt, receipt)
    print("GIRLS_TOILET_SEMANTIC_REPAIR_PASS", before, after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
