#!/usr/bin/env python3
"""Patch the live school inventory gate for proven semantic/compound fixtures.

The patch is digest-bound and atomic.  It does not lower any required quantity;
it recognizes physically valid labels emitted by the generated-asset pipeline and
expands only dimension-attested two-fixture meshes into two auditable components.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import tempfile
from pathlib import Path


EXPECTED_INPUT_SHA256 = (
    "41c8387b00630032bcabdbc82fc7f3424dc1eac1cdc29a528c23b85dc160a4e9"
)


MULTIPLICITY_HELPER = r'''
def _expand_verified_fixture_multiplicity(
    requirements: tuple[InventoryRequirement, ...],
    semantic_objects: list[tuple[str, str, dict[str, Any]]],
) -> tuple[
    list[tuple[str, str, dict[str, Any]]],
    dict[str, dict[str, Any]],
    list[str],
]:
    """Represent real two-fixture meshes as independently auditable components.

    Generated restroom assets can legitimately contain two visible basins or two
    visible urinals in one mesh.  Quantity words alone are insufficient evidence,
    so expansion is limited to an exclusive inventory role, an explicit double/two
    fixture description, a passing dimension contract, and role-specific plausible
    measured dimensions.  The component IDs remain bound to the physical parent.
    """

    requirements_by_key = {requirement.key: requirement for requirement in requirements}
    expanded: list[tuple[str, str, dict[str, Any]]] = []
    evidence: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    for object_id, text, obj in semantic_objects:
        matching = [
            requirement.key
            for requirement in requirements
            if _matches(text, requirement)
        ]
        role: str | None = None
        component_label = ""
        minimum_dimensions = (0.0, 0.0, 0.0)
        component_suffixes: tuple[str, str] = ("left", "right")
        if re.search(
            r"\b(?:double|dual|two)[- ](?:wash[- ]?)?(?:basin|sink)s?\b",
            text,
            flags=re.IGNORECASE,
        ):
            role = "sinks"
            component_label = "sink basin"
            minimum_dimensions = (1.0, 0.35, 0.45)
            component_suffixes = ("sink_basin_left", "sink_basin_right")
        elif re.search(
            r"\b(?:double|dual|two)[- ]urinals?\b",
            text,
            flags=re.IGNORECASE,
        ):
            role = "urinals"
            component_label = "urinal"
            minimum_dimensions = (1.0, 0.25, 0.75)
            component_suffixes = ("urinal_left", "urinal_right")
        if role is None:
            expanded.append((object_id, text, obj))
            continue
        if matching != [role] or role not in requirements_by_key:
            issues.append(
                f"Multi-fixture {object_id} does not map exclusively to the {role} requirement."
            )
            expanded.append((object_id, text, obj))
            continue
        metadata = obj.get("metadata")
        contract = metadata.get("dimension_contract") if isinstance(metadata, dict) else None
        measured = (
            contract.get("measured_final_scene_dimensions_m")
            if isinstance(contract, dict)
            else None
        )
        valid_dimensions = (
            isinstance(contract, dict)
            and contract.get("status") == "pass"
            and isinstance(measured, list)
            and len(measured) == 3
            and all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in measured
            )
            and all(
                float(value) >= minimum
                for value, minimum in zip(measured, minimum_dimensions, strict=True)
            )
        )
        if not valid_dimensions:
            issues.append(
                f"Multi-fixture {object_id} lacks a passing, plausible measured-dimension contract."
            )
            expanded.append((object_id, text, obj))
            continue
        component_ids = [f"{object_id}#{suffix}" for suffix in component_suffixes]
        for component_id in component_ids:
            expanded.append((component_id, component_label, obj))
        evidence[object_id] = {
            "status": "pass",
            "requirement_key": role,
            "component_count": 2,
            "component_object_ids": component_ids,
            "parent_object_id": object_id,
            "parent_description": str(obj.get("description", "")),
            "measured_final_scene_dimensions_m": [float(value) for value in measured],
            "dimension_contract_status": contract.get("status"),
            "policy": f"verified_double_{role}_fixture_components",
        }
    return expanded, evidence, issues
'''.strip("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one source match, found {count}")
    return source.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    args = parser.parse_args()

    info = args.target.lstat()
    if not stat.S_ISREG(info.st_mode) or args.target.is_symlink():
        raise SystemExit("target is not a regular unlinked file")
    before = sha256(args.target)
    if before != EXPECTED_INPUT_SHA256:
        raise SystemExit(f"target digest mismatch: {before}")
    source = args.target.read_text(encoding="utf-8")

    source = replace_once(
        source,
        'r"\\bstudent\\s+(?:school\\s+)?desk\\b"',
        'r"\\bstudent\\s+(?:(?:school|classroom)\\s+)?desk\\b"',
        "student desk semantic variant",
    )
    source = replace_once(
        source,
        '_req("student_chairs", "student chairs", (r"\\bstudent\\s+chair\\b", r"\\bpupil\\s+chair\\b", r"\\blearner\\s+chair\\b"), 12, 12),',
        '_req("student_chairs", "student chairs", (r"\\bstudent\\s+(?:classroom\\s+)?chair\\b", r"\\bclassroom\\s+chair\\b", r"\\bpupil\\s+chair\\b", r"\\blearner\\s+chair\\b"), 12, 12),',
        "student chair semantic variants",
    )
    source = replace_once(
        source,
        '            r"\\binstructor\\s+chair\\b",\n',
        '            r"\\binstructor\\s+chair\\b",\n            r"\\bteacher\\b.*\\bchair\\b",\n',
        "teacher chair semantic variant",
    )
    source = replace_once(
        source,
        '        exclude=(r"\\b(?:white\\s*board|board)\\s+markers?\\b",),',
        '        exclude=(\n            r"\\b(?:white\\s*board|board)\\s+markers?\\b",\n            r"\\b(?:white\\s*board|chalk\\s*board|teaching\\s+board)\\b.*\\bmarkers?\\b",\n            r"\\bmarkers?\\s+(?:holder|cup|pot|tray)\\b",\n        ),',
        "general marker false-positive exclusions",
    )
    source = replace_once(
        source,
        '_req("board_markers", "board markers", (r"\\b(?:white\\s*board|board)\\s+markers?\\b",), 1),',
        '_req("board_markers", "board markers", (r"\\b(?:white\\s*board|board)\\s+markers?\\b", r"\\bdry[- ]erase\\s+markers?\\b"), 1),',
        "dry erase marker semantic variant",
    )
    partition_old = '_req("partitions", "stall partitions", (r"\\bstall\\s+partition\\b", r"\\brestroom\\s+partition\\b"), 2),'
    partition_new = '_req("partitions", "stall partitions", (r"\\bstall\\s+partition\\b", r"\\brestroom\\s+partition\\b", r"\\bprivacy\\s+screen\\s+divider\\b"), 2),'
    if source.count(partition_old) != 2:
        raise SystemExit(
            "restroom privacy divider semantics: expected the Boys and Girls rules"
        )
    source = source.replace(partition_old, partition_new)

    start = source.index("def _expand_verified_fixture_multiplicity(")
    end = source.index("\n\ndef evaluate_room_inventory(", start)
    source = source[:start] + MULTIPLICITY_HELPER + source[end:]

    args.backup_dir.mkdir(parents=True, exist_ok=True)
    backup = args.backup_dir / f"{args.target.name}.{before}.backup"
    if not backup.exists():
        shutil.copy2(args.target, backup)
    if sha256(backup) != before:
        raise SystemExit("backup digest mismatch")

    fd, temporary = tempfile.mkstemp(prefix=f".{args.target.name}.", dir=args.target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, stat.S_IMODE(info.st_mode))
        os.replace(temporary, args.target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print("SCHOOL_ROOM_CONTRACT_SEMANTIC_FIX_PASS", before, sha256(args.target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
