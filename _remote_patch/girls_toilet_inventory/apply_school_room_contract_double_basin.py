#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import tempfile
from pathlib import Path


HELPER = r'''
def _expand_verified_fixture_multiplicity(
    requirements: tuple[InventoryRequirement, ...],
    semantic_objects: list[tuple[str, str, dict[str, Any]]],
) -> tuple[
    list[tuple[str, str, dict[str, Any]]],
    dict[str, dict[str, Any]],
    list[str],
]:
    """Represent real multi-fixture assets as independently auditable components.

    A generated double-basin vanity is one furniture model but contains two real
    sink fixtures. Treating it as one sink rejects a visually and physically
    correct school-restroom layout; blindly trusting a quantity word would be too
    weak. This narrow rule therefore requires the object to match only the sink
    inventory role, carry a passing dimension contract, and have plausible measured
    dimensions for two basins. The two component identities remain bound to the
    one hash-verified parent mesh and are exposed in the gate evidence.
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
        is_double_basin = bool(
            re.search(r"\b(?:double|dual|two)[- ]basin\b", text, flags=re.IGNORECASE)
        )
        if not is_double_basin:
            expanded.append((object_id, text, obj))
            continue
        if matching != ["sinks"] or "sinks" not in requirements_by_key:
            issues.append(
                f"Multi-basin fixture {object_id} does not map exclusively to the sinks requirement."
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
            and float(measured[0]) >= 1.0
            and float(measured[1]) >= 0.35
            and float(measured[2]) >= 0.45
        )
        if not valid_dimensions:
            issues.append(
                f"Multi-basin fixture {object_id} lacks a passing, plausible measured-dimension contract."
            )
            expanded.append((object_id, text, obj))
            continue
        component_ids = [
            f"{object_id}#sink_basin_left",
            f"{object_id}#sink_basin_right",
        ]
        for component_id in component_ids:
            expanded.append((component_id, "sink basin", obj))
        evidence[object_id] = {
            "status": "pass",
            "requirement_key": "sinks",
            "component_count": 2,
            "component_object_ids": component_ids,
            "parent_object_id": object_id,
            "parent_description": str(obj.get("description", "")),
            "measured_final_scene_dimensions_m": [float(value) for value in measured],
            "dimension_contract_status": contract.get("status"),
            "policy": "verified_double_basin_fixture_components",
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
    if source.count(old) != 1:
        raise SystemExit(f"{label}: expected exactly one source match, found {source.count(old)}")
    return source.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--backup-dir", required=True, type=Path)
    args = parser.parse_args()
    info = args.target.lstat()
    if not stat.S_ISREG(info.st_mode) or args.target.is_symlink():
        raise SystemExit("target is not a regular unlinked file")
    before = sha256(args.target)
    if before != args.expected_sha256:
        raise SystemExit(f"target digest mismatch: {before}")
    source = args.target.read_text(encoding="utf-8")
    anchor = '''def inventory_requirements(room_id: str) -> tuple[InventoryRequirement, ...]:
    if room_id.startswith("classroom_"):
        return CLASSROOM_INVENTORY
    return ROOM_INVENTORIES.get(room_id, ())


def evaluate_room_inventory('''
    replacement = anchor.replace("\n\n\ndef evaluate_room_inventory(", f"\n\n\n{HELPER}\n\n\ndef evaluate_room_inventory(")
    source = replace_once(source, anchor, replacement, "helper insertion")
    old_assignment = "    matches, raw_matches = _minimum_unique_assignment(requirements, semantic_objects)"
    new_assignment = '''    assignment_objects, fixture_multiplicity, multiplicity_issues = (
        _expand_verified_fixture_multiplicity(requirements, semantic_objects)
    )
    if multiplicity_issues:
        issues.extend(multiplicity_issues)
        repairs.append(
            "Replace invalid multi-fixture quantity claims with independently represented fixtures."
        )
    matches, raw_matches = _minimum_unique_assignment(requirements, assignment_objects)'''
    source = replace_once(source, old_assignment, new_assignment, "assignment replacement")
    old_return = '''        "physical_object_evidence": physical_evidence,
        "invalid_physical_candidates": invalid_physical_candidates,
        "spatial_checks": spatial_checks,'''
    new_return = '''        "physical_object_evidence": physical_evidence,
        "invalid_physical_candidates": invalid_physical_candidates,
        "fixture_multiplicity_evidence": fixture_multiplicity,
        "spatial_checks": spatial_checks,'''
    source = replace_once(source, old_return, new_return, "evidence insertion")

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
    print("SCHOOL_ROOM_CONTRACT_DOUBLE_BASIN_PATCH_PASS", before, sha256(args.target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
