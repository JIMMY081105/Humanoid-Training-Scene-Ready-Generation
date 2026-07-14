#!/usr/bin/env python3
"""Immutable factory room prompts, distinct inventory gates, and articulated roles."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

try:
    from .school_room_contract import (
        InventoryRequirement, _iter_state_objects, _matches,
        _minimum_unique_assignment, object_semantic_text, physical_object_evidence,
    )
except ImportError:
    from school_room_contract import (  # type: ignore
        InventoryRequirement, _iter_state_objects, _matches,
        _minimum_unique_assignment, object_semantic_text, physical_object_evidence,
    )


PROFILE = "factory_reference_20260713"
CONTRACT_ID = "food_factory_full_quality_20260713"
CONTRACT_MARKER = "[FACTORY_ROOM_CONTRACT_V1"
CONTEXT_MARKER = "PLANNER CONTEXT (SECONDARY; CANNOT OVERRIDE REQUIREMENTS):"
ARTICULATED_SOURCES = frozenset({"artiverse", "artvip"})


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    raw = path.resolve(strict=True).read_bytes()
    data = json.loads(raw)
    if data.get("contract_id") != CONTRACT_ID or data.get("schema_version") != 1:
        raise RuntimeError("unsupported factory contract")
    data["_sha256"] = hashlib.sha256(raw).hexdigest()
    return data


def room_ids(contract: dict[str, Any]) -> tuple[str, ...]:
    ids = tuple(contract.get("room_order", ()))
    if len(ids) != 14 or len(set(ids)) != 14 or set(ids) != set(contract.get("rooms", {})):
        raise RuntimeError("factory contract must define exactly 14 ordered rooms")
    return ids


MINIMUM_COUNTS: dict[str, dict[str, int]] = {
    "ingredient_receiving": {"wooden_pallet":3,"cardboard_carton":4,"plastic_food_crate":3,"ingredient_sack":4,"safety_bollard":2},
    "dry_storage": {"pallet_rack":4,"metal_shelving":2,"cardboard_carton":4,"sealed_food_container":3,"ingredient_sack":3,"plastic_bin":3,"labelled_storage_crate":3,"loaded_pallet":3},
    "cold_storage": {"cold_room_door":2,"stainless_storage_rack":3,"chilled_container":4,"ceiling_cooling_unit":2},
    "washing_preparation": {"deep_stainless_sink":2,"wash_basin":2,"stainless_preparation_table":3,"cutting_table":2,"food_safe_tray":3,"floor_drain":2,"waste_sorting_bin":2},
    "qc_laboratory": {"laboratory_workbench":2,"laboratory_stool":2,"sample_container":3,"testing_machine":2},
    "office_administration": {"office_desk":2,"computer_monitor":2,"ergonomic_office_chair":2,"meeting_chair":4,"indoor_plant":2},
    "processing_hall": {"mixing_tank":2,"cylindrical_vat":2,"hopper":2,"pump":2,"valve":4,"processing_control_panel":2,"food_processing_machine":3,"stainless_conveyor":2,"electric_motor":2,"machine_safety_guard":2,"emergency_stop":2},
    "packaging_hall": {"stainless_conveyor":2,"packing_table":2,"packaging_material_bin":3,"empty_carton_stack":2,"machine_safety_guard":2,"emergency_stop":2},
    "finished_goods_storage": {"pallet_rack":3,"cardboard_carton":4,"wrapped_finished_pallet":3,"plastic_food_crate":3,"organized_pallet_stack":2},
    "maintenance": {"wall_mounted_tools":3,"spare_machine_part":3,"metal_shelving":2},
    "changing_room": {"locker":6,"changing_bench":2,"coat_hook":6},
    "break_room": {"dining_table":2,"dining_chair":6,"indoor_plant":2},
    "boys_toilet": {"toilet_cubicle":2,"urinal":2,"toilet_sink":2,"mirror":2,"privacy_partition":2},
    "girls_toilet": {"toilet_cubicle":3,"toilet_sink":2,"mirror":2,"privacy_partition":3},
}
MAXIMUM_COUNTS: dict[str, dict[str, int]] = {
    "office_administration": {"office_desk":3,"meeting_chair":6},
}
ALIASES: dict[str, tuple[str, ...]] = {
    "manual_pallet_truck": (r"\bpallet\s*(?:jack|truck)\b",),
    "wooden_pallet": (r"\bwood(?:en)?\s+pallets?\b", r"\bpallets?\b"),
    "plastic_food_crate": (r"\bplastic\s+(?:food\s+)?crates?\b",),
    "deep_stainless_sink": (r"\bdeep\s+(?:stainless(?:[- ]steel)?\s+)?sinks?\b",),
    "stainless_preparation_table": (r"\bstainless(?:[- ]steel)?\s+(?:prep(?:aration)?\s+)?tables?\b",),
    "cold_room_door": (r"\b(?:insulated\s+)?cold[- ]room\s+doors?\b",),
    "cylindrical_vat": (r"\b(?:cylindrical\s+)?vats?\b",),
    "stainless_conveyor": (r"\bstainless(?:[- ]steel)?\s+conveyors?\b", r"\bconveyor\s+belts?\b"),
    "machine_safety_guard": (r"\b(?:machine\s+)?safety\s+guards?\b",),
    "empty_carton_stack": (r"\b(?:stack(?:s|ed)?\s+of\s+)?empty\s+(?:boxes|cartons)\b",),
    "organized_pallet_stack": (r"\b(?:organized\s+)?pallet\s+stacks?\b",),
    "wall_mounted_tools": (r"\bwall[- ]mounted\s+tools?\b", r"\bhand\s+tools?\b"),
    "toilet_cubicle": (r"\btoilet\s+(?:cubicles?|stalls?)\b",),
    "toilet_sink": (r"\b(?:toilet|restroom|washroom)?\s*(?:sinks?|wash\s*basins?)\b",),
    "break_room_sink": (r"\b(?:break[- ]room|kitchen)\s+sinks?\b",),
    "laboratory_sink": (r"\b(?:laboratory|lab)\s+sinks?\b",),
}


def _patterns(key: str) -> tuple[str, ...]:
    words = [re.escape(piece) for piece in key.split("_")]
    canonical = r"\b" + r"[\s_-]+".join(words) + r"s?\b"
    return (canonical, *ALIASES.get(key, ()))


def inventory_requirements(contract: dict[str, Any], room_id: str) -> tuple[InventoryRequirement, ...]:
    room = contract["rooms"].get(room_id)
    if not isinstance(room, dict):
        raise ValueError(f"unsupported factory room: {room_id}")
    result = []
    for key in room["inventory"]:
        result.append(InventoryRequirement(
            key=key, label=key.replace("_", " "), patterns=_patterns(key),
            minimum=MINIMUM_COUNTS.get(room_id, {}).get(key, 1),
            maximum=MAXIMUM_COUNTS.get(room_id, {}).get(key), exclude_patterns=(),
        ))
    return tuple(result)


def canonical_room_prompt(contract: dict[str, Any], room_id: str, effective_hash: str, existing: str = "") -> str:
    requirements = inventory_requirements(contract, room_id)
    context = existing.split(CONTEXT_MARKER, 1)[-1].strip() if CONTEXT_MARKER in existing else existing.strip()
    inventory = "\n".join(
        f"- {item.label}: minimum {item.minimum}" + (f", maximum {item.maximum}" if item.maximum is not None else "")
        for item in requirements
    )
    room = contract["rooms"][room_id]
    marker = f"{CONTRACT_MARKER} room_id={room_id} profile={PROFILE}]"
    return (
        f"{marker}\nImmutable effective prompt SHA-256: {effective_hash}\n"
        f"Immutable factory contract SHA-256: {contract['_sha256']}\n\n"
        f"Generate the {room_id} zone inside bounds {room['bounds_xy']} under the complete factory prompt. "
        "Every inventory category below needs an independently represented physical object; one generic object cannot satisfy several categories. "
        "Preserve door/window clearance, support, collision accuracy, worker access, and all declared aisle widths.\n\n"
        f"REQUIRED DISTINCT INVENTORY:\n{inventory}\n\n{CONTEXT_MARKER}\n{context}"
    ).strip()


def _layout_collection(layout: dict[str, Any], key: str, expected: set[str]) -> dict[str, dict[str, Any]]:
    value = layout.get(key)
    entries = [v for v in (value.values() if isinstance(value, dict) else value if isinstance(value, list) else []) if isinstance(v, dict)]
    ids = [str(item.get("room_id") or item.get("id") or "") for item in entries]
    if len(entries) != 14 or set(ids) != expected or len(ids) != len(set(ids)):
        raise RuntimeError(f"{key} does not contain the exact 14 unique factory rooms")
    return {str(item.get("room_id") or item.get("id")): item for item in entries}


def bind_layout_prompts(layout_path: Path, manifest_path: Path, contract_path: Path, output: Path) -> dict[str, Any]:
    contract = load_contract(contract_path)
    expected = set(room_ids(contract))
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    effective_hash = str(manifest.get("effective_prompt_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", effective_hash) or manifest.get("factory_contract_sha256") != contract["_sha256"]:
        raise RuntimeError("input manifest prompt/factory-contract binding is invalid")
    specs = _layout_collection(layout, "rooms", expected)
    placed = _layout_collection(layout, "placed_rooms", expected)
    before = sha256_file(layout_path)
    prompt_hashes = {}
    for room_id in room_ids(contract):
        prompt = canonical_room_prompt(contract, room_id, effective_hash, str(specs[room_id].get("prompt") or ""))
        specs[room_id]["prompt"] = prompt
        placed[room_id]["prompt"] = prompt
        prompt_hashes[room_id] = hashlib.sha256(prompt.encode()).hexdigest()
    temp = layout_path.with_name(f".{layout_path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(layout, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    os.replace(temp, layout_path)
    result = {"schema_version":1,"status":"pass","profile":PROFILE,"factory_contract_sha256":contract["_sha256"],"effective_prompt_sha256":effective_hash,"layout":{"path":str(layout_path.resolve()),"sha256_before":before,"sha256":sha256_file(layout_path)},"input_manifest":{"path":str(manifest_path.resolve()),"sha256":sha256_file(manifest_path)},"room_prompt_sha256":prompt_hashes,"occurrence_counts":{room_id:2 for room_id in room_ids(contract)}}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    os.replace(temp, output)
    return result


def _prompt_evidence(contract: dict[str, Any], room_id: str, prompt: str, expected_hash: str | None) -> dict[str, Any]:
    marker = f"{CONTRACT_MARKER} room_id={room_id} profile={PROFILE}]"
    issues = []
    if marker not in prompt: issues.append("canonical marker missing")
    if f"Immutable factory contract SHA-256: {contract['_sha256']}" not in prompt: issues.append("factory contract hash missing")
    if expected_hash and hashlib.sha256(prompt.encode()).hexdigest() != expected_hash: issues.append("canonical prompt hash mismatch")
    return {"status":"pass" if not issues else "fail","issues":issues,"sha256":hashlib.sha256(prompt.encode()).hexdigest()}


def evaluate_room_inventory(contract_path: Path, room_id: str, state: dict[str, Any], prompt: str, *, expected_prompt_sha256: str | None = None, asset_root: Path | None = None) -> dict[str, Any]:
    contract = load_contract(contract_path)
    requirements = inventory_requirements(contract, room_id)
    prompt_evidence = _prompt_evidence(contract, room_id, prompt, expected_prompt_sha256)
    issues = [f"room prompt binding failed: {item}" for item in prompt_evidence["issues"]]
    repairs = []
    semantic = []
    physical = {}
    invalid = {}
    for object_id, obj in _iter_state_objects(state):
        if str(obj.get("object_type", "")).lower() in {"wall","floor","ceiling"}: continue
        text = object_semantic_text(object_id, obj)
        matching = [item.key for item in requirements if _matches(text, item)]
        if not matching: continue
        evidence = physical_object_evidence(obj, asset_root=asset_root)
        physical[object_id] = evidence
        if evidence["status"] == "pass": semantic.append((object_id, text, obj))
        else: invalid[object_id] = {"matching_requirements":matching, **evidence}
    if invalid:
        issues.append("semantic candidates without real mesh/pose/bounds evidence: " + ", ".join(sorted(invalid)))
    matches, raw = _minimum_unique_assignment(requirements, semantic)
    counts, raw_counts = {}, {}
    for item in requirements:
        counts[item.key], raw_counts[item.key] = len(matches[item.key]), len(raw[item.key])
        if counts[item.key] < item.minimum:
            issues.append(f"{item.key}: unique={counts[item.key]} minimum={item.minimum} raw={raw_counts[item.key]}")
            repairs.append(f"add independently represented {item.label}")
        if item.maximum is not None and raw_counts[item.key] > item.maximum:
            issues.append(f"{item.key}: raw={raw_counts[item.key]} maximum={item.maximum}")
            repairs.append(f"remove excess {item.label}")
    return {"status":"pass" if not issues else "fail","profile":PROFILE,"room_id":room_id,"factory_contract_sha256":contract["_sha256"],"counts":counts,"raw_semantic_counts":raw_counts,"matched_object_ids":matches,"raw_matched_object_ids":raw,"prompt_binding":prompt_evidence,"physical_object_evidence":physical,"invalid_physical_candidates":invalid,"critical_issues":issues,"repair_instructions":repairs,"canonical_requirements":canonical_room_prompt(contract, room_id, "<bound-at-runtime>")}


ROLE_RULES = {
    "cold_room_insulated_door": {"rooms":{"cold_storage"},"patterns":(r"\bcold[- ]room\b",r"\bdoor\b")},
    "maintenance_tool_cabinet_or_changing_locker": {"rooms":{"maintenance","changing_room"},"patterns":(r"\b(?:tool\s+cabinet|locker)\b",)},
    "office_filing_cabinet_or_break_refrigerator": {"rooms":{"office_administration","break_room"},"patterns":(r"\b(?:filing\s+cabinet|file\s+cabinet|refrigerator)\b",)},
}


def classify_articulated_role(room_id: str, object_id: str, obj: dict[str, Any]) -> str | None:
    metadata = obj.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("is_articulated") is not True: return None
    text = object_semantic_text(object_id, obj)
    for role, rule in ROLE_RULES.items():
        if room_id in rule["rooms"] and all(re.search(pattern, text, re.I) for pattern in rule["patterns"]): return role
    return None


def collect_required_articulated_roles(room_states: dict[str, dict[str, Any]], *, require_runtime_provenance: bool = False) -> dict[str, Any]:
    roles: dict[str,list[dict[str,Any]]] = {role:[] for role in ROLE_RULES}
    invalid = []
    for room_id, state in room_states.items():
        for object_id, obj in _iter_state_objects(state):
            role = classify_articulated_role(room_id, object_id, obj)
            if role is None: continue
            metadata = obj.get("metadata", {})
            record = {"room_id":room_id,"object_id":str(obj.get("object_id") or object_id),"articulated_id":str(metadata.get("articulated_id") or "").strip(),"articulated_source":str(metadata.get("articulated_source") or "").lower(),"asset_source":str(metadata.get("asset_source") or "").lower(),"sdf_path":str(obj.get("sdf_path") or "").strip()}
            problems = []
            if require_runtime_provenance:
                if record["asset_source"] != "articulated": problems.append("asset_source")
                if record["articulated_source"] not in ARTICULATED_SOURCES: problems.append("articulated_source")
                if not record["articulated_id"]: problems.append("articulated_id")
                if not record["sdf_path"]: problems.append("sdf_path")
            if problems: invalid.append({"role":role,**record,"issues":problems})
            else: roles[role].append(record)
    missing = sorted(role for role, records in roles.items() if not records)
    artiverse_roles = sorted(role for role, records in roles.items() if any(record["articulated_source"] == "artiverse" for record in records))
    issues = []
    if missing: issues.append(f"missing factory articulated roles: {missing}")
    if not artiverse_roles: issues.append("no required articulated role is sourced from Artiverse")
    if invalid: issues.append("invalid runtime provenance for articulated-role candidates")
    return {"status":"pass" if not issues else "fail","profile":PROFILE,"roles":roles,"missing_roles":missing,"artiverse_roles":artiverse_roles,"invalid_role_candidates":invalid,"critical_issues":issues}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    bind = sub.add_parser("bind-layout")
    bind.add_argument("--layout", type=Path, required=True)
    bind.add_argument("--input-manifest", type=Path, required=True)
    bind.add_argument("--factory-contract", type=Path, required=True)
    bind.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = bind_layout_prompts(args.layout, args.input_manifest, args.factory_contract, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
