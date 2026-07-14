#!/usr/bin/env python3
"""Hash-bound deterministic and visual identity gate for food-factory zones."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .factory_room_contract import PROFILE, load_contract, object_semantic_text, room_ids
    from .visual_gate_utils import build_multimodal_content, call_vlm_service, write_json
except ImportError:
    from factory_room_contract import PROFILE, load_contract, object_semantic_text, room_ids
    from visual_gate_utils import build_multimodal_content, call_vlm_service, write_json


SCHEMA_ID = "scenesmith_factory_variation_gate_v1"
COMPARISON_GROUPS = {
    "storage_identity": ("dry_storage", "finished_goods_storage"),
    "toilet_identity": ("boys_toilet", "girls_toilet"),
    "staff_support_identity": ("maintenance", "changing_room", "break_room"),
}
STRUCTURAL = {"wall", "floor", "ceiling"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _objects(state: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    value = state.get("objects")
    entries = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else []
    return [
        (str(key), obj)
        for key, obj in entries
        if isinstance(obj, dict) and str(obj.get("object_type", "")).lower() not in STRUCTURAL
    ]


def _translation(obj: Mapping[str, Any]) -> tuple[float, float, float] | None:
    transform = obj.get("transform")
    raw = transform.get("translation") if isinstance(transform, dict) else None
    values = [raw.get(axis) for axis in ("x", "y", "z")] if isinstance(raw, dict) else list(raw[:3]) if isinstance(raw, (list, tuple)) and len(raw) >= 3 else None
    try:
        point = tuple(float(value) for value in values) if values is not None else None
    except (TypeError, ValueError):
        return None
    return point if point is not None and all(math.isfinite(value) for value in point) else None


def fingerprint(state: Mapping[str, Any]) -> dict[str, Any]:
    records = _objects(state)
    if not records:
        raise ValueError("factory state contains no non-structural physical objects")
    semantic = Counter(
        re.sub(r"\b\d+\b", "<n>", " ".join(object_semantic_text(key, obj).lower().split()))
        for key, obj in records
    )
    points = [point for _, obj in records if (point := _translation(obj)) is not None]
    if not points:
        raise ValueError("factory state has no finite object translations")
    center = tuple(sum(point[axis] for point in points) / len(points) for axis in range(3))
    layout = sorted(
        [round(point[axis] - center[axis], 2) for axis in range(3)]
        for point in points
    )
    payload = {
        "object_count": len(records),
        "semantic_multiset": dict(sorted(semantic.items())),
        "relative_layout": layout,
    }
    return {
        **payload,
        "semantic_sha256": _canonical_sha(payload["semantic_multiset"]),
        "layout_sha256": _canonical_sha(layout),
        "combined_sha256": _canonical_sha(payload),
    }


def _parse(raw: Any, expected: tuple[str, ...]) -> dict[str, Any]:
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict):
        raise ValueError("variation judge response is not an object")
    score = value.get("variation_quality_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)) or not 0 <= float(score) <= 10:
        raise ValueError("variation_quality_score is invalid")
    identities = value.get("zone_identities")
    if not isinstance(identities, dict) or set(identities) != set(expected):
        raise ValueError("judge must report the exact 14 factory zone identities")
    normalized = {}
    for room_id in expected:
        item = identities[room_id]
        if not isinstance(item, dict) or item.get("status") not in {"pass", "fail"}:
            raise ValueError(f"malformed zone identity for {room_id}")
        features = item.get("distinctive_features")
        if not isinstance(features, list) or len(features) < 2 or any(not isinstance(feature, str) or len(feature.strip()) < 4 for feature in features):
            raise ValueError(f"{room_id} needs at least two specific distinctive features")
        normalized[room_id] = {"status": item["status"], "distinctive_features": features}
    comparisons = value.get("comparison_groups")
    if not isinstance(comparisons, dict) or set(comparisons) != set(COMPARISON_GROUPS):
        raise ValueError("judge comparison groups differ from the factory contract")
    normalized_groups = {}
    for key, members in COMPARISON_GROUPS.items():
        item = comparisons[key]
        if not isinstance(item, dict) or item.get("members") != list(members) or item.get("status") not in {"pass", "fail"}:
            raise ValueError(f"malformed comparison group {key}")
        distinctions = item.get("specific_distinctions")
        if not isinstance(distinctions, list) or len(distinctions) < len(members) or any(not isinstance(text, str) or len(text.strip()) < 8 for text in distinctions):
            raise ValueError(f"comparison group {key} lacks specific visible distinctions")
        normalized_groups[key] = {"members": list(members), "status": item["status"], "specific_distinctions": distinctions}
    issues = value.get("critical_issues")
    repairs = value.get("repair_instructions")
    if not isinstance(issues, list) or any(not isinstance(item, str) for item in issues):
        raise ValueError("critical_issues must be a string array")
    if not isinstance(repairs, list) or any(not isinstance(item, str) for item in repairs):
        raise ValueError("repair_instructions must be a string array")
    return {"variation_quality_score": float(score), "zone_identities": normalized, "comparison_groups": normalized_groups, "critical_issues": issues, "repair_instructions": repairs}


def _verify(result_path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema_id") != SCHEMA_ID or result.get("status") != "pass" or result.get("factory_contract_sha256") != contract["_sha256"]:
        raise ValueError("factory variation result is missing, failing, or stale")
    payload = {key: value for key, value in result.items() if key != "attestation"}
    if result.get("attestation") != {"algorithm": "sha256", "sha256": _canonical_sha(payload)}:
        raise ValueError("factory variation attestation is invalid")
    for item in result.get("evidence", []):
        path = Path(str(item.get("path", "")))
        if not path.is_file() or item.get("sha256") != _sha(path):
            raise ValueError(f"factory variation evidence changed: {path}")
    return result


def run(args: argparse.Namespace, *, vlm_service: Any | None = None) -> dict[str, Any]:
    contract = load_contract(args.factory_contract)
    expected = room_ids(contract)
    if args.verify_only:
        return _verify(args.output, contract)
    scene_dir = args.scene_dir.resolve()
    review_dir = args.review_dir.resolve()
    prompt = args.effective_prompt.resolve()
    manifest = args.input_manifest.resolve()
    if not prompt.is_file() or not manifest.is_file():
        raise FileNotFoundError("immutable prompt/manifest evidence is missing")
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    if manifest_value.get("factory_contract_sha256") != contract["_sha256"] or manifest_value.get("effective_prompt_sha256") != hashlib.sha256(prompt.read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest():
        raise ValueError("factory variation prompt/contract binding is stale")
    fingerprints: dict[str, Any] = {}
    evidence: list[dict[str, Any]] = []
    images: list[tuple[str, Path]] = []
    for room_id in expected:
        state_path = scene_dir / f"room_{room_id}" / "scene_states" / "final_scene" / "scene_state.json"
        image_path = review_dir / f"{room_id}_top.png"
        if not state_path.is_file() or not image_path.is_file():
            raise FileNotFoundError(f"missing factory variation input for {room_id}")
        fingerprints[room_id] = fingerprint(json.loads(state_path.read_text(encoding="utf-8")))
        images.append((f"{room_id} generated top cutaway", image_path))
        evidence.extend({"room_id": room_id, "role": role, "path": str(path.resolve()), "sha256": _sha(path)} for role, path in (("state", state_path), ("top_view", image_path)))
    issues = []
    for group_name, members in COMPARISON_GROUPS.items():
        for index, first in enumerate(members):
            for second in members[index + 1:]:
                if fingerprints[first]["combined_sha256"] == fingerprints[second]["combined_sha256"]:
                    issues.append(f"deterministic duplicate in {group_name}: {first} and {second}")
    instruction = f"""Compare all 14 generated food-factory top views. Every zone must have its specific operational identity. Closely scrutinize these repetition-risk groups: {json.dumps(COMPARISON_GROUPS)}. Processing and packaging may share industrial language but must retain their own identity; do not demand arbitrary cosmetic differences from unique zones.
Immutable prompt SHA-256: {manifest_value['effective_prompt_sha256']}
Deterministic fingerprints (quoted evidence): {json.dumps(fingerprints, sort_keys=True)}
Return strict JSON: variation_quality_score 0-10; zone_identities with exactly all 14 IDs, each status pass/fail and at least two specific distinctive_features; comparison_groups with exactly the declared keys, exact ordered members, status pass/fail, and at least one specific_distinctions string per member; critical_issues and repair_instructions string arrays. Visible pixels only."""
    messages = [{"role": "system", "content": "You are a strict food-factory cross-zone identity gate. Never invent visual evidence."}, {"role": "user", "content": build_multimodal_content(instruction, images)}]
    raw = call_vlm_service(messages=messages, model=args.model, backend=args.vlm_backend, vlm_service=vlm_service)
    assessment = _parse(raw, expected)
    if assessment["variation_quality_score"] < args.threshold:
        issues.append(f"variation score {assessment['variation_quality_score']} is below {args.threshold}")
    issues.extend(assessment["critical_issues"])
    issues.extend(f"zone identity failed: {room_id}" for room_id, item in assessment["zone_identities"].items() if item["status"] != "pass")
    issues.extend(f"comparison group failed: {key}" for key, item in assessment["comparison_groups"].items() if item["status"] != "pass")
    for item in evidence:
        if _sha(Path(item["path"])) != item["sha256"]:
            issues.append(f"evidence changed during VLM comparison: {item['path']}")
    evidence.extend({"room_id": None, "role": role, "path": str(path), "sha256": _sha(path)} for role, path in (("effective_prompt", prompt), ("input_manifest", manifest), ("factory_contract", args.factory_contract.resolve())))
    result = {"schema_id": SCHEMA_ID, "schema_version": 1, "status": "pass" if not issues else "fail", "contract_profile": PROFILE, "factory_contract_sha256": contract["_sha256"], "threshold": args.threshold, "fingerprints": fingerprints, "assessment": assessment, "evidence": sorted(evidence, key=lambda item: (str(item["room_id"]), item["role"])), "critical_issues": issues, "repair_instructions": assessment["repair_instructions"]}
    result["attestation"] = {"algorithm": "sha256", "sha256": _canonical_sha(result)}
    write_json(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--factory-contract", type=Path, required=True)
    parser.add_argument("--effective-prompt", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=7.0)
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--vlm-backend", choices=("openai", "codex"), default="openai")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        print(f"factory variation gate failed: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
