#!/usr/bin/env python3
"""Hash-bound prompt-authority visual gate for the assembled food factory."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .factory_room_contract import PROFILE, load_contract, room_ids
    from .visual_gate_utils import build_multimodal_content, call_vlm_service
except ImportError:
    from factory_room_contract import PROFILE, load_contract, room_ids
    from visual_gate_utils import build_multimodal_content, call_vlm_service


SCHEMA_ID = "scenesmith_whole_factory_prompt_gate_v1"
VERIFY_SCHEMA_ID = "scenesmith_whole_factory_prompt_gate_verify_v1"
SCORE_KEYS = (
    "room_count_and_identity",
    "room_arrangement",
    "visual_finish",
    "circulation_and_access",
    "furnishing_completeness",
    "simulation_readiness",
    "reference_similarity",
)
OVERVIEW_NAMES = ("overview_top", "overview_isometric", "overview_front")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _entry(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise ValueError(f"whole-factory evidence is not a nonempty regular file: {path}")
    return {"path": str(path), "sha256": _sha(path), "size_bytes": path.stat().st_size}


def _rooms(value: Mapping[str, Any], key: str) -> dict[str, dict[str, Any]]:
    raw = value.get(key)
    entries = raw.values() if isinstance(raw, dict) else raw if isinstance(raw, list) else []
    result = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        room_id = str(item.get("room_id") or item.get("id") or "")
        if not room_id or room_id in result:
            raise ValueError(f"{key} contains missing/duplicate room IDs")
        result[room_id] = item
    return result


def _parse(raw: Any) -> dict[str, Any]:
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict) or set(value.get("scores", {})) != set(SCORE_KEYS):
        raise ValueError("whole-factory judge response has incorrect score keys")
    scores = {}
    for key in SCORE_KEYS:
        number = value["scores"][key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)) or not 0 <= float(number) <= 10:
            raise ValueError(f"invalid whole-factory score: {key}")
        scores[key] = float(number)
    identities = value.get("room_evidence")
    if not isinstance(identities, dict):
        raise ValueError("whole-factory response lacks room_evidence")
    issues = value.get("critical_issues")
    repairs = value.get("repair_instructions")
    if not isinstance(issues, list) or any(not isinstance(item, str) for item in issues):
        raise ValueError("critical_issues must be a string array")
    if not isinstance(repairs, list) or any(not isinstance(item, str) for item in repairs):
        raise ValueError("repair_instructions must be a string array")
    return {"scores": scores, "room_evidence": identities, "critical_issues": issues, "repair_instructions": repairs}


def _changed(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    failures = []
    for entry in entries:
        path = Path(str(entry.get("path", "")))
        if not path.is_file() or path.stat().st_size != entry.get("size_bytes") or _sha(path) != entry.get("sha256"):
            failures.append(f"whole-factory evidence changed: {path}")
    return failures


def evaluate(args: argparse.Namespace, *, vlm_service: Any | None = None) -> dict[str, Any]:
    contract_path = args.factory_contract.resolve()
    prompt_path = args.effective_prompt.resolve()
    manifest_path = args.input_manifest.resolve()
    scene_dir = args.scene_dir.resolve()
    layout_path = scene_dir / "house_layout.json"
    combined_path = scene_dir / "combined_house" / "house_state.json"
    contract = load_contract(contract_path)
    expected = room_ids(contract)
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    if set(_rooms(layout, "placed_rooms")) != set(expected):
        raise ValueError("house layout does not contain exactly the 14 factory zones")
    combined_rooms = combined.get("rooms")
    if not isinstance(combined_rooms, dict) or set(combined_rooms) != set(expected):
        raise ValueError("combined state does not contain exactly the 14 factory zones")
    binding = layout.get("factory_contract_binding")
    if not isinstance(binding, dict) or binding.get("sha256") != contract["_sha256"]:
        raise ValueError("assembled layout factory contract binding is missing or stale")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompt_text = prompt_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    prompt_sha = hashlib.sha256(prompt_text.encode()).hexdigest()
    if manifest.get("effective_prompt_sha256") != prompt_sha or manifest.get("factory_contract_sha256") != contract["_sha256"]:
        raise ValueError("whole-factory prompt/contract binding is stale")
    if manifest.get("reference_image_sha256") is not None:
        raise ValueError("factory prompt gate refuses a fabricated reference-image claim")
    overview_paths = [args.review_dir.resolve() / f"{name}.png" for name in OVERVIEW_NAMES]
    evidence = [_entry(path) for path in (contract_path, prompt_path, manifest_path, layout_path, combined_path, *overview_paths)]
    for room_id in expected:
        evidence.append(_entry(scene_dir / f"room_{room_id}" / "scene_states" / "final_scene" / "scene_state.json"))
    room_bounds = {room_id: contract["rooms"][room_id]["bounds_xy"] for room_id in expected}
    instruction = f"""Strictly judge the assembled clean modern food factory shown in exactly three generated overview cutaways. No user reference image exists: the complete immutable text prompt below is the sole reference authority.

Immutable prompt:
{prompt_text}

Required exact ordered room IDs: {json.dumps(expected)}
Required room bounds/arrangement: {json.dumps(room_bounds, sort_keys=True)}

Return strict JSON with scores containing exactly {json.dumps(SCORE_KEYS)}, each 0-10; room_evidence containing exactly all 14 room IDs, each with status pass/fail, view_indices from 1-3, and a specific observation; critical_issues and repair_instructions string arrays. Verify visible room count/identity, arrangement, bright clean finish, loading dock and truck outside, pedestrian/forklift circulation, full operational furnishings, collision/support realism, and similarity to the entire text prompt. Never award evidence based only on the quoted text."""
    messages = [{"role": "system", "content": "You are a strict simulation-ready whole-food-factory visual gate. Visible pixels only."}, {"role": "user", "content": build_multimodal_content(instruction, [(f"generated whole factory {name}", path) for name, path in zip(OVERVIEW_NAMES, overview_paths)])}]
    request = {
        "schema_id": "scenesmith_whole_factory_vlm_request_v1",
        "model": args.model,
        "backend": args.vlm_backend,
        "threshold": args.threshold,
        "room_ids": list(expected),
        "prompt_sha256": prompt_sha,
        "review_sha256": [_sha(path) for path in overview_paths],
        "messages_sha256": _canonical_sha(messages),
    }
    request["request_sha256"] = _canonical_sha(request)
    assessment = _parse(call_vlm_service(messages=messages, model=args.model, backend=args.vlm_backend, vlm_service=vlm_service))
    if set(assessment["room_evidence"]) != set(expected):
        raise ValueError("whole-factory judge room_evidence differs from exact 14-room set")
    issues = list(assessment["critical_issues"])
    for room_id in expected:
        item = assessment["room_evidence"][room_id]
        if not isinstance(item, dict) or item.get("status") not in {"pass", "fail"}:
            raise ValueError(f"malformed whole-factory evidence for {room_id}")
        indices = item.get("view_indices")
        if not isinstance(indices, list) or not indices or indices != sorted(set(indices)) or any(not isinstance(index, int) or not 1 <= index <= 3 for index in indices):
            raise ValueError(f"invalid overview citations for {room_id}")
        observation = item.get("observation")
        if not isinstance(observation, str) or len(observation.strip()) < 8:
            raise ValueError(f"non-specific overview observation for {room_id}")
        if item["status"] != "pass":
            issues.append(f"room evidence failed: {room_id}: {observation}")
    below = [key for key, score in assessment["scores"].items() if score < args.threshold]
    if below:
        issues.append(f"whole-factory scores below {args.threshold:g}: {below}")
    issues.extend(_changed(evidence))
    result = {"schema_id": SCHEMA_ID, "schema_version": 1, "status": "pass" if not issues else "fail", "contract_profile": PROFILE, "factory_contract_sha256": contract["_sha256"], "reference_authority": "prompt_only_no_user_image_supplied", "threshold": args.threshold, "scores": assessment["scores"], "room_evidence": assessment["room_evidence"], "canonical_vlm_request": request, "evidence": evidence, "critical_issues": issues, "repair_instructions": assessment["repair_instructions"]}
    result["attestation"] = {"algorithm": "sha256", "sha256": _canonical_sha(result)}
    return result


def verify(path: Path) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("schema_id") != SCHEMA_ID or source.get("status") != "pass":
        raise ValueError("saved whole-factory gate is not a pass")
    unsigned = dict(source); unsigned.pop("attestation", None)
    if source.get("attestation") != {"algorithm": "sha256", "sha256": _canonical_sha(unsigned)}:
        raise ValueError("saved whole-factory attestation is invalid")
    failures = _changed(source.get("evidence", []))
    if failures:
        raise ValueError("; ".join(failures))
    return {"schema_id": VERIFY_SCHEMA_ID, "schema_version": 1, "status": "pass", "source": _entry(path), "source_attestation_sha256": source["attestation"]["sha256"], "all_evidence_rehashed": True}


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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
    parser.add_argument("--verification-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = verify(args.output) if args.verify_only else evaluate(args)
        destination = args.verification_output if args.verify_only and args.verification_output else args.output
        _write(destination, result)
    except Exception as exc:
        print(f"whole factory gate failed: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
