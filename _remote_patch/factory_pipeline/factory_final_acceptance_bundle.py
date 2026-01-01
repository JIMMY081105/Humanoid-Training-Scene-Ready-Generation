#!/usr/bin/env python3
"""Create or revalidate the factory-specific final acceptance record.

The school acceptance bundler is intentionally school-profile specific.  This
sibling binds the exact 14-room factory package and every already-passed,
repeat-verified factory gate without weakening their individual validators.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


PROFILE = "factory_reference_20260713"
SCHEMA_ID = "scenesmith_factory_final_acceptance_v1"
ROOM_IDS = (
    "ingredient_receiving", "dry_storage", "cold_storage",
    "washing_preparation", "qc_laboratory", "office_administration",
    "processing_hall", "packaging_hall", "finished_goods_storage",
    "maintenance", "changing_room", "break_room", "boys_toilet",
    "girls_toilet",
)
EXPECTED_BINDING = {
    "original_prompt_sha256": "3bb58cf8e07012cc0c5381eb51730da03b25fb6659a08338238d9fd4baa5093b",
    "appendix_sha256": "ffc327f082431d407c15b25b039d7d2676786a03663f186777558e9ba8e1db19",
    "effective_prompt_sha256": "35387fa3fffb0af92b6c9e07e39f2b59a5443d497190f721090f82bd81734322",
    "factory_contract_sha256": "f33e56eef4375d1f68684bd10c02d7d2f82ee72420b8a12570e8fcbe135a94cd",
    "reference_image_sha256": None,
}


class AcceptanceError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _regular(path: Path, *, root: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    if lexical.is_symlink():
        raise AcceptanceError(f"{label} must not be a symlink: {lexical}")
    try:
        info = lexical.stat(follow_symlinks=False)
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise AcceptanceError(f"missing {label}: {lexical}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise AcceptanceError(f"{label} must be a non-empty regular file: {lexical}")
    if not _inside(root, resolved):
        raise AcceptanceError(f"{label} escapes its authority root: {resolved}")
    return resolved


def _read_json(path: Path, *, root: Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = _regular(path, root=root, label=label)
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AcceptanceError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AcceptanceError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise AcceptanceError(f"{label} must be a JSON object")
    return resolved, value


def _entry(path: Path, *, root: Path, label: str) -> dict[str, Any]:
    resolved = _regular(path, root=root, label=label)
    return {
        "path": resolved.relative_to(root).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _require_pass(path: Path, *, root: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved, document = _read_json(path, root=root, label=label)
    if document.get("status") != "pass" and document.get("pass") is not True:
        raise AcceptanceError(f"{label} status is not pass")
    critical = document.get("critical_issues", [])
    if critical not in (None, []) and critical:
        raise AcceptanceError(f"{label} retains critical issues")
    return _entry(resolved, root=root, label=label), document


def _attest(document: dict[str, Any]) -> dict[str, Any]:
    payload = dict(document)
    payload.pop("attestation", None)
    document["attestation"] = {
        "algorithm": "sha256",
        "schema_version": 1,
        "sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
    }
    return document


def _validate_attestation(document: dict[str, Any]) -> None:
    saved = document.get("attestation")
    if not isinstance(saved, dict):
        raise AcceptanceError("acceptance record has no attestation")
    payload = dict(document)
    payload.pop("attestation", None)
    expected = hashlib.sha256(_canonical(payload)).hexdigest()
    if saved != {"algorithm": "sha256", "schema_version": 1, "sha256": expected}:
        raise AcceptanceError("acceptance record attestation is stale")


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _collect(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_dir.resolve(strict=True)
    run_dir = args.run_dir.resolve(strict=True)
    scene_dir = args.scene_dir.resolve(strict=True)
    input_dir = args.input_dir.resolve(strict=True)
    if not _inside(run_dir, scene_dir):
        raise AcceptanceError("scene directory escapes run directory")
    if not _inside(repo, input_dir):
        raise AcceptanceError("input directory escapes canonical checkout")

    contract = repo / "scripts" / "factory_contract.json"
    if _sha256(_regular(contract, root=repo, label="factory contract")) != EXPECTED_BINDING["factory_contract_sha256"]:
        raise AcceptanceError("factory contract digest differs from the immutable binding")
    _, input_manifest = _read_json(input_dir / "input_manifest.json", root=input_dir, label="input manifest")
    for key, expected in EXPECTED_BINDING.items():
        if input_manifest.get(key) != expected:
            raise AcceptanceError(f"input manifest differs from immutable {key}")

    evidence: dict[str, dict[str, Any]] = {}
    evidence["pipeline_code_contract"], code_contract = _require_pass(
        run_dir / "pipeline_code_contract.json", root=run_dir, label="pipeline code contract"
    )
    if int(code_contract.get("artifact_count", 0)) < 200:
        raise AcceptanceError("pipeline code contract artifact coverage is unexpectedly small")
    evidence["input_manifest_validation"], _ = _require_pass(
        run_dir / "factory_input_manifest_validation.json", root=run_dir, label="input manifest validation"
    )
    evidence["layout_seed"], _ = _require_pass(
        scene_dir / "quality_gates" / "reference_factory_layout_seed.json", root=scene_dir, label="native layout seed"
    )
    evidence["layout_gate"], _ = _require_pass(
        scene_dir / "quality_gates" / "floor_plan_layout.json", root=scene_dir, label="factory layout gate"
    )
    _, layout = _read_json(scene_dir / "house_layout.json", root=scene_dir, label="house layout")
    placed = layout.get("placed_rooms")
    layout_ids = [str(item.get("room_id")) for item in placed or [] if isinstance(item, dict)]
    if tuple(layout_ids) != ROOM_IDS:
        raise AcceptanceError(f"house layout is not the exact ordered 14-room factory: {layout_ids}")
    evidence["house_layout"] = _entry(scene_dir / "house_layout.json", root=scene_dir, label="house layout")
    evidence["prompt_binding"], binding = _require_pass(
        scene_dir / "quality_gates" / "room_prompt_binding.json", root=scene_dir, label="factory prompt binding"
    )
    if tuple((binding.get("room_prompt_sha256") or {}).keys()) != ROOM_IDS:
        raise AcceptanceError("prompt binding does not cover the exact 14 factory rooms")

    room_records: dict[str, Any] = {}
    review_root = scene_dir / "review" / "room_review_renders"
    deterministic_root = scene_dir / "quality_gates" / "room_self_exam_deterministic"
    visual_root = scene_dir / "quality_gates" / "room_self_exam"
    for room_id in ROOM_IDS:
        final_state = scene_dir / f"room_{room_id}" / "scene_states" / "final_scene" / "scene_state.json"
        final_blend = final_state.with_name("scene.blend")
        deterministic_entry, deterministic = _require_pass(
            deterministic_root / f"{room_id}.json", root=scene_dir, label=f"{room_id} deterministic gate"
        )
        visual_entry, visual = _require_pass(
            visual_root / f"{room_id}.json", root=scene_dir, label=f"{room_id} visual gate"
        )
        if deterministic.get("room_id") != room_id or visual.get("room_id") != room_id:
            raise AcceptanceError(f"room gate identity mismatch for {room_id}")
        cutaway_entry, cutaway = _require_pass(
            review_root / f"{room_id}_cutaway_evidence.json", root=scene_dir, label=f"{room_id} cutaway evidence"
        )
        images = sorted(review_root.glob(f"{room_id}_*.png"))
        if len(images) != 3 or len({_sha256(path) for path in images}) != 3:
            raise AcceptanceError(f"{room_id} does not have exactly three distinct review images")
        room_records[room_id] = {
            "final_state": _entry(final_state, root=scene_dir, label=f"{room_id} final state"),
            "final_blend": _entry(final_blend, root=scene_dir, label=f"{room_id} final blend"),
            "deterministic_gate": deterministic_entry,
            "visual_gate": visual_entry,
            "cutaway_evidence": cutaway_entry,
            "review_images": [_entry(path, root=scene_dir, label=f"{room_id} review image") for path in images],
            "cutaway_schema_id": cutaway.get("schema_id"),
        }

    gates = {
        "variation": "factory_variation.json",
        "variation_verification": "factory_variation_verification.json",
        "articulated_motion": "factory_articulated_motion.json",
        "articulated_motion_verification": "factory_articulated_motion_verification.json",
        "navigation": "factory_navigation.json",
        "navigation_verification": "factory_navigation_verification.json",
        "whole_factory": "whole_factory_reference.json",
        "whole_factory_verification": "whole_factory_reference_verification.json",
        "drake_load": "drake_load_paracloud.json",
        "simulator_exports": "simulator_exports.json",
        "sage_scene_checker": "sage_scene_checker.json",
    }
    for label, filename in gates.items():
        evidence[label], _ = _require_pass(
            scene_dir / "quality_gates" / filename, root=scene_dir, label=label.replace("_", " ")
        )
    evidence["artiverse_preparation"], _ = _require_pass(
        run_dir / "artiverse_preparation_validation.json", root=run_dir, label="Artiverse preparation"
    )
    evidence["artiverse_visual_resources"], _ = _require_pass(
        run_dir / "artiverse_visual_resources.json", root=run_dir, label="Artiverse visual resources"
    )

    combined = scene_dir / "combined_house"
    for key, relative in {
        "combined_state": "house_state.json",
        "combined_dmd": "house.dmd.yaml",
        "artiverse_usage": "artiverse_usage.json",
        "combined_blend": "house.blend",
    }.items():
        evidence[key] = _entry(combined / relative, root=scene_dir, label=key.replace("_", " "))
    overview_paths = sorted((combined / "outlook_renders").glob("overview_*.png"))
    if len(overview_paths) != 3 or len({_sha256(path) for path in overview_paths}) != 3:
        raise AcceptanceError("combined factory does not have exactly three distinct overview renders")
    evidence["overview_renders"] = [
        _entry(path, root=scene_dir, label="factory overview render") for path in overview_paths
    ]

    return _attest({
        "schema_id": SCHEMA_ID,
        "schema_version": 1,
        "status": "awaiting_2gpu_acceptance",
        "all_preterminal_gates_passed": True,
        "contract_profile": PROFILE,
        "run_attempt_id": args.run_attempt_id,
        "required_room_ids": list(ROOM_IDS),
        "external_input_binding": EXPECTED_BINDING,
        "room_records": room_records,
        "evidence": evidence,
        "critical_issues": [],
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--run-attempt-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        expected = _collect(args)
        output = args.output.resolve(strict=args.verify_only)
        scene = args.scene_dir.resolve(strict=True)
        if not _inside(scene, output):
            raise AcceptanceError("acceptance output must stay inside the scene package")
        if args.verify_only:
            _, saved = _read_json(output, root=scene, label="saved factory acceptance")
            _validate_attestation(saved)
            if saved != expected:
                raise AcceptanceError("saved factory acceptance differs from a full recomputation")
        else:
            if output.exists():
                raise AcceptanceError("refusing to overwrite an existing factory acceptance record")
            _atomic_json(output, expected)
        print(json.dumps({"status": "pass", "attestation": expected["attestation"]}, indent=2, sort_keys=True))
        return 0
    except (AcceptanceError, OSError, ValueError) as exc:
        print(f"factory final acceptance failed: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
