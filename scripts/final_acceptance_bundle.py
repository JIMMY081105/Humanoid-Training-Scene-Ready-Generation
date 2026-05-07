#!/usr/bin/env python3
"""Create and verify the immutable SQZ-side final-acceptance package record.

This is the last single-GPU gate.  It does not claim the required two-GPU
acceptance; a successful record deliberately remains in
``awaiting_2gpu_acceptance`` state.  Creation validates every upstream verdict,
copies evidence which lives outside the scene into the scene atomically, hashes
the complete scene package, and self-attests the resulting receipt.

Verification is intentionally cache-only and service-free.  It rejects a
changed, added, removed, linked, escaped, or non-regular package file and reruns
the label-specific semantic checks on the packaged evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from scripts import artiverse_contract as _artiverse_contract

sys.modules.setdefault("artiverse_contract", _artiverse_contract)
from scripts import preflight_artiverse_visual_resources as artiverse_visual
from scripts import preflight_sam3d_generation as sam3d_generation
from scripts import preflight_sam3d_offline as sam3d_preflight


SCHEMA_ID = "scenesmith_sqz_final_acceptance_bundle_v1"
SCHEMA_VERSION = 1
STATUS = "awaiting_2gpu_acceptance"
HASH_ALGORITHM = "sha256"
PIPELINE_CODE_CONTRACT_SCHEMA_VERSION = 2
EVIDENCE_DIRECTORY = Path("quality_gates") / "final_acceptance_evidence"
ROOM_IDS = (
    "classroom_01",
    "classroom_02",
    "classroom_03",
    "classroom_04",
    "classroom_05",
    "classroom_06",
    "library",
    "boys_toilet",
    "girls_toilet",
    "storage_room",
    "main_corridor",
)
ROOM_SET = frozenset(ROOM_IDS)
INPUT_FILES = (
    "input_manifest.json",
    "prompt_original.txt",
    "scene_contract_appendix.txt",
    "prompt.txt",
    "prompt.csv",
    "reference.png",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ATTEMPT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
SCHOOL_RUN_NAME = "full_quality_school_reference_sam3d_artvip_artiverse_20260710"
SCHOOL_EXTERNAL_INPUT_BINDING = {
    "original_prompt_sha256": "5ddc06e1a9afa60b0417da882c1ec53265eae23f3f9bdd2343360d123923f34c",
    "effective_prompt_sha256": "ac8d297cc9a2d605f41b4bcd7abd52aac29bfd0f195875840342ee1e6a7da86f",
    "reference_image_sha256": "7ba62c39ac98cd2b21d9a5a97fd6f9b90d7efaf447d5ac9a6148524b5a8dbe48",
}
ROOM_SCORE_KEYS = (
    "object_relevance",
    "placement_realism",
    "clearance_and_access",
    "collision_risk",
    "prompt_alignment",
)
FLOOR_SCORE_KEYS = (
    "room_count_and_identity",
    "room_arrangement",
    "warm_visual_style",
    "circulation_and_access",
    "furnishing_completeness",
    "simulation_readiness",
    "reference_similarity",
)
DERIVATION_SCHEMA_ID = "scenesmith_state_blend_render_derivation_v1"
VLM_REQUEST_SCHEMA_ID = "scenesmith_canonical_vlm_request_v1"


class AcceptanceBundleError(RuntimeError):
    """The scene cannot be represented by a passing SQZ acceptance bundle."""


@dataclass(frozen=True)
class EvidenceSource:
    label: str
    source: Path
    packaged_relative: Path
    category: str


@dataclass(frozen=True)
class BundleContext:
    repo_dir: Path
    run_dir: Path
    scene_dir: Path
    input_dir: Path
    packaged_evidence_dir: Path
    run_attempt_id: str
    output: Path


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AcceptanceBundleError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except AcceptanceBundleError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AcceptanceBundleError(f"Cannot parse {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AcceptanceBundleError(f"{label} must be a JSON object: {path}")
    return value


def _is_link_like(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse and getattr(info, "st_file_attributes", 0) & reparse)


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _require_no_link_components(path: Path, *, label: str) -> None:
    path = _absolute_lexical(path)
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.exists() and _is_link_like(current):
            raise AcceptanceBundleError(
                f"{label} contains a symlink/junction component: {current}"
            )


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _require_directory(path: Path, *, label: str) -> Path:
    lexical = _absolute_lexical(path)
    _require_no_link_components(lexical, label=label)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise AcceptanceBundleError(f"{label} is missing: {lexical}") from exc
    if not resolved.is_dir():
        raise AcceptanceBundleError(f"{label} is not a directory: {resolved}")
    return resolved


def _require_regular_file(
    path: Path,
    *,
    label: str,
    containment_root: Path | None = None,
) -> Path:
    lexical = _absolute_lexical(path)
    if containment_root is not None and not _within(lexical, containment_root):
        raise AcceptanceBundleError(
            f"{label} escapes its allowed root {containment_root}: {lexical}"
        )
    _require_no_link_components(lexical, label=label)
    try:
        info = lexical.lstat()
    except OSError as exc:
        raise AcceptanceBundleError(f"{label} is missing: {lexical}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise AcceptanceBundleError(f"{label} is not a regular file: {lexical}")
    if info.st_size < 1:
        raise AcceptanceBundleError(f"{label} is empty: {lexical}")
    resolved = lexical.resolve(strict=True)
    if containment_root is not None and not _within(resolved, containment_root):
        raise AcceptanceBundleError(
            f"{label} resolves outside its allowed root {containment_root}: {resolved}"
        )
    return resolved


def _require_output_path(scene_dir: Path, output: Path) -> Path:
    output = _absolute_lexical(output)
    if not _within(output, scene_dir):
        raise AcceptanceBundleError(f"--output must be inside --scene-dir: {output}")
    _require_no_link_components(output.parent, label="acceptance output parent")
    if output.exists() and _is_link_like(output):
        raise AcceptanceBundleError(f"Acceptance output is link-like: {output}")
    if output == scene_dir / EVIDENCE_DIRECTORY or _within(
        output, scene_dir / EVIDENCE_DIRECTORY
    ):
        raise AcceptanceBundleError(
            "Acceptance record cannot be inside final_acceptance_evidence"
        )
    return output


def _build_context(
    *,
    repo_dir: Path,
    run_dir: Path,
    scene_dir: Path,
    input_dir: Path,
    run_attempt_id: str,
    output: Path,
) -> BundleContext:
    if not ATTEMPT_PATTERN.fullmatch(run_attempt_id):
        raise AcceptanceBundleError(
            "--run-attempt-id must be 1-128 safe identifier characters"
        )
    repo = _require_directory(repo_dir, label="repository directory")
    run = _require_directory(run_dir, label="run directory")
    scene = _require_directory(scene_dir, label="scene directory")
    inputs = _require_directory(input_dir, label="input directory")
    if not _within(run, repo):
        raise AcceptanceBundleError("--run-dir must be inside --repo-dir")
    if not _within(scene, run):
        raise AcceptanceBundleError("--scene-dir must be inside --run-dir")
    if not _within(inputs, repo):
        raise AcceptanceBundleError("--input-dir must be inside --repo-dir")
    target = _require_output_path(scene, output)
    return BundleContext(
        repo_dir=repo,
        run_dir=run,
        scene_dir=scene,
        input_dir=inputs,
        packaged_evidence_dir=scene / EVIDENCE_DIRECTORY,
        run_attempt_id=run_attempt_id,
        output=target,
    )


def _external_sources(context: BundleContext) -> list[EvidenceSource]:
    preflight = context.repo_dir / "outputs" / "preflight" / context.input_dir.name
    generation = preflight / sam3d_generation.ARTIFACT_DIRECTORY_NAME
    sources = [
        EvidenceSource(
            f"input:{name}",
            context.input_dir / name,
            Path("inputs") / name,
            "immutable_input",
        )
        for name in INPUT_FILES
    ]
    sources.extend(
        (
            EvidenceSource(
                "input_manifest_validation",
                context.run_dir / "input_manifest_validation.json",
                Path("run") / "input_manifest_validation.json",
                "run_evidence",
            ),
            EvidenceSource(
                "pipeline_code_contract",
                context.run_dir / "pipeline_code_contract.json",
                Path("run") / "pipeline_code_contract.json",
                "run_evidence",
            ),
            EvidenceSource(
                "materials_contract",
                context.run_dir / "materials_contract_validation.json",
                Path("run") / "materials_contract_validation.json",
                "run_evidence",
            ),
            EvidenceSource(
                "artiverse_preparation",
                context.run_dir / "artiverse_preparation_validation.json",
                Path("run") / "artiverse_preparation_validation.json",
                "run_evidence",
            ),
            EvidenceSource(
                "artiverse_visual_resources",
                context.run_dir / "artiverse_visual_resources.json",
                Path("run") / "artiverse_visual_resources.json",
                "run_evidence",
            ),
            EvidenceSource(
                "articulated_router",
                context.run_dir / "articulated_router_validation.json",
                Path("run") / "articulated_router_validation.json",
                "run_evidence",
            ),
            EvidenceSource(
                "resolved_asset_policy",
                context.run_dir / "resolved_asset_policy.json",
                Path("run") / "resolved_asset_policy.json",
                "run_evidence",
            ),
            EvidenceSource(
                "sam3d_offline_preflight",
                preflight / "sam3d_offline_load.json",
                Path("preflight") / "sam3d_offline_load.json",
                "model_preflight",
            ),
            EvidenceSource(
                "sam3d_generation_preflight",
                generation / sam3d_generation.RECEIPT_NAME,
                Path("preflight")
                / sam3d_generation.ARTIFACT_DIRECTORY_NAME
                / sam3d_generation.RECEIPT_NAME,
                "model_preflight",
            ),
            EvidenceSource(
                "sam3d_generation_input",
                context.repo_dir / sam3d_generation.CANONICAL_INPUT_RELATIVE,
                Path("preflight") / "sam3d_generation_input.png",
                "model_preflight_artifact",
            ),
            EvidenceSource(
                "sam3d_generation_glb",
                generation / sam3d_generation.GLB_NAME,
                Path("preflight")
                / sam3d_generation.ARTIFACT_DIRECTORY_NAME
                / sam3d_generation.GLB_NAME,
                "model_preflight_artifact",
            ),
            EvidenceSource(
                "sam3d_generation_mask",
                generation / sam3d_generation.MASK_NAME,
                Path("preflight")
                / sam3d_generation.ARTIFACT_DIRECTORY_NAME
                / sam3d_generation.MASK_NAME,
                "model_preflight_artifact",
            ),
            EvidenceSource(
                "sam3d_generation_masked_image",
                generation / sam3d_generation.MASKED_IMAGE_NAME,
                Path("preflight")
                / sam3d_generation.ARTIFACT_DIRECTORY_NAME
                / sam3d_generation.MASKED_IMAGE_NAME,
                "model_preflight_artifact",
            ),
            EvidenceSource(
                "objathor_retrieval_preflight",
                preflight / "objathor_retrieval_offline.json",
                Path("preflight") / "objathor_retrieval_offline.json",
                "model_preflight",
            ),
            EvidenceSource(
                "vlm_vision_smoke",
                preflight / "vlm_vision_smoke.json",
                Path("preflight") / "vlm_vision_smoke.json",
                "model_preflight",
            ),
        )
    )
    return sources


def _internal_paths(scene_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {
        "house_layout": scene_dir / "house_layout.json",
        "reference_school_layout_seed": scene_dir
        / "quality_gates"
        / "reference_school_layout_seed.json",
        "floor_plan_layout_gate": scene_dir / "quality_gates" / "floor_plan_layout.json",
        "room_prompt_binding": scene_dir / "quality_gates" / "room_prompt_binding.json",
        "classroom_variation": scene_dir
        / "quality_gates"
        / "classroom_variation.json",
        "articulated_motion": scene_dir
        / "quality_gates"
        / "articulated_motion.json",
        "articulated_motion_verification": scene_dir
        / "quality_gates"
        / "articulated_motion_verification.json",
        "school_navigation": scene_dir
        / "quality_gates"
        / "school_navigation.json",
        "school_navigation_verification": scene_dir
        / "quality_gates"
        / "school_navigation_verification.json",
        "room_deterministic_summary": scene_dir
        / "quality_gates"
        / "room_self_exam_deterministic"
        / "summary.json",
        "room_visual_summary": scene_dir
        / "quality_gates"
        / "room_self_exam"
        / "summary.json",
        "house_state": scene_dir / "combined_house" / "house_state.json",
        "house_dmd": scene_dir / "combined_house" / "house.dmd.yaml",
        "house_blend": scene_dir / "combined_house" / "house.blend",
        "artiverse_usage": scene_dir / "combined_house" / "artiverse_usage.json",
        "house_cutaway": scene_dir
        / "combined_house"
        / "outlook_renders"
        / "overview_cutaway_evidence.json",
        "overview:overview_top": scene_dir
        / "combined_house"
        / "outlook_renders"
        / "overview_top.png",
        "overview:overview_isometric": scene_dir
        / "combined_house"
        / "outlook_renders"
        / "overview_isometric.png",
        "overview:overview_front": scene_dir
        / "combined_house"
        / "outlook_renders"
        / "overview_front.png",
        "artiverse_final_validation": scene_dir
        / "quality_gates"
        / "artiverse_final_validation.json",
        "whole_floor_gate": scene_dir
        / "quality_gates"
        / "whole_floor_reference.json",
        "drake_sqz": scene_dir / "quality_gates" / "drake_load_sqz.json",
        "simulator_exports": scene_dir
        / "quality_gates"
        / "simulator_exports.json",
        "sage_scene_checker": scene_dir
        / "quality_gates"
        / "sage_scene_checker.json",
    }
    for room_id in ROOM_IDS:
        paths[f"room_deterministic:{room_id}"] = (
            scene_dir
            / "quality_gates"
            / "room_self_exam_deterministic"
            / f"{room_id}.json"
        )
        paths[f"room_visual:{room_id}"] = (
            scene_dir / "quality_gates" / "room_self_exam" / f"{room_id}.json"
        )
        paths[f"room_cutaway:{room_id}"] = (
            scene_dir
            / "review"
            / "room_review_renders"
            / f"{room_id}_cutaway_evidence.json"
        )
    return paths


def _packaged_external_paths(
    context: BundleContext, *, physical_root: Path | None = None
) -> dict[str, Path]:
    root = physical_root or context.packaged_evidence_dir
    return {
        source.label: root / source.packaged_relative
        for source in _external_sources(context)
    }


def _all_fixed_paths(
    context: BundleContext, *, physical_evidence_root: Path | None = None
) -> dict[str, Path]:
    values = _internal_paths(context.scene_dir)
    values.update(
        _packaged_external_paths(context, physical_root=physical_evidence_root)
    )
    return values


def _normalized_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def _validate_input_bundle(paths: Mapping[str, Path]) -> None:
    by_name = {
        label.split(":", 1)[1]: path
        for label, path in paths.items()
        if label.startswith("input:")
    }
    if set(by_name) != set(INPUT_FILES):
        raise AcceptanceBundleError("Packaged immutable input file set is incomplete")
    manifest = _read_json(by_name["input_manifest.json"], label="input manifest")
    original = _normalized_text(by_name["prompt_original.txt"])
    appendix = _normalized_text(by_name["scene_contract_appendix.txt"])
    effective = _normalized_text(by_name["prompt.txt"])
    expected = {
        "original_prompt_sha256": _sha256_bytes(original.encode("utf-8")),
        "appendix_sha256": _sha256_bytes(appendix.encode("utf-8")),
        "effective_prompt_sha256": _sha256_bytes(effective.encode("utf-8")),
        "effective_prompt_chars": len(effective),
        "reference_image_sha256": _sha256_file(by_name["reference.png"]),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise AcceptanceBundleError(f"Input manifest mismatch for {key}")
    if manifest.get("run_name") == SCHOOL_RUN_NAME:
        for key, value in SCHOOL_EXTERNAL_INPUT_BINDING.items():
            if expected.get(key) != value:
                raise AcceptanceBundleError(
                    f"Packaged input differs from external school binding for {key}"
                )
    if effective != original + "\n" + appendix:
        raise AcceptanceBundleError(
            "Effective prompt is not original prompt plus the compulsory appendix"
        )
    try:
        with by_name["prompt.csv"].open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error) as exc:
        raise AcceptanceBundleError(f"Cannot parse packaged prompt.csv: {exc}") from exc
    if len(rows) != 1 or str(rows[0].get("prompt", "")).replace("\r\n", "\n") != effective:
        raise AcceptanceBundleError("prompt.csv is not the exact one-row effective prompt")
    if str(rows[0].get("scene_index", "")) != str(manifest.get("scene_index", "")):
        raise AcceptanceBundleError("prompt.csv scene_index differs from input manifest")
    if manifest.get("expected_room_ids") != list(ROOM_IDS):
        raise AcceptanceBundleError("Input manifest does not name the exact 11 rooms")
    if manifest.get("pipeline_contract") != {
        "id": "scenesmith_full_quality_v1",
        "final_assembly_policy": "external_artiverse_gated",
        "required_articulated_source": "artiverse",
    }:
        raise AcceptanceBundleError("Input manifest does not make Artiverse compulsory")


def _empty_string_list(value: Any, *, label: str) -> None:
    if value != []:
        raise AcceptanceBundleError(f"{label} must be an empty list")


def _require_passing_json(
    document: Mapping[str, Any], *, label: str, pass_key: str = "status"
) -> None:
    expected: Any = True if pass_key == "pass" else "pass"
    if document.get(pass_key) != expected:
        raise AcceptanceBundleError(f"{label} is not passing")
    if len(document) < 2:
        raise AcceptanceBundleError(f"{label} is a status-only spoof")
    for issue_key in ("critical_issues", "failures", "failed_checks"):
        if issue_key in document and document.get(issue_key) not in ([], None):
            raise AcceptanceBundleError(f"{label} contains {issue_key}")


def _verify_pipeline_attestation(document: Mapping[str, Any]) -> None:
    attestation = document.get("attestation")
    payload = {key: value for key, value in document.items() if key != "attestation"}
    expected = {
        "schema_version": 1,
        "algorithm": "sha256",
        "sha256": _sha256_bytes(_canonical_json(payload)),
    }
    if attestation != expected:
        raise AcceptanceBundleError("Pipeline/code contract attestation is invalid")


def _sam3d_attestation(document: Mapping[str, Any]) -> str:
    """Recompute the canonical preflight digest without duplicating its schema."""

    return sam3d_preflight._attestation_sha256(document)


def _objathor_attestation(document: Mapping[str, Any]) -> str:
    keys = (
        "schema_version",
        "status",
        "offline",
        "offline_environment",
        "dataset",
        "model",
        "model_smoke",
        "evidence",
    )
    return _sha256_bytes(_canonical_json({key: document.get(key) for key in keys}))


def _require_attestation(
    document: Mapping[str, Any], *, label: str, expected_digest: str
) -> None:
    if document.get("attestation") != {
        "algorithm": "sha256",
        "sha256": expected_digest,
    }:
        raise AcceptanceBundleError(f"{label} JSON attestation is invalid")


def _require_scores(document: Mapping[str, Any], *, label: str) -> None:
    scores = document.get("scores")
    threshold = document.get("threshold")
    if not isinstance(scores, dict) or not scores:
        raise AcceptanceBundleError(f"{label} has no score mapping")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or float(threshold) != 7.0
    ):
        raise AcceptanceBundleError(f"{label} threshold must be exactly 7")
    expected_keys = (
        FLOOR_SCORE_KEYS if label == "whole_floor_gate" else ROOM_SCORE_KEYS
    )
    if set(scores) != set(expected_keys):
        raise AcceptanceBundleError(f"{label} score keys are not exact")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value < float(threshold)
        or value > 10
        for value in scores.values()
    ):
        raise AcceptanceBundleError(f"{label} contains a non-passing score")


def _embedded_file_record(
    record: Any,
    *,
    expected: Path,
    label: str,
    strict_path: bool,
    path_key: str = "path",
    hash_key: str = "sha256",
    size_key: str | None = None,
) -> None:
    if not isinstance(record, dict):
        raise AcceptanceBundleError(f"{label} file evidence is missing")
    expected = _require_regular_file(expected, label=label)
    if strict_path:
        path_value = record.get(path_key)
        if not isinstance(path_value, str) or Path(path_value).resolve() != expected:
            raise AcceptanceBundleError(f"{label} is bound to a different path")
    if record.get(hash_key) != _sha256_file(expected):
        raise AcceptanceBundleError(f"{label} embedded hash is stale")
    if size_key is not None and record.get(size_key) != expected.stat().st_size:
        raise AcceptanceBundleError(f"{label} embedded size is stale")


def _validate_derivation_receipt(
    document: Mapping[str, Any],
    *,
    source_state: Path,
    source_blend: Path,
    renders: Sequence[tuple[str, Path]],
    label: str,
) -> None:
    receipt = document.get("derivation_receipt")
    if not isinstance(receipt, dict) or receipt.get("schema_id") != DERIVATION_SCHEMA_ID:
        raise AcceptanceBundleError(f"{label} derivation receipt is missing")
    _embedded_file_record(
        receipt.get("source_state"),
        expected=source_state,
        label=f"{label} derivation state",
        strict_path=True,
        size_key="size_bytes",
    )
    _embedded_file_record(
        receipt.get("source_blend"),
        expected=source_blend,
        label=f"{label} derivation blend",
        strict_path=True,
        size_key="size_bytes",
    )
    records = receipt.get("renders")
    if not isinstance(records, list) or len(records) != len(renders):
        raise AcceptanceBundleError(f"{label} derivation render set is malformed")
    image_hashes: list[str] = []
    for index, (view_name, path) in enumerate(renders):
        record = records[index]
        if not isinstance(record, dict) or record.get("view_name") != view_name:
            raise AcceptanceBundleError(f"{label} derivation view order is malformed")
        _embedded_file_record(
            record,
            expected=path,
            label=f"{label} derivation {view_name}",
            strict_path=True,
            hash_key="sha256",
            size_key="size_bytes",
        )
        image_hashes.append(str(record.get("sha256")))
    if len(set(image_hashes)) != len(image_hashes):
        raise AcceptanceBundleError(f"{label} render image hashes are not distinct")
    payload = {key: value for key, value in receipt.items() if key != "attestation"}
    expected_attestation = {
        "algorithm": "sha256",
        "sha256": _sha256_bytes(_canonical_json(payload)),
    }
    if receipt.get("attestation") != expected_attestation:
        raise AcceptanceBundleError(f"{label} derivation attestation is invalid")


def _validate_vlm_request_record(
    record: Any,
    *,
    label: str,
    score_keys: Sequence[str],
    threshold: float,
    reference_sha256: str,
    render_sha256: Sequence[str],
    expected_record: Mapping[str, Any] | None = None,
) -> None:
    if (
        not isinstance(record, dict)
        or record.get("schema_id") != VLM_REQUEST_SCHEMA_ID
        or record.get("schema_version") != 1
        or record.get("algorithm") != "sha256"
    ):
        raise AcceptanceBundleError(f"{label} canonical VLM request is missing")
    if record.get("score_keys") != list(score_keys) or record.get("threshold") != threshold:
        raise AcceptanceBundleError(f"{label} VLM score/threshold contract is stale")
    if record.get("reference_image_sha256") != reference_sha256:
        raise AcceptanceBundleError(f"{label} VLM reference binding is stale")
    recorded_renders = record.get("review_image_sha256")
    if recorded_renders is None:
        recorded_renders = record.get("overview_image_sha256")
    if recorded_renders != list(render_sha256):
        raise AcceptanceBundleError(f"{label} VLM render binding is stale")
    if not _is_sha256(record.get("messages_sha256")):
        raise AcceptanceBundleError(f"{label} VLM message digest is malformed")
    payload = {key: value for key, value in record.items() if key != "request_sha256"}
    if record.get("request_sha256") != _sha256_bytes(_canonical_json(payload)):
        raise AcceptanceBundleError(f"{label} canonical VLM request digest is invalid")
    if expected_record is not None and record != dict(expected_record):
        raise AcceptanceBundleError(
            f"{label} canonical VLM request differs from reconstructed messages"
        )


def _validate_cutaway(
    document: Mapping[str, Any],
    *,
    room_id: str,
    scene_dir: Path,
) -> None:
    if document.get("schema_id") != "scenesmith_room_cutaway_review_v1":
        raise AcceptanceBundleError(f"{room_id} cutaway schema_id is unsupported")
    if document.get("schema_version") != 1 or document.get("status") != "pass":
        raise AcceptanceBundleError(f"{room_id} cutaway is not passing schema v1")
    if document.get("room_id") != room_id:
        raise AcceptanceBundleError(f"{room_id} cutaway room identity is stale")
    if document.get("expected_views") != ["top", "oblique_a", "oblique_b"]:
        raise AcceptanceBundleError(f"{room_id} cutaway view contract is malformed")
    if document.get("rendered_view_count") != 3:
        raise AcceptanceBundleError(f"{room_id} cutaway does not prove three views")
    blend = (
        scene_dir
        / f"room_{room_id}"
        / "scene_states"
        / "final_scene"
        / "scene.blend"
    )
    state = blend.with_name("scene_state.json")
    _embedded_file_record(
        document.get("source_state"),
        expected=state,
        label=f"{room_id} cutaway source state",
        strict_path=True,
        size_key="size_bytes",
    )
    _embedded_file_record(
        document.get("source_blend"),
        expected=blend,
        label=f"{room_id} cutaway source blend",
        strict_path=True,
        size_key="size_bytes",
    )
    classification = document.get("classification")
    if not isinstance(classification, dict):
        raise AcceptanceBundleError(f"{room_id} cutaway classification is missing")
    for role in ("wall", "floor", "content"):
        if not isinstance(classification.get(role), list) or not classification[role]:
            raise AcceptanceBundleError(f"{room_id} cutaway has no {role} classification")
    if classification.get("combined_envelope") not in ([], None):
        raise AcceptanceBundleError(f"{room_id} cutaway has an indivisible envelope")
    views = document.get("views")
    names = ("top", "oblique_a", "oblique_b")
    if not isinstance(views, list) or len(views) != 3:
        raise AcceptanceBundleError(f"{room_id} cutaway views are malformed")
    for index, name in enumerate(names):
        view = views[index]
        if not isinstance(view, dict) or view.get("view_name") != name:
            raise AcceptanceBundleError(f"{room_id} cutaway view order is malformed")
        image = scene_dir / "review" / "room_review_renders" / f"{room_id}_{name}.png"
        _embedded_file_record(
            view,
            expected=image,
            label=f"{room_id} cutaway {name}",
            strict_path=True,
            path_key="image",
            hash_key="image_sha256",
            size_key="image_size_bytes",
        )
        proof = view.get("cutaway")
        if not isinstance(proof, dict) or proof.get("established") is not True:
            raise AcceptanceBundleError(f"{room_id} cutaway {name} is unproven")
        if proof.get("view_name") != name:
            raise AcceptanceBundleError(f"{room_id} cutaway {name} identity is stale")
        if proof.get("overhead_state") not in {"hidden", "verified_absent"}:
            raise AcceptanceBundleError(f"{room_id} cutaway {name} overhead is unproven")
        if name != "top" and not proof.get("hidden_camera_side_walls"):
            raise AcceptanceBundleError(f"{room_id} cutaway {name} hides no near wall")
        for key in (
            "visible_far_wall_object_names",
            "visible_floor_object_names",
            "visible_content_object_names",
        ):
            if not isinstance(proof.get(key), list) or not proof[key]:
                raise AcceptanceBundleError(f"{room_id} cutaway {name} has no {key}")
    _validate_derivation_receipt(
        document,
        source_state=state,
        source_blend=blend,
        renders=[
            (
                name,
                scene_dir
                / "review"
                / "room_review_renders"
                / f"{room_id}_{name}.png",
            )
            for name in names
        ],
        label=f"{room_id} cutaway",
    )


def _validate_house_cutaway(document: Mapping[str, Any], scene_dir: Path) -> None:
    if document.get("schema_id") != "scenesmith_house_cutaway_review_v1":
        raise AcceptanceBundleError("Final-house cutaway schema_id is unsupported")
    if document.get("schema_version") != 1 or document.get("status") != "pass":
        raise AcceptanceBundleError("Final-house cutaway is not passing schema v1")
    names = ("overview_top", "overview_isometric", "overview_front")
    if document.get("expected_views") != list(names) or document.get(
        "rendered_view_count"
    ) != 3:
        raise AcceptanceBundleError("Final-house cutaway view contract is malformed")
    _embedded_file_record(
        document.get("source_blend"),
        expected=scene_dir / "combined_house" / "house.blend",
        label="final-house cutaway source blend",
        strict_path=True,
        size_key="size_bytes",
    )
    _embedded_file_record(
        document.get("source_state"),
        expected=scene_dir / "combined_house" / "house_state.json",
        label="final-house cutaway source state",
        strict_path=True,
        size_key="size_bytes",
    )
    classification = document.get("classification")
    if not isinstance(classification, dict):
        raise AcceptanceBundleError("Final-house cutaway classification is missing")
    for role in ("wall", "floor", "content"):
        if not isinstance(classification.get(role), list) or not classification[role]:
            raise AcceptanceBundleError(f"Final-house cutaway has no {role}")
    if classification.get("combined_envelope") not in ([], None):
        raise AcceptanceBundleError("Final-house cutaway has an indivisible envelope")
    views = document.get("views")
    if not isinstance(views, list) or len(views) != 3:
        raise AcceptanceBundleError("Final-house cutaway view records are malformed")
    for index, name in enumerate(names):
        view = views[index]
        if not isinstance(view, dict) or view.get("view_name") != name:
            raise AcceptanceBundleError("Final-house cutaway view order is malformed")
        _embedded_file_record(
            view,
            expected=scene_dir / "combined_house" / "outlook_renders" / f"{name}.png",
            label=f"final-house cutaway {name}",
            strict_path=True,
            path_key="image",
            hash_key="image_sha256",
            size_key="image_size_bytes",
        )
        proof = view.get("cutaway")
        if not isinstance(proof, dict) or proof.get("established") is not True:
            raise AcceptanceBundleError(f"Final-house cutaway {name} is unproven")
        if proof.get("view_name") != name:
            raise AcceptanceBundleError(f"Final-house cutaway {name} identity is stale")
        if name != "overview_top" and not proof.get("hidden_camera_side_walls"):
            raise AcceptanceBundleError(f"Final-house cutaway {name} hides no near wall")
        for key in (
            "visible_far_wall_object_names",
            "visible_floor_object_names",
            "visible_content_object_names",
        ):
            if not isinstance(proof.get(key), list) or not proof[key]:
                raise AcceptanceBundleError(f"Final-house cutaway {name} has no {key}")
    _validate_derivation_receipt(
        document,
        source_state=scene_dir / "combined_house" / "house_state.json",
        source_blend=scene_dir / "combined_house" / "house.blend",
        renders=[
            (
                name,
                scene_dir / "combined_house" / "outlook_renders" / f"{name}.png",
            )
            for name in names
        ],
        label="final-house cutaway",
    )


def _validate_classroom_variation(
    document: Mapping[str, Any], *, scene_dir: Path
) -> None:
    if document.get("schema_version") != 1 or document.get("status") != "pass":
        raise AcceptanceBundleError(
            "Classroom-variation gate is not a passing schema-v1 record"
        )
    _empty_string_list(
        document.get("critical_issues"),
        label="classroom variation critical_issues",
    )
    minimum = document.get("minimum_variation_score")
    assessment = document.get("visual_assessment")
    if (
        not isinstance(minimum, (int, float))
        or isinstance(minimum, bool)
        or minimum < 7
        or not isinstance(assessment, dict)
        or not isinstance(assessment.get("variation_quality_score"), (int, float))
        or isinstance(assessment.get("variation_quality_score"), bool)
        or assessment["variation_quality_score"] < minimum
    ):
        raise AcceptanceBundleError(
            "Classroom-variation visual score does not meet its threshold"
        )
    classroom_ids = frozenset(f"classroom_{index:02d}" for index in range(1, 7))
    classrooms = assessment.get("classrooms")
    if not isinstance(classrooms, dict) or set(classrooms) != classroom_ids:
        raise AcceptanceBundleError(
            "Classroom-variation visual proof does not cover exactly six classrooms"
        )
    for room_id, value in classrooms.items():
        if (
            not isinstance(value, dict)
            or value.get("status") != "pass"
            or not isinstance(value.get("distinctive_features"), list)
            or len(value["distinctive_features"]) < 2
            or not isinstance(value.get("seating_layout"), str)
            or not value["seating_layout"].strip()
        ):
            raise AcceptanceBundleError(
                f"Classroom-variation visual proof is incomplete for {room_id}"
            )
    _empty_string_list(
        assessment.get("too_similar_pairs"),
        label="classroom variation too_similar_pairs",
    )
    fingerprints = document.get("fingerprints")
    if not isinstance(fingerprints, dict) or set(fingerprints) != classroom_ids:
        raise AcceptanceBundleError(
            "Classroom-variation deterministic fingerprints are incomplete"
        )
    combined: list[str] = []
    for room_id, value in fingerprints.items():
        if not isinstance(value, dict):
            raise AcceptanceBundleError(
                f"Classroom-variation fingerprint is malformed for {room_id}"
            )
        for key in (
            "semantic_sha256",
            "seating_sha256",
            "decoration_sha256",
            "combined_sha256",
        ):
            if not _is_sha256(value.get(key)):
                raise AcceptanceBundleError(
                    f"Classroom-variation fingerprint has malformed {key} for {room_id}"
                )
        combined.append(value["combined_sha256"])
    if len(set(combined)) != 6:
        raise AcceptanceBundleError(
            "Classroom-variation deterministic fingerprints are not all distinct"
        )
    evidence = document.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 12:
        raise AcceptanceBundleError(
            "Classroom-variation gate must bind six states and six top views"
        )
    identities: set[tuple[str, str]] = set()
    for record in evidence:
        if not isinstance(record, dict):
            raise AcceptanceBundleError(
                "Classroom-variation file evidence is malformed"
            )
        room_id = str(record.get("room_id", ""))
        role = str(record.get("role", ""))
        identity = (room_id, role)
        if (
            room_id not in classroom_ids
            or role not in {"state", "top_view"}
            or identity in identities
        ):
            raise AcceptanceBundleError(
                "Classroom-variation file evidence identity is invalid or duplicated"
            )
        identities.add(identity)
        expected = (
            scene_dir
            / f"room_{room_id}"
            / "scene_states"
            / "final_scene"
            / "scene_state.json"
            if role == "state"
            else scene_dir
            / "review"
            / "room_review_renders"
            / f"{room_id}_top.png"
        )
        _embedded_file_record(
            record,
            expected=expected,
            label=f"classroom variation {room_id} {role}",
            strict_path=True,
            size_key="size_bytes",
        )
    if identities != {
        (room_id, role)
        for room_id in classroom_ids
        for role in ("state", "top_view")
    }:
        raise AcceptanceBundleError(
            "Classroom-variation file evidence set is incomplete"
        )
    payload = {
        key: value for key, value in document.items() if key != "attestation_sha256"
    }
    if document.get("attestation_sha256") != _sha256_bytes(
        _canonical_json(payload)
    ):
        raise AcceptanceBundleError(
            "Classroom-variation JSON attestation is invalid"
        )


def _validate_articulated_motion(
    document: Mapping[str, Any], *, scene_dir: Path
) -> None:
    expected_roles = {
        "library_glass_door_bookcase": ("library", "revolute", 1),
        "school_supply_two_door_utility_cabinet": (
            "storage_room",
            "revolute",
            2,
        ),
        "teacher_filing_drawer_cabinet": ("classroom_01", "prismatic", 1),
    }
    if (
        document.get("schema_version") != 1
        or document.get("status") != "pass"
        or document.get("profile") != "school_reference_20260710"
        or document.get("drake_motion_exercised") is not True
    ):
        raise AcceptanceBundleError(
            "Articulated-motion gate is not a passing school-profile proof"
        )
    _empty_string_list(
        document.get("critical_issues"),
        label="articulated motion critical_issues",
    )
    if set(document.get("required_roles", [])) != set(expected_roles):
        raise AcceptanceBundleError(
            "Articulated-motion proof does not name the exact three roles"
        )
    artiverse_roles = document.get("artiverse_roles")
    if (
        not isinstance(artiverse_roles, list)
        or not artiverse_roles
        or not set(artiverse_roles).issubset(expected_roles)
    ):
        raise AcceptanceBundleError(
            "Articulated-motion proof has no required Artiverse role"
        )
    state_evidence = document.get("state_evidence")
    required_rooms = {"library", "storage_room", "classroom_01"}
    if not isinstance(state_evidence, dict) or set(state_evidence) != required_rooms:
        raise AcceptanceBundleError(
            "Articulated-motion proof does not bind its three room states"
        )
    for room_id, record in state_evidence.items():
        _embedded_file_record(
            record,
            expected=scene_dir
            / f"room_{room_id}"
            / "scene_states"
            / "final_scene"
            / "scene_state.json",
            label=f"articulated motion room state {room_id}",
            strict_path=True,
            size_key="size_bytes",
        )
    roles = document.get("roles")
    if not isinstance(roles, dict) or set(roles) != set(expected_roles):
        raise AcceptanceBundleError(
            "Articulated-motion role evidence set is malformed"
        )
    proven_artiverse: set[str] = set()
    for role, (expected_room, expected_type, minimum_joints) in expected_roles.items():
        value = roles[role]
        if not isinstance(value, dict) or value.get("room_id") != expected_room:
            raise AcceptanceBundleError(
                f"Articulated-motion role {role} is bound to the wrong room"
            )
        if not value.get("object_id") or not value.get("articulated_id"):
            raise AcceptanceBundleError(
                f"Articulated-motion role {role} lacks a concrete asset identity"
            )
        if value.get("articulated_source") == "artiverse":
            proven_artiverse.add(role)
        state_hash = state_evidence[expected_room].get("sha256")
        if value.get("state_sha256") != state_hash:
            raise AcceptanceBundleError(
                f"Articulated-motion role {role} has a stale state hash"
            )
        sdf = value.get("sdf")
        if not isinstance(sdf, dict):
            raise AcceptanceBundleError(
                f"Articulated-motion role {role} has no SDF proof"
            )
        sdf_path_value = sdf.get("path")
        if not isinstance(sdf_path_value, str):
            raise AcceptanceBundleError(
                f"Articulated-motion role {role} SDF path is missing"
            )
        sdf_path = Path(sdf_path_value)
        if not _within(sdf_path.resolve(), scene_dir / f"room_{expected_room}"):
            raise AcceptanceBundleError(
                f"Articulated-motion role {role} SDF escapes its room"
            )
        _embedded_file_record(
            sdf,
            expected=sdf_path,
            label=f"articulated motion SDF {role}",
            strict_path=True,
            size_key="size_bytes",
        )
        motion = value.get("drake_motion")
        if not isinstance(motion, dict) or motion.get("status") != "pass":
            raise AcceptanceBundleError(
                f"Articulated-motion role {role} has no passing Drake exercise"
            )
        exercises = motion.get("joint_exercises")
        if not isinstance(exercises, list) or len(exercises) < minimum_joints:
            raise AcceptanceBundleError(
                f"Articulated-motion role {role} exercised too few joints"
            )
        for exercise in exercises:
            delta = exercise.get("transform_delta") if isinstance(exercise, dict) else None
            if (
                not isinstance(exercise, dict)
                or exercise.get("joint_type") != expected_type
                or exercise.get("child_body_pose_changed") is not True
                or not isinstance(exercise.get("tested_positions"), list)
                or len(exercise["tested_positions"]) != 2
                or not isinstance(delta, dict)
                or not isinstance(delta.get("max_abs_matrix_delta"), (int, float))
                or delta["max_abs_matrix_delta"] <= 0
            ):
                raise AcceptanceBundleError(
                    f"Articulated-motion role {role} contains an invalid joint exercise"
                )
    if not set(artiverse_roles).issubset(proven_artiverse) or not proven_artiverse:
        raise AcceptanceBundleError(
            "Articulated-motion Artiverse role summary differs from role provenance"
        )
    evidence_payload = {
        "schema_version": document.get("schema_version"),
        "profile": document.get("profile"),
        "scene_dir": document.get("scene_dir"),
        "required_roles": document.get("required_roles"),
        "artiverse_roles": document.get("artiverse_roles"),
        "state_evidence": document.get("state_evidence"),
        "roles": document.get("roles"),
        "drake_motion_exercised": document.get("drake_motion_exercised"),
    }
    if document.get("evidence_sha256") != _sha256_bytes(
        _canonical_json(evidence_payload)
    ):
        raise AcceptanceBundleError(
            "Articulated-motion evidence digest is stale"
        )
    unsigned = {key: value for key, value in document.items() if key != "attestation"}
    if document.get("attestation") != {
        "algorithm": "sha256",
        "sha256": _sha256_bytes(_canonical_json(unsigned)),
    }:
        raise AcceptanceBundleError(
            "Articulated-motion JSON attestation is invalid"
        )


def _validate_school_navigation(
    document: Mapping[str, Any], *, paths: Mapping[str, Path], scene_dir: Path
) -> None:
    if (
        document.get("schema_version") != 1
        or document.get("algorithm") != "school_navigation_grid_v1"
        or document.get("status") != "pass"
    ):
        raise AcceptanceBundleError(
            "School-navigation gate schema/algorithm/status is invalid"
        )
    _empty_string_list(
        document.get("critical_issues"),
        label="school navigation critical_issues",
    )
    attestation = document.get("attestation")
    unsigned = {key: value for key, value in document.items() if key != "attestation"}
    if (
        not isinstance(attestation, dict)
        or attestation.get("algorithm")
        != "sha256(canonical JSON excluding attestation)"
        or attestation.get("payload_sha256")
        != _sha256_bytes(_canonical_json(unsigned))
        or attestation.get("self_attested_status") != "pass"
        or attestation.get("all_inputs_hashed") is not True
        or attestation.get("all_routes_hashed") is not True
    ):
        raise AcceptanceBundleError(
            "School-navigation self-attestation is invalid"
        )
    parameters = document.get("parameters")
    if (
        not isinstance(parameters, dict)
        or parameters.get("minimum_door_width_m") != 0.9
        or parameters.get("minimum_entrance_width_m") != 1.6
        or any(
            not isinstance(parameters.get(key), (int, float))
            or isinstance(parameters.get(key), bool)
            or parameters[key] <= 0
            for key in (
                "humanoid_radius_m",
                "humanoid_height_m",
                "grid_resolution_m",
                "turning_diameter_m",
            )
        )
    ):
        raise AcceptanceBundleError(
            "School-navigation humanoid/grid parameters are malformed"
        )
    inputs = document.get("inputs")
    if not isinstance(inputs, dict):
        raise AcceptanceBundleError("School-navigation input evidence is missing")
    _embedded_file_record(
        inputs.get("house_layout"),
        expected=paths["house_layout"],
        label="school navigation house layout",
        strict_path=True,
    )
    _embedded_file_record(
        inputs.get("house_state"),
        expected=paths["house_state"],
        label="school navigation house state",
        strict_path=True,
    )
    states = inputs.get("final_room_states")
    if not isinstance(states, dict) or set(states) != ROOM_SET:
        raise AcceptanceBundleError(
            "School-navigation proof does not bind exactly 11 final room states"
        )
    for room_id, record in states.items():
        state_path = (
            scene_dir
            / f"room_{room_id}"
            / "scene_states"
            / "final_scene"
            / "scene_state.json"
        )
        _embedded_file_record(
            record,
            expected=state_path,
            label=f"school navigation room state {room_id}",
            strict_path=True,
        )
        state = _read_json(state_path, label=f"school navigation state {room_id}")
        objects = state.get("objects")
        if (
            not isinstance(objects, (dict, list))
            or record.get("objects_sha256")
            != _sha256_bytes(_canonical_json(objects))
            or record.get("object_count") != len(objects)
        ):
            raise AcceptanceBundleError(
                f"School-navigation object evidence is stale for {room_id}"
            )
    topology = document.get("topology")
    direct = topology.get("direct_common_portals_by_room") if isinstance(topology, dict) else None
    if not isinstance(direct, dict) or set(direct) != ROOM_SET:
        raise AcceptanceBundleError(
            "School-navigation direct-threshold topology is malformed"
        )
    common_zones = set(topology.get("common_zone_ids", []))
    if not {"library", "main_corridor"}.issubset(common_zones):
        raise AcceptanceBundleError(
            "School-navigation common circulation core is missing"
        )
    if any(not direct[room_id] for room_id in ROOM_SET - common_zones):
        raise AcceptanceBundleError(
            "School-navigation proof has a room without a direct common threshold"
        )
    turning = document.get("turning_areas")
    turning_rooms = {
        "library",
        "main_corridor",
        *(f"classroom_{index:02d}" for index in range(1, 7)),
    }
    if (
        not isinstance(turning, dict)
        or set(turning) != turning_rooms
        or any(
            not isinstance(value, dict)
            or value.get("status") != "pass"
            or value.get("candidate_count", 0) < 1
            for value in turning.values()
        )
    ):
        raise AcceptanceBundleError(
            "School-navigation turning-area proof is incomplete"
        )
    routes = document.get("routes")
    if not isinstance(routes, dict) or set(routes) != ROOM_SET:
        raise AcceptanceBundleError(
            "School-navigation route set is not the exact 11-room set"
        )
    for room_id, route in routes.items():
        if (
            not isinstance(route, dict)
            or route.get("status") != "pass"
            or not isinstance(route.get("cells"), list)
            or not route["cells"]
            or not isinstance(route.get("waypoints"), list)
            or not route["waypoints"]
            or route.get("route_sha256")
            != _sha256_bytes(
                _canonical_json(
                    {"cells": route["cells"], "waypoints": route["waypoints"]}
                )
            )
        ):
            raise AcceptanceBundleError(
                f"School-navigation route is invalid for {room_id}"
            )
    if document.get("routes_sha256") != _sha256_bytes(_canonical_json(routes)):
        raise AcceptanceBundleError("School-navigation route-set digest is stale")
    occupancy = document.get("occupancy")
    if not isinstance(occupancy, dict) or occupancy.get(
        "obstacles_sha256"
    ) != _sha256_bytes(_canonical_json(occupancy.get("obstacles"))):
        raise AcceptanceBundleError(
            "School-navigation obstacle evidence digest is stale"
        )


def _validate_articulated_motion_verification(
    document: Mapping[str, Any], *, paths: Mapping[str, Path], scene_dir: Path
) -> None:
    if (
        document.get("verification_schema_id")
        != "scenesmith_articulated_motion_verification_v1"
        or document.get("schema_version") != 1
        or document.get("status") != "pass"
        or document.get("mode") != "verify-only"
        or document.get("scene_dir") != str(scene_dir.resolve())
        or document.get("saved_output")
        != str(paths["articulated_motion"].resolve())
        or document.get("drake_motion_repeated") is not True
    ):
        raise AcceptanceBundleError(
            "Articulated-motion repeat proof schema/status/binding is invalid"
        )
    _empty_string_list(
        document.get("critical_issues"),
        label="articulated motion verification critical_issues",
    )
    elapsed = document.get("elapsed_seconds")
    if (
        not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or elapsed < 0
    ):
        raise AcceptanceBundleError(
            "Articulated-motion repeat proof elapsed time is malformed"
        )

    source = document.get("source_gate")
    _embedded_file_record(
        source,
        expected=paths["articulated_motion"],
        label="articulated motion repeat source gate",
        strict_path=True,
        size_key="size_bytes",
    )
    source_document = _read_json(
        paths["articulated_motion"], label="articulated motion repeat source gate"
    )
    _validate_articulated_motion(source_document, scene_dir=scene_dir)
    source_attestation = source_document.get("attestation")
    source_evidence_sha256 = source_document.get("evidence_sha256")
    if (
        not isinstance(source, dict)
        or source.get("evidence_sha256") != source_evidence_sha256
        or source.get("attestation_sha256")
        != (
            source_attestation.get("sha256")
            if isinstance(source_attestation, dict)
            else None
        )
        or document.get("saved_evidence_sha256") != source_evidence_sha256
    ):
        raise AcceptanceBundleError(
            "Articulated-motion repeat proof source digest is stale"
        )

    recomputed = document.get("recomputed_gate")
    repeated_result = recomputed.get("result") if isinstance(recomputed, dict) else None
    if not isinstance(repeated_result, dict):
        raise AcceptanceBundleError(
            "Articulated-motion repeat proof has no full recomputed result"
        )
    _validate_articulated_motion(repeated_result, scene_dir=scene_dir)
    if (
        recomputed.get("status") != "pass"
        or recomputed.get("role_count") != 3
        or recomputed.get("evidence_sha256") != source_evidence_sha256
        or document.get("fresh_evidence_sha256") != source_evidence_sha256
        or recomputed.get("attestation") != repeated_result.get("attestation")
        or recomputed.get("result_sha256")
        != _sha256_bytes(_canonical_json(repeated_result))
    ):
        raise AcceptanceBundleError(
            "Articulated-motion repeated Drake result differs from its source gate"
        )
    unsigned = {key: value for key, value in document.items() if key != "attestation"}
    if document.get("attestation") != {
        "algorithm": "sha256",
        "sha256": _sha256_bytes(_canonical_json(unsigned)),
    }:
        raise AcceptanceBundleError(
            "Articulated-motion repeat-proof attestation is invalid"
        )


def _validate_school_navigation_verification(
    document: Mapping[str, Any], *, paths: Mapping[str, Path], scene_dir: Path
) -> None:
    if (
        document.get("verification_schema_id")
        != "scenesmith_school_navigation_verification_v1"
        or document.get("schema_version") != 1
        or document.get("status") != "pass"
        or document.get("mode") != "verify-only"
        or document.get("scene_dir") != str(scene_dir.resolve())
        or document.get("verified_output")
        != str(paths["school_navigation"].resolve())
        or document.get("navigation_recomputed") is not True
    ):
        raise AcceptanceBundleError(
            "School-navigation repeat proof schema/status/binding is invalid"
        )
    _empty_string_list(
        document.get("critical_issues"),
        label="school navigation verification critical_issues",
    )
    source = document.get("source_gate")
    _embedded_file_record(
        source,
        expected=paths["school_navigation"],
        label="school navigation repeat source gate",
        strict_path=True,
        size_key="size_bytes",
    )
    source_document = _read_json(
        paths["school_navigation"], label="school navigation repeat source gate"
    )
    _validate_school_navigation(source_document, paths=paths, scene_dir=scene_dir)
    source_attestation = source_document.get("attestation")
    source_payload_sha256 = (
        source_attestation.get("payload_sha256")
        if isinstance(source_attestation, dict)
        else None
    )
    source_routes_sha256 = source_document.get("routes_sha256")
    if (
        not isinstance(source, dict)
        or source.get("payload_sha256") != source_payload_sha256
        or source.get("routes_sha256") != source_routes_sha256
        or document.get("payload_sha256") != source_payload_sha256
        or document.get("routes_sha256") != source_routes_sha256
    ):
        raise AcceptanceBundleError(
            "School-navigation repeat proof source digest is stale"
        )

    recomputed = document.get("recomputed_gate")
    repeated_result = recomputed.get("result") if isinstance(recomputed, dict) else None
    if not isinstance(repeated_result, dict):
        raise AcceptanceBundleError(
            "School-navigation repeat proof has no full recomputed result"
        )
    _validate_school_navigation(repeated_result, paths=paths, scene_dir=scene_dir)
    repeated_attestation = repeated_result.get("attestation")
    repeated_payload_sha256 = (
        repeated_attestation.get("payload_sha256")
        if isinstance(repeated_attestation, dict)
        else None
    )
    if (
        recomputed.get("status") != "pass"
        or recomputed.get("route_count") != len(ROOM_IDS)
        or recomputed.get("payload_sha256") != source_payload_sha256
        or recomputed.get("routes_sha256") != source_routes_sha256
        or recomputed.get("attestation") != repeated_attestation
        or recomputed.get("result_sha256")
        != _sha256_bytes(_canonical_json(repeated_result))
        or _sha256_bytes(_canonical_json(repeated_result))
        != _sha256_bytes(_canonical_json(source_document))
    ):
        raise AcceptanceBundleError(
            "School-navigation full recomputation differs from its source gate"
        )
    unsigned = {key: value for key, value in document.items() if key != "attestation"}
    if document.get("attestation") != {
        "algorithm": "sha256(canonical JSON excluding attestation)",
        "payload_sha256": _sha256_bytes(_canonical_json(unsigned)),
        "self_attested_status": "pass",
    }:
        raise AcceptanceBundleError(
            "School-navigation repeat-proof attestation is invalid"
        )


def _validate_artiverse_usage(
    document: Mapping[str, Any], *, paths: Mapping[str, Path], scene_dir: Path
) -> None:
    if document.get("schema_version") != 2 or document.get("status") != "pass":
        raise AcceptanceBundleError("Artiverse usage is not a passing schema-v2 record")
    if document.get("dataset") != "artiverse":
        raise AcceptanceBundleError("Artiverse usage dataset label is invalid")
    placed = document.get("placed_assets")
    final = document.get("final_surviving_assets")
    if not isinstance(placed, list) or not placed or not isinstance(final, list) or not final:
        raise AcceptanceBundleError("Artiverse usage has no placed/surviving assets")
    if document.get("placed_asset_count") != len(placed) or document.get(
        "final_surviving_asset_count"
    ) != len(final):
        raise AcceptanceBundleError("Artiverse usage counts are stale")
    def identity(record: Any) -> tuple[str, str, str]:
        if not isinstance(record, dict):
            raise AcceptanceBundleError("Artiverse usage asset record is malformed")
        value = (
            str(record.get("room_id", "")),
            str(record.get("object_id", "")),
            str(record.get("articulated_id", "")),
        )
        if not all(value) or value[0] not in ROOM_SET:
            raise AcceptanceBundleError("Artiverse usage asset identity is incomplete")
        if (
            str(record.get("asset_source", "")).strip().lower() != "articulated"
            or str(record.get("articulated_source", "")).strip().lower()
            != "artiverse"
        ):
            raise AcceptanceBundleError(
                "Artiverse usage asset has forged or incomplete runtime source labels"
            )
        for key in ("sdf_sha256", "sdf_tree_sha256", "source_sdf_sha256", "source_tree_sha256"):
            if not _is_sha256(record.get(key)):
                raise AcceptanceBundleError(f"Artiverse usage has malformed {key}")
        return value
    placed_values = [identity(record) for record in placed]
    final_values = [identity(record) for record in final]
    placed_ids = set(placed_values)
    final_ids = set(final_values)
    if len(placed_ids) != len(placed_values) or len(final_ids) != len(final_values):
        raise AcceptanceBundleError("Artiverse usage contains duplicate asset identities")
    if not placed_ids.issubset(final_ids):
        raise AcceptanceBundleError("Final house lost a placed Artiverse asset")
    _embedded_file_record(
        document.get("house_state"),
        expected=paths["house_state"],
        label="Artiverse usage house state",
        strict_path=True,
    )
    room_records = document.get("room_states")
    if not isinstance(room_records, list) or {
        record.get("room_id") for record in room_records if isinstance(record, dict)
    } != ROOM_SET:
        raise AcceptanceBundleError("Artiverse usage does not bind all 11 room states")
    for record in room_records:
        room_id = str(record["room_id"])
        state = (
            scene_dir
            / f"room_{room_id}"
            / "scene_states"
            / "final_scene"
            / "scene_state.json"
        )
        _embedded_file_record(
            record,
            expected=state,
            label=f"Artiverse usage room state {room_id}",
            strict_path=True,
        )

    def state_artiverse_identities(
        state: Any, room_id: str, *, label: str
    ) -> set[tuple[str, str, str]]:
        if not isinstance(state, dict):
            raise AcceptanceBundleError(f"{label} is not a JSON object")
        objects = state.get("objects")
        if isinstance(objects, dict):
            items = objects.items()
        elif isinstance(objects, list):
            items = enumerate(objects)
        else:
            raise AcceptanceBundleError(f"{label} has no object collection")
        identities: set[tuple[str, str, str]] = set()
        for state_object_id, obj in items:
            if not isinstance(obj, dict):
                continue
            metadata = obj.get("metadata")
            if not isinstance(metadata, dict) or str(
                metadata.get("articulated_source", "")
            ).strip().lower() != "artiverse":
                continue
            value = (
                room_id,
                str(obj.get("object_id") or state_object_id),
                str(metadata.get("articulated_id", "")).strip(),
            )
            if (
                not all(value)
                or str(metadata.get("asset_source", "")).strip().lower()
                != "articulated"
                or metadata.get("is_articulated") is not True
            ):
                raise AcceptanceBundleError(
                    f"{label} has forged or incomplete Artiverse runtime provenance"
                )
            if value in identities:
                raise AcceptanceBundleError(
                    f"{label} has duplicate Artiverse object identity"
                )
            identities.add(value)
        return identities

    actual_placed: set[tuple[str, str, str]] = set()
    for record in room_records:
        room_id = str(record["room_id"])
        state_path = (
            scene_dir
            / f"room_{room_id}"
            / "scene_states"
            / "final_scene"
            / "scene_state.json"
        )
        actual_placed.update(
            state_artiverse_identities(
                _read_json(state_path, label=f"room state {room_id}"),
                room_id,
                label=f"room state {room_id}",
            )
        )
    if actual_placed != placed_ids:
        raise AcceptanceBundleError(
            "Artiverse placed-asset evidence differs from the bound room states"
        )

    house_state = _read_json(paths["house_state"], label="combined house state")
    rooms = house_state.get("rooms")
    if not isinstance(rooms, dict):
        raise AcceptanceBundleError("Combined house state has no room mapping")
    actual_final: set[tuple[str, str, str]] = set()
    for room_id, room_state in rooms.items():
        actual_final.update(
            state_artiverse_identities(
                room_state,
                str(room_id),
                label=f"combined house room {room_id}",
            )
        )
    if actual_final != final_ids:
        raise AcceptanceBundleError(
            "Artiverse final-survivor evidence differs from the combined house state"
        )
    roles = document.get("required_articulated_roles")
    if not isinstance(roles, dict) or roles.get("status") != "pass":
        raise AcceptanceBundleError("Artiverse usage lacks passing articulated-role proof")


def _validate_policy(document: Mapping[str, Any]) -> None:
    expected_sources = {
        "furniture_agent": "generated",
        "wall_agent": "generated",
        "ceiling_agent": "generated",
        "manipuland_agent": "objaverse",
    }
    for agent, source in expected_sources.items():
        record = document.get(agent)
        if not isinstance(record, dict) or record.get("general_asset_source") != source:
            raise AcceptanceBundleError(f"Resolved asset policy is invalid for {agent}")
        if int(record.get("coacd_max_convex_hull", 10**9)) > 32 or int(
            record.get("vhacd_max_convex_hulls", 10**9)
        ) > 32:
            raise AcceptanceBundleError(f"Resolved asset policy exceeds collision cap for {agent}")
        if agent != "manipuland_agent" and record.get("backend") != "sam3d":
            raise AcceptanceBundleError(f"Resolved asset policy does not use SAM3D for {agent}")
    services = document.get("worker_services")
    if not isinstance(services, list) or "hssd" in {str(value).lower() for value in services}:
        raise AcceptanceBundleError("Resolved asset policy service plan is malformed or uses HSSD")
    articulated = document.get("articulated_contract")
    if not isinstance(articulated, dict):
        raise AcceptanceBundleError("Resolved articulated asset policy is missing")
    if articulated.get("articulated_strategy_enabled") is not True or articulated.get(
        "artiverse_strategy_enabled"
    ) is not True:
        raise AcceptanceBundleError("Resolved asset policy does not enable Artiverse routing")
    for source in ("artiverse", "artvip"):
        value = articulated.get(source)
        if not isinstance(value, dict) or not all(
            (
                value.get("enabled") is True,
                value.get("data_path_exists") is True,
                value.get("embeddings_path_exists") is True,
                value.get("missing_embedding_files") == [],
            )
        ):
            raise AcceptanceBundleError(f"Resolved {source} policy is incomplete")
    materials = document.get("materials_contract")
    if not isinstance(materials, dict) or materials.get("status") != "pass":
        raise AcceptanceBundleError("Resolved asset policy lacks materials authority")


def _simulator_tree_inventory(root: Path) -> list[dict[str, Any]]:
    """Inventory the live simulator tree without following link-like entries."""

    export_root = _require_directory(root, label="simulator export root")
    records: list[dict[str, Any]] = []
    for root_text, directory_names, file_names in os.walk(
        export_root, followlinks=False
    ):
        current = Path(root_text)
        if not _within(current.resolve(strict=True), export_root):
            raise AcceptanceBundleError(
                f"Simulator export directory escapes its root: {current}"
            )
        for name in directory_names:
            directory = current / name
            if _is_link_like(directory):
                raise AcceptanceBundleError(
                    f"Simulator export contains a link-like directory: {directory}"
                )
            try:
                info = directory.lstat()
                resolved = directory.resolve(strict=True)
            except OSError as exc:
                raise AcceptanceBundleError(
                    f"Cannot inspect simulator export directory: {directory}"
                ) from exc
            if not stat.S_ISDIR(info.st_mode):
                raise AcceptanceBundleError(
                    f"Simulator export contains a special directory entry: {directory}"
                )
            if not _within(resolved, export_root):
                raise AcceptanceBundleError(
                    f"Simulator export directory escapes its root: {directory}"
                )
        for name in file_names:
            path = current / name
            if _is_link_like(path):
                raise AcceptanceBundleError(
                    f"Simulator export contains a link-like file: {path}"
                )
            try:
                info = path.lstat()
                resolved = path.resolve(strict=True)
            except OSError as exc:
                raise AcceptanceBundleError(
                    f"Cannot inspect simulator export file: {path}"
                ) from exc
            if not stat.S_ISREG(info.st_mode):
                raise AcceptanceBundleError(
                    f"Simulator export contains a special file: {path}"
                )
            if info.st_size < 1:
                raise AcceptanceBundleError(
                    f"Simulator export contains an empty file: {path}"
                )
            if not _within(resolved, export_root):
                raise AcceptanceBundleError(
                    f"Simulator export file escapes its root: {path}"
                )
            records.append(
                {
                    "path": resolved.relative_to(export_root).as_posix(),
                    "size": info.st_size,
                    "sha256": _sha256_file(resolved),
                }
            )
    if not records:
        raise AcceptanceBundleError("Simulator export tree is empty")
    return sorted(records, key=lambda value: str(value["path"]))


def _simulator_relative_file(
    root: Path,
    value: Any,
    *,
    label: str,
) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise AcceptanceBundleError(f"{label} has no relative path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "\\" in value
        or relative.as_posix() != value
    ):
        raise AcceptanceBundleError(f"{label} path is unsafe: {value!r}")
    lexical = _absolute_lexical(root.joinpath(*relative.parts))
    if not _within(lexical, root):
        raise AcceptanceBundleError(f"{label} path escapes simulator root: {value!r}")
    path = _require_regular_file(
        lexical,
        label=label,
        containment_root=root,
    )
    return value, path


def _simulator_record(
    root: Path,
    value: Any,
    *,
    label: str,
) -> tuple[str, Path, dict[str, Any]]:
    if not isinstance(value, dict):
        raise AcceptanceBundleError(f"{label} record is malformed")
    if set(value) != {"path", "size", "sha256"}:
        raise AcceptanceBundleError(f"{label} record fields are malformed")
    relative, path = _simulator_relative_file(root, value.get("path"), label=label)
    expected = {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if value != expected:
        raise AcceptanceBundleError(f"{label} record is stale")
    return relative, path, expected


def _validate_simulator_exports(
    document: Mapping[str, Any],
    *,
    context: BundleContext,
) -> None:
    """Bind the simulator report to the exact, current published export tree."""

    _require_passing_json(document, label="simulator_exports")
    if document.get("schema_version") != 2 or document.get("require_usd") is not True:
        raise AcceptanceBundleError("Simulator export report does not require USD")

    expected_root = _require_directory(
        context.scene_dir / "mujoco_export",
        label="published simulator export",
    )
    reported_root = document.get("output_dir")
    if not isinstance(reported_root, str) or not Path(reported_root).is_absolute():
        raise AcceptanceBundleError(
            "Simulator export report has no absolute published output root"
        )
    if _absolute_lexical(Path(reported_root)) != expected_root:
        raise AcceptanceBundleError(
            "Simulator export report points to a different output root"
        )

    live_inventory = _simulator_tree_inventory(expected_root)
    report_inventory = document.get("file_inventory")
    if (
        not isinstance(report_inventory, list)
        or not report_inventory
        or document.get("file_count") != len(report_inventory)
    ):
        raise AcceptanceBundleError("Simulator export inventory is malformed")
    checked_inventory = [
        _simulator_record(
            expected_root,
            value,
            label=f"Simulator inventory[{index}]",
        )[2]
        for index, value in enumerate(report_inventory)
    ]
    if checked_inventory != live_inventory:
        raise AcceptanceBundleError(
            "Simulator export live file set, sizes, or hashes differ from its report"
        )
    inventory_digest = _sha256_bytes(_canonical_json(live_inventory))
    if document.get("inventory_sha256") != inventory_digest:
        raise AcceptanceBundleError("Simulator export inventory attestation is stale")

    marker = document.get("attempt_marker")
    if not isinstance(marker, dict) or set(marker) != {
        "path",
        "sha256",
        "run_attempt_id",
    }:
        raise AcceptanceBundleError("Simulator attempt-marker mapping is malformed")
    marker_relative, marker_path = _simulator_relative_file(
        expected_root,
        marker.get("path"),
        label="Simulator attempt marker",
    )
    if marker_relative != ".export_attempt.json":
        raise AcceptanceBundleError(
            "Simulator report maps the attempt marker to a noncanonical file"
        )
    if (
        marker.get("run_attempt_id") != context.run_attempt_id
        or marker.get("sha256") != _sha256_file(marker_path)
    ):
        raise AcceptanceBundleError(
            "Simulator export is not bound to the current run attempt"
        )
    marker_document = _read_json(marker_path, label="simulator attempt marker")
    created_at = marker_document.get("created_at")
    try:
        created_time = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
    except ValueError as exc:
        raise AcceptanceBundleError(
            "Published simulator attempt marker has an invalid creation time"
        ) from exc
    if (
        set(marker_document) != {"schema_version", "run_attempt_id", "created_at"}
        or marker_document.get("schema_version") != 1
        or marker_document.get("run_attempt_id") != context.run_attempt_id
        or created_time.tzinfo is None
    ):
        raise AcceptanceBundleError(
            "Published simulator attempt marker has stale or unexpected contents"
        )

    publication = document.get("publication")
    if (
        not isinstance(publication, dict)
        or publication.get("run_attempt_id") != context.run_attempt_id
        or publication.get("atomic_promotion") is not True
        or publication.get("previous_backup_cleanup")
        not in {"removed", "not_applicable"}
        or publication.get("staging_inventory_sha256") != inventory_digest
    ):
        raise AcceptanceBundleError(
            "Simulator export lacks clean atomic-publication evidence"
        )
    published_dir = publication.get("published_dir")
    if (
        not isinstance(published_dir, str)
        or not Path(published_dir).is_absolute()
        or _absolute_lexical(Path(published_dir)) != expected_root
    ):
        raise AcceptanceBundleError(
            "Simulator publication report points to a different output root"
        )
    exporter_contract = publication.get("exporter_contract")
    expected_exporter = _require_regular_file(
        context.repo_dir / "scripts" / "export_scene_to_mujoco.py",
        label="canonical simulator exporter",
        containment_root=context.repo_dir,
    )
    if not isinstance(exporter_contract, dict) or exporter_contract != {
        "path": str(expected_exporter),
        "sha256": _sha256_file(expected_exporter),
        "usd_failure_contract": "pass",
    }:
        raise AcceptanceBundleError(
            "Simulator exporter-contract path or hash is not canonical"
        )

    mujoco = document.get("mujoco")
    if not isinstance(mujoco, dict):
        raise AcceptanceBundleError("Simulator export has no MuJoCo proof")
    scene_relative, scene_xml = _simulator_relative_file(
        expected_root,
        mujoco.get("scene_xml"),
        label="Simulator MJCF root",
    )
    if scene_relative != "scene.xml":
        raise AcceptanceBundleError("Simulator report maps MJCF to a noncanonical root")
    if mujoco.get("scene_xml_sha256") != _sha256_file(scene_xml):
        raise AcceptanceBundleError("Simulator MJCF root hash is stale")
    try:
        mjcf_root = ET.parse(scene_xml).getroot()
    except (OSError, ET.ParseError) as exc:
        raise AcceptanceBundleError(f"Published MJCF cannot be parsed: {exc}") from exc
    model_name = (mjcf_root.get("model") or "").strip()
    if (
        not model_name
        or PurePosixPath(model_name).name != model_name
        or "\\" in model_name
        or model_name in {".", ".."}
    ):
        raise AcceptanceBundleError(
            "Published MJCF has no safe model name for the USD root"
        )

    supported_file_elements = {"mesh", "texture"}
    for element in mjcf_root.iter():
        if element.get("file") and element.tag.rsplit("}", 1)[-1] not in supported_file_elements:
            raise AcceptanceBundleError(
                "Published MJCF contains an unsupported external file mapping"
            )
    compiler = mjcf_root.find("compiler")
    mesh_dir = compiler.get("meshdir", "") if compiler is not None else ""
    texture_dir = (
        compiler.get("texturedir", mesh_dir) if compiler is not None else mesh_dir
    )
    referenced: dict[str, dict[str, Any]] = {}
    for selector, directory in (
        (".//asset/mesh", mesh_dir),
        (".//asset/texture", texture_dir),
    ):
        for element in mjcf_root.findall(selector):
            file_value = element.get("file")
            if not file_value:
                continue
            combined = PurePosixPath(directory) / PurePosixPath(file_value)
            relative, path = _simulator_relative_file(
                expected_root,
                combined.as_posix(),
                label="Referenced simulator MJCF asset",
            )
            referenced[relative] = {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    expected_references = [referenced[path] for path in sorted(referenced)]
    if (
        mujoco.get("referenced_asset_count") != len(expected_references)
        or mujoco.get("referenced_assets") != expected_references
    ):
        raise AcceptanceBundleError(
            "Simulator MJCF referenced-asset mappings are stale or noncanonical"
        )
    model_counts = mujoco.get("model_counts")
    nbody = model_counts.get("nbody") if isinstance(model_counts, dict) else None
    if (
        not isinstance(model_counts, dict)
        or not model_counts
        or not isinstance(nbody, int)
        or isinstance(nbody, bool)
        or nbody < 1
    ):
        raise AcceptanceBundleError("Simulator export has no MuJoCo load/step proof")

    usd = document.get("usd")
    if not isinstance(usd, dict) or usd.get("candidate_failures") != []:
        raise AcceptanceBundleError("Simulator export has no passing USD proof")
    if usd.get("usd_dir") != "usd":
        raise AcceptanceBundleError("Simulator report maps USD to a noncanonical root")
    usd_root = _require_directory(expected_root / "usd", label="simulator USD root")
    if usd_root.parent != expected_root:
        raise AcceptanceBundleError("Simulator USD root escapes the export root")

    usd_suffixes = {".usd", ".usda", ".usdc"}
    inventory_by_path = {str(value["path"]): value for value in live_inventory}
    actual_layers = {
        path: value
        for path, value in inventory_by_path.items()
        if PurePosixPath(path).parts[:1] == ("usd",)
        and PurePosixPath(path).suffix.lower() in usd_suffixes
    }
    if not actual_layers or usd.get("usd_layer_count") != len(actual_layers):
        raise AcceptanceBundleError("Simulator USD layer accounting is stale")
    top_level = sorted(
        path for path in actual_layers if len(PurePosixPath(path).parts) == 2
    )
    expected_root_layer = f"usd/{model_name}.usda"
    if top_level != [expected_root_layer]:
        raise AcceptanceBundleError(
            "Simulator USD export does not have the exact MJCF-named root layer"
        )

    expected_layer_paths = {
        expected_root_layer,
        "usd/Payload/Contents.usda",
        "usd/Payload/Geometry.usda",
        "usd/Payload/Physics.usda",
    }
    if mjcf_root.findall(".//asset/mesh"):
        expected_layer_paths.add("usd/Payload/GeometryLibrary.usdc")
    non_grid_materials = [
        material
        for material in mjcf_root.findall(".//asset/material")
        if material.get("name") != "grid"
    ]
    if non_grid_materials:
        expected_layer_paths.update(
            {
                "usd/Payload/Materials.usda",
                "usd/Payload/MaterialsLibrary.usdc",
            }
        )
    if not expected_layer_paths.issubset(actual_layers):
        missing = sorted(expected_layer_paths - set(actual_layers))
        raise AcceptanceBundleError(
            f"Simulator export is missing expected USD layers: {missing}"
        )
    expected_artifacts = [
        actual_layers[path] for path in sorted(expected_layer_paths)
    ]
    if usd.get("expected_artifacts") != expected_artifacts:
        raise AcceptanceBundleError(
            "Simulator expected-USD-layer mappings are stale or noncanonical"
        )

    stages = usd.get("validated_stages")
    if not isinstance(stages, list) or len(stages) != len(actual_layers):
        raise AcceptanceBundleError("Simulator validated-stage accounting is stale")
    stage_paths: set[str] = set()
    root_used_layers: set[str] = set()
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict) or set(stage) != {
            "path",
            "sha256",
            "prim_count",
            "used_layers",
        }:
            raise AcceptanceBundleError(
                f"Simulator validated stage[{index}] is malformed"
            )
        relative, path = _simulator_relative_file(
            expected_root,
            stage.get("path"),
            label=f"Simulator validated stage[{index}]",
        )
        if relative not in actual_layers or relative in stage_paths:
            raise AcceptanceBundleError(
                "Simulator validated-stage paths are duplicate or not USD layers"
            )
        stage_paths.add(relative)
        if stage.get("sha256") != _sha256_file(path):
            raise AcceptanceBundleError("Simulator validated-stage hash is stale")
        prim_count = stage.get("prim_count")
        if not isinstance(prim_count, int) or isinstance(prim_count, bool) or prim_count < 1:
            raise AcceptanceBundleError("Simulator validated stage has no prim proof")
        used = stage.get("used_layers")
        if not isinstance(used, list) or any(not isinstance(item, str) for item in used):
            raise AcceptanceBundleError("Simulator used-layer mapping is malformed")
        if len(used) != len(set(used)):
            raise AcceptanceBundleError("Simulator used-layer mapping has duplicates")
        for used_path in used:
            mapped, _ = _simulator_relative_file(
                expected_root,
                used_path,
                label=f"Simulator used layer for {relative}",
            )
            if mapped not in actual_layers:
                raise AcceptanceBundleError(
                    "Simulator used-layer mapping does not name a current USD layer"
                )
        if relative == expected_root_layer:
            root_used_layers.update(used)
    if stage_paths != set(actual_layers):
        raise AcceptanceBundleError(
            "Simulator report does not validate the exact current USD layer set"
        )
    required_payloads = expected_layer_paths - {expected_root_layer}
    if not required_payloads.issubset(root_used_layers):
        raise AcceptanceBundleError(
            "Simulator USD root does not bind every expected payload layer"
        )


def _validate_document(
    label: str,
    document: Mapping[str, Any],
    *,
    paths: Mapping[str, Path],
    context: BundleContext,
) -> None:
    if label == "input:input_manifest.json":
        return
    if label == "input_manifest_validation":
        _require_passing_json(document, label=label)
        if document.get("expected_room_ids") != list(ROOM_IDS):
            raise AcceptanceBundleError("Input validation has a stale room set")
        if document.get("pipeline_contract", {}).get("required_articulated_source") != "artiverse":
            raise AcceptanceBundleError("Input validation does not require Artiverse")
        _empty_string_list(document.get("critical_issues"), label=f"{label}.critical_issues")
        return
    if label == "pipeline_code_contract":
        _require_passing_json(document, label=label)
        if (
            document.get("schema_version")
            != PIPELINE_CODE_CONTRACT_SCHEMA_VERSION
            or document.get("contract")
            != "scenesmith-full-quality-pipeline-code"
        ):
            raise AcceptanceBundleError("Pipeline/code contract schema is unsupported")
        specification = document.get("specification")
        if not isinstance(specification, dict) or specification.get(
            "read_before_generation_code"
        ) is not True:
            raise AcceptanceBundleError("Pipeline/code contract lacks read-before-code proof")
        if not isinstance(document.get("artifacts"), list) or not document["artifacts"]:
            raise AcceptanceBundleError("Pipeline/code contract inventory is empty")
        _verify_pipeline_attestation(document)
        return
    if label == "sam3d_offline_preflight":
        _require_passing_json(document, label=label)
        if (
            document.get("schema_version")
            != sam3d_preflight.RESULT_SCHEMA_VERSION
            or document.get("offline") is not True
        ):
            raise AcceptanceBundleError("SAM3D preflight schema/offline proof is invalid")
        if document.get("model_loaded") is not True or document.get("pipeline_loaded") is not True:
            raise AcceptanceBundleError("SAM3D preflight did not load both models")
        if not isinstance(document.get("visible_gpu_count"), int) or document.get("visible_gpu_count", 0) < 1:
            raise AcceptanceBundleError("SAM3D preflight has no GPU load proof")
        if not isinstance(document.get("evidence"), dict) or document.get(
            "evidence_verification", {}
        ).get("status") != "pass":
            raise AcceptanceBundleError("SAM3D artifact evidence is missing or stale")
        inference_failures = sam3d_preflight.verify_inference_smoke(
            document.get("inference_smoke")
        )
        if inference_failures:
            raise AcceptanceBundleError(
                "SAM3D image/text inference proof is invalid: "
                + "; ".join(inference_failures)
            )
        _require_attestation(
            document,
            label=label,
            expected_digest=_sam3d_attestation(document),
        )
        return
    if label == "sam3d_generation_preflight":
        generation_paths = {
            key: paths[key]
            for key in (
                "sam3d_generation_glb",
                "sam3d_generation_mask",
                "sam3d_generation_masked_image",
            )
        }
        parents = {path.parent for path in generation_paths.values()}
        if len(parents) != 1:
            raise AcceptanceBundleError(
                "SAM3D generation artifacts do not share one canonical directory"
            )
        failures = sam3d_generation.verify_bound_receipt(
            document,
            input_path=paths["sam3d_generation_input"],
            artifact_dir=next(iter(parents)),
        )
        if failures:
            raise AcceptanceBundleError(
                "SAM3D full offline generation proof is invalid: "
                + "; ".join(failures)
            )
        return
    if label == "objathor_retrieval_preflight":
        _require_passing_json(document, label=label)
        if document.get("schema_version") != 1 or document.get("offline") is not True:
            raise AcceptanceBundleError("ObjectThor preflight schema/offline proof is invalid")
        for key in ("dataset", "model", "model_smoke", "evidence"):
            if not isinstance(document.get(key), dict) or not document[key]:
                raise AcceptanceBundleError(f"ObjectThor preflight has no {key} proof")
        if document.get("verification", {}).get("status") != "pass":
            raise AcceptanceBundleError("ObjectThor cache-only repeat verification did not pass")
        _require_attestation(
            document,
            label=label,
            expected_digest=_objathor_attestation(document),
        )
        return
    if label == "vlm_vision_smoke":
        _require_passing_json(document, label=label)
        if document.get("backend") not in {"openai", "codex"}:
            raise AcceptanceBundleError("VLM smoke backend is unsupported")
        if document.get("image_sha256") != _sha256_file(paths["input:reference.png"]):
            raise AcceptanceBundleError("VLM smoke is bound to a different reference image")
        assessment = document.get("assessment")
        if not isinstance(assessment, dict) or assessment.get("critical_issues") != []:
            raise AcceptanceBundleError("VLM smoke assessment is malformed")
        score = assessment.get("scores", {}).get("image_readability")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or score < 1:
            raise AcceptanceBundleError("VLM smoke did not prove image readability")
        return
    if label == "materials_contract":
        _require_passing_json(document, label=label)
        if document.get("schema_id") != "scenesmith_materials_contract_v1":
            raise AcceptanceBundleError("Materials contract schema is unsupported")
        source_count = document.get("source_count")
        retained = document.get("retained_count")
        pruned = document.get("pruned_count")
        if not all(isinstance(value, int) for value in (source_count, retained, pruned)):
            raise AcceptanceBundleError("Materials contract counts are malformed")
        if retained < 1900 or pruned > 15 or source_count != retained + pruned:
            raise AcceptanceBundleError("Materials contract counts violate the quality profile")
        for key in ("asset_inventory_sha256", "manifest_sha256"):
            if not _is_sha256(document.get(key)):
                raise AcceptanceBundleError(f"Materials contract has malformed {key}")
        return
    if label == "artiverse_preparation":
        _require_passing_json(document, label=label)
        authority = document.get("authority")
        if not isinstance(authority, dict):
            raise AcceptanceBundleError("Artiverse preparation authority is missing")
        if authority.get("source_repository") != "3dlg-hcvc/artiverse" or authority.get(
            "source_revision"
        ) != "8c4b120418e7cbdf9ac4c9580c5dbfdbf128a248":
            raise AcceptanceBundleError("Artiverse preparation is not pinned to official bytes")
        if not isinstance(authority.get("indexed_count"), int) or authority.get("indexed_count", 0) < 1:
            raise AcceptanceBundleError("Artiverse preparation has no indexed assets")
        if not isinstance(authority.get("index_sha256"), dict) or not authority["index_sha256"]:
            raise AcceptanceBundleError("Artiverse preparation index hashes are missing")
        return
    if label == "artiverse_visual_resources":
        try:
            artiverse_visual._validate_saved_receipt(document)
        except artiverse_visual.ArtiverseVisualResourceError as exc:
            raise AcceptanceBundleError(
                f"Artiverse visual-resource receipt is invalid: {exc}"
            ) from exc
        expected_policy = artiverse_visual.mapping_policy()
        if document.get("mapping_policy") != expected_policy:
            raise AcceptanceBundleError(
                "Artiverse visual-resource receipt changed its exact mapping policy"
            )
        audit = document.get("audit")
        if not isinstance(audit, dict):
            raise AcceptanceBundleError(
                "Artiverse visual-resource receipt has no complete audit"
            )

        def required_audit_int(name: str, *, minimum: int) -> int:
            value = audit.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise AcceptanceBundleError(
                    f"Artiverse visual-resource receipt has invalid {name}"
                )
            return value

        indexed = required_audit_int("indexed_asset_count", minimum=500)
        audited = required_audit_int("audited_asset_count", minimum=500)
        if indexed != audited:
            raise AcceptanceBundleError(
                "Artiverse visual-resource receipt did not audit every indexed asset"
            )
        for name in (
            "mapped_link_count",
            "visual_mapping_count",
            "unique_glb_count",
            "glb_primitive_count",
            "derived_gltf_count",
            "derived_bin_count",
            "gltf_buffer_view_count",
            "gltf_accessor_count",
            "gltf_material_count",
            "gltf_texture_count",
            "gltf_image_count",
            "collision_binding_count",
        ):
            required_audit_int(name, minimum=1)
        for name in (
            "mapping_inventory_sha256",
            "collision_inventory_sha256",
            "asset_inventory_sha256",
        ):
            if not _is_sha256(audit.get(name)):
                raise AcceptanceBundleError(
                    f"Artiverse visual-resource receipt has invalid {name}"
                )
        assets = audit.get("assets")
        if not isinstance(assets, list) or len(assets) != audited:
            raise AcceptanceBundleError(
                "Artiverse visual-resource receipt lacks every per-asset derivation"
            )
        asset_ids: set[str] = set()
        summed = {
            name: 0
            for name in (
                "mapped_link_count",
                "visual_mapping_count",
                "unique_glb_count",
                "glb_primitive_count",
                "derived_gltf_count",
                "derived_bin_count",
                "gltf_buffer_view_count",
                "gltf_accessor_count",
                "gltf_material_count",
                "gltf_texture_count",
                "gltf_image_count",
                "collision_binding_count",
            )
        }
        for item in assets:
            if not isinstance(item, dict):
                raise AcceptanceBundleError(
                    "Artiverse visual-resource per-asset derivation is malformed"
                )
            object_id = item.get("object_id")
            if (
                not isinstance(object_id, str)
                or not object_id
                or object_id in asset_ids
            ):
                raise AcceptanceBundleError(
                    "Artiverse visual-resource per-asset identities are invalid"
                )
            asset_ids.add(object_id)
            for name in summed:
                value = item.get(name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise AcceptanceBundleError(
                        f"Artiverse visual-resource asset has invalid {name}"
                    )
                summed[name] += value
            for name in (
                "source_sdf_sha256",
                "source_tree_sha256",
                "derived_resource_inventory_sha256",
                "mapping_sha256",
                "collision_binding_sha256",
            ):
                if not _is_sha256(item.get(name)):
                    raise AcceptanceBundleError(
                        f"Artiverse visual-resource asset has invalid {name}"
                    )
        if any(summed[name] != audit.get(name) for name in summed):
            raise AcceptanceBundleError(
                "Artiverse visual-resource per-asset derivation totals are stale"
            )
        if (
            audit.get("source_files_written") != 0
            or audit.get("mapping_target")
            != "publisher_glb_derived_external_gltf"
            or audit.get("publisher_glb_json_bin_resources_validated") is not True
            or audit.get("publisher_position_normal_accessors_validated") is not True
            or audit.get("publisher_material_image_references_validated") is not True
            or audit.get("derived_external_gltf_hashes_precomputed") is not True
            or audit.get("source_tree_identity_revalidated_after_audit") is not True
            or audit.get("source_hash_revalidated_after_each_asset_audit") is not True
        ):
            raise AcceptanceBundleError(
                "Artiverse visual-resource receipt does not prove zero source mutation"
            )
        if document.get("runtime_code_revalidated_after_audit") is not True:
            raise AcceptanceBundleError(
                "Artiverse visual-resource runtime code was not revalidated"
            )
        runtime_code = document.get("runtime_code")
        if not isinstance(runtime_code, list):
            raise AcceptanceBundleError(
                "Artiverse visual-resource receipt has no runtime code inventory"
            )
        runtime_paths = {
            str(item.get("path", "")).replace("\\", "/")
            for item in runtime_code
            if isinstance(item, dict)
            and isinstance(item.get("size_bytes"), int)
            and not isinstance(item.get("size_bytes"), bool)
            and item.get("size_bytes", 0) > 0
            and _is_sha256(item.get("sha256"))
        }
        required_suffixes = {
            "scenesmith/agent_utils/artiverse_visual_normalization.py",
            "scenesmith/agent_utils/asset_manager.py",
        }
        missing = sorted(
            suffix
            for suffix in required_suffixes
            if not any(path.endswith(suffix) for path in runtime_paths)
        )
        if missing:
            raise AcceptanceBundleError(
                "Artiverse visual-resource receipt lacks required runtime code: "
                + ", ".join(missing)
            )
        return
    if label == "articulated_router":
        _require_passing_json(document, label=label)
        _empty_string_list(document.get("failures"), label="articulated router failures")
        datasets = document.get("datasets")
        if not isinstance(datasets, dict) or set(datasets) != {"artiverse", "artvip"}:
            raise AcceptanceBundleError("Articulated router source set is malformed")
        for source, value in datasets.items():
            if not isinstance(value, dict) or not all(
                (
                    value.get("data_path_exists") is True,
                    value.get("embeddings_path_exists") is True,
                    value.get("missing_embedding_files") == [],
                )
            ):
                raise AcceptanceBundleError(f"Articulated router {source} source failed")
        selected = document.get("selected_asset_identifiers")
        if not isinstance(selected, dict) or any(not selected.get(name) for name in datasets):
            raise AcceptanceBundleError("Articulated router selected no concrete source assets")
        prompts = document.get("prompts")
        if not isinstance(prompts, list) or not prompts:
            raise AcceptanceBundleError("Articulated router has no prompt evidence")
        if any(
            item.get("passes_artiverse_router_strategy_check") is not True
            or item.get("passes_artvip_router_strategy_check") is not True
            or item.get("passes_artiverse_candidate_check") is not True
            or item.get("passes_artvip_candidate_check") is not True
            for item in prompts
            if isinstance(item, dict)
        ):
            raise AcceptanceBundleError("Articulated router contains a failed strategy/candidate check")
        return
    if label == "resolved_asset_policy":
        _validate_policy(document)
        return
    if label == "reference_school_layout_seed":
        _require_passing_json(document, label=label)
        try:
            from scripts import seed_reference_school_layout as reference_layout
        except ImportError:
            import seed_reference_school_layout as reference_layout  # type: ignore[no-redef]

        if (
            document.get("schema_version") != reference_layout.SEED_SCHEMA_VERSION
            or document.get("profile") != reference_layout.PROFILE
            or document.get("implementation")
            != "native_scenesmith_deterministic_reference_layout"
        ):
            raise AcceptanceBundleError(
                "Reference-school native layout seed schema/profile is unsupported"
            )
        if document.get("seed_spec") != reference_layout._seed_spec_evidence():
            raise AcceptanceBundleError(
                "Reference-school native layout seed does not bind the exact seed specification"
            )

        layout_document = _read_json(paths["house_layout"], label="house layout")
        if document.get(
            "structural_layout_sha256"
        ) != reference_layout._structural_layout_sha256(layout_document):
            raise AcceptanceBundleError(
                "House layout differs structurally from the native reference-school seed"
            )
        try:
            native_geometry = reference_layout._tree_manifest(
                context.scene_dir,
                ("floor_plans", "room_geometry", "package.xml"),
            )
        except reference_layout.SeedError as exc:
            raise AcceptanceBundleError(
                f"Reference-school native geometry is invalid: {exc}"
            ) from exc
        if (
            document.get("artifacts") != native_geometry
            or document.get("artifact_count") != len(native_geometry)
            or document.get("artifact_manifest_sha256")
            != reference_layout._manifest_sha256(native_geometry)
        ):
            raise AcceptanceBundleError(
                "Reference-school native geometry differs from its seed manifest"
            )
        return
    if label == "floor_plan_layout_gate":
        _require_passing_json(document, label=label)
        if document.get("required_room_count") != 11 or document.get("actual_unique_room_count") != 11:
            raise AcceptanceBundleError("Floor-plan layout gate does not prove 11 unique rooms")
        for key in ("missing_room_ids", "unexpected_room_ids", "critical_issues"):
            _empty_string_list(document.get(key), label=f"floor layout {key}")
        evidence = document.get("evidence")
        if not isinstance(evidence, dict) or not evidence.get("entrance_door_ids"):
            raise AcceptanceBundleError("Floor-plan layout gate has no entrance evidence")
        return
    if label == "room_prompt_binding":
        _require_passing_json(document, label=label)
        if document.get("schema_version") != 1 or document.get("profile") != "school_reference_20260710":
            raise AcceptanceBundleError("Room prompt binding profile is unsupported")
        if set(document.get("room_prompt_sha256", {})) != ROOM_SET:
            raise AcceptanceBundleError("Room prompt binding does not cover exactly 11 rooms")
        if document.get("effective_prompt_sha256") != _sha256_bytes(
            _normalized_text(paths["input:prompt.txt"]).encode("utf-8")
        ):
            raise AcceptanceBundleError("Room prompt binding is tied to a different prompt")
        _embedded_file_record(
            document.get("layout"),
            expected=paths["house_layout"],
            label="Room prompt binding layout",
            strict_path=True,
        )
        _embedded_file_record(
            document.get("input_manifest"),
            expected=paths["input:input_manifest.json"],
            label="Room prompt binding input manifest",
            strict_path=False,
        )
        return
    if label == "classroom_variation":
        _validate_classroom_variation(document, scene_dir=context.scene_dir)
        return
    if label == "articulated_motion":
        _validate_articulated_motion(document, scene_dir=context.scene_dir)
        return
    if label == "articulated_motion_verification":
        _validate_articulated_motion_verification(
            document,
            paths=paths,
            scene_dir=context.scene_dir,
        )
        return
    if label == "school_navigation":
        _validate_school_navigation(
            document,
            paths=paths,
            scene_dir=context.scene_dir,
        )
        return
    if label == "school_navigation_verification":
        _validate_school_navigation_verification(
            document,
            paths=paths,
            scene_dir=context.scene_dir,
        )
        return
    if label in {"room_deterministic_summary", "room_visual_summary"}:
        _require_passing_json(document, label=label)
        if document.get("room_count") != 11 or set(document.get("passed_rooms", [])) != ROOM_SET:
            raise AcceptanceBundleError(f"{label} does not pass the exact 11-room set")
        _empty_string_list(document.get("failed_rooms"), label=f"{label}.failed_rooms")
        return
    if label.startswith("room_deterministic:"):
        room_id = label.split(":", 1)[1]
        _require_passing_json(document, label=label)
        if document.get("room_id") != room_id or document.get("contract_profile") != "school_reference_20260710":
            raise AcceptanceBundleError(f"{label} identity/profile is invalid")
        _empty_string_list(document.get("critical_issues"), label=f"{label}.critical_issues")
        _require_scores(document, label=label)
        semantic = document.get("metrics", {}).get("semantic_inventory")
        if not isinstance(semantic, dict) or semantic.get("status") != "pass" or semantic.get("critical_issues") != []:
            raise AcceptanceBundleError(f"{label} has no passing semantic inventory")
        return
    if label.startswith("room_cutaway:"):
        _validate_cutaway(
            document,
            room_id=label.split(":", 1)[1],
            scene_dir=context.scene_dir,
        )
        return
    if label.startswith("room_visual:"):
        room_id = label.split(":", 1)[1]
        _require_passing_json(document, label=label)
        if document.get("room_id") != room_id or document.get("contract_profile") != "school_reference_20260710":
            raise AcceptanceBundleError(f"{label} identity/profile is invalid")
        if document.get("deterministic_gate_status") != "pass":
            raise AcceptanceBundleError(f"{label} is not based on a passing deterministic gate")
        _empty_string_list(document.get("critical_issues"), label=f"{label}.critical_issues")
        _require_scores(document, label=label)
        assessment = document.get("visual_assessment")
        if not isinstance(assessment, dict) or assessment.get("critical_issues") != []:
            raise AcceptanceBundleError(f"{label} visual assessment is malformed")
        assessment_scores = assessment.get("scores")
        if (
            not isinstance(assessment_scores, dict)
            or set(assessment_scores) != set(ROOM_SCORE_KEYS)
            or any(
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not 7.0 <= float(score) <= 10.0
                for score in assessment_scores.values()
            )
        ):
            raise AcceptanceBundleError(
                f"{label} visual assessment scores are not the exact passing contract"
            )
        requirements = assessment.get("requirement_evidence")
        try:
            from scripts.room_visual_self_exam import (
                SCHOOL_VLM_BACKEND,
                SCHOOL_VLM_MODEL,
                _room_judge_instruction,
                _room_messages,
                _vlm_request_contract as room_vlm_request_contract,
                room_visual_requirements,
            )
        except ImportError:
            from room_visual_self_exam import (  # type: ignore[no-redef]
                SCHOOL_VLM_BACKEND,
                SCHOOL_VLM_MODEL,
                _room_judge_instruction,
                _room_messages,
                _vlm_request_contract as room_vlm_request_contract,
                room_visual_requirements,
            )
        expected_requirement_keys = set(room_visual_requirements(room_id))
        if (
            not isinstance(requirements, dict)
            or set(requirements) != expected_requirement_keys
            or any(
            not isinstance(value, dict)
            or value.get("status") != "pass"
            or not isinstance(value.get("view_indices"), list)
            or not value.get("view_indices")
            or value["view_indices"] != sorted(set(value["view_indices"]))
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index not in {1, 2, 3}
                for index in value["view_indices"]
            )
            or not isinstance(value.get("observation"), str)
            or len(value["observation"].strip()) < 8
            for value in requirements.values()
            )
        ):
            raise AcceptanceBundleError(f"{label} lacks itemized passing visual evidence")
        evidence = document.get("evidence")
        if not isinstance(evidence, dict) or evidence.get("schema_version") != 1:
            raise AcceptanceBundleError(f"{label} immutable evidence is missing")
        state = context.scene_dir / f"room_{room_id}" / "scene_states" / "final_scene" / "scene_state.json"
        blend = context.scene_dir / f"room_{room_id}" / "scene_states" / "final_scene" / "scene.blend"
        _embedded_file_record(evidence.get("scene_state"), expected=state, label=f"{label} state", strict_path=True)
        _embedded_file_record(evidence.get("source_blend"), expected=blend, label=f"{label} blend", strict_path=True)
        _embedded_file_record(evidence.get("house_layout"), expected=paths["house_layout"], label=f"{label} layout", strict_path=True)
        _embedded_file_record(evidence.get("reference_image"), expected=paths["input:reference.png"], label=f"{label} reference", strict_path=False)
        _embedded_file_record(evidence.get("effective_prompt"), expected=paths["input:prompt.txt"], label=f"{label} prompt", strict_path=False)
        _embedded_file_record(evidence.get("input_manifest"), expected=paths["input:input_manifest.json"], label=f"{label} manifest", strict_path=False)
        _embedded_file_record(evidence.get("prompt_binding"), expected=paths["room_prompt_binding"], label=f"{label} prompt binding", strict_path=True)
        _embedded_file_record(evidence.get("deterministic_gate"), expected=paths[f"room_deterministic:{room_id}"], label=f"{label} deterministic gate", strict_path=True)
        _embedded_file_record(evidence.get("cutaway_evidence"), expected=paths[f"room_cutaway:{room_id}"], label=f"{label} cutaway", strict_path=True)
        reviews = evidence.get("review_images")
        if not isinstance(reviews, list) or len(reviews) != 3:
            raise AcceptanceBundleError(f"{label} does not bind exactly three reviews")
        for index, name in enumerate(("top", "oblique_a", "oblique_b")):
            _embedded_file_record(
                reviews[index],
                expected=context.scene_dir / "review" / "room_review_renders" / f"{room_id}_{name}.png",
                label=f"{label} review {name}",
                strict_path=True,
            )
        reference_record = evidence.get("reference_image")
        review_hashes = [str(record.get("sha256")) for record in reviews]
        if len(set(review_hashes)) != 3:
            raise AcceptanceBundleError(f"{label} review image hashes are not distinct")
        deterministic_document = _read_json(
            paths[f"room_deterministic:{room_id}"],
            label=f"{label} deterministic gate",
        )
        room_prompt = document.get("room_prompt")
        if not isinstance(room_prompt, str) or not room_prompt:
            raise AcceptanceBundleError(f"{label} has no bound generated room prompt")
        review_paths = [
            context.scene_dir
            / "review"
            / "room_review_renders"
            / f"{room_id}_{name}.png"
            for name in ("top", "oblique_a", "oblique_b")
        ]
        instruction = _room_judge_instruction(
            room_id=room_id,
            room_prompt=room_prompt,
            deterministic_result=deterministic_document,
            threshold=7.0,
            review_count=3,
            immutable_effective_prompt=_normalized_text(paths["input:prompt.txt"]),
            contract_profile="school_reference_20260710",
            visual_requirements=room_visual_requirements(room_id),
        )
        messages = _room_messages(
            instruction, paths["input:reference.png"], review_paths
        )
        expected_request = room_vlm_request_contract(
            messages=messages,
            room_id=room_id,
            model=SCHOOL_VLM_MODEL,
            backend=SCHOOL_VLM_BACKEND,
            threshold=7.0,
            requirement_keys=list(room_visual_requirements(room_id)),
            reference_image=paths["input:reference.png"],
            review_images=review_paths,
        )
        _validate_vlm_request_record(
            evidence.get("vlm_request"),
            label=label,
            score_keys=ROOM_SCORE_KEYS,
            threshold=7.0,
            reference_sha256=str(reference_record.get("sha256")),
            render_sha256=review_hashes,
            expected_record=expected_request,
        )
        request_record = evidence.get("vlm_request")
        if (
            request_record.get("requirement_keys")
            != list(room_visual_requirements(room_id))
            or request_record.get("generated_view_indices") != [1, 2, 3]
        ):
            raise AcceptanceBundleError(
                f"{label} VLM requirement/view-index contract is stale"
            )
        return
    if label == "house_cutaway":
        _validate_house_cutaway(document, context.scene_dir)
        return
    if label == "artiverse_usage":
        _validate_artiverse_usage(document, paths=paths, scene_dir=context.scene_dir)
        return
    if label == "artiverse_final_validation":
        _require_passing_json(document, label=label)
        if document.get("schema_version") != 1 or document.get("placed_asset_count", 0) < 1 or document.get("final_surviving_asset_count", 0) < 1:
            raise AcceptanceBundleError("Final Artiverse validation is incomplete")
        _embedded_file_record(document.get("usage_manifest"), expected=paths["artiverse_usage"], label="Final Artiverse usage", strict_path=True)
        _embedded_file_record(document.get("house_state"), expected=paths["house_state"], label="Final Artiverse house state", strict_path=True)
        return
    if label == "whole_floor_gate":
        _require_passing_json(document, label=label)
        _empty_string_list(document.get("critical_issues"), label="whole-floor critical issues")
        _require_scores(document, label=label)
        floor_assessment = document.get("visual_assessment")
        floor_assessment_scores = (
            floor_assessment.get("scores")
            if isinstance(floor_assessment, dict)
            else None
        )
        if (
            not isinstance(floor_assessment, dict)
            or floor_assessment.get("critical_issues") != []
            or not isinstance(floor_assessment_scores, dict)
            or set(floor_assessment_scores) != set(FLOOR_SCORE_KEYS)
            or any(
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not 7.0 <= float(score) <= 10.0
                for score in floor_assessment_scores.values()
            )
        ):
            raise AcceptanceBundleError(
                "Whole-floor visual assessment is not the exact passing contract"
            )
        if document.get("evidence_verification", {}).get("status") != "pass":
            raise AcceptanceBundleError("Whole-floor evidence repeat verification did not pass")
        evidence = document.get("evidence")
        if not isinstance(evidence, dict) or evidence.get("schema_version") != 1:
            raise AcceptanceBundleError("Whole-floor immutable evidence is missing")
        cross = {
            "house_layout": "house_layout",
            "house_state": "house_state",
            "artiverse_usage": "artiverse_usage",
            "house_cutaway": "house_cutaway",
            "input_manifest": "input:input_manifest.json",
        }
        for key, expected_label in cross.items():
            _embedded_file_record(
                evidence.get(key),
                expected=paths[expected_label],
                label=f"whole floor {key}",
                strict_path=key != "input_manifest",
            )
        _embedded_file_record(evidence.get("reference_image"), expected=paths["input:reference.png"], label="whole floor reference", strict_path=False)
        overviews = evidence.get("overview_images")
        if not isinstance(overviews, dict) or set(overviews) != {"overview_top", "overview_isometric", "overview_front"}:
            raise AcceptanceBundleError("Whole-floor overview evidence set is malformed")
        for name in overviews:
            _embedded_file_record(overviews[name], expected=paths[f"overview:{name}"], label=f"whole floor {name}", strict_path=True)
        reference_record = evidence.get("reference_image")
        overview_hashes = [
            str(overviews[name].get("sha256"))
            for name in ("overview_top", "overview_isometric", "overview_front")
        ]
        if len(set(overview_hashes)) != 3:
            raise AcceptanceBundleError("Whole-floor overview image hashes are not distinct")
        manifest = _read_json(paths["input:input_manifest.json"], label="input manifest")
        if manifest.get("reference_image_sha256") != reference_record.get("sha256"):
            raise AcceptanceBundleError(
                "Whole-floor reference hash differs from input manifest"
            )
        try:
            from scripts.whole_floor_reference_gate import (
                _floor_messages,
                _vlm_request_contract as floor_vlm_request_contract,
                validate_exact_layout,
            )
        except ImportError:
            from whole_floor_reference_gate import (  # type: ignore[no-redef]
                _floor_messages,
                _vlm_request_contract as floor_vlm_request_contract,
                validate_exact_layout,
            )
        layout_document = _read_json(paths["house_layout"], label="house layout")
        deterministic_layout = validate_exact_layout(layout_document)
        if document.get("deterministic_layout_gate") != deterministic_layout:
            raise AcceptanceBundleError(
                "Whole-floor deterministic layout evidence is stale"
            )
        overview_paths = {
            name: paths[f"overview:{name}"]
            for name in ("overview_top", "overview_isometric", "overview_front")
        }
        floor_messages = _floor_messages(
            deterministic=deterministic_layout,
            threshold=7.0,
            reference_image=paths["input:reference.png"],
            overview_images=overview_paths,
        )
        floor_request_record = evidence.get("vlm_request")
        model = floor_request_record.get("model") if isinstance(
            floor_request_record, dict
        ) else None
        backend = floor_request_record.get("backend") if isinstance(
            floor_request_record, dict
        ) else None
        if not isinstance(model, str) or not model or backend not in {
            "openai",
            "codex",
        }:
            raise AcceptanceBundleError("Whole-floor VLM model/backend is malformed")
        expected_floor_request = floor_vlm_request_contract(
            messages=floor_messages,
            model=model,
            backend=backend,
            reference_image=paths["input:reference.png"],
            overview_images=overview_paths,
        )
        _validate_vlm_request_record(
            evidence.get("vlm_request"),
            label="whole_floor_gate",
            score_keys=FLOOR_SCORE_KEYS,
            threshold=7.0,
            reference_sha256=str(reference_record.get("sha256")),
            render_sha256=overview_hashes,
            expected_record=expected_floor_request,
        )
        request_record = evidence.get("vlm_request")
        if (
            request_record.get("generated_view_indices") != [1, 2, 3]
            or request_record.get("overview_names")
            != ["overview_top", "overview_isometric", "overview_front"]
        ):
            raise AcceptanceBundleError(
                "Whole-floor VLM view-index/name contract is stale"
            )
        return
    if label == "drake_sqz":
        _require_passing_json(document, label=label)
        checks = document.get("checks")
        if not isinstance(checks, dict) or not checks or any(value is not True for value in checks.values()):
            raise AcceptanceBundleError("SQZ Drake report contains a failed check")
        requirements = document.get("acceptance_requirements")
        if (
            not isinstance(requirements, dict)
            or requirements.get("required_visible_gpu_count") != 0
            or requirements.get("max_collision_elements_per_sdf") != 32
            or requirements.get("expected_room_count") != 11
        ):
            raise AcceptanceBundleError("SQZ Drake requirements are not the school profile")
        if document.get("dmd_inventory", {}).get("sha256") != _sha256_file(paths["house_dmd"]):
            raise AcceptanceBundleError("SQZ Drake report is bound to a different DMD")
        if document.get("house_state_inventory", {}).get("sha256") != _sha256_file(paths["house_state"]):
            raise AcceptanceBundleError("SQZ Drake report is bound to a different house state")
        collision = document.get("collision_report")
        if not isinstance(collision, dict) or collision.get("parse_failures") != [] or collision.get("assets_over_collision_cap") != [] or collision.get("sdf_asset_count", 0) < 1:
            raise AcceptanceBundleError("SQZ Drake collision report is not passing")
        return
    if label == "simulator_exports":
        _validate_simulator_exports(document, context=context)
        return
    if label == "sage_scene_checker":
        _require_passing_json(document, label=label, pass_key="pass")
        policy = document.get("acceptance_policy")
        if not isinstance(policy, dict) or policy.get("fail_on_warnings") is not True or policy.get("fatal_failed_check_ids") != []:
            raise AcceptanceBundleError("SAGE report did not fail on warnings")
        summary = document.get("summary")
        if not isinstance(summary, dict) or summary.get("num_rooms") != 11 or summary.get("num_errors") != 0 or summary.get("num_warnings") != 0:
            raise AcceptanceBundleError("SAGE report summary is not clean")
        checks = document.get("checks")
        if not isinstance(checks, list) or not checks or any(
            isinstance(item, dict)
            and item.get("severity") in {"error", "warning"}
            and item.get("status") != "pass"
            for item in checks
        ):
            raise AcceptanceBundleError("SAGE report contains a failed required check")
        return


def _validate_cross_file_state(paths: Mapping[str, Path], context: BundleContext) -> None:
    _validate_input_bundle(paths)
    layout = _read_json(paths["house_layout"], label="house layout")
    placed = layout.get("placed_rooms")
    if isinstance(placed, dict):
        room_ids = set(map(str, placed))
    elif isinstance(placed, list):
        room_ids = {
            str(value.get("room_id") or value.get("id"))
            for value in placed
            if isinstance(value, dict)
        }
    else:
        room_ids = set()
    if room_ids != ROOM_SET or layout.get("placement_valid") is not True or layout.get("connectivity_valid") is not True:
        raise AcceptanceBundleError("House layout is not the exact valid 11-room layout")
    house_state = _read_json(paths["house_state"], label="combined house state")
    rooms = house_state.get("rooms")
    if not isinstance(rooms, dict) or set(map(str, rooms)) != ROOM_SET:
        raise AcceptanceBundleError("Combined house state does not contain exactly 11 rooms")
    if any(
        not isinstance(value, dict) or not isinstance(value.get("objects"), (dict, list))
        for value in rooms.values()
    ):
        raise AcceptanceBundleError("Combined house state has malformed room/object data")

    json_labels = {
        label
        for label, path in paths.items()
        if path.suffix.lower() == ".json"
        and label not in {"house_layout", "house_state"}
    }
    for label in sorted(json_labels):
        document = _read_json(paths[label], label=label)
        _validate_document(label, document, paths=paths, context=context)

    expected_room_labels = {
        prefix: {label.split(":", 1)[1] for label in paths if label.startswith(prefix + ":")}
        for prefix in ("room_deterministic", "room_visual", "room_cutaway")
    }
    if any(values != ROOM_SET for values in expected_room_labels.values()):
        raise AcceptanceBundleError(
            f"Required 11-room evidence sets are incomplete: {expected_room_labels}"
        )


def _copy_external_atomically(
    context: BundleContext,
    sources: Sequence[EvidenceSource],
) -> Path:
    target = context.packaged_evidence_dir
    if target.exists():
        raise AcceptanceBundleError(
            f"Packaged evidence already exists; use --verify-only: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    _require_no_link_components(target.parent, label="final acceptance evidence parent")
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.{os.getpid()}.",
            suffix=".tmp",
            dir=target.parent,
        )
    )
    try:
        seen_destinations: set[Path] = set()
        for source in sources:
            source_path = _require_regular_file(
                source.source,
                label=source.label,
                containment_root=(
                    context.input_dir
                    if source.category == "immutable_input"
                    else context.repo_dir
                ),
            )
            destination = stage / source.packaged_relative
            if destination in seen_destinations:
                raise AcceptanceBundleError(
                    f"Duplicate packaged evidence destination: {destination}"
                )
            seen_destinations.add(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            before = _sha256_file(source_path)
            shutil.copyfile(source_path, destination)
            after = _sha256_file(source_path)
            copied = _sha256_file(destination)
            if before != after or copied != before:
                raise AcceptanceBundleError(
                    f"Evidence changed during atomic copy: {source.label}"
                )
            _require_regular_file(destination, label=f"packaged {source.label}", containment_root=stage)
        return stage
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _scene_relative(scene_dir: Path, path: Path) -> str:
    try:
        relative = path.relative_to(scene_dir)
    except ValueError as exc:
        raise AcceptanceBundleError(f"Evidence path escapes scene: {path}") from exc
    value = relative.as_posix()
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or not parsed.parts or any(part in {"", ".", ".."} for part in parsed.parts):
        raise AcceptanceBundleError(f"Evidence path is not canonical scene-relative POSIX: {value}")
    return value


def _entry(
    *,
    label: str,
    physical_path: Path,
    scene_path: Path,
    scene_dir: Path,
    category: str,
) -> dict[str, Any]:
    path = _require_regular_file(physical_path, label=label)
    return {
        "label": label,
        "scene_relative_path": _scene_relative(scene_dir, scene_path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "category": category,
    }


def _walk_scene_files(scene_dir: Path, *, output: Path) -> list[Path]:
    files: list[Path] = []
    for root_text, directory_names, file_names in os.walk(scene_dir, followlinks=False):
        root = Path(root_text)
        for name in list(directory_names):
            directory = root / name
            if _is_link_like(directory):
                raise AcceptanceBundleError(f"Scene package contains link-like directory: {directory}")
        for name in file_names:
            path = root / name
            if path == output:
                continue
            if _is_link_like(path):
                raise AcceptanceBundleError(f"Scene package contains link-like file: {path}")
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise AcceptanceBundleError(f"Scene package contains special file: {path}")
            if info.st_size < 1:
                raise AcceptanceBundleError(f"Scene package contains empty file: {path}")
            files.append(path.resolve())
    return sorted(files, key=lambda value: value.relative_to(scene_dir).as_posix())


def _reject_leaked_work_directories(scene_dir: Path) -> None:
    """Refuse unpublished simulator/assembly attempts anywhere in the package."""

    simulator = re.compile(r"^\.mujoco_export\.(?:staging|previous|failed)\.")
    hidden_combined = re.compile(r"^\.combined_house(?:\.|$)")
    combined_backup = re.compile(
        r"^combined_house\.(?:pre_final_assemble_backup|rejected)_"
    )
    evidence_stage = re.compile(r"^\.final_acceptance_evidence\..*\.tmp$")
    for path in scene_dir.rglob("*"):
        if not path.is_dir():
            continue
        name = path.name
        if (
            simulator.match(name)
            or hidden_combined.match(name)
            or combined_backup.match(name)
            or evidence_stage.match(name)
        ):
            raise AcceptanceBundleError(
                f"Scene contains leaked staging/backup directory: {path}"
            )


def _build_inventory(
    context: BundleContext,
    *,
    staged_evidence_root: Path,
    fixed_paths: Mapping[str, Path],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    fixed_scene_paths: dict[str, Path] = {}
    staged_external = _packaged_external_paths(context, physical_root=staged_evidence_root)
    target_external = _packaged_external_paths(context)
    for label, path in fixed_paths.items():
        if label in staged_external:
            physical = staged_external[label]
            scene_path = target_external[label]
            category = next(
                source.category for source in _external_sources(context) if source.label == label
            )
        else:
            physical = path
            scene_path = path
            category = "pipeline_evidence"
        entries.append(
            _entry(
                label=label,
                physical_path=physical,
                scene_path=scene_path,
                scene_dir=context.scene_dir,
                category=category,
            )
        )
        fixed_scene_paths[_scene_relative(context.scene_dir, scene_path)] = scene_path

    evidence_stage_prefix = staged_evidence_root.resolve()
    for path in _walk_scene_files(context.scene_dir, output=context.output):
        if _within(path, context.packaged_evidence_dir):
            continue
        relative = _scene_relative(context.scene_dir, path)
        if relative in fixed_scene_paths:
            continue
        # A stage is a sibling of the target evidence directory and must never
        # become part of the accepted package inventory.
        if _within(path, evidence_stage_prefix):
            continue
        entries.append(
            _entry(
                label=f"package_file:{relative}",
                physical_path=path,
                scene_path=path,
                scene_dir=context.scene_dir,
                category="scene_payload",
            )
        )
    entries.sort(key=lambda value: value["label"])
    labels = [entry["label"] for entry in entries]
    paths = [entry["scene_relative_path"] for entry in entries]
    if len(labels) != len(set(labels)):
        raise AcceptanceBundleError("Evidence inventory contains duplicate labels")
    if len(paths) != len(set(paths)):
        raise AcceptanceBundleError("Evidence inventory contains duplicate paths")
    return entries


def required_labels() -> list[str]:
    labels = set(_internal_paths(Path(".")))
    # _internal_paths with a relative root is used for names only.
    labels.update(f"input:{name}" for name in INPUT_FILES)
    labels.update(
        {
            "input_manifest_validation",
            "pipeline_code_contract",
            "materials_contract",
            "artiverse_preparation",
            "artiverse_visual_resources",
            "articulated_router",
            "resolved_asset_policy",
            "sam3d_offline_preflight",
            "sam3d_generation_preflight",
            "sam3d_generation_input",
            "sam3d_generation_glb",
            "sam3d_generation_mask",
            "sam3d_generation_masked_image",
            "objathor_retrieval_preflight",
            "vlm_vision_smoke",
        }
    )
    return sorted(labels)


def _record_without_attestation(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "self_attestation"}


def _attest_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "algorithm": HASH_ALGORITHM,
        "scope": "canonical_json_of_all_fields_except_self_attestation",
        "sha256": _sha256_bytes(_canonical_json(_record_without_attestation(record))),
    }


def _build_record(
    context: BundleContext,
    inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required = required_labels()
    labels = {str(entry.get("label")) for entry in inventory}
    missing = sorted(set(required) - labels)
    if missing:
        raise AcceptanceBundleError(f"Acceptance inventory is missing required labels: {missing}")
    by_label = {str(entry["label"]): entry for entry in inventory}
    package_items = sorted(
        (
            {
                "scene_relative_path": entry["scene_relative_path"],
                "size_bytes": entry["size_bytes"],
                "sha256": entry["sha256"],
            }
            for entry in inventory
        ),
        key=lambda value: value["scene_relative_path"],
    )
    record_relative = _scene_relative(
        context.scene_dir,
        context.output,
    )
    record: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "run_attempt_id": context.run_attempt_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "hash_algorithm": HASH_ALGORITHM,
        "path_encoding": "scene-relative-posix",
        "required_room_ids": list(ROOM_IDS),
        "required_labels": required,
        "scene_identity": {
            "scene_relative_to_run": context.scene_dir.relative_to(
                context.run_dir
            ).as_posix(),
            "house_layout_sha256": by_label["house_layout"]["sha256"],
            "house_state_sha256": by_label["house_state"]["sha256"],
            "house_dmd_sha256": by_label["house_dmd"]["sha256"],
        },
        "package_inventory": {
            "scope": "all_regular_scene_files_except_this_record",
            "record_scene_relative_path": record_relative,
            "file_count": len(package_items),
            "content_sha256": _sha256_bytes(_canonical_json(package_items)),
        },
        "external_package_manifest": {
            "required": True,
            "status": "pending_creation_after_this_record",
            "must_be_outside_scene": True,
            "must_cover_exact_scene_file_set_including_this_record": True,
            "record_scene_relative_path": record_relative,
        },
        "evidence_count": len(inventory),
        "evidence": list(inventory),
        "evidence_attestation": {
            "algorithm": HASH_ALGORITHM,
            "sha256": _sha256_bytes(_canonical_json(list(inventory))),
        },
        "label_specific_validation": {
            label: "pass" for label in required
        },
        "classroom_variation_gate": {
            "required": True,
            "status": "pass",
            "evidence_label": "classroom_variation",
        },
        "two_gpu_acceptance": {
            "required": True,
            "status": "pending",
            "minimum_visible_gpu_count": 2,
        },
    }
    record["self_attestation"] = _attest_record(record)
    return record


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_no_link_components(path.parent, label="acceptance output parent")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() and _is_link_like(temporary):
        raise AcceptanceBundleError(f"Acceptance temporary output is link-like: {temporary}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def create_bundle(
    *,
    repo_dir: Path,
    run_dir: Path,
    scene_dir: Path,
    input_dir: Path,
    run_attempt_id: str,
    output: Path,
) -> dict[str, Any]:
    context = _build_context(
        repo_dir=repo_dir,
        run_dir=run_dir,
        scene_dir=scene_dir,
        input_dir=input_dir,
        run_attempt_id=run_attempt_id,
        output=output,
    )
    if context.output.exists():
        raise AcceptanceBundleError(
            f"Acceptance record already exists; use --verify-only: {context.output}"
        )
    _reject_leaked_work_directories(context.scene_dir)
    sources = _external_sources(context)
    # Validate every source before creating a staging directory.  A failed gate
    # therefore cannot leave even a partial final_acceptance_evidence tree.
    for source in sources:
        _require_regular_file(
            source.source,
            label=source.label,
            containment_root=(context.input_dir if source.category == "immutable_input" else context.repo_dir),
        )
    for label, path in _internal_paths(context.scene_dir).items():
        _require_regular_file(path, label=label, containment_root=context.scene_dir)

    stage = _copy_external_atomically(context, sources)
    promoted = False
    try:
        staged_paths = _all_fixed_paths(context, physical_evidence_root=stage)
        _validate_cross_file_state(staged_paths, context)
        inventory = _build_inventory(
            context,
            staged_evidence_root=stage,
            fixed_paths=staged_paths,
        )
        record = _build_record(context, inventory)
        os.replace(stage, context.packaged_evidence_dir)
        promoted = True
        _atomic_write_json(context.output, record)
        return record
    except Exception:
        if promoted and context.packaged_evidence_dir.exists():
            shutil.rmtree(context.packaged_evidence_dir, ignore_errors=True)
        elif stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        try:
            context.output.unlink()
        except FileNotFoundError:
            pass
        raise


def _resolve_record_path(scene_dir: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AcceptanceBundleError(f"{label} has no scene-relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise AcceptanceBundleError(f"{label} path is unsafe: {value!r}")
    if "\\" in value or relative.as_posix() != value:
        raise AcceptanceBundleError(f"{label} path is not canonical POSIX: {value!r}")
    path = scene_dir.joinpath(*relative.parts)
    return _require_regular_file(path, label=label, containment_root=scene_dir)


def _validate_record_shape(
    record: Mapping[str, Any], *, context: BundleContext
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    if record.get("schema_id") != SCHEMA_ID or record.get("schema_version") != SCHEMA_VERSION:
        raise AcceptanceBundleError("Acceptance record schema is unsupported")
    if record.get("status") != STATUS:
        raise AcceptanceBundleError("Acceptance record is not awaiting two-GPU acceptance")
    if record.get("run_attempt_id") != context.run_attempt_id:
        raise AcceptanceBundleError("Acceptance record run_attempt_id differs from the CLI")
    if record.get("hash_algorithm") != HASH_ALGORITHM or record.get("path_encoding") != "scene-relative-posix":
        raise AcceptanceBundleError("Acceptance record hash/path contract is malformed")
    if record.get("required_room_ids") != list(ROOM_IDS):
        raise AcceptanceBundleError("Acceptance record room set is not the exact 11-room set")
    if record.get("required_labels") != required_labels():
        raise AcceptanceBundleError("Acceptance record required-label contract was changed")
    if record.get("classroom_variation_gate") != {
        "required": True,
        "status": "pass",
        "evidence_label": "classroom_variation",
    }:
        raise AcceptanceBundleError("Acceptance record classroom-variation state is invalid")
    if record.get("two_gpu_acceptance") != {
        "required": True,
        "status": "pending",
        "minimum_visible_gpu_count": 2,
    }:
        raise AcceptanceBundleError("SQZ record improperly claims two-GPU acceptance")
    if record.get("self_attestation") != _attest_record(record):
        raise AcceptanceBundleError("Acceptance receipt self-attestation was tampered with")
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise AcceptanceBundleError("Acceptance evidence inventory is missing")
    if record.get("evidence_count") != len(evidence):
        raise AcceptanceBundleError("Acceptance evidence_count is stale")
    if record.get("evidence_attestation") != {
        "algorithm": HASH_ALGORITHM,
        "sha256": _sha256_bytes(_canonical_json(evidence)),
    }:
        raise AcceptanceBundleError("Acceptance evidence attestation was tampered with")
    labels: set[str] = set()
    relative_paths: set[str] = set()
    resolved: dict[str, Path] = {}
    normalized: list[dict[str, Any]] = []
    for index, value in enumerate(evidence):
        if not isinstance(value, dict):
            raise AcceptanceBundleError(f"Acceptance evidence[{index}] is malformed")
        label = value.get("label")
        if not isinstance(label, str) or not label or label in labels:
            raise AcceptanceBundleError(f"Acceptance evidence has duplicate/invalid label: {label!r}")
        labels.add(label)
        relative = value.get("scene_relative_path")
        if not isinstance(relative, str) or relative in relative_paths:
            raise AcceptanceBundleError(f"Acceptance evidence has duplicate/invalid path: {relative!r}")
        relative_paths.add(relative)
        path = _resolve_record_path(context.scene_dir, relative, label=label)
        if value.get("size_bytes") != path.stat().st_size or value.get("sha256") != _sha256_file(path):
            raise AcceptanceBundleError(f"Acceptance evidence changed: {label}")
        if not isinstance(value.get("category"), str) or not value["category"]:
            raise AcceptanceBundleError(f"Acceptance evidence category is missing: {label}")
        resolved[label] = path
        normalized.append(value)
    if not set(required_labels()).issubset(labels):
        raise AcceptanceBundleError("Acceptance evidence lacks one or more required labels")
    by_label = {str(item["label"]): item for item in normalized}
    package_items = sorted(
        (
            {
                "scene_relative_path": item["scene_relative_path"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in normalized
        ),
        key=lambda value: value["scene_relative_path"],
    )
    record_relative = _scene_relative(context.scene_dir, context.output)
    if record.get("scene_identity") != {
        "scene_relative_to_run": context.scene_dir.relative_to(
            context.run_dir
        ).as_posix(),
        "house_layout_sha256": by_label["house_layout"]["sha256"],
        "house_state_sha256": by_label["house_state"]["sha256"],
        "house_dmd_sha256": by_label["house_dmd"]["sha256"],
    }:
        raise AcceptanceBundleError("Acceptance scene identity is stale")
    if record.get("package_inventory") != {
        "scope": "all_regular_scene_files_except_this_record",
        "record_scene_relative_path": record_relative,
        "file_count": len(package_items),
        "content_sha256": _sha256_bytes(_canonical_json(package_items)),
    }:
        raise AcceptanceBundleError("Acceptance package fingerprint is stale")
    if record.get("external_package_manifest") != {
        "required": True,
        "status": "pending_creation_after_this_record",
        "must_be_outside_scene": True,
        "must_cover_exact_scene_file_set_including_this_record": True,
        "record_scene_relative_path": record_relative,
    }:
        raise AcceptanceBundleError(
            "Acceptance external-package-manifest contract is malformed"
        )
    claims = record.get("label_specific_validation")
    if not isinstance(claims, dict) or claims != {label: "pass" for label in required_labels()}:
        raise AcceptanceBundleError("Acceptance label-specific validation claims were changed")
    return normalized, resolved


def verify_bundle(
    *,
    repo_dir: Path,
    run_dir: Path,
    scene_dir: Path,
    input_dir: Path,
    run_attempt_id: str,
    output: Path,
) -> dict[str, Any]:
    context = _build_context(
        repo_dir=repo_dir,
        run_dir=run_dir,
        scene_dir=scene_dir,
        input_dir=input_dir,
        run_attempt_id=run_attempt_id,
        output=output,
    )
    _require_directory(context.packaged_evidence_dir, label="packaged final acceptance evidence")
    _reject_leaked_work_directories(context.scene_dir)
    record = _read_json(
        _require_regular_file(context.output, label="SQZ acceptance record", containment_root=context.scene_dir),
        label="SQZ acceptance record",
    )
    inventory, resolved = _validate_record_shape(record, context=context)
    fixed = _all_fixed_paths(context)
    for label, expected in fixed.items():
        actual = resolved.get(label)
        if actual is None or actual != expected.resolve():
            raise AcceptanceBundleError(f"Acceptance label {label} is not at its canonical package path")
    _validate_cross_file_state(fixed, context)

    inventory_paths = {entry["scene_relative_path"] for entry in inventory}
    current_paths = {
        _scene_relative(context.scene_dir, path)
        for path in _walk_scene_files(context.scene_dir, output=context.output)
    }
    missing = sorted(inventory_paths - current_paths)
    added = sorted(current_paths - inventory_paths)
    if missing or added:
        raise AcceptanceBundleError(
            f"Scene package file set changed: missing={missing}, added={added}"
        )
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--run-attempt-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Rehash and semantically revalidate the existing in-package record.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    operation = verify_bundle if args.verify_only else create_bundle
    try:
        result = operation(
            repo_dir=args.repo_dir,
            run_dir=args.run_dir,
            scene_dir=args.scene_dir,
            input_dir=args.input_dir,
            run_attempt_id=args.run_attempt_id,
            output=args.output,
        )
    except Exception as exc:  # Fail closed on I/O races and malformed evidence too.
        print(
            json.dumps(
                {
                    "schema_id": SCHEMA_ID,
                    "schema_version": SCHEMA_VERSION,
                    "status": "fail",
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
