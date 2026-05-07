#!/usr/bin/env python3
"""Create a hash-bound receipt for the real classroom_01 production benchmark.

The benchmark deliberately stops after the room's deterministic and image-aware
gates.  It is timing evidence, not a whole-school acceptance artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any, Iterable

from scripts import artiverse_contract as _artiverse_contract

sys.modules.setdefault("artiverse_contract", _artiverse_contract)
from scripts import preflight_artiverse_visual_resources as artiverse_visual
from scripts import preflight_sam3d_generation as sam3d_generation
from scripts import preflight_sam3d_offline as sam3d_preflight
from scripts import seed_reference_school_layout as reference_layout


SCHEMA_ID = "scenesmith_classroom_full_quality_benchmark_v1"
ROOM_ID = "classroom_01"
ROOM_SCORE_KEYS = {
    "object_relevance",
    "placement_realism",
    "clearance_and_access",
    "collision_risk",
    "prompt_alignment",
}
EXPECTED_EXTERNAL_INPUT_BINDING = {
    "original_prompt_sha256": "5ddc06e1a9afa60b0417da882c1ec53265eae23f3f9bdd2343360d123923f34c",
    "effective_prompt_sha256": "ac8d297cc9a2d605f41b4bcd7abd52aac29bfd0f195875840342ee1e6a7da86f",
    "reference_image_sha256": "7ba62c39ac98cd2b21d9a5a97fd6f9b90d7efaf447d5ac9a6148524b5a8dbe48",
}


class BenchmarkReceiptError(RuntimeError):
    """The benchmark evidence is absent, stale, or not production quality."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or resolved.is_symlink():
        raise BenchmarkReceiptError(f"{label} must not be a symlink: {path}")
    is_junction = getattr(path, "is_junction", None)
    if is_junction is not None and is_junction():
        raise BenchmarkReceiptError(f"{label} must not be a junction: {path}")
    if not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
        raise BenchmarkReceiptError(f"{label} is not a regular file: {path}")
    if resolved.stat().st_size <= 0:
        raise BenchmarkReceiptError(f"{label} is empty: {path}")
    return resolved


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    regular = _regular_file(path, label=label)
    try:
        value = json.loads(regular.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BenchmarkReceiptError(f"cannot read {label}: {regular}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkReceiptError(f"{label} must be a JSON object")
    return value


def _require_pass(path: Path, *, label: str) -> dict[str, Any]:
    value = _load_json(path, label=label)
    if value.get("status") != "pass":
        raise BenchmarkReceiptError(f"{label} is not passing: {path}")
    return value


def _validate_sam3d_preflight(document: dict[str, Any]) -> None:
    """Reject load-only, status-only, stale, or tampered SAM3D proof JSON."""

    failures: list[str] = []
    if document.get("schema_version") != sam3d_preflight.RESULT_SCHEMA_VERSION:
        failures.append("schema_version is unsupported")
    if document.get("offline") is not True:
        failures.append("offline proof is missing")
    offline_environment = document.get("offline_environment")
    if not isinstance(offline_environment, dict) or any(
        offline_environment.get(name) != "1"
        for name in sam3d_preflight.OFFLINE_VARIABLES
    ):
        failures.append("offline environment is incomplete")
    if document.get("model_loaded") is not True:
        failures.append("SAM3 image model load proof is missing")
    if document.get("pipeline_loaded") is not True:
        failures.append("SAM 3D Objects pipeline load proof is missing")
    if (
        not isinstance(document.get("visible_gpu_count"), int)
        or isinstance(document.get("visible_gpu_count"), bool)
        or document.get("visible_gpu_count", 0) < 1
    ):
        failures.append("GPU load proof is missing")
    if not isinstance(document.get("evidence"), dict) or document.get(
        "evidence_verification", {}
    ).get("status") != "pass":
        failures.append("artifact evidence is missing or stale")
    failures.extend(
        sam3d_preflight.verify_inference_smoke(document.get("inference_smoke"))
    )
    expected_attestation = {
        "algorithm": "sha256",
        "sha256": sam3d_preflight._attestation_sha256(document),
    }
    if document.get("attestation") != expected_attestation:
        failures.append("JSON attestation is invalid")
    if failures:
        raise BenchmarkReceiptError(
            "SAM3D offline preflight is invalid: " + "; ".join(failures)
        )


def _validate_artiverse_visual_resources(document: dict[str, Any]) -> None:
    """Require the complete publisher-GLB to external-glTF derivation audit."""

    try:
        artiverse_visual._validate_saved_receipt(document)
    except artiverse_visual.ArtiverseVisualResourceError as exc:
        raise BenchmarkReceiptError(
            f"Artiverse visual-resource preflight is invalid: {exc}"
        ) from exc

    expected_policy = artiverse_visual.mapping_policy()
    if document.get("mapping_policy") != expected_policy:
        raise BenchmarkReceiptError(
            "Artiverse visual-resource preflight changed its exact mapping policy"
        )

    audit = document.get("audit")
    if not isinstance(audit, dict):
        raise BenchmarkReceiptError(
            "Artiverse visual-resource preflight has no complete audit"
        )

    def required_int(name: str, *, minimum: int) -> int:
        value = audit.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise BenchmarkReceiptError(
                f"Artiverse visual-resource preflight has invalid {name}"
            )
        return value

    indexed = required_int("indexed_asset_count", minimum=500)
    audited = required_int("audited_asset_count", minimum=500)
    if indexed != audited:
        raise BenchmarkReceiptError(
            "Artiverse visual-resource preflight did not audit every indexed asset"
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
        required_int(name, minimum=1)
    for name in (
        "mapping_inventory_sha256",
        "collision_inventory_sha256",
        "asset_inventory_sha256",
    ):
        value = audit.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise BenchmarkReceiptError(
                f"Artiverse visual-resource preflight has invalid {name}"
            )
    assets = audit.get("assets")
    if not isinstance(assets, list) or len(assets) != audited:
        raise BenchmarkReceiptError(
            "Artiverse visual-resource preflight lacks every per-asset derivation"
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
            raise BenchmarkReceiptError(
                "Artiverse visual-resource per-asset derivation is malformed"
            )
        object_id = item.get("object_id")
        if not isinstance(object_id, str) or not object_id or object_id in asset_ids:
            raise BenchmarkReceiptError(
                "Artiverse visual-resource per-asset identities are invalid"
            )
        asset_ids.add(object_id)
        for name in summed:
            value = item.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise BenchmarkReceiptError(
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
            value = item.get(name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise BenchmarkReceiptError(
                    f"Artiverse visual-resource asset has invalid {name}"
                )
    if any(summed[name] != audit.get(name) for name in summed):
        raise BenchmarkReceiptError(
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
        raise BenchmarkReceiptError(
            "Artiverse visual-resource preflight does not prove zero source mutation"
        )
    if document.get("runtime_code_revalidated_after_audit") is not True:
        raise BenchmarkReceiptError(
            "Artiverse visual-resource runtime code was not revalidated"
        )
    runtime_code = document.get("runtime_code")
    if not isinstance(runtime_code, list):
        raise BenchmarkReceiptError(
            "Artiverse visual-resource preflight has no runtime code inventory"
        )
    runtime_paths = {
        str(item.get("path", "")).replace("\\", "/")
        for item in runtime_code
        if isinstance(item, dict)
        and isinstance(item.get("size_bytes"), int)
        and not isinstance(item.get("size_bytes"), bool)
        and item.get("size_bytes", 0) > 0
        and isinstance(item.get("sha256"), str)
        and len(item.get("sha256", "")) == 64
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
        raise BenchmarkReceiptError(
            "Artiverse visual-resource preflight lacks required runtime code: "
            + ", ".join(missing)
        )


def _artifact(path: Path, *, label: str) -> dict[str, Any]:
    regular = _regular_file(path, label=label)
    return {
        "label": label,
        "path": os.fspath(regular),
        "size_bytes": regular.stat().st_size,
        "sha256": _sha256(regular),
    }


def _validate_asset_policy(path: Path) -> dict[str, Any]:
    policy = _load_json(path, label="resolved asset policy")
    expected_sources = {
        "furniture_agent": "generated",
        "wall_agent": "generated",
        "ceiling_agent": "generated",
        "manipuland_agent": "objaverse",
    }
    errors: list[str] = []
    for agent, expected_source in expected_sources.items():
        record = policy.get(agent)
        if not isinstance(record, dict):
            errors.append(f"missing {agent}")
            continue
        if record.get("general_asset_source") != expected_source:
            errors.append(f"{agent} source is not {expected_source}")
        if agent != "manipuland_agent" and record.get("backend") != "sam3d":
            errors.append(f"{agent} backend is not sam3d")
        for key in ("coacd_max_convex_hull", "vhacd_max_convex_hulls"):
            value = record.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or not (1 <= value <= 32):
                errors.append(f"{agent} {key} is invalid or exceeds 32")

    articulated = policy.get("articulated_contract")
    if not isinstance(articulated, dict):
        errors.append("articulated contract is absent")
    else:
        if articulated.get("articulated_strategy_enabled") is not True:
            errors.append("generic articulated strategy is disabled")
        if articulated.get("artiverse_strategy_enabled") is not True:
            errors.append("Artiverse strategy is disabled")
        enabled_agents = articulated.get("artvip_enabled_agents")
        if not isinstance(enabled_agents, dict) or set(enabled_agents) != set(
            expected_sources
        ) or not all(value is True for value in enabled_agents.values()):
            errors.append("ArtVIP is not enabled for every production agent")
        for source in ("artvip", "artiverse"):
            record = articulated.get(source)
            if not isinstance(record, dict):
                errors.append(f"{source} source contract is absent")
                continue
            if record.get("enabled") is not True:
                errors.append(f"{source} source is disabled")
            if record.get("data_path_exists") is not True:
                errors.append(f"{source} data path is absent")
            if record.get("embeddings_path_exists") is not True:
                errors.append(f"{source} embeddings path is absent")
            if record.get("missing_embedding_files") != []:
                errors.append(f"{source} embeddings are incomplete")
    if errors:
        raise BenchmarkReceiptError("resolved full-quality policy failed: " + "; ".join(errors))
    return policy


def _events(path: Path) -> list[dict[str, Any]]:
    regular = _regular_file(path, label="benchmark event log")
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    previous_epoch: int | None = None
    for line_number, raw in enumerate(regular.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.split("\t")
        if len(parts) != 3:
            raise BenchmarkReceiptError(
                f"benchmark event line {line_number} is malformed"
            )
        epoch_text, timestamp, name = parts
        try:
            epoch = int(epoch_text)
        except ValueError as exc:
            raise BenchmarkReceiptError(
                f"benchmark event line {line_number} has an invalid epoch"
            ) from exc
        if not name or name in seen:
            raise BenchmarkReceiptError(f"duplicate/empty benchmark event: {name!r}")
        if previous_epoch is not None and epoch < previous_epoch:
            raise BenchmarkReceiptError("benchmark event timestamps are not monotonic")
        seen.add(name)
        previous_epoch = epoch
        events.append({"name": name, "epoch": epoch, "timestamp_utc": timestamp})

    required = {
        "benchmark_start",
        "artiverse_visual_preflight_complete",
        "sam3d_generation_preflight_complete",
        "preflight_complete",
        "floor_layout_gate_complete",
        "room_start",
        "benchmark_complete",
    }
    names = [item["name"] for item in events]
    missing = sorted(required - set(names))
    if missing or not events or names[0] != "benchmark_start" or names[-1] != "benchmark_complete":
        raise BenchmarkReceiptError(
            f"benchmark event chain is incomplete or misordered; missing={missing}"
        )
    if "room_evidence_reused" not in names:
        detailed = [
            "room_generation_start",
            "room_generation_complete",
            "blend_refresh_complete",
            "render_complete",
            "deterministic_gate_complete",
            "visual_gate_complete",
        ]
        missing_detail = [name for name in detailed if name not in names]
        if missing_detail:
            raise BenchmarkReceiptError(
                f"fresh benchmark event chain is incomplete: {missing_detail}"
            )
        positions = [names.index(name) for name in detailed]
        if positions != sorted(positions):
            raise BenchmarkReceiptError("fresh benchmark stages are out of order")

    for current, following in zip(events, events[1:]):
        current["seconds_until_next_event"] = following["epoch"] - current["epoch"]
    events[-1]["seconds_until_next_event"] = 0
    return events


def _gpu_samples(path: Path) -> dict[str, Any]:
    regular = _regular_file(path, label="benchmark GPU samples")
    samples: list[dict[str, Any]] = []
    with regular.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) != 6:
                continue
            timestamp, index_text, name, utilization_text, used_text, total_text = (
                item.strip() for item in row
            )
            try:
                index = int(index_text)
                utilization = float(utilization_text)
                used_mib = float(used_text)
                total_mib = float(total_text)
            except ValueError:
                continue
            if not all(
                math.isfinite(item)
                for item in (utilization, used_mib, total_mib)
            ):
                continue
            if index < 0 or not name or not (0 <= utilization <= 100):
                continue
            if used_mib < 0 or total_mib <= 0 or used_mib > total_mib:
                continue
            samples.append(
                {
                    "timestamp": timestamp,
                    "index": index,
                    "name": name,
                    "utilization_percent": utilization,
                    "memory_used_mib": used_mib,
                    "memory_total_mib": total_mib,
                }
            )
    if not samples:
        raise BenchmarkReceiptError("benchmark recorded no valid GPU samples")
    return {
        "sample_count": len(samples),
        "gpu_indices": sorted({item["index"] for item in samples}),
        "gpu_names": sorted({item["name"] for item in samples}),
        "peak_utilization_percent": max(
            item["utilization_percent"] for item in samples
        ),
        "mean_utilization_percent": sum(
            item["utilization_percent"] for item in samples
        )
        / len(samples),
        "peak_memory_used_mib": max(item["memory_used_mib"] for item in samples),
        "memory_total_mib": max(item["memory_total_mib"] for item in samples),
    }


def _bound_evidence_files(value: Any) -> Iterable[tuple[str, Path, str]]:
    if isinstance(value, dict):
        path = value.get("path")
        digest = value.get("sha256")
        if isinstance(path, str) and path and isinstance(digest, str) and digest:
            yield "visual gate bound evidence", Path(path), digest
        for nested in value.values():
            yield from _bound_evidence_files(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _bound_evidence_files(nested)


def create_receipt(
    *,
    repo_dir: Path,
    run_dir: Path,
    scene_dir: Path,
    input_dir: Path,
    events_path: Path,
    gpu_samples_path: Path,
    output: Path,
    run_attempt_id: str,
) -> dict[str, Any]:
    repo = repo_dir.resolve(strict=True)
    run = run_dir.resolve(strict=True)
    scene = scene_dir.resolve(strict=True)
    inputs = input_dir.resolve(strict=True)
    if not run_attempt_id or any(character.isspace() for character in run_attempt_id):
        raise BenchmarkReceiptError("run_attempt_id is empty or contains whitespace")
    if scene.parent != run:
        raise BenchmarkReceiptError("scene_dir must be scene_000 directly under run_dir")
    combined_house = scene / "combined_house"
    if combined_house.exists() or combined_house.is_symlink():
        raise BenchmarkReceiptError(
            "benchmark is not isolated: combined_house already exists"
        )

    preflight = repo / "outputs" / "preflight" / "full_quality_school_reference_20260710"
    sam3d_generation_paths = sam3d_generation.canonical_paths(preflight)
    deterministic_path = (
        scene
        / "quality_gates"
        / "room_self_exam_deterministic"
        / f"{ROOM_ID}.json"
    )
    visual_path = scene / "quality_gates" / "room_self_exam" / f"{ROOM_ID}.json"
    deterministic = _require_pass(deterministic_path, label="deterministic room gate")
    visual = _require_pass(visual_path, label="visual room gate")
    if deterministic.get("room_id") != ROOM_ID or visual.get("room_id") != ROOM_ID:
        raise BenchmarkReceiptError("room gate identity is not classroom_01")
    if visual.get("threshold") != 7:
        raise BenchmarkReceiptError("visual room gate threshold is not exactly 7")
    scores = visual.get("scores")
    if not isinstance(scores, dict) or set(scores) != ROOM_SCORE_KEYS:
        raise BenchmarkReceiptError("visual room gate score keys are not exact")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 7
        for value in scores.values()
    ):
        raise BenchmarkReceiptError("visual room gate has a malformed/below-threshold score")
    reviews = visual.get("review_images")
    if not isinstance(reviews, list) or len(reviews) != 3:
        raise BenchmarkReceiptError("benchmark needs exactly three bound review images")
    review_paths = [
        _regular_file(Path(item), label="benchmark review image")
        for item in reviews
        if isinstance(item, str)
    ]
    if len(review_paths) != 3 or len({_sha256(path) for path in review_paths}) != 3:
        raise BenchmarkReceiptError("benchmark review images are missing or byte-identical")

    status_artifacts = {
        "pipeline code contract": run / "pipeline_code_contract.json",
        "input manifest validation": run / "input_manifest_validation.json",
        "materials contract": run / "materials_contract_validation.json",
        "Artiverse preparation contract": run / "artiverse_preparation_validation.json",
        "Artiverse visual-resource preflight": run / "artiverse_visual_resources.json",
        "articulated router validation": run / "articulated_router_validation.json",
        "ObjectThor offline preflight": preflight / "objathor_retrieval_offline.json",
        "SAM3D offline preflight": preflight / "sam3d_offline_load.json",
        "SAM3D full offline generation preflight": sam3d_generation_paths[
            "receipt"
        ],
        "VLM reference-image preflight": preflight / "vlm_vision_smoke.json",
        "floor layout gate": scene / "quality_gates" / "floor_plan_layout.json",
        "native reference layout seed": scene
        / "quality_gates"
        / "reference_school_layout_seed.json",
        "room prompt binding": scene / "quality_gates" / "room_prompt_binding.json",
    }
    status_documents = {
        label: _require_pass(path, label=label)
        for label, path in status_artifacts.items()
    }
    _validate_sam3d_preflight(status_documents["SAM3D offline preflight"])
    _validate_artiverse_visual_resources(
        status_documents["Artiverse visual-resource preflight"]
    )
    generation_failures = sam3d_generation.verify_bound_receipt(
        status_documents["SAM3D full offline generation preflight"],
        input_path=repo / sam3d_generation.CANONICAL_INPUT_RELATIVE,
        artifact_dir=sam3d_generation_paths["artifact_dir"],
    )
    if generation_failures:
        raise BenchmarkReceiptError(
            "SAM3D full offline generation preflight is invalid: "
            + "; ".join(generation_failures)
        )
    for label in ("pipeline code contract", "input manifest validation"):
        binding = status_documents[label].get("external_input_binding")
        if not isinstance(binding, dict) or any(
            binding.get(key) != value
            for key, value in EXPECTED_EXTERNAL_INPUT_BINDING.items()
        ):
            raise BenchmarkReceiptError(
                f"{label} does not bind the immutable prompt/reference anchors"
            )

    seed_receipt = status_documents["native reference layout seed"]
    if (
        seed_receipt.get("schema_version") != reference_layout.SEED_SCHEMA_VERSION
        or seed_receipt.get("profile") != reference_layout.PROFILE
        or seed_receipt.get("implementation")
        != "native_scenesmith_deterministic_reference_layout"
    ):
        raise BenchmarkReceiptError("native reference layout seed receipt is malformed")
    layout_document = _load_json(scene / "house_layout.json", label="house layout")
    if seed_receipt.get(
        "structural_layout_sha256"
    ) != reference_layout._structural_layout_sha256(layout_document):
        raise BenchmarkReceiptError(
            "house layout differs structurally from the native reference seed"
        )
    try:
        seed_artifacts = reference_layout._tree_manifest(
            scene, ("floor_plans", "room_geometry", "package.xml")
        )
    except reference_layout.SeedError as exc:
        raise BenchmarkReceiptError(f"native floor geometry is invalid: {exc}") from exc
    if (
        seed_receipt.get("artifacts") != seed_artifacts
        or seed_receipt.get("artifact_count") != len(seed_artifacts)
        or seed_receipt.get("artifact_manifest_sha256")
        != reference_layout._manifest_sha256(seed_artifacts)
    ):
        raise BenchmarkReceiptError("native floor geometry differs from its seed receipt")

    policy_path = run / "resolved_asset_policy.json"
    _validate_asset_policy(policy_path)

    fixed_artifacts = {
        "full-quality specification": repo / "CODEX_SCENESMITH_FULL_QUALITY_PIPELINE.md",
        "production runner": repo / "remote_jobs" / "run_full_quality_school_sqz.sh",
        "original prompt": inputs / "prompt_original.txt",
        "scene contract appendix": inputs / "scene_contract_appendix.txt",
        "effective prompt": inputs / "prompt.txt",
        "prompt CSV": inputs / "prompt.csv",
        "input manifest": inputs / "input_manifest.json",
        "reference image": inputs / "reference.png",
        "SAM3D generation GLB": sam3d_generation_paths["glb"],
        "SAM3D generation mask": sam3d_generation_paths["mask"],
        "SAM3D generation masked image": sam3d_generation_paths["masked_image"],
        "resolved asset policy": policy_path,
        "house layout": scene / "house_layout.json",
        "final room state": scene / f"room_{ROOM_ID}" / "scene_states" / "final_scene" / "scene_state.json",
        "final room blend": scene / f"room_{ROOM_ID}" / "scene_states" / "final_scene" / "scene.blend",
        "deterministic room gate": deterministic_path,
        "visual room gate": visual_path,
        "benchmark event log": events_path,
        "benchmark GPU samples": gpu_samples_path,
    }
    artifacts = [
        _artifact(path, label=label)
        for label, path in {**status_artifacts, **fixed_artifacts}.items()
    ]
    recorded_paths = {item["path"] for item in artifacts}
    for label, path, expected_digest in _bound_evidence_files(visual.get("evidence")):
        regular = _regular_file(path, label=label)
        actual_digest = _sha256(regular)
        if actual_digest != expected_digest:
            raise BenchmarkReceiptError(f"visual bound evidence changed: {regular}")
        if os.fspath(regular) not in recorded_paths:
            artifacts.append(_artifact(regular, label=label))
            recorded_paths.add(os.fspath(regular))

    events = _events(events_path)
    gpu = _gpu_samples(gpu_samples_path)
    payload: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "status": "pass",
        "scope": "single_room_full_quality_benchmark_not_final_acceptance",
        "room_id": ROOM_ID,
        "run_attempt_id": run_attempt_id,
        "execution_mode": "benchmark_classroom_01",
        "asset_policy": "generated_sam3d_artvip_compulsory_artiverse_objathor",
        "asset_policy_verified": True,
        "hssd_used": False,
        "whole_house_assembly_performed": False,
        "whole_house_export_performed": False,
        "paid_api_usage": {
            "calls_expected": True,
            "cost_measured": False,
            "cost_usd": None,
            "reason": "SceneSmith does not expose one authoritative run-level usage ledger; inspect provider billing and request logs separately.",
        },
        "timing": {
            "started_at_utc": events[0]["timestamp_utc"],
            "completed_at_utc": events[-1]["timestamp_utc"],
            "elapsed_seconds": events[-1]["epoch"] - events[0]["epoch"],
            "events": events,
            "reused_existing_room_evidence": any(
                item["name"] == "room_evidence_reused" for item in events
            ),
        },
        "gpu": gpu,
        "artifacts": sorted(artifacts, key=lambda item: (item["label"], item["path"])),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["attestation_sha256"] = hashlib.sha256(canonical).hexdigest()

    destination = output.resolve()
    if destination.parent != run:
        raise BenchmarkReceiptError("benchmark receipt must be directly under run_dir")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{run_attempt_id}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--gpu-samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-attempt-id", required=True)
    args = parser.parse_args(argv)
    result = create_receipt(
        repo_dir=args.repo_dir,
        run_dir=args.run_dir,
        scene_dir=args.scene_dir,
        input_dir=args.input_dir,
        events_path=args.events,
        gpu_samples_path=args.gpu_samples,
        output=args.output,
        run_attempt_id=args.run_attempt_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
