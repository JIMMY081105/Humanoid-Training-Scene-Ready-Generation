"""Assemble a completed room-sharded SceneSmith house and render outlook images."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

try:
    from .artiverse_contract import (
        ArtiverseAuthority,
        ArtiverseContractError,
        load_artiverse_authority,
        sha256_directory_tree,
        sha256_file,
        validate_usage_manifest,
    )
    from .cutaway_evidence_contract import (
        VIEW_NAMES as CUTAWAY_VIEW_NAMES,
        validate_cutaway_evidence,
    )
    from .house_cutaway_evidence_contract import (
        VIEW_NAMES as HOUSE_CUTAWAY_VIEW_NAMES,
        validate_house_cutaway_evidence,
    )
    from .school_room_contract import ARTICULATED_ROLE_RULES
    from .school_room_contract import PROFILE as SCHOOL_CONTRACT_PROFILE
    from .school_room_contract import collect_required_articulated_roles
    from .factory_room_contract import ROLE_RULES as FACTORY_ARTICULATED_ROLE_RULES
    from .factory_room_contract import PROFILE as FACTORY_CONTRACT_PROFILE
    from .factory_room_contract import collect_required_articulated_roles as collect_factory_articulated_roles
    from .factory_room_contract import load_contract as load_factory_contract
    from .factory_room_contract import room_ids as factory_room_ids
    from .factory_room_visual_self_exam import _validate_existing as validate_existing_factory_visual
    from .room_visual_self_exam import (
        ROOM_SCORE_KEYS,
        SCHOOL_GATE_THRESHOLD,
        SCHOOL_VLM_BACKEND,
        SCHOOL_VLM_MODEL,
        _room_judge_instruction,
        _room_messages,
        _validate_derivation_receipt,
        _validate_vlm_request_record,
        _vlm_request_contract,
        room_visual_requirements,
        validate_requirement_evidence,
    )
except ImportError:  # Direct execution: python scripts/assemble_final_house_and_render.py
    from artiverse_contract import (  # type: ignore[no-redef]
        ArtiverseAuthority,
        ArtiverseContractError,
        load_artiverse_authority,
        sha256_directory_tree,
        sha256_file,
        validate_usage_manifest,
    )
    from cutaway_evidence_contract import (  # type: ignore[no-redef]
        VIEW_NAMES as CUTAWAY_VIEW_NAMES,
        validate_cutaway_evidence,
    )
    from house_cutaway_evidence_contract import (  # type: ignore[no-redef]
        VIEW_NAMES as HOUSE_CUTAWAY_VIEW_NAMES,
        validate_house_cutaway_evidence,
    )
    from school_room_contract import ARTICULATED_ROLE_RULES  # type: ignore[no-redef]
    from school_room_contract import PROFILE as SCHOOL_CONTRACT_PROFILE  # type: ignore[no-redef]
    from school_room_contract import collect_required_articulated_roles  # type: ignore[no-redef]
    from factory_room_contract import ROLE_RULES as FACTORY_ARTICULATED_ROLE_RULES  # type: ignore[no-redef]
    from factory_room_contract import PROFILE as FACTORY_CONTRACT_PROFILE  # type: ignore[no-redef]
    from factory_room_contract import collect_required_articulated_roles as collect_factory_articulated_roles  # type: ignore[no-redef]
    from factory_room_contract import load_contract as load_factory_contract  # type: ignore[no-redef]
    from factory_room_contract import room_ids as factory_room_ids  # type: ignore[no-redef]
    from factory_room_visual_self_exam import _validate_existing as validate_existing_factory_visual  # type: ignore[no-redef]
    from room_visual_self_exam import (  # type: ignore[no-redef]
        ROOM_SCORE_KEYS,
        SCHOOL_GATE_THRESHOLD,
        SCHOOL_VLM_BACKEND,
        SCHOOL_VLM_MODEL,
        _room_judge_instruction,
        _room_messages,
        _validate_derivation_receipt,
        _validate_vlm_request_record,
        _vlm_request_contract,
        room_visual_requirements,
        validate_requirement_evidence,
    )


ROOMS = [
    "main_corridor",
    "classroom_01",
    "classroom_02",
    "classroom_03",
    "classroom_04",
    "classroom_05",
    "classroom_06",
    "classroom_07",
    "classroom_08",
    "lobby_waiting_area",
    "stair_entrance",
    "teacher_office",
    "storage_room",
    "male_restroom",
    "female_restroom",
    "cleaning_closet",
    "library",
    "emergency_exit_corridor",
]

LOGGER = logging.getLogger(__name__)
HASH_CHUNK_SIZE = 1024 * 1024
SHA256_HEX_LENGTH = 64
DERIVATION_SCHEMA_ID = "scenesmith_state_blend_render_derivation_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _derivation_file_record(path: Path, *, canonical_path: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Derivation source is missing or empty: {path}")
    return {
        "path": str((canonical_path or path).resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _make_derivation_receipt(
    *, state_record: dict[str, Any], blend_record: dict[str, Any], views: list[dict[str, Any]]
) -> dict[str, Any]:
    payload = {
        "schema_id": DERIVATION_SCHEMA_ID,
        "schema_version": 1,
        "algorithm": "sha256",
        "source_state": state_record,
        "source_blend": blend_record,
        "renders": [
            {
                "view_name": view["view_name"],
                "path": view["image"],
                "size_bytes": view["image_size_bytes"],
                "sha256": view["image_sha256"],
            }
            for view in views
        ],
    }
    return {
        **payload,
        "attestation": {
            "algorithm": "sha256",
            "sha256": hashlib.sha256(_canonical_json(payload)).hexdigest(),
        },
    }


def _load_cfg(repo_dir: Path, run_dir: Path, csv_path: str, run_name: str):
    import hydra

    from omegaconf import OmegaConf, open_dict
    from scenesmith.utils.omegaconf import register_resolvers

    register_resolvers()
    config_dir = repo_dir.resolve() / "configurations"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                f"+name={run_name}",
                f"experiment.csv_path={csv_path}",
                "experiment.num_workers=1",
                "experiment.pipeline.start_stage=wall_mounted",
                "experiment.pipeline.stop_stage=manipuland",
                "experiment.pipeline.parallel_rooms=false",
                "experiment.pipeline.final_assembly_policy=external_artiverse_gated",
                "floor_plan_agent.mode=house",
                "++codex.enabled=false",
            ],
        )

    with open_dict(cfg):
        cfg.experiment._name = "indoor_scene_generation"
        cfg.floor_plan_agent._name = "stateful_floor_plan_agent"
        cfg.furniture_agent._name = "stateful_furniture_agent"
        cfg.wall_agent._name = "stateful_wall_agent"
        cfg.ceiling_agent._name = "stateful_ceiling_agent"
        cfg.manipuland_agent._name = "stateful_manipuland_agent"
        cfg.experiment.output_dir = str(run_dir.resolve())

    OmegaConf.resolve(cfg)
    return cfg


def _verify_final_rooms(scene_dir: Path, room_ids: list[str]) -> list[str]:
    missing: list[str] = []
    for room_id in room_ids:
        final_state = (
            scene_dir
            / f"room_{room_id}"
            / "scene_states"
            / "final_scene"
            / "scene_state.json"
        )
        if not final_state.exists():
            missing.append(room_id)
    return missing


def _room_ids_from_layout(layout: Any) -> list[str]:
    room_ids = [room.room_id for room in layout.placed_rooms]
    return room_ids or ROOMS


def _room_ids_from_layout_data(layout_data: dict[str, Any]) -> list[str]:
    """Read room IDs without instantiating HouseLayout, which may write package.xml."""
    room_ids = [
        str(room["room_id"])
        for room in layout_data.get("placed_rooms", [])
        if isinstance(room, dict) and room.get("room_id")
    ]
    if room_ids:
        return room_ids

    for room in layout_data.get("rooms", []):
        if not isinstance(room, dict):
            continue
        room_id = room.get("room_id") or room.get("id")
        if room_id:
            room_ids.append(str(room_id))
    return room_ids or ROOMS


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_evidence_entry(
    entry: Any,
    *,
    label: str,
    failures: list[str],
    expected_path: Path | None = None,
) -> tuple[str, str] | None:
    """Rehash one bound file and optionally require its canonical scene path."""

    if not isinstance(entry, dict):
        failures.append(f"{label}: missing or malformed evidence record")
        return None
    path_value = entry.get("path")
    expected_hash = entry.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        failures.append(f"{label}: evidence path is missing")
        return None
    if not _is_sha256(expected_hash):
        failures.append(f"{label}: evidence SHA-256 is missing or malformed")
        return None

    path = Path(path_value)
    if not path.is_absolute():
        failures.append(f"{label}: evidence path is not absolute: {path}")
        return None
    resolved = path.resolve()
    if expected_path is not None and resolved != expected_path.resolve():
        failures.append(
            f"{label}: evidence path does not match canonical file; "
            f"recorded={resolved}, expected={expected_path.resolve()}"
        )
        return None
    if not resolved.is_file():
        failures.append(f"{label}: evidence file is missing: {resolved}")
        return None
    try:
        actual_hash = _sha256_file(resolved)
    except OSError as exc:
        failures.append(f"{label}: cannot read evidence file {resolved}: {exc}")
        return None
    if actual_hash != expected_hash:
        failures.append(
            f"{label}: SHA-256 mismatch for {resolved}; "
            f"gate={expected_hash}, current={actual_hash}"
        )
        return None
    return str(resolved), expected_hash


def _verify_room_gate_evidence(
    *,
    scene_dir: Path,
    room_id: str,
    result: dict[str, Any],
    contract_profile: str | None = None,
    input_dir: Path | None = None,
) -> tuple[list[str], tuple[str, str] | None]:
    failures: list[str] = []
    evidence = result.get("evidence")
    if not isinstance(evidence, dict):
        return [f"{room_id}: passing gate has no evidence manifest"], None
    if evidence.get("schema_version") != 1:
        failures.append(f"{room_id}: unsupported evidence schema version")
    if evidence.get("algorithm") != "sha256":
        failures.append(f"{room_id}: evidence algorithm is not sha256")

    expected_state = (
        scene_dir
        / f"room_{room_id}"
        / "scene_states"
        / "final_scene"
        / "scene_state.json"
    )
    _verify_evidence_entry(
        evidence.get("scene_state"),
        label=f"{room_id}/scene_state",
        failures=failures,
        expected_path=expected_state,
    )
    _verify_evidence_entry(
        evidence.get("house_layout"),
        label=f"{room_id}/house_layout",
        failures=failures,
        expected_path=scene_dir / "house_layout.json",
    )
    reference_binding = _verify_evidence_entry(
        evidence.get("reference_image"),
        label=f"{room_id}/reference_image",
        failures=failures,
    )
    deterministic_entry = evidence.get("deterministic_gate")
    deterministic_path = (
        scene_dir
        / "quality_gates"
        / "room_self_exam_deterministic"
        / f"{room_id}.json"
    )
    deterministic_document: dict[str, Any] | None = None
    if contract_profile or deterministic_entry is not None:
        _verify_evidence_entry(
            deterministic_entry,
            label=f"{room_id}/deterministic_gate",
            failures=failures,
            expected_path=deterministic_path if contract_profile else None,
        )
        if contract_profile and deterministic_path.is_file():
            try:
                loaded = json.loads(deterministic_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    deterministic_document = loaded
                else:
                    failures.append(
                        f"{room_id}: deterministic gate JSON is not an object"
                    )
            except (OSError, json.JSONDecodeError) as exc:
                failures.append(
                    f"{room_id}: cannot reconstruct deterministic VLM input: {exc}"
                )
    if contract_profile:
        if input_dir is None:
            failures.append(f"{room_id}: contract input directory was not supplied")
        else:
            _verify_evidence_entry(
                evidence.get("effective_prompt"),
                label=f"{room_id}/effective_prompt",
                failures=failures,
                expected_path=input_dir / "prompt.txt",
            )
            _verify_evidence_entry(
                evidence.get("input_manifest"),
                label=f"{room_id}/input_manifest",
                failures=failures,
                expected_path=input_dir / "input_manifest.json",
            )
            _verify_evidence_entry(
                evidence.get("prompt_binding"),
                label=f"{room_id}/prompt_binding",
                failures=failures,
                expected_path=(
                    scene_dir / "quality_gates" / "room_prompt_binding.json"
                ),
            )

        contract_review_images = [
            (
                scene_dir
                / "review"
                / "room_review_renders"
                / f"{room_id}_{view_name}.png"
            )
            for view_name in CUTAWAY_VIEW_NAMES
        ]
        source_blend = (
            scene_dir
            / f"room_{room_id}"
            / "scene_states"
            / "final_scene"
            / "scene.blend"
        )
        cutaway_path = (
            scene_dir
            / "review"
            / "room_review_renders"
            / f"{room_id}_cutaway_evidence.json"
        )
        _verify_evidence_entry(
            evidence.get("source_blend"),
            label=f"{room_id}/source_blend",
            failures=failures,
            expected_path=source_blend,
        )
        _verify_evidence_entry(
            evidence.get("cutaway_evidence"),
            label=f"{room_id}/cutaway_evidence",
            failures=failures,
            expected_path=cutaway_path,
        )
        failures.extend(
            f"{room_id}/cutaway_evidence: {failure}"
            for failure in validate_cutaway_evidence(
                cutaway_path,
                room_id=room_id,
                review_images=contract_review_images,
                source_blend=source_blend,
            )
        )
        failures.extend(
            f"{room_id}/derivation: {failure}"
            for failure in _validate_derivation_receipt(
                cutaway_path,
                scene_state=expected_state,
                source_blend=source_blend,
                review_images=contract_review_images,
            )
        )
    else:
        contract_review_images = None

    review_entries = evidence.get("review_images")
    review_bindings: list[tuple[str, str]] = []
    if not isinstance(review_entries, list) or (
        len(review_entries) != 3 if contract_profile else len(review_entries) < 3
    ):
        failures.append(
            f"{room_id}: passing gate must bind "
            + ("exactly three canonical cutaway images" if contract_profile else "at least three review images")
        )
    else:
        for index, entry in enumerate(review_entries):
            binding = _verify_evidence_entry(
                entry,
                label=f"{room_id}/review_images[{index}]",
                failures=failures,
                expected_path=(
                    contract_review_images[index]
                    if contract_review_images is not None
                    else None
                ),
            )
            if binding is not None:
                review_bindings.append(binding)
        if len({path for path, _digest in review_bindings}) != len(review_bindings):
            failures.append(f"{room_id}: review-image evidence paths are not distinct")
        if len({digest for _path, digest in review_bindings}) != len(review_bindings):
            failures.append(f"{room_id}: review-image evidence hashes are not distinct")

    if contract_profile and reference_binding is not None:
        reference_hash = reference_binding[1]
        review_hashes = [digest for _path, digest in review_bindings]
        expected_vlm_record: dict[str, Any] | None = None
        if (
            input_dir is not None
            and deterministic_document is not None
            and len(review_bindings) == 3
        ):
            try:
                reference_path = Path(reference_binding[0])
                review_paths = [Path(path) for path, _digest in review_bindings]
                immutable_prompt = (input_dir / "prompt.txt").read_text(
                    encoding="utf-8"
                )
                instruction = _room_judge_instruction(
                    room_id=room_id,
                    room_prompt=str(result.get("room_prompt", "")),
                    deterministic_result=deterministic_document,
                    threshold=SCHOOL_GATE_THRESHOLD,
                    review_count=3,
                    immutable_effective_prompt=immutable_prompt,
                    contract_profile=contract_profile,
                    visual_requirements=room_visual_requirements(room_id),
                )
                messages = _room_messages(
                    instruction, reference_path, review_paths
                )
                expected_vlm_record = _vlm_request_contract(
                    messages=messages,
                    room_id=room_id,
                    model=SCHOOL_VLM_MODEL,
                    backend=SCHOOL_VLM_BACKEND,
                    threshold=SCHOOL_GATE_THRESHOLD,
                    requirement_keys=list(room_visual_requirements(room_id)),
                    reference_image=reference_path,
                    review_images=review_paths,
                )
            except (OSError, ValueError, TypeError) as exc:
                failures.append(
                    f"{room_id}: cannot reconstruct canonical VLM request: {exc}"
                )
        failures.extend(
            f"{room_id}/vlm_request: {failure}"
            for failure in _validate_vlm_request_record(
                evidence.get("vlm_request"),
                room_id=room_id,
                threshold=SCHOOL_GATE_THRESHOLD,
                requirement_keys=list(room_visual_requirements(room_id)),
                reference_sha256=reference_hash,
                review_sha256=review_hashes,
                expected_record=expected_vlm_record,
            )
        )
        if input_dir is not None:
            try:
                manifest = json.loads(
                    (input_dir / "input_manifest.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                failures.append(f"{room_id}: cannot read input manifest: {exc}")
            else:
                if manifest.get("reference_image_sha256") != reference_hash:
                    failures.append(
                        f"{room_id}: reference hash differs from input manifest"
                    )

    recorded_reviews = result.get("review_images")
    if not isinstance(recorded_reviews, list) or not all(
        isinstance(path, str) and Path(path).is_absolute() for path in recorded_reviews
    ):
        failures.append(f"{room_id}: top-level review image paths are malformed")
    elif [str(Path(path).resolve()) for path in recorded_reviews] != [
        path for path, _digest in review_bindings
    ]:
        failures.append(
            f"{room_id}: evidence review-image paths differ from the reviewed paths"
        )

    recorded_reference = result.get("reference_image")
    if (
        reference_binding is not None
        and (
            not isinstance(recorded_reference, str)
            or not Path(recorded_reference).is_absolute()
            or str(Path(recorded_reference).resolve()) != reference_binding[0]
        )
    ):
        failures.append(
            f"{room_id}: evidence reference path differs from the reviewed reference"
        )
    return failures, reference_binding


def _verify_room_gates(
    gate_dir: Path,
    room_ids: list[str],
    scene_dir: Path,
    contract_profile: str | None = None,
    input_dir: Path | None = None,
    factory_contract: Path | None = None,
) -> list[str]:
    if contract_profile == FACTORY_CONTRACT_PROFILE:
        if factory_contract is None:
            return ["factory contract path was not supplied to assembly"]
        try:
            contract = load_factory_contract(factory_contract)
            expected = list(factory_room_ids(contract))
        except Exception as exc:
            return [f"cannot load immutable factory contract: {exc}"]
        failures: list[str] = []
        if room_ids != expected:
            failures.append(
                f"assembly layout room order differs from exact factory contract: {room_ids}"
            )
        for room_id in expected:
            failures.extend(
                validate_existing_factory_visual(
                    gate_dir / f"{room_id}.json",
                    room_id=room_id,
                    contract=contract,
                    threshold=SCHOOL_GATE_THRESHOLD,
                )
            )
        return failures

    failures: list[str] = []
    shared_reference: tuple[str, str] | None = None
    for room_id in room_ids:
        gate_path = gate_dir / f"{room_id}.json"
        if not gate_path.exists():
            failures.append(f"{room_id}: missing gate JSON")
            continue
        try:
            with gate_path.open(encoding="utf-8") as f:
                result = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"{room_id}: cannot read gate JSON: {exc}")
            continue
        if not isinstance(result, dict):
            failures.append(f"{room_id}: gate JSON is not an object")
            continue
        if result.get("status") != "pass":
            failures.append(f"{room_id}: status={result.get('status')}")
            continue
        if result.get("room_id") != room_id:
            failures.append(
                f"{room_id}: gate room_id={result.get('room_id')!r} does not match"
            )
        if contract_profile and result.get("contract_profile") != contract_profile:
            failures.append(
                f"{room_id}: gate contract_profile={result.get('contract_profile')!r} "
                f"does not match {contract_profile!r}"
            )
        if contract_profile:
            raw_scores = result.get("scores")
            if (
                result.get("threshold") != SCHOOL_GATE_THRESHOLD
                or not isinstance(raw_scores, dict)
                or set(raw_scores) != set(ROOM_SCORE_KEYS)
                or any(
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not SCHOOL_GATE_THRESHOLD <= float(score) <= 10.0
                    for score in raw_scores.values()
                )
            ):
                failures.append(
                    f"{room_id}: score keys or threshold differ from school contract"
                )
            visual = result.get("visual_assessment")
            review_images = result.get("review_images")
            if not isinstance(visual, dict) or not isinstance(review_images, list):
                failures.append(
                    f"{room_id}: visual assessment or reviewed-image list is malformed"
                )
            else:
                visual_scores = visual.get("scores")
                if not isinstance(visual_scores, dict) or set(visual_scores) != set(
                    ROOM_SCORE_KEYS
                ) or any(
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not SCHOOL_GATE_THRESHOLD <= float(score) <= 10.0
                    for score in visual_scores.values()
                ):
                    failures.append(
                        f"{room_id}: visual score keys differ from school contract"
                    )
                if visual.get("critical_issues") != []:
                    failures.append(
                        f"{room_id}: visual assessment contains critical issues"
                    )
                try:
                    _normalized, checklist_failures = validate_requirement_evidence(
                        visual.get("requirement_evidence"),
                        room_visual_requirements(room_id),
                        review_count=len(review_images),
                    )
                except ValueError as exc:
                    failures.append(
                        f"{room_id}: itemized visual checklist is malformed: {exc}"
                    )
                else:
                    failures.extend(
                        f"{room_id}: {failure}" for failure in checklist_failures
                    )

        evidence_failures, reference_binding = _verify_room_gate_evidence(
            scene_dir=scene_dir,
            room_id=room_id,
            result=result,
            contract_profile=contract_profile,
            input_dir=input_dir,
        )
        failures.extend(evidence_failures)
        if reference_binding is not None:
            if shared_reference is None:
                shared_reference = reference_binding
            elif reference_binding != shared_reference:
                failures.append(
                    f"{room_id}: reference evidence differs from other passing rooms"
                )
    return failures


def _load_room(scene_dir: Path, room_id: str):
    from scenesmith.agent_utils.room import RoomScene

    room_dir = scene_dir / f"room_{room_id}"
    state_path = room_dir / "scene_states" / "final_scene" / "scene_state.json"
    with state_path.open() as f:
        state = json.load(f)

    room = RoomScene(room_geometry=None, scene_dir=room_dir, room_id=room_id)
    room.restore_from_state_dict(state)
    return room


def _iter_state_objects(state: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    objects = state.get("objects", {})
    if isinstance(objects, dict):
        return [
            (str(object_id), obj)
            for object_id, obj in objects.items()
            if isinstance(obj, dict)
        ]
    if isinstance(objects, list):
        return [
            (str(obj.get("object_id", index)), obj)
            for index, obj in enumerate(objects)
            if isinstance(obj, dict)
        ]
    return []


def _artiverse_records_from_room_state(
    room_state: dict[str, Any],
    room_id: str,
    room_dir: Path,
    authority: ArtiverseAuthority,
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for state_object_id, obj in _iter_state_objects(room_state):
        metadata = obj.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        articulated_source = str(metadata.get("articulated_source", "")).lower()
        if articulated_source != "artiverse":
            continue

        object_id = str(obj.get("object_id") or state_object_id)
        articulated_id = str(metadata.get("articulated_id") or "").strip()
        asset_source = str(metadata.get("asset_source") or "").lower()
        sdf_value = obj.get("sdf_path")
        sdf_path = Path(str(sdf_value)) if sdf_value else None
        resolved_sdf = None
        if sdf_path is not None:
            resolved_sdf = (
                sdf_path.resolve()
                if sdf_path.is_absolute()
                else (room_dir / sdf_path).resolve()
            )

        object_failures = []
        if asset_source != "articulated":
            object_failures.append(
                f"asset_source={asset_source or '<missing>'}, expected exactly articulated"
            )
        if metadata.get("is_articulated") is not True:
            object_failures.append("is_articulated is not true")
        if not articulated_id:
            object_failures.append("articulated_id is missing")
        if resolved_sdf is None:
            object_failures.append("sdf_path is missing")
        elif not resolved_sdf.is_file():
            object_failures.append(f"SDF does not exist: {resolved_sdf}")

        expected: dict[str, Any] | None = None
        if articulated_id:
            try:
                expected = authority.asset(articulated_id)
            except ArtiverseContractError as exc:
                object_failures.append(str(exc))

        copied_sdf_sha256 = None
        copied_tree_sha256 = None
        if resolved_sdf is not None and resolved_sdf.is_file():
            expected_copy_root = (room_dir / "generated_assets" / "sdf").resolve()
            try:
                resolved_sdf.relative_to(expected_copy_root)
            except ValueError:
                object_failures.append(
                    f"placed SDF is outside the room articulated-copy root: {resolved_sdf}"
                )
            copied_sdf_sha256 = sha256_file(resolved_sdf)
            copied_tree_sha256 = sha256_directory_tree(resolved_sdf.parent)

        if expected is not None:
            source_sdf = expected["source_sdf_path"]
            source_hash = expected["source_sdf_sha256"]
            source_tree_hash = expected["source_tree_sha256"]
            recorded_source = Path(
                str(metadata.get("articulated_source_sdf_path", ""))
            ).resolve()
            if recorded_source != source_sdf:
                object_failures.append(
                    f"source SDF path does not match prepared index: {recorded_source}"
                )
            if metadata.get("articulated_source_sdf_sha256") != source_hash:
                object_failures.append("source SDF hash does not match prepared index")
            if metadata.get("articulated_source_tree_sha256") != source_tree_hash:
                object_failures.append("source asset-tree hash does not match prepared index")
            if resolved_sdf is not None and resolved_sdf.name != source_sdf.name:
                object_failures.append("placed SDF filename differs from indexed source")
            if metadata.get("articulated_copied_sdf_sha256") != copied_sdf_sha256:
                object_failures.append("placed SDF content hash is missing or stale")
            if metadata.get("articulated_copied_tree_sha256") != copied_tree_sha256:
                object_failures.append("placed asset-tree hash is missing or stale")

        if object_failures:
            failures.append(
                f"{room_id}/{object_id}: invalid Artiverse provenance: "
                + "; ".join(object_failures)
            )
            continue

        records.append(
            {
                "room_id": room_id,
                "object_id": object_id,
                "articulated_id": articulated_id,
                "asset_source": asset_source,
                "articulated_source": "artiverse",
                "sdf_path": str(resolved_sdf),
                "sdf_sha256": copied_sdf_sha256,
                "sdf_tree_sha256": copied_tree_sha256,
                "source_sdf_path": str(expected["source_sdf_path"]),
                "source_sdf_sha256": expected["source_sdf_sha256"],
                "source_tree_sha256": expected["source_tree_sha256"],
            }
        )
    return records, failures


def _collect_artiverse_from_room_states(
    scene_dir: Path, room_ids: list[str], authority: ArtiverseAuthority
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for room_id in room_ids:
        room_dir = scene_dir / f"room_{room_id}"
        state_path = room_dir / "scene_states" / "final_scene" / "scene_state.json"
        try:
            with state_path.open(encoding="utf-8") as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"{room_id}: cannot read final room state: {exc}")
            continue
        room_records, room_failures = _artiverse_records_from_room_state(
            state, room_id, room_dir, authority
        )
        records.extend(room_records)
        failures.extend(room_failures)
    return records, failures


def _collect_artiverse_from_house_state(
    house_state_path: Path,
    scene_dir: Path,
    authority: ArtiverseAuthority,
) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        with house_state_path.open(encoding="utf-8") as f:
            house_state = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return [], [f"cannot read combined house state: {exc}"]

    rooms = house_state.get("rooms", {})
    if not isinstance(rooms, dict):
        return [], ["combined house state has no room mapping"]

    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for room_id, room_state in rooms.items():
        if not isinstance(room_state, dict):
            failures.append(f"{room_id}: combined room state is not an object")
            continue
        room_records, room_failures = _artiverse_records_from_room_state(
            room_state,
            str(room_id),
            scene_dir / f"room_{room_id}",
            authority,
        )
        records.extend(room_records)
        failures.extend(room_failures)
    return records, failures


def _load_final_room_states(
    scene_dir: Path, room_ids: list[str]
) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for room_id in room_ids:
        path = (
            scene_dir
            / f"room_{room_id}"
            / "scene_states"
            / "final_scene"
            / "scene_state.json"
        )
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot load final room state for {room_id}: {exc}") from exc
        if not isinstance(state, dict):
            raise RuntimeError(f"Final room state is not an object: {path}")
        states[room_id] = state
    return states


def _load_combined_room_states(house_state_path: Path) -> dict[str, dict[str, Any]]:
    try:
        house_state = json.loads(house_state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load combined house state: {exc}") from exc
    rooms = house_state.get("rooms")
    if not isinstance(rooms, dict):
        raise RuntimeError("Combined house state has no room mapping")
    return {
        str(room_id): state
        for room_id, state in rooms.items()
        if isinstance(state, dict)
    }


def _require_articulated_roles(
    role_result: dict[str, Any],
    valid_artiverse_records: list[dict[str, Any]],
    context: str,
    *,
    profile: str,
    role_rules: Mapping[str, Any],
) -> None:
    if role_result.get("status") != "pass":
        raise RuntimeError(
            f"Refusing final assembly: articulated-role contract failed in {context}: "
            + "; ".join(str(issue) for issue in role_result.get("critical_issues", []))
        )
    if role_result.get("profile") != profile:
        raise RuntimeError(
            f"Refusing final assembly: articulated-role profile is invalid in {context}"
        )
    expected_roles = set(role_rules)
    role_records = role_result.get("roles")
    if not isinstance(role_records, dict):
        raise RuntimeError(
            f"Refusing final assembly: articulated-role records are malformed in {context}"
        )
    missing_roles = sorted(
        role
        for role in expected_roles
        if not isinstance(role_records.get(role), list) or not role_records[role]
    )
    if missing_roles:
        raise RuntimeError(
            f"Refusing final assembly: missing required roles in {context}: {missing_roles}"
        )
    for role in sorted(expected_roles):
        for record in role_records[role]:
            if not isinstance(record, dict):
                raise RuntimeError(
                    f"Refusing final assembly: malformed record for required role {role}"
                )
            source = str(record.get("articulated_source", ""))
            invalid = (
                record.get("asset_source") != "articulated"
                or source not in {"artiverse", "artvip"}
                or not str(record.get("articulated_id", "")).strip()
                or not str(record.get("sdf_path", "")).strip()
            )
            if invalid:
                raise RuntimeError(
                    f"Refusing final assembly: required role {role} has forged or "
                    "incomplete runtime provenance"
                )

    computed_artiverse_roles = {
        role
        for role in expected_roles
        if any(
            record.get("articulated_source") == "artiverse"
            for record in role_records[role]
        )
    }
    reported_artiverse_roles = set(role_result.get("artiverse_roles", []))
    if reported_artiverse_roles != computed_artiverse_roles:
        raise RuntimeError(
            f"Refusing final assembly: Artiverse role summary is stale in {context}"
        )
    if not computed_artiverse_roles:
        raise RuntimeError(
            f"Refusing final assembly: no required role uses Artiverse in {context}"
        )

    valid_artiverse = {_record_identity(record) for record in valid_artiverse_records}
    for role in sorted(computed_artiverse_roles):
        records = role_records[role]
        if not any(
            (
                str(record.get("room_id", "")),
                str(record.get("object_id", "")),
                str(record.get("articulated_id", "")),
            )
            in valid_artiverse
            for record in records
            if isinstance(record, dict)
        ):
            raise RuntimeError(
                f"Refusing final assembly: Artiverse role {role} lacks validated provenance"
            )


def _verify_articulated_role_survival(
    placed: dict[str, Any], final: dict[str, Any], role_rules: Mapping[str, Any]
) -> None:
    for role in role_rules:
        placed_records = placed.get("roles", {}).get(role, [])
        final_records = final.get("roles", {}).get(role, [])
        placed_ids = {
            (
                str(record.get("room_id", "")),
                str(record.get("object_id", "")),
                str(record.get("articulated_id", "")),
                str(record.get("articulated_source", "")),
            )
            for record in placed_records
        }
        final_ids = {
            (
                str(record.get("room_id", "")),
                str(record.get("object_id", "")),
                str(record.get("articulated_id", "")),
                str(record.get("articulated_source", "")),
            )
            for record in final_records
        }
        if not placed_ids.issubset(final_ids):
            raise RuntimeError(f"Combined house lost required articulated role {role}")


def _require_artiverse_records(
    records: list[dict[str, Any]], failures: list[str], context: str
) -> None:
    if failures:
        raise RuntimeError(
            f"Refusing final assembly: invalid Artiverse evidence in {context}: "
            + "; ".join(failures)
        )
    if not records:
        raise RuntimeError(
            f"Refusing final assembly: zero surviving Artiverse assets in {context}"
        )


def _record_identity(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record["room_id"]),
        str(record["object_id"]),
        str(record["articulated_id"]),
    )


def _verify_combined_survival(
    placed_records: list[dict[str, Any]], final_records: list[dict[str, Any]]
) -> None:
    placed = {_record_identity(record) for record in placed_records}
    final = {_record_identity(record) for record in final_records}
    missing = sorted(placed - final)
    if missing:
        raise RuntimeError(
            "Final combined house lost placed Artiverse assets: "
            + ", ".join("/".join(identity) for identity in missing)
        )


def _write_artiverse_usage_manifest(
    manifest_path: Path,
    placed_records: list[dict[str, Any]],
    final_records: list[dict[str, Any]],
    authority: ArtiverseAuthority,
    scene_dir: Path,
    room_ids: list[str],
    house_state_path: Path,
    canonical_house_state_path: Path,
    articulated_roles: dict[str, Any] | None = None,
) -> None:
    room_states = []
    for room_id in room_ids:
        path = (
            scene_dir
            / f"room_{room_id}"
            / "scene_states"
            / "final_scene"
            / "scene_state.json"
        ).resolve()
        room_states.append(
            {"room_id": room_id, "path": str(path), "sha256": sha256_file(path)}
        )
    manifest = {
        "schema_version": 2,
        "status": "pass",
        "dataset": "artiverse",
        "authority": authority.evidence(),
        "placed_asset_count": len(placed_records),
        "final_surviving_asset_count": len(final_records),
        "placed_asset_identifiers": sorted(
            {record["articulated_id"] for record in placed_records}
        ),
        "final_surviving_asset_identifiers": sorted(
            {record["articulated_id"] for record in final_records}
        ),
        "placed_assets": placed_records,
        "final_surviving_assets": final_records,
        "room_states": room_states,
        "house_state": {
            "path": str(canonical_house_state_path.resolve()),
            "sha256": sha256_file(house_state_path),
        },
        "required_articulated_roles": articulated_roles,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, manifest_path)


def _backup_existing(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = path.with_name(
        f"{path.name}.pre_final_assemble_backup_{stamp}_{uuid.uuid4().hex[:8]}"
    )
    LOGGER.info("Preserving existing %s as %s", path, backup)
    shutil.move(str(path), str(backup))
    return backup


def _promote_combined_candidate(
    candidate: Path,
    target: Path,
    post_promote_validator,
) -> Path:
    """Publish a validated candidate and restore the prior target on failure."""

    if not candidate.is_dir():
        raise RuntimeError(f"Combined-house candidate is missing: {candidate}")
    backup = _backup_existing(target)
    os.replace(candidate, target)
    try:
        post_promote_validator(target)
    except Exception:
        rejected = target.with_name(
            f"{target.name}.rejected_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_"
            f"{uuid.uuid4().hex[:8]}"
        )
        os.replace(target, rejected)
        if backup is not None:
            os.replace(backup, target)
        raise
    return target


def _render_blend_overviews(
    blend_path: Path,
    house_state_path: Path,
    output_dir: Path,
    *,
    published_blend_path: Path | None = None,
    published_house_state_path: Path | None = None,
    published_output_dir: Path | None = None,
) -> dict[str, Any]:
    """Render three final-house cutaways and bind their exact Blender state."""

    try:
        import bpy
        from mathutils import Vector
    except Exception as exc:
        raise RuntimeError("Cannot render required outlook views because bpy is unavailable") from exc

    blend_path = blend_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()
    canonical_blend = (published_blend_path or blend_path).resolve()
    house_state_path = house_state_path.resolve()
    canonical_house_state = (published_house_state_path or house_state_path).resolve()
    canonical_output = (published_output_dir or output_dir).resolve()
    evidence_path = output_dir / "overview_cutaway_evidence.json"

    def write_evidence(value: dict[str, Any]) -> None:
        temporary = evidence_path.with_name(
            f".{evidence_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, evidence_path)

    evidence: dict[str, Any] = {
        "schema_id": "scenesmith_house_cutaway_review_v1",
        "schema_version": 1,
        "status": "rendering",
        "expected_views": list(HOUSE_CUTAWAY_VIEW_NAMES),
        "source_blend": {
            "path": str(canonical_blend),
            "size_bytes": blend_path.stat().st_size,
            "sha256": _sha256_file(blend_path),
        },
        "source_state": _derivation_file_record(
            house_state_path, canonical_path=canonical_house_state
        ),
        "views": [],
    }
    write_evidence(evidence)
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objects:
        raise RuntimeError(f"No mesh objects found in required render source {blend_path}")

    visibility = {
        obj: (
            bool(getattr(obj, "hide_render", False)),
            bool(getattr(obj, "hide_viewport", False)),
        )
        for obj in objects
    }

    def restore_visibility() -> None:
        for obj, (hide_render, hide_viewport) in visibility.items():
            obj.hide_render = hide_render
            obj.hide_viewport = hide_viewport
        view_layer = getattr(bpy.context, "view_layer", None)
        update = getattr(view_layer, "update", None)
        if callable(update):
            update()

    mins = Vector((math.inf, math.inf, math.inf))
    maxs = Vector((-math.inf, -math.inf, -math.inf))
    raw_records: list[dict[str, Any]] = []
    for obj in objects:
        obj_min = Vector((math.inf, math.inf, math.inf))
        obj_max = Vector((-math.inf, -math.inf, -math.inf))
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            obj_min.x, obj_min.y, obj_min.z = (
                min(obj_min.x, world.x),
                min(obj_min.y, world.y),
                min(obj_min.z, world.z),
            )
            obj_max.x, obj_max.y, obj_max.z = (
                max(obj_max.x, world.x),
                max(obj_max.y, world.y),
                max(obj_max.z, world.z),
            )
            mins.x, mins.y, mins.z = min(mins.x, world.x), min(mins.y, world.y), min(
                mins.z, world.z
            )
            maxs.x, maxs.y, maxs.z = max(maxs.x, world.x), max(maxs.y, world.y), max(
                maxs.z, world.z
            )
        raw_records.append(
            {
                "object": obj,
                "name": str(obj.name),
                "minimum": [float(obj_min.x), float(obj_min.y), float(obj_min.z)],
                "maximum": [float(obj_max.x), float(obj_max.y), float(obj_max.z)],
            }
        )

    center = (mins + maxs) * 0.5
    span = max((maxs - mins).length, 1.0)
    span_xyz = [float(maxs[index] - mins[index]) for index in range(3)]
    height = max(span_xyz[2], 1e-6)
    classification: dict[str, list[dict[str, Any]]] = {
        "overhead": [],
        "wall": [],
        "floor": [],
        "combined_envelope": [],
        "content": [],
    }

    for record in raw_records:
        obj = record["object"]
        minimum = record["minimum"]
        maximum = record["maximum"]
        dimensions = [maximum[index] - minimum[index] for index in range(3)]
        obj_center = [
            (maximum[index] + minimum[index]) * 0.5 for index in range(3)
        ]
        dx, dy, dz = dimensions
        label_text = " ".join(
            filter(
                None,
                (
                    str(getattr(obj, "name", "")),
                    str(getattr(getattr(obj, "data", None), "name", "")),
                    str(getattr(getattr(obj, "parent", None), "name", "")),
                ),
            )
        ).lower()
        labels = set(re.findall(r"[a-z0-9]+", label_text))
        high = obj_center[2] >= float(mins.z) + 0.65 * height
        low = obj_center[2] <= float(mins.z) + 0.2 * height
        thin_horizontal = dz <= max(0.2, 0.08 * height)
        combined = (
            dx >= 0.75 * max(span_xyz[0], 1e-6)
            and dy >= 0.75 * max(span_xyz[1], 1e-6)
            and dz >= 0.55 * height
        )
        semantic_overhead = bool(labels & {"ceiling", "roof", "overhead"})
        semantic_floor = bool(labels & {"floor", "ground"})
        semantic_wall = bool(labels & {"wall", "window", "doorframe"})
        vertical_wall = (
            dz >= max(1.2, 0.4 * height)
            and min(dx, dy) <= 0.35
            and max(dx, dy) >= 0.8
        )
        if combined:
            role = "combined_envelope"
            reason = "single_mesh_spans_house_xyz"
        elif high and (
            semantic_overhead
            or (thin_horizontal and dx >= 0.5 and dy >= 0.5)
        ):
            role = "overhead"
            reason = "semantic_or_high_horizontal_envelope"
        elif low and (
            semantic_floor or (thin_horizontal and dx >= 1.0 and dy >= 1.0)
        ):
            role = "floor"
            reason = "semantic_or_low_horizontal_surface"
        elif semantic_wall or vertical_wall:
            role = "wall"
            reason = "semantic_or_tall_thin_envelope"
        else:
            role = "content"
            reason = "non_envelope_mesh"
        public = {
            **record,
            "role": role,
            "classification_reason": reason,
            "bounds": {
                "minimum": [round(value, 6) for value in minimum],
                "maximum": [round(value, 6) for value in maximum],
                "center": [round(value, 6) for value in obj_center],
                "dimensions": [round(value, 6) for value in dimensions],
            },
        }
        classification[role].append(public)

    if classification["combined_envelope"]:
        raise RuntimeError(
            "Cannot prove final-house cutaway; indivisible envelope meshes: "
            + ", ".join(item["name"] for item in classification["combined_envelope"])
        )
    for required_role in ("wall", "floor", "content"):
        if not classification[required_role]:
            raise RuntimeError(
                f"Cannot prove final-house cutaway; no {required_role} meshes classified"
            )
    evidence["house_bounds"] = {
        "minimum": [round(float(mins[index]), 6) for index in range(3)],
        "maximum": [round(float(maxs[index]), 6) for index in range(3)],
        "center": [round(float(center[index]), 6) for index in range(3)],
    }
    evidence["classification"] = {
        role: [
            {key: value for key, value in record.items() if key != "object"}
            for record in records
        ]
        for role, records in classification.items()
    }

    for obj in list(bpy.context.scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.object.light_add(type="SUN", location=(center.x, center.y, center.z + span))
    bpy.context.object.data.energy = 3.0
    bpy.ops.object.light_add(
        type="AREA", location=(center.x, center.y - span * 0.35, center.z + span)
    )
    bpy.context.object.data.energy = 600.0
    bpy.context.object.data.size = max(span * 0.6, 5.0)

    scene = bpy.context.scene
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.film_transparent = False
    if hasattr(scene, "eevee"):
        scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in {
            item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items
        } else "BLENDER_EEVEE"

    def render_view(
        name: str, direction: tuple[float, float, float], scale: float
    ) -> None:
        restore_visibility()
        direction_vec = Vector(direction).normalized()
        near_walls: list[dict[str, Any]] = []
        wall_dots: dict[str, float] = {}
        if name != "overview_top":
            horizontal_norm = math.hypot(direction_vec.x, direction_vec.y)
            if horizontal_norm <= 1e-6:
                raise RuntimeError(f"{name} has no horizontal cutaway direction")
            for wall in classification["wall"]:
                wall_center = wall["bounds"]["center"]
                wall_x = wall_center[0] - float(center.x)
                wall_y = wall_center[1] - float(center.y)
                wall_norm = math.hypot(wall_x, wall_y)
                dot = (
                    (direction_vec.x * wall_x + direction_vec.y * wall_y)
                    / (horizontal_norm * wall_norm)
                    if wall_norm > 1e-6
                    else -1.0
                )
                wall_dots[wall["name"]] = round(float(dot), 6)
                if dot > 0.15:
                    near_walls.append(wall)
            if not near_walls:
                raise RuntimeError(f"No camera-side wall found for {name}")
        hidden = [*classification["overhead"], *near_walls]
        for record in hidden:
            record["object"].hide_render = True
            record["object"].hide_viewport = True
        view_layer = getattr(bpy.context, "view_layer", None)
        update = getattr(view_layer, "update", None)
        if callable(update):
            update()
        if any(
            not record["object"].hide_render
            or not record["object"].hide_viewport
            for record in hidden
        ):
            raise RuntimeError(f"Cutaway visibility flags failed for {name}")

        camera = bpy.data.objects.get(f"camera_{name}")
        if camera is None:
            bpy.ops.object.camera_add()
            camera = bpy.context.object
            camera.name = f"camera_{name}"
        camera.location = center + direction_vec * span * scale
        target = center
        track = (target - camera.location).to_track_quat("-Z", "Y")
        camera.rotation_euler = track.to_euler()
        camera.data.lens = 24
        camera.data.clip_end = max(span * 8, 1000)
        scene.camera = camera
        scene.render.filepath = str(output_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        image_path = Path(scene.render.filepath).resolve()
        if not image_path.is_file() or image_path.stat().st_size < 1:
            raise RuntimeError(f"Final-house renderer produced no image for {name}")
        hidden_names = {record["name"] for record in hidden}
        far_walls = [
            record["name"]
            for record in classification["wall"]
            if record["name"] not in hidden_names
            and not record["object"].hide_render
        ]
        visible_floors = [
            record["name"]
            for record in classification["floor"]
            if not record["object"].hide_render
        ]
        visible_content = [
            record["name"]
            for record in classification["content"]
            if not record["object"].hide_render
        ]
        if not far_walls or not visible_floors or not visible_content:
            raise RuntimeError(
                f"Final-house cutaway {name} lacks visible walls, floors, or content"
            )
        evidence["views"].append(
            {
                "view_name": name,
                "image": str((canonical_output / f"{name}.png").resolve()),
                "image_size_bytes": image_path.stat().st_size,
                "image_sha256": _sha256_file(image_path),
                "camera": {
                    "location": [
                        round(float(camera.location[index]), 6) for index in range(3)
                    ],
                    "target": [round(float(center[index]), 6) for index in range(3)],
                },
                "cutaway": {
                    "established": True,
                    "view_name": name,
                    "overhead_state": (
                        "hidden" if classification["overhead"] else "verified_absent"
                    ),
                    "hidden_overhead": [
                        record["name"] for record in classification["overhead"]
                    ],
                    "hidden_camera_side_walls": [
                        record["name"] for record in near_walls
                    ],
                    "wall_camera_dot_products": wall_dots,
                    "visible_far_wall_object_names": sorted(far_walls),
                    "visible_floor_object_names": sorted(visible_floors),
                    "visible_content_object_names": sorted(visible_content),
                },
            }
        )
        LOGGER.info("Rendered outlook image: %s", scene.render.filepath)

    try:
        render_view("overview_top", (0.0, 0.0, 1.0), 1.7)
        render_view("overview_isometric", (1.0, -1.0, 0.75), 1.4)
        render_view("overview_front", (0.0, -1.0, 0.35), 1.25)
        if [view["view_name"] for view in evidence["views"]] != list(
            HOUSE_CUTAWAY_VIEW_NAMES
        ):
            raise RuntimeError("Final-house renderer violated the exact view contract")
        image_hashes = [view["image_sha256"] for view in evidence["views"]]
        if len(set(image_hashes)) != len(image_hashes):
            raise RuntimeError("Distinct final-house viewpoints produced duplicate image bytes")
        evidence["derivation_receipt"] = _make_derivation_receipt(
            state_record=evidence["source_state"],
            blend_record=evidence["source_blend"],
            views=evidence["views"],
        )
        evidence["status"] = "pass"
        evidence["rendered_view_count"] = len(evidence["views"])
        write_evidence(evidence)
        return evidence
    except Exception as exc:
        evidence["status"] = "fail"
        evidence["error_type"] = type(exc).__name__
        evidence["error"] = str(exc)
        write_evidence(evidence)
        raise
    finally:
        restore_visibility()


def _require_outlook_renders(
    combined_dir: Path, *, validate_cutaway: bool = True
) -> None:
    output_dir = combined_dir / "outlook_renders"
    overview_images = {
        name: output_dir / f"{name}.png" for name in HOUSE_CUTAWAY_VIEW_NAMES
    }
    required = list(overview_images.values())
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size < 1]
    if missing:
        raise RuntimeError(f"Required combined-house outlook renders are missing: {missing}")
    if validate_cutaway:
        failures = validate_house_cutaway_evidence(
            output_dir / "overview_cutaway_evidence.json",
            source_blend=combined_dir / "house.blend",
            overview_images=overview_images,
        )
        if failures:
            raise RuntimeError(
                "Published combined-house cutaway evidence failed: "
                + "; ".join(failures)
            )
        evidence_path = output_dir / "overview_cutaway_evidence.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        receipt = evidence.get("derivation_receipt")
        if not isinstance(receipt, dict) or receipt.get("schema_id") != DERIVATION_SCHEMA_ID:
            raise RuntimeError("Published combined-house derivation receipt is missing")
        expected_state = _derivation_file_record(combined_dir / "house_state.json")
        expected_blend = _derivation_file_record(combined_dir / "house.blend")
        expected_renders = [
            {
                "view_name": name,
                "path": str(overview_images[name].resolve()),
                "size_bytes": overview_images[name].stat().st_size,
                "sha256": _sha256_file(overview_images[name]),
            }
            for name in HOUSE_CUTAWAY_VIEW_NAMES
        ]
        if receipt.get("source_state") != expected_state:
            raise RuntimeError("Published house render is bound to a substituted house state")
        if receipt.get("source_blend") != expected_blend:
            raise RuntimeError("Published house render is bound to a substituted blend")
        if receipt.get("renders") != expected_renders:
            raise RuntimeError("Published house derivation render bindings are stale")
        payload = {key: value for key, value in receipt.items() if key != "attestation"}
        if receipt.get("attestation") != {
            "algorithm": "sha256",
            "sha256": hashlib.sha256(_canonical_json(payload)).hexdigest(),
        }:
            raise RuntimeError("Published house derivation receipt attestation is invalid")
        if len({record["sha256"] for record in expected_renders}) != len(expected_renders):
            raise RuntimeError("Published overview images have duplicate bytes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--csv",
        default="inputs/full_school_floor_20260703.csv",
        help="Prompt CSV used for this run. Must match the floor-plan/room stage.",
    )
    parser.add_argument(
        "--run-name",
        default="scenesmith_final_assemble",
        help="Hydra run name for assembly.",
    )
    parser.add_argument(
        "--gate-dir",
        help=(
            "Directory containing room self-exam JSON files. Defaults to "
            "scene_000/quality_gates/room_self_exam."
        ),
    )
    parser.add_argument(
        "--allow-ungated",
        action="store_true",
        help="Bypass room self-exam gate. Use only for debugging, never final runs.",
    )
    parser.add_argument(
        "--artiverse-data",
        default="data/artiverse",
        help="Official prepared Artiverse dataset root, relative to --repo-dir by default.",
    )
    parser.add_argument(
        "--verify-published-only",
        action="store_true",
        help="Revalidate the already-published combined_house without assembling it.",
    )
    parser.add_argument(
        "--contract-profile",
        choices=(SCHOOL_CONTRACT_PROFILE, FACTORY_CONTRACT_PROFILE),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Immutable staged input directory required by the school contract.",
    )
    parser.add_argument(
        "--factory-contract",
        type=Path,
        help="Immutable factory_contract.json; required by the factory profile.",
    )
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOGLEVEL", "INFO").upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    repo_dir = Path(args.repo_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    scene_dir = run_dir / "scene_000"
    input_dir = args.input_dir.resolve() if args.input_dir else None
    factory_contract = args.factory_contract.resolve() if args.factory_contract else None
    if args.contract_profile and input_dir is None:
        raise RuntimeError("--input-dir is required with --contract-profile")
    if args.contract_profile == FACTORY_CONTRACT_PROFILE and factory_contract is None:
        raise RuntimeError("--factory-contract is required by the factory profile")

    if args.contract_profile == FACTORY_CONTRACT_PROFILE:
        collect_articulated_roles = collect_factory_articulated_roles
        articulated_role_rules = FACTORY_ARTICULATED_ROLE_RULES
        articulated_profile = FACTORY_CONTRACT_PROFILE
    else:
        collect_articulated_roles = collect_required_articulated_roles
        articulated_role_rules = ARTICULATED_ROLE_RULES
        articulated_profile = SCHOOL_CONTRACT_PROFILE

    layout_path = scene_dir / "house_layout.json"
    with layout_path.open(encoding="utf-8") as f:
        layout_data = json.load(f)
    room_ids = _room_ids_from_layout_data(layout_data)

    missing = _verify_final_rooms(scene_dir, room_ids)
    if missing:
        raise RuntimeError(f"Refusing to assemble: missing final rooms: {missing}")

    gate_dir = (
        Path(args.gate_dir).resolve()
        if args.gate_dir
        else scene_dir / "quality_gates" / "room_self_exam"
    )
    if not args.allow_ungated:
        gate_failures = _verify_room_gates(
            gate_dir,
            room_ids,
            scene_dir,
            contract_profile=args.contract_profile,
            input_dir=input_dir,
            factory_contract=factory_contract,
        )
        if gate_failures:
            raise RuntimeError(
                "Refusing to assemble: room self-exam gate has not passed: "
                + "; ".join(gate_failures)
            )

    artiverse_data = Path(args.artiverse_data)
    if not artiverse_data.is_absolute():
        artiverse_data = repo_dir / artiverse_data
    authority = load_artiverse_authority(artiverse_data)

    # This check deliberately happens before HouseLayout construction, config loading,
    # backup, or assembly. Those operations may write package/export files.
    placed_artiverse, artiverse_failures = _collect_artiverse_from_room_states(
        scene_dir, room_ids, authority
    )
    _require_artiverse_records(
        placed_artiverse, artiverse_failures, "passing final room states"
    )
    placed_roles = None
    if args.contract_profile:
        placed_roles = collect_articulated_roles(
            _load_final_room_states(scene_dir, room_ids),
            require_runtime_provenance=True,
        )
        _require_articulated_roles(
            placed_roles,
            placed_artiverse,
            "passing final room states",
            profile=articulated_profile,
            role_rules=articulated_role_rules,
        )

    if args.verify_published_only:
        combined_dir = scene_dir / "combined_house"
        required = [
            combined_dir / "house_state.json",
            combined_dir / "sceneeval_state.json",
            combined_dir / "house.dmd.yaml",
            combined_dir / "house.blend",
            combined_dir / "artiverse_usage.json",
        ]
        missing_outputs = [
            str(path) for path in required if not path.is_file() or path.stat().st_size < 1
        ]
        if missing_outputs:
            raise RuntimeError(f"Published combined export is incomplete: {missing_outputs}")
        final_artiverse, final_artiverse_failures = _collect_artiverse_from_house_state(
            combined_dir / "house_state.json", scene_dir, authority
        )
        _require_artiverse_records(
            final_artiverse, final_artiverse_failures, "published combined house state"
        )
        _verify_combined_survival(placed_artiverse, final_artiverse)
        final_roles = None
        if args.contract_profile:
            final_roles = collect_articulated_roles(
                _load_combined_room_states(combined_dir / "house_state.json"),
                require_runtime_provenance=True,
            )
            _require_articulated_roles(
                final_roles,
                final_artiverse,
                "published combined house state",
                profile=articulated_profile,
                role_rules=articulated_role_rules,
            )
            _verify_articulated_role_survival(
                placed_roles, final_roles, articulated_role_rules
            )
            usage_data = json.loads(
                (combined_dir / "artiverse_usage.json").read_text(encoding="utf-8")
            )
            expected_roles = {"placed": placed_roles, "final": final_roles}
            if usage_data.get("required_articulated_roles") != expected_roles:
                raise RuntimeError(
                    "Published Artiverse usage manifest has stale articulated-role evidence"
                )
        validate_usage_manifest(
            combined_dir / "artiverse_usage.json",
            combined_dir / "house_state.json",
            authority,
            scene_dir=scene_dir,
        )
        if args.render:
            _require_outlook_renders(combined_dir)
        LOGGER.info("Published combined-house Artiverse evidence revalidated")
        return

    from omegaconf import OmegaConf
    from scenesmith.agent_utils.house import HouseLayout, HouseScene

    layout = HouseLayout.from_dict(layout_data, house_dir=scene_dir)

    cfg = _load_cfg(
        repo_dir=repo_dir, run_dir=run_dir, csv_path=args.csv, run_name=args.run_name
    )
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    rooms = {room_id: _load_room(scene_dir, room_id) for room_id in room_ids}
    house = HouseScene(layout=layout, rooms=rooms)

    target_dir = scene_dir / "combined_house"
    candidate_name = f".combined_house.staging.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    candidate_dir = house.assemble(cfg=cfg_dict, output_name=candidate_name)
    LOGGER.info("Assembled combined-house candidate at %s", candidate_dir)

    required = [
        candidate_dir / "house_state.json",
        candidate_dir / "sceneeval_state.json",
        candidate_dir / "house.dmd.yaml",
        candidate_dir / "house.blend",
    ]
    missing_outputs = [
        str(path) for path in required if not path.is_file() or path.stat().st_size < 1
    ]
    if missing_outputs:
        raise RuntimeError(f"Combined export missing outputs: {missing_outputs}")

    final_artiverse, final_artiverse_failures = _collect_artiverse_from_house_state(
        candidate_dir / "house_state.json", scene_dir, authority
    )
    _require_artiverse_records(
        final_artiverse, final_artiverse_failures, "combined house state"
    )
    _verify_combined_survival(placed_artiverse, final_artiverse)
    final_roles = None
    if args.contract_profile:
        final_roles = collect_articulated_roles(
            _load_combined_room_states(candidate_dir / "house_state.json"),
            require_runtime_provenance=True,
        )
        _require_articulated_roles(
            final_roles,
            final_artiverse,
            "combined house candidate",
            profile=articulated_profile,
            role_rules=articulated_role_rules,
        )
        _verify_articulated_role_survival(
            placed_roles, final_roles, articulated_role_rules
        )
    usage_manifest = candidate_dir / "artiverse_usage.json"
    _write_artiverse_usage_manifest(
        usage_manifest,
        placed_artiverse,
        final_artiverse,
        authority,
        scene_dir,
        room_ids,
        candidate_dir / "house_state.json",
        target_dir / "house_state.json",
        (
            {"placed": placed_roles, "final": final_roles}
            if args.contract_profile
            else None
        ),
    )
    LOGGER.info(
        "Verified %s surviving Artiverse assets; manifest=%s",
        len(final_artiverse),
        usage_manifest,
    )

    if args.render:
        _render_blend_overviews(
            blend_path=candidate_dir / "house.blend",
            house_state_path=candidate_dir / "house_state.json",
            output_dir=candidate_dir / "outlook_renders",
            published_blend_path=target_dir / "house.blend",
            published_house_state_path=target_dir / "house_state.json",
            published_output_dir=target_dir / "outlook_renders",
        )
        _require_outlook_renders(candidate_dir, validate_cutaway=False)

    def validate_published(published_dir: Path) -> None:
        validate_usage_manifest(
            published_dir / "artiverse_usage.json",
            published_dir / "house_state.json",
            authority,
            scene_dir=scene_dir,
        )
        if args.render:
            _require_outlook_renders(published_dir)

    combined_dir = _promote_combined_candidate(
        candidate_dir, target_dir, validate_published
    )
    LOGGER.info("Published validated final combined house at %s", combined_dir)

    LOGGER.info("Final assembly/postprocess completed")


if __name__ == "__main__":
    main()
