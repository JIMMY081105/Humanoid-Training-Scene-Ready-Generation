from __future__ import annotations

import csv
import json
import os

from pathlib import Path

import pytest

from scripts import final_acceptance_bundle as bundle
from scripts import preflight_sam3d_offline as sam3d_preflight
from scripts import preflight_sam3d_generation as sam3d_generation
from tests.unit.test_classroom_benchmark_receipt import (
    TEST_SAM3D_MESH_STATS,
    _artiverse_visual_receipt,
    _write_sam3d_generation,
)
from scripts import seed_reference_school_layout as reference_layout
from scripts.room_visual_self_exam import (
    SCHOOL_VLM_BACKEND,
    SCHOOL_VLM_MODEL,
    _room_judge_instruction,
    _room_messages,
    _vlm_request_contract as _room_vlm_request_contract,
    room_visual_requirements,
)
from scripts.whole_floor_reference_gate import (
    _floor_messages,
    _vlm_request_contract as _floor_vlm_request_contract,
    validate_exact_layout,
)


ATTEMPT_ID = "20260710T120000Z-1234"


@pytest.fixture(autouse=True)
def _fixture_sam3d_mesh_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sam3d_generation,
        "_mesh_stats",
        lambda _path: dict(TEST_SAM3D_MESH_STATS),
    )


def _write(path: Path, value: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")
    return path


def _write_json(path: Path, value: object) -> Path:
    return _write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _hash(path: Path) -> str:
    return bundle._sha256_file(path)


def _record(path: Path, *, size: bool = False) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(path.resolve()),
        "sha256": _hash(path),
    }
    if size:
        result["size_bytes"] = path.stat().st_size
    return result


def test_acceptance_score_contract_rejects_wrong_threshold_and_keys() -> None:
    valid_scores = {key: 8 for key in bundle.ROOM_SCORE_KEYS}
    with pytest.raises(bundle.AcceptanceBundleError, match="threshold must be exactly 7"):
        bundle._require_scores(
            {"scores": valid_scores, "threshold": 8}, label="room_visual:test"
        )
    with pytest.raises(bundle.AcceptanceBundleError, match="score keys are not exact"):
        bundle._require_scores(
            {"scores": {**valid_scores, "extra": 8}, "threshold": 7},
            label="room_visual:test",
        )


def test_acceptance_recomputes_canonical_vlm_request_digest() -> None:
    expected = {
        "schema_id": bundle.VLM_REQUEST_SCHEMA_ID,
        "schema_version": 1,
        "algorithm": "sha256",
        "score_keys": list(bundle.ROOM_SCORE_KEYS),
        "threshold": 7.0,
        "reference_image_sha256": "1" * 64,
        "review_image_sha256": ["2" * 64, "3" * 64, "4" * 64],
        "messages_sha256": "5" * 64,
    }
    expected["request_sha256"] = bundle._sha256_bytes(
        bundle._canonical_json(expected)
    )
    substituted = dict(expected)
    substituted["messages_sha256"] = "6" * 64
    substituted_payload = {
        key: value
        for key, value in substituted.items()
        if key != "request_sha256"
    }
    substituted["request_sha256"] = bundle._sha256_bytes(
        bundle._canonical_json(substituted_payload)
    )

    with pytest.raises(
        bundle.AcceptanceBundleError, match="differs from reconstructed messages"
    ):
        bundle._validate_vlm_request_record(
            substituted,
            label="room_visual:test",
            score_keys=bundle.ROOM_SCORE_KEYS,
            threshold=7.0,
            reference_sha256="1" * 64,
            render_sha256=["2" * 64, "3" * 64, "4" * 64],
            expected_record=expected,
        )


def _cutaway_proof(name: str) -> dict[str, object]:
    return {
        "established": True,
        "view_name": name,
        "overhead_state": "verified_absent",
        "hidden_overhead": [],
        "hidden_camera_side_walls": [] if name in {"top", "overview_top"} else ["near_wall"],
        "visible_far_wall_object_names": ["far_wall"],
        "visible_floor_object_names": ["floor"],
        "visible_content_object_names": ["desk"],
        "classified_content_count": 1,
    }


def _room_cutaway(scene: Path, room_id: str) -> dict[str, object]:
    final = scene / f"room_{room_id}" / "scene_states" / "final_scene"
    review = scene / "review" / "room_review_renders"
    views = []
    for name in ("top", "oblique_a", "oblique_b"):
        image = review / f"{room_id}_{name}.png"
        image_record = _record(image, size=True)
        views.append(
            {
                "view_name": name,
                "image": image_record["path"],
                "image_sha256": image_record["sha256"],
                "image_size_bytes": image_record["size_bytes"],
                "cutaway": _cutaway_proof(name),
            }
        )
    state_record = _record(final / "scene_state.json", size=True)
    blend_record = _record(final / "scene.blend", size=True)
    derivation_payload = {
        "schema_id": bundle.DERIVATION_SCHEMA_ID,
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
        "schema_id": "scenesmith_room_cutaway_review_v1",
        "schema_version": 1,
        "status": "pass",
        "room_id": room_id,
        "expected_views": ["top", "oblique_a", "oblique_b"],
        "rendered_view_count": 3,
        "source_state": state_record,
        "source_blend": blend_record,
        "classification": {
            "wall": [{"name": "wall"}],
            "floor": [{"name": "floor"}],
            "content": [{"name": "desk"}],
            "combined_envelope": [],
        },
        "views": views,
        "derivation_receipt": {
            **derivation_payload,
            "attestation": {
                "algorithm": "sha256",
                "sha256": bundle._sha256_bytes(
                    bundle._canonical_json(derivation_payload)
                ),
            },
        },
    }


def _house_cutaway(scene: Path) -> dict[str, object]:
    combined = scene / "combined_house"
    views = []
    for name in ("overview_top", "overview_isometric", "overview_front"):
        image = combined / "outlook_renders" / f"{name}.png"
        record = _record(image, size=True)
        views.append(
            {
                "view_name": name,
                "image": record.pop("path"),
                "image_sha256": record.pop("sha256"),
                "image_size_bytes": record.pop("size_bytes"),
                "cutaway": _cutaway_proof(name),
            }
        )
    state_record = _record(combined / "house_state.json", size=True)
    blend_record = _record(combined / "house.blend", size=True)
    derivation_payload = {
        "schema_id": bundle.DERIVATION_SCHEMA_ID,
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
        "schema_id": "scenesmith_house_cutaway_review_v1",
        "schema_version": 1,
        "status": "pass",
        "expected_views": ["overview_top", "overview_isometric", "overview_front"],
        "rendered_view_count": 3,
        "source_state": state_record,
        "source_blend": blend_record,
        "classification": {
            "wall": [{"name": "wall"}],
            "floor": [{"name": "floor"}],
            "content": [{"name": "desk"}],
            "combined_envelope": [],
        },
        "views": views,
        "derivation_receipt": {
            **derivation_payload,
            "attestation": {
                "algorithm": "sha256",
                "sha256": bundle._sha256_bytes(
                    bundle._canonical_json(derivation_payload)
                ),
            },
        },
    }


def _pipeline_contract() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": bundle.PIPELINE_CODE_CONTRACT_SCHEMA_VERSION,
        "contract": "scenesmith-full-quality-pipeline-code",
        "status": "pass",
        "specification": {
            "read_before_generation_code": True,
            "required_clauses": [{"id": "compulsory_artiverse"}],
        },
        "artifacts": [{"path": "scripts/example.py", "sha256": "a" * 64}],
        "artifact_count": 1,
    }
    payload["attestation"] = {
        "schema_version": 1,
        "algorithm": "sha256",
        "sha256": bundle._sha256_bytes(bundle._canonical_json(payload)),
    }
    return payload


def _sam3d() -> dict[str, object]:
    _image, target_mask = sam3d_preflight._canonical_inference_input()
    value: dict[str, object] = {
        "schema_version": sam3d_preflight.RESULT_SCHEMA_VERSION,
        "status": "pass",
        "offline": True,
        "offline_environment": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DIFFUSERS_OFFLINE": "1",
        },
        "sam3_checkpoint": "/cache/sam3.pt",
        "pipeline_config": "/cache/pipeline.yaml",
        "model_loaded": True,
        "pipeline_loaded": True,
        "visible_gpu_count": 1,
        "gpu": "A10",
        "inference_smoke": sam3d_preflight._build_inference_smoke(target_mask),
        "inference_smoke_verification": {
            "status": "pass",
            "critical_issues": [],
        },
        "evidence": {"schema_version": 1, "artifact_count": 3},
        "evidence_verification": {"status": "pass", "critical_issues": []},
    }
    value["attestation"] = {
        "algorithm": "sha256",
        "sha256": bundle._sam3d_attestation(value),
    }
    return value


@pytest.mark.parametrize("mutation", ("missing", "failed", "tampered"))
def test_sam3d_acceptance_rejects_invalid_real_inference_proof(
    mutation: str,
) -> None:
    document = _sam3d()
    if mutation == "missing":
        document.pop("inference_smoke")
    elif mutation == "failed":
        document["inference_smoke"]["status"] = "fail"
    else:
        document["inference_smoke"]["output"]["foreground_pixels"] = 0

    # Reattest the mutated document so this specifically proves semantic
    # inference validation cannot be bypassed by recomputing the outer digest.
    document["attestation"] = {
        "algorithm": "sha256",
        "sha256": bundle._sam3d_attestation(document),
    }
    with pytest.raises(
        bundle.AcceptanceBundleError,
        match="SAM3D image/text inference proof is invalid",
    ):
        bundle._validate_document(
            "sam3d_offline_preflight",
            document,
            paths={},
            context=None,  # type: ignore[arg-type]
        )


def _objathor() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "offline": True,
        "offline_environment": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DIFFUSERS_OFFLINE": "1",
        },
        "dataset": {"row_count": 50092},
        "model": {"name": "ViT-L-14"},
        "model_smoke": {"dimension": 768},
        "evidence": {"schema_version": 1, "artifact_count": 10},
        "verification": {"status": "pass", "critical_issues": []},
    }
    value["attestation"] = {
        "algorithm": "sha256",
        "sha256": bundle._objathor_attestation(value),
    }
    return value


def _policy() -> dict[str, object]:
    agents = {}
    for name, source in {
        "furniture_agent": "generated",
        "wall_agent": "generated",
        "ceiling_agent": "generated",
        "manipuland_agent": "objaverse",
    }.items():
        agents[name] = {
            "general_asset_source": source,
            "backend": "objaverse" if name == "manipuland_agent" else "sam3d",
            "coacd_max_convex_hull": 32,
            "vhacd_max_convex_hulls": 32,
        }
    ready_source = {
        "enabled": True,
        "data_path_exists": True,
        "embeddings_path_exists": True,
        "missing_embedding_files": [],
    }
    return {
        **agents,
        "worker_services": ["objaverse", "sam3d"],
        "articulated_contract": {
            "articulated_strategy_enabled": True,
            "artiverse_strategy_enabled": True,
            "artiverse": ready_source,
            "artvip": ready_source,
        },
        "materials_contract": {"status": "pass", "retained_count": 1934},
    }


def _make_fixture(tmp_path: Path) -> dict[str, Path | str]:
    repo = tmp_path / "repo"
    run = repo / "outputs" / "run"
    scene = run / "scene_000"
    inputs = repo / "inputs" / "school_reference_20260710"
    output = scene / "combined_house" / "sqz_acceptance_record.json"
    for directory in (repo, run, scene, inputs):
        directory.mkdir(parents=True, exist_ok=True)
    simulator_exporter = _write(
        repo / "scripts" / "export_scene_to_mujoco.py",
        "def export_to_usd():\n    pass\n",
    )

    original = "Generate the school shown in the reference."
    appendix = "Artiverse is compulsory and every quality gate must pass."
    effective = original + "\n" + appendix
    _write(inputs / "prompt_original.txt", original)
    _write(inputs / "scene_contract_appendix.txt", appendix)
    _write(inputs / "prompt.txt", effective)
    with (inputs / "prompt.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("scene_index", "prompt"))
        writer.writeheader()
        writer.writerow({"scene_index": 0, "prompt": effective})
    _write(inputs / "reference.png", b"reference-image")
    manifest = {
        "original_prompt_sha256": bundle._sha256_bytes(original.encode()),
        "appendix_sha256": bundle._sha256_bytes(appendix.encode()),
        "effective_prompt_sha256": bundle._sha256_bytes(effective.encode()),
        "effective_prompt_chars": len(effective),
        "reference_image_sha256": _hash(inputs / "reference.png"),
        "expected_room_ids": list(bundle.ROOM_IDS),
        "pipeline_contract": {
            "id": "scenesmith_full_quality_v1",
            "final_assembly_policy": "external_artiverse_gated",
            "required_articulated_source": "artiverse",
        },
        "scene_index": 0,
        "run_name": "test_school",
    }
    _write_json(inputs / "input_manifest.json", manifest)

    room_geometry = {
        "classroom_01": ([0.0, 0.0], 9.0, 7.5),
        "classroom_03": ([0.0, 10.0], 9.0, 7.5),
        "classroom_04": ([0.0, 20.0], 9.0, 7.5),
        "classroom_02": ([21.0, 0.0], 9.0, 7.5),
        "classroom_06": ([21.0, 12.0], 9.0, 7.5),
        "classroom_05": ([21.0, 20.0], 9.0, 7.5),
        "library": ([10.0, -9.0], 10.0, 9.0),
        "boys_toilet": ([1.0, -5.0], 4.0, 4.0),
        "girls_toilet": ([5.0, -5.0], 4.0, 4.0),
        "storage_room": ([21.0, 8.0], 5.0, 3.7),
        "main_corridor": ([9.0, 0.0], 12.0, 22.5),
    }
    layout = {
        "placement_valid": True,
        "connectivity_valid": True,
        "placed_rooms": {
            room_id: {
                "room_id": room_id,
                "prompt": f"contract prompt {room_id}",
                "position": room_geometry[room_id][0],
                "width": room_geometry[room_id][1],
                "depth": room_geometry[room_id][2],
            }
            for room_id in bundle.ROOM_IDS
        },
    }
    layout_path = _write_json(scene / "house_layout.json", layout)
    gate_root = scene / "quality_gates"
    _write(scene / "floor_plans" / "final_floor_plan" / "floor_plan.dmd.yaml", "directives: []\n")
    _write(scene / "room_geometry" / "reference_school.sdf", "<sdf version='1.9'/>\n")
    _write(scene / "package.xml", "<package/>\n")
    native_geometry = reference_layout._tree_manifest(
        scene, ("floor_plans", "room_geometry", "package.xml")
    )
    _write_json(
        gate_root / "reference_school_layout_seed.json",
        {
            "schema_version": reference_layout.SEED_SCHEMA_VERSION,
            "status": "pass",
            "profile": reference_layout.PROFILE,
            "implementation": "native_scenesmith_deterministic_reference_layout",
            "seed_spec": reference_layout._seed_spec_evidence(),
            "structural_layout_sha256": reference_layout._structural_layout_sha256(
                layout
            ),
            "artifact_count": len(native_geometry),
            "artifact_manifest_sha256": reference_layout._manifest_sha256(
                native_geometry
            ),
            "artifacts": native_geometry,
        },
    )
    deterministic_root = gate_root / "room_self_exam_deterministic"
    visual_root = gate_root / "room_self_exam"
    review = scene / "review" / "room_review_renders"
    room_states: dict[str, object] = {}
    deterministic_scores = {
        "object_relevance": 8,
        "placement_realism": 8,
        "clearance_and_access": 8,
        "collision_risk": 8,
        "prompt_alignment": 8,
    }
    for room_id in bundle.ROOM_IDS:
        final = scene / f"room_{room_id}" / "scene_states" / "final_scene"
        state = {"objects": {"object_1": {"object_type": "chair"}}}
        if room_id == "library":
            state["objects"]["bookcase_1"] = {
                "object_id": "bookcase_1",
                "object_type": "cabinet",
                "sdf_path": "generated_assets/sdf/bookcase/model.sdf",
                "metadata": {
                    "asset_source": "articulated",
                    "articulated_source": "artiverse",
                    "articulated_id": "official_bookcase",
                    "is_articulated": True,
                },
            }
        state_path = _write_json(final / "scene_state.json", state)
        room_states[room_id] = state
        blend = _write(final / "scene.blend", f"blend-{room_id}".encode())
        images = {}
        for name in ("top", "oblique_a", "oblique_b"):
            images[name] = _write(review / f"{room_id}_{name}.png", f"{room_id}-{name}".encode())
        deterministic = {
            "room_id": room_id,
            "status": "pass",
            "contract_profile": "school_reference_20260710",
            "scores": deterministic_scores,
            "threshold": 7,
            "critical_issues": [],
            "repair_instructions": [],
            "metrics": {
                "semantic_inventory": {
                    "status": "pass",
                    "critical_issues": [],
                    "matched_requirements": ["school inventory"],
                }
            },
        }
        deterministic_path = _write_json(deterministic_root / f"{room_id}.json", deterministic)
        cutaway_path = _write_json(
            review / f"{room_id}_cutaway_evidence.json",
            _room_cutaway(scene, room_id),
        )
        visual_evidence = {
            "schema_version": 1,
            "algorithm": "sha256",
            "scene_state": _record(state_path),
            "source_blend": _record(blend),
            "house_layout": _record(layout_path),
            "reference_image": _record(inputs / "reference.png"),
            "effective_prompt": _record(inputs / "prompt.txt"),
            "input_manifest": _record(inputs / "input_manifest.json"),
            "prompt_binding": {},
            "deterministic_gate": _record(deterministic_path),
            "cutaway_evidence": _record(cutaway_path),
            "review_images": [_record(images[name]) for name in ("top", "oblique_a", "oblique_b")],
        }
        # Prompt binding does not exist until after the room loop; fill/write the
        # visual JSON below once that shared artifact is available.
        _write_json(visual_root / f".{room_id}.pending.json", visual_evidence)

    prompt_binding = {
        "schema_version": 1,
        "status": "pass",
        "profile": "school_reference_20260710",
        "effective_prompt_sha256": manifest["effective_prompt_sha256"],
        "room_prompt_sha256": {
            room_id: bundle._sha256_bytes(room_id.encode())
            for room_id in bundle.ROOM_IDS
        },
        "layout": _record(layout_path),
        "input_manifest": _record(inputs / "input_manifest.json"),
    }
    prompt_binding_path = _write_json(gate_root / "room_prompt_binding.json", prompt_binding)
    for room_id in bundle.ROOM_IDS:
        pending = visual_root / f".{room_id}.pending.json"
        visual_evidence = json.loads(pending.read_text(encoding="utf-8"))
        pending.unlink()
        visual_evidence["prompt_binding"] = _record(prompt_binding_path)
        room_prompt = f"contract prompt {room_id}"
        deterministic_document = json.loads(
            (deterministic_root / f"{room_id}.json").read_text(encoding="utf-8")
        )
        review_paths = [
            review / f"{room_id}_{name}.png"
            for name in ("top", "oblique_a", "oblique_b")
        ]
        requirement_map = room_visual_requirements(room_id)
        instruction = _room_judge_instruction(
            room_id=room_id,
            room_prompt=room_prompt,
            deterministic_result=deterministic_document,
            threshold=7.0,
            review_count=3,
            immutable_effective_prompt=effective,
            contract_profile="school_reference_20260710",
            visual_requirements=requirement_map,
        )
        visual_evidence["vlm_request"] = _room_vlm_request_contract(
            messages=_room_messages(
                instruction, inputs / "reference.png", review_paths
            ),
            room_id=room_id,
            model=SCHOOL_VLM_MODEL,
            backend=SCHOOL_VLM_BACKEND,
            threshold=7.0,
            requirement_keys=list(requirement_map),
            reference_image=inputs / "reference.png",
            review_images=review_paths,
        )
        visual = {
            "room_id": room_id,
            "status": "pass",
            "contract_profile": "school_reference_20260710",
            "scores": deterministic_scores,
            "threshold": 7,
            "critical_issues": [],
            "repair_instructions": [],
            "deterministic_gate_status": "pass",
            "room_prompt": room_prompt,
            "reference_image": str((inputs / "reference.png").resolve()),
            "review_images": [str(path.resolve()) for path in review_paths],
            "visual_assessment": {
                "scores": deterministic_scores,
                "critical_issues": [],
                "requirement_evidence": {
                    key: {
                        "status": "pass",
                        "view_indices": [1],
                        "observation": f"Visible evidence for {key} is present.",
                    }
                    for key in requirement_map
                },
            },
            "evidence": visual_evidence,
        }
        _write_json(visual_root / f"{room_id}.json", visual)
    summary = {
        "status": "pass",
        "room_count": 11,
        "passed_rooms": list(bundle.ROOM_IDS),
        "failed_rooms": [],
        "contract_profile": "school_reference_20260710",
    }
    _write_json(deterministic_root / "summary.json", summary)
    _write_json(visual_root / "summary.json", summary)

    _write_json(
        gate_root / "floor_plan_layout.json",
        {
            "status": "pass",
            "required_room_count": 11,
            "actual_room_count": 11,
            "actual_unique_room_count": 11,
            "missing_room_ids": [],
            "unexpected_room_ids": [],
            "critical_issues": [],
            "evidence": {"entrance_door_ids": ["main_double_door"]},
        },
    )

    variation: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "minimum_variation_score": 7.0,
        "fingerprints": {
            room_id: {
                "semantic_sha256": bundle._sha256_bytes(f"semantic-{room_id}".encode()),
                "seating_sha256": bundle._sha256_bytes(f"seating-{room_id}".encode()),
                "decoration_sha256": bundle._sha256_bytes(f"decor-{room_id}".encode()),
                "combined_sha256": bundle._sha256_bytes(f"combined-{room_id}".encode()),
            }
            for room_id in bundle.ROOM_IDS[:6]
        },
        "visual_assessment": {
            "variation_quality_score": 8.0,
            "classrooms": {
                room_id: {
                    "status": "pass",
                    "distinctive_features": ["distinct teaching wall", "unique reading corner"],
                    "seating_layout": "A visibly distinct seating arrangement.",
                }
                for room_id in bundle.ROOM_IDS[:6]
            },
            "too_similar_pairs": [],
        },
        "evidence": [
            {
                "room_id": room_id,
                "role": role,
                **_record(
                    scene / f"room_{room_id}" / "scene_states" / "final_scene" / "scene_state.json"
                    if role == "state"
                    else review / f"{room_id}_top.png",
                    size=True,
                ),
            }
            for room_id in bundle.ROOM_IDS[:6]
            for role in ("state", "top_view")
        ],
        "critical_issues": [],
    }
    variation["attestation_sha256"] = bundle._sha256_bytes(bundle._canonical_json(variation))
    _write_json(gate_root / "classroom_variation.json", variation)

    combined = scene / "combined_house"
    house_state_path = _write_json(
        combined / "house_state.json",
        {"layout": layout, "rooms": room_states},
    )
    dmd = _write(combined / "house.dmd.yaml", "directives:\n  - add_model: {name: school, file: room.sdf}\n")
    house_blend = _write(combined / "house.blend", b"combined-house-blend")
    overviews = {}
    for name in ("overview_top", "overview_isometric", "overview_front"):
        overviews[name] = _write(
            combined / "outlook_renders" / f"{name}.png",
            name.encode(),
        )
    house_cutaway_path = _write_json(
        combined / "outlook_renders" / "overview_cutaway_evidence.json",
        _house_cutaway(scene),
    )
    artiverse_asset = {
        "room_id": "library",
        "object_id": "bookcase_1",
        "articulated_id": "official_bookcase",
        "asset_source": "articulated",
        "articulated_source": "artiverse",
        "sdf_path": "generated_assets/sdf/bookcase/model.sdf",
        "sdf_sha256": "1" * 64,
        "sdf_tree_sha256": "2" * 64,
        "source_sdf_sha256": "3" * 64,
        "source_tree_sha256": "4" * 64,
    }
    artiverse_usage = {
        "schema_version": 2,
        "status": "pass",
        "dataset": "artiverse",
        "placed_asset_count": 1,
        "final_surviving_asset_count": 1,
        "placed_assets": [artiverse_asset],
        "final_surviving_assets": [artiverse_asset],
        "house_state": _record(house_state_path),
        "room_states": [
            {
                "room_id": room_id,
                **_record(
                    scene / f"room_{room_id}" / "scene_states" / "final_scene" / "scene_state.json"
                ),
            }
            for room_id in bundle.ROOM_IDS
        ],
        "required_articulated_roles": {
            "status": "pass",
            "roles": ["library_bookcase", "storage_cabinet", "teacher_drawers"],
        },
    }
    artiverse_usage_path = _write_json(combined / "artiverse_usage.json", artiverse_usage)
    motion_role_specs = {
        "library_glass_door_bookcase": ("library", "revolute", 1, "artiverse"),
        "school_supply_two_door_utility_cabinet": (
            "storage_room",
            "revolute",
            2,
            "artvip",
        ),
        "teacher_filing_drawer_cabinet": (
            "classroom_01",
            "prismatic",
            1,
            "artvip",
        ),
    }
    motion_states = {
        room_id: _record(
            scene
            / f"room_{room_id}"
            / "scene_states"
            / "final_scene"
            / "scene_state.json",
            size=True,
        )
        for room_id in ("library", "storage_room", "classroom_01")
    }
    motion_roles: dict[str, object] = {}
    for role, (room_id, joint_type, joint_count, source) in motion_role_specs.items():
        sdf = _write(
            scene
            / f"room_{room_id}"
            / "generated_assets"
            / "sdf"
            / role
            / "model.sdf",
            f"<sdf version='1.10'><model name='{role}'/></sdf>",
        )
        exercises = [
            {
                "joint_name": f"joint_{index}",
                "joint_type": joint_type,
                "tested_positions": [0.0, 0.5],
                "child_body_pose_changed": True,
                "transform_delta": {
                    "max_abs_matrix_delta": 0.5,
                    "translation_norm": 0.5,
                    "rotation_frobenius_norm": 0.5,
                },
            }
            for index in range(joint_count)
        ]
        motion_roles[role] = {
            "room_id": room_id,
            "object_id": f"{role}_object",
            "articulated_id": f"{role}_asset",
            "articulated_source": source,
            "asset_source": "articulated",
            "state_sha256": motion_states[room_id]["sha256"],
            "sdf": {**_record(sdf, size=True), "referenced_resources": []},
            "resource_tree": {"sha256": bundle._sha256_bytes(role.encode())},
            "drake_motion": {
                "status": "pass",
                "num_joints": joint_count,
                "joint_exercises": exercises,
            },
        }
    motion: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "profile": "school_reference_20260710",
        "scene_dir": str(scene.resolve()),
        "required_roles": sorted(motion_role_specs),
        "artiverse_roles": ["library_glass_door_bookcase"],
        "state_evidence": motion_states,
        "roles": motion_roles,
        "drake_motion_exercised": True,
        "critical_issues": [],
        "elapsed_seconds": 1.0,
    }
    motion_payload = {
        key: motion.get(key)
        for key in (
            "schema_version",
            "profile",
            "scene_dir",
            "required_roles",
            "artiverse_roles",
            "state_evidence",
            "roles",
            "drake_motion_exercised",
        )
    }
    motion["evidence_sha256"] = bundle._sha256_bytes(
        bundle._canonical_json(motion_payload)
    )
    motion["attestation"] = {
        "algorithm": "sha256",
        "sha256": bundle._sha256_bytes(bundle._canonical_json(motion)),
    }
    motion_path = _write_json(gate_root / "articulated_motion.json", motion)
    repeated_motion = json.loads(motion_path.read_text(encoding="utf-8"))
    motion_source = {
        **_record(motion_path, size=True),
        "evidence_sha256": motion["evidence_sha256"],
        "attestation_sha256": motion["attestation"]["sha256"],
    }
    motion_verification: dict[str, object] = {
        "verification_schema_id": "scenesmith_articulated_motion_verification_v1",
        "schema_version": 1,
        "status": "pass",
        "mode": "verify-only",
        "scene_dir": str(scene.resolve()),
        "saved_output": str(motion_path.resolve()),
        "source_gate": motion_source,
        "saved_evidence_sha256": motion["evidence_sha256"],
        "fresh_evidence_sha256": motion["evidence_sha256"],
        "recomputed_gate": {
            "result": repeated_motion,
            "status": "pass",
            "evidence_sha256": motion["evidence_sha256"],
            "attestation": repeated_motion["attestation"],
            "result_sha256": bundle._sha256_bytes(
                bundle._canonical_json(repeated_motion)
            ),
            "role_count": 3,
        },
        "drake_motion_repeated": True,
        "critical_issues": [],
        "elapsed_seconds": 1.0,
    }
    motion_verification["attestation"] = {
        "algorithm": "sha256",
        "sha256": bundle._sha256_bytes(
            bundle._canonical_json(motion_verification)
        ),
    }
    _write_json(
        gate_root / "articulated_motion_verification.json",
        motion_verification,
    )

    final_state_inputs = {}
    for room_id in bundle.ROOM_IDS:
        state_path = (
            scene
            / f"room_{room_id}"
            / "scene_states"
            / "final_scene"
            / "scene_state.json"
        )
        state_value = json.loads(state_path.read_text(encoding="utf-8"))
        final_state_inputs[room_id] = {
            **_record(state_path),
            "objects_sha256": bundle._sha256_bytes(
                bundle._canonical_json(state_value["objects"])
            ),
            "object_count": len(state_value["objects"]),
        }
    routes = {}
    for index, room_id in enumerate(bundle.ROOM_IDS):
        cells = [[index, 0], [index, 1]]
        waypoints = [
            {"zone": "library", "cell": cells[0], "point_m": [index, 0.0]},
            {"zone": room_id, "cell": cells[1], "point_m": [index, 1.0]},
        ]
        routes[room_id] = {
            "status": "pass",
            "allowed_zones": ["library", "main_corridor", room_id],
            "traversed_zones": ["library", room_id],
            "cells": cells,
            "waypoints": waypoints,
            "route_sha256": bundle._sha256_bytes(
                bundle._canonical_json({"cells": cells, "waypoints": waypoints})
            ),
        }
    common_zones = {"library", "main_corridor"}
    navigation: dict[str, object] = {
        "schema_version": 1,
        "algorithm": "school_navigation_grid_v1",
        "status": "pass",
        "parameters": {
            "humanoid_radius_m": 0.35,
            "humanoid_height_m": 2.0,
            "grid_resolution_m": 0.15,
            "minimum_door_width_m": 0.9,
            "minimum_entrance_width_m": 1.6,
            "turning_diameter_m": 1.5,
        },
        "inputs": {
            "house_layout": _record(layout_path),
            "house_state": _record(house_state_path),
            "final_room_states": final_state_inputs,
        },
        "topology": {
            "common_zone_ids": sorted(common_zones),
            "common_zones_reachable_from_core": sorted(common_zones),
            "direct_common_portals_by_room": {
                room_id: [] if room_id in common_zones else [f"door_{room_id}"]
                for room_id in bundle.ROOM_IDS
            },
            "forbidden_transit_policy": "Only common circulation and the target room.",
        },
        "occupancy": {
            "grid_dimensions": [10, 10],
            "obstacles": [],
            "obstacles_sha256": bundle._sha256_bytes(bundle._canonical_json([])),
        },
        "turning_areas": {
            room_id: {
                "status": "pass",
                "candidate_count": 1,
                "selected_cell": [0, 0],
            }
            for room_id in {
                "library",
                "main_corridor",
                *(f"classroom_{index:02d}" for index in range(1, 7)),
            }
        },
        "routes": routes,
        "routes_sha256": bundle._sha256_bytes(bundle._canonical_json(routes)),
        "critical_issues": [],
    }
    navigation["attestation"] = {
        "algorithm": "sha256(canonical JSON excluding attestation)",
        "payload_sha256": bundle._sha256_bytes(bundle._canonical_json(navigation)),
        "self_attested_status": "pass",
        "all_inputs_hashed": True,
        "all_routes_hashed": True,
    }
    navigation_path = _write_json(
        gate_root / "school_navigation.json", navigation
    )
    repeated_navigation = json.loads(
        navigation_path.read_text(encoding="utf-8")
    )
    navigation_source = {
        **_record(navigation_path, size=True),
        "payload_sha256": navigation["attestation"]["payload_sha256"],
        "routes_sha256": navigation["routes_sha256"],
    }
    navigation_verification: dict[str, object] = {
        "verification_schema_id": "scenesmith_school_navigation_verification_v1",
        "schema_version": 1,
        "status": "pass",
        "mode": "verify-only",
        "scene_dir": str(scene.resolve()),
        "verified_output": str(navigation_path.resolve()),
        "source_gate": navigation_source,
        "payload_sha256": navigation["attestation"]["payload_sha256"],
        "routes_sha256": navigation["routes_sha256"],
        "recomputed_gate": {
            "result": repeated_navigation,
            "status": "pass",
            "payload_sha256": navigation["attestation"]["payload_sha256"],
            "routes_sha256": navigation["routes_sha256"],
            "attestation": repeated_navigation["attestation"],
            "result_sha256": bundle._sha256_bytes(
                bundle._canonical_json(repeated_navigation)
            ),
            "route_count": len(bundle.ROOM_IDS),
        },
        "navigation_recomputed": True,
        "critical_issues": [],
        "message": "navigation evidence and every input/route hash recomputed exactly",
    }
    navigation_verification["attestation"] = {
        "algorithm": "sha256(canonical JSON excluding attestation)",
        "payload_sha256": bundle._sha256_bytes(
            bundle._canonical_json(navigation_verification)
        ),
        "self_attested_status": "pass",
    }
    _write_json(
        gate_root / "school_navigation_verification.json",
        navigation_verification,
    )
    _write_json(
        gate_root / "artiverse_final_validation.json",
        {
            "schema_version": 1,
            "status": "pass",
            "placed_asset_count": 1,
            "final_surviving_asset_count": 1,
            "usage_manifest": _record(artiverse_usage_path),
            "house_state": _record(house_state_path),
        },
    )
    whole_evidence = {
        "schema_version": 1,
        "algorithm": "sha256",
        "house_layout": _record(layout_path),
        "house_state": _record(house_state_path),
        "artiverse_usage": _record(artiverse_usage_path),
        "house_cutaway": _record(house_cutaway_path),
        "input_manifest": _record(inputs / "input_manifest.json"),
        "reference_image": _record(inputs / "reference.png"),
        "overview_images": {name: _record(path) for name, path in overviews.items()},
    }
    deterministic_layout = validate_exact_layout(layout)
    floor_messages = _floor_messages(
        deterministic=deterministic_layout,
        threshold=7.0,
        reference_image=inputs / "reference.png",
        overview_images=overviews,
    )
    whole_evidence["vlm_request"] = _floor_vlm_request_contract(
        messages=floor_messages,
        model="gpt-5.2",
        backend="openai",
        reference_image=inputs / "reference.png",
        overview_images=overviews,
    )
    _write_json(
        gate_root / "whole_floor_reference.json",
        {
            "status": "pass",
            "scores": {
                "room_count_and_identity": 8,
                "room_arrangement": 8,
                "warm_visual_style": 8,
                "circulation_and_access": 8,
                "furnishing_completeness": 8,
                "simulation_readiness": 8,
                "reference_similarity": 8,
            },
            "threshold": 7,
            "critical_issues": [],
            "deterministic_layout_gate": deterministic_layout,
            "visual_assessment": {
                "scores": {
                    "room_count_and_identity": 8,
                    "room_arrangement": 8,
                    "warm_visual_style": 8,
                    "circulation_and_access": 8,
                    "furnishing_completeness": 8,
                    "simulation_readiness": 8,
                    "reference_similarity": 8,
                },
                "critical_issues": [],
                "repair_instructions": [],
                "observations": ["All whole-floor requirements are visible."],
            },
            "evidence_verification": {"status": "pass", "critical_issues": []},
            "evidence": whole_evidence,
        },
    )
    _write_json(
        gate_root / "drake_load_sqz.json",
        {
            "status": "pass",
            "checks": {
                "dmd_file_exists": True,
                "package_root_exists": True,
                "drake_load_succeeded": True,
            },
            "acceptance_requirements": {
                "required_visible_gpu_count": 0,
                "max_collision_elements_per_sdf": 32,
                "expected_room_count": 11,
            },
            "two_gpu_acceptance_environment": False,
            "dmd_inventory": {"status": "pass", "sha256": _hash(dmd)},
            "house_state_inventory": {"status": "pass", "sha256": _hash(house_state_path)},
            "collision_report": {
                "parse_failures": [],
                "assets_over_collision_cap": [],
                "sdf_asset_count": 1,
            },
        },
    )
    export_root = scene / "mujoco_export"
    attempt_marker = _write_json(
        export_root / ".export_attempt.json",
        {
            "schema_version": 1,
            "run_attempt_id": ATTEMPT_ID,
            "created_at": "2026-07-10T12:00:00+00:00",
        },
    )
    scene_xml = _write(export_root / "scene.xml", "<mujoco model='school'/>")
    usd_layers = [
        _write(export_root / relative, "#usda 1.0\n" + relative)
        for relative in (
            "usd/school.usda",
            "usd/Payload/Contents.usda",
            "usd/Payload/Geometry.usda",
            "usd/Payload/Physics.usda",
        )
    ]
    exported = [attempt_marker, scene_xml, *usd_layers]
    export_inventory = [
        {
            "path": path.relative_to(export_root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _hash(path),
        }
        for path in sorted(exported, key=lambda value: value.relative_to(export_root).as_posix())
    ]
    expected_usd = [
        value
        for value in export_inventory
        if value["path"]
        in {
            "usd/school.usda",
            "usd/Payload/Contents.usda",
            "usd/Payload/Geometry.usda",
            "usd/Payload/Physics.usda",
        }
    ]
    expected_usd.sort(key=lambda value: str(value["path"]))
    payload_layers = [
        "usd/Payload/Contents.usda",
        "usd/Payload/Geometry.usda",
        "usd/Payload/Physics.usda",
    ]
    validated_stages = [
        {
            "path": record["path"],
            "sha256": record["sha256"],
            "prim_count": 1,
            "used_layers": (
                ["usd/school.usda", *payload_layers]
                if record["path"] == "usd/school.usda"
                else [record["path"]]
            ),
        }
        for record in expected_usd
    ]
    _write_json(
        gate_root / "simulator_exports.json",
        {
            "schema_version": 2,
            "status": "pass",
            "output_dir": str(export_root.resolve()),
            "require_usd": True,
            "attempt_marker": {
                "path": ".export_attempt.json",
                "sha256": _hash(attempt_marker),
                "run_attempt_id": ATTEMPT_ID,
            },
            "publication": {
                "run_attempt_id": ATTEMPT_ID,
                "published_dir": str(export_root.resolve()),
                "atomic_promotion": True,
                "staging_inventory_sha256": bundle._sha256_bytes(
                    bundle._canonical_json(export_inventory)
                ),
                "previous_backup_cleanup": "not_applicable",
                "exporter_contract": {
                    "path": str(simulator_exporter.resolve()),
                    "sha256": _hash(simulator_exporter),
                    "usd_failure_contract": "pass",
                },
            },
            "mujoco": {
                "scene_xml": "scene.xml",
                "scene_xml_sha256": _hash(scene_xml),
                "referenced_asset_count": 0,
                "referenced_assets": [],
                "model_counts": {
                    "nbody": 2,
                    "ngeom": 1,
                    "njnt": 0,
                    "nq": 0,
                    "nv": 0,
                },
            },
            "usd": {
                "usd_dir": "usd",
                "usd_layer_count": len(expected_usd),
                "expected_artifacts": expected_usd,
                "validated_stages": validated_stages,
                "candidate_failures": [],
            },
            "file_count": len(export_inventory),
            "file_inventory": export_inventory,
            "inventory_sha256": bundle._sha256_bytes(bundle._canonical_json(export_inventory)),
        },
    )
    _write_json(
        gate_root / "sage_scene_checker.json",
        {
            "scene_id": "scene_000",
            "pass": True,
            "acceptance_policy": {
                "fail_on_warnings": True,
                "fatal_failed_check_ids": [],
            },
            "summary": {
                "num_rooms": 11,
                "num_objects": 11,
                "num_errors": 0,
                "num_warnings": 0,
            },
            "checks": [{"id": "json_parse", "severity": "error", "status": "pass"}],
        },
    )

    _write_json(
        run / "input_manifest_validation.json",
        {
            "status": "pass",
            "expected_room_ids": list(bundle.ROOM_IDS),
            "pipeline_contract": manifest["pipeline_contract"],
            "critical_issues": [],
        },
    )
    _write_json(run / "pipeline_code_contract.json", _pipeline_contract())
    _write_json(
        run / "materials_contract_validation.json",
        {
            "status": "pass",
            "schema_id": "scenesmith_materials_contract_v1",
            "source_count": 1949,
            "retained_count": 1934,
            "pruned_count": 15,
            "asset_inventory_sha256": "5" * 64,
            "manifest_sha256": "6" * 64,
        },
    )
    authority = {
        "source_repository": "3dlg-hcvc/artiverse",
        "source_revision": "8c4b120418e7cbdf9ac4c9580c5dbfdbf128a248",
        "indexed_count": 100,
        "index_sha256": {"clip_embeddings.npy": "7" * 64},
    }
    _write_json(
        run / "artiverse_preparation_validation.json",
        {"status": "pass", "authority": authority, "schema_version": 1},
    )
    _write_json(
        run / "artiverse_visual_resources.json",
        _artiverse_visual_receipt(repo),
    )
    ready_dataset = {
        "data_path_exists": True,
        "embeddings_path_exists": True,
        "missing_embedding_files": [],
    }
    _write_json(
        run / "articulated_router_validation.json",
        {
            "status": "pass",
            "failures": [],
            "datasets": {"artiverse": ready_dataset, "artvip": ready_dataset},
            "selected_asset_identifiers": {"artiverse": ["a"], "artvip": ["b"]},
            "prompts": [
                {
                    "passes_artiverse_router_strategy_check": True,
                    "passes_artvip_router_strategy_check": True,
                    "passes_artiverse_candidate_check": True,
                    "passes_artvip_candidate_check": True,
                }
            ],
        },
    )
    _write_json(run / "resolved_asset_policy.json", _policy())
    preflight = repo / "outputs" / "preflight" / inputs.name
    _write_json(preflight / "sam3d_offline_load.json", _sam3d())
    _write_sam3d_generation(preflight, repo)
    _write_json(preflight / "objathor_retrieval_offline.json", _objathor())
    _write_json(
        preflight / "vlm_vision_smoke.json",
        {
            "status": "pass",
            "backend": "openai",
            "model": "gpt-5.2",
            "image_sha256": _hash(inputs / "reference.png"),
            "assessment": {
                "scores": {"image_readability": 10},
                "critical_issues": [],
                "repair_instructions": [],
            },
        },
    )
    return {
        "repo_dir": repo,
        "run_dir": run,
        "scene_dir": scene,
        "input_dir": inputs,
        "run_attempt_id": ATTEMPT_ID,
        "output": output,
    }


def test_complete_bundle_is_created_and_verifies(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    result = bundle.create_bundle(**fixture)
    assert result["status"] == bundle.STATUS
    assert result["classroom_variation_gate"] == {
        "required": True,
        "status": "pass",
        "evidence_label": "classroom_variation",
    }
    assert result["required_room_ids"] == list(bundle.ROOM_IDS)
    assert "classroom_variation" in result["required_labels"]
    assert "reference_school_layout_seed" in result["required_labels"]
    assert "sam3d_generation_preflight" in result["required_labels"]
    assert "sam3d_generation_glb" in result["required_labels"]
    assert "artiverse_visual_resources" in result["required_labels"]
    packaged_visual_receipt = (
        Path(fixture["scene_dir"])
        / bundle.EVIDENCE_DIRECTORY
        / "run"
        / "artiverse_visual_resources.json"
    )
    assert packaged_visual_receipt.is_file()
    assert any(
        item["label"] == "artiverse_visual_resources"
        and item["sha256"] == bundle._sha256_file(packaged_visual_receipt)
        for item in result["evidence"]
    )
    assert bundle.verify_bundle(**fixture) == result


def test_artiverse_visual_resource_receipt_requires_complete_zero_mutation_audit(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    receipt_path = Path(fixture["run_dir"]) / "artiverse_visual_resources.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["audit"]["source_files_written"] = 1
    payload = {key: value for key, value in receipt.items() if key != "attestation"}
    receipt["attestation"] = bundle.artiverse_visual._attestation(payload)
    _write_json(receipt_path, receipt)

    with pytest.raises(
        bundle.AcceptanceBundleError, match="zero source mutation"
    ):
        bundle.create_bundle(**fixture)


def test_artiverse_visual_resource_receipt_rejects_direct_glb_target(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    receipt_path = Path(fixture["run_dir"]) / "artiverse_visual_resources.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["mapping_policy"]["target"] = "direct_glb"
    payload = {key: value for key, value in receipt.items() if key != "attestation"}
    receipt["attestation"] = bundle.artiverse_visual._attestation(payload)
    _write_json(receipt_path, receipt)

    with pytest.raises(
        bundle.AcceptanceBundleError,
        match="exact mapping policy",
    ):
        bundle.create_bundle(**fixture)


def test_artiverse_visual_resource_document_requires_all_500_derivations(
    tmp_path: Path,
) -> None:
    receipt = _artiverse_visual_receipt(tmp_path)
    receipt["audit"]["assets"].pop()  # type: ignore[index,union-attr]
    payload = {key: value for key, value in receipt.items() if key != "attestation"}
    receipt["attestation"] = bundle.artiverse_visual._attestation(payload)

    with pytest.raises(
        bundle.AcceptanceBundleError,
        match="every per-asset derivation",
    ):
        bundle._validate_document(
            "artiverse_visual_resources",
            receipt,
            paths={},
            context=None,  # type: ignore[arg-type]
        )


def test_full_sam3d_generation_glb_is_required_and_hash_bound(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    repo = Path(fixture["repo_dir"])
    preflight = repo / "outputs" / "preflight" / Path(fixture["input_dir"]).name
    glb = sam3d_generation.canonical_paths(preflight)["glb"]
    glb.write_bytes(glb.read_bytes() + b"mutated")

    with pytest.raises(
        bundle.AcceptanceBundleError,
        match="SAM3D full offline generation proof is invalid",
    ):
        bundle.create_bundle(**fixture)


def test_reference_school_native_layout_seed_is_required(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    scene = Path(fixture["scene_dir"])
    (scene / "quality_gates" / "reference_school_layout_seed.json").unlink()

    with pytest.raises(
        bundle.AcceptanceBundleError, match="reference_school_layout_seed"
    ):
        bundle.create_bundle(**fixture)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("schema_version", 999, "schema/profile"),
        ("profile", "different_school", "schema/profile"),
        ("structural_layout_sha256", "0" * 64, "differs structurally"),
    ],
)
def test_reference_school_native_layout_seed_rejects_stale_identity_or_layout(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    fixture = _make_fixture(tmp_path)
    receipt_path = (
        Path(fixture["scene_dir"])
        / "quality_gates"
        / "reference_school_layout_seed.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt[field] = replacement
    _write_json(receipt_path, receipt)

    with pytest.raises(bundle.AcceptanceBundleError, match=message):
        bundle.create_bundle(**fixture)


def test_reference_school_native_layout_seed_rehashes_geometry_manifest(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    geometry = (
        Path(fixture["scene_dir"])
        / "room_geometry"
        / "reference_school.sdf"
    )
    geometry.write_text("<sdf version='1.9'><changed/></sdf>\n", encoding="utf-8")

    with pytest.raises(
        bundle.AcceptanceBundleError, match="native geometry differs"
    ):
        bundle.create_bundle(**fixture)


def test_external_manifest_semantically_binds_the_sqz_record(tmp_path: Path) -> None:
    from scripts import two_gpu_drake_acceptance_contract as transfer

    fixture = _make_fixture(tmp_path)
    result = bundle.create_bundle(**fixture)
    manifest = tmp_path / "scene_000.sha256"
    verified = transfer.create_package_manifest(
        Path(fixture["scene_dir"]),
        manifest,
        ATTEMPT_ID,
    )
    sqz = verified["sqz_acceptance_record"]
    assert sqz["run_attempt_id"] == ATTEMPT_ID
    assert sqz["relative_path"] == "combined_house/sqz_acceptance_record.json"
    assert sqz["self_attestation_sha256"] == result["self_attestation"]["sha256"]
    assert sqz["evidence_attestation_sha256"] == result["evidence_attestation"]["sha256"]


def test_missing_required_gate_leaves_no_packaged_evidence(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    scene = fixture["scene_dir"]
    assert isinstance(scene, Path)
    (scene / "quality_gates" / "floor_plan_layout.json").unlink()
    with pytest.raises(bundle.AcceptanceBundleError, match="floor_plan_layout_gate"):
        bundle.create_bundle(**fixture)
    assert not (scene / bundle.EVIDENCE_DIRECTORY).exists()
    assert not Path(fixture["output"]).exists()


def test_artiverse_survivor_must_exist_in_combined_house_state(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    scene = Path(fixture["scene_dir"])
    usage_path = scene / "combined_house" / "artiverse_usage.json"
    usage = json.loads(usage_path.read_text(encoding="utf-8"))
    house_path = scene / "combined_house" / "house_state.json"
    house = json.loads(house_path.read_text(encoding="utf-8"))
    house["rooms"]["library"]["objects"].pop("bookcase_1")
    _write_json(house_path, house)
    usage["house_state"] = _record(house_path)

    with pytest.raises(
        bundle.AcceptanceBundleError,
        match="final-survivor evidence differs from the combined house state",
    ):
        bundle._validate_artiverse_usage(
            usage,
            paths={"house_state": house_path},
            scene_dir=scene,
        )


def test_mutated_package_file_invalidates_verify_only(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    bundle.create_bundle(**fixture)
    scene = Path(fixture["scene_dir"])
    (scene / "combined_house" / "house_state.json").write_text(
        "{\"rooms\": {}}\n", encoding="utf-8"
    )
    with pytest.raises(bundle.AcceptanceBundleError, match="changed: house_state"):
        bundle.verify_bundle(**fixture)


def test_receipt_field_tamper_breaks_self_attestation(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    bundle.create_bundle(**fixture)
    output = Path(fixture["output"])
    receipt = json.loads(output.read_text(encoding="utf-8"))
    receipt["created_at_utc"] = "2099-01-01T00:00:00Z"
    _write_json(output, receipt)
    with pytest.raises(bundle.AcceptanceBundleError, match="self-attestation"):
        bundle.verify_bundle(**fixture)


def test_external_copy_failure_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path)
    scene = Path(fixture["scene_dir"])
    original = bundle.shutil.copyfile
    calls = 0

    def fail_second(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected copy failure")
        return original(source, destination)

    monkeypatch.setattr(bundle.shutil, "copyfile", fail_second)
    with pytest.raises(OSError, match="injected copy failure"):
        bundle.create_bundle(**fixture)
    assert not (scene / bundle.EVIDENCE_DIRECTORY).exists()
    assert not list((scene / "quality_gates").glob(".final_acceptance_evidence.*.tmp"))
    assert not Path(fixture["output"]).exists()


def test_all_three_room_evidence_sets_must_have_exactly_eleven_rooms(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    scene = Path(fixture["scene_dir"])
    (scene / "quality_gates" / "room_self_exam" / "classroom_06.json").unlink()
    with pytest.raises(bundle.AcceptanceBundleError, match="room_visual:classroom_06"):
        bundle.create_bundle(**fixture)
    assert not (scene / bundle.EVIDENCE_DIRECTORY).exists()


def test_classroom_variation_and_leaked_staging_are_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    scene = Path(fixture["scene_dir"])
    leaked = scene / "nested" / ".mujoco_export.failed.attempt"
    leaked.mkdir(parents=True)
    with pytest.raises(bundle.AcceptanceBundleError, match="leaked staging/backup"):
        bundle.create_bundle(**fixture)
    leaked.rmdir()
    variation_path = scene / "quality_gates" / "classroom_variation.json"
    variation = json.loads(variation_path.read_text(encoding="utf-8"))
    variation["status"] = "fail"
    variation["attestation_sha256"] = bundle._sha256_bytes(
        bundle._canonical_json(
            {key: value for key, value in variation.items() if key != "attestation_sha256"}
        )
    )
    _write_json(variation_path, variation)
    with pytest.raises(bundle.AcceptanceBundleError, match="Classroom-variation gate"):
        bundle.create_bundle(**fixture)


def test_articulated_motion_repeat_receipt_cannot_be_status_only(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    scene = Path(fixture["scene_dir"])
    receipt_path = (
        scene / "quality_gates" / "articulated_motion_verification.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["drake_motion_repeated"] = False
    receipt.pop("attestation")
    receipt["attestation"] = {
        "algorithm": "sha256",
        "sha256": bundle._sha256_bytes(bundle._canonical_json(receipt)),
    }
    _write_json(receipt_path, receipt)

    with pytest.raises(
        bundle.AcceptanceBundleError,
        match="Articulated-motion repeat proof schema/status/binding",
    ):
        bundle.create_bundle(**fixture)


def test_navigation_repeat_receipt_requires_full_eleven_route_recompute(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    scene = Path(fixture["scene_dir"])
    receipt_path = (
        scene / "quality_gates" / "school_navigation_verification.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["recomputed_gate"]["route_count"] = 10
    receipt.pop("attestation")
    receipt["attestation"] = {
        "algorithm": "sha256(canonical JSON excluding attestation)",
        "payload_sha256": bundle._sha256_bytes(bundle._canonical_json(receipt)),
        "self_attested_status": "pass",
    }
    _write_json(receipt_path, receipt)

    with pytest.raises(
        bundle.AcceptanceBundleError,
        match="School-navigation full recomputation differs",
    ):
        bundle.create_bundle(**fixture)


@pytest.mark.parametrize("change", ["mutate", "add", "remove"])
def test_simulator_report_is_rehashed_against_exact_live_file_set(
    tmp_path: Path,
    change: str,
) -> None:
    fixture = _make_fixture(tmp_path)
    export_root = Path(fixture["scene_dir"]) / "mujoco_export"
    if change == "mutate":
        (export_root / "scene.xml").write_text(
            "<mujoco model='changed'/>", encoding="utf-8"
        )
    elif change == "add":
        _write(export_root / "unreported.bin", b"unreported")
    else:
        (export_root / "usd" / "Payload" / "Physics.usda").unlink()

    with pytest.raises(
        bundle.AcceptanceBundleError,
        match="Simulator inventory|live file set|missing expected USD",
    ):
        bundle.create_bundle(**fixture)


def test_simulator_report_cannot_map_to_a_different_export_root(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    scene = Path(fixture["scene_dir"])
    report_path = scene / "quality_gates" / "simulator_exports.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    elsewhere = tmp_path / "elsewhere" / "mujoco_export"
    elsewhere.mkdir(parents=True)
    report["output_dir"] = str(elsewhere.resolve())
    _write_json(report_path, report)

    with pytest.raises(
        bundle.AcceptanceBundleError,
        match="different output root",
    ):
        bundle.create_bundle(**fixture)


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ("attempt_marker", "attempt marker.*unsafe"),
        ("mjcf_root", "MJCF root.*unsafe"),
        ("validated_stage", "validated stage.*unsafe"),
    ],
)
def test_simulator_report_path_mappings_cannot_escape(
    tmp_path: Path,
    mapping: str,
    message: str,
) -> None:
    fixture = _make_fixture(tmp_path)
    scene = Path(fixture["scene_dir"])
    outside = _write(tmp_path / "outside.usda", "outside")
    report_path = scene / "quality_gates" / "simulator_exports.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    escaping = "../" + outside.name
    if mapping == "attempt_marker":
        report["attempt_marker"]["path"] = escaping
    elif mapping == "mjcf_root":
        report["mujoco"]["scene_xml"] = escaping
    else:
        report["usd"]["validated_stages"][0]["path"] = escaping
    _write_json(report_path, report)

    with pytest.raises(bundle.AcceptanceBundleError, match=message):
        bundle.create_bundle(**fixture)


def test_simulator_export_rejects_link_like_entries(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    scene = Path(fixture["scene_dir"])
    outside = _write(tmp_path / "outside.bin", b"outside")
    linked = scene / "mujoco_export" / "linked.bin"
    try:
        linked.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Symlinks are unavailable in this test environment: {exc}")

    with pytest.raises(bundle.AcceptanceBundleError, match="link-like"):
        bundle.create_bundle(**fixture)


def test_simulator_export_rejects_special_entries(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable on this platform")
    fixture = _make_fixture(tmp_path)
    scene = Path(fixture["scene_dir"])
    fifo = scene / "mujoco_export" / "unexpected.fifo"
    try:
        os.mkfifo(fifo)
    except OSError as exc:
        pytest.skip(f"FIFO creation is unavailable in this test environment: {exc}")

    with pytest.raises(bundle.AcceptanceBundleError, match="special"):
        bundle.create_bundle(**fixture)
