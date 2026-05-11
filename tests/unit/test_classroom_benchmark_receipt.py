from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scripts import preflight_sam3d_generation as sam3d_generation
from scripts import preflight_sam3d_offline as sam3d_preflight
from scripts import seed_reference_school_layout as reference_layout
from scripts.classroom_benchmark_receipt import (
    BenchmarkReceiptError,
    EXPECTED_EXTERNAL_INPUT_BINDING,
    _events,
    _gpu_samples,
    _validate_artiverse_visual_resources,
    _validate_sam3d_preflight,
    artiverse_visual,
    create_receipt,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_SAM3D_MESH_STATS = {
    "loader": "trimesh.load(force='scene', process=False)",
    "geometry_count": 1,
    "vertex_count": 8,
    "face_count": 12,
    "bounds": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
}


def _artiverse_visual_receipt(root: Path) -> dict[str, object]:
    assets = [
        {
            "object_id": f"artiverse/armoire/fixture/model_{index:04d}",
            "source_sdf_path": (
                f"data/armoire/fixture/model_{index:04d}/scenesmith_artiverse.sdf"
            ),
            "source_sdf_sha256": "3" * 64,
            "source_tree_sha256": "4" * 64,
            "mapped_link_count": 1,
            "visual_mapping_count": 2,
            "unique_glb_count": 1,
            "glb_primitive_count": 1,
            "derived_gltf_count": 1,
            "derived_bin_count": 1,
            "gltf_buffer_view_count": 4,
            "gltf_accessor_count": 3,
            "gltf_material_count": 1,
            "gltf_texture_count": 1,
            "gltf_image_count": 1,
            "collision_binding_count": 2,
            "derived_resource_inventory_sha256": "5" * 64,
            "mapping_sha256": "6" * 64,
            "collision_binding_sha256": "7" * 64,
        }
        for index in range(500)
    ]
    payload: dict[str, object] = {
        "schema_version": artiverse_visual.SCHEMA_VERSION,
        "contract": artiverse_visual.CONTRACT_NAME,
        "status": "pass",
        "hash_algorithm": "sha256",
        "mapping_policy": artiverse_visual.mapping_policy(),
        "authority": {"indexed_count": 500},
        "runtime_code": [
            {
                "path": str(
                    root
                    / "scenesmith"
                    / "agent_utils"
                    / "artiverse_visual_normalization.py"
                ),
                "size_bytes": 100,
                "sha256": "1" * 64,
            },
            {
                "path": str(
                    root / "scenesmith" / "agent_utils" / "asset_manager.py"
                ),
                "size_bytes": 100,
                "sha256": "2" * 64,
            },
        ],
        "runtime_code_revalidated_after_audit": True,
        "audit": {
            "indexed_asset_count": 500,
            "audited_asset_count": 500,
            "mapped_link_count": 500,
            "visual_mapping_count": 1000,
            "unique_glb_count": 500,
            "glb_primitive_count": 500,
            "derived_gltf_count": 500,
            "derived_bin_count": 500,
            "gltf_buffer_view_count": 2000,
            "gltf_accessor_count": 1500,
            "gltf_material_count": 500,
            "gltf_texture_count": 500,
            "gltf_image_count": 500,
            "collision_binding_count": 1000,
            "mapping_inventory_sha256": "8" * 64,
            "collision_inventory_sha256": "9" * 64,
            "asset_inventory_sha256": "a" * 64,
            "assets": assets,
            "mapping_target": "publisher_glb_derived_external_gltf",
            "publisher_glb_json_bin_resources_validated": True,
            "publisher_position_normal_accessors_validated": True,
            "publisher_material_image_references_validated": True,
            "derived_external_gltf_hashes_precomputed": True,
            "source_tree_identity_revalidated_after_audit": True,
            "source_hash_revalidated_after_each_asset_audit": True,
            "source_files_written": 0,
        },
    }
    return {**payload, "attestation": artiverse_visual._attestation(payload)}


def test_artiverse_visual_receipt_requires_external_gltf_target(
    tmp_path: Path,
) -> None:
    receipt = _artiverse_visual_receipt(tmp_path)
    receipt["mapping_policy"]["target"] = "direct_glb"  # type: ignore[index]
    payload = {key: value for key, value in receipt.items() if key != "attestation"}
    receipt["attestation"] = artiverse_visual._attestation(payload)

    with pytest.raises(BenchmarkReceiptError, match="exact mapping policy"):
        _validate_artiverse_visual_resources(receipt)  # type: ignore[arg-type]


def test_artiverse_visual_receipt_requires_all_500_derivations(
    tmp_path: Path,
) -> None:
    receipt = _artiverse_visual_receipt(tmp_path)
    receipt["audit"]["assets"].pop()  # type: ignore[index,union-attr]
    payload = {key: value for key, value in receipt.items() if key != "attestation"}
    receipt["attestation"] = artiverse_visual._attestation(payload)

    with pytest.raises(BenchmarkReceiptError, match="every per-asset derivation"):
        _validate_artiverse_visual_resources(receipt)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _fixture_sam3d_mesh_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sam3d_generation,
        "_mesh_stats",
        lambda _path: dict(TEST_SAM3D_MESH_STATS),
    )


def _sam3d_preflight() -> dict[str, object]:
    _image, target_mask = sam3d_preflight._canonical_inference_input()
    document: dict[str, object] = {
        "schema_version": sam3d_preflight.RESULT_SCHEMA_VERSION,
        "status": "pass",
        "offline": True,
        "offline_environment": {
            name: "1" for name in sam3d_preflight.OFFLINE_VARIABLES
        },
        "sam3_checkpoint": "/cache/sam3.pt",
        "pipeline_config": "/cache/pipeline.yaml",
        "model_loaded": True,
        "pipeline_loaded": True,
        "gpu": "NVIDIA A10",
        "visible_gpu_count": 1,
        "inference_smoke": sam3d_preflight._build_inference_smoke(target_mask),
        "inference_smoke_verification": {
            "status": "pass",
            "critical_issues": [],
        },
        "evidence": {"schema_version": 1, "artifact_count": 1},
        "evidence_verification": {"status": "pass", "critical_issues": []},
    }
    document["attestation"] = {
        "algorithm": "sha256",
        "sha256": sam3d_preflight._attestation_sha256(document),
    }
    return document


def _write_sam3d_generation(preflight_dir: Path, repo: Path) -> None:
    source = REPO_ROOT / sam3d_generation.CANONICAL_INPUT_RELATIVE
    target = repo / sam3d_generation.CANONICAL_INPUT_RELATIVE
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    paths = sam3d_generation.canonical_paths(preflight_dir)
    paths["artifact_dir"].mkdir(parents=True, exist_ok=True)
    paths["glb"].write_bytes(b"glb-proof")
    with Image.open(target) as opened:
        source_image = opened.convert("RGB")
    mask_image = Image.new("L", source_image.size, 0)
    ImageDraw.Draw(mask_image).rectangle((256, 384, 767, 1151), fill=255)
    mask_image.save(paths["mask"])
    masked_image = Image.new("RGB", source_image.size, "black")
    masked_image.paste(source_image, mask=mask_image)
    masked_image.save(paths["masked_image"])

    def artifact(key: str, name: str) -> dict[str, object]:
        path = paths[key]
        return {
            "relative_path": name,
            "size_bytes": path.stat().st_size,
            "sha256": sam3d_generation._sha256_file(path),
        }

    models = [
        {
            "role": role,
            "path": f"/cache/{name}",
            "kind": "file",
            "size_bytes": 1,
            "sha256": digest * 64,
        }
        for role, name, digest in (
            ("sam3_checkpoint", "sam3.pt", "b"),
            ("pipeline_config", "pipeline.yaml", "c"),
        )
    ]
    runtime_root = (repo / "synthetic_runtime_identity").resolve()
    runtime_distributions: list[dict[str, object]] = []
    for (
        distribution_name,
        module_name,
        extension_globs,
        resource_globs,
    ) in sam3d_generation.RUNTIME_COMPONENTS:
        slug = distribution_name.replace("-", "_")

        def bound_files(
            patterns: tuple[str, ...], prefix: str, digest: str
        ) -> list[dict[str, object]]:
            return [
                {
                    "distribution_relative_path": f"{slug}/{prefix}_{index}",
                    "resolved_path": str(
                        runtime_root / slug / f"{prefix}_{index}"
                    ),
                    "size_bytes": 1,
                    "sha256": digest * 64,
                }
                for index, _pattern in enumerate(patterns)
            ]

        runtime_distributions.append(
            {
                "requested_name": distribution_name,
                "module_name": module_name,
                "canonical_name": distribution_name,
                "version": "1.0",
                "record_path": str(runtime_root / slug / "RECORD"),
                "record_size_bytes": 1,
                "record_sha256": "9" * 64,
                "module_origin": str(runtime_root / slug / "__init__.py"),
                "module_origin_size_bytes": 1,
                "module_origin_sha256": "6" * 64,
                "required_extension_globs": list(extension_globs),
                "core_extension_files": bound_files(
                    extension_globs, "extension", "7"
                ),
                "required_resource_globs": list(resource_globs),
                "runtime_resource_files": bound_files(
                    resource_globs, "resource", "8"
                ),
            }
        )
    result: dict[str, object] = {
        "schema_id": sam3d_generation.SCHEMA_ID,
        "schema_version": sam3d_generation.SCHEMA_VERSION,
        "status": "pass",
        "offline": True,
        "offline_environment": {name: "1" for name in sam3d_generation.OFFLINE_VARIABLES},
        "paid_api_calls": 0,
        "network_access_required": False,
        "production_entrypoint": "scenesmith.agent_utils.geometry_generation_server.geometry_generation.generate_geometry_from_image",
        "generation_parameters": {
            "backend": "sam3d",
            "mode": "foreground",
            "object_description": None,
            "threshold": 0.5,
            "use_pipeline_caching": False,
        },
        "input": {
            "repository_relative_path": sam3d_generation.CANONICAL_INPUT_RELATIVE.as_posix(),
            "size_bytes": target.stat().st_size,
            "sha256": sam3d_generation.CANONICAL_INPUT_SHA256,
            "width": sam3d_generation.CANONICAL_INPUT_WIDTH,
            "height": sam3d_generation.CANONICAL_INPUT_HEIGHT,
        },
        "sam3_checkpoint": "/cache/sam3.pt",
        "pipeline_config": "/cache/pipeline.yaml",
        "model_artifacts": {
            "schema_version": 1,
            "algorithm": "sha256",
            "artifact_count": 2,
            "artifacts": models,
        },
        "code_artifacts": [
            {
                "path": relative,
                "relative_path": relative,
                "size_bytes": 1,
                "sha256": "a" * 64,
            }
            for relative in sam3d_generation.CODE_RELATIVE_PATHS
        ],
        "executable_source_trees": [
            {
                "role": role,
                "path": path,
                "resolved_path": str((repo / "resolved" / role).resolve()),
                "kind": "package_source_tree",
                "import_name": sam3d_generation.SOURCE_TREE_MODULES[role],
                "import_origin_path": (
                    Path(path) / "__init__.py"
                ).as_posix(),
                "resolved_import_origin_path": str(
                    (repo / "resolved" / role / "__init__.py").resolve()
                ),
                "import_origin_size_bytes": 1,
                "import_origin_sha256": "5" * 64,
                "sha256": digest * 64,
                "file_count": 1,
                "total_bytes": 1,
                "selection": {
                    "regular_package_files": (
                        "all regular files excluding __pycache__ contents and *.pyc/*.pyo"
                    ),
                    "required_runtime_resources": (
                        ["assets/bpe_simple_vocab_16e6.txt.gz"]
                        if role == "external_sam3_python_source"
                        else []
                    ),
                    "links_and_special_files_forbidden": True,
                },
            }
            for role, path, digest in (
                (
                    "external_sam3_python_source",
                    "external/SAM3/sam3",
                    "d",
                ),
                (
                    "external_sam3d_objects_python_source",
                    "external/sam-3d-objects/sam3d_objects",
                    "e",
                ),
                (
                    sam3d_generation.MOGE_SOURCE_ROLE,
                    "/site-packages/moge",
                    "f",
                ),
            )
        ],
        "runtime_identity": {
            "python_implementation": "cpython",
            "python_version": "3.11.0",
            "distribution_count": len(runtime_distributions),
            "distributions": runtime_distributions,
            "scope_note": "synthetic package/runtime identity",
        },
        "outputs": {
            "glb": {
                **artifact("glb", sam3d_generation.GLB_NAME),
                "mesh": dict(TEST_SAM3D_MESH_STATS),
            },
            "mask": artifact("mask", sam3d_generation.MASK_NAME),
            "masked_image": artifact(
                "masked_image", sam3d_generation.MASKED_IMAGE_NAME
            ),
            "image_validation": sam3d_generation._image_stats_and_semantics(
                target, paths["mask"], paths["masked_image"]
            ),
        },
        "validation": {
            "status": "pass",
            "trimesh_reloaded": True,
            "critical_issues": [],
        },
        "elapsed_seconds": 1.0,
    }
    result["attestation"] = sam3d_generation._attestation(result)
    paths["receipt"].write_text(json.dumps(result) + "\n", encoding="utf-8")


@pytest.mark.parametrize("mutation", ("missing", "failed", "tampered"))
def test_benchmark_rejects_invalid_sam3_image_text_inference_proof(
    mutation: str,
) -> None:
    document = _sam3d_preflight()
    if mutation == "missing":
        document.pop("inference_smoke")
    elif mutation == "failed":
        document["inference_smoke"]["status"] = "fail"
    else:
        document["inference_smoke"]["output"]["foreground_pixels"] = 0
    document["attestation"] = {
        "algorithm": "sha256",
        "sha256": sam3d_preflight._attestation_sha256(document),
    }

    with pytest.raises(BenchmarkReceiptError, match="SAM3D offline preflight"):
        _validate_sam3d_preflight(document)


def _write_events(path: Path, names: list[str]) -> None:
    path.write_text(
        "".join(
            f"{1_700_000_000 + index}\t2026-07-11T00:00:{index:02d}Z\t{name}\n"
            for index, name in enumerate(names)
        ),
        encoding="utf-8",
    )


def test_fresh_benchmark_event_chain_records_every_real_room_stage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.tsv"
    names = [
        "benchmark_start",
        "artiverse_visual_preflight_complete",
        "sam3d_generation_preflight_complete",
        "preflight_complete",
        "floor_layout_gate_complete",
        "room_start",
        "room_generation_start",
        "room_generation_complete",
        "blend_refresh_complete",
        "render_complete",
        "deterministic_gate_complete",
        "visual_gate_complete",
        "benchmark_complete",
    ]
    _write_events(path, names)

    events = _events(path)

    assert [event["name"] for event in events] == names
    assert events[-1]["epoch"] - events[0]["epoch"] == len(names) - 1
    assert all(event["seconds_until_next_event"] == 1 for event in events[:-1])


def test_benchmark_event_chain_rejects_skipped_real_stage(tmp_path: Path) -> None:
    path = tmp_path / "events.tsv"
    _write_events(
        path,
        [
            "benchmark_start",
            "artiverse_visual_preflight_complete",
            "sam3d_generation_preflight_complete",
            "preflight_complete",
            "floor_layout_gate_complete",
            "room_start",
            "room_generation_start",
            "room_generation_complete",
            "blend_refresh_complete",
            "render_complete",
            "visual_gate_complete",
            "benchmark_complete",
        ],
    )

    with pytest.raises(BenchmarkReceiptError, match="deterministic_gate_complete"):
        _events(path)


def test_gpu_sampler_reports_peak_vram_and_utilization(tmp_path: Path) -> None:
    path = tmp_path / "gpu.csv"
    path.write_text(
        "2026/07/11 00:00:00.000, 0, NVIDIA A10, 25, 1024, 23028\n"
        "2026/07/11 00:00:05.000, 0, NVIDIA A10, 91, 17400, 23028\n",
        encoding="utf-8",
    )

    result = _gpu_samples(path)

    assert result["sample_count"] == 2
    assert result["gpu_indices"] == [0]
    assert result["gpu_names"] == ["NVIDIA A10"]
    assert result["peak_utilization_percent"] == 91
    assert result["mean_utilization_percent"] == 58
    assert result["peak_memory_used_mib"] == 17400


def test_production_benchmark_mode_uses_exact_runner_and_stops_before_global_gates() -> None:
    runner = (
        REPO_ROOT / "remote_jobs" / "run_full_quality_school_sqz.sh"
    ).read_text(encoding="utf-8")

    assert "SCENESMITH_EXECUTION_MODE:-full" in runner
    assert "benchmark_classroom_01)" in runner
    assert "ROOMS=(classroom_01)" in runner
    assert "python scripts/classroom_benchmark_receipt.py" in runner
    assert "python scripts/preflight_sam3d_generation.py" in runner
    assert "room_evidence_reused" in runner
    completion_function = runner.split("complete_classroom_benchmark() {", 1)[1].split(
        "\n}", 1
    )[0]
    assert "scripts/pipeline_code_contract.py" in completion_function
    assert "scripts/preflight_sam3d_offline.py" in completion_function
    assert "scripts/preflight_sam3d_generation.py" in completion_function
    assert "--verify-only" in completion_function
    assert "scripts/validate_input_manifest.py" in completion_function
    assert completion_function.index("--verify-only") < completion_function.index(
        "record_benchmark_event benchmark_complete"
    )
    visual_event = runner.index("record_benchmark_event visual_gate_complete")
    completion_call = runner.index("complete_classroom_benchmark", visual_event)
    global_gate = runner.index('echo "[gate] confirming every room gate together"')
    assert visual_event < completion_call < global_gate
    assert "full-house stages were not started" in runner


def test_create_receipt_binds_passing_room_timing_gpu_and_artifacts(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    run = repo / "outputs" / "2026-07-10" / "run"
    scene = run / "scene_000"
    inputs = repo / "inputs" / "full_quality_school_reference_20260710"
    preflight = repo / "outputs" / "preflight" / "full_quality_school_reference_20260710"
    room_final = scene / "room_classroom_01" / "scene_states" / "final_scene"
    deterministic_dir = scene / "quality_gates" / "room_self_exam_deterministic"
    visual_dir = scene / "quality_gates" / "room_self_exam"
    review_dir = scene / "review" / "room_review_renders"
    for directory in (
        repo / "remote_jobs",
        inputs,
        preflight,
        room_final,
        deterministic_dir,
        visual_dir,
        review_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    (repo / "CODEX_SCENESMITH_FULL_QUALITY_PIPELINE.md").write_text(
        "full quality\n", encoding="utf-8"
    )
    (repo / "remote_jobs" / "run_full_quality_school_sqz.sh").write_text(
        "#!/usr/bin/env bash\n", encoding="utf-8"
    )
    for name in ("prompt_original.txt", "scene_contract_appendix.txt", "prompt.txt"):
        (inputs / name).write_text("school prompt\n", encoding="utf-8")
    (inputs / "prompt.csv").write_text("prompt\nschool prompt\n", encoding="utf-8")
    (inputs / "input_manifest.json").write_text("{}\n", encoding="utf-8")
    (inputs / "reference.png").write_bytes(b"reference")
    policy = {
        agent: {
            "general_asset_source": "objaverse"
            if agent == "manipuland_agent"
            else "generated",
            "backend": "" if agent == "manipuland_agent" else "sam3d",
            "coacd_max_convex_hull": 32,
            "vhacd_max_convex_hulls": 32,
        }
        for agent in (
            "furniture_agent",
            "wall_agent",
            "ceiling_agent",
            "manipuland_agent",
        )
    }
    policy["articulated_contract"] = {
        "articulated_strategy_enabled": True,
        "artiverse_strategy_enabled": True,
        "artvip_enabled_agents": {
            agent: True
            for agent in (
                "furniture_agent",
                "wall_agent",
                "ceiling_agent",
                "manipuland_agent",
            )
        },
        "artvip": {
            "enabled": True,
            "data_path_exists": True,
            "embeddings_path_exists": True,
            "missing_embedding_files": [],
        },
        "artiverse": {
            "enabled": True,
            "data_path_exists": True,
            "embeddings_path_exists": True,
            "missing_embedding_files": [],
        },
    }
    (run / "resolved_asset_policy.json").write_text(
        json.dumps(policy) + "\n", encoding="utf-8"
    )
    (run / "artiverse_visual_resources.json").write_text(
        json.dumps(_artiverse_visual_receipt(repo)) + "\n", encoding="utf-8"
    )
    (scene / "house_layout.json").write_text("{}\n", encoding="utf-8")
    (scene / "floor_plans").mkdir(parents=True)
    (scene / "floor_plans" / "floor.gltf").write_bytes(b"floor")
    (scene / "room_geometry").mkdir(parents=True)
    (scene / "room_geometry" / "room.sdf").write_bytes(b"sdf")
    (scene / "package.xml").write_bytes(b"package")
    (room_final / "scene_state.json").write_text("{}\n", encoding="utf-8")
    (room_final / "scene.blend").write_bytes(b"blend")

    status_paths = [
        run / "pipeline_code_contract.json",
        run / "input_manifest_validation.json",
        run / "materials_contract_validation.json",
        run / "artiverse_preparation_validation.json",
        run / "articulated_router_validation.json",
        preflight / "objathor_retrieval_offline.json",
        preflight / "sam3d_offline_load.json",
        preflight / "vlm_vision_smoke.json",
        scene / "quality_gates" / "floor_plan_layout.json",
        scene / "quality_gates" / "reference_school_layout_seed.json",
        scene / "quality_gates" / "room_prompt_binding.json",
    ]
    for path in status_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass"}\n', encoding="utf-8")
    (preflight / "sam3d_offline_load.json").write_text(
        json.dumps(_sam3d_preflight()) + "\n",
        encoding="utf-8",
    )
    _write_sam3d_generation(preflight, repo)
    anchored = {
        "status": "pass",
        "external_input_binding": EXPECTED_EXTERNAL_INPUT_BINDING,
    }
    for path in (
        run / "pipeline_code_contract.json",
        run / "input_manifest_validation.json",
    ):
        path.write_text(json.dumps(anchored) + "\n", encoding="utf-8")
    seed_artifacts = reference_layout._tree_manifest(
        scene, ("floor_plans", "room_geometry", "package.xml")
    )
    (scene / "quality_gates" / "reference_school_layout_seed.json").write_text(
        json.dumps(
            {
                "schema_version": reference_layout.SEED_SCHEMA_VERSION,
                "status": "pass",
                "profile": reference_layout.PROFILE,
                "implementation": "native_scenesmith_deterministic_reference_layout",
                "structural_layout_sha256": reference_layout._structural_layout_sha256(
                    {}
                ),
                "artifact_count": len(seed_artifacts),
                "artifact_manifest_sha256": reference_layout._manifest_sha256(
                    seed_artifacts
                ),
                "artifacts": seed_artifacts,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    deterministic_path = deterministic_dir / "classroom_01.json"
    deterministic_path.write_text(
        '{"room_id":"classroom_01","status":"pass"}\n', encoding="utf-8"
    )
    reviews = []
    for index in range(3):
        path = review_dir / f"classroom_01_view_{index}.png"
        path.write_bytes(f"image-{index}".encode())
        reviews.append(str(path.resolve()))
    visual_path = visual_dir / "classroom_01.json"
    visual_path.write_text(
        json.dumps(
            {
                "room_id": "classroom_01",
                "status": "pass",
                "threshold": 7,
                "scores": {
                    "object_relevance": 8,
                    "placement_realism": 8,
                    "clearance_and_access": 8,
                    "collision_risk": 8,
                    "prompt_alignment": 8,
                },
                "review_images": reviews,
                "evidence": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    events = run / "classroom_01_full_quality_benchmark.events.tsv"
    _write_events(
        events,
        [
            "benchmark_start",
            "artiverse_visual_preflight_complete",
            "sam3d_generation_preflight_complete",
            "preflight_complete",
            "floor_layout_gate_complete",
            "room_start",
            "room_generation_start",
            "room_generation_complete",
            "blend_refresh_complete",
            "render_complete",
            "deterministic_gate_complete",
            "visual_gate_complete",
            "benchmark_complete",
        ],
    )
    gpu = run / "classroom_01_full_quality_benchmark.gpu.csv"
    gpu.write_text(
        "2026/07/11 00:00:00.000, 0, NVIDIA A10, 75, 17000, 23028\n",
        encoding="utf-8",
    )
    output = run / "classroom_01_full_quality_benchmark.json"

    receipt = create_receipt(
        repo_dir=repo,
        run_dir=run,
        scene_dir=scene,
        input_dir=inputs,
        events_path=events,
        gpu_samples_path=gpu,
        output=output,
        run_attempt_id="20260711T000000Z-123",
    )

    assert receipt["status"] == "pass"
    assert receipt["whole_house_assembly_performed"] is False
    assert receipt["timing"]["elapsed_seconds"] == 12
    assert receipt["gpu"]["peak_memory_used_mib"] == 17000
    assert output.is_file()


def test_benchmark_event_chain_requires_artiverse_visual_preflight(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.tsv"
    _write_events(
        path,
        [
            "benchmark_start",
            "sam3d_generation_preflight_complete",
            "preflight_complete",
            "floor_layout_gate_complete",
            "room_start",
            "room_evidence_reused",
            "benchmark_complete",
        ],
    )

    with pytest.raises(
        BenchmarkReceiptError, match="artiverse_visual_preflight_complete"
    ):
        _events(path)
