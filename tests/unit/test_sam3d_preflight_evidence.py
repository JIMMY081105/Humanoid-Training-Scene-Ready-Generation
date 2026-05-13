from __future__ import annotations

import json

from pathlib import Path

import pytest

import scripts.preflight_sam3d_offline as sam3d_preflight


def _runtime_evidence() -> dict:
    _, target_mask = sam3d_preflight._canonical_inference_input()
    return {
        "model_loaded": True,
        "pipeline_loaded": True,
        "gpu": "Test GPU",
        "visible_gpu_count": 1,
        "total_memory_bytes": 24 * 1024**3,
        "allocated_bytes": 10,
        "reserved_bytes": 20,
        "peak_allocated_bytes": 30,
        "peak_reserved_bytes": 40,
        "memory_stats_reset": True,
        "memory_stats_reset_error": None,
        "inference_smoke": sam3d_preflight._build_inference_smoke(target_mask),
    }


def _fixture(tmp_path: Path) -> dict[str, object]:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    sam3_checkpoint = checkpoint_dir / "sam3.pt"
    sam3_checkpoint.write_bytes(b"sam3-checkpoint")
    component_config = checkpoint_dir / "generator.yaml"
    component_config.write_text("module:\n  width: 4\n", encoding="utf-8")
    component_checkpoint = checkpoint_dir / "generator.ckpt"
    component_checkpoint.write_bytes(b"generator-checkpoint")
    pipeline_config = checkpoint_dir / "pipeline.yaml"
    pipeline_config.write_text(
        "_target_: test.Pipeline\n"
        "ss_generator_config_path: generator.yaml\n"
        "ss_generator_ckpt_path: generator.ckpt\n",
        encoding="utf-8",
    )

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    moge = cache_dir / "moge-model.pt"
    moge.write_bytes(b"moge-cache")
    dino_weights = cache_dir / "dinov2-weights.pth"
    dino_weights.write_bytes(b"dino-weights")
    dino_source = cache_dir / "facebookresearch_dinov2_main"
    dino_source.mkdir()
    (dino_source / "hubconf.py").write_text("MODEL = 'dino'\n", encoding="utf-8")
    package_dir = dino_source / "dinov2"
    package_dir.mkdir()
    (package_dir / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    # Bytecode and non-Python files are deliberately outside the source-tree hash.
    (dino_source / "README.md").write_text("documentation", encoding="utf-8")

    cache_artifacts = {
        "huggingface_cache:Ruicheng/moge-vitl:model.pt": moge,
        "torch_hub_cache:dinov2_vitl14_reg:source": dino_source,
        "torch_hub_cache:dinov2_vitl14_reg:weights": dino_weights,
    }
    output = tmp_path / "sam3d_offline_load.json"
    return {
        "sam3": sam3_checkpoint,
        "pipeline": pipeline_config,
        "component_config": component_config,
        "component_checkpoint": component_checkpoint,
        "moge": moge,
        "dino_weights": dino_weights,
        "dino_source": dino_source,
        "cache_artifacts": cache_artifacts,
        "output": output,
    }


def _run_passing(fixture: dict[str, object], monkeypatch) -> dict:
    monkeypatch.setattr(
        sam3d_preflight,
        "_load_pipelines_with_cuda",
        lambda _sam3, _pipeline: _runtime_evidence(),
    )
    return sam3d_preflight.run(
        fixture["sam3"],
        fixture["pipeline"],
        fixture["output"],
        cache_artifacts=fixture["cache_artifacts"],
    )


def test_pass_binds_every_local_and_cache_artifact(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path)
    result = _run_passing(fixture, monkeypatch)

    assert result["status"] == "pass", result.get("error")
    assert result["schema_version"] == 3
    assert result["evidence_verification"] == {
        "status": "pass",
        "critical_issues": [],
    }
    assert result["attestation"]["algorithm"] == "sha256"
    assert result["inference_smoke_verification"] == {
        "status": "pass",
        "critical_issues": [],
    }
    smoke = result["inference_smoke"]
    assert smoke["input"] == {
        "mode": "RGB",
        "size": [512, 512],
        "background_rgb": [255, 255, 255],
        "shape": "ellipse",
        "shape_bbox_xyxy": [156, 156, 356, 356],
        "shape_fill_rgb": [0, 0, 255],
        "raw_rgb_sha256": smoke["input"]["raw_rgb_sha256"],
    }
    assert smoke["request"] == {
        "function": "generate_mask",
        "mode": "object_description",
        "object_description": "circle",
    }
    assert smoke["runtime"] == {
        "device_type": "cuda",
        "autocast_dtype": "bfloat16",
        "bf16_supported": True,
    }
    assert sam3d_preflight._is_sha256(smoke["input"]["raw_rgb_sha256"])
    assert sam3d_preflight.verify_inference_smoke(smoke) == []
    roles = {entry["role"] for entry in result["evidence"]["artifacts"]}
    assert "sam3_checkpoint" in roles
    assert "pipeline_config" in roles
    assert any("ss_generator_config_path" in role for role in roles)
    assert any("ss_generator_ckpt_path" in role for role in roles)
    assert set(fixture["cache_artifacts"]) <= roles
    assert sam3d_preflight.verify_saved_result(
        result,
        fixture["sam3"],
        fixture["pipeline"],
        cache_artifacts=fixture["cache_artifacts"],
    ) == []


def test_artifact_mutation_during_model_load_fails_closed(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path)

    def mutate_during_load(_sam3: Path, _pipeline: Path) -> dict:
        target = fixture["component_checkpoint"]
        target.write_bytes(target.read_bytes() + b"-changed-during-load")
        return _runtime_evidence()

    monkeypatch.setattr(
        sam3d_preflight, "_load_pipelines_with_cuda", mutate_during_load
    )
    result = sam3d_preflight.run(
        fixture["sam3"],
        fixture["pipeline"],
        fixture["output"],
        cache_artifacts=fixture["cache_artifacts"],
    )

    assert result["status"] == "fail"
    assert result["model_loaded"] is True
    assert result["pipeline_loaded"] is True
    assert result["evidence_verification"]["status"] == "fail"
    assert "changed during offline load" in result["error"]
    assert "ss_generator_ckpt_path" in result["error"]


@pytest.mark.parametrize(
    "artifact",
    [
        "sam3",
        "pipeline",
        "component_config",
        "component_checkpoint",
        "moge",
        "dino_weights",
    ],
)
def test_file_mutation_after_pass_invalidates_saved_proof(
    tmp_path: Path, monkeypatch, artifact: str
) -> None:
    fixture = _fixture(tmp_path)
    result = _run_passing(fixture, monkeypatch)
    assert result["status"] == "pass"
    target = fixture[artifact]
    if target.suffix in {".yaml", ".yml"}:
        target.write_text(
            target.read_text(encoding="utf-8") + "\n# changed-after-pass\n",
            encoding="utf-8",
        )
    else:
        target.write_bytes(target.read_bytes() + b"-changed-after-pass")

    failures = sam3d_preflight.verify_saved_result(
        result,
        fixture["sam3"],
        fixture["pipeline"],
        cache_artifacts=fixture["cache_artifacts"],
    )

    assert any("SHA-256 mismatch" in failure for failure in failures)


def test_cached_source_tree_mutation_invalidates_saved_proof(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path)
    result = _run_passing(fixture, monkeypatch)
    source_file = fixture["dino_source"] / "dinov2" / "model.py"
    source_file.write_text("VALUE = 2\n", encoding="utf-8")

    failures = sam3d_preflight.verify_saved_result(
        result,
        fixture["sam3"],
        fixture["pipeline"],
        cache_artifacts=fixture["cache_artifacts"],
    )

    assert any("SHA-256 mismatch" in failure for failure in failures)


def test_verify_only_rejects_stale_proof_without_loading_models(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path)
    result = _run_passing(fixture, monkeypatch)
    assert result["status"] == "pass"
    fixture["moge"].write_bytes(b"replaced-moge-cache")

    monkeypatch.setattr(
        sam3d_preflight,
        "_load_pipelines_with_cuda",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("verify-only must not load models")
        ),
    )
    verified = sam3d_preflight.verify_output(
        fixture["output"],
        fixture["sam3"],
        fixture["pipeline"],
        cache_artifacts=fixture["cache_artifacts"],
    )

    assert verified["status"] == "fail"
    assert verified["evidence_verification"]["status"] == "fail"
    assert verified["inference_smoke_verification"] == {
        "status": "pass",
        "critical_issues": [],
    }
    assert "moge-vitl" in verified["verification_error"]


def test_json_field_tampering_breaks_attestation(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path)
    result = _run_passing(fixture, monkeypatch)
    result["gpu"] = "forged GPU"

    failures = sam3d_preflight.verify_saved_result(
        result,
        fixture["sam3"],
        fixture["pipeline"],
        cache_artifacts=fixture["cache_artifacts"],
    )

    assert "SAM3D preflight JSON attestation does not match its contents" in failures


def test_inference_smoke_tampering_breaks_attestation(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path)
    result = _run_passing(fixture, monkeypatch)
    result["inference_smoke"]["output"]["raw_mask_sha256"] = "0" * 64

    failures = sam3d_preflight.verify_saved_result(
        result,
        fixture["sam3"],
        fixture["pipeline"],
        cache_artifacts=fixture["cache_artifacts"],
    )

    assert "SAM3D preflight JSON attestation does not match its contents" in failures


def test_verify_only_recomputes_inference_status_without_polluting_artifact_status(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path)
    result = _run_passing(fixture, monkeypatch)
    result["inference_smoke"]["output"]["target_iou"] = 0.0
    result["inference_smoke_verification"] = {
        "status": "pass",
        "critical_issues": [],
    }
    fixture["output"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    verified = sam3d_preflight.verify_output(
        fixture["output"],
        fixture["sam3"],
        fixture["pipeline"],
        cache_artifacts=fixture["cache_artifacts"],
    )

    assert verified["status"] == "fail"
    assert verified["evidence_verification"] == {
        "status": "pass",
        "critical_issues": [],
    }
    assert verified["inference_smoke_verification"]["status"] == "fail"
    assert any(
        "target_iou" in issue
        for issue in verified["inference_smoke_verification"]["critical_issues"]
    )


def test_semantically_wrong_inference_mask_fails_closed() -> None:
    import numpy as np

    empty = np.zeros((512, 512), dtype=np.uint8)
    smoke = sam3d_preflight._build_inference_smoke(empty)

    assert smoke["status"] == "fail"
    failures = sam3d_preflight.verify_inference_smoke(smoke)
    assert any("binary mask" in failure for failure in failures)
    assert any("target IoU" in failure for failure in failures)


def test_inference_smoke_rejects_missing_cuda_bfloat16_support() -> None:
    _, target_mask = sam3d_preflight._canonical_inference_input()
    smoke = sam3d_preflight._build_inference_smoke(
        target_mask,
        runtime={
            "device_type": "cuda",
            "autocast_dtype": "bfloat16",
            "bf16_supported": False,
        },
    )

    assert smoke["status"] == "fail"
    assert any(
        "CUDA bfloat16" in failure
        for failure in sam3d_preflight.verify_inference_smoke(smoke)
    )


def test_canonical_input_hash_is_raw_rgb_not_encoded_image_bytes() -> None:
    import hashlib
    import io

    image, target_mask = sam3d_preflight._canonical_inference_input()
    smoke = sam3d_preflight._build_inference_smoke(target_mask)
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")

    assert smoke["input"]["raw_rgb_sha256"] == hashlib.sha256(
        image.tobytes()
    ).hexdigest()
    assert smoke["input"]["raw_rgb_sha256"] != hashlib.sha256(
        encoded.getvalue()
    ).hexdigest()


def test_inference_smoke_uses_production_generate_mask_contract() -> None:
    processor = object()
    calls: list[dict[str, object]] = []

    def fake_generate_mask(**kwargs):
        calls.append(kwargs)
        image = kwargs["image"]
        assert image.mode == "RGB"
        assert image.size == (512, 512)
        assert image.getpixel((0, 0)) == (255, 255, 255)
        assert image.getpixel((256, 256)) == (0, 0, 255)
        _, target_mask = sam3d_preflight._canonical_inference_input()
        return target_mask

    smoke = sam3d_preflight._run_inference_smoke(
        processor,
        generate_mask_function=fake_generate_mask,
    )

    assert len(calls) == 1
    assert calls[0]["sam3_processor"] is processor
    assert calls[0]["mode"] == "object_description"
    assert calls[0]["object_description"] == "circle"
    assert sam3d_preflight.verify_inference_smoke(smoke) == []


def test_missing_inference_smoke_prevents_pass(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path)
    runtime = _runtime_evidence()
    runtime.pop("inference_smoke")
    monkeypatch.setattr(
        sam3d_preflight,
        "_load_pipelines_with_cuda",
        lambda _sam3, _pipeline: runtime,
    )

    result = sam3d_preflight.run(
        fixture["sam3"],
        fixture["pipeline"],
        fixture["output"],
        cache_artifacts=fixture["cache_artifacts"],
    )

    assert result["status"] == "fail"
    assert result["inference_smoke_verification"]["status"] == "fail"
    assert "inference smoke is missing" in result["error"]


def test_gpu_memory_evidence_tampering_breaks_attestation(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path)
    result = _run_passing(fixture, monkeypatch)
    result["peak_allocated_bytes"] += 1

    failures = sam3d_preflight.verify_saved_result(
        result,
        fixture["sam3"],
        fixture["pipeline"],
        cache_artifacts=fixture["cache_artifacts"],
    )

    assert "SAM3D preflight JSON attestation does not match its contents" in failures


def test_memory_counter_reset_outcome_is_attested(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path)
    result = _run_passing(fixture, monkeypatch)
    result["memory_stats_reset"] = False
    result["memory_stats_reset_error"] = "forged"

    failures = sam3d_preflight.verify_saved_result(
        result,
        fixture["sam3"],
        fixture["pipeline"],
        cache_artifacts=fixture["cache_artifacts"],
    )

    assert "SAM3D preflight JSON attestation does not match its contents" in failures


def test_missing_evidence_cannot_be_upgraded_by_status_field(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path)
    result = _run_passing(fixture, monkeypatch)
    result.pop("evidence")
    result["status"] = "pass"

    failures = sam3d_preflight.verify_saved_result(
        result,
        fixture["sam3"],
        fixture["pipeline"],
        cache_artifacts=fixture["cache_artifacts"],
    )

    assert any("evidence manifest is missing" in failure for failure in failures)
    assert any("attestation" in failure for failure in failures)


def test_duplicate_artifact_role_is_rejected_even_when_receipt_is_reattested(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path)
    result = _run_passing(fixture, monkeypatch)
    duplicate = dict(result["evidence"]["artifacts"][0])
    result["evidence"]["artifacts"].append(duplicate)
    result["evidence"]["artifact_count"] += 1
    result["attestation"] = {
        "algorithm": "sha256",
        "sha256": sam3d_preflight._attestation_sha256(result),
    }

    failures = sam3d_preflight.verify_saved_result(
        result,
        fixture["sam3"],
        fixture["pipeline"],
        cache_artifacts=fixture["cache_artifacts"],
    )

    assert "SAM3D artifact evidence has duplicate or malformed roles" in failures
    assert "SAM3D artifact evidence count differs from the exact resolver" in failures
