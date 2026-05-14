import hashlib
import json
import shutil

from pathlib import Path

import pytest

from scripts import two_gpu_drake_acceptance_contract as contract


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ATTEMPT_ID = "sqz-attempt-001"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pipeline_contract_payload() -> dict:
    artifacts = []
    for relative_value in contract.REQUIRED_RUNTIME_ARTIFACTS:
        path = REPO_ROOT.joinpath(*relative_value.split("/"))
        artifacts.append(
            {
                "path": relative_value,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "roles": ["test_runtime"],
            }
        )
    payload = {
        "schema_version": 1,
        "status": "pass",
        "hash_algorithm": "sha256",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    payload["attestation"] = {
        "schema_version": 1,
        "algorithm": "sha256",
        "sha256": _canonical_sha256(payload),
    }
    return payload


def _write_manifest(package: Path, manifest: Path) -> str:
    records = []
    for path in sorted(value for value in package.rglob("*") if value.is_file()):
        records.append(f"{_sha256(path)}  {path.relative_to(package).as_posix()}\n")
    manifest.write_text("".join(records), encoding="utf-8")
    return _sha256(manifest)


def _write_sqz_record(package: Path, *, run_attempt_id: str) -> Path:
    dmd = package / "combined_house" / "house.dmd.yaml"
    label_paths: dict[str, Path] = {
        "house_dmd": dmd,
        "asset_sdf": package / "room_01" / "asset.sdf",
    }
    for label in sorted(contract.CRITICAL_SQZ_EVIDENCE_LABELS - {"house_dmd"}):
        path = package / "quality_gates" / "contract_fixture" / f"{label}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if label == contract.PIPELINE_CODE_CONTRACT_LABEL:
            path.write_text(
                json.dumps(_pipeline_contract_payload(), sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text(
                f'{{"label":"{label}","status":"pass"}}\n', encoding="utf-8"
            )
        label_paths[label] = path
    evidence = []
    for label, path in sorted(label_paths.items()):
        evidence.append(
            {
                "label": label,
                "scene_relative_path": path.relative_to(package).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "category": "test_evidence",
            }
        )
    required_labels = sorted(contract.CRITICAL_SQZ_EVIDENCE_LABELS)
    by_label = {item["label"]: item for item in evidence}
    package_items = sorted(
        (
            {
                "scene_relative_path": item["scene_relative_path"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in evidence
        ),
        key=lambda item: item["scene_relative_path"],
    )
    record = {
        "schema_id": contract.SQZ_RECORD_SCHEMA_ID,
        "schema_version": 1,
        "status": contract.SQZ_AWAITING_STATUS,
        "run_attempt_id": run_attempt_id,
        "created_at_utc": "2026-07-10T00:00:00Z",
        "hash_algorithm": "sha256",
        "path_encoding": "scene-relative-posix",
        "required_room_ids": list(contract.REQUIRED_ROOM_IDS),
        "required_labels": required_labels,
        "scene_identity": {
            "scene_relative_to_run": "scene_000",
            "house_layout_sha256": by_label["house_layout"]["sha256"],
            "house_state_sha256": by_label["house_state"]["sha256"],
            "house_dmd_sha256": by_label["house_dmd"]["sha256"],
        },
        "package_inventory": {
            "scope": "all_regular_scene_files_except_this_record",
            "record_scene_relative_path": contract.SQZ_RECORD_RELATIVE_PATH,
            "file_count": len(package_items),
            "content_sha256": _canonical_sha256(package_items),
        },
        "external_package_manifest": {
            "required": True,
            "status": "pending_creation_after_this_record",
            "must_be_outside_scene": True,
            "must_cover_exact_scene_file_set_including_this_record": True,
            "record_scene_relative_path": contract.SQZ_RECORD_RELATIVE_PATH,
        },
        "evidence_count": len(evidence),
        "evidence": evidence,
        "evidence_attestation": {
            "algorithm": "sha256",
            "sha256": _canonical_sha256(evidence),
        },
        "label_specific_validation": {label: "pass" for label in required_labels},
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
    record["self_attestation"] = {
        "algorithm": "sha256",
        "scope": "canonical_json_of_all_fields_except_self_attestation",
        "sha256": _canonical_sha256(record),
    }
    output = package / contract.SQZ_RECORD_RELATIVE_PATH
    output.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _package(
    tmp_path: Path, *, run_attempt_id: str = RUN_ATTEMPT_ID
) -> tuple[Path, Path, str]:
    package = tmp_path / "scene_000"
    dmd = package / "combined_house" / "house.dmd.yaml"
    sdf = package / "room_01" / "asset.sdf"
    dmd.parent.mkdir(parents=True)
    sdf.parent.mkdir(parents=True)
    dmd.write_text("directives: []\n", encoding="utf-8")
    sdf.write_text('<sdf version="1.9"/>\n', encoding="utf-8")
    _write_sqz_record(package, run_attempt_id=run_attempt_id)
    manifest = tmp_path / "scene_000.sha256"
    return package, manifest, _write_manifest(package, manifest)


def _write_evidence(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_completion(
    path: Path,
    package_validation_path: Path,
) -> tuple[dict, str]:
    completion = contract.create_sqz_completion(
        package_validation_path,
        expected_run_attempt_id=RUN_ATTEMPT_ID,
        output_path=path,
    )
    _write_evidence(path, completion)
    return completion, _sha256(path)


def _write_runtime_validation(
    path: Path, package_preflight_path: Path
) -> dict:
    payload = contract.verify_runtime(
        package_preflight_path,
        REPO_ROOT,
        expected_run_attempt_id=RUN_ATTEMPT_ID,
    )
    _write_evidence(path, payload)
    return payload


def _passing_drake_report(package: Path) -> dict:
    return {
        "status": "pass",
        "dmd_path": str(package / "combined_house" / "house.dmd.yaml"),
        "package_root": str(package),
        "acceptance_requirements": {
            "required_visible_gpu_count": 2,
            "required_gpu_execution_count": 2,
            "max_collision_elements_per_sdf": 32,
            "minimum_model_directive_count": 12,
            "expected_room_count": 11,
        },
        "visible_gpu_count": 2,
        "visible_gpus": ["gpu0", "gpu1"],
        "two_gpu_acceptance_environment": True,
        "gpu_inventory": {
            "count": 2,
            "names": ["gpu0", "gpu1"],
            "required_execution_count": 2,
            "all_required_devices_exercised": True,
            "execution_exercises": [
                {
                    "index": 0,
                    "name": "gpu0",
                    "status": "pass",
                    "allocated_bytes": 1024,
                    "synchronized": True,
                },
                {
                    "index": 1,
                    "name": "gpu1",
                    "status": "pass",
                    "allocated_bytes": 1024,
                    "synchronized": True,
                },
            ],
        },
        "checks": {
            "dmd_file_exists": True,
            "package_root_exists": True,
            "required_gpus_available": True,
            "required_gpus_exercised": True,
            "all_sdf_files_parsed": True,
            "collision_cap_satisfied": True,
            "sdf_assets_present": True,
            "dmd_inventory_nonempty": True,
            "house_state_inventory_nonempty": True,
            "expected_room_count_satisfied": True,
            "drake_model_count_matches_directives": True,
            "drake_load_succeeded": True,
        },
        "house_state_inventory": {
            "status": "pass",
            "room_count": 11,
            "object_count": 100,
        },
        "drake_load": {
            "status": "pass",
            "added_model_count": 111,
            "num_bodies": 112,
        },
    }


def test_verify_package_binds_exact_manifest_and_file_set(tmp_path: Path) -> None:
    package, manifest, expected = _package(tmp_path)

    result = contract.verify_package(package, manifest, expected, RUN_ATTEMPT_ID)

    assert result["status"] == "pass"
    assert result["manifest_sha256"] == expected
    assert result["file_count"] == len(
        [path for path in package.rglob("*") if path.is_file()]
    )
    assert result["run_attempt_id"] == RUN_ATTEMPT_ID
    assert result["sqz_acceptance_record"]["relative_path"] == (
        contract.SQZ_RECORD_RELATIVE_PATH
    )
    assert len(result["package_content_sha256"]) == 64
    assert contract.SQZ_RECORD_RELATIVE_PATH in {
        item["path"] for item in result["files"]
    }


def test_verify_runtime_binds_live_files_to_packaged_pipeline_contract(
    tmp_path: Path,
) -> None:
    package, manifest, expected = _package(tmp_path)
    preflight = contract.verify_package(package, manifest, expected, RUN_ATTEMPT_ID)
    preflight_path = tmp_path / "preflight.json"
    _write_evidence(preflight_path, preflight)

    result = contract.verify_runtime(
        preflight_path,
        REPO_ROOT,
        expected_run_attempt_id=RUN_ATTEMPT_ID,
    )

    assert result["status"] == "pass"
    assert [item["path"] for item in result["runtime_artifacts"]] == list(
        contract.REQUIRED_RUNTIME_ARTIFACTS
    )
    assert len(result["runtime_content_sha256"]) == 64


def test_verify_runtime_rejects_live_code_mutation(tmp_path: Path) -> None:
    package, manifest, expected = _package(tmp_path)
    preflight = contract.verify_package(package, manifest, expected, RUN_ATTEMPT_ID)
    preflight_path = tmp_path / "preflight.json"
    _write_evidence(preflight_path, preflight)
    runtime_repo = tmp_path / "runtime_repo"
    for relative_value in contract.REQUIRED_RUNTIME_ARTIFACTS:
        source = REPO_ROOT.joinpath(*relative_value.split("/"))
        target = runtime_repo.joinpath(*relative_value.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    target = runtime_repo.joinpath(
        *contract.REQUIRED_RUNTIME_ARTIFACTS[0].split("/")
    )
    target.write_text(target.read_text(encoding="utf-8") + "\n# mutated\n")

    with pytest.raises(contract.AcceptanceContractError, match="runtime differs"):
        contract.verify_runtime(
            preflight_path,
            runtime_repo,
            expected_run_attempt_id=RUN_ATTEMPT_ID,
        )


def test_create_manifest_is_external_deterministic_and_self_verified(
    tmp_path: Path,
) -> None:
    package, _old_manifest, _old_expected = _package(tmp_path)
    manifest = tmp_path / "generated.sha256"

    first = contract.create_package_manifest(package, manifest, RUN_ATTEMPT_ID)
    first_bytes = manifest.read_bytes()
    second = contract.create_package_manifest(package, manifest, RUN_ATTEMPT_ID)

    assert first["status"] == "pass"
    assert second["status"] == "pass"
    assert first_bytes == manifest.read_bytes()
    assert first["manifest_sha256"] == _sha256(manifest)
    assert first["package_content_sha256"] == second["package_content_sha256"]


@pytest.mark.parametrize("mutation", ["changed", "missing", "extra"])
def test_verify_package_rejects_any_file_set_or_content_drift(
    tmp_path: Path, mutation: str
) -> None:
    package, manifest, expected = _package(tmp_path)
    target = package / "room_01" / "asset.sdf"
    if mutation == "changed":
        target.write_text("mutated", encoding="utf-8")
    elif mutation == "missing":
        target.unlink()
    else:
        (package / "unexpected.txt").write_text("extra", encoding="utf-8")

    with pytest.raises(contract.AcceptanceContractError):
        contract.verify_package(package, manifest, expected, RUN_ATTEMPT_ID)


def test_verify_package_rejects_wrong_manifest_digest(tmp_path: Path) -> None:
    package, manifest, _expected = _package(tmp_path)

    with pytest.raises(contract.AcceptanceContractError, match="manifest digest mismatch"):
        contract.verify_package(package, manifest, "0" * 64, RUN_ATTEMPT_ID)


def test_verify_package_rejects_wrong_run_attempt_id(tmp_path: Path) -> None:
    package, manifest, expected = _package(tmp_path)

    with pytest.raises(contract.AcceptanceContractError, match="run_attempt_id"):
        contract.verify_package(package, manifest, expected, "different-attempt")


def test_verify_package_semantically_rejects_tampered_sqz_record(
    tmp_path: Path,
) -> None:
    package, manifest, _expected = _package(tmp_path)
    record_path = package / contract.SQZ_RECORD_RELATIVE_PATH
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["two_gpu_acceptance"]["status"] = "pass"
    record["self_attestation"]["sha256"] = _canonical_sha256(
        {key: value for key, value in record.items() if key != "self_attestation"}
    )
    record_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    expected = _write_manifest(package, manifest)

    with pytest.raises(contract.AcceptanceContractError, match="improperly claims"):
        contract.verify_package(package, manifest, expected, RUN_ATTEMPT_ID)


@pytest.mark.parametrize("unsafe_path", ["../escape.sdf", "/absolute/file.sdf"])
def test_verify_package_rejects_unsafe_manifest_paths(
    tmp_path: Path, unsafe_path: str
) -> None:
    package, manifest, _expected = _package(tmp_path)
    manifest.write_text(f"{'0' * 64}  {unsafe_path}\n", encoding="utf-8")

    with pytest.raises(contract.AcceptanceContractError, match="unsafe path"):
        contract.verify_package(
            package, manifest, _sha256(manifest), RUN_ATTEMPT_ID
        )


def test_verify_package_rejects_symlinks(tmp_path: Path) -> None:
    package, manifest, _expected = _package(tmp_path)
    target = package / "room_01" / "asset.sdf"
    link = package / "linked.sdf"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable for this test user")
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + f"{_sha256(target)}  linked.sdf\n",
        encoding="utf-8",
    )

    with pytest.raises(contract.AcceptanceContractError, match="symlink"):
        contract.verify_package(
            package, manifest, _sha256(manifest), RUN_ATTEMPT_ID
        )


def test_sqz_completion_binds_manifest_record_and_attempt(tmp_path: Path) -> None:
    package, manifest, expected = _package(tmp_path)
    validation = contract.verify_package(
        package, manifest, expected, RUN_ATTEMPT_ID
    )
    validation_path = tmp_path / "manifest_validation.json"
    _write_evidence(validation_path, validation)
    output = tmp_path / "pipeline_completion.json"

    completion, completion_sha256 = _write_completion(output, validation_path)

    assert completion["status"] == contract.SQZ_AWAITING_STATUS
    assert completion["run_attempt_id"] == RUN_ATTEMPT_ID
    assert completion["package"]["manifest_sha256"] == expected
    assert completion["sqz_acceptance_record"]["sha256"] == _sha256(
        package / contract.SQZ_RECORD_RELATIVE_PATH
    )
    assert len(completion["completion_payload_sha256"]) == 64
    assert completion_sha256 == _sha256(output)

    with pytest.raises(contract.AcceptanceContractError, match="outside"):
        contract.create_sqz_completion(
            validation_path,
            expected_run_attempt_id=RUN_ATTEMPT_ID,
            output_path=package / "pipeline_completion.json",
        )


def test_finalize_receipt_binds_package_and_drake_evidence(tmp_path: Path) -> None:
    package, manifest, expected = _package(tmp_path)
    preflight = contract.verify_package(package, manifest, expected, RUN_ATTEMPT_ID)
    postflight = contract.verify_package(package, manifest, expected, RUN_ATTEMPT_ID)
    preflight_path = tmp_path / "preflight.json"
    postflight_path = tmp_path / "postflight.json"
    report_path = tmp_path / "drake.json"
    _write_evidence(preflight_path, preflight)
    _write_evidence(postflight_path, postflight)
    _write_evidence(report_path, _passing_drake_report(package))
    runtime_path = tmp_path / "runtime_validation.json"
    runtime = _write_runtime_validation(runtime_path, preflight_path)
    completion_path = tmp_path / "pipeline_completion.json"
    completion, completion_sha256 = _write_completion(
        completion_path, preflight_path
    )

    receipt = contract.finalize_receipt(
        preflight_path,
        postflight_path,
        report_path,
        runtime_path,
        completion_path,
        expected_run_attempt_id=RUN_ATTEMPT_ID,
        expected_completion_sha256=completion_sha256,
        slurm_job_id="12345",
        node_name="gpu-node-1",
    )

    assert receipt["status"] == "pass"
    assert receipt["schema_id"] == contract.TWO_GPU_RECEIPT_SCHEMA_ID
    assert receipt["acceptance_type"] == "two_gpu_drake"
    assert receipt["run_attempt_id"] == RUN_ATTEMPT_ID
    assert receipt["package"]["manifest_sha256"] == expected
    assert receipt["package"]["dmd_sha256"] == _sha256(
        package / "combined_house" / "house.dmd.yaml"
    )
    assert receipt["drake"]["visible_gpu_count"] == 2
    assert receipt["schema_version"] == 2
    assert receipt["drake"]["gpu_execution_exercises"] == (
        _passing_drake_report(package)["gpu_inventory"]["execution_exercises"]
    )
    assert receipt["runtime"]["runtime_content_sha256"] == runtime[
        "runtime_content_sha256"
    ]
    assert receipt["sqz_acceptance_record"]["run_attempt_id"] == RUN_ATTEMPT_ID
    assert receipt["sqz_pipeline_completion"]["sha256"] == completion_sha256
    assert receipt["sqz_pipeline_completion"]["completion_payload_sha256"] == (
        completion["completion_payload_sha256"]
    )
    assert len(receipt["receipt_payload_sha256"]) == 64


def test_finalize_rejects_package_change_or_nonpassing_report(tmp_path: Path) -> None:
    package, manifest, expected = _package(tmp_path)
    preflight = contract.verify_package(package, manifest, expected, RUN_ATTEMPT_ID)
    postflight = dict(preflight)
    postflight["package_content_sha256"] = "f" * 64
    preflight_path = tmp_path / "preflight.json"
    postflight_path = tmp_path / "postflight.json"
    report_path = tmp_path / "drake.json"
    _write_evidence(preflight_path, preflight)
    _write_evidence(postflight_path, postflight)
    _write_evidence(report_path, _passing_drake_report(package))
    runtime_path = tmp_path / "runtime_validation.json"
    _write_runtime_validation(runtime_path, preflight_path)
    completion_path = tmp_path / "pipeline_completion.json"
    _completion, completion_sha256 = _write_completion(
        completion_path, preflight_path
    )

    with pytest.raises(contract.AcceptanceContractError, match="inconsistent"):
        contract.finalize_receipt(
            preflight_path,
            postflight_path,
            report_path,
            runtime_path,
            completion_path,
            expected_run_attempt_id=RUN_ATTEMPT_ID,
            expected_completion_sha256=completion_sha256,
            slurm_job_id="12345",
            node_name="gpu-node-1",
        )

    _write_evidence(postflight_path, preflight)
    report = _passing_drake_report(package)
    report["status"] = "fail"
    _write_evidence(report_path, report)
    with pytest.raises(contract.AcceptanceContractError, match="not passing"):
        contract.finalize_receipt(
            preflight_path,
            postflight_path,
            report_path,
            runtime_path,
            completion_path,
            expected_run_attempt_id=RUN_ATTEMPT_ID,
            expected_completion_sha256=completion_sha256,
            slurm_job_id="12345",
            node_name="gpu-node-1",
        )


def test_finalize_does_not_trust_pass_status_without_two_gpu_contract(
    tmp_path: Path,
) -> None:
    package, manifest, expected = _package(tmp_path)
    evidence = contract.verify_package(package, manifest, expected, RUN_ATTEMPT_ID)
    preflight_path = tmp_path / "preflight.json"
    postflight_path = tmp_path / "postflight.json"
    report_path = tmp_path / "drake.json"
    _write_evidence(preflight_path, evidence)
    _write_evidence(postflight_path, evidence)
    report = _passing_drake_report(package)
    report["visible_gpu_count"] = 1
    report["visible_gpus"] = ["gpu0"]
    _write_evidence(report_path, report)
    runtime_path = tmp_path / "runtime_validation.json"
    _write_runtime_validation(runtime_path, preflight_path)
    completion_path = tmp_path / "pipeline_completion.json"
    _completion, completion_sha256 = _write_completion(
        completion_path, preflight_path
    )

    with pytest.raises(contract.AcceptanceContractError, match="fewer than two"):
        contract.finalize_receipt(
            preflight_path,
            postflight_path,
            report_path,
            runtime_path,
            completion_path,
            expected_run_attempt_id=RUN_ATTEMPT_ID,
            expected_completion_sha256=completion_sha256,
            slurm_job_id="12345",
            node_name="gpu-node-1",
        )


def test_finalize_requires_concrete_slurm_identity(tmp_path: Path) -> None:
    package, manifest, expected = _package(tmp_path)
    evidence = contract.verify_package(package, manifest, expected, RUN_ATTEMPT_ID)
    preflight_path = tmp_path / "preflight.json"
    postflight_path = tmp_path / "postflight.json"
    report_path = tmp_path / "drake.json"
    completion_path = tmp_path / "pipeline_completion.json"
    _write_evidence(preflight_path, evidence)
    _write_evidence(postflight_path, evidence)
    _write_evidence(report_path, _passing_drake_report(package))
    runtime_path = tmp_path / "runtime_validation.json"
    _write_runtime_validation(runtime_path, preflight_path)
    _completion, completion_sha256 = _write_completion(
        completion_path, preflight_path
    )

    with pytest.raises(contract.AcceptanceContractError, match="SLURM job id"):
        contract.finalize_receipt(
            preflight_path,
            postflight_path,
            report_path,
            runtime_path,
            completion_path,
            expected_run_attempt_id=RUN_ATTEMPT_ID,
            expected_completion_sha256=completion_sha256,
            slurm_job_id="unknown",
            node_name="gpu-node-1",
        )

def test_paracloud_template_is_strict_two_gpu_acceptance() -> None:
    template = (
        REPO_ROOT / "remote_jobs" / "TEMPLATE_2gpu_drake_acceptance.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --nodes=1" in template
    assert "#SBATCH --gres=gpu:2" in template
    assert "EXPECTED_MANIFEST_SHA256:?" in template
    assert "EXPECTED_RUN_ATTEMPT_ID:?" in template
    assert "SQZ_PIPELINE_COMPLETION:?" in template
    assert "EXPECTED_SQZ_COMPLETION_SHA256:?" in template
    assert template.count("verify-package") == 2
    assert template.count("verify-runtime") == 1
    assert template.index("verify-runtime") < template.index(
        "source local_setup/compute_node_env.sh"
    )
    assert template.count('--expected-run-attempt-id "$EXPECTED_RUN_ATTEMPT_ID"') == 4
    assert "scripts/validate_drake_scene.py" in template
    assert "--require-gpus 2" in template
    assert "--max-collision-elements 32" in template
    assert "--minimum-models 12" in template
    assert "--expected-rooms 11" in template
    assert "two_gpu_drake_acceptance_receipt.json" in template
    assert '--runtime-validation "$RUNTIME_VALIDATION"' in template
    assert '--sqz-pipeline-completion "$SQZ_PIPELINE_COMPLETION"' in template
    assert (
        '--expected-sqz-completion-sha256 '
        '"$EXPECTED_SQZ_COMPLETION_SHA256"'
    ) in template
    assert "DMD_RELATIVE_PATH resolves outside" in template
