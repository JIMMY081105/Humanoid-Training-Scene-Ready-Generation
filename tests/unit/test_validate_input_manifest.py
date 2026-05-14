from __future__ import annotations

import csv
import hashlib
import json

from scripts.validate_input_manifest import SCHOOL_REFERENCE_RUN_NAME, validate


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _valid_inputs(tmp_path):
    original = "original prompt\n"
    appendix = "contract appendix\n"
    effective = original + "\n" + appendix
    reference = b"reference-image"
    (tmp_path / "prompt_original.txt").write_text(original, encoding="utf-8")
    (tmp_path / "scene_contract_appendix.txt").write_text(appendix, encoding="utf-8")
    (tmp_path / "prompt.txt").write_text(effective, encoding="utf-8")
    (tmp_path / "reference.png").write_bytes(reference)
    with (tmp_path / "prompt.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene_index", "prompt"])
        writer.writerow([0, effective])
    manifest = {
        "original_prompt_sha256": _hash_text(original),
        "appendix_sha256": _hash_text(appendix),
        "effective_prompt_sha256": _hash_text(effective),
        "effective_prompt_chars": len(effective),
        "reference_image_sha256": hashlib.sha256(reference).hexdigest(),
        "scene_index": 0,
        "run_name": "test",
        "expected_room_ids": [f"room_{index}" for index in range(11)],
        "pipeline_contract": {
            "id": "scenesmith_full_quality_v1",
            "final_assembly_policy": "external_artiverse_gated",
            "required_articulated_source": "artiverse",
        },
    }
    (tmp_path / "input_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return tmp_path


def test_manifest_and_reconstruction_pass(tmp_path) -> None:
    result = validate(_valid_inputs(tmp_path))
    assert result["status"] == "pass"
    assert result["critical_issues"] == []


def test_changed_prompt_fails_even_if_other_files_exist(tmp_path) -> None:
    input_dir = _valid_inputs(tmp_path)
    (input_dir / "prompt.txt").write_text("changed", encoding="utf-8")
    result = validate(input_dir)
    assert result["status"] == "fail"
    assert any("effective_prompt_sha256" in issue for issue in result["critical_issues"])
    assert any("not exactly" in issue for issue in result["critical_issues"])


def test_changed_reference_fails_hash_check(tmp_path) -> None:
    input_dir = _valid_inputs(tmp_path)
    (input_dir / "reference.png").write_bytes(b"different")
    result = validate(input_dir)
    assert result["status"] == "fail"
    assert any("reference_image_sha256" in issue for issue in result["critical_issues"])


def test_missing_external_artiverse_contract_fails(tmp_path) -> None:
    input_dir = _valid_inputs(tmp_path)
    manifest_path = input_dir / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("pipeline_contract")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate(input_dir)

    assert result["status"] == "fail"
    assert any("pipeline_contract" in issue for issue in result["critical_issues"])


def test_school_profile_cannot_replace_manifest_and_all_staged_inputs(tmp_path) -> None:
    input_dir = _valid_inputs(tmp_path)
    manifest_path = input_dir / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_name"] = SCHOOL_REFERENCE_RUN_NAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate(input_dir)

    assert result["status"] == "fail"
    assert any(
        "External school-reference binding mismatch" in issue
        for issue in result["critical_issues"]
    )
