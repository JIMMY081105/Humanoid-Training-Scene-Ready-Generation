from __future__ import annotations

import json

from scripts.preflight_sam3d_offline import OFFLINE_VARIABLES, run, verify_output


def test_missing_checkpoint_writes_fail_closed_result(tmp_path) -> None:
    output = tmp_path / "sam3d_offline.json"
    result = run(
        tmp_path / "missing-sam3.pt",
        tmp_path / "missing-pipeline.yaml",
        output,
    )

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "fail"
    assert persisted["status"] == "fail"
    assert persisted["error_type"] == "FileNotFoundError"
    assert persisted["model_loaded"] is False
    assert persisted["pipeline_loaded"] is False
    assert persisted["inference_smoke"] is None
    assert persisted["schema_version"] == 3
    assert persisted["offline"] is True
    assert all(persisted["offline_environment"][name] == "1" for name in OFFLINE_VARIABLES)


def test_verify_only_rewrites_non_object_json_as_fail_closed(tmp_path) -> None:
    output = tmp_path / "sam3d_offline.json"
    output.write_text("[]\n", encoding="utf-8")

    result = verify_output(
        output,
        tmp_path / "missing-sam3.pt",
        tmp_path / "missing-pipeline.yaml",
    )

    assert result["status"] == "fail"
    assert result["error_type"] == "TypeError"
    assert result["evidence_verification"]["status"] == "fail"
    assert result["inference_smoke_verification"]["status"] == "fail"
