from __future__ import annotations

import json
import os
import re

from pathlib import Path

import pytest

import scripts.pipeline_code_contract as contract


def _spec_text() -> str:
    clauses = "\n".join(clause for _identifier, clause in contract.REQUIRED_SPEC_CLAUSES)
    return f"# Test full-quality pipeline\n\n{clauses}\n"


def _fixture_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "tools" / "sage_scene_checker").mkdir(parents=True)
    (repo / "remote_jobs").mkdir(parents=True)
    (repo / "upstream-patches").mkdir(parents=True)
    (repo / "local_setup").mkdir(parents=True)
    for runtime_path in contract.SCENESMITH_RUNTIME_FILES:
        path = repo / runtime_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"# fixture SceneSmith runtime dependency: {runtime_path}\n",
            encoding="utf-8",
        )
    for regression_path in contract.CONTRACT_REGRESSION_FILES:
        path = repo / regression_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"# fixture contract regression: {regression_path}\n",
            encoding="utf-8",
        )

    (repo / contract.DEFAULT_SPEC).write_text(_spec_text(), encoding="utf-8")
    (repo / contract.CONTRACT_SCRIPT).write_text(
        "# fixture pipeline contract\nVALUE = 1\n", encoding="utf-8"
    )
    (repo / "scripts" / "worker.py").write_text(
        "def generate():\n    return 'scene'\n", encoding="utf-8"
    )
    (repo / contract.SAGE_CHECKER).write_text(
        "def check_scene():\n    return True\n", encoding="utf-8"
    )
    (repo / contract.DEFAULT_RUNNER).write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "\n".join(
            f"readonly {name}={value}"
            for name, value in contract.RUNNER_EXTERNAL_BINDING_VARIABLES.items()
        )
        + "\n",
        encoding="utf-8",
    )
    (repo / "remote_jobs" / "room_worker.sbatch").write_text(
        "#!/usr/bin/env bash\n#SBATCH --job-name=room\n", encoding="utf-8"
    )
    (repo / "upstream-patches" / "first.patch").write_text(
        "diff --git a/a b/a\n", encoding="utf-8"
    )
    (repo / contract.PATCH_ORDER).write_text(
        "# explicit order\nfirst.patch\n", encoding="utf-8"
    )
    (repo / contract.COMPUTE_NODE_ENV).write_text(
        "#!/usr/bin/env bash\n# fixture compute-node environment\n",
        encoding="utf-8",
    )
    output = tmp_path / "pipeline_code_contract.json"
    return repo, output


def _create(repo: Path, output: Path) -> dict:
    return contract.run(
        repo,
        contract.DEFAULT_SPEC,
        contract.DEFAULT_RUNNER,
        output,
    )


def _verify(repo: Path, output: Path) -> dict:
    return contract.run(
        repo,
        contract.DEFAULT_SPEC,
        contract.DEFAULT_RUNNER,
        output,
        verify_only=True,
    )


def test_create_and_verify_binds_canonical_complete_inventory(tmp_path: Path) -> None:
    repo, output = _fixture_repo(tmp_path)

    manifest = _create(repo, output)

    assert manifest["schema_version"] == contract.SCHEMA_VERSION
    assert manifest["attestation"]["schema_version"] == contract.ATTESTATION_SCHEMA_VERSION
    assert manifest["status"] == "pass"
    assert manifest["snapshot_reverified_after_inventory"] is True
    assert manifest["external_input_binding"] == contract.REQUIRED_EXTERNAL_INPUT_BINDING
    assert manifest["specification"]["read_before_generation_code"] is True
    assert manifest["compute_node_environment_path"] == contract.COMPUTE_NODE_ENV
    assert {
        clause["id"] for clause in manifest["specification"]["required_clauses"]
    } == {identifier for identifier, _text in contract.REQUIRED_SPEC_CLAUSES}
    assert manifest["inventory"] == {
        "scripts_python": [
            "scripts/pipeline_code_contract.py",
            "scripts/worker.py",
        ],
        "remote_jobs": [
            "remote_jobs/room_worker.sbatch",
            "remote_jobs/run_full_quality_school_sqz.sh",
        ],
        "scenesmith_runtime": list(contract.SCENESMITH_RUNTIME_FILES),
        "contract_regressions": list(contract.CONTRACT_REGRESSION_FILES),
        "upstream_patches": ["upstream-patches/first.patch"],
    }
    by_path = {entry["path"]: entry for entry in manifest["artifacts"]}
    assert by_path["scripts/pipeline_code_contract.py"]["roles"] == [
        "pipeline_contract",
        "scripts_python",
    ]
    assert by_path["remote_jobs/run_full_quality_school_sqz.sh"]["roles"] == [
        "production_runner",
        "remote_job",
    ]
    assert by_path[contract.COMPUTE_NODE_ENV]["roles"] == [
        "compute_node_environment",
    ]
    for runtime_path in contract.SCENESMITH_RUNTIME_FILES:
        assert by_path[runtime_path]["roles"] == [
            "scenesmith_runtime_dependency",
        ]
    for regression_path in contract.CONTRACT_REGRESSION_FILES:
        assert by_path[regression_path]["roles"] == ["contract_regression"]
    assert all(not Path(entry["path"]).is_absolute() for entry in manifest["artifacts"])
    assert _verify(repo, output) == manifest


def test_specification_is_fully_read_before_generation_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _output = _fixture_repo(tmp_path)
    labels: list[str] = []
    original = contract._read_regular_file

    def recording_read(
        repo_path: Path, path: Path, *, label: str
    ) -> tuple[bytes, dict]:
        labels.append(label)
        return original(repo_path, path, label=label)

    monkeypatch.setattr(contract, "_read_regular_file", recording_read)
    contract.build_manifest(repo)

    assert labels[0] == "pipeline specification"
    assert labels.index("pipeline specification") < labels.index("production runner")
    assert labels.index("pipeline specification") < labels.index("scripts Python file")


def test_identical_checkouts_have_identical_repo_relative_attestations(
    tmp_path: Path,
) -> None:
    first_repo, _first_output = _fixture_repo(tmp_path / "first")
    second_repo, _second_output = _fixture_repo(tmp_path / "second")

    first = contract.build_manifest(first_repo)
    second = contract.build_manifest(second_repo)

    assert first == second
    serialized = json.dumps(first, sort_keys=True)
    assert os.fspath(first_repo) not in serialized
    assert os.fspath(second_repo) not in serialized


@pytest.mark.parametrize(
    "clause_id",
    [
        "full_quality",
        "no_hssd",
        "sam3d",
        "compulsory_artiverse",
        "artiverse_publisher_external_gltf_visuals",
        "room_gate",
        "two_gpu_acceptance",
        "responses_visual_tool_transport",
        "sam3d_full_generation_preflight",
        "sam3d_request_memory_guard",
    ],
)
def test_initial_creation_rejects_absent_exact_clause(
    tmp_path: Path, clause_id: str
) -> None:
    repo, output = _fixture_repo(tmp_path)
    spec = repo / contract.DEFAULT_SPEC
    identifier_to_clause = dict(contract.REQUIRED_SPEC_CLAUSES)
    spec.write_text(
        _spec_text().replace(identifier_to_clause[clause_id], "removed clause"),
        encoding="utf-8",
    )

    with pytest.raises(
        contract.PipelineCodeContractError,
        match=f"missing exact required clauses: {clause_id}",
    ):
        _create(repo, output)


def test_verify_rejects_code_mutation(tmp_path: Path) -> None:
    repo, output = _fixture_repo(tmp_path)
    _create(repo, output)
    (repo / "scripts" / "worker.py").write_text(
        "def generate():\n    return 'changed'\n", encoding="utf-8"
    )

    with pytest.raises(
        contract.PipelineCodeContractError,
        match=r"mutated artifacts: scripts/worker\.py",
    ):
        _verify(repo, output)


def test_creation_rejects_missing_compute_node_environment(tmp_path: Path) -> None:
    repo, output = _fixture_repo(tmp_path)
    (repo / contract.COMPUTE_NODE_ENV).unlink()

    with pytest.raises(
        contract.PipelineCodeContractError,
        match="compute-node environment bootstrap is missing",
    ):
        _create(repo, output)


def test_verify_rejects_compute_node_environment_mutation(tmp_path: Path) -> None:
    repo, output = _fixture_repo(tmp_path)
    _create(repo, output)
    (repo / contract.COMPUTE_NODE_ENV).write_text(
        "#!/usr/bin/env bash\n# changed compute-node environment\n",
        encoding="utf-8",
    )

    with pytest.raises(
        contract.PipelineCodeContractError,
        match=r"mutated artifacts: local_setup/compute_node_env\.sh",
    ):
        _verify(repo, output)


def test_verify_rejects_compute_node_environment_removal(tmp_path: Path) -> None:
    repo, output = _fixture_repo(tmp_path)
    _create(repo, output)
    (repo / contract.COMPUTE_NODE_ENV).unlink()

    with pytest.raises(
        contract.PipelineCodeContractError,
        match="compute-node environment bootstrap is missing",
    ):
        _verify(repo, output)


@pytest.mark.parametrize("runtime_path", contract.SCENESMITH_RUNTIME_FILES)
def test_verify_rejects_scenesmith_runtime_mutation(
    tmp_path: Path, runtime_path: str
) -> None:
    repo, output = _fixture_repo(tmp_path)
    _create(repo, output)
    (repo / runtime_path).write_text(
        "# mutated live SceneSmith runtime dependency\n",
        encoding="utf-8",
    )

    with pytest.raises(
        contract.PipelineCodeContractError,
        match=rf"mutated artifacts: {re.escape(runtime_path)}",
    ):
        _verify(repo, output)


@pytest.mark.parametrize("runtime_path", contract.SCENESMITH_RUNTIME_FILES)
def test_verify_rejects_scenesmith_runtime_removal(
    tmp_path: Path, runtime_path: str
) -> None:
    repo, output = _fixture_repo(tmp_path)
    _create(repo, output)
    (repo / runtime_path).unlink()

    with pytest.raises(
        contract.PipelineCodeContractError,
        match=rf"SceneSmith runtime dependency {re.escape(runtime_path)} is missing",
    ):
        _verify(repo, output)


@pytest.mark.parametrize("regression_path", contract.CONTRACT_REGRESSION_FILES)
def test_verify_rejects_contract_regression_mutation(
    tmp_path: Path, regression_path: str
) -> None:
    repo, output = _fixture_repo(tmp_path)
    _create(repo, output)
    (repo / regression_path).write_text(
        "# mutated contract regression\n",
        encoding="utf-8",
    )

    with pytest.raises(
        contract.PipelineCodeContractError,
        match=rf"mutated artifacts: {re.escape(regression_path)}",
    ):
        _verify(repo, output)


@pytest.mark.parametrize("runtime_path", contract.SCENESMITH_RUNTIME_FILES)
def test_verify_rejects_scenesmith_runtime_link_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_path: str,
) -> None:
    repo, output = _fixture_repo(tmp_path)
    _create(repo, output)
    substituted = (repo / runtime_path).absolute()
    original_is_link_like = contract._is_link_like

    def substituted_runtime_is_link_like(path: Path) -> bool:
        if path.absolute() == substituted:
            return True
        return original_is_link_like(path)

    monkeypatch.setattr(contract, "_is_link_like", substituted_runtime_is_link_like)
    with pytest.raises(
        contract.PipelineCodeContractError,
        match=(
            rf"SceneSmith runtime dependency {re.escape(runtime_path)} path "
            r"contains a symlink/junction"
        ),
    ):
        _verify(repo, output)


def test_compute_node_environment_symlink_is_rejected_when_supported(
    tmp_path: Path,
) -> None:
    repo, output = _fixture_repo(tmp_path)
    helper = repo / contract.COMPUTE_NODE_ENV
    target = repo / "real_compute_node_env.sh"
    target.write_bytes(helper.read_bytes())
    helper.unlink()
    try:
        helper.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("creating symlinks is not supported for this test user")

    with pytest.raises(
        contract.PipelineCodeContractError,
        match="compute-node environment bootstrap path contains a symlink/junction",
    ):
        _create(repo, output)


def test_creation_rejects_runner_without_external_input_binding(tmp_path: Path) -> None:
    repo, output = _fixture_repo(tmp_path)
    (repo / contract.DEFAULT_RUNNER).write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8"
    )

    with pytest.raises(
        contract.PipelineCodeContractError,
        match="missing exact external input bindings",
    ):
        _create(repo, output)


def test_verify_rejects_patch_omitted_from_apply_order(tmp_path: Path) -> None:
    repo, output = _fixture_repo(tmp_path)
    _create(repo, output)
    (repo / contract.PATCH_ORDER).write_text(
        "# patch was improperly omitted\n", encoding="utf-8"
    )

    with pytest.raises(
        contract.PipelineCodeContractError,
        match="contains no patches|unlisted patch files",
    ):
        _verify(repo, output)


def test_verify_rejects_added_scripts_code(tmp_path: Path) -> None:
    repo, output = _fixture_repo(tmp_path)
    _create(repo, output)
    (repo / "scripts" / "new_generation_stage.py").write_text(
        "VALUE = 'unattested'\n", encoding="utf-8"
    )

    with pytest.raises(
        contract.PipelineCodeContractError,
        match=r"added artifacts: scripts/new_generation_stage\.py",
    ):
        _verify(repo, output)


def test_verify_rejects_removed_remote_job(tmp_path: Path) -> None:
    repo, output = _fixture_repo(tmp_path)
    _create(repo, output)
    (repo / "remote_jobs" / "room_worker.sbatch").unlink()

    with pytest.raises(
        contract.PipelineCodeContractError,
        match=r"removed artifacts: remote_jobs/room_worker\.sbatch",
    ):
        _verify(repo, output)


def test_verify_rejects_manifest_tampering(tmp_path: Path) -> None:
    repo, output = _fixture_repo(tmp_path)
    _create(repo, output)
    value = json.loads(output.read_text(encoding="utf-8"))
    value["artifact_count"] += 1
    output.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        contract.PipelineCodeContractError,
        match="attestation was tampered with",
    ):
        _verify(repo, output)


def test_scripts_symlink_is_rejected_when_supported(tmp_path: Path) -> None:
    repo, output = _fixture_repo(tmp_path)
    link = repo / "scripts" / "linked.py"
    try:
        link.symlink_to(repo / "scripts" / "worker.py")
    except (NotImplementedError, OSError):
        pytest.skip("creating symlinks is not supported for this test user")

    with pytest.raises(
        contract.PipelineCodeContractError,
        match="scripts tree contains a symlink/junction",
    ):
        _create(repo, output)


def test_patch_escape_and_duplicate_are_rejected(tmp_path: Path) -> None:
    repo, output = _fixture_repo(tmp_path)
    order = repo / contract.PATCH_ORDER
    order.write_text("first.patch\n../first.patch\n", encoding="utf-8")
    with pytest.raises(contract.PipelineCodeContractError, match="Unsafe patch entry"):
        _create(repo, output)

    order.write_text("first.patch\nfirst.patch\n", encoding="utf-8")
    with pytest.raises(contract.PipelineCodeContractError, match="duplicate patches"):
        _create(repo, output)


def test_cli_verify_only_does_not_rewrite_manifest(tmp_path: Path) -> None:
    repo, output = _fixture_repo(tmp_path)
    assert contract.main(
        ["--repo-dir", os.fspath(repo), "--output", os.fspath(output)]
    ) == 0
    before = output.read_bytes()
    assert contract.main(
        [
            "--repo-dir",
            os.fspath(repo),
            "--output",
            os.fspath(output),
            "--verify-only",
        ]
    ) == 0
    assert output.read_bytes() == before


def test_sqz_continuation_has_fail_closed_benchmark_only_mode() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    continuation = (
        repo_root / "remote_jobs" / "continue_full_quality_school_sqz.sh"
    ).read_text(encoding="utf-8")

    assert "SCENESMITH_CONTINUATION_MODE:-full" in continuation
    assert "full|benchmark_classroom_01" in continuation
    assert "waiting_for_existing_artiverse" in continuation
    assert "^complete exit=0 " in continuation
    assert "--query-compute-apps=pid" in continuation
    assert "SCENESMITH_EXECUTION_MODE=benchmark_classroom_01" in continuation
    assert "^benchmark_complete exit=0 " in continuation
    wait = continuation.index("while artiverse_worker_running")
    complete_check = continuation.index("if ! artiverse_complete", wait)
    gpu_wait = continuation.index("phase=waiting_for_idle_gpu", complete_check)
    benchmark = continuation.index("phase=benchmark_classroom_01", gpu_wait)
    assert wait < complete_check < gpu_wait < benchmark


def test_production_runner_requires_full_sam3d_generation_before_paid_probe() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    runner = (repo_root / contract.DEFAULT_RUNNER).read_text(encoding="utf-8")

    preflight_start = runner.index('echo "[preflight] $(date -Is)')
    generation = runner.index(
        "python scripts/preflight_sam3d_generation.py", preflight_start
    )
    artiverse_visual = runner.index(
        "python scripts/preflight_artiverse_visual_resources.py", preflight_start
    )
    paid_probe = runner.index("probe_openai", generation)
    room_worker = runner.index("python scripts/run_single_room_worker.py", paid_probe)
    assert artiverse_visual < paid_probe < room_worker
    assert generation < paid_probe < room_worker
    assert "SAM3D_GENERATION_RECEIPT" in runner
    assert "--verify-only" in runner[generation:paid_probe]
    assert runner.count("python scripts/preflight_sam3d_offline.py") >= 3
    assert runner.count("python scripts/preflight_sam3d_generation.py") >= 4
    assert runner.count("python scripts/preflight_artiverse_visual_resources.py") >= 4
    assert "tests/unit/test_preflight_sam3d_generation.py" in contract.CONTRACT_REGRESSION_FILES
    assert "tests/unit/test_artiverse_visual_resources.py" in contract.CONTRACT_REGRESSION_FILES
    assert "tests/unit/test_artiverse_visual_normalization.py" in contract.CONTRACT_REGRESSION_FILES
    assert (
        "tests/unit/test_artiverse_retrieval_integration.py"
        in contract.CONTRACT_REGRESSION_FILES
    )
    assert "tests/unit/test_manipuland_agent.py" in contract.CONTRACT_REGRESSION_FILES
    assert "tests/unit/test_asset_registry_autosave.py" in contract.CONTRACT_REGRESSION_FILES
    assert "tests/unit/test_objathor_manipuland_routing.py" in contract.CONTRACT_REGRESSION_FILES
    assert (
        "tests/unit/test_artiverse_support_surface_extraction.py"
        in contract.CONTRACT_REGRESSION_FILES
    )
    assert (
        "tests/unit/test_manipuland_furniture_checkpoint.py"
        in contract.CONTRACT_REGRESSION_FILES
    )
    assert (
        "tests/unit/test_manipuland_furniture_scope.py"
        in contract.CONTRACT_REGRESSION_FILES
    )
    assert (
        "tests/unit/test_asset_registry_resume_contract.py"
        in contract.CONTRACT_REGRESSION_FILES
    )
    assert (
        "tests/unit/test_physical_feasibility_thread_timeout.py"
        in contract.CONTRACT_REGRESSION_FILES
    )
    assert (
        "tests/unit/test_inprocess_server_shutdown.py"
        in contract.CONTRACT_REGRESSION_FILES
    )
    assert "scenesmith/agent_utils/asset_router/router.py" in contract.SCENESMITH_RUNTIME_FILES
    assert "scenesmith/agent_utils/asset_manager.py" in contract.SCENESMITH_RUNTIME_FILES
    assert "scenesmith/agent_utils/asset_registry.py" in contract.SCENESMITH_RUNTIME_FILES
    assert (
        "scenesmith/agent_utils/physical_feasibility.py"
        in contract.SCENESMITH_RUNTIME_FILES
    )
    assert "scenesmith/agent_utils/physics_tools.py" in contract.SCENESMITH_RUNTIME_FILES
    assert (
        "scenesmith/agent_utils/support_surface_extraction.py"
        in contract.SCENESMITH_RUNTIME_FILES
    )
    assert "scenesmith/experiments/base_experiment.py" in contract.SCENESMITH_RUNTIME_FILES
    assert (
        "scenesmith/experiments/indoor_scene_generation.py"
        in contract.SCENESMITH_RUNTIME_FILES
    )
    assert (
        "scenesmith/manipuland_agents/base_manipuland_agent.py"
        in contract.SCENESMITH_RUNTIME_FILES
    )
    assert (
        "scenesmith/manipuland_agents/stateful_manipuland_agent.py"
        in contract.SCENESMITH_RUNTIME_FILES
    )
    assert (
        "scenesmith/manipuland_agents/tools/manipuland_tools.py"
        in contract.SCENESMITH_RUNTIME_FILES
    )
    assert (
        "scenesmith/manipuland_agents/tools/vision_tools.py"
        in contract.SCENESMITH_RUNTIME_FILES
    )
    for manager_path in (
        "scenesmith/agent_utils/geometry_generation_server/server_manager.py",
        "scenesmith/agent_utils/objaverse_retrieval_server/server_manager.py",
        "scenesmith/agent_utils/articulated_retrieval_server/server_manager.py",
        "scenesmith/agent_utils/materials_retrieval_server/server_manager.py",
        "scenesmith/agent_utils/hssd_retrieval_server/server_manager.py",
    ):
        assert manager_path in contract.SCENESMITH_RUNTIME_FILES
