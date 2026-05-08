#!/usr/bin/env python3
"""Bind the full-quality pipeline specification to its executable code.

The contract is deliberately repository-relative: the same verified checkout can
be copied to another machine without changing the attestation.  Initial creation
reads and hashes the specification *before* discovering or reading generation
code.  Verification re-discovers the complete in-scope inventory, so adding,
removing, or changing executable code invalidates a saved proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat

from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 2
ATTESTATION_SCHEMA_VERSION = 1
CONTRACT_NAME = "scenesmith-full-quality-pipeline-code"
HASH_ALGORITHM = "sha256"
HASH_CHUNK_SIZE = 1024 * 1024
DEFAULT_SPEC = "CODEX_SCENESMITH_FULL_QUALITY_PIPELINE.md"
DEFAULT_RUNNER = "remote_jobs/run_full_quality_school_sqz.sh"
CONTRACT_SCRIPT = "scripts/pipeline_code_contract.py"
SAGE_CHECKER = "tools/sage_scene_checker/check_scenesmith_output.py"
PATCH_ORDER = "upstream-patches/APPLY_ORDER.txt"
COMPUTE_NODE_ENV = "local_setup/compute_node_env.sh"
SCENESMITH_RUNTIME_FILES: tuple[str, ...] = (
    "scenesmith/agent_utils/action_logger.py",
    "scenesmith/agent_utils/asset_manager.py",
    "scenesmith/agent_utils/asset_registry.py",
    "scenesmith/agent_utils/asset_router/router.py",
    "scenesmith/agent_utils/articulated_retrieval_server/server_manager.py",
    "scenesmith/agent_utils/base_stateful_agent.py",
    "scenesmith/agent_utils/codex_cli.py",
    "scenesmith/agent_utils/physical_feasibility.py",
    "scenesmith/agent_utils/physics_tools.py",
    "scenesmith/agent_utils/physics_validation.py",
    "scenesmith/agent_utils/room.py",
    "scenesmith/agent_utils/sceneeval_exporter.py",
    "scenesmith/agent_utils/support_surface_extraction.py",
    "scenesmith/agent_utils/workflow_tools.py",
    "scenesmith/furniture_agents/stateful_furniture_agent.py",
    "scenesmith/furniture_agents/tools/vision_tools.py",
    "scenesmith/experiments/base_experiment.py",
    "scenesmith/experiments/indoor_scene_generation.py",
    "scenesmith/manipuland_agents/base_manipuland_agent.py",
    "scenesmith/manipuland_agents/stateful_manipuland_agent.py",
    "scenesmith/manipuland_agents/tools/manipuland_tools.py",
    "scenesmith/manipuland_agents/tools/vision_tools.py",
    "scenesmith/agent_utils/geometry_generation_server/server_manager.py",
    "scenesmith/agent_utils/geometry_generation_server/cuda_env_setup.py",
    "scenesmith/agent_utils/geometry_generation_server/geometry_generation.py",
    "scenesmith/agent_utils/geometry_generation_server/sam3d_pipeline_manager.py",
    "scenesmith/agent_utils/hssd_retrieval_server/server_manager.py",
    "scenesmith/agent_utils/materials_retrieval_server/server_manager.py",
    "scenesmith/agent_utils/objaverse_retrieval_server/server_manager.py",
    "scenesmith/agent_utils/artiverse_visual_normalization.py",
    "scenesmith/agent_utils/mesh_utils.py",
    "scenesmith/agent_utils/urdf_to_sdf.py",
    "scenesmith/utils/inertia_utils.py",
    "scenesmith/utils/openai.py",
    "scenesmith/utils/sdf_utils.py",
    "scenesmith/wall_agents/tools/wall_surface.py",
)
CONTRACT_REGRESSION_FILES: tuple[str, ...] = (
    "tests/integration/test_image_persistence.py",
    "tests/unit/test_action_logger.py",
    "tests/unit/test_asset_registry_autosave.py",
    "tests/unit/test_artiverse_retrieval_integration.py",
    "tests/unit/test_artiverse_visual_normalization.py",
    "tests/unit/test_artiverse_visual_resources.py",
    "tests/unit/test_artiverse_support_surface_extraction.py",
    "tests/unit/test_apply_room_pose_correction.py",
    "tests/unit/test_prove_recovered_room_render.py",
    "tests/unit/test_recover_reference_classroom_furniture.py",
    "tests/unit/test_manipuland_agent.py",
    "tests/unit/test_manipuland_furniture_checkpoint.py",
    "tests/unit/test_manipuland_furniture_scope.py",
    "tests/unit/test_asset_registry_resume_contract.py",
    "tests/unit/test_inprocess_server_shutdown.py",
    "tests/unit/test_objathor_manipuland_routing.py",
    "tests/unit/test_physical_feasibility_thread_timeout.py",
    "tests/unit/test_sam3d_dtype_contract.py",
    "tests/unit/test_sam3d_memory_guard.py",
    "tests/unit/test_sam3d_preflight_evidence.py",
    "tests/unit/test_preflight_sam3d_generation.py",
    "tests/unit/test_workflow_and_vision_guards.py",
)

REQUIRED_EXTERNAL_INPUT_BINDING: dict[str, Any] = {
    "profile": "school_reference_20260710",
    "run_name": "full_quality_school_reference_sam3d_artvip_artiverse_20260710",
    "original_prompt_sha256": "5ddc06e1a9afa60b0417da882c1ec53265eae23f3f9bdd2343360d123923f34c",
    "effective_prompt_sha256": "ac8d297cc9a2d605f41b4bcd7abd52aac29bfd0f195875840342ee1e6a7da86f",
    "reference_image_sha256": "7ba62c39ac98cd2b21d9a5a97fd6f9b90d7efaf447d5ac9a6148524b5a8dbe48",
}
RUNNER_EXTERNAL_BINDING_VARIABLES = {
    "EXPECTED_ORIGINAL_PROMPT_SHA256": REQUIRED_EXTERNAL_INPUT_BINDING[
        "original_prompt_sha256"
    ],
    "EXPECTED_EFFECTIVE_PROMPT_SHA256": REQUIRED_EXTERNAL_INPUT_BINDING[
        "effective_prompt_sha256"
    ],
    "EXPECTED_REFERENCE_IMAGE_SHA256": REQUIRED_EXTERNAL_INPUT_BINDING[
        "reference_image_sha256"
    ],
}

# These clauses are intentional literal bindings, not keyword checks.  Changing
# their wording is a contract change and requires creating a new proof.
REQUIRED_SPEC_CLAUSES: tuple[tuple[str, str], ...] = (
    (
        "full_quality",
        "- Do not use an `hssd_only_fast` run as the final-quality run unless the user explicitly approves it.",
    ),
    (
        "no_hssd",
        "- Do not set `general_asset_source=hssd` for a full-quality scene unless the user explicitly asks for HSSD-only speed.",
    ),
    (
        "sam3d",
        "- Keep SAM3D available by using `general_asset_source=generated` and `backend=sam3d` for furniture, wall, ceiling, and manipuland agents when the target is quality.",
    ),
    (
        "compulsory_artiverse",
        "- Artiverse is compulsory for every full-quality run. The local `data/artiverse` dataset and the `artiverse_articulated` route must exist, be enabled, pass validation, and contribute at least one asset that survives into the final assembled scene. An enabled flag without proven final asset usage does not satisfy this contract.",
    ),
    (
        "artiverse_publisher_external_gltf_visuals",
        "- Every copied Artiverse SDF used by a room must normalize its material-split OBJ visuals through the matching publisher-authored GLB into deterministic room-local external `.gltf` plus `.bin` resources with explicit normals before Drake/Blender rendering; only the exact derived resource directory and grouped copied visual elements may be added or changed, while publisher sources, collisions, joints, inertials, poses, scales, prompt content, and all quality settings remain unchanged. Direct `.glb` SDF mesh URIs are forbidden because Drake ignores them.",
    ),
    (
        "room_gate",
        "- Do not assemble `combined_house`, final Drake exports, Isaac/USD exports, or final renders until every required room has passed a room-level quality gate.",
    ),
    (
        "two_gpu_acceptance",
        "- Copy or mount the exact hash-verified final package on a 2-GPU ParaCloud allocation and rerun `scripts/validate_drake_scene.py` there with `--require-gpus 2 --max-collision-elements 32`. The report must show `status=pass`, `visible_gpu_count >= 2`, `two_gpu_acceptance_environment=true`, no malformed/over-cap SDFs, and the full-house load must complete without OOM.",
    ),
    (
        "responses_visual_tool_transport",
        "- Production room workers must use the OpenAI Responses API for multimodal agent turns. Do not force official Chat Completions: Agents SDK 0.17.4 reduces function-tool results to text on that transport and silently drops every `ToolOutputImage`, leaving the designer visually blind. The Responses request must preserve the mixed image/text `function_call_output` unchanged, and every `data:image/png;base64` payload must decode to real PNG bytes before generation starts.",
    ),
    (
        "sam3d_full_generation_preflight",
        "- Before any paid API probe or room generation, run the production `generate_geometry_from_image` entrypoint with `backend=sam3d` and all model hubs offline on the exact committed `tests/test_data/office_shelf.png`; require an atomically published, hash-attested, nonempty loadable GLB plus mask and masked-image evidence, then use only its model-free `--verify-only` path on later benchmark/final acceptance checks.",
    ),
    (
        "sam3d_request_memory_guard",
        "- Every production SAM3D room worker must enable PyTorch `expandable_segments:True` before importing CUDA and must release the full request-scoped output graph plus unused CUDA allocator cache before and after each mesh decode, including the exception path. This memory guard may not unload the cached models or reduce segmentation, inference steps, texture baking, layout postprocessing, mesh validation, or any other quality setting; a conflicting allocator setting or repeated CUDA OOM is a hard stop before further paid retries.",
    ),
)


class PipelineCodeContractError(RuntimeError):
    """Raised when the immutable pipeline/code contract cannot be proven."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_runner_external_bindings(runner_text: str) -> None:
    missing = [
        name
        for name, value in RUNNER_EXTERNAL_BINDING_VARIABLES.items()
        if f"readonly {name}={value}" not in runner_text
    ]
    if missing:
        raise PipelineCodeContractError(
            "Production runner is missing exact external input bindings: "
            + ", ".join(sorted(missing))
        )


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if is_junction is not None and is_junction():
        return True
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, FileNotFoundError, OSError):
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_flag and attributes & reparse_flag)


def _normal_path(path: Path) -> str:
    return os.path.normcase(os.path.normpath(os.fspath(path)))


def _inside(root: Path, candidate: Path) -> bool:
    try:
        return os.path.commonpath((_normal_path(root), _normal_path(candidate))) == _normal_path(root)
    except ValueError:
        return False


def _repository_root(repo_dir: Path | str) -> Path:
    supplied = Path(repo_dir).expanduser().absolute()
    if not supplied.exists():
        raise PipelineCodeContractError(f"Repository directory is missing: {supplied}")
    if _is_link_like(supplied):
        raise PipelineCodeContractError(
            f"Repository directory must not be a symlink/junction: {supplied}"
        )
    try:
        mode = supplied.lstat().st_mode
    except OSError as exc:
        raise PipelineCodeContractError(f"Cannot inspect repository directory: {exc}") from exc
    if not stat.S_ISDIR(mode):
        raise PipelineCodeContractError(f"Repository path is not a directory: {supplied}")
    return supplied.resolve(strict=True)


def _repo_path(repo: Path, value: Path | str, *, label: str) -> Path:
    supplied = Path(value).expanduser()
    if not supplied.is_absolute():
        if ".." in supplied.parts:
            raise PipelineCodeContractError(f"{label} path contains an escape: {value}")
        candidate = (repo / supplied).absolute()
    else:
        candidate = supplied.absolute()
    if not _inside(repo, candidate):
        raise PipelineCodeContractError(f"{label} path escapes repository: {candidate}")

    try:
        relative = candidate.relative_to(repo)
    except ValueError as exc:
        raise PipelineCodeContractError(f"{label} path escapes repository: {candidate}") from exc
    current = repo
    for part in relative.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            if _is_link_like(current):
                raise PipelineCodeContractError(
                    f"{label} path contains a symlink/junction: {current}"
                )
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PipelineCodeContractError(f"{label} is missing: {candidate}") from exc
    if not _inside(repo, resolved):
        raise PipelineCodeContractError(f"{label} resolves outside repository: {candidate}")
    return resolved


def _relative(repo: Path, path: Path) -> str:
    try:
        relative = path.relative_to(repo)
    except ValueError as exc:
        raise PipelineCodeContractError(f"Artifact escapes repository: {path}") from exc
    value = relative.as_posix()
    if not value or value == "." or value.startswith("../"):
        raise PipelineCodeContractError(f"Invalid canonical artifact path: {value!r}")
    return value


def _read_regular_file(repo: Path, path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    # Revalidate the path immediately before reading; this also rejects a link in
    # any parent component rather than trusting resolve() to hide it.
    safe_path = _repo_path(repo, _relative(repo, path), label=label)
    try:
        before = safe_path.lstat()
    except OSError as exc:
        raise PipelineCodeContractError(f"Cannot inspect {label}: {exc}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise PipelineCodeContractError(f"{label} is not a regular file: {safe_path}")
    try:
        data = safe_path.read_bytes()
        after = safe_path.lstat()
    except OSError as exc:
        raise PipelineCodeContractError(f"Cannot fully read {label}: {exc}") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(data) != after.st_size:
        raise PipelineCodeContractError(f"{label} changed while it was being read")
    return data, {
        "path": _relative(repo, safe_path),
        "size_bytes": len(data),
        "sha256": _sha256_bytes(data),
    }


def _scan_regular_files(
    repo: Path,
    root_relative: str,
    *,
    suffixes: frozenset[str],
    label: str,
) -> list[Path]:
    root = _repo_path(repo, root_relative, label=label)
    if not root.is_dir():
        raise PipelineCodeContractError(f"{label} is not a directory: {root}")
    selected: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise PipelineCodeContractError(f"Cannot scan {label}: {exc}") from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise PipelineCodeContractError(f"Cannot inspect {label} entry {path}: {exc}") from exc
            if entry.is_symlink() or _is_link_like(path):
                raise PipelineCodeContractError(
                    f"{label} contains a symlink/junction: {_relative(repo, path)}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(path)
            elif stat.S_ISREG(metadata.st_mode):
                if path.suffix.lower() in suffixes:
                    selected.append(path.resolve(strict=True))
            else:
                raise PipelineCodeContractError(
                    f"{label} contains a special filesystem entry: {_relative(repo, path)}"
                )
    return sorted(selected, key=lambda path: _relative(repo, path))


def _required_clause_records(spec_text: str) -> list[dict[str, str]]:
    lines = set(spec_text.splitlines())
    missing = [identifier for identifier, clause in REQUIRED_SPEC_CLAUSES if clause not in lines]
    if missing:
        raise PipelineCodeContractError(
            "Pipeline specification is missing exact required clauses: " + ", ".join(missing)
        )
    return [
        {
            "id": identifier,
            "text": clause,
            "sha256": _sha256_bytes(clause.encode("utf-8")),
        }
        for identifier, clause in REQUIRED_SPEC_CLAUSES
    ]


def _parse_patch_order(repo: Path, order_path: Path) -> tuple[list[str], dict[str, Any]]:
    raw, evidence = _read_regular_file(repo, order_path, label="upstream patch order")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PipelineCodeContractError("APPLY_ORDER.txt is not valid UTF-8") from exc
    names = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not names:
        raise PipelineCodeContractError("APPLY_ORDER.txt contains no patches")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise PipelineCodeContractError(
            "APPLY_ORDER.txt contains duplicate patches: " + ", ".join(duplicates)
        )
    for name in names:
        parsed = Path(name)
        if (
            parsed.is_absolute()
            or name != parsed.name
            or ".." in parsed.parts
            or parsed.suffix.lower() != ".patch"
        ):
            raise PipelineCodeContractError(f"Unsafe patch entry in APPLY_ORDER.txt: {name!r}")

    patch_dir = order_path.parent
    discovered = _scan_regular_files(
        repo,
        _relative(repo, patch_dir),
        suffixes=frozenset({".patch"}),
        label="upstream patch directory",
    )
    discovered_names = {_relative(repo, path) for path in discovered}
    listed_paths = [f"{_relative(repo, patch_dir)}/{name}" for name in names]
    listed_set = set(listed_paths)
    unlisted = sorted(discovered_names - listed_set)
    missing = sorted(listed_set - discovered_names)
    if unlisted or missing:
        details = []
        if unlisted:
            details.append("unlisted patch files: " + ", ".join(unlisted))
        if missing:
            details.append("listed patches missing: " + ", ".join(missing))
        raise PipelineCodeContractError("Patch inventory mismatch; " + "; ".join(details))
    return listed_paths, evidence


def _add_artifact(
    artifacts: dict[str, dict[str, Any]],
    evidence: Mapping[str, Any],
    role: str,
) -> None:
    path = str(evidence["path"])
    record = artifacts.get(path)
    if record is None:
        record = {
            "path": path,
            "size_bytes": int(evidence["size_bytes"]),
            "sha256": str(evidence["sha256"]),
            "roles": [],
        }
        artifacts[path] = record
    elif (
        record["size_bytes"] != evidence["size_bytes"]
        or record["sha256"] != evidence["sha256"]
    ):
        raise PipelineCodeContractError(f"Artifact changed during contract creation: {path}")
    if role not in record["roles"]:
        record["roles"].append(role)


def _attest(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "algorithm": HASH_ALGORITHM,
        "sha256": _sha256_bytes(_canonical_json(payload)),
    }


def _verify_build_snapshot(repo: Path, payload: Mapping[str, Any]) -> None:
    """Close inventory/read races before a new attestation is published."""

    inventory = payload["inventory"]
    current_scripts = [
        _relative(repo, path)
        for path in _scan_regular_files(
            repo,
            "scripts",
            suffixes=frozenset({".py"}),
            label="scripts tree",
        )
    ]
    current_remote_jobs = [
        _relative(repo, path)
        for path in _scan_regular_files(
            repo,
            "remote_jobs",
            suffixes=frozenset({".sh", ".sbatch"}),
            label="remote jobs tree",
        )
    ]
    order_path = _repo_path(
        repo,
        str(payload["patch_order_path"]),
        label="upstream patch order",
    )
    current_patches, _order_evidence = _parse_patch_order(repo, order_path)
    current_scenesmith_runtime = [
        _relative(
            repo,
            _repo_path(
                repo,
                relative_path,
                label=f"SceneSmith runtime dependency {relative_path}",
            ),
        )
        for relative_path in SCENESMITH_RUNTIME_FILES
    ]
    current_contract_regressions = [
        _relative(
            repo,
            _repo_path(
                repo,
                relative_path,
                label=f"contract regression {relative_path}",
            ),
        )
        for relative_path in CONTRACT_REGRESSION_FILES
    ]
    if current_scripts != inventory["scripts_python"]:
        raise PipelineCodeContractError("Scripts inventory changed during contract creation")
    if current_remote_jobs != inventory["remote_jobs"]:
        raise PipelineCodeContractError("Remote-jobs inventory changed during contract creation")
    if current_patches != inventory["upstream_patches"]:
        raise PipelineCodeContractError("Upstream patch inventory changed during contract creation")
    if current_scenesmith_runtime != inventory["scenesmith_runtime"]:
        raise PipelineCodeContractError(
            "SceneSmith runtime inventory changed during contract creation"
        )
    if current_contract_regressions != inventory["contract_regressions"]:
        raise PipelineCodeContractError(
            "Contract-regression inventory changed during contract creation"
        )

    for expected in payload["artifacts"]:
        path = _repo_path(repo, str(expected["path"]), label="attested artifact")
        raw, current = _read_regular_file(repo, path, label="attested artifact")
        del raw
        if (
            current["path"] != expected["path"]
            or current["size_bytes"] != expected["size_bytes"]
            or current["sha256"] != expected["sha256"]
        ):
            raise PipelineCodeContractError(
                f"Artifact changed during contract creation: {expected['path']}"
            )


def build_manifest(
    repo_dir: Path | str,
    spec: Path | str = DEFAULT_SPEC,
    runner: Path | str = DEFAULT_RUNNER,
) -> dict[str, Any]:
    """Create a deterministic manifest, reading the specification first."""

    repo = _repository_root(repo_dir)
    spec_path = _repo_path(repo, spec, label="pipeline specification")

    # Ordering is a security property: fully consume and hash the specification
    # before discovering or reading any generation-stage code.
    spec_raw, spec_evidence = _read_regular_file(
        repo, spec_path, label="pipeline specification"
    )
    try:
        spec_text = spec_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PipelineCodeContractError("Pipeline specification is not valid UTF-8") from exc
    clause_records = _required_clause_records(spec_text)

    runner_path = _repo_path(repo, runner, label="production runner")
    script_path = _repo_path(repo, CONTRACT_SCRIPT, label="pipeline contract script")
    sage_path = _repo_path(repo, SAGE_CHECKER, label="SAGE scene checker")
    order_path = _repo_path(repo, PATCH_ORDER, label="upstream patch order")
    compute_node_env_path = _repo_path(
        repo,
        COMPUTE_NODE_ENV,
        label="compute-node environment bootstrap",
    )
    scenesmith_runtime_paths = [
        _repo_path(
            repo,
            relative_path,
            label=f"SceneSmith runtime dependency {relative_path}",
        )
        for relative_path in SCENESMITH_RUNTIME_FILES
    ]
    contract_regression_paths = [
        _repo_path(
            repo,
            relative_path,
            label=f"contract regression {relative_path}",
        )
        for relative_path in CONTRACT_REGRESSION_FILES
    ]

    scripts_python = _scan_regular_files(
        repo,
        "scripts",
        suffixes=frozenset({".py"}),
        label="scripts tree",
    )
    remote_jobs = _scan_regular_files(
        repo,
        "remote_jobs",
        suffixes=frozenset({".sh", ".sbatch"}),
        label="remote jobs tree",
    )
    patch_paths, order_evidence = _parse_patch_order(repo, order_path)

    artifacts: dict[str, dict[str, Any]] = {}
    _add_artifact(artifacts, spec_evidence, "pipeline_specification")

    runner_raw, runner_evidence = _read_regular_file(repo, runner_path, label="production runner")
    try:
        runner_text = runner_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PipelineCodeContractError("Production runner is not valid UTF-8") from exc
    _require_runner_external_bindings(runner_text)
    del runner_raw, runner_text
    _add_artifact(artifacts, runner_evidence, "production_runner")

    script_raw, script_evidence = _read_regular_file(repo, script_path, label="pipeline contract script")
    del script_raw
    _add_artifact(artifacts, script_evidence, "pipeline_contract")

    sage_raw, sage_evidence = _read_regular_file(repo, sage_path, label="SAGE scene checker")
    del sage_raw
    _add_artifact(artifacts, sage_evidence, "sage_scene_checker")
    _add_artifact(artifacts, order_evidence, "upstream_patch_order")

    compute_node_env_raw, compute_node_env_evidence = _read_regular_file(
        repo,
        compute_node_env_path,
        label="compute-node environment bootstrap",
    )
    del compute_node_env_raw
    _add_artifact(
        artifacts,
        compute_node_env_evidence,
        "compute_node_environment",
    )

    for path in scenesmith_runtime_paths:
        raw, evidence = _read_regular_file(
            repo,
            path,
            label="SceneSmith runtime dependency",
        )
        del raw
        _add_artifact(
            artifacts,
            evidence,
            "scenesmith_runtime_dependency",
        )

    for path in contract_regression_paths:
        raw, evidence = _read_regular_file(
            repo,
            path,
            label="contract regression",
        )
        del raw
        _add_artifact(
            artifacts,
            evidence,
            "contract_regression",
        )

    for path in scripts_python:
        raw, evidence = _read_regular_file(repo, path, label="scripts Python file")
        del raw
        _add_artifact(artifacts, evidence, "scripts_python")
    for path in remote_jobs:
        raw, evidence = _read_regular_file(repo, path, label="remote job")
        del raw
        _add_artifact(artifacts, evidence, "remote_job")
    for relative_path in patch_paths:
        path = _repo_path(repo, relative_path, label="ordered upstream patch")
        raw, evidence = _read_regular_file(repo, path, label="ordered upstream patch")
        del raw
        _add_artifact(artifacts, evidence, "upstream_patch")

    artifact_records = []
    for path in sorted(artifacts):
        record = artifacts[path]
        record["roles"] = sorted(record["roles"])
        artifact_records.append(record)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_NAME,
        "status": "pass",
        "hash_algorithm": HASH_ALGORITHM,
        "path_encoding": "repository-relative-posix",
        "snapshot_reverified_after_inventory": True,
        "external_input_binding": dict(REQUIRED_EXTERNAL_INPUT_BINDING),
        "specification": {
            **spec_evidence,
            "read_before_generation_code": True,
            "required_clauses": clause_records,
        },
        "runner_path": _relative(repo, runner_path),
        "contract_script_path": _relative(repo, script_path),
        "sage_checker_path": _relative(repo, sage_path),
        "patch_order_path": _relative(repo, order_path),
        "compute_node_environment_path": _relative(repo, compute_node_env_path),
        "ordered_patches": patch_paths,
        "inventory": {
            "scripts_python": [_relative(repo, path) for path in scripts_python],
            "remote_jobs": [_relative(repo, path) for path in remote_jobs],
            "scenesmith_runtime": [
                _relative(repo, path) for path in scenesmith_runtime_paths
            ],
            "contract_regressions": [
                _relative(repo, path) for path in contract_regression_paths
            ],
            "upstream_patches": patch_paths,
        },
        "artifact_count": len(artifact_records),
        "artifacts": artifact_records,
    }
    _verify_build_snapshot(repo, payload)
    return {**payload, "attestation": _attest(payload)}


def _without_attestation(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "attestation"}


def _load_manifest(path: Path) -> dict[str, Any]:
    if _is_link_like(path):
        raise PipelineCodeContractError(f"Manifest must not be a symlink/junction: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise PipelineCodeContractError(f"Cannot read saved pipeline/code manifest: {exc}") from exc

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PipelineCodeContractError(f"Manifest contains duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except PipelineCodeContractError:
        raise
    except (json.JSONDecodeError, TypeError) as exc:
        raise PipelineCodeContractError(f"Saved pipeline/code manifest is malformed: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineCodeContractError("Saved pipeline/code manifest must be a JSON object")
    return value


def _validate_saved_attestation(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise PipelineCodeContractError("Saved manifest schema_version is unsupported")
    if manifest.get("contract") != CONTRACT_NAME or manifest.get("status") != "pass":
        raise PipelineCodeContractError("Saved manifest is not a passing pipeline/code contract")
    attestation = manifest.get("attestation")
    if (
        not isinstance(attestation, dict)
        or attestation.get("schema_version") != ATTESTATION_SCHEMA_VERSION
        or attestation.get("algorithm") != HASH_ALGORITHM
    ):
        raise PipelineCodeContractError("Saved manifest attestation is missing or malformed")
    expected = _attest(_without_attestation(manifest))
    if attestation != expected:
        raise PipelineCodeContractError("Saved pipeline/code manifest attestation was tampered with")


def _artifact_index(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return {}
    return {
        str(item.get("path")): item
        for item in artifacts
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }


def _manifest_difference(saved: Mapping[str, Any], current: Mapping[str, Any]) -> str:
    saved_artifacts = _artifact_index(saved)
    current_artifacts = _artifact_index(current)
    added = sorted(current_artifacts.keys() - saved_artifacts.keys())
    removed = sorted(saved_artifacts.keys() - current_artifacts.keys())
    mutated = sorted(
        path
        for path in saved_artifacts.keys() & current_artifacts.keys()
        if saved_artifacts[path] != current_artifacts[path]
    )
    details = []
    if added:
        details.append("added artifacts: " + ", ".join(added))
    if removed:
        details.append("removed artifacts: " + ", ".join(removed))
    if mutated:
        details.append("mutated artifacts: " + ", ".join(mutated))
    if not details:
        details.append("manifest metadata or ordered inventory changed")
    return "; ".join(details)


def verify_manifest(
    output: Path | str,
    repo_dir: Path | str,
    spec: Path | str = DEFAULT_SPEC,
    runner: Path | str = DEFAULT_RUNNER,
) -> dict[str, Any]:
    saved = _load_manifest(Path(output).expanduser().absolute())
    _validate_saved_attestation(saved)
    current = build_manifest(repo_dir, spec, runner)
    if saved != current:
        raise PipelineCodeContractError(
            "Pipeline/code contract verification failed: "
            + _manifest_difference(saved, current)
        )
    return saved


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    if path.exists() and _is_link_like(path):
        raise PipelineCodeContractError(f"Output manifest must not be a symlink/junction: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() and _is_link_like(temporary):
        raise PipelineCodeContractError(f"Temporary manifest path is a symlink/junction: {temporary}")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run(
    repo_dir: Path | str,
    spec: Path | str,
    runner: Path | str,
    output: Path | str,
    *,
    verify_only: bool = False,
) -> dict[str, Any]:
    output_path = Path(output).expanduser().absolute()
    if verify_only:
        return verify_manifest(output_path, repo_dir, spec, runner)
    manifest = build_manifest(repo_dir, spec, runner)
    _write_manifest(output_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=Path(DEFAULT_SPEC))
    parser.add_argument("--runner", type=Path, default=Path(DEFAULT_RUNNER))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Re-read and re-hash the saved proof without rewriting it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run(
            args.repo_dir,
            args.spec,
            args.runner,
            args.output,
            verify_only=args.verify_only,
        )
    except PipelineCodeContractError as exc:
        print(f"FAIL: {exc}")
        return 1
    action = "verified" if args.verify_only else "created"
    print(
        f"PASS: {action} pipeline/code contract with "
        f"{manifest['artifact_count']} artifacts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
