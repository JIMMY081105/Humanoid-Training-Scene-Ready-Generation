#!/usr/bin/env python3
"""Factory-specific immutable package and terminal two-GPU acceptance contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


PROFILE = "factory_reference_20260713"
MANIFEST_SCHEMA = "scenesmith_factory_package_manifest_v1"
VALIDATION_SCHEMA = "scenesmith_factory_package_validation_v1"
COMPLETION_SCHEMA = "scenesmith_factory_pipeline_completion_v1"
RUNTIME_SCHEMA = "scenesmith_factory_2gpu_runtime_v1"
FINAL_SCHEMA = "scenesmith_factory_2gpu_acceptance_v1"


class ContractError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _regular(path: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    if lexical.is_symlink():
        raise ContractError(f"{label} must not be a symlink: {lexical}")
    try:
        info = lexical.stat(follow_symlinks=False)
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"missing {label}: {lexical}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise ContractError(f"{label} must be a non-empty regular file: {lexical}")
    return resolved


def _read_json(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = _regular(path, label)
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"duplicate key in {label}: {key}")
            result[key] = value
        return result
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(document, dict):
        raise ContractError(f"{label} must be a JSON object")
    return resolved, document


def _attest(document: dict[str, Any]) -> dict[str, Any]:
    payload = dict(document)
    payload.pop("attestation", None)
    document["attestation"] = {
        "algorithm": "sha256", "schema_version": 1,
        "sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
    }
    return document


def _validate_attestation(document: dict[str, Any], label: str) -> None:
    saved = document.get("attestation")
    payload = dict(document)
    payload.pop("attestation", None)
    expected = {"algorithm": "sha256", "schema_version": 1,
                "sha256": hashlib.sha256(_canonical(payload)).hexdigest()}
    if saved != expected:
        raise ContractError(f"{label} attestation is stale")


def _atomic_json(path: Path, document: dict[str, Any], *, overwrite: bool = False) -> None:
    path = Path(os.path.abspath(os.fspath(path)))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise ContractError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _inventory(package_root: Path) -> list[dict[str, Any]]:
    root = package_root.resolve(strict=True)
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        lexical = Path(os.path.abspath(os.fspath(path)))
        info = lexical.stat(follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
            raise ContractError(f"package contains a symlink: {lexical}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ContractError(f"package contains a special file: {lexical}")
        resolved = lexical.resolve(strict=True)
        if not _inside(root, resolved) or info.st_size <= 0:
            raise ContractError(f"invalid package file: {lexical}")
        records.append({
            "path": resolved.relative_to(root).as_posix(),
            "size_bytes": info.st_size,
            "sha256": _sha256(resolved),
        })
    if not records:
        raise ContractError("package inventory is empty")
    return records


def _manifest_document(package_root: Path, run_attempt_id: str) -> dict[str, Any]:
    files = _inventory(package_root)
    return _attest({
        "schema_id": MANIFEST_SCHEMA,
        "schema_version": 1,
        "status": "pass",
        "contract_profile": PROFILE,
        "run_attempt_id": run_attempt_id,
        "package_root_name": package_root.resolve(strict=True).name,
        "file_count": len(files),
        "total_bytes": sum(int(item["size_bytes"]) for item in files),
        "inventory_sha256": hashlib.sha256(_canonical(files)).hexdigest(),
        "files": files,
    })


def _validation(manifest_path: Path, manifest: dict[str, Any], package_root: Path) -> dict[str, Any]:
    _validate_attestation(manifest, "package manifest")
    if manifest.get("schema_id") != MANIFEST_SCHEMA or manifest.get("status") != "pass":
        raise ContractError("package manifest schema/status is invalid")
    fresh = _manifest_document(package_root, str(manifest.get("run_attempt_id") or ""))
    if fresh != manifest:
        raise ContractError("package differs from the immutable manifest")
    return _attest({
        "schema_id": VALIDATION_SCHEMA,
        "schema_version": 1,
        "status": "pass",
        "contract_profile": PROFILE,
        "run_attempt_id": manifest["run_attempt_id"],
        "package_root": str(package_root.resolve(strict=True)),
        "manifest": {"path": str(manifest_path.resolve(strict=True)),
                     "sha256": _sha256(manifest_path)},
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "inventory_sha256": manifest["inventory_sha256"],
        "all_files_rehashed": True,
    })


def create_manifest(args: argparse.Namespace) -> None:
    root = args.package_root.resolve(strict=True)
    manifest = _manifest_document(root, args.expected_run_attempt_id)
    _atomic_json(args.manifest, manifest)
    validation = _validation(args.manifest.resolve(strict=True), manifest, root)
    _atomic_json(args.output, validation, overwrite=True)


def verify_package(args: argparse.Namespace) -> None:
    manifest_path, manifest = _read_json(args.manifest, "package manifest")
    if _sha256(manifest_path) != args.expected_manifest_sha256:
        raise ContractError("out-of-band package-manifest digest mismatch")
    if manifest.get("run_attempt_id") != args.expected_run_attempt_id:
        raise ContractError("package manifest run-attempt mismatch")
    validation = _validation(manifest_path, manifest, args.package_root.resolve(strict=True))
    _atomic_json(args.output, validation, overwrite=True)


def create_completion(args: argparse.Namespace) -> None:
    validation_path, validation = _read_json(args.package_validation, "package validation")
    _validate_attestation(validation, "package validation")
    if validation.get("schema_id") != VALIDATION_SCHEMA or validation.get("status") != "pass":
        raise ContractError("package validation did not pass")
    if validation.get("run_attempt_id") != args.expected_run_attempt_id:
        raise ContractError("package validation run-attempt mismatch")
    result = _attest({
        "schema_id": COMPLETION_SCHEMA, "schema_version": 1,
        "status": "awaiting_2gpu_acceptance", "contract_profile": PROFILE,
        "run_attempt_id": args.expected_run_attempt_id,
        "package_validation": {"path": str(validation_path), "sha256": _sha256(validation_path)},
        "manifest_sha256": validation["manifest"]["sha256"],
        "inventory_sha256": validation["inventory_sha256"],
    })
    _atomic_json(args.output, result, overwrite=True)


def _visible_gpu_count() -> int:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    return len([line for line in completed.stdout.splitlines() if line.strip()])


def verify_runtime(args: argparse.Namespace) -> None:
    preflight_path, preflight = _read_json(args.package_preflight, "package preflight")
    _validate_attestation(preflight, "package preflight")
    if preflight.get("status") != "pass" or preflight.get("run_attempt_id") != args.expected_run_attempt_id:
        raise ContractError("package preflight status/attempt is invalid")
    repo = args.repo_root.resolve(strict=True)
    count = _visible_gpu_count()
    if count < 2:
        raise ContractError(f"terminal runtime exposes only {count} GPUs")
    try:
        import pydrake  # noqa: F401
    except Exception as exc:
        raise ContractError(f"pydrake import failed: {exc}") from exc
    result = _attest({
        "schema_id": RUNTIME_SCHEMA, "schema_version": 1, "status": "pass",
        "contract_profile": PROFILE, "run_attempt_id": args.expected_run_attempt_id,
        "repo_root": str(repo), "python_executable": sys.executable,
        "visible_gpu_count": count,
        "package_preflight": {"path": str(preflight_path), "sha256": _sha256(preflight_path)},
    })
    _atomic_json(args.output, result, overwrite=True)


def finalize(args: argparse.Namespace) -> None:
    paths_docs = {}
    for key, path in {
        "preflight": args.package_preflight, "postflight": args.package_postflight,
        "drake": args.drake_report, "runtime": args.runtime_validation,
        "completion": args.sqz_pipeline_completion,
    }.items():
        paths_docs[key] = _read_json(path, key)
    for key in ("preflight", "postflight", "runtime", "completion"):
        _validate_attestation(paths_docs[key][1], key)
    pre = paths_docs["preflight"][1]
    post = paths_docs["postflight"][1]
    runtime = paths_docs["runtime"][1]
    completion_path, completion = paths_docs["completion"]
    drake_path, drake = paths_docs["drake"]
    if any(doc.get("run_attempt_id") != args.expected_run_attempt_id for doc in (pre, post, runtime, completion)):
        raise ContractError("terminal evidence run-attempt mismatch")
    for key in ("manifest", "inventory_sha256", "file_count", "total_bytes"):
        if pre.get(key) != post.get(key):
            raise ContractError(f"package changed between pre/post flight: {key}")
    if runtime.get("status") != "pass" or int(runtime.get("visible_gpu_count", 0)) < 2:
        raise ContractError("two-GPU runtime validation did not pass")
    if _sha256(completion_path) != args.expected_sqz_completion_sha256:
        raise ContractError("out-of-band pipeline-completion digest mismatch")
    if completion.get("status") != "awaiting_2gpu_acceptance":
        raise ContractError("pipeline completion is not awaiting terminal acceptance")
    requirements = drake.get("acceptance_requirements") or {}
    checks = drake.get("checks") or {}
    if (
        drake.get("status") != "pass"
        or requirements.get("required_visible_gpu_count") != 2
        or requirements.get("required_gpu_execution_count") != 2
        or requirements.get("expected_room_count") != 14
        or drake.get("two_gpu_acceptance_environment") is not True
        or checks.get("required_gpus_available") is not True
        or checks.get("required_gpus_exercised") is not True
        or checks.get("expected_room_count_satisfied") is not True
    ):
        raise ContractError("terminal Drake report does not prove the exact 2-GPU/14-room contract")
    result = _attest({
        "schema_id": FINAL_SCHEMA, "schema_version": 1, "status": "pass",
        "contract_profile": PROFILE, "run_attempt_id": args.expected_run_attempt_id,
        "slurm_job_id": args.slurm_job_id, "node_name": args.node_name,
        "manifest_sha256": pre["manifest"]["sha256"],
        "inventory_sha256": pre["inventory_sha256"],
        "package_preflight": {"path": str(paths_docs["preflight"][0]), "sha256": _sha256(paths_docs["preflight"][0])},
        "package_postflight": {"path": str(paths_docs["postflight"][0]), "sha256": _sha256(paths_docs["postflight"][0])},
        "runtime_validation": {"path": str(paths_docs["runtime"][0]), "sha256": _sha256(paths_docs["runtime"][0])},
        "drake_report": {"path": str(drake_path), "sha256": _sha256(drake_path)},
        "pipeline_completion": {"path": str(completion_path), "sha256": _sha256(completion_path)},
        "critical_issues": [],
    })
    _atomic_json(args.output, result, overwrite=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-manifest")
    create.add_argument("--package-root", type=Path, required=True)
    create.add_argument("--manifest", type=Path, required=True)
    create.add_argument("--expected-run-attempt-id", required=True)
    create.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify-package")
    verify.add_argument("--package-root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--expected-manifest-sha256", required=True)
    verify.add_argument("--expected-run-attempt-id", required=True)
    verify.add_argument("--output", type=Path, required=True)
    completion = sub.add_parser("create-sqz-completion")
    completion.add_argument("--package-validation", type=Path, required=True)
    completion.add_argument("--expected-run-attempt-id", required=True)
    completion.add_argument("--output", type=Path, required=True)
    runtime = sub.add_parser("verify-runtime")
    runtime.add_argument("--package-preflight", type=Path, required=True)
    runtime.add_argument("--repo-root", type=Path, required=True)
    runtime.add_argument("--expected-run-attempt-id", required=True)
    runtime.add_argument("--output", type=Path, required=True)
    final = sub.add_parser("finalize")
    final.add_argument("--package-preflight", type=Path, required=True)
    final.add_argument("--package-postflight", type=Path, required=True)
    final.add_argument("--drake-report", type=Path, required=True)
    final.add_argument("--runtime-validation", type=Path, required=True)
    final.add_argument("--sqz-pipeline-completion", type=Path, required=True)
    final.add_argument("--expected-sqz-completion-sha256", required=True)
    final.add_argument("--expected-run-attempt-id", required=True)
    final.add_argument("--slurm-job-id", required=True)
    final.add_argument("--node-name", required=True)
    final.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        {"create-manifest": create_manifest, "verify-package": verify_package,
         "create-sqz-completion": create_completion, "verify-runtime": verify_runtime,
         "finalize": finalize}[args.command](args)
        print(json.dumps({"status": "pass", "command": args.command}, indent=2, sort_keys=True))
        return 0
    except (ContractError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"factory two-GPU contract failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
