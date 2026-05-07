#!/usr/bin/env python3
"""Export, strictly validate, and rollback-safely publish MuJoCo/USD artifacts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

if __package__:
    from .validate_simulator_exports import ATTEMPT_MARKER, validate_exports
else:
    from validate_simulator_exports import ATTEMPT_MARKER, validate_exports


_RUN_ATTEMPT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _unique_sibling(path: Path, role: str, run_attempt_id: str) -> Path:
    return path.with_name(
        f".{path.name}.{role}.{run_attempt_id}.{os.getpid()}.{uuid.uuid4().hex}"
    )


def _write_attempt_marker(staging_dir: Path, run_attempt_id: str) -> None:
    marker = {
        "schema_version": 1,
        "run_attempt_id": run_attempt_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    marker_path = staging_dir / ATTEMPT_MARKER
    with marker_path.open("x", encoding="utf-8") as stream:
        json.dump(marker, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_exporter_fail_closed_contract(exporter: Path) -> dict[str, Any]:
    """Reject an execution checkout that has not applied the USD-fatal patch."""

    source = exporter.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(exporter))
    except SyntaxError as exc:
        raise RuntimeError(f"Cannot parse simulator exporter: {exc}") from exc
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "export_to_usd"
    ]
    if len(functions) != 1:
        raise RuntimeError(
            "Simulator exporter must define exactly one top-level export_to_usd function"
        )
    function = functions[0]
    function_source = ast.get_source_segment(source, function) or ""
    required_guards = (
        "Requested USD export cannot run while bpy is importable",
        "Requested USD export dependencies are unavailable",
        "Requested USD export failed",
        "USD converter did not create its requested root layer",
        "OpenUSD could not load converted root layer",
    )
    missing_guards = [guard for guard in required_guards if guard not in function_source]

    bpy_try_is_fatal = False
    conversion_handlers_are_fatal = False
    for candidate in ast.walk(function):
        if not isinstance(candidate, ast.Try):
            continue
        body_nodes = [node for statement in candidate.body for node in ast.walk(statement)]
        imported_names = {
            alias.name
            for node in body_nodes
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        if "bpy" in imported_names and any(
            isinstance(node, ast.Raise) for node in body_nodes
        ):
            bpy_try_is_fatal = True
        if "mujoco_usd_converter" in imported_names:
            conversion_handlers_are_fatal = bool(candidate.handlers) and all(
                any(
                    isinstance(node, ast.Raise)
                    for statement in handler.body
                    for node in ast.walk(statement)
                )
                for handler in candidate.handlers
            )

    if missing_guards or not bpy_try_is_fatal or not conversion_handlers_are_fatal:
        raise RuntimeError(
            "SceneSmith simulator exporter is not the fail-closed patched version; "
            f"missing_guards={missing_guards}, bpy_fatal={bpy_try_is_fatal}, "
            f"conversion_handlers_fatal={conversion_handlers_are_fatal}. "
            "Reapply the ordered upstream patch stack before launch."
        )
    return {
        "path": str(exporter),
        "sha256": _sha256_file(exporter),
        "usd_failure_contract": "pass",
    }


def _remove_tree(path: Path) -> None:
    if not _path_exists(path):
        return
    if path.is_symlink() or not path.is_dir():
        path.unlink()
    else:
        shutil.rmtree(path)


def _run_exporter(command: Sequence[str]) -> None:
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Simulator exporter failed with exit code {completed.returncode}: "
            + " ".join(command)
        )


def export_validate_and_publish(
    *,
    scene_dir: Path,
    published_dir: Path,
    validation_output: Path,
    exporter: Path,
    run_attempt_id: str,
    require_usd: bool,
    expected_mjcf: str = "scene.xml",
    additional_expected_usd_artifacts: Iterable[str] | None = None,
    exporter_runner: Callable[[Sequence[str]], None] = _run_exporter,
) -> dict[str, Any]:
    """Publish only an export created and validated for this exact attempt."""

    if _RUN_ATTEMPT_PATTERN.fullmatch(run_attempt_id) is None:
        raise RuntimeError(
            "RUN_ATTEMPT_ID must be 1-128 safe filename characters and start "
            "with an alphanumeric character"
        )

    scene_dir = scene_dir.resolve()
    published_dir = published_dir.absolute()
    validation_output = validation_output.absolute()
    exporter = exporter.resolve()
    if not scene_dir.is_dir():
        raise RuntimeError(f"Scene directory is missing: {scene_dir}")
    if not exporter.is_file():
        raise RuntimeError(f"Simulator exporter is missing: {exporter}")
    if validation_output == published_dir or published_dir in validation_output.parents:
        raise RuntimeError("Validation output must be outside the published export directory")

    _atomic_write_json(
        validation_output,
        {
            "schema_version": 2,
            "status": "running",
            "run_attempt_id": run_attempt_id,
            "published_dir": str(published_dir),
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    exporter_contract = _verify_exporter_fail_closed_contract(exporter)
    published_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = _unique_sibling(published_dir, "staging", run_attempt_id)
    staging_dir.mkdir(mode=0o700)
    _write_attempt_marker(staging_dir, run_attempt_id)

    command = [
        sys.executable,
        str(exporter),
        str(scene_dir),
        "-o",
        str(staging_dir),
    ]
    if require_usd:
        command.append("--usd")

    exporter_runner(command)
    expected_usd = tuple(additional_expected_usd_artifacts or ())
    staged_result = validate_exports(
        staging_dir,
        require_usd,
        expected_mjcf=expected_mjcf,
        additional_expected_usd_artifacts=expected_usd,
        expected_run_attempt_id=run_attempt_id,
    )

    backup_dir: Path | None = None
    displaced_new_export: Path | None = None
    if _path_exists(published_dir):
        if published_dir.is_symlink() or not published_dir.is_dir():
            raise RuntimeError(
                f"Published simulator export is not a real directory: {published_dir}"
            )
        backup_dir = _unique_sibling(published_dir, "previous", run_attempt_id)
        os.replace(published_dir, backup_dir)

    try:
        os.replace(staging_dir, published_dir)
        published_result = validate_exports(
            published_dir,
            require_usd,
            expected_mjcf=expected_mjcf,
            additional_expected_usd_artifacts=expected_usd,
            expected_run_attempt_id=run_attempt_id,
        )
        if published_result["inventory_sha256"] != staged_result["inventory_sha256"]:
            raise RuntimeError("Published simulator inventory changed during atomic promotion")
        published_result["publication"] = {
            "run_attempt_id": run_attempt_id,
            "published_dir": str(published_dir),
            "staging_inventory_sha256": staged_result["inventory_sha256"],
            "previous_export_replaced": backup_dir is not None,
            "atomic_promotion": True,
            "exporter_contract": exporter_contract,
            "previous_backup_cleanup": (
                "pending" if backup_dir is not None else "not_applicable"
            ),
        }
        _atomic_write_json(validation_output, published_result)
    except Exception as promotion_error:
        restore_errors: list[str] = []
        if _path_exists(published_dir):
            displaced_new_export = _unique_sibling(
                published_dir, "failed", run_attempt_id
            )
            try:
                os.replace(published_dir, displaced_new_export)
            except OSError as exc:
                restore_errors.append(f"could not displace failed new export: {exc}")
        if backup_dir is not None and _path_exists(backup_dir):
            try:
                os.replace(backup_dir, published_dir)
            except OSError as exc:
                restore_errors.append(f"could not restore previous export: {exc}")
        detail = str(promotion_error)
        if restore_errors:
            detail += "; rollback errors: " + "; ".join(restore_errors)
        raise RuntimeError(f"Simulator export promotion failed: {detail}") from promotion_error

    if backup_dir is not None:
        try:
            _remove_tree(backup_dir)
            published_result["publication"]["previous_backup_cleanup"] = "removed"
        except Exception as cleanup_error:
            published_result["publication"]["previous_backup_cleanup"] = "retained"
            published_result["publication"]["cleanup_warning"] = str(cleanup_error)
        try:
            _atomic_write_json(validation_output, published_result)
        except Exception as evidence_update_error:
            print(
                "Simulator export passed, but could not update nonfatal backup "
                f"cleanup evidence: {evidence_update_error}",
                file=sys.stderr,
            )
    return published_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--published-dir", required=True, type=Path)
    parser.add_argument("--validation-output", required=True, type=Path)
    parser.add_argument(
        "--exporter", type=Path, default=Path("scripts/export_scene_to_mujoco.py")
    )
    parser.add_argument(
        "--run-attempt-id",
        default=os.environ.get("RUN_ATTEMPT_ID"),
        help="Unique attempt ID (defaults to RUN_ATTEMPT_ID)",
    )
    parser.add_argument("--require-usd", action="store_true")
    parser.add_argument("--expected-mjcf", default="scene.xml")
    parser.add_argument("--expected-usd-artifact", action="append", default=[])
    args = parser.parse_args()

    if not args.run_attempt_id:
        parser.error("--run-attempt-id or RUN_ATTEMPT_ID is required")

    try:
        result = export_validate_and_publish(
            scene_dir=args.scene_dir,
            published_dir=args.published_dir,
            validation_output=args.validation_output,
            exporter=args.exporter,
            run_attempt_id=args.run_attempt_id,
            require_usd=args.require_usd,
            expected_mjcf=args.expected_mjcf,
            additional_expected_usd_artifacts=args.expected_usd_artifact,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": 2,
            "status": "fail",
            "run_attempt_id": args.run_attempt_id,
            "published_dir": str(args.published_dir.resolve()),
            "error": str(exc),
        }
        try:
            _atomic_write_json(args.validation_output.resolve(), failure)
        except Exception as write_exc:
            print(
                f"Failed to write simulator-export failure evidence: {write_exc}",
                file=sys.stderr,
            )
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
