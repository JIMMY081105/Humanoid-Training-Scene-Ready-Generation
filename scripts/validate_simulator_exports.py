#!/usr/bin/env python3
"""Fail-closed MuJoCo/USD validation with a hash-bound export inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET

from pathlib import Path
from typing import Any, Iterable


ATTEMPT_MARKER = ".export_attempt.json"
_USD_SUFFIXES = {".usd", ".usda", ".usdc"}
_BASE_USD_PAYLOADS = (
    "usd/Payload/Contents.usda",
    "usd/Payload/Geometry.usda",
    "usd/Payload/Physics.usda",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _inventory(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Simulator export contains a symlink: {path}")
        if not path.is_file():
            continue
        if not _is_within(path, root):
            raise RuntimeError(f"Simulator export path escapes its root: {path}")
        size = path.stat().st_size
        if size < 1:
            raise RuntimeError(f"Simulator export contains an empty file: {path}")
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": size,
                "sha256": _sha256_file(path),
            }
        )
    if not records:
        raise RuntimeError(f"Simulator export directory is empty: {root}")
    return records


def _resolve_mjcf_asset(output_dir: Path, directory: str, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        candidate = relative
    else:
        candidate = output_dir / directory / relative
    resolved = candidate.resolve()
    if not _is_within(resolved, output_dir):
        raise RuntimeError(f"MJCF asset escapes export root: {value}")
    return resolved


def _safe_export_path(output_dir: Path, value: str, label: str) -> Path:
    relative = Path(value)
    if not value or relative.is_absolute() or relative == Path("."):
        raise RuntimeError(f"{label} must be a nonempty relative path: {value!r}")
    candidate = (output_dir / relative).resolve()
    if not _is_within(candidate, output_dir):
        raise RuntimeError(f"{label} escapes export root: {value}")
    return candidate


def _require_file(path: Path, label: str) -> None:
    if path.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink: {path}")
    if not path.is_file() or path.stat().st_size < 1:
        raise RuntimeError(f"{label} is missing or empty: {path}")


def _validate_attempt_marker(output_dir: Path, expected_run_attempt_id: str) -> dict[str, Any]:
    marker_path = output_dir / ATTEMPT_MARKER
    _require_file(marker_path, "Simulator export attempt marker")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot parse simulator export attempt marker: {exc}") from exc
    if marker.get("run_attempt_id") != expected_run_attempt_id:
        raise RuntimeError(
            "Simulator export attempt marker mismatch: "
            f"expected {expected_run_attempt_id!r}, found {marker.get('run_attempt_id')!r}"
        )
    return {
        "path": ATTEMPT_MARKER,
        "sha256": _sha256_file(marker_path),
        "run_attempt_id": expected_run_attempt_id,
    }


def _validate_mujoco(output_dir: Path, expected_mjcf: str) -> tuple[dict[str, Any], ET.Element]:
    scene_xml = _safe_export_path(output_dir, expected_mjcf, "Expected MJCF path")
    _require_file(scene_xml, "Expected MuJoCo MJCF")
    try:
        root = ET.parse(scene_xml).getroot()
    except (OSError, ET.ParseError) as exc:
        raise RuntimeError(f"Cannot parse MuJoCo scene.xml: {exc}") from exc
    supported_file_elements = {"mesh", "texture"}
    unsupported_file_references = []
    for element in root.iter():
        if not element.get("file"):
            continue
        local_tag = element.tag.rsplit("}", 1)[-1]
        if local_tag not in supported_file_elements:
            unsupported_file_references.append(
                f"<{local_tag} file={element.get('file')!r}>"
            )
    if unsupported_file_references:
        raise RuntimeError(
            "MJCF contains external file reference types that are not root/hash-bound; "
            "the exporter must inline them or the validator must explicitly support them: "
            + ", ".join(unsupported_file_references)
        )
    compiler = root.find("compiler")
    mesh_dir = compiler.get("meshdir", "") if compiler is not None else ""
    texture_dir = compiler.get("texturedir", mesh_dir) if compiler is not None else mesh_dir
    referenced: dict[str, dict[str, Any]] = {}
    for element, directory in (
        (".//asset/mesh", mesh_dir),
        (".//asset/texture", texture_dir),
    ):
        for asset in root.findall(element):
            file_value = asset.get("file")
            if not file_value:
                continue
            path = _resolve_mjcf_asset(output_dir, directory, file_value)
            if not path.is_file() or path.stat().st_size < 1:
                raise RuntimeError(f"Referenced MuJoCo asset is missing: {path}")
            relative = path.relative_to(output_dir).as_posix()
            referenced[relative] = {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }

    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(scene_xml))
        data = mujoco.MjData(model)
        for _ in range(10):
            mujoco.mj_step(model, data)
    except Exception as exc:
        raise RuntimeError(f"MuJoCo load/step validation failed: {exc}") from exc
    return {
        "scene_xml": scene_xml.relative_to(output_dir).as_posix(),
        "scene_xml_sha256": _sha256_file(scene_xml),
        "referenced_asset_count": len(referenced),
        "referenced_assets": [referenced[path] for path in sorted(referenced)],
        "model_counts": {
            "nbody": int(model.nbody),
            "ngeom": int(model.ngeom),
            "njnt": int(model.njnt),
            "nq": int(model.nq),
            "nv": int(model.nv),
        },
    }, root


def _derive_expected_usd_artifacts(
    mjcf_root: ET.Element,
    additional_expected: Iterable[str] | None,
) -> list[str]:
    model_name = (mjcf_root.get("model") or "").strip()
    if not model_name:
        raise RuntimeError("MJCF root must have a nonempty model name for exact USD binding")
    if Path(model_name).name != model_name or model_name in {".", ".."}:
        raise RuntimeError(f"MJCF model name cannot identify a top-level USD layer: {model_name!r}")

    expected = {
        f"usd/{model_name}.usda",
        *_BASE_USD_PAYLOADS,
    }
    if mjcf_root.findall(".//asset/mesh"):
        expected.add("usd/Payload/GeometryLibrary.usdc")
    non_grid_materials = [
        material
        for material in mjcf_root.findall(".//asset/material")
        if material.get("name") != "grid"
    ]
    if non_grid_materials:
        expected.update(
            {
                "usd/Payload/Materials.usda",
                "usd/Payload/MaterialsLibrary.usdc",
            }
        )
    expected.update(additional_expected or ())
    return sorted(expected)


def _validate_usd(
    output_dir: Path,
    mjcf_root: ET.Element,
    additional_expected: Iterable[str] | None,
) -> dict[str, Any]:
    usd_dir = output_dir / "usd"
    if not usd_dir.is_dir():
        raise RuntimeError(f"Required USD directory is missing: {usd_dir}")
    candidates = sorted(
        path
        for path in usd_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _USD_SUFFIXES
    )
    if not candidates:
        raise RuntimeError(f"USD export contains no USD stage/layer: {usd_dir}")
    try:
        from pxr import Usd
    except ImportError as exc:
        raise RuntimeError("OpenUSD pxr module is unavailable in export environment") from exc

    expected = _derive_expected_usd_artifacts(mjcf_root, additional_expected)
    expected_paths: list[Path] = []
    for relative in expected:
        path = _safe_export_path(output_dir, relative, "Expected USD artifact")
        if not _is_within(path, usd_dir):
            raise RuntimeError(f"Expected USD artifact is outside the USD directory: {relative}")
        _require_file(path, "Expected USD artifact")
        if path.suffix.lower() not in _USD_SUFFIXES:
            raise RuntimeError(f"Expected USD artifact is not a USD layer: {relative}")
        expected_paths.append(path)

    candidate_set = {path.resolve() for path in candidates}
    missing_candidates = [
        path.relative_to(output_dir).as_posix()
        for path in expected_paths
        if path.resolve() not in candidate_set
    ]
    if missing_candidates:
        raise RuntimeError(
            "Expected USD artifacts were not included in the candidate inventory: "
            + ", ".join(missing_candidates)
        )

    top_level = [path for path in candidates if path.parent == usd_dir]
    if len(top_level) != 1:
        raise RuntimeError(
            "USD export must contain exactly one top-level root layer; found "
            + repr([path.name for path in top_level])
        )
    model_root = _safe_export_path(
        output_dir,
        f"usd/{(mjcf_root.get('model') or '').strip()}.usda",
        "Expected USD root layer",
    )
    if top_level[0].resolve() != model_root.resolve():
        raise RuntimeError(
            f"USD root layer mismatch: expected {model_root.name}, found {top_level[0].name}"
        )

    stages: list[dict[str, Any]] = []
    failures: list[str] = []
    root_used_layers: set[str] = set()
    for path in candidates:
        try:
            stage = Usd.Stage.Open(str(path))
            if stage is None:
                raise RuntimeError("Usd.Stage.Open returned None")
            prim_count = sum(1 for _ in stage.TraverseAll())
            if prim_count < 1:
                raise RuntimeError("stage contains no prims")
            used_layers: list[str] = []
            for layer in stage.GetUsedLayers():
                real_path = Path(layer.realPath) if layer.realPath else None
                if real_path is None:
                    continue
                if not real_path.is_absolute():
                    raise RuntimeError(
                        "USD used layer has an ambiguous relative realPath: "
                        f"{real_path}"
                    )
                if not _is_within(real_path, usd_dir) or not real_path.is_file():
                    raise RuntimeError(f"USD layer escapes/is missing: {real_path}")
                used_layers.append(real_path.relative_to(output_dir).as_posix())
            if path.resolve() == top_level[0].resolve():
                root_used_layers.update(used_layers)
            stages.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "sha256": _sha256_file(path),
                    "prim_count": prim_count,
                    "used_layers": sorted(set(used_layers)),
                }
            )
        except Exception as exc:
            failures.append(f"{path}: {exc}")
    if failures:
        raise RuntimeError(
            "One or more USD candidate layers failed load validation: " + "; ".join(failures)
        )
    if len(stages) != len(candidates):
        raise RuntimeError(
            f"USD candidate accounting mismatch: validated {len(stages)} of {len(candidates)}"
        )

    required_used_layers = {
        path.relative_to(output_dir).as_posix() for path in expected_paths
    } - {top_level[0].relative_to(output_dir).as_posix()}
    missing_from_root = sorted(required_used_layers - root_used_layers)
    if missing_from_root:
        raise RuntimeError(
            "USD root stage does not use every expected payload layer: "
            + ", ".join(missing_from_root)
        )

    expected_inventory = [
        {
            "path": path.relative_to(output_dir).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in expected_paths
    ]
    return {
        "usd_dir": "usd",
        "usd_layer_count": len(candidates),
        "expected_artifacts": expected_inventory,
        "validated_stages": stages,
        "candidate_failures": [],
    }


def validate_exports(
    output_dir: Path,
    require_usd: bool,
    *,
    expected_mjcf: str = "scene.xml",
    additional_expected_usd_artifacts: Iterable[str] | None = None,
    expected_run_attempt_id: str | None = None,
) -> dict[str, Any]:
    root = output_dir.resolve()
    if not root.is_dir():
        raise RuntimeError(f"Simulator export directory is missing: {root}")
    marker_result = (
        _validate_attempt_marker(root, expected_run_attempt_id)
        if expected_run_attempt_id is not None
        else None
    )
    mujoco_result, mjcf_root = _validate_mujoco(root, expected_mjcf)
    usd_result = (
        _validate_usd(root, mjcf_root, additional_expected_usd_artifacts)
        if require_usd
        else None
    )
    inventory = _inventory(root)
    inventory_payload = json.dumps(inventory, separators=(",", ":"), sort_keys=True)
    return {
        "schema_version": 2,
        "status": "pass",
        "output_dir": str(root),
        "require_usd": require_usd,
        "attempt_marker": marker_result,
        "mujoco": mujoco_result,
        "usd": usd_result,
        "file_count": len(inventory),
        "file_inventory": inventory,
        "inventory_sha256": hashlib.sha256(inventory_payload.encode("utf-8")).hexdigest(),
    }


def _atomic_write(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--require-usd", action="store_true")
    parser.add_argument("--expected-mjcf", default="scene.xml")
    parser.add_argument("--expected-usd-artifact", action="append", default=[])
    parser.add_argument("--expected-run-attempt-id")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = validate_exports(
            args.output_dir,
            args.require_usd,
            expected_mjcf=args.expected_mjcf,
            additional_expected_usd_artifacts=args.expected_usd_artifact,
            expected_run_attempt_id=args.expected_run_attempt_id,
        )
        _atomic_write(args.output.resolve(), result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        result = {"schema_version": 2, "status": "fail", "error": str(exc)}
        _atomic_write(args.output.resolve(), result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
