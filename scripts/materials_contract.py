#!/usr/bin/env python3
"""Prepare and validate an isolated, fail-closed materials retrieval index.

The publisher/shared materials directory is treated as read-only.  ``prepare``
copies only the valid embedding rows and metadata into a separate contract
directory; ``validate`` binds that derivative to the current source indexes and
to a deterministic inventory of the retained material asset files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile

from pathlib import Path
from typing import Any

import numpy as np
import yaml


SCHEMA_VERSION = 1
SCHEMA_ID = "scenesmith_materials_contract_v1"
REQUIRED_INDEX_FILES = (
    "clip_embeddings.npy",
    "embedding_index.yaml",
    "metadata_index.yaml",
)
MANIFEST_NAME = "materials_contract_manifest.json"


class MaterialsContractError(RuntimeError):
    """Raised when the isolated materials index is missing, stale, or invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exact YAML exceptions vary.
        raise MaterialsContractError(f"Cannot parse YAML {path}: {exc}") from exc


def _require_index_files(directory: Path, *, label: str) -> None:
    if not directory.is_dir():
        raise MaterialsContractError(f"{label} directory is missing: {directory}")
    missing = [
        name
        for name in REQUIRED_INDEX_FILES
        if not (directory / name).is_file() or (directory / name).stat().st_size == 0
    ]
    if missing:
        raise MaterialsContractError(
            f"{label} is incomplete; missing or empty: {', '.join(missing)}"
        )


def _load_index_set(
    directory: Path, *, label: str
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    _require_index_files(directory, label=label)
    try:
        embeddings = np.load(directory / "clip_embeddings.npy", mmap_mode="r")
    except Exception as exc:
        raise MaterialsContractError(
            f"Cannot load {label} clip embeddings: {exc}"
        ) from exc
    embedding_index = _load_yaml(directory / "embedding_index.yaml")
    metadata_index = _load_yaml(directory / "metadata_index.yaml")
    if embeddings.ndim != 2 or embeddings.shape[0] < 1 or embeddings.shape[1] < 1:
        raise MaterialsContractError(
            f"{label} embeddings must be a non-empty matrix, got {embeddings.shape}"
        )
    if not isinstance(embedding_index, list) or not all(
        isinstance(item, str) and item for item in embedding_index
    ):
        raise MaterialsContractError(
            f"{label} embedding_index.yaml must be a string list"
        )
    if len(set(embedding_index)) != len(embedding_index):
        raise MaterialsContractError(f"{label} embedding index contains duplicate IDs")
    if embeddings.shape[0] != len(embedding_index):
        raise MaterialsContractError(
            f"{label} row/index mismatch: {embeddings.shape[0]} != {len(embedding_index)}"
        )
    if not isinstance(metadata_index, dict):
        raise MaterialsContractError(f"{label} metadata_index.yaml must be a mapping")
    if set(metadata_index) != set(embedding_index):
        missing = sorted(set(embedding_index) - set(metadata_index))
        extra = sorted(set(metadata_index) - set(embedding_index))
        raise MaterialsContractError(
            f"{label} metadata/index mismatch; missing={missing}, extra={extra}"
        )
    return embeddings, embedding_index, metadata_index


def _safe_material_dir(data_root: Path, material_id: str) -> Path:
    relative = Path(material_id)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise MaterialsContractError(f"Unsafe material ID in index: {material_id!r}")
    root = data_root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise MaterialsContractError(
            f"Material ID escapes data root: {material_id!r}"
        ) from exc
    return candidate


def _material_exclusion_reason(data_root: Path, material_id: str) -> str | None:
    material_dir = _safe_material_dir(data_root, material_id)
    if not material_dir.is_dir():
        return "missing_material_directory"
    color_maps = sorted(material_dir.glob("*_Color.jpg"))
    if not any(path.is_file() and path.stat().st_size > 0 for path in color_maps):
        return "missing_nonempty_color_texture"
    return None


def _asset_inventory_sha256(data_root: Path, material_ids: list[str]) -> str:
    """Hash a portable inventory of every retained asset path and byte size.

    The large shared texture corpus remains read-only, and validation stays fast:
    index files are content-hashed while the 45 GB asset tree is bound by its
    complete relative-path/size inventory plus explicit non-empty color-map checks.
    """

    digest = hashlib.sha256()
    root = data_root.resolve()
    for material_id in material_ids:
        directory = _safe_material_dir(root, material_id)
        files = sorted(path for path in directory.rglob("*") if path.is_file())
        if not files:
            raise MaterialsContractError(
                f"Retained material directory contains no files: {material_id}"
            )
        for path in files:
            relative = path.relative_to(root).as_posix()
            record = f"{relative}\0{path.stat().st_size}\n".encode("utf-8")
            digest.update(record)
    return digest.hexdigest()


def _hashes(directory: Path) -> dict[str, str]:
    return {name: sha256_file(directory / name) for name in REQUIRED_INDEX_FILES}


def _guard_isolated_output(
    *, data_root: Path, source_embeddings: Path, contract_embeddings: Path
) -> None:
    data_root = data_root.resolve()
    source_embeddings = source_embeddings.resolve()
    contract_embeddings = contract_embeddings.resolve()
    if contract_embeddings == source_embeddings:
        raise MaterialsContractError(
            "Contract embeddings must not overwrite the shared source embeddings"
        )
    for protected in (data_root, source_embeddings):
        try:
            contract_embeddings.relative_to(protected)
        except ValueError:
            continue
        raise MaterialsContractError(
            f"Contract output must be outside protected shared path: {protected}"
        )


def prepare_contract(
    *,
    data_root: Path,
    source_embeddings: Path,
    contract_embeddings: Path,
    min_retained: int = 1,
    max_pruned: int = 100,
) -> dict[str, Any]:
    data_root = data_root.resolve()
    source_embeddings = source_embeddings.resolve()
    contract_embeddings = contract_embeddings.resolve()
    _guard_isolated_output(
        data_root=data_root,
        source_embeddings=source_embeddings,
        contract_embeddings=contract_embeddings,
    )
    if not data_root.is_dir():
        raise MaterialsContractError(f"Materials data root is missing: {data_root}")

    source_matrix, source_ids, source_metadata = _load_index_set(
        source_embeddings, label="source materials index"
    )
    retained_ids: list[str] = []
    retained_rows: list[int] = []
    excluded: list[dict[str, str]] = []
    for row, material_id in enumerate(source_ids):
        reason = _material_exclusion_reason(data_root, material_id)
        if reason is None:
            retained_ids.append(material_id)
            retained_rows.append(row)
        else:
            excluded.append({"material_id": material_id, "reason": reason})

    if len(retained_ids) < min_retained:
        raise MaterialsContractError(
            f"Only {len(retained_ids)} valid materials remain; minimum is {min_retained}"
        )
    if len(excluded) > max_pruned:
        raise MaterialsContractError(
            f"Would prune {len(excluded)} materials; maximum is {max_pruned}"
        )

    contract_metadata = {
        material_id: source_metadata[material_id] for material_id in retained_ids
    }
    contract_matrix = np.asarray(source_matrix[retained_rows])
    contract_embeddings.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=contract_embeddings.parent, prefix=".materials-contract-"
    ) as temporary_name:
        temporary = Path(temporary_name)
        np.save(temporary / "clip_embeddings.npy", contract_matrix)
        (temporary / "embedding_index.yaml").write_text(
            yaml.safe_dump(retained_ids, sort_keys=False), encoding="utf-8"
        )
        (temporary / "metadata_index.yaml").write_text(
            yaml.safe_dump(contract_metadata, sort_keys=True), encoding="utf-8"
        )
        manifest = {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "status": "pass",
            "source": {
                "row_count": len(source_ids),
                "embedding_dimension": int(source_matrix.shape[1]),
                "embedding_dtype": str(source_matrix.dtype),
                "index_sha256": _hashes(source_embeddings),
            },
            "contract": {
                "retained_count": len(retained_ids),
                "pruned_count": len(excluded),
                "retained_ids_sha256": hashlib.sha256(
                    ("\n".join(retained_ids) + "\n").encode("utf-8")
                ).hexdigest(),
                "asset_inventory_sha256": _asset_inventory_sha256(
                    data_root, retained_ids
                ),
                "index_sha256": _hashes(temporary),
            },
            "excluded": excluded,
        }
        (temporary / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # Prove the complete candidate before publishing any file.  The manifest
        # is moved last below, so even an interrupted per-file replacement leaves
        # a fail-closed hash mismatch instead of an apparently valid mixed set.
        load_materials_authority(
            data_root=data_root,
            source_embeddings=source_embeddings,
            contract_embeddings=temporary,
            min_retained=min_retained,
            max_pruned=max_pruned,
        )

        contract_embeddings.mkdir(parents=True, exist_ok=True)
        for filename in (*REQUIRED_INDEX_FILES, MANIFEST_NAME):
            os.replace(temporary / filename, contract_embeddings / filename)

    return load_materials_authority(
        data_root=data_root,
        source_embeddings=source_embeddings,
        contract_embeddings=contract_embeddings,
        min_retained=min_retained,
        max_pruned=max_pruned,
    )


def load_materials_authority(
    *,
    data_root: Path,
    source_embeddings: Path,
    contract_embeddings: Path,
    min_retained: int = 1,
    max_pruned: int = 100,
) -> dict[str, Any]:
    """Validate the derivative against source indexes and current asset inventory."""

    data_root = data_root.resolve()
    source_embeddings = source_embeddings.resolve()
    contract_embeddings = contract_embeddings.resolve()
    _guard_isolated_output(
        data_root=data_root,
        source_embeddings=source_embeddings,
        contract_embeddings=contract_embeddings,
    )
    if not data_root.is_dir():
        raise MaterialsContractError(f"Materials data root is missing: {data_root}")
    _require_index_files(contract_embeddings, label="contract materials index")
    manifest_path = contract_embeddings / MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        raise MaterialsContractError(
            f"Materials contract manifest is missing: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MaterialsContractError(f"Cannot parse materials manifest: {exc}") from exc
    if (
        manifest.get("schema_id") != SCHEMA_ID
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise MaterialsContractError("Unsupported materials contract manifest schema")
    if manifest.get("status") != "pass":
        raise MaterialsContractError("Materials contract manifest is not passing")

    source_matrix, source_ids, source_metadata = _load_index_set(
        source_embeddings, label="source materials index"
    )
    contract_matrix, contract_ids, contract_metadata = _load_index_set(
        contract_embeddings, label="contract materials index"
    )
    if manifest.get("source", {}).get("index_sha256") != _hashes(source_embeddings):
        raise MaterialsContractError(
            "Source materials index hashes do not match manifest"
        )
    if manifest.get("contract", {}).get("index_sha256") != _hashes(contract_embeddings):
        raise MaterialsContractError(
            "Contract materials index hashes do not match manifest"
        )
    if manifest.get("source", {}).get("row_count") != len(source_ids):
        raise MaterialsContractError(
            "Source materials row count does not match manifest"
        )
    if manifest.get("source", {}).get("embedding_dimension") != int(
        source_matrix.shape[1]
    ):
        raise MaterialsContractError(
            "Source embedding dimension does not match manifest"
        )
    if manifest.get("source", {}).get("embedding_dtype") != str(source_matrix.dtype):
        raise MaterialsContractError("Source embedding dtype does not match manifest")

    exclusions: list[dict[str, str]] = []
    expected_ids: list[str] = []
    expected_rows: list[int] = []
    for row, material_id in enumerate(source_ids):
        reason = _material_exclusion_reason(data_root, material_id)
        if reason is None:
            expected_ids.append(material_id)
            expected_rows.append(row)
        else:
            exclusions.append({"material_id": material_id, "reason": reason})
    if contract_ids != expected_ids:
        raise MaterialsContractError(
            "Contract material IDs are stale or are not the exact valid source subset"
        )
    if manifest.get("excluded") != exclusions:
        raise MaterialsContractError("Materials exclusion list is stale or malformed")
    if len(contract_ids) < min_retained:
        raise MaterialsContractError(
            f"Only {len(contract_ids)} contract materials remain; minimum is {min_retained}"
        )
    if len(exclusions) > max_pruned:
        raise MaterialsContractError(
            f"Contract prunes {len(exclusions)} materials; maximum is {max_pruned}"
        )
    expected_metadata = {
        material_id: source_metadata[material_id] for material_id in expected_ids
    }
    if contract_metadata != expected_metadata:
        raise MaterialsContractError(
            "Contract metadata is not the exact valid source subset"
        )
    if contract_matrix.shape != (len(expected_ids), source_matrix.shape[1]):
        raise MaterialsContractError("Contract embedding matrix has the wrong shape")
    if contract_matrix.dtype != source_matrix.dtype:
        raise MaterialsContractError("Contract embedding dtype differs from source")
    if not np.array_equal(
        np.asarray(contract_matrix),
        np.asarray(source_matrix[expected_rows]),
        equal_nan=True,
    ):
        raise MaterialsContractError("Contract embeddings are not exact source rows")

    retained_ids_hash = hashlib.sha256(
        ("\n".join(contract_ids) + "\n").encode("utf-8")
    ).hexdigest()
    if manifest.get("contract", {}).get("retained_ids_sha256") != retained_ids_hash:
        raise MaterialsContractError(
            "Retained material ID hash does not match manifest"
        )
    inventory_hash = _asset_inventory_sha256(data_root, contract_ids)
    if manifest.get("contract", {}).get("asset_inventory_sha256") != inventory_hash:
        raise MaterialsContractError("Retained material asset inventory is stale")
    if manifest.get("contract", {}).get("retained_count") != len(contract_ids):
        raise MaterialsContractError("Retained material count does not match manifest")
    if manifest.get("contract", {}).get("pruned_count") != len(exclusions):
        raise MaterialsContractError("Pruned material count does not match manifest")

    return {
        "status": "pass",
        "schema_id": SCHEMA_ID,
        "data_root": str(data_root),
        "source_embeddings": str(source_embeddings),
        "contract_embeddings": str(contract_embeddings),
        "source_count": len(source_ids),
        "retained_count": len(contract_ids),
        "pruned_count": len(exclusions),
        "excluded": exclusions,
        "source_index_sha256": _hashes(source_embeddings),
        "contract_index_sha256": _hashes(contract_embeddings),
        "asset_inventory_sha256": inventory_hash,
        "manifest_sha256": sha256_file(manifest_path),
    }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=Path("data/materials"))
    parser.add_argument(
        "--source-embeddings",
        type=Path,
        default=Path("data/materials/embeddings"),
    )
    parser.add_argument(
        "--contract-embeddings",
        type=Path,
        default=Path("data/materials_full_quality_contract/embeddings"),
    )
    parser.add_argument("--min-retained", type=int, default=1900)
    parser.add_argument("--max-pruned", type=int, default=15)
    parser.add_argument("--output", type=Path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    validate_parser = subparsers.add_parser("validate")
    _add_common_arguments(prepare_parser)
    _add_common_arguments(validate_parser)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare_contract(
                data_root=args.data_root,
                source_embeddings=args.source_embeddings,
                contract_embeddings=args.contract_embeddings,
                min_retained=args.min_retained,
                max_pruned=args.max_pruned,
            )
        else:
            result = load_materials_authority(
                data_root=args.data_root,
                source_embeddings=args.source_embeddings,
                contract_embeddings=args.contract_embeddings,
                min_retained=args.min_retained,
                max_pruned=args.max_pruned,
            )
    except MaterialsContractError as exc:
        result = {"status": "fail", "error": str(exc)}
        if args.output:
            _atomic_write_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    if args.output:
        _atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
