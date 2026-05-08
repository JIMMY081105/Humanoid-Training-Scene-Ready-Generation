#!/usr/bin/env python3
"""Create and reverify one real, service-free SAM3D geometry-generation proof.

The existing SAM3D load preflight proves that the two model objects initialize.
This separate gate exercises the exact production ``generate_geometry_from_image``
route, publishes a GLB plus its segmentation diagnostics atomically, and binds the
input, model/cache, executable code, and outputs in one attested receipt.

``--verify-only`` never imports the production generator and never performs model
inference.  It rehashes all bound inputs and reloads the published GLB with
``trimesh``.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
import shutil
import stat
import tempfile
import time
import sys

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from scripts import preflight_sam3d_offline as model_preflight


SCHEMA_ID = "scenesmith_sam3d_offline_generation_preflight_v1"
SCHEMA_VERSION = 1
ATTESTATION_SCOPE = "canonical_json_of_all_fields_except_attestation"
OFFLINE_VARIABLES = model_preflight.OFFLINE_VARIABLES
CANONICAL_INPUT_RELATIVE = Path("tests/test_data/office_shelf.png")
CANONICAL_INPUT_SHA256 = (
    "a69065c31da63e6604a76c4baba486718292df4bd3edadaba08a93e2a8fb09e7"
)
CANONICAL_INPUT_WIDTH = 1024
CANONICAL_INPUT_HEIGHT = 1536
ARTIFACT_DIRECTORY_NAME = "sam3d_offline_generation"
RECEIPT_NAME = "receipt.json"
GLB_NAME = "office_shelf_3d.glb"
MASK_NAME = "office_shelf_mask.png"
MASKED_IMAGE_NAME = "office_shelf_masked.png"
MODE = "foreground"
THRESHOLD = 0.5
HASH_CHUNK_SIZE = 1024 * 1024
SHA256_LENGTH = 64
CODE_RELATIVE_PATHS = (
    "scripts/preflight_sam3d_generation.py",
    "scripts/preflight_sam3d_offline.py",
    "scenesmith/agent_utils/geometry_generation_server/geometry_generation.py",
    "scenesmith/agent_utils/geometry_generation_server/sam3d_pipeline_manager.py",
    "scenesmith/agent_utils/geometry_generation_server/cuda_env_setup.py",
    "scenesmith/agent_utils/mesh_utils.py",
    "scripts/run_single_room_worker.py",
    "tests/unit/test_sam3d_memory_guard.py",
)
REPOSITORY_SOURCE_TREES = (
    ("external_sam3_python_source", "external/SAM3/sam3"),
    (
        "external_sam3d_objects_python_source",
        "external/sam-3d-objects/sam3d_objects",
    ),
)
MOGE_SOURCE_ROLE = "environment_moge_python_source"
SOURCE_TREE_MODULES = {
    "external_sam3_python_source": "sam3",
    "external_sam3d_objects_python_source": "sam3d_objects",
    MOGE_SOURCE_ROLE: "moge",
}
SOURCE_TREE_EXTRA_FILES = {
    "external_sam3_python_source": (
        "assets/bpe_simple_vocab_16e6.txt.gz",
    ),
}
RUNTIME_COMPONENTS = (
    ("torch", "torch", ("torch/_C*.so",), ()),
    ("torchvision", "torchvision", ("torchvision/_C*.so",), ()),
    ("pytorch3d", "pytorch3d", ("pytorch3d/_C*.so",), ()),
    ("spconv-cu121", "spconv", ("spconv/core_cc*.so",), ()),
    ("cumm-cu121", "cumm", ("cumm/core_cc*.so",), ()),
    # fnmatch's '*' spans '/' here, intentionally binding all Kaolin extensions.
    ("kaolin", "kaolin", ("kaolin/*.so",), ()),
    ("xformers", "xformers", ("xformers/_C*.so",), ()),
    ("gsplat", "gsplat", ("gsplat/csrc.so",), ()),
    ("nvdiffrast", "nvdiffrast", ("_nvdiffrast_c*.so",), ()),
    ("warp-lang", "warp", ("warp/bin/*.so",), ()),
    # utils3d loads shader/resource files at runtime; bind every RECORD-listed
    # package file, not merely its version or RECORD text.
    ("utils3d", "utils3d", (), ("utils3d/*",)),
    ("trimesh", "trimesh", (), ()),
    ("numpy", "numpy", ("numpy/core/_multiarray_umath*.so",), ()),
    ("Pillow", "PIL", ("PIL/_imaging*.so",), ()),
    ("scipy", "scipy", ("scipy/_lib/_ccallback_c*.so",), ()),
    ("hydra-core", "hydra", (), ()),
    ("omegaconf", "omegaconf", (), ()),
)
RUNTIME_DISTRIBUTIONS = tuple(item[0] for item in RUNTIME_COMPONENTS)


class SAM3DGenerationPreflightError(RuntimeError):
    """The full offline generation proof is absent, stale, or malformed."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _regular_file(
    path: Path, *, label: str, allow_empty: bool = False
) -> Path:
    if _is_link_like(path):
        raise SAM3DGenerationPreflightError(f"{label} must not be link-like: {path}")
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise SAM3DGenerationPreflightError(f"cannot inspect {label}: {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or (
        metadata.st_size < 1 and not allow_empty
    ):
        raise SAM3DGenerationPreflightError(
            f"{label} must be a nonempty regular file: {path}"
        )
    return path.resolve(strict=True)


def _real_directory(path: Path, *, label: str) -> Path:
    supplied = path.expanduser().absolute()
    if _is_link_like(supplied):
        raise SAM3DGenerationPreflightError(f"{label} must not be link-like: {supplied}")
    try:
        metadata = supplied.stat(follow_symlinks=False)
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise SAM3DGenerationPreflightError(
            f"cannot inspect {label}: {supplied}: {exc}"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise SAM3DGenerationPreflightError(f"{label} is not a directory: {supplied}")
    if os.path.normcase(os.fspath(resolved)) != os.path.normcase(os.fspath(supplied)):
        raise SAM3DGenerationPreflightError(
            f"{label} contains a link-like parent component: {supplied}"
        )
    return resolved


def _inside(root: Path, candidate: Path) -> bool:
    try:
        return os.path.commonpath((os.fspath(root), os.fspath(candidate))) == os.fspath(root)
    except ValueError:
        return False


def _repo_file(repo: Path, relative: Path | str, *, label: str) -> Path:
    parsed = Path(relative)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise SAM3DGenerationPreflightError(f"unsafe {label} path: {relative}")
    candidate = (repo / parsed).absolute()
    current = repo
    for part in parsed.parts:
        current = current / part
        if (current.exists() or current.is_symlink()) and _is_link_like(current):
            raise SAM3DGenerationPreflightError(
                f"{label} path contains a link-like component: {current}"
            )
    resolved = _regular_file(candidate, label=label)
    if not _inside(repo, resolved):
        raise SAM3DGenerationPreflightError(f"{label} escapes repository: {resolved}")
    return resolved


def canonical_paths(preflight_dir: Path) -> dict[str, Path]:
    root = preflight_dir.expanduser().absolute()
    artifact_dir = root / ARTIFACT_DIRECTORY_NAME
    return {
        "preflight_dir": root,
        "artifact_dir": artifact_dir,
        "receipt": artifact_dir / RECEIPT_NAME,
        "glb": artifact_dir / GLB_NAME,
        "mask": artifact_dir / MASK_NAME,
        "masked_image": artifact_dir / MASKED_IMAGE_NAME,
    }


def _reject_transaction_orphans(preflight_dir: Path) -> None:
    root = _real_directory(preflight_dir, label="SAM3D preflight directory")
    prefix = f".{ARTIFACT_DIRECTORY_NAME}."
    for entry in os.scandir(root):
        if entry.name.startswith(prefix) and entry.name.endswith(".tmp"):
            raise SAM3DGenerationPreflightError(
                f"orphan SAM3D generation transaction exists: {entry.path}"
            )


def _verify_exact_artifact_inventory(artifact_dir: Path) -> None:
    root = _real_directory(
        artifact_dir, label="SAM3D generation artifact directory"
    )
    expected = {RECEIPT_NAME, GLB_NAME, MASK_NAME, MASKED_IMAGE_NAME}
    actual: set[str] = set()
    for entry in os.scandir(root):
        candidate = Path(entry.path)
        if entry.is_symlink() or _is_link_like(candidate):
            raise SAM3DGenerationPreflightError(
                f"SAM3D generation artifact inventory contains a link: {candidate}"
            )
        metadata = entry.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size < 1:
            raise SAM3DGenerationPreflightError(
                "SAM3D generation artifact inventory contains a directory, special, "
                f"or empty entry: {candidate}"
            )
        actual.add(entry.name)
    if actual != expected:
        raise SAM3DGenerationPreflightError(
            "SAM3D generation artifact inventory is not exact; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _artifact(path: Path, *, relative_path: str, label: str) -> dict[str, Any]:
    regular = _regular_file(path, label=label)
    return {
        "relative_path": relative_path,
        "size_bytes": regular.stat().st_size,
        "sha256": _sha256_file(regular),
    }


def _code_evidence(repo: Path) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for relative in CODE_RELATIVE_PATHS:
        path = _repo_file(repo, relative, label=f"generation code {relative}")
        evidence.append({"path": relative, **_artifact(path, relative_path=relative, label=relative)})
    return evidence


def _verify_code_evidence(repo: Path, evidence: Any) -> list[str]:
    if not isinstance(evidence, list):
        return ["generation code evidence is missing or malformed"]
    by_path = {
        str(item.get("path")): item
        for item in evidence
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    failures: list[str] = []
    if set(by_path) != set(CODE_RELATIVE_PATHS):
        failures.append("generation code evidence does not contain the exact code set")
    for relative in CODE_RELATIVE_PATHS:
        item = by_path.get(relative)
        if not isinstance(item, dict):
            continue
        try:
            current = _artifact(
                _repo_file(repo, relative, label=f"generation code {relative}"),
                relative_path=relative,
                label=relative,
            )
        except SAM3DGenerationPreflightError as exc:
            failures.append(str(exc))
            continue
        if item.get("sha256") != current["sha256"] or item.get("size_bytes") != current["size_bytes"]:
            failures.append(f"generation code changed: {relative}")
    return failures


def _python_source_tree(
    path: Path,
    *,
    role: str,
    recorded_path: str,
    import_name: str,
    import_origin: Path,
    recorded_import_origin: str,
    extra_relative_files: Sequence[str] = (),
) -> dict[str, Any]:
    root = _real_directory(path, label=f"{role} source tree")
    resolved_import_origin = _regular_file(
        import_origin, label=f"{role} import origin", allow_empty=True
    )
    if not _inside(root, resolved_import_origin):
        raise SAM3DGenerationPreflightError(
            f"{role} import origin escapes source tree: {resolved_import_origin}"
        )
    selected: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise SAM3DGenerationPreflightError(
                f"cannot scan {role} source tree: {directory}: {exc}"
            ) from exc
        for entry in entries:
            candidate = Path(entry.path)
            if entry.is_symlink() or _is_link_like(candidate):
                raise SAM3DGenerationPreflightError(
                    f"{role} source tree contains a link: {candidate}"
                )
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                # Traverse ignored cache directories too so a nested link or special
                # file cannot be hidden there, but never bind generated bytecode.
                pending.append(candidate)
            elif stat.S_ISREG(metadata.st_mode):
                relative = candidate.relative_to(root).as_posix()
                relative_parts = Path(relative).parts
                if (
                    "__pycache__" not in relative_parts
                    and candidate.suffix.lower() not in {".pyc", ".pyo"}
                ):
                    selected.append(candidate.resolve(strict=True))
            else:
                raise SAM3DGenerationPreflightError(
                    f"{role} source tree contains a special file: {candidate}"
                )
    if not selected:
        raise SAM3DGenerationPreflightError(
            f"{role} source tree has no selected regular package files"
        )
    selected_relatives = {
        source.relative_to(root).as_posix() for source in selected
    }
    missing_extras = sorted(set(extra_relative_files) - selected_relatives)
    if missing_extras:
        raise SAM3DGenerationPreflightError(
            f"{role} source tree lacks required runtime resources: {missing_extras}"
        )
    import_origin_relative = resolved_import_origin.relative_to(root).as_posix()
    if import_origin_relative not in selected_relatives:
        raise SAM3DGenerationPreflightError(
            f"{role} import origin is excluded from its package inventory"
        )
    digest = hashlib.sha256()
    total_bytes = 0
    for source in sorted(selected, key=lambda item: item.relative_to(root).as_posix()):
        relative = source.relative_to(root).as_posix()
        size = source.stat().st_size
        file_hash = _sha256_file(source)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        total_bytes += size
    return {
        "role": role,
        "path": recorded_path,
        "resolved_path": str(root),
        "kind": "package_source_tree",
        "import_name": import_name,
        "import_origin_path": recorded_import_origin,
        "resolved_import_origin_path": str(resolved_import_origin),
        "import_origin_size_bytes": resolved_import_origin.stat().st_size,
        "import_origin_sha256": _sha256_file(resolved_import_origin),
        "sha256": digest.hexdigest(),
        "file_count": len(selected),
        "total_bytes": total_bytes,
        "selection": {
            "regular_package_files": (
                "all regular files excluding __pycache__ contents and *.pyc/*.pyo"
            ),
            "required_runtime_resources": sorted(extra_relative_files),
            "links_and_special_files_forbidden": True,
        },
    }


def _package_import_spec(
    *,
    root: Path,
    module_name: str,
    recorded_path: str,
    module_spec: Any,
    label: str,
) -> dict[str, Any]:
    locations = list(module_spec.submodule_search_locations or []) if module_spec else []
    resolved_locations: list[Path] = []
    for location in locations:
        try:
            resolved_locations.append(Path(location).resolve(strict=True))
        except OSError as exc:
            raise SAM3DGenerationPreflightError(
                f"cannot resolve {label} import location {location}: {exc}"
            ) from exc
    if resolved_locations != [root]:
        raise SAM3DGenerationPreflightError(
            f"{label} import resolves outside the bound package root: "
            f"expected={[str(root)]}, actual={[str(item) for item in resolved_locations]}"
        )
    if not isinstance(getattr(module_spec, "origin", None), str):
        raise SAM3DGenerationPreflightError(
            f"{label} import has no regular package origin"
        )
    origin = _regular_file(
        Path(module_spec.origin), label=f"{label} import origin", allow_empty=True
    )
    if not _inside(root, origin):
        raise SAM3DGenerationPreflightError(
            f"{label} import origin escapes bound package root: {origin}"
        )
    origin_relative = origin.relative_to(root)
    recorded_origin = (Path(recorded_path) / origin_relative).as_posix()
    return {
        "import_name": module_name,
        "import_origin": origin,
        "recorded_import_origin": recorded_origin,
    }


def _discover_source_tree_specs(repo: Path) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for role, relative in REPOSITORY_SOURCE_TREES:
        logical = (repo / relative).absolute()
        try:
            root = logical.resolve(strict=True)
        except OSError as exc:
            raise SAM3DGenerationPreflightError(
                f"cannot resolve intentional external source root {logical}: {exc}"
            ) from exc
        root = _real_directory(root, label=f"{role} resolved source tree")
        module_name = SOURCE_TREE_MODULES[role]
        module_spec = importlib.machinery.PathFinder.find_spec(
            module_name, [str(root.parent)]
        )
        specs.append(
            {
                "role": role,
                "path": root,
                "recorded_path": relative,
                "extra_relative_files": SOURCE_TREE_EXTRA_FILES.get(role, ()),
                **_package_import_spec(
                    root=root,
                    module_name=module_name,
                    recorded_path=relative,
                    module_spec=module_spec,
                    label=role,
                ),
            }
        )
    module_spec = importlib.util.find_spec("moge")
    locations = (
        list(module_spec.submodule_search_locations or [])
        if module_spec is not None
        else []
    )
    if len(locations) != 1:
        raise SAM3DGenerationPreflightError(
            f"expected one installed MoGe package source root, found {locations}"
        )
    moge_root = _real_directory(
        Path(locations[0]), label="installed MoGe Python source tree"
    )
    specs.append(
        {
            "role": MOGE_SOURCE_ROLE,
            "path": moge_root,
            "recorded_path": str(moge_root),
            "extra_relative_files": (),
            **_package_import_spec(
                root=moge_root,
                module_name="moge",
                recorded_path=str(moge_root),
                module_spec=module_spec,
                label=MOGE_SOURCE_ROLE,
            ),
        }
    )
    return specs


def _build_source_tree_evidence(
    specs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    roles = [str(spec.get("role")) for spec in specs]
    if len(roles) != len(set(roles)):
        raise SAM3DGenerationPreflightError("source-tree specifications have duplicate roles")
    return [
        _python_source_tree(
            Path(spec["path"]),
            role=str(spec["role"]),
            recorded_path=str(spec["recorded_path"]),
            import_name=str(spec["import_name"]),
            import_origin=Path(spec["import_origin"]),
            recorded_import_origin=str(spec["recorded_import_origin"]),
            extra_relative_files=tuple(spec.get("extra_relative_files", ())),
        )
        for spec in sorted(specs, key=lambda item: str(item["role"]))
    ]


def _verify_source_tree_evidence(
    repo: Path,
    evidence: Any,
    *,
    source_tree_specs: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    if not isinstance(evidence, list):
        return ["SAM3D executable source-tree evidence is missing or malformed"]
    roles = [item.get("role") for item in evidence if isinstance(item, dict)]
    if len(roles) != len(evidence) or len(roles) != len(set(roles)):
        return ["SAM3D executable source-tree evidence has duplicate/malformed roles"]
    try:
        specs = (
            list(source_tree_specs)
            if source_tree_specs is not None
            else _discover_source_tree_specs(repo)
        )
        current = _build_source_tree_evidence(specs)
    except Exception as exc:
        return [f"cannot rediscover SAM3D executable source trees: {exc}"]
    if evidence != current:
        return ["SAM3D executable source-tree evidence is stale or redirected"]
    return []


def _build_runtime_identity() -> dict[str, Any]:
    distributions: list[dict[str, Any]] = []
    for (
        requested_name,
        module_name,
        extension_globs,
        resource_globs,
    ) in RUNTIME_COMPONENTS:
        try:
            distribution = importlib.metadata.distribution(requested_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise SAM3DGenerationPreflightError(
                f"required SAM3D runtime distribution is missing: {requested_name}"
            ) from exc
        record_candidates = [
            entry
            for entry in (distribution.files or ())
            if str(entry).endswith(".dist-info/RECORD")
        ]
        if len(record_candidates) != 1:
            raise SAM3DGenerationPreflightError(
                f"{requested_name} has no unique installed RECORD identity"
            )
        record = _regular_file(
            Path(distribution.locate_file(record_candidates[0])),
            label=f"{requested_name} distribution RECORD",
        )
        module_spec = importlib.util.find_spec(module_name)
        if module_spec is None or not isinstance(module_spec.origin, str):
            raise SAM3DGenerationPreflightError(
                f"required SAM3D runtime module has no import origin: {module_name}"
            )
        module_origin = _regular_file(
            Path(module_spec.origin),
            label=f"{module_name} import origin",
            allow_empty=True,
        )
        distribution_files = list(distribution.files or ())

        def bind_distribution_files(
            patterns: Sequence[str], *, kind: str, allow_empty: bool
        ) -> list[dict[str, Any]]:
            bound: dict[str, dict[str, Any]] = {}
            for pattern in patterns:
                matching = sorted(
                    (
                        entry
                        for entry in distribution_files
                        if fnmatch.fnmatchcase(
                            str(entry).replace("\\", "/"), pattern
                        )
                    ),
                    key=str,
                )
                if not matching:
                    raise SAM3DGenerationPreflightError(
                        f"{requested_name} lacks required {kind} matching {pattern}"
                    )
                for entry in matching:
                    relative = str(entry).replace("\\", "/")
                    runtime_file = _regular_file(
                        Path(distribution.locate_file(entry)),
                        label=f"{requested_name} {kind} {entry}",
                        allow_empty=allow_empty,
                    )
                    bound[relative] = {
                        "distribution_relative_path": relative,
                        "resolved_path": str(runtime_file),
                        "size_bytes": runtime_file.stat().st_size,
                        "sha256": _sha256_file(runtime_file),
                    }
            return [bound[key] for key in sorted(bound)]

        extension_files = bind_distribution_files(
            extension_globs, kind="core extension", allow_empty=False
        )
        resource_files = bind_distribution_files(
            resource_globs, kind="runtime package resource", allow_empty=True
        )
        distributions.append(
            {
                "requested_name": requested_name,
                "module_name": module_name,
                "canonical_name": str(distribution.metadata["Name"]),
                "version": distribution.version,
                "record_path": str(record),
                "record_size_bytes": record.stat().st_size,
                "record_sha256": _sha256_file(record),
                "module_origin": str(module_origin),
                "module_origin_size_bytes": module_origin.stat().st_size,
                "module_origin_sha256": _sha256_file(module_origin),
                "required_extension_globs": list(extension_globs),
                "core_extension_files": extension_files,
                "required_resource_globs": list(resource_globs),
                "runtime_resource_files": resource_files,
            }
        )
    return {
        "python_implementation": sys.implementation.name,
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "distribution_count": len(distributions),
        "distributions": distributions,
        "scope_note": (
            "Distribution versions, installed RECORD manifests, import origins, direct "
            "core extensions, and runtime resources bind the package/runtime boundary; "
            "full executable SAM3, SAM-3D Objects, and MoGe package inventories plus "
            "model/config/cache artifacts and the generated GLB are hashed separately."
        ),
    }


def _verify_runtime_identity(saved: Any, current: Mapping[str, Any]) -> list[str]:
    if not isinstance(saved, dict):
        return ["SAM3D runtime distribution identity is missing or malformed"]
    if saved != current:
        return ["SAM3D runtime distribution identity is stale"]
    return []


def _set_offline_environment() -> dict[str, str]:
    for variable in OFFLINE_VARIABLES:
        os.environ[variable] = "1"
    return {variable: os.environ[variable] for variable in OFFLINE_VARIABLES}


def _production_generate(
    image_path: Path,
    output_path: Path,
    debug_folder: Path,
    sam3_checkpoint: Path,
    pipeline_config: Path,
) -> None:
    # This import is intentionally confined to the non-verify path.  It triggers
    # CUDA/SAM3D setup in upstream SceneSmith.
    from scenesmith.agent_utils.geometry_generation_server.geometry_generation import (
        generate_geometry_from_image,
    )

    generate_geometry_from_image(
        image_path=image_path,
        output_path=output_path,
        debug_folder=debug_folder,
        use_pipeline_caching=False,
        backend="sam3d",
        sam3d_config={
            "sam3_checkpoint": sam3_checkpoint,
            "sam3d_checkpoint": pipeline_config,
            "mode": MODE,
            "object_description": None,
            "threshold": THRESHOLD,
        },
    )


def _mesh_stats(path: Path) -> dict[str, Any]:
    regular = _regular_file(path, label="SAM3D output GLB")
    try:
        import numpy as np
        import trimesh

        loaded = trimesh.load(regular, force="scene", process=False)
        geometries = list(loaded.geometry.values())
    except Exception as exc:
        raise SAM3DGenerationPreflightError(
            f"cannot load SAM3D output GLB with trimesh: {type(exc).__name__}: {exc}"
        ) from exc
    if not geometries:
        raise SAM3DGenerationPreflightError("SAM3D output GLB contains no mesh geometry")
    vertex_count = sum(len(mesh.vertices) for mesh in geometries)
    face_count = sum(len(mesh.faces) for mesh in geometries)
    if vertex_count < 1 or face_count < 1:
        raise SAM3DGenerationPreflightError(
            "SAM3D output GLB has no vertices or faces"
        )
    bounds = loaded.bounds
    if bounds is None or tuple(bounds.shape) != (2, 3) or not bool(np.isfinite(bounds).all()):
        raise SAM3DGenerationPreflightError("SAM3D output GLB has invalid bounds")
    return {
        "loader": "trimesh.load(force='scene', process=False)",
        "geometry_count": len(geometries),
        "vertex_count": vertex_count,
        "face_count": face_count,
        "bounds": [[float(value) for value in row] for row in bounds.tolist()],
    }


def _image_stats_and_semantics(
    input_path: Path, mask_path: Path, masked_path: Path
) -> dict[str, Any]:
    try:
        from PIL import Image

        with Image.open(input_path) as opened:
            if opened.format != "PNG":
                raise SAM3DGenerationPreflightError("canonical SAM3D input is not PNG")
            source = opened.convert("RGB")
            source.load()
        with Image.open(mask_path) as opened:
            if opened.format != "PNG" or opened.mode != "L":
                raise SAM3DGenerationPreflightError(
                    "SAM3D mask must be an exact single-channel PNG"
                )
            mask = opened.copy()
            mask.load()
        with Image.open(masked_path) as opened:
            if opened.format != "PNG":
                raise SAM3DGenerationPreflightError("SAM3D masked image is not PNG")
            masked = opened.convert("RGB")
            masked.load()
    except SAM3DGenerationPreflightError:
        raise
    except Exception as exc:
        raise SAM3DGenerationPreflightError(
            f"cannot load SAM3D image artifact: {type(exc).__name__}: {exc}"
        ) from exc
    if source.size != (CANONICAL_INPUT_WIDTH, CANONICAL_INPUT_HEIGHT):
        raise SAM3DGenerationPreflightError(
            f"canonical input dimensions changed: {source.size}"
        )
    if mask.size != source.size or masked.size != source.size:
        raise SAM3DGenerationPreflightError("SAM3D mask/masked dimensions differ from input")
    mask_bytes = mask.tobytes()
    source_bytes = source.tobytes()
    masked_bytes = masked.tobytes()
    mask_values = sorted(set(mask_bytes))
    if mask_values != [0, 255]:
        raise SAM3DGenerationPreflightError(
            f"SAM3D mask values are not exactly binary {{0,255}}: {mask_values}"
        )
    foreground = 0
    background = 0
    min_x = source.width
    min_y = source.height
    max_x = -1
    max_y = -1
    for index, mask_value in enumerate(mask_bytes):
        offset = index * 3
        source_pixel = source_bytes[offset : offset + 3]
        masked_pixel = masked_bytes[offset : offset + 3]
        if mask_value == 0:
            background += 1
            if masked_pixel != b"\0\0\0":
                raise SAM3DGenerationPreflightError(
                    "masked image contains nonblack background pixels"
                )
        elif mask_value == 255:
            foreground += 1
            x = index % source.width
            y = index // source.width
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            if masked_pixel != source_pixel:
                raise SAM3DGenerationPreflightError(
                    "masked image changed a foreground input pixel"
                )
    total = source.width * source.height
    fraction = foreground / total
    if not (0.01 <= fraction <= 0.95):
        raise SAM3DGenerationPreflightError(
            f"SAM3D foreground fraction is implausible for the canonical shelf: {fraction:.6f}"
        )
    bbox_width = max_x - min_x + 1
    bbox_height = max_y - min_y + 1
    if bbox_width < source.width * 0.05 or bbox_height < source.height * 0.05:
        raise SAM3DGenerationPreflightError(
            "SAM3D foreground bbox is implausibly small for the canonical shelf"
        )
    return {
        "width": source.width,
        "height": source.height,
        "mask_values": mask_values,
        "foreground_pixel_count": foreground,
        "background_pixel_count": background,
        "foreground_fraction": round(fraction, 8),
        "foreground_bbox_xyxy": [min_x, min_y, max_x, max_y],
        "masked_image_matches_mask": True,
    }


def _outputs(
    artifact_dir: Path,
    input_path: Path,
    *,
    mesh_inspector: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    glb = artifact_dir / GLB_NAME
    mask = artifact_dir / MASK_NAME
    masked = artifact_dir / MASKED_IMAGE_NAME
    inspector = mesh_inspector or _mesh_stats
    mesh = dict(inspector(glb))
    for key in ("geometry_count", "vertex_count", "face_count"):
        value = mesh.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise SAM3DGenerationPreflightError(f"invalid GLB {key}: {value!r}")
    images = _image_stats_and_semantics(input_path, mask, masked)
    return {
        "glb": {**_artifact(glb, relative_path=GLB_NAME, label="SAM3D output GLB"), "mesh": mesh},
        "mask": _artifact(mask, relative_path=MASK_NAME, label="SAM3D mask"),
        "masked_image": _artifact(
            masked, relative_path=MASKED_IMAGE_NAME, label="SAM3D masked image"
        ),
        "image_validation": images,
    }


def _without_attestation(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "attestation"}


def _attestation(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "algorithm": "sha256",
        "scope": ATTESTATION_SCOPE,
        "sha256": hashlib.sha256(_canonical_json(_without_attestation(result))).hexdigest(),
    }


def verify_bound_receipt(
    result: Any,
    *,
    input_path: Path,
    artifact_dir: Path,
    expected_input_sha256: str = CANONICAL_INPUT_SHA256,
    mesh_inspector: Callable[[Path], Mapping[str, Any]] | None = None,
) -> list[str]:
    """Validate the portable receipt/output graph without model inference.

    This is used by benchmark/final-acceptance consumers after the runner has
    already executed full ``--verify-only``.  It intentionally does not require
    the model cache to be present in a transferred package, but it does rehash
    the canonical input and all three copied generation outputs.
    """

    if not isinstance(result, dict):
        return ["SAM3D generation receipt is missing or malformed"]
    failures: list[str] = []
    try:
        artifact_dir = _real_directory(
            artifact_dir, label="SAM3D generation artifact directory"
        )
        _verify_exact_artifact_inventory(artifact_dir)
    except Exception as exc:
        failures.append(str(exc))
    if result.get("schema_id") != SCHEMA_ID or result.get("schema_version") != SCHEMA_VERSION:
        failures.append("SAM3D generation receipt schema is unsupported")
    if result.get("status") != "pass":
        failures.append("SAM3D generation receipt is not passing")
    if (
        result.get("offline") is not True
        or result.get("paid_api_calls") != 0
        or result.get("network_access_required") is not False
    ):
        failures.append("SAM3D generation receipt lacks no-API offline proof")
    if result.get("production_entrypoint") != (
        "scenesmith.agent_utils.geometry_generation_server.geometry_generation."
        "generate_geometry_from_image"
    ):
        failures.append("SAM3D generation receipt names a nonproduction entrypoint")
    if result.get("generation_parameters") != {
        "backend": "sam3d",
        "mode": MODE,
        "object_description": None,
        "threshold": THRESHOLD,
        "use_pipeline_caching": False,
    }:
        failures.append("SAM3D generation parameters are not canonical")
    environment = result.get("offline_environment")
    if not isinstance(environment, dict) or any(
        environment.get(name) != "1" for name in OFFLINE_VARIABLES
    ):
        failures.append("SAM3D generation offline environment is incomplete")

    try:
        input_file = _regular_file(input_path, label="canonical SAM3D input")
        input_hash = _sha256_file(input_file)
    except Exception as exc:
        failures.append(f"cannot rehash canonical SAM3D input: {exc}")
        input_file = None
        input_hash = None
    recorded_input = result.get("input")
    if not isinstance(recorded_input, dict) or (
        input_hash != expected_input_sha256
        or recorded_input.get("repository_relative_path")
        != CANONICAL_INPUT_RELATIVE.as_posix()
        or recorded_input.get("sha256") != input_hash
        or isinstance(recorded_input.get("size_bytes"), bool)
        or not isinstance(recorded_input.get("size_bytes"), int)
        or recorded_input["size_bytes"] < 1
        or (input_file is not None and recorded_input.get("size_bytes") != input_file.stat().st_size)
        or recorded_input.get("width") != CANONICAL_INPUT_WIDTH
        or recorded_input.get("height") != CANONICAL_INPUT_HEIGHT
    ):
        failures.append("SAM3D canonical input evidence is stale")

    code = result.get("code_artifacts")
    if (
        not isinstance(code, list)
        or len(code) != len(CODE_RELATIVE_PATHS)
        or any(not isinstance(item, dict) for item in code)
        or any(not isinstance(item.get("path"), str) for item in code)
        or {item.get("path") for item in code} != set(CODE_RELATIVE_PATHS)
    ):
        failures.append("SAM3D generation code evidence is malformed")
    elif any(
        not _is_sha256(item.get("sha256"))
        or isinstance(item.get("size_bytes"), bool)
        or not isinstance(item.get("size_bytes"), int)
        or item["size_bytes"] < 1
        for item in code
    ):
        failures.append("SAM3D generation code hashes are malformed")

    source_trees = result.get("executable_source_trees")
    required_source_roles = {
        role for role, _relative in REPOSITORY_SOURCE_TREES
    } | {MOGE_SOURCE_ROLE}
    if (
        not isinstance(source_trees, list)
        or len(source_trees) != len(required_source_roles)
        or any(not isinstance(item, dict) for item in source_trees)
        or any(not isinstance(item.get("role"), str) for item in source_trees)
        or {item.get("role") for item in source_trees} != required_source_roles
        or any(
            item.get("kind") != "package_source_tree"
            or not isinstance(item.get("path"), str)
            or not item["path"]
            or not isinstance(item.get("resolved_path"), str)
            or not Path(item["resolved_path"]).is_absolute()
            or not isinstance(item.get("import_name"), str)
            or not item["import_name"]
            or item.get("import_name")
            != SOURCE_TREE_MODULES.get(str(item.get("role")))
            or not isinstance(item.get("import_origin_path"), str)
            or not item["import_origin_path"]
            or not isinstance(item.get("resolved_import_origin_path"), str)
            or not Path(item["resolved_import_origin_path"]).is_absolute()
            or not _is_sha256(item.get("import_origin_sha256"))
            or isinstance(item.get("import_origin_size_bytes"), bool)
            or not isinstance(item.get("import_origin_size_bytes"), int)
            or item["import_origin_size_bytes"] < 0
            or item.get("selection") != {
                "regular_package_files": (
                    "all regular files excluding __pycache__ contents and *.pyc/*.pyo"
                ),
                "required_runtime_resources": sorted(
                    SOURCE_TREE_EXTRA_FILES.get(str(item.get("role")), ())
                ),
                "links_and_special_files_forbidden": True,
            }
            or not _is_sha256(item.get("sha256"))
            or isinstance(item.get("file_count"), bool)
            or not isinstance(item.get("file_count"), int)
            or item["file_count"] < 1
            or isinstance(item.get("total_bytes"), bool)
            or not isinstance(item.get("total_bytes"), int)
            or item["total_bytes"] < 1
            for item in source_trees
        )
    ):
        failures.append("SAM3D executable source-tree evidence is malformed")

    runtime_identity = result.get("runtime_identity")
    runtime_distributions = (
        runtime_identity.get("distributions")
        if isinstance(runtime_identity, dict)
        else None
    )
    if (
        not isinstance(runtime_identity, dict)
        or not isinstance(runtime_identity.get("python_implementation"), str)
        or not isinstance(runtime_identity.get("python_version"), str)
        or not isinstance(runtime_identity.get("scope_note"), str)
        or not isinstance(runtime_distributions, list)
        or len(runtime_distributions) != len(RUNTIME_DISTRIBUTIONS)
        or runtime_identity.get("distribution_count") != len(runtime_distributions)
        or any(not isinstance(item, dict) for item in runtime_distributions)
        or [item.get("requested_name") for item in runtime_distributions]
        != list(RUNTIME_DISTRIBUTIONS)
        or any(
            not isinstance(item.get("version"), str)
            or not item["version"]
            or not isinstance(item.get("canonical_name"), str)
            or not item["canonical_name"]
            or not isinstance(item.get("record_path"), str)
            or not Path(item["record_path"]).is_absolute()
            or not isinstance(item.get("module_name"), str)
            or not isinstance(item.get("module_origin"), str)
            or not Path(item["module_origin"]).is_absolute()
            or not _is_sha256(item.get("module_origin_sha256"))
            or isinstance(item.get("module_origin_size_bytes"), bool)
            or not isinstance(item.get("module_origin_size_bytes"), int)
            or item["module_origin_size_bytes"] < 1
            or not _is_sha256(item.get("record_sha256"))
            or isinstance(item.get("record_size_bytes"), bool)
            or not isinstance(item.get("record_size_bytes"), int)
            or item["record_size_bytes"] < 1
            for item in runtime_distributions
        )
        or any(
            item.get("module_name") != module_name
            or item.get("required_extension_globs") != list(extension_globs)
            or not isinstance(item.get("core_extension_files"), list)
            or any(
                not isinstance(extension, dict)
                or not isinstance(extension.get("distribution_relative_path"), str)
                or not isinstance(extension.get("resolved_path"), str)
                or not Path(extension["resolved_path"]).is_absolute()
                or not _is_sha256(extension.get("sha256"))
                or isinstance(extension.get("size_bytes"), bool)
                or not isinstance(extension.get("size_bytes"), int)
                or extension["size_bytes"] < 1
                for extension in item.get("core_extension_files", [])
            )
            or (bool(extension_globs) and not item.get("core_extension_files"))
            or item.get("required_resource_globs") != list(resource_globs)
            or not isinstance(item.get("runtime_resource_files"), list)
            or any(
                not isinstance(resource, dict)
                or not isinstance(resource.get("distribution_relative_path"), str)
                or not isinstance(resource.get("resolved_path"), str)
                or not Path(resource["resolved_path"]).is_absolute()
                or not _is_sha256(resource.get("sha256"))
                or isinstance(resource.get("size_bytes"), bool)
                or not isinstance(resource.get("size_bytes"), int)
                or resource["size_bytes"] < 0
                for resource in item.get("runtime_resource_files", [])
            )
            or (bool(resource_globs) and not item.get("runtime_resource_files"))
            for item, (
                _distribution,
                module_name,
                extension_globs,
                resource_globs,
            ) in zip(
                runtime_distributions, RUNTIME_COMPONENTS
            )
        )
    ):
        failures.append("SAM3D runtime distribution identity is malformed")

    models = result.get("model_artifacts")
    model_entries = models.get("artifacts") if isinstance(models, dict) else None
    if (
        not isinstance(models, dict)
        or models.get("algorithm") != "sha256"
        or not isinstance(model_entries, list)
        or not model_entries
        or models.get("artifact_count") != len(model_entries)
    ):
        failures.append("SAM3D generation model evidence is malformed")
    else:
        model_roles = [
            item.get("role") for item in model_entries if isinstance(item, dict)
        ]
        if (
            len(model_roles) != len(model_entries)
            or any(not isinstance(role, str) for role in model_roles)
            or len(model_roles) != len(set(map(str, model_roles)))
        ):
            failures.append("SAM3D generation model evidence has duplicate roles")
        if not {"sam3_checkpoint", "pipeline_config"}.issubset(model_roles):
            failures.append("SAM3D generation model evidence lacks canonical roots")
        if any(
            not isinstance(item, dict)
            or not _is_sha256(item.get("sha256"))
            or item.get("kind") not in {"file", "python_source_tree"}
            for item in model_entries
        ):
            failures.append("SAM3D generation model artifact hashes are malformed")

    recorded_outputs = result.get("outputs")
    recorded_glb = (
        recorded_outputs.get("glb") if isinstance(recorded_outputs, dict) else None
    )
    recorded_mesh = recorded_glb.get("mesh") if isinstance(recorded_glb, dict) else None
    if not isinstance(recorded_mesh, dict) or any(
        isinstance(recorded_mesh.get(key), bool)
        or not isinstance(recorded_mesh.get(key), int)
        or recorded_mesh[key] < 1
        for key in ("geometry_count", "vertex_count", "face_count")
    ):
        failures.append("SAM3D GLB mesh-count evidence is invalid")

    if input_file is not None:
        try:
            current_outputs = _outputs(
                artifact_dir,
                input_file,
                mesh_inspector=mesh_inspector,
            )
        except Exception as exc:
            failures.append(f"cannot independently recompute SAM3D outputs: {exc}")
        else:
            if result.get("outputs") != current_outputs:
                failures.append("SAM3D generation output evidence is stale")
    validation = result.get("validation")
    if validation != {
        "status": "pass",
        "trimesh_reloaded": True,
        "critical_issues": [],
    }:
        failures.append("SAM3D generation validation proof is missing")
    if result.get("attestation") != _attestation(result):
        failures.append("SAM3D generation receipt attestation is invalid")
    return list(dict.fromkeys(failures))


def _write_json(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_generation(
    *,
    repo_dir: Path,
    preflight_dir: Path,
    sam3_checkpoint: Path,
    pipeline_config: Path,
    expected_input_sha256: str = CANONICAL_INPUT_SHA256,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    source_tree_specs: Sequence[Mapping[str, Any]] | None = None,
    runtime_identity: Mapping[str, Any] | None = None,
    generator: Callable[[Path, Path, Path, Path, Path], None] = _production_generate,
    mesh_inspector: Callable[[Path], Mapping[str, Any]] = _mesh_stats,
) -> dict[str, Any]:
    repo = repo_dir.expanduser().resolve(strict=True)
    input_path = _repo_file(repo, CANONICAL_INPUT_RELATIVE, label="canonical SAM3D input")
    if _sha256_file(input_path) != expected_input_sha256:
        raise SAM3DGenerationPreflightError("canonical SAM3D input SHA-256 changed")
    sam3_checkpoint = _regular_file(sam3_checkpoint.expanduser().absolute(), label="SAM3 checkpoint")
    pipeline_config = _regular_file(pipeline_config.expanduser().absolute(), label="SAM3D pipeline config")
    paths = canonical_paths(preflight_dir)
    target = paths["artifact_dir"]
    if target.exists() or target.is_symlink():
        raise SAM3DGenerationPreflightError(
            f"canonical generation proof already exists; use --verify-only: {target}"
        )
    paths["preflight_dir"].mkdir(parents=True, exist_ok=True)
    _real_directory(paths["preflight_dir"], label="SAM3D preflight directory")
    _reject_transaction_orphans(paths["preflight_dir"])
    offline_environment = _set_offline_environment()
    specs = list(model_specs) if model_specs is not None else model_preflight.discover_artifacts(
        sam3_checkpoint, pipeline_config
    )
    model_evidence = model_preflight.build_artifact_manifest(specs)
    code_evidence = _code_evidence(repo)
    resolved_source_specs = (
        list(source_tree_specs)
        if source_tree_specs is not None
        else _discover_source_tree_specs(repo)
    )
    source_tree_evidence = _build_source_tree_evidence(resolved_source_specs)
    resolved_runtime_identity = dict(runtime_identity or _build_runtime_identity())
    started = time.monotonic()
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{ARTIFACT_DIRECTORY_NAME}.",
            suffix=".tmp",
            dir=paths["preflight_dir"],
        )
    )
    try:
        generator(input_path, stage / GLB_NAME, stage, sam3_checkpoint, pipeline_config)
        output_evidence = _outputs(stage, input_path, mesh_inspector=mesh_inspector)
        model_failures = model_preflight.verify_artifact_manifest(model_evidence, specs)
        if model_failures:
            raise SAM3DGenerationPreflightError(
                "SAM3D model artifacts changed during generation: " + "; ".join(model_failures)
            )
        result: dict[str, Any] = {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "status": "pass",
            "offline": True,
            "offline_environment": offline_environment,
            "paid_api_calls": 0,
            "network_access_required": False,
            "production_entrypoint": (
                "scenesmith.agent_utils.geometry_generation_server.geometry_generation."
                "generate_geometry_from_image"
            ),
            "generation_parameters": {
                "backend": "sam3d",
                "mode": MODE,
                "object_description": None,
                "threshold": THRESHOLD,
                "use_pipeline_caching": False,
            },
            "input": {
                "repository_relative_path": CANONICAL_INPUT_RELATIVE.as_posix(),
                "size_bytes": input_path.stat().st_size,
                "sha256": _sha256_file(input_path),
                "width": CANONICAL_INPUT_WIDTH,
                "height": CANONICAL_INPUT_HEIGHT,
            },
            "sam3_checkpoint": str(sam3_checkpoint),
            "pipeline_config": str(pipeline_config),
            "model_artifacts": model_evidence,
            "code_artifacts": code_evidence,
            "executable_source_trees": source_tree_evidence,
            "runtime_identity": resolved_runtime_identity,
            "outputs": output_evidence,
            "validation": {
                "status": "pass",
                "trimesh_reloaded": True,
                "critical_issues": [],
            },
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        result["attestation"] = _attestation(result)
        _write_json(stage / RECEIPT_NAME, result)
        _verify_exact_artifact_inventory(stage)
        os.replace(stage, target)
        verified = verify_output(
            repo_dir=repo,
            preflight_dir=paths["preflight_dir"],
            sam3_checkpoint=sam3_checkpoint,
            pipeline_config=pipeline_config,
            expected_input_sha256=expected_input_sha256,
            mesh_inspector=mesh_inspector,
            source_tree_specs=resolved_source_specs,
            runtime_identity=resolved_runtime_identity,
        )
        return verified
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def verify_saved_result(
    result: Any,
    *,
    repo_dir: Path,
    artifact_dir: Path,
    sam3_checkpoint: Path,
    pipeline_config: Path,
    expected_input_sha256: str = CANONICAL_INPUT_SHA256,
    mesh_inspector: Callable[[Path], Mapping[str, Any]] | None = None,
    source_tree_specs: Sequence[Mapping[str, Any]] | None = None,
    runtime_identity: Mapping[str, Any] | None = None,
) -> list[str]:
    if not isinstance(result, dict):
        return ["SAM3D generation receipt is missing or malformed"]
    repo = repo_dir.expanduser().resolve()
    try:
        input_path = _repo_file(repo, CANONICAL_INPUT_RELATIVE, label="canonical SAM3D input")
    except Exception as exc:
        return [f"cannot rehash canonical SAM3D input: {exc}"]
    failures = verify_bound_receipt(
        result,
        input_path=input_path,
        artifact_dir=artifact_dir,
        expected_input_sha256=expected_input_sha256,
        mesh_inspector=mesh_inspector,
    )
    failures.extend(_verify_code_evidence(repo, result.get("code_artifacts")))
    failures.extend(
        _verify_source_tree_evidence(
            repo,
            result.get("executable_source_trees"),
            source_tree_specs=source_tree_specs,
        )
    )
    try:
        current_runtime_identity = dict(runtime_identity or _build_runtime_identity())
    except Exception as exc:
        failures.append(f"cannot rediscover SAM3D runtime identity: {exc}")
    else:
        failures.extend(
            _verify_runtime_identity(
                result.get("runtime_identity"), current_runtime_identity
            )
        )
    resolved_sam3 = sam3_checkpoint.expanduser().resolve()
    resolved_pipeline = pipeline_config.expanduser().resolve()
    if result.get("sam3_checkpoint") != str(resolved_sam3):
        failures.append("SAM3 checkpoint path differs from canonical path")
    if result.get("pipeline_config") != str(resolved_pipeline):
        failures.append("SAM3D pipeline config path differs from canonical path")
    try:
        # Independent rediscovery is essential: never let a re-attested receipt
        # choose the cache/source paths that verification subsequently trusts.
        specs = model_preflight.discover_artifacts(resolved_sam3, resolved_pipeline)
    except Exception as exc:
        failures.append(
            f"cannot independently rediscover SAM3D model artifacts: "
            f"{type(exc).__name__}: {exc}"
        )
    else:
        manifest = result.get("model_artifacts")
        entries = manifest.get("artifacts") if isinstance(manifest, dict) else None
        expected_triples = {
            (
                str(spec["role"]),
                str(Path(spec["path"]).resolve()),
                str(spec.get("kind", "file")),
            )
            for spec in specs
        }
        if not isinstance(entries, list) or any(
            not isinstance(item, dict) for item in entries
        ):
            failures.append("SAM3D model artifact inventory is malformed")
        else:
            recorded_triples = [
                (
                    str(item.get("role")),
                    str(item.get("path")),
                    str(item.get("kind")),
                )
                for item in entries
            ]
            if (
                len(recorded_triples) != len(set(recorded_triples))
                or set(recorded_triples) != expected_triples
            ):
                failures.append(
                    "SAM3D model artifact role/path/kind inventory is not exact"
                )
        failures.extend(
            model_preflight.verify_artifact_manifest(manifest, specs)
        )
    return list(dict.fromkeys(failures))


def verify_output(
    *,
    repo_dir: Path,
    preflight_dir: Path,
    sam3_checkpoint: Path,
    pipeline_config: Path,
    expected_input_sha256: str = CANONICAL_INPUT_SHA256,
    mesh_inspector: Callable[[Path], Mapping[str, Any]] | None = None,
    source_tree_specs: Sequence[Mapping[str, Any]] | None = None,
    runtime_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    paths = canonical_paths(preflight_dir)
    preflight_root = _real_directory(
        paths["preflight_dir"], label="SAM3D preflight directory"
    )
    artifact_dir = _real_directory(
        paths["artifact_dir"], label="SAM3D generation artifact directory"
    )
    if artifact_dir.parent != preflight_root:
        raise SAM3DGenerationPreflightError(
            "SAM3D generation artifact directory escapes the preflight directory"
        )
    _reject_transaction_orphans(preflight_root)
    _verify_exact_artifact_inventory(artifact_dir)
    receipt = paths["receipt"]
    try:
        value = json.loads(_regular_file(receipt, label="SAM3D generation receipt").read_text(encoding="utf-8"))
    except Exception as exc:
        raise SAM3DGenerationPreflightError(
            f"cannot read SAM3D generation receipt: {type(exc).__name__}: {exc}"
        ) from exc
    failures = verify_saved_result(
        value,
        repo_dir=repo_dir,
        artifact_dir=artifact_dir,
        sam3_checkpoint=sam3_checkpoint,
        pipeline_config=pipeline_config,
        expected_input_sha256=expected_input_sha256,
        mesh_inspector=mesh_inspector,
        source_tree_specs=source_tree_specs,
        runtime_identity=runtime_identity,
    )
    if failures:
        raise SAM3DGenerationPreflightError("; ".join(failures))
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path("."))
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument(
        "--sam3-checkpoint", type=Path, default=Path("external/checkpoints/sam3.pt")
    )
    parser.add_argument(
        "--pipeline-config", type=Path, default=Path("external/checkpoints/pipeline.yaml")
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Rehash/reload the saved GLB proof without importing SAM3D or using a GPU.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.verify_only:
            result = verify_output(
                repo_dir=args.repo_dir,
                preflight_dir=args.preflight_dir,
                sam3_checkpoint=args.sam3_checkpoint,
                pipeline_config=args.pipeline_config,
            )
        else:
            result = run_generation(
                repo_dir=args.repo_dir,
                preflight_dir=args.preflight_dir,
                sam3_checkpoint=args.sam3_checkpoint,
                pipeline_config=args.pipeline_config,
            )
    except Exception as exc:
        print(f"SAM3D generation preflight failed: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
