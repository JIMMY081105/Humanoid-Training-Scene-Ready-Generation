#!/usr/bin/env python3
"""Prove that ObjectThor retrieval is complete and can run fully offline.

The gate binds the four preprocessed retrieval files, a deterministic sample of
the mesh payloads returned by SceneSmith's production path resolver, and the
exact pinned OpenCLIP checkpoint blob.  ``--verify-only`` is intentionally not
a status-file check: it revalidates and rehashes the dataset and repeats the
cache-only CPU model load and text-embedding smoke test.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import struct
import time

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


RESULT_SCHEMA_VERSION = 1
EVIDENCE_SCHEMA_VERSION = 1
HASH_CHUNK_SIZE = 1024 * 1024
EMBEDDING_DIMENSION = 768
DEFAULT_MINIMUM_ROWS = 50_000
DEFAULT_MAXIMUM_ROWS = 51_000
DEFAULT_SAMPLE_COUNT = 5
OFFLINE_VARIABLES = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "DIFFUSERS_OFFLINE",
)

MODEL_NAME = "ViT-L-14"
PRETRAINED_TAG = "laion2b_s32b_b82k"
MODEL_REPO_ID = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
MODEL_REVISION = "1627032197142fbe2a7cfec626f4ced3ae60d07a"
MODEL_FILENAME = "open_clip_pytorch_model.safetensors"
SMOKE_QUERY = "a wooden school desk"

REQUIRED_INDEX_FILES = (
    "metadata_index.json",
    "clip_embeddings.npy",
    "embedding_index.yaml",
    "object_categories.json",
)

WeightResolver = Callable[[], Mapping[str, Any]]
ModelLoader = Callable[[Mapping[str, Any], str], Any]
MeshPathResolver = Callable[[Path, str], Path]
MeshSmokeLoader = Callable[[Path], Mapping[str, Any]]


def _set_offline_environment() -> dict[str, str]:
    for variable in OFFLINE_VARIABLES:
        os.environ[variable] = "1"
    return {variable: os.environ[variable] for variable in OFFLINE_VARIABLES}


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _artifact_spec(role: str, path: Path) -> dict[str, Any]:
    return {"role": role, "path": _absolute(path)}


def _artifact_evidence(spec: Mapping[str, Any]) -> dict[str, Any]:
    role = str(spec["role"])
    logical_path = _absolute(Path(spec["path"]))
    if not logical_path.is_file():
        raise FileNotFoundError(f"ObjectThor evidence artifact is missing ({role}): {logical_path}")
    if logical_path.stat().st_size < 1:
        raise ValueError(f"ObjectThor evidence artifact is empty ({role}): {logical_path}")
    resolved_path = logical_path.resolve(strict=True)
    return {
        "role": role,
        "path": str(logical_path),
        "resolved_path": str(resolved_path),
        "size_bytes": resolved_path.stat().st_size,
        "sha256": _sha256_file(resolved_path),
    }


def build_artifact_manifest(
    specs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    roles = [str(spec["role"]) for spec in specs]
    duplicates = sorted(role for role in set(roles) if roles.count(role) > 1)
    if duplicates:
        raise ValueError(f"Duplicate ObjectThor evidence roles: {duplicates}")
    artifacts = [_artifact_evidence(spec) for spec in sorted(specs, key=lambda x: str(x["role"]))]
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "algorithm": "sha256",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def _load_embedding_index(path: Path) -> list[str]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for the ObjectThor preflight") from exc
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    values = document.get("uids") if isinstance(document, dict) else document
    if not isinstance(values, list) or not values:
        raise ValueError("embedding_index.yaml must contain a nonempty UID list")
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError("embedding_index.yaml contains an empty or non-string UID")
    if len(values) != len(set(values)):
        raise ValueError("embedding_index.yaml contains duplicate UIDs")
    return values


def _load_json(path: Path, label: str) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc


def _select_sample_uids(
    embedding_index: Sequence[str],
    object_categories: Mapping[str, Sequence[str]],
    sample_count: int,
) -> list[str]:
    if sample_count < 1:
        raise ValueError("sample_count must be at least one")
    selected: list[str] = []

    def add(uid: str) -> None:
        if uid not in selected and len(selected) < sample_count:
            selected.append(uid)

    # Cover the production placement categories before filling deterministic
    # positions across the full index.
    for category in sorted(object_categories):
        category_uids = object_categories[category]
        if category_uids:
            add(str(category_uids[0]))
    if sample_count == 1:
        add(str(embedding_index[0]))
    else:
        for sample_index in range(sample_count):
            position = round(sample_index * (len(embedding_index) - 1) / (sample_count - 1))
            add(str(embedding_index[position]))
    for uid in embedding_index:
        add(str(uid))
        if len(selected) == sample_count:
            break
    return selected


def resolve_production_mesh_path(dataset_root: Path, uid: str) -> Path:
    """Call the same resolver used by the ObjectThor retrieval consumer."""

    try:
        from scenesmith.agent_utils.objaverse_retrieval.data_loader import (
            construct_objaverse_mesh_path,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import SceneSmith's ObjectThor production mesh resolver"
        ) from exc
    resolved = Path(construct_objaverse_mesh_path(dataset_root, uid))
    if not resolved.is_absolute():
        resolved = dataset_root / resolved
    return _absolute(resolved)


def smoke_load_mesh_payload(path: Path) -> dict[str, Any]:
    """Parse enough of a GLB or canonical ObjectThor pickle to reject truncation."""

    path = _absolute(path)
    if path.suffix.lower() == ".glb":
        with path.open("rb") as stream:
            header = stream.read(12)
        if len(header) != 12:
            raise ValueError(f"GLB payload is truncated: {path}")
        magic, version, declared_size = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2:
            raise ValueError(f"GLB payload has an invalid header: {path}")
        if declared_size != path.stat().st_size:
            raise ValueError(
                f"GLB declared length differs from file size: {declared_size} != {path.stat().st_size}"
            )
        return {
            "format": "glb",
            "version": version,
            "declared_size_bytes": declared_size,
        }

    if path.name.endswith(".pkl.gz"):
        with gzip.open(path, "rb") as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict):
            raise ValueError(f"ObjectThor pickle payload is not a mapping: {path}")
        vertices = payload.get("vertices")
        triangles = payload.get("triangles")
        if not isinstance(vertices, list) or len(vertices) < 3:
            raise ValueError(f"ObjectThor payload has no usable vertices: {path}")
        if not isinstance(triangles, list) or len(triangles) < 3 or len(triangles) % 3:
            raise ValueError(f"ObjectThor payload has no usable triangle faces: {path}")
        if not all(
            isinstance(vertex, dict)
            and all(
                isinstance(vertex.get(axis), (int, float))
                and math.isfinite(float(vertex[axis]))
                for axis in ("x", "y", "z")
            )
            for vertex in vertices
        ):
            raise ValueError(f"ObjectThor payload contains invalid vertices: {path}")
        if not all(
            isinstance(index, int) and 0 <= index < len(vertices)
            for index in triangles
        ):
            raise ValueError(f"ObjectThor payload contains invalid triangle indices: {path}")
        return {
            "format": "objathor_pickle_gzip",
            "vertex_count": len(vertices),
            "face_count": len(triangles) // 3,
        }

    raise ValueError(f"Unsupported production ObjectThor mesh payload: {path}")


def inspect_dataset(
    dataset_root: Path,
    preprocessed_path: Path,
    *,
    minimum_rows: int = DEFAULT_MINIMUM_ROWS,
    maximum_rows: int = DEFAULT_MAXIMUM_ROWS,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    mesh_path_resolver: MeshPathResolver = resolve_production_mesh_path,
    mesh_smoke_loader: MeshSmokeLoader = smoke_load_mesh_payload,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate the complete index contract and sampled production payloads."""

    import numpy as np

    dataset_root = _absolute(dataset_root)
    preprocessed_path = _absolute(preprocessed_path)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"ObjectThor dataset root is missing: {dataset_root}")
    if not preprocessed_path.is_dir():
        raise FileNotFoundError(
            f"ObjectThor preprocessed directory is missing: {preprocessed_path}"
        )
    if minimum_rows < 1 or maximum_rows < minimum_rows:
        raise ValueError("ObjectThor row bounds are invalid")

    required_paths = {
        filename: preprocessed_path / filename for filename in REQUIRED_INDEX_FILES
    }
    for filename, path in required_paths.items():
        if not path.is_file() or path.stat().st_size < 1:
            raise FileNotFoundError(f"Required ObjectThor index is missing or empty: {filename}")

    metadata = _load_json(required_paths["metadata_index.json"], "metadata_index.json")
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("metadata_index.json must be a nonempty UID mapping")
    if not all(
        isinstance(uid, str) and uid and isinstance(entry, dict)
        for uid, entry in metadata.items()
    ):
        raise ValueError("metadata_index.json has an invalid UID or metadata entry")

    embedding_index = _load_embedding_index(required_paths["embedding_index.yaml"])
    index_uid_set = set(embedding_index)
    if set(metadata) != index_uid_set:
        raise ValueError(
            "metadata_index.json and embedding_index.yaml do not contain the same UIDs"
        )

    categories = _load_json(required_paths["object_categories.json"], "object_categories.json")
    if not isinstance(categories, dict) or not categories:
        raise ValueError("object_categories.json must be a nonempty category mapping")
    category_uids: list[str] = []
    for category, values in categories.items():
        if not isinstance(category, str) or not category:
            raise ValueError("object_categories.json contains an invalid category name")
        if not isinstance(values, list) or not all(
            isinstance(uid, str) and uid for uid in values
        ):
            raise ValueError(f"ObjectThor category {category!r} is not a UID list")
        category_uids.extend(values)
    if len(category_uids) != len(set(category_uids)):
        raise ValueError("An ObjectThor UID is assigned to more than one category")
    if set(category_uids) != index_uid_set:
        raise ValueError(
            "object_categories.json does not assign every indexed UID exactly once"
        )

    try:
        embeddings = np.load(
            required_paths["clip_embeddings.npy"], mmap_mode="r", allow_pickle=False
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"clip_embeddings.npy cannot be loaded safely: {exc}") from exc
    if embeddings.ndim != 2:
        raise ValueError(f"clip_embeddings.npy must be rank 2, got {embeddings.shape}")
    rows, dimensions = map(int, embeddings.shape)
    if not minimum_rows <= rows <= maximum_rows:
        raise ValueError(
            f"ObjectThor index has {rows} rows; expected {minimum_rows}..{maximum_rows}"
        )
    if dimensions != EMBEDDING_DIMENSION:
        raise ValueError(
            f"ObjectThor embeddings have {dimensions} dimensions, expected {EMBEDDING_DIMENSION}"
        )
    if rows != len(embedding_index):
        raise ValueError(
            f"ObjectThor row/index mismatch: {rows} != {len(embedding_index)}"
        )
    if embeddings.dtype.kind not in "fc":
        raise ValueError(f"ObjectThor embeddings are not floating point: {embeddings.dtype}")
    for start in range(0, rows, 4096):
        chunk = np.asarray(embeddings[start : start + 4096], dtype=np.float32)
        if not np.isfinite(chunk).all():
            raise ValueError(f"ObjectThor embeddings contain non-finite values near row {start}")
        if np.any(np.linalg.norm(chunk, axis=1) <= 1e-12):
            raise ValueError(f"ObjectThor embeddings contain a zero vector near row {start}")

    sample_uids = _select_sample_uids(embedding_index, categories, sample_count)
    samples: list[dict[str, Any]] = []
    artifact_specs = [
        _artifact_spec(f"index:{filename}", path)
        for filename, path in required_paths.items()
    ]
    for uid in sample_uids:
        computed_glb = dataset_root / "assets" / uid / f"{uid}.glb"
        canonical_pickle = dataset_root / "assets" / uid / f"{uid}.pkl.gz"
        try:
            selected_path = _absolute(mesh_path_resolver(dataset_root, uid))
        except Exception as exc:
            raise RuntimeError(
                f"Production ObjectThor mesh resolver failed for {uid}: {type(exc).__name__}: {exc}"
            ) from exc
        if not selected_path.is_file() or selected_path.stat().st_size < 1:
            raise FileNotFoundError(
                f"Production ObjectThor mesh path is missing or empty for {uid}: {selected_path}"
            )
        smoke = dict(mesh_smoke_loader(selected_path))
        sample = {
            "uid": uid,
            "production_mesh_path": str(selected_path),
            "computed_glb_path": str(_absolute(computed_glb)),
            "computed_glb_exists": computed_glb.is_file(),
            "canonical_pickle_path": str(_absolute(canonical_pickle)),
            "canonical_pickle_exists": canonical_pickle.is_file(),
            "smoke_load": smoke,
        }
        samples.append(sample)
        artifact_specs.append(_artifact_spec(f"sample_mesh:{uid}", selected_path))

    summary = {
        "dataset_root": str(dataset_root),
        "preprocessed_path": str(preprocessed_path),
        "row_count": rows,
        "embedding_dimensions": dimensions,
        "embedding_dtype": str(embeddings.dtype),
        "metadata_count": len(metadata),
        "embedding_index_count": len(embedding_index),
        "category_count": len(categories),
        "category_membership_count": len(category_uids),
        "sample_count": len(samples),
        "samples": samples,
    }
    return summary, artifact_specs


def _snapshot_revision(cache_path: Path) -> str | None:
    parts = cache_path.parts
    try:
        position = parts.index("snapshots")
    except ValueError:
        return None
    if position + 1 >= len(parts):
        return None
    return parts[position + 1]


def resolve_openclip_weight_blob() -> dict[str, Any]:
    """Resolve the production-pinned OpenCLIP blob without network access."""

    _set_offline_environment()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to resolve the ObjectThor OpenCLIP cache"
        ) from exc
    cache_path = Path(
        hf_hub_download(
            repo_id=MODEL_REPO_ID,
            repo_type="model",
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    )
    cache_path = _absolute(cache_path)
    if not cache_path.is_file() or cache_path.stat().st_size < 1:
        raise FileNotFoundError(f"Pinned OpenCLIP weight blob is missing: {cache_path}")
    snapshot_revision = _snapshot_revision(cache_path)
    if snapshot_revision is not None and snapshot_revision != MODEL_REVISION:
        raise ValueError(
            "Pinned OpenCLIP cache resolved to the wrong revision: "
            f"{snapshot_revision} != {MODEL_REVISION}"
        )
    try:
        open_clip_version = importlib.metadata.version("open-clip-torch")
    except importlib.metadata.PackageNotFoundError:
        open_clip_version = None
    return {
        "model_name": MODEL_NAME,
        "pretrained_tag": PRETRAINED_TAG,
        "repo_id": MODEL_REPO_ID,
        "requested_revision": MODEL_REVISION,
        "snapshot_revision": snapshot_revision,
        "filename": MODEL_FILENAME,
        "cache_path": str(cache_path),
        "blob_path": str(cache_path.resolve(strict=True)),
        "open_clip_version": open_clip_version,
    }


def load_openclip_text_embedding(weight: Mapping[str, Any], query: str) -> dict[str, Any]:
    """Load the exact local checkpoint on CPU and encode one retrieval query."""

    _set_offline_environment()
    import numpy as np
    import open_clip
    import torch

    cache_path = _absolute(Path(str(weight["cache_path"])))
    if cache_path.suffix != ".safetensors" or not cache_path.is_file():
        raise FileNotFoundError(
            f"Pinned OpenCLIP safetensors snapshot path is missing: {cache_path}"
        )
    blob_path = cache_path.resolve(strict=True)
    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=str(cache_path),
        device="cpu",
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    tokens = tokenizer([query]).to("cpu")
    model.eval()
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    embedding = np.asarray(features.detach().cpu().numpy()[0], dtype=np.float32)
    return {
        "embedding": embedding,
        "device": "cpu",
        "loaded_weight_blob": str(cache_path),
    }


def validate_model_smoke(
    raw_smoke: Any, weight: Mapping[str, Any], query: str
) -> dict[str, Any]:
    import numpy as np

    if isinstance(raw_smoke, Mapping):
        if "embedding" not in raw_smoke:
            raise ValueError("OpenCLIP loader did not return an embedding")
        embedding = np.asarray(raw_smoke["embedding"])
        device = raw_smoke.get("device")
        loaded_weight_blob = raw_smoke.get("loaded_weight_blob")
    else:
        embedding = np.asarray(raw_smoke)
        device = None
        loaded_weight_blob = None
    if embedding.shape != (EMBEDDING_DIMENSION,):
        raise ValueError(
            f"OpenCLIP text embedding has shape {embedding.shape}, expected ({EMBEDDING_DIMENSION},)"
        )
    if not np.isfinite(embedding).all():
        raise ValueError("OpenCLIP text embedding contains non-finite values")
    norm = float(np.linalg.norm(embedding.astype(np.float64)))
    if not math.isfinite(norm) or abs(norm - 1.0) > 1e-4:
        raise ValueError(f"OpenCLIP text embedding is not normalized: norm={norm}")
    expected_blob = Path(str(weight["blob_path"])).resolve(strict=True)
    if loaded_weight_blob is not None and Path(str(loaded_weight_blob)).resolve(strict=True) != expected_blob:
        raise ValueError(
            "OpenCLIP loader used a different weight blob: "
            f"{loaded_weight_blob} != {expected_blob}"
        )
    if device is not None and str(device) != "cpu":
        raise ValueError(f"ObjectThor OpenCLIP smoke did not run on CPU: {device}")
    return {
        "query": query,
        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "embedding_shape": [EMBEDDING_DIMENSION],
        "embedding_dtype": str(embedding.dtype),
        "embedding_norm": norm,
        "finite": True,
        "device": "cpu" if device is None else str(device),
        "loaded_weight_blob": str(expected_blob),
    }


def _attestation_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": result.get("schema_version"),
        "status": result.get("status"),
        "offline": result.get("offline"),
        "offline_environment": result.get("offline_environment"),
        "dataset": result.get("dataset"),
        "model": result.get("model"),
        "model_smoke": result.get("model_smoke"),
        "evidence": result.get("evidence"),
    }


def _attestation_sha256(result: Mapping[str, Any]) -> str:
    return _sha256_json(_attestation_payload(result))


def _current_proof(
    dataset_root: Path,
    preprocessed_path: Path,
    *,
    minimum_rows: int,
    maximum_rows: int,
    sample_count: int,
    weight_resolver: WeightResolver,
    model_loader: ModelLoader,
    mesh_path_resolver: MeshPathResolver,
    mesh_smoke_loader: MeshSmokeLoader,
) -> dict[str, Any]:
    dataset, specs = inspect_dataset(
        dataset_root,
        preprocessed_path,
        minimum_rows=minimum_rows,
        maximum_rows=maximum_rows,
        sample_count=sample_count,
        mesh_path_resolver=mesh_path_resolver,
        mesh_smoke_loader=mesh_smoke_loader,
    )
    model = dict(weight_resolver())
    required_model_fields = {
        "model_name": MODEL_NAME,
        "pretrained_tag": PRETRAINED_TAG,
        "repo_id": MODEL_REPO_ID,
        "requested_revision": MODEL_REVISION,
        "filename": MODEL_FILENAME,
    }
    for field, expected in required_model_fields.items():
        if model.get(field) != expected:
            raise ValueError(
                f"Resolved OpenCLIP {field} is {model.get(field)!r}, expected {expected!r}"
            )
    blob_path = Path(str(model.get("blob_path", "")))
    if not blob_path.is_file():
        raise FileNotFoundError(f"Resolved OpenCLIP blob is missing: {blob_path}")
    specs.append(_artifact_spec("openclip_weight_blob", blob_path))
    evidence = build_artifact_manifest(specs)
    raw_smoke = model_loader(model, SMOKE_QUERY)
    model_smoke = validate_model_smoke(raw_smoke, model, SMOKE_QUERY)
    # Close the mutation window across the actual model load.
    evidence_after_load = build_artifact_manifest(specs)
    if evidence_after_load != evidence:
        raise RuntimeError("ObjectThor index, mesh, or OpenCLIP blob changed during model load")
    return {
        "dataset": dataset,
        "model": model,
        "model_smoke": model_smoke,
        "evidence": evidence,
    }


def run(
    dataset_root: Path,
    preprocessed_path: Path,
    output: Path,
    *,
    minimum_rows: int = DEFAULT_MINIMUM_ROWS,
    maximum_rows: int = DEFAULT_MAXIMUM_ROWS,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    weight_resolver: WeightResolver = resolve_openclip_weight_blob,
    model_loader: ModelLoader = load_openclip_text_embedding,
    mesh_path_resolver: MeshPathResolver = resolve_production_mesh_path,
    mesh_smoke_loader: MeshSmokeLoader = smoke_load_mesh_payload,
) -> dict[str, Any]:
    started = time.monotonic()
    offline_environment = _set_offline_environment()
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "fail",
        "offline": all(value == "1" for value in offline_environment.values()),
        "offline_environment": offline_environment,
        "dataset": None,
        "model": None,
        "model_smoke": None,
        "evidence": None,
    }
    try:
        result.update(
            _current_proof(
                dataset_root,
                preprocessed_path,
                minimum_rows=minimum_rows,
                maximum_rows=maximum_rows,
                sample_count=sample_count,
                weight_resolver=weight_resolver,
                model_loader=model_loader,
                mesh_path_resolver=mesh_path_resolver,
                mesh_smoke_loader=mesh_smoke_loader,
            )
        )
        result["status"] = "pass"
        result["attestation"] = {
            "algorithm": "sha256",
            "sha256": _attestation_sha256(result),
        }
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    finally:
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        _write_json(output, result)
    return result


def _compare_current_to_saved(
    saved: Mapping[str, Any], current: Mapping[str, Any]
) -> list[str]:
    failures: list[str] = []
    for field in ("dataset", "model", "model_smoke", "evidence"):
        if saved.get(field) != current.get(field):
            failures.append(f"ObjectThor saved {field} evidence differs from current offline proof")
    return failures


def verify_output(
    output: Path,
    dataset_root: Path,
    preprocessed_path: Path,
    *,
    minimum_rows: int = DEFAULT_MINIMUM_ROWS,
    maximum_rows: int = DEFAULT_MAXIMUM_ROWS,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    weight_resolver: WeightResolver = resolve_openclip_weight_blob,
    model_loader: ModelLoader = load_openclip_text_embedding,
    mesh_path_resolver: MeshPathResolver = resolve_production_mesh_path,
    mesh_smoke_loader: MeshSmokeLoader = smoke_load_mesh_payload,
) -> dict[str, Any]:
    started = time.monotonic()
    _set_offline_environment()
    output = _absolute(output)
    try:
        saved = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        saved = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "fail",
            "error_type": type(exc).__name__,
            "error": f"Cannot read ObjectThor preflight JSON {output}: {exc}",
        }

    failures: list[str] = []
    if not isinstance(saved, dict):
        saved = {"schema_version": RESULT_SCHEMA_VERSION, "status": "fail"}
        failures.append("ObjectThor preflight JSON is not an object")
    if saved.get("schema_version") != RESULT_SCHEMA_VERSION:
        failures.append("ObjectThor preflight schema_version is unsupported")
    if saved.get("status") != "pass":
        failures.append(f"ObjectThor preflight status is {saved.get('status')!r}, not pass")
    if saved.get("offline") is not True or any(
        not isinstance(saved.get("offline_environment"), dict)
        or saved["offline_environment"].get(name) != "1"
        for name in OFFLINE_VARIABLES
    ):
        failures.append("ObjectThor preflight did not attest the complete offline environment")

    attestation = saved.get("attestation")
    if not isinstance(attestation, dict) or attestation.get("algorithm") != "sha256":
        failures.append("ObjectThor preflight attestation is missing or malformed")
    elif attestation.get("sha256") != _attestation_sha256(saved):
        failures.append("ObjectThor preflight JSON attestation does not match its contents")

    try:
        current = _current_proof(
            dataset_root,
            preprocessed_path,
            minimum_rows=minimum_rows,
            maximum_rows=maximum_rows,
            sample_count=sample_count,
            weight_resolver=weight_resolver,
            model_loader=model_loader,
            mesh_path_resolver=mesh_path_resolver,
            mesh_smoke_loader=mesh_smoke_loader,
        )
    except Exception as exc:
        failures.append(
            f"Current cache-only ObjectThor proof failed: {type(exc).__name__}: {exc}"
        )
    else:
        failures.extend(_compare_current_to_saved(saved, current))

    failures = list(dict.fromkeys(failures))
    saved["verification"] = {
        "status": "pass" if not failures else "fail",
        "critical_issues": failures,
        "cache_only_model_load_repeated": not any(
            issue.startswith("Current cache-only ObjectThor proof failed") for issue in failures
        ),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    if failures:
        saved["status"] = "fail"
        saved["verification_error"] = "; ".join(failures)
    _write_json(output, saved)
    return saved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("data/objathor-assets")
    )
    parser.add_argument(
        "--preprocessed-path",
        type=Path,
        help="Defaults to <dataset-root>/preprocessed.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-rows", type=int, default=DEFAULT_MINIMUM_ROWS)
    parser.add_argument("--maximum-rows", type=int, default=DEFAULT_MAXIMUM_ROWS)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Rehash all evidence and repeat the cache-only CPU OpenCLIP load.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = _absolute(args.dataset_root)
    preprocessed_path = _absolute(
        args.preprocessed_path or (dataset_root / "preprocessed")
    )
    output = _absolute(args.output)
    kwargs = {
        "minimum_rows": args.minimum_rows,
        "maximum_rows": args.maximum_rows,
        "sample_count": args.sample_count,
    }
    if args.verify_only:
        result = verify_output(
            output, dataset_root, preprocessed_path, **kwargs
        )
    else:
        result = run(output=output, dataset_root=dataset_root, preprocessed_path=preprocessed_path, **kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
