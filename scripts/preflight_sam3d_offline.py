#!/usr/bin/env python3
"""Load SAM3D offline and bind the exact model/config/cache artifacts used."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

from pathlib import Path
from typing import Any, Mapping, Sequence


OFFLINE_VARIABLES = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "DIFFUSERS_OFFLINE")
EVIDENCE_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 3
INFERENCE_SMOKE_SCHEMA_VERSION = 1
HASH_CHUNK_SIZE = 1024 * 1024
SHA256_HEX_LENGTH = 64
INFERENCE_IMAGE_SIZE = (512, 512)
INFERENCE_ELLIPSE_BBOX = (156, 156, 356, 356)
INFERENCE_PROMPT = "circle"
MIN_INFERENCE_TARGET_IOU = 0.5
MIN_INFERENCE_TARGET_COVERAGE = 0.5
MIN_INFERENCE_FOREGROUND_PRECISION = 0.5


def _write(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_evidence(path: Path) -> dict[str, Any]:
    """Hash the executable Python source of one cached torch-hub repository."""

    root = path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SAM3D cache source tree is missing: {root}")
    files = sorted(
        candidate
        for candidate in root.rglob("*.py")
        if "__pycache__" not in candidate.parts
    )
    if not files:
        raise ValueError(f"SAM3D cache source tree has no Python files: {root}")
    digest = hashlib.sha256()
    total_bytes = 0
    for file_path in files:
        relative = file_path.relative_to(root).as_posix()
        file_hash = _sha256_file(file_path)
        size = file_path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        total_bytes += size
    return {
        "path": str(root),
        "kind": "python_source_tree",
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "selection": "**/*.py excluding __pycache__",
    }


def _artifact_spec(role: str, path: Path, *, kind: str = "file") -> dict[str, Any]:
    return {"role": role, "path": path.resolve(), "kind": kind}


def _walk_mapping(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield child_prefix, str(key), child
            yield from _walk_mapping(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            yield from _walk_mapping(child, child_prefix)


def _read_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to resolve SAM3D artifacts") from exc
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _discover_local_pipeline_artifacts(
    sam3_checkpoint: Path, pipeline_config: Path
) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    """Resolve every local config/checkpoint referenced by the Hydra pipeline."""

    sam3_checkpoint = sam3_checkpoint.resolve()
    pipeline_config = pipeline_config.resolve()
    if not sam3_checkpoint.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint is missing: {sam3_checkpoint}")
    if not pipeline_config.is_file():
        raise FileNotFoundError(f"SAM3D pipeline config is missing: {pipeline_config}")

    specs = [
        _artifact_spec("sam3_checkpoint", sam3_checkpoint),
        _artifact_spec("pipeline_config", pipeline_config),
    ]
    pretrained_repositories: set[str] = set()
    dino_models: set[str] = set()
    queued_configs = [pipeline_config]
    parsed_configs: set[Path] = set()
    seen_paths = {sam3_checkpoint, pipeline_config}

    while queued_configs:
        config_path = queued_configs.pop(0).resolve()
        if config_path in parsed_configs:
            continue
        parsed_configs.add(config_path)
        document = _read_yaml(config_path)
        for key_path, key, value in _walk_mapping(document):
            if key == "pretrained_model_name_or_path" and isinstance(value, str):
                pretrained_repositories.add(value)
            if key == "dino_model" and isinstance(value, str):
                dino_models.add(value)
            if not (
                isinstance(value, str)
                and key != "pretrained_model_name_or_path"
                and key.endswith(("_path", "_ckpt_path", "_config_path"))
            ):
                continue
            referenced = Path(value).expanduser()
            if not referenced.is_absolute():
                referenced = config_path.parent / referenced
            referenced = referenced.resolve()
            if referenced in seen_paths:
                continue
            seen_paths.add(referenced)
            role = f"pipeline_reference:{config_path.name}:{key_path}"
            specs.append(_artifact_spec(role, referenced))
            if referenced.suffix.lower() in {".yaml", ".yml"}:
                queued_configs.append(referenced)

    return specs, pretrained_repositories, dino_models


def _discover_cache_artifacts(
    pretrained_repositories: set[str], dino_models: set[str]
) -> list[dict[str, Any]]:
    """Resolve the Hugging Face and torch-hub cache objects loaded by SAM3D."""

    specs: list[dict[str, Any]] = []
    for model_name in sorted(pretrained_repositories):
        local_candidate = Path(model_name).expanduser()
        if local_candidate.exists():
            specs.append(
                _artifact_spec(
                    f"pretrained_model:{model_name}", local_candidate.resolve()
                )
            )
            continue
        if "/" not in model_name:
            raise ValueError(
                f"Cannot resolve non-local pretrained model reference: {model_name}"
            )
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required to resolve SAM3D cache evidence"
            ) from exc
        cached = Path(
            hf_hub_download(
                repo_id=model_name,
                repo_type="model",
                filename="model.pt",
                local_files_only=True,
            )
        )
        specs.append(
            _artifact_spec(
                f"huggingface_cache:{model_name}:model.pt", cached.resolve()
            )
        )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to resolve SAM3D cache evidence") from exc
    hub_root = Path(torch.hub.get_dir()).resolve()
    for model_name in sorted(dino_models):
        source_candidates = sorted(hub_root.glob("facebookresearch_dinov2_*"))
        main_candidates = [path for path in source_candidates if path.name.endswith("_main")]
        if main_candidates:
            source_candidates = main_candidates
        if len(source_candidates) != 1:
            raise RuntimeError(
                "Expected exactly one cached facebookresearch/dinov2 source tree for "
                f"{model_name}, found {[str(path) for path in source_candidates]}"
            )
        weight_candidates = sorted(
            (hub_root / "checkpoints").glob(f"{model_name}*.pth")
        )
        if len(weight_candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one cached weight for {model_name}, found "
                f"{[str(path) for path in weight_candidates]}"
            )
        specs.extend(
            [
                _artifact_spec(
                    f"torch_hub_cache:{model_name}:source",
                    source_candidates[0],
                    kind="python_source_tree",
                ),
                _artifact_spec(
                    f"torch_hub_cache:{model_name}:weights",
                    weight_candidates[0],
                ),
            ]
        )
    return specs


def discover_artifacts(
    sam3_checkpoint: Path,
    pipeline_config: Path,
    *,
    cache_artifacts: Mapping[str, Path] | None = None,
) -> list[dict[str, Any]]:
    local_specs, repositories, dino_models = _discover_local_pipeline_artifacts(
        sam3_checkpoint, pipeline_config
    )
    if cache_artifacts is None:
        cache_specs = _discover_cache_artifacts(repositories, dino_models)
    else:
        cache_specs = [
            _artifact_spec(
                role,
                path,
                kind="python_source_tree" if path.is_dir() else "file",
            )
            for role, path in sorted(cache_artifacts.items())
        ]
    specs = [*local_specs, *cache_specs]
    roles = [str(spec["role"]) for spec in specs]
    duplicate_roles = sorted(role for role in set(roles) if roles.count(role) > 1)
    if duplicate_roles:
        raise ValueError(f"Duplicate SAM3D artifact roles: {duplicate_roles}")
    return sorted(specs, key=lambda spec: str(spec["role"]))


def _artifact_evidence(spec: Mapping[str, Any]) -> dict[str, Any]:
    role = str(spec["role"])
    path = Path(spec["path"]).resolve()
    kind = str(spec.get("kind", "file"))
    if kind == "python_source_tree":
        return {"role": role, **_source_tree_evidence(path)}
    if kind != "file":
        raise ValueError(f"Unsupported SAM3D artifact kind {kind!r} for {role}")
    if not path.is_file():
        raise FileNotFoundError(f"SAM3D artifact is missing ({role}): {path}")
    size = path.stat().st_size
    if size < 1:
        raise ValueError(f"SAM3D artifact is empty ({role}): {path}")
    return {
        "role": role,
        "path": str(path),
        "kind": "file",
        "sha256": _sha256_file(path),
        "size_bytes": size,
    }


def build_artifact_manifest(specs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    artifacts = [_artifact_evidence(spec) for spec in specs]
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "algorithm": "sha256",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_inference_input() -> tuple[Any, Any]:
    """Build the fixed RGB smoke image and its independently drawn target mask."""

    import numpy as np
    from PIL import Image, ImageDraw

    image = Image.new("RGB", INFERENCE_IMAGE_SIZE, (255, 255, 255))
    ImageDraw.Draw(image).ellipse(
        INFERENCE_ELLIPSE_BBOX,
        fill=(0, 0, 255),
    )
    target = Image.new("L", INFERENCE_IMAGE_SIZE, 0)
    ImageDraw.Draw(target).ellipse(INFERENCE_ELLIPSE_BBOX, fill=1)
    return image, np.asarray(target, dtype=np.uint8)


def _foreground_bbox(mask: Any) -> list[int] | None:
    import numpy as np

    rows, columns = np.nonzero(mask)
    if not len(rows):
        return None
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max()),
        int(rows.max()),
    ]


def _build_inference_smoke(
    mask: Any,
    *,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe one real production ``generate_mask`` result deterministically."""

    import numpy as np

    image, target = _canonical_inference_input()
    mask_array = np.asarray(mask)
    mask_bytes = np.ascontiguousarray(mask_array).tobytes()
    foreground = mask_array == 1
    target_foreground = target == 1
    foreground_pixels = int(np.count_nonzero(foreground))
    target_pixels = int(np.count_nonzero(target_foreground))
    intersection_pixels = int(np.count_nonzero(foreground & target_foreground))
    union_pixels = int(np.count_nonzero(foreground | target_foreground))
    total_pixels = INFERENCE_IMAGE_SIZE[0] * INFERENCE_IMAGE_SIZE[1]
    if runtime is None:
        runtime = {
            "device_type": "cuda",
            "autocast_dtype": "bfloat16",
            "bf16_supported": True,
        }
    smoke: dict[str, Any] = {
        "schema_version": INFERENCE_SMOKE_SCHEMA_VERSION,
        "status": "pass",
        "input": {
            "mode": "RGB",
            "size": list(INFERENCE_IMAGE_SIZE),
            "background_rgb": [255, 255, 255],
            "shape": "ellipse",
            "shape_bbox_xyxy": list(INFERENCE_ELLIPSE_BBOX),
            "shape_fill_rgb": [0, 0, 255],
            "raw_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        },
        "request": {
            "function": "generate_mask",
            "mode": "object_description",
            "object_description": INFERENCE_PROMPT,
        },
        "runtime": dict(runtime),
        "output": {
            "shape": list(mask_array.shape),
            "dtype": str(mask_array.dtype),
            "binary_values": [int(value) for value in np.unique(mask_array)],
            "raw_mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
            "foreground_pixels": foreground_pixels,
            "foreground_fraction": foreground_pixels / total_pixels,
            "foreground_bbox_xyxy": _foreground_bbox(foreground),
            "target_pixels": target_pixels,
            "target_intersection_pixels": intersection_pixels,
            "target_union_pixels": union_pixels,
            "target_iou": intersection_pixels / union_pixels if union_pixels else 0.0,
            "target_coverage": (
                intersection_pixels / target_pixels if target_pixels else 0.0
            ),
            "foreground_precision": (
                intersection_pixels / foreground_pixels if foreground_pixels else 0.0
            ),
            "center_pixel_foreground": bool(
                foreground[INFERENCE_IMAGE_SIZE[1] // 2, INFERENCE_IMAGE_SIZE[0] // 2]
            ),
            "touches_image_edge": bool(
                foreground[0, :].any()
                or foreground[-1, :].any()
                or foreground[:, 0].any()
                or foreground[:, -1].any()
            ),
        },
        "semantic_verification": {"status": "pass", "critical_issues": []},
    }
    failures = verify_inference_smoke(smoke)
    if failures:
        smoke["status"] = "fail"
        smoke["semantic_verification"] = {
            "status": "fail",
            "critical_issues": failures,
        }
    return smoke


def _run_inference_smoke(
    sam3_processor: Any,
    *,
    generate_mask_function: Any | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Exercise the production SAM3 mask path with the canonical request."""

    if generate_mask_function is None:
        from scenesmith.agent_utils.geometry_generation_server import (
            sam3d_pipeline_manager,
        )

        generate_mask_function = sam3d_pipeline_manager.generate_mask
    image, _ = _canonical_inference_input()
    mask = generate_mask_function(
        image=image,
        sam3_processor=sam3_processor,
        mode="object_description",
        object_description=INFERENCE_PROMPT,
    )
    return _build_inference_smoke(mask, runtime=runtime)


def verify_inference_smoke(smoke: Any) -> list[str]:
    """Validate exact SAM3 smoke provenance and meaningful circle segmentation."""

    import math

    if not isinstance(smoke, dict):
        return ["SAM3 inference smoke is missing or malformed"]
    failures: list[str] = []
    expected_top_keys = {
        "schema_version",
        "status",
        "input",
        "request",
        "runtime",
        "output",
        "semantic_verification",
    }
    if set(smoke) != expected_top_keys:
        failures.append("SAM3 inference smoke fields do not match the exact schema")
    if smoke.get("schema_version") != INFERENCE_SMOKE_SCHEMA_VERSION:
        failures.append("SAM3 inference smoke schema_version is unsupported")
    if smoke.get("status") != "pass":
        failures.append("SAM3 inference smoke status is not pass")

    image, target = _canonical_inference_input()
    expected_input = {
        "mode": "RGB",
        "size": list(INFERENCE_IMAGE_SIZE),
        "background_rgb": [255, 255, 255],
        "shape": "ellipse",
        "shape_bbox_xyxy": list(INFERENCE_ELLIPSE_BBOX),
        "shape_fill_rgb": [0, 0, 255],
        "raw_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    }
    if smoke.get("input") != expected_input:
        failures.append(
            "SAM3 inference smoke input is not the canonical raw-RGB blue ellipse"
        )
    expected_request = {
        "function": "generate_mask",
        "mode": "object_description",
        "object_description": INFERENCE_PROMPT,
    }
    if smoke.get("request") != expected_request:
        failures.append(
            "SAM3 inference smoke did not use the production circle request"
        )
    expected_runtime = {
        "device_type": "cuda",
        "autocast_dtype": "bfloat16",
        "bf16_supported": True,
    }
    if smoke.get("runtime") != expected_runtime:
        failures.append(
            "SAM3 inference smoke lacks attested CUDA bfloat16 execution support"
        )

    output = smoke.get("output")
    expected_output_keys = {
        "shape",
        "dtype",
        "binary_values",
        "raw_mask_sha256",
        "foreground_pixels",
        "foreground_fraction",
        "foreground_bbox_xyxy",
        "target_pixels",
        "target_intersection_pixels",
        "target_union_pixels",
        "target_iou",
        "target_coverage",
        "foreground_precision",
        "center_pixel_foreground",
        "touches_image_edge",
    }
    if not isinstance(output, dict):
        failures.append("SAM3 inference smoke output is missing or malformed")
    else:
        if set(output) != expected_output_keys:
            failures.append(
                "SAM3 inference smoke output fields do not match the exact schema"
            )
        if output.get("shape") != [INFERENCE_IMAGE_SIZE[1], INFERENCE_IMAGE_SIZE[0]]:
            failures.append("SAM3 inference mask shape is not 512x512")
        if output.get("dtype") != "uint8":
            failures.append("SAM3 inference mask dtype is not uint8")
        if output.get("binary_values") != [0, 1]:
            failures.append("SAM3 inference mask is not a nonempty binary mask")
        if not _is_sha256(output.get("raw_mask_sha256")):
            failures.append("SAM3 inference raw-mask SHA-256 is malformed")

        total_pixels = INFERENCE_IMAGE_SIZE[0] * INFERENCE_IMAGE_SIZE[1]
        target_pixels = int(target.sum())
        foreground_pixels = output.get("foreground_pixels")
        intersection_pixels = output.get("target_intersection_pixels")
        union_pixels = output.get("target_union_pixels")
        if not isinstance(foreground_pixels, int) or isinstance(
            foreground_pixels, bool
        ) or not (0 < foreground_pixels < total_pixels):
            failures.append("SAM3 inference foreground pixel count is invalid")
        if output.get("target_pixels") != target_pixels:
            failures.append("SAM3 inference target pixel count is not canonical")
        if not isinstance(intersection_pixels, int) or isinstance(
            intersection_pixels, bool
        ) or not (0 < intersection_pixels <= target_pixels):
            failures.append("SAM3 inference target intersection is invalid")
        if not isinstance(union_pixels, int) or isinstance(union_pixels, bool):
            failures.append("SAM3 inference target union is invalid")
        elif isinstance(foreground_pixels, int) and isinstance(
            intersection_pixels, int
        ) and union_pixels != target_pixels + foreground_pixels - intersection_pixels:
            failures.append("SAM3 inference overlap counts are inconsistent")

        def verify_ratio(field: str, numerator: Any, denominator: Any) -> float | None:
            value = output.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                failures.append(f"SAM3 inference {field} is invalid")
                return None
            if (
                not isinstance(numerator, int)
                or not isinstance(denominator, int)
                or denominator <= 0
            ):
                return float(value)
            expected = numerator / denominator
            if not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12):
                failures.append(f"SAM3 inference {field} does not match its counts")
            return float(value)

        foreground_fraction = verify_ratio(
            "foreground_fraction", foreground_pixels, total_pixels
        )
        target_iou = verify_ratio("target_iou", intersection_pixels, union_pixels)
        target_coverage = verify_ratio(
            "target_coverage", intersection_pixels, target_pixels
        )
        foreground_precision = verify_ratio(
            "foreground_precision", intersection_pixels, foreground_pixels
        )
        if foreground_fraction is not None and not 0.03 <= foreground_fraction <= 0.30:
            failures.append("SAM3 inference foreground area is not circle-like")
        if target_iou is not None and target_iou < MIN_INFERENCE_TARGET_IOU:
            failures.append("SAM3 inference mask has insufficient target IoU")
        if (
            target_coverage is not None
            and target_coverage < MIN_INFERENCE_TARGET_COVERAGE
        ):
            failures.append("SAM3 inference mask has insufficient target coverage")
        if (
            foreground_precision is not None
            and foreground_precision < MIN_INFERENCE_FOREGROUND_PRECISION
        ):
            failures.append("SAM3 inference mask has insufficient foreground precision")

        bbox = output.get("foreground_bbox_xyxy")
        if not (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in bbox
            )
            and 0 <= bbox[0] < bbox[2] < INFERENCE_IMAGE_SIZE[0]
            and 0 <= bbox[1] < bbox[3] < INFERENCE_IMAGE_SIZE[1]
        ):
            failures.append("SAM3 inference foreground bbox is invalid")
        elif any(
            abs(actual - expected) > 64
            for actual, expected in zip(bbox, INFERENCE_ELLIPSE_BBOX)
        ):
            failures.append(
                "SAM3 inference foreground bbox is not aligned to the circle"
            )
        if output.get("center_pixel_foreground") is not True:
            failures.append("SAM3 inference mask does not contain the circle center")
        if output.get("touches_image_edge") is not False:
            failures.append("SAM3 inference mask incorrectly reaches the image edge")

    semantic = smoke.get("semantic_verification")
    if semantic != {"status": "pass", "critical_issues": []}:
        failures.append("SAM3 inference semantic verification is not pass")
    return list(dict.fromkeys(failures))


def verify_artifact_manifest(
    manifest: Any, specs: Sequence[Mapping[str, Any]]
) -> list[str]:
    if not isinstance(manifest, dict):
        return ["SAM3D artifact evidence manifest is missing or malformed"]
    failures: list[str] = []
    if manifest.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        failures.append("SAM3D artifact evidence schema_version is unsupported")
    if manifest.get("algorithm") != "sha256":
        failures.append("SAM3D artifact evidence algorithm is not sha256")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        return [*failures, "SAM3D artifact evidence list is missing or malformed"]
    entry_map = {
        str(entry.get("role")): entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("role")
    }
    expected_map = {str(spec["role"]): spec for spec in specs}
    if len(entry_map) != len(entries):
        failures.append("SAM3D artifact evidence has duplicate or malformed roles")
    if len(entries) != len(expected_map):
        failures.append("SAM3D artifact evidence count differs from the exact resolver")
    missing = sorted(set(expected_map) - set(entry_map))
    unexpected = sorted(set(entry_map) - set(expected_map))
    if missing:
        failures.append(f"SAM3D artifact evidence is missing roles: {missing}")
    if unexpected:
        failures.append(f"SAM3D artifact evidence has unexpected roles: {unexpected}")
    if manifest.get("artifact_count") != len(entries):
        failures.append("SAM3D artifact_count does not match the evidence list")

    for role in sorted(set(expected_map) & set(entry_map)):
        spec = expected_map[role]
        entry = entry_map[role]
        expected_path = Path(spec["path"]).resolve()
        recorded_path = entry.get("path")
        if not isinstance(recorded_path, str) or not Path(recorded_path).is_absolute():
            failures.append(f"{role}: evidence path is missing or not absolute")
            continue
        if Path(recorded_path).resolve() != expected_path:
            failures.append(
                f"{role}: evidence path does not match current artifact; "
                f"recorded={Path(recorded_path).resolve()}, expected={expected_path}"
            )
            continue
        if entry.get("kind") != spec.get("kind", "file"):
            failures.append(f"{role}: artifact kind does not match current resolver")
            continue
        if not _is_sha256(entry.get("sha256")):
            failures.append(f"{role}: evidence SHA-256 is missing or malformed")
            continue
        try:
            current = _artifact_evidence(spec)
        except (OSError, ValueError) as exc:
            failures.append(f"{role}: cannot rehash current artifact: {exc}")
            continue
        if current["sha256"] != entry["sha256"]:
            failures.append(
                f"{role}: SHA-256 mismatch; gate={entry['sha256']}, "
                f"current={current['sha256']}"
            )
        if entry.get("kind") == "file" and current["size_bytes"] != entry.get(
            "size_bytes"
        ):
            failures.append(f"{role}: file size changed")
        if entry.get("kind") == "python_source_tree" and (
            current["file_count"] != entry.get("file_count")
            or current["total_bytes"] != entry.get("total_bytes")
        ):
            failures.append(f"{role}: cached source-tree inventory changed")
    return failures


def _attestation_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": result.get("schema_version"),
        "status": result.get("status"),
        "sam3_checkpoint": result.get("sam3_checkpoint"),
        "pipeline_config": result.get("pipeline_config"),
        "offline": result.get("offline"),
        "offline_environment": result.get("offline_environment"),
        "model_loaded": result.get("model_loaded"),
        "pipeline_loaded": result.get("pipeline_loaded"),
        "gpu": result.get("gpu"),
        "visible_gpu_count": result.get("visible_gpu_count"),
        "total_memory_bytes": result.get("total_memory_bytes"),
        "allocated_bytes": result.get("allocated_bytes"),
        "reserved_bytes": result.get("reserved_bytes"),
        "peak_allocated_bytes": result.get("peak_allocated_bytes"),
        "peak_reserved_bytes": result.get("peak_reserved_bytes"),
        "memory_stats_reset": result.get("memory_stats_reset"),
        "memory_stats_reset_error": result.get("memory_stats_reset_error"),
        "inference_smoke": result.get("inference_smoke"),
        "evidence": result.get("evidence"),
    }


def _attestation_sha256(result: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _attestation_payload(result), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_pipelines_with_cuda(
    sam3_checkpoint: Path, pipeline_config: Path
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("SAM3D offline preflight requires a visible CUDA GPU")
    torch.cuda.empty_cache()
    # Some valid container/driver combinations expose CUDA allocation counters but
    # reject the reset ioctl with ``Invalid device argument``. This process has not
    # loaded a model yet, so its counters already start at zero; retain the exact
    # reset outcome as attested evidence instead of treating that ioctl as a model
    # initialization failure.
    memory_stats_reset = True
    memory_stats_reset_error = None
    try:
        torch.cuda.reset_peak_memory_stats(0)
    except RuntimeError as exc:
        memory_stats_reset = False
        memory_stats_reset_error = str(exc)

    from scenesmith.agent_utils.geometry_generation_server.sam3d_pipeline_manager import (
        SAM3DPipelineManager,
    )

    model, pipeline = SAM3DPipelineManager.get_pipelines(
        sam3_checkpoint=sam3_checkpoint,
        sam3d_checkpoint=pipeline_config,
    )
    inference_smoke = _run_inference_smoke(
        model,
        runtime={
            "device_type": "cuda",
            "autocast_dtype": str(torch.bfloat16).removeprefix("torch."),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        },
    )
    torch.cuda.synchronize(0)
    return {
        "model_loaded": model is not None,
        "pipeline_loaded": pipeline is not None,
        "gpu": torch.cuda.get_device_name(0),
        "visible_gpu_count": torch.cuda.device_count(),
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "allocated_bytes": torch.cuda.memory_allocated(0),
        "reserved_bytes": torch.cuda.memory_reserved(0),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "memory_stats_reset": memory_stats_reset,
        "memory_stats_reset_error": memory_stats_reset_error,
        "inference_smoke": inference_smoke,
    }


def run(
    sam3_checkpoint: Path,
    pipeline_config: Path,
    output: Path,
    *,
    cache_artifacts: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    # Set these before importing SAM3D/Transformers so any missing cache entry is
    # a hard failure instead of a surprise network transfer.
    for variable in OFFLINE_VARIABLES:
        os.environ[variable] = "1"

    sam3_checkpoint = sam3_checkpoint.resolve()
    pipeline_config = pipeline_config.resolve()
    started = time.monotonic()
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "fail",
        "offline": all(os.environ.get(name) == "1" for name in OFFLINE_VARIABLES),
        "offline_environment": {
            name: os.environ.get(name) for name in OFFLINE_VARIABLES
        },
        "sam3_checkpoint": str(sam3_checkpoint),
        "pipeline_config": str(pipeline_config),
        "model_loaded": False,
        "pipeline_loaded": False,
        "inference_smoke": None,
        "evidence": None,
    }
    try:
        specs = discover_artifacts(
            sam3_checkpoint,
            pipeline_config,
            cache_artifacts=cache_artifacts,
        )
        result["evidence"] = build_artifact_manifest(specs)
        runtime = _load_pipelines_with_cuda(sam3_checkpoint, pipeline_config)
        result.update(runtime)
        evidence_failures = verify_artifact_manifest(result["evidence"], specs)
        inference_failures = verify_inference_smoke(result.get("inference_smoke"))
        result["evidence_verification"] = {
            "status": "pass" if not evidence_failures else "fail",
            "critical_issues": evidence_failures,
        }
        result["inference_smoke_verification"] = {
            "status": "pass" if not inference_failures else "fail",
            "critical_issues": inference_failures,
        }
        if evidence_failures:
            result["error"] = (
                "SAM3D artifacts changed during offline load: "
                + "; ".join(evidence_failures)
            )
            if inference_failures:
                result["error"] += "; SAM3 inference smoke failed: " + "; ".join(
                    inference_failures
                )
        elif inference_failures:
            result["error"] = "SAM3 inference smoke failed: " + "; ".join(
                inference_failures
            )
        elif result["model_loaded"] and result["pipeline_loaded"]:
            result["status"] = "pass"
            result["attestation"] = {
                "algorithm": "sha256",
                "sha256": _attestation_sha256(result),
            }
        else:
            result["error"] = "SAM3D manager returned an empty model or pipeline"
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
    finally:
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        _write(output, result)
    return result


def verify_saved_result(
    result: Any,
    sam3_checkpoint: Path,
    pipeline_config: Path,
    *,
    cache_artifacts: Mapping[str, Path] | None = None,
    _verification_sections: dict[str, list[str]] | None = None,
) -> list[str]:
    if not isinstance(result, dict):
        failure = "SAM3D preflight JSON is missing or malformed"
        if _verification_sections is not None:
            _verification_sections["artifact"] = [failure]
            _verification_sections["inference_smoke"] = [failure]
        return [failure]
    failures: list[str] = []
    artifact_failures: list[str] = []
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        failures.append("SAM3D preflight schema_version is unsupported")
    if result.get("status") != "pass":
        failures.append(f"SAM3D preflight status is {result.get('status')!r}, not pass")
    if result.get("offline") is not True:
        failures.append("SAM3D preflight did not run offline")
    offline_environment = result.get("offline_environment")
    if not isinstance(offline_environment, dict) or any(
        offline_environment.get(name) != "1" for name in OFFLINE_VARIABLES
    ):
        failures.append("SAM3D preflight offline environment is incomplete")
    if result.get("model_loaded") is not True:
        failures.append("SAM3 image model was not loaded")
    if result.get("pipeline_loaded") is not True:
        failures.append("SAM 3D Objects pipeline was not loaded")
    inference_failures = verify_inference_smoke(result.get("inference_smoke"))
    failures.extend(inference_failures)
    if not isinstance(result.get("visible_gpu_count"), int) or result.get(
        "visible_gpu_count", 0
    ) < 1:
        failures.append("SAM3D preflight has no valid GPU-load evidence")

    sam3_checkpoint = sam3_checkpoint.resolve()
    pipeline_config = pipeline_config.resolve()
    if result.get("sam3_checkpoint") != str(sam3_checkpoint):
        failures.append("SAM3 checkpoint path differs from the current canonical path")
    if result.get("pipeline_config") != str(pipeline_config):
        failures.append("SAM3D pipeline config path differs from the current canonical path")
    try:
        specs = discover_artifacts(
            sam3_checkpoint,
            pipeline_config,
            cache_artifacts=cache_artifacts,
        )
    except Exception as exc:
        artifact_failures.append(
            f"Cannot resolve current SAM3D artifacts: {type(exc).__name__}: {exc}"
        )
    else:
        artifact_failures.extend(
            verify_artifact_manifest(result.get("evidence"), specs)
        )
    failures.extend(artifact_failures)

    attestation = result.get("attestation")
    if not isinstance(attestation, dict) or attestation.get("algorithm") != "sha256":
        failures.append("SAM3D preflight attestation is missing or malformed")
    elif not _is_sha256(attestation.get("sha256")):
        failures.append("SAM3D preflight attestation SHA-256 is malformed")
    elif attestation["sha256"] != _attestation_sha256(result):
        failures.append("SAM3D preflight JSON attestation does not match its contents")
    if _verification_sections is not None:
        _verification_sections["artifact"] = artifact_failures
        _verification_sections["inference_smoke"] = inference_failures
    return list(dict.fromkeys(failures))


def verify_output(
    output: Path,
    sam3_checkpoint: Path,
    pipeline_config: Path,
    *,
    cache_artifacts: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    output = output.resolve()
    try:
        with output.open(encoding="utf-8") as stream:
            result = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "fail",
            "error_type": type(exc).__name__,
            "error": f"Cannot read SAM3D preflight JSON {output}: {exc}",
        }
    if not isinstance(result, dict):
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "fail",
            "error_type": "TypeError",
            "error": f"SAM3D preflight JSON {output} is not an object",
        }
    verification_sections: dict[str, list[str]] = {}
    failures = verify_saved_result(
        result,
        sam3_checkpoint,
        pipeline_config,
        cache_artifacts=cache_artifacts,
        _verification_sections=verification_sections,
    )
    artifact_failures = verification_sections.get("artifact", [])
    inference_failures = verification_sections.get("inference_smoke", [])
    result["evidence_verification"] = {
        "status": "pass" if not artifact_failures else "fail",
        "critical_issues": artifact_failures,
    }
    result["inference_smoke_verification"] = {
        "status": "pass" if not inference_failures else "fail",
        "critical_issues": inference_failures,
    }
    if failures:
        result["status"] = "fail"
        result["verification_error"] = "; ".join(failures)
    _write(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sam3-checkpoint", type=Path, default=Path("external/checkpoints/sam3.pt")
    )
    parser.add_argument(
        "--pipeline-config", type=Path, default=Path("external/checkpoints/pipeline.yaml")
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Rehash a saved proof without loading SAM3D or requiring a GPU.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sam3_checkpoint = args.sam3_checkpoint.resolve()
    pipeline_config = args.pipeline_config.resolve()
    output = args.output.resolve()
    if args.verify_only:
        result = verify_output(output, sam3_checkpoint, pipeline_config)
    else:
        result = run(sam3_checkpoint, pipeline_config, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
