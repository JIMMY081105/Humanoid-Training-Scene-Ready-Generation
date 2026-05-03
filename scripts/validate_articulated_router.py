"""Validate compulsory Artiverse and ArtVIP articulated retrieval routes.

This is a fail-fast preflight, not a scene run.  A passing result proves all of:

1. The furniture router explicitly selects ``artiverse_articulated`` and the
   generic ``articulated`` fallback for legitimate openable-furniture prompts.
2. Artiverse and ArtVIP are loaded as separate articulated sources.
3. Each source returns concrete candidates with IDs and existing SDF paths under
   its configured dataset root.

Final placed/surviving Artiverse usage is enforced separately by
``assemble_final_house_and_render.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from .artiverse_contract import (
        ArtiverseAuthority,
        ArtiverseContractError,
        load_artiverse_authority,
    )
except ImportError:  # Direct execution: python scripts/validate_articulated_router.py
    from artiverse_contract import (  # type: ignore[no-redef]
        ArtiverseAuthority,
        ArtiverseContractError,
        load_artiverse_authority,
    )

REQUIRED_EMBEDDING_FILES = (
    "clip_embeddings.npy",
    "embedding_index.yaml",
    "metadata_index.yaml",
)

DEFAULT_PROMPTS = [
    "openable classroom storage cabinet with two doors",
    "teacher office filing cabinet with drawers",
    "library bookcase cabinet with glass doors",
    "school supply closet utility cabinet with hinged doors",
]


def _resolve_repo_path(path_value: str | Path, repo_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_dir / path).resolve()


def _source_status(
    source_name: str, data_path: Path, embeddings_path: Path
) -> dict[str, Any]:
    missing_embedding_files = [
        name for name in REQUIRED_EMBEDDING_FILES if not (embeddings_path / name).is_file()
    ]
    return {
        "source": source_name,
        "data_path": str(data_path),
        "data_path_exists": data_path.is_dir(),
        "embeddings_path": str(embeddings_path),
        "embeddings_path_exists": embeddings_path.is_dir(),
        "missing_embedding_files": missing_embedding_files,
    }


def _source_preflight_errors(status: dict[str, Any]) -> list[str]:
    source_name = status["source"]
    errors = []
    if not status["data_path_exists"]:
        errors.append(f"{source_name}: data path is missing: {status['data_path']}")
    if not status["embeddings_path_exists"]:
        errors.append(
            f"{source_name}: embeddings path is missing: {status['embeddings_path']}"
        )
    if status["missing_embedding_files"]:
        errors.append(
            f"{source_name}: embeddings are incomplete; missing "
            + ", ".join(status["missing_embedding_files"])
        )
    return errors


def _is_path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _candidate_passes_source_contract(
    candidate: dict[str, Any],
    source_name: str,
    data_path: Path,
    authority: ArtiverseAuthority | None = None,
) -> bool:
    sdf_path_value = candidate.get("sdf_path")
    if not sdf_path_value:
        return False
    sdf_path = Path(str(sdf_path_value))
    basic_pass = (
        candidate.get("source") == source_name
        and bool(candidate.get("object_id"))
        and sdf_path.is_file()
        and _is_path_within(sdf_path, data_path)
    )
    if not basic_pass:
        return False
    if source_name != "artiverse":
        return True
    if authority is None:
        return False
    try:
        expected = authority.asset(str(candidate.get("object_id", "")))
    except ArtiverseContractError:
        return False
    return sdf_path.resolve() == expected["source_sdf_path"]


def _load_cfg(
    model: str,
    repo_dir: Path,
    artiverse_data: Path,
    artiverse_embeddings: Path,
    artvip_data: Path,
    artvip_embeddings: Path,
):
    from omegaconf import OmegaConf

    base = OmegaConf.load(
        repo_dir / "configurations" / "furniture_agent" / "base_furniture_agent.yaml"
    )
    overrides = OmegaConf.create(
        {
            "openai": {
                "model": model,
                "vision_detail": "low",
                "reasoning_effort": {"asset_analysis": "low"},
                "verbosity": {"asset_analysis": "low"},
            },
            "asset_manager": {
                "general_asset_source": "generated",
                "backend": "sam3d",
                "router": {
                    "analysis_max_retries": 1,
                    "strategies": {
                        "generated": {"enabled": True, "max_retries": 1},
                        "articulated": {
                            "enabled": True,
                            "max_retries": 3,
                            "use_lenient_validation": True,
                        },
                        "artiverse_articulated": {
                            "enabled": True,
                            "max_retries": 3,
                            "use_lenient_validation": True,
                        },
                        "thin_covering": {"enabled": True, "max_retries": 1},
                    },
                },
                "articulated": {
                    "sources": {
                        "partnet_mobility": {"enabled": False},
                        "artvip": {
                            "enabled": True,
                            "data_path": str(artvip_data),
                            "embeddings_path": str(artvip_embeddings),
                        },
                        "artiverse": {
                            "enabled": True,
                            "data_path": str(artiverse_data),
                            "embeddings_path": str(artiverse_embeddings),
                        },
                    }
                },
            },
        }
    )
    return OmegaConf.merge(base, overrides)


def _analyze_prompt(router: Any, prompt: str, dimensions: list[float]) -> dict[str, Any]:
    analysis = router.analyze_request(prompt, dimensions)
    if analysis.error:
        return {"prompt": prompt, "error": analysis.error, "items": []}
    return {
        "prompt": prompt,
        "error": None,
        "items": [
            {
                "description": item.description,
                "short_name": item.short_name,
                "object_type": item.object_type.value,
                "strategies": item.strategies,
                "dimensions": item.dimensions,
            }
            for item in analysis.items
        ],
    }


def _load_source_index(
    source_name: str, data_path: Path, embeddings_path: Path
) -> dict[str, Any]:
    from scenesmith.agent_utils.articulated_retrieval_server.config import (
        ArticulatedSourceConfig,
    )
    from scenesmith.agent_utils.articulated_retrieval_server.data_loader import (
        load_preprocessed_data,
    )

    source_cfg = ArticulatedSourceConfig(
        name=source_name,
        enabled=True,
        data_path=data_path,
        embeddings_path=embeddings_path,
    )
    data = load_preprocessed_data(source_cfg)
    if data is None:
        raise RuntimeError(f"{source_name}: articulated source failed to load")
    matching_indices = data.filter_by_object_type("FURNITURE")
    if not matching_indices:
        raise RuntimeError(f"{source_name}: no FURNITURE candidates in embeddings")
    object_ids = [data.embedding_index[index] for index in matching_indices]
    return {
        "source": source_name,
        "data_path": data_path,
        "embeddings": data.clip_embeddings[matching_indices],
        "object_ids": object_ids,
        "metadata": {object_id: data.metadata_by_id[object_id] for object_id in object_ids},
    }


def _retrieve_candidates(
    prompt: str,
    top_k: int,
    source_index: dict[str, Any],
    authority: ArtiverseAuthority | None = None,
) -> list[dict[str, Any]]:
    from test_asset_retrieval import retrieve_top_k

    results = retrieve_top_k(
        query=prompt,
        embeddings=source_index["embeddings"],
        object_ids=source_index["object_ids"],
        top_k=top_k,
        source="articulated",
    )
    candidates = []
    for object_id, score in results:
        meta = source_index["metadata"][object_id]
        sdf_path = Path(meta.sdf_path)
        candidate = {
            "object_id": object_id,
            "score": float(score),
            "source": meta.source,
            "description": meta.description,
            "category": meta.category,
            "sdf_path": str(sdf_path),
            "sdf_exists": sdf_path.is_file(),
            "sdf_within_dataset": _is_path_within(
                sdf_path, source_index["data_path"]
            ),
        }
        candidate["passes_source_contract"] = _candidate_passes_source_contract(
            candidate,
            source_index["source"],
            source_index["data_path"],
            authority,
        )
        candidates.append(candidate)
    return candidates


def _write_summary(output_path: Path, summary: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument(
        "--vlm-backend", default=os.getenv("SCENESMITH_VLM_BACKEND", "openai")
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--dimensions", default="1.2,0.5,1.8")
    parser.add_argument("--artiverse-data", default="data/artiverse")
    parser.add_argument("--artiverse-embeddings", default="data/artiverse/embeddings")
    parser.add_argument("--artvip-data", default="data/artvip_sdf")
    parser.add_argument("--artvip-embeddings", default="data/artvip_sdf/embeddings")
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    repo_dir = Path(args.repo_dir).resolve()
    source_paths = {
        "artiverse": {
            "data": _resolve_repo_path(args.artiverse_data, repo_dir),
            "embeddings": _resolve_repo_path(args.artiverse_embeddings, repo_dir),
        },
        "artvip": {
            "data": _resolve_repo_path(args.artvip_data, repo_dir),
            "embeddings": _resolve_repo_path(args.artvip_embeddings, repo_dir),
        },
    }
    dataset_status = {
        source: _source_status(source, paths["data"], paths["embeddings"])
        for source, paths in source_paths.items()
    }
    failures = [
        failure
        for status in dataset_status.values()
        for failure in _source_preflight_errors(status)
    ]

    summary: dict[str, Any] = {
        "status": "fail",
        "failures": failures,
        "vlm_backend": args.vlm_backend,
        "model": args.model,
        "datasets": dataset_status,
        "prompts": [],
        "selected_asset_identifiers": {"artiverse": [], "artvip": []},
    }
    if failures:
        _write_summary(output_path, summary)
        return 2

    try:
        artiverse_authority = load_artiverse_authority(
            source_paths["artiverse"]["data"],
            source_paths["artiverse"]["embeddings"],
        )
        summary["artiverse_authority"] = artiverse_authority.evidence()
    except ArtiverseContractError as exc:
        summary["failures"].append(f"Artiverse prepared-data contract failed: {exc}")
        _write_summary(output_path, summary)
        return 2

    try:
        source_indexes = {
            source: _load_source_index(source, paths["data"], paths["embeddings"])
            for source, paths in source_paths.items()
        }
    except Exception as exc:
        summary["failures"].append(f"articulated source loading failed: {exc}")
        _write_summary(output_path, summary)
        return 2

    from scenesmith.agent_utils.asset_router import AssetRouter
    from scenesmith.agent_utils.room import AgentType
    from scenesmith.agent_utils.vlm_service import VLMService

    cfg = _load_cfg(
        model=args.model,
        repo_dir=repo_dir,
        artiverse_data=source_paths["artiverse"]["data"],
        artiverse_embeddings=source_paths["artiverse"]["embeddings"],
        artvip_data=source_paths["artvip"]["data"],
        artvip_embeddings=source_paths["artvip"]["embeddings"],
    )
    vlm_service = VLMService(backend=args.vlm_backend)
    router = AssetRouter(
        agent_type=AgentType.FURNITURE,
        vlm_service=vlm_service,
        cfg=cfg,
        blender_server=None,
    )

    dimensions = [float(value.strip()) for value in args.dimensions.split(",")]
    prompts = args.prompts or DEFAULT_PROMPTS
    selected_ids: dict[str, set[str]] = {"artiverse": set(), "artvip": set()}
    for prompt in prompts:
        analysis = _analyze_prompt(router, prompt, dimensions)
        source_candidates: dict[str, list[dict[str, Any]]] = {}
        for source_name, source_index in source_indexes.items():
            try:
                candidates = _retrieve_candidates(
                    prompt,
                    args.top_k,
                    source_index,
                    artiverse_authority if source_name == "artiverse" else None,
                )
            except Exception as exc:
                candidates = []
                failures.append(f"{prompt}: {source_name} retrieval failed: {exc}")
            source_candidates[source_name] = candidates
            analysis[f"{source_name}_candidates"] = candidates

            valid_candidates = [
                candidate
                for candidate in candidates
                if candidate.get("passes_source_contract")
            ]
            analysis[f"passes_{source_name}_candidate_check"] = bool(valid_candidates)
            if not valid_candidates:
                failures.append(
                    f"{prompt}: no valid {source_name} candidate with an existing "
                    f"SDF under {source_paths[source_name]['data']}"
                )
            selected_ids[source_name].update(
                str(candidate["object_id"]) for candidate in valid_candidates
            )

        has_artiverse_strategy = any(
            "artiverse_articulated" in item.get("strategies", [])
            for item in analysis["items"]
        )
        has_articulated_strategy = any(
            "articulated" in item.get("strategies", []) for item in analysis["items"]
        )
        analysis["passes_artiverse_router_strategy_check"] = has_artiverse_strategy
        analysis["passes_artvip_router_strategy_check"] = has_articulated_strategy
        if not has_artiverse_strategy:
            failures.append(
                f"{prompt}: router did not select artiverse_articulated strategy"
            )
        if not has_articulated_strategy:
            failures.append(f"{prompt}: router did not select articulated fallback")
        summary["prompts"].append(analysis)

    summary["selected_asset_identifiers"] = {
        source: sorted(object_ids) for source, object_ids in selected_ids.items()
    }
    for source_name, object_ids in selected_ids.items():
        if not object_ids:
            failures.append(f"no validated {source_name} asset identifier was selected")

    summary["failures"] = failures
    summary["status"] = "pass" if not failures else "fail"
    _write_summary(output_path, summary)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
