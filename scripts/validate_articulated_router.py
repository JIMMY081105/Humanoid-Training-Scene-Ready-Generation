"""Validate that openable-furniture prompts route to articulated retrieval.

This is a preflight, not a full scene run. It records:

1. AssetRouter analysis strategies for openable-furniture prompts.
2. Concrete articulated retrieval candidates and SDF paths from ArtVIP/PartNet.

The run should not be considered full-quality if this script reports failures.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from scenesmith.agent_utils.asset_router import AssetRouter
from scenesmith.agent_utils.room import AgentType
from scenesmith.agent_utils.vlm_service import VLMService

from test_asset_retrieval import load_articulated_data, retrieve_top_k


DEFAULT_PROMPTS = [
    "openable classroom storage cabinet with two doors",
    "teacher office filing cabinet with drawers",
    "library bookcase cabinet with glass doors",
    "school supply closet utility cabinet with hinged doors",
]


def _load_cfg(model: str):
    base = OmegaConf.load("configurations/furniture_agent/base_furniture_agent.yaml")
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
                        "thin_covering": {"enabled": True, "max_retries": 1},
                    },
                },
                "articulated": {
                    "sources": {
                        "partnet_mobility": {"enabled": True},
                        "artvip": {"enabled": True},
                    }
                },
            },
        }
    )
    return OmegaConf.merge(base, overrides)


def _analyze_prompt(router: AssetRouter, prompt: str, dimensions: list[float]) -> dict[str, Any]:
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


def _retrieve_candidates(prompt: str, top_k: int) -> list[dict[str, Any]]:
    embeddings, object_ids, metadata, source_map = load_articulated_data(
        source="combined", object_type="FURNITURE"
    )
    results = retrieve_top_k(
        query=prompt,
        embeddings=embeddings,
        object_ids=object_ids,
        top_k=top_k,
        source="combined",
    )
    candidates = []
    for object_id, score in results:
        meta = metadata.get(object_id, {})
        source_name, _data_path = source_map.get(object_id, ("unknown", None))
        candidates.append(
            {
                "object_id": object_id,
                "score": float(score),
                "source": source_name,
                "description": meta.get("description"),
                "sdf_path": meta.get("sdf_path"),
                "sdf_exists": Path(str(meta.get("sdf_path", ""))).exists(),
            }
        )
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--vlm-backend", default=os.getenv("SCENESMITH_VLM_BACKEND", "openai"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--dimensions", default="1.2,0.5,1.8")
    args = parser.parse_args()

    dimensions = [float(value.strip()) for value in args.dimensions.split(",")]
    prompts = args.prompts or DEFAULT_PROMPTS
    cfg = _load_cfg(args.model)
    vlm_service = VLMService(backend=args.vlm_backend)
    router = AssetRouter(
        agent_type=AgentType.FURNITURE,
        vlm_service=vlm_service,
        cfg=cfg,
        blender_server=None,
    )

    prompt_results = []
    failures = []
    for prompt in prompts:
        analysis = _analyze_prompt(router, prompt, dimensions)
        try:
            candidates = _retrieve_candidates(prompt, args.top_k)
        except Exception as exc:
            candidates = []
            failures.append(f"{prompt}: articulated retrieval failed: {exc}")

        analysis["articulated_candidates"] = candidates
        has_articulated_strategy = any(
            "articulated" in item.get("strategies", []) for item in analysis["items"]
        )
        has_existing_sdf = any(c.get("sdf_exists") for c in candidates)
        analysis["passes_router_strategy_check"] = has_articulated_strategy
        analysis["passes_retrieval_path_check"] = has_existing_sdf
        if not has_articulated_strategy:
            failures.append(f"{prompt}: router did not select articulated strategy")
        if not has_existing_sdf:
            failures.append(f"{prompt}: no articulated SDF candidate exists")
        prompt_results.append(analysis)

    summary = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "vlm_backend": args.vlm_backend,
        "model": args.model,
        "prompts": prompt_results,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
