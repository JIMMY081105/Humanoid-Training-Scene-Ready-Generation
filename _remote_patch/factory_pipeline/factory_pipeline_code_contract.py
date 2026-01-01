#!/usr/bin/env python3
"""Factory profile for the complete SceneSmith pipeline/code attestation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

try:
    from . import pipeline_code_contract as base
except ImportError:
    import pipeline_code_contract as base


FACTORY_BINDING = {
    "profile": "factory_reference_20260713",
    "run_name": "full_quality_factory_sam3d_artvip_artiverse_20260713",
    "original_prompt_sha256": "3bb58cf8e07012cc0c5381eb51730da03b25fb6659a08338238d9fd4baa5093b",
    "appendix_sha256": "ffc327f082431d407c15b25b039d7d2676786a03663f186777558e9ba8e1db19",
    "effective_prompt_sha256": "35387fa3fffb0af92b6c9e07e39f2b59a5443d497190f721090f82bd81734322",
    "factory_contract_sha256": "f33e56eef4375d1f68684bd10c02d7d2f82ee72420b8a12570e8fcbe135a94cd",
    "reference_image_sha256": None,
    "reference_authority": "prompt_only_no_user_image_supplied",
}
RUNNER_BINDINGS = {
    "EXPECTED_ORIGINAL_PROMPT_SHA256": FACTORY_BINDING["original_prompt_sha256"],
    "EXPECTED_APPENDIX_SHA256": FACTORY_BINDING["appendix_sha256"],
    "EXPECTED_EFFECTIVE_PROMPT_SHA256": FACTORY_BINDING["effective_prompt_sha256"],
    "EXPECTED_FACTORY_CONTRACT_SHA256": FACTORY_BINDING["factory_contract_sha256"],
}
FACTORY_CLAUSES = (
    ("no_fast_hssd", "- Do not use an `hssd_only_fast` run as the final-quality run unless the user explicitly approves it."),
    ("compulsory_artiverse", "- Artiverse is compulsory for every full-quality run. The local `data/artiverse` dataset and the `artiverse_articulated` route must exist, be enabled, pass validation, and contribute at least one asset that survives into the final assembled scene. An enabled flag without proven final asset usage does not satisfy this contract."),
    ("gated_assembly", "- Do not assemble `combined_house`, final Drake exports, Isaac/USD exports, or final renders until every required zone has passed a zone-level quality gate."),
    ("canonical_checkout", "- Record the canonical execution checkout for this run and never sync with `--delete` unless explicitly approved. The canonical factory checkout is the authorized `/data/run01/scvj260/scenesmith-factory-codex` on ParaCloud; ParaCloud is also where the final 2-GPU Drake acceptance runs."),
    ("fixed_shell", "- The building is a single fixed shell of **44 m x 32 m x 5 m** containing exactly **14 enclosed functional rooms**. Exterior/common areas (loading dock, entrance transition, internal circulation, exterior truck road, landscaping) are explicit non-room zones, not extra rooms."),
    ("circulation", "- Circulation minimums are hard gates: **3.0 m forklift routes, 2.5 m production aisles, 1.2-1.5 m worker clearances**, and door/emergency-exit/window clearances. Pedestrian and forklift paths must be separable; the delivery truck stays outside the building shell."),
    ("build_gate", "- All `[BUILD]` scripts implemented and covered by `pipeline_code_contract.json`: factory layout seed + gate, factory room contract, factory profile in the self-exam gates, factory variation gate, factory navigation, factory articulated-motion roles, factory benchmark mode, whole-factory reference gate teaching."),
)
REQUIRED_FACTORY_ARTIFACTS = {
    "CODEX_FACTORY_FULL_QUALITY_PIPELINE.md",
    "CODEX_FOOD_FACTORY_ASSET_AUDIT.md",
    "CODEX_FOOD_FACTORY_ASSET_INVENTORY.csv",
    "FACTORY_8GPU_EXECUTION_TASKS.md",
    "scripts/factory_contract.json",
    "scripts/seed_reference_factory_layout.py",
    "scripts/validate_factory_floor_layout.py",
    "scripts/factory_room_contract.py",
    "scripts/factory_room_visual_self_exam.py",
    "scripts/factory_variation_gate.py",
    "scripts/validate_factory_articulated_motion.py",
    "scripts/validate_factory_navigation.py",
    "scripts/whole_factory_reference_gate.py",
    "scripts/factory_benchmark_receipt.py",
    "scripts/factory_pipeline_code_contract.py",
    "scripts/factory_final_acceptance_bundle.py",
    "scripts/factory_two_gpu_drake_acceptance_contract.py",
    "scripts/room_self_exam.py",
    "scripts/room_visual_self_exam.py",
    "scripts/assemble_final_house_and_render.py",
    "remote_jobs/run_full_quality_factory_paracloud.sh",
    "remote_jobs/factory_compute_env.sh",
    "remote_jobs/prepare_factory_assets_cpu.sh",
    "remote_jobs/TEMPLATE_factory_assets_1gpu.sbatch",
    "remote_jobs/TEMPLATE_factory_benchmark_processing_hall.sbatch",
    "remote_jobs/TEMPLATE_factory_full_parallel.sbatch",
    "remote_jobs/TEMPLATE_factory_2gpu_acceptance.sbatch",
}


def _configure() -> None:
    base.CONTRACT_NAME = "scenesmith-full-quality-factory-pipeline-code"
    base.REQUIRED_EXTERNAL_INPUT_BINDING = dict(FACTORY_BINDING)
    base.RUNNER_EXTERNAL_BINDING_VARIABLES = dict(RUNNER_BINDINGS)
    base.REQUIRED_SPEC_CLAUSES = FACTORY_CLAUSES
    extra = (
        "CODEX_FOOD_FACTORY_ASSET_AUDIT.md",
        "CODEX_FOOD_FACTORY_ASSET_INVENTORY.csv",
        "FACTORY_8GPU_EXECUTION_TASKS.md",
        "remote_jobs/factory_compute_env.sh",
        "remote_jobs/prepare_factory_assets_cpu.sh",
        "remote_jobs/TEMPLATE_factory_assets_1gpu.sbatch",
        "remote_jobs/TEMPLATE_factory_benchmark_processing_hall.sbatch",
        "remote_jobs/TEMPLATE_factory_full_parallel.sbatch",
        "remote_jobs/TEMPLATE_factory_2gpu_acceptance.sbatch",
        "scripts/factory_contract.json",
        "scripts/scene_contract_appendix.txt",
        "scripts/factory_final_acceptance_bundle.py",
        "scripts/factory_two_gpu_drake_acceptance_contract.py",
    )
    base.CONTRACT_REGRESSION_FILES = tuple(dict.fromkeys((*base.CONTRACT_REGRESSION_FILES, *extra)))


def _validate_factory_coverage(manifest: dict[str, Any]) -> None:
    artifacts = {
        str(item.get("path"))
        for item in manifest.get("artifacts", [])
        if isinstance(item, dict)
    }
    missing = sorted(REQUIRED_FACTORY_ARTIFACTS - artifacts)
    if missing:
        raise base.PipelineCodeContractError(
            "factory [BUILD] artifact coverage is incomplete: " + ", ".join(missing)
        )
    external = manifest.get("external_input_binding")
    if external != FACTORY_BINDING:
        raise base.PipelineCodeContractError("factory external input binding is stale")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _configure()
    manifest = base.run(
        args.repo_dir,
        args.spec,
        args.runner,
        args.output,
        verify_only=args.verify_only,
    )
    _validate_factory_coverage(manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=Path("CODEX_FACTORY_FULL_QUALITY_PIPELINE.md"))
    parser.add_argument("--runner", type=Path, default=Path("remote_jobs/run_full_quality_factory_paracloud.sh"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run(args)
    except Exception as exc:
        print(f"factory pipeline/code contract failed: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps({"status": manifest["status"], "artifact_count": manifest["artifact_count"], "attestation": manifest["attestation"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
