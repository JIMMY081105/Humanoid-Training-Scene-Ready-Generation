"""Run exactly one SceneSmith room stage pipeline inside an existing house output.

This is intentionally narrower than main.py: it loads an existing house layout and
room checkpoint, runs one room only, and never assembles combined_house.
"""

import argparse
import json
import logging
import os
import signal
import uuid
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

try:
    from .artiverse_contract import ArtiverseContractError, load_artiverse_authority
    from .materials_contract import MaterialsContractError, load_materials_authority
except ImportError:  # Direct execution: python scripts/run_single_room_worker.py
    from artiverse_contract import (  # type: ignore[no-redef]
        ArtiverseContractError,
        load_artiverse_authority,
    )
    from materials_contract import (  # type: ignore[no-redef]
        MaterialsContractError,
        load_materials_authority,
    )


console_logger = logging.getLogger(__name__)

REQUIRED_EMBEDDING_FILES = (
    "clip_embeddings.npy",
    "embedding_index.yaml",
    "metadata_index.yaml",
)

PYTORCH_CUDA_ALLOC_CONF = "PYTORCH_CUDA_ALLOC_CONF"
REQUIRED_CUDA_ALLOCATOR_SETTING = "expandable_segments:True"
OPENAI_MAX_RETRIES_ENV = "SCENESMITH_OPENAI_MAX_RETRIES"
DEFAULT_OPENAI_MAX_RETRIES = 24
OPENAI_REQUEST_TIMEOUT_SECONDS = 120.0


def _graceful_termination_handler(signum: int, _frame: Any) -> None:
    """Turn scheduler termination into ``SystemExit`` so service cleanup runs."""
    signal_name = signal.Signals(signum).name
    console_logger.warning("Received %s; stopping worker services cleanly", signal_name)
    raise SystemExit(128 + signum)


def _configure_pytorch_cuda_allocator(
    environ: MutableMapping[str, str] | None = None,
) -> str:
    """Enable fragmentation-resistant CUDA segments before importing torch.

    The geometry server is spawned later by this process and inherits the
    environment.  Preserve unrelated allocator settings, append the required
    setting when absent, and fail closed on an explicit contradictory or
    duplicate value instead of silently running with ambiguous behavior.
    """
    target = os.environ if environ is None else environ
    raw = target.get(PYTORCH_CUDA_ALLOC_CONF, "")
    entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
    expandable_entries = []
    for entry in entries:
        name, separator, value = entry.partition(":")
        if name.strip() == "expandable_segments":
            expandable_entries.append((separator, value.strip()))

    if len(expandable_entries) > 1:
        raise RuntimeError(
            f"{PYTORCH_CUDA_ALLOC_CONF} contains duplicate "
            "expandable_segments settings"
        )
    if expandable_entries:
        separator, value = expandable_entries[0]
        if not separator or value.lower() != "true":
            raise RuntimeError(
                f"{PYTORCH_CUDA_ALLOC_CONF} conflicts with the required "
                f"{REQUIRED_CUDA_ALLOCATOR_SETTING}: {raw!r}"
            )
    else:
        entries.append(REQUIRED_CUDA_ALLOCATOR_SETTING)

    configured = ",".join(entries)
    target[PYTORCH_CUDA_ALLOC_CONF] = configured
    return configured


def _shift_range(values: list[int], offset: int) -> list[int]:
    return [int(values[0]) + offset, int(values[1]) + offset]


def _configure_ports(cfg, offset: int) -> None:
    """Move all localhost servers/ranges away from other shard workers."""
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.experiment.geometry_generation_server.port = 7005 + offset
        cfg.experiment.hssd_retrieval_server.port = 7006 + offset
        cfg.experiment.articulated_retrieval_server.port = 7007 + offset
        cfg.experiment.materials_retrieval_server.port = 7008 + offset
        cfg.experiment.objaverse_retrieval_server.port = 7009 + offset

        for agent_name in ("wall_agent", "ceiling_agent", "manipuland_agent"):
            agent = cfg[agent_name]
            if "rendering" in agent and "blender_server_port_range" in agent.rendering:
                agent.rendering.blender_server_port_range = _shift_range(
                    list(agent.rendering.blender_server_port_range), offset
                )
            if (
                "collision_geometry" in agent
                and "server_port_range" in agent.collision_geometry
            ):
                agent.collision_geometry.server_port_range = _shift_range(
                    list(agent.collision_geometry.server_port_range), offset
                )


def _asset_pipeline_overrides(
    asset_pipeline: str,
    *,
    artiverse_data: str = "data/artiverse",
    artiverse_embeddings: str = "data/artiverse/embeddings",
    artvip_data: str = "data/artvip_sdf",
    artvip_embeddings: str = "data/artvip_sdf/embeddings",
    materials_data: str = "data/materials",
    materials_embeddings: str = "data/materials_full_quality_contract/embeddings",
) -> list[str]:
    """Return Hydra overrides for the requested asset source policy.

    Keep the policy explicit because using an HSSD-only run when the target is
    full-quality generation silently bypasses SAM3D and most richer asset routes.
    """
    common = [
        "furniture_agent.asset_manager.hssd.use_top_k=20",
        "wall_agent.asset_manager.hssd.use_top_k=20",
        "ceiling_agent.asset_manager.hssd.use_top_k=20",
        "manipuland_agent.asset_manager.hssd.use_top_k=20",
        "furniture_agent.collision_geometry.coacd.max_convex_hull=32",
        "wall_agent.collision_geometry.coacd.max_convex_hull=32",
        "ceiling_agent.collision_geometry.coacd.max_convex_hull=32",
        "manipuland_agent.collision_geometry.coacd.max_convex_hull=32",
        "furniture_agent.collision_geometry.vhacd.max_convex_hulls=32",
        "wall_agent.collision_geometry.vhacd.max_convex_hulls=32",
        "ceiling_agent.collision_geometry.vhacd.max_convex_hulls=32",
        "manipuland_agent.collision_geometry.vhacd.max_convex_hulls=32",
    ]

    if asset_pipeline == "hssd":
        return common + [
            "furniture_agent.asset_manager.general_asset_source=hssd",
            "wall_agent.asset_manager.general_asset_source=hssd",
            "ceiling_agent.asset_manager.general_asset_source=hssd",
            "manipuland_agent.asset_manager.general_asset_source=hssd",
        ]

    if asset_pipeline == "generated_sam3d":
        return common + [
            f"experiment.materials_retrieval_server.data_path={materials_data}",
            f"experiment.materials_retrieval_server.embeddings_path={materials_embeddings}",
            "furniture_agent.asset_manager.general_asset_source=generated",
            "wall_agent.asset_manager.general_asset_source=generated",
            "ceiling_agent.asset_manager.general_asset_source=generated",
            "manipuland_agent.asset_manager.general_asset_source=objaverse",
            "furniture_agent.asset_manager.backend=sam3d",
            "wall_agent.asset_manager.backend=sam3d",
            "ceiling_agent.asset_manager.backend=sam3d",
            "manipuland_agent.asset_manager.backend=sam3d",
            "manipuland_agent.asset_manager.objaverse.use_top_k=10",
            "manipuland_agent.asset_manager.objaverse.use_lenient_validation=true",
            "furniture_agent.asset_manager.router.strategies.generated.enabled=true",
            "wall_agent.asset_manager.router.strategies.generated.enabled=true",
            "ceiling_agent.asset_manager.router.strategies.generated.enabled=true",
            "manipuland_agent.asset_manager.router.strategies.generated.enabled=true",
            "furniture_agent.asset_manager.router.strategies.articulated.enabled=true",
            "wall_agent.asset_manager.router.strategies.articulated.enabled=true",
            "ceiling_agent.asset_manager.router.strategies.articulated.enabled=true",
            "manipuland_agent.asset_manager.router.strategies.articulated.enabled=true",
            "furniture_agent.asset_manager.articulated.sources.artvip.enabled=true",
            "wall_agent.asset_manager.articulated.sources.artvip.enabled=true",
            "ceiling_agent.asset_manager.articulated.sources.artvip.enabled=true",
            "manipuland_agent.asset_manager.articulated.sources.artvip.enabled=true",
            f"furniture_agent.asset_manager.articulated.sources.artvip.data_path={artvip_data}",
            f"furniture_agent.asset_manager.articulated.sources.artvip.embeddings_path={artvip_embeddings}",
            "++furniture_agent.asset_manager.articulated.sources.artiverse.enabled=true",
            f"++furniture_agent.asset_manager.articulated.sources.artiverse.data_path={artiverse_data}",
            f"++furniture_agent.asset_manager.articulated.sources.artiverse.embeddings_path={artiverse_embeddings}",
            "++furniture_agent.asset_manager.router.strategies.artiverse_articulated.enabled=true",
            "++furniture_agent.asset_manager.router.strategies.artiverse_articulated.max_retries=3",
            "++furniture_agent.asset_manager.router.strategies.artiverse_articulated.use_lenient_validation=true",
        ]

    raise ValueError(f"Unsupported asset pipeline: {asset_pipeline}")


def _get_child(node: Any, key: str, default: Any = None) -> Any:
    """Read a key from dict-like or OmegaConf nodes without importing OmegaConf."""
    if node is None:
        return default
    try:
        return node.get(key, default)
    except (AttributeError, TypeError):
        try:
            return node[key]
        except (KeyError, TypeError):
            return default


def _resolve_repo_path(path_value: Any, repo_dir: Path) -> Path:
    path = Path(str(path_value)).expanduser()
    return path.resolve() if path.is_absolute() else (repo_dir / path).resolve()


def _articulated_source_summary(
    source_cfg: Any, repo_dir: Path
) -> dict[str, Any]:
    if source_cfg is None:
        return {
            "enabled": False,
            "data_path": None,
            "data_path_exists": False,
            "embeddings_path": None,
            "embeddings_path_exists": False,
            "missing_embedding_files": list(REQUIRED_EMBEDDING_FILES),
        }

    data_path_value = _get_child(source_cfg, "data_path", "")
    embeddings_path_value = _get_child(source_cfg, "embeddings_path", "")
    data_path = (
        _resolve_repo_path(data_path_value, repo_dir) if data_path_value else None
    )
    embeddings_path = (
        _resolve_repo_path(embeddings_path_value, repo_dir)
        if embeddings_path_value
        else None
    )
    missing_embedding_files = [
        name
        for name in REQUIRED_EMBEDDING_FILES
        if embeddings_path is None or not (embeddings_path / name).is_file()
    ]
    return {
        "enabled": bool(_get_child(source_cfg, "enabled", False)),
        "data_path": str(data_path) if data_path is not None else None,
        "data_path_exists": data_path.is_dir() if data_path is not None else False,
        "embeddings_path": (
            str(embeddings_path) if embeddings_path is not None else None
        ),
        "embeddings_path_exists": (
            embeddings_path.is_dir() if embeddings_path is not None else False
        ),
        "missing_embedding_files": missing_embedding_files,
    }


def _asset_policy_summary(cfg, repo_dir: Path | None = None) -> dict:
    repo_dir = (repo_dir or Path.cwd()).resolve()
    agents = ("furniture_agent", "wall_agent", "ceiling_agent", "manipuland_agent")
    summary = {}
    artvip_enabled_agents: dict[str, bool] = {}
    for agent_name in agents:
        agent = cfg[agent_name]
        collision = agent.collision_geometry
        articulated = _get_child(agent.asset_manager, "articulated")
        sources = _get_child(articulated, "sources")
        artvip_enabled_agents[agent_name] = bool(
            _get_child(_get_child(sources, "artvip"), "enabled", False)
        )
        summary[agent_name] = {
            "general_asset_source": str(agent.asset_manager.general_asset_source),
            "backend": str(agent.asset_manager.get("backend", "")),
            "objaverse_top_k": int(agent.asset_manager.objaverse.use_top_k)
            if "objaverse" in agent.asset_manager
            else None,
            "coacd_max_convex_hull": int(collision.coacd.max_convex_hull),
            "vhacd_max_convex_hulls": int(collision.vhacd.max_convex_hulls),
        }

    furniture_manager = cfg["furniture_agent"].asset_manager
    furniture_sources = _get_child(
        _get_child(furniture_manager, "articulated"), "sources"
    )
    strategies = _get_child(_get_child(furniture_manager, "router"), "strategies")
    summary["articulated_contract"] = {
        "articulated_strategy_enabled": bool(
            _get_child(_get_child(strategies, "articulated"), "enabled", False)
        ),
        "artiverse_strategy_enabled": bool(
            _get_child(
                _get_child(strategies, "artiverse_articulated"), "enabled", False
            )
        ),
        "artvip_enabled_agents": artvip_enabled_agents,
        "artvip": _articulated_source_summary(
            _get_child(furniture_sources, "artvip"), repo_dir
        ),
        "artiverse": _articulated_source_summary(
            _get_child(furniture_sources, "artiverse"), repo_dir
        ),
    }
    return summary


def _validate_asset_policy(
    cfg, asset_pipeline: str, repo_dir: Path | None = None
) -> None:
    summary = _asset_policy_summary(cfg, repo_dir=repo_dir)
    if asset_pipeline != "generated_sam3d":
        return

    expected_sources = {
        "furniture_agent": "generated",
        "wall_agent": "generated",
        "ceiling_agent": "generated",
        "manipuland_agent": "objaverse",
    }
    errors = []
    if "hssd" in _service_names(asset_pipeline):
        errors.append("generated_sam3d service plan must not start the HSSD server")
    for agent_name, expected in expected_sources.items():
        actual = summary[agent_name]["general_asset_source"]
        if actual != expected:
            errors.append(f"{agent_name}.asset_manager.general_asset_source={actual}, expected {expected}")

    for agent_name in ("furniture_agent", "wall_agent", "ceiling_agent"):
        if summary[agent_name]["backend"] != "sam3d":
            errors.append(f"{agent_name}.asset_manager.backend is not sam3d")

    for agent_name, values in summary.items():
        if agent_name == "articulated_contract":
            continue
        if values["coacd_max_convex_hull"] > 32:
            errors.append(f"{agent_name}.collision_geometry.coacd.max_convex_hull > 32")
        if values["vhacd_max_convex_hulls"] > 32:
            errors.append(f"{agent_name}.collision_geometry.vhacd.max_convex_hulls > 32")

    articulated = summary["articulated_contract"]
    if not articulated["articulated_strategy_enabled"]:
        errors.append("furniture articulated strategy is disabled")
    if not articulated["artiverse_strategy_enabled"]:
        errors.append("furniture artiverse_articulated strategy is disabled")
    for agent_name, enabled in articulated["artvip_enabled_agents"].items():
        if not enabled:
            errors.append(f"{agent_name} ArtVIP source is disabled")

    for source_name in ("artvip", "artiverse"):
        source = articulated[source_name]
        if not source["enabled"]:
            errors.append(f"{source_name} articulated source is disabled")
        if not source["data_path_exists"]:
            errors.append(f"{source_name} data path is missing: {source['data_path']}")
        if not source["embeddings_path_exists"]:
            errors.append(
                f"{source_name} embeddings path is missing: {source['embeddings_path']}"
            )
        if source["missing_embedding_files"]:
            errors.append(
                f"{source_name} embeddings are incomplete; missing "
                + ", ".join(source["missing_embedding_files"])
            )

    artiverse = articulated["artiverse"]
    if artiverse["data_path_exists"] and artiverse["embeddings_path_exists"]:
        try:
            load_artiverse_authority(
                Path(artiverse["data_path"]), Path(artiverse["embeddings_path"])
            )
        except ArtiverseContractError as exc:
            errors.append(f"Artiverse prepared-data contract failed: {exc}")

    if errors:
        raise RuntimeError(
            "Resolved Hydra asset policy does not match generated_sam3d contract: "
            + "; ".join(errors)
        )


def _load_cfg(args: argparse.Namespace):
    import hydra

    from omegaconf import OmegaConf, open_dict
    from scenesmith.utils.omegaconf import register_resolvers

    register_resolvers()
    config_dir = Path(args.repo_dir).resolve() / "configurations"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                f"+name={args.run_name}",
                f"experiment.csv_path={args.csv}",
                "experiment.num_workers=1",
                f"experiment.pipeline.start_stage={args.start_stage}",
                f"experiment.pipeline.stop_stage={args.stop_stage}",
                "experiment.pipeline.parallel_rooms=false",
                "experiment.pipeline.final_assembly_policy=external_artiverse_gated",
                "floor_plan_agent.mode=house",
                "++codex.enabled=true",
                f"++codex.cwd={args.repo_dir}",
                "++codex.timeout_seconds=1800",
                "furniture_agent.asset_manager.router.parallel_workers=1",
                "wall_agent.asset_manager.router.parallel_workers=1",
                "ceiling_agent.asset_manager.router.parallel_workers=1",
                "manipuland_agent.asset_manager.router.parallel_workers=1",
                "furniture_agent.asset_manager.router.strategies.generated.max_retries=5",
                "wall_agent.asset_manager.router.strategies.generated.max_retries=5",
                "ceiling_agent.asset_manager.router.strategies.generated.max_retries=5",
                "manipuland_agent.asset_manager.router.strategies.generated.max_retries=5",
                "furniture_agent.asset_manager.router.strategies.thin_covering.generator.enabled=false",
                "wall_agent.asset_manager.router.strategies.thin_covering.generator.enabled=false",
                "ceiling_agent.asset_manager.router.strategies.thin_covering.generator.enabled=false",
                "manipuland_agent.asset_manager.router.strategies.thin_covering.generator.enabled=false",
                "furniture_agent.context_image_generation.enabled=false",
                "manipuland_agent.context_image_generation.enabled=false",
            ]
            + _asset_pipeline_overrides(
                args.asset_pipeline,
                artiverse_data=args.artiverse_data,
                artiverse_embeddings=args.artiverse_embeddings,
                artvip_data=args.artvip_data,
                artvip_embeddings=args.artvip_embeddings,
                materials_data=args.materials_data,
                materials_embeddings=args.materials_embeddings,
            ),
        )

    with open_dict(cfg):
        cfg.experiment._name = "indoor_scene_generation"
        cfg.floor_plan_agent._name = "stateful_floor_plan_agent"
        cfg.furniture_agent._name = "stateful_furniture_agent"
        cfg.wall_agent._name = "stateful_wall_agent"
        cfg.ceiling_agent._name = "stateful_ceiling_agent"
        cfg.manipuland_agent._name = "stateful_manipuland_agent"
        cfg.experiment.output_dir = str(Path(args.run_dir).resolve())

    _configure_ports(cfg, args.port_offset)
    OmegaConf.resolve(cfg)
    return cfg


def _service_names(asset_pipeline: str) -> list[str]:
    if asset_pipeline == "generated_sam3d":
        return ["geometry", "objaverse", "articulated", "materials"]
    return ["hssd", "articulated", "materials"]


def _start_worker_services(experiment: Any, asset_pipeline: str) -> list[str]:
    """Start every retrieval/generation service needed by a one-room worker."""
    started: list[str] = []
    try:
        for service_name in _service_names(asset_pipeline):
            method_name = f"_start_{service_name}_server"
            method = getattr(experiment, method_name, None)
            if method is None:
                raise RuntimeError(
                    f"IndoorSceneGenerationExperiment lacks required {method_name}()"
                )
            method()
            started.append(service_name)
    except Exception:
        _stop_worker_services(experiment, started)
        raise
    return started


def _stop_worker_services(experiment: Any, started: list[str]) -> None:
    for service_name in reversed(started):
        method = getattr(experiment, f"_stop_{service_name}_server", None)
        if method is None:
            console_logger.warning("No stop method for worker service %s", service_name)
            continue
        try:
            method()
        except Exception:
            console_logger.exception("Failed to stop worker service %s", service_name)


def _recover_final_blend(
    *,
    room_dir: Path,
    room_id: str,
    cfg_dict: dict[str, Any],
    room_scene_cls: Any | None = None,
    export_scene_blend: Any | None = None,
    force_refresh: bool = False,
) -> Path:
    """Re-export a missing final blend from an already-complete room state.

    The upstream room generator treats ``final_scene/scene_state.json`` as the
    completion marker and returns before exporting when that checkpoint already
    exists.  Recovery therefore has to restore that exact state and export it
    without starting generation/retrieval services or placing any new objects.
    """
    state_dir = room_dir / "scene_states" / "final_scene"
    state_path = state_dir / "scene_state.json"
    blend_path = state_dir / "scene.blend"
    if blend_path.is_file() and blend_path.stat().st_size > 0 and not force_refresh:
        return blend_path
    if not state_path.is_file() or state_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Cannot recover final blend; room state is missing or empty: {state_path}"
        )

    if room_scene_cls is None:
        from scenesmith.agent_utils.room import RoomScene

        room_scene_cls = RoomScene
    if export_scene_blend is None:
        from scenesmith.experiments.indoor_scene_generation import (
            _export_scene_blend_file,
        )

        export_scene_blend = _export_scene_blend_file

    backup_path: Path | None = None
    if blend_path.exists():
        if not blend_path.is_file():
            raise RuntimeError(f"Final room blend is not a regular file: {blend_path}")
        backup_path = blend_path.with_name(
            f".{blend_path.name}.previous.{os.getpid()}.{uuid.uuid4().hex}"
        )
        os.replace(blend_path, backup_path)
    try:
        with state_path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        scene = room_scene_cls(room_geometry=None, scene_dir=room_dir, room_id=room_id)
        scene.restore_from_state_dict(state)
        export_scene_blend(
            scene=scene,
            scene_dir=room_dir,
            cfg_dict=cfg_dict,
            name="final_scene",
        )
        if not blend_path.is_file() or blend_path.stat().st_size == 0:
            raise RuntimeError(
                "Final room blend recovery did not produce a non-empty file: "
                f"{blend_path}"
            )
    except Exception:
        if blend_path.exists():
            blend_path.unlink()
        if backup_path is not None and backup_path.exists():
            os.replace(backup_path, blend_path)
        raise
    if backup_path is not None:
        backup_path.unlink()
    return blend_path


def _configure_agents_transport() -> str:
    """Use the Responses API so visual tool outputs reach the next model turn.

    Agents SDK 0.17.4's Chat Completions converter deliberately reduces function
    tool outputs to text because the official Chat Completions API cannot carry
    image content in a tool message. SceneSmith's ``observe_scene`` returns its
    pixels as ``ToolOutputImage`` records, so forcing Chat Completions silently
    left the designer visually blind. Responses preserves the canonical mixed
    image/text function output end to end.
    """

    from agents import set_default_openai_api, set_tracing_disabled

    transport = "responses"
    set_default_openai_api(transport)
    set_tracing_disabled(True)
    return transport


def _configure_openai_request_resilience(
    environ: MutableMapping[str, str] | None = None,
) -> int:
    """Keep an in-flight agent turn alive across a brief proxy interruption.

    Retrying at the OpenAI request boundary is safe even after earlier tool calls
    mutated the room: no new tool result is applied until one Responses request
    completes. This avoids rolling back and regenerating a whole furniture target
    when the laptop-to-cluster proxy reconnects for a minute or two.
    """

    target = os.environ if environ is None else environ
    raw_retries = target.get(OPENAI_MAX_RETRIES_ENV, str(DEFAULT_OPENAI_MAX_RETRIES))
    try:
        max_retries = int(raw_retries)
    except ValueError as error:
        raise RuntimeError(
            f"{OPENAI_MAX_RETRIES_ENV} must be an integer, got {raw_retries!r}"
        ) from error
    if not 2 <= max_retries <= 40:
        raise RuntimeError(
            f"{OPENAI_MAX_RETRIES_ENV} must be between 2 and 40, got {max_retries}"
        )

    from agents import set_default_openai_client
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        max_retries=max_retries,
        timeout=OPENAI_REQUEST_TIMEOUT_SECONDS,
    )
    set_default_openai_client(client, use_for_tracing=False)
    return max_retries


def main() -> None:
    # This must run before the SceneSmith imports below initialize torch/CUDA.
    # Geometry-server subprocesses inherit the validated allocator policy.
    cuda_allocator_config = _configure_pytorch_cuda_allocator()

    from omegaconf import OmegaConf
    from scenesmith.agent_utils.house import HouseLayout
    from scenesmith.experiments.indoor_scene_generation import (
        IndoorSceneGenerationExperiment,
        _generate_room,
    )
    from scenesmith.utils.logging import ConsoleLogger, FileLoggingContext

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--csv",
        default="inputs/full_school_floor_20260703.csv",
        help="Prompt CSV used for this run. Must match the floor-plan stage.",
    )
    parser.add_argument(
        "--run-name",
        default="scenesmith_room_worker",
        help="Hydra run name for this room worker.",
    )
    parser.add_argument("--room-id", required=True)
    parser.add_argument("--start-stage", required=True)
    parser.add_argument("--stop-stage", default="manipuland")
    parser.add_argument(
        "--asset-pipeline",
        choices=["hssd", "generated_sam3d"],
        default="hssd",
        help=(
            "Asset source policy. Use generated_sam3d for full-quality runs; "
            "hssd preserves the old fast continuation behavior."
        ),
    )
    parser.add_argument("--port-offset", type=int, required=True)
    parser.add_argument("--render-gpu-id", type=int, default=0)
    parser.add_argument("--artiverse-data", default="data/artiverse")
    parser.add_argument("--artiverse-embeddings", default="data/artiverse/embeddings")
    parser.add_argument("--artvip-data", default="data/artvip_sdf")
    parser.add_argument("--artvip-embeddings", default="data/artvip_sdf/embeddings")
    parser.add_argument("--materials-data", default="data/materials")
    parser.add_argument(
        "--materials-source-embeddings", default="data/materials/embeddings"
    )
    parser.add_argument(
        "--materials-embeddings",
        default="data/materials_full_quality_contract/embeddings",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Resolve and validate Hydra config, print asset policy JSON, then exit.",
    )
    parser.add_argument(
        "--recover-final-blend",
        action="store_true",
        help=(
            "Restore final_scene/scene_state.json and re-export only scene.blend. "
            "No generation or retrieval services are started."
        ),
    )
    parser.add_argument(
        "--refresh-final-blend",
        action="store_true",
        help=(
            "Atomically re-export scene.blend from the exact current final state, "
            "even when a prior blend exists."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOGLEVEL", "INFO").upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    console_logger.info(
        "Configured %s=%s before SceneSmith CUDA imports",
        PYTORCH_CUDA_ALLOC_CONF,
        cuda_allocator_config,
    )

    repo_dir = Path(args.repo_dir).resolve()
    materials_authority: dict[str, Any] | None = None
    if args.asset_pipeline == "generated_sam3d":
        try:
            materials_authority = load_materials_authority(
                data_root=_resolve_repo_path(args.materials_data, repo_dir),
                source_embeddings=_resolve_repo_path(
                    args.materials_source_embeddings, repo_dir
                ),
                contract_embeddings=_resolve_repo_path(
                    args.materials_embeddings, repo_dir
                ),
                min_retained=1900,
                max_pruned=15,
            )
        except MaterialsContractError as exc:
            raise RuntimeError(
                f"Full-quality materials contract failed: {exc}"
            ) from exc

    cfg = _load_cfg(args)
    _validate_asset_policy(cfg, args.asset_pipeline, repo_dir=repo_dir)
    asset_policy = _asset_policy_summary(cfg, repo_dir=repo_dir)
    asset_policy["worker_services"] = _service_names(args.asset_pipeline)
    if materials_authority is not None:
        asset_policy["materials_contract"] = materials_authority
    console_logger.info(
        "Resolved asset policy: %s", json.dumps(asset_policy, sort_keys=True)
    )
    if args.config_only:
        print(json.dumps(asset_policy, indent=2, sort_keys=True))
        return

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    run_dir = Path(args.run_dir).resolve()
    scene_dir = run_dir / "scene_000"
    room_dir = scene_dir / f"room_{args.room_id}"
    if not room_dir.exists():
        raise FileNotFoundError(f"Room directory not found: {room_dir}")

    if args.recover_final_blend or args.refresh_final_blend:
        blend_path = _recover_final_blend(
            room_dir=room_dir,
            room_id=args.room_id,
            cfg_dict=cfg_dict,
            force_refresh=args.refresh_final_blend,
        )
        console_logger.info("Recovered final room blend: %s", blend_path)
        return

    _configure_agents_transport()
    openai_max_retries = _configure_openai_request_resilience()

    house_layout_path = scene_dir / "house_layout.json"
    with house_layout_path.open() as f:
        house_layout = HouseLayout.from_dict(
            __import__("json").load(f), house_dir=scene_dir
        )

    room_spec = house_layout.get_room_spec(args.room_id)
    room_geometry = house_layout.get_room_geometry(args.room_id)
    if room_geometry is None:
        raise RuntimeError(f"Room geometry not found for {args.room_id}")

    log_path = room_dir / f"room_worker_{args.start_stage}_{os.getpid()}.log"
    logger = ConsoleLogger(output_dir=room_dir)
    experiment = IndoorSceneGenerationExperiment(cfg)

    # A scheduler or recovery controller uses SIGTERM to stop a worker. Raise
    # SystemExit so the service-cleanup ``finally`` below still runs.
    signal.signal(signal.SIGTERM, _graceful_termination_handler)
    signal.signal(signal.SIGINT, _graceful_termination_handler)

    with FileLoggingContext(log_file_path=log_path, suppress_stdout=False):
        console_logger.info(
            "Starting one-room worker: room=%s start=%s stop=%s port_offset=%s",
            args.room_id,
            args.start_stage,
            args.stop_stage,
            args.port_offset,
        )
        console_logger.info(
            "OpenAI request resilience enabled: max_retries=%s timeout_s=%.1f",
            openai_max_retries,
            OPENAI_REQUEST_TIMEOUT_SECONDS,
        )
        started_services: list[str] = []
        try:
            started_services = _start_worker_services(experiment, args.asset_pipeline)
            scene = _generate_room(
                room_id=args.room_id,
                room_prompt=room_spec.prompt,
                room_geometry=room_geometry,
                room_dir=room_dir,
                logger=logger,
                cfg_dict=cfg_dict,
                start_stage=args.start_stage,
                stop_stage=args.stop_stage,
                house_layout=house_layout,
                render_gpu_id=args.render_gpu_id,
            )
            console_logger.info(
                "Completed one-room worker: room=%s objects=%s",
                args.room_id,
                len(scene.objects),
            )
        finally:
            _stop_worker_services(experiment, started_services)


if __name__ == "__main__":
    main()
