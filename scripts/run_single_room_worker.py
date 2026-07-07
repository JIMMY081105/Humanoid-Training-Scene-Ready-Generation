"""Run exactly one SceneSmith room stage pipeline inside an existing house output.

This is intentionally narrower than main.py: it loads an existing house layout and
room checkpoint, runs one room only, and never assembles combined_house.
"""

import argparse
import logging
import os
from pathlib import Path

import hydra

from omegaconf import OmegaConf, open_dict

from scenesmith.agent_utils.house import HouseLayout
from scenesmith.experiments.indoor_scene_generation import (
    IndoorSceneGenerationExperiment,
    _generate_room,
)
from scenesmith.utils.logging import ConsoleLogger, FileLoggingContext
from scenesmith.utils.omegaconf import register_resolvers


console_logger = logging.getLogger(__name__)


def _shift_range(values: list[int], offset: int) -> list[int]:
    return [int(values[0]) + offset, int(values[1]) + offset]


def _configure_ports(cfg, offset: int) -> None:
    """Move all localhost servers/ranges away from other shard workers."""
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


def _asset_pipeline_overrides(asset_pipeline: str) -> list[str]:
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
            "furniture_agent.asset_manager.artiverse_articulated.enabled=true",
            "furniture_agent.asset_manager.artiverse_articulated.data_path=data/artiverse",
            "furniture_agent.asset_manager.router.strategies.artiverse_articulated.enabled=true",
            "wall_agent.asset_manager.router.strategies.artiverse_articulated.enabled=false",
            "ceiling_agent.asset_manager.router.strategies.artiverse_articulated.enabled=false",
            "manipuland_agent.asset_manager.router.strategies.artiverse_articulated.enabled=false",
        ]

    if asset_pipeline == "generated_sam3d":
        return common + [
            "furniture_agent.asset_manager.general_asset_source=generated",
            "wall_agent.asset_manager.general_asset_source=generated",
            "ceiling_agent.asset_manager.general_asset_source=generated",
            "manipuland_agent.asset_manager.general_asset_source=generated",
            "furniture_agent.asset_manager.backend=sam3d",
            "wall_agent.asset_manager.backend=sam3d",
            "ceiling_agent.asset_manager.backend=sam3d",
            "manipuland_agent.asset_manager.backend=sam3d",
            "furniture_agent.asset_manager.router.strategies.generated.enabled=true",
            "wall_agent.asset_manager.router.strategies.generated.enabled=true",
            "ceiling_agent.asset_manager.router.strategies.generated.enabled=true",
            "manipuland_agent.asset_manager.router.strategies.generated.enabled=true",
            "furniture_agent.asset_manager.artiverse_articulated.enabled=true",
            "furniture_agent.asset_manager.artiverse_articulated.data_path=data/artiverse",
            "furniture_agent.asset_manager.router.strategies.artiverse_articulated.enabled=true",
        ]

    raise ValueError(f"Unsupported asset pipeline: {asset_pipeline}")


def _load_cfg(args: argparse.Namespace):
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
                "floor_plan_agent.mode=house",
                "codex.enabled=true",
                f"codex.cwd={args.repo_dir}",
                "codex.timeout_seconds=1800",
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
            + _asset_pipeline_overrides(args.asset_pipeline),
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


def main() -> None:
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
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOGLEVEL", "INFO").upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from agents import set_default_openai_api, set_tracing_disabled

    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)

    os.environ.setdefault("SCENESMITH_MANIPULAND_MAX_FURNITURE", "3")

    cfg = _load_cfg(args)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    run_dir = Path(args.run_dir).resolve()
    scene_dir = run_dir / "scene_000"
    room_dir = scene_dir / f"room_{args.room_id}"
    if not room_dir.exists():
        raise FileNotFoundError(f"Room directory not found: {room_dir}")

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

    with FileLoggingContext(log_file_path=log_path, suppress_stdout=False):
        console_logger.info(
            "Starting one-room worker: room=%s start=%s stop=%s port_offset=%s",
            args.room_id,
            args.start_stage,
            args.stop_stage,
            args.port_offset,
        )
        try:
            experiment._start_hssd_server()
            experiment._start_materials_server()
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
            experiment._stop_materials_server()
            experiment._stop_hssd_server()


if __name__ == "__main__":
    main()
