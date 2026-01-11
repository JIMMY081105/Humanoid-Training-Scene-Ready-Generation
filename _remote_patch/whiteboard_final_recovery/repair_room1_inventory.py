#!/usr/bin/env python3
"""Repair the narrowly failed Room 1 inventory gate without rerunning generation.

This helper retrieves one real ObjectThor ruler through SceneSmith's ordinary
asset pipeline, places it on the existing teacher-desk support surface, runs
the same local and final physical gates as the room worker, then atomically
publishes a new final-scene directory.  It deliberately never regenerates or
mutates the existing furniture/manipuland assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
import uuid

from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _publish_staged_final(*, staged_dir: Path, final_dir: Path) -> Path:
    """Swap an already-validated final directory while preserving the prior one."""
    if not staged_dir.is_dir():
        raise RuntimeError(f"Repair staging directory is missing: {staged_dir}")
    for name in ("scene_state.json", "scene.dmd.yaml", "scene.blend"):
        candidate = staged_dir / name
        if not candidate.is_file() or candidate.stat().st_size == 0:
            raise RuntimeError(f"Staged final artifact is missing or empty: {candidate}")

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup_dir = final_dir.with_name(f"final_scene.backup_inventory_repair_{stamp}")
    if backup_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing final-scene backup: {backup_dir}")
    os.replace(final_dir, backup_dir)
    try:
        os.replace(staged_dir, final_dir)
    except Exception:
        os.replace(backup_dir, final_dir)
        raise
    return backup_dir


def _worker_config(args: argparse.Namespace):
    """Resolve the normal full-quality worker configuration for this repair."""
    from run_single_room_worker import _configure_ports, _load_cfg

    worker_args = SimpleNamespace(
        repo_dir=str(args.repo_dir),
        run_dir=str(args.run_dir),
        csv=str(args.csv),
        run_name="room1_inventory_repair",
        start_stage="manipuland",
        stop_stage="manipuland",
        asset_pipeline="generated_sam3d",
        port_offset=args.port_offset,
        artiverse_data="data/artiverse",
        artiverse_embeddings="data/artiverse/embeddings",
        artvip_data="data/artvip_sdf",
        artvip_embeddings="data/artvip_sdf/embeddings",
        materials_data="data/materials",
        materials_embeddings="data/materials_full_quality_contract/embeddings",
    )
    cfg = _load_cfg(worker_args)
    _configure_ports(cfg, args.port_offset)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--csv",
        default="inputs/full_school_floor_20260703.csv",
        help="The canonical floor-plan prompt CSV.",
    )
    parser.add_argument("--room-id", default="classroom_01")
    parser.add_argument("--port-offset", type=int, default=70)
    parser.add_argument("--render-gpu-id", type=int, default=0)
    args = parser.parse_args()

    # This must precede SceneSmith imports, matching the normal room worker.
    from run_single_room_worker import _configure_pytorch_cuda_allocator

    _configure_pytorch_cuda_allocator()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from omegaconf import OmegaConf

    from scenesmith.agent_utils.asset_manager import AssetGenerationRequest
    from scenesmith.agent_utils.physics_validation import compute_scene_collisions
    from scenesmith.agent_utils.physical_feasibility import (
        apply_per_furniture_postprocessing,
        apply_physical_feasibility_postprocessing,
        clone_scene_with_collision_repairs,
    )
    from scenesmith.agent_utils.room import ObjectType, RoomScene, UniqueID
    from scenesmith.agent_utils.rendering import save_scene_as_blend
    from scenesmith.experiments.base_experiment import BaseExperiment
    from scenesmith.experiments.indoor_scene_generation import (
        IndoorSceneGenerationExperiment,
    )
    from scenesmith.manipuland_agents.tools.manipuland_tools import ManipulandTools
    from scenesmith.utils.logging import ConsoleLogger

    repo_dir = args.repo_dir.resolve()
    run_dir = args.run_dir.resolve()
    # Hydra/ObjectThor configuration contains repository-relative asset paths.
    # Detached invocations otherwise inherit /root and incorrectly resolve the
    # cache as /root/data/objathor-assets instead of this execution checkout.
    os.chdir(repo_dir)
    room_dir = run_dir / "scene_000" / f"room_{args.room_id}"
    final_dir = room_dir / "scene_states" / "final_scene"
    state_path = final_dir / "scene_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Room final state is missing: {state_path}")

    cfg = _worker_config(args)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    logger = ConsoleLogger(output_dir=room_dir)
    experiment = IndoorSceneGenerationExperiment(cfg)
    agent = None
    stage_dir: Path | None = None
    prior_state_sha256 = _sha256_file(state_path)
    try:
        # Only ObjectThor retrieval is needed.  The full agent still supplies the
        # production Blender, VLM physics analysis, CoACD, SDF, and placement
        # checks; it does not call SAM3D or regenerate any prior object.
        agent = BaseExperiment.build_manipuland_agent(
            cfg_dict=cfg_dict,
            compatible_agents=IndoorSceneGenerationExperiment.compatible_manipuland_agents,
            logger=logger,
            render_gpu_id=args.render_gpu_id,
        )

        with state_path.open(encoding="utf-8") as handle:
            original_state = json.load(handle)
        scene = RoomScene(room_geometry=None, scene_dir=room_dir, room_id=args.room_id)
        scene.restore_from_state_dict(original_state)
        agent.scene = scene

        teacher_desk_id = UniqueID("teacher_desk_0")
        teacher_desk = scene.get_object(teacher_desk_id)
        if teacher_desk is None:
            raise RuntimeError("Room 1 has no teacher_desk_0 for ruler placement")
        support_surfaces = {
            str(surface.surface_id): surface for surface in teacher_desk.support_surfaces
        }
        if "S_0" not in support_surfaces:
            raise RuntimeError(
                "Room 1 teacher desk does not expose the attested top support surface S_0"
            )

        asset_manager = agent.asset_manager
        if asset_manager.general_asset_source != "objaverse":
            raise RuntimeError(
                "Inventory repair requires the full-quality ObjectThor route, got "
                f"{asset_manager.general_asset_source!r}"
            )
        # Keep the retrieval deterministic and narrow: use the configured real
        # ObjectThor source directly, while retaining its normal VLM/collision/SDF
        # processing.  Routing would add unrelated generative strategy choices.
        asset_manager.router = None
        ruler_asset = next(
            (
                candidate
                for candidate in asset_manager.list_available_assets()
                if candidate.name == "classroom_ruler"
                and candidate.metadata.get("asset_source") == "objaverse"
                and candidate.geometry_path
                and Path(candidate.geometry_path).is_file()
                and candidate.sdf_path
                and Path(candidate.sdf_path).is_file()
            ),
            None,
        )
        asset_acquisition = "registry_reuse"
        if ruler_asset is None:
            experiment._start_objaverse_server()
            retrieval = asset_manager.generate_assets(
                AssetGenerationRequest(
                    object_descriptions=[
                        "wooden 30 centimeter classroom ruler with clear metric markings, "
                        "lying flat for a teacher desk"
                    ],
                    short_names=["classroom_ruler"],
                    object_type=ObjectType.MANIPULAND,
                    desired_dimensions=[[0.30, 0.04, 0.006]],
                    style_context="warm, practical primary-school classroom",
                    scene_id=args.room_id,
                )
            )
            if retrieval.failed_assets or len(retrieval.successful_assets) != 1:
                details = [failure.error_message for failure in retrieval.failed_assets]
                raise RuntimeError(
                    "Expected exactly one retrievable ObjectThor ruler; "
                    f"successes={len(retrieval.successful_assets)} failures={details}"
                )
            ruler_asset = retrieval.successful_assets[0]
            asset_acquisition = "objectthor_retrieval"
        if ruler_asset.metadata.get("asset_source") != "objaverse":
            raise RuntimeError("Retrieved ruler does not carry ObjectThor provenance")
        if not ruler_asset.geometry_path or not Path(ruler_asset.geometry_path).is_file():
            raise RuntimeError("Retrieved ruler visual mesh is missing")
        if not ruler_asset.sdf_path or not Path(ruler_asset.sdf_path).is_file():
            raise RuntimeError("Retrieved ruler collision SDF is missing")

        protected_ids = set(scene.objects)
        placement_tools = ManipulandTools(
            scene=scene,
            asset_manager=asset_manager,
            cfg=agent.cfg,
            current_furniture_id=teacher_desk_id,
            support_surfaces=support_surfaces,
            protected_object_ids=protected_ids,
        )
        # S_0 spans 1.44m x 0.65m.  This free corner is clear of the retained
        # laptop, gradebook, trays, notebook stacks, and pencil cup.
        placement = json.loads(
            placement_tools._place_manipuland_on_surface_impl(
                asset_id=str(ruler_asset.object_id),
                surface_id="S_0",
                position_x=0.30,
                position_z=0.16,
                rotation_degrees=90.0,
            )
        )
        if not placement.get("success"):
            raise RuntimeError(f"Ruler placement was rejected: {placement}")
        ruler_id = UniqueID(str(placement["object_id"]))
        ruler = scene.get_object(ruler_id)
        if ruler is None or ruler.placement_info is None:
            raise RuntimeError("Ruler placement reported success but no placed object exists")
        if str(ruler.placement_info.parent_surface_id) != "S_0":
            raise RuntimeError("Ruler was not retained on the teacher desk surface")
        if "ruler" not in f"{ruler.name} {ruler.description}".lower():
            raise RuntimeError("Ruler semantic labeling was not retained")

        # Require both the per-support dynamic gate and the final whole-room
        # projection gate.  Neither pass is allowed to remove the ruler.
        scene = apply_per_furniture_postprocessing(
            full_scene=scene,
            furniture_id=teacher_desk_id,
            manipuland_ids=[ruler_id],
            config=agent.cfg.per_furniture_postprocessing,
            simulation_html_path=(room_dir / "simulation" / "room1_ruler_repair.html"),
        )
        physics_cfg = cfg_dict["manipuland_agent"]["physics_validation"]
        projection_cfg = cfg_dict["experiment"]["projection"]
        final_cfg = projection_cfg["final"]
        simulation_cfg = projection_cfg["simulation"]
        scene, projection_success, removed_ids = apply_physical_feasibility_postprocessing(
            scene=scene,
            weld_furniture=True,
            projection_enabled=True,
            projection_influence_distance=final_cfg["influence_distance"],
            projection_solver_name=final_cfg["solver_name"],
            projection_iteration_limit=final_cfg["iteration_limit"],
            projection_time_limit_s=final_cfg["time_limit_s"],
            projection_xy_only=final_cfg["xy_only"],
            projection_fix_rotation=final_cfg["fix_rotation"],
            simulation_enabled=simulation_cfg["enabled"],
            simulation_time_s=simulation_cfg["simulation_time_s"],
            simulation_time_step_s=simulation_cfg["time_step_s"],
            simulation_timeout_s=simulation_cfg["timeout_s"],
            remove_fallen_furniture=False,
            remove_fallen_manipulands=physics_cfg["remove_fallen_manipulands"],
            fallen_manipuland_floor_z=physics_cfg["fallen_manipuland_floor_z"],
            fallen_manipuland_near_floor_z=physics_cfg["fallen_manipuland_near_floor_z"],
            fallen_manipuland_z_displacement=physics_cfg["fallen_manipuland_z_displacement"],
        )
        if not projection_success or ruler_id in removed_ids or scene.get_object(ruler_id) is None:
            raise RuntimeError(
                "Final physical validation did not retain the ruler: "
                f"projection_success={projection_success} removed={removed_ids}"
            )
        ruler_collisions = placement_tools._compute_object_collisions(ruler_id)
        if ruler_collisions:
            raise RuntimeError(
                "Ruler collision validation failed after physical processing: "
                + "; ".join(collision.to_description() for collision in ruler_collisions)
            )

        stage_dir = final_dir.parent / f".room1_inventory_repair_{uuid.uuid4().hex}"
        stage_dir.mkdir(parents=False, exist_ok=False)
        repaired_state = scene.to_state_dict()
        repaired_state["timestamp"] = time.time()
        _write_json(stage_dir / "scene_state.json", repaired_state)
        (stage_dir / "scene.dmd.yaml").write_text(
            scene.to_drake_directive(), encoding="utf-8"
        )

        # The repair agent owns a Blender server for asset canonicalization; stop
        # it before the independent staged blend exporter starts its own server.
        agent.cleanup()
        agent = None
        experiment._stop_objaverse_server()

        rendering_cfg = cfg_dict["furniture_agent"]["rendering"]
        save_scene_as_blend(
            scene=scene,
            output_path=stage_dir / "scene.blend",
            blender_server_host=rendering_cfg.get("blender_server_host", "127.0.0.1"),
            blender_server_port_range=tuple(rendering_cfg["blender_server_port_range"]),
            server_startup_delay=rendering_cfg["server_startup_delay"],
            port_cleanup_delay=rendering_cfg["port_cleanup_delay"],
        )
        backup_dir = _publish_staged_final(staged_dir=stage_dir, final_dir=final_dir)
        stage_dir = None

        receipt = {
            "status": "pass",
            "room_id": args.room_id,
            "repair": "real_objectthor_ruler_on_existing_teacher_desk_support",
            "prior_final_state_sha256": prior_state_sha256,
            "final_state_sha256": _sha256_file(final_dir / "scene_state.json"),
            "backup_final_scene": str(backup_dir),
            "ruler": {
                "object_id": str(ruler_id),
                "name": ruler.name,
                "description": ruler.description,
                "asset_source": ruler.metadata.get("asset_source"),
                "acquisition": asset_acquisition,
                "objaverse_uid": ruler.metadata.get("objaverse_uid"),
                "geometry_path": str(ruler.geometry_path),
                "sdf_path": str(ruler.sdf_path),
                "parent_surface_id": str(ruler.placement_info.parent_surface_id),
            },
            "placement": placement,
            "physical_validation": {
                "final_projection_success": projection_success,
                "removed_ids": [str(object_id) for object_id in removed_ids],
                "ruler_collision_count": len(ruler_collisions),
            },
        }
        _write_json(room_dir / "quality_gates" / "room1_inventory_repair.json", receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
    finally:
        if agent is not None:
            agent.cleanup()
        if experiment.objaverse_server and experiment.objaverse_server.is_running():
            experiment._stop_objaverse_server()
        # Keep an unpublished staging directory on failure as repair evidence.


if __name__ == "__main__":
    main()
