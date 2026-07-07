"""Assemble a completed room-sharded SceneSmith house and render outlook images."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path

import hydra
from omegaconf import OmegaConf, open_dict

from scenesmith.agent_utils.house import HouseLayout, HouseScene
from scenesmith.agent_utils.room import RoomScene
from scenesmith.utils.omegaconf import register_resolvers


ROOMS = [
    "main_corridor",
    "classroom_01",
    "classroom_02",
    "classroom_03",
    "classroom_04",
    "classroom_05",
    "classroom_06",
    "classroom_07",
    "classroom_08",
    "lobby_waiting_area",
    "stair_entrance",
    "teacher_office",
    "storage_room",
    "male_restroom",
    "female_restroom",
    "cleaning_closet",
    "library",
    "emergency_exit_corridor",
]

LOGGER = logging.getLogger(__name__)


def _load_cfg(repo_dir: Path, run_dir: Path, csv_path: str, run_name: str):
    register_resolvers()
    config_dir = repo_dir.resolve() / "configurations"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                f"+name={run_name}",
                f"experiment.csv_path={csv_path}",
                "experiment.num_workers=1",
                "experiment.pipeline.start_stage=wall_mounted",
                "experiment.pipeline.stop_stage=manipuland",
                "experiment.pipeline.parallel_rooms=false",
                "floor_plan_agent.mode=house",
                "codex.enabled=false",
            ],
        )

    with open_dict(cfg):
        cfg.experiment._name = "indoor_scene_generation"
        cfg.floor_plan_agent._name = "stateful_floor_plan_agent"
        cfg.furniture_agent._name = "stateful_furniture_agent"
        cfg.wall_agent._name = "stateful_wall_agent"
        cfg.ceiling_agent._name = "stateful_ceiling_agent"
        cfg.manipuland_agent._name = "stateful_manipuland_agent"
        cfg.experiment.output_dir = str(run_dir.resolve())

    OmegaConf.resolve(cfg)
    return cfg


def _verify_final_rooms(scene_dir: Path, room_ids: list[str]) -> list[str]:
    missing: list[str] = []
    for room_id in room_ids:
        final_state = (
            scene_dir
            / f"room_{room_id}"
            / "scene_states"
            / "final_scene"
            / "scene_state.json"
        )
        if not final_state.exists():
            missing.append(room_id)
    return missing


def _room_ids_from_layout(layout: HouseLayout) -> list[str]:
    room_ids = [room.room_id for room in layout.placed_rooms]
    return room_ids or ROOMS


def _verify_room_gates(gate_dir: Path, room_ids: list[str]) -> list[str]:
    failures: list[str] = []
    for room_id in room_ids:
        gate_path = gate_dir / f"{room_id}.json"
        if not gate_path.exists():
            failures.append(f"{room_id}: missing gate JSON")
            continue
        with gate_path.open() as f:
            result = json.load(f)
        if result.get("status") != "pass":
            failures.append(f"{room_id}: status={result.get('status')}")
    return failures


def _load_room(scene_dir: Path, room_id: str) -> RoomScene:
    room_dir = scene_dir / f"room_{room_id}"
    state_path = room_dir / "scene_states" / "final_scene" / "scene_state.json"
    with state_path.open() as f:
        state = json.load(f)

    room = RoomScene(room_geometry=None, scene_dir=room_dir, room_id=room_id)
    room.restore_from_state_dict(state)
    return room


def _backup_existing(path: Path) -> None:
    if not path.exists():
        return
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = path.with_name(f"{path.name}.pre_final_assemble_backup_{stamp}")
    LOGGER.info("Preserving existing %s as %s", path, backup)
    shutil.move(str(path), str(backup))


def _render_blend_overviews(blend_path: Path, output_dir: Path) -> None:
    try:
        import bpy
        from mathutils import Vector
    except Exception as exc:
        LOGGER.warning("Skipping outlook renders because bpy is unavailable: %s", exc)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objects:
        LOGGER.warning("No mesh objects found in %s; skipping outlook renders", blend_path)
        return

    mins = Vector((math.inf, math.inf, math.inf))
    maxs = Vector((-math.inf, -math.inf, -math.inf))
    for obj in objects:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            mins.x, mins.y, mins.z = min(mins.x, world.x), min(mins.y, world.y), min(
                mins.z, world.z
            )
            maxs.x, maxs.y, maxs.z = max(maxs.x, world.x), max(maxs.y, world.y), max(
                maxs.z, world.z
            )

    center = (mins + maxs) * 0.5
    span = max((maxs - mins).length, 1.0)

    for obj in list(bpy.context.scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.object.light_add(type="SUN", location=(center.x, center.y, center.z + span))
    bpy.context.object.data.energy = 3.0
    bpy.ops.object.light_add(
        type="AREA", location=(center.x, center.y - span * 0.35, center.z + span)
    )
    bpy.context.object.data.energy = 600.0
    bpy.context.object.data.size = max(span * 0.6, 5.0)

    scene = bpy.context.scene
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.film_transparent = False
    if hasattr(scene, "eevee"):
        scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in {
            item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items
        } else "BLENDER_EEVEE"

    def render_view(name: str, direction: tuple[float, float, float], scale: float) -> None:
        camera = bpy.data.objects.get(f"camera_{name}")
        if camera is None:
            bpy.ops.object.camera_add()
            camera = bpy.context.object
            camera.name = f"camera_{name}"
        direction_vec = Vector(direction).normalized()
        camera.location = center + direction_vec * span * scale
        target = center
        track = (target - camera.location).to_track_quat("-Z", "Y")
        camera.rotation_euler = track.to_euler()
        camera.data.lens = 24
        camera.data.clip_end = max(span * 8, 1000)
        scene.camera = camera
        scene.render.filepath = str(output_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        LOGGER.info("Rendered outlook image: %s", scene.render.filepath)

    render_view("overview_isometric", (1.0, -1.0, 0.75), 1.4)
    render_view("overview_top", (0.0, 0.0, 1.0), 1.7)
    render_view("overview_front", (0.0, -1.0, 0.35), 1.25)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--csv",
        default="inputs/full_school_floor_20260703.csv",
        help="Prompt CSV used for this run. Must match the floor-plan/room stage.",
    )
    parser.add_argument(
        "--run-name",
        default="scenesmith_final_assemble",
        help="Hydra run name for assembly.",
    )
    parser.add_argument(
        "--gate-dir",
        help=(
            "Directory containing room self-exam JSON files. Defaults to "
            "scene_000/quality_gates/room_self_exam."
        ),
    )
    parser.add_argument(
        "--allow-ungated",
        action="store_true",
        help="Bypass room self-exam gate. Use only for debugging, never final runs.",
    )
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOGLEVEL", "INFO").upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    repo_dir = Path(args.repo_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    scene_dir = run_dir / "scene_000"

    layout_path = scene_dir / "house_layout.json"
    with layout_path.open() as f:
        layout = HouseLayout.from_dict(json.load(f), house_dir=scene_dir)
    room_ids = _room_ids_from_layout(layout)

    missing = _verify_final_rooms(scene_dir, room_ids)
    if missing:
        raise RuntimeError(f"Refusing to assemble: missing final rooms: {missing}")

    gate_dir = (
        Path(args.gate_dir).resolve()
        if args.gate_dir
        else scene_dir / "quality_gates" / "room_self_exam"
    )
    if not args.allow_ungated:
        gate_failures = _verify_room_gates(gate_dir, room_ids)
        if gate_failures:
            raise RuntimeError(
                "Refusing to assemble: room self-exam gate has not passed: "
                + "; ".join(gate_failures)
            )

    cfg = _load_cfg(
        repo_dir=repo_dir, run_dir=run_dir, csv_path=args.csv, run_name=args.run_name
    )
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    rooms = {room_id: _load_room(scene_dir, room_id) for room_id in room_ids}
    house = HouseScene(layout=layout, rooms=rooms)

    combined_dir = scene_dir / "combined_house"
    _backup_existing(combined_dir)
    combined_dir = house.assemble(cfg=cfg_dict, output_name="combined_house")
    LOGGER.info("Assembled final combined house at %s", combined_dir)

    required = [
        combined_dir / "house_state.json",
        combined_dir / "sceneeval_state.json",
        combined_dir / "house.dmd.yaml",
        combined_dir / "house.blend",
    ]
    missing_outputs = [str(path) for path in required if not path.exists()]
    if missing_outputs:
        raise RuntimeError(f"Combined export missing outputs: {missing_outputs}")

    if args.render:
        try:
            _render_blend_overviews(
                blend_path=combined_dir / "house.blend",
                output_dir=combined_dir / "outlook_renders",
            )
        except Exception as exc:
            LOGGER.exception("Outlook render step failed after successful export: %s", exc)

    LOGGER.info("Final assembly/postprocess completed")


if __name__ == "__main__":
    main()
