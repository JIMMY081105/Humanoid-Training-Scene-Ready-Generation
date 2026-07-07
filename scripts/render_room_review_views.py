"""Render per-room top-down review images from a combined visual Blender file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import bpy
from mathutils import Vector


def clear_existing_cameras() -> None:
    for obj in list(bpy.context.scene.objects):
        if obj.type == "CAMERA" and obj.name.startswith("room_review_camera"):
            bpy.data.objects.remove(obj, do_unlink=True)


def set_render_defaults() -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 1200
    scene.render.film_transparent = False
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        pass


def ensure_lighting() -> None:
    if any(obj.type == "LIGHT" for obj in bpy.context.scene.objects):
        return
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 40))
    bpy.context.object.data.energy = 2.5


def render_room(room: dict, output_dir: Path) -> None:
    room_id = room["room_id"]
    x, y = room["position"]
    width = float(room["width"])
    depth = float(room["depth"])
    center = Vector((float(x) + width / 2.0, float(y) + depth / 2.0, 0.0))
    scale = max(width, depth) * 1.18

    bpy.ops.object.camera_add(location=(center.x, center.y, 40.0))
    camera = bpy.context.object
    camera.name = f"room_review_camera_{room_id}"
    camera.rotation_euler = (0.0, 0.0, 0.0)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = scale
    camera.data.clip_end = 200.0

    scene = bpy.context.scene
    scene.camera = camera
    scene.render.filepath = str(output_dir / f"{room_id}_top.png")
    bpy.ops.render.render(write_still=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", required=True)
    parser.add_argument("--house-state", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    blend_path = Path(args.blend).resolve()
    state_path = Path(args.house_state).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    clear_existing_cameras()
    set_render_defaults()
    ensure_lighting()

    state = json.loads(state_path.read_text())
    placed_rooms = state["layout"].get("placed_rooms", [])
    if isinstance(placed_rooms, dict):
        placed_rooms = list(placed_rooms.values())

    for room in placed_rooms:
        render_room(room, output_dir)


if __name__ == "__main__":
    main()
