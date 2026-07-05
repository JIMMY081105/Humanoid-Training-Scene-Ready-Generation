"""Create a visual Blender scene and outlook renders from combined house_state.json."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector


TYPE_COLORS = {
    "floor": (0.58, 0.55, 0.50, 1.0),
    "wall": (0.82, 0.84, 0.82, 1.0),
    "furniture": (0.55, 0.48, 0.38, 1.0),
    "wall_mounted": (0.72, 0.76, 0.80, 1.0),
    "ceiling_mounted": (0.95, 0.92, 0.72, 1.0),
    "manipuland": (0.45, 0.58, 0.72, 1.0),
}


def material(name: str, color: tuple[float, float, float, float]):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = color
    return mat


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def room_positions(layout: dict) -> dict[str, Vector]:
    placed = layout.get("placed_rooms") or []
    positions: dict[str, Vector] = {}
    if isinstance(placed, dict):
        iterator = placed.values()
    else:
        iterator = placed
    for item in iterator:
        rid = item["room_id"]
        pos = item.get("position", [0.0, 0.0])
        positions[rid] = Vector((float(pos[0]), float(pos[1]), 0.0))
    return positions


def object_matrix(room_pos: Vector, transform: dict) -> Matrix:
    t = transform.get("translation", [0.0, 0.0, 0.0])
    q = transform.get("rotation_wxyz", [1.0, 0.0, 0.0, 0.0])
    loc = room_pos + Vector((float(t[0]), float(t[1]), float(t[2])))
    quat = Quaternion((float(q[0]), float(q[1]), float(q[2]), float(q[3])))
    return Matrix.Translation(loc) @ quat.to_matrix().to_4x4()


def add_box_object(room_pos: Vector, obj: dict, name: str) -> None:
    bbox_min = Vector(tuple(float(v) for v in obj.get("bbox_min", [-0.5, -0.5, -0.5])))
    bbox_max = Vector(tuple(float(v) for v in obj.get("bbox_max", [0.5, 0.5, 0.5])))
    dims = bbox_max - bbox_min
    if min(dims) <= 0:
        return

    bpy.ops.mesh.primitive_cube_add(size=1)
    cube = bpy.context.object
    cube.name = name
    cube.dimensions = (dims.x, dims.y, dims.z)
    center_offset = Matrix.Translation((bbox_min + bbox_max) * 0.5)
    cube.matrix_world = object_matrix(room_pos, obj.get("transform") or {}) @ center_offset
    mat = material(
        f"mat_{obj.get('object_type', 'object')}",
        TYPE_COLORS.get(obj.get("object_type"), (0.7, 0.7, 0.7, 1.0)),
    )
    cube.data.materials.append(mat)


def imported_objects_before() -> set[str]:
    return {obj.name for obj in bpy.context.scene.objects}


def import_gltf(path: Path, matrix: Matrix, name_prefix: str) -> int:
    before = imported_objects_before()
    try:
        bpy.ops.import_scene.gltf(filepath=str(path))
    except Exception:
        return 0
    new_objects = [obj for obj in bpy.context.scene.objects if obj.name not in before]
    for obj in new_objects:
        obj.name = f"{name_prefix}_{obj.name}"[:63]
        obj.matrix_world = matrix @ obj.matrix_world
    return len(new_objects)


def add_room_geometry_from_layout(layout: dict) -> None:
    wall_mat = material("mat_layout_wall", TYPE_COLORS["wall"])
    floor_mat = material("mat_layout_floor", TYPE_COLORS["floor"])
    for placed in layout.get("placed_rooms", []):
        rid = placed["room_id"]
        x, y = placed["position"]
        width = placed["width"]
        depth = placed["depth"]
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x + width / 2, y + depth / 2, -0.035))
        floor = bpy.context.object
        floor.name = f"{rid}_layout_floor"
        floor.dimensions = (width, depth, 0.07)
        floor.data.materials.append(floor_mat)
        for wall in placed.get("walls", []):
            sx, sy = wall["start_point"]
            ex, ey = wall["end_point"]
            length = max(((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5, 0.01)
            cx, cy = (sx + ex) / 2, (sy + ey) / 2
            horizontal = abs(ex - sx) >= abs(ey - sy)
            dims = (length, 0.05, 3.2) if horizontal else (0.05, length, 3.2)
            bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, cy, 1.6))
            wall_obj = bpy.context.object
            wall_obj.name = wall["wall_id"]
            wall_obj.dimensions = dims
            wall_obj.data.materials.append(wall_mat)


def build_scene(scene_dir: Path, state: dict) -> dict:
    positions = room_positions(state["layout"])
    add_room_geometry_from_layout(state["layout"])
    stats = {"imported_assets": 0, "box_objects": 0, "missing_assets": []}

    for room_id, room in state["rooms"].items():
        room_pos = positions.get(room_id, Vector((0.0, 0.0, 0.0)))
        room_dir = scene_dir / f"room_{room_id}"
        for object_id, obj in (room.get("objects") or {}).items():
            object_type = obj.get("object_type")
            if object_type in {"wall", "floor"}:
                continue
            matrix = object_matrix(room_pos, obj.get("transform") or {})
            geometry_path = obj.get("geometry_path")
            imported = 0
            if geometry_path:
                asset_path = room_dir / geometry_path
                if asset_path.exists():
                    imported = import_gltf(asset_path, matrix, f"{room_id}_{object_id}")
                else:
                    stats["missing_assets"].append(str(asset_path))
            if imported:
                stats["imported_assets"] += 1
            else:
                add_box_object(room_pos, obj, f"{room_id}_{object_id}")
                stats["box_objects"] += 1
    return stats


def scene_bounds() -> tuple[Vector, Vector]:
    objs = [obj for obj in bpy.context.scene.objects if obj.type in {"MESH", "EMPTY"}]
    mins = Vector((math.inf, math.inf, math.inf))
    maxs = Vector((-math.inf, -math.inf, -math.inf))
    for obj in objs:
        if obj.type == "EMPTY":
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            mins.x, mins.y, mins.z = min(mins.x, world.x), min(mins.y, world.y), min(
                mins.z, world.z
            )
            maxs.x, maxs.y, maxs.z = max(maxs.x, world.x), max(maxs.y, world.y), max(
                maxs.z, world.z
            )
    return mins, maxs


def render_views(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mins, maxs = scene_bounds()
    center = (mins + maxs) * 0.5
    span = max((maxs - mins).length, 1.0)

    bpy.ops.object.light_add(type="SUN", location=(center.x, center.y, center.z + span))
    bpy.context.object.data.energy = 2.5
    bpy.ops.object.light_add(type="AREA", location=(center.x, center.y - span * 0.25, center.z + span * 0.7))
    bpy.context.object.data.energy = 900
    bpy.context.object.data.size = max(span * 0.5, 8)

    scene = bpy.context.scene
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        pass

    def render(name: str, direction: tuple[float, float, float], scale: float, lens: float) -> None:
        bpy.ops.object.camera_add()
        cam = bpy.context.object
        cam.name = f"camera_{name}"
        direction_vec = Vector(direction).normalized()
        cam.location = center + direction_vec * span * scale
        cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
        cam.data.lens = lens
        cam.data.clip_end = max(span * 10, 1000)
        scene.camera = cam
        scene.render.filepath = str(out_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)

    render("overview_isometric", (1, -1, 0.78), 1.25, 28)
    render("overview_top", (0, 0, 1), 1.55, 35)
    render("overview_front", (0, -1, 0.28), 1.15, 24)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output-blend", required=True)
    parser.add_argument("--render-dir", required=True)
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    combined_dir = scene_dir / "combined_house"
    state_path = combined_dir / "house_state.json"
    state = json.loads(state_path.read_text())

    reset_scene()
    stats = build_scene(scene_dir=scene_dir, state=state)
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.output_blend).resolve()))
    render_views(Path(args.render_dir).resolve())

    manifest = {
        "source": str(state_path),
        "output_blend": str(Path(args.output_blend).resolve()),
        "render_dir": str(Path(args.render_dir).resolve()),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **stats,
    }
    (combined_dir / "direct_visual_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
