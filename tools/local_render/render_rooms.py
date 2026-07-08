"""Render eye-level perspective views of SceneSmith rooms from house.blend.

Usage (after blender -b house.blend -P render_rooms.py --):
  --rooms classroom_01,library   rooms to render (or 'all')
  --out <dir>                    output directory
  --views N                      views per room (1 or 2, default 2)
  --samples N                    cycles samples (default 64)
"""
import bpy
import math
import re
import sys
import os
from mathutils import Vector

ROOMS_ALL = [
    "classroom_01", "classroom_02", "classroom_03", "classroom_04",
    "classroom_05", "classroom_06", "classroom_07", "classroom_08",
    "cleaning_closet", "emergency_exit_corridor", "female_restroom",
    "library", "lobby_waiting_area", "main_corridor", "male_restroom",
    "stair_entrance", "storage_room", "teacher_office", "FLOOR",
]

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
opts = {"rooms": "classroom_01", "out": ".", "views": "2", "samples": "64"}
for i in range(0, len(argv), 2):
    opts[argv[i].lstrip("-")] = argv[i + 1]

rooms = ROOMS_ALL if opts["rooms"] == "all" else opts["rooms"].split(",")
outdir = opts["out"]
nviews = int(opts["views"])
os.makedirs(outdir, exist_ok=True)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = int(opts["samples"])
scene.cycles.use_denoising = True
scene.cycles.texture_limit_render = "1024"  # keep 8GB laptop GPU within VRAM
scene.render.use_persistent_data = False
scene.render.resolution_x = 1600
scene.render.resolution_y = 1200
scene.render.image_settings.file_format = "PNG"

prefs = bpy.context.preferences.addons["cycles"].preferences
for dev_type in ("OPTIX", "CUDA"):
    try:
        prefs.compute_device_type = dev_type
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == dev_type]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == dev_type
            scene.cycles.device = "GPU"
            print(f"USING {dev_type}: {[d.name for d in gpus]}")
            break
    except Exception as e:
        print("device setup failed:", dev_type, e)
else:
    scene.cycles.device = "CPU"
    print("USING CPU")

# soft ambient so nothing is pitch black
world = bpy.data.worlds.new("render_world") if not scene.world else scene.world
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg:
    # rooms have no ceiling geometry: the background IS the visible ceiling
    bg.inputs[0].default_value = (0.90, 0.89, 0.87, 1.0)
    bg.inputs[1].default_value = 0.55

# punchier tone mapping, closer to archviz look
for look in ("AgX - Punchy", "Punchy", "AgX - Medium High Contrast", "Medium High Contrast"):
    try:
        scene.view_settings.look = look
        break
    except TypeError:
        continue


def _new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    return m, m.node_tree, m.node_tree.nodes["Principled BSDF"]


def make_wood_floor():
    m, nt, bsdf = _new_mat("shell_wood_floor")
    tc = nt.nodes.new("ShaderNodeNewGeometry")  # world position: object coords collapse on unit-cube slabs
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (0.35, 2.2, 0.35)  # stretch grain along x
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.inputs["Scale"].default_value = 0.7
    wave.inputs["Distortion"].default_value = 2.5
    wave.inputs["Detail"].default_value = 2.0
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 6.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.30, 0.19, 0.11, 1)
    ramp.color_ramp.elements[1].color = (0.46, 0.31, 0.19, 1)
    mid = ramp.color_ramp.elements.new(0.55)
    mid.color = (0.38, 0.25, 0.14, 1)
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "FLOAT"
    mix.inputs["Factor"].default_value = 0.5
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.06
    ln = nt.links
    ln.new(tc.outputs["Position"], mp.inputs["Vector"])
    ln.new(mp.outputs["Vector"], wave.inputs["Vector"])
    ln.new(mp.outputs["Vector"], noise.inputs["Vector"])
    ln.new(wave.outputs["Fac"], mix.inputs["A"])
    ln.new(noise.outputs["Fac"], mix.inputs["B"])
    ln.new(mix.outputs["Result"], ramp.inputs["Fac"])
    ln.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    ln.new(noise.outputs["Fac"], bump.inputs["Height"])
    ln.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Roughness"].default_value = 0.38
    return m


def make_plaster_wall():
    m, nt, bsdf = _new_mat("shell_plaster_wall")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 35.0
    noise.inputs["Detail"].default_value = 6.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.035
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Base Color"].default_value = (0.60, 0.585, 0.56, 1)
    bsdf.inputs["Roughness"].default_value = 0.85
    return m


def make_ceiling_white():
    m, nt, bsdf = _new_mat("shell_ceiling_white")
    bsdf.inputs["Base Color"].default_value = (0.87, 0.87, 0.86, 1)
    bsdf.inputs["Roughness"].default_value = 0.9
    return m


WOOD = make_wood_floor()
PLASTER = make_plaster_wall()
CEIL_WHITE = make_ceiling_white()
WALL_DIRS = {"north", "south", "east", "west"}


def dress_shell(room, objs):
    """Give the bare room shell (floor/walls/ceiling) archviz materials."""
    for o in objs:
        if o.type != "MESH":
            continue
        rest = re.sub(r"\.\d+$", "", o.name[len(room) + 1:])
        mat = None
        if "layout_floor" in rest:
            mat = WOOD
        elif rest in WALL_DIRS or re.fullmatch(r"(north|south|east|west)_\d+", rest):
            mat = PLASTER
        elif "ceiling" in rest and "light" not in rest:
            mat = CEIL_WHITE
        if mat is not None:
            o.data.materials.clear()
            o.data.materials.append(mat)

room_objs = {r: [] for r in rooms}
sorted_rooms = sorted(rooms, key=len, reverse=True)  # longest prefix wins
for o in bpy.data.objects:
    for r in sorted_rooms:
        if o.name.startswith(r + "_"):
            room_objs[r].append(o)
            break

cam_data = bpy.data.cameras.new("render_cam")
cam_data.lens = 24
cam_data.clip_start = 0.05
cam_data.clip_end = 500
cam = bpy.data.objects.new("render_cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

light_data = bpy.data.lights.new("room_light", type="AREA")
light_data.energy = 400
light_data.shape = "RECTANGLE"
light_data.color = (1.0, 0.97, 0.92)
light = bpy.data.objects.new("room_light", light_data)
light.visible_camera = False
scene.collection.objects.link(light)

# warm side light: fakes low sun through a window for directional shadows
warm_data = bpy.data.lights.new("warm_window", type="AREA")
warm_data.shape = "RECTANGLE"
warm_data.color = (1.0, 0.86, 0.70)
warm = bpy.data.objects.new("warm_window", warm_data)
warm.visible_camera = False
scene.collection.objects.link(warm)

sun_data = bpy.data.lights.new("floor_sun", type="SUN")
sun_data.energy = 3.0
sun_data.angle = 0.2
sun = bpy.data.objects.new("floor_sun", sun_data)
sun.rotation_euler = (math.radians(50), 0, math.radians(30))
sun.hide_render = True
scene.collection.objects.link(sun)


def bbox_of(objs):
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for o in objs:
        if o.type != "MESH":
            continue
        for corner in o.bound_box:
            w = o.matrix_world @ Vector(corner)
            lo = Vector(map(min, lo, w))
            hi = Vector(map(max, hi, w))
    return lo, hi


def aim(cam_obj, frm, to):
    cam_obj.location = frm
    direction = Vector(to) - Vector(frm)
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


for r in rooms:
    if r == "FLOOR":
        # whole-floor perspective: everything visible except ceiling slabs.
        # decoding all 1300+ packed images at once exhausts RAM, so bake each
        # image to its average color (one at a time), then purge all images.
        import numpy as np

        avg = {}
        for img in bpy.data.images:
            try:
                n = img.size[0] * img.size[1]
                if n == 0:
                    continue
                arr = np.empty(n * 4, dtype=np.float32)
                img.pixels.foreach_get(arr)
                rgb = arr.reshape(-1, 4)[:, :3].mean(axis=0)
                avg[img.name] = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
            except Exception as e:
                print("avg fail:", img.name, e)
            finally:
                img.buffers_free()
        print(f"averaged {len(avg)} images")

        trees = [m.node_tree for m in bpy.data.materials if m.use_nodes]
        trees += list(bpy.data.node_groups)
        for nt in trees:
            for node in [n for n in nt.nodes if n.type == "TEX_IMAGE"]:
                col = avg.get(node.image.name) if node.image else None
                for out in node.outputs:
                    for link in list(out.links):
                        sock = link.to_socket
                        nt.links.remove(link)
                        try:
                            if col is None:
                                pass
                            elif sock.type == "RGBA":
                                sock.default_value = col
                            elif sock.type == "VALUE":
                                sock.default_value = sum(col[:3]) / 3
                        except Exception:
                            pass
                nt.nodes.remove(node)
        for img in list(bpy.data.images):
            bpy.data.images.remove(img)
        print("textures stripped")

        for o in bpy.data.objects:
            if o.type == "MESH":
                o.hide_render = "ceiling" in o.name.lower() and "light" not in o.name.lower()
        light.hide_render = True
        warm.hide_render = True
        sun.hide_render = False
        prefixes = tuple(r + "_" for r in ROOMS_ALL if r != "FLOOR")
        mains = [o for o in bpy.data.objects if o.type == "MESH"
                 and not o.hide_render and o.name.startswith(prefixes)]
        lo, hi = bbox_of(mains)
        print(f"FLOOR bbox: {lo[:]} -> {hi[:]}")
        cx, cy = (lo.x + hi.x) / 2, (lo.y + hi.y) / 2
        diag = max(hi.x - lo.x, hi.y - lo.y)
        target = (cx, cy, lo.z + 1.0)
        corners8 = [(x, y, z) for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)]
        coords = [c for corner in corners8 for c in corner]
        cam_data.lens = 32
        scene.render.resolution_x = 1920
        scene.render.resolution_y = 1080
        from bpy_extras.object_utils import world_to_camera_view
        for vi, (sx, sy) in enumerate([(-0.55, -0.75), (0.55, 0.75)]):
            direction = Vector((sx * diag, sy * diag, diag * 0.85)).normalized()
            dist = diag * 0.4
            for _ in range(60):
                aim(cam, Vector(target) + direction * dist, target)
                bpy.context.view_layer.update()
                proj = [world_to_camera_view(scene, cam, Vector(c)) for c in corners8]
                if all(0.03 <= p.x <= 0.97 and 0.03 <= p.y <= 0.97 and p.z > 0 for p in proj):
                    break
                dist *= 1.08
            print(f"floor cam dist {dist:.1f}")
            scene.render.filepath = os.path.join(outdir, f"whole_floor_perspective_{vi + 1}.png")
            print(f"RENDERING {scene.render.filepath}")
            bpy.ops.render.render(write_still=True)
        cam_data.lens = 20
        scene.render.resolution_x = 1600
        scene.render.resolution_y = 1200
        light.hide_render = False
        sun.hide_render = True
        continue

    objs = room_objs[r]
    if not objs:
        print(f"SKIP {r}: no objects")
        continue
    # only this room visible: hide EVERY object (any type — stray global
    # debris floats above the building), then unhide room + our cam/lights
    for o in bpy.data.objects:
        o.hide_render = True
    for o in objs:
        o.hide_render = False
    for o in (cam, light, warm):
        o.hide_render = False

    dress_shell(r, objs)

    lo, hi = bbox_of(objs)
    cx, cy = (lo.x + hi.x) / 2, (lo.y + hi.y) / 2
    zceil = hi.z
    dx, dy = hi.x - lo.x, hi.y - lo.y
    print(f"ROOM {r}: bbox x {dx:.1f} y {dy:.1f} zfloor {lo.z:.2f} zceil {zceil:.2f}")

    # hide botched ceiling assets (run 3 "recessed panel"/detector retrievals
    # render as ragged smears); keep lights, pendants, projectors
    BAD_CEIL = ("panel", "detector", "vent", "alarm", "sprinkler", "speaker")
    for o in objs:
        if o.type != "MESH" or "light" in o.name.lower():
            continue
        d = o.dimensions
        oz = o.matrix_world.translation.z
        if oz > zceil - 0.6 and (d.z < 0.12 or any(k in o.name.lower() for k in BAD_CEIL)):
            o.hide_render = True
            print(f"  hid ceiling debris: {o.name}")

    # ceiling-height area light over room center, slightly below ceiling
    light.location = (cx, cy, max(lo.z + 2.4, zceil - 0.3))
    light_data.size = max(dx * 0.7, 1.0)
    light_data.size_y = max(dy * 0.7, 1.0)
    light_data.energy = 11 * max(dx * dy, 4)  # scale with room area

    # warm side light along the long wall, tilted down into the room
    hgt = zceil - lo.z
    if dx >= dy:
        warm.location = (cx, lo.y + 0.25, lo.z + hgt * 0.72)
        warm.rotation_euler = (math.radians(-65), 0, 0)
        warm_data.size = dx * 0.55
    else:
        warm.location = (lo.x + 0.25, cy, lo.z + hgt * 0.72)
        warm.rotation_euler = (0, math.radians(65), 0)
        warm_data.size = dy * 0.55
    warm_data.size_y = hgt * 0.45
    warm_data.energy = 5 * max(dx * dy, 4)

    # pick camera corners with the most clearance from furniture
    inset = 0.7
    eye_z = lo.z + 1.45
    target = (cx, cy, lo.z + 1.05)
    centers = []
    for o in objs:
        if o.type != "MESH":
            continue
        rest = re.sub(r"\.\d+$", "", o.name[len(r) + 1:])
        if "layout_floor" in rest or rest in WALL_DIRS or "ceiling" in rest:
            continue
        c = o.matrix_world.translation
        centers.append((c.x, c.y))
    cands = [
        (lo.x + inset, lo.y + inset), (hi.x - inset, lo.y + inset),
        (lo.x + inset, hi.y - inset), (hi.x - inset, hi.y - inset),
    ]
    def clearance(p):
        if not centers:
            return 99.0
        return min(math.hypot(p[0] - c[0], p[1] - c[1]) for c in centers)
    corners = sorted(cands, key=clearance, reverse=True)

    for vi in range(nviews):
        px, py = corners[vi % len(corners)]
        aim(cam, (px, py, eye_z), target)
        scene.render.filepath = os.path.join(outdir, f"{r}_view{vi + 1}.png")
        print(f"RENDERING {scene.render.filepath}")
        bpy.ops.render.render(write_still=True)

print("ALL RENDERS DONE")
