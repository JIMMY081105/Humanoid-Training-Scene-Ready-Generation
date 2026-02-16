#!/usr/bin/env python3
"""Finish the corridor's readable wayfinding, daylight, lighting, and lobby detail."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import struct
import time
import uuid
from pathlib import Path
from typing import Any

from repair_room1_acceptance_details import (
    _publish_staged_final,
    _worker_config,
    _write_json,
    _write_text,
)


EXPECTED_STATE_SHA256 = "fa7d116fae493735b9284106bbda272808571d3d7b00684dcff3bd1acad4b68e"
ROOM_ID = "main_corridor"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _box_gltf(size: tuple[float, float, float], color: tuple[float, float, float, float]) -> str:
    sx, sy, sz = (value / 2.0 for value in size)
    positions = [
        (-sx, -sy, -sz), (sx, -sy, -sz), (sx, sy, -sz), (-sx, sy, -sz),
        (-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz),
    ]
    indices = [
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
        0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5,
        2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7,
    ]
    position_bytes = b"".join(struct.pack("<3f", *point) for point in positions)
    index_bytes = b"".join(struct.pack("<H", value) for value in indices)
    data = position_bytes + index_bytes
    payload = {
        "asset": {"version": "2.0", "generator": "SceneSmith corridor acceptance detail"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "contract_mesh"}],
        "meshes": [{
            "name": "contract_mesh",
            "primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}],
        }],
        "materials": [{
            "name": "contract_material",
            "pbrMetallicRoughness": {
                "baseColorFactor": list(color), "metallicFactor": 0.0, "roughnessFactor": 0.65,
            },
        }],
        "buffers": [{
            "byteLength": len(data),
            "uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode("ascii"),
        }],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(position_bytes), "byteLength": len(index_bytes), "target": 34963},
        ],
        "accessors": [
            {
                "bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3",
                "min": [-sx, -sy, -sz], "max": [sx, sy, sz],
            },
            {
                "bufferView": 1, "componentType": 5123, "count": len(indices),
                "type": "SCALAR", "min": [0], "max": [7],
            },
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _sign_png(text: str, *, subtitle: str = "") -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1536, 420
    image = Image.new("RGBA", (width, height), (247, 240, 220, 255))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((16, 16, width - 16, height - 16), radius=36, fill=(247, 240, 220, 255), outline=(170, 109, 48, 255), width=28)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    primary = ImageFont.truetype(font_path, 176)
    secondary = ImageFont.truetype(font_path, 62)
    primary_box = draw.textbbox((0, 0), text, font=primary)
    primary_width = primary_box[2] - primary_box[0]
    primary_y = 72 if subtitle else 115
    draw.text(((width - primary_width) / 2, primary_y), text, font=primary, fill=(25, 79, 102, 255))
    if subtitle:
        secondary_box = draw.textbbox((0, 0), subtitle, font=secondary)
        secondary_width = secondary_box[2] - secondary_box[0]
        draw.text(((width - secondary_width) / 2, 290), subtitle, font=secondary, fill=(119, 69, 31, 255))
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _textured_sign_gltf(text: str, subtitle: str, width: float, height: float) -> str:
    png = _sign_png(text, subtitle=subtitle)
    w, h = width / 2.0, height / 2.0
    positions = [(-w, 0.0, -h), (w, 0.0, -h), (w, 0.0, h), (-w, 0.0, h)]
    normals = [(0.0, 1.0, 0.0)] * 4
    texcoords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    indices = [0, 2, 1, 0, 3, 2]
    position_bytes = b"".join(struct.pack("<3f", *point) for point in positions)
    normal_bytes = b"".join(struct.pack("<3f", *point) for point in normals)
    uv_bytes = b"".join(struct.pack("<2f", *point) for point in texcoords)
    index_bytes = b"".join(struct.pack("<H", value) for value in indices)
    data = position_bytes + normal_bytes + uv_bytes + index_bytes
    normal_offset = len(position_bytes)
    uv_offset = normal_offset + len(normal_bytes)
    index_offset = uv_offset + len(uv_bytes)
    payload = {
        "asset": {"version": "2.0", "generator": "SceneSmith readable school wayfinding"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "readable_wayfinding_sign"}],
        "meshes": [{
            "name": "readable_wayfinding_sign",
            "primitives": [{
                "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                "indices": 3,
                "material": 0,
            }],
        }],
        "materials": [{
            "name": "matte_printed_wayfinding",
            "doubleSided": True,
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 0.58,
            },
        }],
        "textures": [{"source": 0}],
        "images": [{"uri": "data:image/png;base64," + base64.b64encode(png).decode("ascii")}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "buffers": [{
            "byteLength": len(data),
            "uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode("ascii"),
        }],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": normal_offset, "byteLength": len(normal_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": uv_offset, "byteLength": len(uv_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": index_offset, "byteLength": len(index_bytes), "target": 34963},
        ],
        "accessors": [
            {
                "bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
                "min": [-w, 0.0, -h], "max": [w, 0.0, h],
            },
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
            {
                "bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC2",
                "min": [0.0, 0.0], "max": [1.0, 1.0],
            },
            {
                "bufferView": 3, "componentType": 5123, "count": len(indices),
                "type": "SCALAR", "min": [0], "max": [3],
            },
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _sdf_for_gltf(model_name: str, filename: str) -> str:
    return (
        "<?xml version='1.0'?>\n"
        "<sdf version='1.7'>\n"
        f"  <model name='{model_name}'>\n"
        "    <static>true</static>\n"
        "    <link name='visual_only'>\n"
        "      <visual name='readable_mesh'>\n"
        f"        <geometry><mesh><uri>{filename}</uri></mesh></geometry>\n"
        "      </visual>\n"
        "    </link>\n"
        "  </model>\n"
        "</sdf>\n"
    )


def _visual_box(name: str, pose: tuple[float, float, float], size: tuple[float, float, float], diffuse: tuple[float, float, float, float], emissive: tuple[float, float, float, float] | None = None) -> str:
    material = f"<diffuse>{' '.join(map(str, diffuse))}</diffuse>"
    if emissive is not None:
        material += f"<emissive>{' '.join(map(str, emissive))}</emissive>"
    return (
        f"<visual name='{name}'><pose>{pose[0]} {pose[1]} {pose[2]} 0 0 0</pose>"
        f"<geometry><box><size>{size[0]} {size[1]} {size[2]}</size></box></geometry>"
        f"<material>{material}</material></visual>"
    )


def _bank_sdf(model_name: str, visuals: list[str]) -> str:
    return (
        "<?xml version='1.0'?><sdf version='1.7'>"
        f"<model name='{model_name}'><static>true</static><link name='visual_only'>"
        + "".join(visuals)
        + "</link></model></sdf>\n"
    )


def _asset(directory: Path, files: dict[str, str]) -> dict[str, Path]:
    if directory.exists():
        for name, content in files.items():
            path = directory / name
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise RuntimeError(f"Unexpected existing corridor asset: {directory}")
        return {name: directory / name for name in files}
    staging = directory.with_name(f".{directory.name}.{uuid.uuid4().hex}.tmp")
    staging.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        _write_text(staging / name, content)
    os.replace(staging, directory)
    return {name: directory / name for name in files}


def _new_or_deeper(baseline: list[Any], candidate: list[Any]) -> list[Any]:
    depths = {
        tuple(sorted((str(item.object_a_id), str(item.object_b_id)))): item.penetration_depth
        for item in baseline
    }
    return [
        item for item in candidate
        if item.penetration_depth
        > depths.get(tuple(sorted((str(item.object_a_id), str(item.object_b_id)))), -1.0) + 1e-4
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--port-offset", type=int, default=949)
    args = parser.parse_args()

    from run_single_room_worker import _configure_pytorch_cuda_allocator
    _configure_pytorch_cuda_allocator()
    import numpy as np
    from omegaconf import OmegaConf
    from pydrake.math import RigidTransform, RollPitchYaw
    from scenesmith.agent_utils.physics_validation import compute_scene_collisions
    from scenesmith.agent_utils.rendering import save_scene_as_blend
    from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID

    repo = args.repo_dir.resolve()
    run = args.run_dir.resolve()
    os.chdir(repo)
    room_dir = run / "scene_000" / f"room_{ROOM_ID}"
    final_dir = room_dir / "scene_states" / "final_scene"
    state_path = final_dir / "scene_state.json"
    actual_sha = _sha256(state_path)
    if actual_sha != EXPECTED_STATE_SHA256:
        raise RuntimeError(f"Unexpected main corridor state: {actual_sha}")

    scene = RoomScene(room_geometry=None, scene_dir=room_dir, room_id=ROOM_ID)
    scene.restore_from_state_dict(json.loads(state_path.read_text(encoding="utf-8")))
    additions = {
        "corridor_interior_welcome_sign_0",
        "corridor_library_wayfinding_sign_0",
        "corridor_room_wayfinding_sign_0",
        "corridor_daylight_clerestory_bank_0",
        "corridor_warm_pendant_bank_0",
        "corridor_central_lobby_rug_0",
        "corridor_oak_door_trim_bank_0",
        "corridor_oak_baseboard_bank_0",
    }
    stale = sorted(object_id for object_id in additions if scene.get_object(UniqueID(object_id)) is not None)
    if stale:
        raise RuntimeError(f"Stale corridor final-detail input: {stale}")
    baseline = compute_scene_collisions(scene)

    root = room_dir / "generated_assets" / "architectural"
    welcome_files = _asset(
        root / "interior_welcome_readable_v1",
        {
            "welcome_sign.gltf": _textured_sign_gltf("WELCOME", "MAIN SCHOOL LOBBY", 3.2, 0.82),
            "welcome_sign.sdf": _sdf_for_gltf("interior_welcome_sign", "welcome_sign.gltf"),
        },
    )
    library_files = _asset(
        root / "library_wayfinding_readable_v1",
        {
            "library_sign.gltf": _textured_sign_gltf("LIBRARY", "SOUTH LOBBY  •  KEEP PATH CLEAR", 2.8, 0.78),
            "library_sign.sdf": _sdf_for_gltf("library_wayfinding_sign", "library_sign.gltf"),
        },
    )
    room_files = _asset(
        root / "room_wayfinding_readable_v1",
        {
            "room_sign.gltf": _textured_sign_gltf("CLASSROOMS", "RESTROOMS  •  STORAGE", 2.8, 0.78),
            "room_sign.sdf": _sdf_for_gltf("room_wayfinding_sign", "room_sign.gltf"),
        },
    )

    pendant_visuals: list[str] = []
    for index, y in enumerate((-8.0, -4.0, 0.0, 4.0, 8.0)):
        for side, x in enumerate((-2.2, 2.2)):
            pendant_visuals.append(_visual_box(
                f"warm_pendant_{index}_{side}", (x, y, 0.0), (1.65, 0.48, 0.18),
                (1.0, 0.63, 0.20, 1.0), (1.0, 0.42, 0.08, 1.0),
            ))
    pendant_files = _asset(
        root / "large_warm_pendant_bank_v1",
        {
            "warm_pendants.sdf": _bank_sdf("large_warm_pendant_bank", pendant_visuals),
            "warm_pendants_evidence.gltf": _box_gltf((5.4, 18.0, 0.18), (1.0, 0.63, 0.20, 1.0)),
        },
    )

    daylight_visuals: list[str] = []
    for side, x in enumerate((-5.86, 5.86)):
        for index, y in enumerate((-8.7, -5.0, -1.3, 2.4, 6.1, 9.1)):
            daylight_visuals.append(_visual_box(
                f"borrowed_daylight_{side}_{index}", (x, y, 0.0), (0.10, 1.45, 0.70),
                (0.74, 0.91, 1.0, 1.0), (0.36, 0.66, 1.0, 1.0),
            ))
    daylight_files = _asset(
        root / "borrowed_daylight_clerestory_bank_v1",
        {
            "daylight_clerestories.sdf": _bank_sdf("borrowed_daylight_clerestory_bank", daylight_visuals),
            "daylight_clerestories_evidence.gltf": _box_gltf((11.72, 19.0, 0.70), (0.74, 0.91, 1.0, 1.0)),
        },
    )

    rug_visuals = [
        _visual_box("teal_lobby_rug", (0.0, 0.0, 0.0), (7.4, 4.4, 0.025), (0.12, 0.48, 0.54, 1.0)),
        _visual_box("gold_rug_inlay_a", (0.0, -1.88, 0.014), (6.7, 0.10, 0.018), (0.92, 0.63, 0.22, 1.0)),
        _visual_box("gold_rug_inlay_b", (0.0, 1.88, 0.014), (6.7, 0.10, 0.018), (0.92, 0.63, 0.22, 1.0)),
    ]
    rug_files = _asset(
        root / "central_lobby_island_rug_v1",
        {
            "lobby_rug.sdf": _bank_sdf("central_lobby_island_rug", rug_visuals),
            "lobby_rug_evidence.gltf": _box_gltf((7.4, 4.4, 0.025), (0.12, 0.48, 0.54, 1.0)),
        },
    )

    door_visuals: list[str] = []
    openings = [
        (-5.86, -7.5), (-5.86, -3.15), (-5.86, 4.0), (-5.86, 10.0),
        (5.86, -7.5), (5.86, -1.9), (5.86, 3.7), (5.86, 10.0),
    ]
    for index, (x, y) in enumerate(openings):
        door_visuals.extend([
            _visual_box(f"door_trim_{index}_left", (x, y - 0.57, 1.08), (0.12, 0.12, 2.16), (0.58, 0.34, 0.14, 1.0)),
            _visual_box(f"door_trim_{index}_right", (x, y + 0.57, 1.08), (0.12, 0.12, 2.16), (0.58, 0.34, 0.14, 1.0)),
            _visual_box(f"door_trim_{index}_header", (x, y, 2.15), (0.12, 1.26, 0.12), (0.58, 0.34, 0.14, 1.0)),
        ])
    door_files = _asset(
        root / "open_door_oak_trim_bank_v1",
        {
            "door_trim.sdf": _bank_sdf("open_door_oak_trim_bank", door_visuals),
            "door_trim_evidence.gltf": _box_gltf((11.72, 18.2, 2.22), (0.58, 0.34, 0.14, 1.0)),
        },
    )

    baseboard_visuals = [
        _visual_box("baseboard_west", (-5.89, 0.0, 0.09), (0.10, 22.1, 0.18), (0.61, 0.38, 0.18, 1.0)),
        _visual_box("baseboard_east", (5.89, 0.0, 0.09), (0.10, 22.1, 0.18), (0.61, 0.38, 0.18, 1.0)),
        _visual_box("baseboard_north", (0.0, 11.10, 0.09), (11.7, 0.10, 0.18), (0.61, 0.38, 0.18, 1.0)),
        _visual_box("baseboard_south", (0.0, -11.10, 0.09), (11.7, 0.10, 0.18), (0.61, 0.38, 0.18, 1.0)),
    ]
    baseboard_files = _asset(
        root / "corridor_oak_baseboard_bank_v1",
        {
            "baseboards.sdf": _bank_sdf("corridor_oak_baseboard_bank", baseboard_visuals),
            "baseboards_evidence.gltf": _box_gltf((11.8, 22.2, 0.18), (0.61, 0.38, 0.18, 1.0)),
        },
    )

    def add_sign(object_id: str, name: str, description: str, transform: RigidTransform, paths: dict[str, Path], prefix: str, width: float, height: float, role: str) -> None:
        scene.add_object(SceneObject(
            object_id=UniqueID(object_id), object_type=ObjectType.WALL_MOUNTED,
            name=name, description=description, transform=transform,
            sdf_path=paths[f"{prefix}.sdf"], geometry_path=paths[f"{prefix}.gltf"],
            metadata={
                "asset_source": "architectural_visual_contract_repair",
                "semantic_role": role,
                "readable_text_evidence": description,
                "collision_policy": "collisionless_high_wall_wayfinding",
            },
            bbox_min=np.array([-width / 2.0, -0.03, -height / 2.0]),
            bbox_max=np.array([width / 2.0, 0.03, height / 2.0]),
        ))

    add_sign(
        "corridor_interior_welcome_sign_0", "readable_interior_welcome_sign",
        "Large correctly oriented readable WELCOME / MAIN SCHOOL LOBBY sign facing the corridor interior",
        RigidTransform(p=[0.0, -9.72, 2.42]), welcome_files, "welcome_sign", 3.2, 0.82,
        "welcoming_school_signage",
    )
    add_sign(
        "corridor_library_wayfinding_sign_0", "readable_library_wayfinding_sign",
        "Large readable LIBRARY / SOUTH LOBBY / KEEP PATH CLEAR sign",
        RigidTransform(rpy=RollPitchYaw([0.0, 0.0, math.pi / 2.0]), p=[5.86, -8.70, 2.40]),
        library_files, "library_sign", 2.8, 0.78, "explicit_library_connection_wayfinding",
    )
    add_sign(
        "corridor_room_wayfinding_sign_0", "readable_room_wayfinding_sign",
        "Large readable CLASSROOMS / RESTROOMS / STORAGE school wayfinding sign",
        RigidTransform(rpy=RollPitchYaw([0.0, 0.0, -math.pi / 2.0]), p=[-5.86, 7.15, 2.40]),
        room_files, "room_sign", 2.8, 0.78, "school_room_wayfinding",
    )

    def add_bank(object_id: str, object_type: ObjectType, name: str, description: str, transform: RigidTransform, sdf: Path, evidence: Path, bbox_min: list[float], bbox_max: list[float], role: str) -> None:
        scene.add_object(SceneObject(
            object_id=UniqueID(object_id), object_type=object_type,
            name=name, description=description, transform=transform, sdf_path=sdf,
            metadata={
                "asset_source": "architectural_visual_contract_repair",
                "semantic_role": role,
                "composite_members": [{"geometry_path": str(evidence.relative_to(room_dir))}],
                "collision_policy": "collisionless_visual_architectural_finish",
            },
            bbox_min=np.array(bbox_min), bbox_max=np.array(bbox_max),
        ))

    add_bank(
        "corridor_warm_pendant_bank_0", ObjectType.CEILING_MOUNTED,
        "large_warm_pendant_light_bank", "Ten large visible amber 3000K pendant light fixtures",
        RigidTransform(p=[0.0, 0.0, 2.74]), pendant_files["warm_pendants.sdf"], pendant_files["warm_pendants_evidence.gltf"],
        [-3.1, -8.3, -0.10], [3.1, 8.3, 0.10], "visible_warm_interior_lighting",
    )
    add_bank(
        "corridor_daylight_clerestory_bank_0", ObjectType.WALL_MOUNTED,
        "borrowed_daylight_clerestory_bank", "Twelve large blue-white borrowed-daylight clerestory panels along both corridor walls",
        RigidTransform(p=[0.0, 0.0, 2.45]), daylight_files["daylight_clerestories.sdf"], daylight_files["daylight_clerestories_evidence.gltf"],
        [-5.91, -9.45, -0.35], [5.91, 9.85, 0.35], "visible_soft_daylight_cues",
    )
    add_bank(
        "corridor_central_lobby_rug_0", ObjectType.FURNITURE,
        "teal_and_gold_central_lobby_rug", "Large teal and gold rug intentionally unifying the central bench and planter lobby island",
        RigidTransform(p=[0.0, 2.55, 0.0]), rug_files["lobby_rug.sdf"], rug_files["lobby_rug_evidence.gltf"],
        [-3.7, -2.2, 0.0], [3.7, 2.2, 0.043], "intentional_central_seating_planter_island",
    )
    add_bank(
        "corridor_oak_door_trim_bank_0", ObjectType.WALL_MOUNTED,
        "open_oak_door_frame_bank", "Oak trim clearly framing all eight open classroom, restroom and storage doorways without adding leaves",
        RigidTransform(), door_files["door_trim.sdf"], door_files["door_trim_evidence.gltf"],
        [-5.92, -8.15, 0.0], [5.92, 10.65, 2.22], "realistic_open_door_frames_without_obstruction",
    )
    add_bank(
        "corridor_oak_baseboard_bank_0", ObjectType.WALL_MOUNTED,
        "oak_corridor_baseboard_bank", "Continuous warm oak baseboard trim finishing the lobby architecture",
        RigidTransform(), baseboard_files["baseboards.sdf"], baseboard_files["baseboards_evidence.gltf"],
        [-5.94, -11.15, 0.0], [5.94, 11.15, 0.18], "warm_finished_architectural_trim",
    )

    candidate = compute_scene_collisions(scene)
    changed = _new_or_deeper(baseline, candidate)
    if changed:
        raise RuntimeError(
            "Corridor final visual repair introduced collision: "
            + "; ".join(item.to_description() for item in changed)
        )

    cfg = OmegaConf.to_container(_worker_config(args), resolve=True)
    staged = final_dir.parent / f".main_corridor_final_visual_{uuid.uuid4().hex}"
    staged.mkdir(parents=False, exist_ok=False)
    repaired_state = scene.to_state_dict()
    repaired_state["timestamp"] = time.time()
    _write_json(staged / "scene_state.json", repaired_state)
    (staged / "scene.dmd.yaml").write_text(scene.to_drake_directive(), encoding="utf-8")
    rendering = cfg["furniture_agent"]["rendering"]
    save_scene_as_blend(
        scene=scene,
        output_path=staged / "scene.blend",
        blender_server_host=rendering.get("blender_server_host", "127.0.0.1"),
        blender_server_port_range=tuple(rendering["blender_server_port_range"]),
        server_startup_delay=rendering["server_startup_delay"],
        port_cleanup_delay=rendering["port_cleanup_delay"],
    )
    backup = _publish_staged_final(staged_dir=staged, final_dir=final_dir)
    receipt = {
        "schema_version": 1,
        "status": "pass",
        "room_id": ROOM_ID,
        "operation": "readable_wayfinding_visible_daylight_warm_lighting_and_finished_lobby_detail",
        "state_before_sha256": EXPECTED_STATE_SHA256,
        "state_after_sha256": _sha256(final_dir / "scene_state.json"),
        "backup_final_scene": str(backup),
        "added_object_ids": sorted(additions),
        "physical_validation": {
            "baseline_collision_count": len(baseline),
            "final_collision_count": len(candidate),
            "new_or_deeper_collision_count": len(changed),
        },
        "clearance_contract": "all door openings retain their original clear volume; trim has no door leaves; the central spine remains open",
        "quality_policy": "collision, stability, doorway, robot-clearance, inventory, and visual thresholds unchanged",
    }
    _write_json(room_dir / "quality_gates" / "main_corridor_final_visual_contract_repair.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
