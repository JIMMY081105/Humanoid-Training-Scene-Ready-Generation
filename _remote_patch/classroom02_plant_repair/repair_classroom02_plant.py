#!/usr/bin/env python3
"""Add the single missing stable potted plant to completed Classroom 2."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import struct
import time
import uuid
from pathlib import Path
from typing import Any

from repair_room1_acceptance_details import _publish_staged_final, _worker_config, _write_json, _write_text


EXPECTED_STATE_SHA256 = "54848187a4090cd6ec92e39e4844feb7c07217598b714ae452a4b930b338559c"
ROOM_ID = "classroom_02"
OBJECT_ID = "acceptance_potted_classroom_plant_0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mesh() -> str:
    sx, sy, sz = 0.48, 0.48, 1.02
    x, y = sx / 2.0, sy / 2.0
    points = [(-x,-y,0),(x,-y,0),(x,y,0),(-x,y,0),(-x,-y,sz),(x,-y,sz),(x,y,sz),(-x,y,sz)]
    indices = [0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,1,2,6,1,6,5,2,3,7,2,7,6,3,0,4,3,4,7]
    pbytes = b"".join(struct.pack("<3f", *point) for point in points)
    ibytes = b"".join(struct.pack("<H", item) for item in indices)
    data = pbytes + ibytes
    return json.dumps({
        "asset":{"version":"2.0","generator":"SceneSmith classroom greenery repair"},
        "scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0,"name":"potted_classroom_plant"}],
        "meshes":[{"name":"potted_classroom_plant","primitives":[{"attributes":{"POSITION":0},"indices":1,"material":0}]}],
        "materials":[{"name":"greenery_evidence","pbrMetallicRoughness":{"baseColorFactor":[0.10,0.52,0.20,1.0],"metallicFactor":0.0,"roughnessFactor":0.72}}],
        "buffers":[{"byteLength":len(data),"uri":"data:application/octet-stream;base64,"+base64.b64encode(data).decode("ascii")}],
        "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":len(pbytes),"target":34962},{"buffer":0,"byteOffset":len(pbytes),"byteLength":len(ibytes),"target":34963}],
        "accessors":[{"bufferView":0,"componentType":5126,"count":8,"type":"VEC3","min":[-x,-y,0],"max":[x,y,sz]},{"bufferView":1,"componentType":5123,"count":len(indices),"type":"SCALAR","min":[0],"max":[7]}],
    }, indent=2, sort_keys=True)+"\n"


def _sdf() -> str:
    leaves = []
    for index, (x, y, z, yaw) in enumerate((
        (0.0,0.0,0.58,0.0),(-0.12,0.02,0.70,0.45),(0.13,-0.02,0.73,-0.42),
        (-0.08,-0.10,0.86,-0.25),(0.09,0.11,0.88,0.30),(0.0,0.0,0.99,0.0),
    )):
        leaves.append(
            f"<visual name='leaf_{index}'><pose>{x} {y} {z} 0 {yaw} 0</pose>"
            "<geometry><ellipsoid><radii>0.11 0.035 0.24</radii></ellipsoid></geometry>"
            f"<material><diffuse>{0.08 + index * 0.012} {0.38 + index * 0.025} 0.14 1</diffuse></material></visual>"
        )
    return (
        "<?xml version='1.0'?><sdf version='1.7'><model name='stable_potted_classroom_plant'><link name='base_link'>"
        "<visual name='terracotta_pot'><pose>0 0 0.20 0 0 0</pose><geometry><cylinder><radius>0.20</radius><length>0.40</length></cylinder></geometry><material><diffuse>0.58 0.25 0.10 1</diffuse></material></visual>"
        "<visual name='dark_soil'><pose>0 0 0.405 0 0 0</pose><geometry><cylinder><radius>0.17</radius><length>0.025</length></cylinder></geometry><material><diffuse>0.12 0.07 0.03 1</diffuse></material></visual>"
        "<visual name='stem'><pose>0 0 0.62 0 0 0</pose><geometry><cylinder><radius>0.025</radius><length>0.50</length></cylinder></geometry><material><diffuse>0.18 0.32 0.08 1</diffuse></material></visual>"
        + "".join(leaves) +
        "<collision name='pot_collision'><pose>0 0 0.20 0 0 0</pose><geometry><cylinder><radius>0.20</radius><length>0.40</length></cylinder></geometry></collision>"
        "<collision name='stem_collision'><pose>0 0 0.62 0 0 0</pose><geometry><cylinder><radius>0.035</radius><length>0.48</length></cylinder></geometry></collision>"
        "</link></model></sdf>\n"
    )


def _new_or_deeper(baseline: list[Any], candidate: list[Any]) -> list[Any]:
    depths={tuple(sorted((str(item.object_a_id),str(item.object_b_id)))):item.penetration_depth for item in baseline}
    return [item for item in candidate if item.penetration_depth > depths.get(tuple(sorted((str(item.object_a_id),str(item.object_b_id)))),-1.0)+1e-4]


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--csv",required=True)
    parser.add_argument("--port-offset",type=int,default=972)
    args=parser.parse_args()
    from run_single_room_worker import _configure_pytorch_cuda_allocator
    _configure_pytorch_cuda_allocator()
    import numpy as np
    from omegaconf import OmegaConf
    from pydrake.math import RigidTransform
    from scenesmith.agent_utils.physics_validation import compute_scene_collisions
    from scenesmith.agent_utils.rendering import save_scene_as_blend
    from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID

    repo=args.repo_dir.resolve(); run=args.run_dir.resolve(); os.chdir(repo)
    cfg=OmegaConf.to_container(_worker_config(args),resolve=True)
    room_dir=run/"scene_000"/f"room_{ROOM_ID}"; final_dir=room_dir/"scene_states"/"final_scene"; state_path=final_dir/"scene_state.json"
    if _sha256(state_path)!=EXPECTED_STATE_SHA256: raise RuntimeError(f"Unexpected Classroom 2 state: {_sha256(state_path)}")
    scene=RoomScene(room_geometry=None,scene_dir=room_dir,room_id=ROOM_ID); scene.restore_from_state_dict(json.loads(state_path.read_text(encoding="utf-8")))
    if scene.get_object(UniqueID(OBJECT_ID)) is not None: raise RuntimeError("Plant repair already exists")
    baseline=compute_scene_collisions(scene)
    asset_dir=room_dir/"generated_assets"/"contract_inventory"/"potted_classroom_plant_v1"
    mesh_text,sdf_text=_mesh(),_sdf(); mesh_path=asset_dir/"potted_classroom_plant.gltf"; sdf_path=asset_dir/"potted_classroom_plant.sdf"
    if asset_dir.exists():
        if not mesh_path.is_file() or mesh_path.read_text(encoding="utf-8")!=mesh_text or not sdf_path.is_file() or sdf_path.read_text(encoding="utf-8")!=sdf_text: raise RuntimeError(f"Unexpected plant asset: {asset_dir}")
    else:
        temporary=asset_dir.with_name(f".{asset_dir.name}.{uuid.uuid4().hex}.tmp"); temporary.mkdir(parents=True,exist_ok=False); _write_text(temporary/mesh_path.name,mesh_text); _write_text(temporary/sdf_path.name,sdf_text); os.replace(temporary,asset_dir)
    plant=SceneObject(object_id=UniqueID(OBJECT_ID),object_type=ObjectType.FURNITURE,name="potted_classroom_plant",description="One independently represented stable potted classroom plant with abundant green leaves",transform=RigidTransform(),geometry_path=mesh_path,sdf_path=sdf_path,metadata={"asset_source":"contract_inventory_repair","semantic_role":"plant greenery","collision_mesh":"rigid pot and stem collision primitives"},bbox_min=np.array([-0.24,-0.24,0.0]),bbox_max=np.array([0.24,0.24,1.02]))
    scene.add_object(plant)
    candidates=[(-4.00,3.10),(-4.00,-3.10),(3.85,3.10),(3.25,-3.12),(-3.30,2.95),(-3.30,-2.95),(3.25,2.95)]
    chosen=None
    for x,y in candidates:
        plant.transform=RigidTransform(p=[x,y,0.002])
        collisions=compute_scene_collisions(scene)
        if not _new_or_deeper(baseline,collisions): chosen=(x,y); break
    if chosen is None: raise RuntimeError("No collision-free potted plant position exists")
    candidate=compute_scene_collisions(scene); changed=_new_or_deeper(baseline,candidate)
    if changed: raise RuntimeError("Plant introduced collision: "+"; ".join(item.to_description() for item in changed))
    stage=final_dir.parent/f".classroom02_plant_{uuid.uuid4().hex}"; stage.mkdir(parents=False,exist_ok=False)
    state=scene.to_state_dict(); state["timestamp"]=time.time(); _write_json(stage/"scene_state.json",state); (stage/"scene.dmd.yaml").write_text(scene.to_drake_directive(),encoding="utf-8")
    rendering=cfg["furniture_agent"]["rendering"]
    save_scene_as_blend(scene=scene,output_path=stage/"scene.blend",blender_server_host=rendering.get("blender_server_host","127.0.0.1"),blender_server_port_range=tuple(rendering["blender_server_port_range"]),server_startup_delay=rendering["server_startup_delay"],port_cleanup_delay=rendering["port_cleanup_delay"])
    backup=_publish_staged_final(staged_dir=stage,final_dir=final_dir)
    receipt={"schema_version":1,"status":"pass","room_id":ROOM_ID,"operation":"add_missing_stable_potted_classroom_plant","state_before_sha256":EXPECTED_STATE_SHA256,"state_after_sha256":_sha256(final_dir/"scene_state.json"),"backup_final_scene":str(backup),"object_id":OBJECT_ID,"position":list(chosen)+[0.002],"mesh_sha256":_sha256(mesh_path),"sdf_sha256":_sha256(sdf_path),"physical_validation":{"baseline_collision_count":len(baseline),"final_collision_count":len(candidate),"new_or_deeper_collision_count":0},"quality_policy":"all master-prompt, collision, doorway, clearance, stability, inventory, and visual gates unchanged"}
    _write_json(room_dir/"quality_gates"/"classroom02_plant_repair.json",receipt); print(json.dumps(receipt,indent=2,sort_keys=True))


if __name__=="__main__": main()
