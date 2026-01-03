#!/usr/bin/env python3
"""Fail-closed deterministic validation of the native food-factory layout."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from factory_room_contract import CONTRACT_ID, PROFILE, load_contract, room_ids

EPS = 1e-6
MIN_DOOR = 0.9
REQUIRED_WINDOWS = frozenset({"ingredient_receiving","dry_storage","cold_storage","washing_preparation","qc_laboratory","office_administration","finished_goods_storage","maintenance","changing_room","break_room","boys_toilet","girls_toilet"})
WORKFLOW_EDGES = (
    ("ingredient_receiving","dry_storage"),
    ("ingredient_receiving","washing_preparation"),
    ("dry_storage","processing_hall"),
    ("cold_storage","processing_hall"),
    ("washing_preparation","processing_hall"),
    ("processing_hall","packaging_hall"),
    ("packaging_hall","finished_goods_storage"),
)


class LayoutError(RuntimeError): pass


def _num(value: Any, label: str) -> float:
    try: result=float(value)
    except (TypeError,ValueError) as exc: raise LayoutError(f"{label} is not numeric") from exc
    if not math.isfinite(result): raise LayoutError(f"{label} is not finite")
    return result


def _entries(layout: Mapping[str,Any], key: str) -> list[dict[str,Any]]:
    raw=layout.get(key)
    values=list(raw.values()) if isinstance(raw,dict) else raw if isinstance(raw,list) else []
    return [item for item in values if isinstance(item,dict)]


def _map(layout: Mapping[str,Any], key: str, expected: set[str]) -> dict[str,dict[str,Any]]:
    entries=_entries(layout,key); ids=[str(v.get("room_id") or v.get("id") or "") for v in entries]
    if len(entries)!=14 or set(ids)!=expected or len(ids)!=len(set(ids)):
        raise LayoutError(f"{key} must contain exact 14 rooms; actual={ids}")
    return {str(v.get("room_id") or v.get("id")):v for v in entries}


def _bounds(room: Mapping[str,Any]) -> tuple[float,float,float,float]:
    position=room.get("position")
    if not isinstance(position,(list,tuple)) or len(position)<2: raise LayoutError("room position missing")
    x,y=_num(position[0],"x"),_num(position[1],"y")
    width=_num(room.get("width"),"width")
    depth=_num(room.get("depth",room.get("length")),"depth")
    if width<=0 or depth<=0: raise LayoutError("room dimensions must be positive")
    return x,y,x+width,y+depth


def _overlap(a:tuple[float,...],b:tuple[float,...])->float:
    return max(0.0,min(a[2],b[2])-max(a[0],b[0]))*max(0.0,min(a[3],b[3])-max(a[1],b[1]))


def _shared(a:tuple[float,...],b:tuple[float,...])->tuple[str,float,float,float]|None:
    if abs(a[2]-b[0])<=EPS or abs(b[2]-a[0])<=EPS:
        lo,hi=max(a[1],b[1]),min(a[3],b[3]); return ("vertical",a[2] if abs(a[2]-b[0])<=EPS else a[0],lo,hi) if hi-lo>=MIN_DOOR else None
    if abs(a[3]-b[1])<=EPS or abs(b[3]-a[1])<=EPS:
        lo,hi=max(a[0],b[0]),min(a[2],b[2]); return ("horizontal",a[3] if abs(a[3]-b[1])<=EPS else a[1],lo,hi) if hi-lo>=MIN_DOOR else None
    return None


def _walls(room: Mapping[str,Any])->list[dict[str,Any]]:
    raw=room.get("walls"); return [v for v in raw if isinstance(v,dict)] if isinstance(raw,list) else []


def _openings(room: Mapping[str,Any], opening_id: str)->list[tuple[dict[str,Any],dict[str,Any]]]:
    found=[]
    for wall in _walls(room):
        for opening in wall.get("openings",[]) if isinstance(wall.get("openings"),list) else []:
            if isinstance(opening,dict) and str(opening.get("opening_id"))==opening_id: found.append((wall,opening))
    return found


def _door_evidence(layout: Mapping[str,Any], placed: Mapping[str,dict[str,Any]])->tuple[dict[frozenset[str],list[dict[str,Any]]],list[str]]:
    graph:dict[frozenset[str],list[dict[str,Any]]]={}; issues=[]
    for door in layout.get("doors",[]) if isinstance(layout.get("doors"),list) else []:
        if not isinstance(door,dict): continue
        door_id=str(door.get("id") or ""); a=str(door.get("room_a") or ""); b=str(door.get("room_b") or "")
        width=_num(door.get("width"),f"door {door_id} width")
        if width<MIN_DOOR: issues.append(f"door {door_id} width {width} < {MIN_DOOR}")
        if a in placed and b in placed:
            pair=frozenset({a,b}); shared=_shared(_bounds(placed[a]),_bounds(placed[b]))
            paired=_openings(placed[a],door_id)+_openings(placed[b],door_id)
            if shared is None: issues.append(f"door {door_id} endpoints do not share a boundary")
            if len(paired)!=2 or any(str(opening.get("opening_type","")).lower()!="door" or abs(_num(opening.get("width"),"opening width")-width)>EPS for _,opening in paired): issues.append(f"door {door_id} lacks two matching generated wall openings")
            graph.setdefault(pair,[]).append({"id":door_id,"width":width,"paired_openings":len(paired)})
        elif a in placed and not b and str(door.get("door_type","")).lower()=="exterior":
            paired=_openings(placed[a],door_id)
            if len(paired)!=1: issues.append(f"exterior door {door_id} lacks one generated wall opening")
    return graph,issues


def _window_evidence(layout: Mapping[str,Any], placed: Mapping[str,dict[str,Any]])->tuple[dict[str,list[str]],list[str]]:
    result={room:[] for room in placed}; issues=[]
    for room_id,room in placed.items():
        for wall in _walls(room):
            for opening in wall.get("openings",[]) if isinstance(wall.get("openings"),list) else []:
                if not isinstance(opening,dict) or str(opening.get("opening_type","")).lower()!="window": continue
                if wall.get("is_exterior") is not True: issues.append(f"{room_id} window {opening.get('opening_id')} is not on an exterior wall")
                if _num(opening.get("width"),"window width")<1.0 or _num(opening.get("height"),"window height")<0.8: issues.append(f"{room_id} window is undersized")
                result[room_id].append(str(opening.get("opening_id")))
    for room_id in REQUIRED_WINDOWS:
        if not result.get(room_id): issues.append(f"{room_id} has no real exterior window opening")
    return result,issues


def _common_zone_evidence(layout: Mapping[str,Any], doors: Mapping[frozenset[str],list[dict[str,Any]]])->tuple[dict[str,Any],list[str]]:
    issues=[]; zones=layout.get("navigation_common_zones")
    if not isinstance(zones,list): return {},["navigation_common_zones missing"]
    by_id={str(z.get("id")):z for z in zones if isinstance(z,dict)}
    for required in ("entrance_transition","internal_circulation","toilet_foyer"):
        if required not in by_id: issues.append(f"navigation common zone {required} missing")
    foyer=by_id.get("toilet_foyer",{})
    if foyer.get("carved_from")!=["packaging_hall"]: issues.append("toilet_foyer must be carved from packaging_hall")
    connections=foyer.get("connections",[]) if isinstance(foyer.get("connections"),list) else []
    endpoints={str(c.get("to")) for c in connections if isinstance(c,dict)}
    if not {"boys_toilet","girls_toilet","internal_circulation"}.issubset(endpoints): issues.append("toilet_foyer lacks independent boys/girls/internal portals")
    backed={str(c.get("backing_door_id")) for c in connections if isinstance(c,dict) and c.get("backing_door_id")}
    if not {"door_packaging_boys","door_packaging_girls"}.issubset(backed): issues.append("toilet foyer portals are not backed by both real toilet doors")
    factory=layout.get("factory_common_zones")
    if not isinstance(factory,list): issues.append("factory_common_zones missing") ; factory=[]
    factory_ids={str(v.get("id")) for v in factory if isinstance(v,dict)}
    required_factory={"loading_dock","entrance_transition","internal_circulation","toilet_foyer","exterior_truck_road","landscaping"}
    if factory_ids!=required_factory: issues.append(f"factory common/exterior zone IDs differ: {sorted(factory_ids)}")
    return {"navigation_zone_ids":sorted(by_id),"factory_zone_ids":sorted(factory_ids)},issues


def validate(layout: dict[str,Any], contract_path: Path|None=None)->dict[str,Any]:
    issues=[]
    contract_path=contract_path or Path(__file__).with_name("factory_contract.json")
    contract=load_contract(contract_path); expected=set(room_ids(contract))
    try:
        specs=_map(layout,"rooms",expected); placed=_map(layout,"placed_rooms",expected)
        actual_bounds={room_id:_bounds(room) for room_id,room in placed.items()}
    except (LayoutError,KeyError,TypeError,ValueError) as exc:
        return {"schema_version":1,"status":"fail","profile":PROFILE,"critical_issues":[str(exc)]}
    expected_bounds={room_id:tuple(map(float,contract["rooms"][room_id]["bounds_xy"])) for room_id in expected}
    for room_id in expected:
        if any(abs(a-b)>EPS for a,b in zip(actual_bounds[room_id],expected_bounds[room_id])): issues.append(f"{room_id} bounds changed: {actual_bounds[room_id]} expected {expected_bounds[room_id]}")
    shell=(min(v[0] for v in actual_bounds.values()),min(v[1] for v in actual_bounds.values()),max(v[2] for v in actual_bounds.values()),max(v[3] for v in actual_bounds.values()))
    if shell!=(0.0,0.0,44.0,32.0) or _num(layout.get("wall_height"),"wall_height")!=5.0: issues.append(f"shell must be 44x32x5; bounds={shell} height={layout.get('wall_height')}")
    overlaps=[]
    ids=sorted(expected)
    for index,left in enumerate(ids):
        for right in ids[index+1:]:
            area=_overlap(actual_bounds[left],actual_bounds[right])
            if area>EPS: overlaps.append({"left":left,"right":right,"area_m2":area})
    if overlaps: issues.append(f"positive-area room overlaps: {overlaps}")
    try: doors,door_issues=_door_evidence(layout,placed); issues.extend(door_issues)
    except LayoutError as exc: doors={}; issues.append(str(exc))
    missing_workflow=[list(edge) for edge in WORKFLOW_EDGES if frozenset(edge) not in doors]
    if missing_workflow: issues.append(f"workflow edges lack real doors: {missing_workflow}")
    windows,window_issues=_window_evidence(layout,placed); issues.extend(window_issues)
    exterior={str(d.get("id")):d for d in layout.get("doors",[]) if isinstance(d,dict) and not d.get("room_b")}
    entrance=exterior.get("main_entrance"); loading=exterior.get("loading_dock_door")
    if not entrance or entrance.get("room_a")!="finished_goods_storage" or _num(entrance.get("width",0),"entrance")<2.0 or entrance.get("leaf_count")!=2: issues.append("south double pedestrian entrance is missing/invalid")
    if not loading or loading.get("room_a")!="finished_goods_storage" or _num(loading.get("width",0),"loading")<3.0: issues.append("southeast loading-dock roll-up door is missing/invalid")
    common,common_issues=_common_zone_evidence(layout,doors); issues.extend(common_issues)
    binding=layout.get("factory_contract_binding")
    if not isinstance(binding,dict) or binding.get("id")!=CONTRACT_ID or binding.get("sha256")!=contract["_sha256"]: issues.append("factory contract binding missing or stale")
    result={"schema_version":1,"status":"pass" if not issues else "fail","profile":PROFILE,"factory_contract_sha256":contract["_sha256"],"room_ids":list(room_ids(contract)),"room_bounds":{k:list(v) for k,v in sorted(actual_bounds.items())},"shell_bounds_xy":list(shell),"positive_area_overlaps":overlaps,"door_graph":{"|".join(sorted(k)):v for k,v in doors.items()},"missing_workflow_edges":missing_workflow,"windows":windows,"common_zones":common,"critical_issues":issues}
    result["attestation_sha256"]=hashlib.sha256(json.dumps({k:v for k,v in result.items() if k!="attestation_sha256"},sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return result


def main()->int:
    parser=argparse.ArgumentParser(); parser.add_argument("--layout",type=Path,required=True); parser.add_argument("--factory-contract",type=Path); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    layout=json.loads(args.layout.read_text(encoding="utf-8")); result=validate(layout,args.factory_contract)
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True)); return 0 if result["status"]=="pass" else 2


if __name__=="__main__": raise SystemExit(main())
