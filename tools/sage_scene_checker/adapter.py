"""Adapter from SceneSmith outputs to a SAGE-like FloorPlan dictionary."""

from __future__ import annotations

import json
import math
import re

from pathlib import Path
from typing import Any


KNOWN_CATEGORIES = {
    "furniture",
    "manipuland",
    "wall_mounted",
    "ceiling_mounted",
    "wall",
    "floor",
    "object",
    "unknown",
}


def load_scenesmith_output(scene_dir: str | Path) -> dict[str, Any]:
    """Load a SceneSmith scene directory and return a SAGE-like floor plan."""

    root = Path(scene_dir)
    combined = root / "combined_house"
    house_state_path = combined / "house_state.json"
    sceneeval_path = combined / "sceneeval_state.json"
    dmd_path = combined / "house.dmd.yaml"

    house_state = _load_json(house_state_path)
    sceneeval_state = _load_json(sceneeval_path) if sceneeval_path.exists() else None
    layout = house_state.get("layout", {})
    rooms = _convert_rooms(layout)
    room_lookup = {room["id"]: room for room in rooms}
    objects = _convert_objects(
        house_state=house_state,
        sceneeval_state=sceneeval_state,
        scene_dir=root,
        room_lookup=room_lookup,
    )

    return {
        "id": root.name,
        "source_scene_dir": str(root),
        "source_files": {
            "house_state": str(house_state_path),
            "sceneeval_state": str(sceneeval_path),
            "house_dmd": str(dmd_path),
        },
        "has_house_dmd": dmd_path.exists(),
        "building_style": "scenesmith",
        "description": layout.get("house_prompt", ""),
        "created_from_text": layout.get("house_prompt", ""),
        "rooms": rooms,
        "objects": objects,
        "raw": {
            "house_state": house_state,
            "sceneeval_state": sceneeval_state,
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _convert_rooms(layout: dict[str, Any]) -> list[dict[str, Any]]:
    specs = {room.get("id") or room.get("room_id"): room for room in layout.get("rooms", [])}
    placed_rooms = layout.get("placed_rooms") or []
    rooms: list[dict[str, Any]] = []

    if placed_rooms:
        for placed in placed_rooms:
            room_id = str(placed.get("room_id") or placed.get("id") or "room")
            spec = specs.get(room_id, {})
            x, y = _xy(placed.get("position"), default=(0.0, 0.0))
            width = _float_or(placed.get("width"), spec.get("length", 0.0))
            length = _float_or(placed.get("depth"), spec.get("width", 0.0))
            walls, doors, windows = _convert_walls_doors_windows(placed)
            rooms.append(
                {
                    "id": room_id,
                    "room_type": str(spec.get("type") or spec.get("room_type") or room_id),
                    "position": {"x": x, "y": y, "z": 0.0},
                    "dimensions": {
                        "width": width,
                        "length": length,
                        "height": _float_or(layout.get("wall_height"), 2.7),
                    },
                    "walls": walls,
                    "doors": doors,
                    "windows": windows,
                    "objects": [],
                }
            )
    else:
        for room_id, spec in specs.items():
            if room_id is None:
                continue
            x, y = _xy(spec.get("position"), default=(0.0, 0.0))
            width = _float_or(spec.get("length"), 0.0)
            length = _float_or(spec.get("width"), 0.0)
            rooms.append(
                {
                    "id": str(room_id),
                    "room_type": str(spec.get("type") or room_id),
                    "position": {"x": x, "y": y, "z": 0.0},
                    "dimensions": {
                        "width": width,
                        "length": length,
                        "height": _float_or(layout.get("wall_height"), 2.7),
                    },
                    "walls": [],
                    "doors": [],
                    "windows": [],
                    "objects": [],
                }
            )
    return rooms


def _convert_walls_doors_windows(placed: dict[str, Any]) -> tuple[list[dict], list[dict], list[dict]]:
    walls: list[dict] = []
    doors: list[dict] = []
    windows: list[dict] = []
    for wall in placed.get("walls", []) or []:
        wall_id = str(wall.get("wall_id") or wall.get("id") or "")
        start = _point3(wall.get("start_point"))
        end = _point3(wall.get("end_point"))
        walls.append(
            {
                "id": wall_id,
                "wall_id": wall_id,
                "room_id": wall.get("room_id") or placed.get("room_id"),
                "direction": wall.get("direction"),
                "start_point": start,
                "end_point": end,
                "height": _float_or(wall.get("height"), 0.0),
                "length": _float_or(wall.get("length"), _distance_2d(start, end)),
            }
        )
        for opening in wall.get("openings", []) or []:
            item = _opening_to_sage(opening, wall, wall_id)
            if item["kind"] == "door":
                doors.append(item)
            elif item["kind"] == "window":
                windows.append(item)
    return walls, doors, windows


def _opening_to_sage(opening: dict[str, Any], wall: dict[str, Any], wall_id: str) -> dict:
    kind = str(opening.get("opening_type") or opening.get("type") or "").lower()
    if "window" in kind:
        normalized_kind = "window"
    else:
        normalized_kind = "door"
    wall_length = max(_float_or(wall.get("length"), 0.0), 1e-9)
    raw_pos = _float_or(opening.get("position_along_wall"), 0.0)
    return {
        "kind": normalized_kind,
        "id": str(opening.get("opening_id") or opening.get("id") or f"{wall_id}_{normalized_kind}"),
        "wall_id": wall_id,
        "wall_side": wall.get("direction"),
        "position_on_wall": max(0.0, min(1.0, raw_pos / wall_length)),
        "position_along_wall": raw_pos,
        "width": _float_or(opening.get("width"), 0.0),
        "height": _float_or(opening.get("height"), 0.0),
        "sill_height": _float_or(opening.get("sill_height"), 0.0),
    }


def _convert_objects(
    *,
    house_state: dict[str, Any],
    sceneeval_state: dict[str, Any] | None,
    scene_dir: Path,
    room_lookup: dict[str, dict],
) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    sceneeval_objects = _sceneeval_object_lookup(sceneeval_state)
    for room_id, room_state in (house_state.get("rooms") or {}).items():
        room_base = scene_dir / f"room_{room_id}"
        for object_id, obj in (room_state.get("objects") or {}).items():
            converted = _convert_object(
                object_id=str(object_id),
                room_id=str(room_id),
                obj=obj,
                room_base=room_base,
                sceneeval_obj=sceneeval_objects.get(str(object_id)),
            )
            objects.append(converted)
            if room_id in room_lookup:
                room_lookup[room_id]["objects"].append(converted["id"])
    return objects


def _sceneeval_object_lookup(sceneeval_state: dict[str, Any] | None) -> dict[str, dict]:
    if not sceneeval_state:
        return {}
    scene = sceneeval_state.get("scene") or {}
    raw_objects = scene.get("object") or scene.get("objects") or []
    return {str(obj.get("id")): obj for obj in raw_objects if obj.get("id") is not None}


def _convert_object(
    *,
    object_id: str,
    room_id: str,
    obj: dict[str, Any],
    room_base: Path,
    sceneeval_obj: dict[str, Any] | None,
) -> dict[str, Any]:
    transform = obj.get("transform") or {}
    translation = _translation_from_sceneeval(sceneeval_obj) or _vector3(
        transform.get("translation")
    )
    rotation_wxyz = _vector4(transform.get("rotation_wxyz"), default=(1.0, 0.0, 0.0, 0.0))
    bbox_min = _vector3_or_none(obj.get("bbox_min"))
    bbox_max = _vector3_or_none(obj.get("bbox_max"))
    dims = _dimensions_from_bbox(bbox_min, bbox_max)
    category = normalize_category(
        obj.get("object_type")
        or (obj.get("metadata") or {}).get("category")
        or obj.get("name")
        or obj.get("description")
    )
    asset_path = _first_present_path(obj, room_base)
    return {
        "id": object_id,
        "room_id": room_id,
        "type": category,
        "raw_type": obj.get("object_type"),
        "description": obj.get("description") or obj.get("name") or object_id,
        "position": {"x": translation[0], "y": translation[1], "z": translation[2]},
        "rotation": _quaternion_wxyz_to_euler_degrees(rotation_wxyz),
        "dimensions": {
            "width": dims[0],
            "length": dims[1],
            "height": dims[2],
        },
        "bbox_min": list(bbox_min) if bbox_min else None,
        "bbox_max": list(bbox_max) if bbox_max else None,
        "asset_path": str(asset_path) if asset_path else None,
        "asset_path_exists": bool(asset_path and asset_path.exists()),
        "source": "scenesmith",
        "source_id": obj.get("hssd_id") or obj.get("objaverse_uid") or object_id,
        "place_id": _place_id(obj),
        "placement_info": obj.get("placement_info"),
        "support_surfaces": obj.get("support_surfaces") or [],
        "metadata": obj.get("metadata") or {},
    }


def normalize_category(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return "unknown"
    text = re.sub(r"[^a-z0-9_ -]+", "", text).replace("-", "_").replace(" ", "_")
    if text in KNOWN_CATEGORIES:
        return text
    if "wall" in text and "mount" in text:
        return "wall_mounted"
    if "ceiling" in text:
        return "ceiling_mounted"
    if any(token in text for token in ("desk", "chair", "table", "sofa", "shelf", "cabinet", "bed")):
        return "furniture"
    return text


def _first_present_path(obj: dict[str, Any], room_base: Path) -> Path | None:
    for key in ("sdf_path", "geometry_path", "image_path"):
        raw = obj.get(key)
        if not raw:
            continue
        path = Path(str(raw))
        if not path.is_absolute():
            path = room_base / path
        return path
    return None


def _place_id(obj: dict[str, Any]) -> str:
    placement = obj.get("placement_info") or {}
    if placement.get("parent_surface_id"):
        return str(placement["parent_surface_id"])
    raw_type = str(obj.get("object_type") or "").lower()
    if "wall" in raw_type:
        return "wall"
    if "ceiling" in raw_type:
        return "ceiling"
    return "floor"


def _translation_from_sceneeval(sceneeval_obj: dict[str, Any] | None) -> tuple[float, float, float] | None:
    if not sceneeval_obj:
        return None
    matrix = ((sceneeval_obj.get("transform") or {}).get("data") or [])
    if len(matrix) >= 15:
        return (_float_or(matrix[12], 0.0), _float_or(matrix[13], 0.0), _float_or(matrix[14], 0.0))
    return None


def _dimensions_from_bbox(
    bbox_min: tuple[float, float, float] | None,
    bbox_max: tuple[float, float, float] | None,
) -> tuple[float, float, float]:
    if not bbox_min or not bbox_max:
        return (0.0, 0.0, 0.0)
    return tuple(max(0.0, bbox_max[i] - bbox_min[i]) for i in range(3))  # type: ignore[return-value]


def _quaternion_wxyz_to_euler_degrees(q: tuple[float, float, float, float]) -> dict[str, float]:
    w, x, y, z = q
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0:
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    w, x, y, z = (w / norm, x / norm, y / norm, z / norm)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return {
        "x": math.degrees(roll),
        "y": math.degrees(pitch),
        "z": math.degrees(yaw),
    }


def _xy(value: Any, *, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, dict):
        return (_float_or(value.get("x"), default[0]), _float_or(value.get("y"), default[1]))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (_float_or(value[0], default[0]), _float_or(value[1], default[1]))
    return default


def _point3(value: Any) -> dict[str, float]:
    x, y = _xy(value, default=(0.0, 0.0))
    z = _float_or(value.get("z"), 0.0) if isinstance(value, dict) else 0.0
    return {"x": x, "y": y, "z": z}


def _vector3(value: Any) -> tuple[float, float, float]:
    return _vector3_or_none(value) or (0.0, 0.0, 0.0)


def _vector3_or_none(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, dict):
        return (_float_or(value.get("x"), 0.0), _float_or(value.get("y"), 0.0), _float_or(value.get("z"), 0.0))
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return (_float_or(value[0], 0.0), _float_or(value[1], 0.0), _float_or(value[2], 0.0))
    return None


def _vector4(value: Any, *, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return (_float_or(value[0], default[0]), _float_or(value[1], default[1]), _float_or(value[2], default[2]), _float_or(value[3], default[3]))
    return default


def _distance_2d(a: dict[str, float], b: dict[str, float]) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _float_or(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)

