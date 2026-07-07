"""Room-level quality gate for SceneSmith outputs.

This script is intentionally conservative. It performs deterministic checks that
can block obviously broken rooms before final assembly: missing review images,
objects outside room bounds, extreme collision-complexity risk, suspicious
density, and missing/empty room state. It emits one JSON verdict per room plus a
summary file that assembly can enforce.

It does not replace a true VLM/SAGE reviewer. If a stronger visual judge is
available, write its result into the same JSON schema and keep assembly pointed
at the gate directory.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PASS_THRESHOLD = 7
DEFAULT_GATE_DIR = Path("quality_gates") / "room_self_exam"
IGNORED_TYPES = {"wall", "floor", "ceiling"}


@dataclass
class RoomSpec:
    room_id: str
    width: float
    depth: float
    prompt: str = ""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _as_room_list(placed_rooms: Any) -> list[dict[str, Any]]:
    if isinstance(placed_rooms, dict):
        return list(placed_rooms.values())
    if isinstance(placed_rooms, list):
        return placed_rooms
    return []


def _load_specs_from_layout(scene_dir: Path) -> dict[str, RoomSpec]:
    layout_path = scene_dir / "house_layout.json"
    if not layout_path.exists():
        raise FileNotFoundError(f"Missing house layout: {layout_path}")
    layout = _read_json(layout_path)
    placed_rooms = _as_room_list(layout.get("placed_rooms", []))
    specs: dict[str, RoomSpec] = {}
    for room in placed_rooms:
        room_id = str(room.get("room_id", ""))
        if not room_id:
            continue
        specs[room_id] = RoomSpec(
            room_id=room_id,
            width=float(room.get("width", room.get("length", 0.0))),
            depth=float(room.get("depth", room.get("length", 0.0))),
            prompt=str(room.get("prompt", "")),
        )
    return specs


def _load_rooms_from_scene_dir(
    scene_dir: Path, specs: dict[str, RoomSpec]
) -> dict[str, dict[str, Any]]:
    rooms: dict[str, dict[str, Any]] = {}
    for room_id in specs:
        state_path = (
            scene_dir
            / f"room_{room_id}"
            / "scene_states"
            / "final_scene"
            / "scene_state.json"
        )
        if state_path.exists():
            rooms[room_id] = _read_json(state_path)
    return rooms


def _load_from_house_state(house_state: Path) -> tuple[dict[str, RoomSpec], dict[str, dict[str, Any]]]:
    state = _read_json(house_state)
    layout = state.get("layout", {})
    specs: dict[str, RoomSpec] = {}
    for room in _as_room_list(layout.get("placed_rooms", [])):
        room_id = str(room.get("room_id", ""))
        if room_id:
            specs[room_id] = RoomSpec(
                room_id=room_id,
                width=float(room.get("width", 0.0)),
                depth=float(room.get("depth", 0.0)),
                prompt=str(room.get("prompt", "")),
            )
    rooms = state.get("rooms", {})
    if not isinstance(rooms, dict):
        rooms = {}
    return specs, rooms


def _object_iter(room_state: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    objects = room_state.get("objects", {})
    if isinstance(objects, dict):
        return [(str(k), v) for k, v in objects.items() if isinstance(v, dict)]
    if isinstance(objects, list):
        out = []
        for idx, obj in enumerate(objects):
            if isinstance(obj, dict):
                out.append((str(obj.get("object_id", idx)), obj))
        return out
    return []


def _room_dimensions(spec: RoomSpec, room_state: dict[str, Any]) -> tuple[float, float]:
    width = spec.width
    depth = spec.depth
    geometry = room_state.get("room_geometry", {})
    if isinstance(geometry, dict):
        width = width or float(geometry.get("width", 0.0))
        depth = depth or float(geometry.get("depth", geometry.get("length", 0.0)))
    return width, depth


def _bbox_bounds(obj: dict[str, Any]) -> tuple[float, float, float, float] | None:
    transform = obj.get("transform", {})
    translation = transform.get("translation") if isinstance(transform, dict) else None
    bbox_min = obj.get("bbox_min")
    bbox_max = obj.get("bbox_max")
    if not (
        isinstance(translation, list)
        and len(translation) >= 2
        and isinstance(bbox_min, list)
        and isinstance(bbox_max, list)
        and len(bbox_min) >= 2
        and len(bbox_max) >= 2
    ):
        return None
    tx = float(translation[0])
    ty = float(translation[1])
    return (
        tx + float(bbox_min[0]),
        tx + float(bbox_max[0]),
        ty + float(bbox_min[1]),
        ty + float(bbox_max[1]),
    )


def _find_review_images(review_dir: Path, room_id: str) -> list[str]:
    if not review_dir.exists():
        return []
    patterns = [
        f"{room_id}*.png",
        f"{room_id}*.jpg",
        f"{room_id}*.jpeg",
        f"*{room_id}*.png",
        f"*{room_id}*.jpg",
        f"*{room_id}*.jpeg",
    ]
    found: dict[str, Path] = {}
    for pattern in patterns:
        for path in review_dir.rglob(pattern):
            if path.is_file():
                found[str(path.resolve())] = path
    return [str(path) for path in sorted(found.values())]


def _score_room(
    room_id: str,
    spec: RoomSpec,
    room_state: dict[str, Any] | None,
    review_images: list[str],
    max_collision_hulls: int,
) -> dict[str, Any]:
    issues: list[str] = []
    repairs: list[str] = []

    if room_state is None:
        return {
            "room_id": room_id,
            "status": "fail",
            "scores": {
                "object_relevance": 0,
                "placement_realism": 0,
                "clearance_and_access": 0,
                "collision_risk": 0,
                "prompt_alignment": 0,
            },
            "critical_issues": [f"Missing final room state for {room_id}."],
            "repair_instructions": ["Regenerate this room before assembly."],
            "review_images": review_images,
            "metrics": {},
        }

    width, depth = _room_dimensions(spec, room_state)
    if width <= 0.0 or depth <= 0.0:
        issues.append("Room dimensions are missing or invalid.")
        repairs.append("Regenerate or repair the floor-plan/room geometry state.")

    objects = [
        (oid, obj)
        for oid, obj in _object_iter(room_state)
        if str(obj.get("object_type", "")).lower() not in IGNORED_TYPES
        and not bool(obj.get("immutable", False))
    ]
    object_count = len(objects)
    if object_count == 0:
        issues.append("Room contains no non-structural objects.")
        repairs.append("Rerun room generation from the furniture stage.")

    if not review_images:
        issues.append("No room review image found.")
        repairs.append("Render review images before running the gate.")

    x_min_room = -width / 2.0
    x_max_room = width / 2.0
    y_min_room = -depth / 2.0
    y_max_room = depth / 2.0
    margin = 0.15
    overflow_objects: list[str] = []
    unknown_bbox = 0
    object_area = 0.0

    for oid, obj in objects:
        bounds = _bbox_bounds(obj)
        if bounds is None:
            unknown_bbox += 1
            continue
        x0, x1, y0, y1 = bounds
        object_area += max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if (
            x0 < x_min_room - margin
            or x1 > x_max_room + margin
            or y0 < y_min_room - margin
            or y1 > y_max_room + margin
        ):
            overflow_objects.append(oid)

    if overflow_objects:
        issues.append(
            f"{len(overflow_objects)} objects extend outside room bounds: "
            + ", ".join(overflow_objects[:10])
        )
        repairs.append("Repair placement or regenerate the room from placement/furniture stage.")

    room_area = max(width * depth, 1e-6)
    density = object_area / room_area
    if density > 0.65:
        issues.append(f"Object footprint density is too high ({density:.2f}).")
        repairs.append("Remove or reposition objects to keep navigable free space.")

    collision_hull_risk = 0
    for oid, obj in objects:
        metadata = obj.get("metadata", {})
        if isinstance(metadata, dict):
            hull_count = metadata.get("collision_hulls") or metadata.get("convex_hulls")
            if isinstance(hull_count, int) and hull_count > max_collision_hulls:
                collision_hull_risk += 1
    if collision_hull_risk:
        issues.append(
            f"{collision_hull_risk} objects exceed collision hull cap {max_collision_hulls}."
        )
        repairs.append("Regenerate collision meshes with max hull cap <= 32.")

    placement = 10
    clearance = 10
    collision = 10
    relevance = 8
    prompt_alignment = 8

    if overflow_objects:
        placement -= min(6, 2 + len(overflow_objects))
        clearance -= min(6, 2 + len(overflow_objects))
    if density > 0.65:
        clearance -= 4
        placement -= 2
    elif density > 0.45:
        clearance -= 2
    if unknown_bbox:
        collision -= min(3, unknown_bbox)
    if collision_hull_risk:
        collision -= min(5, collision_hull_risk)
    if object_count == 0:
        relevance = 0
        prompt_alignment = 0
    if not review_images:
        placement = min(placement, 6)
        prompt_alignment = min(prompt_alignment, 6)

    scores = {
        "object_relevance": max(0, min(10, relevance)),
        "placement_realism": max(0, min(10, placement)),
        "clearance_and_access": max(0, min(10, clearance)),
        "collision_risk": max(0, min(10, collision)),
        "prompt_alignment": max(0, min(10, prompt_alignment)),
    }
    status = (
        "pass"
        if not issues
        and scores["placement_realism"] >= PASS_THRESHOLD
        and scores["clearance_and_access"] >= PASS_THRESHOLD
        and scores["collision_risk"] >= PASS_THRESHOLD
        else "fail"
    )

    return {
        "room_id": room_id,
        "status": status,
        "scores": scores,
        "critical_issues": issues,
        "repair_instructions": repairs,
        "review_images": review_images,
        "metrics": {
            "room_width": width,
            "room_depth": depth,
            "object_count": object_count,
            "unknown_bbox_count": unknown_bbox,
            "overflow_object_count": len(overflow_objects),
            "object_footprint_density": round(density, 4)
            if math.isfinite(density)
            else None,
            "max_collision_hulls": max_collision_hulls,
        },
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", help="Scene directory containing house_layout.json.")
    parser.add_argument("--house-state", help="Combined house_state.json alternative.")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--rooms", nargs="*", help="Optional room IDs to check.")
    parser.add_argument("--max-collision-hulls", type=int, default=32)
    args = parser.parse_args()

    if not args.scene_dir and not args.house_state:
        raise SystemExit("Provide --scene-dir or --house-state.")

    review_dir = Path(args.review_dir).resolve()
    if args.scene_dir:
        scene_dir = Path(args.scene_dir).resolve()
        specs = _load_specs_from_layout(scene_dir)
        rooms = _load_rooms_from_scene_dir(scene_dir, specs)
        default_output = scene_dir / DEFAULT_GATE_DIR
    else:
        house_state = Path(args.house_state).resolve()
        specs, rooms = _load_from_house_state(house_state)
        default_output = house_state.parent / DEFAULT_GATE_DIR

    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output
    room_ids = args.rooms or sorted(specs)
    if not room_ids:
        raise RuntimeError("No rooms found to examine.")

    results = []
    for room_id in room_ids:
        if room_id not in specs:
            result = {
                "room_id": room_id,
                "status": "fail",
                "scores": {
                    "object_relevance": 0,
                    "placement_realism": 0,
                    "clearance_and_access": 0,
                    "collision_risk": 0,
                    "prompt_alignment": 0,
                },
                "critical_issues": [f"Room {room_id} is missing from layout."],
                "repair_instructions": ["Do not assemble until layout/room state is fixed."],
                "review_images": [],
                "metrics": {},
            }
        else:
            result = _score_room(
                room_id=room_id,
                spec=specs[room_id],
                room_state=rooms.get(room_id),
                review_images=_find_review_images(review_dir, room_id),
                max_collision_hulls=args.max_collision_hulls,
            )
        results.append(result)
        _write_json(output_dir / f"{room_id}.json", result)

    failed = [r["room_id"] for r in results if r["status"] != "pass"]
    summary = {
        "status": "pass" if not failed else "fail",
        "passed_rooms": [r["room_id"] for r in results if r["status"] == "pass"],
        "failed_rooms": failed,
        "room_count": len(results),
        "gate_dir": str(output_dir),
    }
    _write_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
