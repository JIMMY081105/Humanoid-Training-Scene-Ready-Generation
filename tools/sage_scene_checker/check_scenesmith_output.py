"""CLI for validating SceneSmith outputs with cheap SAGE-style checks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys

from pathlib import Path
from typing import Any, Callable

try:
    from .adapter import load_scenesmith_output
    from .money_guard import BackendUsage, MoneyGuardError, check_money_guard
    from .repair_prompt_builder import build_repair_suggestions
except ImportError:  # Allows `python tools/sage_scene_checker/check_...py`.
    from adapter import load_scenesmith_output
    from money_guard import BackendUsage, MoneyGuardError, check_money_guard
    from repair_prompt_builder import build_repair_suggestions


Check = dict[str, Any]

FORBIDDEN_SAGE_IMPORT_MARKERS = (
    "import key",
    "from key",
    "import utils",
    "from utils",
    "openai",
    "anthropic",
    "gemini",
    "google.generativeai",
    "dashscope",
    "nvidia",
    "ark_api",
    "beaver3d",
    "isaacsim",
    "omni.",
)

REQUIRED_SCENE_FILES = {
    "house_state": Path("combined_house/house_state.json"),
    "sceneeval_state": Path("combined_house/sceneeval_state.json"),
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_checker(
        scene_dir=args.scene_dir,
        out=args.out,
        sage_root=args.sage_root,
        no_paid_api=args.no_paid_api,
        fail_on_warnings=args.fail_on_warnings,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["pass"] else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--sage-root")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-paid-api", action="store_true")
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help=(
            "treat every failed warning check as fatal; intended for production "
            "acceptance while the default remains warning-tolerant"
        ),
    )
    return parser.parse_args(argv)


def run_checker(
    *,
    scene_dir: str | Path,
    out: str | Path | None = None,
    sage_root: str | Path | None = None,
    no_paid_api: bool = False,
    fail_on_warnings: bool = False,
) -> dict[str, Any]:
    scene_path = Path(scene_dir)
    checks: list[Check] = []
    usage = BackendUsage(codex_cli_calls=0)

    try:
        money_guard = check_money_guard(no_paid_api=no_paid_api, usage=usage)
    except MoneyGuardError as exc:
        money_guard = exc.result
        checks.append(
            _check(
                "money_guard.paid_env_present",
                "error",
                "fail",
                f"Paid API env vars present: {', '.join(exc.result.present_env_vars)}",
            )
        )

    required = _required_files(scene_path)
    for name, path in required.items():
        checks.append(
            _check(
                f"required_file.{name}",
                "error",
                "pass" if path.exists() else "fail",
                f"Required file {'exists' if path.exists() else 'missing'}: {path}",
            )
        )

    floor_plan: dict[str, Any] | None = None
    if all(path.exists() for path in (required["house_state"], required["sceneeval_state"])):
        try:
            floor_plan = load_scenesmith_output(scene_path)
            checks.append(_check("json_parse.scene_state", "error", "pass", "SceneSmith JSON parsed"))
        except Exception as exc:  # noqa: BLE001 - report every parse failure.
            checks.append(_check("json_parse.scene_state", "error", "fail", str(exc)))
    else:
        checks.append(_check("json_parse.scene_state", "error", "fail", "Cannot parse missing required JSON files"))

    if floor_plan is not None:
        checks.extend(_run_local_checks(floor_plan))
        checks.extend(_run_optional_sage_checks(floor_plan, sage_root))

    errors = [c for c in checks if c["status"] == "fail" and c["severity"] == "error"]
    warnings = [c for c in checks if c["status"] == "fail" and c["severity"] == "warning"]
    report = {
        "scene_id": scene_path.name,
        "pass": not errors and (not fail_on_warnings or not warnings),
        "acceptance_policy": {
            "fail_on_warnings": fail_on_warnings,
            "fatal_failed_check_ids": [
                check["id"]
                for check in errors + (warnings if fail_on_warnings else [])
            ],
        },
        "summary": {
            "num_rooms": len(floor_plan.get("rooms", [])) if floor_plan else 0,
            "num_objects": len(floor_plan.get("objects", [])) if floor_plan else 0,
            "num_errors": len(errors),
            "num_warnings": len(warnings),
        },
        "money_guard": money_guard.to_report_dict(),
        "checks": checks,
        "repair_suggestions": build_repair_suggestions(checks),
    }
    if out:
        report["output_path"] = str(out)
    return report


def _required_files(scene_path: Path) -> dict[str, Path]:
    return {name: scene_path / relative_path for name, relative_path in REQUIRED_SCENE_FILES.items()}


def _run_local_checks(floor_plan: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    rooms = floor_plan.get("rooms", [])
    objects = floor_plan.get("objects", [])
    checks.append(
        _check(
            "count.rooms",
            "error",
            "pass" if rooms else "fail",
            f"Room count: {len(rooms)}",
        )
    )
    checks.append(
        _check(
            "count.objects",
            "warning",
            "pass" if objects else "fail",
            f"Object count: {len(objects)}",
        )
    )

    for room in rooms:
        checks.extend(_check_room(room))
    for obj in objects:
        checks.extend(_check_object(obj, rooms, objects))
    checks.extend(_check_collisions(objects))
    checks.extend(_check_door_clearance(rooms, objects))
    return checks


def _check_room(room: dict[str, Any]) -> list[Check]:
    dims = room.get("dimensions", {})
    ok = _positive_finite(dims.get("width")) and _positive_finite(dims.get("length"))
    return [
        _check(
            "room.dimensions",
            "error",
            "pass" if ok else "fail",
            f"Room {room.get('id')} has valid dimensions",
            room_id=room.get("id"),
        )
    ]


def _check_object(obj: dict[str, Any], rooms: list[dict], objects: list[dict]) -> list[Check]:
    checks: list[Check] = []
    object_id = obj.get("id")
    position = obj.get("position", {})
    dims = obj.get("dimensions", {})
    bbox_min = obj.get("bbox_min")
    bbox_max = obj.get("bbox_max")
    finite_transform = all(
        _finite(position.get(axis)) for axis in ("x", "y", "z")
    )
    checks.append(
        _check(
            "object.transform_finite",
            "error",
            "pass" if finite_transform else "fail",
            f"Object {object_id} has finite transform",
            object_id=object_id,
        )
    )
    valid_bbox = (
        bbox_min is not None
        and bbox_max is not None
        and all(_finite(v) for v in bbox_min + bbox_max)
        and all(_positive_finite(dims.get(axis)) for axis in ("width", "length", "height"))
    )
    checks.append(
        _check(
            "object.bbox_valid",
            "error",
            "pass" if valid_bbox else "fail",
            f"Object {object_id} has valid bbox dimensions",
            object_id=object_id,
        )
    )
    room = next((r for r in rooms if r.get("id") == obj.get("room_id")), None)
    inside = _object_inside_room(obj, room) if room else False
    checks.append(
        _check(
            "object.inside_room",
            "warning",
            "pass" if inside else "fail",
            f"Object {object_id} is inside assigned room",
            object_id=object_id,
            room_id=obj.get("room_id"),
        )
    )
    asset_ok = bool(obj.get("asset_path") and obj.get("asset_path_exists"))
    checks.append(
        _check(
            "object.asset_path",
            "warning",
            "pass" if asset_ok else "fail",
            f"Object {object_id} has an existing local asset path",
            object_id=object_id,
        )
    )
    category_ok = bool(obj.get("type") and obj.get("type") != "unknown")
    checks.append(
        _check(
            "object.semantic_category",
            "warning",
            "pass" if category_ok else "fail",
            f"Object {object_id} has semantic category",
            object_id=object_id,
        )
    )
    checks.append(_check_support_surface_parent(obj, objects))
    return checks


def _check_support_surface_parent(obj: dict[str, Any], objects: list[dict]) -> Check:
    placement = obj.get("placement_info") or {}
    parent = placement.get("parent_surface_id")
    if not parent:
        return _check(
            "object.support_surface_parent",
            "info",
            "pass",
            f"Object {obj.get('id')} has no support parent metadata",
            object_id=obj.get("id"),
        )
    support_ids = {
        str(surface.get("surface_id"))
        for candidate in objects
        for surface in candidate.get("support_surfaces", [])
    }
    return _check(
        "object.support_surface_parent",
        "warning",
        "pass" if str(parent) in support_ids else "fail",
        f"Object {obj.get('id')} support parent exists",
        object_id=obj.get("id"),
    )


def _check_collisions(objects: list[dict]) -> list[Check]:
    checks: list[Check] = []
    for i, first in enumerate(objects):
        for second in objects[i + 1 :]:
            if first.get("room_id") != second.get("room_id"):
                continue
            overlap = _aabb_overlap(first, second)
            if overlap > 0.01:
                checks.append(
                    _check(
                        "object.coarse_collision",
                        "warning",
                        "fail",
                        f"Coarse AABB overlap {overlap:.4f} between {first.get('id')} and {second.get('id')}",
                        object_id=first.get("id"),
                        related_object_id=second.get("id"),
                    )
                )
    if not checks:
        checks.append(_check("object.coarse_collision", "warning", "pass", "No coarse AABB collisions detected"))
    return checks


def _check_door_clearance(rooms: list[dict], objects: list[dict]) -> list[Check]:
    checks: list[Check] = []
    for room in rooms:
        for door in room.get("doors", []):
            clearance = _door_clearance_box(room, door)
            if not clearance:
                continue
            for obj in [item for item in objects if item.get("room_id") == room.get("id")]:
                if _aabb_intersection_volume(_object_world_aabb(obj), clearance) > 0.01:
                    checks.append(
                        _check(
                            "door.clearance_blocked",
                            "warning",
                            "fail",
                            f"Object {obj.get('id')} blocks door {door.get('id')}",
                            object_id=obj.get("id"),
                            room_id=room.get("id"),
                        )
                    )
    if not checks:
        checks.append(_check("door.clearance_blocked", "warning", "pass", "No door clearance blockage detected"))
    return checks


def _run_optional_sage_checks(floor_plan: dict[str, Any], sage_root: str | Path | None) -> list[Check]:
    if not sage_root:
        return [_check("sage.optional", "info", "pass", "SAGE root not provided; used built-in checks")]
    validation_path = Path(sage_root) / "server" / "validation.py"
    if not validation_path.exists():
        return [_check("sage.optional", "warning", "fail", f"SAGE validation.py not found: {validation_path}")]
    safe, reason = _sage_validation_file_is_safe(validation_path)
    if not safe:
        return [_check("sage.optional", "warning", "fail", f"SAGE lightweight import skipped: {reason}")]
    try:
        module = _load_module_from_path("sage_validation_lightweight", validation_path)
        validate_room_only_layout: Callable[[list[dict]], dict] = module.validate_room_only_layout
        check_room_overlap: Callable[[dict, dict], bool] = module.check_room_overlap
    except Exception as exc:  # noqa: BLE001 - optional integration must not crash checker.
        return [_check("sage.optional", "warning", "fail", f"SAGE lightweight import skipped: {exc}")]
    rooms_data = [
        {
            "room_type": room.get("room_type") or room.get("id"),
            "position": {
                "x": room.get("position", {}).get("x", 0.0),
                "y": room.get("position", {}).get("y", 0.0),
            },
            "dimensions": {
                "width": room.get("dimensions", {}).get("width", 0.0),
                "length": room.get("dimensions", {}).get("length", 0.0),
            },
        }
        for room in floor_plan.get("rooms", [])
    ]
    try:
        result = validate_room_only_layout(rooms_data)
    except Exception as exc:  # noqa: BLE001
        return [_check("sage.room_only_layout", "warning", "fail", f"SAGE room-only validation failed: {exc}")]
    overlap_pairs = []
    for index, first in enumerate(rooms_data):
        for second in rooms_data[index + 1 :]:
            try:
                if check_room_overlap(first, second):
                    overlap_pairs.append((first.get("room_type"), second.get("room_type")))
            except Exception as exc:  # noqa: BLE001
                return [_check("sage.check_room_overlap", "warning", "fail", f"SAGE room-overlap check failed: {exc}")]
    return [
        _check(
            "sage.room_only_layout",
            "warning",
            "pass" if result.get("valid") else "fail",
            f"SAGE room-only layout result: {result}",
        ),
        _check(
            "sage.check_room_overlap",
            "warning",
            "pass" if not overlap_pairs else "fail",
            f"SAGE room-overlap pairs: {overlap_pairs}",
        )
    ]


def _sage_validation_file_is_safe(path: Path) -> tuple[bool, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")
    lower = text.lower()
    found = [marker for marker in FORBIDDEN_SAGE_IMPORT_MARKERS if marker in lower]
    if found:
        return False, f"forbidden markers in validation.py: {', '.join(found)}"
    return True, "validation.py has no forbidden lightweight-import markers"


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _object_inside_room(obj: dict, room: dict | None) -> bool:
    if room is None:
        return False
    aabb = _object_world_aabb(obj)
    if aabb is None:
        return False
    room_pos = room.get("position", {})
    dims = room.get("dimensions", {})
    min_x = _num(room_pos.get("x"))
    min_y = _num(room_pos.get("y"))
    max_x = min_x + _num(dims.get("width"))
    max_y = min_y + _num(dims.get("length"))
    return (
        aabb[0][0] >= min_x - 0.05
        and aabb[1][0] <= max_x + 0.05
        and aabb[0][1] >= min_y - 0.05
        and aabb[1][1] <= max_y + 0.05
    )


def _aabb_overlap(first: dict, second: dict) -> float:
    return _aabb_intersection_volume(_object_world_aabb(first), _object_world_aabb(second))


def _aabb_intersection_volume(
    first: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    second: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
) -> float:
    if first is None or second is None:
        return 0.0
    dx = min(first[1][0], second[1][0]) - max(first[0][0], second[0][0])
    dy = min(first[1][1], second[1][1]) - max(first[0][1], second[0][1])
    dz = min(first[1][2], second[1][2]) - max(first[0][2], second[0][2])
    if dx <= 0 or dy <= 0 or dz <= 0:
        return 0.0
    return dx * dy * dz


def _object_world_aabb(obj: dict) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    bbox_min = obj.get("bbox_min")
    bbox_max = obj.get("bbox_max")
    pos = obj.get("position", {})
    if not bbox_min or not bbox_max:
        return None
    center = (_num(pos.get("x")), _num(pos.get("y")), _num(pos.get("z")))
    return (
        tuple(center[i] + _num(bbox_min[i]) for i in range(3)),  # type: ignore[return-value]
        tuple(center[i] + _num(bbox_max[i]) for i in range(3)),  # type: ignore[return-value]
    )


def _door_clearance_box(room: dict, door: dict) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    width = _num(door.get("width"))
    if width <= 0:
        return None
    side = str(door.get("wall_side") or "").lower()
    if side not in {"north", "south", "east", "west"}:
        return None
    room_pos = room.get("position", {})
    dims = room.get("dimensions", {})
    x0 = _num(room_pos.get("x"))
    y0 = _num(room_pos.get("y"))
    room_w = _num(dims.get("width"))
    room_l = _num(dims.get("length"))
    t = _num(door.get("position_on_wall"))
    depth = 1.0
    if side in {"north", "south"}:
        cx = x0 + room_w * t
        y = y0 + room_l if side == "north" else y0
        ymin, ymax = (y - depth, y + 0.05) if side == "north" else (y - 0.05, y + depth)
        return ((cx - width / 2, ymin, 0.0), (cx + width / 2, ymax, 2.2))
    cy = y0 + room_l * t
    x = x0 + room_w if side == "east" else x0
    xmin, xmax = (x - depth, x + 0.05) if side == "east" else (x - 0.05, x + depth)
    return ((xmin, cy - width / 2, 0.0), (xmax, cy + width / 2, 2.2))


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _positive_finite(value: Any) -> bool:
    return _finite(value) and float(value) > 0


def _num(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _check(
    check_id: str,
    severity: str,
    status: str,
    message: str,
    **extra: Any,
) -> Check:
    check = {
        "id": check_id,
        "severity": severity,
        "status": status,
        "message": message,
    }
    check.update({key: value for key, value in extra.items() if value is not None})
    return check


if __name__ == "__main__":
    raise SystemExit(main())
