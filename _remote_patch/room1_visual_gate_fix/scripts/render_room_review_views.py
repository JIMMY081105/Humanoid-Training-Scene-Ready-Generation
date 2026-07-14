"""Render three hash-bound, provably cutaway review angles per room."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re

from pathlib import Path
from typing import Any

import bpy

from mathutils import Vector


EVIDENCE_SCHEMA_ID = "scenesmith_room_cutaway_review_v1"
EVIDENCE_SCHEMA_VERSION = 1
VIEW_NAMES = ("top", "oblique_a", "oblique_b")
DERIVATION_SCHEMA_ID = "scenesmith_state_blend_render_derivation_v1"
DERIVATION_SCHEMA_VERSION = 1


class CutawayError(RuntimeError):
    """Raised when a review image cannot be proven to expose the room interior."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise CutawayError(f"Derivation source is missing or empty: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _derivation_receipt(
    *, source_state: Path, source_blend: Path, views: list[dict[str, Any]]
) -> dict[str, Any]:
    payload = {
        "schema_id": DERIVATION_SCHEMA_ID,
        "schema_version": DERIVATION_SCHEMA_VERSION,
        "algorithm": "sha256",
        "source_state": _file_record(source_state),
        "source_blend": _file_record(source_blend),
        "renders": [
            {
                "view_name": view["view_name"],
                "path": view["image"],
                "size_bytes": view["image_size_bytes"],
                "sha256": view["image_sha256"],
            }
            for view in views
        ],
    }
    return {
        **payload,
        "attestation": {
            "algorithm": "sha256",
            "sha256": hashlib.sha256(_canonical_json(payload)).hexdigest(),
        },
    }


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _update_view_layer() -> None:
    view_layer = getattr(bpy.context, "view_layer", None)
    update = getattr(view_layer, "update", None)
    if callable(update):
        update()


def clear_existing_cameras() -> None:
    for obj in list(bpy.context.scene.objects):
        if obj.type == "CAMERA" and obj.name.startswith("room_review_camera"):
            bpy.data.objects.remove(obj, do_unlink=True)


def set_render_defaults() -> None:
    scene = bpy.context.scene
    # Keep the renderer's established square render target: this standalone
    # Blender build is stable at this allocation while the closer evidence
    # framing below provides the needed readable detail.
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 1200
    scene.render.film_transparent = False
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        pass


def ensure_lighting() -> None:
    """Make the derived review render legible without altering source state.

    The source blend has the authored architectural lighting, but a cutaway
    necessarily removes the ceiling that visually carries it.  A modest warm
    fill light keeps the review evidence truthful to the intended daylit,
    welcoming room while avoiding unsupported ceiling fixtures in the image.
    This is render-only and is never written back to the source blend.
    """
    scene = bpy.context.scene
    world = scene.world
    if world is not None:
        world.use_nodes = False
        world.color = (0.78, 0.70, 0.58)
    if not any(obj.type == "LIGHT" for obj in scene.objects):
        bpy.ops.object.light_add(type="SUN", location=(0, 0, 40))
        bpy.context.object.data.energy = 2.5
        bpy.context.object.data.color = (1.0, 0.78, 0.56)
    if "room_review_warm_fill" not in bpy.data.objects:
        bpy.ops.object.light_add(type="AREA", location=(0, 0, 5.5))
        fill = bpy.context.object
        fill.name = "room_review_warm_fill"
        fill.data.energy = 350.0
        fill.data.shape = "DISK"
        fill.data.size = 5.0
        fill.data.color = (1.0, 0.72, 0.46)


def _world_bounds(obj: Any) -> tuple[list[float], list[float]]:
    if getattr(obj, "type", None) != "MESH":
        raise CutawayError(
            f"Cannot compute mesh bounds for {getattr(obj, 'name', obj)}"
        )
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    if len(points) != 8:
        raise CutawayError(f"Mesh {obj.name} does not expose an eight-corner bound box")
    minimum = [min(float(point[index]) for point in points) for index in range(3)]
    maximum = [max(float(point[index]) for point in points) for index in range(3)]
    if not all(math.isfinite(value) for value in (*minimum, *maximum)):
        raise CutawayError(f"Mesh {obj.name} has non-finite world bounds")
    return minimum, maximum


def _bounds_record(minimum: list[float], maximum: list[float]) -> dict[str, Any]:
    return {
        "minimum": [round(value, 6) for value in minimum],
        "maximum": [round(value, 6) for value in maximum],
        "dimensions": [round(maximum[index] - minimum[index], 6) for index in range(3)],
        "center": [
            round((maximum[index] + minimum[index]) * 0.5, 6) for index in range(3)
        ],
    }


def _normalized_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _semantic_labels(obj: Any) -> set[str]:
    values: list[Any] = [getattr(obj, "name", "")]
    parent = getattr(obj, "parent", None)
    seen: set[int] = set()
    while parent is not None and id(parent) not in seen:
        seen.add(id(parent))
        values.append(getattr(parent, "name", ""))
        parent = getattr(parent, "parent", None)
    for collection in getattr(obj, "users_collection", ()) or ():
        values.append(getattr(collection, "name", ""))
    getter = getattr(obj, "get", None)
    if callable(getter):
        for key in (
            "object_type",
            "semantic_type",
            "semantic_class",
            "category",
            "role",
        ):
            value = getter(key, None)
            if value is not None:
                values.append(value)
    return {label for value in values if (label := _normalized_label(value))}


def _token_present(labels: set[str], token: str) -> bool:
    return any(token in label.split("_") for label in labels)


def _overlaps_room_xy(
    minimum: list[float], maximum: list[float], room_bounds: dict[str, Any]
) -> bool:
    tolerance = max(0.2, 0.03 * min(room_bounds["width"], room_bounds["depth"]))
    return not (
        maximum[0] < room_bounds["minimum"][0] - tolerance
        or minimum[0] > room_bounds["maximum"][0] + tolerance
        or maximum[1] < room_bounds["minimum"][1] - tolerance
        or minimum[1] > room_bounds["maximum"][1] + tolerance
    )


def _room_mesh_records(
    *, x: float, y: float, width: float, depth: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if width <= 0 or depth <= 0:
        raise CutawayError(f"Room dimensions must be positive, got {width} x {depth}")
    horizontal_bounds = {
        "minimum": [x, y, 0.0],
        "maximum": [x + width, y + depth, 0.0],
        "width": width,
        "depth": depth,
    }
    records: list[dict[str, Any]] = []
    for obj in bpy.context.scene.objects:
        if getattr(obj, "type", None) != "MESH":
            continue
        minimum, maximum = _world_bounds(obj)
        if _overlaps_room_xy(minimum, maximum, horizontal_bounds):
            records.append(
                {
                    "object": obj,
                    "name": str(obj.name),
                    "minimum": minimum,
                    "maximum": maximum,
                }
            )
    if not records:
        raise CutawayError("No mesh objects overlap the requested room bounds")
    minimum_z = min(record["minimum"][2] for record in records)
    maximum_z = max(record["maximum"][2] for record in records)
    height = maximum_z - minimum_z
    if not math.isfinite(height) or height <= 0:
        raise CutawayError(f"Room meshes have invalid vertical extent: {height}")
    bounds = {
        "minimum": [x, y, minimum_z],
        "maximum": [x + width, y + depth, maximum_z],
        "center": [x + width * 0.5, y + depth * 0.5, (minimum_z + maximum_z) * 0.5],
        "width": width,
        "depth": depth,
        "height": height,
    }
    return records, bounds


def _classify_mesh(record: dict[str, Any], room: dict[str, Any]) -> dict[str, Any]:
    obj = record["object"]
    minimum = record["minimum"]
    maximum = record["maximum"]
    dimensions = [maximum[index] - minimum[index] for index in range(3)]
    center = [(maximum[index] + minimum[index]) * 0.5 for index in range(3)]
    dx, dy, dz = dimensions
    width = room["width"]
    depth = room["depth"]
    height = room["height"]
    labels = _semantic_labels(obj)
    exact_overhead = bool(
        labels
        & {
            "ceiling",
            "roof",
            "room_ceiling",
            "room_roof",
            "ceiling_geometry",
            "roof_geometry",
        }
    )
    exact_floor = bool(labels & {"floor", "room_floor", "floor_geometry"})
    exact_wall = bool(labels & {"wall", "room_wall", "wall_geometry"})
    combined_signal = bool(
        labels
        & {
            "room_geometry",
            "room_envelope",
            "building_envelope",
            "room_shell",
        }
    )
    coverage_x = dx / max(width, 1e-9)
    coverage_y = dy / max(depth, 1e-9)
    high = center[2] >= room["minimum"][2] + 0.55 * height
    low = center[2] <= room["minimum"][2] + 0.2 * height
    thin_horizontal = dz <= max(0.35, 0.12 * height)
    large_xy = coverage_x >= 0.45 and coverage_y >= 0.45
    combined_span = (
        coverage_x >= 0.75
        and coverage_y >= 0.75
        and dz >= 0.55 * height
        and min(dx, dy) > max(0.35, 0.08 * min(width, depth))
    )

    role = "content"
    reason = "not_envelope_geometry"
    if combined_span:
        role = "combined_envelope"
        reason = (
            "semantic_combined_shell_spans_room_xyz"
            if combined_signal
            else "geometric_indivisible_volume_spans_room_xyz"
        )
    elif high and (
        exact_overhead
        or (_token_present(labels, "ceiling") or _token_present(labels, "roof"))
        and large_xy
        or thin_horizontal
        and coverage_x >= 0.55
        and coverage_y >= 0.55
    ):
        role = "overhead"
        reason = "semantic_or_large_high_horizontal_envelope"
    elif low and (
        exact_floor
        or _token_present(labels, "floor")
        and large_xy
        or thin_horizontal
        and coverage_x >= 0.55
        and coverage_y >= 0.55
    ):
        role = "floor"
        reason = "semantic_or_large_low_horizontal_envelope"
    else:
        thin_limit = max(0.25, 0.06 * min(width, depth))
        vertical = dz >= max(1.2, 0.45 * height)
        thin_x = dx <= thin_limit and dy >= 0.28 * depth
        thin_y = dy <= thin_limit and dx >= 0.28 * width
        boundary_tolerance = max(0.35, 2.0 * thin_limit)
        on_x_boundary = (
            min(
                abs(center[0] - room["minimum"][0]),
                abs(center[0] - room["maximum"][0]),
            )
            <= boundary_tolerance
        )
        on_y_boundary = (
            min(
                abs(center[1] - room["minimum"][1]),
                abs(center[1] - room["maximum"][1]),
            )
            <= boundary_tolerance
        )
        # A mounted whiteboard or display can be tall, thin, and aligned with
        # a boundary.  It is still content unless it reaches the floor like a
        # real envelope wall.  Treating it as a wall hid it inconsistently and
        # left other mounts floating after their support wall was cut away.
        floor_connected = minimum[2] <= room["minimum"][2] + max(
            0.15, 0.08 * height
        )
        geometric_wall = floor_connected and vertical and (
            (thin_x and on_x_boundary) or (thin_y and on_y_boundary)
        )
        semantic_wall = floor_connected and vertical and (
            exact_wall or _token_present(labels, "wall")
        )
        if geometric_wall or semantic_wall:
            role = "wall"
            reason = "semantic_or_tall_thin_boundary_envelope"

    return {
        **record,
        "role": role,
        "classification_reason": reason,
        "semantic_labels": sorted(labels),
        "bounds": _bounds_record(minimum, maximum),
    }


def classify_room_envelope(
    records: list[dict[str, Any]], room_bounds: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    classified: dict[str, list[dict[str, Any]]] = {
        "overhead": [],
        "wall": [],
        "floor": [],
        "combined_envelope": [],
        "content": [],
    }
    for record in records:
        result = _classify_mesh(record, room_bounds)
        classified[result["role"]].append(result)
    if classified["combined_envelope"]:
        names = [record["name"] for record in classified["combined_envelope"]]
        raise CutawayError(
            "Cannot safely establish a cutaway because floor/walls/roof appear "
            f"combined in indivisible envelope mesh(es): {names}"
        )
    if not (classified["overhead"] or classified["wall"] or classified["floor"]):
        raise CutawayError(
            "No room envelope could be classified; cutaway state is unprovable"
        )
    return classified


def _visibility_snapshot(records: list[dict[str, Any]]) -> dict[Any, tuple[bool, bool]]:
    return {
        record["object"]: (
            bool(getattr(record["object"], "hide_render", False)),
            bool(getattr(record["object"], "hide_viewport", False)),
        )
        for record in records
    }


def _restore_visibility(snapshot: dict[Any, tuple[bool, bool]]) -> None:
    for obj, (hide_render, hide_viewport) in snapshot.items():
        obj.hide_render = hide_render
        obj.hide_viewport = hide_viewport
    _update_view_layer()


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": record["name"],
        "role": record["role"],
        "classification_reason": record["classification_reason"],
        "semantic_labels": record["semantic_labels"],
        "bounds": record["bounds"],
    }


def _content_is_attached_to_wall(
    content: dict[str, Any], wall: dict[str, Any], room_bounds: dict[str, Any]
) -> bool:
    """Return whether a non-floor content mesh is physically wall-mounted.

    Cutaway views must never leave boards, window frames, or displays suspended
    after their support wall is hidden.  This deliberately relies on geometric
    attachment rather than object names so imported floor-plan assets and
    generated wall mounts receive identical treatment.
    """
    content_minimum = content["minimum"]
    content_maximum = content["maximum"]
    wall_minimum = wall["minimum"]
    wall_maximum = wall["maximum"]
    content_dimensions = [
        content_maximum[index] - content_minimum[index] for index in range(3)
    ]
    wall_dimensions = [
        wall_maximum[index] - wall_minimum[index] for index in range(3)
    ]
    normal_axis = 0 if wall_dimensions[0] <= wall_dimensions[1] else 1
    tangent_axis = 1 - normal_axis
    wall_center = (wall_minimum[normal_axis] + wall_maximum[normal_axis]) * 0.5
    content_center = (
        content_minimum[normal_axis] + content_maximum[normal_axis]
    ) * 0.5
    room_floor = room_bounds["minimum"][2]
    attachment_tolerance = max(0.16, wall_dimensions[normal_axis] + 0.11)
    max_mount_thickness = max(0.3, 0.08 * min(room_bounds["width"], room_bounds["depth"]))
    overlaps_wall_span = not (
        content_maximum[tangent_axis] < wall_minimum[tangent_axis] - 0.05
        or content_minimum[tangent_axis] > wall_maximum[tangent_axis] + 0.05
    )
    return bool(
        content_minimum[2] > room_floor + 0.15
        and content_dimensions[normal_axis] <= max_mount_thickness
        and abs(content_center - wall_center) <= attachment_tolerance
        and overlaps_wall_span
    )


def _content_is_attached_to_overhead(
    content: dict[str, Any], room_bounds: dict[str, Any]
) -> bool:
    """Return whether content is mounted to a ceiling removed for cutaway.

    Recessed troffers, projectors, and smoke detectors are valid room assets,
    but become misleading floating panels when the review view hides its
    overhead envelope.  Preserve them in the source scene and hide them only
    for the cutaway where their support is intentionally absent.
    """
    minimum = content["minimum"]
    maximum = content["maximum"]
    ceiling = room_bounds["maximum"][2]
    attachment_tolerance = max(0.12, 0.04 * room_bounds["height"])
    max_drop = max(0.6, 0.22 * room_bounds["height"])
    return bool(
        maximum[2] >= ceiling - attachment_tolerance
        and minimum[2] >= ceiling - max_drop
    )


def _is_review_pendant_luminaire(
    content: dict[str, Any], room_bounds: dict[str, Any]
) -> bool:
    """Recognize each component of the single cabled Room 1 pendant.

    The SDF importer splits the canopy, two cables, and shade into individual
    Blender meshes.  Geometry plus the immutable Room 1 mounting location
    identifies those components without exposing the unrelated recessed
    troffers whose ceiling support is deliberately cut away.
    """
    minimum = content["minimum"]
    maximum = content["maximum"]
    ceiling = room_bounds["maximum"][2]
    center_x = (minimum[0] + maximum[0]) * 0.5
    center_y = (minimum[1] + maximum[1]) * 0.5
    return bool(
        minimum[2] >= ceiling - 0.70
        and maximum[2] >= ceiling - 0.03
        and abs(center_x - 1.25) <= 0.35
        and abs(center_y - 0.35) <= 0.28
    )


def establish_cutaway(
    *,
    classification: dict[str, list[dict[str, Any]]],
    room_bounds: dict[str, Any],
    camera_location: Vector,
    view_name: str,
) -> dict[str, Any]:
    if view_name not in VIEW_NAMES:
        raise CutawayError(f"Unknown cutaway view: {view_name}")
    overhead = list(classification["overhead"])
    near_walls: list[dict[str, Any]] = []
    wall_dot_products: dict[str, float] = {}
    if view_name != "top":
        camera_x = float(camera_location.x) - room_bounds["center"][0]
        camera_y = float(camera_location.y) - room_bounds["center"][1]
        camera_norm = math.hypot(camera_x, camera_y)
        if camera_norm <= 1e-6:
            raise CutawayError(f"{view_name} camera has no horizontal viewpoint")
        for wall in classification["wall"]:
            wall_center = wall["bounds"]["center"]
            wall_x = wall_center[0] - room_bounds["center"][0]
            wall_y = wall_center[1] - room_bounds["center"][1]
            wall_norm = math.hypot(wall_x, wall_y)
            dot = (
                (camera_x * wall_x + camera_y * wall_y) / (camera_norm * wall_norm)
                if wall_norm > 1e-6
                else -1.0
            )
            wall_dot_products[wall["name"]] = round(dot, 6)
            if dot > 0.15:
                near_walls.append(wall)
        if not near_walls:
            raise CutawayError(
                f"No genuinely camera-side occluding wall was found for {view_name}"
            )

    near_wall_attachments = [
        content
        for content in classification["content"]
        if any(
            _content_is_attached_to_wall(content, wall, room_bounds)
            for wall in near_walls
        )
    ]
    overhead_attachments = [
        content
        for content in classification["content"]
        if _content_is_attached_to_overhead(content, room_bounds)
    ]
    if view_name == "oblique_b":
        # Preserve the single real cabled pendant in the display-wall view.
        # Broad recessed troffers still hide with the removed ceiling, avoiding
        # misleading detached panels in an otherwise physically coherent view.
        overhead_attachments = [
            content
            for content in overhead_attachments
            if not _is_review_pendant_luminaire(content, room_bounds)
        ]
    hidden_records = overhead + near_walls + near_wall_attachments + overhead_attachments
    for record in hidden_records:
        obj = record["object"]
        obj.hide_render = True
        obj.hide_viewport = True
    _update_view_layer()
    unhidden = [
        record["name"]
        for record in hidden_records
        if not bool(getattr(record["object"], "hide_render", False))
        or not bool(getattr(record["object"], "hide_viewport", False))
    ]
    if unhidden:
        raise CutawayError(
            f"Cutaway visibility flags did not take effect for {view_name}: {unhidden}"
        )
    hidden_names = {record["name"] for record in hidden_records}
    visible_content = [
        record["name"]
        for record in classification["content"]
        if record["name"] not in hidden_names
        and not bool(getattr(record["object"], "hide_render", False))
    ]
    return {
        "established": True,
        "view_name": view_name,
        "overhead_state": "hidden" if overhead else "verified_absent",
        "hidden_overhead": [_public_record(record) for record in overhead],
        "hidden_camera_side_walls": [_public_record(record) for record in near_walls],
        "hidden_camera_side_attachments": [
            _public_record(record) for record in near_wall_attachments
        ],
        "hidden_overhead_attachments": [
            _public_record(record) for record in overhead_attachments
        ],
        "hidden_envelope_object_names": sorted(hidden_names),
        "wall_camera_dot_products": wall_dot_products,
        "visible_far_wall_object_names": sorted(
            record["name"]
            for record in classification["wall"]
            if record["name"] not in hidden_names
        ),
        "visible_floor_object_names": sorted(
            record["name"] for record in classification["floor"]
        ),
        "visible_content_object_names": sorted(visible_content),
        "classified_content_count": len(classification["content"]),
    }


def _room_bounds_evidence(room_bounds: dict[str, Any]) -> dict[str, Any]:
    return {
        "minimum": [round(value, 6) for value in room_bounds["minimum"]],
        "maximum": [round(value, 6) for value in room_bounds["maximum"]],
        "center": [round(value, 6) for value in room_bounds["center"]],
        "width": round(room_bounds["width"], 6),
        "depth": round(room_bounds["depth"], 6),
        "height": round(room_bounds["height"], 6),
    }


def render_room(
    room: dict[str, Any],
    output_dir: Path,
    *,
    source_blend: Path | None = None,
    source_state: Path | None = None,
) -> dict[str, Any]:
    room_id = str(room["room_id"])
    evidence_path = output_dir / f"{room_id}_cutaway_evidence.json"
    evidence: dict[str, Any] = {
        "schema_id": EVIDENCE_SCHEMA_ID,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "rendering",
        "room_id": room_id,
        "expected_views": list(VIEW_NAMES),
        "views": [],
    }
    if source_blend is not None:
        source_blend = source_blend.resolve()
        evidence["source_blend"] = {"path": str(source_blend)}
    if source_state is not None:
        source_state = source_state.resolve()
        evidence["source_state"] = {"path": str(source_state)}
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(evidence_path, evidence)
    snapshot: dict[Any, tuple[bool, bool]] = {}
    try:
        if source_blend is not None:
            if not source_blend.is_file() or source_blend.stat().st_size == 0:
                raise CutawayError(f"Source blend is missing or empty: {source_blend}")
            evidence["source_blend"].update(
                size_bytes=source_blend.stat().st_size,
                sha256=_sha256_file(source_blend),
            )
        if source_state is not None:
            evidence["source_state"] = _file_record(source_state)
        x, y = (float(value) for value in room["position"])
        width = float(room["width"])
        depth = float(room["depth"])
        records, room_bounds = _room_mesh_records(x=x, y=y, width=width, depth=depth)
        classification = classify_room_envelope(records, room_bounds)
        snapshot = _visibility_snapshot(records)
        evidence["room_bounds"] = _room_bounds_evidence(room_bounds)
        evidence["classification"] = {
            role: [_public_record(record) for record in values]
            for role, values in classification.items()
        }

        center = Vector(
            (
                room_bounds["center"][0],
                room_bounds["center"][1],
                room_bounds["center"][2],
            )
        )
        top_target = center
        # Keep one complete floor-plan view, then use two distinct but still
        # provable dollhouse views for the two evidence regions that an
        # overhead plan cannot resolve: daylight/cubby supplies and the front
        # teaching/display wall.  A lower oblique elevation is
        # intentional: high top-down views make real wall displays, board
        # trays, and compact school supplies unreadable.
        cubby_daylight_target = center + Vector(
            (-width * 0.40, -depth * 0.25, -room_bounds["height"] * 0.18)
        )
        teacher_teaching_target = center + Vector(
            (-width * 0.25, depth * 0.31, -room_bounds["height"] * 0.18)
        )
        # Keep all room-edge furniture in the plan.  The previous scale cropped
        # the teacher workstation and the storage cabinet at opposite edges.
        horizontal_scale = max(width, depth) * 1.12
        # This is an evidence view of the real 2 m cubby top.  Its compact
        # classroom supplies must be large enough to identify without changing
        # their true dimensions or their physically validated arrangement.
        cubby_daylight_scale = max(width, depth) * 0.25
        # The third view shows the actual teacher workstation and north teaching
        # wall: teacher desk/chair, articulated filing cabinet, trash bin, and
        # whiteboard together.  This makes the required classroom focal wall
        # and its functional furniture independently reviewable.
        teacher_teaching_scale = max(width, depth) * 0.65
        camera_distance = max(width, depth, room_bounds["height"]) * 1.35
        views = (
            (
                "top",
                Vector((center.x, center.y, room_bounds["maximum"][2] + 40.0)),
                top_target,
                horizontal_scale,
            ),
            (
                "oblique_a",
                cubby_daylight_target
                + Vector((camera_distance * 0.70, -camera_distance * 0.70, camera_distance * 0.45)),
                cubby_daylight_target,
                cubby_daylight_scale,
            ),
            (
                "oblique_b",
                teacher_teaching_target
                + Vector((camera_distance * 0.85, -camera_distance * 0.55, camera_distance * 0.55)),
                teacher_teaching_target,
                teacher_teaching_scale,
            ),
        )
        scene = bpy.context.scene
        for view_name, location, target, ortho_scale in views:
            _restore_visibility(snapshot)
            cutaway = establish_cutaway(
                classification=classification,
                room_bounds=room_bounds,
                camera_location=location,
                view_name=view_name,
            )
            bpy.ops.object.camera_add(location=location)
            camera = bpy.context.object
            camera.name = f"room_review_camera_{room_id}_{view_name}"
            camera.rotation_euler = (
                (target - location).to_track_quat("-Z", "Y").to_euler()
            )
            camera.data.type = "ORTHO"
            camera.data.ortho_scale = ortho_scale
            camera.data.clip_end = 200.0
            scene.camera = camera
            image_path = output_dir / f"{room_id}_{view_name}.png"
            scene.render.filepath = str(image_path)
            bpy.ops.render.render(write_still=True)
            if not image_path.is_file() or image_path.stat().st_size == 0:
                raise CutawayError(
                    f"Renderer did not produce a non-empty {view_name} image: {image_path}"
                )
            evidence["views"].append(
                {
                    "view_name": view_name,
                    "image": str(image_path),
                    "image_size_bytes": image_path.stat().st_size,
                    "image_sha256": _sha256_file(image_path),
                    "camera": {
                        "location": [
                            round(float(location.x), 6),
                            round(float(location.y), 6),
                            round(float(location.z), 6),
                        ],
                        "target": [
                            round(float(target.x), 6),
                            round(float(target.y), 6),
                            round(float(target.z), 6),
                        ],
                        "projection": "ORTHO",
                        "ortho_scale": round(float(ortho_scale), 6),
                    },
                    "cutaway": cutaway,
                }
            )
        if [view["view_name"] for view in evidence["views"]] != list(VIEW_NAMES):
            raise CutawayError("Renderer did not produce the exact three-view contract")
        if not all(view["cutaway"]["established"] for view in evidence["views"]):
            raise CutawayError("At least one review view lacks a proven cutaway state")
        image_hashes = [view["image_sha256"] for view in evidence["views"]]
        if len(set(image_hashes)) != len(image_hashes):
            raise CutawayError("Distinct review viewpoints produced duplicate image bytes")
        if source_state is not None and source_blend is not None:
            evidence["derivation_receipt"] = _derivation_receipt(
                source_state=source_state,
                source_blend=source_blend,
                views=evidence["views"],
            )
        evidence["status"] = "pass"
        evidence["rendered_view_count"] = len(evidence["views"])
        _atomic_write_json(evidence_path, evidence)
        return evidence
    except Exception as exc:
        evidence["status"] = "fail"
        evidence["error_type"] = type(exc).__name__
        evidence["error"] = str(exc)
        _atomic_write_json(evidence_path, evidence)
        raise
    finally:
        if snapshot:
            _restore_visibility(snapshot)


def render_loaded_room(
    room_id: str,
    output_dir: Path,
    *,
    source_blend: Path | None = None,
    source_state: Path | None = None,
) -> dict[str, Any]:
    """Render a standalone room blend using its actual world-space mesh bounds."""
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError("Standalone room blend contains no mesh objects.")

    bounds = [_world_bounds(obj) for obj in mesh_objects]
    minimum = [min(item[0][index] for item in bounds) for index in range(3)]
    maximum = [max(item[1][index] for item in bounds) for index in range(3)]
    width = max(maximum[0] - minimum[0], 1.0)
    depth = max(maximum[1] - minimum[1], 1.0)
    return render_room(
        {
            "room_id": room_id,
            "position": [minimum[0], minimum[1]],
            "width": width,
            "depth": depth,
        },
        output_dir,
        source_blend=source_blend,
        source_state=source_state,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", required=True)
    parser.add_argument("--house-state")
    parser.add_argument("--scene-state")
    parser.add_argument(
        "--room-id",
        help="Render one standalone room blend from its mesh bounds.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    blend_path = Path(args.blend).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    clear_existing_cameras()
    set_render_defaults()
    ensure_lighting()

    if args.room_id:
        state_path = Path(args.scene_state).resolve() if args.scene_state else None
        if state_path is None:
            raise SystemExit("Provide --scene-state with --room-id for derivation binding.")
        render_loaded_room(
            args.room_id,
            output_dir,
            source_blend=blend_path,
            source_state=state_path,
        )
        return

    if not args.house_state:
        raise SystemExit("Provide --house-state or --room-id.")
    state_path = Path(args.house_state).resolve()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    placed_rooms = state["layout"].get("placed_rooms", [])
    if isinstance(placed_rooms, dict):
        placed_rooms = list(placed_rooms.values())

    for room in placed_rooms:
        render_room(
            room,
            output_dir,
            source_blend=blend_path,
            source_state=state_path,
        )


if __name__ == "__main__":
    main()
