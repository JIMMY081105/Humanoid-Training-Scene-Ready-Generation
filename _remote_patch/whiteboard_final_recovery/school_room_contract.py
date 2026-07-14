#!/usr/bin/env python3
"""Immutable per-room prompt binding and semantic inventory for the school run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROFILE = "school_reference_20260710"
CONTRACT_MARKER = "[SCENESMITH_SCHOOL_CONTRACT_V1"
CONTEXT_MARKER = "--- FLOOR-PLANNER CONTEXT (SECONDARY) ---"
ROOM_IDS = (
    "classroom_01",
    "classroom_02",
    "classroom_03",
    "classroom_04",
    "classroom_05",
    "classroom_06",
    "library",
    "boys_toilet",
    "girls_toilet",
    "storage_room",
    "main_corridor",
)

STYLE_CONTRACT = (
    "Use warm cream/beige/off-white walls, light wood furniture and flooring, "
    "soft daylight, warm ceiling light, organized school displays, realistic scale, "
    "greenery, and clear humanoid/robot circulation. Nothing may float, visibly "
    "interpenetrate, block a door, or eliminate a usable walking/turning path."
)

CLASSROOM_COMMON = (
    "Create exactly 12 student desks and exactly 12 corresponding student chairs. "
    "Orient them toward a strong front teaching wall/whiteboard with clear aisles. "
    "Include a teacher desk and teacher chair, storage cabinet, bookshelf or cubbies, "
    "bulletin board/educational posters, classroom clock, trash bin, at least one plant, "
    "and visible exterior daylight/windows. Add believable school-use details including "
    "textbooks, notebooks, folders/worksheets, pencil/marker holders, erasers/rulers, "
    "glue/scissors, water bottles, backpacks, storage bins and paper trays without clutter."
)

ROOM_REQUIREMENTS = {
    "classroom_01": CLASSROOM_COMMON
    + " Use a balanced moderate-density row or cluster layout. Include one functional "
    "teacher filing cabinet with operable drawers as an articulated furniture role.",
    "classroom_02": CLASSROOM_COMMON
    + " Use a visibly different grouped or compact-row arrangement from classroom_01, "
    "with accessible walkways and supplies on selected desks.",
    "classroom_03": CLASSROOM_COMMON
    + " Make the room active but organized, with grouped desks, open cubbies, colorful "
    "student work, classroom bins, and a wall notice area.",
    "classroom_04": CLASSROOM_COMMON
    + " Make the room especially tidy and naturally lit, with a seating arrangement and "
    "decoration placement distinct from the other classrooms.",
    "classroom_05": CLASSROOM_COMMON
    + " Give the room its own identity with a small art or reading corner and logical flow "
    "from the door to teacher and student zones.",
    "classroom_06": CLASSROOM_COMMON
    + " Make it slightly denser but practical, retaining clear paths between all desks.",
    "library": (
        "Create the warm academic heart near the entrance. Include perimeter shelving plus "
        "at least four bookcases/shelf units; one must be a functional articulated bookcase "
        "cabinet with hinged glass doors. Include 2-4 round reading tables, at least six "
        "reading chairs, at least two lounge chairs, a low lounge table, librarian/circulation "
        "counter, book displays, plants, reading posters/notices, storage cubbies, and visible "
        "books/magazines/binders/stationery. Preserve browsing and robot paths and exterior light."
    ),
    "boys_toilet": (
        "Create a clean compact boys school toilet with at least two toilet fixtures/stalls, "
        "at least two urinals, at least two sinks and mirrors, stall partitions, soap dispenser, "
        "paper-towel dispenser or hand dryer, trash bin, tiled floor/walls, and unblocked access."
    ),
    "girls_toilet": (
        "Create a clean compact girls school toilet beside the boys toilet with at least two "
        "toilet fixtures/stalls, at least two sinks and mirrors, stall partitions, soap dispenser, "
        "paper-towel dispenser or hand dryer, trash bin, sanitary disposal bin, tiled floor/walls, "
        "and unblocked access."
    ),
    "storage_room": (
        "Create navigable but dense school storage with at least two sturdy shelf units, at least "
        "four labeled bins, boxes, paper/textbook/art supplies, cleaning buckets/mops, maintenance "
        "tools, and spare classroom materials. Include a functional articulated school-supply "
        "utility cabinet with two hinged doors."
    ),
    "main_corridor": (
        "Create the broad open central circulation/lobby spine with clear sightlines and door "
        "clearance. Include at least two benches/seating elements, at least four plants/planters, "
        "at least two bulletin/display areas, welcoming school signage, and a clear path to the "
        "bottom-center double entrance and library. Preserve wheelchair and humanoid turning space."
    ),
}


@dataclass(frozen=True)
class InventoryRequirement:
    key: str
    label: str
    patterns: tuple[str, ...]
    minimum: int
    maximum: int | None = None
    exclude_patterns: tuple[str, ...] = ()


def _req(
    key: str,
    label: str,
    patterns: Iterable[str],
    minimum: int,
    maximum: int | None = None,
    exclude: Iterable[str] = (),
) -> InventoryRequirement:
    return InventoryRequirement(
        key, label, tuple(patterns), minimum, maximum, tuple(exclude)
    )


CLASSROOM_INVENTORY = (
    _req("student_desks", "student desks", (r"\bstudent\s+(?:school\s+)?desk\b", r"\bpupil\s+desk\b", r"\blearner\s+desk\b"), 12, 12),
    _req("student_chairs", "student chairs", (r"\bstudent\s+chair\b", r"\bpupil\s+chair\b", r"\blearner\s+chair\b"), 12, 12),
    _req("teacher_desk", "teacher desk/station", (r"\bteacher(?:'s)?\s+(?:desk|station)\b", r"\binstructor\s+desk\b"), 1),
    _req(
        "teacher_chair",
        "teacher chair",
        (
            r"\bteacher(?:'s)?\s+(?:office\s+)?chair\b",
            r"\binstructor\s+chair\b",
        ),
        1,
    ),
    _req("whiteboard", "whiteboard/teaching board", (r"\bwhite\s*board\b", r"\bteaching\s+board\b", r"\bchalk\s*board\b"), 1),
    _req("storage", "storage cabinet/shelf", (r"\bstorage\s+(?:cabinet|shelf|unit)\b", r"\bsupply\s+cabinet\b"), 1),
    _req("bookshelf", "bookshelf/cubby", (r"\bbook\s*(?:shelf|case)\b", r"\bcubb(?:y|ies)\b"), 1),
    _req("display", "educational display/poster", (r"\bbulletin\s+board\b", r"\bnotice\s+board\b", r"\bposter\b", r"\bstudent\s+(?:work|art)\b"), 1),
    _req("clock", "classroom clock", (r"\bclock\b",), 1),
    _req("trash", "trash bin", (r"\btrash\s+(?:bin|can)\b", r"\bwaste\s+(?:bin|basket)\b"), 1),
    _req("plant", "plant/greenery", (r"\bplant\b", r"\bgreenery\b"), 1),
    _req("textbooks", "textbooks", (r"\btextbooks?\b",), 1),
    _req("notebooks", "notebooks", (r"\bnotebooks?\b",), 1),
    _req("folders", "folders", (r"\bfolders?\b",), 1),
    _req("worksheets", "worksheets", (r"\bworksheets?\b",), 1),
    _req("pencil_holders", "pencil holders", (r"\bpencil\s+(?:holder|cup|pot)s?\b",), 1),
    _req(
        "markers",
        "general classroom markers",
        (r"\bmarkers?\b",),
        1,
        exclude=(r"\b(?:white\s*board|board)\s+markers?\b",),
    ),
    _req(
        "erasers",
        "general erasers",
        (r"\berasers?\b",),
        1,
        exclude=(r"\b(?:white\s*board|board)\s+erasers?\b",),
    ),
    _req("rulers", "rulers", (r"\brulers?\b",), 1),
    _req("glue_sticks", "glue sticks", (r"\bglue\s+sticks?\b",), 1),
    _req("scissors", "scissors", (r"\bscissors?\b",), 1),
    _req("backpack", "backpack", (r"\bback\s*pack\b", r"\bschool\s+bag\b"), 1),
    _req("water_bottle", "water bottle", (r"\bwater\s+bottle\b",), 1),
    _req("storage_bins", "classroom storage bins", (r"\b(?:classroom\s+)?storage\s+bins?\b",), 1),
    _req("paper_trays", "paper trays", (r"\b(?:paper\s+tray|tray\s+for\s+papers?)s?\b",), 1),
    _req("board_markers", "board markers", (r"\b(?:white\s*board|board)\s+markers?\b",), 1),
    _req("whiteboard_eraser", "whiteboard eraser", (r"\bwhite\s*board\s+erasers?\b",), 1),
)

ROOM_INVENTORIES = {
    "library": (
        _req("bookcases", "bookcases/shelf units", (r"\bbook\s*(?:case|shelf)\b", r"\blibrary\s+shel"), 4),
        _req("reading_tables", "round reading tables", (r"\breading\s+table\b", r"\bround\s+table\b"), 2, 4),
        _req("reading_chairs", "reading chairs", (r"\breading\s+chair\b", r"\blibrary\s+chair\b", r"\bstudent\s+chair\b"), 6),
        _req("lounge_chairs", "lounge/casual chairs", (r"\blounge\s+chair\b", r"\bcasual\s+reading\s+chair\b", r"\barm\s*chair\b"), 2),
        _req("low_table", "low lounge table", (r"\blow\s+table\b", r"\bcoffee\s+table\b"), 1),
        _req("counter", "librarian/circulation desk", (r"\blibrarian\s+(?:counter|desk)\b", r"\bcirculation\s+(?:counter|desk)\b"), 1),
        _req("plants", "library plants", (r"\bplant\b", r"\bgreenery\b"), 2),
        _req("displays", "reading poster/notice", (r"\bposter\b", r"\bnotice\s+board\b", r"\bpin\s*board\b"), 1),
        _req("books", "visible books/materials", (r"\bbook\b", r"\bmagazine\b", r"\bbinder\b", r"\bbook\s+bin\b"), 4),
        _req("cubbies", "storage cubbies", (r"\bcubb(?:y|ies)\b",), 1),
    ),
    "boys_toilet": (
        _req("toilets", "toilet fixtures/stalls", (r"\btoilet\b", r"\bwater\s+closet\b"), 2, exclude=(r"\bboys?\s+toilet\b",)),
        _req("urinals", "urinals", (r"\burinal\b",), 2),
        _req("sinks", "sinks/basins", (r"\bsink\b", r"\bwash\s*basin\b"), 2),
        _req("mirrors", "mirrors", (r"\bmirror\b",), 2),
        _req("partitions", "stall partitions", (r"\bstall\s+partition\b", r"\brestroom\s+partition\b"), 2),
        _req("soap", "soap dispenser", (r"\bsoap\s+dispenser\b",), 1),
        _req("drying", "paper towel/hand dryer", (r"\bpaper\s+towel\b", r"\bhand\s+dryer\b"), 1),
        _req("trash", "trash bin", (r"\btrash\s+(?:bin|can)\b", r"\bwaste\s+bin\b"), 1),
    ),
    "girls_toilet": (
        _req("toilets", "toilet fixtures/stalls", (r"\btoilet\b", r"\bwater\s+closet\b"), 2, exclude=(r"\bgirls?\s+toilet\b",)),
        _req("sinks", "sinks/basins", (r"\bsink\b", r"\bwash\s*basin\b"), 2),
        _req("mirrors", "mirrors", (r"\bmirror\b",), 2),
        _req("partitions", "stall partitions", (r"\bstall\s+partition\b", r"\brestroom\s+partition\b"), 2),
        _req("soap", "soap dispenser", (r"\bsoap\s+dispenser\b",), 1),
        _req("drying", "paper towel/hand dryer", (r"\bpaper\s+towel\b", r"\bhand\s+dryer\b"), 1),
        _req("trash", "trash bin", (r"\btrash\s+(?:bin|can)\b", r"\bwaste\s+bin\b"), 1),
        _req("sanitary", "sanitary disposal bin", (r"\bsanitary\s+(?:disposal\s+)?bin\b",), 1),
    ),
    "storage_room": (
        _req("shelves", "storage shelf units", (r"\b(?:metal|storage|sturdy)\s+shel", r"\bshelving\s+unit\b"), 2),
        _req("bins", "labeled/storage bins", (r"\blabeled\s+bin\b", r"\bstorage\s+bin\b", r"\bsupply\s+bin\b"), 4),
        _req("boxes", "storage boxes", (r"\bbox\b", r"\bcarton\b"), 2),
        _req("school_stock", "paper/books/art supplies", (r"\bprinter\s+paper\b", r"\bpaper\s+(?:stack|box)\b", r"\btextbook\b", r"\bart\s+suppl", r"\bstationery\b"), 2),
        _req("cleaning", "cleaning supplies", (r"\bmop\b", r"\bbucket\b", r"\bcleaning\s+suppl", r"\bbroom\b"), 2),
        _req("maintenance", "maintenance tools", (r"\btool\s*(?:kit|box)\b", r"\bmaintenance\s+(?:tool|suppl)"), 1),
    ),
    "main_corridor": (
        _req("benches", "benches/seating", (r"\bbench\b", r"\bbuilt[- ]in\s+seat", r"\bcorridor\s+seat"), 2),
        _req("plants", "plants/planters", (r"\bplant\b", r"\bplanter\b", r"\bgreenery\b"), 4),
        _req("displays", "bulletin/display areas", (r"\bbulletin\s+board\b", r"\bdisplay\s+(?:area|board)\b", r"\bnotice\s+board\b"), 2),
        _req("signage", "welcome/school signage", (r"\bwelcome\s+(?:sign|signage|mat)\b", r"\bschool\s+sign", r"\bwayfinding\b"), 1),
    ),
}

ARTICULATED_ROLE_RULES = {
    "library_glass_door_bookcase": {
        "rooms": {"library"},
        "patterns": (r"\bbook\s*(?:case|shelf)\b", r"\bglass\b", r"\bdoors?\b"),
    },
    "school_supply_two_door_utility_cabinet": {
        "rooms": {"storage_room"},
        "patterns": (r"\b(?:utility|supply|storage)\b", r"\bcabinet\b", r"\b(?:two|2)\b", r"\bdoors?\b"),
    },
    "teacher_filing_drawer_cabinet": {
        "rooms": {"classroom_01"},
        "patterns": (r"\b(?:teacher|filing|file)\b", r"\bcabinet\b", r"\bdrawers?\b"),
    },
}
ARTICULATED_ROLE_SOURCES = frozenset({"artiverse", "artvip"})
MESH_SUFFIXES = frozenset(
    {".dae", ".fbx", ".glb", ".gltf", ".obj", ".ply", ".stl", ".usd", ".usda", ".usdc"}
)
PLACEHOLDER_OBJECT_TYPES = frozenset({"primitive", "placeholder", "preview", "proxy"})
PLACEHOLDER_ASSET_SOURCES = frozenset({"primitive", "placeholder", "preview", "proxy", "fake"})
PLACEHOLDER_MESH_TOKEN_RE = re.compile(
    r"(?:^|[_.\-])(?:cube|placeholder|dummy|fake|preview|proxy)(?:[_.\-]|$)",
    flags=re.IGNORECASE,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def canonical_room_prompt(room_id: str, effective_prompt_sha256: str, existing: str = "") -> str:
    if room_id not in ROOM_REQUIREMENTS:
        raise ValueError(f"Unsupported school room ID: {room_id}")
    context = existing
    if CONTEXT_MARKER in context:
        context = context.split(CONTEXT_MARKER, 1)[1].strip()
    marker = f"{CONTRACT_MARKER} room_id={room_id} profile={PROFILE}]"
    return (
        f"{marker}\nImmutable effective prompt SHA-256: {effective_prompt_sha256}\n\n"
        f"{STYLE_CONTRACT}\n\nROOM-SPECIFIC REQUIRED CONTENTS:\n"
        f"{ROOM_REQUIREMENTS[room_id]}\n\n{CONTEXT_MARKER}\n{context.strip()}"
    ).strip()


def _layout_collection(
    layout: dict[str, Any], key: str
) -> dict[str, dict[str, Any]]:
    value = layout.get(key)
    if isinstance(value, dict):
        entries = [item for item in value.values() if isinstance(item, dict)]
    elif isinstance(value, list):
        entries = [item for item in value if isinstance(item, dict)]
    else:
        raise RuntimeError(f"house_layout.json {key} must be a room collection")
    ids = [str(entry.get("room_id") or entry.get("id") or "") for entry in entries]
    missing_ids = len(entries) - sum(bool(room_id) for room_id in ids)
    duplicates = sorted(
        room_id for room_id in set(ids) if room_id and ids.count(room_id) > 1
    )
    present = {room_id for room_id in ids if room_id}
    missing = sorted(set(ROOM_IDS) - present)
    unexpected = sorted(present - set(ROOM_IDS))
    if missing_ids or duplicates or missing or unexpected or len(entries) != len(ROOM_IDS):
        raise RuntimeError(
            f"Cannot bind {key}; entries={len(entries)}, missing_ids={missing_ids}, "
            f"duplicates={duplicates}, missing={missing}, unexpected={unexpected}"
        )
    return {
        str(entry.get("room_id") or entry.get("id")): entry for entry in entries
    }


def bind_layout_prompts(
    layout_path: Path, input_manifest_path: Path, evidence_path: Path
) -> dict[str, Any]:
    layout_path = layout_path.resolve()
    manifest_path = input_manifest_path.resolve()
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    effective_hash = str(manifest.get("effective_prompt_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", effective_hash):
        raise RuntimeError("Input manifest has no valid effective prompt hash")
    contract = manifest.get("pipeline_contract")
    if not isinstance(contract, dict) or contract.get("id") != "scenesmith_full_quality_v1":
        raise RuntimeError("Input manifest is not the full-quality school contract")

    room_specs = _layout_collection(layout, "rooms")
    placed_rooms = _layout_collection(layout, "placed_rooms")

    before_hash = sha256_file(layout_path)
    prompt_hashes: dict[str, str] = {}
    occurrence_counts = {room_id: 2 for room_id in ROOM_IDS}
    for room_id in ROOM_IDS:
        canonical_prompt = canonical_room_prompt(
            room_id,
            effective_hash,
            str(room_specs[room_id].get("prompt") or ""),
        )
        # SceneSmith workers consume RoomSpec while later geometry/evidence may
        # consume PlacedRoom. Both serializations must carry identical bytes.
        room_specs[room_id]["prompt"] = canonical_prompt
        placed_rooms[room_id]["prompt"] = canonical_prompt
        prompt_hashes[room_id] = _sha256_bytes(canonical_prompt.encode("utf-8"))

    temporary = layout_path.with_name(f".{layout_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(layout, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, layout_path)
    result = {
        "schema_version": 1,
        "status": "pass",
        "profile": PROFILE,
        "layout": {"path": str(layout_path), "sha256_before": before_hash, "sha256": sha256_file(layout_path)},
        "input_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "effective_prompt_sha256": effective_hash,
        "room_prompt_sha256": dict(sorted(prompt_hashes.items())),
        "occurrence_counts": occurrence_counts,
    }
    evidence_path = evidence_path.resolve()
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_tmp = evidence_path.with_name(f".{evidence_path.name}.{os.getpid()}.tmp")
    evidence_tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(evidence_tmp, evidence_path)
    return result


def _iter_state_objects(state: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    objects = state.get("objects", {})
    if isinstance(objects, dict):
        return [(str(key), value) for key, value in objects.items() if isinstance(value, dict)]
    if isinstance(objects, list):
        return [
            (str(value.get("object_id", index)), value)
            for index, value in enumerate(objects)
            if isinstance(value, dict)
        ]
    return []


def object_semantic_text(object_id: str, obj: dict[str, Any]) -> str:
    values: list[str] = [object_id]
    for key in ("object_id", "name", "description", "object_type", "category", "short_name"):
        value = obj.get(key)
        if isinstance(value, (str, int, float)):
            values.append(str(value))
    metadata = obj.get("metadata")
    if isinstance(metadata, dict):
        for key in (
            "description",
            "category",
            "short_name",
            "asset_name",
            "articulated_id",
            "articulated_source",
        ):
            value = metadata.get(key)
            if isinstance(value, (str, int, float)):
                values.append(str(value))
    return re.sub(r"[_\-/]+", " ", " ".join(values)).lower()


def _matches(text: str, requirement: InventoryRequirement) -> bool:
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in requirement.exclude_patterns):
        return False
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in requirement.patterns)


def _minimum_unique_assignment(
    requirements: tuple[InventoryRequirement, ...],
    semantic_objects: list[tuple[str, str, dict[str, Any]]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Assign one physical object to at most one required inventory slot."""

    candidates = {
        requirement.key: [
            object_id
            for object_id, text, _obj in semantic_objects
            if _matches(text, requirement)
        ]
        for requirement in requirements
    }
    requirements_by_key = {requirement.key: requirement for requirement in requirements}
    slots = [
        (requirement.key, slot_index)
        for requirement in requirements
        for slot_index in range(requirement.minimum)
    ]
    slots.sort(key=lambda slot: (len(candidates[slot[0]]), slot[0], slot[1]))
    object_to_slot: dict[str, tuple[str, int]] = {}
    slot_to_object: dict[tuple[str, int], str] = {}

    def assign(slot: tuple[str, int], visited: set[str]) -> bool:
        for object_id in candidates[slot[0]]:
            if object_id in visited:
                continue
            visited.add(object_id)
            occupied = object_to_slot.get(object_id)
            if occupied is None or assign(occupied, visited):
                object_to_slot[object_id] = slot
                slot_to_object[slot] = object_id
                return True
        return False

    for slot in slots:
        assign(slot, set())

    assigned = {requirement.key: [] for requirement in requirements}
    for (requirement_key, slot_index), object_id in sorted(slot_to_object.items()):
        if slot_index < requirements_by_key[requirement_key].minimum:
            assigned[requirement_key].append(object_id)
    return assigned, candidates


def _object_translation(obj: dict[str, Any]) -> tuple[float, float, float] | None:
    transform = obj.get("transform")
    if not isinstance(transform, dict):
        return None
    translation = transform.get("translation")
    if isinstance(translation, dict):
        values = [translation.get(axis) for axis in ("x", "y", "z")]
    elif isinstance(translation, (list, tuple)) and len(translation) >= 3:
        values = list(translation[:3])
    else:
        return None
    try:
        point = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    return point if all(math.isfinite(value) for value in point) else None


def _finite_vector(value: Any, length: int) -> tuple[float, ...] | None:
    if isinstance(value, dict):
        keys = ("x", "y", "z", "w")[:length]
        raw = [value.get(key) for key in keys]
    elif isinstance(value, (list, tuple)) and len(value) == length:
        raw = list(value)
    else:
        return None
    try:
        vector = tuple(float(item) for item in raw)
    except (TypeError, ValueError):
        return None
    return vector if all(math.isfinite(item) for item in vector) else None


def _object_rotation_wxyz(obj: dict[str, Any]) -> tuple[float, float, float, float] | None:
    transform = obj.get("transform")
    if not isinstance(transform, dict):
        return None
    rotation = _finite_vector(transform.get("rotation_wxyz"), 4)
    if rotation is None:
        return None
    norm = math.sqrt(sum(value * value for value in rotation))
    if norm <= 1.0e-9 or abs(norm - 1.0) > 1.0e-3:
        return None
    return tuple(value / norm for value in rotation)  # type: ignore[return-value]


def _rotation_matrix_wxyz(
    quaternion: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    w, x, y, z = quaternion
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _object_local_bounds(
    obj: dict[str, Any],
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    minimum = _finite_vector(obj.get("bbox_min"), 3)
    maximum = _finite_vector(obj.get("bbox_max"), 3)
    if minimum is None or maximum is None:
        return None
    if any(upper - lower <= 1.0e-6 for lower, upper in zip(minimum, maximum)):
        return None
    return minimum, maximum  # type: ignore[return-value]


def object_world_bounds(
    obj: dict[str, Any],
) -> tuple[float, float, float, float, float, float] | None:
    """Return a rotation-aware world AABB for a serialized SceneObject."""

    translation = _object_translation(obj)
    quaternion = _object_rotation_wxyz(obj)
    bounds = _object_local_bounds(obj)
    if translation is None or quaternion is None or bounds is None:
        return None
    minimum, maximum = bounds
    rotation = _rotation_matrix_wxyz(quaternion)
    world: list[tuple[float, float, float]] = []
    for x in (minimum[0], maximum[0]):
        for y in (minimum[1], maximum[1]):
            for z in (minimum[2], maximum[2]):
                local = (x, y, z)
                world.append(
                    tuple(
                        translation[row]
                        + sum(rotation[row][column] * local[column] for column in range(3))
                        for row in range(3)
                    )
                )
    return (
        min(point[0] for point in world),
        max(point[0] for point in world),
        min(point[1] for point in world),
        max(point[1] for point in world),
        min(point[2] for point in world),
        max(point[2] for point in world),
    )


def _object_mesh_paths(obj: dict[str, Any]) -> list[str]:
    paths: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key == "geometry_path" and isinstance(nested, str) and nested.strip():
                    paths.append(nested.strip())
                elif isinstance(nested, (dict, list, tuple)):
                    visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)

    visit({"geometry_path": obj.get("geometry_path"), "metadata": obj.get("metadata")})
    return sorted(set(paths))


def physical_object_evidence(
    obj: dict[str, Any], *, asset_root: Path | None = None
) -> dict[str, Any]:
    """Validate native, non-textual evidence for one serialized scene object."""

    issues: list[str] = []
    if obj.get("immutable") is not False:
        issues.append("immutable must be explicitly false")
    object_type = str(obj.get("object_type", "")).strip().lower()
    if object_type in PLACEHOLDER_OBJECT_TYPES:
        issues.append(f"placeholder object_type is forbidden: {object_type}")
    metadata = obj.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    asset_source = str(metadata.get("asset_source", "")).strip().lower()
    if asset_source in PLACEHOLDER_ASSET_SOURCES:
        issues.append(f"placeholder asset_source is forbidden: {asset_source}")
    for key in ("is_placeholder", "placeholder", "preview_only", "is_primitive_placeholder"):
        if metadata.get(key) is True:
            issues.append(f"placeholder metadata flag is true: {key}")
    translation = _object_translation(obj)
    if translation is None:
        issues.append("finite 3D translation is missing")
    quaternion = _object_rotation_wxyz(obj)
    if quaternion is None:
        issues.append("finite unit rotation_wxyz is missing")
    local_bounds = _object_local_bounds(obj)
    if local_bounds is None:
        issues.append("finite positive-volume 3D bounds are missing")
    world_bounds = object_world_bounds(obj)
    if world_bounds is None:
        issues.append("rotation-aware world bounds cannot be computed")
    scale = obj.get("scale_factor")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        issues.append("positive finite scale_factor is missing")
    elif not math.isfinite(float(scale)) or float(scale) <= 0:
        issues.append("scale_factor is not positive and finite")

    mesh_paths = _object_mesh_paths(obj)
    if not mesh_paths:
        issues.append("no geometry_path or composite-member mesh is recorded")
    resolved_meshes: list[str] = []
    for raw_path in mesh_paths:
        candidate = Path(raw_path)
        if PLACEHOLDER_MESH_TOKEN_RE.search(candidate.stem):
            issues.append(f"generic placeholder mesh is forbidden: {raw_path}")
        if candidate.suffix.lower() not in MESH_SUFFIXES:
            issues.append(f"mesh path has unsupported suffix: {raw_path}")
            continue
        if not candidate.is_absolute() and asset_root is not None:
            candidate = asset_root / candidate
        candidate = candidate.resolve()
        resolved_meshes.append(str(candidate))
        if asset_root is not None and (
            not candidate.is_file() or candidate.stat().st_size < 1
        ):
            issues.append(f"mesh path is missing or empty: {candidate}")

    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "translation": list(translation) if translation is not None else None,
        "rotation_wxyz": list(quaternion) if quaternion is not None else None,
        "local_bounds": (
            {"minimum": list(local_bounds[0]), "maximum": list(local_bounds[1])}
            if local_bounds is not None
            else None
        ),
        "world_bounds": list(world_bounds) if world_bounds is not None else None,
        "mesh_paths": mesh_paths,
        "resolved_mesh_paths": resolved_meshes,
    }


def _prompt_contract_evidence(
    room_id: str,
    prompt: str,
    expected_prompt_sha256: str | None,
    *,
    require_expected_prompt_sha256: bool = False,
) -> dict[str, Any]:
    marker = f"{CONTRACT_MARKER} room_id={room_id} profile={PROFILE}]"
    issues: list[str] = []
    match = re.match(
        rf"^{re.escape(marker)}\nImmutable effective prompt SHA-256: ([0-9a-f]{{64}})\n",
        prompt,
    )
    effective_hash = match.group(1) if match else ""
    if match is None:
        issues.append("prompt header/hash is not the canonical immutable contract")
    if prompt.count(CONTRACT_MARKER) != 1 or prompt.count(CONTEXT_MARKER) != 1:
        issues.append("prompt contract/context markers are duplicated or missing")
    if effective_hash and CONTEXT_MARKER in prompt:
        context = prompt.split(CONTEXT_MARKER, 1)[1].strip()
        if prompt != canonical_room_prompt(room_id, effective_hash, context):
            issues.append("prompt bytes differ from the canonical room contract")
    actual_hash = _sha256_bytes(prompt.encode("utf-8"))
    if expected_prompt_sha256 is None and require_expected_prompt_sha256:
        issues.append("bound room prompt SHA-256 evidence is required")
    elif expected_prompt_sha256 is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_prompt_sha256):
            issues.append("bound room prompt SHA-256 is malformed")
        elif actual_hash != expected_prompt_sha256:
            issues.append("room prompt SHA-256 differs from binding evidence")
    return {
        "status": "pass" if not issues else "fail",
        "effective_prompt_sha256": effective_hash or None,
        "room_prompt_sha256": actual_hash,
        "expected_room_prompt_sha256": expected_prompt_sha256,
        "issues": issues,
    }


def _desk_chair_pairing(
    objects_by_id: dict[str, dict[str, Any]],
    desk_ids: list[str],
    chair_ids: list[str],
    *,
    minimum_distance: float = 0.20,
    maximum_distance: float = 1.60,
    maximum_axis_delta_degrees: float = 45.0,
    maximum_pair_overlap_ratio: float = 0.65,
    maximum_lateral_offset_m: float = 0.75,
    minimum_axial_offset_m: float = 0.10,
    maximum_same_type_overlap_ratio: float = 0.10,
) -> dict[str, Any]:
    """Prove 12 one-to-one, supported, oriented desk/chair pairs."""

    invalid_physical_evidence = sorted(
        object_id
        for object_id in [*desk_ids, *chair_ids]
        if object_id not in objects_by_id
        or physical_object_evidence(objects_by_id[object_id])["status"] != "pass"
    )

    def yaw(obj: dict[str, Any]) -> float | None:
        quaternion = _object_rotation_wxyz(obj)
        if quaternion is None:
            return None
        w, x, y, z = quaternion
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def axis_delta(left: float, right: float) -> float:
        signed = abs((left - right + math.pi) % (2 * math.pi) - math.pi)
        return min(signed, abs(math.pi - signed))

    def horizontal_overlap_ratio(
        left: tuple[float, float, float, float, float, float],
        right: tuple[float, float, float, float, float, float],
    ) -> float:
        overlap_x = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
        overlap_y = max(0.0, min(left[3], right[3]) - max(left[2], right[2]))
        left_area = max(0.0, left[1] - left[0]) * max(0.0, left[3] - left[2])
        right_area = max(0.0, right[1] - right[0]) * max(0.0, right[3] - right[2])
        denominator = min(left_area, right_area)
        return overlap_x * overlap_y / denominator if denominator > 1.0e-9 else 1.0

    maximum_axis_delta = math.radians(maximum_axis_delta_degrees)
    candidates: dict[str, list[tuple[str, float, dict[str, float]]]] = {}
    for desk_id in desk_ids:
        desk_point = _object_translation(objects_by_id.get(desk_id, {}))
        candidates[desk_id] = []
        if desk_point is None:
            continue
        desk_bounds = object_world_bounds(objects_by_id[desk_id])
        desk_yaw = yaw(objects_by_id[desk_id])
        if desk_bounds is None or desk_yaw is None:
            continue
        for chair_id in chair_ids:
            chair_point = _object_translation(objects_by_id.get(chair_id, {}))
            if chair_point is None:
                continue
            chair_bounds = object_world_bounds(objects_by_id[chair_id])
            chair_yaw = yaw(objects_by_id[chair_id])
            if chair_bounds is None or chair_yaw is None:
                continue
            distance = math.dist(desk_point[:2], chair_point[:2])
            orientation_delta = axis_delta(desk_yaw, chair_yaw)
            bottom_delta = abs(desk_bounds[4] - chair_bounds[4])
            overlap_ratio = horizontal_overlap_ratio(desk_bounds, chair_bounds)
            delta_x = chair_point[0] - desk_point[0]
            delta_y = chair_point[1] - desk_point[1]
            lateral_offset = abs(delta_x * math.cos(desk_yaw) + delta_y * math.sin(desk_yaw))
            axial_offset = abs(-delta_x * math.sin(desk_yaw) + delta_y * math.cos(desk_yaw))
            directly_floor_placed = (
                objects_by_id[desk_id].get("placement_info") is None
                and objects_by_id[chair_id].get("placement_info") is None
            )
            floor_supported = (
                -0.15 <= desk_bounds[4] <= 0.25
                and -0.15 <= chair_bounds[4] <= 0.25
            )
            if (
                minimum_distance <= distance <= maximum_distance
                and orientation_delta <= maximum_axis_delta
                and bottom_delta <= 0.15
                and floor_supported
                and directly_floor_placed
                and overlap_ratio <= maximum_pair_overlap_ratio
                and lateral_offset <= maximum_lateral_offset_m
                and axial_offset >= minimum_axial_offset_m
            ):
                candidates[desk_id].append(
                    (
                        chair_id,
                        distance,
                        {
                            "axis_delta_degrees": math.degrees(orientation_delta),
                            "bottom_delta_m": bottom_delta,
                            "horizontal_overlap_ratio": overlap_ratio,
                            "lateral_offset_m": lateral_offset,
                            "axial_offset_m": axial_offset,
                        },
                    )
                )
        candidates[desk_id].sort(key=lambda item: (item[1], item[0]))

    chair_to_desk: dict[str, str] = {}
    desk_to_chair: dict[str, tuple[str, float, dict[str, float]]] = {}

    def assign(desk_id: str, visited: set[str]) -> bool:
        for chair_id, distance, evidence in candidates.get(desk_id, []):
            if chair_id in visited:
                continue
            visited.add(chair_id)
            occupied_desk = chair_to_desk.get(chair_id)
            if occupied_desk is None or assign(occupied_desk, visited):
                chair_to_desk[chair_id] = desk_id
                desk_to_chair[desk_id] = (chair_id, distance, evidence)
                return True
        return False

    for desk_id in sorted(desk_ids, key=lambda key: (len(candidates[key]), key)):
        assign(desk_id, set())

    pairs = [
        {
            "desk_object_id": desk_id,
            "chair_object_id": chair_id,
            "center_distance_m": round(distance, 4),
            **{key: round(value, 4) for key, value in evidence.items()},
        }
        for desk_id, (chair_id, distance, evidence) in sorted(desk_to_chair.items())
    ]
    same_type_clearance_issues: list[dict[str, Any]] = []
    for object_type, object_ids, minimum_separation in (
        ("desk", desk_ids, 0.35),
        ("chair", chair_ids, 0.30),
    ):
        for index, first_id in enumerate(sorted(object_ids)):
            first = _object_translation(objects_by_id.get(first_id, {}))
            if first is None:
                continue
            for second_id in sorted(object_ids)[index + 1 :]:
                second = _object_translation(objects_by_id.get(second_id, {}))
                if second is None:
                    continue
                separation = math.dist(first[:2], second[:2])
                first_bounds = object_world_bounds(objects_by_id.get(first_id, {}))
                second_bounds = object_world_bounds(objects_by_id.get(second_id, {}))
                overlap_ratio = (
                    horizontal_overlap_ratio(first_bounds, second_bounds)
                    if first_bounds is not None and second_bounds is not None
                    else 1.0
                )
                if (
                    separation < minimum_separation
                    or overlap_ratio > maximum_same_type_overlap_ratio
                ):
                    same_type_clearance_issues.append(
                        {
                            "object_type": object_type,
                            "object_ids": [first_id, second_id],
                            "center_distance_m": round(separation, 4),
                            "minimum_center_distance_m": minimum_separation,
                            "horizontal_overlap_ratio": round(overlap_ratio, 4),
                            "maximum_horizontal_overlap_ratio": maximum_same_type_overlap_ratio,
                        }
                    )
    expected_pairs = min(len(desk_ids), len(chair_ids))
    return {
        "status": (
            "pass"
            if not invalid_physical_evidence
            and not same_type_clearance_issues
            and expected_pairs == 12
            and len(pairs) == expected_pairs
            else "fail"
        ),
        "expected_pair_count": 12,
        "pair_count": len(pairs),
        "minimum_center_distance_m": minimum_distance,
        "maximum_center_distance_m": maximum_distance,
        "maximum_axis_delta_degrees": maximum_axis_delta_degrees,
        "maximum_pair_overlap_ratio": maximum_pair_overlap_ratio,
        "maximum_lateral_offset_m": maximum_lateral_offset_m,
        "minimum_axial_offset_m": minimum_axial_offset_m,
        "maximum_same_type_overlap_ratio": maximum_same_type_overlap_ratio,
        "invalid_physical_evidence_object_ids": invalid_physical_evidence,
        "same_type_clearance_issues": same_type_clearance_issues,
        "pairs": pairs,
    }


def inventory_requirements(room_id: str) -> tuple[InventoryRequirement, ...]:
    if room_id.startswith("classroom_"):
        return CLASSROOM_INVENTORY
    return ROOM_INVENTORIES.get(room_id, ())


def evaluate_room_inventory(
    room_id: str,
    state: dict[str, Any],
    prompt: str,
    *,
    expected_prompt_sha256: str | None = None,
    asset_root: Path | None = None,
    require_prompt_binding: bool = False,
) -> dict[str, Any]:
    issues: list[str] = []
    repairs: list[str] = []
    prompt_evidence = _prompt_contract_evidence(
        room_id,
        prompt,
        expected_prompt_sha256,
        require_expected_prompt_sha256=require_prompt_binding,
    )
    if prompt_evidence["status"] != "pass":
        issues.extend(
            f"Per-room prompt binding failed: {item}."
            for item in prompt_evidence["issues"]
        )
        repairs.append("Re-run school_room_contract.py bind-layout before room generation.")

    all_semantic_objects = [
        (object_id, object_semantic_text(object_id, obj), obj)
        for object_id, obj in _iter_state_objects(state)
        if str(obj.get("object_type", "")).lower() not in {"wall", "floor", "ceiling"}
    ]
    requirements = inventory_requirements(room_id)
    semantic_objects: list[tuple[str, str, dict[str, Any]]] = []
    physical_evidence: dict[str, dict[str, Any]] = {}
    invalid_physical_candidates: dict[str, dict[str, Any]] = {}
    for object_id, text, obj in all_semantic_objects:
        matching_requirements = [
            requirement.key
            for requirement in requirements
            if _matches(text, requirement)
        ]
        if not matching_requirements:
            continue
        evidence = physical_object_evidence(obj, asset_root=asset_root)
        physical_evidence[object_id] = evidence
        if evidence["status"] == "pass":
            semantic_objects.append((object_id, text, obj))
        else:
            invalid_physical_candidates[object_id] = {
                "matching_requirements": matching_requirements,
                **evidence,
            }
    if invalid_physical_candidates:
        issues.append(
            "Semantic inventory candidates lack real mutable mesh/pose/bounds evidence: "
            + "; ".join(
                f"{object_id}: {', '.join(record['issues'])}"
                for object_id, record in sorted(invalid_physical_candidates.items())
            )
        )
        repairs.append(
            "Replace label-only inventory candidates with non-immutable scene objects "
            "whose mesh files, finite poses, and positive 3D bounds are recorded."
        )
    matches, raw_matches = _minimum_unique_assignment(requirements, semantic_objects)
    counts: dict[str, int] = {}
    raw_counts: dict[str, int] = {}
    for requirement in requirements:
        count = len(matches[requirement.key])
        raw_count = len(raw_matches[requirement.key])
        counts[requirement.key] = count
        raw_counts[requirement.key] = raw_count
        if count < requirement.minimum:
            issues.append(
                f"Missing independently represented {requirement.label}: uniquely assigned "
                f"{count}, minimum {requirement.minimum} (raw semantic matches={raw_count})."
            )
            repairs.append(f"Add correctly labeled {requirement.label} and rerun this room.")
        if requirement.maximum is not None and raw_count > requirement.maximum:
            issues.append(
                f"Too many {requirement.label}: found {raw_count}, maximum {requirement.maximum}."
            )
            repairs.append(f"Remove excess {requirement.label}; preserve clear circulation.")

    spatial_checks: dict[str, Any] = {}
    if room_id.startswith("classroom_"):
        objects_by_id = {object_id: obj for object_id, _text, obj in semantic_objects}
        desk_chair_pairing = _desk_chair_pairing(
            objects_by_id,
            matches.get("student_desks", []),
            matches.get("student_chairs", []),
        )
        spatial_checks["desk_chair_pairing"] = desk_chair_pairing
        if desk_chair_pairing["status"] != "pass":
            issues.append(
                "The 12 student desks and chairs do not form 12 independently placed, "
                "one-to-one spatial pairs."
            )
            repairs.append(
                "Place one distinct chair 0.20-1.60 m from each distinct student desk, "
                "with floor support, aligned orientation, usable lateral clearance, finite "
                "transforms, and no reused chair."
            )
    return {
        "status": "pass" if not issues else "fail",
        "profile": PROFILE,
        "room_id": room_id,
        "counts": counts,
        "raw_semantic_counts": raw_counts,
        "matched_object_ids": matches,
        "raw_matched_object_ids": raw_matches,
        "prompt_binding": prompt_evidence,
        "physical_object_evidence": physical_evidence,
        "invalid_physical_candidates": invalid_physical_candidates,
        "spatial_checks": spatial_checks,
        "critical_issues": issues,
        "repair_instructions": repairs,
        "canonical_requirements": ROOM_REQUIREMENTS.get(room_id, ""),
    }


def classify_articulated_role(room_id: str, object_id: str, obj: dict[str, Any]) -> str | None:
    metadata = obj.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("is_articulated") is not True:
        return None
    text = object_semantic_text(object_id, obj)
    for role, rule in ARTICULATED_ROLE_RULES.items():
        if room_id in rule["rooms"] and all(
            re.search(pattern, text, flags=re.IGNORECASE) for pattern in rule["patterns"]
        ):
            return role
    return None


def collect_required_articulated_roles(
    room_states: dict[str, dict[str, Any]],
    *,
    require_runtime_provenance: bool = False,
) -> dict[str, Any]:
    roles: dict[str, list[dict[str, Any]]] = {role: [] for role in ARTICULATED_ROLE_RULES}
    invalid_role_candidates: list[dict[str, Any]] = []
    for room_id, state in room_states.items():
        for object_id, obj in _iter_state_objects(state):
            role = classify_articulated_role(room_id, object_id, obj)
            if role is None:
                continue
            metadata = obj.get("metadata", {})
            record = {
                "room_id": room_id,
                "object_id": str(obj.get("object_id") or object_id),
                "articulated_id": str(metadata.get("articulated_id") or "").strip(),
                "articulated_source": str(
                    metadata.get("articulated_source") or ""
                ).lower(),
                "asset_source": str(metadata.get("asset_source") or "").lower(),
                "sdf_path": str(obj.get("sdf_path") or "").strip(),
            }
            if require_runtime_provenance:
                provenance_issues = []
                if record["asset_source"] != "articulated":
                    provenance_issues.append("asset_source is not exactly articulated")
                if record["articulated_source"] not in ARTICULATED_ROLE_SOURCES:
                    provenance_issues.append(
                        "articulated_source is not an enabled Artiverse/ArtVIP source"
                    )
                if not record["articulated_id"]:
                    provenance_issues.append("articulated_id is missing")
                if not record["sdf_path"]:
                    provenance_issues.append("runtime SDF path is missing")
                if provenance_issues:
                    invalid_role_candidates.append(
                        {
                            "role": role,
                            **record,
                            "issues": provenance_issues,
                        }
                    )
                    continue
            roles[role].append(record)
    missing = sorted(role for role, records in roles.items() if not records)
    artiverse_roles = sorted(
        role
        for role, records in roles.items()
        if any(record["articulated_source"] == "artiverse" for record in records)
    )
    issues = []
    if missing:
        issues.append(f"Missing required articulated furniture roles: {missing}")
    if not artiverse_roles:
        issues.append("None of the three required articulated furniture roles is sourced from Artiverse.")
    if invalid_role_candidates:
        issues.append(
            "Required articulated-role candidates have invalid runtime provenance: "
            + "; ".join(
                f"{record['role']}/{record['room_id']}/{record['object_id']}: "
                + ", ".join(record["issues"])
                for record in invalid_role_candidates
            )
        )
    return {
        "status": "pass" if not issues else "fail",
        "profile": PROFILE,
        "roles": roles,
        "missing_roles": missing,
        "artiverse_roles": artiverse_roles,
        "invalid_role_candidates": invalid_role_candidates,
        "critical_issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    bind = subparsers.add_parser("bind-layout")
    bind.add_argument("--layout", required=True, type=Path)
    bind.add_argument("--input-manifest", required=True, type=Path)
    bind.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "bind-layout":
        result = bind_layout_prompts(args.layout, args.input_manifest, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
