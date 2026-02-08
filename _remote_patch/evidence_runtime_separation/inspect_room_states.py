#!/usr/bin/env python3
"""Print concise state hashes and metadata needed for targeted room repairs."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    run = Path(sys.argv[1])
    concise = "--concise" in sys.argv[2:]
    room_ids = [value for value in sys.argv[2:] if value != "--concise"]
    for room_id in room_ids:
        state_path = run / "scene_000" / f"room_{room_id}" / "scene_states" / "final_scene" / "scene_state.json"
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        objects = payload.get("objects", payload.get("scene_objects", []))
        if isinstance(objects, dict):
            objects = list(objects.values())
        records = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            metadata = obj.get("metadata") or {}
            records.append({
                "id": obj.get("object_id", obj.get("id")),
                "name": obj.get("name"),
                "type": obj.get("object_type"),
                "translation": (obj.get("transform") or {}).get("translation"),
                "bbox_min": obj.get("bbox_min"),
                "bbox_max": obj.get("bbox_max"),
                "geometry_path": obj.get("geometry_path"),
                "sdf_path": obj.get("sdf_path"),
                "composite_members": metadata.get("composite_members"),
                "semantic_role": metadata.get("semantic_role"),
            })
        result = {
            "room_id": room_id,
            "sha256": sha256(state_path),
            "room_geometry": payload.get("room_geometry"),
            "composite_objects": [r for r in records if r["composite_members"]],
            "repair_relevant": [
                r for r in records
                if any(token in str(r["id"]).lower() for token in (
                    "desk", "cubby", "whiteboard", "board", "display", "screen", "frame",
                    "shelf", "drawer", "teacher", "supply", "dryer", "entrance", "welcome",
                    "daylight", "rug", "pendant", "partition", "mirror", "soap", "sanitary",
                ))
            ],
        }
        if concise:
            result.pop("room_geometry")
            result.pop("repair_relevant")
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
