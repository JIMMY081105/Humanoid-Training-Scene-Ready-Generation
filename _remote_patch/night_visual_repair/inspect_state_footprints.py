#!/usr/bin/env python3
"""Print the largest rotation-aware object footprints in a saved room state."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path)
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    records = []
    for object_id, obj in state.get("objects", {}).items():
        minimum, maximum = obj.get("bbox_min"), obj.get("bbox_max")
        transform = obj.get("transform", {})
        if not minimum or not maximum:
            continue
        rotation = transform.get("rotation", transform.get("rotation_wxyz", [1, 0, 0, 0]))
        if isinstance(rotation, dict):
            rotation = rotation.get("wxyz", [1, 0, 0, 0])
        w, x, y, z = map(float, rotation)
        # XY extent of a rotated axis-aligned 3-D box.
        matrix = (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        )
        extents = [(float(maximum[i]) - float(minimum[i])) / 2 for i in range(3)]
        ex = sum(abs(matrix[0][i]) * extents[i] for i in range(3))
        ey = sum(abs(matrix[1][i]) * extents[i] for i in range(3))
        area = 4 * ex * ey
        records.append((area, str(object_id), obj.get("name"), obj.get("object_type"), minimum, maximum))
    for area, object_id, name, object_type, minimum, maximum in sorted(records, reverse=True)[: args.limit]:
        print(json.dumps({"area": round(area, 5), "object_id": object_id, "name": name, "type": object_type, "bbox_min": minimum, "bbox_max": maximum}, sort_keys=True))
    print(json.dumps({"object_count": len(records), "summed_area": round(sum(item[0] for item in records), 5)}, sort_keys=True))


if __name__ == "__main__":
    main()
