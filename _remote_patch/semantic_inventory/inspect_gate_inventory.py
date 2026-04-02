#!/usr/bin/env python3
"""Print compact deterministic gate inventory diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path


root = Path(sys.argv[1])
for room_id in sys.argv[2:]:
    payload = json.loads((root / f"{room_id}.json").read_text(encoding="utf-8"))
    inventory = payload.get("inventory") or payload.get("semantic_inventory") or {}
    print(json.dumps({
        "room_id": room_id,
        "status": payload.get("status"),
        "scores": payload.get("scores", payload.get("score")),
        "critical_issues": payload.get("critical_issues"),
        "inventory": {
            "status": inventory.get("status"),
            "counts": inventory.get("counts"),
            "raw_counts": inventory.get("raw_semantic_counts"),
            "matched": inventory.get("matched_object_ids"),
            "raw_matched": inventory.get("raw_matched_object_ids"),
            "issues": inventory.get("critical_issues"),
        },
    }, sort_keys=True))
