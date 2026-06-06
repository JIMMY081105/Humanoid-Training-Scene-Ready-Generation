import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools" / "sage_scene_checker"
sys.path.insert(0, str(TOOLS_DIR))

from repair_prompt_builder import build_repair_suggestions  # noqa: E402


def test_repair_suggestions_skip_passes_and_deduplicate_failures():
    checks = [
        {"id": "object.bbox_valid", "status": "pass", "object_id": "chair_1"},
        {"id": "object.bbox_valid", "status": "fail", "object_id": "chair_1"},
        {"id": "object.bbox_valid", "status": "fail", "object_id": "chair_1"},
        {"id": "door.clearance_blocked", "status": "fail", "object_id": "chair_2"},
    ]

    suggestions = build_repair_suggestions(checks)

    assert [item["check_id"] for item in suggestions] == [
        "object.bbox_valid",
        "door.clearance_blocked",
    ]
    assert suggestions[0]["object_id"] == "chair_1"
    assert "bbox" in suggestions[0]["suggestion"]
