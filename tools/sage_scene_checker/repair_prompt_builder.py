"""Build deterministic repair suggestions from checker failures."""

from __future__ import annotations


def build_repair_suggestions(checks: list[dict]) -> list[dict]:
    """Convert failed checks into simple local repair suggestions."""

    suggestions: list[dict] = []
    seen: set[tuple[str, str | None]] = set()
    for check in checks:
        if check.get("status") == "pass":
            continue
        check_id = str(check.get("id", "unknown"))
        object_id = check.get("object_id")
        key = (check_id, object_id)
        if key in seen:
            continue
        seen.add(key)
        suggestions.append(
            {
                "check_id": check_id,
                "object_id": object_id,
                "suggestion": _suggestion_for(check),
            }
        )
    return suggestions


def _suggestion_for(check: dict) -> str:
    check_id = str(check.get("id", ""))
    object_id = check.get("object_id") or "the object"
    if "door" in check_id:
        return f"move {object_id} away from the door clearance region"
    if "collision" in check_id or "overlap" in check_id:
        return f"remove or move overlapping furniture involving {object_id}"
    if "asset" in check_id:
        return f"assign a valid local asset path for {object_id}"
    if "bbox" in check_id:
        return f"fix invalid bbox dimensions for {object_id}"
    if "category" in check_id:
        return f"add a semantic category for {object_id}"
    if "support" in check_id:
        return f"fix support surface parent metadata for {object_id}"
    if "room" in check_id or "inside" in check_id:
        return f"move {object_id} inside its assigned room bounds"
    if "transform" in check_id:
        return f"regenerate or repair the finite transform for {object_id}"
    return str(check.get("message") or "inspect and repair this scene issue")

