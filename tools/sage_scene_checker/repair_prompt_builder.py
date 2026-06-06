"""Build deterministic repair suggestions from checker failures."""

from __future__ import annotations


SUGGESTION_RULES = (
    (("door",), "move {object_id} away from the door clearance region"),
    (("collision", "overlap"), "remove or move overlapping furniture involving {object_id}"),
    (("asset",), "assign a valid local asset path for {object_id}"),
    (("bbox",), "fix invalid bbox dimensions for {object_id}"),
    (("category",), "add a semantic category for {object_id}"),
    (("support",), "fix support surface parent metadata for {object_id}"),
    (("room", "inside"), "move {object_id} inside its assigned room bounds"),
    (("transform",), "regenerate or repair the finite transform for {object_id}"),
)


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
    for keywords, template in SUGGESTION_RULES:
        if any(keyword in check_id for keyword in keywords):
            return template.format(object_id=object_id)
    return str(check.get("message") or "inspect and repair this scene issue")

