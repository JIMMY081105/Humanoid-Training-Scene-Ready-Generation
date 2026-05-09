"""Shared fail-closed helpers for SceneSmith visual quality gates."""

from __future__ import annotations

import base64
import json
import math
import mimetypes

from pathlib import Path
from typing import Any, Iterable, Sequence


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object, rejecting arrays and scalar JSON values."""

    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def image_to_data_url(path: Path) -> str:
    """Encode a local image for both OpenAI and Codex VLMService backends."""

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"Image is empty: {path}")

    mime_type, _encoding = mimetypes.guess_type(path.name)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image extension: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_multimodal_content(
    instruction: str, labeled_images: Sequence[tuple[str, Path]]
) -> list[dict[str, Any]]:
    """Build chat-style multimodal content accepted by SceneSmith VLMService."""

    content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
    for index, (label, path) in enumerate(labeled_images, start=1):
        content.append(
            {"type": "text", "text": f"Image {index}: {label} ({path.name})"}
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(path)},
            }
        )
    return content


def call_vlm_service(
    *,
    messages: list[dict[str, Any]],
    model: str,
    backend: str,
    vlm_service: Any | None = None,
) -> Any:
    """Call SceneSmith's VLMService; the import stays lazy for local unit tests."""

    if vlm_service is None:
        from scenesmith.agent_utils.vlm_service import VLMService

        vlm_service = VLMService(backend=backend)
    return vlm_service.create_completion(
        model=model,
        messages=messages,
        reasoning_effort="high",
        verbosity="low",
        response_format={"type": "json_object"},
        vision_detail="high",
    )


def parse_visual_assessment(
    raw_response: Any, required_score_keys: Sequence[str]
) -> dict[str, Any]:
    """Strictly validate the visual judge response.

    Invalid or incomplete responses raise. Gate callers catch these errors and
    emit a failure result; a malformed model response can therefore never pass.
    """

    if isinstance(raw_response, str):
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Visual judge returned invalid JSON: {exc}") from exc
    else:
        payload = raw_response
    if not isinstance(payload, dict):
        raise ValueError("Visual judge response is not a JSON object")

    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, dict):
        raise ValueError("Visual judge response has no scores object")

    scores: dict[str, float] = {}
    for key in required_score_keys:
        value = raw_scores.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Visual judge score {key!r} is not numeric")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 10.0:
            raise ValueError(f"Visual judge score {key!r} is outside 0-10")
        scores[key] = number

    critical_issues = _string_list(payload.get("critical_issues"), "critical_issues")
    repair_instructions = _string_list(
        payload.get("repair_instructions"), "repair_instructions"
    )
    result: dict[str, Any] = {
        "scores": scores,
        "critical_issues": critical_issues,
        "repair_instructions": repair_instructions,
    }
    if "observations" in payload:
        result["observations"] = payload["observations"]
    if "requirement_evidence" in payload:
        result["requirement_evidence"] = payload["requirement_evidence"]
    return result


def normalize_scores(
    raw_scores: Any, required_score_keys: Sequence[str]
) -> dict[str, float]:
    """Validate an existing gate's scores using the same 0-10 contract."""

    if not isinstance(raw_scores, dict):
        raise ValueError("Gate result has no scores object")
    scores: dict[str, float] = {}
    for key in required_score_keys:
        value = raw_scores.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Gate score {key!r} is not numeric")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 10.0:
            raise ValueError(f"Gate score {key!r} is outside 0-10")
        scores[key] = number
    return scores


def zero_scores(required_score_keys: Sequence[str]) -> dict[str, float]:
    return {key: 0.0 for key in required_score_keys}


def unique_strings(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"Visual judge field {field!r} is not a list")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"Visual judge field {field!r} contains non-string values")
    return unique_strings(value)
