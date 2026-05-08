#!/usr/bin/env python3
"""Prove that SceneSmith's configured VLM can receive an image and return JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import os

from pathlib import Path
from typing import Any

try:
    from .visual_gate_utils import (
        build_multimodal_content,
        call_vlm_service,
        parse_visual_assessment,
    )
except ImportError:
    from visual_gate_utils import (  # type: ignore[no-redef]
        build_multimodal_content,
        call_vlm_service,
        parse_visual_assessment,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def run(
    *,
    image: Path,
    output: Path,
    model: str,
    backend: str,
    vlm_service: Any | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "fail",
        "model": model,
        "backend": backend,
        "image": str(image),
    }
    try:
        if not image.is_file() or image.stat().st_size <= 0:
            raise FileNotFoundError(f"Vision-smoke image is missing or empty: {image}")
        result["image_sha256"] = _sha256(image)
        messages = [
            {
                "role": "system",
                "content": (
                    "This is a transport/schema smoke test. Inspect only the supplied "
                    "image and return the exact requested JSON shape."
                ),
            },
            {
                "role": "user",
                "content": build_multimodal_content(
                    "Confirm the image is readable. Return only this JSON object, with "
                    "image_readability from 0 to 10 and empty lists when readable: "
                    '{"scores":{"image_readability":0},"critical_issues":[],'
                    '"repair_instructions":[]}',
                    [("required pipeline reference image", image)],
                ),
            },
        ]
        raw = call_vlm_service(
            messages=messages,
            model=model,
            backend=backend,
            vlm_service=vlm_service,
        )
        result["raw_response"] = raw if isinstance(raw, str) else raw
        assessment = parse_visual_assessment(raw, ("image_readability",))
        result["assessment"] = assessment
        if assessment["scores"]["image_readability"] < 1:
            raise ValueError("VLM reported that the supplied reference image is unreadable")
        if assessment["critical_issues"]:
            raise ValueError(
                "VLM vision smoke returned critical issues: "
                + "; ".join(assessment["critical_issues"])
            )
        result["status"] = "pass"
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
    _write(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--vlm-backend", choices=("openai", "codex"), default="openai")
    args = parser.parse_args()
    result = run(
        image=args.image.resolve(),
        output=args.output.resolve(),
        model=args.model,
        backend=args.vlm_backend,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
