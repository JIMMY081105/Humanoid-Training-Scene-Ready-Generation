#!/usr/bin/env python3
"""Fail closed unless staged prompt/reference inputs match their immutable manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json

from pathlib import Path
from typing import Any


REQUIRED_FILES = (
    "prompt_original.txt",
    "scene_contract_appendix.txt",
    "prompt.txt",
    "prompt.csv",
    "reference.png",
    "input_manifest.json",
)
REQUIRED_PIPELINE_CONTRACT = {
    "id": "scenesmith_full_quality_v1",
    "final_assembly_policy": "external_artiverse_gated",
    "required_articulated_source": "artiverse",
}

# These values come from the user's external source prompt and reference image,
# not from the mutable staged manifest.  Keeping the binding in executable code
# makes replacement of the manifest *and* all staged inputs fail closed for the
# one production profile this repository is intended to run.
SCHOOL_REFERENCE_RUN_NAME = (
    "full_quality_school_reference_sam3d_artvip_artiverse_20260710"
)
SCHOOL_REFERENCE_EXTERNAL_BINDING = {
    "original_prompt_sha256": "5ddc06e1a9afa60b0417da882c1ec53265eae23f3f9bdd2343360d123923f34c",
    "appendix_sha256": "7f7afbe051a26563d47b778f8f4a0286fc298f36c0256fba230b811020ab8e6d",
    "effective_prompt_sha256": "ac8d297cc9a2d605f41b4bcd7abd52aac29bfd0f195875840342ee1e6a7da86f",
    "effective_prompt_chars": 17181,
    "reference_image_sha256": "7ba62c39ac98cd2b21d9a5a97fd6f9b90d7efaf447d5ac9a6148524b5a8dbe48",
}


def _normalized_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _text_hash(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def validate(input_dir: Path) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    errors: list[str] = []
    missing = [name for name in REQUIRED_FILES if not (input_dir / name).is_file()]
    if missing:
        return {
            "status": "fail",
            "input_dir": str(input_dir),
            "missing_files": missing,
            "critical_issues": [f"Missing required input files: {missing}"],
        }

    manifest_path = input_dir / "input_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "fail",
            "input_dir": str(input_dir),
            "critical_issues": [f"Cannot parse input manifest: {exc}"],
        }
    if not isinstance(manifest, dict):
        return {
            "status": "fail",
            "input_dir": str(input_dir),
            "critical_issues": ["Input manifest is not a JSON object"],
        }

    original = _normalized_text(input_dir / "prompt_original.txt")
    appendix = _normalized_text(input_dir / "scene_contract_appendix.txt")
    effective = _normalized_text(input_dir / "prompt.txt")
    reconstructed = original + "\n" + appendix
    evidence = {
        "original_prompt_sha256": _text_hash(original),
        "appendix_sha256": _text_hash(appendix),
        "effective_prompt_sha256": _text_hash(effective),
        "effective_prompt_chars": len(effective),
        "reference_image_sha256": _sha256_bytes(
            (input_dir / "reference.png").read_bytes()
        ),
    }
    for key, actual in evidence.items():
        expected = manifest.get(key)
        if actual != expected:
            errors.append(f"Manifest mismatch for {key}: actual={actual} expected={expected}")
    external_binding = None
    if manifest.get("run_name") == SCHOOL_REFERENCE_RUN_NAME:
        external_binding = dict(SCHOOL_REFERENCE_EXTERNAL_BINDING)
        for key, expected in SCHOOL_REFERENCE_EXTERNAL_BINDING.items():
            actual = evidence.get(key)
            if actual != expected:
                errors.append(
                    f"External school-reference binding mismatch for {key}: "
                    f"actual={actual} expected={expected}"
                )
    if effective != reconstructed:
        errors.append(
            "prompt.txt is not exactly normalized prompt_original.txt + one LF + "
            "scene_contract_appendix.txt"
        )

    try:
        with (input_dir / "prompt.csv").open(
            newline="", encoding="utf-8-sig"
        ) as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error) as exc:
        rows = []
        errors.append(f"Cannot parse prompt.csv: {exc}")
    if len(rows) != 1:
        errors.append(f"prompt.csv must contain exactly one scene row, found {len(rows)}")
    else:
        row = rows[0]
        if str(row.get("scene_index", "")) != str(manifest.get("scene_index", "")):
            errors.append("prompt.csv scene_index differs from input_manifest.json")
        csv_prompt = str(row.get("prompt", "")).replace("\r\n", "\n")
        if csv_prompt != effective:
            errors.append("prompt.csv prompt differs from prompt.txt")

    expected_ids = manifest.get("expected_room_ids")
    if not isinstance(expected_ids, list) or len(expected_ids) != 11:
        errors.append("Manifest must contain exactly 11 expected_room_ids")
    elif len(set(expected_ids)) != 11:
        errors.append("Manifest expected_room_ids contains duplicates")

    if manifest.get("pipeline_contract") != REQUIRED_PIPELINE_CONTRACT:
        errors.append(
            "Manifest pipeline_contract must require external_artiverse_gated "
            "assembly and the artiverse articulated source"
        )

    return {
        "status": "pass" if not errors else "fail",
        "input_dir": str(input_dir),
        "manifest": str(manifest_path.resolve()),
        "run_name": manifest.get("run_name"),
        "scene_index": manifest.get("scene_index"),
        "expected_room_ids": expected_ids,
        "pipeline_contract": manifest.get("pipeline_contract"),
        "evidence": evidence,
        "external_input_binding": external_binding,
        "critical_issues": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.input_dir)
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
