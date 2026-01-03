#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from stage_factory_inputs import ROOM_IDS, RUN_NAME


EXTERNAL_BINDING = {
    "original_prompt_sha256": "3bb58cf8e07012cc0c5381eb51730da03b25fb6659a08338238d9fd4baa5093b",
    "appendix_sha256": "ffc327f082431d407c15b25b039d7d2676786a03663f186777558e9ba8e1db19",
    "effective_prompt_sha256": "35387fa3fffb0af92b6c9e07e39f2b59a5443d497190f721090f82bd81734322",
    "effective_prompt_chars": 21330,
    "factory_contract_sha256": "f33e56eef4375d1f68684bd10c02d7d2f82ee72420b8a12570e8fcbe135a94cd",
}


def norm(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--factory-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    errors: list[str] = []
    required = ["prompt_original.txt", "scene_contract_appendix.txt", "prompt.txt", "prompt.csv", "input_manifest.json"]
    missing = [name for name in required if not (root / name).is_file() or (root / name).is_symlink()]
    if missing:
        errors.append(f"missing/untrusted input files: {missing}")
    manifest = json.loads((root / "input_manifest.json").read_text(encoding="utf-8")) if not missing else {}
    if not missing:
        original, appendix, effective = (norm(root / name) for name in ("prompt_original.txt", "scene_contract_appendix.txt", "prompt.txt"))
        contract_raw = args.factory_contract.resolve(strict=True).read_bytes()
        evidence = {
            "original_prompt_sha256": sha(original.encode()),
            "appendix_sha256": sha(appendix.encode()),
            "effective_prompt_sha256": sha(effective.encode()),
            "effective_prompt_chars": len(effective),
            "factory_contract_sha256": sha(contract_raw),
        }
        if effective != original + "\n" + appendix:
            errors.append("effective prompt is not exact original + LF + appendix")
        for key, value in evidence.items():
            if manifest.get(key) != value:
                errors.append(f"manifest mismatch for {key}")
            if EXTERNAL_BINDING.get(key) != value:
                errors.append(f"external executable binding mismatch for {key}")
        with (root / "prompt.csv").open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1 or rows[0].get("scene_index") != "0" or rows[0].get("prompt", "").replace("\r\n", "\n") != effective:
            errors.append("prompt.csv is not the exact one-row effective prompt")
    else:
        evidence = {}
    if manifest.get("run_name") != RUN_NAME or manifest.get("expected_room_ids") != ROOM_IDS:
        errors.append("run name or exact 14 room IDs changed")
    expected_pipeline = {"id":"scenesmith_full_quality_v1","final_assembly_policy":"external_artiverse_gated","required_articulated_source":"artiverse"}
    if manifest.get("pipeline_contract") != expected_pipeline:
        errors.append("pipeline contract markers changed")
    if manifest.get("reference_image_sha256") is not None or manifest.get("reference_authority") != "prompt_only_no_user_image_supplied":
        errors.append("factory run must not claim an unsupplied reference image")
    result = {"schema_version":1,"status":"pass" if not errors else "fail","input_dir":str(root),"evidence":evidence,"external_binding":EXTERNAL_BINDING,"critical_issues":errors}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
