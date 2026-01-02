#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOM_IDS = [
    "ingredient_receiving", "dry_storage", "cold_storage",
    "washing_preparation", "qc_laboratory", "office_administration",
    "processing_hall", "packaging_hall", "finished_goods_storage",
    "maintenance", "changing_room", "break_room", "boys_toilet",
    "girls_toilet",
]
RUN_NAME = "full_quality_factory_sam3d_artvip_artiverse_20260713"


def normalized(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--appendix", type=Path, required=True)
    parser.add_argument("--factory-contract", type=Path, required=True)
    args = parser.parse_args()
    target = args.input_dir.resolve()
    original_path = target / "prompt_original.txt"
    if not original_path.is_file() or original_path.is_symlink():
        raise RuntimeError("prompt_original.txt must be a regular, unlinked user input")
    original = normalized(original_path)
    appendix = normalized(args.appendix.resolve(strict=True))
    contract_raw = args.factory_contract.resolve(strict=True).read_bytes()
    contract = json.loads(contract_raw)
    if contract.get("room_order") != ROOM_IDS:
        raise RuntimeError("factory contract room order differs from the immutable 14 IDs")
    effective = original + "\n" + appendix
    target.mkdir(parents=True, exist_ok=True)
    (target / "scene_contract_appendix.txt").write_text(appendix, encoding="utf-8", newline="\n")
    (target / "prompt.txt").write_text(effective, encoding="utf-8", newline="\n")
    with (target / "prompt.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scene_index", "prompt"])
        writer.writeheader()
        writer.writerow({"scene_index": 0, "prompt": effective})
    manifest = {
        "schema_version": 1,
        "run_name": RUN_NAME,
        "scene_index": 0,
        "expected_room_ids": ROOM_IDS,
        "hash_normalization": "utf-8 with CRLF normalized to LF for text",
        "original_prompt_sha256": sha(original.encode()),
        "appendix_sha256": sha(appendix.encode()),
        "effective_prompt_sha256": sha(effective.encode()),
        "effective_prompt_chars": len(effective),
        "reference_image_sha256": None,
        "reference_authority": "prompt_only_no_user_image_supplied",
        "factory_contract_sha256": sha(contract_raw),
        "pipeline_contract": {
            "id": "scenesmith_full_quality_v1",
            "final_assembly_policy": "external_artiverse_gated",
            "required_articulated_source": "artiverse"
        }
    }
    (target / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
