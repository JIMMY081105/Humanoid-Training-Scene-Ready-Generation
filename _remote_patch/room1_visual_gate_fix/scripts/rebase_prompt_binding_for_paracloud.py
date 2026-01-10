#!/usr/bin/env python3
"""Rebase the two location fields in an immutable prompt-binding receipt.

The bound layout and manifest bytes must already match their recorded digests.
Only their absolute locations are rewritten, with an atomic backup retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time

from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--layout", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    binding_path = args.binding.resolve()
    layout = args.layout.resolve()
    manifest = args.input_manifest.resolve()
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    for key, artifact in (("layout", layout), ("input_manifest", manifest)):
        record = binding.get(key)
        if not isinstance(record, dict) or record.get("sha256") != sha256_file(artifact):
            raise RuntimeError(f"{key} digest does not match the active immutable artifact")
        record["path"] = str(artifact)

    temporary = binding_path.with_name(f".{binding_path.name}.paracloud_rebase.tmp")
    temporary.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    backup = binding_path.with_name(f"{binding_path.name}.sqz_pathbound_{int(time.time())}")
    shutil.copy2(binding_path, backup)
    os.replace(temporary, binding_path)
    payload = {
        "status": "pass",
        "operation": "prompt_binding_path_rebase_only",
        "binding": str(binding_path),
        "backup": str(backup),
        "layout": str(layout),
        "layout_sha256": sha256_file(layout),
        "input_manifest": str(manifest),
        "input_manifest_sha256": sha256_file(manifest),
        "rebased_binding_sha256": sha256_file(binding_path),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temp_receipt = args.receipt.with_name(f".{args.receipt.name}.tmp")
    temp_receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp_receipt, args.receipt)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
