#!/usr/bin/env python3
"""Atomically rebase a copied Room 1 state from SQZ to its ParaCloud topology.

The source checkpoint remains preserved verbatim beside the rewritten state.
Only absolute filesystem references are changed; object transforms, assets,
physics metadata, prompts, and inventory are left byte-for-byte as loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time

from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_prefix(value: Any, *, source: str, target: str, changed: list[str], path: str = "$") -> Any:
    if isinstance(value, dict):
        return {
            key: replace_prefix(item, source=source, target=target, changed=changed, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            replace_prefix(item, source=source, target=target, changed=changed, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, str) and (value == source or value.startswith(source + "/")):
        changed.append(path)
        return target + value[len(source):]
    return value


def string_values(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from string_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from string_values(item)
    elif isinstance(value, str):
        yield value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--source-scene", required=True, type=Path)
    parser.add_argument("--target-scene", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    state_path = args.state.resolve()
    source = str(args.source_scene)
    target = str(args.target_scene.resolve())
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    if not args.target_scene.is_dir():
        raise FileNotFoundError(args.target_scene)

    original_sha256 = sha256_file(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    changed: list[str] = []
    rewritten = replace_prefix(state, source=source, target=target, changed=changed)
    if not changed:
        raise RuntimeError("No SQZ scene paths were found to rebase")
    unresolved = [value for value in string_values(rewritten) if value == source or value.startswith(source + "/")]
    if unresolved:
        raise RuntimeError(f"SQZ scene paths remain after rebasing: {len(unresolved)}")
    missing_targets = sorted(
        {value for value in string_values(rewritten) if value.startswith(target + "/") and not Path(value).exists()}
    )
    if missing_targets:
        raise FileNotFoundError(
            "Rebased scene references are unavailable: " + "; ".join(missing_targets[:5])
        )

    temporary = state_path.with_name(f".{state_path.name}.paracloud_rebase.tmp")
    temporary.write_text(json.dumps(rewritten, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        reloaded = json.loads(temporary.read_text(encoding="utf-8"))
        if reloaded != rewritten:
            raise RuntimeError("Temporary rebased state did not round-trip")
        backup = state_path.with_name(f"{state_path.name}.sqz_pathbound_{int(time.time())}")
        shutil.copy2(state_path, backup)
        os.replace(temporary, state_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    receipt = {
        "status": "pass",
        "operation": "absolute_path_rebase_only",
        "source_scene": source,
        "target_scene": target,
        "prior_state_sha256": original_sha256,
        "rebased_state_sha256": sha256_file(state_path),
        "rewritten_path_count": len(changed),
        "backup": str(backup),
        "changed_json_paths": changed,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary_receipt = args.receipt.with_name(f".{args.receipt.name}.tmp")
    temporary_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_receipt, args.receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
