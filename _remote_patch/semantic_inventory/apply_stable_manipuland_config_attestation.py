#!/usr/bin/env python3
"""Make manipuland config attestations independent of constructor side effects."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
from pathlib import Path


EXPECTED_BEFORE = "32bd6b5869a774f155eda5093f9879df2f1c62e18f2497caee6d604dc551fefe"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source: str, before: str, after: str, label: str) -> str:
    if source.count(before) != 1:
        raise SystemExit(f"{label} source scope mismatch")
    return source.replace(before, after, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    args = parser.parse_args()
    target = args.target.resolve(strict=True)
    if sha(target) != EXPECTED_BEFORE:
        raise SystemExit(f"unexpected target hash: {sha(target)}")

    source = target.read_text(encoding="utf-8")
    source = replace_once(
        source,
        """    ):
        # Initialize base agent (sessions, checkpoint state, prompt registry).
        BaseStatefulAgent.__init__(
""",
        """    ):
        # Bind checkpoints to the resolved production config before constructors
        # can normalize or otherwise mutate the shared DictConfig in place.
        resolved_checkpoint_config = OmegaConf.to_container(cfg, resolve=True)
        self._checkpoint_config_sha256 = _sha256_bytes(
            _canonical_json_bytes(resolved_checkpoint_config)
        )

        # Initialize base agent (sessions, checkpoint state, prompt registry).
        BaseStatefulAgent.__init__(
""",
        "constructor config capture",
    )
    source = replace_once(
        source,
        """        selection = _selection_payload(furniture_selection)
        resolved_config = OmegaConf.to_container(self.cfg, resolve=True)
        return {
            "furniture_index": furniture_index,
            "furniture_id": str(furniture_selection.furniture_id),
            "selection": selection,
            "selection_sha256": _sha256_bytes(_canonical_json_bytes(selection)),
            "config_sha256": _sha256_bytes(_canonical_json_bytes(resolved_config)),
            "checkpoint_runtime_sha256": _sha256_file(Path(__file__).resolve()),
        }
""",
        """        selection = _selection_payload(furniture_selection)
        return {
            "furniture_index": furniture_index,
            "furniture_id": str(furniture_selection.furniture_id),
            "selection": selection,
            "selection_sha256": _sha256_bytes(_canonical_json_bytes(selection)),
            "config_sha256": self._checkpoint_config_sha256,
            "checkpoint_runtime_sha256": _sha256_file(Path(__file__).resolve()),
        }
""",
        "checkpoint context hash",
    )
    source = replace_once(
        source,
        """    def _plan_context(self, input_scene_hash: str) -> dict[str, Any]:
        resolved_config = OmegaConf.to_container(self.cfg, resolve=True)
        return {
            "room_id": str(getattr(self.scene, "room_id", "")),
            "input_scene_content_hash": input_scene_hash,
            "config_sha256": _sha256_bytes(_canonical_json_bytes(resolved_config)),
            "checkpoint_runtime_sha256": _sha256_file(Path(__file__).resolve()),
        }
""",
        """    def _plan_context(self, input_scene_hash: str) -> dict[str, Any]:
        return {
            "room_id": str(getattr(self.scene, "room_id", "")),
            "input_scene_content_hash": input_scene_hash,
            "config_sha256": self._checkpoint_config_sha256,
            "checkpoint_runtime_sha256": _sha256_file(Path(__file__).resolve()),
        }
""",
        "plan context hash",
    )

    args.backup_dir.mkdir(parents=True, exist_ok=True)
    backup = args.backup_dir / f"{target.name}.{EXPECTED_BEFORE}"
    if not backup.exists():
        shutil.copy2(target, backup, follow_symlinks=False)
    elif sha(backup) != EXPECTED_BEFORE:
        raise SystemExit("backup hash mismatch")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp.", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    print("STABLE_MANIPULAND_CONFIG_ATTESTATION_APPLIED", sha(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
