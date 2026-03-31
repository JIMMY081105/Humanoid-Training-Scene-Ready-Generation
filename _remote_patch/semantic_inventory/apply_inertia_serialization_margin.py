#!/usr/bin/env python3
"""Make generated inertia repair survive SDF decimal serialization."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path


EXPECTED_SHA256 = "30ae785da0ed08f390bb5059dfe2a44f595e832a24585e09e96ada54aa6fa745"


OLD_DEFICIT = """        deficit = (
            sorted_eigs[2] + _EIGENVALUE_EPSILON - (sorted_eigs[0] + sorted_eigs[1])
        )
"""
NEW_DEFICIT = """        # Keep a scale-aware margin that survives text serialization and
        # Drake's subsequent eigendecomposition.  The old absolute 1e-10
        # margin was rounded away by six-digit scientific notation.
        triangle_margin = max(
            _EIGENVALUE_EPSILON,
            float(sorted_eigs[2]) * 1e-7,
        )
        deficit = (
            sorted_eigs[2] + triangle_margin - (sorted_eigs[0] + sorted_eigs[1])
        )
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    args = parser.parse_args()
    target = args.target.resolve(strict=True)
    before = digest(target)
    if before != EXPECTED_SHA256:
        raise SystemExit(f"unexpected inertia-utils digest: {before}")
    source = target.read_text(encoding="utf-8")
    if source.count(OLD_DEFICIT) != 1:
        raise SystemExit("inertia triangle-margin anchor mismatch")
    if source.count(':.6e}') != 6:
        raise SystemExit("inertia serialization anchor mismatch")
    updated = source.replace(OLD_DEFICIT, NEW_DEFICIT, 1).replace(':.6e}', ':.17e}')

    args.backup_dir.mkdir(parents=True, exist_ok=True)
    backup = args.backup_dir / f"{target.name}.{before}.backup"
    if not backup.exists():
        shutil.copy2(target, backup, follow_symlinks=False)
    elif digest(backup) != before:
        raise SystemExit("inertia-utils backup mismatch")

    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        target.stat().st_mode & 0o777,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(updated)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    print("INERTIA_SERIALIZATION_MARGIN_PATCH_PASS", before, digest(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
