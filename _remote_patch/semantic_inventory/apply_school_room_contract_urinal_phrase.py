#!/usr/bin/env python3
"""Accept an attested two-urinal description with a material adjective."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import tempfile
from pathlib import Path


EXPECTED_SHA256 = "41f176bf755f53f7b90bbe63dfee25344e500a2017beeb164f2b5ddb405e8163"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    args = parser.parse_args()
    info = args.target.lstat()
    if not stat.S_ISREG(info.st_mode) or args.target.is_symlink():
        raise SystemExit("unsafe target")
    before = sha256(args.target)
    if before != EXPECTED_SHA256:
        raise SystemExit(f"unexpected target digest {before}")
    source = args.target.read_text(encoding="utf-8")
    old = 'r"\\b(?:double|dual|two)[- ]urinals?\\b",'
    new = 'r"\\b(?:double|dual)[- ]urinals?\\b|\\btwo\\b.{0,24}\\burinals?\\b",'
    if source.count(old) != 1:
        raise SystemExit("two-urinal regex anchor mismatch")
    source = source.replace(old, new, 1)
    args.backup_dir.mkdir(parents=True, exist_ok=True)
    backup = args.backup_dir / f"{args.target.name}.{before}.backup"
    if not backup.exists():
        shutil.copy2(args.target, backup)
    if sha256(backup) != before:
        raise SystemExit("backup digest mismatch")
    fd, temporary = tempfile.mkstemp(prefix=f".{args.target.name}.", dir=args.target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, stat.S_IMODE(info.st_mode))
        os.replace(temporary, args.target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print("SCHOOL_ROOM_CONTRACT_URINAL_PHRASE_PASS", before, sha256(args.target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
