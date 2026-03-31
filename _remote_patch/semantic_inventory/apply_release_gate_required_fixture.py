#!/usr/bin/env python3
"""Digest-bound atomic hook for the two required fixture reuse repairs."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import tempfile
from pathlib import Path


EXPECTED_SHA256 = "5bdf8adfb8b8a738efaf17c57fc216fc7f14ba298cc7d4c8c23f9b336e51e97e"


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
        raise SystemExit(f"unexpected release gate digest {before}")
    source = args.target.read_text(encoding="utf-8")
    old = 'python scripts/render_room_review_views.py --blend "$ROOM/scene_states/final_scene/scene.blend"'
    hook = '''case "$ROOM_ID" in
  classroom_02) FIXTURE_PORT_OFFSET=882 ;;
  classroom_06) FIXTURE_PORT_OFFSET=886 ;;
  *) FIXTURE_PORT_OFFSET= ;;
esac
if [ -n "$FIXTURE_PORT_OFFSET" ]; then
  python "$FACTORY/repair_required_room_fixture.py" \\
    --repo-dir "$EXEC" \\
    --run-dir "$RUN" \\
    --csv "$EXEC/inputs/full_quality_school_reference_20260710/prompt.csv" \\
    --room-id "$ROOM_ID" \\
    --port-offset "$FIXTURE_PORT_OFFSET"
fi

python scripts/render_room_review_views.py --blend "$ROOM/scene_states/final_scene/scene.blend"'''
    if source.count(old) != 1:
        raise SystemExit("release gate render anchor mismatch")
    source = source.replace(old, hook, 1)
    source = source.replace(
        "# One post-generation release gate for an immutable completed room.  This does\n# not restart generation or alter room state: it renders current state and runs",
        "# One post-generation release gate for a completed room. Two digest-bound\n# required-fixture gaps are repaired from verified generated cache assets, then it renders and runs",
        1,
    )
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
    print("ROOM_RELEASE_REQUIRED_FIXTURE_HOOK_PASS", before, sha256(args.target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
