#!/usr/bin/env bash
# Controller-side handoff: submit final assembly exactly once, only after every
# room has a current deterministic pass and image-aware visual pass. The final
# job independently rehashes all evidence before it can assemble the school.

set -euo pipefail

readonly FACTORY=/data/run01/scvj260/codex_factory
readonly SCENE="$FACTORY/room1_execution/outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710/scene_000"
readonly DETERMINISTIC="$SCENE/quality_gates/room_self_exam_deterministic"
readonly VISUAL="$SCENE/quality_gates/room_visual_self_exam"
readonly JOB="$FACTORY/paracloud_school_post_room_pipeline.sbatch"
readonly RECEIPT="$FACTORY/post_room_pipeline_submission_current.txt"
readonly LOCK="$FACTORY/.post_room_pipeline_submission_current.lock"

mkdir -p "$FACTORY/logs"
exec 9>"$LOCK"
flock -n 9 || exit 0

if [[ -s "$RECEIPT" ]]; then
  exit 0
fi
test -s "$JOB"

all_rooms_pass() {
  /usr/bin/python3 - "$DETERMINISTIC" "$VISUAL" <<'PY'
import json
import sys
from pathlib import Path

rooms = (
    "library", "storage_room", "classroom_01", "classroom_02",
    "classroom_03", "classroom_04", "classroom_05", "classroom_06",
    "boys_toilet", "girls_toilet", "main_corridor",
)
for root_text in sys.argv[1:]:
    root = Path(root_text)
    for room in rooms:
        try:
            decision = json.loads((root / f"{room}.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise SystemExit(1)
        if decision.get("room_id") != room or decision.get("status") != "pass":
            raise SystemExit(1)
PY
}

while true; do
  if all_rooms_pass; then
    if output=$(sbatch "$JOB" 2>&1); then
      temporary="$RECEIPT.tmp.$$"
      printf '%s\n' "$output" >"$temporary"
      mv -f "$temporary" "$RECEIPT"
      exit 0
    fi
    if ! printf '%s' "$output" | grep -q 'AssocMaxSubmitJobLimit'; then
      echo "post-room pipeline submission failed: $output" >&2
      exit 71
    fi
  fi
  sleep 30
done
