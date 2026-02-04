#!/usr/bin/env bash
# Scheduler-only handoff from the eleven independent room release gates to the
# serialized global variation, assembly, simulator-export, and acceptance run.

set -euo pipefail
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly JOB="$FACTORY/paracloud_school_post_room_pipeline.sbatch"
readonly ROOM_RECEIPTS="$FACTORY/remaining_gate_submissions"
readonly RECEIPT="$FACTORY/post_room_pipeline_submission.txt"
readonly LOG="$FACTORY/logs/post_room_pipeline_dispatcher.log"
readonly STARTED="$FACTORY/post_room_pipeline_dispatcher_started.txt"
readonly CLASSROOM04_GATE_RECEIPT="$FACTORY/classroom04_gate_replacement_submission.txt"
readonly STORAGE_GATE_RECEIPT="$FACTORY/storage_gate_replacement_submission.txt"
readonly CLASSROOM05_GATE_RECEIPT="$FACTORY/classroom05_gate_replacement_submission.txt"
readonly CLASSROOM02_GATE_RECEIPT="$FACTORY/classroom02_gate_replacement_submission.txt"
readonly CLASSROOM03_GATE_RECEIPT="$FACTORY/classroom03_gate_replacement_submission.txt"
readonly LIBRARY_GATE_RECEIPT="$FACTORY/library_gate_replacement_submission.txt"
readonly BOYS_GATE_RECEIPT="$FACTORY/boys_toilet_gate_replacement_submission.txt"
readonly ROOM1_RECEIPT="$FACTORY/room1_finish_replacement_submission.txt"

test -s "$JOB"
test -s "$RECEIPT" && exit 0
temporary="$STARTED.tmp.$$"
printf 'pid=%s started=%s\n' "$$" "$(date -Is)" > "$temporary"
mv -f "$temporary" "$STARTED"

while true; do
  ready=1
  for room in classroom_06 girls_toilet main_corridor; do
    test -s "$ROOM_RECEIPTS/$room.txt" || ready=0
  done
  test -s "$CLASSROOM04_GATE_RECEIPT" || ready=0
  test -s "$STORAGE_GATE_RECEIPT" || ready=0
  test -s "$CLASSROOM05_GATE_RECEIPT" || ready=0
  test -s "$CLASSROOM02_GATE_RECEIPT" || ready=0
  test -s "$CLASSROOM03_GATE_RECEIPT" || ready=0
  test -s "$LIBRARY_GATE_RECEIPT" || ready=0
  test -s "$BOYS_GATE_RECEIPT" || ready=0
  test -s "$ROOM1_RECEIPT" || ready=0
  (( ready == 1 )) && break
  sleep 30
done

dependencies=()
for receipt in "$CLASSROOM02_GATE_RECEIPT" "$CLASSROOM03_GATE_RECEIPT" "$CLASSROOM04_GATE_RECEIPT" "$STORAGE_GATE_RECEIPT" "$CLASSROOM05_GATE_RECEIPT" "$LIBRARY_GATE_RECEIPT" "$BOYS_GATE_RECEIPT" "$ROOM1_RECEIPT"; do
  job_id=$(awk '/Submitted batch job/ {print $4}' "$receipt")
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "invalid replacement submission receipt: $receipt" >&2
    exit 2
  }
  dependencies+=("$job_id")
done
for room in classroom_06 girls_toilet main_corridor; do
  job_id=$(awk '/Submitted batch job/ {print $4}' "$ROOM_RECEIPTS/$room.txt")
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "invalid gate submission receipt for $room" >&2
    exit 2
  }
  dependencies+=("$job_id")
done
dependency_csv=$(IFS=:; echo "${dependencies[*]}")

while true; do
  if output=$(sbatch --dependency="afterok:$dependency_csv" "$JOB" 2>&1); then
    temporary="$RECEIPT.tmp.$$"
    printf '%s\n' "$output" > "$temporary"
    mv -f "$temporary" "$RECEIPT"
    exit 0
  fi
  if printf '%s' "$output" | grep -q 'AssocMaxSubmitJobLimit'; then
    sleep 30
    continue
  fi
  printf '%s submission failed: %s\n' "$(date -Is)" "$output" >> "$LOG"
  exit 1
done
