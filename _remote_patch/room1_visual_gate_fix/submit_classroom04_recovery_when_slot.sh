#!/usr/bin/env bash
# Replace only the dependency-dead Classroom 4 and Room 1 jobs. Existing live
# generation tasks are never signalled or restarted.

set -euo pipefail
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly RESUME_JOB="$FACTORY/paracloud_classroom04_manipuland_resume.sbatch"
readonly GATE_JOB="$FACTORY/paracloud_room_release_gate.sbatch"
readonly ROOM1_JOB="$FACTORY/paracloud_room1_finalize_after_wave.sbatch"
readonly RESUME_RECEIPT="$FACTORY/classroom04_resume_submission.txt"
readonly GATE_RECEIPT="$FACTORY/classroom04_gate_replacement_submission.txt"
readonly ROOM1_RECEIPT="$FACTORY/room1_finish_replacement_submission.txt"
readonly STARTED="$FACTORY/classroom04_recovery_dispatcher_started.txt"
readonly LOG="$FACTORY/logs/classroom04_recovery_dispatcher.log"

for job in "$RESUME_JOB" "$GATE_JOB" "$ROOM1_JOB"; do test -s "$job"; done
temporary="$STARTED.tmp.$$"
printf 'pid=%s started=%s\n' "$$" "$(date -Is)" > "$temporary"
mv -f "$temporary" "$STARTED"

submit_when_slot() {
  receipt=$1
  shift
  test -s "$receipt" && return 0
  while true; do
    if output=$(sbatch "$@" 2>&1); then
      temporary="$receipt.tmp.$$"
      printf '%s\n' "$output" > "$temporary"
      mv -f "$temporary" "$receipt"
      return 0
    fi
    if printf '%s' "$output" | grep -q 'AssocMaxSubmitJobLimit'; then
      sleep 30
      continue
    fi
    printf '%s submission failed: %s\n' "$(date -Is)" "$output" >> "$LOG"
    exit 1
  done
}

submit_when_slot "$RESUME_RECEIPT" "$RESUME_JOB"
resume_id=$(awk '/Submitted batch job/ {print $4}' "$RESUME_RECEIPT")
[[ "$resume_id" =~ ^[0-9]+$ ]]

room1_dependency="afterok:168922_0:168922_1:168922_2:168922_3:168922_5:168922_6:168922_7:$resume_id"
submit_when_slot "$ROOM1_RECEIPT" --dependency="$room1_dependency" "$ROOM1_JOB"

submit_when_slot "$GATE_RECEIPT" \
  --dependency="afterok:$resume_id" \
  --job-name=gate_classroom_04_recovery \
  --export=ALL,ROOM_ID=classroom_04,PROXY_PORT=18444 \
  "$GATE_JOB"

