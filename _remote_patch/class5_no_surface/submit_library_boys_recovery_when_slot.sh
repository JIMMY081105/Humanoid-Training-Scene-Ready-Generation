#!/usr/bin/env bash
# Replace only the dependency-dead Library, Boys Toilet, and Room 1 jobs.
# Existing live generation tasks are never signalled or restarted.

set -euo pipefail
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly RESUME_JOB="$FACTORY/paracloud_manipuland_resume.sbatch"
readonly GATE_JOB="$FACTORY/paracloud_room_release_gate.sbatch"
readonly ROOM1_JOB="$FACTORY/paracloud_room1_finalize_after_wave.sbatch"
readonly LIBRARY_RESUME="$FACTORY/library_resume_submission.txt"
readonly BOYS_RESUME="$FACTORY/boys_toilet_resume_submission.txt"
readonly LIBRARY_GATE="$FACTORY/library_gate_replacement_submission.txt"
readonly BOYS_GATE="$FACTORY/boys_toilet_gate_replacement_submission.txt"
readonly ROOM1_RECEIPT="$FACTORY/room1_finish_replacement_submission.txt"
readonly COMPLETE="$FACTORY/library_boys_recovery_submissions_complete.txt"
readonly STARTED="$FACTORY/library_boys_recovery_dispatcher_started.txt"
readonly LOG="$FACTORY/logs/library_boys_recovery_dispatcher.log"

for job in "$RESUME_JOB" "$GATE_JOB" "$ROOM1_JOB"; do test -s "$job"; done
test -s "$COMPLETE" && exit 0
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

submit_when_slot "$LIBRARY_RESUME" \
  --job-name=resume_library \
  --export=ALL,ROOM_ID=library,GPU_PROXY_PORT=18610,PORT_OFFSET=410,RUN_NAME=paracloud_resume_library \
  "$RESUME_JOB"
library_id=$(awk '/Submitted batch job/ {print $4}' "$LIBRARY_RESUME")
[[ "$library_id" =~ ^[0-9]+$ ]]

submit_when_slot "$BOYS_RESUME" \
  --job-name=resume_boys_toilet \
  --export=ALL,ROOM_ID=boys_toilet,GPU_PROXY_PORT=18617,PORT_OFFSET=417,RUN_NAME=paracloud_resume_boys_toilet \
  "$RESUME_JOB"
boys_id=$(awk '/Submitted batch job/ {print $4}' "$BOYS_RESUME")
[[ "$boys_id" =~ ^[0-9]+$ ]]

room1_dependency="afterok:168922_2:168922_3:168922_6:169026:169049:169061:$library_id:$boys_id"
submit_when_slot "$ROOM1_RECEIPT" --dependency="$room1_dependency" "$ROOM1_JOB"

submit_when_slot "$LIBRARY_GATE" \
  --dependency="afterok:$library_id" \
  --job-name=gate_library_recovery \
  --export=ALL,ROOM_ID=library,PROXY_PORT=18410 \
  "$GATE_JOB"

submit_when_slot "$BOYS_GATE" \
  --dependency="afterok:$boys_id" \
  --job-name=gate_boys_toilet_recovery \
  --export=ALL,ROOM_ID=boys_toilet,PROXY_PORT=18417 \
  "$GATE_JOB"

temporary="$COMPLETE.tmp.$$"
printf 'library_resume=%s boys_resume=%s completed=%s\n' \
  "$library_id" "$boys_id" "$(date -Is)" > "$temporary"
mv -f "$temporary" "$COMPLETE"
