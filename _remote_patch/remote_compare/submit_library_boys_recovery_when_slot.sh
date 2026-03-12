#!/usr/bin/env bash
# Replace dependency-dead Library/Boys gates and Room 1 handoff without
# signalling any healthy generation worker.

set -euo pipefail
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly RESUME_JOB="$FACTORY/paracloud_manipuland_resume.sbatch"
readonly GATE_JOB="$FACTORY/paracloud_room_release_gate.sbatch"
readonly ROOM1_JOB="$FACTORY/paracloud_room1_finalize_after_wave.sbatch"
readonly LOG="$FACTORY/logs/library_boys_recovery_dispatcher.log"
readonly STARTED="$FACTORY/library_boys_recovery_dispatcher_started.txt"
readonly LIB_RESUME="$FACTORY/library_resume_replacement_submission.txt"
readonly BOYS_RESUME="$FACTORY/boys_resume_replacement_submission.txt"
readonly LIB_GATE="$FACTORY/library_gate_replacement_submission.txt"
readonly BOYS_GATE="$FACTORY/remaining_gate_submissions/boys_toilet.txt"
readonly ROOM1_RECEIPT="$FACTORY/room1_finish_replacement_submission.txt"

for job in "$RESUME_JOB" "$GATE_JOB" "$ROOM1_JOB"; do test -s "$job"; done
temporary="$STARTED.tmp.$$"
printf 'pid=%s started=%s\n' "$$" "$(date -Is)" > "$temporary"
mv -f "$temporary" "$STARTED"

submit_when_slot() {
  local receipt=$1
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

submit_when_slot "$LIB_RESUME" \
  --job-name=resume_library \
  --export=ALL,ROOM_ID=library,PROXY_PORT=18700,PORT_OFFSET=700 \
  "$RESUME_JOB"
submit_when_slot "$BOYS_RESUME" \
  --job-name=resume_boys_toilet \
  --export=ALL,ROOM_ID=boys_toilet,PROXY_PORT=18707,PORT_OFFSET=707 \
  "$RESUME_JOB"

library_id=$(awk '/Submitted batch job/ {print $4}' "$LIB_RESUME")
boys_id=$(awk '/Submitted batch job/ {print $4}' "$BOYS_RESUME")
[[ "$library_id" =~ ^[0-9]+$ && "$boys_id" =~ ^[0-9]+$ ]]

submit_when_slot "$LIB_GATE" \
  --dependency="afterok:$library_id" \
  --job-name=gate_library_recovery \
  --export=ALL,ROOM_ID=library,PROXY_PORT=18740 \
  "$GATE_JOB"
submit_when_slot "$BOYS_GATE" \
  --dependency="afterok:$boys_id" \
  --job-name=gate_boys_recovery \
  --export=ALL,ROOM_ID=boys_toilet,PROXY_PORT=18747 \
  "$GATE_JOB"

room1_dependency="afterok:168922_2:168922_3:168922_6:169026:169049:169061:$library_id:$boys_id"
submit_when_slot "$ROOM1_RECEIPT" \
  --dependency="$room1_dependency" \
  "$ROOM1_JOB"

echo "LIBRARY_BOYS_RECOVERY_CHAIN_SUBMITTED"
