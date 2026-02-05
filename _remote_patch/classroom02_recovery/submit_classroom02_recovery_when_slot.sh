#!/usr/bin/env bash
# Submit only Classroom 2's manipuland resume and replacement release gate.

set -euo pipefail
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly RESUME_JOB="$FACTORY/paracloud_manipuland_resume.sbatch"
readonly GATE_JOB="$FACTORY/paracloud_room_release_gate.sbatch"
readonly RESUME_RECEIPT="$FACTORY/classroom02_resume_replacement_submission.txt"
readonly GATE_RECEIPT="$FACTORY/classroom02_gate_replacement_submission.txt"
readonly STARTED="$FACTORY/classroom02_recovery_dispatcher_started.txt"
readonly LOG="$FACTORY/logs/classroom02_recovery_dispatcher.log"

test -s "$RESUME_JOB"
test -s "$GATE_JOB"
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

submit_when_slot "$RESUME_RECEIPT" \
  --job-name=resume_classroom_02 \
  --export=ALL,ROOM_ID=classroom_02,GPU_PROXY_PORT=18612,PORT_OFFSET=412,RUN_NAME=paracloud_resume_classroom_02 \
  "$RESUME_JOB"
resume_id=$(awk '/Submitted batch job/ {print $4}' "$RESUME_RECEIPT")
[[ "$resume_id" =~ ^[0-9]+$ ]]

submit_when_slot "$GATE_RECEIPT" \
  --dependency="afterok:$resume_id" \
  --job-name=gate_classroom_02_recovery \
  --export=ALL,ROOM_ID=classroom_02,PROXY_PORT=18742 \
  "$GATE_JOB"

echo CLASSROOM02_RECOVERY_CHAIN_SUBMITTED
