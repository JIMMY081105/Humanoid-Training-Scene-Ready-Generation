#!/usr/bin/env bash
# Keep the scheduler-only handoff dispatchers alive independently of the
# operator laptop. This never signals, restarts, or edits a generation job.

set -euo pipefail

readonly FACTORY=/data/run01/scvj260/codex_factory
readonly LOG="$FACTORY/logs/dispatcher_supervisor.log"
readonly LOCK="$FACTORY/.dispatcher_supervisor.lock"

mkdir -p "$FACTORY/logs"
exec 9>"$LOCK"
flock -n 9 || exit 0

ensure_dispatcher() {
  local script_name=$1
  local receipt=$2
  local script="$FACTORY/$script_name"

  test -s "$receipt" && return 0
  test -s "$script"
  if ! pgrep -f -x "bash $script" >/dev/null 2>&1; then
    printf '%s restarting %s\n' "$(date -Is)" "$script_name" >> "$LOG"
    nohup setsid bash "$script" >> "$LOG" 2>&1 </dev/null &
  fi
  return 1
}

while true; do
  remaining_done=0
  classroom04_done=0
  storage_done=0
  classroom05_done=0
  post_done=0

  ensure_dispatcher \
    submit_remaining_gate_batch_when_slot.sh \
    "$FACTORY/remaining_release_gate_batch_submission.txt" \
    && remaining_done=1 || true
  ensure_dispatcher \
    submit_classroom04_recovery_when_slot.sh \
    "$FACTORY/classroom04_gate_replacement_submission.txt" \
    && classroom04_done=1 || true
  ensure_dispatcher \
    submit_storage_recovery_when_slot.sh \
    "$FACTORY/storage_gate_replacement_submission.txt" \
    && storage_done=1 || true
  ensure_dispatcher \
    submit_classroom05_recovery_when_slot.sh \
    "$FACTORY/classroom05_gate_replacement_submission.txt" \
    && classroom05_done=1 || true
  ensure_dispatcher \
    submit_post_room_pipeline_when_ready.sh \
    "$FACTORY/post_room_pipeline_submission.txt" \
    && post_done=1 || true

  if ((
    remaining_done == 1 &&
    classroom04_done == 1 &&
    storage_done == 1 &&
    classroom05_done == 1 &&
    post_done == 1
  )); then
    printf '%s all scheduler handoffs submitted\n' "$(date -Is)" >> "$LOG"
    exit 0
  fi
  sleep 30
done
