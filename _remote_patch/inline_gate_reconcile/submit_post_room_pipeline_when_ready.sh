#!/usr/bin/env bash
# Submit global assembly after eight inline recovery+gate jobs, both tail-room
# gates, and the independent Room 1 finalizer. Generation is never restarted.

set -euo pipefail
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly JOB="$FACTORY/paracloud_school_post_room_pipeline.sbatch"
readonly ROOM_RECEIPTS="$FACTORY/remaining_gate_submissions"
readonly RECEIPT="$FACTORY/post_room_pipeline_submission.txt"
readonly LOG="$FACTORY/logs/post_room_pipeline_dispatcher.log"
readonly STARTED="$FACTORY/post_room_pipeline_dispatcher_started.txt"
readonly ROOM1_RECEIPT="$FACTORY/room1_finish_replacement_submission.txt"
readonly RELEASE_GATE="$FACTORY/paracloud_room_release_gate.sbatch"
readonly RECOVERY_RECEIPTS="$FACTORY/inline_recovery_submissions"

readonly -a RECOVERY_ROOMS=(
  'classroom_04|paracloud_classroom04_manipuland_resume.sbatch'
  'storage_room|paracloud_storage_manipuland_resume.sbatch'
  'classroom_05|paracloud_classroom05_manipuland_resume.sbatch'
  'library|paracloud_manipuland_resume.sbatch'
  'boys_toilet|paracloud_manipuland_resume.sbatch'
  'classroom_06|paracloud_manipuland_resume.sbatch'
  'classroom_02|paracloud_manipuland_resume.sbatch'
  'classroom_03|paracloud_manipuland_resume.sbatch'
)

test -s "$JOB"
test -s "$RELEASE_GATE"
test -s "$RECEIPT" && exit 0
temporary="$STARTED.tmp.$$"
printf 'pid=%s started=%s mode=inline_recovery_gates\n' "$$" "$(date -Is)" > "$temporary"
mv -f "$temporary" "$STARTED"

# Verify the exact batch snapshots held by Slurm execute the release gate before
# exit. Checking only the live file is insufficient because sbatch snapshots it.
dependencies=()
for entry in "${RECOVERY_ROOMS[@]}"; do
  IFS='|' read -r room_id script_name <<< "$entry"
  receipt="$RECOVERY_RECEIPTS/$room_id.txt"
  test -s "$receipt"
  job_id=$(awk '/Submitted batch job/ {print $4}' "$receipt")
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "invalid inline-recovery receipt for $room_id" >&2
    exit 2
  }
  script="$FACTORY/$script_name"
  test -s "$script"
  job_record=$(scontrol show job "$job_id" -o)
  grep -q "JobId=$job_id" <<< "$job_record"
  grep -q "Command=$script" <<< "$job_record"
  batch_script=$(scontrol write batch_script "$job_id" -)
  grep -q 'RECOVERY_AND_RELEASE_GATE_PASS' <<< "$batch_script"
  grep -q 'paracloud_room_release_gate.sbatch' <<< "$batch_script"
  dependencies+=("$job_id")
  printf '%s recovery_job=%s script=%s\n' "$room_id" "$job_id" "$script_name" >> "$LOG"
done

# The two currently running tail rooms retain their existing dependent release
# jobs. Wait only for their submission receipts, not for execution.
while true; do
  ready=1
  for room in girls_toilet main_corridor; do
    test -s "$ROOM_RECEIPTS/$room.txt" || ready=0
  done
  test -s "$ROOM1_RECEIPT" || ready=0
  (( ready == 1 )) && break
  sleep 30
done

for room in girls_toilet main_corridor; do
  job_id=$(awk '/Submitted batch job/ {print $4}' "$ROOM_RECEIPTS/$room.txt")
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "invalid gate submission receipt for $room" >&2
    exit 2
  }
  dependencies+=("$job_id")
done
room1_job_id=$(awk '/Submitted batch job/ {print $4}' "$ROOM1_RECEIPT")
[[ "$room1_job_id" =~ ^[0-9]+$ ]] || {
  echo "invalid Room 1 finalizer receipt" >&2
  exit 2
}
dependencies+=("$room1_job_id")
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
