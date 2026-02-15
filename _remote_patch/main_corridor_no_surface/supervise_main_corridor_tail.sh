#!/usr/bin/env bash
# Recover only the known optional bulletin-board zero-surface failure. Any other
# terminal state is left untouched and recorded for operator diagnosis.

set -euo pipefail
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly EXEC="$FACTORY/room1_execution"
readonly VENV=/data/run01/scvj260/scenesmith/.venv
readonly RUN="$EXEC/outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710"
readonly SCENE="$RUN/scene_000"
readonly ROOM="$SCENE/room_main_corridor"
readonly TAIL_JOB=168956_1
readonly STALE_GATE_JOB=169156
readonly STALE_POST_JOB=169171
readonly LOG="$FACTORY/logs/codex_room_tail-168956_1.out"
readonly REPAIR="$FACTORY/repair_main_corridor_plan_after_checkpoint.py"
readonly RECEIPT="$FACTORY/main_corridor_tail_supervision.txt"
readonly RECOVERY_RECEIPT="$FACTORY/main_corridor_recovery_submission.txt"
readonly POST_RECEIPT="$FACTORY/post_room_pipeline_submission.txt"

test -s "$REPAIR"
test -s "$LOG"
test -s "$FACTORY/paracloud_manipuland_resume.sbatch"
test -s "$FACTORY/paracloud_school_post_room_pipeline.sbatch"
test ! -e "$RECEIPT"

while squeue -h -j "$TAIL_JOB" | grep -q .; do
  sleep 30
done
state=$(sacct -j "$TAIL_JOB" -X -n -o State%30 | awk 'NF {print $1; exit}')
if [[ "$state" == COMPLETED ]]; then
  temporary="$RECEIPT.tmp.$$"
  printf 'status=pass action=none job=%s state=%s finished=%s\n' \
    "$TAIL_JOB" "$state" "$(date -Is)" > "$temporary"
  mv -f "$temporary" "$RECEIPT"
  exit 0
fi

if [[ "$state" != FAILED ]] || ! grep -Eq \
  'Required manipuland target bulletin_board_[01] failed|bulletin_board_[01].*no usable support surfaces' \
  "$LOG"; then
  temporary="$RECEIPT.tmp.$$"
  printf 'status=needs_diagnosis action=none job=%s state=%s finished=%s\n' \
    "$TAIL_JOB" "$state" "$(date -Is)" > "$temporary"
  mv -f "$temporary" "$RECEIPT"
  exit 3
fi

checkpoint="$ROOM/scene_states/manipuland_checkpoint_000_console_table_0/completion_receipt.json"
test -s "$checkpoint"
checkpoint_sha=$(sha256sum "$checkpoint" | awk '{print $1}')
"$VENV/bin/python" "$REPAIR" \
  --room-dir "$ROOM" \
  --expected-checkpoint-sha256 "$checkpoint_sha" \
  --backup-dir "$FACTORY/plan_repair_backups/main_corridor" \
  --receipt "$FACTORY/main_corridor_plan_repair_receipt.json"
stage=$("$VENV/bin/python" "$EXEC/scripts/select_room_resume_stage.py" \
  --scene-dir "$SCENE" \
  --room-id main_corridor \
  --prompt-binding "$SCENE/quality_gates/room_prompt_binding.json")
test "$stage" = manipuland

while true; do
  if output=$(sbatch \
    --job-name=resume_main_corridor \
    --export=ALL,ROOM_ID=main_corridor,GPU_PROXY_PORT=18709,PORT_OFFSET=650,RUN_NAME=paracloud_resume_main_corridor \
    "$FACTORY/paracloud_manipuland_resume.sbatch" 2>&1); then
    break
  fi
  if printf '%s' "$output" | grep -q 'AssocMaxSubmitJobLimit'; then
    sleep 30
    continue
  fi
  echo "main-corridor recovery submission failed: $output" >&2
  exit 4
done
recovery_job=$(awk '/Submitted batch job/ {print $4}' <<< "$output")
[[ "$recovery_job" =~ ^[0-9]+$ ]]
temporary="$RECOVERY_RECEIPT.tmp.$$"
printf '%s\n' "$output" > "$temporary"
mv -f "$temporary" "$RECOVERY_RECEIPT"

# Replace only the two dependencies invalidated by the exact tail failure.
if stale_gate=$(scontrol show job "$STALE_GATE_JOB" -o 2>/dev/null); then
  grep -q 'JobName=gate_main_corridor' <<< "$stale_gate"
  scancel "$STALE_GATE_JOB"
fi
stale_post=$(scontrol show job "$STALE_POST_JOB" -o)
grep -q 'JobName=codex_school_final' <<< "$stale_post"
scancel "$STALE_POST_JOB"
if test -s "$POST_RECEIPT"; then
  mv -f "$POST_RECEIPT" "$POST_RECEIPT.stale_$STALE_POST_JOB"
fi

dependency="afterok:169162:169163:169164:169165:169166:169167:169168:169169:169155:169170:$recovery_job"
while true; do
  if post_output=$(sbatch --dependency="$dependency" \
    "$FACTORY/paracloud_school_post_room_pipeline.sbatch" 2>&1); then
    break
  fi
  if printf '%s' "$post_output" | grep -q 'AssocMaxSubmitJobLimit'; then
    sleep 30
    continue
  fi
  echo "replacement post-pipeline submission failed: $post_output" >&2
  exit 5
done
temporary="$POST_RECEIPT.tmp.$$"
printf '%s\n' "$post_output" > "$temporary"
mv -f "$temporary" "$POST_RECEIPT"

temporary="$RECEIPT.tmp.$$"
printf 'status=recovered original_job=%s recovery_job=%s checkpoint_sha256=%s replacement_post=%s finished=%s\n' \
  "$TAIL_JOB" "$recovery_job" "$checkpoint_sha" \
  "$(awk '/Submitted batch job/ {print $4}' <<< "$post_output")" "$(date -Is)" \
  > "$temporary"
mv -f "$temporary" "$RECEIPT"
