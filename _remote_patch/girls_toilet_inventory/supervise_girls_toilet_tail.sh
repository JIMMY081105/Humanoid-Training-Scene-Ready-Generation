#!/usr/bin/env bash
set -euo pipefail

readonly F=/data/run01/scvj260/codex_factory
readonly E="$F/room1_execution"
readonly V=/data/run01/scvj260/scenesmith/.venv/bin/python
readonly RUN="$E/outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710"
readonly S="$RUN/scene_000"
readonly ROOM="$S/room_girls_toilet"
readonly STATE="$ROOM/scene_states/final_scene/scene_state.json"
readonly TAIL_JOB=168956_0
readonly GATE_JOB=169155
readonly REPAIR="$F/repair_girls_toilet_final_state.py"
readonly PLAN_REPAIR="$F/repair_girls_toilet_plan_after_checkpoint.py"
readonly RECEIPT="$F/girls_toilet_tail_supervision.txt"
readonly REPAIR_RECEIPT="$F/girls_toilet_semantic_repair_receipt.json"
readonly INVENTORY="$F/girls_toilet_inventory_after_repair.json"
readonly LOG="$F/logs/codex_room_tail-168956_0.out"
readonly RECOVERY_RECEIPT="$F/girls_toilet_recovery_submission.txt"
readonly POST_RECEIPT="$F/post_room_pipeline_submission.txt"

test -s "$REPAIR"
test ! -e "$RECEIPT"
gate=$(scontrol show job "$GATE_JOB" -o)
grep -q 'JobName=gate_girls_toilet' <<< "$gate"
grep -q 'Reason=JobHeldUser' <<< "$gate"

while squeue -h -j "$TAIL_JOB" | grep -q .; do
  sleep 20
done
state=$(sacct -j "$TAIL_JOB" -X -n -o State%30 | awk 'NF {print $1; exit}')
if [[ "$state" != COMPLETED ]] && ! {
  [[ "$state" == FAILED ]] && grep -Eq \
    'Required manipuland target hygiene_station_column_0 failed|hygiene_station_column_0.*no usable support surfaces' \
    "$LOG"
}; then
  temporary="$RECEIPT.tmp.$$"
  printf 'status=needs_diagnosis action=gate_remains_held job=%s state=%s finished=%s\n' \
    "$TAIL_JOB" "$state" "$(date -Is)" > "$temporary"
  mv -f "$temporary" "$RECEIPT"
  exit 3
fi

if [[ "$state" == FAILED ]]; then
  checkpoint="$ROOM/scene_states/manipuland_checkpoint_000_sink_vanity_0/completion_receipt.json"
  test -s "$checkpoint"
  checkpoint_sha=$(sha256sum "$checkpoint" | awk '{print $1}')
  "$V" "$PLAN_REPAIR" \
    --room-dir "$ROOM" \
    --expected-checkpoint-sha256 "$checkpoint_sha" \
    --backup-dir "$F/plan_repair_backups/girls_toilet" \
    --receipt "$F/girls_toilet_plan_repair_receipt.json"
  stage=$("$V" "$E/scripts/select_room_resume_stage.py" \
    --scene-dir "$S" \
    --room-id girls_toilet \
    --prompt-binding "$S/quality_gates/room_prompt_binding.json")
  test "$stage" = manipuland
  while true; do
    if recovery_output=$(sbatch \
      --job-name=resume_girls_toilet \
      --export=ALL,ROOM_ID=girls_toilet,GPU_PROXY_PORT=18711,PORT_OFFSET=675,RUN_NAME=paracloud_resume_girls_toilet \
      "$F/paracloud_manipuland_resume.sbatch" 2>&1); then
      break
    fi
    if grep -q 'AssocMaxSubmitJobLimit' <<< "$recovery_output"; then sleep 20; continue; fi
    echo "girls-toilet recovery submission failed: $recovery_output" >&2
    exit 4
  done
  recovery_job=$(awk '/Submitted batch job/ {print $4}' <<< "$recovery_output")
  [[ "$recovery_job" =~ ^[0-9]+$ ]]
  temporary="$RECOVERY_RECEIPT.tmp.$$"
  printf '%s\n' "$recovery_output" > "$temporary"
  mv -f "$temporary" "$RECOVERY_RECEIPT"

  if gate=$(scontrol show job "$GATE_JOB" -o 2>/dev/null); then
    grep -q 'JobName=gate_girls_toilet' <<< "$gate"
    scancel "$GATE_JOB"
  fi
  current_post=$(awk '/Submitted batch job/ {print $4}' "$POST_RECEIPT")
  [[ "$current_post" =~ ^[0-9]+$ ]]
  if post=$(scontrol show job "$current_post" -o 2>/dev/null); then
    grep -q 'JobName=codex_school_final' <<< "$post"
    scancel "$current_post"
  fi
  mv -f "$POST_RECEIPT" "$POST_RECEIPT.stale_$current_post"
  dependency="afterok:169162:169163:169164:169165:169166:169167:169168:169169:169170:169176:$recovery_job"
  while true; do
    if post_output=$(sbatch --dependency="$dependency" \
      "$F/paracloud_school_post_room_pipeline.sbatch" 2>&1); then
      break
    fi
    if grep -q 'AssocMaxSubmitJobLimit' <<< "$post_output"; then sleep 20; continue; fi
    echo "replacement post-pipeline submission failed: $post_output" >&2
    exit 5
  done
  replacement_post=$(awk '/Submitted batch job/ {print $4}' <<< "$post_output")
  [[ "$replacement_post" =~ ^[0-9]+$ ]]
  temporary="$POST_RECEIPT.tmp.$$"
  printf '%s\n' "$post_output" > "$temporary"
  mv -f "$temporary" "$POST_RECEIPT"
  temporary="$RECEIPT.tmp.$$"
  printf 'status=recovery_submitted original_job=%s recovery_job=%s checkpoint_sha256=%s replacement_post=%s finished=%s\n' \
    "$TAIL_JOB" "$recovery_job" "$checkpoint_sha" "$replacement_post" "$(date -Is)" > "$temporary"
  mv -f "$temporary" "$RECEIPT"
  exit 0
fi

test -s "$STATE"
state_sha=$(sha256sum "$STATE" | awk '{print $1}')
"$V" "$REPAIR" \
  --state "$STATE" \
  --expected-state-sha256 "$state_sha" \
  --backup-dir "$F/girls_toilet_inventory_backups" \
  --receipt "$REPAIR_RECEIPT"

PYTHONPATH="$E" "$V" - "$STATE" "$INVENTORY" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

from scripts.school_room_contract import evaluate_room_inventory

state_path = Path(sys.argv[1])
output = Path(sys.argv[2])
document = json.loads(state_path.read_text(encoding="utf-8"))
scene = state_path.parents[3]
result = evaluate_room_inventory(
    "girls_toilet",
    document,
    document["text_description"],
    expected_prompt_sha256="3cb610ef085d9b9540517f833d7d282233692e18e56724580bab824c49b6ebd5",
    asset_root=scene / "room_girls_toilet",
    require_prompt_binding=True,
)
fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
if result["status"] != "pass":
    raise SystemExit(json.dumps(result["critical_issues"], indent=2))
if result["counts"]["sinks"] < 2 or result["counts"]["mirrors"] < 2:
    raise SystemExit("sink/mirror inventory did not pass after repair")
PY

gate=$(scontrol show job "$GATE_JOB" -o)
grep -q 'JobName=gate_girls_toilet' <<< "$gate"
grep -q 'Reason=JobHeldUser' <<< "$gate"
scontrol release "$GATE_JOB"
temporary="$RECEIPT.tmp.$$"
printf 'status=repaired_and_released tail_job=%s gate_job=%s state_sha256_before=%s state_sha256_after=%s inventory_sha256=%s finished=%s\n' \
  "$TAIL_JOB" "$GATE_JOB" "$state_sha" "$(sha256sum "$STATE" | awk '{print $1}')" \
  "$(sha256sum "$INVENTORY" | awk '{print $1}')" "$(date -Is)" > "$temporary"
mv -f "$temporary" "$RECEIPT"
