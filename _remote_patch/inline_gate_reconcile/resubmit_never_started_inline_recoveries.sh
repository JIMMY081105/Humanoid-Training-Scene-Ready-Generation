#!/usr/bin/env bash
# Replace only never-started recovery jobs so Slurm snapshots the inline gate.

set -euo pipefail
readonly F=/data/run01/scvj260/codex_factory
readonly B="$F/backups"
readonly R="$F/inline_recovery_submissions"
readonly STAMP="$(date +%Y%m%d_%H%M%S)"
readonly OLD_RECOVERIES=(169026 169049 169061 169072 169073 169083 169104 169105)
readonly OLD_DEPENDENTS=(169123 169152 169153 169154 169158)

mkdir -p "$B" "$R"
test "$(sha256sum "$F/paracloud_manipuland_resume.sbatch" | awk '{print $1}')" = \
  5cd8183daf50a414f6ac5ebe000faa611475ca8b13e06b55ebfa0be93585a26a
test "$(sha256sum "$F/paracloud_classroom04_manipuland_resume.sbatch" | awk '{print $1}')" = \
  70c14f1dadf7efa8ea5d96a3a5039a68134a12b0c59ec04a38c9eee604b171b2
test "$(sha256sum "$F/paracloud_storage_manipuland_resume.sbatch" | awk '{print $1}')" = \
  ff6db7d57dfd36910aec30c2ee4107023a2bed93f5a0e527c76d6cfb57a8fda9
test "$(sha256sum "$F/paracloud_classroom05_manipuland_resume.sbatch" | awk '{print $1}')" = \
  289854232bb2f372acf2281d2b3d76307c120c1143b63e578d8fd619dd9c3fbe
for script in \
  paracloud_manipuland_resume.sbatch \
  paracloud_classroom04_manipuland_resume.sbatch \
  paracloud_storage_manipuland_resume.sbatch \
  paracloud_classroom05_manipuland_resume.sbatch
do
  bash -n "$F/$script"
  grep -q RECOVERY_AND_RELEASE_GATE_PASS "$F/$script"
  grep -q paracloud_room_release_gate.sbatch "$F/$script"
done

scontrol hold "${OLD_RECOVERIES[@]}"
for id in "${OLD_RECOVERIES[@]}"; do
  record=$(scontrol show job "$id" -o)
  grep -q 'JobState=PENDING' <<< "$record"
  grep -q 'RunTime=00:00:00' <<< "$record"
  if scontrol write batch_script "$id" - | grep -q RECOVERY_AND_RELEASE_GATE_PASS; then
    echo "old job unexpectedly already contains inline gate: $id" >&2
    exit 2
  fi
done

for path in \
  "$F/post_room_pipeline_submission.txt" \
  "$F/room1_finish_replacement_submission.txt" \
  "$F/classroom02_gate_replacement_submission.txt" \
  "$F/classroom03_gate_replacement_submission.txt" \
  "$F/classroom05_gate_replacement_submission.txt"
do
  if [ -s "$path" ]; then
    mv "$path" "$B/$(basename "$path").pre_inline_snapshot.$STAMP"
  fi
done
if compgen -G "$R/*.txt" >/dev/null; then
  mkdir -p "$B/inline_recovery_submissions.$STAMP"
  mv "$R"/*.txt "$B/inline_recovery_submissions.$STAMP/"
fi

scancel "${OLD_DEPENDENTS[@]}" "${OLD_RECOVERIES[@]}" 2>/dev/null || true
sleep 1
old_csv=$(IFS=,; echo "${OLD_DEPENDENTS[*]},${OLD_RECOVERIES[*]}")
if squeue -h -j "$old_csv" | grep -q .; then
  echo "old pending jobs survived cancellation" >&2
  squeue -h -j "$old_csv" >&2
  exit 2
fi

submit() {
  local room=$1
  shift
  local output
  output=$(sbatch "$@")
  printf '%s\n' "$output" > "$R/$room.txt.tmp.$$"
  mv "$R/$room.txt.tmp.$$" "$R/$room.txt"
  awk '/Submitted batch job/ {print $4}' <<< "$output"
}

id_c04=$(submit classroom_04 "$F/paracloud_classroom04_manipuland_resume.sbatch")
id_storage=$(submit storage_room "$F/paracloud_storage_manipuland_resume.sbatch")
id_c05=$(submit classroom_05 "$F/paracloud_classroom05_manipuland_resume.sbatch")
id_library=$(submit library \
  --job-name=resume_library \
  --export=ALL,ROOM_ID=library,GPU_PROXY_PORT=18610,PORT_OFFSET=410,RUN_NAME=paracloud_resume_library \
  "$F/paracloud_manipuland_resume.sbatch")
id_boys=$(submit boys_toilet \
  --job-name=resume_boys_toilet \
  --export=ALL,ROOM_ID=boys_toilet,GPU_PROXY_PORT=18617,PORT_OFFSET=417,RUN_NAME=paracloud_resume_boys_toilet \
  "$F/paracloud_manipuland_resume.sbatch")
id_c06=$(submit classroom_06 \
  --job-name=resume_classroom_06 \
  --export=ALL,ROOM_ID=classroom_06,GPU_PROXY_PORT=18616,PORT_OFFSET=416,RUN_NAME=paracloud_resume_classroom_06 \
  "$F/paracloud_manipuland_resume.sbatch")
id_c02=$(submit classroom_02 \
  --job-name=resume_classroom_02 \
  --export=ALL,ROOM_ID=classroom_02,GPU_PROXY_PORT=18612,PORT_OFFSET=412,RUN_NAME=paracloud_resume_classroom_02 \
  "$F/paracloud_manipuland_resume.sbatch")
id_c03=$(submit classroom_03 \
  --job-name=resume_classroom_03 \
  --export=ALL,ROOM_ID=classroom_03,GPU_PROXY_PORT=18613,PORT_OFFSET=413,RUN_NAME=paracloud_resume_classroom_03 \
  "$F/paracloud_manipuland_resume.sbatch")

dependency="afterok:$id_c04:$id_storage:$id_c05:$id_library:$id_boys:$id_c06:$id_c02:$id_c03"
room1_output=$(sbatch --dependency="$dependency" "$F/paracloud_room1_finalize_after_wave.sbatch")
printf '%s\n' "$room1_output" > "$F/room1_finish_replacement_submission.txt.tmp.$$"
mv "$F/room1_finish_replacement_submission.txt.tmp.$$" \
  "$F/room1_finish_replacement_submission.txt"

manifest="$F/inline_recovery_resubmission_receipt.$STAMP.txt"
{
  printf 'timestamp=%s\n' "$(date -Is)"
  printf 'old_recoveries=%s\n' "${OLD_RECOVERIES[*]}"
  printf 'old_dependents=%s\n' "${OLD_DEPENDENTS[*]}"
  printf 'classroom_04=%s\n' "$id_c04"
  printf 'storage_room=%s\n' "$id_storage"
  printf 'classroom_05=%s\n' "$id_c05"
  printf 'library=%s\n' "$id_library"
  printf 'boys_toilet=%s\n' "$id_boys"
  printf 'classroom_06=%s\n' "$id_c06"
  printf 'classroom_02=%s\n' "$id_c02"
  printf 'classroom_03=%s\n' "$id_c03"
  printf 'room1=%s\n' "$(awk '/Submitted batch job/ {print $4}' <<< "$room1_output")"
} > "$manifest"
echo "INLINE_RECOVERY_RESUBMISSION_PASS $manifest"
