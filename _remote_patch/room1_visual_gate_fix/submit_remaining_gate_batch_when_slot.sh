#!/usr/bin/env bash
# Submit each remaining room gate as soon as ParaCloud's per-user submission
# limit releases a slot. This keeps independent gates parallel without ever
# signalling, restarting, modifying, or cancelling a generation task.

set -euo pipefail
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly GATE_JOB="$FACTORY/paracloud_room_release_gate.sbatch"
readonly RECEIPT="$FACTORY/remaining_release_gate_batch_submission.txt"
readonly LOG="$FACTORY/logs/remaining_release_gate_dispatcher.log"
readonly RECEIPT_DIR="$FACTORY/remaining_gate_submissions"

test -s "$GATE_JOB"
if test -s "$RECEIPT"; then
  exit 0
fi
mkdir -p "$RECEIPT_DIR"

for entry in \
  'classroom_05|18405|168922_5' \
  'classroom_06|18406|168922_6' \
  'boys_toilet|18407|168922_7' \
  'girls_toilet|18408|168956_0' \
  'main_corridor|18409|168956_1'
do
  IFS='|' read -r room_id proxy_port dependency <<< "$entry"
  room_receipt="$RECEIPT_DIR/$room_id.txt"
  test -s "$room_receipt" && continue

  while true; do
    if output=$(sbatch \
      --dependency="afterok:$dependency" \
      --job-name="gate_$room_id" \
      --export="ALL,ROOM_ID=$room_id,PROXY_PORT=$proxy_port" \
      "$GATE_JOB" 2>&1)
    then
      temporary="$room_receipt.tmp.$$"
      printf '%s\n' "$output" > "$temporary"
      mv -f "$temporary" "$room_receipt"
      break
    fi
    if printf '%s' "$output" | grep -q 'AssocMaxSubmitJobLimit'; then
      sleep 30
      continue
    fi
    printf '%s submission failed for %s: %s\n' \
      "$(date -Is)" "$room_id" "$output" >> "$LOG"
    exit 1
  done
done

temporary="$RECEIPT.tmp.$$"
for entry in "$RECEIPT_DIR"/*.txt; do
  printf '%s %s\n' "$(basename "$entry" .txt)" "$(cat "$entry")"
done > "$temporary"
mv -f "$temporary" "$RECEIPT"
