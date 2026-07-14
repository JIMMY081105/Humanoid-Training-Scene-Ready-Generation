#!/usr/bin/env bash
# Compact release-gate batch for rooms whose worker submissions were already
# live when the scheduler's submit cap was reached.  It never re-runs a room
# worker: each child invocation renders and gates the completed final state.

#SBATCH -J codex_remaining_gates
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%j.out

set -euo pipefail
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly GATE="$FACTORY/paracloud_room_release_gate.sbatch"
test -x "$GATE"

for entry in \
  'classroom_05|18405' \
  'classroom_06|18406' \
  'boys_toilet|18407' \
  'girls_toilet|18408' \
  'main_corridor|18409'
do
  export ROOM_ID="$(echo "$entry" | cut -d'|' -f1)"
  export PROXY_PORT="$(echo "$entry" | cut -d'|' -f2)"
  bash "$GATE"
done

echo REMAINING_ROOM_RELEASE_GATES_PASS
