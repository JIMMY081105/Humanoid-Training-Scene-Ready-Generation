#!/usr/bin/env bash
# Validate already-published Room 1 and restroom repairs without reapplying them.

#SBATCH -J repaired_room_gates
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%j.out

set -uo pipefail
readonly FACTORY=/data/run01/scvj260/codex_factory
failures=0

run_gate() {
  local room_id=$1
  local proxy_port=$2
  echo "BEGIN_REPAIRED_ROOM_GATE room=${room_id}"
  if ROOM_ID="$room_id" PROXY_PORT="$proxy_port" \
      bash "$FACTORY/paracloud_room_release_gate.sbatch"; then
    echo "REPAIRED_ROOM_GATE_PASS room=${room_id}"
  else
    rc=$?
    echo "REPAIRED_ROOM_GATE_FAIL room=${room_id} rc=${rc}" >&2
    failures=$((failures + 1))
  fi
}

run_gate classroom_01 19939
run_gate boys_toilet 19927
run_gate girls_toilet 19928

if (( failures != 0 )); then
  echo "REPAIRED_ROOM_GATES_FAILED count=${failures}" >&2
  exit 1
fi
echo REPAIRED_ROOM_GATES_PASS
