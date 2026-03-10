#!/usr/bin/env bash
# Repair deterministic mesh evidence, then run every unchanged strict room gate.

#SBATCH -J repaired_evidence_gate
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%j.out

set -euo pipefail
readonly EXEC=/data/run01/scvj260/codex_factory/room1_execution
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly VENV=/data/run01/scvj260/scenesmith/.venv
readonly RUN="$EXEC/outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710"

case "$(hostname)" in ln*|Ln*|lN*|LN*) echo 'Refusing login node.' >&2; exit 65;; esac
source "$VENV/bin/activate"
cd "$EXEC"
set -a; source .env; set +a
export PYTHONPATH="$FACTORY/python_overlay_agents_0_17_4:$EXEC"
export CUDA_HOME=/data/run01/scvj260/cuda128_conda
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python scripts/repair_repaired_room_mesh_evidence.py \
  --repo-dir "$EXEC" \
  --run-dir "$RUN" \
  --csv "$EXEC/inputs/full_quality_school_reference_20260710/prompt.csv" \
  --port-offset 929

failures=0
run_gate() {
  local room_id=$1
  local proxy_port=$2
  if ROOM_ID="$room_id" PROXY_PORT="$proxy_port" \
      bash "$FACTORY/paracloud_room_release_gate.sbatch"; then
    echo "REPAIRED_EVIDENCE_GATE_PASS room=${room_id}"
  else
    rc=$?
    echo "REPAIRED_EVIDENCE_GATE_FAIL room=${room_id} rc=${rc}" >&2
    failures=$((failures + 1))
  fi
}

run_gate classroom_01 19939
run_gate boys_toilet 19927
run_gate girls_toilet 19928

if (( failures != 0 )); then
  echo "REPAIRED_EVIDENCE_GATES_FAILED count=${failures}" >&2
  exit 1
fi
echo REPAIRED_EVIDENCE_GATES_PASS
