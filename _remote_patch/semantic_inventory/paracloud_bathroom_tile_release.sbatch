#!/usr/bin/env bash
# Refresh the two repaired restroom blends and keep one allocation through both
# deterministic/reference-aware release gates.

#SBATCH -J bathroom_tile_release
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%j.out

set -euo pipefail
readonly EXEC=/data/run01/scvj260/codex_factory/room1_execution
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly VENV=/data/run01/scvj260/scenesmith/.venv
readonly RUN="$EXEC/outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710"

case "$(hostname)" in ln*|Ln*|lN*|LN*) echo 'Refusing login node.' >&2; exit 65;; esac
test -s "$FACTORY/bathroom_tile_material_repair_receipt.json"
source "$VENV/bin/activate"
cd "$EXEC"
set -a; source .env; set +a
export PYTHONPATH="$FACTORY/python_overlay_agents_0_17_4:$EXEC"
export CUDA_HOME=/data/run01/scvj260/cuda128_conda
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

refresh_room() {
  local room_id="$1" run_name="$2" port_offset="$3"
  python scripts/run_single_room_worker.py \
    --repo-dir "$EXEC" \
    --run-dir "$RUN" \
    --csv "$EXEC/inputs/full_quality_school_reference_20260710/prompt.csv" \
    --run-name "$run_name" \
    --room-id "$room_id" \
    --start-stage manipuland \
    --stop-stage manipuland \
    --asset-pipeline generated_sam3d \
    --port-offset "$port_offset" \
    --render-gpu-id 0 \
    --refresh-final-blend
}

refresh_room boys_toilet bathroom_tile_refresh_boys 704
refresh_room girls_toilet bathroom_tile_refresh_girls 724

export ROOM_ID=boys_toilet PROXY_PORT=19907
bash "$FACTORY/paracloud_room_release_gate.sbatch"
export ROOM_ID=girls_toilet PROXY_PORT=19908
bash "$FACTORY/paracloud_room_release_gate.sbatch"
echo BATHROOM_TILE_RELEASE_PASS
