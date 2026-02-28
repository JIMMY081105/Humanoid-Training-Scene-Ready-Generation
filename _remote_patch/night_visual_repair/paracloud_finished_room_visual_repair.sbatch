#!/usr/bin/env bash
#SBATCH -J night_visual_repair
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --array=0-6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%A_%a.out

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

ROOMS=(classroom_01 classroom_02 classroom_03 classroom_04 boys_toilet girls_toilet main_corridor)
PROXIES=(20010 20011 20012 20013 20014 20015 20016)
OFFSETS=(1010 1011 1012 1013 1014 1015 1016)
ROOM_ID=${ROOMS[$SLURM_ARRAY_TASK_ID]}
PROXY_PORT=${PROXIES[$SLURM_ARRAY_TASK_ID]}
PORT_OFFSET=${OFFSETS[$SLURM_ARRAY_TASK_ID]}

python scripts/repair_finished_room_visuals.py \
  --repo-dir "$EXEC" --run-dir "$RUN" \
  --csv "$EXEC/inputs/full_quality_school_reference_20260710/prompt.csv" \
  --room-id "$ROOM_ID" --port-offset "$PORT_OFFSET"

ROOM_ID="$ROOM_ID" PROXY_PORT="$PROXY_PORT" bash "$FACTORY/paracloud_room_release_gate.sbatch"
echo "NIGHT_VISUAL_RELEASE_PASS room=${ROOM_ID}"
