#!/usr/bin/env bash
# Physical Room 1 finish, then only its release gates.  Submitted dependent on
# the eight-room production wave so the account never exceeds eight active GPUs.

#SBATCH -J codex_r1_finish
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
readonly SCENE="$RUN/scene_000"
readonly ROOM="$SCENE/room_classroom_01"

case "$(hostname)" in ln*|Ln*|lN*|LN*) echo 'Refusing login node.' >&2; exit 65;; esac
source "$VENV/bin/activate"
cd "$EXEC"
set -a; source .env; set +a
export PYTHONPATH="$FACTORY/python_overlay_agents_0_17_4:$EXEC"
export CUDA_HOME=/data/run01/scvj260/cuda128_conda
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python scripts/repair_room1_warm_finish.py --repo-dir "$EXEC" --run-dir "$RUN" --csv "$EXEC/inputs/full_quality_school_reference_20260710/prompt.csv" --room-id classroom_01 --port-offset 79

# Reuse the maintained release wrapper so Room 1 receives the same robust
# keepalive proxy, canonical review path, hull cap, and fail-closed gates as
# every other room.
export ROOM_ID=classroom_01
export PROXY_PORT=18109
bash "$FACTORY/paracloud_room_release_gate.sbatch"
echo ROOM1_FINALIZE_PASS
