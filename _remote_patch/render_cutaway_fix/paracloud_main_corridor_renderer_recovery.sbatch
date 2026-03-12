#!/usr/bin/env bash
# Rerun the unchanged main-corridor release gate after strict wall recovery fix.
#SBATCH -J corridor_cutaway_fix
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%j.out
set -euo pipefail
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly EXEC="$FACTORY/room1_execution"
readonly VENV=/data/run01/scvj260/scenesmith/.venv
case "$(hostname)" in ln*|Ln*|lN*|LN*) exit 65;; esac
source "$VENV/bin/activate"; cd "$EXEC"; set -a; source .env; set +a
export PYTHONPATH="$FACTORY/python_overlay_agents_0_17_4:$EXEC"
export CUDA_HOME=/data/run01/scvj260/cuda128_conda CUDA_PATH=/data/run01/scvj260/cuda128_conda
export PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m py_compile scripts/render_room_review_views.py
ROOM_ID=main_corridor PROXY_PORT=19959 bash "$FACTORY/paracloud_room_release_gate.sbatch"
echo MAIN_CORRIDOR_CUTAWAY_RELEASE_PASS
