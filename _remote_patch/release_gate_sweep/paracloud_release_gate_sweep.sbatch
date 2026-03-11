#!/usr/bin/env bash
# Strict gate-only sweep for checkpoint-complete rooms after renderer/API recovery.
#SBATCH -J room_gate_sweep
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --array=0-6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:30:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%A_%a.out
set -euo pipefail
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly EXEC="$FACTORY/room1_execution"
readonly VENV=/data/run01/scvj260/scenesmith/.venv
case "$(hostname)" in ln*|Ln*|lN*|LN*) exit 65;; esac
source "$VENV/bin/activate"; cd "$EXEC"; set -a; source .env; set +a
export PYTHONPATH="$FACTORY/python_overlay_agents_0_17_4:$EXEC"
export CUDA_HOME=/data/run01/scvj260/cuda128_conda CUDA_PATH=/data/run01/scvj260/cuda128_conda
export PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ROOMS=(classroom_01 classroom_02 classroom_03 classroom_04 boys_toilet girls_toilet main_corridor)
PORTS=(19960 19961 19962 19963 19964 19965 19966)
ROOM_ID=${ROOMS[$SLURM_ARRAY_TASK_ID]}
PROXY_PORT=${PORTS[$SLURM_ARRAY_TASK_ID]}
ROOM_ID="$ROOM_ID" PROXY_PORT="$PROXY_PORT" bash "$FACTORY/paracloud_room_release_gate.sbatch"
echo "ROOM_GATE_SWEEP_PASS room=${ROOM_ID}"
