#!/usr/bin/env bash
#SBATCH -J classroom02_plant
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:30:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%j.out
set -euo pipefail
readonly EXEC=/data/run01/scvj260/codex_factory/room1_execution
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly VENV=/data/run01/scvj260/scenesmith/.venv
readonly RUN="$EXEC/outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710"
case "$(hostname)" in ln*|Ln*|lN*|LN*) exit 65;; esac
source "$VENV/bin/activate"; cd "$EXEC"; set -a; source .env; set +a
export PYTHONPATH="$FACTORY/python_overlay_agents_0_17_4:$EXEC"
export CUDA_HOME=/data/run01/scvj260/cuda128_conda CUDA_PATH=/data/run01/scvj260/cuda128_conda
export PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python scripts/repair_classroom02_plant.py --repo-dir "$EXEC" --run-dir "$RUN" --csv "$EXEC/inputs/full_quality_school_reference_20260710/prompt.csv" --port-offset 972
ROOM_ID=classroom_02 PROXY_PORT=19972 bash "$FACTORY/paracloud_room_release_gate.sbatch"
echo CLASSROOM02_PLANT_RELEASE_PASS
