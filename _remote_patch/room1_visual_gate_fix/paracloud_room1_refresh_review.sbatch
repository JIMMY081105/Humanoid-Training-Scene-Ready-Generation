#!/usr/bin/env bash
# Derive fresh Room 1 review evidence and re-run its deterministic gate from
# the published repaired state.  This job performs no generation and no scene
# mutation.

#SBATCH -J codex_r1_review
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%j.out

set -euo pipefail
readonly EXEC=/data/run01/scvj260/codex_factory/room1_execution
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly VENV=/data/run01/scvj260/scenesmith/.venv
readonly RUN="$EXEC/outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710"
readonly ROOM="$RUN/scene_000/room_classroom_01"

source "$VENV/bin/activate"
cd "$EXEC"
export PYTHONPATH="$FACTORY/python_overlay_agents_0_17_4:$EXEC"
export CUDA_HOME=/data/run01/scvj260/cuda128_conda
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

python scripts/render_room_review_views.py --blend "$ROOM/scene_states/final_scene/scene.blend" --scene-state "$ROOM/scene_states/final_scene/scene_state.json" --room-id classroom_01 --output-dir "$EXEC/review/room_review_renders"
python scripts/room_self_exam.py --run-dir "$RUN" --room-id classroom_01 --csv "$EXEC/inputs/full_quality_school_reference_20260710/prompt.csv" --effective-prompt "$EXEC/inputs/full_quality_school_reference_20260710/prompt.txt" --prompt-binding "$EXEC/quality_gates/room_prompt_binding.json" --output-dir "$EXEC/quality_gates/room_self_exam_deterministic"
echo ROOM1_REVIEW_REFRESH_PASS
