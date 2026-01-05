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
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

# The VLM request uses the user's existing laptop proxy, forwarded only through
# ln07 and only on this compute node's loopback interface.
PROXY_PORT=18109
ssh -N -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$FACTORY/ln07_known_hosts" -L "127.0.0.1:${PROXY_PORT}:127.0.0.1:18093" ln07 &
TUNNEL_PID=$!
cleanup() { kill "$TUNNEL_PID" 2>/dev/null || true; }
trap cleanup EXIT
sleep 2
export HTTP_PROXY="http://127.0.0.1:${PROXY_PORT}"
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTP_PROXY"

python scripts/repair_room1_warm_finish.py --repo-dir "$EXEC" --run-dir "$RUN" --csv "$EXEC/inputs/full_quality_school_reference_20260710/prompt.csv" --room-id classroom_01 --port-offset 79
python scripts/render_room_review_views.py --blend "$ROOM/scene_states/final_scene/scene.blend" --scene-state "$ROOM/scene_states/final_scene/scene_state.json" --room-id classroom_01 --output-dir "$EXEC/review/room_review_renders"
python scripts/room_self_exam.py --scene-dir "$SCENE" --review-dir "$RUN/review/room_review_renders" --output-dir "$SCENE/quality_gates/room_self_exam_deterministic" --rooms classroom_01 --contract-profile school_reference_20260710 --prompt-binding "$SCENE/quality_gates/room_prompt_binding.json"
python scripts/pipeline_code_contract.py --repo-dir "$EXEC" --output "$EXEC/outputs/pipeline_code_contract.json"
python scripts/pipeline_code_contract.py --repo-dir "$EXEC" --output "$EXEC/outputs/pipeline_code_contract.json" --verify-only
set +e
python scripts/room_visual_self_exam.py --scene-dir "$SCENE" --deterministic-gate-dir "$SCENE/quality_gates/room_self_exam_deterministic" --review-dir "$RUN/review/room_review_renders" --reference-image "$EXEC/inputs/full_quality_school_reference_20260710/reference.png" --output-dir "$SCENE/quality_gates/room_visual_self_exam" --rooms classroom_01 --contract-profile school_reference_20260710 --effective-prompt "$EXEC/inputs/full_quality_school_reference_20260710/prompt.txt" --input-manifest "$EXEC/inputs/full_quality_school_reference_20260710/input_manifest.json" --prompt-binding "$SCENE/quality_gates/room_prompt_binding.json"
STATUS=$?
set -e
test "$STATUS" -eq 0
echo ROOM1_FINALIZE_PASS
