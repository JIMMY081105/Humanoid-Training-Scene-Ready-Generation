#!/usr/bin/env bash
# One post-generation release gate for an immutable completed room.  This does
# not restart generation or alter room state: it renders current state and runs
# the contract's deterministic and reference-aware visual gates exactly once.

#SBATCH -J codex_room_gate
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%j.out

set -euo pipefail
: "${ROOM_ID:?ROOM_ID must be exported at submission}"
: "${PROXY_PORT:?PROXY_PORT must be exported at submission}"
readonly EXEC=/data/run01/scvj260/codex_factory/room1_execution
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly VENV=/data/run01/scvj260/scenesmith/.venv
readonly RUN="$EXEC/outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710"
readonly SCENE="$RUN/scene_000"
readonly ROOM="$SCENE/room_${ROOM_ID}"

case "$(hostname)" in ln*|Ln*|lN*|LN*) echo 'Refusing login node.' >&2; exit 65;; esac
test -s "$ROOM/scene_states/final_scene/scene_state.json"
test -s "$ROOM/scene_states/final_scene/scene.blend"
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

ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ConnectTimeout=15 \
  -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$FACTORY/ln07_known_hosts" \
  -L "127.0.0.1:${PROXY_PORT}:127.0.0.1:18093" ln07 &
TUNNEL_PID=$!
cleanup() { kill "$TUNNEL_PID" 2>/dev/null || true; wait "$TUNNEL_PID" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 10); do
  if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then wait "$TUNNEL_PID"; fi
  status=$(curl --connect-timeout 10 --max-time 20 -sS -o /dev/null -w '%{http_code}' -x "http://127.0.0.1:${PROXY_PORT}" https://api.openai.com/v1/models || true)
  [ "$status" = 401 ] && break
  sleep 2
done
[ "${status:-000}" = 401 ] || { echo "VLM proxy unavailable: ${status:-000}" >&2; exit 70; }
export HTTP_PROXY="http://127.0.0.1:${PROXY_PORT}"
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTP_PROXY"

python scripts/render_room_review_views.py --blend "$ROOM/scene_states/final_scene/scene.blend" --scene-state "$ROOM/scene_states/final_scene/scene_state.json" --room-id "$ROOM_ID" --output-dir "$RUN/review/room_review_renders"
python scripts/room_self_exam.py --scene-dir "$SCENE" --review-dir "$RUN/review/room_review_renders" --output-dir "$SCENE/quality_gates/room_self_exam_deterministic" --rooms "$ROOM_ID" --max-collision-hulls 32 --contract-profile school_reference_20260710 --prompt-binding "$SCENE/quality_gates/room_prompt_binding.json"
python scripts/room_visual_self_exam.py --scene-dir "$SCENE" --deterministic-gate-dir "$SCENE/quality_gates/room_self_exam_deterministic" --review-dir "$RUN/review/room_review_renders" --reference-image "$EXEC/inputs/full_quality_school_reference_20260710/reference.png" --output-dir "$SCENE/quality_gates/room_visual_self_exam" --rooms "$ROOM_ID" --minimum-review-images 3 --threshold 7 --contract-profile school_reference_20260710 --effective-prompt "$EXEC/inputs/full_quality_school_reference_20260710/prompt.txt" --input-manifest "$EXEC/inputs/full_quality_school_reference_20260710/input_manifest.json" --prompt-binding "$SCENE/quality_gates/room_prompt_binding.json"
echo ROOM_RELEASE_GATE_PASS "$ROOM_ID"
