#!/usr/bin/env bash
#SBATCH -J visual_completion_v2
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --array=0-6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%A_%a.out

set -euo pipefail
readonly EXEC=/data/run01/scvj260/codex_factory/room1_execution
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly VENV=/data/run01/scvj260/scenesmith/.venv
readonly RUN="$EXEC/outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710"
readonly ROOM1_BASELINE="$FACTORY/room1_baseline_ba67.json"
mkdir -p "$FACTORY/logs"
exec > >(tee -a "$FACTORY/logs/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.runtime.out") 2>&1
case "$(hostname)" in ln*|Ln*|lN*|LN*) echo 'Refusing login node.' >&2; exit 65;; esac

source "$VENV/bin/activate"
cd "$EXEC"
set -a; source .env; set +a
export PYTHONPATH="$FACTORY/python_overlay_agents_0_17_4:$EXEC/scripts:$EXEC"
export CUDA_HOME=/data/run01/scvj260/cuda128_conda
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ROOMS=(classroom_01 classroom_02 classroom_03 classroom_04 boys_toilet girls_toilet main_corridor)
PROXIES=(20130 20131 20132 20133 20134 20135 20136)
OFFSETS=(1130 1131 1132 1133 1134 1135 1136)
ROOM_ID=${ROOMS[$SLURM_ARRAY_TASK_ID]}
PROXY_PORT=${PROXIES[$SLURM_ARRAY_TASK_ID]}
PORT_OFFSET=${OFFSETS[$SLURM_ARRAY_TASK_ID]}

python "$FACTORY/repair_visual_completion_v2.py" \
  --repo-dir "$EXEC" \
  --run-dir "$RUN" \
  --csv "$EXEC/inputs/full_quality_school_reference_20260710/prompt.csv" \
  --room-id "$ROOM_ID" \
  --port-offset "$PORT_OFFSET" \
  --room1-baseline "$ROOM1_BASELINE"

readonly SCENE="$RUN/scene_000"
readonly ROOM="$SCENE/room_${ROOM_ID}"
test -s "$ROOM/scene_states/final_scene/scene_state.json"
test -s "$ROOM/scene_states/final_scene/scene.blend"

# Run the unchanged strict release gate inline from this immutable Slurm batch
# snapshot.  This avoids a second shared-storage script lookup and binds every
# repair task to the healthy 18114 login-node bridge established by the helper.
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ConnectTimeout=15 \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
  -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$FACTORY/ln07_known_hosts" \
  -L "127.0.0.1:${PROXY_PORT}:127.0.0.1:18114" ln07 &
TUNNEL_PID=$!
cleanup() { kill "$TUNNEL_PID" 2>/dev/null || true; wait "$TUNNEL_PID" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 10); do
  if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then wait "$TUNNEL_PID"; fi
  status=$(curl --connect-timeout 10 --max-time 20 -sS -o /dev/null \
    -w '%{http_code}' -x "http://127.0.0.1:${PROXY_PORT}" \
    https://api.openai.com/v1/models || true)
  [[ "$status" == 401 ]] && break
  sleep 2
done
[[ "${status:-000}" == 401 ]] || { echo "VLM proxy unavailable: ${status:-000}" >&2; exit 70; }
export HTTP_PROXY="http://127.0.0.1:${PROXY_PORT}"
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTP_PROXY"

python scripts/render_room_review_views.py \
  --blend "$ROOM/scene_states/final_scene/scene.blend" \
  --scene-state "$ROOM/scene_states/final_scene/scene_state.json" \
  --room-id "$ROOM_ID" --output-dir "$RUN/review/room_review_renders"
python scripts/room_self_exam.py \
  --scene-dir "$SCENE" --review-dir "$RUN/review/room_review_renders" \
  --output-dir "$SCENE/quality_gates/room_self_exam_deterministic" \
  --rooms "$ROOM_ID" --max-collision-hulls 32 \
  --contract-profile school_reference_20260710 \
  --prompt-binding "$SCENE/quality_gates/room_prompt_binding.json"
python scripts/room_visual_self_exam.py \
  --scene-dir "$SCENE" \
  --deterministic-gate-dir "$SCENE/quality_gates/room_self_exam_deterministic" \
  --review-dir "$RUN/review/room_review_renders" \
  --reference-image "$EXEC/inputs/full_quality_school_reference_20260710/reference.png" \
  --output-dir "$SCENE/quality_gates/room_visual_self_exam" \
  --rooms "$ROOM_ID" --minimum-review-images 3 --threshold 7 \
  --contract-profile school_reference_20260710 \
  --effective-prompt "$EXEC/inputs/full_quality_school_reference_20260710/prompt.txt" \
  --input-manifest "$EXEC/inputs/full_quality_school_reference_20260710/input_manifest.json" \
  --prompt-binding "$SCENE/quality_gates/room_prompt_binding.json"
echo "VISUAL_COMPLETION_V2_PASS room=${ROOM_ID}"
