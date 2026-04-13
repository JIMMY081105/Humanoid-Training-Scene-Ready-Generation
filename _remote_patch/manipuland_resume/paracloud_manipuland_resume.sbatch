#!/usr/bin/env bash
# Generic stage-only recovery for a room whose furniture, wall, and ceiling
# checkpoints are already prompt-bound and valid.

#SBATCH -J resume_manipuland
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 05:00:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%j.out

set -euo pipefail
readonly EXEC=/data/run01/scvj260/codex_factory/room1_execution
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly VENV=/data/run01/scvj260/scenesmith/.venv
readonly RUN="$EXEC/outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710"
readonly SCENE="$RUN/scene_000"
: "${ROOM_ID:?ROOM_ID is required}"
: "${GPU_PROXY_PORT:?GPU_PROXY_PORT is required}"
: "${PORT_OFFSET:?PORT_OFFSET is required}"
: "${RUN_NAME:?RUN_NAME is required}"
readonly ROOM_ID GPU_PROXY_PORT PORT_OFFSET RUN_NAME

mkdir -p "$FACTORY/logs"
exec > >(tee -a "$FACTORY/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.runtime.out") 2>&1

case "$(hostname)" in
  ln*|Ln*|lN*|LN*) echo 'Refusing login node.' >&2; exit 65 ;;
esac
test -d "$SCENE/floor_plans/$ROOM_ID"
test ! -L "$SCENE/floor_plans"
test -d "$SCENE/room_geometry"
test ! -L "$SCENE/room_geometry"
test -s "$SCENE/room_$ROOM_ID/scene_states/scene_after_ceiling_objects/scene_state.json"
test "$("$VENV/bin/python" "$EXEC/scripts/select_room_resume_stage.py" \
  --scene-dir "$SCENE" \
  --room-id "$ROOM_ID" \
  --prompt-binding "$SCENE/quality_gates/room_prompt_binding.json")" = manipuland

# Rebinding imports the production worker/config stack, so establish the same
# checked-out code path and strip user/session proxy variables before that
# import.  A SOCKS-valued ALL_PROXY makes httpx import fail before the job can
# create its own supported HTTP tunnel; installing socksio would hide the bad
# launch order and is intentionally not required by this pipeline.
cd "$EXEC"
export PYTHONPATH="$FACTORY/python_overlay_agents_0_17_4:$EXEC"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

if [[ "${REBIND_PLAN:-0}" == "1" ]]; then
  "$VENV/bin/python" "$FACTORY/rebind_one_manipuland_plan.py" \
    --repo-dir "$EXEC" \
    --run-dir "$RUN" \
    --room-id "$ROOM_ID" \
    --run-name "$RUN_NAME" \
    --port-offset "$PORT_OFFSET" \
    --backup-dir "$FACTORY/manipuland_plan_rebind_backups"
fi

source "$VENV/bin/activate"
set -a
source .env
set +a
export HF_HOME="$FACTORY/huggingface"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TORCH_HOME="$FACTORY/torch"
export CUDA_HOME=/data/run01/scvj260/cuda128_conda
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"

if ss -ltn "sport = :${GPU_PROXY_PORT}" | grep -q 'LISTEN'; then
  echo "GPU proxy local port is already occupied: ${GPU_PROXY_PORT}" >&2
  exit 70
fi

ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ConnectTimeout=15 \
  -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$FACTORY/ln07_known_hosts" \
  -L "127.0.0.1:${GPU_PROXY_PORT}:127.0.0.1:18114" ln07 &
GPU_PROXY_PID=$!
cleanup() {
  kill "$GPU_PROXY_PID" 2>/dev/null || true
  wait "$GPU_PROXY_PID" 2>/dev/null || true
}
trap cleanup EXIT
for _ in $(seq 1 10); do
  if ! kill -0 "$GPU_PROXY_PID" 2>/dev/null; then
    wait "$GPU_PROXY_PID" 2>/dev/null || true
    echo "GPU proxy tunnel exited before authentication" >&2
    exit 70
  fi
  status=$(curl --connect-timeout 10 --max-time 20 -sS -o /dev/null \
    -w '%{http_code}' -x "http://127.0.0.1:${GPU_PROXY_PORT}" \
    https://api.openai.com/v1/models || true)
  [[ "$status" == 401 ]] && break
  sleep 2
done
[[ "${status:-000}" == 401 ]] || { echo "GPU proxy unavailable: ${status:-000}" >&2; exit 70; }
export HTTP_PROXY="http://127.0.0.1:${GPU_PROXY_PORT}"
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTP_PROXY"

runtime_log="$FACTORY/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.runtime.out"
for attempt in 1 2; do
  if python scripts/run_single_room_worker.py \
    --repo-dir "$EXEC" \
    --run-dir "$RUN" \
    --csv "$EXEC/inputs/full_quality_school_reference_20260710/prompt.csv" \
    --run-name "$RUN_NAME" \
    --room-id "$ROOM_ID" \
    --start-stage manipuland \
    --stop-stage manipuland \
    --asset-pipeline generated_sam3d \
    --artiverse-data data/artiverse \
    --artiverse-embeddings data/artiverse/embeddings \
    --artvip-data data/artvip_sdf \
    --artvip-embeddings data/artvip_sdf/embeddings \
    --materials-data data/materials \
    --materials-source-embeddings data/materials_full_quality_contract/embeddings \
    --port-offset "$PORT_OFFSET" \
    --render-gpu-id 0; then
    break
  else
    worker_status=$?
  fi

  # A durable checkpoint makes a single in-allocation rerun safe after a lost
  # API connection.  Every quality, physics, VLM, and asset failure remains
  # fail-closed: only this exact transport class gets one retry.
  if (( attempt == 1 )) \
    && tail -n 260 "$runtime_log" | grep -Eq \
      'openai\.APIConnectionError|httpx\.RemoteProtocolError|Server disconnected without sending a response'; then
    status=$(curl --connect-timeout 10 --max-time 20 -sS -o /dev/null \
      -w '%{http_code}' -x "http://127.0.0.1:${GPU_PROXY_PORT}" \
      https://api.openai.com/v1/models || true)
    [[ "$status" == 401 ]] || { echo "API proxy lost before retry: $status" >&2; exit 70; }
    echo "Transient API transport failure; retrying durable manipuland resume once."
    sleep 10
    continue
  fi
  exit "$worker_status"
done

echo MANIPULAND_RESUME_PASS "$ROOM_ID"
