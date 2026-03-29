#!/usr/bin/env bash
# Resume only Classroom 4's committed manipuland stage after canonical scene
# assets have been materialized from migration symlinks.

#SBATCH -J resume_classroom_04
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%j.out

set -euo pipefail
readonly EXEC=/data/run01/scvj260/codex_factory/room1_execution
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly VENV=/data/run01/scvj260/scenesmith/.venv
readonly RUN="$EXEC/outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710"
readonly SCENE="$RUN/scene_000"
readonly ROOM_ID=classroom_04
readonly GPU_PROXY_PORT=18604

case "$(hostname)" in
  ln*|Ln*|lN*|LN*) echo 'Refusing login node.' >&2; exit 65 ;;
esac
test -d "$SCENE/floor_plans/$ROOM_ID"
test ! -L "$SCENE/floor_plans"
test -d "$SCENE/room_geometry"
test ! -L "$SCENE/room_geometry"
test -s "$SCENE/room_$ROOM_ID/scene_states/scene_after_ceiling_objects/scene_state.json"

source "$VENV/bin/activate"
cd "$EXEC"
set -a
source .env
set +a
export PYTHONPATH="$FACTORY/python_overlay_agents_0_17_4:$EXEC"
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

ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ConnectTimeout=15 \
  -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$FACTORY/ln07_known_hosts" \
  -L "127.0.0.1:${GPU_PROXY_PORT}:127.0.0.1:18093" ln07 &
GPU_PROXY_PID=$!
cleanup() {
  kill "$GPU_PROXY_PID" 2>/dev/null || true
  wait "$GPU_PROXY_PID" 2>/dev/null || true
}
trap cleanup EXIT
for _ in $(seq 1 10); do
  if ! kill -0 "$GPU_PROXY_PID" 2>/dev/null; then wait "$GPU_PROXY_PID"; fi
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

python scripts/run_single_room_worker.py \
  --repo-dir "$EXEC" \
  --run-dir "$RUN" \
  --csv "$EXEC/inputs/full_quality_school_reference_20260710/prompt.csv" \
  --run-name paracloud_resume_classroom_04 \
  --room-id "$ROOM_ID" \
  --start-stage manipuland \
  --stop-stage manipuland \
  --asset-pipeline generated_sam3d \
  --artiverse-data data/artiverse \
  --artiverse-embeddings data/artiverse/embeddings \
  --artvip-data data/artvip_sdf \
  --artvip-embeddings data/artvip_sdf/embeddings \
  --materials-data data/materials \
  --materials-source-embeddings data/materials/embeddings \
  --materials-embeddings data/materials_full_quality_contract/embeddings \
  --port-offset 404 \
  --render-gpu-id 0

echo CLASSROOM04_MANIPULAND_RESUME_PASS

