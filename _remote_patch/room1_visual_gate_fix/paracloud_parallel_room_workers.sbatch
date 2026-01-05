#!/usr/bin/env bash
# Eight independent full-quality room workers. Shared target preflight,
# immutable layout, and code contract must already be passing before submit.

#SBATCH -J codex_room_wave
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --array=0-7%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 1-00:00:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%A_%a.out

set -euo pipefail

readonly EXEC=/data/run01/scvj260/codex_factory/room1_execution
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly VENV=/data/run01/scvj260/scenesmith/.venv
readonly RUN="$EXEC/outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710"
readonly SCENE="$RUN/scene_000"

case "$(hostname)" in
  ln*|Ln*|lN*|LN*) echo "Refusing to run on a login node." >&2; exit 65 ;;
esac

ROOM_ID="$(printf '%s\n' library storage_room classroom_02 classroom_03 classroom_04 classroom_05 classroom_06 boys_toilet | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")"
test -n "$ROOM_ID"
test -s "$SCENE/house_layout.json"
test -s "$EXEC/outputs/preflight/full_quality_school_reference_20260710/sam3d_offline_load.json"
test -s "$EXEC/outputs/preflight/full_quality_school_reference_20260710/sam3d_offline_generation/receipt.json"
test -s "$EXEC/outputs/pipeline_code_contract.json"

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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
# The cluster's historic ln08 proxy is unavailable.  Each GPU node instead
# makes a loopback-only forward to ln07's loopback-only reverse tunnel; ln07
# then reaches the existing laptop proxy.  Ports are per-array-task so two
# one-GPU workers on one physical host never collide.  Nothing listens on a
# cluster-facing address and the tunnel is removed as the Slurm task exits.
readonly GPU_PROXY_PORT="$((18100 + SLURM_ARRAY_TASK_ID))"
ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ConnectTimeout=15 \
  -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$FACTORY/ln07_known_hosts" \
  -L "127.0.0.1:${GPU_PROXY_PORT}:127.0.0.1:18093" ln07 &
GPU_PROXY_PID=$!
cleanup_gpu_proxy() {
  kill "$GPU_PROXY_PID" 2>/dev/null || true
  wait "$GPU_PROXY_PID" 2>/dev/null || true
}
trap cleanup_gpu_proxy EXIT
for _ in $(seq 1 10); do
  if ! kill -0 "$GPU_PROXY_PID" 2>/dev/null; then
    wait "$GPU_PROXY_PID"
  fi
  PROXY_STATUS=$(curl --connect-timeout 10 --max-time 20 -sS -o /dev/null -w '%{http_code}' \
    -x "http://127.0.0.1:${GPU_PROXY_PORT}" https://api.openai.com/v1/models || true)
  [ "$PROXY_STATUS" = 401 ] && break
  sleep 2
done
[ "${PROXY_STATUS:-000}" = 401 ] || { echo "GPU proxy tunnel unavailable: ${PROXY_STATUS:-000}" >&2; exit 70; }
export HTTP_PROXY="http://127.0.0.1:${GPU_PROXY_PORT}"
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTP_PROXY"
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
unset ALL_PROXY all_proxy

nvidia-smi
python scripts/run_single_room_worker.py \
  --repo-dir "$EXEC" \
  --run-dir "$RUN" \
  --csv "$EXEC/inputs/full_quality_school_reference_20260710/prompt.csv" \
  --run-name "paracloud_wave_$ROOM_ID" \
  --room-id "$ROOM_ID" \
  --start-stage furniture \
  --stop-stage manipuland \
  --asset-pipeline generated_sam3d \
  --artiverse-data data/artiverse \
  --artiverse-embeddings data/artiverse/embeddings \
  --artvip-data data/artvip_sdf \
  --artvip-embeddings data/artvip_sdf/embeddings \
  --materials-data data/materials \
  --materials-source-embeddings data/materials/embeddings \
  --materials-embeddings data/materials_full_quality_contract/embeddings \
  --port-offset "$((200 + SLURM_ARRAY_TASK_ID * 50))" \
  --render-gpu-id 0

echo ROOM_WORKER_PASS "$ROOM_ID"
