#!/usr/bin/env bash
# Template: rerun one unfinished/failed room with full-quality generated SAM3D policy.
# Copy this file per room and replace ROOM_ID, RUN_DIR, PORT_OFFSET, and log path.

#SBATCH -J room_full_quality
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 1-00:00:00
#SBATCH -o /data/run01/scvj260/scenesmith/outputs/%x-%j.out

set -euo pipefail

cd /data/run01/scvj260/scenesmith
source local_setup/compute_node_env.sh

RUN_DIR="${RUN_DIR:?Set RUN_DIR, e.g. outputs/2026-07-07/full_quality_scene_sam3d_artvip}"
PROMPT_CSV="${PROMPT_CSV:?Set PROMPT_CSV used by floor-plan stage}"
RUN_NAME="${RUN_NAME:-full_quality_room_worker}"
ROOM_ID="${ROOM_ID:?Set ROOM_ID, e.g. classroom_01}"
START_STAGE="${START_STAGE:-furniture}"
STOP_STAGE="${STOP_STAGE:-manipuland}"
PORT_OFFSET="${PORT_OFFSET:?Set unique PORT_OFFSET, e.g. 0, 100, 200}"
RENDER_GPU_ID="${RENDER_GPU_ID:-0}"

if [[ "$(hostname)" == ln* ]]; then
  echo "[guard] Refusing to run production on login node $(hostname)." >&2
  exit 10
fi

export PYTHONUNBUFFERED=1
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export PYTHONPATH="/data/run01/scvj260/scenesmith${PYTHONPATH:+:$PYTHONPATH}"
export HTTP_PROXY="${HTTP_PROXY:-http://ln08:18092}"
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTP_PROXY"
unset ALL_PROXY
unset all_proxy
export NO_PROXY="127.0.0.1,localhost,dashscope.aliyuncs.com,hf-mirror.com"
export no_proxy="$NO_PROXY"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/data/run01/scvj260/.cache/huggingface"
export TRANSFORMERS_CACHE="/data/run01/scvj260/.cache/huggingface/transformers"
export UV_CACHE_DIR="/data/run01/scvj260/uv-cache"
export UV_LINK_MODE="copy"
export UV_HTTP_TIMEOUT=600

echo "[slurm] job_id=${SLURM_JOB_ID:-unknown} node=$(hostname) start=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
nvidia-smi
test -f external/checkpoints/sam3.pt
test -f external/checkpoints/pipeline.yaml

PROBE_CODE="$(curl --max-time 60 --proxy "$HTTP_PROXY" -s -o /dev/null -w "%{http_code}" https://api.openai.com/v1/models || true)"
echo "[probe] openai_http=$PROBE_CODE proxy=$HTTP_PROXY"
case "$PROBE_CODE" in
  200|401|403) ;;
  *) echo "[guard] Proxy probe failed from GPU node; refusing to start." >&2; exit 20 ;;
esac

echo "[policy] generated_sam3d; this must not write combined_house"
echo "[preflight] resolving worker Hydra config and checking it does not force HSSD"
.venv/bin/python scripts/run_single_room_worker.py \
  --repo-dir /data/run01/scvj260/scenesmith \
  --run-dir "$RUN_DIR" \
  --csv "$PROMPT_CSV" \
  --run-name "$RUN_NAME" \
  --room-id "$ROOM_ID" \
  --start-stage "$START_STAGE" \
  --stop-stage "$STOP_STAGE" \
  --asset-pipeline generated_sam3d \
  --port-offset "$PORT_OFFSET" \
  --render-gpu-id "$RENDER_GPU_ID" \
  --config-only

.venv/bin/python scripts/run_single_room_worker.py \
  --repo-dir /data/run01/scvj260/scenesmith \
  --run-dir "$RUN_DIR" \
  --csv "$PROMPT_CSV" \
  --run-name "$RUN_NAME" \
  --room-id "$ROOM_ID" \
  --start-stage "$START_STAGE" \
  --stop-stage "$STOP_STAGE" \
  --asset-pipeline generated_sam3d \
  --port-offset "$PORT_OFFSET" \
  --render-gpu-id "$RENDER_GPU_ID"

echo "[required-next] rerender this room and run SAGE/Codex gate before final assembly"
echo "[slurm] end=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
