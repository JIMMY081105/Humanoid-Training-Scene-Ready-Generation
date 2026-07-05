#!/usr/bin/env bash
#SBATCH -J room_cleaning_closet
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 1-00:00:00
#SBATCH -o /data/run01/scvj260/scenesmith/outputs/2026-07-03/brand_new_school_floor_02-36-38_furniture_hssd_only_fast/room_worker_cleaning_closet-%j.out

set -euo pipefail
cd /data/run01/scvj260/scenesmith
source local_setup/compute_node_env.sh

RUN_DIR="outputs/2026-07-03/brand_new_school_floor_02-36-38_furniture_hssd_only_fast"

export PYTHONUNBUFFERED=1
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export PATH="/data/home/scvj260/bin:/data/run01/scvj260/mingyuan/.config/nvm/versions/node/v24.13.0/bin:$PATH"
export HTTP_PROXY="http://ln08:18092"
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTP_PROXY"
unset ALL_PROXY
unset all_proxy
export NO_PROXY="localhost,127.0.0.1,::1,dashscope.aliyuncs.com,hf-mirror.com"
export no_proxy="$NO_PROXY"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/data/run01/scvj260/.cache/huggingface"
export TRANSFORMERS_CACHE="/data/run01/scvj260/.cache/huggingface/transformers"
export UV_CACHE_DIR="/data/run01/scvj260/uv-cache"
export UV_LINK_MODE="copy"
export UV_HTTP_TIMEOUT=600
export SCENESMITH_CODEX_ISOLATE_HOME=1
export SCENESMITH_MANIPULAND_MAX_FURNITURE=3
export SCENESMITH_CODEX_CLI_ATTEMPTS=5
export SCENESMITH_CODEX_CLI_RETRY_DELAY_SECONDS=150
export SCENESMITH_CODEX_CLI_USAGE_LIMIT_WAITS=8
export SCENESMITH_CODEX_CLI_USAGE_LIMIT_PADDING_SECONDS=180
export SCENESMITH_CODEX_CLI_USAGE_LIMIT_MAX_SLEEP_SECONDS=14400
export SCENESMITH_CODEX_CLI_USAGE_LIMIT_FALLBACK_SECONDS=1800

echo "[slurm] job_id=${SLURM_JOB_ID:-unknown} node=$(hostname) start=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
nvidia-smi

.venv/bin/python scripts/run_single_room_worker.py \
  --repo-dir /data/run01/scvj260/scenesmith \
  --run-dir "$RUN_DIR" \
  --room-id cleaning_closet \
  --start-stage wall_mounted \
  --stop-stage manipuland \
  --port-offset 260 \
  --render-gpu-id 0

echo "[slurm] end=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
