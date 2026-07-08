#!/usr/bin/env bash
# Template: validate AssetRouter actually chooses articulated retrieval.

#SBATCH -J validate_router
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o /data/run01/scvj260/scenesmith/outputs/%x-%j.out

set -euo pipefail

cd /data/run01/scvj260/scenesmith
source local_setup/compute_node_env.sh

RUN_DIR="${RUN_DIR:-outputs/router_validation_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN_DIR"

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

echo "[slurm] job_id=${SLURM_JOB_ID:-unknown} node=$(hostname) start=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
nvidia-smi || true

PROBE_CODE="$(curl --max-time 60 --proxy "$HTTP_PROXY" -s -o /dev/null -w "%{http_code}" https://api.openai.com/v1/models || true)"
echo "[probe] openai_http=$PROBE_CODE proxy=$HTTP_PROXY"
case "$PROBE_CODE" in
  200|401|403) ;;
  *) echo "[guard] Proxy probe failed from GPU node; refusing to start." >&2; exit 20 ;;
esac

.venv/bin/python scripts/validate_articulated_router.py \
  --output "$RUN_DIR/articulated_router_validation.json" \
  --vlm-backend openai \
  --top-k 3

echo "[output] $RUN_DIR/articulated_router_validation.json"
echo "[slurm] end=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
