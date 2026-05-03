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
ARTIVERSE_DATA="${ARTIVERSE_DATA:-data/artiverse}"
ARTIVERSE_EMBEDDINGS="${ARTIVERSE_EMBEDDINGS:-$ARTIVERSE_DATA/embeddings}"
ARTVIP_DATA="${ARTVIP_DATA:-data/artvip_sdf}"
ARTVIP_EMBEDDINGS="${ARTVIP_EMBEDDINGS:-$ARTVIP_DATA/embeddings}"
MATERIALS_DATA="${MATERIALS_DATA:-data/materials}"
MATERIALS_SOURCE_EMBEDDINGS="${MATERIALS_SOURCE_EMBEDDINGS:-$MATERIALS_DATA/embeddings}"
MATERIALS_EMBEDDINGS="${MATERIALS_EMBEDDINGS:-data/materials_full_quality_contract/embeddings}"
OBJATHOR_DATA="${OBJATHOR_DATA:-data/objathor-assets}"
OBJATHOR_PREPROCESSED="${OBJATHOR_PREPROCESSED:-$OBJATHOR_DATA/preprocessed}"
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
export HF_HUB_OFFLINE=1
export SCENESMITH_REQUIRE_GATED_FINAL_ASSEMBLY=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1

require_embedding_index() {
  local directory="$1"
  test -s "$directory/clip_embeddings.npy"
  test -s "$directory/embedding_index.yaml"
  test -s "$directory/metadata_index.yaml" || test -s "$directory/metadata_index.json"
}

require_objathor_index() {
  local directory="$1"
  test -s "$directory/clip_embeddings.npy"
  test -s "$directory/embedding_index.yaml"
  test -s "$directory/metadata_index.json"
}

echo "[slurm] job_id=${SLURM_JOB_ID:-unknown} node=$(hostname) start=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
nvidia-smi || true

test -d "$ARTIVERSE_DATA"
test -d "$ARTVIP_DATA"
test -d "$MATERIALS_DATA"
test -d "$OBJATHOR_DATA"
require_embedding_index "$ARTIVERSE_EMBEDDINGS"
require_embedding_index "$ARTVIP_EMBEDDINGS"
require_embedding_index "$MATERIALS_SOURCE_EMBEDDINGS"
require_embedding_index "$MATERIALS_EMBEDDINGS"
require_objathor_index "$OBJATHOR_PREPROCESSED"

.venv/bin/python scripts/materials_contract.py validate \
  --data-root "$MATERIALS_DATA" \
  --source-embeddings "$MATERIALS_SOURCE_EMBEDDINGS" \
  --contract-embeddings "$MATERIALS_EMBEDDINGS" \
  --min-retained 1900 \
  --max-pruned 15 \
  --output "$RUN_DIR/materials_contract_validation.json"
.venv/bin/python scripts/artiverse_contract.py \
  --dataset-root "$ARTIVERSE_DATA" \
  --embeddings-path "$ARTIVERSE_EMBEDDINGS" \
  --output "$RUN_DIR/artiverse_preparation_validation.json"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY for the authenticated production probe}"
PROBE_CODE="$(curl --max-time 60 --proxy "$HTTP_PROXY" -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer ${OPENAI_API_KEY}" https://api.openai.com/v1/models || true)"
echo "[probe] openai_http=$PROBE_CODE proxy=$HTTP_PROXY"
case "$PROBE_CODE" in
  200) ;;
  *) echo "[guard] Authenticated OpenAI probe failed from GPU node; refusing to start." >&2; exit 20 ;;
esac

.venv/bin/python scripts/validate_articulated_router.py \
  --output "$RUN_DIR/articulated_router_validation.json" \
  --repo-dir /data/run01/scvj260/scenesmith \
  --artiverse-data "$ARTIVERSE_DATA" \
  --artiverse-embeddings "$ARTIVERSE_EMBEDDINGS" \
  --artvip-data "$ARTVIP_DATA" \
  --artvip-embeddings "$ARTVIP_EMBEDDINGS" \
  --vlm-backend openai \
  --top-k 3

echo "[output] $RUN_DIR/articulated_router_validation.json"
echo "[slurm] end=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
