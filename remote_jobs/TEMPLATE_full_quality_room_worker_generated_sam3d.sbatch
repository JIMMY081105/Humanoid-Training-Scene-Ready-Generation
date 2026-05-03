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

RUN_DIR="${RUN_DIR:?Set RUN_DIR, e.g. outputs/2026-07-07/full_quality_scene_sam3d_artvip_artiverse}"
PROMPT_CSV="${PROMPT_CSV:?Set PROMPT_CSV used by floor-plan stage}"
RUN_NAME="${RUN_NAME:-full_quality_room_worker}"
ROOM_ID="${ROOM_ID:?Set ROOM_ID, e.g. classroom_01}"
START_STAGE="${START_STAGE:-furniture}"
STOP_STAGE="${STOP_STAGE:-manipuland}"
PORT_OFFSET="${PORT_OFFSET:?Set unique PORT_OFFSET, e.g. 0, 100, 200}"
RENDER_GPU_ID="${RENDER_GPU_ID:-0}"
ARTIVERSE_DATA="${ARTIVERSE_DATA:-data/artiverse}"
ARTIVERSE_EMBEDDINGS="${ARTIVERSE_EMBEDDINGS:-$ARTIVERSE_DATA/embeddings}"
ARTVIP_DATA="${ARTVIP_DATA:-data/artvip_sdf}"
ARTVIP_EMBEDDINGS="${ARTVIP_EMBEDDINGS:-$ARTVIP_DATA/embeddings}"
MATERIALS_DATA="${MATERIALS_DATA:-data/materials}"
MATERIALS_SOURCE_EMBEDDINGS="${MATERIALS_SOURCE_EMBEDDINGS:-$MATERIALS_DATA/embeddings}"
MATERIALS_EMBEDDINGS="${MATERIALS_EMBEDDINGS:-data/materials_full_quality_contract/embeddings}"
OBJATHOR_DATA="${OBJATHOR_DATA:-data/objathor-assets}"
OBJATHOR_PREPROCESSED="${OBJATHOR_PREPROCESSED:-$OBJATHOR_DATA/preprocessed}"
ROUTER_VALIDATION="${ROUTER_VALIDATION:-$RUN_DIR/articulated_router_validation.json}"

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
nvidia-smi
test -f external/checkpoints/sam3.pt
test -f external/checkpoints/pipeline.yaml
test -s "$PROMPT_CSV"
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
  --output "$RUN_DIR/materials_contract_validation_${ROOM_ID}.json"
.venv/bin/python scripts/artiverse_contract.py \
  --dataset-root "$ARTIVERSE_DATA" \
  --embeddings-path "$ARTIVERSE_EMBEDDINGS" \
  --output "$RUN_DIR/artiverse_preparation_validation_${ROOM_ID}.json"
test -f "$ROUTER_VALIDATION"
.venv/bin/python - "$ROUTER_VALIDATION" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    result = json.load(handle)
if result.get("status") != "pass":
    raise SystemExit(f"Articulated router validation is not passing: {sys.argv[1]}")
if not result.get("selected_asset_identifiers", {}).get("artiverse"):
    raise SystemExit(f"Router validation has no Artiverse asset IDs: {sys.argv[1]}")
PY

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY for the authenticated production probe}"
PROBE_CODE="$(curl --max-time 60 --proxy "$HTTP_PROXY" -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer ${OPENAI_API_KEY}" https://api.openai.com/v1/models || true)"
echo "[probe] openai_http=$PROBE_CODE proxy=$HTTP_PROXY"
case "$PROBE_CODE" in
  200) ;;
  *) echo "[guard] Authenticated OpenAI probe failed from GPU node; refusing to start." >&2; exit 20 ;;
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
  --artiverse-data "$ARTIVERSE_DATA" \
  --artiverse-embeddings "$ARTIVERSE_EMBEDDINGS" \
  --artvip-data "$ARTVIP_DATA" \
  --artvip-embeddings "$ARTVIP_EMBEDDINGS" \
  --materials-data "$MATERIALS_DATA" \
  --materials-source-embeddings "$MATERIALS_SOURCE_EMBEDDINGS" \
  --materials-embeddings "$MATERIALS_EMBEDDINGS" \
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
  --artiverse-data "$ARTIVERSE_DATA" \
  --artiverse-embeddings "$ARTIVERSE_EMBEDDINGS" \
  --artvip-data "$ARTVIP_DATA" \
  --artvip-embeddings "$ARTVIP_EMBEDDINGS" \
  --materials-data "$MATERIALS_DATA" \
  --materials-source-embeddings "$MATERIALS_SOURCE_EMBEDDINGS" \
  --materials-embeddings "$MATERIALS_EMBEDDINGS" \
  --port-offset "$PORT_OFFSET" \
  --render-gpu-id "$RENDER_GPU_ID"

echo "[required-next] rerender this room and run SAGE/Codex gate before final assembly"
echo "[slurm] end=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
