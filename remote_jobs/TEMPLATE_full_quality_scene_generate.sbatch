#!/usr/bin/env bash
# Template: generate and validate only the floor plan for a full-quality run.
# Room generation is deliberately sharded and gated by the worker template.
# Copy this file to a run-specific sbatch and replace the variables below.

#SBATCH -J scenesmith_full_quality
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 1-00:00:00
#SBATCH -o /data/run01/scvj260/scenesmith/outputs/%x-%j.out

set -euo pipefail

cd /data/run01/scvj260/scenesmith

RUN_NAME="${RUN_NAME:-full_quality_scene_sam3d_artvip_artiverse}"
PROMPT_CSV="${PROMPT_CSV:?Set the run-specific prompt CSV}"
RUN_DIR="${RUN_DIR:-outputs/$(date -u +%Y-%m-%d)/${RUN_NAME}}"
LAYOUT_VALIDATOR="${LAYOUT_VALIDATOR:?Set a scene-specific layout validator, e.g. scripts/validate_school_floor_layout.py}"
LAYOUT_GATE_OUTPUT="${LAYOUT_GATE_OUTPUT:-$RUN_DIR/scene_000/quality_gates/floor_plan_layout.json}"
ARTIVERSE_DATA="${ARTIVERSE_DATA:-data/artiverse}"
ARTIVERSE_EMBEDDINGS="${ARTIVERSE_EMBEDDINGS:-$ARTIVERSE_DATA/embeddings}"
ARTVIP_DATA="${ARTVIP_DATA:-data/artvip_sdf}"
ARTVIP_EMBEDDINGS="${ARTVIP_EMBEDDINGS:-$ARTVIP_DATA/embeddings}"
MATERIALS_DATA="${MATERIALS_DATA:-data/materials}"
MATERIALS_SOURCE_EMBEDDINGS="${MATERIALS_SOURCE_EMBEDDINGS:-$MATERIALS_DATA/embeddings}"
MATERIALS_EMBEDDINGS="${MATERIALS_EMBEDDINGS:-data/materials_full_quality_contract/embeddings}"
OBJATHOR_DATA="${OBJATHOR_DATA:-data/objathor-assets}"
OBJATHOR_PREPROCESSED="${OBJATHOR_PREPROCESSED:-$OBJATHOR_DATA/preprocessed}"

if [[ "$(hostname)" == ln* ]]; then
  echo "[guard] Refusing to run production on login node $(hostname)." >&2
  exit 10
fi

source local_setup/compute_node_env.sh

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
test -f "$LAYOUT_VALIDATOR"
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
echo "[probe] openai_http=$PROBE_CODE"
case "$PROBE_CODE" in
  200) ;;
  *) echo "[guard] Authenticated OpenAI probe failed; refusing to start." >&2; exit 20 ;;
esac

echo "[policy] full quality: generated SAM3D; compulsory Artiverse + ArtVIP; HSSD not used as global source"
echo "[run] RUN_DIR=$RUN_DIR PROMPT_CSV=$PROMPT_CSV"

mkdir -p "$RUN_DIR"
echo "[preflight] validating articulated router behavior"
.venv/bin/python scripts/validate_articulated_router.py \
  --output "$RUN_DIR/articulated_router_validation.json" \
  --repo-dir /data/run01/scvj260/scenesmith \
  --artiverse-data "$ARTIVERSE_DATA" \
  --artiverse-embeddings "$ARTIVERSE_EMBEDDINGS" \
  --artvip-data "$ARTVIP_DATA" \
  --artvip-embeddings "$ARTVIP_EMBEDDINGS" \
  --vlm-backend openai \
  --top-k 3

.venv/bin/python main.py \
  +name="$RUN_NAME" \
  experiment.csv_path="$PROMPT_CSV" \
  experiment.num_workers=1 \
  experiment.pipeline.start_stage=floor_plan \
  experiment.pipeline.stop_stage=floor_plan \
  experiment.pipeline.parallel_rooms=false \
  experiment.pipeline.final_assembly_policy=external_artiverse_gated \
  experiment.materials_retrieval_server.data_path="$MATERIALS_DATA" \
  experiment.materials_retrieval_server.embeddings_path="$MATERIALS_EMBEDDINGS" \
  floor_plan_agent.mode=house \
  ++codex.enabled=true \
  ++codex.cwd=/data/run01/scvj260/scenesmith \
  ++codex.timeout_seconds=1800 \
  furniture_agent.asset_manager.general_asset_source=generated \
  wall_agent.asset_manager.general_asset_source=generated \
  ceiling_agent.asset_manager.general_asset_source=generated \
  manipuland_agent.asset_manager.general_asset_source=objaverse \
  furniture_agent.asset_manager.backend=sam3d \
  wall_agent.asset_manager.backend=sam3d \
  ceiling_agent.asset_manager.backend=sam3d \
  manipuland_agent.asset_manager.backend=sam3d \
  manipuland_agent.asset_manager.objaverse.use_top_k=10 \
  manipuland_agent.asset_manager.objaverse.use_lenient_validation=true \
  furniture_agent.asset_manager.router.strategies.generated.enabled=true \
  wall_agent.asset_manager.router.strategies.generated.enabled=true \
  ceiling_agent.asset_manager.router.strategies.generated.enabled=true \
  manipuland_agent.asset_manager.router.strategies.generated.enabled=true \
  furniture_agent.asset_manager.router.strategies.articulated.enabled=true \
  wall_agent.asset_manager.router.strategies.articulated.enabled=true \
  ceiling_agent.asset_manager.router.strategies.articulated.enabled=true \
  manipuland_agent.asset_manager.router.strategies.articulated.enabled=true \
  furniture_agent.asset_manager.articulated.sources.artvip.enabled=true \
  wall_agent.asset_manager.articulated.sources.artvip.enabled=true \
  ceiling_agent.asset_manager.articulated.sources.artvip.enabled=true \
  manipuland_agent.asset_manager.articulated.sources.artvip.enabled=true \
  furniture_agent.asset_manager.articulated.sources.artvip.data_path="$ARTVIP_DATA" \
  furniture_agent.asset_manager.articulated.sources.artvip.embeddings_path="$ARTVIP_EMBEDDINGS" \
  ++furniture_agent.asset_manager.articulated.sources.artiverse.enabled=true \
  ++furniture_agent.asset_manager.articulated.sources.artiverse.data_path="$ARTIVERSE_DATA" \
  ++furniture_agent.asset_manager.articulated.sources.artiverse.embeddings_path="$ARTIVERSE_EMBEDDINGS" \
  ++furniture_agent.asset_manager.router.strategies.artiverse_articulated.enabled=true \
  ++furniture_agent.asset_manager.router.strategies.artiverse_articulated.max_retries=3 \
  ++furniture_agent.asset_manager.router.strategies.artiverse_articulated.use_lenient_validation=true \
  furniture_agent.collision_geometry.coacd.max_convex_hull=32 \
  wall_agent.collision_geometry.coacd.max_convex_hull=32 \
  ceiling_agent.collision_geometry.coacd.max_convex_hull=32 \
  manipuland_agent.collision_geometry.coacd.max_convex_hull=32 \
  furniture_agent.collision_geometry.vhacd.max_convex_hulls=32 \
  wall_agent.collision_geometry.vhacd.max_convex_hulls=32 \
  ceiling_agent.collision_geometry.vhacd.max_convex_hulls=32 \
  manipuland_agent.collision_geometry.vhacd.max_convex_hulls=32 \
  furniture_agent.asset_manager.router.parallel_workers=1 \
  wall_agent.asset_manager.router.parallel_workers=1 \
  ceiling_agent.asset_manager.router.parallel_workers=1 \
  manipuland_agent.asset_manager.router.parallel_workers=1 \
  hydra.run.dir="$RUN_DIR"

echo "[gate] running scene-specific layout validator: $LAYOUT_VALIDATOR"
.venv/bin/python "$LAYOUT_VALIDATOR" \
  --layout "$RUN_DIR/scene_000/house_layout.json" \
  --output "$LAYOUT_GATE_OUTPUT"

echo "[required-next] run each room with generated_sam3d; render three room views; run deterministic + visual gates; repair failures; only then assemble/export"
echo "[slurm] end=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
