#!/usr/bin/env bash
# Template: start a full-quality SceneSmith run with SAM3D/generated assets.
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

RUN_NAME="${RUN_NAME:-full_quality_scene_sam3d_artvip}"
PROMPT_CSV="${PROMPT_CSV:-inputs/my_scene.csv}"
RUN_DIR="${RUN_DIR:-outputs/$(date -u +%Y-%m-%d)/${RUN_NAME}}"

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

echo "[slurm] job_id=${SLURM_JOB_ID:-unknown} node=$(hostname) start=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
nvidia-smi

test -f external/checkpoints/sam3.pt
test -f external/checkpoints/pipeline.yaml

PROBE_CODE="$(curl --max-time 60 --proxy "$HTTP_PROXY" -s -o /dev/null -w "%{http_code}" https://api.openai.com/v1/models || true)"
echo "[probe] openai_http=$PROBE_CODE"
case "$PROBE_CODE" in
  200|401|403) ;;
  *) echo "[guard] Proxy probe failed; refusing to start." >&2; exit 20 ;;
esac

echo "[policy] full quality: generated SAM3D, ArtVIP articulated enabled, HSSD not used as global source"
echo "[run] RUN_DIR=$RUN_DIR PROMPT_CSV=$PROMPT_CSV"

mkdir -p "$RUN_DIR"
echo "[preflight] validating articulated router behavior"
.venv/bin/python scripts/validate_articulated_router.py \
  --output "$RUN_DIR/articulated_router_validation.json" \
  --vlm-backend openai \
  --top-k 3

.venv/bin/python main.py \
  +name="$RUN_NAME" \
  experiment.csv_path="$PROMPT_CSV" \
  experiment.num_workers=1 \
  floor_plan_agent.mode=house \
  codex.enabled=true \
  codex.cwd=/data/run01/scvj260/scenesmith \
  codex.timeout_seconds=1800 \
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
  furniture_agent.asset_manager.articulated.sources.artvip.enabled=true \
  wall_agent.asset_manager.articulated.sources.artvip.enabled=true \
  ceiling_agent.asset_manager.articulated.sources.artvip.enabled=true \
  manipuland_agent.asset_manager.articulated.sources.artvip.enabled=true \
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

echo "[required-next] render room reviews, run SAGE/Codex room gate, repair failures, then assemble/export"
echo "[slurm] end=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
