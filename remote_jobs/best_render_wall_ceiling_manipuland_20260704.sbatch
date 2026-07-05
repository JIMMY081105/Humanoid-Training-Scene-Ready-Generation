#!/usr/bin/env bash
# Resume the full SceneSmith production pipeline for the school floor from the
# saved furniture states. This must run under a SLURM GPU allocation.
# The login-node proxy bridge is expected at http://ln08:18092.

#SBATCH -J school_best_render
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 1-00:00:00
#SBATCH -o /data/run01/scvj260/scenesmith/outputs/2026-07-03/brand_new_school_floor_02-36-38_furniture_hssd_only_fast/school_best_render-%j.out

set -euo pipefail

cd /data/run01/scvj260/scenesmith

RUN_DIR="outputs/2026-07-03/brand_new_school_floor_02-36-38_furniture_hssd_only_fast"
SCENE="$RUN_DIR/scene_000"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$RUN_DIR/best_render_wall_ceiling_manipuland_${STAMP}.log"

exec > >(tee -a "$LOG") 2>&1

echo "[slurm] job_id=${SLURM_JOB_ID:-unknown}"
echo "[slurm] node=$(hostname)"
echo "[slurm] start=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"

if [[ "$(hostname)" == ln* ]]; then
  echo "[guard] Refusing to run production on login node $(hostname)." >&2
  exit 10
fi

nvidia-smi

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
export NO_PROXY="127.0.0.1,localhost,dashscope.aliyuncs.com,hf-mirror.com"
export no_proxy="$NO_PROXY"

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/data/run01/scvj260/.cache/huggingface"
export TRANSFORMERS_CACHE="/data/run01/scvj260/.cache/huggingface/transformers"
export UV_CACHE_DIR="/data/run01/scvj260/uv-cache"
export UV_LINK_MODE="copy"
export UV_HTTP_TIMEOUT=600

export SCENESMITH_CODEX_CLI_ATTEMPTS=5
export SCENESMITH_CODEX_CLI_RETRY_DELAY_SECONDS=150
export SCENESMITH_CODEX_CLI_USAGE_LIMIT_WAITS=8
export SCENESMITH_CODEX_CLI_USAGE_LIMIT_PADDING_SECONDS=180
export SCENESMITH_CODEX_CLI_USAGE_LIMIT_MAX_SLEEP_SECONDS=14400
export SCENESMITH_CODEX_CLI_USAGE_LIMIT_FALLBACK_SECONDS=1800

echo "[env] python=$(.venv/bin/python -V 2>&1)"
echo "[env] codex=$(command -v codex || true)"
codex --version || true
echo "[env] proxy=$HTTP_PROXY"

PROBE_CODE="$(curl --max-time 60 --proxy "$HTTP_PROXY" -s -o /dev/null -w "%{http_code}" https://api.openai.com/v1/models || true)"
echo "[probe] openai_http=$PROBE_CODE"
case "$PROBE_CODE" in
  200|401|403) ;;
  *)
    echo "[guard] Proxy probe failed from GPU node; refusing to start SceneSmith." >&2
    exit 20
    ;;
esac

echo "[archive] preserving scripted shortcut final outputs"
for d in "$SCENE"/room_*/scene_states/final_scene; do
  if [[ -d "$d" ]]; then
    mv "$d" "$d.scripted_shortcut_backup_$STAMP"
    echo "[archive] moved $d"
  fi
done
if [[ -d "$SCENE/combined_house" ]]; then
  mv "$SCENE/combined_house" "$SCENE/combined_house.scripted_shortcut_backup_$STAMP"
  echo "[archive] moved $SCENE/combined_house"
fi

.venv/bin/python - <<'PY'
from pathlib import Path
base = Path("outputs/2026-07-03/brand_new_school_floor_02-36-38_furniture_hssd_only_fast/scene_000")
print("[state] furniture states", sum(1 for _ in base.glob("room_*/scene_states/scene_after_furniture/scene_state.json")))
print("[state] final states", sum(1 for _ in base.glob("room_*/scene_states/final_scene/scene_state.json")))
PY

echo "[run] starting SceneSmith production continuation"
.venv/bin/python main.py \
  +name=brand_new_school_floor_best_render \
  experiment.csv_path=inputs/full_school_floor_20260703.csv \
  experiment.num_workers=1 \
  experiment.pipeline.start_stage=wall_mounted \
  experiment.pipeline.stop_stage=manipuland \
  experiment.pipeline.parallel_rooms=false \
  floor_plan_agent.mode=house \
  codex.enabled=true \
  codex.cwd=/data/run01/scvj260/scenesmith \
  codex.timeout_seconds=1800 \
  furniture_agent.asset_manager.router.parallel_workers=1 \
  wall_agent.asset_manager.router.parallel_workers=1 \
  ceiling_agent.asset_manager.router.parallel_workers=1 \
  manipuland_agent.asset_manager.router.parallel_workers=1 \
  furniture_agent.asset_manager.router.strategies.generated.max_retries=5 \
  wall_agent.asset_manager.router.strategies.generated.max_retries=5 \
  ceiling_agent.asset_manager.router.strategies.generated.max_retries=5 \
  manipuland_agent.asset_manager.router.strategies.generated.max_retries=5 \
  furniture_agent.asset_manager.router.strategies.thin_covering.generator.enabled=false \
  wall_agent.asset_manager.router.strategies.thin_covering.generator.enabled=false \
  ceiling_agent.asset_manager.router.strategies.thin_covering.generator.enabled=false \
  manipuland_agent.asset_manager.router.strategies.thin_covering.generator.enabled=false \
  furniture_agent.context_image_generation.enabled=false \
  manipuland_agent.context_image_generation.enabled=false \
  furniture_agent.asset_manager.hssd.use_top_k=20 \
  wall_agent.asset_manager.hssd.use_top_k=20 \
  ceiling_agent.asset_manager.hssd.use_top_k=20 \
  manipuland_agent.asset_manager.hssd.use_top_k=20 \
  furniture_agent.asset_manager.general_asset_source=hssd \
  wall_agent.asset_manager.general_asset_source=hssd \
  ceiling_agent.asset_manager.general_asset_source=hssd \
  manipuland_agent.asset_manager.general_asset_source=hssd \
  furniture_agent.asset_manager.artiverse_articulated.enabled=true \
  furniture_agent.asset_manager.artiverse_articulated.data_path=data/artiverse \
  furniture_agent.asset_manager.router.strategies.artiverse_articulated.enabled=true \
  wall_agent.asset_manager.router.strategies.artiverse_articulated.enabled=false \
  ceiling_agent.asset_manager.router.strategies.artiverse_articulated.enabled=false \
  manipuland_agent.asset_manager.router.strategies.artiverse_articulated.enabled=false \
  hydra.run.dir="$RUN_DIR"

echo "[slurm] end=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
