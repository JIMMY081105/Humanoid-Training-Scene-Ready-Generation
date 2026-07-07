#!/usr/bin/env bash
#SBATCH -J run3_room_reviews
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o /data/run01/scvj260/scenesmith/outputs/2026-07-03/brand_new_school_floor_02-36-38_furniture_hssd_only_fast/run3_room_reviews-%j.out

set -euo pipefail
cd /data/run01/scvj260/scenesmith
source local_setup/compute_node_env.sh

SCENE_DIR="/data/run01/scvj260/scenesmith/outputs/2026-07-03/brand_new_school_floor_02-36-38_furniture_hssd_only_fast/scene_000"
RUN3_VISUAL_DIR="$SCENE_DIR/combined_house/run3_centered_visual"

mkdir -p "$RUN3_VISUAL_DIR/room_review_renders"

export PYTHONUNBUFFERED=1

echo "[slurm] job_id=${SLURM_JOB_ID:-unknown} node=$(hostname) start=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
nvidia-smi || true

.venv/bin/python scripts/render_room_review_views.py \
  --blend "$RUN3_VISUAL_DIR/house.blend" \
  --house-state "$SCENE_DIR/combined_house/house_state.json" \
  --output-dir "$RUN3_VISUAL_DIR/room_review_renders"

echo "[slurm] end=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
