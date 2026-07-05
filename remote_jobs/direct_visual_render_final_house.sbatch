#!/usr/bin/env bash
#SBATCH -J direct_visual_render_house
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00
#SBATCH -o /data/run01/scvj260/scenesmith/outputs/2026-07-03/brand_new_school_floor_02-36-38_furniture_hssd_only_fast/direct_visual_render_house-%j.out

set -euo pipefail
cd /data/run01/scvj260/scenesmith
source local_setup/compute_node_env.sh

SCENE_DIR="/data/run01/scvj260/scenesmith/outputs/2026-07-03/brand_new_school_floor_02-36-38_furniture_hssd_only_fast/scene_000"
COMBINED_DIR="$SCENE_DIR/combined_house"

export PYTHONUNBUFFERED=1
export LOGLEVEL=INFO

echo "[slurm] job_id=${SLURM_JOB_ID:-unknown} node=$(hostname) start=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
nvidia-smi || true

.venv/bin/python scripts/render_direct_visual_house.py \
  --scene-dir "$SCENE_DIR" \
  --output-blend "$COMBINED_DIR/house.blend" \
  --render-dir "$COMBINED_DIR/outlook_renders"

echo "[slurm] end=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
