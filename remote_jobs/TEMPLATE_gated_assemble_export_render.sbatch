#!/usr/bin/env bash
# Template: assemble only after room_self_exam JSON files are passing.

#SBATCH -J gated_assemble
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00
#SBATCH -o /data/run01/scvj260/scenesmith/outputs/%x-%j.out

set -euo pipefail

cd /data/run01/scvj260/scenesmith
source local_setup/compute_node_env.sh

RUN_DIR="${RUN_DIR:?Set RUN_DIR, e.g. outputs/2026-07-07/full_quality_scene_sam3d_artvip}"
PROMPT_CSV="${PROMPT_CSV:?Set PROMPT_CSV used by the run}"
RUN_NAME="${RUN_NAME:-full_quality_gated_assemble}"
SCENE_DIR="$RUN_DIR/scene_000"
REVIEW_DIR="${REVIEW_DIR:-$SCENE_DIR/review/room_review_renders}"
GATE_DIR="${GATE_DIR:-$SCENE_DIR/quality_gates/room_self_exam}"

export PYTHONUNBUFFERED=1
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1

echo "[slurm] job_id=${SLURM_JOB_ID:-unknown} node=$(hostname) start=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
nvidia-smi || true

echo "[gate] running room self-exam"
.venv/bin/python scripts/room_self_exam.py \
  --scene-dir "$SCENE_DIR" \
  --review-dir "$REVIEW_DIR" \
  --output-dir "$GATE_DIR" \
  --max-collision-hulls 32

echo "[assemble] gate passed; assembling"
.venv/bin/python scripts/assemble_final_house_and_render.py \
  --repo-dir /data/run01/scvj260/scenesmith \
  --run-dir "$RUN_DIR" \
  --csv "$PROMPT_CSV" \
  --run-name "$RUN_NAME" \
  --gate-dir "$GATE_DIR" \
  --render

echo "[slurm] end=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
