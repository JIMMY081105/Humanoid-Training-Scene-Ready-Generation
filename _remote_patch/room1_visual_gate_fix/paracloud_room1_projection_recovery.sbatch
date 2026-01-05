#!/usr/bin/env bash
# Restore only the projection-lifted desk pair, then regenerate all Room 1
# render and quality evidence.  No asset generation or broad physics pass runs.

#SBATCH -J codex_room1_recover
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%j.out

set -euo pipefail

readonly EXEC=/data/run01/scvj260/codex_factory/room1_execution
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly VENV=/data/run01/scvj260/scenesmith/.venv
readonly RUN="$EXEC/outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710"
readonly SCENE="$RUN/scene_000"
readonly ROOM="$SCENE/room_classroom_01"
readonly BASELINE="$ROOM/scene_states/final_scene.backup_acceptance_details_20260714T010023Z/scene_state.json"

case "$(hostname)" in
  ln*|Ln*|lN*|LN*) echo "Refusing to run on a login node." >&2; exit 65 ;;
esac

test -s "$ROOM/scene_states/final_scene/scene_state.json"
test -s "$BASELINE"
test -x "$VENV/bin/python"

source "$VENV/bin/activate"
cd "$EXEC"
set -a
source .env
set +a

export PYTHONPATH="$FACTORY/python_overlay_agents_0_17_4:$EXEC"
export HF_HOME="$FACTORY/huggingface"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TORCH_HOME="$FACTORY/torch"
export CUDA_HOME=/data/run01/scvj260/cuda128_conda
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HTTP_PROXY=http://ln08:18092
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTP_PROXY"
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
unset ALL_PROXY all_proxy

python scripts/restore_room1_projection_anomaly.py \
  --room-dir "$ROOM" \
  --baseline-state "$BASELINE" \
  --receipt "$ROOM/quality_gates/room1_projection_recovery.json"

python scripts/run_single_room_worker.py \
  --repo-dir "$EXEC" \
  --run-dir "$RUN" \
  --csv "$EXEC/inputs/full_quality_school_reference_20260710/prompt.csv" \
  --run-name room1_projection_recovery_refresh \
  --room-id classroom_01 \
  --start-stage furniture \
  --stop-stage manipuland \
  --asset-pipeline generated_sam3d \
  --artiverse-data data/artiverse \
  --artiverse-embeddings data/artiverse/embeddings \
  --artvip-data data/artvip_sdf \
  --artvip-embeddings data/artvip_sdf/embeddings \
  --materials-data data/materials \
  --materials-source-embeddings data/materials/embeddings \
  --materials-embeddings data/materials_full_quality_contract/embeddings \
  --port-offset 72 \
  --render-gpu-id 0 \
  --refresh-final-blend

python scripts/render_room_review_views.py \
  --blend "$ROOM/scene_states/final_scene/scene.blend" \
  --scene-state "$ROOM/scene_states/final_scene/scene_state.json" \
  --room-id classroom_01 \
  --output-dir "$RUN/review/room_review_renders"

python scripts/pipeline_code_contract.py \
  --repo-dir "$EXEC" \
  --output "$EXEC/outputs/pipeline_code_contract.json"
python scripts/pipeline_code_contract.py \
  --repo-dir "$EXEC" \
  --output "$EXEC/outputs/pipeline_code_contract.json" \
  --verify-only

python scripts/room_self_exam.py \
  --scene-dir "$SCENE" \
  --review-dir "$RUN/review/room_review_renders" \
  --output-dir "$SCENE/quality_gates/room_self_exam_deterministic" \
  --rooms classroom_01 \
  --contract-profile school_reference_20260710 \
  --prompt-binding "$SCENE/quality_gates/room_prompt_binding.json"

python scripts/room_visual_self_exam.py \
  --scene-dir "$SCENE" \
  --deterministic-gate-dir "$SCENE/quality_gates/room_self_exam_deterministic" \
  --review-dir "$RUN/review/room_review_renders" \
  --reference-image "$EXEC/inputs/full_quality_school_reference_20260710/reference.png" \
  --output-dir "$SCENE/quality_gates/room_visual_self_exam" \
  --rooms classroom_01 \
  --contract-profile school_reference_20260710 \
  --effective-prompt "$EXEC/inputs/full_quality_school_reference_20260710/prompt.txt" \
  --input-manifest "$EXEC/inputs/full_quality_school_reference_20260710/input_manifest.json" \
  --prompt-binding "$SCENE/quality_gates/room_prompt_binding.json"

echo ROOM1_PROJECTION_RECOVERY_PASS
