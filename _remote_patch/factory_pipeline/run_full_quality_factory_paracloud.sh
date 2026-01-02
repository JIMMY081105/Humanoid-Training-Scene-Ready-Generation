#!/usr/bin/env bash
# Resumable, fail-closed ParaCloud execution for the food-factory profile.
set -euo pipefail

readonly REPO="${SCENESMITH_FACTORY_REPO:-/data/run01/scvj260/scenesmith-factory-codex}"
readonly OUTPUT_ROOT="${SCENESMITH_FACTORY_OUTPUT_ROOT:-/data/run01/scvj260/factory_outputs}"
readonly RUN_NAME=full_quality_factory_sam3d_artvip_artiverse_20260713
readonly RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
readonly SCENE_DIR="${RUN_DIR}/scene_000"
readonly INPUT_DIR="${REPO}/inputs/full_quality_factory_reference_20260713"
readonly PROMPT_CSV="${INPUT_DIR}/prompt.csv"
readonly FACTORY_CONTRACT="${REPO}/scripts/factory_contract.json"
readonly REVIEW_DIR="${SCENE_DIR}/review/room_review_renders"
readonly DETERMINISTIC_GATE_DIR="${SCENE_DIR}/quality_gates/room_self_exam_deterministic"
readonly VISUAL_GATE_DIR="${SCENE_DIR}/quality_gates/room_self_exam"
readonly PROMPT_BINDING="${SCENE_DIR}/quality_gates/room_prompt_binding.json"
readonly STATUS_FILE="${RUN_DIR}/pipeline.status"
readonly CONTRACT_PROFILE=factory_reference_20260713
readonly MATERIALS_DATA="${REPO}/data/materials"
readonly MATERIALS_SOURCE_EMBEDDINGS="${MATERIALS_DATA}/embeddings"
readonly MATERIALS_EMBEDDINGS="${REPO}/data/materials_full_quality_contract/embeddings"
readonly SAM3D_PREFLIGHT_DIR="${OUTPUT_ROOT}/preflight/${RUN_NAME}"
readonly ARTIVERSE_VISUAL_RECEIPT="${RUN_DIR}/artiverse_visual_resources.json"
readonly BENCHMARK_RECEIPT="${RUN_DIR}/processing_hall_full_quality_benchmark.json"
readonly BENCHMARK_EVENTS="${RUN_DIR}/processing_hall_full_quality_benchmark.events.tsv"
readonly BENCHMARK_GPU_SAMPLES="${RUN_DIR}/processing_hall_full_quality_benchmark.gpu.csv"
readonly EXPECTED_ORIGINAL_PROMPT_SHA256=3bb58cf8e07012cc0c5381eb51730da03b25fb6659a08338238d9fd4baa5093b
readonly EXPECTED_APPENDIX_SHA256=ffc327f082431d407c15b25b039d7d2676786a03663f186777558e9ba8e1db19
readonly EXPECTED_EFFECTIVE_PROMPT_SHA256=35387fa3fffb0af92b6c9e07e39f2b59a5443d497190f721090f82bd81734322
readonly EXPECTED_FACTORY_CONTRACT_SHA256=f33e56eef4375d1f68684bd10c02d7d2f82ee72420b8a12570e8fcbe135a94cd

# The shared virtualenv is an editable install.  Force every Python entrypoint
# to resolve SceneSmith from this isolated, hash-attested factory checkout.
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

EXECUTION_MODE="${SCENESMITH_EXECUTION_MODE:-full}"
MAX_PARALLEL_GPUS="${SCENESMITH_MAX_PARALLEL_GPUS:-8}"
RUN_ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
GPU_MONITOR_PID=""
PIPELINE_ACCEPTED=0
SUCCESS_STATUS=awaiting_2gpu_acceptance

ROOMS=(
  cold_storage maintenance office_administration
  changing_room break_room ingredient_receiving dry_storage
  processing_hall packaging_hall finished_goods_storage
  washing_preparation qc_laboratory boys_toilet girls_toilet
)
case "${EXECUTION_MODE}" in
  benchmark_processing_hall)
    SUCCESS_STATUS=benchmark_complete
    ROOMS=(processing_hall)
    ;;
  full) ;;
  *) echo "unsupported SCENESMITH_EXECUTION_MODE=${EXECUTION_MODE}" >&2; exit 2 ;;
esac

mkdir -p "${RUN_DIR}" "${REVIEW_DIR}" "${DETERMINISTIC_GATE_DIR}" "${VISUAL_GATE_DIR}"
exec 9>"${RUN_DIR}/.pipeline.lock"
flock -n 9 || { echo "another factory runner holds the pipeline lock" >&2; exit 75; }

write_status() {
  local state="$1" code="$2" temporary="${STATUS_FILE}.${RUN_ATTEMPT_ID}.tmp"
  printf '%s exit=%s attempt=%s %s\n' "${state}" "${code}" "${RUN_ATTEMPT_ID}" "$(date -Is)" >"${temporary}"
  mv -f "${temporary}" "${STATUS_FILE}"
}
record_event() {
  [[ "${EXECUTION_MODE}" == benchmark_processing_hall ]] || return 0
  printf '%s\t%s\t%s\n' "$(date +%s)" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1" >>"${BENCHMARK_EVENTS}"
}
stop_gpu_monitor() {
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    GPU_MONITOR_PID=""
  fi
}
on_exit() {
  local code=$? state=stopped_for_failure_or_repair
  trap - EXIT
  stop_gpu_monitor
  if (( code == 0 && PIPELINE_ACCEPTED == 1 )); then state="${SUCCESS_STATUS}"; else (( code != 0 )) || code=70; fi
  write_status "${state}" "${code}"
  exit "${code}"
}
write_status running 0
trap on_exit EXIT

cd "${REPO}"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 LOGLEVEL=INFO
export HF_HOME="${REPO}/data/hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
export SCENESMITH_RETRIEVAL_DEVICE=cpu SCENESMITH_REQUIRE_GATED_FINAL_ASSEMBLY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RUN_ATTEMPT_ID
exec > >(tee -a "${RUN_DIR}/pipeline.log") 2>&1

if [[ "${EXECUTION_MODE}" == benchmark_processing_hall ]]; then
  : >"${BENCHMARK_EVENTS}"
  : >"${BENCHMARK_GPU_SAMPLES}"
  record_event benchmark_start
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits -l 5 >"${BENCHMARK_GPU_SAMPLES}" 2>&1 &
  GPU_MONITOR_PID=$!
fi

require_index() {
  local root="$1"
  test -s "${root}/clip_embeddings.npy"
  test -s "${root}/embedding_index.yaml"
  test -s "${root}/metadata_index.yaml" || test -s "${root}/metadata_index.json"
}
receipt_status_pass() {
  local receipt="$1"
  [[ -s "${receipt}" ]] || return 1
  python - "${receipt}" <<'PY'
import json, sys
try:
    document = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if document.get("status") == "pass" else 1)
PY
}
probe_openai() {
  local code
  code="$(curl --connect-timeout 10 --max-time 30 -s -o /dev/null -w '%{http_code}' \
    -H "Authorization: Bearer ${OPENAI_API_KEY}" https://api.openai.com/v1/models || true)"
  [[ "${code}" == 200 ]] || { echo "OpenAI production probe failed HTTP ${code}" >&2; return 1; }
}

# Read and bind the factory specification before any generation-capable import.
python scripts/factory_pipeline_code_contract.py \
  --repo-dir "${REPO}" \
  --spec CODEX_FACTORY_FULL_QUALITY_PIPELINE.md \
  --runner remote_jobs/run_full_quality_factory_paracloud.sh \
  --output "${RUN_DIR}/pipeline_code_contract.json"
python scripts/validate_factory_input_manifest.py \
  --input-dir "${INPUT_DIR}" \
  --factory-contract "${FACTORY_CONTRACT}" \
  --output "${RUN_DIR}/factory_input_manifest_validation.json"
python - "${INPUT_DIR}/input_manifest.json" <<'PY'
import json, os, sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
expected={
 "original_prompt_sha256":os.environ.get("EXPECTED_ORIGINAL_PROMPT_SHA256","3bb58cf8e07012cc0c5381eb51730da03b25fb6659a08338238d9fd4baa5093b"),
 "appendix_sha256":os.environ.get("EXPECTED_APPENDIX_SHA256","ffc327f082431d407c15b25b039d7d2676786a03663f186777558e9ba8e1db19"),
 "effective_prompt_sha256":os.environ.get("EXPECTED_EFFECTIVE_PROMPT_SHA256","35387fa3fffb0af92b6c9e07e39f2b59a5443d497190f721090f82bd81734322"),
 "factory_contract_sha256":os.environ.get("EXPECTED_FACTORY_CONTRACT_SHA256","f33e56eef4375d1f68684bd10c02d7d2f82ee72420b8a12570e8fcbe135a94cd")}
if any(m.get(k)!=v for k,v in expected.items()): raise SystemExit("runner/input manifest binding mismatch")
if m.get("reference_image_sha256") is not None: raise SystemExit("factory must not claim a reference image")
PY
record_event code_input_preflight_complete

test -s external/checkpoints/sam3.pt
test -s external/checkpoints/pipeline.yaml
nvidia-smi
VISIBLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( VISIBLE_GPUS >= 1 )) || { echo "no visible GPU" >&2; exit 20; }
test -d "${MATERIALS_DATA}"; require_index "${MATERIALS_SOURCE_EMBEDDINGS}"; require_index "${MATERIALS_EMBEDDINGS}"
python scripts/materials_contract.py validate --data-root "${MATERIALS_DATA}" \
  --source-embeddings "${MATERIALS_SOURCE_EMBEDDINGS}" --contract-embeddings "${MATERIALS_EMBEDDINGS}" \
  --min-retained 1900 --max-pruned 15 --output "${RUN_DIR}/materials_contract_validation.json"
test -d data/artvip_sdf; require_index data/artvip_sdf/embeddings
test -d data/objathor-assets; require_index data/objathor-assets/preprocessed
test -d data/artiverse; require_index data/artiverse/embeddings
if receipt_status_pass "${SAM3D_PREFLIGHT_DIR}/objathor_retrieval_offline.json"; then
  OBJATHOR_PREFLIGHT_MODE=(--verify-only)
else
  OBJATHOR_PREFLIGHT_MODE=()
fi
python scripts/preflight_objathor_retrieval.py --dataset-root data/objathor-assets \
  --preprocessed-path data/objathor-assets/preprocessed \
  --output "${SAM3D_PREFLIGHT_DIR}/objathor_retrieval_offline.json" "${OBJATHOR_PREFLIGHT_MODE[@]}"
python scripts/artiverse_contract.py --dataset-root data/artiverse --embeddings-path data/artiverse/embeddings \
  --output "${RUN_DIR}/artiverse_preparation_validation.json"
if receipt_status_pass "${ARTIVERSE_VISUAL_RECEIPT}"; then
  ARTIVERSE_VISUAL_MODE=(--verify-only)
else
  ARTIVERSE_VISUAL_MODE=()
fi
python scripts/preflight_artiverse_visual_resources.py --dataset-root data/artiverse \
  --embeddings-path data/artiverse/embeddings \
  --runtime-code scenesmith/agent_utils/artiverse_visual_normalization.py \
  --runtime-code scenesmith/agent_utils/asset_manager.py \
  --output "${ARTIVERSE_VISUAL_RECEIPT}" "${ARTIVERSE_VISUAL_MODE[@]}"
python scripts/preflight_sam3d_offline.py --sam3-checkpoint external/checkpoints/sam3.pt \
  --pipeline-config external/checkpoints/pipeline.yaml \
  --output "${SAM3D_PREFLIGHT_DIR}/sam3d_offline_load.json" --verify-only
if receipt_status_pass "${SAM3D_PREFLIGHT_DIR}/sam3d_offline_generation/receipt.json"; then
  python scripts/preflight_sam3d_generation.py --repo-dir "${REPO}" --preflight-dir "${SAM3D_PREFLIGHT_DIR}" \
    --sam3-checkpoint external/checkpoints/sam3.pt --pipeline-config external/checkpoints/pipeline.yaml --verify-only
else
  python scripts/preflight_sam3d_generation.py --repo-dir "${REPO}" --preflight-dir "${SAM3D_PREFLIGHT_DIR}" \
    --sam3-checkpoint external/checkpoints/sam3.pt --pipeline-config external/checkpoints/pipeline.yaml
fi
probe_openai
python -m pytest -q tests/unit/test_artiverse_retrieval_integration.py
python scripts/validate_articulated_router.py --output "${RUN_DIR}/articulated_router_validation.json" \
  --repo-dir "${REPO}" --artiverse-data data/artiverse --artiverse-embeddings data/artiverse/embeddings \
  --artvip-data data/artvip_sdf --artvip-embeddings data/artvip_sdf/embeddings --vlm-backend openai --top-k 3
record_event model_asset_router_preflight_complete

python scripts/run_single_room_worker.py --repo-dir "${REPO}" --run-dir "${RUN_DIR}" \
  --csv "${PROMPT_CSV}" --run-name "${RUN_NAME}_config_preflight" --room-id processing_hall \
  --start-stage furniture --stop-stage manipuland --asset-pipeline generated_sam3d \
  --materials-data "${MATERIALS_DATA}" --materials-source-embeddings "${MATERIALS_SOURCE_EMBEDDINGS}" \
  --materials-embeddings "${MATERIALS_EMBEDDINGS}" --port-offset 0 --render-gpu-id 0 --config-only \
  | tee "${RUN_DIR}/resolved_asset_policy.json"
python scripts/seed_reference_factory_layout.py --repo-dir "${REPO}" --output-root "${OUTPUT_ROOT}" --scene-dir "${SCENE_DIR}" \
  --config "${RUN_DIR}/resolved_config.yaml" --prompt "${INPUT_DIR}/prompt.txt" \
  --expected-prompt-sha256 "${EXPECTED_EFFECTIVE_PROMPT_SHA256}" --profile "${CONTRACT_PROFILE}" \
  --factory-contract "${FACTORY_CONTRACT}"
python scripts/factory_room_contract.py bind-layout --layout "${SCENE_DIR}/house_layout.json" \
  --input-manifest "${INPUT_DIR}/input_manifest.json" --factory-contract "${FACTORY_CONTRACT}" \
  --output "${PROMPT_BINDING}"
python scripts/validate_factory_floor_layout.py --layout "${SCENE_DIR}/house_layout.json" \
  --factory-contract "${FACTORY_CONTRACT}" --output "${SCENE_DIR}/quality_gates/floor_plan_layout.json"
record_event floor_layout_gate_complete

run_room() {
  local room_id="$1" gpu_id="$2" ordinal="$3"
  local room_dir="${SCENE_DIR}/room_${room_id}"
  local final_dir="${room_dir}/scene_states/final_scene"
  local final_state="${final_dir}/scene_state.json" final_blend="${final_dir}/scene.blend"
  local port_offset=$((100 * (ordinal + 1))) resume_stage
  if [[ "${EXECUTION_MODE}" == benchmark_processing_hall ]]; then record_event room_generation_start; fi
  resume_stage="$(python scripts/select_room_resume_stage.py --scene-dir "${SCENE_DIR}" --room-id "${room_id}" --prompt-binding "${PROMPT_BINDING}")"
  if [[ "${resume_stage}" != complete ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_id}" python scripts/run_single_room_worker.py \
      --repo-dir "${REPO}" --run-dir "${RUN_DIR}" --csv "${PROMPT_CSV}" \
      --run-name "${RUN_NAME}_${room_id}" --room-id "${room_id}" --start-stage "${resume_stage}" \
      --stop-stage manipuland --asset-pipeline generated_sam3d --materials-data "${MATERIALS_DATA}" \
      --materials-source-embeddings "${MATERIALS_SOURCE_EMBEDDINGS}" --materials-embeddings "${MATERIALS_EMBEDDINGS}" \
      --port-offset "${port_offset}" --render-gpu-id 0
  fi
  if [[ "${EXECUTION_MODE}" == benchmark_processing_hall ]]; then record_event room_generation_complete; fi
  test -s "${final_state}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" python scripts/run_single_room_worker.py \
    --repo-dir "${REPO}" --run-dir "${RUN_DIR}" --csv "${PROMPT_CSV}" \
    --run-name "${RUN_NAME}_${room_id}_blend_refresh" --room-id "${room_id}" \
    --start-stage furniture --stop-stage manipuland --asset-pipeline generated_sam3d \
    --materials-data "${MATERIALS_DATA}" --materials-source-embeddings "${MATERIALS_SOURCE_EMBEDDINGS}" \
    --materials-embeddings "${MATERIALS_EMBEDDINGS}" --port-offset "${port_offset}" --render-gpu-id 0 --refresh-final-blend
  if [[ "${EXECUTION_MODE}" == benchmark_processing_hall ]]; then record_event blend_refresh_complete; fi
  CUDA_VISIBLE_DEVICES="${gpu_id}" python scripts/render_room_review_views.py --blend "${final_blend}" \
    --scene-state "${final_state}" --room-id "${room_id}" --output-dir "${REVIEW_DIR}"
  if [[ "${EXECUTION_MODE}" == benchmark_processing_hall ]]; then record_event render_complete; fi
  python scripts/room_self_exam.py --scene-dir "${SCENE_DIR}" --review-dir "${REVIEW_DIR}" \
    --output-dir "${DETERMINISTIC_GATE_DIR}" --rooms "${room_id}" --max-collision-hulls 32 \
    --contract-profile "${CONTRACT_PROFILE}" --factory-contract "${FACTORY_CONTRACT}"
  if [[ "${EXECUTION_MODE}" == benchmark_processing_hall ]]; then record_event deterministic_gate_complete; fi
  python scripts/room_visual_self_exam.py --scene-dir "${SCENE_DIR}" --deterministic-gate-dir "${DETERMINISTIC_GATE_DIR}" \
    --review-dir "${REVIEW_DIR}" --output-dir "${VISUAL_GATE_DIR}" --rooms "${room_id}" --threshold 7 \
    --contract-profile "${CONTRACT_PROFILE}" --factory-contract "${FACTORY_CONTRACT}" \
    --effective-prompt "${INPUT_DIR}/prompt.txt" --input-manifest "${INPUT_DIR}/input_manifest.json" \
    --prompt-binding "${PROMPT_BINDING}"
  if [[ "${EXECUTION_MODE}" == benchmark_processing_hall ]]; then record_event visual_gate_complete; fi
  python scripts/room_visual_self_exam.py --scene-dir "${SCENE_DIR}" --deterministic-gate-dir "${DETERMINISTIC_GATE_DIR}" \
    --review-dir "${REVIEW_DIR}" --output-dir "${VISUAL_GATE_DIR}" --rooms "${room_id}" --summarize-existing \
    --threshold 7 --contract-profile "${CONTRACT_PROFILE}" --factory-contract "${FACTORY_CONTRACT}" \
    --effective-prompt "${INPUT_DIR}/prompt.txt" --input-manifest "${INPUT_DIR}/input_manifest.json" \
    --prompt-binding "${PROMPT_BINDING}"
  if [[ "${EXECUTION_MODE}" == benchmark_processing_hall ]]; then record_event saved_gate_revalidation_complete; fi
}

if [[ "${EXECUTION_MODE}" == benchmark_processing_hall ]]; then
  run_room processing_hall 0 0
  python scripts/factory_pipeline_code_contract.py --repo-dir "${REPO}" \
    --spec CODEX_FACTORY_FULL_QUALITY_PIPELINE.md --runner remote_jobs/run_full_quality_factory_paracloud.sh \
    --output "${RUN_DIR}/pipeline_code_contract.json" --verify-only
  python scripts/validate_factory_input_manifest.py --input-dir "${INPUT_DIR}" \
    --factory-contract "${FACTORY_CONTRACT}" --output "${RUN_DIR}/factory_input_manifest_validation.json"
  record_event benchmark_complete
  stop_gpu_monitor
  python scripts/factory_benchmark_receipt.py --run-dir "${RUN_DIR}" --scene-dir "${SCENE_DIR}" \
    --input-dir "${INPUT_DIR}" --factory-contract "${FACTORY_CONTRACT}" --events "${BENCHMARK_EVENTS}" \
    --gpu-samples "${BENCHMARK_GPU_SAMPLES}" --output "${BENCHMARK_RECEIPT}" --run-attempt-id "${RUN_ATTEMPT_ID}"
  PIPELINE_ACCEPTED=1
  exit 0
fi

# One background lane per visible GPU keeps every allocation busy without two rooms sharing a GPU.
GPU_LANES=$(( VISIBLE_GPUS < MAX_PARALLEL_GPUS ? VISIBLE_GPUS : MAX_PARALLEL_GPUS ))
(( GPU_LANES >= 1 ))
pids=()
for ((lane=0; lane<GPU_LANES; lane++)); do
  (
    for ((index=lane; index<${#ROOMS[@]}; index+=GPU_LANES)); do
      run_room "${ROOMS[$index]}" "${lane}" "${index}"
    done
  ) > >(tee -a "${RUN_DIR}/gpu_lane_${lane}.log") 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done

python scripts/room_self_exam.py --scene-dir "${SCENE_DIR}" --review-dir "${REVIEW_DIR}" \
  --output-dir "${DETERMINISTIC_GATE_DIR}" --max-collision-hulls 32 --contract-profile "${CONTRACT_PROFILE}" \
  --factory-contract "${FACTORY_CONTRACT}"
python scripts/room_visual_self_exam.py --scene-dir "${SCENE_DIR}" --deterministic-gate-dir "${DETERMINISTIC_GATE_DIR}" \
  --review-dir "${REVIEW_DIR}" --output-dir "${VISUAL_GATE_DIR}" --summarize-existing --threshold 7 \
  --contract-profile "${CONTRACT_PROFILE}" --factory-contract "${FACTORY_CONTRACT}" \
  --effective-prompt "${INPUT_DIR}/prompt.txt" --input-manifest "${INPUT_DIR}/input_manifest.json" --prompt-binding "${PROMPT_BINDING}"
python scripts/factory_variation_gate.py --scene-dir "${SCENE_DIR}" --review-dir "${REVIEW_DIR}" \
  --factory-contract "${FACTORY_CONTRACT}" --effective-prompt "${INPUT_DIR}/prompt.txt" \
  --input-manifest "${INPUT_DIR}/input_manifest.json" --output "${SCENE_DIR}/quality_gates/factory_variation.json"
python scripts/validate_factory_articulated_motion.py --scene-dir "${SCENE_DIR}" \
  --output "${SCENE_DIR}/quality_gates/factory_articulated_motion.json"
python scripts/assemble_final_house_and_render.py --repo-dir "${REPO}" --run-dir "${RUN_DIR}" --csv "${PROMPT_CSV}" \
  --run-name "${RUN_NAME}_final_assemble" --gate-dir "${VISUAL_GATE_DIR}" --artiverse-data data/artiverse \
  --contract-profile "${CONTRACT_PROFILE}" --factory-contract "${FACTORY_CONTRACT}" --input-dir "${INPUT_DIR}" --render
python scripts/whole_factory_reference_gate.py --scene-dir "${SCENE_DIR}" \
  --review-dir "${SCENE_DIR}/combined_house/outlook_renders" --factory-contract "${FACTORY_CONTRACT}" \
  --effective-prompt "${INPUT_DIR}/prompt.txt" --input-manifest "${INPUT_DIR}/input_manifest.json" \
  --output "${SCENE_DIR}/quality_gates/whole_factory_reference.json" --threshold 7
python scripts/validate_factory_navigation.py --scene-dir "${SCENE_DIR}" \
  --output "${SCENE_DIR}/quality_gates/factory_navigation.json"
python scripts/validate_drake_scene.py --dmd "${SCENE_DIR}/combined_house/house.dmd.yaml" --package-root "${SCENE_DIR}" \
  --output "${SCENE_DIR}/quality_gates/drake_load_paracloud.json" --require-gpus 0 --max-collision-elements 32 \
  --minimum-models 15 --expected-rooms 14
.mujoco_venv/bin/python scripts/export_simulator_artifacts_atomic.py --scene-dir "${SCENE_DIR}" \
  --published-dir "${SCENE_DIR}/mujoco_export" --validation-output "${SCENE_DIR}/quality_gates/simulator_exports.json" \
  --exporter scripts/export_scene_to_mujoco.py --run-attempt-id "${RUN_ATTEMPT_ID}" --require-usd
python tools/sage_scene_checker/check_scenesmith_output.py --scene-dir "${SCENE_DIR}" \
  --out "${SCENE_DIR}/quality_gates/sage_scene_checker.json" --fail-on-warnings

python scripts/assemble_final_house_and_render.py --repo-dir "${REPO}" --run-dir "${RUN_DIR}" --csv "${PROMPT_CSV}" \
  --run-name "${RUN_NAME}_published_verify" --gate-dir "${VISUAL_GATE_DIR}" --artiverse-data data/artiverse \
  --contract-profile "${CONTRACT_PROFILE}" --factory-contract "${FACTORY_CONTRACT}" --input-dir "${INPUT_DIR}" \
  --verify-published-only --render
python scripts/whole_factory_reference_gate.py --scene-dir "${SCENE_DIR}" \
  --review-dir "${SCENE_DIR}/combined_house/outlook_renders" --factory-contract "${FACTORY_CONTRACT}" \
  --effective-prompt "${INPUT_DIR}/prompt.txt" --input-manifest "${INPUT_DIR}/input_manifest.json" \
  --output "${SCENE_DIR}/quality_gates/whole_factory_reference.json" --verify-only \
  --verification-output "${SCENE_DIR}/quality_gates/whole_factory_reference_verification.json"
python scripts/validate_factory_navigation.py --scene-dir "${SCENE_DIR}" \
  --output "${SCENE_DIR}/quality_gates/factory_navigation.json" --verify-only \
  --verification-output "${SCENE_DIR}/quality_gates/factory_navigation_verification.json"
python scripts/factory_variation_gate.py --scene-dir "${SCENE_DIR}" --review-dir "${REVIEW_DIR}" \
  --factory-contract "${FACTORY_CONTRACT}" --effective-prompt "${INPUT_DIR}/prompt.txt" \
  --input-manifest "${INPUT_DIR}/input_manifest.json" --output "${SCENE_DIR}/quality_gates/factory_variation.json" --verify-only
python scripts/validate_factory_articulated_motion.py --scene-dir "${SCENE_DIR}" \
  --output "${SCENE_DIR}/quality_gates/factory_articulated_motion.json" --verify-only \
  --verification-output "${SCENE_DIR}/quality_gates/factory_articulated_motion_verification.json"
python scripts/factory_pipeline_code_contract.py --repo-dir "${REPO}" \
  --spec CODEX_FACTORY_FULL_QUALITY_PIPELINE.md --runner remote_jobs/run_full_quality_factory_paracloud.sh \
  --output "${RUN_DIR}/pipeline_code_contract.json" --verify-only

# Factory siblings are required here because the school acceptance scripts bind
# 11 school rooms and school-only gate labels.
python scripts/factory_final_acceptance_bundle.py --repo-dir "${REPO}" --run-dir "${RUN_DIR}" --scene-dir "${SCENE_DIR}" \
  --input-dir "${INPUT_DIR}" --run-attempt-id "${RUN_ATTEMPT_ID}" \
  --output "${SCENE_DIR}/combined_house/sqz_acceptance_record.json"
python scripts/factory_final_acceptance_bundle.py --repo-dir "${REPO}" --run-dir "${RUN_DIR}" --scene-dir "${SCENE_DIR}" \
  --input-dir "${INPUT_DIR}" --run-attempt-id "${RUN_ATTEMPT_ID}" \
  --output "${SCENE_DIR}/combined_house/sqz_acceptance_record.json" --verify-only
python scripts/factory_two_gpu_drake_acceptance_contract.py create-manifest --package-root "${SCENE_DIR}" \
  --manifest "${RUN_DIR}/scene_000.sha256" --expected-run-attempt-id "${RUN_ATTEMPT_ID}" \
  --output "${RUN_DIR}/package_manifest_validation.json"
python scripts/factory_two_gpu_drake_acceptance_contract.py create-sqz-completion \
  --package-validation "${RUN_DIR}/package_manifest_validation.json" --expected-run-attempt-id "${RUN_ATTEMPT_ID}" \
  --output "${RUN_DIR}/pipeline_completion.json"
PIPELINE_ACCEPTED=1
echo "factory package ready for terminal 2-GPU Drake acceptance"
