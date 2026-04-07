#!/usr/bin/env bash
# Resumable, fail-closed execution of the reference school pipeline.  SQZ is
# the default runtime; the same attested runner can be used on ParaCloud by
# setting SCENESMITH_RUNTIME=paracloud and SCENESMITH_REPO explicitly.
# The script stops at the first failed floor/room/final gate so Codex can repair
# only the failed stage. Re-running resumes completed, still-valid artifacts.

set -euo pipefail

REPO="${SCENESMITH_REPO:-/root/workspace/scenesmith-hts}"
RUN_NAME=full_quality_school_reference_sam3d_artvip_artiverse_20260710
RUN_DIR="${REPO}/outputs/2026-07-10/${RUN_NAME}"
SCENE_DIR="${RUN_DIR}/scene_000"
INPUT_DIR="${REPO}/inputs/full_quality_school_reference_20260710"
PROMPT_CSV="${INPUT_DIR}/prompt.csv"
REFERENCE_IMAGE="${INPUT_DIR}/reference.png"
REVIEW_DIR="${SCENE_DIR}/review/room_review_renders"
DETERMINISTIC_GATE_DIR="${SCENE_DIR}/quality_gates/room_self_exam_deterministic"
VISUAL_GATE_DIR="${SCENE_DIR}/quality_gates/room_self_exam"
STATUS_FILE="${RUN_DIR}/pipeline.status"
COMPLETION_RECEIPT="${RUN_DIR}/pipeline_completion.json"
MATERIALS_DATA="${REPO}/data/materials"
MATERIALS_SOURCE_EMBEDDINGS="${MATERIALS_DATA}/embeddings"
MATERIALS_EMBEDDINGS="${REPO}/data/materials_full_quality_contract/embeddings"
SUCCESS_STATUS=awaiting_2gpu_acceptance
CONTRACT_PROFILE=school_reference_20260710
readonly EXPECTED_ORIGINAL_PROMPT_SHA256=5ddc06e1a9afa60b0417da882c1ec53265eae23f3f9bdd2343360d123923f34c
readonly EXPECTED_EFFECTIVE_PROMPT_SHA256=ac8d297cc9a2d605f41b4bcd7abd52aac29bfd0f195875840342ee1e6a7da86f
readonly EXPECTED_REFERENCE_IMAGE_SHA256=7ba62c39ac98cd2b21d9a5a97fd6f9b90d7efaf447d5ac9a6148524b5a8dbe48
PROMPT_BINDING="${SCENE_DIR}/quality_gates/room_prompt_binding.json"
PIPELINE_ACCEPTED=0
RUN_ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
EXECUTION_MODE="${SCENESMITH_EXECUTION_MODE:-full}"
BENCHMARK_RECEIPT="${RUN_DIR}/classroom_01_full_quality_benchmark.json"
BENCHMARK_EVENTS="${RUN_DIR}/classroom_01_full_quality_benchmark.events.tsv"
BENCHMARK_GPU_SAMPLES="${RUN_DIR}/classroom_01_full_quality_benchmark.gpu.csv"
BENCHMARK_GPU_MONITOR_PID=""
SAM3D_PREFLIGHT_DIR="${REPO}/outputs/preflight/full_quality_school_reference_20260710"
SAM3D_GENERATION_DIR="${SAM3D_PREFLIGHT_DIR}/sam3d_offline_generation"
SAM3D_GENERATION_RECEIPT="${SAM3D_GENERATION_DIR}/receipt.json"
ARTIVERSE_VISUAL_RECEIPT="${RUN_DIR}/artiverse_visual_resources.json"

ROOMS=(
  # Exercise the three compulsory articulated roles first so a retrieval or
  # provenance failure stops before days are spent on the remaining rooms.
  library storage_room classroom_01
  classroom_02 classroom_03 classroom_04 classroom_05 classroom_06
  boys_toilet girls_toilet main_corridor
)

case "${EXECUTION_MODE}" in
  full)
    ;;
  benchmark_classroom_01)
    # This is the smallest measured production path: all shared preflights,
    # floor generation/layout validation, and one dense real SAM3D classroom
    # through blend refresh, three renders, and both room gates.  It exits before
    # all-room summaries, variation, articulation, assembly, or export.
    SUCCESS_STATUS=benchmark_complete
    ROOMS=(classroom_01)
    ;;
  *)
    echo "Unsupported SCENESMITH_EXECUTION_MODE=${EXECUTION_MODE@Q}" >&2
    exit 2
    ;;
esac

mkdir -p "${RUN_DIR}" "${REVIEW_DIR}"
exec 9>"${RUN_DIR}/.pipeline.lock"
if ! flock -n 9; then
  echo "Another full-quality pipeline attempt holds ${RUN_DIR}/.pipeline.lock" >&2
  exit 75
fi

write_status() {
  local state="$1"
  local code="$2"
  local temporary="${STATUS_FILE}.${RUN_ATTEMPT_ID}.tmp"
  printf '%s exit=%s attempt=%s %s\n' \
    "${state}" "${code}" "${RUN_ATTEMPT_ID}" "$(date -Is)" >"${temporary}"
  mv -f "${temporary}" "${STATUS_FILE}"
}

record_benchmark_event() {
  if [[ "${EXECUTION_MODE}" != benchmark_classroom_01 ]]; then
    return 0
  fi
  local event="$1"
  printf '%s\t%s\t%s\n' \
    "$(date +%s)" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${event}" \
    >>"${BENCHMARK_EVENTS}"
}

stop_benchmark_gpu_monitor() {
  if [[ -n "${BENCHMARK_GPU_MONITOR_PID}" ]]; then
    kill "${BENCHMARK_GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${BENCHMARK_GPU_MONITOR_PID}" 2>/dev/null || true
    BENCHMARK_GPU_MONITOR_PID=""
  fi
}

on_pipeline_exit() {
  local code=$?
  local state
  trap - EXIT
  stop_benchmark_gpu_monitor
  if (( code == 0 && PIPELINE_ACCEPTED == 1 )); then
    state="${SUCCESS_STATUS}"
  else
    state=stopped_for_failure_or_repair
    if (( code == 0 )); then
      code=70
    fi
  fi
  write_status "${state}" "${code}"
  exit "${code}"
}

# Install failure accounting before moving any receipt/evidence from a prior
# attempt, so even an archival failure cannot leave a stale success status.
write_status running 0
trap on_pipeline_exit EXIT

PRIOR_ACCEPTANCE_DIR="${RUN_DIR}/prior_acceptance/${RUN_ATTEMPT_ID}"
PRIOR_ACCEPTANCE_PATHS=(
  "${COMPLETION_RECEIPT}"
  "${RUN_DIR}/scene_000.sha256"
  "${RUN_DIR}/package_manifest_validation.json"
  "${SCENE_DIR}/combined_house/sqz_acceptance_record.json"
  "${SCENE_DIR}/quality_gates/final_acceptance_evidence"
)
if [[ "${EXECUTION_MODE}" == full ]]; then
  for prior_path in "${PRIOR_ACCEPTANCE_PATHS[@]}"; do
    if [[ -e "${prior_path}" || -L "${prior_path}" ]]; then
      mkdir -p "${PRIOR_ACCEPTANCE_DIR}"
      mv "${prior_path}" "${PRIOR_ACCEPTANCE_DIR}/"
    fi
  done
elif [[ -e "${SCENE_DIR}/combined_house" || -L "${SCENE_DIR}/combined_house" ]]; then
  echo "[benchmark] refusing to run against an already assembled scene" >&2
  exit 26
else
  BENCHMARK_PRIOR_DIR="${RUN_DIR}/prior_benchmarks/${RUN_ATTEMPT_ID}"
  for prior_benchmark_path in \
    "${BENCHMARK_RECEIPT}" "${BENCHMARK_EVENTS}" "${BENCHMARK_GPU_SAMPLES}"; do
    if [[ -e "${prior_benchmark_path}" || -L "${prior_benchmark_path}" ]]; then
      mkdir -p "${BENCHMARK_PRIOR_DIR}"
      mv "${prior_benchmark_path}" "${BENCHMARK_PRIOR_DIR}/"
    fi
  done
fi

if [[ "${SCENESMITH_RUNTIME:-sqz}" == "paracloud" ]]; then
  PARACLOUD_VENV="${SCENESMITH_PARACLOUD_VENV:-/data/run01/scvj260/scenesmith/.venv}"
  test -r "${PARACLOUD_VENV}/bin/activate"
  source "${PARACLOUD_VENV}/bin/activate"
  if [[ -n "${SCENESMITH_PYTHON_OVERLAY:-}" ]]; then
    test -d "${SCENESMITH_PYTHON_OVERLAY}"
    export PYTHONPATH="${SCENESMITH_PYTHON_OVERLAY}:${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
  else
    export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
  fi
  agent_proxy() {
    local proxy="${SCENESMITH_AGENT_HTTP_PROXY:-http://ln08:18092}"
    export HTTP_PROXY="${proxy}" HTTPS_PROXY="${proxy}"
    export http_proxy="${proxy}" https_proxy="${proxy}"
    export NO_PROXY="127.0.0.1,localhost,dashscope.aliyuncs.com,hf-mirror.com"
    export no_proxy="${NO_PROXY}"
  }
else
  source /root/workspace/Humanoid-Training-Scene-Ready-Generation/local_setup/setup_env_sqz.sh
fi
cd "${REPO}"
set -a
source .env
set +a
if [[ "${SCENESMITH_RUNTIME:-sqz}" == "paracloud" ]]; then
  # Keep the immutable ObjectThor OpenCLIP checkpoint in Codex's owned
  # ParaCloud workspace.  Production stays offline; this merely selects the
  # pre-populated cache that the preflight must re-verify on every run.
  export HF_HOME="${SCENESMITH_HF_HOME:-/data/run01/scvj260/codex_factory/huggingface}"
  # The checkpoint was downloaded with an explicit ``cache_dir=$HF_HOME``;
  # preserve that layout rather than duplicating a 1.7-GB immutable blob.
  export HF_HUB_CACHE="${HF_HOME}"
  export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
  # SAM3D's DINO dependency is resolved through torch.hub rather than the
  # Hugging Face cache.  Pin it to the same owned, pre-verified ParaCloud
  # runtime cache so offline proof generation cannot silently use a shared
  # mutable cache or reach the network.
  export TORCH_HOME="${SCENESMITH_TORCH_HOME:-/data/run01/scvj260/codex_factory/torch}"
fi
agent_proxy
# SceneSmith/OpenAI uses httpx without the optional SOCKS extra. The HTTP reverse
# proxy is sufficient; leaving ALL_PROXY=socks5h would fail before the first VLM call.
unset ALL_PROXY all_proxy

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export LOGLEVEL=INFO
export SCENESMITH_RETRIEVAL_DEVICE=cpu
export SCENESMITH_REQUIRE_GATED_FINAL_ASSEMBLY=1
export RUN_ATTEMPT_ID
# Every required model must already be present. Hidden Hugging Face downloads are
# a preflight failure, not something a multi-day production run may discover.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1

exec > >(tee -a "${RUN_DIR}/pipeline.log") 2>&1

if [[ "${EXECUTION_MODE}" == benchmark_classroom_01 ]]; then
  : >"${BENCHMARK_EVENTS}"
  : >"${BENCHMARK_GPU_SAMPLES}"
  record_benchmark_event benchmark_start
  nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits \
    -l 5 >"${BENCHMARK_GPU_SAMPLES}" 2>&1 &
  BENCHMARK_GPU_MONITOR_PID=$!
fi

complete_classroom_benchmark() {
  python scripts/preflight_artiverse_visual_resources.py \
    --dataset-root data/artiverse \
    --embeddings-path data/artiverse/embeddings \
    --runtime-code scenesmith/agent_utils/artiverse_visual_normalization.py \
    --runtime-code scenesmith/agent_utils/asset_manager.py \
    --output "${ARTIVERSE_VISUAL_RECEIPT}" \
    --verify-only
  python scripts/preflight_sam3d_offline.py \
    --sam3-checkpoint external/checkpoints/sam3.pt \
    --pipeline-config external/checkpoints/pipeline.yaml \
    --output "${SAM3D_PREFLIGHT_DIR}/sam3d_offline_load.json" \
    --verify-only
  python scripts/preflight_sam3d_generation.py \
    --repo-dir "${REPO}" \
    --preflight-dir "${SAM3D_PREFLIGHT_DIR}" \
    --sam3-checkpoint external/checkpoints/sam3.pt \
    --pipeline-config external/checkpoints/pipeline.yaml \
    --verify-only
  python scripts/pipeline_code_contract.py \
    --repo-dir "${REPO}" \
    --spec CODEX_SCENESMITH_FULL_QUALITY_PIPELINE.md \
    --runner remote_jobs/run_full_quality_school_sqz.sh \
    --output "${RUN_DIR}/pipeline_code_contract.json" \
    --verify-only
  python scripts/validate_input_manifest.py \
    --input-dir "${INPUT_DIR}" \
    --output "${RUN_DIR}/input_manifest_validation.json"
  record_benchmark_event benchmark_complete
  stop_benchmark_gpu_monitor
  python scripts/classroom_benchmark_receipt.py \
    --repo-dir "${REPO}" \
    --run-dir "${RUN_DIR}" \
    --scene-dir "${SCENE_DIR}" \
    --input-dir "${INPUT_DIR}" \
    --events "${BENCHMARK_EVENTS}" \
    --gpu-samples "${BENCHMARK_GPU_SAMPLES}" \
    --output "${BENCHMARK_RECEIPT}" \
    --run-attempt-id "${RUN_ATTEMPT_ID}"
  PIPELINE_ACCEPTED=1
  echo "[benchmark complete] receipt=${BENCHMARK_RECEIPT}; full-house stages were not started"
  exit 0
}

# The user requires the current full-quality specification to be read before
# generation code. This proof fully reads it first, then binds the runner,
# scripts, SAGE checker, remote jobs, and exact ordered upstream patch stack.
python scripts/pipeline_code_contract.py \
  --repo-dir "${REPO}" \
  --spec CODEX_SCENESMITH_FULL_QUALITY_PIPELINE.md \
  --runner remote_jobs/run_full_quality_school_sqz.sh \
  --output "${RUN_DIR}/pipeline_code_contract.json"

probe_openai() {
  local attempt=0
  local code
  while true; do
    attempt=$((attempt + 1))
    code="$(curl --connect-timeout 10 --max-time 30 --proxy "${HTTP_PROXY}" \
      -s -o /dev/null -w '%{http_code}' \
      -H "Authorization: Bearer ${OPENAI_API_KEY}" \
      https://api.openai.com/v1/models || true)"
    echo "[preflight] OpenAI authenticated HTTP ${code} attempt=${attempt}"
    case "${code}" in
      200)
        return 0
        ;;
      401|403)
        echo "[preflight] OpenAI authentication/authorization failed; refusing to retry with the same credential." >&2
        return 1
        ;;
      *)
        echo "[preflight] OpenAI path is temporarily unavailable; waiting 20s for the VPN/reverse tunnel supervisor." >&2
        sleep 20
        ;;
    esac
  done
}

require_index() {
  local directory="$1"
  test -s "${directory}/clip_embeddings.npy"
  test -s "${directory}/embedding_index.yaml"
  test -s "${directory}/metadata_index.yaml" || test -s "${directory}/metadata_index.json"
}

echo "[preflight] $(date -Is) run=${RUN_NAME}"
test -s "${PROMPT_CSV}"
test -s "${REFERENCE_IMAGE}"
test -s "${INPUT_DIR}/input_manifest.json"
python scripts/validate_input_manifest.py \
  --input-dir "${INPUT_DIR}" \
  --output "${RUN_DIR}/input_manifest_validation.json"
python - "${RUN_DIR}/input_manifest_validation.json" \
  "${EXPECTED_ORIGINAL_PROMPT_SHA256}" \
  "${EXPECTED_EFFECTIVE_PROMPT_SHA256}" \
  "${EXPECTED_REFERENCE_IMAGE_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "original_prompt_sha256": sys.argv[2],
    "effective_prompt_sha256": sys.argv[3],
    "reference_image_sha256": sys.argv[4],
}
if result.get("status") != "pass":
    raise SystemExit("immutable input validation did not pass")
external = result.get("external_input_binding", {})
if any(external.get(key) != value for key, value in expected.items()):
    raise SystemExit("validator did not attest the runner's external input binding")
if any(result.get("evidence", {}).get(key) != value for key, value in expected.items()):
    raise SystemExit("staged inputs differ from the runner's external input binding")
PY
test -s external/checkpoints/sam3.pt
test -s external/checkpoints/pipeline.yaml
test -x .mujoco_venv/bin/python
.mujoco_venv/bin/python - <<'PY'
import mujoco
import mujoco_usd_converter
import usdex.core
from pxr import Usd
print("MuJoCo/OpenUSD isolated runtime preflight passed")
PY
test -d "${MATERIALS_DATA}"
require_index "${MATERIALS_SOURCE_EMBEDDINGS}"
require_index "${MATERIALS_EMBEDDINGS}"
python scripts/materials_contract.py validate \
  --data-root "${MATERIALS_DATA}" \
  --source-embeddings "${MATERIALS_SOURCE_EMBEDDINGS}" \
  --contract-embeddings "${MATERIALS_EMBEDDINGS}" \
  --min-retained 1900 \
  --max-pruned 15 \
  --output "${RUN_DIR}/materials_contract_validation.json"
test -d data/artvip_sdf
require_index data/artvip_sdf/embeddings
test -d data/objathor-assets
require_index data/objathor-assets/preprocessed
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python scripts/preflight_objathor_retrieval.py \
  --dataset-root data/objathor-assets \
  --preprocessed-path data/objathor-assets/preprocessed \
  --output outputs/preflight/full_quality_school_reference_20260710/objathor_retrieval_offline.json \
  --verify-only
if [[ ! -d data/artiverse ]]; then
  echo "[blocker] compulsory Artiverse is not installed; authorize Hugging Face and run remote_jobs/download_prepare_artiverse_sqz.sh" >&2
  exit 21
fi
if ! require_index data/artiverse/embeddings; then
  echo "[blocker] compulsory Artiverse indexes are missing or incomplete; rerun its pinned preparation job" >&2
  exit 22
fi
python scripts/artiverse_contract.py \
  --dataset-root data/artiverse \
  --embeddings-path data/artiverse/embeddings \
  --output "${RUN_DIR}/artiverse_preparation_validation.json"
if [[ -s "${ARTIVERSE_VISUAL_RECEIPT}" ]]; then
  echo "[preflight] reverifying all Artiverse publisher-GLB to external-glTF derivations"
  python scripts/preflight_artiverse_visual_resources.py \
    --dataset-root data/artiverse \
    --embeddings-path data/artiverse/embeddings \
    --runtime-code scenesmith/agent_utils/artiverse_visual_normalization.py \
    --runtime-code scenesmith/agent_utils/asset_manager.py \
    --output "${ARTIVERSE_VISUAL_RECEIPT}" \
    --verify-only
elif [[ -e "${ARTIVERSE_VISUAL_RECEIPT}" || -L "${ARTIVERSE_VISUAL_RECEIPT}" ]]; then
  echo "[blocker] incomplete/stale Artiverse visual-resource receipt: ${ARTIVERSE_VISUAL_RECEIPT}" >&2
  exit 28
else
  echo "[preflight] auditing all prepared Artiverse external-glTF derivations before any paid room call"
  python scripts/preflight_artiverse_visual_resources.py \
    --dataset-root data/artiverse \
    --embeddings-path data/artiverse/embeddings \
    --runtime-code scenesmith/agent_utils/artiverse_visual_normalization.py \
    --runtime-code scenesmith/agent_utils/asset_manager.py \
    --output "${ARTIVERSE_VISUAL_RECEIPT}"
fi
record_benchmark_event artiverse_visual_preflight_complete
python scripts/preflight_sam3d_offline.py \
  --sam3-checkpoint external/checkpoints/sam3.pt \
  --pipeline-config external/checkpoints/pipeline.yaml \
  --output outputs/preflight/full_quality_school_reference_20260710/sam3d_offline_load.json \
  --verify-only
if [[ -s "${SAM3D_GENERATION_RECEIPT}" ]]; then
  echo "[preflight] reverifying the existing full offline SAM3D GLB proof"
  python scripts/preflight_sam3d_generation.py \
    --repo-dir "${REPO}" \
    --preflight-dir "${SAM3D_PREFLIGHT_DIR}" \
    --sam3-checkpoint external/checkpoints/sam3.pt \
    --pipeline-config external/checkpoints/pipeline.yaml \
    --verify-only
elif [[ -e "${SAM3D_GENERATION_DIR}" || -L "${SAM3D_GENERATION_DIR}" ]]; then
  echo "[preflight] incomplete/stale SAM3D generation directory exists without a receipt: ${SAM3D_GENERATION_DIR}" >&2
  exit 27
else
  echo "[preflight] creating the one-time full offline SAM3D GLB proof before any paid API probe"
  python scripts/preflight_sam3d_generation.py \
    --repo-dir "${REPO}" \
    --preflight-dir "${SAM3D_PREFLIGHT_DIR}" \
    --sam3-checkpoint external/checkpoints/sam3.pt \
    --pipeline-config external/checkpoints/pipeline.yaml
fi
record_benchmark_event sam3d_generation_preflight_complete
python - <<'PY'
import hashlib
import json
from pathlib import Path

reference = Path("inputs/full_quality_school_reference_20260710/reference.png")
result = json.loads(
    Path("outputs/preflight/full_quality_school_reference_20260710/vlm_vision_smoke.json").read_text()
)
actual_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
if result.get("status") != "pass" or result.get("image_sha256") != actual_hash:
    raise SystemExit(f"VLM vision smoke is missing, failed, or stale: {result}")
PY
probe_openai
nvidia-smi

echo "[preflight] compulsory Artiverse and ArtVIP router validation"
test -s tests/unit/test_artiverse_retrieval_integration.py
python -m pytest -q tests/unit/test_artiverse_retrieval_integration.py
python scripts/validate_articulated_router.py \
  --output "${RUN_DIR}/articulated_router_validation.json" \
  --repo-dir "${REPO}" \
  --artiverse-data data/artiverse \
  --artiverse-embeddings data/artiverse/embeddings \
  --artvip-data data/artvip_sdf \
  --artvip-embeddings data/artvip_sdf/embeddings \
  --vlm-backend openai \
  --top-k 3

echo "[preflight] resolved generated-SAM3D policy"
python scripts/run_single_room_worker.py \
  --repo-dir "${REPO}" \
  --run-dir "${RUN_DIR}" \
  --csv "${PROMPT_CSV}" \
  --run-name "${RUN_NAME}_config_preflight" \
  --room-id classroom_01 \
  --start-stage furniture \
  --stop-stage manipuland \
  --asset-pipeline generated_sam3d \
  --materials-data "${MATERIALS_DATA}" \
  --materials-source-embeddings "${MATERIALS_SOURCE_EMBEDDINGS}" \
  --materials-embeddings "${MATERIALS_EMBEDDINGS}" \
  --port-offset 0 \
  --render-gpu-id 0 \
  --config-only \
  | tee "${RUN_DIR}/resolved_asset_policy.json"
record_benchmark_event preflight_complete

echo "[floor_plan] materializing or revalidating the exact native SceneSmith reference layout"
test -s "${RUN_DIR}/resolved_config.yaml"
python scripts/seed_reference_school_layout.py \
  --repo-dir "${REPO}" \
  --scene-dir "${SCENE_DIR}" \
  --config "${RUN_DIR}/resolved_config.yaml" \
  --prompt "${INPUT_DIR}/prompt.txt" \
  --expected-prompt-sha256 "${EXPECTED_EFFECTIVE_PROMPT_SHA256}" \
  --profile "${CONTRACT_PROFILE}"

echo "[floor_plan] binding immutable room-specific generation prompts"
python scripts/school_room_contract.py bind-layout \
  --layout "${SCENE_DIR}/house_layout.json" \
  --input-manifest "${INPUT_DIR}/input_manifest.json" \
  --output "${PROMPT_BINDING}"

echo "[floor_plan] enforcing exact reference-relative layout"
python scripts/validate_school_floor_layout.py \
  --layout "${SCENE_DIR}/house_layout.json" \
  --output "${SCENE_DIR}/quality_gates/floor_plan_layout.json"
record_benchmark_event floor_layout_gate_complete

for index in "${!ROOMS[@]}"; do
  room_id="${ROOMS[$index]}"
  room_dir="${SCENE_DIR}/room_${room_id}"
  final_dir="${room_dir}/scene_states/final_scene"
  final_state="${final_dir}/scene_state.json"
  final_blend="${final_dir}/scene.blend"
  port_offset=$((100 * (index + 1)))
  record_benchmark_event room_start

  # A rerun first rehashes every artifact bound by an existing passing visual
  # decision. Only a byte-identical pass may skip blend refresh, rendering,
  # deterministic checks, and the paid VLM call; the final all-room summary
  # independently verifies every room again before assembly.
  if [[ -s "${VISUAL_GATE_DIR}/${room_id}.json" ]] && \
    python scripts/room_visual_self_exam.py \
      --scene-dir "${SCENE_DIR}" \
      --deterministic-gate-dir "${DETERMINISTIC_GATE_DIR}" \
      --review-dir "${REVIEW_DIR}" \
      --reference-image "${REFERENCE_IMAGE}" \
      --output-dir "${VISUAL_GATE_DIR}" \
      --rooms "${room_id}" \
      --summarize-existing \
      --threshold 7 \
      --contract-profile "${CONTRACT_PROFILE}" \
      --effective-prompt "${INPUT_DIR}/prompt.txt" \
      --input-manifest "${INPUT_DIR}/input_manifest.json" \
      --prompt-binding "${PROMPT_BINDING}"; then
    echo "[room ${room_id}] exact passing evidence is unchanged; reusing it without rendering or VLM"
    if [[ "${EXECUTION_MODE}" == benchmark_classroom_01 ]]; then
      record_benchmark_event room_evidence_reused
      complete_classroom_benchmark
    fi
    continue
  fi

  record_benchmark_event room_generation_start
  resume_stage="$(python scripts/select_room_resume_stage.py \
    --scene-dir "${SCENE_DIR}" \
    --room-id "${room_id}" \
    --prompt-binding "${PROMPT_BINDING}")"
  if [[ "${resume_stage}" != complete ]]; then
    echo "[room ${room_id}] generating from highest bound checkpoint stage=${resume_stage}"
    probe_openai
    python scripts/run_single_room_worker.py \
      --repo-dir "${REPO}" \
      --run-dir "${RUN_DIR}" \
      --csv "${PROMPT_CSV}" \
      --run-name "${RUN_NAME}_${room_id}" \
      --room-id "${room_id}" \
      --start-stage "${resume_stage}" \
      --stop-stage manipuland \
      --asset-pipeline generated_sam3d \
      --materials-data "${MATERIALS_DATA}" \
      --materials-source-embeddings "${MATERIALS_SOURCE_EMBEDDINGS}" \
      --materials-embeddings "${MATERIALS_EMBEDDINGS}" \
      --port-offset "${port_offset}" \
      --render-gpu-id 0
  else
    echo "[room ${room_id}] final state is prompt-bound and complete; rebuilding its evidence without regeneration"
  fi
  record_benchmark_event room_generation_complete

  test -s "${final_state}"
  echo "[room ${room_id}] atomically binding the review blend to the exact current state"
  python scripts/run_single_room_worker.py \
    --repo-dir "${REPO}" \
    --run-dir "${RUN_DIR}" \
    --csv "${PROMPT_CSV}" \
    --run-name "${RUN_NAME}_${room_id}_blend_refresh" \
    --room-id "${room_id}" \
    --start-stage furniture \
    --stop-stage manipuland \
    --asset-pipeline generated_sam3d \
    --materials-data "${MATERIALS_DATA}" \
    --materials-source-embeddings "${MATERIALS_SOURCE_EMBEDDINGS}" \
    --materials-embeddings "${MATERIALS_EMBEDDINGS}" \
    --port-offset "${port_offset}" \
    --render-gpu-id 0 \
    --refresh-final-blend
  test -s "${final_blend}"
  record_benchmark_event blend_refresh_complete

  echo "[room ${room_id}] rendering top and two oblique review views"
  python scripts/render_room_review_views.py \
    --blend "${final_blend}" \
    --scene-state "${final_state}" \
    --room-id "${room_id}" \
    --output-dir "${REVIEW_DIR}"
  record_benchmark_event render_complete

  echo "[room ${room_id}] deterministic gate"
  python scripts/room_self_exam.py \
    --scene-dir "${SCENE_DIR}" \
    --review-dir "${REVIEW_DIR}" \
    --output-dir "${DETERMINISTIC_GATE_DIR}" \
    --rooms "${room_id}" \
    --max-collision-hulls 32 \
    --contract-profile "${CONTRACT_PROFILE}"
  record_benchmark_event deterministic_gate_complete

  echo "[room ${room_id}] image-aware reference gate"
  probe_openai
  python scripts/room_visual_self_exam.py \
    --scene-dir "${SCENE_DIR}" \
    --deterministic-gate-dir "${DETERMINISTIC_GATE_DIR}" \
    --review-dir "${REVIEW_DIR}" \
    --reference-image "${REFERENCE_IMAGE}" \
    --output-dir "${VISUAL_GATE_DIR}" \
    --rooms "${room_id}" \
    --threshold 7 \
    --contract-profile "${CONTRACT_PROFILE}" \
    --effective-prompt "${INPUT_DIR}/prompt.txt" \
    --input-manifest "${INPUT_DIR}/input_manifest.json" \
    --prompt-binding "${PROMPT_BINDING}"
  record_benchmark_event visual_gate_complete
  if [[ "${EXECUTION_MODE}" == benchmark_classroom_01 ]]; then
    # Rehash the newly written decision through the same production verifier
    # before creating timing evidence and exiting before all global stages.
    python scripts/room_visual_self_exam.py \
      --scene-dir "${SCENE_DIR}" \
      --deterministic-gate-dir "${DETERMINISTIC_GATE_DIR}" \
      --review-dir "${REVIEW_DIR}" \
      --reference-image "${REFERENCE_IMAGE}" \
      --output-dir "${VISUAL_GATE_DIR}" \
      --rooms "${room_id}" \
      --summarize-existing \
      --threshold 7 \
      --contract-profile "${CONTRACT_PROFILE}" \
      --effective-prompt "${INPUT_DIR}/prompt.txt" \
      --input-manifest "${INPUT_DIR}/input_manifest.json" \
      --prompt-binding "${PROMPT_BINDING}"
    complete_classroom_benchmark
  fi
done

echo "[gate] confirming every room gate together"
python scripts/room_self_exam.py \
  --scene-dir "${SCENE_DIR}" \
  --review-dir "${REVIEW_DIR}" \
  --output-dir "${DETERMINISTIC_GATE_DIR}" \
  --max-collision-hulls 32 \
  --contract-profile "${CONTRACT_PROFILE}"
python scripts/room_visual_self_exam.py \
  --scene-dir "${SCENE_DIR}" \
  --deterministic-gate-dir "${DETERMINISTIC_GATE_DIR}" \
  --review-dir "${REVIEW_DIR}" \
  --reference-image "${REFERENCE_IMAGE}" \
  --output-dir "${VISUAL_GATE_DIR}" \
  --summarize-existing \
  --threshold 7 \
  --contract-profile "${CONTRACT_PROFILE}" \
  --effective-prompt "${INPUT_DIR}/prompt.txt" \
  --input-manifest "${INPUT_DIR}/input_manifest.json" \
  --prompt-binding "${PROMPT_BINDING}"

echo "[gate] proving all six classrooms are spatially and visually distinct"
CLASSROOM_VARIATION_OUTPUT="${SCENE_DIR}/quality_gates/classroom_variation.json"
if [[ -s "${CLASSROOM_VARIATION_OUTPUT}" ]] && \
  python scripts/classroom_variation_gate.py \
    --scene-dir "${SCENE_DIR}" \
    --review-dir "${REVIEW_DIR}" \
    --output "${CLASSROOM_VARIATION_OUTPUT}" \
    --vlm-backend openai \
    --verify-only; then
  echo "[gate] reusing hash-verified classroom-variation decision without a VLM call"
else
  probe_openai
  python scripts/classroom_variation_gate.py \
    --scene-dir "${SCENE_DIR}" \
    --review-dir "${REVIEW_DIR}" \
    --output "${CLASSROOM_VARIATION_OUTPUT}" \
    --vlm-backend openai
fi

echo "[gate] exercising all three required articulated furniture roles in Drake"
python scripts/validate_articulated_motion.py \
  --scene-dir "${SCENE_DIR}" \
  --output "${SCENE_DIR}/quality_gates/articulated_motion.json"

echo "[assemble] all room gates passed; enforcing Artiverse survival during assembly"
python scripts/assemble_final_house_and_render.py \
  --repo-dir "${REPO}" \
  --run-dir "${RUN_DIR}" \
  --csv "${PROMPT_CSV}" \
  --run-name "${RUN_NAME}_final_assemble" \
  --gate-dir "${VISUAL_GATE_DIR}" \
  --artiverse-data data/artiverse \
  --contract-profile "${CONTRACT_PROFILE}" \
  --input-dir "${INPUT_DIR}" \
  --render

echo "[final gate] comparing the final floor with the supplied reference"
probe_openai
python scripts/whole_floor_reference_gate.py \
  --scene-dir "${SCENE_DIR}" \
  --reference-image "${REFERENCE_IMAGE}" \
  --input-manifest "${INPUT_DIR}/input_manifest.json" \
  --output "${SCENE_DIR}/quality_gates/whole_floor_reference.json" \
  --threshold 7

echo "[final gate] proving collision-aware humanoid routes to all 11 rooms"
python scripts/validate_school_navigation.py \
  --scene-dir "${SCENE_DIR}" \
  --output "${SCENE_DIR}/quality_gates/school_navigation.json"

echo "[validate] loading final Drake directives and reporting collision complexity"
python scripts/validate_drake_scene.py \
  --dmd "${SCENE_DIR}/combined_house/house.dmd.yaml" \
  --package-root "${SCENE_DIR}" \
  --output "${SCENE_DIR}/quality_gates/drake_load_sqz.json" \
  --require-gpus 0 \
  --max-collision-elements 32 \
  --minimum-models 12 \
  --expected-rooms 11

echo "[export] MuJoCo and USD"
.mujoco_venv/bin/python scripts/export_simulator_artifacts_atomic.py \
  --scene-dir "${SCENE_DIR}" \
  --published-dir "${SCENE_DIR}/mujoco_export" \
  --validation-output "${SCENE_DIR}/quality_gates/simulator_exports.json" \
  --exporter scripts/export_scene_to_mujoco.py \
  --run-attempt-id "${RUN_ATTEMPT_ID}" \
  --require-usd

echo "[validate] SAGE-style structural checks"
python tools/sage_scene_checker/check_scenesmith_output.py \
  --scene-dir "${SCENE_DIR}" \
  --out "${SCENE_DIR}/quality_gates/sage_scene_checker.json" \
  --fail-on-warnings

echo "[final evidence] revalidating published Artiverse lineage and gate bindings"
python scripts/assemble_final_house_and_render.py \
  --repo-dir "${REPO}" \
  --run-dir "${RUN_DIR}" \
  --csv "${PROMPT_CSV}" \
  --run-name "${RUN_NAME}_published_verify" \
  --gate-dir "${VISUAL_GATE_DIR}" \
  --artiverse-data data/artiverse \
  --contract-profile "${CONTRACT_PROFILE}" \
  --input-dir "${INPUT_DIR}" \
  --verify-published-only \
  --render
python scripts/artiverse_contract.py \
  --dataset-root data/artiverse \
  --embeddings-path data/artiverse/embeddings \
  --usage-manifest "${SCENE_DIR}/combined_house/artiverse_usage.json" \
  --house-state "${SCENE_DIR}/combined_house/house_state.json" \
  --scene-dir "${SCENE_DIR}" \
  --output "${SCENE_DIR}/quality_gates/artiverse_final_validation.json"
python scripts/whole_floor_reference_gate.py \
  --scene-dir "${SCENE_DIR}" \
  --reference-image "${REFERENCE_IMAGE}" \
  --input-manifest "${INPUT_DIR}/input_manifest.json" \
  --output "${SCENE_DIR}/quality_gates/whole_floor_reference.json" \
  --threshold 7 \
  --verify-only
python scripts/validate_school_navigation.py \
  --scene-dir "${SCENE_DIR}" \
  --output "${SCENE_DIR}/quality_gates/school_navigation.json" \
  --verification-output "${SCENE_DIR}/quality_gates/school_navigation_verification.json" \
  --verify-only
python scripts/classroom_variation_gate.py \
  --scene-dir "${SCENE_DIR}" \
  --review-dir "${REVIEW_DIR}" \
  --output "${SCENE_DIR}/quality_gates/classroom_variation.json" \
  --verify-only
python scripts/validate_articulated_motion.py \
  --scene-dir "${SCENE_DIR}" \
  --output "${SCENE_DIR}/quality_gates/articulated_motion.json" \
  --verification-output "${SCENE_DIR}/quality_gates/articulated_motion_verification.json" \
  --verify-only
python scripts/materials_contract.py validate \
  --data-root "${MATERIALS_DATA}" \
  --source-embeddings "${MATERIALS_SOURCE_EMBEDDINGS}" \
  --contract-embeddings "${MATERIALS_EMBEDDINGS}" \
  --min-retained 1900 \
  --max-pruned 15 \
  --output "${RUN_DIR}/materials_contract_validation.json"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python scripts/preflight_objathor_retrieval.py \
  --dataset-root data/objathor-assets \
  --preprocessed-path data/objathor-assets/preprocessed \
  --output outputs/preflight/full_quality_school_reference_20260710/objathor_retrieval_offline.json \
  --verify-only
python scripts/preflight_sam3d_offline.py \
  --sam3-checkpoint external/checkpoints/sam3.pt \
  --pipeline-config external/checkpoints/pipeline.yaml \
  --output "${SAM3D_PREFLIGHT_DIR}/sam3d_offline_load.json" \
  --verify-only
python scripts/preflight_sam3d_generation.py \
  --repo-dir "${REPO}" \
  --preflight-dir "${SAM3D_PREFLIGHT_DIR}" \
  --sam3-checkpoint external/checkpoints/sam3.pt \
  --pipeline-config external/checkpoints/pipeline.yaml \
  --verify-only
python scripts/preflight_artiverse_visual_resources.py \
  --dataset-root data/artiverse \
  --embeddings-path data/artiverse/embeddings \
  --runtime-code scenesmith/agent_utils/artiverse_visual_normalization.py \
  --runtime-code scenesmith/agent_utils/asset_manager.py \
  --output "${ARTIVERSE_VISUAL_RECEIPT}" \
  --verify-only
python scripts/pipeline_code_contract.py \
  --repo-dir "${REPO}" \
  --spec CODEX_SCENESMITH_FULL_QUALITY_PIPELINE.md \
  --runner remote_jobs/run_full_quality_school_sqz.sh \
  --output "${RUN_DIR}/pipeline_code_contract.json" \
  --verify-only

echo "[acceptance] building the immutable in-package SQZ evidence record"
python scripts/final_acceptance_bundle.py \
  --repo-dir "${REPO}" \
  --run-dir "${RUN_DIR}" \
  --scene-dir "${SCENE_DIR}" \
  --input-dir "${INPUT_DIR}" \
  --run-attempt-id "${RUN_ATTEMPT_ID}" \
  --output "${SCENE_DIR}/combined_house/sqz_acceptance_record.json"
python scripts/final_acceptance_bundle.py \
  --repo-dir "${REPO}" \
  --run-dir "${RUN_DIR}" \
  --scene-dir "${SCENE_DIR}" \
  --input-dir "${INPUT_DIR}" \
  --run-attempt-id "${RUN_ATTEMPT_ID}" \
  --output "${SCENE_DIR}/combined_house/sqz_acceptance_record.json" \
  --verify-only

echo "[acceptance] creating the external exact-file package manifest"
python scripts/two_gpu_drake_acceptance_contract.py create-manifest \
  --package-root "${SCENE_DIR}" \
  --manifest "${RUN_DIR}/scene_000.sha256" \
  --expected-run-attempt-id "${RUN_ATTEMPT_ID}" \
  --output "${RUN_DIR}/package_manifest_validation.json"

echo "[acceptance] writing the outside-scene pending 2-GPU completion receipt"
python scripts/two_gpu_drake_acceptance_contract.py create-sqz-completion \
  --package-validation "${RUN_DIR}/package_manifest_validation.json" \
  --expected-run-attempt-id "${RUN_ATTEMPT_ID}" \
  --output "${COMPLETION_RECEIPT}"

PIPELINE_ACCEPTED=1

echo "[sqz precheck complete] $(date -Is) ${RUN_DIR}; awaiting required 2-GPU Drake acceptance"
