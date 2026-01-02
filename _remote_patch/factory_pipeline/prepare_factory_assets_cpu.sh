#!/usr/bin/env bash
# Resumable preparation of factory retrieval inputs on ParaCloud. Extraction
# is CPU/I/O bound; the final 500-item Artiverse CLIP index uses one GPU.
set -euo pipefail

readonly REPO=/data/run01/scvj260/scenesmith-factory-codex
readonly OUTPUT_ROOT=/data/run01/scvj260/factory_outputs
readonly PREFLIGHT_DIR="${OUTPUT_ROOT}/preflight/full_quality_factory_sam3d_artvip_artiverse_20260713"
readonly RUN_DIR="${OUTPUT_ROOT}/full_quality_factory_sam3d_artvip_artiverse_20260713"
readonly ARTIVERSE_ROOT="${REPO}/data/artiverse"
readonly ARTIVERSE_SOURCE=/data/run01/scvj260/hts_migration/archives/artiverse
readonly CHUNKS_SOURCE="${ARTIVERSE_SOURCE}/dataset_chunks"
readonly MODEL_REVISION=1627032197142fbe2a7cfec626f4ced3ae60d07a
readonly MODEL_FILENAME=open_clip_pytorch_model.safetensors
readonly MODEL_SHA256=7d129ed747e0ed53e82dfcc140382b51be66b56e6a9bdc3258afd2846e3bb019
readonly STATUS="${PREFLIGHT_DIR}/factory_asset_prepare.status"

mkdir -p "${PREFLIGHT_DIR}" "${RUN_DIR}" "${ARTIVERSE_ROOT}/dataset_chunks" "${REPO}/data/hf_cache/hub"
exec 9>"${PREFLIGHT_DIR}/.factory_asset_prepare.lock"
flock -n 9 || { echo "another factory asset-preparation job holds the lock" >&2; exit 75; }

write_status() {
  local state="$1" code="$2" tmp="${STATUS}.$$.tmp"
  printf '%s exit=%s %s\n' "${state}" "${code}" "$(date -Is)" >"${tmp}"
  mv -f "${tmp}" "${STATUS}"
}
on_exit() {
  local code=$?
  trap - EXIT
  if (( code == 0 )); then write_status complete 0; else write_status failed "${code}"; fi
  exit "${code}"
}
trap on_exit EXIT
write_status running 0

cd "${REPO}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${REPO}/data/hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

copy_exact() {
  local source="$1" target="$2" expected_sha="$3"
  if [[ ! -e "${target}" ]]; then
    cp --reflink=auto --sparse=always -- "${source}" "${target}"
  fi
  test -f "${target}" && test ! -L "${target}"
  printf '%s  %s\n' "${expected_sha}" "${target}" | sha256sum -c -
}

copy_archive_for_safe_extract() {
  local source="$1" target="$2" expected_size="$3"
  if [[ -e "${target}" && "$(stat -c %s -- "${target}")" != "${expected_size}" ]]; then
    rm -f -- "${target}"
  fi
  if [[ ! -e "${target}" ]]; then
    cp --reflink=auto --sparse=always -- "${source}" "${target}"
  fi
  test -f "${target}" && test ! -L "${target}"
  test "$(stat -c %s -- "${target}")" = "${expected_size}"
  # safe_extract_artiverse.py authenticates the pinned SHA-256 immediately
  # before it consumes the archive, avoiding a redundant 65 GB hash pass here.
}

copy_exact "${CHUNKS_SOURCE}/manifest.json" \
  "${ARTIVERSE_ROOT}/dataset_chunks/manifest.json" \
  8fa6468254a1f74c58f0c25699598bf88f622fabdaf74f0cd9268ee5663c5586
copy_exact "${ARTIVERSE_SOURCE}/pack_dataset_chunks.py" \
  "${ARTIVERSE_ROOT}/pack_dataset_chunks.py" \
  f438e6fa147514f5260a205bc09d4b6c6ff3c0ce2d3022af424d220a9c933b99

if [[ ! -d "${ARTIVERSE_ROOT}/data" || ! -s "${ARTIVERSE_ROOT}/artiverse_safe_extraction_receipt.json" ]]; then
  # JuiceFS does not support reflinks. These authenticated copies are only
  # temporary inputs to the safe extractor and are removed after publication.
  copy_archive_for_safe_extract "${CHUNKS_SOURCE}/artiverse_data-00001-of-00002.tar.gz" \
    "${ARTIVERSE_ROOT}/dataset_chunks/artiverse_data-00001-of-00002.tar.gz" \
    38163580631
  copy_archive_for_safe_extract "${CHUNKS_SOURCE}/artiverse_data-00002-of-00002.tar.gz" \
    "${ARTIVERSE_ROOT}/dataset_chunks/artiverse_data-00002-of-00002.tar.gz" \
    27170560473
fi

# ParaCloud has no working direct/mirror/proxy route to Hugging Face.  The
# exact already-verified SQZ blob is relayed into this factory-owned cache;
# preparation remains offline and refuses any missing or altered cache.
model_blob="${HUGGINGFACE_HUB_CACHE}/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K/blobs/${MODEL_SHA256}"
model_snapshot="${HUGGINGFACE_HUB_CACHE}/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K/snapshots/${MODEL_REVISION}/${MODEL_FILENAME}"
test -f "${model_blob}" && test ! -L "${model_blob}"
test "$(stat -c %s -- "${model_blob}")" = 1710517748
printf '%s  %s\n' "${MODEL_SHA256}" "${model_blob}" | sha256sum -c -
test -L "${model_snapshot}"
test "$(readlink -f -- "${model_snapshot}")" = "${model_blob}"

if [[ -d "${ARTIVERSE_ROOT}/data" && -s "${ARTIVERSE_ROOT}/artiverse_safe_extraction_receipt.json" ]]; then
  .venv/bin/python scripts/safe_extract_artiverse.py \
    --repository-root "${ARTIVERSE_ROOT}" --verify-existing
else
  .venv/bin/python scripts/safe_extract_artiverse.py \
    --repository-root "${ARTIVERSE_ROOT}"
fi
rm -f -- \
  "${ARTIVERSE_ROOT}/dataset_chunks/artiverse_data-00001-of-00002.tar.gz" \
  "${ARTIVERSE_ROOT}/dataset_chunks/artiverse_data-00002-of-00002.tar.gz"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
.venv/bin/python scripts/prepare_artiverse.py \
  --dataset-root data/artiverse \
  --output-path data/artiverse/embeddings \
  --source-revision 8c4b120418e7cbdf9ac4c9580c5dbfdbf128a248 \
  --max-collision-elements 32 \
  --minimum-indexed 10 \
  --minimum-indexed-per-category 3 \
  --max-failure-fraction 0.25 \
  --categories armoire bookcase chest_of_drawers
.venv/bin/python scripts/artiverse_contract.py \
  --dataset-root data/artiverse \
  --embeddings-path data/artiverse/embeddings \
  --output "${RUN_DIR}/artiverse_preparation_validation.json"
.venv/bin/python scripts/preflight_artiverse_visual_resources.py \
  --dataset-root data/artiverse \
  --embeddings-path data/artiverse/embeddings \
  --runtime-code scenesmith/agent_utils/artiverse_visual_normalization.py \
  --runtime-code scenesmith/agent_utils/asset_manager.py \
  --output "${RUN_DIR}/artiverse_visual_resources.json"

.venv/bin/python scripts/materials_contract.py validate \
  --data-root data/materials \
  --source-embeddings data/materials/embeddings \
  --contract-embeddings data/materials_full_quality_contract/embeddings \
  --min-retained 1900 --max-pruned 15 \
  --output "${PREFLIGHT_DIR}/materials_contract_validation.json"

if [[ ! -s data/objathor-assets/preprocessed/clip_embeddings.npy ]]; then
  .venv/bin/python scripts/prepare_objaverse.py --data-path data/objathor-assets
fi
objathor_receipt="${PREFLIGHT_DIR}/objathor_retrieval_offline.json"
if [[ -s "${objathor_receipt}" ]] && python - "${objathor_receipt}" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("status") == "pass" else 1)
PY
then
  objathor_mode=(--verify-only)
else
  objathor_mode=()
fi
.venv/bin/python scripts/preflight_objathor_retrieval.py \
  --dataset-root data/objathor-assets \
  --preprocessed-path data/objathor-assets/preprocessed \
  --output "${objathor_receipt}" "${objathor_mode[@]}"

df -h "${OUTPUT_ROOT}"
echo "factory one-GPU asset preparation passed"
