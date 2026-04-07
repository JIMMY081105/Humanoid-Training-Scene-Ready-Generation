#!/usr/bin/env bash
# Download, verify, unpack, and index the access-gated official Artiverse release.
# Prerequisite: the user has accepted the dataset terms and completed `hf auth login`.

set -euo pipefail

REPO=/root/workspace/scenesmith-hts
ASSET_ROOT=/localssd/scenesmith-hts-assets/artiverse
DOWNLOAD_DIR="${ASSET_ROOT}/repository"
RUN_META="${REPO}/outputs/preflight/full_quality_school_reference_20260710"
STATUS_FILE="${RUN_META}/artiverse_download_prepare.status"
LOCK_FILE="${ASSET_ROOT}/.download_prepare.lock"
ARTIVERSE_REVISION=8c4b120418e7cbdf9ac4c9580c5dbfdbf128a248
PACK_SCRIPT=pack_dataset_chunks.py
PACK_SCRIPT_SIZE=11919
PACK_SCRIPT_GIT_OID=13875091144e2ac4b7bf4210300b940f6f63b8d8
MANIFEST_PATH=dataset_chunks/manifest.json
MANIFEST_SIZE=255798
MANIFEST_GIT_OID=e5d6df0cef9c564a043666e6b7a02013c9e74eb1
AUDITED_PACK_SCRIPT_SHA256=f438e6fa147514f5260a205bc09d4b6c6ff3c0ce2d3022af424d220a9c933b99
AUDITED_MANIFEST_SHA256=8fa6468254a1f74c58f0c25699598bf88f622fabdaf74f0cd9268ee5663c5586
CHUNK_1_PATH=dataset_chunks/artiverse_data-00001-of-00002.tar.gz
CHUNK_1_SIZE=38163580631
CHUNK_1_SHA256=695d2d602faafab922ce66359ea104d81505f5b0fdee8f461d8905f0ccb4ef3b
CHUNK_2_PATH=dataset_chunks/artiverse_data-00002-of-00002.tar.gz
CHUNK_2_SIZE=27170560473
CHUNK_2_SHA256=56dffa50f1c8c20d3b1eef626046805a6c7cd997141e8ab5fac9ebdae8ffab81
FULL_DOWNLOAD_APPROVED="${ARTIVERSE_APPROVE_FULL_DOWNLOAD:-0}"

mkdir -p "${RUN_META}" "${ASSET_ROOT}"
exec 8>"${LOCK_FILE}"
if ! flock -n 8; then
    echo "ERROR: another Artiverse download/preparation attempt holds ${LOCK_FILE}." >&2
    exit 75
fi
source /root/workspace/Humanoid-Training-Scene-Ready-Generation/local_setup/setup_env_sqz.sh
agent_proxy
unset ALL_PROXY all_proxy
# The installed Xet transport truncates local-dir partial files when restarted.
# Force huggingface_hub's standard Range-resumable HTTP path. Both optional
# accelerators have restart semantics that are unsuitable for this tunnel.
export HF_HUB_DISABLE_XET=1
unset HF_HUB_ENABLE_HF_TRANSFER
# huggingface_hub defaults streamed reads to 10 seconds. The laptop reverse
# tunnel can legitimately pause longer while carrying both 27-38 GB archives;
# a 120-second socket read timeout avoids needless reconnect loops while the
# outer exact-offset retry remains responsible for genuinely broken streams.
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
cd "${REPO}"

if [[ -L data ]]; then
    echo "ERROR: data must be an isolated directory, not the vanilla checkout symlink." >&2
    exit 2
fi
if [[ ! -L data/materials || ! -L data/artvip_sdf ]]; then
    echo "ERROR: expected read-only shared materials and ArtVIP links are missing." >&2
    exit 2
fi

# Compute state-aware headroom before a multi-hour phase.  A fresh setup needs
# the still-missing archive bytes plus the complete 87 GB staging tree; a
# receipt-bound existing tree only needs derived/index/output headroom.
AVAILABLE_BYTES="$(df --output=avail -B1 "${ASSET_ROOT}" | tail -n 1 | tr -d ' ')"
AVAILABLE_INODES="$(df --output=iavail "${ASSET_ROOT}" | tail -n 1 | tr -d ' ')"
if [[ -d "${DOWNLOAD_DIR}/data" && -f "${DOWNLOAD_DIR}/artiverse_safe_extraction_receipt.json" ]]; then
    REQUIRED_BYTES=30000000000
    REQUIRED_INODES=100000
else
    MISSING_ARCHIVE_BYTES=0
    for archive_record in \
        "${CHUNK_1_PATH}:${CHUNK_1_SIZE}" \
        "${CHUNK_2_PATH}:${CHUNK_2_SIZE}"; do
        archive_path="${archive_record%%:*}"
        archive_size="${archive_record##*:}"
        current_size=0
        if [[ -f "${DOWNLOAD_DIR}/${archive_path}" ]]; then
            current_size="$(stat -c %s "${DOWNLOAD_DIR}/${archive_path}")"
        fi
        if (( current_size < archive_size )); then
            MISSING_ARCHIVE_BYTES=$((MISSING_ARCHIVE_BYTES + archive_size - current_size))
        fi
    done
    REQUIRED_BYTES=$((MISSING_ARCHIVE_BYTES + 86992752890 + 30000000000))
    REQUIRED_INODES=750000
fi
if (( AVAILABLE_BYTES < REQUIRED_BYTES )); then
    echo "ERROR: Artiverse needs ${REQUIRED_BYTES} free bytes for the remaining state; available=${AVAILABLE_BYTES}." >&2
    exit 25
fi
if (( AVAILABLE_INODES < REQUIRED_INODES )); then
    echo "ERROR: Artiverse needs ${REQUIRED_INODES} free inodes for the remaining state; available=${AVAILABLE_INODES}." >&2
    exit 25
fi

echo "running $(date -Is)" >"${STATUS_FILE}"
trap 'code=$?; if (( code == 0 )); then state=complete; else state=failed; fi; printf "%s exit=%s %s\n" "$state" "$code" "$(date -Is)" >"${STATUS_FILE}"' EXIT

# The 65.3 GB transfer crosses the laptop's SSH reverse proxy. A VPN, Wi-Fi, or
# proxy interruption must turn into an automatic retry. huggingface_hub's
# standard HTTP transport resumes each etag-bound ``*.incomplete`` file with a
# Range request when the CDN honors Range (as verified on SQZ), so retry the
# exact pinned download indefinitely without manually altering partial data.
download_pinned_chunks_with_retry() {
    local attempt=0
    local exit_code=0
    local retry_delay_seconds=20

    while true; do
        attempt=$((attempt + 1))
        printf 'downloading_chunks attempt=%s %s\n' \
            "${attempt}" "$(date -Is)" >"${STATUS_FILE}"

        set +e
        hf download 3dlg-hcvc/artiverse \
            "${CHUNK_1_PATH}" "${CHUNK_2_PATH}" \
            --repo-type dataset \
            --revision "${ARTIVERSE_REVISION}" \
            --local-dir "${DOWNLOAD_DIR}" \
            --max-workers 2
        exit_code=$?
        set -e

        if (( exit_code == 0 )); then
            # huggingface_hub may deliberately return the existing local_dir
            # with exit status zero after a transport failure.  That fallback is
            # useful for small optional snapshots but is not success for these
            # two compulsory archives.  Accept the attempt only after both final
            # (not cache-partial) paths exist at their exact pinned sizes.
            local final_chunks_complete=1
            local final_path
            local expected_size
            for archive_record in \
                "${CHUNK_1_PATH}:${CHUNK_1_SIZE}" \
                "${CHUNK_2_PATH}:${CHUNK_2_SIZE}"; do
                final_path="${DOWNLOAD_DIR}/${archive_record%%:*}"
                expected_size="${archive_record##*:}"
                if [[ ! -f "${final_path}" ]] || \
                    [[ "$(stat -c %s "${final_path}" 2>/dev/null || printf 0)" != "${expected_size}" ]]; then
                    final_chunks_complete=0
                fi
            done
            if (( final_chunks_complete == 1 )); then
                printf 'downloaded_chunks attempt=%s %s\n' \
                    "${attempt}" "$(date -Is)" >"${STATUS_FILE}"
                return 0
            fi
            exit_code=26
            printf 'Artiverse downloader returned success without both exact final archives; treating the local-dir fallback as incomplete.\n' >&2
        fi

        printf 'Artiverse chunk transfer disconnected (attempt=%s exit=%s); preserving partial files and retrying in %ss.\n' \
            "${attempt}" "${exit_code}" "${retry_delay_seconds}" >&2
        printf 'retrying_chunks attempt=%s previous_exit=%s delay_seconds=%s %s\n' \
            "${attempt}" "${exit_code}" "${retry_delay_seconds}" "$(date -Is)" \
            >"${STATUS_FILE}"
        sleep "${retry_delay_seconds}"
    done
}

set +e
HF_WHOAMI="$(hf auth whoami 2>&1)"
HF_WHOAMI_STATUS=$?
set -e
if (( HF_WHOAMI_STATUS != 0 )) || grep -qi "not logged in" <<<"${HF_WHOAMI}"; then
    echo "ERROR: SQZ is not logged into Hugging Face." >&2
    echo "Accept https://huggingface.co/datasets/3dlg-hcvc/artiverse and run 'hf auth login'." >&2
    exit 21
fi
printf '%s\n' "${HF_WHOAMI}"

# Prove the saved token has manual-gate approval and that both small supply-chain
# files still match the pinned repository tree. These are HEAD requests only;
# they cannot accidentally begin either 27-38 GB chunk.
python - "${ARTIVERSE_REVISION}" \
    "${PACK_SCRIPT}" "${PACK_SCRIPT_SIZE}" "${PACK_SCRIPT_GIT_OID}" \
    "${MANIFEST_PATH}" "${MANIFEST_SIZE}" "${MANIFEST_GIT_OID}" <<'PY'
from __future__ import annotations

import sys

from huggingface_hub import get_hf_file_metadata, hf_hub_url

repo_id = "3dlg-hcvc/artiverse"
revision = sys.argv[1]
values = sys.argv[2:]
for offset in range(0, len(values), 3):
    filename, expected_size, expected_oid = values[offset : offset + 3]
    metadata = get_hf_file_metadata(
        hf_hub_url(
            repo_id,
            filename,
            repo_type="dataset",
            revision=revision,
        ),
        token=True,
        timeout=30,
    )
    if metadata.commit_hash != revision:
        raise SystemExit(
            f"Pinned Artiverse HEAD resolved {metadata.commit_hash}, expected {revision}"
        )
    if metadata.etag != expected_oid or metadata.size != int(expected_size):
        raise SystemExit(
            f"Pinned Artiverse metadata mismatch for {filename}: "
            f"etag={metadata.etag} size={metadata.size}"
        )
    print(f"authorized pinned HEAD passed: {filename}")
PY

# Fetch only the two reviewable files first. The CLI and later chunk transfer are
# resumable, but publisher code is never executed merely because access succeeds.
hf download 3dlg-hcvc/artiverse \
    "${PACK_SCRIPT}" "${MANIFEST_PATH}" \
    --repo-type dataset \
    --revision "${ARTIVERSE_REVISION}" \
    --local-dir "${DOWNLOAD_DIR}" \
    --max-workers 1

test "$(stat -c %s "${DOWNLOAD_DIR}/${PACK_SCRIPT}")" -eq "${PACK_SCRIPT_SIZE}"
test "$(stat -c %s "${DOWNLOAD_DIR}/${MANIFEST_PATH}")" -eq "${MANIFEST_SIZE}"
test "$(git hash-object "${DOWNLOAD_DIR}/${PACK_SCRIPT}")" = "${PACK_SCRIPT_GIT_OID}"
test "$(git hash-object "${DOWNLOAD_DIR}/${MANIFEST_PATH}")" = "${MANIFEST_GIT_OID}"
PACK_SCRIPT_SHA256="$(sha256sum "${DOWNLOAD_DIR}/${PACK_SCRIPT}" | awk '{print $1}')"
MANIFEST_SHA256="$(sha256sum "${DOWNLOAD_DIR}/${MANIFEST_PATH}" | awk '{print $1}')"
printf 'gated metadata sha256: %s=%s %s=%s\n' \
    "${PACK_SCRIPT}" "${PACK_SCRIPT_SHA256}" \
    "${MANIFEST_PATH}" "${MANIFEST_SHA256}"

test "${PACK_SCRIPT_SHA256}" = "${AUDITED_PACK_SCRIPT_SHA256}"
test "${MANIFEST_SHA256}" = "${AUDITED_MANIFEST_SHA256}"

if [[ "${FULL_DOWNLOAD_APPROVED}" != 1 ]]; then
    echo "Metadata audit passed. Set ARTIVERSE_APPROVE_FULL_DOWNLOAD=1 for the explicit 65.3 GB transfer." >&2
    echo "The audited publisher unpack command will never be executed; extraction uses the local fail-closed implementation." >&2
    exit 23
fi

# Fetch only the two pinned archives. Standard HTTP appends to authenticated
# partials and the retry loop survives tunnel reconnections automatically.
download_pinned_chunks_with_retry

test "$(stat -c %s "${DOWNLOAD_DIR}/${CHUNK_1_PATH}")" -eq "${CHUNK_1_SIZE}"
test "$(stat -c %s "${DOWNLOAD_DIR}/${CHUNK_2_PATH}")" -eq "${CHUNK_2_SIZE}"
# The safe extractor performs two authoritative reads: it authenticates both
# complete archives before staging, then authenticates the exact compressed
# stream consumed during extraction. Avoid two redundant 65.3 GB hash passes.

if [[ -d "${DOWNLOAD_DIR}/data" && -f "${DOWNLOAD_DIR}/artiverse_safe_extraction_receipt.json" ]]; then
    python scripts/safe_extract_artiverse.py \
        --repository-root "${DOWNLOAD_DIR}" \
        --verify-existing
elif [[ -e "${DOWNLOAD_DIR}/data" || -e "${DOWNLOAD_DIR}/artiverse_safe_extraction_receipt.json" ]]; then
    echo "ERROR: incomplete Artiverse extraction state; refusing automatic overwrite." >&2
    exit 24
else
    python scripts/safe_extract_artiverse.py \
        --repository-root "${DOWNLOAD_DIR}"
fi

test -d "${DOWNLOAD_DIR}/data"
test -f "${DOWNLOAD_DIR}/dataset_chunks/manifest.json"

if [[ -e data/artiverse && ! -L data/artiverse ]]; then
    echo "ERROR: data/artiverse exists and is not a symlink; refusing to overwrite it." >&2
    exit 22
fi
ln -sfn "${DOWNLOAD_DIR}" data/artiverse

# Only school-relevant articulated furniture is indexed for this scene. The full
# verified dataset remains available at the same path for later re-indexing.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
python - <<'PY'
from scenesmith.agent_utils.clip_embeddings import get_text_embedding

embedding = get_text_embedding("openable school storage cabinet", device="cpu")
if embedding.shape != (1024,):
    raise SystemExit(f"Unexpected offline OpenCLIP embedding shape: {embedding.shape}")
print("OpenCLIP offline cache preflight passed")
PY
python scripts/prepare_artiverse.py \
    --dataset-root data/artiverse \
    --output-path data/artiverse/embeddings \
    --source-revision "${ARTIVERSE_REVISION}" \
    --max-collision-elements 32 \
    --minimum-indexed 10 \
    --minimum-indexed-per-category 3 \
    --max-failure-fraction 0.25 \
    --categories \
      armoire bookcase chest_of_drawers

for filename in clip_embeddings.npy embedding_index.yaml metadata_index.yaml; do
    test -s "data/artiverse/embeddings/${filename}"
done
python - <<'PY'
import json
from pathlib import Path

path = Path("data/artiverse/embeddings/artiverse_preparation_manifest.json")
manifest = json.loads(path.read_text(encoding="utf-8"))
if (
    manifest.get("status") != "pass"
    or manifest.get("source_revision")
    != "8c4b120418e7cbdf9ac4c9580c5dbfdbf128a248"
    or int(manifest.get("indexed_count", 0)) < 1
):
    raise SystemExit(f"Artiverse preparation did not produce a usable index: {manifest}")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

python scripts/artiverse_contract.py \
    --dataset-root data/artiverse \
    --embeddings-path data/artiverse/embeddings \
    --output "${RUN_META}/artiverse_preparation_validation.json"

echo "Artiverse download, checksum verification, unpack, and indexing completed."
