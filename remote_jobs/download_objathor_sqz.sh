#!/usr/bin/env bash
# Download and preprocess the ObjectThor assets required by the full-quality
# manipuland route on SQZ. Run inside tmux; this script is safe to resume even
# after a connection dies partway through one of the large tar files.

set -euo pipefail

REPO=/root/workspace/scenesmith-hts
RUN_META="${REPO}/outputs/preflight/full_quality_school_reference_20260710"
STATUS_FILE="${RUN_META}/objathor_download.status"
VERSION=2023_09_23
CACHE_ROOT="${HOME}/.objathor-assets/${VERSION}"
BASE_URL="https://pub-daedd7738a984186a00f2ab264d06a07.r2.dev/${VERSION}"
ASSETS_TAR="${CACHE_ROOT}/assets.tar"
ANNOTATIONS_GZ="${CACHE_ROOT}/annotations.json.gz"
FEATURES_TAR="${CACHE_ROOT}/features.tar"

# Pinned Content-Length values from the publisher's versioned bucket. A short
# transfer must never be mistaken for a complete archive (the objathor helper
# only checks for path existence and is therefore unsafe after interruption).
ASSETS_SIZE=23177386496
ANNOTATIONS_SIZE=9740343
FEATURES_SIZE=388221440
ASSETS_ETAG='"31b8681f98917e74216a836ebf244cef-4421"'
ANNOTATIONS_ETAG='"3031c430789b80331d63485b527c5d89"'
FEATURES_ETAG='"1a556b459af2da002655b8262f24c92d-75"'

mkdir -p "${RUN_META}" "${CACHE_ROOT}"
source /root/workspace/Humanoid-Training-Scene-Ready-Generation/local_setup/setup_env_sqz.sh
cd "${REPO}"

if [[ -L data ]]; then
    echo "ERROR: data must be an isolated directory, not the vanilla checkout symlink." >&2
    exit 2
fi
if [[ ! -L data/materials || ! -L data/artvip_sdf ]]; then
    echo "ERROR: expected read-only shared materials and ArtVIP links are missing." >&2
    exit 2
fi

echo "running $(date -Is)" >"${STATUS_FILE}"
trap 'code=$?; if (( code == 0 )); then state=complete; else state=failed; fi; printf "%s exit=%s %s\n" "$state" "$code" "$(date -Is)" >"${STATUS_FILE}"' EXIT

# Coordinate with objathor's own FileLock and prevent two recovery jobs from
# modifying the cache concurrently.
exec 9>"${CACHE_ROOT}/objects.lock"
flock -n 9 || { echo "ERROR: ObjectThor objects.lock is busy" >&2; exit 30; }

file_size() {
    local path="$1"
    if [[ -f "${path}" ]]; then
        stat -c %s "${path}"
    else
        printf '0\n'
    fi
}

resume_download() {
    local url="$1"
    local destination="$2"
    local expected_size="$3"
    local expected_etag="$4"
    local label="$5"
    local candidate="${destination}.resume"
    local current_size
    local attempt=0

    current_size="$(file_size "${destination}")"
    if (( current_size > expected_size )); then
        echo "ERROR: ${label} is larger than the pinned publisher size: ${current_size} > ${expected_size}" >&2
        exit 31
    fi

    if (( current_size == expected_size )); then
        candidate="${destination}"
    elif [[ ! -e "${candidate}" ]]; then
        if (( current_size > 0 )); then
            # Keep the original interrupted file unchanged. Resume a same-filesystem
            # candidate so publication can be atomic after verification.
            cp --reflink=auto --sparse=always "${destination}" "${candidate}"
        else
            : >"${candidate}"
        fi
    fi

    current_size="$(file_size "${candidate}")"
    if (( current_size > expected_size )); then
        echo "ERROR: ${label} resume candidate is oversized: ${current_size} > ${expected_size}" >&2
        exit 32
    fi

    while (( current_size < expected_size )); do
        attempt=$((attempt + 1))
        if (( attempt > 50 )); then
            echo "ERROR: ${label} did not complete after 50 resumable attempts" >&2
            exit 33
        fi
        echo "[download] ${label}: resuming at ${current_size}/${expected_size} bytes"

        # Prove that the versioned object still matches the pinned publisher
        # object before appending bytes to the candidate.
        local headers code content_range remote_etag
        headers="$(curl --noproxy '*' --fail --silent --show-error --location --head \
            -H "Range: bytes=${current_size}-" "${url}")"
        code="$(printf '%s\n' "${headers}" | awk '/^HTTP/{code=$2} END{print code}' | tr -d '\r')"
        content_range="$(printf '%s\n' "${headers}" | awk 'tolower($1)=="content-range:"{$1=""; sub(/^ /,""); value=$0} END{print value}' | tr -d '\r')"
        remote_etag="$(printf '%s\n' "${headers}" | awk 'tolower($1)=="etag:"{$1=""; sub(/^ /,""); value=$0} END{print value}' | tr -d '\r')"
        test "${code}" = 206
        test "${content_range}" = "bytes ${current_size}-$((expected_size - 1))/${expected_size}"
        test "${remote_etag}" = "${expected_etag}"

        # --continue-at - preserves the existing prefix. If-Range prevents an
        # in-place mix of two publisher object versions.
        curl --noproxy '*' --fail --location --show-error --continue-at - \
            --connect-timeout 30 --speed-limit 1024 --speed-time 300 \
            --retry 5 --retry-delay 5 \
            -H "If-Range: ${expected_etag}" \
            --output "${candidate}" \
            "${url}" || true

        local new_size
        new_size="$(file_size "${candidate}")"
        if (( new_size > expected_size )); then
            echo "ERROR: ${label} exceeded expected size after download: ${new_size} > ${expected_size}" >&2
            exit 33
        fi
        if (( new_size <= current_size )); then
            echo "[download] ${label}: no progress; retrying after 10 seconds" >&2
            sleep 10
        fi
        current_size="${new_size}"
    done

    test "$(file_size "${candidate}")" -eq "${expected_size}"

    # R2 exposes either a whole-file MD5 or an S3-compatible multipart ETag.
    # Recompute it locally before the candidate is allowed to replace anything.
    python - "${candidate}" "${expected_etag}" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = sys.argv[2].strip('"')
if "-" in expected:
    digests = []
    with path.open("rb") as stream:
        while chunk := stream.read(5 * 1024 * 1024):
            digests.append(hashlib.md5(chunk, usedforsecurity=False).digest())
    actual = (
        f"{hashlib.md5(b''.join(digests), usedforsecurity=False).hexdigest()}"
        f"-{len(digests)}"
    )
else:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    actual = digest.hexdigest()
print(f"verified_etag={actual}")
if actual != expected:
    raise SystemExit(f"ETag mismatch: actual={actual} expected={expected}")
PY

    if [[ "${candidate}" != "${destination}" ]]; then
        if [[ -e "${destination}" ]]; then
            local backup="${destination}.interrupted-$(file_size "${destination}")"
            if [[ -e "${backup}" ]]; then
                echo "ERROR: refusing to overwrite interrupted-file backup ${backup}" >&2
                exit 34
            fi
            mv "${destination}" "${backup}"
            echo "[download] preserved interrupted prefix at ${backup}"
        fi
        mv "${candidate}" "${destination}"
        sync -f "${destination}"
    fi
    echo "[download] ${label}: complete and ETag-verified (${expected_size} bytes)"
}

resume_download "${BASE_URL}/assets.tar" "${ASSETS_TAR}" "${ASSETS_SIZE}" "${ASSETS_ETAG}" assets
resume_download "${BASE_URL}/annotations.json.gz" "${ANNOTATIONS_GZ}" "${ANNOTATIONS_SIZE}" "${ANNOTATIONS_ETAG}" annotations
resume_download "${BASE_URL}/features.tar" "${FEATURES_TAR}" "${FEATURES_SIZE}" "${FEATURES_ETAG}" features

gzip -t "${ANNOTATIONS_GZ}"
tar -tf "${ASSETS_TAR}" >/dev/null
tar -tf "${FEATURES_TAR}" >/dev/null

if [[ ! -d "${CACHE_ROOT}/assets" ]]; then
    ASSETS_STAGE="${CACHE_ROOT}/assets.extracting"
    mkdir -p "${ASSETS_STAGE}"
    echo "[extract] ObjectThor assets (safe to rerun over an interrupted extraction)"
    tar -xf "${ASSETS_TAR}" -C "${ASSETS_STAGE}"
    mapfile -t ASSET_TOP_LEVEL < <(find "${ASSETS_STAGE}" -mindepth 1 -maxdepth 1 -type d -print)
    if (( ${#ASSET_TOP_LEVEL[@]} != 1 )); then
        echo "ERROR: expected one top-level assets directory, found ${#ASSET_TOP_LEVEL[@]}" >&2
        exit 34
    fi
    mv "${ASSET_TOP_LEVEL[0]}" "${CACHE_ROOT}/assets"
    # Publisher archives can leave harmless AppleDouble metadata beside the
    # top-level assets directory. Cleanup must not abort an otherwise complete
    # extraction; the staged directory is ignored once CACHE_ROOT/assets exists.
    if ! rmdir "${ASSETS_STAGE}" 2>/dev/null; then
        echo "[extract] preserving non-empty staging residue at ${ASSETS_STAGE}"
    fi
fi
test -n "$(find "${CACHE_ROOT}/assets" -mindepth 1 -print -quit)"

if [[ ! -s "${CACHE_ROOT}/features/clip_features.pkl" ]]; then
    FEATURES_STAGE="${CACHE_ROOT}/features.extracting"
    mkdir -p "${FEATURES_STAGE}" "${CACHE_ROOT}/features"
    echo "[extract] ObjectThor CLIP features"
    tar -xf "${FEATURES_TAR}" -C "${FEATURES_STAGE}"
    feature_count=0
    while IFS= read -r -d '' feature_path; do
        cp -f "${feature_path}" "${CACHE_ROOT}/features/$(basename "${feature_path}")"
        feature_count=$((feature_count + 1))
    done < <(find "${FEATURES_STAGE}" -type f -name '*.pkl' -print0)
    if (( feature_count < 1 )); then
        echo "ERROR: no PKL files found in ObjectThor features archive" >&2
        exit 35
    fi
fi
test -s "${CACHE_ROOT}/features/clip_features.pkl"

if [[ -e data/objathor-assets && ! -L data/objathor-assets ]]; then
    echo "ERROR: data/objathor-assets exists and is not a symlink; refusing to overwrite it." >&2
    exit 36
fi
ln -sfn "${CACHE_ROOT}" data/objathor-assets

python scripts/prepare_objaverse.py

test -d data/objathor-assets/preprocessed
test -f data/objathor-assets/preprocessed/clip_embeddings.pkl \
    -o -f data/objathor-assets/preprocessed/clip_embeddings.npy

# The prepared image embeddings are unusable without the exact matching text
# encoder. Keep its separately resumable, revision-pinned setup in this job so a
# fresh ObjectThor acquisition cannot appear complete while retrieval would make
# a hidden network call (or fail) during a room run.
bash remote_jobs/download_objathor_retrieval_model_sqz.sh
python scripts/preflight_objathor_retrieval.py \
    --dataset-root data/objathor-assets \
    --preprocessed-path data/objathor-assets/preprocessed \
    --output "${RUN_META}/objathor_retrieval_offline.json"
python scripts/preflight_objathor_retrieval.py \
    --dataset-root data/objathor-assets \
    --preprocessed-path data/objathor-assets/preprocessed \
    --output "${RUN_META}/objathor_retrieval_offline.json" \
    --verify-only

echo "ObjectThor download and preprocessing completed."
