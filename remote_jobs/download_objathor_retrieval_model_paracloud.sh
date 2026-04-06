#!/usr/bin/env bash
# Publish the exact ObjectThor text-retrieval checkpoint into Codex's owned
# ParaCloud cache.  Production itself remains cache-only and offline.

set -euo pipefail

readonly FACTORY=/data/run01/scvj260/codex_factory
readonly CACHE="${FACTORY}/huggingface"
readonly RECEIPT="${FACTORY}/objathor_openclip_download.json"
readonly REPO_ID=laion/CLIP-ViT-L-14-laion2B-s32B-b82K
readonly REVISION=1627032197142fbe2a7cfec626f4ced3ae60d07a
readonly FILENAME=open_clip_pytorch_model.safetensors
readonly EXPECTED_SHA256=7d129ed747e0ed53e82dfcc140382b51be66b56e6a9bdc3258afd2846e3bb019
readonly EXPECTED_BYTES=1710517748
readonly MODEL_DIR="${CACHE}/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K"
readonly URL="https://hf-mirror.com/${REPO_ID}/resolve/${REVISION}/${FILENAME}"

mkdir -p "${CACHE}"
exec 9>"${FACTORY}/.objathor_openclip_download.lock"
flock -n 9 || { echo "Another ObjectThor model download is active." >&2; exit 30; }

existing_blob="${MODEL_DIR}/blobs/${EXPECTED_SHA256}"
if [[ -f "${existing_blob}" ]] \
  && [[ "$(stat -c%s "${existing_blob}")" == "${EXPECTED_BYTES}" ]] \
  && [[ "$(sha256sum "${existing_blob}" | awk '{print $1}')" == "${EXPECTED_SHA256}" ]]; then
  echo "Pinned ObjectThor checkpoint is already verified."
else
  if [[ -e "${MODEL_DIR}" ]]; then
    mv "${MODEL_DIR}" "${MODEL_DIR}.corrupt_$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  stage="$(mktemp -d "${CACHE}/.models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K.stage.XXXXXX")"
  trap 'rm -rf "${stage}"' EXIT
  mkdir -p "${stage}/blobs" "${stage}/snapshots/${REVISION}" "${stage}/refs"
  curl --fail --location --retry 3 --retry-all-errors --connect-timeout 30 \
    --output "${stage}/blobs/.${EXPECTED_SHA256}.download" "${URL}"
  test "$(stat -c%s "${stage}/blobs/.${EXPECTED_SHA256}.download")" = "${EXPECTED_BYTES}"
  test "$(sha256sum "${stage}/blobs/.${EXPECTED_SHA256}.download" | awk '{print $1}')" = "${EXPECTED_SHA256}"
  mv "${stage}/blobs/.${EXPECTED_SHA256}.download" "${stage}/blobs/${EXPECTED_SHA256}"
  ln -s "../../blobs/${EXPECTED_SHA256}" "${stage}/snapshots/${REVISION}/${FILENAME}"
  printf '%s\n' "${REVISION}" >"${stage}/refs/main"
  mv "${stage}" "${MODEL_DIR}"
  trap - EXIT
fi

"/data/run01/scvj260/scenesmith/.venv/bin/python" - "${existing_blob}" "${RECEIPT}" \
  "${REPO_ID}" "${REVISION}" "${FILENAME}" "${EXPECTED_SHA256}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

blob, receipt, repo_id, revision, filename, expected = sys.argv[1:]
path = Path(blob).resolve(strict=True)
digest = hashlib.sha256()
with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
actual = digest.hexdigest()
if actual != expected:
    raise SystemExit(f"SHA-256 mismatch: {actual} != {expected}")

payload = {
    "status": "pass",
    "repo_id": repo_id,
    "revision": revision,
    "filename": filename,
    "cache_path": str(path),
    "blob_path": str(path),
    "size_bytes": path.stat().st_size,
    "sha256": actual,
}
temporary = Path(receipt).with_name(f".{Path(receipt).name}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(receipt)
print(json.dumps(payload, sort_keys=True))
PY
