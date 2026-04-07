#!/usr/bin/env bash
# Download only the immutable OpenCLIP text encoder used by ObjectThor retrieval.
# This is a model/data setup action; it does not install or upgrade packages.

set -euo pipefail

REPO=/root/workspace/scenesmith-hts
MODEL_REPO=laion/CLIP-ViT-L-14-laion2B-s32B-b82K
MODEL_REVISION=1627032197142fbe2a7cfec626f4ced3ae60d07a
MODEL_FILENAME=open_clip_pytorch_model.safetensors
RUN_META="${REPO}/outputs/preflight/full_quality_school_reference_20260710"
STATUS_FILE="${RUN_META}/objathor_retrieval_model_download.status"

mkdir -p "${RUN_META}"
source /root/workspace/Humanoid-Training-Scene-Ready-Generation/local_setup/setup_env_sqz.sh
cd "${REPO}"
agent_proxy
unset ALL_PROXY all_proxy
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE DIFFUSERS_OFFLINE

printf 'running revision=%s %s\n' "${MODEL_REVISION}" "$(date -Is)" >"${STATUS_FILE}"
trap 'code=$?; if (( code == 0 )); then state=complete; else state=failed; fi; printf "%s exit=%s revision=%s %s\n" "$state" "$code" "${MODEL_REVISION}" "$(date -Is)" >"${STATUS_FILE}"' EXIT

python - "${MODEL_REPO}" "${MODEL_REVISION}" "${MODEL_FILENAME}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

repo_id, revision, filename = sys.argv[1:]
snapshot_path = Path(
    hf_hub_download(
        repo_id=repo_id,
        revision=revision,
        filename=filename,
    )
).absolute()
path = snapshot_path.resolve()

digest = hashlib.sha256()
with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
        digest.update(chunk)

snapshot_revision = snapshot_path.parent.name
if snapshot_revision != revision:
    raise SystemExit(
        f"Hugging Face snapshot mismatch: resolved={snapshot_revision} expected={revision}"
    )

print(
    json.dumps(
        {
            "repo_id": repo_id,
            "revision": revision,
            "filename": filename,
            "snapshot_path": str(snapshot_path),
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        },
        indent=2,
        sort_keys=True,
    )
)
PY

echo "Pinned ObjectThor OpenCLIP retrieval model downloaded; run the offline preflight next."
