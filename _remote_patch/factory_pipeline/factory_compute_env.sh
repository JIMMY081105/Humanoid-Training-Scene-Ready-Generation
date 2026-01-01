#!/usr/bin/env bash
# Isolated ParaCloud compute-node environment for the factory checkout.
set -euo pipefail

readonly FACTORY_REPO=/data/run01/scvj260/scenesmith-factory-codex
if [[ "$(hostname)" == ln* ]]; then
  echo "factory compute environment refuses login node $(hostname)" >&2
  return 1 2>/dev/null || exit 1
fi
test -x "${FACTORY_REPO}/.venv/bin/python"
cd "${FACTORY_REPO}"
# shellcheck disable=SC1091
source "${FACTORY_REPO}/.venv/bin/activate"

export PYTHONPATH="${FACTORY_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export HTTPS_PROXY=http://ln08:7890
export HTTP_PROXY=http://ln08:7890
export NO_PROXY=127.0.0.1,localhost,dashscope.aliyuncs.com,hf-mirror.com
export no_proxy="${NO_PROXY}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="${FACTORY_REPO}/data/hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export UV_CACHE_DIR=/data/run01/scvj260/uv-cache
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=600

echo "[factory-env] node=$(hostname) repo=${FACTORY_REPO} python=$(command -v python)"
