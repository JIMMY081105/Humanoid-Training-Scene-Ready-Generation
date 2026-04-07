#!/usr/bin/env bash
# Acquire the SAM3D dependency proven missing by the offline initialization test.

set -euo pipefail

REPO=/root/workspace/scenesmith-hts
RUN_META="${REPO}/outputs/preflight/full_quality_school_reference_20260710"
STATUS_FILE="${RUN_META}/sam3d_dependency_download.status"

mkdir -p "${RUN_META}"
source /root/workspace/Humanoid-Training-Scene-Ready-Generation/local_setup/setup_env_sqz.sh
agent_proxy
cd "${REPO}"

echo "running $(date -Is)" >"${STATUS_FILE}"
trap 'code=$?; if (( code == 0 )); then state=complete; else state=failed; fi; printf "%s exit=%s %s\n" "$state" "$code" "$(date -Is)" >"${STATUS_FILE}"' EXIT

hf download Ruicheng/moge-vitl model.pt

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
python - <<'PY'
from huggingface_hub import hf_hub_download

path = hf_hub_download("Ruicheng/moge-vitl", "model.pt", local_files_only=True)
print(path)
PY

python scripts/preflight_sam3d_offline.py \
    --sam3-checkpoint external/checkpoints/sam3.pt \
    --pipeline-config external/checkpoints/pipeline.yaml \
    --output "${RUN_META}/sam3d_offline_load.json"
