#!/usr/bin/env bash
# Strict ParaCloud resumption of the existing Room 1 benchmark.  This wrapper
# never bypasses the full runner preflight, SAM3D evidence verification, or
# deterministic/visual self-examination gates.

#SBATCH -J room1_strict_resume
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
# Four CPU cores and 64 GiB cover the strict model preflights plus the
# checkpointed pipeline without reserving the cluster's 126-GiB default.
# This is a scheduling optimisation only; it does not change any quality gate.
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%j.out

set -euo pipefail

readonly REPO=/data/run01/scvj260/scenesmith-hts
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly RUN_DIR="${REPO}/outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710"
readonly OVERLAY="${FACTORY}/python_overlay_agents_0_17_4"
readonly ARTIVERSE_RUNTIME="${FACTORY}/artiverse_runtime_v3"
readonly ARTIVERSE_VALIDATION="${FACTORY}/artiverse_preparation_validation.json"
readonly SAM3D_PREFLIGHT_DIR="${REPO}/outputs/preflight/full_quality_school_reference_20260710"
readonly SAM3D_GENERATION_DIR="${SAM3D_PREFLIGHT_DIR}/sam3d_offline_generation"

case "$(hostname)" in
  ln*|Ln*|lN*|LN*)
    echo "Refusing to run Room 1 production on a login node." >&2
    exit 65
    ;;
esac

test -d "${REPO}"
test -d "${OVERLAY}"
test -s "${RUN_DIR}/scene_000/room_classroom_01/scene_states/manipuland_checkpoint_004_storage_cabinet_0/scene_state.json"
test -d "${ARTIVERSE_RUNTIME}"
"${SCENESMITH_PARACLOUD_VENV:-/data/run01/scvj260/scenesmith/.venv}/bin/python" - "${ARTIVERSE_VALIDATION}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if result.get("status") != "pass":
    raise SystemExit("Artiverse preparation validation is not passing")
PY
if [[ -e "${REPO}/data/artiverse" || -L "${REPO}/data/artiverse" ]]; then
  [[ "$(readlink -f "${REPO}/data/artiverse")" == "$(readlink -f "${ARTIVERSE_RUNTIME}")" ]] || {
    echo "Refusing to replace a different Artiverse dataset link." >&2
    exit 66
  }
else
  link_tmp="${REPO}/data/.artiverse.${SLURM_JOB_ID}.tmp"
  ln -s "${ARTIVERSE_RUNTIME}" "${link_tmp}"
  mv -T "${link_tmp}" "${REPO}/data/artiverse"
fi
test -d "${REPO}/data/artiverse"
test -s "${REPO}/data/artiverse/embeddings/clip_embeddings.npy"
test -s "${REPO}/data/artiverse/embeddings/embedding_index.yaml"
test -s "${REPO}/data/artiverse/embeddings/metadata_index.yaml"

# The prior SAM3D receipts were created on SQZ and therefore attest paths that
# do not exist on ParaCloud.  Rebuild both strict, offline GPU proofs against
# the immutable byte-verified Codex cache before allowing the checkpointed
# Room 1 runner to start.  Preserve the old evidence rather than deleting it.
export HF_HOME="${FACTORY}/huggingface"
export HF_HUB_CACHE="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TORCH_HOME="${FACTORY}/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export PYTHONPATH="${OVERLAY}:${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO}"
PYTHON_BIN="${SCENESMITH_PARACLOUD_VENV:-/data/run01/scvj260/scenesmith/.venv}/bin/python"
"${PYTHON_BIN}" scripts/preflight_sam3d_offline.py \
  --sam3-checkpoint external/checkpoints/sam3.pt \
  --pipeline-config external/checkpoints/pipeline.yaml \
  --output "${SAM3D_PREFLIGHT_DIR}/sam3d_offline_load.json"
"${PYTHON_BIN}" scripts/preflight_sam3d_offline.py \
  --sam3-checkpoint external/checkpoints/sam3.pt \
  --pipeline-config external/checkpoints/pipeline.yaml \
  --output "${SAM3D_PREFLIGHT_DIR}/sam3d_offline_load.json" \
  --verify-only
if [[ -e "${SAM3D_GENERATION_DIR}" || -L "${SAM3D_GENERATION_DIR}" ]]; then
  mv "${SAM3D_GENERATION_DIR}" \
    "${SAM3D_PREFLIGHT_DIR}/sam3d_offline_generation.prior_${SLURM_JOB_ID}"
fi
"${PYTHON_BIN}" scripts/preflight_sam3d_generation.py \
  --repo-dir "${REPO}" \
  --preflight-dir "${SAM3D_PREFLIGHT_DIR}" \
  --sam3-checkpoint external/checkpoints/sam3.pt \
  --pipeline-config external/checkpoints/pipeline.yaml
"${PYTHON_BIN}" scripts/preflight_sam3d_generation.py \
  --repo-dir "${REPO}" \
  --preflight-dir "${SAM3D_PREFLIGHT_DIR}" \
  --sam3-checkpoint external/checkpoints/sam3.pt \
  --pipeline-config external/checkpoints/pipeline.yaml \
  --verify-only

export SCENESMITH_RUNTIME=paracloud
export SCENESMITH_REPO="${REPO}"
export SCENESMITH_PARACLOUD_VENV=/data/run01/scvj260/scenesmith/.venv
export SCENESMITH_PYTHON_OVERLAY="${OVERLAY}"
export SCENESMITH_HF_HOME="${HF_HOME}"
export SCENESMITH_TORCH_HOME="${TORCH_HOME}"
export SCENESMITH_AGENT_HTTP_PROXY="${SCENESMITH_AGENT_HTTP_PROXY:-http://ln08:18092}"
export SCENESMITH_EXECUTION_MODE=benchmark_classroom_01

exec bash "${REPO}/remote_jobs/run_full_quality_school_sqz.sh"
