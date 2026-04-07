#!/usr/bin/env bash
# Build the missing school Artiverse retrieval index without redownloading or
# modifying the transferred publisher asset tree.  The isolated runtime must
# first be published by copy_artiverse_school_paracloud.sbatch: Artiverse
# rejects hard-linked publisher files as a provenance violation.

#SBATCH -J artiverse_school_prepare
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 1-00:00:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%j.out

set -euo pipefail

readonly REPO=/data/run01/scvj260/scenesmith-hts
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly RUNTIME="${FACTORY}/artiverse_runtime_v3"
readonly OVERLAY="${FACTORY}/python_overlay_agents_0_17_4"

case "$(hostname)" in
  ln*|Ln*|lN*|LN*)
    echo "Refusing Artiverse preparation on a login node." >&2
    exit 65
    ;;
esac

test -d "${REPO}"
test -d "${OVERLAY}"

test -d "${RUNTIME}/data"
test -s "${RUNTIME}/artiverse_safe_extraction_receipt.json"
test -s "${RUNTIME}/dataset_chunks/manifest.json"
test -f "${RUNTIME}/.source_copy_complete"

source /data/run01/scvj260/scenesmith/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${OVERLAY}:${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HOME=/data/run01/scvj260/.cache/huggingface
export TRANSFORMERS_CACHE=/data/run01/scvj260/.cache/huggingface/transformers

cd "${REPO}"
python - <<'PY'
from scenesmith.agent_utils.clip_embeddings import get_text_embedding

embedding = get_text_embedding("openable school storage cabinet", device="cpu")
if embedding.shape != (1024,):
    raise SystemExit(f"Unexpected offline OpenCLIP embedding shape: {embedding.shape}")
print("OpenCLIP offline cache preflight passed")
PY
python scripts/prepare_artiverse.py \
  --dataset-root "${RUNTIME}" \
  --output-path "${RUNTIME}/embeddings" \
  --source-revision 8c4b120418e7cbdf9ac4c9580c5dbfdbf128a248 \
  --max-collision-elements 32 \
  --minimum-indexed 10 \
  --minimum-indexed-per-category 3 \
  --max-failure-fraction 0.25 \
  --categories armoire bookcase chest_of_drawers

for filename in clip_embeddings.npy embedding_index.yaml metadata_index.yaml; do
  test -s "${RUNTIME}/embeddings/${filename}"
done
python scripts/artiverse_contract.py \
  --dataset-root "${RUNTIME}" \
  --embeddings-path "${RUNTIME}/embeddings" \
  --output "${FACTORY}/artiverse_preparation_validation.json"
