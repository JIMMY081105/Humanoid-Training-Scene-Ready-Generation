#!/usr/bin/env bash
# Rebuild then independently verify the immutable Artiverse visual-resource
# receipt against the current authoritative runtime tree.

#SBATCH -J artiverse_visual_receipt
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -t 1-00:00:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%j.out

set -euo pipefail

readonly REPO=/data/run01/scvj260/scenesmith-hts
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly RUN_DIR="${REPO}/outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710"

case "$(hostname)" in
  ln*|Ln*|lN*|LN*)
    echo "Refusing visual-resource preparation on a login node." >&2
    exit 65
    ;;
esac

source /data/run01/scvj260/scenesmith/.venv/bin/activate
export PYTHONPATH="${FACTORY}/python_overlay_agents_0_17_4:${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO}"

python scripts/preflight_artiverse_visual_resources.py \
  --dataset-root data/artiverse \
  --embeddings-path data/artiverse/embeddings \
  --runtime-code scenesmith/agent_utils/artiverse_visual_normalization.py \
  --runtime-code scenesmith/agent_utils/asset_manager.py \
  --output "${RUN_DIR}/artiverse_visual_resources.json"
python scripts/preflight_artiverse_visual_resources.py \
  --dataset-root data/artiverse \
  --embeddings-path data/artiverse/embeddings \
  --runtime-code scenesmith/agent_utils/artiverse_visual_normalization.py \
  --runtime-code scenesmith/agent_utils/asset_manager.py \
  --output "${RUN_DIR}/artiverse_visual_resources.json" \
  --verify-only
