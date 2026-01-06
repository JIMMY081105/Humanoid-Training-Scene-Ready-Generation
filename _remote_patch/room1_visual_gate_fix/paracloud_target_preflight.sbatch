#!/usr/bin/env bash
# Fail-closed target preflight for the Codex-owned ParaCloud execution checkout.
# It validates runtime evidence only; it does not generate a room or alter
# Claude-managed files.

#SBATCH -J codex_target_preflight
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%j.out

set -euo pipefail

readonly EXEC=/data/run01/scvj260/codex_factory/room1_execution
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly VENV=/data/run01/scvj260/scenesmith/.venv
readonly PREFLIGHT_DIR="$EXEC/outputs/preflight/full_quality_school_reference_20260710"

case "$(hostname)" in
  ln*|Ln*|lN*|LN*) echo "Refusing to run on a login node." >&2; exit 65 ;;
esac

test -x "$VENV/bin/python"
test -r "$EXEC/.env"
test -d "$EXEC/data/artiverse"
test -d "$EXEC/data/artvip_sdf"
test -d "$EXEC/data/materials"
test -d "$EXEC/data/objathor-assets"
test -f "$EXEC/external/checkpoints/sam3.pt"
test -f "$EXEC/external/checkpoints/pipeline.yaml"

source "$VENV/bin/activate"
cd "$EXEC"
set -a
source .env
set +a

export PYTHONPATH="$FACTORY/python_overlay_agents_0_17_4:$EXEC"
export HF_HOME="$FACTORY/huggingface"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TORCH_HOME="$FACTORY/torch"
export CUDA_HOME=/data/run01/scvj260/cuda128_conda
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HTTP_PROXY=http://ln08:18092
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTP_PROXY"
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
unset ALL_PROXY all_proxy

nvidia-smi
python -c "import scenesmith; print(scenesmith.__file__)"
mkdir -p "$PREFLIGHT_DIR"

python scripts/preflight_sam3d_offline.py \
  --sam3-checkpoint external/checkpoints/sam3.pt \
  --pipeline-config external/checkpoints/pipeline.yaml \
  --output "$PREFLIGHT_DIR/sam3d_offline_load.json"
python scripts/preflight_sam3d_offline.py \
  --sam3-checkpoint external/checkpoints/sam3.pt \
  --pipeline-config external/checkpoints/pipeline.yaml \
  --output "$PREFLIGHT_DIR/sam3d_offline_load.json" \
  --verify-only
python scripts/preflight_sam3d_generation.py \
  --repo-dir "$EXEC" \
  --preflight-dir "$PREFLIGHT_DIR" \
  --sam3-checkpoint external/checkpoints/sam3.pt \
  --pipeline-config external/checkpoints/pipeline.yaml
python scripts/preflight_sam3d_generation.py \
  --repo-dir "$EXEC" \
  --preflight-dir "$PREFLIGHT_DIR" \
  --sam3-checkpoint external/checkpoints/sam3.pt \
  --pipeline-config external/checkpoints/pipeline.yaml \
  --verify-only

python scripts/materials_contract.py validate \
  --data-root data/materials \
  --source-embeddings data/materials/embeddings \
  --contract-embeddings data/materials_full_quality_contract/embeddings \
  --min-retained 1900 \
  --max-pruned 15 \
  --output "$EXEC/outputs/materials_contract_validation.json"
python scripts/artiverse_contract.py \
  --dataset-root data/artiverse \
  --embeddings-path data/artiverse/embeddings \
  --output "$EXEC/outputs/artiverse_preparation_validation.json"

python scripts/pipeline_code_contract.py \
  --repo-dir "$EXEC" \
  --output "$EXEC/outputs/pipeline_code_contract.json"
python scripts/pipeline_code_contract.py \
  --repo-dir "$EXEC" \
  --output "$EXEC/outputs/pipeline_code_contract.json" \
  --verify-only

echo TARGET_PREFLIGHT_PASS
