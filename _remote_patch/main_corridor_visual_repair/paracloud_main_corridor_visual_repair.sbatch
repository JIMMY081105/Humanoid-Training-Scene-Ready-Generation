#!/usr/bin/env bash
# One transactional main-corridor scene repair followed by the unchanged strict
# deterministic and image-aware room release gates in the same GPU allocation.

#SBATCH -J main_corridor_visual_v2
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%j.out

set -euo pipefail
readonly F=/data/run01/scvj260/codex_factory
readonly E="$F/room1_execution"
readonly V=/data/run01/scvj260/scenesmith/.venv
readonly RUN="$E/outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710"

case "$(hostname)" in ln*|Ln*|lN*|LN*) echo 'Refusing login node.' >&2; exit 65;; esac
source "$V/bin/activate"
cd "$E"
set -a; source .env; set +a
export PYTHONPATH="$F/python_overlay_agents_0_17_4:$E:$E/scripts"
export CUDA_HOME=/data/run01/scvj260/cuda128_conda
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

python scripts/repair_main_corridor_visual_contract.py \
  --repo-dir "$E" \
  --run-dir "$RUN" \
  --room-id main_corridor \
  --port-offset 929

export ROOM_ID=main_corridor
export PROXY_PORT=19929
bash "$F/paracloud_room_release_gate.sbatch"
