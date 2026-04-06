#!/usr/bin/env bash
# Read-only compute-node diagnostic for ObjectThor preflight reproducibility.

#SBATCH -J objathor_repro_diag
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH -t 0-00:10:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%j.out

set -euo pipefail

readonly REPO=/data/run01/scvj260/scenesmith-hts
readonly FACTORY=/data/run01/scvj260/codex_factory
source /data/run01/scvj260/scenesmith/.venv/bin/activate
export PYTHONPATH="${FACTORY}/python_overlay_agents_0_17_4:${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${FACTORY}/huggingface"
export HF_HUB_CACHE="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "${REPO}"

python - <<'PY'
import json

from scripts.preflight_objathor_retrieval import (
    SMOKE_QUERY,
    load_openclip_text_embedding,
    resolve_openclip_weight_blob,
    validate_model_smoke,
)

receipt = "outputs/preflight/full_quality_school_reference_20260710/objathor_retrieval_offline.json"
saved = json.load(open(receipt, encoding="utf-8"))["model_smoke"]
weight = resolve_openclip_weight_blob()
current = validate_model_smoke(load_openclip_text_embedding(weight, SMOKE_QUERY), weight, SMOKE_QUERY)
print(json.dumps({"saved": saved, "current": current, "equal": saved == current}, sort_keys=True))
for key in sorted(set(saved) | set(current)):
    if saved.get(key) != current.get(key):
        print(f"DIFF {key} saved={saved.get(key)!r} current={current.get(key)!r}")
PY
