#!/usr/bin/env bash
# Template: final, hash-bound Drake acceptance on one ParaCloud node with 2 GPUs.
#
# Transfer the complete scene directory and a separate GNU sha256sum manifest.
# Set the six required variables out-of-band when submitting, for example:
#   sbatch --export=ALL,SCENE_PACKAGE_ROOT=/data/.../scene_000,\
#PACKAGE_SHA256_MANIFEST=/data/.../scene_000.sha256,\
#EXPECTED_MANIFEST_SHA256=<sha256-of-scene_000.sha256>,\
#EXPECTED_RUN_ATTEMPT_ID=<immutable-run-attempt-id>,\
#SQZ_PIPELINE_COMPLETION=/data/.../pipeline_completion.json,\
#EXPECTED_SQZ_COMPLETION_SHA256=<sha256-of-pipeline_completion.json> \
#     remote_jobs/TEMPLATE_2gpu_drake_acceptance.sbatch
# The manifest must be outside SCENE_PACKAGE_ROOT and list every package file once,
# using paths relative to SCENE_PACKAGE_ROOT. Extra, missing, symlinked, or mutated
# files are fatal. A pass requires both pre- and post-Drake verification.

#SBATCH -J drake_accept_2gpu
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH -t 04:00:00
#SBATCH -o /data/run01/scvj260/scenesmith/outputs/%x-%j.out

set -euo pipefail
umask 077

REPO_ROOT="${REPO_ROOT:-/data/run01/scvj260/scenesmith}"
cd "$REPO_ROOT"

: "${SLURM_JOB_ID:?This template must run inside a SLURM allocation}"
: "${SCENE_PACKAGE_ROOT:?Set the exact transferred scene package directory}"
: "${PACKAGE_SHA256_MANIFEST:?Set the external GNU sha256sum package manifest}"
: "${EXPECTED_MANIFEST_SHA256:?Set the out-of-band SHA-256 of that manifest}"
: "${EXPECTED_RUN_ATTEMPT_ID:?Set the exact SQZ run_attempt_id out-of-band}"
: "${SQZ_PIPELINE_COMPLETION:?Set the outside-scene SQZ pipeline completion receipt}"
: "${EXPECTED_SQZ_COMPLETION_SHA256:?Set its out-of-band SHA-256}"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
DMD_RELATIVE_PATH="${DMD_RELATIVE_PATH:-combined_house/house.dmd.yaml}"
ACCEPTANCE_DIR="${ACCEPTANCE_DIR:-$REPO_ROOT/outputs/two_gpu_drake_acceptance/$SLURM_JOB_ID}"

test -x "$PYTHON_BIN"
test -f scripts/two_gpu_drake_acceptance_contract.py
test -f scripts/validate_drake_scene.py
test -d "$SCENE_PACKAGE_ROOT"
test -f "$PACKAGE_SHA256_MANIFEST"
test -f "$SQZ_PIPELINE_COMPLETION"

SCENE_PACKAGE_ROOT="$(realpath "$SCENE_PACKAGE_ROOT")"
PACKAGE_SHA256_MANIFEST="$(realpath "$PACKAGE_SHA256_MANIFEST")"
SQZ_PIPELINE_COMPLETION="$(realpath "$SQZ_PIPELINE_COMPLETION")"
mkdir -p "$ACCEPTANCE_DIR"
ACCEPTANCE_DIR="$(realpath "$ACCEPTANCE_DIR")"
case "$ACCEPTANCE_DIR/" in
  "$SCENE_PACKAGE_ROOT/"*)
    echo "[guard] ACCEPTANCE_DIR must be outside SCENE_PACKAGE_ROOT" >&2
    exit 21
    ;;
esac
case "$SQZ_PIPELINE_COMPLETION" in
  "$SCENE_PACKAGE_ROOT"|"$SCENE_PACKAGE_ROOT/"*)
    echo "[guard] SQZ_PIPELINE_COMPLETION must be outside SCENE_PACKAGE_ROOT" >&2
    exit 24
    ;;
esac
if find "$ACCEPTANCE_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "[guard] acceptance directory is not empty: $ACCEPTANCE_DIR" >&2
  exit 22
fi

DMD_PATH="$(realpath "$SCENE_PACKAGE_ROOT/$DMD_RELATIVE_PATH")"
case "$DMD_PATH" in
  "$SCENE_PACKAGE_ROOT/"*) ;;
  *)
    echo "[guard] DMD_RELATIVE_PATH resolves outside SCENE_PACKAGE_ROOT" >&2
    exit 23
    ;;
esac
test -f "$DMD_PATH"

PACKAGE_PREFLIGHT="$ACCEPTANCE_DIR/package_preflight.json"
PACKAGE_POSTFLIGHT="$ACCEPTANCE_DIR/package_postflight.json"
RUNTIME_VALIDATION="$ACCEPTANCE_DIR/runtime_validation.json"
DRAKE_REPORT="$ACCEPTANCE_DIR/drake_validation.json"
DRAKE_REPORT_TMP="$ACCEPTANCE_DIR/.drake_validation.json.$SLURM_JOB_ID.tmp"
ACCEPTANCE_RECEIPT="$ACCEPTANCE_DIR/two_gpu_drake_acceptance_receipt.json"

cleanup() {
  rm -f "$DRAKE_REPORT_TMP"
}
trap cleanup EXIT

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1

echo "[slurm] job_id=$SLURM_JOB_ID node=$(hostname) start=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
nvidia-smi -L

"$PYTHON_BIN" scripts/two_gpu_drake_acceptance_contract.py verify-package \
  --package-root "$SCENE_PACKAGE_ROOT" \
  --manifest "$PACKAGE_SHA256_MANIFEST" \
  --expected-manifest-sha256 "$EXPECTED_MANIFEST_SHA256" \
  --expected-run-attempt-id "$EXPECTED_RUN_ATTEMPT_ID" \
  --output "$PACKAGE_PREFLIGHT"

# This runs before sourcing the environment file: every executable runtime
# artifact, including that environment file and this template, must match the
# immutable pipeline/code proof shipped inside the hash-verified scene package.
"$PYTHON_BIN" scripts/two_gpu_drake_acceptance_contract.py verify-runtime \
  --package-preflight "$PACKAGE_PREFLIGHT" \
  --repo-root "$REPO_ROOT" \
  --expected-run-attempt-id "$EXPECTED_RUN_ATTEMPT_ID" \
  --output "$RUNTIME_VALIDATION"

source local_setup/compute_node_env.sh
cd "$REPO_ROOT"
test -x "$PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import torch
import pydrake

if torch.cuda.device_count() < 2:
    raise SystemExit("fewer than two CUDA GPUs are visible")
print("ParaCloud runtime preflight passed", pydrake.__file__)
PY

"$PYTHON_BIN" scripts/validate_drake_scene.py \
  --dmd "$DMD_PATH" \
  --package-root "$SCENE_PACKAGE_ROOT" \
  --output "$DRAKE_REPORT_TMP" \
  --require-gpus 2 \
  --max-collision-elements 32 \
  --minimum-models 12 \
  --expected-rooms 11
mv -f "$DRAKE_REPORT_TMP" "$DRAKE_REPORT"

# Drake validation must not mutate the accepted package. Rehash every file before
# issuing a receipt, even though validate_drake_scene.py returned zero.
"$PYTHON_BIN" scripts/two_gpu_drake_acceptance_contract.py verify-package \
  --package-root "$SCENE_PACKAGE_ROOT" \
  --manifest "$PACKAGE_SHA256_MANIFEST" \
  --expected-manifest-sha256 "$EXPECTED_MANIFEST_SHA256" \
  --expected-run-attempt-id "$EXPECTED_RUN_ATTEMPT_ID" \
  --output "$PACKAGE_POSTFLIGHT"

"$PYTHON_BIN" scripts/two_gpu_drake_acceptance_contract.py finalize \
  --package-preflight "$PACKAGE_PREFLIGHT" \
  --package-postflight "$PACKAGE_POSTFLIGHT" \
  --drake-report "$DRAKE_REPORT" \
  --runtime-validation "$RUNTIME_VALIDATION" \
  --sqz-pipeline-completion "$SQZ_PIPELINE_COMPLETION" \
  --expected-sqz-completion-sha256 "$EXPECTED_SQZ_COMPLETION_SHA256" \
  --expected-run-attempt-id "$EXPECTED_RUN_ATTEMPT_ID" \
  --slurm-job-id "$SLURM_JOB_ID" \
  --node-name "$(hostname)" \
  --output "$ACCEPTANCE_RECEIPT"

test -s "$ACCEPTANCE_RECEIPT"
echo "[acceptance] receipt=$ACCEPTANCE_RECEIPT sha256=$(sha256sum "$ACCEPTANCE_RECEIPT" | awk '{print $1}')"
echo "[slurm] end=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
