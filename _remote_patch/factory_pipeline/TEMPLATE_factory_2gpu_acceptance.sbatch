#!/usr/bin/env bash
# Terminal two-GPU Drake acceptance for the already hash-manifested factory package.
#SBATCH -J factory_accept_2gpu
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH -t 04:00:00
#SBATCH -o /data/run01/scvj260/factory_outputs/slurm/%x-%j.out

set -euo pipefail
readonly REPO=/data/run01/scvj260/scenesmith-factory-codex
readonly RUN_DIR=/data/run01/scvj260/factory_outputs/full_quality_factory_sam3d_artvip_artiverse_20260713
readonly SCENE_DIR="${RUN_DIR}/scene_000"
cd "${REPO}"
source remote_jobs/factory_compute_env.sh
: "${EXPECTED_MANIFEST_SHA256:?export the out-of-band scene_000.sha256 digest}"
: "${EXPECTED_RUN_ATTEMPT_ID:?export the immutable run attempt ID}"
: "${EXPECTED_PIPELINE_COMPLETION_SHA256:?export the out-of-band completion receipt digest}"
python scripts/factory_two_gpu_drake_acceptance_contract.py verify-package \
  --package-root "${SCENE_DIR}" --manifest "${RUN_DIR}/scene_000.sha256" \
  --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA256}" \
  --expected-run-attempt-id "${EXPECTED_RUN_ATTEMPT_ID}" \
  --output "${RUN_DIR}/factory_2gpu_package_preflight.json"
python scripts/factory_two_gpu_drake_acceptance_contract.py verify-runtime \
  --package-preflight "${RUN_DIR}/factory_2gpu_package_preflight.json" \
  --repo-root "${REPO}" --expected-run-attempt-id "${EXPECTED_RUN_ATTEMPT_ID}" \
  --output "${RUN_DIR}/factory_2gpu_runtime_validation.json"
python scripts/validate_drake_scene.py \
  --dmd "${SCENE_DIR}/combined_house/house.dmd.yaml" --package-root "${SCENE_DIR}" \
  --output "${RUN_DIR}/factory_2gpu_drake_validation.json" --require-gpus 2 \
  --max-collision-elements 32 --minimum-models 15 --expected-rooms 14
python scripts/factory_two_gpu_drake_acceptance_contract.py verify-package \
  --package-root "${SCENE_DIR}" --manifest "${RUN_DIR}/scene_000.sha256" \
  --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA256}" \
  --expected-run-attempt-id "${EXPECTED_RUN_ATTEMPT_ID}" \
  --output "${RUN_DIR}/factory_2gpu_package_postflight.json"
python scripts/factory_two_gpu_drake_acceptance_contract.py finalize \
  --package-preflight "${RUN_DIR}/factory_2gpu_package_preflight.json" \
  --package-postflight "${RUN_DIR}/factory_2gpu_package_postflight.json" \
  --drake-report "${RUN_DIR}/factory_2gpu_drake_validation.json" \
  --runtime-validation "${RUN_DIR}/factory_2gpu_runtime_validation.json" \
  --sqz-pipeline-completion "${RUN_DIR}/pipeline_completion.json" \
  --expected-sqz-completion-sha256 "${EXPECTED_PIPELINE_COMPLETION_SHA256}" \
  --expected-run-attempt-id "${EXPECTED_RUN_ATTEMPT_ID}" \
  --slurm-job-id "${SLURM_JOB_ID}" --node-name "$(hostname)" \
  --output "${RUN_DIR}/factory_2gpu_acceptance_receipt.json"
