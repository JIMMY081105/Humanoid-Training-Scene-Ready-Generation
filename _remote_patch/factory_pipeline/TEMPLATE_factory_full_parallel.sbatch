#!/usr/bin/env bash
# Maximum account-safe parallel factory production (observed scvj260 cap: 8 GPUs).
#SBATCH -J factory_full_8gpu
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -t 7-00:00:00
#SBATCH -o /data/run01/scvj260/factory_outputs/slurm/%x-%j.out

set -euo pipefail
readonly REPO=/data/run01/scvj260/scenesmith-factory-codex
mkdir -p /data/run01/scvj260/factory_outputs/slurm
cd "${REPO}"
source remote_jobs/factory_compute_env.sh
export SCENESMITH_EXECUTION_MODE=full
export SCENESMITH_MAX_PARALLEL_GPUS=8
exec bash remote_jobs/run_full_quality_factory_paracloud.sh
