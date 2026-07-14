#!/usr/bin/env bash
# One-GPU calibration; exits benchmark_complete and releases the allocation.
#SBATCH -J factory_benchmark
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -t 1-00:00:00
#SBATCH -o /data/run01/scvj260/factory_outputs/slurm/%x-%j.out

set -euo pipefail
readonly REPO=/data/run01/scvj260/scenesmith-factory-codex
mkdir -p /data/run01/scvj260/factory_outputs/slurm
cd "${REPO}"
source remote_jobs/factory_compute_env.sh
export SCENESMITH_EXECUTION_MODE=benchmark_processing_hall
export SCENESMITH_MAX_PARALLEL_GPUS=1
exec bash remote_jobs/run_full_quality_factory_paracloud.sh
