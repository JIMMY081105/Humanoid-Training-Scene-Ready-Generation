#!/usr/bin/env bash
# Extract and prepare all retrieval assets; one GPU accelerates Artiverse CLIP indexing.
#SBATCH -J factory_assets_1gpu
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -t 2-00:00:00
#SBATCH -o /data/run01/scvj260/factory_outputs/slurm/%x-%j.out

set -euo pipefail
readonly REPO=/data/run01/scvj260/scenesmith-factory-codex
mkdir -p /data/run01/scvj260/factory_outputs/slurm
cd "${REPO}"
source remote_jobs/factory_compute_env.sh
exec bash remote_jobs/prepare_factory_assets_cpu.sh
