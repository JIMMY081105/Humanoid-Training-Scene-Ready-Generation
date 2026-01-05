#!/usr/bin/env bash
# The last two manifest rooms.  They remain pending under the account's hard
# eight-GPU cap and start automatically as current workers finish.

#SBATCH -J codex_room_tail
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 1-00:00:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%A_%a.out

set -euo pipefail
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly SOURCE="$FACTORY/paracloud_parallel_room_workers.sbatch"
readonly DERIVED="$FACTORY/.tail_room_worker_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.sh"
trap 'rm -f "$DERIVED"' EXIT

# Preserve the validated worker implementation while giving this wave unique
# proxy/Blender ports and its two remaining manifest room ids.
sed \
  -e 's/library storage_room classroom_02 classroom_03 classroom_04 classroom_05 classroom_06 boys_toilet/girls_toilet main_corridor/' \
  -e 's/18100 + SLURM_ARRAY_TASK_ID/18200 + SLURM_ARRAY_TASK_ID/' \
  -e 's/200 + SLURM_ARRAY_TASK_ID \* 50/600 + SLURM_ARRAY_TASK_ID * 50/' \
  "$SOURCE" > "$DERIVED"
bash "$DERIVED"
