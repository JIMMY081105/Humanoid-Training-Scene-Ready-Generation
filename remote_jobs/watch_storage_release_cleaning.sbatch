#!/usr/bin/env bash
#SBATCH -J watch_storage_release_cleaning
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --cpus-per-task=1
#SBATCH -t 12:00:00
#SBATCH -o /data/run01/scvj260/scenesmith/outputs/2026-07-03/brand_new_school_floor_02-36-38_furniture_hssd_only_fast/watch_storage_release_cleaning-%j.out

set -euo pipefail

BASE="/data/run01/scvj260/scenesmith/outputs/2026-07-03/brand_new_school_floor_02-36-38_furniture_hssd_only_fast"
SCENE="$BASE/scene_000"
STORAGE_JOB="165649"
CLEANING_JOB="165719"

room_complete() {
  local room="$1"
  local final="$SCENE/room_${room}/scene_states/final_scene/scene_state.json"
  local log
  log="$(ls -t "$BASE"/room_worker_"${room}"*.out 2>/dev/null | head -1 || true)"
  [[ -f "$final" ]] && [[ -n "$log" ]] && grep -q "Completed one-room worker" "$log"
}

job_state() {
  squeue -j "$1" -h -o "%T" 2>/dev/null | tr -d ' ' || true
}

echo "[watcher] start $(date -u '+%Y-%m-%d %H:%M:%S UTC') host=$(hostname)"
echo "[watcher] will release cleaning only after storage final_scene and completed log exist"

while true; do
  now="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  storage_state="$(job_state "$STORAGE_JOB")"
  cleaning_state="$(job_state "$CLEANING_JOB")"
  echo "[watcher] $now storage_state=${storage_state:-not_in_queue} cleaning_state=${cleaning_state:-not_in_queue}"

  if room_complete storage_room; then
    echo "[watcher] storage_room complete"
    if [[ "$storage_state" == "RUNNING" ]]; then
      echo "[watcher] cancelling completed-hung storage job $STORAGE_JOB"
      scancel "$STORAGE_JOB" || true
      sleep 20
    fi
    cleaning_state="$(job_state "$CLEANING_JOB")"
    if [[ "$cleaning_state" == "PENDING" ]]; then
      echo "[watcher] releasing cleaning_closet job $CLEANING_JOB"
      scontrol release "$CLEANING_JOB" || true
    fi
  elif [[ -z "$storage_state" ]]; then
    echo "[watcher] storage job is gone but verified storage final is missing; not releasing cleaning"
    exit 2
  fi

  if room_complete cleaning_closet; then
    echo "[watcher] cleaning_closet complete"
    cleaning_state="$(job_state "$CLEANING_JOB")"
    if [[ "$cleaning_state" == "RUNNING" ]]; then
      echo "[watcher] cancelling completed-hung cleaning job $CLEANING_JOB"
      scancel "$CLEANING_JOB" || true
    fi
    echo "[watcher] done $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    exit 0
  fi

  sleep 300
done
