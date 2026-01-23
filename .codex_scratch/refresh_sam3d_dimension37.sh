#!/usr/bin/env bash
set -euo pipefail
REPO=/root/workspace/scenesmith-hts
RUN="$REPO/outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710"
PREFLIGHT="$REPO/outputs/preflight/full_quality_school_reference_20260710"
TARGET="$PREFLIGHT/sam3d_offline_generation"
BACKUP="/root/workspace/.codex_scratch/sam3d_offline_generation_pre_dimension37_$(date -u +%Y%m%dT%H%M%SZ)"
exec 9>"$RUN/.pipeline.lock"
flock -w 30 9
if pgrep -f '^bash remote_jobs/run_full_quality_school_sqz.sh' >/dev/null; then
  echo "ERROR: pipeline runner is active" >&2
  exit 70
fi
test -d "$TARGET"
test -f "$TARGET/receipt.json"
mv "$TARGET" "$BACKUP"
rollback() {
  rc=$?
  if test "$rc" -ne 0; then
    rm -rf "$TARGET"
    mv "$BACKUP" "$TARGET"
    echo "RESTORED_OLD_PROOF rc=$rc backup=$BACKUP" >&2
  fi
  exit "$rc"
}
trap rollback EXIT
source /root/workspace/Humanoid-Training-Scene-Ready-Generation/local_setup/setup_env_sqz.sh
cd "$REPO"
python scripts/preflight_sam3d_generation.py \
  --repo-dir "$REPO" \
  --preflight-dir "$PREFLIGHT" \
  --sam3-checkpoint external/checkpoints/sam3.pt \
  --pipeline-config external/checkpoints/pipeline.yaml
python scripts/preflight_sam3d_generation.py \
  --repo-dir "$REPO" \
  --preflight-dir "$PREFLIGHT" \
  --sam3-checkpoint external/checkpoints/sam3.pt \
  --pipeline-config external/checkpoints/pipeline.yaml \
  --verify-only
trap - EXIT
sha256sum "$TARGET/receipt.json"
echo "REFRESHED backup=$BACKUP"
