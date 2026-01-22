#!/usr/bin/env bash
set -euo pipefail
REPO=/root/workspace/scenesmith-hts
RUN="$REPO/outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710"
SOURCE=/root/workspace/.codex_scratch/dimension_contract_verify_v2_20260713/tests/unit/test_artiverse_retrieval_integration.py
PATCH=/root/workspace/.codex_scratch/380036e-dimension-contract-v2.patch
BACKUP="/root/workspace/.codex_scratch/dimension37_v2_backup_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$BACKUP/tests/unit" "$BACKUP/upstream-patches" "$BACKUP/run"
exec 9>"$RUN/.pipeline.lock"
flock -w 30 9
cd "$REPO"
if pgrep -f '^bash remote_jobs/run_full_quality_school_sqz.sh' >/dev/null; then
  echo "ERROR: pipeline runner is active" >&2
  exit 70
fi
test "$(sha256sum "$SOURCE" | awk '{print $1}')" = "acd6b438c2f31ed634a7698ba7e043b6a58fb8fead41a30a363d5af5a1c35b8b"
test "$(sha256sum "$PATCH" | awk '{print $1}')" = "58965984c21a53c89c61e60edd7f23ccc244f4f01d010fe170f54e1a214d9d80"
test "$(sha256sum upstream-patches/APPLY_ORDER.txt | awk '{print $1}')" = "21d2f3f9daba118a0ed013cae10ad40c8a2ad881a074db5e17c226a54186ca0e"
cp -a tests/unit/test_artiverse_retrieval_integration.py "$BACKUP/tests/unit/"
cp -a upstream-patches/380036e-dimension-contract.patch "$BACKUP/upstream-patches/"
cp -a "$RUN/pipeline_code_contract.json" "$BACKUP/run/"
rollback() {
  rc=$?
  if test "$rc" -ne 0; then
    cp -a "$BACKUP/tests/unit/test_artiverse_retrieval_integration.py" tests/unit/
    cp -a "$BACKUP/upstream-patches/380036e-dimension-contract.patch" upstream-patches/
    cp -a "$BACKUP/run/pipeline_code_contract.json" "$RUN/"
    echo "ROLLED_BACK_V2 rc=$rc backup=$BACKUP" >&2
  fi
  exit "$rc"
}
trap rollback EXIT
install -m 0644 "$SOURCE" tests/unit/test_artiverse_retrieval_integration.py
install -m 0644 "$PATCH" upstream-patches/380036e-dimension-contract.patch
test "$(sha256sum tests/unit/test_artiverse_retrieval_integration.py | awk '{print $1}')" = "acd6b438c2f31ed634a7698ba7e043b6a58fb8fead41a30a363d5af5a1c35b8b"
test "$(sha256sum upstream-patches/380036e-dimension-contract.patch | awk '{print $1}')" = "58965984c21a53c89c61e60edd7f23ccc244f4f01d010fe170f54e1a214d9d80"
git diff --check -- tests/unit/test_artiverse_retrieval_integration.py
source /root/workspace/Humanoid-Training-Scene-Ready-Generation/local_setup/setup_env_sqz.sh
cd "$REPO"
python -m pytest -q \
  tests/unit/test_artiverse_retrieval_integration.py::test_asset_manager_records_visual_normalization_in_artiverse_metadata \
  tests/unit/test_dimension_contract.py \
  tests/unit/test_mesh_utils.py \
  tests/unit/test_asset_router.py \
  tests/unit/test_asset_manager.py \
  tests/unit/test_objaverse_retrieval.py \
  tests/unit/test_objathor_manipuland_routing.py
python scripts/pipeline_code_contract.py \
  --repo-dir "$REPO" \
  --spec CODEX_SCENESMITH_FULL_QUALITY_PIPELINE.md \
  --runner remote_jobs/run_full_quality_school_sqz.sh \
  --output "$RUN/pipeline_code_contract.json"
python scripts/pipeline_code_contract.py \
  --repo-dir "$REPO" \
  --spec CODEX_SCENESMITH_FULL_QUALITY_PIPELINE.md \
  --runner remote_jobs/run_full_quality_school_sqz.sh \
  --output "$RUN/pipeline_code_contract.json" \
  --verify-only
trap - EXIT
sha256sum upstream-patches/380036e-dimension-contract.patch tests/unit/test_artiverse_retrieval_integration.py "$RUN/pipeline_code_contract.json"
echo "DEPLOYED_V2 backup=$BACKUP"
