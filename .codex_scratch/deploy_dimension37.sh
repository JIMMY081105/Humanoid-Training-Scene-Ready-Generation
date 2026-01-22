#!/usr/bin/env bash
set -euo pipefail
REPO=/root/workspace/scenesmith-hts
RUN="$REPO/outputs/full_quality_school_reference_20260710"
BUNDLE=/root/workspace/.codex_scratch/dimension37_deploy_20260712T235617Z
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BACKUP="/root/workspace/.codex_scratch/dimension37_backup_$STAMP"
mkdir -p "$RUN" "$BACKUP"
exec 9>"$RUN/.pipeline.lock"
flock -w 30 9
cd "$REPO"
if pgrep -f '^bash remote_jobs/run_full_quality_school_sqz.sh' >/dev/null; then
  echo "ERROR: pipeline runner is active" >&2
  exit 70
fi
test "$(sha256sum "$BUNDLE/380036e-dimension-contract.patch" | awk '{print $1}')" = "21facf73219c96ae180433eabe0be81d8c23eb1e103d613c7a6f1db4bfedb604"
test "$(sha256sum "$BUNDLE/APPLY_ORDER.txt" | awk '{print $1}')" = "21d2f3f9daba118a0ed013cae10ad40c8a2ad881a074db5e17c226a54186ca0e"
test "$(grep -Ec '\.patch$' "$BUNDLE/APPLY_ORDER.txt")" = "37"
test "$(tail -n 1 "$BUNDLE/APPLY_ORDER.txt")" = "380036e-dimension-contract.patch"
targets=(
  scenesmith/agent_utils/asset_manager.py
  scenesmith/agent_utils/asset_router/router.py
  scenesmith/agent_utils/mesh_utils.py
  tests/unit/test_asset_manager.py
  tests/unit/test_dimension_contract.py
  tests/unit/test_objathor_manipuland_routing.py
  upstream-patches/APPLY_ORDER.txt
  upstream-patches/380036e-dimension-contract.patch
)
: > "$BACKUP/existed.list"
for f in "${targets[@]}"; do
  if test -e "$f"; then
    mkdir -p "$BACKUP/$(dirname "$f")"
    cp -a "$f" "$BACKUP/$f"
    printf '%s\n' "$f" >> "$BACKUP/existed.list"
  fi
done
rollback() {
  rc=$?
  if test "$rc" -ne 0; then
    for f in "${targets[@]}"; do rm -f "$f"; done
    while IFS= read -r f; do
      mkdir -p "$(dirname "$f")"
      cp -a "$BACKUP/$f" "$f"
    done < "$BACKUP/existed.list"
    echo "ROLLED_BACK rc=$rc backup=$BACKUP" >&2
  fi
  exit "$rc"
}
trap rollback EXIT
git apply --check --whitespace=error-all "$BUNDLE/380036e-dimension-contract.patch"
git apply --whitespace=error-all "$BUNDLE/380036e-dimension-contract.patch"
install -m 0644 "$BUNDLE/380036e-dimension-contract.patch" upstream-patches/380036e-dimension-contract.patch
install -m 0644 "$BUNDLE/APPLY_ORDER.txt" upstream-patches/APPLY_ORDER.txt
cat > "$BACKUP/expected.sha256" <<'EOF'
2a8679e2b7d91a8027925c31bea19a5983e6d7a816d225b10ec72eeb07e35455  scenesmith/agent_utils/asset_manager.py
3bfcdeaf736ff42bf7e01d9db1a8a9d1a8baf9f4f642c8651773f30e68969fa0  scenesmith/agent_utils/asset_router/router.py
7e7db1c7bf6a6acc072357adab8895678f0fe6b581a45b2b37b01f79df762673  scenesmith/agent_utils/mesh_utils.py
599cd53d729ed3e72934aea87bb45b2dac3e2c48928b1a16132456b84df7bcb7  tests/unit/test_asset_manager.py
97e3f36bbc7b593a1464fe4d6201a0a6c0d8c81d72229e3e11be666bfb895004  tests/unit/test_dimension_contract.py
2a609ebf239626bc49f51ec949650ce55555023acb1fd666e7e5b8cce4e9e738  tests/unit/test_objathor_manipuland_routing.py
EOF
sha256sum -c "$BACKUP/expected.sha256"
test "$(sha256sum upstream-patches/380036e-dimension-contract.patch | awk '{print $1}')" = "21facf73219c96ae180433eabe0be81d8c23eb1e103d613c7a6f1db4bfedb604"
test "$(sha256sum upstream-patches/APPLY_ORDER.txt | awk '{print $1}')" = "21d2f3f9daba118a0ed013cae10ad40c8a2ad881a074db5e17c226a54186ca0e"
test "$(grep -Ec '\.patch$' upstream-patches/APPLY_ORDER.txt)" = "37"
git diff --check -- \
  scenesmith/agent_utils/asset_manager.py \
  scenesmith/agent_utils/asset_router/router.py \
  scenesmith/agent_utils/mesh_utils.py \
  tests/unit/test_asset_manager.py \
  tests/unit/test_objathor_manipuland_routing.py
/root/workspace/miniconda3/envs/scenesmith/bin/python -m py_compile \
  scenesmith/agent_utils/asset_manager.py \
  scenesmith/agent_utils/asset_router/router.py \
  scenesmith/agent_utils/mesh_utils.py \
  tests/unit/test_dimension_contract.py
trap - EXIT
echo "DEPLOYED backup=$BACKUP"
