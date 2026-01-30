#!/usr/bin/env bash
# Continue the exact full-quality school runner only after every independently
# generated room has passed its release gate. No room generation is restarted
# here; the production runner reuses hash-valid final room evidence.

#SBATCH -J codex_school_final
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%j.out

set -euo pipefail
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly REPO="$FACTORY/room1_execution"
readonly VENV=/data/run01/scvj260/scenesmith/.venv
readonly RUN="$REPO/outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710"
readonly SCENE="$RUN/scene_000"
readonly STAGED_VISUAL="$SCENE/quality_gates/room_visual_self_exam"
readonly CANONICAL_VISUAL="$SCENE/quality_gates/room_self_exam"
readonly PROXY_PORT=18500

case "$(hostname)" in
  ln*|Ln*|lN*|LN*) echo 'Refusing login node.' >&2; exit 65 ;;
esac

# The parallel release jobs intentionally share one staging directory. Promote
# that complete evidence set atomically to the contract's canonical directory.
# A later evidence-driven rerun may reuse the already-promoted directory.
if [[ -d "$STAGED_VISUAL" && ! -e "$CANONICAL_VISUAL" && ! -L "$CANONICAL_VISUAL" ]]; then
  VISUAL_SOURCE="$STAGED_VISUAL"
elif [[ -d "$CANONICAL_VISUAL" && ! -e "$STAGED_VISUAL" && ! -L "$STAGED_VISUAL" ]]; then
  VISUAL_SOURCE="$CANONICAL_VISUAL"
else
  echo "visual-gate staging/canonical directories are missing or ambiguous" >&2
  exit 66
fi
"$VENV/bin/python" - "$VISUAL_SOURCE" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rooms = (
    "library", "storage_room", "classroom_01", "classroom_02",
    "classroom_03", "classroom_04", "classroom_05", "classroom_06",
    "boys_toilet", "girls_toilet", "main_corridor",
)
for room in rooms:
    path = root / f"{room}.json"
    try:
        decision = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"missing or malformed release decision {path}: {error}")
    if decision.get("room_id") != room or decision.get("status") != "pass":
        raise SystemExit(f"room release gate did not pass: {path}")
PY

if [[ "$VISUAL_SOURCE" == "$STAGED_VISUAL" ]]; then
  mv "$STAGED_VISUAL" "$CANONICAL_VISUAL"
fi

source "$VENV/bin/activate"
cd "$REPO"
set -a
source .env
set +a
export PYTHONDONTWRITEBYTECODE=1
export SCENESMITH_RUNTIME=paracloud
export SCENESMITH_REPO="$REPO"
export SCENESMITH_PARACLOUD_VENV="$VENV"
export SCENESMITH_PYTHON_OVERLAY="$FACTORY/python_overlay_agents_0_17_4"
export SCENESMITH_HF_HOME="$FACTORY/huggingface"
export SCENESMITH_TORCH_HOME="$FACTORY/torch"
export SCENESMITH_EXECUTION_MODE=full

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ConnectTimeout=15 \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
  -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$FACTORY/ln07_known_hosts" \
  -L "127.0.0.1:${PROXY_PORT}:127.0.0.1:18114" ln07 &
TUNNEL_PID=$!
cleanup() {
  kill "$TUNNEL_PID" 2>/dev/null || true
  wait "$TUNNEL_PID" 2>/dev/null || true
}
trap cleanup EXIT
for _ in $(seq 1 10); do
  if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then wait "$TUNNEL_PID"; fi
  status=$(curl --connect-timeout 10 --max-time 20 -sS -o /dev/null \
    -w '%{http_code}' -x "http://127.0.0.1:${PROXY_PORT}" \
    https://api.openai.com/v1/models || true)
  [[ "$status" == 401 ]] && break
  sleep 2
done
[[ "${status:-000}" == 401 ]] || {
  echo "final-pipeline proxy unavailable: ${status:-000}" >&2
  exit 70
}
export SCENESMITH_AGENT_HTTP_PROXY="http://127.0.0.1:${PROXY_PORT}"

bash remote_jobs/run_full_quality_school_sqz.sh

# The full runner has now created the immutable outside-scene manifest and
# pending completion receipt. Preserve their digests in this parent job and
# pass them explicitly to the independent two-GPU acceptance allocation.
readonly PACKAGE_MANIFEST="$RUN/scene_000.sha256"
readonly COMPLETION_RECEIPT="$RUN/pipeline_completion.json"
readonly ACCEPTANCE_TEMPLATE="$REPO/remote_jobs/TEMPLATE_2gpu_drake_acceptance.sbatch"
readonly ACCEPTANCE_SUBMISSION="$FACTORY/two_gpu_acceptance_submission.txt"
test -s "$PACKAGE_MANIFEST"
test -s "$COMPLETION_RECEIPT"
test -s "$ACCEPTANCE_TEMPLATE"
read -r RUN_ATTEMPT_ID EXPECTED_MANIFEST_SHA256 < <(
  "$VENV/bin/python" - "$COMPLETION_RECEIPT" <<'PY'
import json
import sys
from pathlib import Path

document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if document.get("status") != "awaiting_2gpu_acceptance":
    raise SystemExit("pipeline completion is not awaiting two-GPU acceptance")
attempt = document.get("run_attempt_id")
manifest = document.get("package", {}).get("manifest_sha256")
if not isinstance(attempt, str) or not attempt:
    raise SystemExit("missing final run_attempt_id")
if not isinstance(manifest, str) or len(manifest) != 64:
    raise SystemExit("missing package-manifest digest")
print(attempt, manifest)
PY
)
test "$(sha256sum "$PACKAGE_MANIFEST" | awk '{print $1}')" = "$EXPECTED_MANIFEST_SHA256"
EXPECTED_COMPLETION_SHA256=$(sha256sum "$COMPLETION_RECEIPT" | awk '{print $1}')

while true; do
  if output=$(sbatch \
    --dependency="afterok:$SLURM_JOB_ID" \
    --output="$FACTORY/logs/drake_accept_2gpu-%j.out" \
    --export="ALL,REPO_ROOT=$REPO,PYTHON_BIN=$VENV/bin/python,SCENE_PACKAGE_ROOT=$SCENE,PACKAGE_SHA256_MANIFEST=$PACKAGE_MANIFEST,EXPECTED_MANIFEST_SHA256=$EXPECTED_MANIFEST_SHA256,EXPECTED_RUN_ATTEMPT_ID=$RUN_ATTEMPT_ID,SQZ_PIPELINE_COMPLETION=$COMPLETION_RECEIPT,EXPECTED_SQZ_COMPLETION_SHA256=$EXPECTED_COMPLETION_SHA256" \
    "$ACCEPTANCE_TEMPLATE" 2>&1)
  then
    temporary="$ACCEPTANCE_SUBMISSION.tmp.$$"
    printf '%s\n' "$output" > "$temporary"
    mv -f "$temporary" "$ACCEPTANCE_SUBMISSION"
    break
  fi
  if printf '%s' "$output" | grep -q 'AssocMaxSubmitJobLimit'; then
    sleep 30
    continue
  fi
  echo "two-GPU acceptance submission failed: $output" >&2
  exit 71
done
