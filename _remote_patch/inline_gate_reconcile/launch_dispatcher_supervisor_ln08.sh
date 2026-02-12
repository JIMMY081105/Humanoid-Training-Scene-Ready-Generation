#!/usr/bin/env bash
set -euo pipefail
readonly FACTORY=/data/run01/scvj260/codex_factory
mkdir -p "$FACTORY/logs"
nohup setsid bash "$FACTORY/paracloud_dispatcher_supervisor.sh" \
  >> "$FACTORY/logs/paracloud_dispatcher_supervisor.log" 2>&1 </dev/null &
echo "SUPERVISOR_LAUNCHED $!"
