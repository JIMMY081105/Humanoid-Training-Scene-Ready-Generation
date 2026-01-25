#!/usr/bin/env bash
set -euo pipefail

readonly DIRECT_PORT=18113
readonly STABLE_PORT=18114
readonly URL=https://api.openai.com/v1/models

stop_relays() {
  for node in ln07 ln08; do
    ssh -n "$node" "pkill -f '^ssh -N .*127\\.0\\.0\\.1:18114:127\\.0\\.0\\.1:18113 ln0[78]$' 2>/dev/null || true" || true
  done
}

probe_port() {
  local node="$1" port="$2"
  local code
  code="$(ssh -n "$node" "curl -sS -o /dev/null -w '%{http_code}' --max-time 8 -x http://127.0.0.1:${port} $URL" 2>/dev/null || true)"
  [[ "$code" == "401" ]]
}

start_relays() {
  local source="$1" destination="$2"
  ssh -n "$source" "
    if ! pgrep -f '^ssh -N .* -L 127\\.0\\.0\\.1:18114:127\\.0\\.0\\.1:18113 ${source}$' >/dev/null; then
      nohup ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        -L 127.0.0.1:18114:127.0.0.1:18113 '${source}' \
        >/dev/null 2>&1 &
    fi
    if ! pgrep -f '^ssh -N .* -R 127\\.0\\.0\\.1:18114:127\\.0\\.0\\.1:18113 ${destination}$' >/dev/null; then
      nohup ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        -R 127.0.0.1:18114:127.0.0.1:18113 '${destination}' \
        >/dev/null 2>&1 &
    fi
  "
}

if [[ "${1:-}" == "--reset-relays" ]]; then
  stop_relays
  exit 0
fi

ln07_stable=0
ln08_stable=0
probe_port ln07 "$STABLE_PORT" && ln07_stable=1
probe_port ln08 "$STABLE_PORT" && ln08_stable=1

if (( ln07_stable && ln08_stable )); then
  exit 0
fi

ln07_direct=0
ln08_direct=0
probe_port ln07 "$DIRECT_PORT" && ln07_direct=1
probe_port ln08 "$DIRECT_PORT" && ln08_direct=1
stop_relays
sleep 1
if (( ln08_direct )); then
  start_relays ln08 ln07
elif (( ln07_direct )); then
  start_relays ln07 ln08
else
  exit 1
fi

sleep 3
probe_port ln07 "$STABLE_PORT"
probe_port ln08 "$STABLE_PORT"
