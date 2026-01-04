#!/usr/bin/env bash
# Verify that one GPU node can reach the loopback-only login tunnel.  It never
# exposes the user's proxy to the cluster network.

#SBATCH -J codex_gpu_proxy
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:05:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%j.out

set -euo pipefail
readonly FACTORY=/data/run01/scvj260/codex_factory
readonly LOCAL_PORT=18094

ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ConnectTimeout=15 \
  -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$FACTORY/ln07_known_hosts" \
  -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:18093" ln07 &
TUNNEL_PID=$!
cleanup() { kill "$TUNNEL_PID" 2>/dev/null || true; wait "$TUNNEL_PID" 2>/dev/null || true; }
trap cleanup EXIT

for _ in $(seq 1 10); do
  if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    wait "$TUNNEL_PID"
  fi
  CODE=$(curl --connect-timeout 10 --max-time 20 -sS -o /dev/null -w '%{http_code}' \
    -x "http://127.0.0.1:${LOCAL_PORT}" https://api.openai.com/v1/models || true)
  if [ "$CODE" = 401 ]; then
    echo GPU_PROXY_TUNNEL_OK
    exit 0
  fi
  sleep 2
done
echo "GPU_PROXY_TUNNEL_BAD_STATUS=${CODE:-none}" >&2
exit 1
